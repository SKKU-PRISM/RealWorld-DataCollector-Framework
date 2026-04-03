"""
Generic HuggingFace VLM backend.

Supports any HuggingFace vision-language model as a VLA by adding
a linear action head on top of the VLM's hidden states.

Useful for experimenting with models not specifically designed for robotics,
such as Qwen-VL, LLaVA, or other multimodal models.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base_vla import BaseVLA
from .registry import register_vla
from .types import VLAInput, VLAModelConfig, VLAOutput

logger = logging.getLogger(__name__)


@register_vla("hf_vlm")
class HFVLMBackend(BaseVLA):
    """
    Generic HuggingFace VLM backend.

    Architecture: VLM encoder -> hidden states -> linear action head -> action
    Adds a trainable linear layer that maps VLM hidden states to action space.
    LoRA is applied to the VLM backbone; the action head is also trainable.
    """

    def __init__(self, config: VLAModelConfig):
        super().__init__(config)
        self._action_head: Any = None

    def load_model(self) -> None:
        """Load HuggingFace VLM with optional 4-bit quantization."""
        import torch
        from transformers import AutoModel, AutoProcessor

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.config.quantize_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"Loading HF VLM: {self.config.model_name}")
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        # Try Vision2Seq first, then generic AutoModel
        try:
            from transformers import AutoModelForVision2Seq

            self._model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name, **load_kwargs
            )
        except (ValueError, OSError):
            self._model = AutoModel.from_pretrained(
                self.config.model_name, **load_kwargs
            )

        # Create action head: hidden_dim -> action_dim
        hidden_dim = self._get_hidden_dim()
        self._action_head = torch.nn.Linear(hidden_dim, self.config.action_dim)
        self._action_head = self._action_head.to(
            dtype=torch.bfloat16, device=self._get_device()
        )

        logger.info(
            f"HF VLM loaded with action head: {hidden_dim} -> {self.config.action_dim}"
        )

    def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load LoRA adapter + action head weights."""
        import torch
        from peft import PeftModel

        if not self._loaded_adapters:
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, adapter_name=adapter_name
            )
        else:
            self._model.load_adapter(adapter_path, adapter_name=adapter_name)

        # Load action head if saved alongside adapter
        import os

        head_path = os.path.join(adapter_path, "action_head.pt")
        if os.path.exists(head_path):
            state = torch.load(head_path, map_location=self._get_device())
            self._action_head.load_state_dict(state)
            logger.info(f"Loaded action head from {head_path}")

        self._loaded_adapters[adapter_name] = adapter_path
        logger.info(f"Loaded LoRA adapter: {adapter_name}")

    def set_active_adapter(self, adapter_name: str) -> None:
        """Switch active LoRA adapter."""
        if adapter_name not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not loaded. "
                f"Available: {list(self._loaded_adapters.keys())}"
            )
        self._model.set_adapter(adapter_name)
        self._active_adapter = adapter_name

    def predict(self, vla_input: VLAInput) -> VLAOutput:
        """Run HF VLM inference with action head."""
        import torch
        from PIL import Image

        # Prepare image
        image_array = next(iter(vla_input.images.values()))
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array).convert("RGB")

        # Process inputs
        prompt = f"Robot action for: {vla_input.instruction}"
        inputs = self._processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(self._get_device()) for k, v in inputs.items()}

        # Get hidden states
        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)
            # Use last hidden state, take mean over sequence
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden = outputs.hidden_states[-1].mean(dim=1)
            elif hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state.mean(dim=1)
            else:
                hidden = outputs.logits.mean(dim=1)

            # Project to action space
            action_tensor = self._action_head(hidden.to(self._action_head.weight.dtype))
            action = action_tensor.squeeze(0).cpu().float().numpy()

        return VLAOutput(
            action=action,
            metadata={"adapter": self._active_adapter, "backend": "hf_vlm"},
        )

    def _get_hidden_dim(self) -> int:
        """Detect hidden dimension of the VLM."""
        config = self._model.config
        for attr in ["hidden_size", "d_model", "n_embd"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        # Fallback
        return 768

    def _get_device(self):
        """Get model device."""
        import torch

        if hasattr(self._model, "device"):
            return self._model.device
        return torch.device(self.config.device)
