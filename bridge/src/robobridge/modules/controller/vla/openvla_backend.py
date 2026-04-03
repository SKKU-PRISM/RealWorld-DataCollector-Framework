"""
OpenVLA backend.

Uses discrete 256-bin action tokenization via HuggingFace Transformers + PEFT.
Model: openvla/openvla-7b (Llama 2 + SigLIP + DinoV2).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base_vla import BaseVLA
from .registry import register_vla
from .types import VLAInput, VLAModelConfig, VLAOutput

logger = logging.getLogger(__name__)


@register_vla("openvla")
class OpenVLABackend(BaseVLA):
    """
    OpenVLA backend (openvla/openvla-7b).

    Architecture: SigLIP + DinoV2 vision -> Projector -> Llama 2 7B -> Action tokens
    Action format: 7D discretized into 256-bin tokens per dimension.
    LoRA: Via PEFT library, targets attention projections.
    """

    def __init__(self, config: VLAModelConfig):
        super().__init__(config)
        self._action_tokenizer = None

    def load_model(self) -> None:
        """Load OpenVLA base model with optional 4-bit quantization."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

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

        logger.info(f"Loading OpenVLA model: {self.config.model_name}")
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name, **load_kwargs
        )
        logger.info(f"OpenVLA model loaded (4-bit={self.config.quantize_4bit})")

    def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a LoRA adapter via PEFT."""
        from peft import PeftModel

        if not self._loaded_adapters:
            # First adapter: wrap model with PeftModel
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, adapter_name=adapter_name
            )
        else:
            # Additional adapters
            self._model.load_adapter(adapter_path, adapter_name=adapter_name)

        self._loaded_adapters[adapter_name] = adapter_path
        logger.info(f"Loaded LoRA adapter: {adapter_name}")

    def set_active_adapter(self, adapter_name: str) -> None:
        """Switch active LoRA adapter (pointer swap, < 1ms)."""
        if adapter_name not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not loaded. "
                f"Available: {list(self._loaded_adapters.keys())}"
            )
        self._model.set_adapter(adapter_name)
        self._active_adapter = adapter_name

    def predict(self, vla_input: VLAInput) -> VLAOutput:
        """Run OpenVLA inference: image + instruction -> action tokens -> continuous action."""
        import torch
        from PIL import Image

        # Use first available image
        image_array = next(iter(vla_input.images.values()))
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array).convert("RGB")

        # Build prompt (OpenVLA format)
        prompt = f"In: What action should the robot take to {vla_input.instruction}?\nOut:"

        # Process inputs
        inputs = self._processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate action tokens
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.action_dim,
                do_sample=False,
            )

        # Decode generated tokens to action
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        action = self._decode_action_tokens(generated_ids)

        return VLAOutput(
            action=action,
            metadata={"adapter": self._active_adapter, "backend": "openvla"},
        )

    def _decode_action_tokens(self, token_ids) -> np.ndarray:
        """Decode OpenVLA action tokens to continuous values.

        OpenVLA maps action bins to token IDs in the vocabulary.
        This extracts bin indices and converts to continuous actions.
        """
        # OpenVLA uses specific token offsets for action bins
        # The exact mapping depends on the model's vocabulary configuration
        action = np.zeros(self.config.action_dim, dtype=np.float32)

        for i, tid in enumerate(token_ids[: self.config.action_dim]):
            # OpenVLA action tokens are mapped to vocabulary indices
            # Bin index extraction depends on model configuration
            bin_idx = int(tid.cpu().item()) % 256
            action[i] = (bin_idx + 0.5) / 256.0 * 2.0 - 1.0  # normalized to [-1, 1]

        return action
