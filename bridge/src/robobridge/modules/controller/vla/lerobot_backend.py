"""
LeRobot backend for VLA models.

Supports SmolVLA, pi0.5, GROOT N1.5, and other LeRobot-compatible policies.
These models use flow matching (continuous actions) and share the LeRobot API.

Models:
    - smolvla: lerobot/smolvla_base (450M, SigLIP + SmolLM2)
    - pi0.5:   lerobot/pi05_base (2.6B, PaliGemma + Gemma)
    - groot: nvidia/GR00T-N1.5-3B (3B, SigLIP2 + DiT flow matching)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .base_vla import BaseVLA
from .registry import register_vla
from .types import VLAInput, VLAModelConfig, VLAOutput

logger = logging.getLogger(__name__)

# Map backend names to LeRobot policy types
_LEROBOT_POLICY_MAP = {
    "smolvla": "smolvla",
    "pi0.5": "pi05",
    "pi05": "pi05",
    "groot": "groot",
    "groot_n1.5": "groot",
    "groot_n15": "groot",
}


@register_vla("smolvla")
@register_vla("pi05")
@register_vla("pi0.5")
@register_vla("groot")
@register_vla("groot_n1.5")
@register_vla("lerobot")
class LeRobotBackend(BaseVLA):
    """
    LeRobot-based VLA backend.

    Supports SmolVLA, pi0.5, GROOT N1.5 via LeRobot's unified policy API.
    All these models use flow matching for continuous action prediction.
    LoRA is applied via PEFT through LeRobot's training infrastructure.
    """

    def __init__(self, config: VLAModelConfig):
        super().__init__(config)
        self._policy: Any = None
        self._policy_type: Optional[str] = None
        self._groot_preprocessor: Any = None
        self._groot_postprocessor: Any = None
        self._eagle_processor: Any = None
        self._language_tokenizer: Any = None  # SmolVLA language tokenizer
        self._chunk_buffer: Optional[np.ndarray] = None  # (T, action_dim)
        self._chunk_idx: int = 0
        self._chunk_stride: int = config.chunk_stride

    def load_model(self) -> None:
        """Load LeRobot policy model."""
        import torch

        policy_type = _LEROBOT_POLICY_MAP.get(
            self.config.backend.lower(), self.config.backend.lower()
        )
        self._policy_type = policy_type

        logger.info(f"Loading LeRobot policy: {self.config.model_name} (type={policy_type})")

        try:
            self._policy = self._load_policy(policy_type)
            self._model = self._policy  # For is_loaded check
            logger.info(f"LeRobot policy loaded: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load LeRobot policy: {e}")
            raise

    def _load_policy(self, policy_type: str) -> Any:
        """Load the appropriate LeRobot policy class."""
        import torch

        device = torch.device(self.config.device)

        if policy_type == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.configs.types import PolicyFeature, FeatureType

            policy = SmolVLAPolicy.from_pretrained(self.config.model_name)
            # Remap image features: training uses "camera1", eval provides single camera
            # Get image shape from existing visual features before removing them
            img_shape = (3, 256, 256)  # default
            for v in policy.config.input_features.values():
                if v.type == FeatureType.VISUAL:
                    img_shape = v.shape
                    break
            keys_to_remove = [k for k, v in policy.config.input_features.items()
                              if v.type == FeatureType.VISUAL]
            for key in keys_to_remove:
                del policy.config.input_features[key]
            policy.config.input_features["observation.images.camera1"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=img_shape,
            )
            policy = policy.to(device)
            # Extract tokenizer for language encoding at inference
            self._language_tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            self._tokenizer_max_length = getattr(policy.config, 'tokenizer_max_length', 48)

        elif policy_type in ("pi05", "pi0.5"):
            from transformers import AutoTokenizer

            use_tensor_state = getattr(self, '_use_tensor_state', False)
            if use_tensor_state:
                from .pi05_tensor_state import PI05TensorStatePolicy
                policy = PI05TensorStatePolicy.from_pretrained(self.config.model_name)
                logger.info("PI0.5 tensor-state mode: using PI05TensorStatePolicy")
            else:
                from lerobot.policies.pi05.modeling_pi05 import PI05Policy
                policy = PI05Policy.from_pretrained(self.config.model_name)

            policy = policy.to(device)
            # PI0.5 training used SmolVLM2 tokenizer with state-in-prompt format
            self._language_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
            self._tokenizer_max_length = 200  # Match training max_length
            # State quantile stats will be set by controller via set_state_stats()
            self._pi05_state_q01: Optional[np.ndarray] = None
            self._pi05_state_q99: Optional[np.ndarray] = None

        elif policy_type == "groot":
            # GROOT N1.5 uses LeRobot's groot policy
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

            is_full_ft = getattr(self, '_is_full_ft', False)
            if is_full_ft:
                # Full fine-tuning: load complete model from checkpoint
                policy = GrootPolicy.from_pretrained(self.config.model_name)
                logger.info(f"GROOT full-FT model loaded from {self.config.model_name}")
            else:
                # LoRA: load base model with lora_rank/alpha
                policy = GrootPolicy.from_pretrained(
                    self.config.model_name,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                )
            policy = policy.to(device)

            # GROOT needs its own pre/post processor pipeline (Eagle VLM encoding)
            self._groot_preprocessor, self._groot_postprocessor = (
                make_groot_pre_post_processors(policy.config)
            )

            # Also build standalone Eagle processor for manual batch building
            from lerobot.policies.groot.processor_groot import _build_eagle_processor
            self._eagle_processor = _build_eagle_processor()
            # Patch: ensure Eagle image processor has _prepare_image_like_inputs
            # (HuggingFace cache auto-updates can break compatibility with installed transformers)
            ip = getattr(self._eagle_processor, 'image_processor', None)
            if ip is not None and not hasattr(ip, '_prepare_image_like_inputs'):
                if hasattr(ip, '_prepare_input_images'):
                    ip._prepare_image_like_inputs = ip._prepare_input_images
                else:
                    from transformers.image_processing_utils_fast import BaseImageProcessorFast
                    import types
                    ip._prepare_image_like_inputs = types.MethodType(
                        BaseImageProcessorFast._prepare_image_like_inputs, ip
                    )

        else:
            raise ValueError(f"Unknown LeRobot policy type: {policy_type}")

        policy.eval()
        return policy

    def set_state_stats(self, state_stats: Dict) -> None:
        """Set state quantile stats for PI0.5 state-in-prompt encoding."""
        if self._policy_type not in ("pi05", "pi0.5"):
            return
        _STATE_KEEP_IDX_12D = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
        q01 = np.array(state_stats["q01"], dtype=np.float32)
        q99 = np.array(state_stats["q99"], dtype=np.float32)
        if len(q01) == 12:
            q01 = q01[_STATE_KEEP_IDX_12D]
            q99 = q99[_STATE_KEEP_IDX_12D]
        self._pi05_state_q01 = q01
        self._pi05_state_q99 = q99
        logger.info(f"PI0.5 state quantile stats set (dim={len(q01)})")

    def _discretize_state_pi05(self, state: np.ndarray) -> str:
        """Discretize state to 256-bin string for PI0.5 prompt (matches training)."""
        q01 = self._pi05_state_q01
        q99 = self._pi05_state_q99
        state_norm = (state - q01) / (q99 - q01 + 1e-8) * 2 - 1
        state_norm = np.clip(state_norm, -1, 1)
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm
        bins = np.linspace(-1, 1, 257)[:-1]
        discretized = np.digitize(state_padded, bins) - 1
        discretized = np.clip(discretized, 0, 255)
        return " ".join(map(str, discretized))

    def _normalize_state_tensor_pi05(self, state: np.ndarray) -> np.ndarray:
        """Quantile-normalize state to [-1,1] and pad to 32D for tensor-state mode.

        Matches PI05TensorStateChunkDataset preprocessing in training.
        """
        _STATE_KEEP_IDX_12D = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
        q01 = self._pi05_state_q01
        q99 = self._pi05_state_q99

        # Drop gripper indices if 12D input
        if len(state) == 12:
            state = state[_STATE_KEEP_IDX_12D]

        state_norm = (state - q01) / (q99 - q01 + 1e-8) * 2 - 1
        state_norm = np.clip(state_norm, -1, 1)

        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm
        return state_padded

    def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load an adapter into the LeRobot policy.

        Auto-detects adapter format:
        - adapter_config.json present → PEFT LoRA adapter
        - model_diff.safetensors present → weight-diff adapter (e.g. GROOT)
        """
        # Full fine-tuning: model already loaded, no adapter needed
        if getattr(self, '_is_full_ft', False):
            self._loaded_adapters[adapter_name] = adapter_path
            logger.info(f"Full-FT mode: registered adapter '{adapter_name}' (no-op)")
            return

        peft_config = os.path.join(adapter_path, "adapter_config.json")
        diff_path = os.path.join(adapter_path, "model_diff.safetensors")

        internal_config = os.path.join(adapter_path, "config.json")
        internal_weights = os.path.join(adapter_path, "model.safetensors")

        if os.path.exists(peft_config):
            self._load_peft_adapter(adapter_path, adapter_name)
        elif os.path.exists(diff_path):
            self._load_weight_diff_adapter(diff_path, adapter_name)
        elif os.path.exists(internal_weights):
            # Internal LoRA format (e.g. GROOT): config.json + model.safetensors
            self._load_weight_diff_adapter(internal_weights, adapter_name)
        else:
            raise FileNotFoundError(
                f"No adapter found at '{adapter_path}'. "
                f"Expected 'adapter_config.json' (PEFT), 'model_diff.safetensors' (weight-diff), "
                f"or 'model.safetensors' (internal LoRA)."
            )

    def _load_peft_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a PEFT LoRA adapter onto the underlying language model."""
        from peft import PeftModel

        base_model = self._get_lora_target_model()

        if not self._loaded_adapters:
            wrapped = PeftModel.from_pretrained(
                base_model, adapter_path, adapter_name=adapter_name
            )
            self._set_lora_target_model(wrapped)
        else:
            lora_model = self._get_lora_target_model()
            lora_model.load_adapter(adapter_path, adapter_name=adapter_name)

        self._loaded_adapters[adapter_name] = adapter_path
        logger.info(f"Loaded PEFT adapter: {adapter_name}")

    def _load_weight_diff_adapter(self, diff_path: str, adapter_name: str) -> None:
        """Load a weight-diff adapter (trained weight subset, e.g. GROOT action head)."""
        from safetensors.torch import load_file

        if not hasattr(self, "_weight_diffs"):
            self._weight_diffs = {}
        self._weight_diffs[adapter_name] = load_file(diff_path)
        self._loaded_adapters[adapter_name] = diff_path
        logger.info(
            f"Loaded weight-diff adapter: {adapter_name} "
            f"({len(self._weight_diffs[adapter_name])} keys)"
        )

    def set_active_adapter(self, adapter_name: str) -> None:
        """Switch active adapter (PEFT or weight-diff)."""
        if adapter_name not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not loaded. "
                f"Available: {list(self._loaded_adapters.keys())}"
            )

        # Full fine-tuning: same model for all adapters, just track the name
        if getattr(self, '_is_full_ft', False):
            self._active_adapter = adapter_name
            return

        # Weight-diff mode: apply trained weights directly
        if hasattr(self, "_weight_diffs") and adapter_name in self._weight_diffs:
            self._policy.load_state_dict(self._weight_diffs[adapter_name], strict=False)
            self._active_adapter = adapter_name
            if hasattr(self._policy, "reset"):
                self._policy.reset()
            self._chunk_buffer = None
            self._chunk_idx = 0
            return

        # PEFT mode
        lora_model = self._get_lora_target_model()
        if hasattr(lora_model, "set_adapter"):
            lora_model.set_adapter(adapter_name)
        self._active_adapter = adapter_name
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._chunk_buffer = None
        self._chunk_idx = 0

    def reset_policy(self) -> None:
        """Reset policy state (clear action queue and chunk buffer). Call at episode start."""
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
        self._chunk_buffer = None
        self._chunk_idx = 0

    def predict(self, vla_input: VLAInput) -> VLAOutput:
        """Run LeRobot policy inference with chunk stride management.

        Instead of using select_action (which caches 16 actions and ignores
        new observations), we manage the action chunk locally with chunk_stride=8
        to match direct mode behavior.
        """
        import torch

        # Build fresh batch from current observation
        if self._groot_preprocessor is not None:
            batch = self._build_groot_batch(vla_input)
        else:
            batch = self._build_observation(vla_input)

        # Check if we need new inference
        need_inference = (
            self._chunk_buffer is None
            or self._chunk_idx >= self._chunk_stride
        )

        if need_inference:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    actions = self._policy.predict_action_chunk(batch)
            # actions shape: (B, T, action_dim) where B=1, T=16
            self._chunk_buffer = actions[0].float().cpu().numpy()  # (T, action_dim)
            self._chunk_idx = 0
            # Clear the internal action queue to prevent stale actions
            if hasattr(self._policy, 'reset'):
                self._policy.reset()

        # Get current action from buffer
        action_np = self._chunk_buffer[self._chunk_idx].copy()
        self._chunk_idx += 1

        # Ensure correct action dimension
        if len(action_np) > self.config.action_dim:
            action_np = action_np[:self.config.action_dim]
        elif len(action_np) < self.config.action_dim:
            padded = np.zeros(self.config.action_dim, dtype=np.float32)
            padded[:len(action_np)] = action_np
            action_np = padded

        return VLAOutput(
            action=action_np,
            metadata={
                "adapter": self._active_adapter,
                "backend": f"lerobot/{self._policy_type}",
                "chunk_idx": self._chunk_idx - 1,
            },
        )

    def predict_full_chunk(self, vla_input: VLAInput, action_dim: int = 9) -> np.ndarray:
        """Run inference and return the FULL action chunk (no stride management).

        Used by MBR decoding which needs multiple independent samples.
        Does NOT update the internal chunk buffer.

        Args:
            vla_input: Model input.
            action_dim: Number of action dimensions to return.

        Returns:
            (T, action_dim) numpy array of the full predicted chunk.
        """
        import torch

        if self._groot_preprocessor is not None:
            batch = self._build_groot_batch(vla_input)
        else:
            batch = self._build_observation(vla_input)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actions = self._policy.predict_action_chunk(batch)
        # actions shape: (B, T, D) where B=1, T=16, D=32 (padded)
        chunk = actions[0].float().cpu().numpy()  # (T, D)
        # Clear internal action queue to prevent stale state
        if hasattr(self._policy, 'reset'):
            self._policy.reset()
        return chunk[:, :action_dim]

    def _build_groot_batch(self, vla_input: VLAInput) -> Dict:
        """Build GROOT input batch.

        If state_mask is present (pre-normalized state from controller),
        builds batch manually with Eagle image encoding.
        Otherwise falls back to the preprocessor pipeline.
        """
        if vla_input.state_mask is not None and self._eagle_processor is not None:
            return self._build_groot_batch_manual(vla_input)

        # Fallback: use preprocessor pipeline (raw state)
        import torch

        batch = {}
        if not vla_input.images:
            logger.warning("No images in VLA input — GROOT requires at least one camera image")
        for cam_name, image in vla_input.images.items():
            if image is None:
                continue
            img = image.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            batch[f"observation.images.{cam_name}"] = img_tensor

        state_tensor = torch.from_numpy(
            vla_input.robot_state.astype(np.float32)
        ).unsqueeze(0)
        batch["observation.state"] = state_tensor
        batch["task"] = vla_input.instruction

        return self._groot_preprocessor(batch)

    def _build_groot_batch_manual(self, vla_input: VLAInput) -> Dict:
        """Build GROOT batch with pre-normalized state + Eagle image encoding.

        Bypasses the preprocessor for state (already normalized+padded to 64D)
        but uses Eagle processor for image encoding.
        """
        import torch
        from PIL import Image as PILImage

        device = torch.device(self.config.device)

        # State: already normalized and padded to 64D
        state_padded = vla_input.robot_state.astype(np.float32)
        state_mask = vla_input.state_mask

        batch = {
            "state": torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0).to(device),
            "state_mask": torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0).to(device),
            "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
        }

        # Eagle image encoding
        if vla_input.images:
            image = next(iter(vla_input.images.values()))
            if image is not None:
                pil_img = PILImage.fromarray(image)
                eagle_inputs = self._eagle_processor(
                    text=[f"<image-1> {vla_input.instruction}"],
                    images=[pil_img],
                    images_kwargs={
                        "min_dynamic_tiles": 1,
                        "max_dynamic_tiles": 1,
                        "use_thumbnail": False,
                    },
                    return_tensors="pt",
                    padding=True,
                )
                for k, v in eagle_inputs.items():
                    batch[f"eagle_{k}"] = v.to(device)

        return batch

    def _build_observation(self, vla_input: VLAInput) -> Dict:
        """Convert VLAInput to LeRobot observation format."""
        import torch

        obs = {}

        # Images: LeRobot expects (C, H, W) float tensors normalized to [0, 1]
        # SmolVLA uses "camera1", pi0.5 uses "base_0_rgb", GROOT uses original cam names
        cam_remap = {"smolvla": "camera1", "pi05": "base_0_rgb", "pi0.5": "base_0_rgb"}
        for cam_name, image in vla_input.images.items():
            if image is None:
                continue
            img = image.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            out_name = cam_remap.get(self._policy_type, cam_name)
            obs[f"observation.images.{out_name}"] = img_tensor.to(self.config.device)

        # PI0.5: state conditioning mode
        is_pi05 = self._policy_type in ("pi05", "pi0.5")
        use_tensor_state = getattr(self, '_use_tensor_state', False)
        if is_pi05 and use_tensor_state and self._pi05_state_q01 is not None:
            # Tensor-state mode: no state in prompt, state as normalized 32D tensor
            prompt = f"Task: {vla_input.instruction};\nAction: "
            tokens = self._language_tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            obs["observation.language.tokens"] = tokens["input_ids"].to(self.config.device)
            obs["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(self.config.device)
        elif is_pi05 and self._pi05_state_q01 is not None:
            # Text-discretized state mode (original)
            state_str = self._discretize_state_pi05(vla_input.robot_state)
            prompt = f"Task: {vla_input.instruction}, State: {state_str};\nAction: "
            tokens = self._language_tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            obs["observation.language.tokens"] = tokens["input_ids"].to(self.config.device)
            obs["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(self.config.device)
        elif self._language_tokenizer is not None:
            # SmolVLA: pre-tokenize language instruction
            tokens = self._language_tokenizer(
                vla_input.instruction,
                padding="max_length",
                max_length=self._tokenizer_max_length,
                return_tensors="pt",
            )
            obs["observation.language.tokens"] = tokens["input_ids"].to(self.config.device)
            obs["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(self.config.device)
        else:
            obs["task"] = vla_input.instruction

        # Robot state tensor
        if is_pi05 and use_tensor_state:
            # PI0.5 tensor-state mode: quantile-normalize state to [-1,1], pad to 32D
            state_tensor = self._normalize_state_tensor_pi05(vla_input.robot_state)
            obs["observation.state"] = torch.from_numpy(state_tensor).unsqueeze(0).to(self.config.device)
        elif is_pi05:
            # PI0.5 text mode: state info is in text prompt; pass zeros
            state_zeros = np.zeros(10, dtype=np.float32)  # 10D matching training
            obs["observation.state"] = torch.from_numpy(state_zeros).unsqueeze(0).to(self.config.device)
        else:
            state_tensor = torch.from_numpy(
                vla_input.robot_state.astype(np.float32)
            ).unsqueeze(0)
            obs["observation.state"] = state_tensor.to(self.config.device)

        return obs

    def _get_lora_target_model(self) -> Any:
        """Get the sub-model that LoRA targets.

        SmolVLA/pi0.5: PEFT wraps full policy (target_modules start with 'model.')
        GROOT: internal LoRA on policy.model (weight-diff, not PEFT)
        """
        # SmolVLA/pi0.5: PEFT was applied to the full policy during training
        if self._policy_type in ("smolvla", "pi05", "pi0.5"):
            return self._policy
        # GROOT: weight-diff on policy.model
        if hasattr(self._policy, "model"):
            return self._policy.model
        return self._policy

    def _set_lora_target_model(self, wrapped_model: Any) -> None:
        """Replace the sub-model with LoRA-wrapped version."""
        if self._policy_type in ("smolvla", "pi05", "pi0.5"):
            self._policy = wrapped_model
        elif hasattr(self._policy, "model"):
            self._policy.model = wrapped_model
