#!/usr/bin/env python3
"""
RoboCasa evaluation script for trained multitask VLA LoRA adapters.

Evaluates PI0.5 and GROOT N1.5 models on RoboCasa single-stage tasks.
Works with the multitask adapter structure in ./outputs/vla_adapters/.

Usage:
    # Evaluate PI0.5 on a single task
    python eval_robocasa.py --model pi05 --tasks CloseDrawer --num-episodes 10

    # Evaluate GROOT on multiple tasks
    python eval_robocasa.py --model groot --tasks CloseDrawer TurnOnMicrowave --num-episodes 25

    # Evaluate specific checkpoint (not best)
    python eval_robocasa.py --model groot --checkpoint 15000 --tasks CloseDrawer

    # Dry run (test model loading + env creation only)
    python eval_robocasa.py --model pi05 --tasks CloseDrawer --dry-run

    # Save rollout videos
    python eval_robocasa.py --model pi05 --tasks CloseDrawer --num-episodes 5 --save-video

    # List all available tasks
    python eval_robocasa.py --list-tasks
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

for _mod in ("robosuite", "OpenGL", "PIL", "urllib3", "transformers", "huggingface_hub", "peft"):
    logging.getLogger(_mod).setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Fetching.*files.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ADAPTER_BASE = Path(os.environ.get("ADAPTER_BASE", "./outputs/vla_adapters"))
METADATA_DIR = Path(os.environ.get("METADATA_DIR", "./bridge/multitask_training_package/data"))

# ---------------------------------------------------------------------------
# Task instructions (all 25 single-stage RoboCasa tasks)
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    # Doors
    "CloseSingleDoor": "close the door",
    "OpenSingleDoor": "open the door",
    "CloseDoubleDoor": "close the double door",
    "OpenDoubleDoor": "open the double door",
    # Drawers
    "CloseDrawer": "close the drawer",
    "OpenDrawer": "open the drawer",
    # Coffee
    "StartCoffeeMachine": "press the coffee machine button",
    "CoffeeServeMug": "serve the coffee mug",
    "CoffeeSetupMug": "set up the coffee mug",
    # Pick and place (v1.0 names)
    "PickPlaceCabinetToCounter": "pick the object from the cabinet and place it on the counter",
    "PickPlaceCounterToCabinet": "pick the object from the counter and place it in the cabinet",
    "PickPlaceCounterToMicrowave": "pick the object from the counter and place it in the microwave",
    "PickPlaceCounterToSink": "pick the object from the counter and place it in the sink",
    "PickPlaceCounterToStove": "pick the object from the counter and place it on the stove",
    "PickPlaceMicrowaveToCounter": "pick the object from the microwave and place it on the counter",
    "PickPlaceSinkToCounter": "pick the object from the sink and place it on the counter",
    "PickPlaceStoveToCounter": "pick the object from the stove and place it on the counter",
    # Microwave
    "TurnOffMicrowave": "turn off the microwave",
    "TurnOnMicrowave": "turn on the microwave",
    # Sink
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "TurnOnSinkFaucet": "turn on the sink faucet",
    "TurnSinkSpout": "turn the sink spout",
    # Stove
    "TurnOffStove": "turn off the stove",
    "TurnOnStove": "turn on the stove",
}

# Legacy name aliases (v0.1 -> v1.0) for backwards compatibility
_TASK_ALIASES = {
    "CoffeePressButton": "StartCoffeeMachine",
    "PnPCabToCounter": "PickPlaceCabinetToCounter",
    "PnPCounterToCab": "PickPlaceCounterToCabinet",
    "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
    "PnPCounterToSink": "PickPlaceCounterToSink",
    "PnPCounterToStove": "PickPlaceCounterToStove",
    "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
    "PnPSinkToCounter": "PickPlaceSinkToCounter",
    "PnPStoveToCounter": "PickPlaceStoveToCounter",
}

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "openvla": {
        "base_model": "openvla/openvla-7b",
        "adapter_dir": ADAPTER_BASE / "openvla" / "general" / "lora_adapter",
        "image_size": 224,
        "chunk_size": 1,
        "chunk_stride": 1,
        "norm_type": "mean_std",
    },
    "pi05": {
        "base_model": "lerobot/pi05_base",
        "adapter_dir": ADAPTER_BASE / "pi0.5" / "general" / "lora_adapter",
        "image_size": 224,
        "chunk_size": 50,
        "chunk_stride": 10,
        "norm_type": "quantile",
    },
    "groot": {
        "base_model": "nvidia/GR00T-N1.5-3B",
        "adapter_dir": ADAPTER_BASE / "groot",
        "image_size": 224,
        "chunk_size": 16,
        "chunk_stride": 8,
        "norm_type": "min_max",
    },
    "smolvla": {
        "base_model": "lerobot/smolvla_base",
        "adapter_dir": ADAPTER_BASE / "smolvla" / "general" / "lora_adapter",
        "image_size": 512,
        "chunk_size": 50,
        "chunk_stride": 10,
        "norm_type": "mean_std",
    },
}


# ---------------------------------------------------------------------------
# Statistics loading
# ---------------------------------------------------------------------------


def load_normalization_stats(stats_path: Optional[str] = None, adapter_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load action/state normalization statistics.

    Resolution order:
    1. Explicit ``stats_path`` (JSON file)
    2. ``data_stats.json`` next to the adapter directory
    3. Global metadata files at METADATA_DIR
    """
    # --- Try explicit path or adapter-local data_stats.json ---
    candidates = []
    if stats_path:
        candidates.append(Path(stats_path))
    if adapter_dir:
        # adapter_dir may point to checkpoint-best; also check parent
        candidates.append(Path(adapter_dir) / "data_stats.json")
        candidates.append(Path(adapter_dir).parent / "data_stats.json")

    for p in candidates:
        if p.exists():
            logger.info(f"Loading normalization stats from {p}")
            with open(p) as f:
                ds = json.load(f)
            action_stats = dict(ds["action_stats"])
            # Merge quantile stats if available
            aq = ds.get("action_quantile_stats", {})
            if "q01" in aq:
                action_stats["q01"] = aq["q01"]
            if "q99" in aq:
                action_stats["q99"] = aq["q99"]
            state_stats = ds.get("state_stats", {})
            # Truncate state_stats to 16D if needed
            RAW_STATE_DIM = 16
            state_stats_raw = {}
            for k, v in state_stats.items():
                state_stats_raw[k] = v[:RAW_STATE_DIM] if len(v) > RAW_STATE_DIM else v
            return {"action_stats": action_stats, "state_stats": state_stats_raw}

    # --- Fallback: global metadata files ---
    meta_path = METADATA_DIR / "metadata.json"
    meta_ext_path = METADATA_DIR / "metadata_extended.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No normalization stats found. Tried: {[str(c) for c in candidates]} and {meta_path}. "
            f"Use --stats-path to provide a data_stats.json file."
        )

    with open(meta_path) as f:
        meta = json.load(f)
    with open(meta_ext_path) as f:
        meta_ext = json.load(f)

    # state_stats in metadata_extended.json are 19D (preprocessed: 16D raw + 3D target_pos).
    # Raw RoboCasa obs gives 16D. The first 16 elements align, so truncate to 16D.
    RAW_STATE_DIM = 16
    state_stats_raw = {}
    for k, v in meta_ext["state_stats"].items():
        state_stats_raw[k] = v[:RAW_STATE_DIM]

    return {
        "action_stats": {
            **meta["action_stats"],
            "q01": meta_ext["action_quantile_stats"]["q01"],
            "q99": meta_ext["action_quantile_stats"]["q99"],
        },
        "state_stats": state_stats_raw,
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_state_mean_std(state, mean, std):
    return (state - mean) / (std + 1e-8)


def normalize_state_min_max(state, vmin, vmax):
    r = vmax - vmin + 1e-8
    return np.clip(2.0 * (state - vmin) / r - 1.0, -1.0, 1.0)


def denormalize_action_quantile(action, q01, q99):
    return (action + 1.0) / 2.0 * (q99 - q01) + q01


def denormalize_action_min_max(action, vmin, vmax):
    return (action + 1.0) / 2.0 * (vmax - vmin) + vmin


def denormalize_action_mean_std(action, mean, std):
    return action * std + mean


def denormalize_action(action_7d, action_stats, norm_type):
    """Denormalize an action vector (6D or 7D)."""
    a = action_7d[:7].copy()
    dim = len(a)
    if norm_type == "quantile":
        q01 = np.array(action_stats["q01"][:dim], dtype=np.float32)
        q99 = np.array(action_stats["q99"][:dim], dtype=np.float32)
        denorm = denormalize_action_quantile(a, q01, q99)
    elif norm_type == "min_max":
        vmin = np.array(action_stats["min"][:dim], dtype=np.float32)
        vmax = np.array(action_stats["max"][:dim], dtype=np.float32)
        denorm = denormalize_action_min_max(a, vmin, vmax)
    elif norm_type == "mean_std":
        mean = np.array(action_stats["mean"][:dim], dtype=np.float32)
        std = np.array(action_stats["std"][:dim], dtype=np.float32)
        denorm = denormalize_action_mean_std(a, mean, std)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    # Pad to 7D if model output is shorter (e.g. 6D arm-only, no gripper)
    if dim < 7:
        padded = np.zeros(7, dtype=np.float32)
        padded[:dim] = denorm
        padded[6] = 1.0  # default gripper open
        return padded
    return denorm


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------


def extract_state_16d(obs):
    """Extract 16D state: eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper(2)."""
    return np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_joint_pos"],
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)


def extract_image(obs, camera="robot0_agentview_left", size=224):
    """Extract and resize camera image. Returns (H, W, 3) uint8."""
    from PIL import Image as PILImage
    img = obs[f"{camera}_image"]
    if img.shape[0] != size or img.shape[1] != size:
        img = np.array(PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR))
    return img


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


# Door tasks require fixture_id in robocasa v1.0
_DOOR_FIXTURE_MAP = {
    "OpenSingleDoor": ("OpenDoor", "CABINET_SINGLE_DOOR"),
    "CloseSingleDoor": ("CloseDoor", "CABINET_SINGLE_DOOR"),
    "OpenDoubleDoor": ("OpenDoor", "CABINET_WITH_DOOR"),
    "CloseDoubleDoor": ("CloseDoor", "CABINET_WITH_DOOR"),
}


def create_env(task_name, image_size=224):
    """Create a RoboCasa environment."""
    import robocasa
    from robocasa.environments.kitchen.kitchen import FixtureType

    kwargs = dict(
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_left"],
        camera_heights=image_size,
        camera_widths=image_size,
        reward_shaping=False,
        ignore_done=True,
        control_freq=20,
    )

    if task_name in _DOOR_FIXTURE_MAP:
        env_name, fixture_attr = _DOOR_FIXTURE_MAP[task_name]
        kwargs["fixture_id"] = getattr(FixtureType, fixture_attr)
        return robocasa.make(env_name=env_name, **kwargs)

    return robocasa.make(env_name=task_name, **kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pi05(adapter_path, device):
    """Load PI0.5 with PEFT LoRA adapter."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel
    from transformers import AutoTokenizer

    logger.info(f"Loading PI0.5 base model...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    policy = policy.to(dtype=torch.bfloat16)

    # Override to single camera
    keys_to_remove = [
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL
    ]
    for key in keys_to_remove:
        del policy.config.input_features[key]
    policy.config.input_features["observation.images.base_0_rgb"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, *policy.config.image_resolution),
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(policy, adapter_path)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b", padding_side="right")
    return model, tokenizer


def load_groot(adapter_path, device):
    """Load GROOT N1.5 with LoRA weights.

    Auto-detects checkpoint format:
    - PEFT adapter: adapter_config.json + adapter_model.safetensors
    - Internal LoRA: config.json + model.safetensors (legacy)
    """
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    is_peft = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

    if is_peft:
        from peft import PeftModel

        logger.info(f"Loading GROOT N1.5 base for PEFT adapter...")
        policy = GrootPolicy.from_pretrained("nvidia/GR00T-N1.5-3B")
        policy = policy.to(dtype=torch.bfloat16)

        logger.info(f"Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(policy, adapter_path)
        model = model.to(device)
        model.eval()
    else:
        from safetensors.torch import load_file

        # Legacy: internal LoRA checkpoint
        config_path = os.path.join(adapter_path, "config.json")
        lora_rank, lora_alpha = 64, 128  # defaults from training
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            lora_rank = cfg.get("lora_rank", lora_rank)
            lora_alpha = cfg.get("lora_alpha", lora_alpha)

        logger.info(f"Loading GROOT N1.5 base (lora_rank={lora_rank}, lora_alpha={lora_alpha})...")
        policy = GrootPolicy.from_pretrained(
            "nvidia/GR00T-N1.5-3B",
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        weights_path = os.path.join(adapter_path, "model.safetensors")
        if os.path.exists(weights_path):
            logger.info(f"Loading GROOT weights from {weights_path}")
            adapter_weights = load_file(weights_path)
            missing, unexpected = policy.load_state_dict(adapter_weights, strict=False)
            logger.info(f"Loaded {len(adapter_weights)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            logger.warning(f"No model.safetensors at {adapter_path}, using base model")

        model = policy.to(dtype=torch.bfloat16, device=device)
        model.eval()

    from lerobot.policies.groot.processor_groot import _build_eagle_processor
    eagle_processor = _build_eagle_processor()
    return model, eagle_processor


def load_smolvla(adapter_path, device):
    """Load SmolVLA with PEFT LoRA adapter."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel
    from transformers import AutoTokenizer

    logger.info("Loading SmolVLA base model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(dtype=torch.bfloat16)

    # Override to single camera
    keys_to_remove = [
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL
    ]
    for key in keys_to_remove:
        del policy.config.input_features[key]
    policy.config.input_features["observation.images.camera1"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 512, 512),
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(policy, adapter_path)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
    return model, tokenizer


def load_openvla(adapter_path, device):
    """Load OpenVLA with PEFT LoRA adapter."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    import timm as _timm

    base_model_name = "openvla/openvla-7b"
    logger.info(f"Loading OpenVLA base model: {base_model_name}")

    # Monkey-patch timm version to bypass overly strict check in cached modeling_prismatic.py
    # (requires 0.9.x but timm 1.0.x is backwards-compatible for the SigLIP backbone)
    _real_timm_version = _timm.__version__
    _timm.__version__ = "0.9.16"

    try:
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        try:
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForVision2Seq.from_pretrained(base_model_name, **load_kwargs)
        except (AttributeError, ValueError, TypeError, NotImplementedError):
            logger.warning("SDPA attention failed, falling back to eager")
            load_kwargs["attn_implementation"] = "eager"
            model = AutoModelForVision2Seq.from_pretrained(base_model_name, **load_kwargs)
    finally:
        _timm.__version__ = _real_timm_version

    # Fix timm 1.0.x compatibility: get_intermediate_layers returns list, not tuple.
    # The cached unpack_tuple only checks isinstance(result, tuple), so lists pass through.
    def _patch_featurizer_forward(featurizer):
        _orig_forward = featurizer.forward
        def _patched(*args, **kwargs):
            result = _orig_forward(*args, **kwargs)
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            return result
        featurizer.forward = _patched

    vb = model.vision_backbone
    _patch_featurizer_forward(vb.featurizer)
    if hasattr(vb, "fused_featurizer"):
        _patch_featurizer_forward(vb.fused_featurizer)

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()

    return model, processor


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


def prepare_pi05_batch(image, state_16d, instruction, state_stats, action_stats, tokenizer, device):
    """Prepare PI0.5 input batch."""
    # Image -> (1, 3, 224, 224) float [0,1]
    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

    # Discretize state for prompt
    q01 = np.array(state_stats["q01"], dtype=np.float32)
    q99 = np.array(state_stats["q99"], dtype=np.float32)
    state_norm = np.clip((state_16d - q01) / (q99 - q01 + 1e-8) * 2 - 1, -1, 1)
    state_padded = np.zeros(32, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.clip(np.digitize(state_padded, bins) - 1, 0, 255)
    state_str = " ".join(map(str, discretized))

    prompt = f"Task: {instruction}, State: {state_str};\nAction: "
    tokens = tokenizer(prompt, max_length=200, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "observation.images.base_0_rgb": img_t.to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }


def prepare_groot_batch(image, state_16d, instruction, state_stats, eagle_processor, device):
    """Prepare GROOT N1.5 input batch."""
    from PIL import Image as PILImage

    # State: min_max normalize, pad to 64
    s_min = np.array(state_stats["min"], dtype=np.float32)
    s_max = np.array(state_stats["max"], dtype=np.float32)
    state_norm = normalize_state_min_max(state_16d, s_min, s_max)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm
    state_mask = np.zeros(64, dtype=bool)
    state_mask[:len(state_norm)] = True

    batch = {
        "state": torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0).to(device),
        "state_mask": torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0).to(device),
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),  # new_embodiment
    }

    if eagle_processor is not None:
        # Image is already uint8 [0,255] - pass directly to eagle_processor
        pil_img = PILImage.fromarray(image)
        eagle_inputs = eagle_processor(
            text=[f"<image-1> {instruction}"],
            images=[pil_img],
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )
        for k, v in eagle_inputs.items():
            batch[f"eagle_{k}"] = v.to(device)

    return batch


def prepare_smolvla_batch(image, state_16d, instruction, state_stats, tokenizer, device):
    """Prepare SmolVLA input batch."""
    from PIL import Image as PILImage
    import torch.nn.functional as F

    # Image -> resize to 512x512, float [0,1]
    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    if img_t.shape[-1] != 512 or img_t.shape[-2] != 512:
        img_t = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)

    # State: mean_std normalize
    s_mean = np.array(state_stats["mean"], dtype=np.float32)
    s_std = np.array(state_stats["std"], dtype=np.float32)
    state_norm = normalize_state_mean_std(state_16d, s_mean, s_std)

    # Tokenize instruction (SmolVLA expects trailing newline)
    text = instruction + "\n"
    tokens = tokenizer(text, max_length=48, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "observation.images.camera1": img_t.to(device),
        "observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }


def prepare_openvla_batch(image, instruction, processor, device):
    """Prepare OpenVLA input batch."""
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(image)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Use full processor to get correctly formatted pixel_values (dual backbone: 6ch)
    inputs = processor(prompt, pil_img)
    return {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def stack_batches(batch_list):
    """Stack list of single-sample batch dicts into one batched dict."""
    if len(batch_list) == 1:
        return batch_list[0]
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in batch_list[0]}


@torch.no_grad()
def predict_action_chunk(model, batch, **kwargs):
    """Run inference, return (batch_size, chunk_size, action_dim) numpy."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch, **kwargs)
    return actions.float().cpu().numpy()


@torch.no_grad()
def predict_openvla_actions(model, batch):
    """Generate action tokens from OpenVLA and decode to normalized actions.

    Uses KV-cache: first forward pass through the full model (vision + LLM),
    then 6 subsequent tokens via language_model only with past_key_values.
    ~5x faster than re-running the full model 7 times.

    Returns (batch_size, 1, 7) numpy array with values in [-1, 1] (z-score normalized).
    """
    # Get the underlying Prismatic model through PEFT layers
    m = model
    if hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "model"):
        m = m.model
    bin_centers = m.bin_centers
    vocab_size = m.vocab_size
    if isinstance(bin_centers, torch.Tensor):
        bin_centers_np = bin_centers.float().cpu().numpy()
    else:
        bin_centers_np = np.array(bin_centers, dtype=np.float32)

    input_ids = batch["input_ids"]
    pixel_values = batch["pixel_values"]
    attention_mask = batch["attention_mask"]

    generated_tokens = []

    # Step 1: Full forward (vision backbone + projector + LLM) with use_cache=True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    generated_tokens.append(next_token.cpu())
    past_kv = outputs.past_key_values

    # Steps 2-7: LLM-only forward with KV cache (skip vision backbone entirely)
    embed_fn = m.get_input_embeddings()
    for _ in range(6):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            new_embeds = embed_fn(next_token.unsqueeze(-1))
            outputs = m.language_model(
                input_ids=None,
                inputs_embeds=new_embeds,
                past_key_values=past_kv,
                use_cache=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_tokens.append(next_token.cpu())
        past_kv = outputs.past_key_values

    # Decode action tokens -> normalized actions
    batch_size = input_ids.shape[0]
    actions = []
    for i in range(batch_size):
        token_ids = torch.stack([t[i] for t in generated_tokens]).numpy()
        bin_indices = vocab_size - 1 - token_ids
        bin_indices = np.clip(bin_indices, 0, len(bin_centers_np) - 1)
        action_norm = bin_centers_np[bin_indices]
        actions.append(action_norm)

    # (batch_size, 1, 7) — single-step "chunk" for compatibility
    return np.array(actions, dtype=np.float32)[:, np.newaxis, :]


# ---------------------------------------------------------------------------
# 7D action -> 12D RoboCasa command
# ---------------------------------------------------------------------------


def action_7d_to_12d(action_7d):
    """Convert denormalized 7D action to 12D RoboCasa action."""
    action_12d = np.zeros(12, dtype=np.float32)
    action_12d[0:6] = np.clip(action_7d[:6], -1.0, 1.0)
    action_12d[10:12] = np.clip(action_7d[6], -1.0, 1.0)
    return action_12d


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------


def save_video(frames, path, fps=20):
    """Save list of (H,W,3) uint8 frames as mp4."""
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        logger.info(f"Video saved: {path}")
    except ImportError:
        logger.warning("imageio not installed, skipping video save")


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------


def run_episodes_batched(
    model,
    model_name: str,
    task_name: str,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    norm_type: str,
    image_size: int,
    chunk_stride: int,
    max_horizon: int,
    device: torch.device,
    batch_size: int = 1,
    tokenizer=None,
    eagle_processor=None,
    record_video: bool = False,
    ep_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Run batch_size episodes in parallel. Returns list of result dicts."""
    n = batch_size

    # Create N environments
    envs = []
    for i in range(n):
        try:
            envs.append(create_env(task_name, image_size=image_size))
        except Exception as e:
            logger.error(f"Failed to create env #{i}: {e}")
            for env in envs:
                env.close()
            return [{"success": False, "steps": 0, "total_reward": 0.0}] * n

    # Reset envs with retry (some RoboCasa fixture seeds produce None fixtures)
    obs_list = []
    for i, env in enumerate(envs):
        obs = None
        for attempt in range(5):
            try:
                obs = env.reset()
                break
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Env #{i} reset failed (attempt {attempt+1}/5): {e}")
                try:
                    env.close()
                except Exception:
                    pass
                envs[i] = create_env(task_name, image_size=image_size)
                env = envs[i]
        if obs is None:
            logger.error(f"Env #{i} reset failed after 5 attempts, returning failures")
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            return [{"success": False, "steps": 0, "total_reward": 0.0}] * n
        obs_list.append(obs)

    # Per-env state
    action_chunks = [None] * n
    chunk_idxs = [0] * n
    active = [True] * n
    ep_successes = [False] * n
    ep_steps = [0] * n
    ep_rewards = [0.0] * n
    ep_frames = [[] for _ in range(n)]
    prev_states = [None] * n
    no_change_counts = [0] * n

    _batch_start = time.time()
    for step in range(max_horizon):
        if not any(active):
            break

        # Status line
        n_active = sum(active)
        n_ok = sum(ep_successes)
        elapsed = time.time() - _batch_start
        print(
            f"\r    step {step+1:>3d}/{max_horizon} | "
            f"active: {n_active}/{n} | "
            f"success: {n_ok} | "
            f"({elapsed:.0f}s)",
            end="", flush=True,
        )

        # Find envs needing new action chunks
        needs_chunk = [
            i for i in range(n)
            if active[i] and (action_chunks[i] is None or chunk_idxs[i] >= chunk_stride)
        ]

        if needs_chunk:
            batch_dicts = []
            valid_indices = []
            for i in needs_chunk:
                image = extract_image(obs_list[i], size=image_size)
                state_16d = extract_state_16d(obs_list[i])
                try:
                    if model_name == "openvla":
                        bd = prepare_openvla_batch(
                            image, instruction, tokenizer, device
                        )
                    elif model_name == "pi05":
                        bd = prepare_pi05_batch(
                            image, state_16d, instruction, state_stats, action_stats, tokenizer, device
                        )
                    elif model_name == "groot":
                        bd = prepare_groot_batch(
                            image, state_16d, instruction, state_stats, eagle_processor, device
                        )
                    elif model_name == "smolvla":
                        bd = prepare_smolvla_batch(
                            image, state_16d, instruction, state_stats, tokenizer, device
                        )
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                    batch_dicts.append(bd)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Batch prep error env {i} step {step}: {e}")
                    active[i] = False

            if batch_dicts:
                try:
                    stacked = stack_batches(batch_dicts)
                    if model_name == "openvla":
                        all_chunks = predict_openvla_actions(model, stacked)
                    else:
                        kwargs = {"num_steps": 10} if model_name in ("pi05", "smolvla") else {}
                        all_chunks = predict_action_chunk(model, stacked, **kwargs)
                    for j, idx in enumerate(valid_indices):
                        action_chunks[idx] = all_chunks[j]
                        chunk_idxs[idx] = 0
                except Exception as e:
                    logger.error(f"Batched inference error at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        # Step all active envs
        for i in range(n):
            if not active[i]:
                continue
            if action_chunks[i] is None or chunk_idxs[i] >= chunk_stride:
                continue

            raw_action = denormalize_action(action_chunks[i][chunk_idxs[i]], action_stats, norm_type)
            chunk_idxs[i] += 1
            action_12d = action_7d_to_12d(raw_action)

            if record_video:
                ep_frames[i].append(extract_image(obs_list[i], size=image_size))

            obs_list[i], reward, done, info = envs[i].step(action_12d)
            ep_rewards[i] += reward
            ep_steps[i] = step + 1

            # Check success
            try:
                if envs[i]._check_success():
                    ep_successes[i] = True
                    active[i] = False
            except Exception:
                pass

            # Early termination: stuck
            current_state = extract_state_16d(obs_list[i])
            if prev_states[i] is not None:
                if np.abs(current_state - prev_states[i]).max() < 1e-4:
                    no_change_counts[i] += 1
                else:
                    no_change_counts[i] = 0
                if no_change_counts[i] >= 50:
                    active[i] = False
            prev_states[i] = current_state

    print()  # newline after step loop

    # Close envs and collect results
    results = []
    for i in range(n):
        envs[i].close()
        tag = "\033[92mSUCCESS\033[0m" if ep_successes[i] else "\033[91mFAIL\033[0m"
        reason = ""
        if not ep_successes[i] and no_change_counts[i] >= 50:
            reason = " [stuck]"
        print(f"    ep{ep_offset+i+1:>2d}: {tag}  steps={ep_steps[i]:<4d}  reward={ep_rewards[i]:.2f}{reason}")

        r = {
            "success": ep_successes[i],
            "steps": ep_steps[i],
            "total_reward": float(ep_rewards[i]),
        }
        if record_video and ep_frames[i]:
            r["frames"] = ep_frames[i]
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Evaluate one (model, task) pair
# ---------------------------------------------------------------------------


def evaluate(
    model_name: str,
    task_name: str,
    num_episodes: int,
    max_horizon: int,
    checkpoint: str,
    device: torch.device,
    dry_run: bool = False,
    save_video_dir: Optional[str] = None,
    batch_size: int = 1,
    adapter_dir_override: Optional[str] = None,
    stats_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a model on a task for N episodes (in batches of batch_size)."""
    cfg = MODEL_CONFIGS[model_name]
    instruction = TASK_INSTRUCTIONS[task_name]

    # Resolve adapter path
    ckpt_name = f"checkpoint-{checkpoint}" if checkpoint != "best" else "checkpoint-best"
    base_dir = Path(adapter_dir_override) if adapter_dir_override else cfg["adapter_dir"]
    adapter_path = str(base_dir / ckpt_name)
    if not os.path.exists(adapter_path):
        logger.error(f"Adapter not found: {adapter_path}")
        return {"success_rate": -1, "error": "adapter_not_found", "path": adapter_path}

    # Load normalization stats
    stats = load_normalization_stats(stats_path=stats_path, adapter_dir=adapter_path)
    action_stats = stats["action_stats"]
    state_stats = stats["state_stats"]

    # Load model
    print(f"  Loading {model_name} from {adapter_path}...", flush=True)
    tokenizer = None
    eagle_processor = None

    try:
        if model_name == "openvla":
            model, tokenizer = load_openvla(adapter_path, device)
        elif model_name == "pi05":
            model, tokenizer = load_pi05(adapter_path, device)
        elif model_name == "groot":
            model, eagle_processor = load_groot(adapter_path, device)
        elif model_name == "smolvla":
            model, tokenizer = load_smolvla(adapter_path, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return {"success_rate": -1, "error": str(e)}

    print(f"  Model loaded.", flush=True)

    if dry_run:
        logger.info("Dry run: testing env creation...")
        try:
            env = create_env(task_name, image_size=cfg["image_size"])
            obs = env.reset()
            logger.info(f"Env created. Obs keys: {list(obs.keys())}")
            env.close()
        except Exception as e:
            logger.error(f"Env creation failed: {e}")
        del model
        torch.cuda.empty_cache()
        return {"success_rate": -1, "dry_run": True}

    # Run episodes in batches
    episode_results = []
    successes = 0
    record_video = save_video_dir is not None
    episodes_done = 0

    while episodes_done < num_episodes:
        bs = min(batch_size, num_episodes - episodes_done)
        print(f"\n  Batch [{episodes_done+1}-{episodes_done+bs}] / {num_episodes}  (batch_size={bs})")

        # Reset model state
        for obj in [model, getattr(model, "base_model", None)]:
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()
            if obj is not None and hasattr(obj, "model") and hasattr(obj.model, "reset"):
                obj.model.reset()

        batch_results = run_episodes_batched(
            model=model,
            model_name=model_name,
            task_name=task_name,
            instruction=instruction,
            action_stats=action_stats,
            state_stats=state_stats,
            norm_type=cfg["norm_type"],
            image_size=cfg["image_size"],
            chunk_stride=cfg["chunk_stride"],
            max_horizon=max_horizon,
            device=device,
            batch_size=bs,
            tokenizer=tokenizer,
            eagle_processor=eagle_processor,
            record_video=record_video,
            ep_offset=episodes_done,
        )

        for i, result in enumerate(batch_results):
            ep_idx = episodes_done + i + 1
            result["episode"] = ep_idx

            # Save video
            if record_video and "frames" in result:
                os.makedirs(save_video_dir, exist_ok=True)
                tag = "success" if result["success"] else "fail"
                vid_path = os.path.join(save_video_dir, f"{model_name}_{task_name}_ep{ep_idx}_{tag}.mp4")
                save_video(result.pop("frames"), vid_path)

            if result["success"]:
                successes += 1
            episode_results.append(result)

        episodes_done += bs
        print(f"  => batch {sum(r['success'] for r in batch_results)}/{bs} | cumulative {successes}/{episodes_done} = {successes/episodes_done:.1%}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    print(f"\n  \033[1m{model_name}/{task_name}: {successes}/{num_episodes} = {success_rate:.1%}\033[0m\n")

    return {
        "model": model_name,
        "task": task_name,
        "checkpoint": checkpoint,
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "episodes": episode_results,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def format_results_table(results, tasks) -> str:
    """Format results as a text table. Returns the table string."""
    col_w = max(max(len(t) for t in tasks), 12)
    lines = []

    lines.append("=" * 70)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 70)
    header = f"{'Task':<{col_w}}  {'Success Rate':>14}  {'Successes':>10}  {'Avg Steps':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    rates = []
    for task in tasks:
        if task in results and results[task].get("success_rate", -1) >= 0:
            r = results[task]
            rate = r["success_rate"]
            rates.append(rate)
            avg_steps = np.mean([e["steps"] for e in r["episodes"]])
            s, t = r["successes"], r["total_episodes"]
            lines.append(f"{task:<{col_w}}  {rate:>13.1%}  {s:>4}/{t:<4}  {avg_steps:>10.1f}")
        else:
            lines.append(f"{task:<{col_w}}  {'N/A':>14}  {'':>10}  {'':>10}")

    if rates:
        avg = np.mean(rates)
        lines.append("-" * len(header))
        lines.append(f"{'AVERAGE':<{col_w}}  {avg:>13.1%}")
    lines.append("=" * 70)

    return "\n".join(lines)


def print_results_table(results, tasks):
    """Print formatted results table."""
    print("\n" + format_results_table(results, tasks) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def list_available_checkpoints():
    """Print available checkpoints for each model."""
    for name, cfg in MODEL_CONFIGS.items():
        adapter_dir = cfg["adapter_dir"]
        if adapter_dir.exists():
            ckpts = sorted([
                d.name.replace("checkpoint-", "")
                for d in adapter_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ])
            print(f"  {name}: {', '.join(ckpts)}")
        else:
            print(f"  {name}: (no adapter directory found)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA LoRA adapters on RoboCasa tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=False, choices=list(MODEL_CONFIGS.keys()),
        help="Model to evaluate (pi05 or groot)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="RoboCasa task names (default: all tasks)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=24,
        help="Episodes per task (default: 24)",
    )
    parser.add_argument(
        "--max-horizon", type=int, default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best",
        help="Checkpoint to load: 'best', 'final', or step number like '15000' (default: best)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/eval_results",
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save rollout videos (requires imageio)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=24,
        help="Number of parallel environments per batch (default: 1)",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=None,
        help="Override adapter directory (e.g. ./outputs/vla_adapters/pi05_lr1e-5/general/lora_adapter)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test model + env loading only")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stats-path", type=str, default=None,
                        help="Path to data_stats.json for normalization (auto-detected from adapter dir if absent)")
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks and exit")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="Merge multiple result JSON files into one and exit")

    args = parser.parse_args()

    # Info-only modes
    if args.list_tasks:
        print("\nAvailable RoboCasa tasks:")
        for t, instr in sorted(TASK_INSTRUCTIONS.items()):
            print(f"  {t:<30s} -> \"{instr}\"")
        return

    if args.list_checkpoints:
        print("\nAvailable checkpoints:")
        list_available_checkpoints()
        return

    if args.merge:
        # Merge multiple result JSON files into one combined output
        merged_results = {}
        merged_tasks = []
        meta = {}
        for jpath in args.merge:
            with open(jpath) as f:
                data = json.load(f)
            if not meta:
                meta = {k: data[k] for k in data if k not in ("results", "summary_table", "tasks")}
            for task_name, task_data in data["results"].items():
                merged_results[task_name] = task_data
                if task_name not in merged_tasks:
                    merged_tasks.append(task_name)
        # Sort tasks to match canonical order
        canonical = list(TASK_INSTRUCTIONS.keys())
        merged_tasks.sort(key=lambda t: canonical.index(t) if t in canonical else 999)
        table_str = format_results_table(merged_results, merged_tasks)
        print("\n" + table_str + "\n")
        # Save merged files
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = meta.get("model", "merged")
        ckpt = meta.get("checkpoint", "best")
        base = f"eval_{model_name}_{ckpt}_{timestamp}"
        meta.update({"tasks": merged_tasks, "results": merged_results, "summary_table": table_str})
        json_path = os.path.join(out_dir, base + ".json")
        txt_path = os.path.join(out_dir, base + ".txt")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")
        print(f"Merged JSON: {json_path}")
        print(f"Merged TXT:  {txt_path}")
        return

    # Validate
    if args.model is None:
        parser.error("--model is required (use --list-tasks or --list-checkpoints for info)")

    tasks = args.tasks or list(TASK_INSTRUCTIONS.keys())
    # Resolve legacy v0.1 task names to v1.0 names
    resolved_tasks = []
    for t in tasks:
        if t in _TASK_ALIASES:
            new_t = _TASK_ALIASES[t]
            logger.info(f"Task alias: {t} -> {new_t}")
            resolved_tasks.append(new_t)
        else:
            resolved_tasks.append(t)
    tasks = resolved_tasks
    for t in tasks:
        if t not in TASK_INSTRUCTIONS:
            logger.warning(f"No instruction for '{t}', using task name")
            TASK_INSTRUCTIONS[t] = t.lower().replace("_", " ")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} | Checkpoint: {args.checkpoint}")
    print(f"Tasks: {len(tasks)} | Episodes: {args.num_episodes} | Horizon: {args.max_horizon} | Batch: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    video_dir = os.path.join(args.output_dir, "videos") if args.save_video else None

    all_results = {}
    start_time = time.time()

    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {args.model} / {task}")
        print("-" * 40)

        result = evaluate(
            model_name=args.model,
            task_name=task,
            num_episodes=args.num_episodes,
            max_horizon=args.max_horizon,
            checkpoint=args.checkpoint,
            device=device,
            dry_run=args.dry_run,
            save_video_dir=video_dir,
            batch_size=args.batch_size,
            adapter_dir_override=args.adapter_dir,
            stats_path=args.stats_path,
        )
        all_results[task] = result

    elapsed = time.time() - start_time

    # Print summary
    if not args.dry_run:
        print_results_table(all_results, tasks)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"eval_{args.model}_{args.checkpoint}_{timestamp}.json")
    table_str = format_results_table(all_results, tasks) if not args.dry_run else ""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "num_episodes": args.num_episodes,
        "max_horizon": args.max_horizon,
        "elapsed_seconds": elapsed,
        "results": all_results,
        "summary_table": table_str,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Save readable text summary
    if table_str:
        txt_path = output_path.replace(".json", ".txt")
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")
        print(f"Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
