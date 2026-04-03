#!/usr/bin/env python3
"""
RoboCasa evaluation script for trained VLA LoRA adapters.

Evaluates SmolVLA, PI0.5, and GROOT N1.5 models with LoRA adapters
on RoboCasa single-stage manipulation tasks.

Usage:
    # Evaluate all models on all tasks
    python eval_robocasa.py --models smolvla pi05 groot_n1.5 \
        --tasks CloseDrawer TurnOnMicrowave CloseSingleDoor TurnOffSinkFaucet TurnOffMicrowave \
        --num-episodes 25

    # Dry run (test env creation + model loading only)
    python eval_robocasa.py --models smolvla --tasks CloseDrawer --dry-run

    # Single model/task
    python eval_robocasa.py --models pi05 --tasks CloseDrawer --num-episodes 10
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
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for _mod in ("robosuite", "OpenGL", "PIL", "urllib3", "transformers", "huggingface_hub", "peft"):
    logging.getLogger(_mod).setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Fetching.*files.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# State indices kept during training (from train_lora.py):
# eef_pos(3) + eef_quat(4) + target_pos(3) = 10D
# Indices into 19D preprocessed state: [0..6, 16..18]
# But raw RoboCasa obs gives 16D: eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper(2)
# The 19D comes from preprocessing that adds 3 extra dims.
# For evaluation we assemble the 16D raw state, then data_stats also has 16D.
# We use indices [0:7] (eef_pos + eef_quat) + [14:16] (gripper) = 9D if following
# the _STATE_KEEP_IDX pattern on 16D... but data_stats.json has 16D stats.
#
# Actually, looking at the training code: state_stats in data_stats.json are 16D
# (the raw concatenation). The _STATE_KEEP_IDX = list(range(0,7)) + list(range(16,19))
# is for the 19D preprocessed format. For raw 16D, SmolVLAChunkDataset and others
# check len and skip the index mapping when len != 19.
#
# So during eval we just provide the full 16D state and let normalization use
# the full 16D stats from data_stats.json.

TASK_INSTRUCTIONS = {
    "CloseDrawer": "close the drawer",
    "TurnOnMicrowave": "turn on the microwave",
    "CloseSingleDoor": "close the door",
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "TurnOffMicrowave": "turn off the microwave",
}

MODEL_CONFIGS = {
    "smolvla": {
        "backend_name": "lerobot/smolvla_base",
        "image_size": 512,
        "chunk_size": 50,
        "norm_type": "mean_std",  # z-score
        "camera_key": "observation.images.camera1",
    },
    "pi05": {
        "backend_name": "lerobot/pi05_base",
        "image_size": 224,
        "chunk_size": 50,
        "norm_type": "quantile",  # quantile [-1, 1]
        "camera_key": "observation.images.base_0_rgb",
    },
    "groot_n1.5": {
        "backend_name": "nvidia/GR00T-N1.5-3B",
        "image_size": 224,
        "chunk_size": 16,
        "norm_type": "min_max",  # min_max [-1, 1]
        "camera_key": None,  # GROOT uses Eagle processor, not direct image key
    },
}


# ---------------------------------------------------------------------------
# Data stats loading
# ---------------------------------------------------------------------------


def load_data_stats(stats_path: str) -> Dict[str, Any]:
    """Load normalization statistics from data_stats.json."""
    with open(stats_path) as f:
        stats = json.load(f)
    return stats


# ---------------------------------------------------------------------------
# Action denormalization
# ---------------------------------------------------------------------------


def denormalize_action_mean_std(
    action: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Denormalize action from z-score: action * std + mean."""
    return action * std + mean


def denormalize_action_quantile(
    action: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """Denormalize action from quantile [-1,1]: (action+1)/2 * (q99-q01) + q01."""
    return (action + 1.0) / 2.0 * (q99 - q01) + q01


def denormalize_action_min_max(
    action: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> np.ndarray:
    """Denormalize action from min_max [-1,1]: (action+1)/2 * (max-min) + min."""
    return (action + 1.0) / 2.0 * (vmax - vmin) + vmin


# ---------------------------------------------------------------------------
# State normalization (for preparing model input)
# ---------------------------------------------------------------------------


def normalize_state_mean_std(
    state: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize state with z-score."""
    return (state - mean) / (std + 1e-8)


def normalize_state_min_max(
    state: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> np.ndarray:
    """Normalize state with min_max to [-1, 1]."""
    r = vmax - vmin + 1e-8
    return np.clip(2.0 * (state - vmin) / r - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# State extraction from RoboCasa observations
# ---------------------------------------------------------------------------


def extract_state_from_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract 16D robot state from RoboCasa observation dict.

    Concatenation order matches training data preprocessing:
    eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper_qpos(2) = 16D
    """
    eef_pos = obs["robot0_eef_pos"]       # (3,)
    eef_quat = obs["robot0_eef_quat"]     # (4,)
    joint_pos = obs["robot0_joint_pos"]    # (7,)
    gripper = obs["robot0_gripper_qpos"]   # (2,)
    return np.concatenate([eef_pos, eef_quat, joint_pos, gripper]).astype(np.float32)


def extract_image_from_obs(
    obs: Dict[str, np.ndarray],
    camera_name: str = "robot0_agentview_left",
    target_size: int = 224,
) -> np.ndarray:
    """Extract and resize camera image from RoboCasa observation.

    Returns: (H, W, 3) uint8 array.
    """
    from PIL import Image as PILImage

    key = f"{camera_name}_image"
    img = obs[key]  # (H, W, 3) uint8

    if img.shape[0] != target_size or img.shape[1] != target_size:
        pil_img = PILImage.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size), PILImage.BILINEAR)
        img = np.array(pil_img)

    return img


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------


def create_robocasa_env(
    task_name: str,
    camera_height: int = 224,
    camera_width: int = 224,
):
    """Create a RoboCasa environment for evaluation."""
    import robocasa

    env = robocasa.make(
        env_name=task_name,
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_left"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=False,
        ignore_done=True,
        control_freq=20,
    )
    return env


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_smolvla_model(adapter_path: str, device: torch.device):
    """Load SmolVLA base model with PEFT LoRA adapter."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel
    from transformers import AutoTokenizer

    # logger.info("Loading SmolVLA base model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    # Override action_feature to 7D (base model defaults to 6D but we trained with 7D)
    policy.config.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    policy = policy.to(dtype=torch.bfloat16)

    # logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(policy, adapter_path)
    model = model.to(device)
    model.eval()

    try:
        model = torch.compile(model, mode="reduce-overhead")
        pass  # torch.compile OK
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")

    # Tokenizer for instruction encoding
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    return model, tokenizer


def load_pi05_model(adapter_path: str, device: torch.device):
    """Load PI0.5 base model with PEFT LoRA adapter."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel

    # logger.info("Loading PI0.5 base model...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    policy = policy.to(dtype=torch.bfloat16)

    # Override input_features to match our single-camera setup
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

    # logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(policy, adapter_path)
    model = model.to(device)
    model.eval()

    try:
        model = torch.compile(model, mode="reduce-overhead")
        pass  # torch.compile OK
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")

    # Use Gemma tokenizer (PaliGemma uses the same 256K Gemma tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/gemma-2-2b",
        padding_side="right",
    )

    return model, tokenizer


def load_groot_model(adapter_path: str, device: torch.device):
    """Load GROOT N1.5 base model with internal LoRA, then load adapter weights."""
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from safetensors.torch import load_file

    # logger.info("Loading GROOT N1.5 base model with LoRA...")
    policy = GrootPolicy.from_pretrained(
        "nvidia/GR00T-N1.5-3B",
        lora_rank=32,
        lora_alpha=32,
    )

    # Load the trained LoRA weights from checkpoint
    weights_path = os.path.join(adapter_path, "model.safetensors")
    if os.path.exists(weights_path):
        logger.info(f"Loading GROOT adapter weights from {weights_path}")
        adapter_weights = load_file(weights_path)
        # Use strict=False because we only saved the trainable parameters
        missing, unexpected = policy.load_state_dict(adapter_weights, strict=False)
        logger.info(
            f"GROOT adapter loaded: {len(adapter_weights)} tensors, "
            f"{len(missing)} missing keys, {len(unexpected)} unexpected keys"
        )
    else:
        logger.warning(f"No model.safetensors found at {adapter_path}, using base model")

    policy = policy.to(dtype=torch.bfloat16, device=device)
    policy.eval()

    try:
        policy = torch.compile(policy, mode="reduce-overhead")
        pass  # torch.compile OK
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")

    # Build Eagle processor for image+text encoding
    from lerobot.policies.groot.processor_groot import _build_eagle_processor
    try:
        eagle_processor = _build_eagle_processor()
    except FileNotFoundError:
        logger.warning(
            "Eagle processor cache not found. GROOT inference may fail. "
            "Ensure the model was loaded at least once to populate the cache."
        )
        eagle_processor = None

    return policy, eagle_processor


# ---------------------------------------------------------------------------
# Inference helpers (per-model)
# ---------------------------------------------------------------------------


def prepare_smolvla_batch(
    image: np.ndarray,
    state_16d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    tokenizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch dict for SmolVLA select_action / predict_action_chunk.

    SmolVLA expects:
      - observation.images.camera1: (B, 3, 512, 512) float32 [0, 1]
      - observation.state: (B, 10) float32 MEAN_STD normalized
        (the model internally pads to max_state_dim)
      - observation.language.tokens: (B, seq_len) long
      - observation.language.attention_mask: (B, seq_len) bool
    """
    # Image: (H, W, 3) uint8 -> (1, 3, 512, 512) float32 [0,1]
    img_t = torch.from_numpy(image).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    # Resize to 512x512 if needed
    if img_t.shape[-1] != 512 or img_t.shape[-2] != 512:
        img_t = torch.nn.functional.interpolate(
            img_t, size=(512, 512), mode="bilinear", align_corners=False
        )

    # State: normalize with MEAN_STD using full 16D stats
    s_mean = np.array(state_stats["mean"], dtype=np.float32)
    s_std = np.array(state_stats["std"], dtype=np.float32)
    state_norm = normalize_state_mean_std(state_16d, s_mean, s_std)
    state_t = torch.from_numpy(state_norm).float().unsqueeze(0)  # (1, 16)

    # Tokenize instruction
    text = instruction + "\n"
    tokens = tokenizer(
        text,
        max_length=48,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    batch = {
        "observation.images.camera1": img_t.to(device),
        "observation.state": state_t.to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }
    return batch


def prepare_pi05_batch(
    image: np.ndarray,
    state_16d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    action_stats: Dict,
    tokenizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch dict for PI0.5 predict_action_chunk.

    PI0.5 expects:
      - observation.images.base_0_rgb: (B, 3, 224, 224) float32 [0, 1]
      - observation.language.tokens: (B, seq_len) long
      - observation.language.attention_mask: (B, seq_len) bool
    Note: PI0.5 encodes state into the language prompt via discretization.
    """
    # Image: (H, W, 3) uint8 -> (1, 3, 224, 224) float32 [0,1]
    img_t = torch.from_numpy(image).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Discretize state for PI0.5 prompt
    q01 = np.array(state_stats["q01"], dtype=np.float32)
    q99 = np.array(state_stats["q99"], dtype=np.float32)
    state_norm = (state_16d - q01) / (q99 - q01 + 1e-8) * 2 - 1
    state_norm = np.clip(state_norm, -1, 1)
    # Pad to 32 dims
    state_padded = np.zeros(32, dtype=np.float32)
    state_padded[: len(state_norm)] = state_norm
    # Discretize to 256 bins
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state_padded, bins) - 1
    discretized = np.clip(discretized, 0, 255)
    state_str = " ".join(map(str, discretized))

    prompt = f"Task: {instruction}, State: {state_str};\nAction: \n"
    tokens = tokenizer(
        prompt,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    batch = {
        "observation.images.base_0_rgb": img_t.to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }
    return batch


def prepare_groot_batch(
    image: np.ndarray,
    state_16d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    eagle_processor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch dict for GROOT predict_action_chunk.

    GROOT expects:
      - state: (B, 1, 64) float32 MIN_MAX normalized
      - state_mask: (B, 1, 64) bool
      - embodiment_id: (B,) long
      - eagle_*: tensors from Eagle processor encoding
    """
    from PIL import Image as PILImage

    # State: MIN_MAX normalize, pad to 64
    s_min = np.array(state_stats["min"], dtype=np.float32)
    s_max = np.array(state_stats["max"], dtype=np.float32)
    state_norm = normalize_state_min_max(state_16d, s_min, s_max)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[: len(state_norm)] = state_norm
    state_mask = np.zeros(64, dtype=bool)
    state_mask[: len(state_norm)] = True

    state_t = torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 64)
    state_mask_t = torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0)        # (1, 1, 64)
    embodiment_id = torch.tensor([31], dtype=torch.long)  # new_embodiment

    batch = {
        "state": state_t.to(device),
        "state_mask": state_mask_t.to(device),
        "embodiment_id": embodiment_id.to(device),
    }

    # Eagle encoding for image + language
    if eagle_processor is not None:
        pil_img = PILImage.fromarray(image)
        text = f"<image-1> {instruction}"
        eagle_inputs = eagle_processor(
            text=[text],
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
            batch["eagle_" + k] = v.to(device)

    return batch


# ---------------------------------------------------------------------------
# Per-model inference wrappers
# ---------------------------------------------------------------------------


def stack_batch_dicts(batch_dicts):
    """Stack multiple batch dicts (each batch_size=1) into a single batched dict."""
    if len(batch_dicts) == 1:
        return batch_dicts[0]
    result = {}
    for key in batch_dicts[0]:
        result[key] = torch.cat([b[key] for b in batch_dicts], dim=0)
    return result


@torch.no_grad()
def generate_action_chunk(
    model,
    batch: Dict[str, torch.Tensor],
    **kwargs,
) -> np.ndarray:
    """Generate a full action chunk from the model.
    Returns: (chunk_size, action_dim) float32 numpy array (still normalized).
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch, **kwargs)  # (1, chunk_size, action_dim)
    return actions[0].float().cpu().numpy()  # (chunk_size, action_dim)


@torch.no_grad()
def generate_action_chunk_batched(
    model,
    batch: Dict[str, torch.Tensor],
    **kwargs,
) -> np.ndarray:
    """Generate action chunks for a batch of observations.
    Returns: (batch_size, chunk_size, action_dim) float32 numpy array.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch, **kwargs)
    return actions.float().cpu().numpy()


def denormalize_action(
    action: np.ndarray,
    action_stats: Dict,
    norm_type: str,
) -> np.ndarray:
    """Denormalize a single 7D action based on norm_type."""
    a = action[:7]
    if norm_type == "mean_std":
        a_mean = np.array(action_stats["mean"][:7], dtype=np.float32)
        a_std = np.array(action_stats["std"][:7], dtype=np.float32)
        return denormalize_action_mean_std(a, a_mean, a_std)
    elif norm_type == "quantile":
        a_q01 = np.array(action_stats["q01"][:7], dtype=np.float32)
        a_q99 = np.array(action_stats["q99"][:7], dtype=np.float32)
        return denormalize_action_quantile(a, a_q01, a_q99)
    elif norm_type == "min_max":
        a_min = np.array(action_stats["min"][:7], dtype=np.float32)
        a_max = np.array(action_stats["max"][:7], dtype=np.float32)
        return denormalize_action_min_max(a, a_min, a_max)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model_task(
    model_name: str,
    task_name: str,
    adapter_base: str,
    num_episodes: int,
    max_horizon: int,
    device: torch.device,
    dry_run: bool = False,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Evaluate a single (model, task) pair.

    Returns dict with success_rate, successes, total, per-episode results.
    """
    cfg = MODEL_CONFIGS[model_name]
    instruction = TASK_INSTRUCTIONS[task_name]

    # Adapter path
    adapter_path = os.path.join(
        adapter_base, model_name, task_name, "lora_adapter", "checkpoint-best"
    )
    stats_path = os.path.join(
        adapter_base, model_name, task_name, "lora_adapter", "data_stats.json"
    )

    # Validate paths
    if not os.path.exists(adapter_path):
        logger.warning(f"Adapter not found: {adapter_path}, skipping")
        return {"success_rate": -1, "error": "adapter_not_found"}
    if not os.path.exists(stats_path):
        logger.warning(f"Stats not found: {stats_path}, skipping")
        return {"success_rate": -1, "error": "stats_not_found"}

    # Load stats
    data_stats = load_data_stats(stats_path)
    action_stats = data_stats["action_stats"]
    state_stats = data_stats["state_stats"]

    # Load model
    print(f"  Loading {model_name} adapter...", end=" ", flush=True)
    tokenizer = None
    eagle_processor = None

    try:
        if model_name == "smolvla":
            model, tokenizer = load_smolvla_model(adapter_path, device)
        elif model_name == "pi05":
            model, tokenizer = load_pi05_model(adapter_path, device)
        elif model_name == "groot_n1.5":
            model, eagle_processor = load_groot_model(adapter_path, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        print("FAILED")
        logger.error(f"Failed to load {model_name}: {e}")
        return {"success_rate": -1, "error": str(e)}

    print("OK")
    image_size = cfg["image_size"]
    norm_type = cfg["norm_type"]

    if dry_run:
        # Create a single env to test
        logger.info(f"Creating RoboCasa env for {task_name} (image_size={image_size})...")
        try:
            env = create_robocasa_env(
                task_name,
                camera_height=image_size,
                camera_width=image_size,
            )
        except Exception as e:
            logger.error(f"Failed to create env for {task_name}: {e}")
            del model
            torch.cuda.empty_cache()
            return {"success_rate": -1, "error": f"env_creation_failed: {e}"}
        logger.info(f"[DRY RUN] {model_name}/{task_name}: model loaded, env created successfully")
        env.close()
        del model
        torch.cuda.empty_cache()
        return {"success_rate": -1, "dry_run": True}

    # Chunk stride: only execute this many steps before re-predicting
    CHUNK_STRIDES = {"smolvla": 10, "pi05": 10, "groot_n1.5": 8}
    chunk_stride = CHUNK_STRIDES.get(model_name, 10)

    # Run episodes in batches
    episodes_done = 0
    episode_results = []
    successes = 0

    while episodes_done < num_episodes:
        n = min(batch_size, num_episodes - episodes_done)
        print(f"\n  batch {episodes_done+1}-{episodes_done+n}/{num_episodes} (stride={chunk_stride})")

        # Reset model state once per batch
        if hasattr(model, "reset"):
            model.reset()
        base = model
        if hasattr(model, "base_model"):
            base = model.base_model
            if hasattr(base, "model") and hasattr(base.model, "reset"):
                base.model.reset()
        if hasattr(base, "reset"):
            base.reset()

        # Create N environments
        envs = []
        for _ in range(n):
            try:
                envs.append(create_robocasa_env(task_name, camera_height=image_size, camera_width=image_size))
            except Exception as e:
                logger.error(f"Failed to create env for {task_name}: {e}")
                for env in envs:
                    env.close()
                del model
                torch.cuda.empty_cache()
                return {"success_rate": -1, "error": f"env_creation_failed: {e}"}
        obs_list = [env.reset() for env in envs]

        # Per-env state
        action_chunks = [None] * n
        chunk_idxs = [0] * n
        active = [True] * n
        ep_successes = [False] * n
        ep_steps = [0] * n
        ep_rewards = [0.0] * n
        prev_states = [None] * n  # for early termination
        no_change_counts = [0] * n

        _batch_start = time.time()
        for step in range(max_horizon):
            if not any(active):
                break

            # Live progress
            n_active = sum(active)
            elapsed_s = time.time() - _batch_start
            print(f"\r  step {step:>3d}/{max_horizon} | active: {n_active}/{n} | {elapsed_s:.0f}s", end="", flush=True)

            # Find envs that need new action chunks
            needs_chunk = []
            for i in range(n):
                if not active[i]:
                    continue
                if action_chunks[i] is None or chunk_idxs[i] >= chunk_stride:
                    needs_chunk.append(i)

            if needs_chunk:
                # Prepare batched observations
                batch_dicts = []
                for i in needs_chunk:
                    image = extract_image_from_obs(obs_list[i], camera_name="robot0_agentview_left", target_size=image_size)
                    state_16d = extract_state_from_obs(obs_list[i])

                    try:
                        if model_name == "smolvla":
                            bd = prepare_smolvla_batch(image, state_16d, instruction, state_stats, tokenizer, device)
                        elif model_name == "pi05":
                            bd = prepare_pi05_batch(image, state_16d, instruction, state_stats, action_stats, tokenizer, device)
                        elif model_name == "groot_n1.5":
                            bd = prepare_groot_batch(image, state_16d, instruction, state_stats, eagle_processor, device)
                        batch_dicts.append(bd)
                    except Exception as e:
                        logger.error(f"  Batch prep error at step {step}, env {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        active[i] = False
                        continue

                if batch_dicts:
                    try:
                        # Stack and run batched inference
                        stacked = stack_batch_dicts(batch_dicts)
                        kwargs = {"num_steps": 10} if model_name == "pi05" else {}
                        all_chunks = generate_action_chunk_batched(model, stacked, **kwargs)

                        # Distribute chunks back to the envs that are still active
                        j = 0
                        for idx in needs_chunk:
                            if not active[idx]:
                                continue
                            action_chunks[idx] = all_chunks[j]
                            chunk_idxs[idx] = 0
                            j += 1
                    except Exception as e:
                        logger.error(f"  Batched inference error at step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        break

            # Step all active environments
            for i in range(n):
                if not active[i]:
                    continue
                if action_chunks[i] is None or chunk_idxs[i] >= chunk_stride:
                    continue

                action = denormalize_action(action_chunks[i][chunk_idxs[i]], action_stats, norm_type)
                chunk_idxs[i] += 1

                # Map 7D -> 12D
                action_12d = np.zeros(12, dtype=np.float32)
                action_12d[0:6] = np.clip(action[:6], -1.0, 1.0)
                action_12d[10:12] = np.clip(action[6], -1.0, 1.0)

                obs_list[i], reward, done, info = envs[i].step(action_12d)
                ep_rewards[i] += reward
                ep_steps[i] = step + 1

                # Check success
                try:
                    if envs[i]._check_success():
                        ep_successes[i] = True
                        active[i] = False
                        pass  # shown in batch summary
                except Exception:
                    pass

                # Early termination: check if state changed
                current_state = extract_state_from_obs(obs_list[i])
                if prev_states[i] is not None:
                    state_delta = np.abs(current_state - prev_states[i]).max()
                    if state_delta < 1e-4:
                        no_change_counts[i] += 1
                    else:
                        no_change_counts[i] = 0
                    if no_change_counts[i] >= 50:
                        active[i] = False
                        pass  # shown in batch summary
                prev_states[i] = current_state

        # Record results for this batch
        print()  # newline after progress bar
        batch_ok = 0
        for i in range(n):
            ep_idx = episodes_done + i
            episode_results.append({
                "episode": ep_idx,
                "success": ep_successes[i],
                "steps": ep_steps[i],
                "total_reward": float(ep_rewards[i]),
            })
            if ep_successes[i]:
                successes += 1
                batch_ok += 1
            tag = "\033[92mSUCCESS\033[0m" if ep_successes[i] else "\033[91mFAIL\033[0m"
            print(f"    ep{ep_idx+1:>2d}: {tag}  steps={ep_steps[i]}")
            envs[i].close()
        print(f"  => batch {batch_ok}/{n} | cumulative {successes}/{episodes_done+n} = {successes/(episodes_done+n):.1%}")

        episodes_done += n

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    result = {
        "model": model_name,
        "task": task_name,
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "episodes": episode_results,
    }

    print(f"\n  \033[1m{model_name}/{task_name}: {successes}/{num_episodes} = {success_rate:.1%}\033[0m")
    return result


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def print_results_table(results: Dict[str, Dict[str, Any]], models: List[str], tasks: List[str]):
    """Print a formatted results table: model x task success rates."""
    # Header
    task_width = max(len(t) for t in tasks)
    model_width = max(len(m) for m in models)
    col_width = max(task_width, 12)

    header = f"{'Model':<{model_width}} | " + " | ".join(
        f"{t:>{col_width}}" for t in tasks
    ) + " | " + f"{'Average':>{col_width}}"
    separator = "-" * len(header)

    print("\n" + separator)
    print("EVALUATION RESULTS")
    print(separator)
    print(header)
    print(separator)

    for model in models:
        rates = []
        cells = []
        for task in tasks:
            key = f"{model}/{task}"
            if key in results and results[key].get("success_rate", -1) >= 0:
                rate = results[key]["success_rate"]
                rates.append(rate)
                s = results[key].get("successes", 0)
                t = results[key].get("total_episodes", 0)
                cells.append(f"{rate:.1%} ({s}/{t})")
            else:
                cells.append("N/A")

        avg = sum(rates) / len(rates) if rates else 0.0
        row = f"{model:<{model_width}} | " + " | ".join(
            f"{c:>{col_width}}" for c in cells
        ) + " | " + f"{avg:.1%}".rjust(col_width)
        print(row)

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA LoRA adapters on RoboCasa tasks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["smolvla", "pi05", "groot_n1.5"],
        choices=["smolvla", "pi05", "groot_n1.5"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "CloseDrawer",
            "TurnOnMicrowave",
            "CloseSingleDoor",
            "TurnOffSinkFaucet",
            "TurnOffMicrowave",
        ],
        help="RoboCasa tasks to evaluate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=25,
        help="Number of evaluation episodes per (model, task)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--adapter-base",
        type=str,
        default="./outputs/raw_lora_results",
        help="Base directory for adapter checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test env creation and model loading, no full episodes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of parallel environments per (model, task)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Models: {args.models} | Tasks: {len(args.tasks)}")
    print(f"Episodes: {args.num_episodes} | Horizon: {args.max_horizon} | Batch: {args.batch_size}")
    print(f"Adapters: {args.adapter_base}")

    # Validate task instructions exist
    for task in args.tasks:
        if task not in TASK_INSTRUCTIONS:
            logger.warning(
                f"No instruction mapping for task '{task}', using task name as instruction"
            )
            TASK_INSTRUCTIONS[task] = task.lower().replace("_", " ")

    # Run evaluation
    all_results = {}
    start_time = time.time()

    total_combos = len(args.models) * len(args.tasks)
    combo_idx = 0
    for model_name in args.models:
        for task_name in args.tasks:
            combo_idx += 1
            print(f"\n{'=' * 60}")
            print(f"[{combo_idx}/{total_combos}] {model_name} / {task_name}  ({args.num_episodes} eps, horizon={args.max_horizon})")
            print(f"{'=' * 60}")

            result = evaluate_model_task(
                model_name=model_name,
                task_name=task_name,
                adapter_base=args.adapter_base,
                num_episodes=args.num_episodes,
                max_horizon=args.max_horizon,
                device=device,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
            all_results[f"{model_name}/{task_name}"] = result

    elapsed = time.time() - start_time
    logger.info(f"\nTotal evaluation time: {elapsed / 60:.1f} minutes")

    # Print results table
    if not args.dry_run:
        print_results_table(all_results, args.models, args.tasks)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_summary.json")
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "models": args.models,
            "tasks": args.tasks,
            "num_episodes": args.num_episodes,
            "max_horizon": args.max_horizon,
            "adapter_base": args.adapter_base,
            "dry_run": args.dry_run,
        },
        "elapsed_seconds": elapsed,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
