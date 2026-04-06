#!/usr/bin/env python3
"""
RoboCasa evaluation for VLA LoRA adapters with move/grip adapter support.

Two execution modes:
  - direct:   Direct model loading + action chunking (like eval_robocasa.py)
  - pipeline: Full RoboBridgeClient pipeline (Perception -> Planner -> Controller)

Usage:
    # Direct mode (default): GROOT with move/grip adapters
    python eval_vla_robocasa.py --model groot \
        --move-adapter ./outputs/vla_adapters/groot_move_grip/general/move_adapter_lr1e-4/checkpoint-best \
        --grip-adapter ./outputs/vla_adapters/groot_move_grip/general/grip_adapter_lr1e-4/checkpoint-best \
        --tasks PnPCounterToCab --num-episodes 10

    # Pipeline mode: RoboBridgeClient with planner + perception + monitor
    python eval_vla_robocasa.py --mode pipeline \
        --vla-backend groot_n1.5 --vla-model nvidia/GR00T-N1.5-3B \
        --move-adapter /path/to/move_adapter --grip-adapter /path/to/grip_adapter \
        --action-stats ./bridge/multitask_training_package/data/metadata.json \
        --dataset /path/to/demo.hdf5 --num-episodes 10

    # List all available tasks
    python eval_vla_robocasa.py --list-tasks
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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
    # Doors (exact HDF5 instructions)
    "OpenDoor": "open the cabinet door",
    "CloseDoor": "close the cabinet door",
    "OpenSingleDoor": "open the cabinet door",
    "CloseSingleDoor": "close the cabinet door",
    "OpenDoubleDoor": "open the cabinet doors",
    "CloseDoubleDoor": "close the cabinet doors",
    # Drawers
    "OpenDrawer": "open the right drawer",
    "CloseDrawer": "close the left drawer",
    # Cabinets
    "OpenCabinet": "open the cabinet door",
    "CloseCabinet": "close the cabinet doors",
    # Coffee
    "StartCoffeeMachine": "press the button on the coffee machine to serve coffee",
    "CoffeeSetupMug": "pick the mug from the counter and place it under the coffee machine dispenser",
    "CoffeeServeMug": "pick the mug from under the coffee machine dispenser and place it on the counter",
    "CoffeePressButton": "press the button on the coffee machine to serve coffee",
    # Pick and place (exact HDF5 instructions)
    "PnPCabToCounter": "pick the hot dog from the cabinet and place it on the counter",
    "PnPCounterToCab": "pick the cake from the counter and place it in the cabinet",
    "PnPCounterToMicrowave": "pick the corn from the counter and place it in the microwave",
    "PnPCounterToSink": "pick the apple from the counter and place it in the sink",
    "PnPCounterToStove": "pick the potato from the plate and place it in the pan",
    "PnPMicrowaveToCounter": "pick the onion from the microwave and place it on plate located on the counter",
    "PnPSinkToCounter": "pick the tomato from the sink and place it on the plate located on the counter",
    "PnPStoveToCounter": "pick the steak from the pan and place it on the plate",
    # Legacy PnP aliases
    "PickPlaceCounterToCabinet": "pick the cake from the counter and place it in the cabinet",
    "PickPlaceCabinetToCounter": "pick the hot dog from the cabinet and place it on the counter",
    "PickPlaceCounterToMicrowave": "pick the corn from the counter and place it in the microwave",
    "PickPlaceMicrowaveToCounter": "pick the onion from the microwave and place it on plate located on the counter",
    "PickPlaceCounterToSink": "pick the apple from the counter and place it in the sink",
    "PickPlaceSinkToCounter": "pick the tomato from the sink and place it on the plate located on the counter",
    "PickPlaceCounterToStove": "pick the potato from the plate and place it in the pan",
    "PickPlaceStoveToCounter": "pick the steak from the pan and place it on the plate",
    # Microwave
    "TurnOnMicrowave": "press the start button on the microwave",
    "TurnOffMicrowave": "press the stop button on the microwave",
    # Sink
    "TurnOnSinkFaucet": "turn on the sink faucet",
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "TurnSinkSpout": "turn the sink spout to the right",
    # Stove
    "TurnOnStove": "turn on the front left burner of the stove",
    "TurnOffStove": "turn off the front right burner of the stove",
}

# ---------------------------------------------------------------------------
# Model configs (direct mode)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "groot": {
        "base_model": "nvidia/GR00T-N1.5-3B",
        "adapter_dir": ADAPTER_BASE / "groot",
        "move_adapter_dir": ADAPTER_BASE / "groot_move_grip" / "general" / "move_adapter_lr1e-4",
        "grip_adapter_dir": ADAPTER_BASE / "groot_move_grip" / "general" / "grip_adapter_lr1e-4",
        "image_size": 128,
        "chunk_size": 16,
        "chunk_stride": 8,
        "norm_type": "min_max",
    },
    "pi05": {
        "base_model": "lerobot/pi05_base",
        "adapter_dir": ADAPTER_BASE / "pi0.5" / "general" / "lora_adapter",
        "image_size": 224,
        "chunk_size": 50,
        "chunk_stride": 10,
        "norm_type": "quantile",
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


# ===========================================================================
# SHARED: Statistics, normalization, observation, environment, video, results
# ===========================================================================


def load_normalization_stats() -> Dict[str, Any]:
    """Load action/state normalization statistics from metadata files."""
    meta_path = METADATA_DIR / "metadata.json"
    meta_ext_path = METADATA_DIR / "metadata_extended.json"

    with open(meta_path) as f:
        meta = json.load(f)
    with open(meta_ext_path) as f:
        meta_ext = json.load(f)

    RAW_STATE_DIM = 12
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


# --- Normalization helpers ---

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


def denormalize_action(action_raw, action_stats, norm_type):
    """Denormalize action vector. Handles both 7D and 12D stats, always returns 7D."""
    stat_dim = len(action_stats["min"])
    a_dim = min(stat_dim, len(action_raw))
    a = action_raw[:a_dim].copy()
    if norm_type == "quantile":
        q01 = np.array(action_stats["q01"][:a_dim], dtype=np.float32)
        q99 = np.array(action_stats["q99"][:a_dim], dtype=np.float32)
        result = denormalize_action_quantile(a, q01, q99)
    elif norm_type == "min_max":
        vmin = np.array(action_stats["min"][:a_dim], dtype=np.float32)
        vmax = np.array(action_stats["max"][:a_dim], dtype=np.float32)
        result = denormalize_action_min_max(a, vmin, vmax)
    elif norm_type == "mean_std":
        mean = np.array(action_stats["mean"][:a_dim], dtype=np.float32)
        std = np.array(action_stats["std"][:a_dim], dtype=np.float32)
        result = denormalize_action_mean_std(a, mean, std)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    return result[:7]


# --- Observation ---

def extract_state_12d(obs):
    """Extract 12D state: eef_pos(3) + eef_quat(4) + gripper(2) + target_pos(3).
    Uses eef_pos as target_pos (no external target in direct mode).
    """
    eef_pos = obs["robot0_eef_pos"]
    return np.concatenate([
        eef_pos,
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
        eef_pos,  # target_pos = eef_pos (self-referential)
    ]).astype(np.float32)


extract_state_16d = extract_state_12d  # backward compat alias


def extract_state_16d_hdf5(obs):
    """Extract 16D state for HDF5-trained models: eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper(2)."""
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


# --- Environment ---

# Door tasks require fixture_id in robocasa v1.0+
_DOOR_FIXTURE_MAP = {
    "OpenDoor": ("OpenDoor", "CABINET_WITH_DOOR"),
    "CloseDoor": ("CloseDoor", "CABINET_WITH_DOOR"),
    "OpenSingleDoor": ("OpenDoor", "CABINET_WITH_DOOR"),
    "CloseSingleDoor": ("CloseDoor", "CABINET_WITH_DOOR"),
    "OpenDoubleDoor": ("OpenDoor", "CABINET_WITH_DOOR"),
    "CloseDoubleDoor": ("CloseDoor", "CABINET_WITH_DOOR"),
}

# Task name aliases (training name -> robocasa env name)
_TASK_ENV_MAP = {
    "CoffeePressButton": "StartCoffeeMachine",
    "PnPCabToCounter": "PickPlaceCabinetToCounter",
    "PnPCounterToCab": "PickPlaceCounterToCabinet",
    "PnPCounterToSink": "PickPlaceCounterToSink",
    "PnPSinkToCounter": "PickPlaceSinkToCounter",
    "PnPCounterToStove": "PickPlaceCounterToStove",
    "PnPStoveToCounter": "PickPlaceStoveToCounter",
    "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
    "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
}


def create_env(task_name, image_size=224, match_training=False):
    """Create a RoboCasa environment with random layout.

    Handles RoboCasa style_ids KeyError (np.int64 vs int dict key mismatch)
    by retrying without style_ids on failure.
    """
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

    if match_training:
        # Match HDF5 training data env kwargs for visual consistency
        # Note: obj_instance_split="A" excluded - causes sampling failures
        kwargs.update(dict(
            generative_textures="100p",
            style_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11],
            translucent_robot=False,
            randomize_cameras=True,
            controller_configs={
                "type": "OSC_POSE", "input_max": 1, "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
                "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
                "position_limits": None, "orientation_limits": None,
                "uncouple_pos_ori": True, "control_delta": True,
                "interpolation": None, "ramp_ratio": 0.2,
            },
        ))

    # Resolve task name aliases
    resolved_name = _TASK_ENV_MAP.get(task_name, task_name)

    if resolved_name in _DOOR_FIXTURE_MAP:
        env_name, fixture_attr = _DOOR_FIXTURE_MAP[resolved_name]
        kwargs["fixture_id"] = getattr(FixtureType, fixture_attr)
    else:
        env_name = resolved_name

    # Retry with fallback: RoboCasa scene_registry can throw KeyError
    # when np.int64 style_id is used as dict key (numpy/Python int mismatch)
    for attempt in range(3):
        try:
            return robocasa.make(env_name=env_name, **kwargs)
        except KeyError as e:
            if attempt < 2:
                logger.warning(f"Env creation KeyError (attempt {attempt+1}/3): {e}. Retrying without style_ids.")
                kwargs.pop("style_ids", None)
            else:
                raise


def create_env_from_demo(dataset_path, camera_names=None):
    """Create a RoboCasa environment from HDF5 dataset metadata."""
    import robosuite
    from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset

    env_meta = get_env_metadata_from_dataset(dataset_path)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["ignore_done"] = True
    env_kwargs["camera_names"] = camera_names or [
        "robot0_agentview_left", "robot0_eye_in_hand", "robot0_robotview",
    ]
    env_kwargs["camera_heights"] = 480
    env_kwargs["camera_widths"] = 640
    return robosuite.make(**env_kwargs)


def load_demo_state(dataset_path, demo_idx):
    """Load initial state and metadata from HDF5 demo."""
    import h5py

    f = h5py.File(dataset_path, "r")
    demo_names = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))

    if demo_idx >= len(demo_names):
        f.close()
        raise ValueError(f"Demo index {demo_idx} out of range (max: {len(demo_names) - 1})")

    demo_name = demo_names[demo_idx]
    demo = f[f"data/{demo_name}"]
    states = demo["states"][()]
    initial_state = {
        "states": states[0],
        "model": demo.attrs["model_file"],
        "ep_meta": demo.attrs.get("ep_meta", None),
    }

    ep_meta = {}
    if initial_state["ep_meta"]:
        ep_meta = json.loads(initial_state["ep_meta"])

    f.close()
    return initial_state, ep_meta


# --- Action conversion ---

def action_7d_to_12d(action_7d):
    """Convert denormalized 7D action to 12D RoboCasa action.

    RoboCasa PandaMobile 12D layout:
      [0:6]  arm (delta pos + delta rot via OSC)
      [6]    gripper command
      [7:11] base (zeros = no base motion)
      [11]   gripper copy
    """
    action_12d = np.zeros(12, dtype=np.float32)
    action_12d[0:7] = np.clip(action_7d[:7], -1.0, 1.0)
    action_12d[11] = np.clip(action_7d[6], -1.0, 1.0)
    return action_12d


# --- Video ---

def save_video(frames, path, fps=20):
    """Save list of (H,W,3) uint8 frames as mp4."""
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        logger.info(f"Video saved: {path}")
    except ImportError:
        logger.warning("imageio not installed, skipping video save")


# --- Results table ---

def format_results_table(results, tasks) -> str:
    """Format results as a text table."""
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
            avg_steps = np.mean([e.get("steps", 0) for e in r["episodes"]])
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
    print("\n" + format_results_table(results, tasks) + "\n")


# ===========================================================================
# DIRECT MODE: Direct model loading + action chunking
# ===========================================================================


def _find_weights_file(adapter_path: str) -> Optional[str]:
    """Find weights file in adapter directory."""
    for filename in ["model_diff.safetensors", "model.safetensors", "adapter_model.pt"]:
        path = os.path.join(adapter_path, filename)
        if os.path.exists(path):
            return path
    return None


def _load_weights_file(weights_path: str) -> dict:
    """Load weights from safetensors or pt file."""
    if weights_path.endswith(".pt"):
        import torch
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        return load_file(weights_path)


def _read_lora_config(adapter_path: str) -> Tuple[int, int]:
    """Read lora_rank/lora_alpha from adapter config.json."""
    config_path = os.path.join(adapter_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        rank = cfg.get("lora_rank", 64)
        alpha = cfg.get("lora_alpha", rank * 2)
        return rank, alpha
    return 64, 128


def load_pi05(adapter_path, device):
    """Load PI0.5 with PEFT LoRA adapter."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel
    from transformers import AutoTokenizer

    logger.info("Loading PI0.5 base model...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    policy = policy.to(dtype=torch.bfloat16)

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
    return model, tokenizer, None


def load_groot(
    adapter_path: Optional[str] = None,
    move_adapter: Optional[str] = None,
    grip_adapter: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
    image_size: int = 224,
    denoise_steps: int = 10,
):
    """Load GROOT N1.5 with optional move/grip weight-diff adapters.

    Returns: (model, eagle_processor, adapter_weights_dict)
    """
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    lora_rank, lora_alpha = 64, 128
    for path in [move_adapter, grip_adapter, adapter_path]:
        if path:
            lora_rank, lora_alpha = _read_lora_config(path)
            break

    is_peft = adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

    if is_peft:
        from peft import PeftModel

        logger.info("Loading GROOT N1.5 base for PEFT adapter...")
        policy = GrootPolicy.from_pretrained("nvidia/GR00T-N1.5-3B")
        policy = policy.to(dtype=torch.bfloat16)

        logger.info(f"Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(policy, adapter_path)
        model = model.to(device)
        model.eval()

        from lerobot.policies.groot.processor_groot import _build_eagle_processor
        eagle_processor = _build_eagle_processor()
        return model, eagle_processor, {}

    logger.info(f"Loading GROOT N1.5 base (lora_rank={lora_rank}, lora_alpha={lora_alpha})...")
    policy = GrootPolicy.from_pretrained(
        "nvidia/GR00T-N1.5-3B",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    adapter_weights = {}

    if move_adapter and grip_adapter:
        for name, path in [("move", move_adapter), ("grip", grip_adapter)]:
            weights_path = _find_weights_file(path)
            if weights_path:
                adapter_weights[name] = _load_weights_file(weights_path)
                logger.info(f"Loaded {name} adapter: {len(adapter_weights[name])} keys from {weights_path}")
            else:
                logger.warning(f"No weights found for {name} adapter at {path}")

        if "move" in adapter_weights:
            policy.load_state_dict(adapter_weights["move"], strict=False)
            logger.info("Applied move adapter as default")

    elif adapter_path:
        weights_path = _find_weights_file(adapter_path)
        if weights_path:
            weights = _load_weights_file(weights_path)
            missing, unexpected = policy.load_state_dict(weights, strict=False)
            logger.info(f"Loaded adapter: {len(weights)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            logger.warning(f"No weights found at {adapter_path}, using base model")

    model = policy.to(dtype=torch.bfloat16, device=device)
    model.eval()

    from lerobot.policies.groot.processor_groot import _build_eagle_processor
    eagle_processor = _build_eagle_processor()
    return model, eagle_processor, adapter_weights


def load_smolvla(adapter_path, device):
    """Load SmolVLA with PEFT LoRA adapter."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from peft import PeftModel
    from transformers import AutoTokenizer

    logger.info("Loading SmolVLA base model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(dtype=torch.bfloat16)

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
    return model, tokenizer, None


# --- Direct mode: batch preparation ---

def prepare_pi05_batch(image, state_16d, instruction, state_stats, action_stats, tokenizer, device):
    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

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
    from PIL import Image as PILImage

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
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    if eagle_processor is not None:
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
    import torch.nn.functional as F

    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    if img_t.shape[-1] != 512 or img_t.shape[-2] != 512:
        img_t = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)

    s_mean = np.array(state_stats["mean"], dtype=np.float32)
    s_std = np.array(state_stats["std"], dtype=np.float32)
    state_norm = normalize_state_mean_std(state_16d, s_mean, s_std)

    text = instruction + "\n"
    tokens = tokenizer(text, max_length=48, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "observation.images.camera1": img_t.to(device),
        "observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }


# --- Direct mode: inference ---

def stack_batches(batch_list):
    if len(batch_list) == 1:
        return batch_list[0]
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in batch_list[0]}


@torch.no_grad()
def predict_action_chunk(model, batch, **kwargs):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch, **kwargs)
    return actions.float().cpu().numpy()


def switch_adapter(model, adapter_weights, adapter_name):
    if adapter_name in adapter_weights:
        model.load_state_dict(adapter_weights[adapter_name], strict=False)


# --- Direct mode: batched episode rollout ---

def _create_vlm_monitor(provider: str, model_name: str):
    """Create a VLM client for direct-mode monitoring."""
    if not provider:
        return None
    if provider == "bedrock":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from robobridge.modules.bedrock_bearer import create_bedrock_bearer_chat
        return create_bedrock_bearer_chat(model=model_name, temperature=0.0, max_tokens=256)
    if provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=model_name, project="prism-485101",
            location="global", temperature=0.0, max_output_tokens=256,
        )
    return None


def _vlm_check_task_failed(vlm_client, image: np.ndarray, instruction: str) -> bool:
    """Ask VLM if the robot appears stuck/failing. Returns True if clearly failing."""
    import base64
    import cv2
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
    b64 = base64.b64encode(buf).decode()
    msg = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Task: \"{instruction}\"\n"
                    "Look at this robot image. Is the robot clearly failing or stuck "
                    "(e.g., not near the target, pushing against a wall, gripper empty when it should hold something)?\n"
                    "Answer ONLY 'FAILING' or 'OK'."
                ),
            },
        ],
    }
    try:
        resp = vlm_client.invoke([msg])
        answer = resp.content.strip().upper()
        return "FAILING" in answer
    except Exception as e:
        logger.warning(f"VLM monitor check failed: {e}")
        return False


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
    adapter_weights: Optional[Dict] = None,
    record_video: bool = False,
    ep_offset: int = 0,
    state_dim: int = 12,
    action_ema_alpha: float = 0.0,
    match_training: bool = False,
    action_scale: float = 1.0,
    temporal_ensemble: bool = False,
    stuck_max_retries: int = 0,
    vlm_check_interval: int = 0,
    vlm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Run batch_size episodes in parallel (direct mode)."""
    n = batch_size

    envs = []
    for i in range(n):
        try:
            envs.append(create_env(task_name, image_size=image_size, match_training=match_training))
        except Exception as e:
            logger.error(f"Failed to create env #{i}: {e}")
            for env in envs:
                env.close()
            return [{"success": False, "steps": 0, "total_reward": 0.0}] * n

    obs_list = []
    env_instructions = []  # per-env instruction from ep_meta
    for env in envs:
        for _reset_attempt in range(5):
            try:
                obs_list.append(env.reset())
                break
            except Exception as e:
                logger.warning(f"env.reset() attempt {_reset_attempt+1}/5 failed: {e}")
                if _reset_attempt == 4:
                    obs_list.append(env.reset())  # let it raise
        # Extract correct instruction from env metadata (handles directional tasks)
        try:
            ep_meta = env.get_ep_meta()
            env_lang = ep_meta.get("lang", instruction) if isinstance(ep_meta, dict) else instruction
        except Exception:
            env_lang = instruction
        env_instructions.append(env_lang)
        logger.info(f"  env instruction: {env_lang}")

    action_chunks = [None] * n
    chunk_idxs = [0] * n
    active = [True] * n
    ep_successes = [False] * n
    ep_steps = [0] * n
    ep_rewards = [0.0] * n
    ep_frames = [[] for _ in range(n)]
    prev_states = [None] * n
    no_change_counts = [0] * n
    prev_actions = [None] * n  # for EMA smoothing
    stuck_retries = [0] * n  # monitor: stuck retry counter
    vlm_step_counters = [0] * n  # monitor: steps since last VLM check

    # Temporal ensemble buffers
    # For each env, stores a dict: timestep -> list of (action_array, weight)
    ensemble_buffers = [{} for _ in range(n)] if temporal_ensemble else None
    global_steps = [0] * n  # Track global timestep for each env

    _batch_start = time.time()
    for step in range(max_horizon):
        if not any(active):
            break

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

        needs_chunk = [
            i for i in range(n)
            if active[i] and (action_chunks[i] is None or chunk_idxs[i] >= chunk_stride)
        ]

        if needs_chunk:
            if adapter_weights and "move" in adapter_weights:
                switch_adapter(model, adapter_weights, "move")

            batch_dicts = []
            valid_indices = []
            for i in needs_chunk:
                image = extract_image(obs_list[i], size=image_size)
                state_vec = extract_state_16d_hdf5(obs_list[i]) if state_dim == 16 else extract_state_12d(obs_list[i])
                # Use per-env instruction (from ep_meta) for correct directional tasks
                env_instr = env_instructions[i] if env_instructions else instruction
                try:
                    if model_name == "pi05":
                        bd = prepare_pi05_batch(image, state_vec, env_instr, state_stats, action_stats, tokenizer, device)
                    elif model_name == "groot":
                        # Dual adapter (movegrip) trained with primitive instructions
                        # Single adapter (HDF5) trained with task-level instructions
                        vla_instr = "move" if adapter_weights and "move" in adapter_weights else env_instr
                        bd = prepare_groot_batch(image, state_vec, vla_instr, state_stats, eagle_processor, device)
                    elif model_name == "smolvla":
                        bd = prepare_smolvla_batch(image, state_vec, env_instr, state_stats, tokenizer, device)
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
                    kwargs = {"num_steps": 10} if model_name in ("pi05", "smolvla") else {}
                    all_chunks = predict_action_chunk(model, stacked, **kwargs)
                    for j, idx in enumerate(valid_indices):
                        action_chunks[idx] = all_chunks[j]
                        chunk_idxs[idx] = 0
                        # Add to temporal ensemble buffer
                        if temporal_ensemble:
                            chunk = all_chunks[j]  # (chunk_size, action_dim)
                            g = global_steps[idx]
                            for t in range(chunk.shape[0]):
                                future_step = g + t
                                if future_step not in ensemble_buffers[idx]:
                                    ensemble_buffers[idx][future_step] = []
                                ensemble_buffers[idx][future_step].append(chunk[t].copy())
                except Exception as e:
                    logger.error(f"Batched inference error at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        for i in range(n):
            if not active[i]:
                continue
            if action_chunks[i] is None or chunk_idxs[i] >= chunk_stride:
                continue

            if temporal_ensemble and global_steps[i] in ensemble_buffers[i]:
                # Average all predictions for this timestep
                preds = ensemble_buffers[i][global_steps[i]]
                n_preds = len(preds)
                if n_preds > 1:
                    # Exponential weighting: newer predictions get higher weight
                    weights = np.array([np.exp(0.01 * k) for k in range(n_preds)])
                    weights /= weights.sum()
                    avg_action = sum(w * p for w, p in zip(weights, preds))
                else:
                    avg_action = preds[0]
                raw_action = denormalize_action(avg_action, action_stats, norm_type)
                # Clean up old entries
                del ensemble_buffers[i][global_steps[i]]
                global_steps[i] += 1
                chunk_idxs[i] += 1
            else:
                raw_action = denormalize_action(action_chunks[i][chunk_idxs[i]], action_stats, norm_type)
                chunk_idxs[i] += 1
                if temporal_ensemble:
                    global_steps[i] += 1
            # Apply action scaling
            if action_scale != 1.0:
                raw_action[:6] *= action_scale  # Scale pos+rot, not gripper
            # Apply EMA smoothing
            if action_ema_alpha > 0 and prev_actions[i] is not None:
                raw_action = action_ema_alpha * prev_actions[i] + (1 - action_ema_alpha) * raw_action
            prev_actions[i] = raw_action.copy()
            action_12d = action_7d_to_12d(raw_action)

            if record_video:
                ep_frames[i].append(extract_image(obs_list[i], size=image_size))

            obs_list[i], reward, done, info = envs[i].step(action_12d)
            ep_rewards[i] += reward
            ep_steps[i] = step + 1

            try:
                if envs[i]._check_success():
                    ep_successes[i] = True
                    active[i] = False
            except Exception:
                pass

            current_state = extract_state_16d_hdf5(obs_list[i]) if state_dim == 16 else extract_state_12d(obs_list[i])
            if prev_states[i] is not None:
                if np.abs(current_state - prev_states[i]).max() < 1e-4:
                    no_change_counts[i] += 1
                else:
                    no_change_counts[i] = 0
                if no_change_counts[i] >= 50:
                    if stuck_retries[i] < stuck_max_retries:
                        # STUCK → reset model state and retry
                        stuck_retries[i] += 1
                        no_change_counts[i] = 0
                        action_chunks[i] = None
                        chunk_idxs[i] = 0
                        prev_actions[i] = None
                        # Reset model internal state (clear chunk buffer / KV cache)
                        for obj in [model, getattr(model, "base_model", None)]:
                            if obj is not None and hasattr(obj, "reset"):
                                obj.reset()
                        logger.info(f"  [STUCK] env{i} retry {stuck_retries[i]}/{stuck_max_retries} at step {step}")
                    else:
                        active[i] = False
            prev_states[i] = current_state

            # VLM periodic task-failure check
            if vlm_client and vlm_check_interval > 0:
                vlm_step_counters[i] += 1
                if vlm_step_counters[i] >= vlm_check_interval:
                    vlm_step_counters[i] = 0
                    img = extract_image(obs_list[i], size=image_size)
                    env_instr = env_instructions[i] if env_instructions else instruction
                    if _vlm_check_task_failed(vlm_client, img, env_instr):
                        if stuck_retries[i] < stuck_max_retries:
                            stuck_retries[i] += 1
                            no_change_counts[i] = 0
                            action_chunks[i] = None
                            chunk_idxs[i] = 0
                            prev_actions[i] = None
                            for obj in [model, getattr(model, "base_model", None)]:
                                if obj is not None and hasattr(obj, "reset"):
                                    obj.reset()
                            logger.info(f"  [VLM-FAIL] env{i} retry {stuck_retries[i]}/{stuck_max_retries} at step {step}")
                        else:
                            active[i] = False
                            logger.info(f"  [VLM-FAIL] env{i} max retries reached, giving up")

    print()

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


def evaluate_direct(
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
    move_adapter_override: Optional[str] = None,
    grip_adapter_override: Optional[str] = None,
    stats_file: Optional[str] = None,
    image_size_override: Optional[int] = None,
    chunk_stride_override: Optional[int] = None,
    action_ema_alpha: float = 0.0,
    match_training: bool = False,
    action_scale: float = 1.0,
    denoise_steps: Optional[int] = None,
    temporal_ensemble: bool = False,
    stuck_max_retries: int = 0,
    vlm_check_interval: int = 0,
    monitor_provider: str = "",
    monitor_model: str = "",
) -> Dict[str, Any]:
    """Evaluate a model on a task (direct mode)."""
    cfg = MODEL_CONFIGS[model_name].copy()
    if image_size_override:
        cfg["image_size"] = image_size_override
    if chunk_stride_override:
        cfg["chunk_stride"] = chunk_stride_override
    instruction = TASK_INSTRUCTIONS[task_name]

    # Resolve adapter paths
    ckpt_name = f"checkpoint-{checkpoint}" if checkpoint != "best" else "checkpoint-best"
    move_adapter = None
    grip_adapter = None
    adapter_path = None

    if move_adapter_override and grip_adapter_override:
        move_adapter = move_adapter_override
        grip_adapter = grip_adapter_override
    elif model_name == "groot" and cfg.get("move_adapter_dir") and not adapter_dir_override:
        move_dir = cfg["move_adapter_dir"] / ckpt_name
        grip_dir = cfg["grip_adapter_dir"] / ckpt_name
        if move_dir.exists() and grip_dir.exists():
            move_adapter = str(move_dir)
            grip_adapter = str(grip_dir)
        else:
            base_dir = Path(adapter_dir_override) if adapter_dir_override else cfg["adapter_dir"]
            adapter_path = str(base_dir / ckpt_name)
    else:
        base_dir = Path(adapter_dir_override) if adapter_dir_override else cfg["adapter_dir"]
        adapter_path = str(base_dir / ckpt_name)

    for label, path in [("move_adapter", move_adapter), ("grip_adapter", grip_adapter), ("adapter", adapter_path)]:
        if path and not os.path.exists(path):
            logger.error(f"{label} not found: {path}")
            return {"success_rate": -1, "error": f"{label}_not_found", "path": path}

    if stats_file and os.path.exists(stats_file):
        with open(stats_file) as f:
            custom_stats = json.load(f)
        action_stats = custom_stats["action_stats"]
        state_stats = custom_stats["state_stats"]
        # Merge quantile stats if present (like load_normalization_stats does)
        if "action_quantile_stats" in custom_stats:
            action_stats["q01"] = custom_stats["action_quantile_stats"]["q01"]
            action_stats["q99"] = custom_stats["action_quantile_stats"]["q99"]
        if "q01" not in state_stats and "state_quantile_stats" in custom_stats:
            state_stats["q01"] = custom_stats["state_quantile_stats"]["q01"]
            state_stats["q99"] = custom_stats["state_quantile_stats"]["q99"]
        logger.info(f"Loaded custom stats from {stats_file} (action_dim={len(action_stats['min'])}, state_dim={len(state_stats['min'])})")
    else:
        stats = load_normalization_stats()
        action_stats = stats["action_stats"]
        state_stats = stats["state_stats"]

    state_dim = len(state_stats["min"])

    if move_adapter and grip_adapter:
        print(f"  Loading {model_name} with move/grip adapters...", flush=True)
        print(f"    move: {move_adapter}", flush=True)
        print(f"    grip: {grip_adapter}", flush=True)
    else:
        print(f"  Loading {model_name} from {adapter_path}...", flush=True)

    tokenizer = None
    eagle_processor = None
    adapter_weights = {}

    try:
        if model_name == "pi05":
            model, tokenizer, _ = load_pi05(adapter_path, device)
        elif model_name == "groot":
            model, eagle_processor, adapter_weights = load_groot(
                adapter_path=adapter_path,
                move_adapter=move_adapter,
                grip_adapter=grip_adapter,
                device=device,
            )
            if denoise_steps and hasattr(model, '_groot_model'):
                model._groot_model.action_head.num_inference_timesteps = denoise_steps
                logger.info(f"Overriding GROOT denoising steps to {denoise_steps}")
        elif model_name == "smolvla":
            model, tokenizer, _ = load_smolvla(adapter_path, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return {"success_rate": -1, "error": str(e)}

    adapter_mode = "move/grip" if adapter_weights else "single"
    print(f"  Model loaded. (adapter_mode={adapter_mode})", flush=True)

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

    # Create VLM monitor client for direct mode
    vlm_client = None
    if monitor_provider and (stuck_max_retries > 0 or vlm_check_interval > 0):
        vlm_client = _create_vlm_monitor(monitor_provider, monitor_model)
        if vlm_client:
            logger.info(f"Direct-mode monitor: provider={monitor_provider}, model={monitor_model}, "
                        f"stuck_retries={stuck_max_retries}, vlm_interval={vlm_check_interval}")

    episode_results = []
    successes = 0
    record_video = save_video_dir is not None
    episodes_done = 0

    while episodes_done < num_episodes:
        bs = min(batch_size, num_episodes - episodes_done)
        print(f"\n  Batch [{episodes_done+1}-{episodes_done+bs}] / {num_episodes}  (batch_size={bs})")

        for obj in [model, getattr(model, "base_model", None)]:
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()
            if obj is not None and hasattr(obj, "model") and hasattr(obj.model, "reset"):
                obj.model.reset()

        batch_results = run_episodes_batched(
            model=model, model_name=model_name,
            task_name=task_name, instruction=instruction,
            action_stats=action_stats, state_stats=state_stats,
            norm_type=action_stats.get("mode", cfg["norm_type"]), image_size=cfg["image_size"],
            chunk_stride=cfg["chunk_stride"], max_horizon=max_horizon,
            device=device, batch_size=bs,
            tokenizer=tokenizer, eagle_processor=eagle_processor,
            adapter_weights=adapter_weights,
            record_video=record_video, ep_offset=episodes_done,
            state_dim=state_dim,
            action_ema_alpha=action_ema_alpha,
            match_training=match_training,
            action_scale=action_scale,
            temporal_ensemble=temporal_ensemble,
            stuck_max_retries=stuck_max_retries,
            vlm_check_interval=vlm_check_interval,
            vlm_client=vlm_client,
        )

        for i, result in enumerate(batch_results):
            ep_idx = episodes_done + i + 1
            result["episode"] = ep_idx

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

    del model
    torch.cuda.empty_cache()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    print(f"\n  \033[1m{model_name}/{task_name}: {successes}/{num_episodes} = {success_rate:.1%}\033[0m\n")

    return {
        "model": model_name,
        "task": task_name,
        "checkpoint": checkpoint,
        "adapter_mode": adapter_mode,
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "episodes": episode_results,
    }


# ===========================================================================
# CYCLEVLA MODE: MBR + Subtask FSM + VLM Failure Prediction + Backtracking
# ===========================================================================


def run_episode_cyclevla(
    model,
    eagle_processor,
    adapter_weights,
    env,
    obs: Dict,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    max_steps: int = 500,
    image_size: int = 224,
    denoise_steps: Optional[int] = None,
    mbr_samples: int = 8,
    stop_threshold: float = 0.5,
    stop_complete_threshold: float = 0.8,
    max_backtracks: int = 2,
    use_vlm_predictor: bool = True,
    vlm_provider: str = "bedrock",
    vlm_model: str = "eu.anthropic.claude-sonnet-4-6-v1",
    n_subtasks: int = 2,
    task_name: str = "",
    record_video: bool = False,
    action_scale: float = 1.0,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
) -> Dict[str, Any]:
    """Run one episode with CycleVLA: MBR decoding + subtask state machine.

    Uses the CycleVLARunner for intelligent action selection and failure recovery.
    """
    from robobridge.modules.controller.cyclevla import (
        CycleVLAConfig, CycleVLARunner, VLMFailurePredictor,
    )

    config = CycleVLAConfig(
        mbr_samples=mbr_samples,
        stop_threshold=stop_threshold,
        stop_complete_threshold=stop_complete_threshold,
        max_backtracks=max_backtracks,
        use_vlm_predictor=use_vlm_predictor,
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        action_dim=9,
    )

    runner = CycleVLARunner(config=config)
    runner.initialize()
    runner.reset_episode(task_description=instruction, n_subtasks=n_subtasks)

    # Build normalization arrays
    a_min = np.array(action_stats.get("q01", action_stats["min"])[:7], dtype=np.float32)
    a_max = np.array(action_stats.get("q99", action_stats["max"])[:7], dtype=np.float32)
    s_min = np.array(state_stats.get("q01", state_stats["min"]), dtype=np.float32)
    s_max = np.array(state_stats.get("q99", state_stats["max"]), dtype=np.float32)

    frames = []
    prev_action_7d = None
    total_reward = 0.0

    # Save initial checkpoint
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    gripper_val = obs.get("robot0_gripper_qpos", np.array([0.0]))[0] if "robot0_gripper_qpos" in obs else 0.0
    runner.save_checkpoint(eef_pos, gripper_val)

    for step_i in range(max_steps):
        if step_i % 50 == 0:
            print(f"    step {step_i}/{max_steps}, subtask={runner.state_machine.current_subtask_idx if runner.state_machine else '?'}", flush=True)

        # Build GROOT batch
        batch = _build_cyclevla_batch(
            obs, model, eagle_processor, instruction, image_size,
            s_min, s_max, denoise_steps,
        )

        # CycleVLA step: MBR decode + state machine
        action_9d_norm, meta = runner.step(
            obs={"image": obs.get("robot0_agentview_left_image", obs.get("robot0_agentview_image"))},
            policy=model,
            batch=batch,
        )

        # Denormalize action (only first 7D)
        action_7d = action_9d_norm.copy()
        action_7d = denormalize_action_quantile(action_7d, a_min, a_max)  # unnormalize from [-1, 1]

        # Apply EMA smoothing
        if prev_action_7d is not None and ema_alpha > 0:
            action_7d = ema_alpha * prev_action_7d + (1 - ema_alpha) * action_7d
        prev_action_7d = action_7d.copy()

        # Apply action scale
        action_7d[:6] *= action_scale

        # Handle backtracking
        if meta.get("backtrack_requested"):
            checkpoint = runner.get_backtrack_target()
            if checkpoint is not None:
                logger.info(f"  Backtracking to checkpoint at pos={checkpoint.eef_pos}")
                # Move back to checkpoint position using small steps
                for _ in range(30):
                    curr_pos = obs.get("robot0_eef_pos", np.zeros(3))
                    delta = checkpoint.eef_pos - curr_pos
                    if np.linalg.norm(delta) < 0.02:
                        break
                    bt_action = np.zeros(7)
                    bt_action[:3] = np.clip(delta * 2.0, -0.3, 0.3)
                    bt_action[6] = -1.0  # open gripper
                    action_12d = action_7d_to_12d(bt_action)
                    obs, reward, done, info = env.step(action_12d)
                    if done:
                        break
                # Save new checkpoint for retry
                eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
                runner.save_checkpoint(eef_pos, gripper_val)
                continue

        # Handle subtask advancement - save new checkpoint
        if meta.get("subtask_advanced"):
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            gripper_val = obs.get("robot0_gripper_qpos", np.array([0.0]))[0] if "robot0_gripper_qpos" in obs else 0.0
            runner.save_checkpoint(eef_pos, gripper_val)

        # Convert to 12D and step env
        action_12d = action_7d_to_12d(action_7d)
        obs, reward, done, info = env.step(action_12d)
        total_reward += reward

        if record_video:
            try:
                rgb = env.sim.render(camera_name="robot0_agentview_left", height=480, width=480)
                frames.append(np.flip(rgb, axis=0).copy())
            except Exception:
                pass

        if done:
            break

    task_success = False
    try:
        task_success = env._check_success()
    except Exception:
        pass

    stats = runner.get_stats()

    return {
        "success": task_success,
        "steps": step_i + 1,
        "reward": total_reward,
        "frames": frames if record_video else [],
        "cyclevla_stats": stats,
    }


def _build_cyclevla_batch(obs, model, eagle_processor, instruction, image_size,
                          s_min, s_max, denoise_steps):
    """Build GROOT input batch for CycleVLA MBR decoding."""
    import torch
    from PIL import Image as PILImage

    device = next(model.parameters()).device

    # Get image
    rgb = obs.get("robot0_agentview_left_image", obs.get("robot0_agentview_image"))
    if rgb is None:
        raise ValueError("No camera image found in observation")

    # Process image with Eagle
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        pil_img = PILImage.fromarray(rgb).resize((image_size, image_size), PILImage.BILINEAR)
        rgb = np.array(pil_img)

    # Build robot state (12D normalized)
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.array([0.04, 0.04], dtype=np.float32))
    base = obs.get("robot0_base_pos", np.zeros(3))
    state_12d = np.concatenate([eef_pos, eef_quat, gripper[:2], base[:3]]).astype(np.float32)

    # Normalize
    state_norm = np.where(
        (s_max - s_min) > 1e-6,
        2.0 * (state_12d - s_min) / (s_max - s_min) - 1.0,
        0.0,
    ).astype(np.float32)

    # Pad to (1, 1, 64) — GROOT expects (B, 1, 64)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm
    state_tensor = torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0).to(device=device)

    # State mask (1, 1, 64)
    state_mask_np = np.zeros(64, dtype=bool)
    state_mask_np[:len(state_norm)] = True
    state_mask_tensor = torch.from_numpy(state_mask_np).unsqueeze(0).unsqueeze(0).to(device=device)

    batch = {
        "state": state_tensor,
        "state_mask": state_mask_tensor,
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    # Eagle image encoding — include ALL keys (input_ids, attention_mask, pixel_values, image_sizes)
    if eagle_processor is not None:
        eagle_inputs = eagle_processor(
            text=[f"<image-1> {instruction}"],
            images=[PILImage.fromarray(rgb)],
            images_kwargs={
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 12,
                "use_thumbnail": False,
            },
            return_tensors="pt",
            padding=True,
        )
        for k, v in eagle_inputs.items():
            batch[f"eagle_{k}"] = v.to(device)
    else:
        pixel_values = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16) / 255.0
        batch["eagle_pixel_values"] = pixel_values

    # Override denoise steps if specified
    if denoise_steps and hasattr(model, "config"):
        if hasattr(model.config, "num_inference_timesteps"):
            model.config.num_inference_timesteps = denoise_steps

    return batch


def _get_perception_target_rb(env, obs, task_name):
    """Get target fixture interaction-site position in robot-base frame.

    Uses RoboCasaPerception to find the target fixture's interaction site
    (handle, knob, spout, etc.) and returns its position in robot-base frame.
    This matches the pipeline's perception_target used in _generate_template_plan().
    """
    try:
        from robobridge.wrappers.robocasa_perception import RoboCasaPerception
        ep_meta = env.get_ep_meta() if hasattr(env, 'get_ep_meta') else {}
        perception = RoboCasaPerception()
        perception.set_environment_state(obs, ep_meta, env)
        detections = perception._detect_interaction_sites()
        for det in detections:
            if det.metadata.get("role") == "target":
                pos = det.pose["position"]
                target_rb = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
                logger.info(f"[HYBRID] Perception target (robot-base): {target_rb}")
                return target_rb
        logger.warning(f"[HYBRID] No target detection found for {task_name}")
    except Exception as e:
        logger.warning(f"[HYBRID] Failed to get perception target: {e}")
    return None


def _build_hybrid_batch(obs, model, eagle_processor, instruction, image_size,
                        s_min, s_max, denoise_steps, direction_vector=None,
                        target_pos_rb=None):
    """Build GROOT input batch for Hybrid CycleVLA with direction_vector injection.

    Same as _build_cyclevla_batch but replaces state[9:12] (robot0_base_pos)
    with direction_vector from template plan config. This provides the model
    with navigation direction, matching pipeline mode's state construction.

    Args:
        direction_vector: [X, Y, Z] direction vector. None values in Y are
                         computed from perception (eef_pos Y toward target).
    """
    import torch
    from PIL import Image as PILImage

    device = next(model.parameters()).device

    # Get image
    rgb = obs.get("robot0_agentview_left_image", obs.get("robot0_agentview_image"))
    if rgb is None:
        raise ValueError("No camera image found in observation")

    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        pil_img = PILImage.fromarray(rgb).resize((image_size, image_size), PILImage.BILINEAR)
        rgb = np.array(pil_img)

    # Build robot state (12D) — matching pipeline's _build_state_vector()
    # State: eef_base(3) + eef_quat_base(4) + gripper(2) + delta_base(3)
    from scipy.spatial.transform import Rotation as Rot

    eef_world = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)
    eef_quat_world = np.array(obs.get("robot0_eef_quat", [1, 0, 0, 0]), dtype=np.float64)
    gripper = obs.get("robot0_gripper_qpos", np.array([0.04, 0.04], dtype=np.float32))
    base_pos = obs.get("robot0_base_pos", None)
    base_quat = obs.get("robot0_base_quat", None)

    # Compute eef_base: robot-centric end-effector position
    # Same as pipeline: eef_base = R_inv @ (eef_world - base_pos)
    gt_eef_base = obs.get("robot0_base_to_eef_pos", None)
    if gt_eef_base is not None:
        eef_base = np.array(gt_eef_base, dtype=np.float32)
        if base_pos is not None and base_quat is not None:
            rot_inv = Rot.from_quat(np.array(base_quat, dtype=np.float64)).inv()
        else:
            rot_inv = None
    elif base_pos is not None and base_quat is not None:
        base_pos_np = np.array(base_pos, dtype=np.float64)
        rot_inv = Rot.from_quat(np.array(base_quat, dtype=np.float64)).inv()
        eef_base = rot_inv.apply(eef_world - base_pos_np).astype(np.float32)
    else:
        eef_base = eef_world.astype(np.float32)
        rot_inv = None

    # Transform quaternion to base frame
    if rot_inv is not None:
        quat_base = (rot_inv * Rot.from_quat(eef_quat_world)).as_quat().astype(np.float32)
    else:
        quat_base = eef_quat_world.astype(np.float32)

    # Direction vector injection (delta_base at indices 9:12)
    if direction_vector is not None:
        dir_vec = np.array(direction_vector, dtype=np.float32)
        # Handle null Y: compute dynamically from perception target (matches pipeline)
        # Pipeline: dvy = target_pos.y - eef_base.y (vla_lora_controller.py:610-612)
        if np.isnan(dir_vec[1]):
            if target_pos_rb is not None:
                dir_vec[1] = float(target_pos_rb[1]) - float(eef_base[1])
            else:
                dir_vec[1] = 0.0  # fallback
        # Unit-normalize direction vector — matches pipeline's _build_state_vector()
        # Training data uses unit-normalized directions, so state stats expect norm≈1.0
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > 1e-6:
            dir_vec = dir_vec / dir_norm
    else:
        dir_vec = np.zeros(3, dtype=np.float32)

    state_12d = np.concatenate([eef_base, quat_base, gripper[:2], dir_vec]).astype(np.float32)

    # Normalize — matches pipeline's _normalize_and_pad_state():
    # r = s_max - s_min + 1e-8; state_norm = clip(2*(state-s_min)/r - 1, -1, 1)
    r = s_max - s_min + 1e-8
    state_norm = np.clip(2.0 * (state_12d - s_min) / r - 1.0, -1.0, 1.0).astype(np.float32)

    # Pad to (1, 1, 64) — GROOT expects (B, 1, 64)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm
    state_tensor = torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0).to(device=device)

    # State mask (1, 1, 64)
    state_mask_np = np.zeros(64, dtype=bool)
    state_mask_np[:len(state_norm)] = True
    state_mask_tensor = torch.from_numpy(state_mask_np).unsqueeze(0).unsqueeze(0).to(device=device)

    batch = {
        "state": state_tensor,
        "state_mask": state_mask_tensor,
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    # Eagle image encoding
    if eagle_processor is not None:
        eagle_inputs = eagle_processor(
            text=[f"<image-1> {instruction}"],
            images=[PILImage.fromarray(rgb)],
            images_kwargs={
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 12,
                "use_thumbnail": False,
            },
            return_tensors="pt",
            padding=True,
        )
        for k, v in eagle_inputs.items():
            batch[f"eagle_{k}"] = v.to(device)
    else:
        pixel_values = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16) / 255.0
        batch["eagle_pixel_values"] = pixel_values

    # Override denoise steps if specified
    if denoise_steps and hasattr(model, "config"):
        if hasattr(model.config, "num_inference_timesteps"):
            model.config.num_inference_timesteps = denoise_steps

    return batch


# ===========================================================================
# HYBRID CYCLEVLA MODE: MBR + direction_vector + budget FSM + VLM checks
# ===========================================================================


def run_episode_hybrid_cyclevla(
    model,
    eagle_processor,
    adapter_weights,
    env,
    obs: Dict,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    template_primitives: List[Dict],
    max_steps: int = 500,
    image_size: int = 224,
    denoise_steps: Optional[int] = None,
    mbr_samples: int = 8,
    max_backtracks: int = 2,
    vlm_check_interval: int = 50,
    vlm_client: Any = None,
    task_name: str = "",
    record_video: bool = False,
    action_scale: float = 1.0,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    target_pos_rb: Optional[np.ndarray] = None,
    heuristic_monitor: bool = False,
    monitor_stuck_window: int = 25,
    monitor_stuck_threshold: float = 0.003,
    monitor_warmup: int = 30,
) -> Dict[str, Any]:
    """Run one episode with Hybrid CycleVLA.

    Combines:
    - Pipeline's direction_vector state injection (proven 40% baseline)
    - CycleVLA's MBR decoding for robust consensus actions
    - Budget-based subtask transitions from template plan
    - VLM failure detection + backtracking for recovery

    Args:
        template_primitives: List of primitive dicts from task_plan_config.json
        vlm_client: LangChain-style VLM client for failure checks (or None)
        target_pos_rb: Target fixture position in robot-base frame (for dynamic Y)
    """
    from robobridge.modules.controller.cyclevla import (
        HybridSubtaskStateMachine, MBRDecoder, SubtaskCheckpoint, SubtaskState,
    )

    # Setup MBR decoder (7D actions)
    mbr_decoder = MBRDecoder(n_samples=mbr_samples, action_dim=7)

    # Setup budget-based state machine
    fsm = HybridSubtaskStateMachine(primitives=template_primitives)

    # Build normalization arrays
    # Action: use min/max for denormalization — matches training mode="min_max"
    # (NOT q01/q99 which gives smaller actions and wrong signs)
    a_min = np.array(action_stats["min"][:7], dtype=np.float32)
    a_max = np.array(action_stats["max"][:7], dtype=np.float32)
    # State: use min/max (NOT q01/q99) — matches pipeline's _normalize_and_pad_state()
    s_min = np.array(state_stats["min"], dtype=np.float32)
    s_max = np.array(state_stats["max"], dtype=np.float32)

    frames = []
    prev_action_7d = None
    total_reward = 0.0
    total_steps = 0
    backtrack_counts: Dict[int, int] = {}
    # Heuristic monitor state
    eef_history: List[np.ndarray] = []
    checkpoints: List[SubtaskCheckpoint] = []

    # Save initial checkpoint
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    gripper_val = obs.get("robot0_gripper_qpos", np.array([0.0]))[0] if "robot0_gripper_qpos" in obs else 0.0
    checkpoints.append(SubtaskCheckpoint(eef_pos=eef_pos.copy(), gripper_state=gripper_val, subtask_idx=0))

    # MBR chunk buffer
    chunk_buffer = None
    chunk_step = 0

    while not fsm.is_done and total_steps < max_steps:
        prim = fsm.current_primitive
        if prim is None:
            break

        prim_type = prim.get("type", "move")
        prim_idx = fsm.current_idx
        budget = prim.get("steps_budget", 250)
        # v4: Cap budget when monitor is enabled → more retry attempts
        if heuristic_monitor and prim_type != "grip":
            budget = min(budget, 150)

        # Get direction vector for this primitive
        direction_vector = prim.get("direction_vector", prim.get("direction_target"))
        # Convert null values to NaN for _build_hybrid_batch
        if direction_vector is not None:
            direction_vector = [float('nan') if v is None else float(v) for v in direction_vector]

        # Select adapter based on primitive type
        if prim_type == "grip":
            if "grip" in adapter_weights:
                switch_adapter(model, adapter_weights, "grip")
        else:
            if "move" in adapter_weights:
                switch_adapter(model, adapter_weights, "move")

        logger.info(
            f"Hybrid CycleVLA: primitive {prim_idx+1}/{len(template_primitives)} "
            f"type={prim_type} budget={budget} dir={direction_vector}"
        )

        # Execute this primitive
        prim_steps = 0
        while prim_steps < budget and total_steps < max_steps:
            if total_steps % 50 == 0:
                print(
                    f"    step {total_steps}/{max_steps}, "
                    f"prim={prim_idx+1}/{len(template_primitives)} ({prim_type}), "
                    f"prim_step={prim_steps}/{budget}",
                    flush=True,
                )

            # Determine if we need a new MBR chunk
            need_new_chunk = (chunk_buffer is None or chunk_step >= chunk_stride)

            if need_new_chunk:
                # Use primitive type as instruction (matches training: "move"/"grip")
                prim_instruction = prim_type  # "move" or "grip"
                # Build batch with direction_vector injection
                batch = _build_hybrid_batch(
                    obs, model, eagle_processor, prim_instruction, image_size,
                    s_min, s_max, denoise_steps,
                    direction_vector=direction_vector,
                    target_pos_rb=target_pos_rb,
                )

                if mbr_samples <= 1:
                    # Single inference — use pipeline's exact path
                    chunk_buffer = predict_action_chunk(model, batch)[0, :, :7]  # (T, 7)
                else:
                    # MBR decode: get consensus action chunk (7D)
                    action_chunk = mbr_decoder.decode(model, batch)  # (T, 7)
                    chunk_buffer = action_chunk
                chunk_step = 0

            # Get action from chunk
            if chunk_buffer is not None and chunk_step < len(chunk_buffer):
                action_7d_norm = chunk_buffer[chunk_step]
                chunk_step += 1
            else:
                action_7d_norm = np.zeros(7)

            # Clip to [-1, 1] before denormalization (same as pipeline)
            action_7d_norm = np.clip(action_7d_norm[:7], -1.0, 1.0)
            # Denormalize (min_max mode — matches training)
            action_7d = denormalize_action_min_max(action_7d_norm, a_min, a_max)

            # Scale FIRST (matches pipeline controller order)
            if action_scale != 1.0:
                action_7d[:6] *= action_scale

            # EMA on first 6 dims only (gripper excluded, matches pipeline)
            if prev_action_7d is not None and ema_alpha > 0:
                action_7d[:6] = ema_alpha * prev_action_7d[:6] + (1 - ema_alpha) * action_7d[:6]
            prev_action_7d = action_7d.copy()

            # Debug: log first few steps per episode
            if prim_steps < 5 or prim_steps % 50 == 0:
                eef = obs.get("robot0_base_to_eef_pos", obs.get("robot0_eef_pos", [0,0,0]))
                logger.info(
                    f"  [DIAG] s={total_steps} norm={[f'{v:.3f}' for v in action_7d_norm[:3]]} "
                    f"denorm={[f'{v:.4f}' for v in action_7d[:3]]} "
                    f"eef={[f'{v:.3f}' for v in eef[:3]]}"
                )

            # For grip primitives, force gripper action
            if prim_type == "grip":
                action_7d[:6] = 0.0  # no arm motion
                action_7d[6] = 1.0   # close gripper

            # Step env
            action_12d = action_7d_to_12d(action_7d)
            obs, reward, done, info = env.step(action_12d)
            total_reward += reward
            total_steps += 1
            prim_steps += 1

            # Heuristic monitor: track EEF and detect stuck/divergence
            if heuristic_monitor and prim_type != "grip":
                eef_now = obs.get("robot0_eef_pos", np.zeros(3))
                eef_history.append(eef_now.copy())

                if prim_steps >= monitor_warmup and len(eef_history) >= monitor_stuck_window:
                    window_positions = eef_history[-monitor_stuck_window:]
                    max_disp = max(
                        np.linalg.norm(p - window_positions[0])
                        for p in window_positions[1:]
                    )
                    if max_disp < monitor_stuck_threshold:
                        logger.info(
                            f"  [MONITOR] Stuck detected at prim_step={prim_steps}: "
                            f"max_disp={max_disp:.4f}m < {monitor_stuck_threshold}m over {monitor_stuck_window} steps"
                        )
                        break  # Exit inner loop → will trigger retry


            if record_video:
                try:
                    rgb = env.sim.render(camera_name="robot0_agentview_left", height=480, width=480)
                    frames.append(np.flip(rgb, axis=0).copy())
                except Exception:
                    pass

            if done:
                break

            # VLM failure check at intervals
            if (
                vlm_client is not None
                and vlm_check_interval > 0
                and prim_steps > vlm_check_interval
                and prim_steps % vlm_check_interval == 0
            ):
                img = obs.get("robot0_agentview_left_image", obs.get("robot0_agentview_image"))
                if img is not None:
                    is_failing = _vlm_check_task_failed(vlm_client, img, instruction)
                    if is_failing:
                        bt_count = backtrack_counts.get(prim_idx, 0)
                        if bt_count < max_backtracks:
                            backtrack_counts[prim_idx] = bt_count + 1
                            logger.info(
                                f"  VLM detected failure at prim {prim_idx+1}, "
                                f"backtrack #{bt_count+1}"
                            )
                            # Backtrack: move back to checkpoint
                            for cp in reversed(checkpoints):
                                if cp.subtask_idx == prim_idx:
                                    for _ in range(30):
                                        curr_pos = obs.get("robot0_eef_pos", np.zeros(3))
                                        delta = cp.eef_pos - curr_pos
                                        if np.linalg.norm(delta) < 0.02:
                                            break
                                        bt_action = np.zeros(7)
                                        bt_action[:3] = np.clip(delta * 2.0, -0.3, 0.3)
                                        bt_action[6] = -1.0
                                        obs, reward, done, info = env.step(action_7d_to_12d(bt_action))
                                        total_steps += 1
                                        if done:
                                            break
                                    break
                            # Reset primitive execution
                            prim_steps = 0
                            chunk_buffer = None
                            prev_action_7d = None
                            continue
                        else:
                            logger.warning(
                                f"  Max backtracks reached for prim {prim_idx+1}, continuing"
                            )

            # Update FSM step counter
            fsm.step()

        if done:
            break

        # Retry: if budget exhausted OR monitor detected stuck, retry same primitive
        retry_count = backtrack_counts.get(f"retry_{prim_idx}", 0)
        max_retries = 10 if heuristic_monitor else 3
        monitor_triggered = (heuristic_monitor and prim_steps < budget and not done)
        budget_exhausted = (prim_steps >= budget)

        if (budget_exhausted or monitor_triggered) and total_steps < max_steps and retry_count < max_retries:
            trigger_reason = "STUCK" if monitor_triggered else "BUDGET"
            backtrack_counts[f"retry_{prim_idx}"] = retry_count + 1
            logger.info(
                f"  [RETRY] Primitive {prim_idx+1} {trigger_reason} at step {prim_steps}/{budget}, "
                f"resetting VLA state and retrying (attempt {retry_count + 1}/{max_retries}), "
                f"total_steps={total_steps}/{max_steps}"
            )

            # v2: Backtrack to checkpoint after STUCK to avoid stuck-loop
            if monitor_triggered and checkpoints:
                for cp in reversed(checkpoints):
                    if cp.subtask_idx <= prim_idx:
                        bt_target = cp.eef_pos
                        for _ in range(30):
                            curr_pos = obs.get("robot0_eef_pos", np.zeros(3))
                            delta = bt_target - curr_pos
                            if np.linalg.norm(delta) < 0.02:
                                break
                            bt_action = np.zeros(7)
                            bt_action[:3] = np.clip(delta * 2.0, -0.3, 0.3)
                            bt_action[6] = -1.0  # keep gripper open
                            obs, reward, done, info = env.step(action_7d_to_12d(bt_action))
                            total_steps += 1
                            if done:
                                break
                        logger.info(
                            f"  [BACKTRACK] Returned to checkpoint (subtask {cp.subtask_idx}), "
                            f"total_steps={total_steps}/{max_steps}"
                        )
                        break
                if done:
                    break

            chunk_buffer = None
            chunk_step = 0
            prev_action_7d = None
            eef_history.clear()  # Reset EEF tracking for new attempt
            continue  # Re-enter outer while loop, same primitive

        # Transition to next primitive
        fsm.advance()
        chunk_buffer = None  # Force new chunk for next primitive
        prev_action_7d = None  # Reset EMA for new primitive

        # Save checkpoint at primitive boundary
        if not fsm.is_done:
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            gripper_val = obs.get("robot0_gripper_qpos", np.array([0.0]))[0] if "robot0_gripper_qpos" in obs else 0.0
            checkpoints.append(SubtaskCheckpoint(
                eef_pos=eef_pos.copy(), gripper_state=gripper_val,
                subtask_idx=fsm.current_idx,
            ))

    # Check success
    task_success = False
    try:
        task_success = env._check_success()
    except Exception:
        pass

    return {
        "success": task_success,
        "steps": total_steps,
        "reward": total_reward,
        "frames": frames if record_video else [],
        "hybrid_stats": {
            "primitives_completed": fsm.current_idx,
            "primitives_total": len(template_primitives),
            "backtrack_counts": dict(backtrack_counts),
            "vlm_checks": vlm_client is not None,
        },
    }


# ===========================================================================
# PIPELINE MODE: RoboBridgeClient (Perception -> Planner -> Controller)
# ===========================================================================


def setup_client(args) -> Any:
    """Initialize RoboBridgeClient with VLA controller."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from robobridge.client.client import RoboBridgeClient

    logger.info("Initializing RoboBridgeClient...")

    # Modules
    modules = ["perception", "planner", "controller"]
    use_monitor = bool(args.monitor_provider)
    if use_monitor:
        modules.append("monitor")

    # API config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "")

    # Planner
    planner_cfg = {
        "provider": args.planner_provider,
        "model": args.planner_model,
    }
    if api_key:
        planner_cfg["api_key"] = api_key
    if api_base:
        planner_cfg["api_base"] = api_base

    # Monitor
    monitor_cfg = {}
    if use_monitor:
        monitor_cfg = {
            "provider": args.monitor_provider,
            "model": args.monitor_model,
        }
        if api_key:
            monitor_cfg["api_key"] = api_key
        if api_base:
            monitor_cfg["api_base"] = api_base
        logger.info(f"Monitor enabled: {args.monitor_provider}/{args.monitor_model}")

    # Perception
    perception_cfg = {
        "provider": args.perception_mode,
        "model": args.perception_model or "",
    }

    # Controller
    controller_cfg = {
        "provider": "custom",
        "backend": "vla_lora",
        "model": args.vla_model,
        "device": args.device,
    }

    client = RoboBridgeClient(
        simulation=True,
        modules=modules,
        auto_start=False,
        perception=perception_cfg,
        planner=planner_cfg,
        controller=controller_cfg,
        **({"monitor": monitor_cfg} if use_monitor else {}),
    )

    # Validate API key for planner/monitor (skip for bedrock - uses bearer token)
    if not api_key and args.planner_provider not in ("bedrock", "vertex"):
        logger.error(
            "No API key found. Set --api-key or OPENAI_API_KEY env var. "
            "Pipeline mode requires an LLM for planning."
        )
        sys.exit(1)

    client._init_modules()

    # Set VLA config on controller
    controller = client._modules["controller"]
    # Validate action_stats for pipeline mode
    action_stats_path = args.action_stats or ""
    if not action_stats_path or not os.path.exists(action_stats_path):
        logger.warning(
            f"action_stats not set or not found: '{action_stats_path}'. "
            "Pipeline denormalization will be IDENTITY (raw VLA output). "
            "Set --action-stats to data_stats.json from training."
        )
    else:
        # Log action stats min/max to verify denormalization is non-trivial
        with open(action_stats_path) as f:
            _stats = json.load(f)
        _act_stats = _stats.get("action_stats", {})
        _a_min = _act_stats.get("min", [])[:7]
        _a_max = _act_stats.get("max", [])[:7]
        logger.info(
            f"Action stats loaded from {action_stats_path}:\n"
            f"  min={[round(v, 4) for v in _a_min]}\n"
            f"  max={[round(v, 4) for v in _a_max]}"
        )
        # Check for identity transform (min=-1, max=1 means no scaling)
        if _a_min and _a_max:
            is_identity = all(abs(v + 1.0) < 0.01 for v in _a_min) and all(abs(v - 1.0) < 0.01 for v in _a_max)
            if is_identity:
                logger.warning(
                    "Action stats min/max are [-1, 1] — denormalization is IDENTITY. "
                    "This usually means the training data spans the full action range. "
                    "Consider using task-specific stats for better scaling."
                )

    vla_cfg = {
        "backend": args.vla_backend,
        "model_name": args.vla_model,
        "move_adapter_path": args.move_adapter or "",
        "grip_adapter_path": args.grip_adapter or "",
        "action_stats_path": action_stats_path,
        "quantize_4bit": False,
        "device": args.device,
        "ik_solver": {"type": "passthrough"},
        "use_absolute_targets": False,
        "action_scale": getattr(args, 'action_scale', 1.0),
        "norm_type": _act_stats.get("mode", "min_max") if _act_stats else "min_max",
    }
    # Pipeline v4 options (only set if explicitly provided, otherwise use defaults)
    if getattr(args, 'pipe_chunk_stride', None) is not None:
        vla_cfg["chunk_stride"] = args.pipe_chunk_stride
    if getattr(args, 'ema_alpha', None) is not None:
        vla_cfg["action_ema_alpha"] = args.ema_alpha
    if getattr(args, 'no_ultimate_target', False):
        vla_cfg["use_ultimate_target"] = False
    controller._vla_config = vla_cfg
    controller.initialize_model()

    # Override GROOT denoising steps (default=4 is too low)
    denoise_steps = getattr(args, 'denoise_steps', None) or 10
    try:
        # controller._vla_lora is VLALoRAController (set by _init_vla_lora)
        vla_lora = getattr(controller, '_vla_lora', None)
        lora_mgr = getattr(vla_lora, '_lora_manager', None) if vla_lora else None
        backend = getattr(lora_mgr, '_backend', None) if lora_mgr else None
        policy = getattr(backend, '_policy', None) if backend else None
        # GrootPolicy wraps GR00TN15: policy._groot_model.action_head
        groot_model = getattr(policy, '_groot_model', None) if policy else None
        action_head = getattr(groot_model, 'action_head', None) if groot_model else None
        if action_head and hasattr(action_head, 'num_inference_timesteps'):
            old_steps = action_head.num_inference_timesteps
            action_head.num_inference_timesteps = denoise_steps
            logger.info(f"Pipeline: GROOT denoising steps = {old_steps} -> {denoise_steps}")
        else:
            logger.warning(f"Could not find action_head for denoise steps override. "
                          f"vla_lora={vla_lora is not None}, lora_mgr={lora_mgr is not None}, "
                          f"backend={backend is not None}, policy={policy is not None}, "
                          f"groot_model={groot_model is not None}")
    except Exception as e:
        logger.warning(f"Could not set denoise steps: {e}")

    logger.info("VLA controller initialized")

    # DIAGNOSTIC: set fixed direction for testing model quality
    # Training mean direction for CloseDrawer: [0.758, -0.001, -0.396]
    _diag_dir = os.environ.get("DIAG_FIXED_DIRECTION")
    if _diag_dir:
        vals = [float(v) for v in _diag_dir.split(",")]
        controller._diag_fixed_direction = vals
        logger.info(f"DIAGNOSTIC: Fixed direction = {vals}")

    logger.info("RoboBridgeClient setup complete")
    return client


def run_episode_pipeline(
    client,
    env,
    obs: Dict,
    instruction: str,
    ep_meta: Dict,
    max_steps: int = 500,
    record_video: bool = False,
    move_timeout_replan: bool = False,
    diverge_margin: float = 1.0,
    stuck_window: int = 200,
    move_max_steps: int = 250,
    replan_cooldown: int = 300,
    trend_window: int = 30,
    task_name: str = "",
    max_plan_attempts: int = 3,
) -> Dict[str, Any]:
    """Run one episode through the RoboBridgeClient pipeline.

    Loops plan generation + execution until max_steps exhausted or task succeeds.
    """
    if task_name:
        client._current_task_name = task_name
    client.connect_env(env, obs, ep_meta)
    # Multi-camera video recording (named + custom wide-angle)
    NAMED_CAMERAS = ["robot0_agentview_left", "robot0_agentview_center"]
    CUSTOM_CAMERAS = {
        "wide_90left":   {"distance": 2.0, "azimuth": 90,  "elevation": -20},
        "wide_90right":  {"distance": 2.0, "azimuth": -90, "elevation": -20},
        "wide_fullbody": {"distance": 2.5, "azimuth": 45,  "elevation": -25},
    }
    ALL_CAMERA_KEYS = NAMED_CAMERAS + list(CUSTOM_CAMERAS.keys())
    multi_frames = {cam: [] for cam in ALL_CAMERA_KEYS}
    frames = multi_frames[NAMED_CAMERAS[0]]  # default frames for backward compat
    _custom_renderer = None
    _custom_mjv_cam = None
    frame_callback = None
    if record_video:
        import mujoco as _mj
        def _init_custom_renderer():
            nonlocal _custom_renderer, _custom_mjv_cam
            if _custom_renderer is None:
                _model = env.sim.model._model
                _custom_renderer = _mj.Renderer(_model, height=480, width=480)
                _custom_mjv_cam = _mj.MjvCamera()
                _custom_mjv_cam.type = _mj.mjtCamera.mjCAMERA_FREE
                # Lookat = EEF position
                eef_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
                if eef_id >= 0:
                    eef_pos = env.sim.data._data.xpos[eef_id].copy()
                    _custom_mjv_cam.lookat[:] = eef_pos
                    _custom_mjv_cam.lookat[2] = 0.6

        def frame_callback(current_obs):
            # Named cameras (v-flip needed)
            for cam in NAMED_CAMERAS:
                try:
                    rgb = env.sim.render(camera_name=cam, height=480, width=480)
                    multi_frames[cam].append(np.flip(rgb, axis=0).copy())
                except Exception:
                    pass
            # Custom wide-angle cameras (no flip needed)
            try:
                _init_custom_renderer()
                _data = env.sim.data._data
                for cam_name, cfg in CUSTOM_CAMERAS.items():
                    _custom_mjv_cam.distance = cfg["distance"]
                    _custom_mjv_cam.azimuth = cfg["azimuth"]
                    _custom_mjv_cam.elevation = cfg["elevation"]
                    _custom_renderer.update_scene(_data, camera=_custom_mjv_cam)
                    multi_frames[cam_name].append(_custom_renderer.render().copy())
            except Exception:
                pass

    start_time = time.time()
    total_steps = 0
    all_plans = []
    # max_plan_attempts passed as function parameter (default: 3)

    for attempt in range(max_plan_attempts):
        # Check task success before generating new plan (skip first attempt)
        if total_steps > 0:
            try:
                if env._check_success():
                    break
            except Exception:
                pass

        # Get fresh observation for subsequent attempts
        if attempt > 0:
            obs = env._get_observations(force_update=True)
            client.update_obs(obs)

        remaining_budget = max(0, max_steps - total_steps)
        result = client.step(
            obs, instruction, execute=True,
            enable_async_perception=False,
            frame_callback=frame_callback,
            max_steps=remaining_budget,
            move_timeout_replan=move_timeout_replan,
            diverge_margin=diverge_margin,
            stuck_window=stuck_window,
            move_max_steps=move_max_steps,
            replan_cooldown=replan_cooldown,
            trend_window=trend_window,
        )

        plans = result.get("plans", [])
        # Use total_steps_used (includes return-to-start) for accurate budget tracking
        steps = result.get("total_steps_used", len(result.get("execution_results", [])))
        total_steps += steps
        all_plans.extend(plans)

        # Print generated plans
        if plans:
            plan_label = f"attempt {attempt+1}" if attempt > 0 else ""
            if plan_label:
                print(f"\n    [Replan {plan_label}] {len(plans)} plan(s):")
            else:
                print(f"\n    [Plans] {len(plans)} plan(s) generated:")
            for i, plan in enumerate(plans):
                action = plan.parent_action if hasattr(plan, "parent_action") else None
                if action:
                    loc = f" -> {action.target_location}" if action.target_location else ""
                    print(f"      Plan {i+1}: {action.action_type}({action.target_object}{loc})")
                prims = plan.primitives if hasattr(plan, "primitives") else []
                for j, prim in enumerate(prims):
                    instr = prim.instruction if hasattr(prim, "instruction") else str(prim)
                    print(f"        [{j+1}] {instr}")

        if steps == 0:
            break  # No progress made

        if total_steps >= max_steps:
            break

    elapsed = time.time() - start_time

    task_success = False
    try:
        task_success = env._check_success()
    except Exception as e:
        logger.warning(f"Could not check success: {e}")

    pipeline_success = result.get("success", False) if 'result' in dir() else False

    # Serialize plan details for logging
    plan_details = []
    for plan in all_plans:
        action = plan.parent_action if hasattr(plan, "parent_action") else None
        plan_info = {"action": None, "primitives": []}
        if action:
            loc = f" -> {action.target_location}" if action.target_location else ""
            plan_info["action"] = f"{action.action_type}({action.target_object}{loc})"
        prims = plan.primitives if hasattr(plan, "primitives") else []
        for prim in prims:
            prim_dict = prim.to_dict() if hasattr(prim, "to_dict") else {}
            prim_str = prim.instruction if hasattr(prim, "instruction") else str(prim)
            plan_info["primitives"].append({"instruction": prim_str, "details": prim_dict})
        plan_details.append(plan_info)

    ep_result = {
        "success": task_success,
        "pipeline_success": pipeline_success,
        "steps": total_steps,
        "elapsed_s": elapsed,
        "num_plans": len(all_plans),
        "num_detections": result.get("num_detections", 0) if 'result' in dir() else 0,
        "replan_count": result.get("replan_count", 0) if 'result' in dir() else 0,
        "plan_attempts": min(attempt + 1, max_plan_attempts) if 'attempt' in dir() else 0,
        "plan_details": plan_details,
    }
    if record_video and frames:
        ep_result["frames"] = frames
        ep_result["multi_frames"] = multi_frames
    return ep_result


def evaluate_pipeline(
    client,
    task_name: str,
    num_episodes: int,
    max_horizon: int,
    dataset_path: Optional[str] = None,
    start_demo: int = 0,
    save_video_dir: Optional[str] = None,
    match_training: bool = False,
    move_timeout_replan: bool = False,
    diverge_margin: float = 1.0,
    stuck_window: int = 200,
    move_max_steps: int = 250,
    replan_cooldown: int = 300,
    trend_window: int = 30,
    max_plan_attempts: int = 3,
) -> Dict[str, Any]:
    """Evaluate using RoboBridgeClient pipeline."""
    instruction = TASK_INSTRUCTIONS.get(task_name, task_name.lower().replace("_", " "))
    record_video = save_video_dir is not None
    # Create timestamped sub-directory per run to avoid overwrites
    if record_video:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video_dir = os.path.join(save_video_dir, f"{run_ts}_{task_name}")
        os.makedirs(save_video_dir, exist_ok=True)

    episode_results = []
    successes = 0

    for i in range(num_episodes):
        ep_idx = start_demo + i
        env = None
        try:
            if dataset_path:
                # Load from HDF5 demo
                from robocasa.scripts.playback_dataset import reset_to
                initial_state, ep_meta = load_demo_state(dataset_path, ep_idx)
                demo_instruction = ep_meta.get("lang", instruction)
                env = create_env_from_demo(dataset_path)
                reset_to(env, initial_state)
                obs = env._get_observations(force_update=True)
            else:
                # Random environment (retry up to 15 times for fixture/robot errors)
                _max_env_attempts = 15
                for _env_attempt in range(_max_env_attempts):
                    try:
                        env = create_env(task_name, image_size=224, match_training=match_training)
                        obs = env.reset()
                        break
                    except Exception as env_err:
                        logger.warning(f"Env creation attempt {_env_attempt + 1}/{_max_env_attempts} failed: {env_err}")
                        if env is not None:
                            try:
                                env.close()
                            except Exception:
                                pass
                            env = None
                        if _env_attempt == _max_env_attempts - 1:
                            raise
                demo_instruction = instruction
                # Get ep_meta from env for fixture/object detection
                try:
                    ep_meta = env.get_ep_meta()
                except Exception:
                    ep_meta = {}

            print(f"  ep{i+1:>2d}/{num_episodes}: '{demo_instruction}'", end="", flush=True)

            result = run_episode_pipeline(
                client, env, obs, demo_instruction, ep_meta,
                max_steps=max_horizon, record_video=record_video,
                move_timeout_replan=move_timeout_replan,
                diverge_margin=diverge_margin,
                stuck_window=stuck_window,
                move_max_steps=move_max_steps,
                replan_cooldown=replan_cooldown,
                trend_window=trend_window,
                task_name=task_name,
                max_plan_attempts=max_plan_attempts,
            )

            tag = "\033[92mSUCCESS\033[0m" if result["success"] else "\033[91mFAIL\033[0m"
            print(f" -> {tag}  ({result['elapsed_s']:.1f}s, plans={result['num_plans']})")

            # Save video (one file per camera view)
            if record_video and "multi_frames" in result:
                os.makedirs(save_video_dir, exist_ok=True)
                tag_str = "success" if result["success"] else "fail"
                for cam_name, cam_frames in result.pop("multi_frames").items():
                    if cam_frames:
                        short_cam = cam_name.replace("robot0_", "")
                        vid_path = os.path.join(save_video_dir, f"pipeline_{task_name}_ep{ep_idx}_{tag_str}_{short_cam}.mp4")
                        save_video(cam_frames, vid_path, fps=20)
                result.pop("frames", None)

            result["episode"] = i + 1
            result["instruction"] = demo_instruction

            if result["success"]:
                successes += 1
            episode_results.append(result)

        except Exception as e:
            logger.error(f"Episode {i+1} error: {e}")
            import traceback
            traceback.print_exc()
            episode_results.append({
                "success": False,
                "episode": i + 1,
                "steps": 0,
                "error": str(e),
            })
        finally:
            if env is not None:
                env.close()
            client._env = None
            client._obs = None
            client._ep_meta = None

        # Running stats
        print(f"    progress: {successes}/{len(episode_results)} ({successes/len(episode_results):.1%})")

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    print(f"\n  \033[1m{task_name}: {successes}/{num_episodes} = {success_rate:.1%}\033[0m\n")

    return {
        "task": task_name,
        "mode": "pipeline",
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "episodes": episode_results,
    }


# ===========================================================================
# Main
# ===========================================================================


def list_available_checkpoints():
    for name, cfg in MODEL_CONFIGS.items():
        print(f"  {name}:")
        adapter_dir = cfg.get("adapter_dir")
        if adapter_dir and adapter_dir.exists():
            ckpts = sorted([
                d.name.replace("checkpoint-", "")
                for d in adapter_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ])
            print(f"    adapter: {', '.join(ckpts)}")
        for label in ["move_adapter_dir", "grip_adapter_dir"]:
            d = cfg.get(label)
            if d and d.exists():
                ckpts = sorted([
                    c.name.replace("checkpoint-", "")
                    for c in d.iterdir()
                    if c.is_dir() and c.name.startswith("checkpoint-")
                ])
                print(f"    {label.replace('_dir', '')}: {', '.join(ckpts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA LoRA adapters on RoboCasa tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="direct",
        choices=["direct", "pipeline", "cyclevla", "hybrid-cyclevla"],
        help="Execution mode: 'direct' | 'pipeline' | 'cyclevla' | 'hybrid-cyclevla'",
    )

    # Shared
    parser.add_argument("--tasks", nargs="+", default=None, help="RoboCasa task names (default: all)")
    parser.add_argument("--num-episodes", type=int, default=25, help="Episodes per task")
    parser.add_argument("--max-horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Test loading only")
    parser.add_argument("--list-tasks", action="store_true", help="List all tasks and exit")
    parser.add_argument("--list-checkpoints", action="store_true", help="List checkpoints and exit")

    # Adapter paths (shared by both modes)
    parser.add_argument("--move-adapter", type=str, default=None, help="Path to move adapter checkpoint")
    parser.add_argument("--grip-adapter", type=str, default=None, help="Path to grip adapter checkpoint")

    # Direct mode options
    direct_group = parser.add_argument_group("direct mode")
    direct_group.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), help="Model to evaluate")
    direct_group.add_argument("--checkpoint", type=str, default="best", help="Checkpoint name (default: best)")
    direct_group.add_argument("--batch-size", type=int, default=1, help="Parallel environments per batch")
    direct_group.add_argument("--adapter-dir", type=str, default=None, help="Override single adapter directory")
    direct_group.add_argument("--stats-file", type=str, default=None, help="Path to custom data_stats.json (overrides default metadata stats)")
    direct_group.add_argument("--image-size", type=int, default=None, help="Override image size (default: from model config)")
    direct_group.add_argument("--chunk-stride", type=int, default=None, help="Override chunk stride (re-predict every N steps)")
    direct_group.add_argument("--action-ema", type=float, default=0.0, help="Action EMA alpha (0=disabled, 0.8=smooth). action = alpha*prev + (1-alpha)*new")
    direct_group.add_argument("--match-training", action="store_true", help="Match training env kwargs (obj_instance_split=A, style_ids, etc.)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Scale factor for predicted actions (0.2-1.0). Reduces overshooting.")
    parser.add_argument("--denoise-steps", type=int, default=None, help="Override GROOT denoising steps (default: 4)")
    direct_group.add_argument("--temporal-ensemble", action="store_true", default=False,
                              help="Enable temporal ensemble: average overlapping action chunks")
    direct_group.add_argument("--stuck-max-retries", type=int, default=0,
                              help="Max STUCK resets before giving up (0=disabled, 3=recommended)")
    direct_group.add_argument("--vlm-check-interval", type=int, default=0,
                              help="VLM task-failure check every N steps (0=disabled, 100=recommended)")
    direct_group.add_argument("--dm-monitor-provider", default="",
                              help="Direct-mode monitor VLM provider (e.g., 'bedrock')")
    direct_group.add_argument("--dm-monitor-model", default="eu.anthropic.claude-sonnet-4-6-v1",
                              help="Direct-mode monitor VLM model")

    # Pipeline mode options
    pipe_group = parser.add_argument_group("pipeline mode")
    pipe_group.add_argument("--vla-backend", default="groot_n1.5",
                            choices=["openvla", "smolvla", "groot", "groot_n1.5", "pi05", "lerobot"],
                            help="VLA backend")
    pipe_group.add_argument("--vla-model", default="nvidia/GR00T-N1.5-3B", help="VLA model ID")
    pipe_group.add_argument("--action-stats", default="", help="Path to action normalization stats")
    pipe_group.add_argument("--dataset", default=None, help="HDF5 dataset path (pipeline mode, optional)")
    pipe_group.add_argument("--start-demo", type=int, default=0, help="Starting demo index")
    pipe_group.add_argument("--planner-provider", default="openai", help="Planner LLM provider")
    pipe_group.add_argument("--planner-model", default="gpt-5.2", help="Planner LLM model")
    pipe_group.add_argument("--perception-mode", default="robocasa_gt",
                            choices=["robocasa_gt", "hf", "florence2", "grounding_dino"],
                            help="Perception provider")
    pipe_group.add_argument("--perception-model", default="", help="Perception model name")
    pipe_group.add_argument("--monitor-provider", default="openai", help="Monitor provider (empty = disabled)")
    pipe_group.add_argument("--monitor-model", default="gpt-5.2", help="Monitor model")
    pipe_group.add_argument("--api-key", default="", help="API key (or OPENAI_API_KEY env)")
    pipe_group.add_argument("--api-base", default="", help="Custom API base URL (e.g., Azure)")

    # Pipeline v4 options (backward-compatible defaults)
    pipe_group.add_argument("--pipe-chunk-stride", type=int, default=None,
                            help="VLA chunk stride for pipeline mode (default: backend default=8)")
    pipe_group.add_argument("--ema-alpha", type=float, default=None,
                            help="Action EMA smoothing alpha for pipeline mode (default: controller default=0.3)")
    pipe_group.add_argument("--no-ultimate-target", action="store_true", default=False,
                            help="Use each primitive's own target for direction (instead of ultimate target)")
    pipe_group.add_argument("--move-timeout-replan", action="store_true", default=False,
                            help="Trigger replan when MOVE_MAX_STEPS reached (instead of treating as converged)")
    pipe_group.add_argument("--diverge-margin", type=float, default=None,
                            help="Diverge detection margin in meters (default: 1.0)")
    pipe_group.add_argument("--stuck-window", type=int, default=None,
                            help="Steps before declaring stuck (default: 200)")
    pipe_group.add_argument("--move-max-steps", type=int, default=None,
                            help="Max steps per move primitive (default: 250)")
    pipe_group.add_argument("--replan-cooldown", type=int, default=None,
                            help="Steps before STUCK/DIVERGE checks activate after replan (default: 300)")
    pipe_group.add_argument("--max-plan-attempts", type=int, default=None,
                            help="Max number of VLM replan attempts per episode (default: 3)")
    pipe_group.add_argument("--trend-window", type=int, default=None,
                            help="Window size for trend-based DIVERGE detection (default: 30)")
    pipe_group.add_argument("--template-plan", type=str, default=None,
                            help="Path to task_plan_config.json for training-data-aligned plans (bypasses VLM planner)")

    # CycleVLA options
    cyclevla_group = parser.add_argument_group("CycleVLA mode")
    cyclevla_group.add_argument("--cyclevla", action="store_true", default=False,
                                help="Enable CycleVLA: MBR decoding + subtask FSM + VLM failure prediction + backtracking")
    cyclevla_group.add_argument("--mbr-samples", type=int, default=8,
                                help="MBR sample count (default: 8)")
    cyclevla_group.add_argument("--stop-threshold", type=float, default=0.5,
                                help="Stop signal threshold for subtask transition check (default: 0.5)")
    cyclevla_group.add_argument("--stop-complete-threshold", type=float, default=0.8,
                                help="Stop signal threshold for subtask completion (default: 0.8)")
    cyclevla_group.add_argument("--max-backtracks", type=int, default=2,
                                help="Max backtrack retries per subtask (default: 2)")
    cyclevla_group.add_argument("--cyclevla-vlm-provider", type=str, default="bedrock",
                                help="VLM provider for CycleVLA failure prediction (default: bedrock)")
    cyclevla_group.add_argument("--cyclevla-vlm-model", type=str, default="eu.anthropic.claude-sonnet-4-6-v1",
                                help="VLM model for CycleVLA failure prediction")
    cyclevla_group.add_argument("--no-vlm-predictor", action="store_true", default=False,
                                help="Disable VLM failure prediction (use stop/progress signals only)")

    # Hybrid CycleVLA options
    hybrid_group = parser.add_argument_group("Hybrid CycleVLA mode")
    hybrid_group.add_argument("--hybrid-vlm-check-interval", type=int, default=50,
                              help="VLM failure check interval in steps (default: 50)")
    hybrid_group.add_argument("--vlm-provider", type=str, default="bedrock",
                              help="VLM provider for hybrid mode failure checks (default: bedrock)")
    hybrid_group.add_argument("--vlm-model", type=str, default="eu.anthropic.claude-sonnet-4-6-v1",
                              help="VLM model for hybrid mode failure checks")
    hybrid_group.add_argument("--heuristic-monitor", action="store_true", default=False,
                              help="Enable heuristic active monitor (stuck detection + fast retry)")
    hybrid_group.add_argument("--monitor-stuck-window", type=int, default=25,
                              help="Steps window for stuck detection (default: 25)")
    hybrid_group.add_argument("--monitor-stuck-threshold", type=float, default=0.003,
                              help="Stuck threshold in meters (default: 0.003)")
    hybrid_group.add_argument("--monitor-warmup", type=int, default=30,
                              help="Warmup steps before stuck detection starts (default: 30)")

    # Seed
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")

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

    tasks = args.tasks or list(TASK_INSTRUCTIONS.keys())
    for t in tasks:
        if t not in TASK_INSTRUCTIONS:
            logger.warning(f"No instruction for '{t}', using task name")
            TASK_INSTRUCTIONS[t] = t.lower().replace("_", " ")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # DIRECT MODE
    # -----------------------------------------------------------------------
    if args.mode == "direct":
        if args.model is None:
            parser.error("--model is required in direct mode")

        if args.move_adapter and args.grip_adapter:
            adapter_info = "move/grip dual adapters"
        elif args.adapter_dir:
            adapter_info = f"adapter: {args.adapter_dir}"
        else:
            adapter_info = f"default adapters (checkpoint: {args.checkpoint})"

        print(f"\n{'=' * 60}")
        print(f"Mode: DIRECT | Model: {args.model} | {adapter_info}")
        print(f"Tasks: {len(tasks)} | Episodes: {args.num_episodes} | Horizon: {args.max_horizon} | Batch: {args.batch_size}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

        video_dir = os.path.join(args.output_dir, "videos") if args.save_video else None
        all_results = {}
        start_time = time.time()

        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {args.model} / {task}")
            print("-" * 40)

            result = evaluate_direct(
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
                move_adapter_override=args.move_adapter,
                grip_adapter_override=args.grip_adapter,
                stats_file=getattr(args, 'stats_file', None),
                image_size_override=getattr(args, 'image_size', None),
                chunk_stride_override=getattr(args, 'chunk_stride', None),
                action_ema_alpha=getattr(args, 'action_ema', 0.0),
                match_training=getattr(args, 'match_training', False),
                action_scale=getattr(args, 'action_scale', 1.0),
                denoise_steps=getattr(args, 'denoise_steps', None),
                temporal_ensemble=getattr(args, 'temporal_ensemble', False),
                stuck_max_retries=getattr(args, 'stuck_max_retries', 0),
                vlm_check_interval=getattr(args, 'vlm_check_interval', 0),
                monitor_provider=getattr(args, 'dm_monitor_provider', ''),
                monitor_model=getattr(args, 'dm_monitor_model', ''),
            )
            all_results[task] = result

        elapsed = time.time() - start_time

        if not args.dry_run:
            print_results_table(all_results, tasks)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"eval_{args.model}_{args.checkpoint}_{timestamp}.json")
        table_str = format_results_table(all_results, tasks) if not args.dry_run else ""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "direct",
            "model": args.model,
            "checkpoint": args.checkpoint,
            "adapter_mode": "move/grip" if (args.move_adapter and args.grip_adapter) else "single",
            "move_adapter": args.move_adapter,
            "grip_adapter": args.grip_adapter,
            "tasks": tasks,
            "num_episodes": args.num_episodes,
            "max_horizon": args.max_horizon,
            "elapsed_seconds": elapsed,
            "results": all_results,
            "summary_table": table_str,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

        if table_str:
            txt_path = output_path.replace(".json", ".txt")
            with open(txt_path, "w") as f:
                f.write(table_str + "\n")
            print(f"Summary saved to: {txt_path}")

    # -----------------------------------------------------------------------
    # PIPELINE MODE
    # -----------------------------------------------------------------------
    elif args.mode == "pipeline":
        print(f"\n{'=' * 60}")
        print(f"Mode: PIPELINE (RoboBridgeClient)")
        print(f"VLA: {args.vla_backend} / {args.vla_model}")
        print(f"Planner: {args.planner_provider}/{args.planner_model}")
        print(f"Monitor: {args.monitor_provider}/{args.monitor_model}" if args.monitor_provider else "Monitor: disabled")
        print(f"Tasks: {len(tasks)} | Episodes: {args.num_episodes} | Horizon: {args.max_horizon}")
        print(f"Device: {args.device}")
        # v4 options
        v4_opts = []
        if getattr(args, 'pipe_chunk_stride', None) is not None:
            v4_opts.append(f"chunk_stride={args.pipe_chunk_stride}")
        if getattr(args, 'ema_alpha', None) is not None:
            v4_opts.append(f"ema_alpha={args.ema_alpha}")
        if getattr(args, 'no_ultimate_target', False):
            v4_opts.append("no_ultimate_target")
        if getattr(args, 'move_timeout_replan', False):
            v4_opts.append("move_timeout_replan")
        if v4_opts:
            print(f"V4 options: {', '.join(v4_opts)}")
        print(f"{'=' * 60}\n")

        client = setup_client(args)

        # Load template plan config if specified
        if getattr(args, 'template_plan', None):
            tp_path = args.template_plan
            with open(tp_path) as f:
                client._template_plan_config = json.load(f)
            logger.info(f"Template plan loaded: {tp_path} ({len(client._template_plan_config)} tasks)")

        all_results = {}
        start_time = time.time()

        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {task}")
            print("-" * 40)

            result = evaluate_pipeline(
                client=client,
                task_name=task,
                num_episodes=args.num_episodes,
                max_horizon=args.max_horizon,
                dataset_path=args.dataset,
                start_demo=args.start_demo,
                save_video_dir=os.path.join(args.output_dir, "videos") if args.save_video else None,
                match_training=getattr(args, 'match_training', False),
                move_timeout_replan=getattr(args, 'move_timeout_replan', False),
                diverge_margin=getattr(args, 'diverge_margin', None) if getattr(args, 'diverge_margin', None) is not None else 1.0,
                stuck_window=getattr(args, 'stuck_window', None) if getattr(args, 'stuck_window', None) is not None else 200,
                move_max_steps=getattr(args, 'move_max_steps', None) if getattr(args, 'move_max_steps', None) is not None else 250,
                replan_cooldown=getattr(args, 'replan_cooldown', None) if getattr(args, 'replan_cooldown', None) is not None else 300,
                trend_window=getattr(args, 'trend_window', None) if getattr(args, 'trend_window', None) is not None else 30,
                max_plan_attempts=getattr(args, 'max_plan_attempts', None) if getattr(args, 'max_plan_attempts', None) is not None else 3,
            )
            all_results[task] = result

        elapsed = time.time() - start_time

        if not args.dry_run:
            print_results_table(all_results, tasks)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"eval_pipeline_{timestamp}.json")
        table_str = format_results_table(all_results, tasks) if not args.dry_run else ""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "pipeline",
            "vla_backend": args.vla_backend,
            "vla_model": args.vla_model,
            "planner": f"{args.planner_provider}/{args.planner_model}",
            "monitor": f"{args.monitor_provider}/{args.monitor_model}" if args.monitor_provider else "disabled",
            "tasks": tasks,
            "num_episodes": args.num_episodes,
            "max_horizon": args.max_horizon,
            "elapsed_seconds": elapsed,
            "v4_options": {
                "chunk_stride": getattr(args, 'pipe_chunk_stride', None),
                "ema_alpha": getattr(args, 'ema_alpha', None),
                "no_ultimate_target": getattr(args, 'no_ultimate_target', False),
                "move_timeout_replan": getattr(args, 'move_timeout_replan', False),
            },
            "results": all_results,
            "summary_table": table_str,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

        if table_str:
            txt_path = output_path.replace(".json", ".txt")
            with open(txt_path, "w") as f:
                f.write(table_str + "\n")
            print(f"Summary saved to: {txt_path}")

        # Save detailed plan log
        plan_log_path = output_path.replace(".json", "_plans.txt")
        with open(plan_log_path, "w") as f:
            f.write(f"Pipeline Plan Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Planner: {args.planner_provider}/{args.planner_model}\n")
            f.write(f"{'=' * 70}\n\n")
            for task_name, task_result in all_results.items():
                f.write(f"Task: {task_name}\n")
                f.write(f"{'-' * 50}\n")
                for ep in task_result.get("episodes", []):
                    ep_num = ep.get("episode", "?")
                    success = ep.get("success", False)
                    steps = ep.get("steps", 0)
                    instr = ep.get("instruction", "")
                    tag = "SUCCESS" if success else "FAIL"
                    f.write(f"\n  Episode {ep_num}: [{tag}] steps={steps} \"{instr}\"\n")
                    for pi, plan_info in enumerate(ep.get("plan_details", [])):
                        action_str = plan_info.get("action", "unknown")
                        f.write(f"    Plan {pi+1}: {action_str}\n")
                        for pj, prim in enumerate(plan_info.get("primitives", [])):
                            f.write(f"      [{pj+1}] {prim.get('instruction', '')}\n")
                            details = prim.get("details", {})
                            if details:
                                tp = details.get("target_position")
                                if tp:
                                    f.write(f"           target_pos: ({tp.get('x',0):.3f}, {tp.get('y',0):.3f}, {tp.get('z',0):.3f})\n")
                                gs = details.get("gripper_state")
                                if gs:
                                    f.write(f"           gripper: {gs}\n")
                f.write(f"\n")
        print(f"Plan log saved to: {plan_log_path}")

    # -----------------------------------------------------------------------
    # CYCLEVLA MODE
    # -----------------------------------------------------------------------
    elif args.mode == "cyclevla":
        # Subtask counts per task (from training data decomposition)
        TASK_SUBTASK_COUNTS = {
            "CoffeePressButton": 3, "PnPCabToCounter": 3, "PnPCounterToCab": 3,
            "PnPCounterToMicrowave": 3, "PnPCounterToSink": 3, "PnPCounterToStove": 3,
            "PnPMicrowaveToCounter": 3, "PnPSinkToCounter": 3, "PnPStoveToCounter": 3,
            "CoffeeServeMug": 3, "CoffeeSetupMug": 3,
        }

        print(f"\n{'=' * 60}")
        print(f"Mode: CYCLEVLA (MBR + Subtask FSM + Backtracking)")
        print(f"VLA: groot_n1.5 / {args.vla_model}")
        print(f"MBR samples: {args.mbr_samples} | Stop threshold: {args.stop_threshold}")
        print(f"Max backtracks: {args.max_backtracks}")
        vlm_status = f"{args.cyclevla_vlm_provider}/{args.cyclevla_vlm_model}" if not args.no_vlm_predictor else "disabled"
        print(f"VLM predictor: {vlm_status}")
        print(f"Tasks: {len(tasks)} | Episodes: {args.num_episodes} | Horizon: {args.max_horizon}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

        # Load GROOT model
        move_adapter = args.move_adapter
        grip_adapter = args.grip_adapter

        # Load stats
        action_stats_path = args.action_stats
        if action_stats_path and os.path.exists(action_stats_path):
            with open(action_stats_path) as f:
                custom_stats = json.load(f)
            # Handle both flat and nested stats formats
            if "action_stats" in custom_stats:
                action_stats = custom_stats["action_stats"]
                state_stats = custom_stats["state_stats"]
            else:
                action_stats = custom_stats
                state_stats = custom_stats.get("state_stats", load_normalization_stats()["state_stats"])
            # Merge quantile stats if present (same as pipeline mode)
            if "action_quantile_stats" in custom_stats:
                action_stats["q01"] = custom_stats["action_quantile_stats"]["q01"]
                action_stats["q99"] = custom_stats["action_quantile_stats"]["q99"]
            if "q01" not in state_stats and "state_quantile_stats" in custom_stats:
                state_stats["q01"] = custom_stats["state_quantile_stats"]["q01"]
                state_stats["q99"] = custom_stats["state_quantile_stats"]["q99"]
        else:
            stats = load_normalization_stats()
            action_stats = stats["action_stats"]
            state_stats = stats["state_stats"]

        print(f"  Loading GROOT with move/grip adapters...", flush=True)
        print(f"    move: {move_adapter}", flush=True)
        print(f"    grip: {grip_adapter}", flush=True)

        model, eagle_processor, adapter_weights = load_groot(
            adapter_path=None,
            move_adapter=move_adapter,
            grip_adapter=grip_adapter,
            image_size=getattr(args, 'image_size', 224) or 224,
            denoise_steps=args.denoise_steps,
            device=device,
        )
        print(f"  Model loaded.", flush=True)

        all_results = {}
        start_time = time.time()

        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task}")
            print("-" * 40)

            instruction = TASK_INSTRUCTIONS.get(task, task.lower().replace("_", " "))
            n_subtasks = TASK_SUBTASK_COUNTS.get(task, 2)
            img_size = getattr(args, 'image_size', 224) or 224

            successes = []
            episodes_data = []

            for ep in range(args.num_episodes):
                env = create_env(task, image_size=img_size,
                                 match_training=getattr(args, 'match_training', False))

                try:
                    obs = env.reset()
                except Exception as e:
                    logger.warning(f"  Episode {ep+1} reset failed: {e}")
                    successes.append(False)
                    env.close()
                    continue

                # Select adapter (use move adapter by default)
                if "move" in adapter_weights:
                    switch_adapter(model, adapter_weights, "move")

                ep_result = run_episode_cyclevla(
                    model=model,
                    eagle_processor=eagle_processor,
                    adapter_weights=adapter_weights,
                    env=env,
                    obs=obs,
                    instruction=instruction,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    max_steps=args.max_horizon,
                    image_size=getattr(args, 'image_size', 224) or 224,
                    denoise_steps=args.denoise_steps,
                    mbr_samples=args.mbr_samples,
                    stop_threshold=args.stop_threshold,
                    stop_complete_threshold=args.stop_complete_threshold,
                    max_backtracks=args.max_backtracks,
                    use_vlm_predictor=not args.no_vlm_predictor,
                    vlm_provider=args.cyclevla_vlm_provider,
                    vlm_model=args.cyclevla_vlm_model,
                    n_subtasks=n_subtasks,
                    task_name=task,
                    record_video=args.save_video,
                    action_scale=getattr(args, 'action_scale', 1.0),
                    chunk_stride=getattr(args, 'pipe_chunk_stride', 4) or 4,
                    ema_alpha=getattr(args, 'ema_alpha', 0.6) or 0.6,
                )

                success = ep_result["success"]
                steps = ep_result["steps"]
                successes.append(success)
                tag = "SUCCESS" if success else "FAIL"
                cv_stats = ep_result.get("cyclevla_stats", {})
                bt_info = cv_stats.get("backtrack_counts", {})
                bt_str = f" bt={bt_info}" if bt_info else ""
                print(f"  Ep {ep+1:2d}/{args.num_episodes}: [{tag:7s}] steps={steps:3d}{bt_str}")

                episodes_data.append({
                    "episode": ep + 1,
                    "success": success,
                    "steps": steps,
                    "instruction": instruction,
                    "cyclevla_stats": cv_stats,
                })

                if args.save_video and ep_result.get("frames"):
                    vid_dir = os.path.join(args.output_dir, "videos")
                    os.makedirs(vid_dir, exist_ok=True)
                    vid_path = os.path.join(vid_dir, f"cyclevla_{task}_ep{ep+1}.mp4")
                    save_video(ep_result["frames"], vid_path)

                env.close()

            sr = sum(successes) / len(successes) if successes else 0.0
            print(f"  {task}: {sr:.0%} ({sum(successes)}/{len(successes)})")
            all_results[task] = {
                "success_rate": sr,
                "successes": sum(successes),
                "total_episodes": len(successes),
                "episodes": episodes_data,
            }

        elapsed = time.time() - start_time

        if not args.dry_run:
            print_results_table(all_results, tasks)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"eval_cyclevla_{timestamp}.json")
        table_str = format_results_table(all_results, tasks) if not args.dry_run else ""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "cyclevla",
            "vla_model": args.vla_model,
            "mbr_samples": args.mbr_samples,
            "stop_threshold": args.stop_threshold,
            "max_backtracks": args.max_backtracks,
            "vlm_predictor": not args.no_vlm_predictor,
            "tasks": tasks,
            "num_episodes": args.num_episodes,
            "max_horizon": args.max_horizon,
            "elapsed_seconds": elapsed,
            "results": all_results,
            "summary_table": table_str,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        if table_str:
            txt_path = output_path.replace(".json", ".txt")
            with open(txt_path, "w") as f:
                f.write(table_str + "\n")
            print(f"Summary saved to: {txt_path}")

    # -----------------------------------------------------------------------
    # HYBRID CYCLEVLA MODE
    # -----------------------------------------------------------------------
    elif args.mode == "hybrid-cyclevla":
        # Load template plan config (required)
        if not getattr(args, 'template_plan', None):
            parser.error("--template-plan is required for hybrid-cyclevla mode")
        with open(args.template_plan) as f:
            template_config = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"Mode: HYBRID CYCLEVLA (MBR + direction_vector + budget FSM)")
        print(f"VLA: groot_n1.5 / {args.vla_model}")
        print(f"MBR samples: {args.mbr_samples} | Chunk stride: {getattr(args, 'pipe_chunk_stride', 4) or 4}")
        print(f"EMA alpha: {getattr(args, 'ema_alpha', 0.6) or 0.6}")
        print(f"VLM check interval: {args.hybrid_vlm_check_interval}")
        print(f"Max backtracks: {args.max_backtracks}")
        print(f"Template plan: {args.template_plan} ({len(template_config)} tasks)")
        if args.heuristic_monitor:
            print(f"Heuristic monitor: ON (stuck_window={args.monitor_stuck_window}, "
                  f"thresh={args.monitor_stuck_threshold}, warmup={args.monitor_warmup})")
        else:
            print(f"Heuristic monitor: OFF")
        print(f"Tasks: {len(tasks)} | Episodes: {args.num_episodes} | Horizon: {args.max_horizon}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

        # Load GROOT model
        move_adapter = args.move_adapter
        grip_adapter = args.grip_adapter

        # Load stats
        action_stats_path = args.action_stats
        if action_stats_path and os.path.exists(action_stats_path):
            with open(action_stats_path) as f:
                custom_stats = json.load(f)
            if "action_stats" in custom_stats:
                action_stats = custom_stats["action_stats"]
                state_stats = custom_stats["state_stats"]
            else:
                action_stats = custom_stats
                state_stats = custom_stats.get("state_stats", load_normalization_stats()["state_stats"])
            if "action_quantile_stats" in custom_stats:
                action_stats["q01"] = custom_stats["action_quantile_stats"]["q01"]
                action_stats["q99"] = custom_stats["action_quantile_stats"]["q99"]
            if "q01" not in state_stats and "state_quantile_stats" in custom_stats:
                state_stats["q01"] = custom_stats["state_quantile_stats"]["q01"]
                state_stats["q99"] = custom_stats["state_quantile_stats"]["q99"]
        else:
            stats = load_normalization_stats()
            action_stats = stats["action_stats"]
            state_stats = stats["state_stats"]

        print(f"  Loading GROOT with move/grip adapters...", flush=True)
        print(f"    move: {move_adapter}", flush=True)
        print(f"    grip: {grip_adapter}", flush=True)

        model, eagle_processor, adapter_weights = load_groot(
            adapter_path=None,
            move_adapter=move_adapter,
            grip_adapter=grip_adapter,
            image_size=getattr(args, 'image_size', 224) or 224,
            denoise_steps=args.denoise_steps,
            device=device,
        )
        print(f"  Model loaded.", flush=True)

        # Create VLM client for failure checks
        vlm_client = None
        if args.hybrid_vlm_check_interval > 0:
            vlm_client = _create_vlm_monitor(args.vlm_provider, args.vlm_model)
            if vlm_client:
                print(f"  VLM monitor: {args.vlm_provider}/{args.vlm_model}")
            else:
                print(f"  VLM monitor: failed to create, running without")

        all_results = {}
        start_time = time.time()

        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task}")
            print("-" * 40)

            if task not in template_config:
                print(f"  SKIP: no template plan for {task}")
                continue

            instruction = TASK_INSTRUCTIONS.get(task, task.lower().replace("_", " "))
            primitives = template_config[task]["primitives"]
            img_size = getattr(args, 'image_size', 224) or 224

            successes = []
            episodes_data = []

            for ep in range(args.num_episodes):
                env = create_env(task, image_size=img_size,
                                 match_training=getattr(args, 'match_training', False))

                try:
                    obs = env.reset()
                except Exception as e:
                    logger.warning(f"  Episode {ep+1} reset failed: {e}")
                    successes.append(False)
                    env.close()
                    continue

                # Get perception target for dynamic direction_vector Y
                target_pos_rb = _get_perception_target_rb(env, obs, task)

                # Select move adapter by default
                if "move" in adapter_weights:
                    switch_adapter(model, adapter_weights, "move")

                ep_result = run_episode_hybrid_cyclevla(
                    model=model,
                    eagle_processor=eagle_processor,
                    adapter_weights=adapter_weights,
                    env=env,
                    obs=obs,
                    instruction=instruction,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    template_primitives=primitives,
                    max_steps=args.max_horizon,
                    image_size=img_size,
                    denoise_steps=args.denoise_steps,
                    mbr_samples=args.mbr_samples,
                    max_backtracks=args.max_backtracks,
                    vlm_check_interval=args.hybrid_vlm_check_interval,
                    vlm_client=vlm_client,
                    task_name=task,
                    record_video=args.save_video,
                    action_scale=getattr(args, 'action_scale', 1.0),
                    chunk_stride=getattr(args, 'pipe_chunk_stride', 4) or 4,
                    ema_alpha=getattr(args, 'ema_alpha', 0.6) or 0.6,
                    target_pos_rb=target_pos_rb,
                    heuristic_monitor=getattr(args, 'heuristic_monitor', False),
                    monitor_stuck_window=getattr(args, 'monitor_stuck_window', 25),
                    monitor_stuck_threshold=getattr(args, 'monitor_stuck_threshold', 0.003),
                    monitor_warmup=getattr(args, 'monitor_warmup', 30),
                )

                success = ep_result["success"]
                steps = ep_result["steps"]
                successes.append(success)
                tag = "SUCCESS" if success else "FAIL"
                h_stats = ep_result.get("hybrid_stats", {})
                bt_info = h_stats.get("backtrack_counts", {})
                prims_done = h_stats.get("primitives_completed", 0)
                prims_total = h_stats.get("primitives_total", 0)
                bt_str = f" bt={bt_info}" if bt_info else ""
                print(f"  Ep {ep+1:2d}/{args.num_episodes}: [{tag:7s}] steps={steps:3d} prims={prims_done}/{prims_total}{bt_str}")

                episodes_data.append({
                    "episode": ep + 1,
                    "success": success,
                    "steps": steps,
                    "instruction": instruction,
                    "hybrid_stats": h_stats,
                })

                if args.save_video and ep_result.get("frames"):
                    vid_dir = os.path.join(args.output_dir, "videos")
                    os.makedirs(vid_dir, exist_ok=True)
                    vid_path = os.path.join(vid_dir, f"hybrid_{task}_ep{ep+1}.mp4")
                    save_video(ep_result["frames"], vid_path)

                env.close()

            sr = sum(successes) / len(successes) if successes else 0.0
            print(f"  {task}: {sr:.0%} ({sum(successes)}/{len(successes)})")
            all_results[task] = {
                "success_rate": sr,
                "successes": sum(successes),
                "total_episodes": len(successes),
                "episodes": episodes_data,
            }

        elapsed = time.time() - start_time

        if not args.dry_run:
            print_results_table(all_results, tasks)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"eval_hybrid_cyclevla_{timestamp}.json")
        table_str = format_results_table(all_results, tasks) if not args.dry_run else ""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "hybrid-cyclevla",
            "vla_model": args.vla_model,
            "mbr_samples": args.mbr_samples,
            "max_backtracks": args.max_backtracks,
            "vlm_check_interval": args.hybrid_vlm_check_interval,
            "vlm_provider": args.vlm_provider,
            "vlm_model": args.vlm_model,
            "heuristic_monitor": getattr(args, 'heuristic_monitor', False),
            "monitor_stuck_window": getattr(args, 'monitor_stuck_window', 25),
            "monitor_stuck_threshold": getattr(args, 'monitor_stuck_threshold', 0.003),
            "monitor_warmup": getattr(args, 'monitor_warmup', 30),
            "template_plan": args.template_plan,
            "chunk_stride": getattr(args, 'pipe_chunk_stride', 4),
            "ema_alpha": getattr(args, 'ema_alpha', 0.6),
            "tasks": tasks,
            "num_episodes": args.num_episodes,
            "max_horizon": args.max_horizon,
            "elapsed_seconds": elapsed,
            "results": all_results,
            "summary_table": table_str,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        if table_str:
            txt_path = output_path.replace(".json", ".txt")
            with open(txt_path, "w") as f:
                f.write(table_str + "\n")
            print(f"Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
