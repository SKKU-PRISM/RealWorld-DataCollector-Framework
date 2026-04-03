#!/usr/bin/env python3
"""
LIBERO evaluation for VLA models (GROOT N1.5, PI0.5, SmolVLA) with pipeline-style execution.

Supports all four LIBERO suites: Object, Spatial, Goal, Long (libero_10).
  - Object: pick X and place in basket (10 tasks)
  - Spatial: pick black bowl from location and place on plate (10 tasks)
  - Goal: diverse manipulation — open, push, turn, place (10 tasks)
  - Long: multi-step manipulation sequences (10 tasks)

Three evaluation modes:
  - template: Fixed FSM plan (Object suite only)
  - pipeline: VLM planner + monitor with suite-aware prompts (all suites)
  - raw: Continuous VLA inference without FSM (all suites)

Architecture (pipeline mode):
  For each episode:
    1. Reset LIBERO env, get initial obs
    2. VLM generates suite-aware plan (or fallback)
    3. For each primitive:
       a. Build state vector (12D/9D/8D depending on training)
       b. Get agentview image (with horizontal flip)
       c. Feed to VLA -> action chunk (16 steps, 7D)
       d. Apply with EMA smoothing + chunk stride
       e. Monitor for STUCK/DIVERGE -> replan if needed
    4. Check env.step() done flag for success

Usage:
    # Object suite (default)
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    PYTHONPATH=$LIBERO_REPO:. \
    python3 eval_vla_libero.py \
        --task alphabet_soup \
        --move-adapter outputs/groot_libero/alphabet_soup/move_adapter/checkpoint-best \
        --action-stats outputs/groot_libero/alphabet_soup/move_adapter/data_stats.json \
        --num-episodes 50 --max-horizon 280 \
        --denoise-steps 10 --ema-alpha 0.6 --chunk-stride 4 --seed 42

    # Spatial suite with pipeline mode
    python3 eval_vla_libero.py --suite spatial --mode pipeline \
        --task from_table_center \
        --move-adapter outputs/groot_libero_spatial/from_table_center/move_adapter/checkpoint-best \
        --planner-provider bedrock --planner-model eu.anthropic.claude-sonnet-4-20250514-v1:0

    # Goal suite with pipeline mode (all tasks)
    python3 eval_vla_libero.py --suite goal --mode pipeline --all-tasks \
        --move-adapter-pattern outputs/groot_libero_goal/{task}/move_adapter/checkpoint-best \
        --stats-pattern outputs/groot_libero_goal/{task}/move_adapter/data_stats.json

    # Run all 10 Object tasks
    python3 eval_vla_libero.py --all-tasks \
        --move-adapter-pattern outputs/groot_libero/{task}/move_adapter/checkpoint-best \
        --stats-pattern outputs/groot_libero/{task}/move_adapter/data_stats.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

# ---------------------------------------------------------------------------
# PyTorch 2.6+ compatibility: torch.load defaults weights_only=True,
# but LIBERO init states are pickled numpy arrays.
# ---------------------------------------------------------------------------
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

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


# ===========================================================================
# LIBERO-Object task definitions
# ===========================================================================

# Map short task name -> (task_id in benchmark, object obs prefix, full instruction)
# Object obs prefix is used to find {prefix}_pos and {prefix}_to_robot0_eef_pos in obs.
LIBERO_OBJECT_TASKS = {
    "alphabet_soup":      (0, "alphabet_soup_1",      "pick up the alphabet soup and place it in the basket"),
    "cream_cheese":       (1, "cream_cheese_1",       "pick up the cream cheese and place it in the basket"),
    "salad_dressing":     (2, "salad_dressing_1",     "pick up the salad dressing and place it in the basket"),
    "bbq_sauce":          (3, "bbq_sauce_1",          "pick up the bbq sauce and place it in the basket"),
    "ketchup":            (4, "ketchup_1",            "pick up the ketchup and place it in the basket"),
    "tomato_sauce":       (5, "tomato_sauce_1",       "pick up the tomato sauce and place it in the basket"),
    "butter":             (6, "butter_1",             "pick up the butter and place it in the basket"),
    "milk":               (7, "milk_1",               "pick up the milk and place it in the basket"),
    "chocolate_pudding":  (8, "chocolate_pudding_1",  "pick up the chocolate pudding and place it in the basket"),
    "orange_juice":       (9, "orange_juice_1",       "pick up the orange juice and place it in the basket"),
}

# ---------------------------------------------------------------------------
# LIBERO-Spatial task definitions (10 tasks)
# Map short name -> (task_id, full instruction)
# ---------------------------------------------------------------------------
LIBERO_SPATIAL_TASKS = {
    "between_the_plate_and_the_ramekin": (0, "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    "next_to_the_ramekin": (1, "pick up the black bowl next to the ramekin and place it on the plate"),
    "from_table_center": (2, "pick up the black bowl from table center and place it on the plate"),
    "on_the_cookie_box": (3, "pick up the black bowl on the cookie box and place it on the plate"),
    "in_the_top_drawer_of_the_wooden_cabinet": (4, "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"),
    "on_the_ramekin": (5, "pick up the black bowl on the ramekin and place it on the plate"),
    "next_to_the_cookie_box": (6, "pick up the black bowl next to the cookie box and place it on the plate"),
    "on_the_stove": (7, "pick up the black bowl on the stove and place it on the plate"),
    "next_to_the_plate": (8, "pick up the black bowl next to the plate and place it on the plate"),
    "on_the_wooden_cabinet": (9, "pick up the black bowl on the wooden cabinet and place it on the plate"),
}

# ---------------------------------------------------------------------------
# LIBERO-Goal task definitions (10 tasks)
# ---------------------------------------------------------------------------
LIBERO_GOAL_TASKS = {
    "open_the_middle_drawer_of_the_cabinet": (0, "open the middle drawer of the cabinet"),
    "put_the_bowl_on_the_stove": (1, "put the bowl on the stove"),
    "put_the_wine_bottle_on_top_of_the_cabinet": (2, "put the wine bottle on top of the cabinet"),
    "open_the_top_drawer_and_put_the_bowl_inside": (3, "open the top drawer and put the bowl inside"),
    "put_the_bowl_on_top_of_the_cabinet": (4, "put the bowl on top of the cabinet"),
    "push_the_plate_to_the_front_of_the_stove": (5, "push the plate to the front of the stove"),
    "put_the_cream_cheese_in_the_bowl": (6, "put the cream cheese in the bowl"),
    "turn_on_the_stove": (7, "turn on the stove"),
    "put_the_bowl_on_the_plate": (8, "put the bowl on the plate"),
    "put_the_wine_bottle_on_the_rack": (9, "put the wine bottle on the rack"),
}

# ---------------------------------------------------------------------------
# LIBERO-Long (libero_10) task definitions (10 tasks)
# ---------------------------------------------------------------------------
LIBERO_LONG_TASKS = {
    "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": (0, "put both the alphabet soup and the tomato sauce in the basket"),
    "put_both_the_cream_cheese_box_and_the_butter_in_the_basket": (1, "put both the cream cheese box and the butter in the basket"),
    "turn_on_the_stove_and_put_the_moka_pot_on_it": (2, "turn on the stove and put the moka pot on it"),
    "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": (3, "put the black bowl in the bottom drawer of the cabinet and close it"),
    "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": (4, "put the white mug on the left plate and put the yellow and white mug on the right plate"),
    "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": (5, "pick up the book and place it in the back compartment of the caddy"),
    "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": (6, "put the white mug on the plate and put the chocolate pudding to the right of the plate"),
    "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": (7, "put both the alphabet soup and the cream cheese box in the basket"),
    "put_both_moka_pots_on_the_stove": (8, "put both moka pots on the stove"),
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it": (9, "put the yellow and white mug in the microwave and close it"),
}

# ---------------------------------------------------------------------------
# Unified suite registry
# ---------------------------------------------------------------------------
SUITE_REGISTRY = {
    "object": LIBERO_OBJECT_TASKS,      # (task_id, obj_prefix, instruction)
    "spatial": LIBERO_SPATIAL_TASKS,     # (task_id, instruction)
    "goal": LIBERO_GOAL_TASKS,           # (task_id, instruction)
    "long": LIBERO_LONG_TASKS,           # (task_id, instruction)
}

SUITE_BENCH_NAME = {
    "object": "libero_object",
    "spatial": "libero_spatial",
    "goal": "libero_goal",
    "long": "libero_10",
}

BASKET_OBS_PREFIX = "basket_1"

# Number of stabilization steps at episode start (objects settle)
NUM_WAIT_STEPS = 10

# Dummy action: no arm motion, gripper open
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Template plan primitives for LIBERO-Object (all tasks: pick X, place in basket)
TEMPLATE_PLAN = [
    {"type": "move", "target": "object",  "instruction": "move", "steps_budget": 120},
    {"type": "grip", "target": "object",  "instruction": "grip", "steps_budget": 30},
    {"type": "move", "target": "basket",  "instruction": "move", "steps_budget": 150},
    {"type": "release", "target": "basket", "instruction": "grip", "steps_budget": 20},
]


# ===========================================================================
# LIBERO environment utilities
# ===========================================================================

def fix_libero_config():
    """Ensure LIBERO config points to correct paths (not /tmp/LIBERO)."""
    import yaml

    config_path = os.path.expanduser("~/.libero/config.yaml")
    libero_root = None

    # Try to find libero repo
    candidates = [
        "$LIBERO_REPO/libero/libero",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "bddl_files")):
            libero_root = c
            break

    if libero_root is None:
        # Try to find from installed package
        try:
            import libero.libero
            pkg_root = os.path.dirname(os.path.abspath(libero.libero.__file__))
            if os.path.isdir(os.path.join(pkg_root, "bddl_files")):
                libero_root = pkg_root
        except Exception:
            pass

    if libero_root is None:
        logger.warning("Cannot find LIBERO bddl_files. Env creation may fail.")
        return

    # Check if current config is valid
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        bddl_path = config.get("bddl_files", "")
        if os.path.isdir(bddl_path):
            return  # Config is fine

    # Fix config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    new_config = {
        "benchmark_root": libero_root,
        "bddl_files": os.path.join(libero_root, "bddl_files"),
        "init_states": os.path.join(libero_root, "init_files"),
        "datasets": os.path.join(os.path.dirname(libero_root), "datasets"),
        "assets": os.path.join(libero_root, "assets"),
    }
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)
    logger.info(f"Fixed LIBERO config: {config_path} -> {libero_root}")


def get_libero_env(task_name: str, resolution: int = 224, suite_name: str = "object", horizon: int = 2000):
    """Create LIBERO environment for a given task and suite.

    Args:
        task_name: Short task name (e.g. 'alphabet_soup').
        resolution: Image resolution.
        suite_name: Suite name ('object', 'spatial', 'goal', 'long').
        horizon: Env internal horizon (must exceed max_steps + wait_steps).

    Returns:
        (env, task_suite, task_id, task_description)
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    registry = SUITE_REGISTRY[suite_name]
    entry = registry[task_name]
    task_id = entry[0]
    task_description = entry[-1]  # last element is always instruction

    bench = benchmark.get_benchmark_dict()
    suite = bench[SUITE_BENCH_NAME[suite_name]]()
    task = suite.get_task(task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=horizon,
    )
    env.seed(0)

    return env, suite, task_id, task_description


_TRAIN_IMG_MEAN = np.array([111.5, 102.2, 93.7], dtype=np.float32)
_TRAIN_IMG_STD = np.array([34.0, 33.4, 33.2], dtype=np.float32)


def get_libero_image(obs: Dict, size: int = 224, match_train_stats: bool = False) -> np.ndarray:
    """Extract agentview image with horizontal flip to match training data.

    Training preprocessing applies 180° rotation (flipV+flipH) to HDF5 images.
    The live env already has correct vertical orientation but opposite horizontal,
    so we apply a horizontal flip to match.
    """
    from PIL import Image as PILImage

    img = obs["agentview_image"]
    # Horizontal flip: live env mirrors training data horizontally
    img = img[:, ::-1].copy()

    if img.shape[0] != size or img.shape[1] != size:
        img = np.array(PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR))

    return img


def get_object_pos(obs: Dict, task_name: str) -> Optional[np.ndarray]:
    """Get target object position from obs (Object suite only)."""
    if task_name not in LIBERO_OBJECT_TASKS:
        return None
    _, obj_prefix, _ = LIBERO_OBJECT_TASKS[task_name]
    key = f"{obj_prefix}_pos"
    if key in obs:
        return np.array(obs[key], dtype=np.float32)
    return None


def get_basket_pos(obs: Dict) -> Optional[np.ndarray]:
    """Get basket position from obs."""
    key = f"{BASKET_OBS_PREFIX}_pos"
    if key in obs:
        return np.array(obs[key], dtype=np.float32)
    return None


def _find_obs_pos(obs: Dict, candidates: List[str]) -> Optional[np.ndarray]:
    """Try candidate obs keys and return the first matching position."""
    for key in candidates:
        if key in obs:
            return np.array(obs[key], dtype=np.float32)
        # Also try with _pos suffix
        k2 = f"{key}_pos" if not key.endswith("_pos") else key
        if k2 in obs:
            return np.array(obs[k2], dtype=np.float32)
    return None


def _extract_keywords_from_task(task_name: str) -> List[str]:
    """Extract object-like keywords from a task name (underscored form).

    E.g. 'put_the_bowl_on_the_stove' -> ['bowl', 'stove']
    """
    # Common non-object words to skip
    skip_words = {
        "the", "a", "an", "and", "or", "to", "from", "in", "on", "of",
        "put", "pick", "place", "push", "pull", "turn", "open", "close",
        "both", "up", "down", "left", "right", "front", "back", "top",
        "bottom", "middle", "inside", "it", "its", "next", "between",
    }
    words = task_name.split("_")
    keywords = [w for w in words if w.lower() not in skip_words and len(w) > 1]
    return keywords


def _get_sim_body_positions(env) -> Dict[str, np.ndarray]:
    """Extract all body positions from MuJoCo sim (includes fixtures like drawers, stoves)."""
    result = {}
    try:
        sim = env.env.sim
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if not name:
                continue
            if any(skip in name.lower() for skip in ("robot", "world", "gripper", "mount", "base")):
                continue
            result[name] = np.array(sim.data.body_xpos[i], dtype=np.float32)
    except Exception:
        pass
    return result


def get_scene_positions(
    obs: Dict,
    suite_name: str,
    task_name: str,
    env=None,
) -> Dict[str, Optional[np.ndarray]]:
    """Extract scene positions from obs in a suite-aware manner.

    Returns dict with keys like 'object', 'target', etc. Values may be None
    if the position cannot be found in the obs.

    For Object suite: {'object': obj_pos, 'target': basket_pos}
    For Spatial suite: {'object': bowl_pos, 'target': plate_pos}
    For Goal/Long suites: keyword-based search from obs keys
    """
    result: Dict[str, Optional[np.ndarray]] = {}

    if suite_name == "object":
        result["object"] = get_object_pos(obs, task_name)
        result["target"] = get_basket_pos(obs)
        return result

    if suite_name == "spatial":
        # All spatial tasks: pick up black bowl, place on plate
        # obs key is "akita_black_bowl_1_pos" in LIBERO
        bowl_candidates = [
            "akita_black_bowl_1_pos", "akita_black_bowl_2_pos",
            "akita_black_bowl_pos", "black_bowl_1_pos", "black_bowl_pos",
            "bowl_1_pos", "bowl_pos",
        ]
        plate_candidates = ["plate_1_pos", "plate_pos"]
        result["object"] = _find_obs_pos(obs, bowl_candidates)
        result["target"] = _find_obs_pos(obs, plate_candidates)
        return result

    # Goal and Long: match object names against task_name using suffix matching.
    # Search BOTH obs keys and sim body names for comprehensive coverage.
    # obs has movable objects (bowl, bottle); sim has fixtures (drawer, stove, rack).

    # Collect all available positions: obs + sim
    pos_keys = {}
    for k in obs:
        if k.endswith("_pos") and k != "robot0_eef_pos" and "_to_" not in k and k != "robot0_joint_pos":
            val = obs[k]
            if hasattr(val, '__len__') and len(val) == 3:
                pos_keys[k.replace("_pos", "")] = np.array(val, dtype=np.float32)

    # Also add sim body positions (fixtures not in obs)
    if env is not None:
        sim_bodies = _get_sim_body_positions(env)
        for body_name, body_pos in sim_bodies.items():
            clean = body_name.replace("_main", "")
            if clean not in pos_keys:
                pos_keys[clean] = body_pos

    if not pos_keys:
        result["object"] = None
        result["target"] = None
        return result

    task_lower = task_name.lower().replace(" ", "_")
    matches = []  # (position_in_task, key, pos_val, matched_name, match_len)
    for key, pos_val in pos_keys.items():
        obj_name = re.sub(r'_\d+$', '', key)
        # Try progressively shorter suffixes: "akita_black_bowl" -> "black_bowl" -> "bowl"
        parts = obj_name.lower().split("_")
        best_idx, best_suffix, best_len = -1, "", 0
        for start in range(len(parts)):
            suffix = "_".join(parts[start:])
            idx = task_lower.find(suffix)
            if idx >= 0 and len(suffix) > best_len:
                best_idx, best_suffix, best_len = idx, suffix, len(suffix)
        if best_idx >= 0:
            matches.append((best_idx, key, pos_val, best_suffix, best_len))

    # Sort by position in task name (first mentioned = object, last = target)
    matches.sort(key=lambda x: x[0])
    # Deduplicate: keep longest match per unique suffix
    seen_suffixes = set()
    unique_matches = []
    for m in matches:
        if m[3] not in seen_suffixes:
            seen_suffixes.add(m[3])
            unique_matches.append(m)

    result["object"] = unique_matches[0][2] if len(unique_matches) >= 1 else None
    result["target"] = unique_matches[-1][2] if len(unique_matches) >= 2 else None

    # For tasks with single object (e.g. "open the drawer"), target = object
    if result["target"] is None and result["object"] is not None:
        result["target"] = result["object"]

    # Include all positions for richer VLM context
    for key, pos_val in pos_keys.items():
        name = key.replace("_", " ")
        if name not in result:
            result[name] = pos_val

    return result


# ===========================================================================
# State construction (matches preprocess_libero.py + vla_lora_controller.py)
# ===========================================================================

def _compute_direction(eef_pos: np.ndarray, target_pos: Optional[np.ndarray], primitive_type: str) -> np.ndarray:
    """Compute direction vector for template state."""
    if primitive_type == "move" and target_pos is not None:
        delta = target_pos - eef_pos
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            return (delta / norm).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def build_state_12d(
    obs: Dict,
    target_pos: Optional[np.ndarray],
    primitive_type: str,
) -> np.ndarray:
    """Build 12D state vector matching training format.

    State: eef_pos(3) + eef_quat(4) + gripper(2) + direction(3) = 12D
    Used by: Object, Spatial, Goal template models.
    """
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
    direction = _compute_direction(eef_pos, target_pos, primitive_type)
    state = np.concatenate([eef_pos, eef_quat, gripper, direction])
    return state.astype(np.float32)


def build_state_11d(
    obs: Dict,
    target_pos: Optional[np.ndarray],
    primitive_type: str,
) -> np.ndarray:
    """Build 11D state vector for Long template models.

    State: eef_pos(3) + eef_euler(3) + gripper(2) + direction(3) = 11D
    Long suite uses euler angles instead of quaternion.
    """
    from scipy.spatial.transform import Rotation
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    eef_euler = Rotation.from_quat(eef_quat[[1, 2, 3, 0]]).as_euler('xyz').astype(np.float32)
    gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
    direction = _compute_direction(eef_pos, target_pos, primitive_type)
    state = np.concatenate([eef_pos, eef_euler, gripper, direction])
    return state.astype(np.float32)


# ===========================================================================
# Normalization (matches eval_vla_robocasa.py + vla_lora_controller.py)
# ===========================================================================

def normalize_state_min_max(state: np.ndarray, s_min: np.ndarray, s_max: np.ndarray) -> np.ndarray:
    """Min-max normalize state to [-1, 1]."""
    r = s_max - s_min + 1e-8
    return np.clip(2.0 * (state - s_min) / r - 1.0, -1.0, 1.0).astype(np.float32)


def denormalize_action_min_max(action: np.ndarray, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
    """Min-max denormalize action from [-1, 1]."""
    return ((action + 1.0) / 2.0 * (a_max - a_min) + a_min).astype(np.float32)


# ===========================================================================
# GROOT model loading (matches eval_vla_robocasa.py load_groot)
# ===========================================================================

def load_groot_model(
    move_adapter: str,
    device: torch.device,
    denoise_steps: int = 10,
    image_size: int = 224,
) -> Tuple[Any, Any, Dict]:
    """Load GROOT N1.5 with move adapter (weight-diff).

    Returns: (model, eagle_processor, adapter_weights)
    """
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import _build_eagle_processor
    from safetensors.torch import load_file

    # Read lora config from adapter
    lora_rank, lora_alpha = 64, 128
    config_path = os.path.join(move_adapter, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        lora_rank = cfg.get("lora_rank", 64)
        lora_alpha = cfg.get("lora_alpha", lora_rank * 2)

    logger.info(f"Loading GROOT N1.5 (lora_rank={lora_rank}, lora_alpha={lora_alpha})...")
    policy = GrootPolicy.from_pretrained(
        "nvidia/GR00T-N1.5-3B",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Load move adapter weights
    adapter_weights = {}
    for filename in ["model_diff.safetensors", "model.safetensors"]:
        weights_path = os.path.join(move_adapter, filename)
        if os.path.exists(weights_path):
            adapter_weights["move"] = load_file(weights_path)
            logger.info(f"Loaded move adapter: {len(adapter_weights['move'])} keys from {weights_path}")
            break

    if "move" in adapter_weights:
        policy.load_state_dict(adapter_weights["move"], strict=False)
        logger.info("Applied move adapter weights")
    else:
        logger.warning(f"No weights found at {move_adapter}, using base model")

    model = policy.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # torch.compile for faster inference
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile applied (reduce-overhead)")
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")

    # Override denoise steps
    groot_model = getattr(model, "_groot_model", None)
    if groot_model and hasattr(groot_model, "action_head"):
        old = groot_model.action_head.num_inference_timesteps
        groot_model.action_head.num_inference_timesteps = denoise_steps
        logger.info(f"GROOT denoise steps: {old} -> {denoise_steps}")

    eagle_processor = _build_eagle_processor()

    # Monkey-patch: fix Eagle processor compat with transformers 4.53.x
    # HF Hub may re-download cached processor that calls _prepare_image_like_inputs (removed in 4.53)
    from transformers.image_processing_utils_fast import BaseImageProcessorFast
    if not hasattr(BaseImageProcessorFast, '_prepare_image_like_inputs') and hasattr(BaseImageProcessorFast, '_prepare_input_images'):
        BaseImageProcessorFast._prepare_image_like_inputs = BaseImageProcessorFast._prepare_input_images
        logger.info("Monkey-patched Eagle processor: _prepare_image_like_inputs -> _prepare_input_images")

    return model, eagle_processor, adapter_weights


def load_pi05_model(move_adapter: str, device: torch.device, denoise_steps: int = 10):
    """Load PI0.5 policy with adapter weights.

    Returns: (model, tokenizer, {})
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    logger.info("Loading PI0.5...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")

    # Load adapter
    loaded = False
    for filename in ["model_diff.safetensors", "model.safetensors", "adapter_model.safetensors"]:
        weights_path = os.path.join(move_adapter, filename)
        if os.path.exists(weights_path):
            if filename == "adapter_model.safetensors":
                # PEFT LoRA format — manually merge lora_A/lora_B into base weights
                import json
                adapter_weights = load_file(weights_path)
                config_path = os.path.join(move_adapter, "adapter_config.json")
                with open(config_path) as f:
                    adapter_cfg = json.load(f)
                lora_alpha = adapter_cfg.get("lora_alpha", 128)
                lora_r = adapter_cfg.get("r", 64)
                scaling = lora_alpha / lora_r

                # Group lora_A and lora_B by module
                lora_pairs = {}
                for key, tensor in adapter_weights.items():
                    # Strip "base_model.model." prefix to get model-relative key
                    clean = key.replace("base_model.model.", "", 1)
                    if ".lora_A.weight" in clean:
                        base_key = clean.replace(".lora_A.weight", ".weight")
                        lora_pairs.setdefault(base_key, {})["A"] = tensor
                    elif ".lora_B.weight" in clean:
                        base_key = clean.replace(".lora_B.weight", ".weight")
                        lora_pairs.setdefault(base_key, {})["B"] = tensor

                state_dict = policy.state_dict()
                merged = 0
                for base_key, ab in lora_pairs.items():
                    if "A" in ab and "B" in ab and base_key in state_dict:
                        w_device = state_dict[base_key].device
                        delta = (ab["B"].to(w_device).float() @ ab["A"].to(w_device).float()) * scaling
                        state_dict[base_key] = state_dict[base_key].float() + delta
                        merged += 1
                policy.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded PI0.5 PEFT adapter: merged {merged} LoRA pairs (alpha={lora_alpha}, r={lora_r}) from {move_adapter}")
            else:
                adapter_weights = load_file(weights_path)
                policy.load_state_dict(adapter_weights, strict=False)
                logger.info(f"Loaded PI0.5 adapter: {len(adapter_weights)} keys from {weights_path}")
            loaded = True
            break

    if not loaded:
        logger.warning(f"WARNING: No adapter weights found at {move_adapter}! Using base model only.")

    model = policy.to(dtype=torch.bfloat16, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
    return model, tokenizer, {}


def load_smolvla_model(move_adapter: str, device: torch.device, denoise_steps: int = 10):
    """Load SmolVLA policy with adapter weights.

    Returns: (model, tokenizer, {})
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from safetensors.torch import load_file

    logger.info("Loading SmolVLA...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

    # Remap image features to camera1
    img_shape = (3, 512, 512)
    for v in policy.config.input_features.values():
        if v.type == FeatureType.VISUAL:
            img_shape = v.shape
            break
    keys_to_remove = [k for k, v in policy.config.input_features.items() if v.type == FeatureType.VISUAL]
    for key in keys_to_remove:
        del policy.config.input_features[key]
    policy.config.input_features["observation.images.camera1"] = PolicyFeature(type=FeatureType.VISUAL, shape=img_shape)

    # Load adapter
    loaded = False
    for filename in ["model_diff.safetensors", "model.safetensors", "adapter_model.safetensors"]:
        weights_path = os.path.join(move_adapter, filename)
        if os.path.exists(weights_path):
            if filename == "adapter_model.safetensors":
                import json as _json
                adapter_weights = load_file(weights_path)
                config_path = os.path.join(move_adapter, "adapter_config.json")
                with open(config_path) as f:
                    adapter_cfg = _json.load(f)
                lora_alpha = adapter_cfg.get("lora_alpha", 128)
                lora_r = adapter_cfg.get("r", 64)
                scaling = lora_alpha / lora_r

                lora_pairs = {}
                for key, tensor in adapter_weights.items():
                    clean = key.replace("base_model.model.", "", 1)
                    if ".lora_A.weight" in clean:
                        base_key = clean.replace(".lora_A.weight", ".weight")
                        lora_pairs.setdefault(base_key, {})["A"] = tensor
                    elif ".lora_B.weight" in clean:
                        base_key = clean.replace(".lora_B.weight", ".weight")
                        lora_pairs.setdefault(base_key, {})["B"] = tensor

                state_dict = policy.state_dict()
                merged = 0
                for base_key, ab in lora_pairs.items():
                    if "A" in ab and "B" in ab and base_key in state_dict:
                        w_device = state_dict[base_key].device
                        delta = (ab["B"].to(w_device).float() @ ab["A"].to(w_device).float()) * scaling
                        state_dict[base_key] = state_dict[base_key].float() + delta
                        merged += 1
                policy.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded SmolVLA PEFT adapter: merged {merged} LoRA pairs (alpha={lora_alpha}, r={lora_r}) from {move_adapter}")
            else:
                adapter_weights = load_file(weights_path)
                policy.load_state_dict(adapter_weights, strict=False)
                logger.info(f"Loaded SmolVLA adapter: {len(adapter_weights)} keys from {weights_path}")
            loaded = True
            break

    if not loaded:
        logger.warning(f"WARNING: No adapter weights found at {move_adapter}! Using base model only.")

    model = policy.to(dtype=torch.bfloat16, device=device)
    model.eval()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    return model, tokenizer, {}


# ===========================================================================
# Batch construction (matches eval_vla_robocasa.py prepare_groot_batch)
# ===========================================================================

def build_groot_batch(
    image: np.ndarray,
    state_12d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    eagle_processor,
    device: torch.device,
) -> Dict:
    """Build GROOT input batch with normalized state + Eagle image encoding."""
    from PIL import Image as PILImage

    s_min = np.array(state_stats["min"], dtype=np.float32)
    s_max = np.array(state_stats["max"], dtype=np.float32)
    # GROOT uses full 12D state (no _STATE_KEEP_IDX_12D trimming)
    # Only trim if stats dim < state dim (e.g. 9D stats with 12D state)
    stat_dim = len(s_min)
    if len(state_12d) > stat_dim:
        state = state_12d[:stat_dim]  # e.g. 12D → 9D
    else:
        state = state_12d
    state_norm = normalize_state_min_max(state, s_min, s_max)

    # Pad to 64D (GROOT DiT expects 64D state)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm
    state_mask = np.zeros(64, dtype=bool)
    state_mask[:len(state_norm)] = True

    batch = {
        "state": torch.from_numpy(state_padded).float().unsqueeze(0).unsqueeze(0).to(device),
        "state_mask": torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0).to(device),
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    # Eagle image encoding
    pil_img = PILImage.fromarray(image)
    eagle_inputs = eagle_processor(
        text=[f"<image-1> {instruction}"],
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


_STATE_KEEP_IDX_12D = list(range(0, 7)) + list(range(9, 12))  # 12D → 10D (match training)


def build_pi05_batch(
    image: np.ndarray,
    state_12d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    tokenizer,
    device: torch.device,
) -> Dict:
    """Build PI0.5 input batch with quantile-normalized state discretized to 256 bins."""
    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

    # Match state dim to stats dim
    q01_full = np.array(state_stats.get("q01", state_stats["min"]), dtype=np.float32)
    q99_full = np.array(state_stats.get("q99", state_stats["max"]), dtype=np.float32)
    stat_dim = len(q01_full)
    if len(state_12d) == 12 and stat_dim == 12:
        state = state_12d[_STATE_KEEP_IDX_12D]  # 12D → 10D
        q01 = q01_full[_STATE_KEEP_IDX_12D]
        q99 = q99_full[_STATE_KEEP_IDX_12D]
    elif len(state_12d) > stat_dim:
        state = state_12d[:stat_dim]  # e.g. 12D → 9D
        q01 = q01_full
        q99 = q99_full
    else:
        state = state_12d
        q01 = q01_full
        q99 = q99_full

    state_norm = np.clip((state - q01) / (q99 - q01 + 1e-8) * 2 - 1, -1, 1)
    state_padded = np.zeros(32, dtype=np.float32)
    state_padded[:len(state_norm)] = state_norm

    # Discretize to 256 bins
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


def build_smolvla_batch(
    image: np.ndarray,
    state_12d: np.ndarray,
    instruction: str,
    state_stats: Dict,
    tokenizer,
    device: torch.device,
) -> Dict:
    """Build SmolVLA input batch with mean-std normalized state."""
    import torch.nn.functional as F

    img_t = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    if img_t.shape[-1] != 512 or img_t.shape[-2] != 512:
        img_t = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)

    # Match state dim to stats dim
    s_mean_full = np.array(state_stats["mean"], dtype=np.float32)
    s_std_full = np.array(state_stats["std"], dtype=np.float32)
    stat_dim = len(s_mean_full)
    if len(state_12d) == 12 and stat_dim == 12:
        state = state_12d[_STATE_KEEP_IDX_12D]  # 12D → 10D
        s_mean = s_mean_full[_STATE_KEEP_IDX_12D]
        s_std = s_std_full[_STATE_KEEP_IDX_12D]
    elif len(state_12d) > stat_dim:
        state = state_12d[:stat_dim]  # e.g. 12D → 9D
        s_mean = s_mean_full
        s_std = s_std_full
    else:
        state = state_12d
        s_mean = s_mean_full
        s_std = s_std_full

    state_norm = ((state - s_mean) / (s_std + 1e-8)).astype(np.float32)

    text = instruction + "\n"
    tokens = tokenizer(text, max_length=48, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "observation.images.camera1": img_t.to(device),
        "observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }


@torch.no_grad()
def predict_action_chunk(model, batch) -> np.ndarray:
    """Run GROOT inference, return (1, T, D) numpy array."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch)
    return actions.float().cpu().numpy()


@torch.no_grad()
def predict_action_chunk_generic(model, batch, num_steps: int = 10) -> np.ndarray:
    """Run PI0.5/SmolVLA inference, return (1, T, D) numpy array."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = model.predict_action_chunk(batch, num_steps=num_steps)
    return actions[0].float().cpu().numpy()  # (T, D)


# ===========================================================================
# Backend dispatch helpers
# ===========================================================================

def _build_batch_and_predict(
    model, image, state, instruction, state_stats,
    eagle_processor, tokenizer, device, backend="groot_n1.5",
    action_stats=None,
):
    """Build batch and predict action chunk for any backend.

    Returns: chunk_buffer as numpy (T, 7) in min-max normalized space [-1, 1].
    All backends' outputs are converted to min-max [-1, 1] space so callers
    can uniformly apply denormalize_action_min_max(chunk, a_min, a_max).
    """
    if backend == "groot_n1.5":
        batch = build_groot_batch(image, state, instruction, state_stats, eagle_processor, device)
        all_chunks = predict_action_chunk(model, batch)
        return all_chunks[0, :, :7]  # (T, 7) already min-max normalized

    elif backend == "pi05":
        batch = build_pi05_batch(image, state, instruction, state_stats, tokenizer, device)
        chunk = predict_action_chunk_generic(model, batch, num_steps=10)
        chunk = chunk[:, :7]  # (T, 7) in q01/q99 [-1, 1] space

        # Convert from q01/q99 space → raw → min/max space
        q01 = np.array(action_stats["q01"][:7], dtype=np.float32)
        q99 = np.array(action_stats["q99"][:7], dtype=np.float32)
        a_min = np.array(action_stats["min"][:7], dtype=np.float32)
        a_max = np.array(action_stats["max"][:7], dtype=np.float32)
        raw = (np.clip(chunk, -1.0, 1.0) + 1.0) / 2.0 * (q99 - q01) + q01
        return np.clip(2.0 * (raw - a_min) / (a_max - a_min + 1e-8) - 1.0, -1.0, 1.0)

    elif backend == "smolvla":
        batch = build_smolvla_batch(image, state, instruction, state_stats, tokenizer, device)
        chunk = predict_action_chunk_generic(model, batch, num_steps=10)
        act_dim = chunk.shape[-1]  # typically 6
        # Pad to 7D (gripper dim = 0 in mean-std space → maps to mean)
        if act_dim < 7:
            pad = np.zeros((chunk.shape[0], 7 - act_dim), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=-1)
        chunk = chunk[:, :7]  # (T, 7) in mean-std space (unbounded)

        # Convert from mean/std space → raw → min/max space
        a_mean = np.array(action_stats["mean"][:7], dtype=np.float32)
        a_std = np.array(action_stats["std"][:7], dtype=np.float32)
        a_min = np.array(action_stats["min"][:7], dtype=np.float32)
        a_max = np.array(action_stats["max"][:7], dtype=np.float32)
        raw = chunk * a_std + a_mean
        return np.clip(2.0 * (raw - a_min) / (a_max - a_min + 1e-8) - 1.0, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown backend: {backend}")


# ===========================================================================
# Episode runner
# ===========================================================================

def run_episode(
    model,
    eagle_processor,
    env,
    obs: Dict,
    task_name: str,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    max_steps: int = 280,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    record_video: bool = False,
    video_size: int = 256,
    episode_idx: int = 0,
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
) -> Dict[str, Any]:
    """Run one episode with template plan FSM.

    Primitives:
      1. move_to_object: VLA drives toward object, direction = normalize(obj - eef)
      2. grip: Close gripper for N steps (rule-based)
      3. move_to_basket: VLA drives toward basket, direction = normalize(basket - eef)
      4. release: Open gripper for N steps (rule-based)
    """
    device = next(model.parameters()).device
    a_min = np.array(action_stats["min"][:7], dtype=np.float32)
    a_max = np.array(action_stats["max"][:7], dtype=np.float32)

    frames = []
    total_steps = 0
    total_reward = 0.0
    prev_action = None

    # Reset model internal state
    if hasattr(model, "reset"):
        model.reset()

    for prim_idx, prim in enumerate(TEMPLATE_PLAN):
        prim_type = prim["type"]
        prim_instruction = prim["instruction"]
        target_key = prim["target"]
        budget = prim["steps_budget"]

        # Determine target position
        if target_key == "object":
            target_pos = get_object_pos(obs, task_name)
        elif target_key == "basket":
            target_pos = get_basket_pos(obs)
        else:
            target_pos = None

        logger.info(
            f"  Primitive {prim_idx + 1}/{len(TEMPLATE_PLAN)}: "
            f"{prim_type} -> {target_key} "
            f"(budget={budget}, target={[round(float(v), 3) for v in target_pos] if target_pos is not None else 'N/A'})"
        )

        # VLA-driven grip/release (direction=[0,0,0] matches training)
        # In training, grip phases include position actions (descent/lift),
        # so the model must predict them — rule-based zeros lose this motion.
        if prim_type in ("grip", "release"):
            gripper_val = 1.0 if prim_type == "grip" else -1.0
            chunk_buffer_g = None
            chunk_step_g = 0
            prev_action_g = None

            if hasattr(model, "reset"):
                model.reset()

            for step_i in range(budget):
                if total_steps >= max_steps:
                    break

                need_new = (chunk_buffer_g is None or chunk_step_g >= chunk_stride)
                if need_new:
                    state_12d = build_state_12d(obs, None, "grip")
                    image = get_libero_image(obs, size=image_size)
                    chunk_buffer_g = _build_batch_and_predict(
                        model, image, state_12d, "grip", state_stats,
                        eagle_processor, tokenizer, device, backend,
                        action_stats=action_stats,
                    )
                    chunk_step_g = 0
                    if hasattr(model, "reset"):
                        model.reset()

                action_norm = chunk_buffer_g[chunk_step_g].copy()
                chunk_step_g += 1
                action_norm = np.clip(action_norm, -1.0, 1.0)
                action_7d = denormalize_action_min_max(action_norm, a_min, a_max)

                # Override gripper to ensure correct close/open
                action_7d[6] = gripper_val

                # EMA smoothing (position + rotation only)
                if prev_action_g is not None and ema_alpha > 0:
                    action_7d[:6] = ema_alpha * prev_action_g[:6] + (1 - ema_alpha) * action_7d[:6]
                prev_action_g = action_7d.copy()

                if step_i < 3 or step_i % 10 == 0:
                    eef = obs.get("robot0_eef_pos", np.zeros(3))
                    logger.info(
                        f"    [{prim_idx+1}] {prim_type} step={step_i} "
                        f"action={[round(float(v), 4) for v in action_7d[:3]]} "
                        f"grip={action_7d[6]:.2f} "
                        f"eef={[round(float(v), 3) for v in eef]}"
                    )

                obs, reward, done, info = env.step(action_7d.tolist())
                total_reward += reward
                total_steps += 1

                if record_video:
                    frames.append(get_libero_image(obs, size=video_size))

                if done:
                    break

            prev_action = None  # Reset EMA at primitive boundary
            if hasattr(model, "reset"):
                model.reset()
            if done:
                break
            continue

        # VLA-driven move primitive
        chunk_buffer = None
        chunk_step = 0
        prev_action = None  # Reset EMA at primitive boundary

        if hasattr(model, "reset"):
            model.reset()

        for step_i in range(budget):
            if total_steps >= max_steps:
                break

            # Update target position each step (objects may move)
            if target_key == "object":
                target_pos = get_object_pos(obs, task_name)
            elif target_key == "basket":
                target_pos = get_basket_pos(obs)

            # Check if we need a new chunk
            need_new_chunk = (chunk_buffer is None or chunk_step >= chunk_stride)

            if need_new_chunk:
                # Build state with direction vector
                state_12d = build_state_12d(obs, target_pos, "move")
                image = get_libero_image(obs, size=image_size)

                chunk_buffer = _build_batch_and_predict(
                    model, image, state_12d, prim_instruction, state_stats,
                    eagle_processor, tokenizer, device, backend,
                    action_stats=action_stats,
                )
                chunk_step = 0

                # Reset model internal queue to prevent stale actions
                if hasattr(model, "reset"):
                    model.reset()

            # Get action from chunk
            action_norm = chunk_buffer[chunk_step].copy()
            chunk_step += 1

            # Denormalize action
            action_norm = np.clip(action_norm, -1.0, 1.0)
            action_7d = denormalize_action_min_max(action_norm, a_min, a_max)

            # EMA smoothing (first 6 dims only, not gripper)
            if prev_action is not None and ema_alpha > 0:
                action_7d[:6] = ema_alpha * prev_action[:6] + (1 - ema_alpha) * action_7d[:6]
            prev_action = action_7d.copy()

            # Debug logging
            if step_i < 3 or step_i % 50 == 0:
                eef = obs.get("robot0_eef_pos", np.zeros(3))
                dist_to_target = (
                    np.linalg.norm(target_pos - eef) if target_pos is not None else -1
                )
                logger.info(
                    f"    [{prim_idx+1}] step={step_i} "
                    f"action={[round(float(v), 4) for v in action_7d[:3]]} "
                    f"grip={action_7d[6]:.2f} "
                    f"eef={[round(float(v), 3) for v in eef]} "
                    f"dist={dist_to_target:.4f}"
                )

            # Step environment
            obs, reward, done, info = env.step(action_7d.tolist())
            total_reward += reward
            total_steps += 1

            if record_video:
                frames.append(get_libero_image(obs, size=video_size))

            if done:
                break

        if done or total_steps >= max_steps:
            break

    # Check success
    success = bool(done)  # LIBERO sets done=True on task completion
    return {
        "success": success,
        "steps": total_steps,
        "reward": float(total_reward),
        "frames": frames if record_video else [],
    }


# ===========================================================================
# Raw mode: continuous VLA inference without FSM
# ===========================================================================

def build_state_9d(obs: Dict) -> np.ndarray:
    """Build 9D state: eef_pos(3) + eef_quat(4) + gripper(2). Standard VLA proprioception."""
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
    return np.concatenate([eef_pos, eef_quat, gripper]).astype(np.float32)


def build_state_8d(obs: Dict) -> np.ndarray:
    """Build 8D state for Long: eef_pos(3) + eef_euler(3) + gripper(2)."""
    from scipy.spatial.transform import Rotation
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    # robosuite quaternion is (w,x,y,z) but scipy expects (x,y,z,w)
    eef_euler = Rotation.from_quat(eef_quat[[1, 2, 3, 0]]).as_euler('xyz').astype(np.float32)
    gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
    return np.concatenate([eef_pos, eef_euler, gripper]).astype(np.float32)


def build_state_15d(obs: Dict, task_name: str) -> np.ndarray:
    """Build 15D state: eef_pos(3) + eef_quat(4) + gripper(2) + object_pos(3) + basket_pos(3)."""
    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
    obj_pos = get_object_pos(obs, task_name)
    if obj_pos is None:
        obj_pos = np.zeros(3, dtype=np.float32)
    basket_pos = get_basket_pos(obs)
    if basket_pos is None:
        basket_pos = np.zeros(3, dtype=np.float32)
    return np.concatenate([eef_pos, eef_quat, gripper, obj_pos, basket_pos]).astype(np.float32)


def run_episode_raw(
    model,
    eagle_processor,
    env,
    obs: Dict,
    task_name: str,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    max_steps: int = 400,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    record_video: bool = False,
    video_size: int = 256,
    episode_idx: int = 0,
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
) -> Dict[str, Any]:
    """Run one episode with raw continuous VLA inference (no FSM).

    Supports both 12D (direction-based) and 15D (object+basket pos) state formats.
    State dim auto-detected from state_stats.
    """
    device = next(model.parameters()).device
    a_min = np.array(action_stats["min"][:7], dtype=np.float32)
    a_max = np.array(action_stats["max"][:7], dtype=np.float32)

    state_dim = len(state_stats["min"])
    use_15d = (state_dim >= 15)
    use_9d = (state_dim == 9)
    use_8d = (state_dim == 8)

    frames = []
    total_reward = 0.0
    done = False
    prev_action = None
    chunk_buffer = None
    chunk_step = 0

    if hasattr(model, "reset"):
        model.reset()

    # Divergence detection: early-terminate if EEF stuck
    STUCK_WINDOW = 60
    STUCK_THRESHOLD = 0.008
    eef_history = []
    consecutive_stuck = 0

    for step_i in range(max_steps):
        need_new = (chunk_buffer is None or chunk_step >= chunk_stride)
        if need_new:
            if use_8d:
                state = build_state_8d(obs)
            elif use_9d:
                state = build_state_9d(obs)
            elif use_15d:
                state = build_state_15d(obs, task_name)
            else:
                target_pos = get_object_pos(obs, task_name)
                state = build_state_12d(obs, target_pos, "move")

            if use_8d or use_9d or use_15d:
                from PIL import Image as PILImage
                img = obs["agentview_image"]
                if suite_name in ("spatial", "goal", "long"):
                    img = img[:, ::-1].copy()
                else:
                    img = img[::-1, ::-1].copy()
                if img.shape[0] != image_size or img.shape[1] != image_size:
                    img = np.array(PILImage.fromarray(img).resize((image_size, image_size), PILImage.BILINEAR))
                image = img
            else:
                image = get_libero_image(obs, size=image_size)

            chunk_buffer = _build_batch_and_predict(
                model, image, state, instruction, state_stats,
                eagle_processor, tokenizer, device, backend,
                action_stats=action_stats,
            )
            chunk_step = 0
            if hasattr(model, "reset"):
                model.reset()

        action_norm = chunk_buffer[chunk_step].copy()
        chunk_step += 1
        action_norm = np.clip(action_norm, -1.0, 1.0)
        action_7d = denormalize_action_min_max(action_norm, a_min, a_max)

        # EMA smoothing (position + rotation only, not gripper)
        if prev_action is not None and ema_alpha > 0:
            action_7d[:6] = ema_alpha * prev_action[:6] + (1 - ema_alpha) * action_7d[:6]
        prev_action = action_7d.copy()

        # Track EEF for divergence detection
        eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
        eef_history.append(eef.copy())

        # Early termination: if stuck for 2 consecutive windows after step 150 → abort
        if len(eef_history) >= STUCK_WINDOW and step_i >= 150:
            displacement = float(np.linalg.norm(eef - eef_history[-STUCK_WINDOW]))
            if displacement < STUCK_THRESHOLD:
                consecutive_stuck += 1
                if consecutive_stuck >= 2:
                    logger.info(
                        f"  [raw] DIVERGED step={step_i} disp={displacement:.4f} — early stop"
                    )
                    break
            else:
                consecutive_stuck = 0

        if step_i < 3 or step_i % 100 == 0:
            positions = get_scene_positions(obs, suite_name, task_name, env=env)
            obj = positions.get("object")
            dist = float(np.linalg.norm(obj - eef)) if obj is not None else -1
            logger.info(
                f"  [raw] step={step_i} "
                f"action={[round(float(v), 3) for v in action_7d[:3]]} "
                f"grip={action_7d[6]:.2f} "
                f"eef={[round(float(v), 3) for v in eef]} "
                f"dist={dist:.4f}"
            )

        try:
            obs, reward, done, info = env.step(action_7d.tolist())
        except ValueError as e:
            if "terminated" in str(e):
                # robosuite raises this when env is internally done
                done = True
                reward = 0.0
            else:
                raise
        total_reward += reward

        if record_video and not done:
            frames.append(get_libero_image(obs, size=video_size))

        if done:
            break

    success = bool(done)
    return {
        "success": success,
        "steps": step_i + 1,
        "reward": float(total_reward),
        "frames": frames if record_video else [],
    }


def run_episode_monitored_raw(
    model,
    eagle_processor,
    env,
    obs: Dict,
    task_name: str,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    max_steps: int = 500,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    record_video: bool = False,
    video_size: int = 256,
    episode_idx: int = 0,
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
    stuck_window: int = 50,
    max_attempts: int = 2,
    init_state=None,
) -> Dict[str, Any]:
    """Raw VLA with env-level retry on failure.

    Total step budget is shared across all attempts (same wall-clock as raw).
    On failure: reset env to init_state and retry with remaining budget.
    Stuck detection: reset chunk buffer when EEF stops moving.
    No VLM calls — same speed as raw for successful episodes.
    """
    device = next(model.parameters()).device
    a_min = np.array(action_stats["min"][:7], dtype=np.float32)
    a_max = np.array(action_stats["max"][:7], dtype=np.float32)

    state_dim = len(state_stats["min"])
    use_15d = (state_dim >= 15)
    use_9d = (state_dim == 9)
    use_8d = (state_dim == 8)

    frames = []
    total_steps = 0
    STUCK_THRESHOLD = 0.008  # EEF displacement threshold over window

    for attempt in range(max_attempts):
        if total_steps >= max_steps:
            break

        if attempt > 0:
            # Reset env to init state for retry
            if init_state is not None:
                obs = env.set_init_state(init_state)
                for _ in range(5):
                    obs, _, _, _ = env.step([0.0] * 7)
            logger.info(f"  [mon-raw] Retry {attempt+1}/{max_attempts} (budget left: {max_steps - total_steps})")

        prev_action = None
        chunk_buffer = None
        chunk_step = 0
        eef_history = []
        buffer_resets = 0
        remaining_steps = max_steps - total_steps

        if hasattr(model, "reset"):
            model.reset()

        done = False
        aborted_early = False
        for step_i in range(remaining_steps):
            need_new = (chunk_buffer is None or chunk_step >= chunk_stride)
            if need_new:
                if use_8d:
                    state = build_state_8d(obs)
                elif use_9d:
                    state = build_state_9d(obs)
                elif use_15d:
                    state = build_state_15d(obs, task_name)
                else:
                    target_pos = get_object_pos(obs, task_name)
                    state = build_state_12d(obs, target_pos, "move")

                if use_8d or use_9d or use_15d:
                    from PIL import Image as PILImage
                    img = obs["agentview_image"]
                    if suite_name in ("spatial", "goal", "long"):
                        img = img[:, ::-1].copy()
                    else:
                        img = img[::-1, ::-1].copy()
                    if img.shape[0] != image_size or img.shape[1] != image_size:
                        img = np.array(PILImage.fromarray(img).resize((image_size, image_size), PILImage.BILINEAR))
                    image = img
                else:
                    image = get_libero_image(obs, size=image_size)

                chunk_buffer = _build_batch_and_predict(
                    model, image, state, instruction, state_stats,
                    eagle_processor, tokenizer, device, backend,
                    action_stats=action_stats,
                )
                chunk_step = 0
                if hasattr(model, "reset"):
                    model.reset()

            action_norm = chunk_buffer[chunk_step].copy()
            chunk_step += 1
            action_norm = np.clip(action_norm, -1.0, 1.0)
            action_7d = denormalize_action_min_max(action_norm, a_min, a_max)

            if prev_action is not None and ema_alpha > 0:
                action_7d[:6] = ema_alpha * prev_action[:6] + (1 - ema_alpha) * action_7d[:6]
            prev_action = action_7d.copy()

            # Track EEF for stuck detection
            eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
            eef_history.append(eef.copy())

            # Stuck detection — reset buffer, or abort attempt early to save budget for retry
            if len(eef_history) >= stuck_window:
                window_start = eef_history[-stuck_window]
                displacement = float(np.linalg.norm(eef - window_start))
                if displacement < STUCK_THRESHOLD:
                    if buffer_resets < 3:
                        buffer_resets += 1
                        chunk_buffer = None
                        chunk_step = 0
                        prev_action = None
                        eef_history = eef_history[-10:]
                        if step_i % 50 == 0 or buffer_resets == 1:
                            logger.info(
                                f"  [mon-raw] STUCK step={total_steps + step_i} "
                                f"disp={displacement:.4f} reset#{buffer_resets}"
                            )
                    elif attempt < max_attempts - 1:
                        # All buffer resets exhausted and still stuck — abort early for retry
                        logger.info(
                            f"  [mon-raw] ABORT attempt {attempt+1} at step={total_steps + step_i} "
                            f"(stuck after {buffer_resets} resets, saving budget for retry)"
                        )
                        total_steps += step_i + 1
                        aborted_early = True
                        break

            if (total_steps + step_i) < 3 or (total_steps + step_i) % 100 == 0:
                positions = get_scene_positions(obs, suite_name, task_name, env=env)
                obj = positions.get("object")
                dist = float(np.linalg.norm(obj - eef)) if obj is not None else -1
                logger.info(
                    f"  [mon-raw] step={total_steps + step_i} "
                    f"eef={[round(float(v), 3) for v in eef]} "
                    f"dist={dist:.4f}"
                )

            try:
                obs, reward, done, info = env.step(action_7d.tolist())
            except ValueError as e:
                if "terminated" in str(e):
                    done = True
                    reward = 0.0
                else:
                    raise

            if record_video and not done:
                frames.append(get_libero_image(obs, size=video_size))

            if done:
                total_steps += step_i + 1
                break

        if not done and not aborted_early:
            total_steps += remaining_steps

        if done:
            break

    success = bool(done)
    return {
        "success": success,
        "steps": total_steps,
        "reward": 1.0 if success else 0.0,
        "attempts": attempt + 1,
        "frames": frames if record_video else [],
    }


# ===========================================================================
# VLM Planner + Monitor Pipeline Mode
# ===========================================================================

def init_vlm(provider: str, model: str, temperature: float = 0.3, max_tokens: int = 1024):
    """Initialize a VLM via langchain for planning/monitoring."""
    if provider == "bedrock":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from robobridge.modules.bedrock_bearer import create_bedrock_bearer_chat
        llm = create_bedrock_bearer_chat(model=model, temperature=temperature, max_tokens=max_tokens)
        if llm is None:
            raise RuntimeError("Bedrock init failed. Set AWS_BEARER_TOKEN_BEDROCK.")
        return llm
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


def _encode_image_b64(img: np.ndarray) -> str:
    """Encode numpy image to base64 JPEG string."""
    from PIL import Image as PILImage
    from io import BytesIO
    import base64
    buf = BytesIO()
    PILImage.fromarray(img).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


LIBERO_PLANNER_SYSTEM = """You are a robot manipulation planner for pick-and-place tasks.
The robot has a parallel-jaw gripper and operates in a tabletop environment.

Coordinate system (robot base frame):
- X: forward (away from robot) / backward (toward robot)
- Y: left (+) / right (-)
- Z: up (+) / down (-)

You must output a JSON plan with a list of primitives. Each primitive is one of:
- {"type": "move", "x": float, "y": float, "z": float, "steps": int}
  Move the end-effector to the given position.
- {"type": "grip", "steps": int}
  Close the gripper to grasp an object.
- {"type": "release", "steps": int}
  Open the gripper to release an object.

Guidelines for pick-and-place:
1. First MOVE to a pre-grasp position slightly above the object (~3-5cm above)
2. Then MOVE down to the grasp position (at or slightly below object center)
3. GRIP to close gripper
4. MOVE up to a safe lift height (~10cm above table)
5. MOVE to above the basket
6. RELEASE to drop the object
7. Allocate reasonable step budgets (move: 60-120 steps, grip: 20-30, release: 15-20)

Output ONLY valid JSON: {"primitives": [...]}"""

# Alias for Object suite
LIBERO_PLANNER_SYSTEM_OBJECT = LIBERO_PLANNER_SYSTEM

LIBERO_PLANNER_SYSTEM_SPATIAL = """You are a robot manipulation planner for pick-and-place tasks.
The robot has a parallel-jaw gripper and operates in a tabletop environment.

Task type: Pick up a black bowl from a specific location and place it on a plate.

Coordinate system (robot base frame):
- X: forward (away from robot) / backward (toward robot)
- Y: left (+) / right (-)
- Z: up (+) / down (-)

You must output a JSON plan with a list of primitives. Each primitive is one of:
- {"type": "move", "x": float, "y": float, "z": float, "steps": int}
  Move the end-effector to the given position.
- {"type": "grip", "steps": int}
  Close the gripper to grasp an object.
- {"type": "release", "steps": int}
  Open the gripper to release an object.

Guidelines:
1. MOVE to pre-grasp position above the black bowl (~3-5cm above)
2. MOVE down to grasp the bowl (at or slightly below bowl center)
3. GRIP to close gripper
4. MOVE up to a safe lift height (~8-10cm above table)
5. MOVE to above the plate
6. RELEASE to place the bowl on the plate
7. Use the IMAGE to identify the bowl's location if coordinates are not available
8. Allocate reasonable step budgets (move: 60-120 steps, grip: 20-30, release: 15-20)

Output ONLY valid JSON: {"primitives": [...]}"""

LIBERO_PLANNER_SYSTEM_GOAL = """You are a robot manipulation planner for diverse tabletop tasks.
The robot has a parallel-jaw gripper and operates in a tabletop environment.

Task types include: opening/closing drawers and doors, pushing objects, turning knobs,
picking and placing objects on various surfaces (stove, cabinet, plate, rack, bowl).

Coordinate system (robot base frame):
- X: forward (away from robot) / backward (toward robot)
- Y: left (+) / right (-)
- Z: up (+) / down (-)

You must output a JSON plan with a list of primitives. Each primitive is one of:
- {"type": "move", "x": float, "y": float, "z": float, "steps": int}
  Move the end-effector to the given position.
- {"type": "grip", "steps": int}
  Close the gripper to grasp an object or grip a handle.
- {"type": "release", "steps": int}
  Open the gripper to release an object.

Guidelines for different task types:
- Pick-and-place: move above object -> descend -> grip -> lift -> move to target -> release
- Open drawer/door: move to handle -> grip -> pull/push motion -> release
- Push object: move behind object -> move forward to push (no grip needed)
- Turn knob/stove: move to knob -> grip -> rotate motion -> release

Key principles:
1. Use the IMAGE to understand the scene layout and object positions
2. Position estimates from coordinates (if available) are approximate — trust the image
3. For grip-based tasks (open, pick, turn): approach, grip, execute motion, release
4. For push tasks: approach from behind, push forward without gripping
5. Allocate reasonable step budgets (move: 60-120 steps, grip: 20-30, release: 15-20)
6. Plan 4-6 primitives for single-action tasks

Output ONLY valid JSON: {"primitives": [...]}"""

LIBERO_PLANNER_SYSTEM_LONG = """You are a robot manipulation planner for multi-step tabletop tasks.
The robot has a parallel-jaw gripper and operates in a tabletop environment.

These are LONG-HORIZON tasks that involve MULTIPLE sequential sub-tasks, such as:
- Pick up two objects and place them in different locations
- Perform an action (open drawer, turn on stove) then manipulate an object
- Place multiple objects on different targets

Coordinate system (robot base frame):
- X: forward (away from robot) / backward (toward robot)
- Y: left (+) / right (-)
- Z: up (+) / down (-)

You must output a JSON plan with a list of primitives. Each primitive is one of:
- {"type": "move", "x": float, "y": float, "z": float, "steps": int}
  Move the end-effector to the given position.
- {"type": "grip", "steps": int}
  Close the gripper to grasp an object.
- {"type": "release", "steps": int}
  Open the gripper to release an object.

Guidelines for multi-step tasks:
1. Break the task into sequential sub-goals (e.g., pick A -> place A -> pick B -> place B)
2. For each sub-goal, follow the standard primitive sequence:
   move above -> descend -> grip -> lift -> move to target -> release
3. Use the IMAGE to understand the full scene and plan the order of operations
4. Plan 8-14 primitives total (2-3 sub-goals x 4-6 primitives each)
5. Allocate step budgets: move 60-100 steps, grip 20-30, release 15-20
6. If a sub-task involves opening/closing, grip the handle and pull/push

Output ONLY valid JSON: {"primitives": [...]}"""

# Map suite name to its planner prompt
_SUITE_PLANNER_PROMPTS = {
    "object": LIBERO_PLANNER_SYSTEM_OBJECT,
    "spatial": LIBERO_PLANNER_SYSTEM_SPATIAL,
    "goal": LIBERO_PLANNER_SYSTEM_GOAL,
    "long": LIBERO_PLANNER_SYSTEM_LONG,
}


def vlm_plan_libero(
    vlm,
    image: np.ndarray,
    instruction: str,
    object_pos: np.ndarray,
    basket_pos: np.ndarray,
    eef_pos: np.ndarray,
    failure_context: Optional[str] = None,
) -> Optional[List[Dict]]:
    """Call VLM to generate a pick-and-place plan for LIBERO."""
    from langchain_core.messages import HumanMessage, SystemMessage
    import re

    user_text = (
        f"Task: {instruction}\n"
        f"Object position: [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]\n"
        f"Basket position: [{basket_pos[0]:.3f}, {basket_pos[1]:.3f}, {basket_pos[2]:.3f}]\n"
        f"Current EEF position: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]\n"
    )
    if failure_context:
        user_text += f"\nPrevious attempt FAILED: {failure_context}\nGenerate a DIFFERENT plan with a different approach angle or strategy.\n"

    img_b64 = _encode_image_b64(image)
    messages = [
        SystemMessage(content=LIBERO_PLANNER_SYSTEM),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]),
    ]

    try:
        response = vlm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*"primitives"[\s\S]*\}', text)
        if json_match:
            plan_data = json.loads(json_match.group())
            primitives = plan_data.get("primitives", [])
            logger.info(f"VLM plan: {len(primitives)} primitives")
            for i, p in enumerate(primitives):
                logger.info(f"  [{i+1}] {p}")
            return primitives
        else:
            logger.warning(f"VLM response has no JSON: {text[:200]}")
            return None
    except Exception as e:
        logger.error(f"VLM planning failed: {e}")
        return None


def make_fallback_plan(object_pos: np.ndarray, basket_pos: np.ndarray) -> List[Dict]:
    """Generate a fallback template plan when VLM fails."""
    return [
        {"type": "move", "x": float(object_pos[0]), "y": float(object_pos[1]),
         "z": float(object_pos[2]) + 0.05, "steps": 80},   # above object
        {"type": "move", "x": float(object_pos[0]), "y": float(object_pos[1]),
         "z": float(object_pos[2]) - 0.01, "steps": 40},   # grasp height
        {"type": "grip", "steps": 25},
        {"type": "move", "x": float(object_pos[0]), "y": float(object_pos[1]),
         "z": float(object_pos[2]) + 0.15, "steps": 50},   # lift
        {"type": "move", "x": float(basket_pos[0]), "y": float(basket_pos[1]),
         "z": float(basket_pos[2]) + 0.15, "steps": 80},   # above basket
        {"type": "release", "steps": 15},
    ]


# ---------------------------------------------------------------------------
# Generic (suite-aware) VLM planner + fallback
# ---------------------------------------------------------------------------

def vlm_plan_generic(
    vlm,
    image: np.ndarray,
    instruction: str,
    eef_pos: np.ndarray,
    scene_positions: Dict[str, Optional[np.ndarray]],
    suite_name: str,
    failure_context: Optional[str] = None,
) -> Optional[List[Dict]]:
    """Call VLM to generate a manipulation plan for any LIBERO suite.

    Uses the suite-specific system prompt and includes whatever position
    information is available (eef_pos always, object/target if found).
    Falls back gracefully when positions are missing -- the VLM relies
    on the image in that case.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    import re

    system_prompt = _SUITE_PLANNER_PROMPTS.get(suite_name, LIBERO_PLANNER_SYSTEM_GOAL)

    # Build user message with available position information
    user_lines = [f"Task: {instruction}"]
    user_lines.append(f"Current EEF position: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

    obj_pos = scene_positions.get("object")
    tgt_pos = scene_positions.get("target")

    if suite_name == "object":
        # Object suite: use object/basket naming for backward compatibility
        if obj_pos is not None:
            user_lines.append(f"Object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        if tgt_pos is not None:
            user_lines.append(f"Basket position: [{tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f}]")
    elif suite_name == "spatial":
        if obj_pos is not None:
            user_lines.append(f"Black bowl position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        if tgt_pos is not None:
            user_lines.append(f"Plate position: [{tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f}]")
    else:
        # Goal / Long: generic naming + all obs positions for spatial context
        if obj_pos is not None:
            user_lines.append(f"Primary object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        if tgt_pos is not None and (obj_pos is None or not np.allclose(tgt_pos, obj_pos)):
            user_lines.append(f"Target position: [{tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f}]")
        # Pass all available scene positions for richer spatial context
        extra_pos = {k: v for k, v in scene_positions.items()
                     if k not in ("object", "target") and v is not None}
        if extra_pos:
            user_lines.append("Scene objects:")
            for k, v in sorted(extra_pos.items()):
                user_lines.append(f"  {k}: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
        elif obj_pos is None and tgt_pos is None:
            user_lines.append("No object positions available. Use the image to determine positions.")

    if failure_context:
        user_lines.append(f"\nPrevious attempt FAILED: {failure_context}")
        user_lines.append("Generate a COMPLETELY DIFFERENT plan. Use different coordinates, approach angles, and strategy.")
        user_lines.append("Do NOT reuse the same target coordinates that failed.")

    user_text = "\n".join(user_lines)
    img_b64 = _encode_image_b64(image)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]),
    ]

    try:
        response = vlm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{[\s\S]*"primitives"[\s\S]*\}', text)
        if json_match:
            plan_data = json.loads(json_match.group())
            primitives = plan_data.get("primitives", [])
            logger.info(f"VLM plan ({suite_name}): {len(primitives)} primitives")
            for i, p in enumerate(primitives):
                logger.info(f"  [{i+1}] {p}")
            return primitives
        else:
            logger.warning(f"VLM response has no JSON: {text[:200]}")
            return None
    except Exception as e:
        logger.error(f"VLM planning failed: {e}")
        return None


def _pick_place_fallback(obj_pos: np.ndarray, tgt_pos: np.ndarray) -> List[Dict]:
    """Standard pick-and-place fallback: approach obj, grip, move to target, release."""
    return [
        {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]),
         "z": float(obj_pos[2]) + 0.05, "steps": 80},
        {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]),
         "z": float(obj_pos[2]) - 0.01, "steps": 40},
        {"type": "grip", "steps": 25},
        {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]),
         "z": float(obj_pos[2]) + 0.15, "steps": 50},
        {"type": "move", "x": float(tgt_pos[0]), "y": float(tgt_pos[1]),
         "z": float(tgt_pos[2]) + 0.15, "steps": 80},
        {"type": "release", "steps": 15},
    ]


def make_fallback_plan_generic(
    suite_name: str,
    task_name: str,
    eef_pos: np.ndarray,
    scene_positions: Dict[str, Optional[np.ndarray]],
) -> List[Dict]:
    """Generate a suite-aware fallback plan when VLM fails.

    Object/Spatial: pick-and-place to basket/plate
    Goal: 2-4 primitive sequence based on task keywords
    Long: 6-8 primitive multi-step sequence based on task keywords
    """
    obj_pos = scene_positions.get("object")
    tgt_pos = scene_positions.get("target")

    # Default positions if not found
    if obj_pos is None:
        obj_pos = eef_pos + np.array([0.05, 0.0, -0.05], dtype=np.float32)
    if tgt_pos is None:
        tgt_pos = eef_pos + np.array([0.0, 0.15, 0.0], dtype=np.float32)

    if suite_name in ("object", "spatial"):
        # Standard pick-and-place
        return _pick_place_fallback(obj_pos, tgt_pos)

    if suite_name == "goal":
        # Detect task type from name keywords
        tn = task_name.lower()
        if "open" in tn:
            # Open drawer/door: approach handle, grip, pull
            return [
                {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]),
                 "z": float(obj_pos[2]), "steps": 80},
                {"type": "grip", "steps": 25},
                {"type": "move", "x": float(obj_pos[0]) - 0.12, "y": float(obj_pos[1]),
                 "z": float(obj_pos[2]), "steps": 80},
                {"type": "release", "steps": 15},
            ]
        elif "push" in tn:
            # Push: approach from behind, push forward
            return [
                {"type": "move", "x": float(obj_pos[0]) - 0.05, "y": float(obj_pos[1]),
                 "z": float(obj_pos[2]) + 0.02, "steps": 60},
                {"type": "move", "x": float(obj_pos[0]) + 0.10, "y": float(obj_pos[1]),
                 "z": float(obj_pos[2]) + 0.02, "steps": 100},
            ]
        elif "turn" in tn:
            # Turn knob: approach, grip, rotate
            return [
                {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]),
                 "z": float(obj_pos[2]), "steps": 80},
                {"type": "grip", "steps": 25},
                {"type": "move", "x": float(obj_pos[0]), "y": float(obj_pos[1]) + 0.03,
                 "z": float(obj_pos[2]) - 0.03, "steps": 60},
                {"type": "release", "steps": 15},
            ]
        else:
            # Generic pick-place for "put X on/in Y"
            return _pick_place_fallback(obj_pos, tgt_pos)

    if suite_name == "long":
        # Long tasks: attempt two pick-place cycles
        # Use object as first target, target as second
        plan = _pick_place_fallback(obj_pos, tgt_pos)
        # For "put both" tasks, add a second cycle with offset positions
        tn = task_name.lower()
        if "both" in tn or "and" in tn:
            second_obj = obj_pos + np.array([0.0, 0.10, 0.0], dtype=np.float32)
            plan += _pick_place_fallback(second_obj, tgt_pos)
        return plan

    # Unknown suite — generic pick-place
    return _pick_place_fallback(obj_pos, tgt_pos)


MONITOR_PROMPT = """Look at this robot image. The robot is trying to: {task_desc}
Current primitive: {prim_desc}
The robot appears to be STUCK (no progress for many steps).
Distance to target: {distance:.3f}m

Analyze the failure briefly and suggest ONE recovery strategy:
- "replan": The approach angle or plan is wrong, needs a new plan
- "retry": Minor issue, just retry the current plan
- "skip": Skip to next primitive

Output ONLY JSON: {{"strategy": "replan"|"retry"|"skip", "reason": "brief explanation"}}"""


def vlm_analyze_failure(
    vlm, image: np.ndarray, instruction: str, prim_desc: str, distance: float,
) -> Tuple[str, str]:
    """Call VLM to analyze a failure and suggest recovery."""
    from langchain_core.messages import HumanMessage
    import re

    prompt = MONITOR_PROMPT.format(
        task_desc=instruction, prim_desc=prim_desc, distance=distance,
    )
    img_b64 = _encode_image_b64(image)
    messages = [HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ])]

    try:
        response = vlm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            strategy = data.get("strategy", "replan")
            reason = data.get("reason", "unknown")
            return strategy, reason
    except Exception as e:
        logger.warning(f"VLM failure analysis failed: {e}")
    return "replan", "VLM analysis failed, defaulting to replan"


def check_stuck_diverge(
    dist_history: List[float],
    stuck_window: int = 60,
    stuck_threshold: float = 0.002,
    diverge_window: int = 30,
    diverge_margin: float = 0.03,
) -> Optional[str]:
    """Check for STUCK or DIVERGE conditions based on distance history.

    Returns 'stuck', 'diverge', or None.
    """
    if len(dist_history) < stuck_window:
        return None

    recent = dist_history[-stuck_window:]
    # STUCK: distance hasn't changed much
    if max(recent) - min(recent) < stuck_threshold:
        return "stuck"

    # DIVERGE: recent distances increasing
    if len(dist_history) >= stuck_window + diverge_window:
        old_avg = np.mean(dist_history[-(stuck_window + diverge_window):-stuck_window])
        new_avg = np.mean(dist_history[-diverge_window:])
        if new_avg > old_avg + diverge_margin:
            return "diverge"

    return None


def run_episode_pipeline(
    model,
    eagle_processor,
    env,
    obs: Dict,
    task_name: str,
    instruction: str,
    action_stats: Dict,
    state_stats: Dict,
    vlm_planner,
    vlm_monitor=None,
    max_steps: int = 400,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    max_attempts: int = 3,
    stuck_window: int = 60,
    pos_threshold: float = 0.02,
    record_video: bool = False,
    video_size: int = 256,
    episode_idx: int = 0,
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
) -> Dict[str, Any]:
    """Run one episode with VLM planner + monitor pipeline.

    Supports all LIBERO suites (object, spatial, goal, long).

    Flow:
      for attempt in 1..max_attempts:
        1. VLM generates plan (or fallback)
        2. Execute each primitive with VLA
        3. Monitor: STUCK/DIVERGE detection
        4. On failure: VLM analyzes → replan or retry
        5. On done: return success
    """
    device = next(model.parameters()).device
    a_min = np.array(action_stats["min"][:7], dtype=np.float32)
    a_max = np.array(action_stats["max"][:7], dtype=np.float32)

    frames = []
    total_steps = 0
    replan_count = 0
    plan_history = []
    failure_context = None

    for attempt in range(max_attempts):
        if total_steps >= max_steps:
            break

        # Get current scene positions (suite-aware)
        scene_positions = get_scene_positions(obs, suite_name, task_name, env=env)
        eef_pos = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)

        # VLM planning with generic planner
        image = get_libero_image(obs, size=image_size)
        plan = vlm_plan_generic(
            vlm_planner, image, instruction, eef_pos,
            scene_positions, suite_name, failure_context,
        )
        if plan is None or len(plan) == 0:
            logger.warning("VLM plan failed or empty, using fallback")
            plan = make_fallback_plan_generic(suite_name, task_name, eef_pos, scene_positions)

        plan_history.append({"attempt": attempt + 1, "primitives": plan})
        logger.info(f"  Attempt {attempt + 1}/{max_attempts}: {len(plan)} primitives, steps_used={total_steps}")

        # Reset model state at start of new plan
        if hasattr(model, "reset"):
            model.reset()

        plan_failed = False
        done = False

        for prim_idx, prim in enumerate(plan):
            if total_steps >= max_steps or done:
                break

            prim_type = prim.get("type", "move")
            prim_steps = prim.get("steps", 80)

            if prim_type in ("grip", "release"):
                # VLA-driven grip/release with gripper override
                gripper_val = 1.0 if prim_type == "grip" else -1.0
                chunk_buf = None
                chunk_idx = 0
                prev_act = None

                if hasattr(model, "reset"):
                    model.reset()

                for step_i in range(prim_steps):
                    if total_steps >= max_steps:
                        break
                    if chunk_buf is None or chunk_idx >= chunk_stride:
                        state_td = build_state_11d(obs, None, "grip") if suite_name == "long" else build_state_12d(obs, None, "grip")
                        img = get_libero_image(obs, size=image_size)
                        chunk_buf = _build_batch_and_predict(
                            model, img, state_td, "grip", state_stats,
                            eagle_processor, tokenizer, device, backend,
                            action_stats=action_stats,
                        )
                        chunk_idx = 0
                        if hasattr(model, "reset"):
                            model.reset()

                    act_norm = chunk_buf[chunk_idx].copy()
                    chunk_idx += 1
                    act_norm = np.clip(act_norm, -1.0, 1.0)
                    act_7d = denormalize_action_min_max(act_norm, a_min, a_max)
                    act_7d[6] = gripper_val

                    if prev_act is not None and ema_alpha > 0:
                        act_7d[:6] = ema_alpha * prev_act[:6] + (1 - ema_alpha) * act_7d[:6]
                    prev_act = act_7d.copy()

                    obs, reward, done, info = env.step(act_7d.tolist())
                    total_steps += 1
                    if record_video:
                        frames.append(get_libero_image(obs, size=video_size))
                    if done:
                        break

                logger.info(f"    [{prim_idx+1}] {prim_type} done ({prim_steps} steps)")
                if hasattr(model, "reset"):
                    model.reset()
                if done:
                    break
                continue

            # --- MOVE primitive with monitoring ---
            vlm_target = np.array([prim.get("x", 0), prim.get("y", 0), prim.get("z", 0)], dtype=np.float32)

            # Use obs/sim GT position ONLY for the first pre-grip move (approach object).
            # All other moves use VLM coordinates — VLM determines placement location,
            # pull/push direction, and intermediate waypoints.
            has_prior_grip = any(
                plan[j].get("type") == "grip" for j in range(prim_idx)
            )
            # Count how many moves came before this one (pre-grip only)
            pre_grip_move_idx = sum(
                1 for j in range(prim_idx)
                if plan[j].get("type") == "move"
                and not any(plan[k].get("type") == "grip" for k in range(j))
            )
            obs_positions = get_scene_positions(obs, suite_name, task_name, env=env)
            obj_pos = obs_positions.get("object")

            if not has_prior_grip and pre_grip_move_idx == 0 and obj_pos is not None:
                # First pre-grip move only: use GT object position
                target_pos = np.array(obj_pos, dtype=np.float32)
            else:
                target_pos = vlm_target
            dist_history = []  # Reset per primitive (target changes between primitives)
            chunk_buf = None
            chunk_idx = 0
            prev_act = None

            if hasattr(model, "reset"):
                model.reset()

            logger.info(
                f"    [{prim_idx+1}] move -> [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] "
                f"(budget={prim_steps})"
            )

            for step_i in range(prim_steps):
                if total_steps >= max_steps:
                    break

                # Direction uses VLM target
                state_td = build_state_11d(obs, target_pos, "move") if suite_name == "long" else build_state_12d(obs, target_pos, "move")
                if chunk_buf is None or chunk_idx >= chunk_stride:
                    img = get_libero_image(obs, size=image_size)
                    chunk_buf = _build_batch_and_predict(
                        model, img, state_td, "move", state_stats,
                        eagle_processor, tokenizer, device, backend,
                        action_stats=action_stats,
                    )
                    chunk_idx = 0
                    if hasattr(model, "reset"):
                        model.reset()

                act_norm = chunk_buf[chunk_idx].copy()
                chunk_idx += 1
                act_norm = np.clip(act_norm, -1.0, 1.0)
                act_7d = denormalize_action_min_max(act_norm, a_min, a_max)

                if prev_act is not None and ema_alpha > 0:
                    act_7d[:6] = ema_alpha * prev_act[:6] + (1 - ema_alpha) * act_7d[:6]
                prev_act = act_7d.copy()

                obs, reward, done, info = env.step(act_7d.tolist())
                total_steps += 1
                if record_video:
                    frames.append(get_libero_image(obs, size=video_size))

                # Track distance for monitoring
                eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
                dist = float(np.linalg.norm(target_pos - eef))
                dist_history.append(dist)

                # Log periodically
                if step_i < 3 or step_i % 40 == 0:
                    logger.info(
                        f"    [{prim_idx+1}] step={step_i} "
                        f"action={[round(float(v), 3) for v in act_7d[:3]]} "
                        f"dist={dist:.4f} eef={[round(float(v), 3) for v in eef]}"
                    )

                if done:
                    break

                # Convergence check
                if dist < pos_threshold:
                    logger.info(f"    [{prim_idx+1}] converged at step {step_i} (dist={dist:.4f})")
                    break

                # STUCK/DIVERGE check (diverge_window auto-fits within budget)
                dw = max(10, prim_steps - stuck_window - 5)
                failure_type = check_stuck_diverge(
                    dist_history, stuck_window=stuck_window,
                    diverge_window=dw,
                )
                if failure_type:
                    logger.warning(f"    [{prim_idx+1}] {failure_type.upper()} detected at step {step_i}")
                    # VLM failure analysis
                    if vlm_monitor is not None:
                        img_now = get_libero_image(obs, size=image_size)
                        prim_desc = f"move to [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
                        strategy, reason = vlm_analyze_failure(
                            vlm_monitor, img_now, instruction, prim_desc, dist,
                        )
                        logger.info(f"    VLM analysis: strategy={strategy}, reason={reason}")
                        if strategy == "skip":
                            break  # skip to next primitive
                        elif strategy == "replan":
                            failure_context = f"{failure_type} during {prim_desc}. {reason}"
                            plan_failed = True
                            break
                        # else "retry" → continue to next primitive
                    else:
                        failure_context = f"{failure_type} at dist={dist:.3f} during move to target"
                        plan_failed = True
                    break

            if done or plan_failed:
                break

        if done:
            break

        if plan_failed:
            replan_count += 1
            logger.info(f"  Plan failed, triggering replan (count={replan_count})")
            if hasattr(model, "reset"):
                model.reset()
            continue

    success = bool(done)
    return {
        "success": success,
        "steps": total_steps,
        "reward": 1.0 if success else 0.0,
        "replan_count": replan_count,
        "attempts": min(attempt + 1, max_attempts) if 'attempt' in dir() else 1,
        "plan_history": plan_history,
        "frames": frames if record_video else [],
    }


def evaluate_task_pipeline(
    task_name: str,
    model,
    eagle_processor,
    action_stats: Dict,
    state_stats: Dict,
    vlm_planner,
    vlm_monitor=None,
    num_episodes: int = 50,
    max_horizon: int = 400,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    seed: int = 42,
    max_attempts: int = 3,
    stuck_window: int = 60,
    save_video_dir: Optional[str] = None,
    video_size: int = 256,
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
) -> Dict[str, Any]:
    """Evaluate a single LIBERO task with VLM pipeline mode."""
    fix_libero_config()

    registry = SUITE_REGISTRY[suite_name]
    entry = registry[task_name]
    task_id = entry[0]
    instruction = entry[-1]
    if suite_name == "object":
        obj_prefix = entry[1]
    else:
        obj_prefix = None
        logger.info(f"Pipeline mode with {suite_name} suite: using generic VLM planner + scene position extraction.")

    logger.info(f"[Pipeline] Evaluating: {task_name} ({instruction})")

    env, suite, _, _ = get_libero_env(task_name, resolution=256, suite_name=suite_name)
    initial_states = suite.get_task_init_states(task_id)

    record_video = save_video_dir is not None
    if record_video:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video_dir = os.path.join(save_video_dir, f"{run_ts}_{task_name}_pipeline")
        os.makedirs(save_video_dir, exist_ok=True)

    episode_results = []
    successes = 0

    for ep_idx in range(num_episodes):
        env.reset()
        obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])
        for _ in range(NUM_WAIT_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        print(f"  ep{ep_idx + 1:>3d}/{num_episodes}: ", end="", flush=True)

        try:
            result = run_episode_pipeline(
                model=model,
                eagle_processor=eagle_processor,
                env=env,
                obs=obs,
                task_name=task_name,
                instruction=instruction,
                action_stats=action_stats,
                state_stats=state_stats,
                vlm_planner=vlm_planner,
                vlm_monitor=vlm_monitor,
                max_steps=max_horizon,
                image_size=image_size,
                chunk_stride=chunk_stride,
                ema_alpha=ema_alpha,
                max_attempts=max_attempts,
                stuck_window=stuck_window,
                record_video=record_video,
                video_size=video_size,
                episode_idx=ep_idx,
                backend=backend,
                tokenizer=tokenizer,
                suite_name=suite_name,
            )

            tag = "\033[92mSUCCESS\033[0m" if result["success"] else "\033[91mFAIL\033[0m"
            replans = result.get("replan_count", 0)
            print(f"{tag}  steps={result['steps']:<4d}  replans={replans}")

            if record_video and result.get("frames"):
                tag_str = "success" if result["success"] else "fail"
                vid_path = os.path.join(
                    save_video_dir, f"libero_pipe_{task_name}_ep{ep_idx + 1}_{tag_str}.mp4"
                )
                save_video(result.pop("frames"), vid_path)

            result["episode"] = ep_idx + 1
            if result["success"]:
                successes += 1
            episode_results.append(result)

        except Exception as e:
            logger.error(f"Episode {ep_idx + 1} error: {e}")
            import traceback
            traceback.print_exc()
            episode_results.append({
                "success": False, "episode": ep_idx + 1, "steps": 0, "error": str(e),
            })
            print(f"\033[91mERROR\033[0m: {e}")

        done_count = len(episode_results)
        print(f"    progress: {successes}/{done_count} ({successes / done_count:.1%})")

    env.close()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    total_replans = sum(r.get("replan_count", 0) for r in episode_results)
    print(f"\n  \033[1m{task_name} [pipeline]: {successes}/{num_episodes} = {success_rate:.1%} "
          f"(total replans: {total_replans})\033[0m\n")

    return {
        "task": task_name,
        "instruction": instruction,
        "mode": "pipeline",
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "total_replans": total_replans,
        "episodes": episode_results,
    }


# ===========================================================================
# Video saving
# ===========================================================================

def save_video(frames: List[np.ndarray], path: str, fps: int = 20):
    """Save frames as MP4."""
    try:
        import imageio
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        logger.info(f"Video saved: {path}")
    except ImportError:
        logger.warning("imageio not installed, skipping video save")


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def evaluate_task(
    task_name: str,
    model,
    eagle_processor,
    action_stats: Dict,
    state_stats: Dict,
    num_episodes: int = 50,
    max_horizon: int = 280,
    image_size: int = 224,
    chunk_stride: int = 4,
    ema_alpha: float = 0.6,
    seed: int = 42,
    save_video_dir: Optional[str] = None,
    video_size: int = 256,
    mode: str = "template",
    backend: str = "groot_n1.5",
    tokenizer=None,
    suite_name: str = "object",
) -> Dict[str, Any]:
    """Evaluate on a single LIBERO task."""

    fix_libero_config()

    registry = SUITE_REGISTRY[suite_name]
    entry = registry[task_name]
    task_id = entry[0]
    instruction = entry[-1]

    logger.info(f"Evaluating: {task_name} ({instruction})")

    env, suite, _, _ = get_libero_env(task_name, resolution=256, suite_name=suite_name)
    initial_states = suite.get_task_init_states(task_id)

    record_video = save_video_dir is not None
    # Create timestamped sub-directory per run to avoid overwrites
    if record_video:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video_dir = os.path.join(save_video_dir, f"{run_ts}_{task_name}")
        os.makedirs(save_video_dir, exist_ok=True)

    episode_results = []
    successes = 0

    for ep_idx in range(num_episodes):
        # Reset env with benchmark initial state
        env.reset()
        obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])

        # Wait steps for stabilization
        for _ in range(NUM_WAIT_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        print(f"  ep{ep_idx + 1:>3d}/{num_episodes}: ", end="", flush=True)

        try:
            if mode == "monitored-raw":
                result = run_episode_monitored_raw(
                    model=model,
                    eagle_processor=eagle_processor,
                    env=env,
                    obs=obs,
                    task_name=task_name,
                    instruction=instruction,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    max_steps=max_horizon,
                    image_size=image_size,
                    chunk_stride=chunk_stride,
                    ema_alpha=ema_alpha,
                    record_video=record_video,
                    video_size=video_size,
                    episode_idx=ep_idx,
                    backend=backend,
                    tokenizer=tokenizer,
                    suite_name=suite_name,
                    stuck_window=40,
                    max_attempts=3,
                    init_state=initial_states[ep_idx % len(initial_states)],
                )
            else:
                runner = run_episode_raw if mode == "raw" else run_episode
                result = runner(
                    model=model,
                    eagle_processor=eagle_processor,
                    env=env,
                    obs=obs,
                    task_name=task_name,
                    instruction=instruction,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    max_steps=max_horizon,
                    image_size=image_size,
                    chunk_stride=chunk_stride,
                    ema_alpha=ema_alpha,
                    record_video=record_video,
                    video_size=video_size,
                    episode_idx=ep_idx,
                    backend=backend,
                    tokenizer=tokenizer,
                    suite_name=suite_name,
                )

            tag = "\033[92mSUCCESS\033[0m" if result["success"] else "\033[91mFAIL\033[0m"
            attempts_str = f"  attempts={result['attempts']}" if "attempts" in result else ""
            print(f"{tag}  steps={result['steps']:<4d}  reward={result['reward']:.2f}{attempts_str}")

            # Save video
            if record_video and result.get("frames"):
                tag_str = "success" if result["success"] else "fail"
                vid_path = os.path.join(
                    save_video_dir, f"libero_{task_name}_ep{ep_idx + 1}_{tag_str}.mp4"
                )
                save_video(result.pop("frames"), vid_path)

            result["episode"] = ep_idx + 1
            if result["success"]:
                successes += 1
            episode_results.append(result)

        except Exception as e:
            logger.error(f"Episode {ep_idx + 1} error: {e}")
            import traceback
            traceback.print_exc()
            episode_results.append({
                "success": False, "episode": ep_idx + 1, "steps": 0, "error": str(e),
            })
            print(f"\033[91mERROR\033[0m: {e}")

        # Running stats
        done_count = len(episode_results)
        print(f"    progress: {successes}/{done_count} ({successes / done_count:.1%})")

    env.close()

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    print(f"\n  \033[1m{task_name}: {successes}/{num_episodes} = {success_rate:.1%}\033[0m\n")

    return {
        "task": task_name,
        "instruction": instruction,
        "success_rate": success_rate,
        "successes": successes,
        "total_episodes": num_episodes,
        "episodes": episode_results,
    }


def format_results_table(results: Dict[str, Dict], tasks: List[str], suite_name: str = "object") -> str:
    """Format results as a text table."""
    col_w = max(max(len(t) for t in tasks), 18)
    lines = []

    suite_label = SUITE_BENCH_NAME.get(suite_name, suite_name).upper()
    lines.append("=" * 75)
    lines.append(f"{suite_label} EVALUATION RESULTS")
    lines.append("=" * 75)
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
    lines.append("=" * 75)

    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation for VLA models (GROOT N1.5, PI0.5, SmolVLA) — supports Object, Spatial, Goal, Long suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Suite / task selection
    parser.add_argument("--suite", type=str, default="object",
                        choices=["object", "spatial", "goal", "long"],
                        help="LIBERO suite to evaluate (default: object)")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to evaluate (use --list-tasks to see available)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Evaluate all tasks in the selected suite")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="List of task names to evaluate")

    # Model
    parser.add_argument("--vla-backend", type=str, default="groot_n1.5",
                        choices=["groot_n1.5", "pi05", "smolvla"],
                        help="VLA model backend (default: groot_n1.5)")
    parser.add_argument("--base-model", type=str, default="nvidia/GR00T-N1.5-3B",
                        help="GROOT base model ID")
    parser.add_argument("--move-adapter", type=str, default=None,
                        help="Path to move adapter checkpoint")
    parser.add_argument("--action-stats", type=str, default=None,
                        help="Path to data_stats.json from training")

    # Multi-task patterns (use {task} placeholder)
    parser.add_argument("--move-adapter-pattern", type=str, default=None,
                        help="Move adapter path pattern with {task} placeholder")
    parser.add_argument("--stats-pattern", type=str, default=None,
                        help="Stats file path pattern with {task} placeholder")

    # Eval settings
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Episodes per task (default: 50)")
    parser.add_argument("--max-horizon", type=int, default=280,
                        help="Max steps per episode (default: 280)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image size for VLA input (default: 224)")
    parser.add_argument("--denoise-steps", type=int, default=10,
                        help="GROOT denoising steps (default: 10)")
    parser.add_argument("--chunk-stride", type=int, default=4,
                        help="Action chunk stride (default: 4)")
    parser.add_argument("--ema-alpha", type=float, default=0.6,
                        help="EMA smoothing alpha (default: 0.6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Output
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--save-video", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--video-size", type=int, default=256,
                        help="Video frame resolution (default: 256)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    # Pipeline mode
    parser.add_argument("--mode", type=str, default="template",
                        choices=["template", "pipeline", "raw", "monitored-raw"],
                        help="Evaluation mode: template (fixed FSM), pipeline (VLM planner+monitor), raw (continuous VLA), monitored-raw (raw+stuck detect+retry)")
    parser.add_argument("--planner-provider", type=str, default="bedrock",
                        help="VLM provider for planner (bedrock/anthropic/openai/google)")
    parser.add_argument("--planner-model", type=str, default="eu.anthropic.claude-opus-4-6-v1",
                        help="VLM model for planner")
    parser.add_argument("--monitor-provider", type=str, default=None,
                        help="VLM provider for monitor (default: same as planner)")
    parser.add_argument("--monitor-model", type=str, default=None,
                        help="VLM model for monitor (default: same as planner)")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Max replan attempts per episode (pipeline mode)")
    parser.add_argument("--stuck-window", type=int, default=60,
                        help="Steps to detect STUCK condition (pipeline mode)")

    # List tasks
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all LIBERO-Object tasks and exit")

    args = parser.parse_args()

    suite_name = args.suite
    registry = SUITE_REGISTRY[suite_name]

    if args.list_tasks:
        bench_name = SUITE_BENCH_NAME[suite_name]
        print(f"\nLIBERO {suite_name} tasks (benchmark: {bench_name}):")
        for name, entry in registry.items():
            tid = entry[0]
            instr = entry[-1]
            print(f"  [{tid}] {name}: {instr}")
        return

    # Determine tasks to evaluate
    if args.all_tasks:
        tasks = list(registry.keys())
    elif args.tasks:
        tasks = args.tasks
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("Specify --task, --tasks, or --all-tasks")

    # Validate tasks
    for t in tasks:
        if t not in registry:
            parser.error(f"Unknown task for suite '{suite_name}': {t}. Use --list-tasks --suite {suite_name} to see available tasks.")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fix_libero_config()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # For single-task mode, load model once
    # For multi-task mode, reload adapter per task if using patterns
    all_results = {}
    model = None
    eagle_processor = None
    tokenizer = None
    loaded_adapter_path = None
    backend = args.vla_backend

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'=' * 70}")
        print(f"TASK [{task_idx + 1}/{len(tasks)}]: {task_name}")
        print(f"{'=' * 70}")

        # Resolve per-task paths
        if args.move_adapter_pattern:
            move_adapter = args.move_adapter_pattern.replace("{task}", task_name)
        elif args.move_adapter:
            move_adapter = args.move_adapter
        else:
            parser.error("Specify --move-adapter or --move-adapter-pattern")

        if args.stats_pattern:
            stats_path = args.stats_pattern.replace("{task}", task_name)
        elif args.action_stats:
            stats_path = args.action_stats
        else:
            # Auto-detect from adapter path
            stats_path = os.path.join(move_adapter, "data_stats.json")

        # Validate paths
        if not os.path.exists(move_adapter):
            logger.error(f"Move adapter not found: {move_adapter}")
            all_results[task_name] = {"success_rate": -1, "error": "adapter_not_found"}
            continue

        if not os.path.exists(stats_path):
            logger.error(f"Stats file not found: {stats_path}")
            all_results[task_name] = {"success_rate": -1, "error": "stats_not_found"}
            continue

        # Load stats
        with open(stats_path) as f:
            data_stats = json.load(f)
        action_stats = data_stats["action_stats"]
        state_stats = data_stats["state_stats"]
        logger.info(f"Loaded stats from {stats_path}")
        logger.info(f"  action dim={len(action_stats['min'])}, state dim={len(state_stats['min'])}")
        logger.info(f"  action min={[round(v, 3) for v in action_stats['min'][:7]]}")
        logger.info(f"  action max={[round(v, 3) for v in action_stats['max'][:7]]}")

        # Load or switch model adapter
        if model is None or loaded_adapter_path != move_adapter:
            if model is None:
                # First load — dispatch by backend
                if backend == "groot_n1.5":
                    model, eagle_processor, _ = load_groot_model(
                        move_adapter=move_adapter,
                        device=device,
                        denoise_steps=args.denoise_steps,
                        image_size=args.image_size,
                    )
                elif backend == "pi05":
                    model, tokenizer, _ = load_pi05_model(
                        move_adapter=move_adapter,
                        device=device,
                        denoise_steps=args.denoise_steps,
                    )
                elif backend == "smolvla":
                    model, tokenizer, _ = load_smolvla_model(
                        move_adapter=move_adapter,
                        device=device,
                        denoise_steps=args.denoise_steps,
                    )
                else:
                    raise ValueError(f"Unknown backend: {backend}")
            else:
                # Switch adapter (load new weights)
                from safetensors.torch import load_file
                for filename in ["model_diff.safetensors", "model.safetensors"]:
                    weights_path = os.path.join(move_adapter, filename)
                    if os.path.exists(weights_path):
                        weights = load_file(weights_path)
                        model.load_state_dict(weights, strict=False)
                        logger.info(f"Switched adapter: {move_adapter} ({len(weights)} keys)")
                        break
            loaded_adapter_path = move_adapter

        # Run evaluation
        video_dir = os.path.join(args.output_dir, "videos", task_name) if args.save_video else None

        if args.mode == "pipeline":
            # Initialize VLM planner/monitor (once)
            if not hasattr(args, '_vlm_planner') or args._vlm_planner is None:
                logger.info(f"Initializing VLM planner: {args.planner_provider}/{args.planner_model}")
                args._vlm_planner = init_vlm(args.planner_provider, args.planner_model)
                mon_provider = args.monitor_provider or args.planner_provider
                mon_model = args.monitor_model or args.planner_model
                if mon_provider and mon_model:
                    logger.info(f"Initializing VLM monitor: {mon_provider}/{mon_model}")
                    args._vlm_monitor = init_vlm(mon_provider, mon_model, temperature=0.1, max_tokens=256)
                else:
                    args._vlm_monitor = None

            result = evaluate_task_pipeline(
                task_name=task_name,
                model=model,
                eagle_processor=eagle_processor,
                action_stats=action_stats,
                state_stats=state_stats,
                vlm_planner=args._vlm_planner,
                vlm_monitor=args._vlm_monitor,
                num_episodes=args.num_episodes,
                max_horizon=args.max_horizon,
                image_size=args.image_size,
                chunk_stride=args.chunk_stride,
                ema_alpha=args.ema_alpha,
                seed=args.seed,
                max_attempts=args.max_attempts,
                stuck_window=args.stuck_window,
                save_video_dir=video_dir,
                video_size=args.video_size,
                backend=backend,
                tokenizer=tokenizer,
                suite_name=suite_name,
            )
        else:
            result = evaluate_task(
                task_name=task_name,
                model=model,
                eagle_processor=eagle_processor,
                action_stats=action_stats,
                state_stats=state_stats,
                num_episodes=args.num_episodes,
                max_horizon=args.max_horizon,
                image_size=args.image_size,
                chunk_stride=args.chunk_stride,
                ema_alpha=args.ema_alpha,
                seed=args.seed,
                save_video_dir=video_dir,
                video_size=args.video_size,
                mode=args.mode,
                backend=backend,
                tokenizer=tokenizer,
                suite_name=suite_name,
            )
        all_results[task_name] = result

        # Save per-task result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(args.output_dir, f"eval_libero_{task_name}_{timestamp}.json")
        with open(result_path, "w") as f:
            # Remove non-serializable items
            save_result = {k: v for k, v in result.items() if k != "frames"}
            json.dump(save_result, f, indent=2, default=str)
        logger.info(f"Results saved: {result_path}")

    # Print summary table
    table = format_results_table(all_results, tasks, suite_name=suite_name)
    print(f"\n{table}\n")

    # Save summary
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "suite": suite_name,
        "mode": args.mode,
        "backend": backend,
        "tasks": tasks,
        "num_episodes": args.num_episodes,
        "max_horizon": args.max_horizon,
        "denoise_steps": args.denoise_steps,
        "chunk_stride": args.chunk_stride,
        "ema_alpha": args.ema_alpha,
        "seed": args.seed,
        **({"planner": f"{args.planner_provider}/{args.planner_model}",
            "max_attempts": args.max_attempts} if args.mode == "pipeline" else {}),
        "results": {
            t: {
                "success_rate": r.get("success_rate", -1),
                "successes": r.get("successes", 0),
                "total_episodes": r.get("total_episodes", 0),
            }
            for t, r in all_results.items()
        },
    }
    summary_path = os.path.join(args.output_dir, f"eval_libero_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # Also save results table as text
    table_path = os.path.join(args.output_dir, f"eval_libero_summary_{timestamp}.txt")
    with open(table_path, "w") as f:
        f.write(table + "\n")
        f.write(f"\nSettings: denoise={args.denoise_steps} chunk_stride={args.chunk_stride} "
                f"ema={args.ema_alpha} episodes={args.num_episodes} horizon={args.max_horizon}\n")
    logger.info(f"Table saved: {table_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
