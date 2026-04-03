#!/usr/bin/env python3
"""
Direction-normalized preprocessor for VLA LoRA training.

State format:
  eef_base(3) + eef_quat(4) + gripper(2) + direction(3) = 12D

Where:
  eef_base    = robot0_base_to_eef_pos (GT from HDF5, no manual conversion)
  eef_quat    = robot0_base_to_eef_quat (GT from HDF5)
  gripper     = gripper_qpos (2D)
  direction   = unit_vector(target_base - eef_base) for move segments
              = [0, 0, 0] for grip segments

Key difference from delta_base:
  - Uses GT base-frame values directly (avoids 4.5mm base_pos drift error)
  - Delta is normalized to unit vector (removes magnitude → eliminates OOD)

Usage:
    python preprocess_direction.py --output data_direction
    python preprocess_direction.py --output data_direction --task CloseDrawer
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "robocasa_data", "v0.1", "single_stage")

CAMERA_KEYS = [
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
]

ACTION_KEYS = ["rel_pos", "rel_rot_axis_angle", "gripper"]


def find_all_datasets(data_root: str, task_filter: Optional[str] = None) -> List[Tuple[str, str]]:
    root = Path(data_root)
    results = []
    for hdf5_path in sorted(root.glob("**/*.hdf5")):
        parts = hdf5_path.parts
        task_name = None
        for i, part in enumerate(parts):
            if part in ("single_stage", "multi_stage") and i + 2 < len(parts):
                task_name = parts[i + 2]
                break
        if task_name is None:
            task_name = hdf5_path.stem
        if task_filter and task_name != task_filter:
            continue
        results.append((task_name, str(hdf5_path)))
    return results


def _remove_short_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Remove short True runs from a boolean mask."""
    if min_len <= 1:
        return mask
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i + 1
            while j < n and out[j]:
                j += 1
            if (j - i) < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def label_phases(
    gripper_actions: np.ndarray,
    rel_pos: Optional[np.ndarray] = None,
    rel_rot_axis_angle: Optional[np.ndarray] = None,
    window: int = 5,
    mode: str = "gripper_only",
    manip_pos_percentile: float = 35.0,
    manip_rot_percentile: float = 65.0,
    manip_expand: int = 3,
    manip_min_len: int = 4,
) -> np.ndarray:
    n = len(gripper_actions)
    phases = np.zeros(n, dtype=np.int32)
    gripper_flat = gripper_actions.flatten()
    transitions = np.where(np.diff(gripper_flat) != 0)[0]
    for t in transitions:
        start = max(0, t - window)
        end = min(n, t + window + 1)
        phases[start:end] = 1

    # Optional fallback: infer manipulation-like (pseudo-grip) segments from motion pattern.
    # Useful for tasks where gripper action is constant in demonstrations.
    if mode in ("gripper_or_manip", "manip_only"):
        if mode == "manip_only":
            phases[:] = 0

        if rel_pos is None or rel_rot_axis_angle is None:
            logger.warning("manip phase mode requested but rel_pos/rel_rot_axis_angle missing; using gripper-only")
            return phases

        if len(rel_pos) != n or len(rel_rot_axis_angle) != n:
            logger.warning("manip phase mode requested but rel arrays have mismatched length; using gripper-only")
            return phases

        pos_norm = np.linalg.norm(rel_pos, axis=1)
        rot_norm = np.linalg.norm(rel_rot_axis_angle, axis=1)

        pos_th = np.percentile(pos_norm, manip_pos_percentile)
        rot_th = np.percentile(rot_norm, manip_rot_percentile)
        manip_mask = (pos_norm <= pos_th) & (rot_norm >= rot_th) & (rot_norm > 1e-4)

        if manip_expand > 0 and manip_mask.any():
            kernel = np.ones(2 * manip_expand + 1, dtype=np.int32)
            manip_mask = np.convolve(manip_mask.astype(np.int32), kernel, mode="same") > 0

        manip_mask = _remove_short_true_runs(manip_mask, manip_min_len)
        if manip_mask.any():
            phases[manip_mask] = 1

    return phases


def find_segments(phases: np.ndarray) -> List[Tuple[int, int, int]]:
    segments = []
    if len(phases) == 0:
        return segments
    seg_start = 0
    seg_phase = phases[0]
    for i in range(1, len(phases)):
        if phases[i] != seg_phase:
            segments.append((seg_start, i - 1, int(seg_phase)))
            seg_start = i
            seg_phase = phases[i]
    segments.append((seg_start, len(phases) - 1, int(seg_phase)))
    return segments


def generate_direction_data(
    phases: np.ndarray,
    eef_base: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """Generate per-timestep instructions and unit-vector direction targets.

    For move segments:
      - target_base = eef_base at end of segment (or start of next)
      - direction = (target_base - eef_base_t) / ||...||  (unit vector)
    For grip segments:
      - direction = [0, 0, 0] (stay in place)
    """
    n = len(phases)
    instructions = [""] * n
    direction = np.zeros((n, 3), dtype=np.float32)
    segments = find_segments(phases)

    for start, end, phase in segments:
        if phase == 0:  # move
            # Target = eef position at end of segment (or start of next)
            target_base = eef_base[end].copy()
            if end + 1 < n:
                target_base = eef_base[end + 1].copy()

            for t in range(start, end + 1):
                instructions[t] = "move"
                delta = target_base - eef_base[t]
                norm = np.linalg.norm(delta)
                if norm > 1e-6:
                    direction[t] = (delta / norm).astype(np.float32)
                # else: stays [0, 0, 0]
        else:  # grip
            for t in range(start, end + 1):
                instructions[t] = "grip"
                # direction stays [0, 0, 0]

    return instructions, direction


def extract_demo(
    f: h5py.File,
    demo_key: str,
    task_name: str,
    demo_id: int,
    grip_window: int,
    phase_mode: str,
    manip_pos_percentile: float,
    manip_rot_percentile: float,
    manip_expand: int,
    manip_min_len: int,
) -> Optional[Dict]:
    demo = f[demo_key]

    if "obs" not in demo or "action_dict" not in demo:
        logger.warning(f"Skipping {demo_key}: missing obs or action_dict")
        return None

    obs = demo["obs"]
    action_dict = demo["action_dict"]
    n_steps = demo["actions"].shape[0]
    if n_steps < 10:
        logger.warning(f"Skipping {demo_key}: too few steps ({n_steps})")
        return None

    # Language instruction
    try:
        ep_meta = json.loads(demo.attrs["ep_meta"])
        instruction = ep_meta.get("lang", "manipulate the object")
    except (KeyError, json.JSONDecodeError):
        instruction = "manipulate the object"

    # Images
    primary_camera = None
    for cam_key in CAMERA_KEYS:
        if cam_key in obs:
            primary_camera = cam_key
            break
    if primary_camera is None:
        logger.warning(f"Skipping {demo_key}: no camera images found")
        return None
    images = obs[primary_camera][:]

    wrist_key = "robot0_eye_in_hand_image"
    wrist_images = obs[wrist_key][:] if wrist_key in obs and wrist_key != primary_camera else None

    # GT base-frame values (no manual R_inv conversion needed)
    eef_base = obs["robot0_base_to_eef_pos"][:].astype(np.float32)  # (N, 3)
    eef_quat = obs["robot0_base_to_eef_quat"][:].astype(np.float32)  # (N, 4)
    gripper_qpos = obs["robot0_gripper_qpos"][:].astype(np.float32)  # (N, 2)

    # Base pose (for metadata, not used in state construction)
    base_pos = obs["robot0_base_pos"][0].astype(np.float64)
    base_quat_raw = obs["robot0_base_quat"][0].astype(np.float64)

    # Actions (already in base frame from OSC_POSE)
    action_parts = []
    for key in ACTION_KEYS:
        if key in action_dict:
            action_parts.append(action_dict[key][:])
    if not action_parts:
        logger.warning(f"Skipping {demo_key}: no action_dict found")
        return None
    actions = np.concatenate(action_parts, axis=1).astype(np.float32)  # (N, 7)

    # Phase labeling
    gripper = action_dict["gripper"][:] if "gripper" in action_dict else actions[:, -1:]
    rel_pos = action_dict["rel_pos"][:] if "rel_pos" in action_dict else None
    rel_rot_axis_angle = (
        action_dict["rel_rot_axis_angle"][:] if "rel_rot_axis_angle" in action_dict else None
    )
    phases = label_phases(
        gripper_actions=gripper,
        rel_pos=rel_pos,
        rel_rot_axis_angle=rel_rot_axis_angle,
        window=grip_window,
        mode=phase_mode,
        manip_pos_percentile=manip_pos_percentile,
        manip_rot_percentile=manip_rot_percentile,
        manip_expand=manip_expand,
        manip_min_len=manip_min_len,
    )

    # Generate unit-vector direction targets (using GT base-frame eef)
    primitive_instructions, direction = generate_direction_data(phases, eef_base)

    # Build 12D state: eef_base(3) + eef_quat(4) + gripper(2) + direction(3)
    robot_state = np.concatenate([eef_base, eef_quat, gripper_qpos, direction], axis=1).astype(np.float32)

    return {
        "images": images,
        "wrist_images": wrist_images,
        "robot_state": robot_state,
        "actions": actions,
        "phases": phases,
        "instruction": instruction,
        "primitive_instructions": primitive_instructions,
        "task_name": task_name,
        "demo_id": demo_id,
        "n_steps": n_steps,
        "base_pos": base_pos.tolist(),
        "base_quat": base_quat_raw.tolist(),
    }


def compute_stats(data: np.ndarray) -> Dict:
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist(),
    }


def process_dataset(
    data_root: str,
    output_dir: str,
    task_filter: Optional[str] = None,
    grip_window: int = 5,
    phase_mode: str = "gripper_only",
    manip_pos_percentile: float = 35.0,
    manip_rot_percentile: float = 65.0,
    manip_expand: int = 3,
    manip_min_len: int = 4,
    val_ratio: float = 0.1,
    max_demos: Optional[int] = None,
) -> None:
    datasets = find_all_datasets(data_root, task_filter)
    if not datasets:
        logger.error(f"No datasets found in {data_root}")
        return

    logger.info(f"Found {len(datasets)} dataset files")

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_actions = []
    all_states = []
    total_samples = 0
    total_demos = 0
    task_stats = {}

    for ds_idx, (task_name, hdf5_path) in enumerate(datasets):
        logger.info(f"[{ds_idx + 1}/{len(datasets)}] Processing: {task_name} ({hdf5_path})")

        try:
            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                if max_demos:
                    demo_keys = demo_keys[:max_demos]

                n_demos = len(demo_keys)
                n_val = max(1, int(n_demos * val_ratio))
                rng = np.random.RandomState(42 + ds_idx)
                indices = rng.permutation(n_demos)
                val_indices = set(indices[:n_val])

                for i, demo_key in enumerate(demo_keys):
                    demo_id = int(demo_key.split("_")[1])
                    demo_data = extract_demo(
                        f,
                        f"data/{demo_key}",
                        task_name,
                        demo_id,
                        grip_window,
                        phase_mode,
                        manip_pos_percentile,
                        manip_rot_percentile,
                        manip_expand,
                        manip_min_len,
                    )
                    if demo_data is None:
                        continue

                    is_val = i in val_indices
                    split_dir = val_dir if is_val else train_dir

                    filename = f"{task_name}_demo{demo_id}.npz"
                    save_path = os.path.join(split_dir, filename)

                    save_data = {
                        "images": demo_data["images"],
                        "robot_state": demo_data["robot_state"],
                        "actions": demo_data["actions"],
                        "phases": demo_data["phases"],
                    }
                    if demo_data["wrist_images"] is not None:
                        save_data["wrist_images"] = demo_data["wrist_images"]

                    meta = {
                        "instruction": demo_data["instruction"],
                        "primitive_instructions": demo_data["primitive_instructions"],
                        "task_name": task_name,
                        "demo_id": demo_id,
                        "n_steps": demo_data["n_steps"],
                        "n_move": int((demo_data["phases"] == 0).sum()),
                        "n_grip": int((demo_data["phases"] == 1).sum()),
                        "split": "val" if is_val else "train",
                        "base_pos": demo_data["base_pos"],
                        "base_quat": demo_data["base_quat"],
                        "state_format": "eef_base(3)+eef_quat(4)+gripper(2)+direction(3)",
                    }

                    np.savez_compressed(save_path, **save_data)
                    meta_path = save_path.replace(".npz", ".json")
                    with open(meta_path, "w") as mf:
                        json.dump(meta, mf, indent=2)

                    all_actions.append(demo_data["actions"])
                    all_states.append(demo_data["robot_state"])
                    total_samples += demo_data["n_steps"]
                    total_demos += 1

                    if task_name not in task_stats:
                        task_stats[task_name] = {
                            "demos": 0, "steps": 0,
                            "move_steps": 0, "grip_steps": 0,
                        }
                    task_stats[task_name]["demos"] += 1
                    task_stats[task_name]["steps"] += demo_data["n_steps"]
                    task_stats[task_name]["move_steps"] += meta["n_move"]
                    task_stats[task_name]["grip_steps"] += meta["n_grip"]

        except Exception as e:
            logger.error(f"Failed to process {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_actions:
        logger.error("No data extracted!")
        return

    all_actions_array = np.concatenate(all_actions, axis=0)
    all_states_array = np.concatenate(all_states, axis=0)

    action_stats = compute_stats(all_actions_array)
    action_stats["mode"] = "min_max"
    state_stats = compute_stats(all_states_array)

    metadata = {
        "action_stats": action_stats,
        "state_stats": state_stats,
        "state_format": "eef_base(3)+eef_quat(4)+gripper(2)+direction(3)",
        "action_dim": all_actions_array.shape[1],
        "state_dim": all_states_array.shape[1],
        "action_keys": ACTION_KEYS,
        "camera_keys": CAMERA_KEYS,
        "total_demos": total_demos,
        "total_samples": total_samples,
        "grip_window": grip_window,
        "phase_mode": phase_mode,
        "manip_pos_percentile": manip_pos_percentile,
        "manip_rot_percentile": manip_rot_percentile,
        "manip_expand": manip_expand,
        "manip_min_len": manip_min_len,
        "val_ratio": val_ratio,
        "task_filter": task_filter,
        "task_stats": task_stats,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also write metadata_extended.json so train_lora_movegrip.py can find state_format
    action_quantile_stats = {
        "q01": np.percentile(all_actions_array, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions_array, 99, axis=0).tolist(),
    }
    metadata_extended = {
        "action_stats": action_stats,
        "state_stats": state_stats,
        "action_quantile_stats": action_quantile_stats,
        "state_format": "eef_base(3)+eef_quat(4)+gripper(2)+direction(3)",
        "action_dim": all_actions_array.shape[1],
        "state_dim": all_states_array.shape[1],
        "action_keys": ACTION_KEYS,
        "camera_keys": CAMERA_KEYS,
        "total_demos": total_demos,
        "total_samples": total_samples,
        "grip_window": grip_window,
        "phase_mode": phase_mode,
        "manip_pos_percentile": manip_pos_percentile,
        "manip_rot_percentile": manip_rot_percentile,
        "manip_expand": manip_expand,
        "manip_min_len": manip_min_len,
        "val_ratio": val_ratio,
        "task_filter": task_filter,
        "task_stats": task_stats,
    }
    ext_metadata_path = os.path.join(output_dir, "metadata_extended.json")
    with open(ext_metadata_path, "w") as f:
        json.dump(metadata_extended, f, indent=2)
    logger.info(f"Saved metadata_extended.json to {ext_metadata_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Direction-normalized preprocessing complete!")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  State format: eef_base(3) + eef_quat(4) + gripper(2) + direction(3)")
    logger.info(f"  Total demos: {total_demos}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info("")

    labels = ["eef_base_x", "eef_base_y", "eef_base_z",
              "quat_x", "quat_y", "quat_z", "quat_w",
              "grip_l", "grip_r",
              "dir_x", "dir_y", "dir_z"]
    logger.info("State statistics:")
    for i, label in enumerate(labels):
        logger.info(f"  {label:12s}: min={state_stats['min'][i]:7.3f}  max={state_stats['max'][i]:7.3f}  "
                     f"std={state_stats['std'][i]:6.3f}")

    # Verify direction is unit vector for move steps
    move_mask = all_states_array[:, 9:12].any(axis=1)  # non-zero direction
    if move_mask.any():
        dir_norms = np.linalg.norm(all_states_array[move_mask, 9:12], axis=1)
        logger.info(f"\n  Direction norms (move steps): mean={dir_norms.mean():.4f}, "
                     f"min={dir_norms.min():.4f}, max={dir_norms.max():.4f}")

    logger.info("")
    logger.info("Per-task statistics:")
    for task, stats in sorted(task_stats.items()):
        move_pct = stats["move_steps"] / max(stats["steps"], 1) * 100
        logger.info(f"  {task}: {stats['demos']} demos, {stats['steps']} steps "
                     f"(move: {move_pct:.0f}%, grip: {100 - move_pct:.0f}%)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Direction-normalized preprocessor for VLA LoRA training")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=str, default="data_direction")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--grip-window", type=int, default=5)
    parser.add_argument(
        "--phase-mode",
        type=str,
        default="gripper_only",
        choices=["gripper_only", "gripper_or_manip", "manip_only"],
        help="How to label grip phases. v4 uses gripper_or_manip for constant-gripper datasets.",
    )
    parser.add_argument("--manip-pos-percentile", type=float, default=35.0)
    parser.add_argument("--manip-rot-percentile", type=float, default=65.0)
    parser.add_argument("--manip-expand", type=int, default=3)
    parser.add_argument("--manip-min-len", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-demos", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        data_root=args.data_root,
        output_dir=args.output,
        task_filter=args.task,
        grip_window=args.grip_window,
        phase_mode=args.phase_mode,
        manip_pos_percentile=args.manip_pos_percentile,
        manip_rot_percentile=args.manip_rot_percentile,
        manip_expand=args.manip_expand,
        manip_min_len=args.manip_min_len,
        val_ratio=args.val_ratio,
        max_demos=args.max_demos,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
