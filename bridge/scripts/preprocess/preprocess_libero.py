#!/usr/bin/env python3
"""
LIBERO-Object preprocessor for GROOT N1.5 LoRA training.

Converts LIBERO-Object HDF5 multicam demo data to per-task NPZ format.

State format:
  eef_base(3) + eef_quat(4) + gripper(2) + direction(3) = 12D

Where:
  eef_base    = obs/ee_states[:, :3] (== obs/ee_pos)
  eef_quat    = obs/ee_states[:, 3:7]
  gripper     = obs/gripper_states (2D)
  direction   = unit_vector(target - eef) for move segments, [0,0,0] for grip

Input:  LIBERO-Object HDF5 files from datasets_multicam/libero_object/
Output: data_libero/{task_name}/train/*.npz + *.json
        data_libero/{task_name}/metadata.json

Usage:
    python preprocess_libero.py --output data_libero
    python preprocess_libero.py --output data_libero --task alphabet_soup
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = os.environ.get("LIBERO_DATA_ROOT", os.path.expanduser("~/SGRPO/data/libero/datasets_multicam/libero_object"))


# ---------------------------------------------------------------------------
# Task name extraction
# ---------------------------------------------------------------------------

def extract_task_name(filename: str) -> Tuple[str, str]:
    """Extract short task name and full language instruction from HDF5 filename.

    Supports libero_object, libero_spatial, and libero_goal naming patterns.

    Returns:
        (task_short, language_instruction)
    """
    stem = filename.replace("_multicam_demo.hdf5", "").replace("_multicam_demo", "")
    lang = stem.replace("_", " ")

    # libero_object: pick_up_the_{item}_and_place_it_in_the_basket
    m = re.match(r"pick_up_the_(.+)_and_place_it_in_the_basket", stem)
    if m:
        return m.group(1), lang

    # libero_spatial: pick_up_the_black_bowl_{location}_and_place_it_on_the_plate
    m = re.match(r"pick_up_the_black_bowl_(.+)_and_place_it_on_the_plate", stem)
    if m:
        return m.group(1), lang

    # libero_10 (long-horizon): KITCHEN_SCENE3_turn_on_the_stove_...
    # Strip scene prefix for short name, keep full stem for lang
    m = re.match(r"(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_(.+)", stem)
    if m:
        return m.group(1), lang

    # libero_goal / fallback: use full stem
    return stem, lang


def find_all_datasets(
    data_root: str, task_filter: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """Find all LIBERO HDF5 files.

    Returns:
        List of (task_short, language_instruction, hdf5_path)
    """
    root = Path(data_root)
    results = []
    for hdf5_path in sorted(root.glob("*.hdf5")):
        task_short, lang = extract_task_name(hdf5_path.name)
        if task_filter and task_short != task_filter:
            continue
        results.append((task_short, lang, str(hdf5_path)))
    return results


# ---------------------------------------------------------------------------
# Phase labeling (reused from preprocess_direction.py)
# ---------------------------------------------------------------------------

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
    window: int = 5,
) -> np.ndarray:
    """Label timesteps as move (0) or grip (1) based on gripper action transitions.

    LIBERO gripper convention:
      -1 = open, +1 = close
      -1 -> +1 = grasp transition
      +1 -> -1 = release transition
    Both transitions get a +-window region labeled as grip (1).
    """
    n = len(gripper_actions)
    phases = np.zeros(n, dtype=np.int32)
    gripper_flat = gripper_actions.flatten()

    transitions = np.where(np.diff(gripper_flat) != 0)[0]
    for t in transitions:
        start = max(0, t - window)
        end = min(n, t + window + 1)
        phases[start:end] = 1

    return phases


def find_segments(phases: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find contiguous segments of same phase.

    Returns:
        List of (start_idx, end_idx, phase) tuples.
    """
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
    language_instruction: str = "",
) -> Tuple[List[str], np.ndarray]:
    """Generate per-timestep instructions and unit-vector direction targets.

    For move segments:
      target = eef_base at end of segment (or start of next)
      direction = normalize(target - eef_current)
    For grip segments:
      direction = [0, 0, 0]

    All timesteps use the original language_instruction (not "move"/"grip").
    """
    n = len(phases)
    instructions = [language_instruction] * n
    direction = np.zeros((n, 3), dtype=np.float32)
    segments = find_segments(phases)

    for start, end, phase in segments:
        if phase == 0:  # move
            target_base = eef_base[end].copy()
            if end + 1 < n:
                target_base = eef_base[end + 1].copy()

            for t in range(start, end + 1):
                delta = target_base - eef_base[t]
                norm = np.linalg.norm(delta)
                if norm > 1e-6:
                    direction[t] = (delta / norm).astype(np.float32)
        else:  # grip
            pass  # direction stays [0, 0, 0]

    return instructions, direction


# ---------------------------------------------------------------------------
# Demo extraction
# ---------------------------------------------------------------------------

def extract_demo(
    f: h5py.File,
    demo_key: str,
    task_name: str,
    language_instruction: str,
    demo_id: int,
    grip_window: int,
    ori_dim: int = 4,
) -> Optional[Dict]:
    """Extract and preprocess a single demo from LIBERO HDF5.

    Args:
        f: Open HDF5 file handle.
        demo_key: e.g. "data/demo_0"
        task_name: Short task name.
        language_instruction: Full language description.
        demo_id: Integer demo index.
        grip_window: Window size for grip phase labeling.
        ori_dim: Orientation dimensions (4=quat for 12D, 3=ori for 11D Long).

    Returns:
        Dict with all preprocessed arrays and metadata, or None on failure.
    """
    demo = f[demo_key]

    if "obs" not in demo:
        logger.warning(f"Skipping {demo_key}: missing obs")
        return None

    obs = demo["obs"]
    n_steps = demo["actions"].shape[0]
    if n_steps < 10:
        logger.warning(f"Skipping {demo_key}: too few steps ({n_steps})")
        return None

    # --- Images (180-degree rotation for LIBERO) ---
    agentview = obs["agentview_rgb"][:]  # (T, 224, 224, 3) uint8
    images = agentview[:, ::-1, ::-1].copy()  # 180-degree rotation

    wrist_raw = obs["eye_in_hand_rgb"][:]  # (T, 224, 224, 3) uint8
    wrist_images = wrist_raw[:, ::-1, ::-1].copy()

    # --- Actions ---
    actions = demo["actions"][:].astype(np.float32)  # (T, 7)
    T = actions.shape[0]

    # --- State construction (truncate to T to match actions) ---
    ee_states = obs["ee_states"][:]  # (T+1, 6 or 7) float64
    eef_base = ee_states[:T, :3].astype(np.float32)   # (T, 3)
    eef_ori = ee_states[:T, 3:3 + ori_dim].astype(np.float32)  # (T, ori_dim)

    gripper_states = obs["gripper_states"][:T].astype(np.float32)  # (T, 2)

    # --- Phase labeling ---
    gripper_actions = actions[:, 6:7]  # (T, 1), -1=open, +1=close
    phases = label_phases(gripper_actions, window=grip_window)

    # --- Direction vector ---
    primitive_instructions, direction = generate_direction_data(phases, eef_base, language_instruction)

    # --- Build state: 12D (ori_dim=4) or 11D (ori_dim=3) ---
    robot_state = np.concatenate(
        [eef_base, eef_ori, gripper_states, direction], axis=1
    ).astype(np.float32)

    return {
        "images": images,
        "wrist_images": wrist_images,
        "robot_state": robot_state,
        "actions": actions,
        "phases": phases,
        "instruction": language_instruction,
        "primitive_instructions": primitive_instructions,
        "task_name": task_name,
        "demo_id": demo_id,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(data: np.ndarray) -> Dict:
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_task(
    task_name: str,
    language_instruction: str,
    hdf5_path: str,
    output_base: str,
    grip_window: int = 5,
    val_ratio: float = 0.1,
    max_demos: Optional[int] = None,
    ori_dim: int = 4,
) -> Optional[Dict]:
    """Process a single LIBERO task: all demos from one HDF5 file.

    Output structure:
        {output_base}/{task_name}/train/*.npz + *.json
        {output_base}/{task_name}/val/*.npz + *.json
        {output_base}/{task_name}/metadata.json

    Returns:
        Summary dict or None on failure.
    """
    task_dir = os.path.join(output_base, task_name)
    train_dir = os.path.join(task_dir, "train")
    val_dir = os.path.join(task_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_actions = []
    all_states = []
    total_samples = 0
    total_demos = 0
    move_steps = 0
    grip_steps = 0

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
            rng = np.random.RandomState(42)
            indices = rng.permutation(n_demos)
            val_indices = set(indices[:n_val].tolist())

            for i, demo_key in enumerate(demo_keys):
                demo_id = int(demo_key.split("_")[1])
                demo_data = extract_demo(
                    f,
                    f"data/{demo_key}",
                    task_name,
                    language_instruction,
                    demo_id,
                    grip_window,
                    ori_dim=ori_dim,
                )
                if demo_data is None:
                    continue

                is_val = i in val_indices
                split_dir = val_dir if is_val else train_dir
                split_name = "val" if is_val else "train"

                # --- Save NPZ ---
                filename = f"{task_name}_demo{demo_id}.npz"
                save_path = os.path.join(split_dir, filename)
                np.savez_compressed(
                    save_path,
                    images=demo_data["images"],
                    wrist_images=demo_data["wrist_images"],
                    robot_state=demo_data["robot_state"],
                    actions=demo_data["actions"],
                    phases=demo_data["phases"],
                )

                # --- Save JSON sidecar ---
                n_move = int((demo_data["phases"] == 0).sum())
                n_grip = int((demo_data["phases"] == 1).sum())
                meta = {
                    "instruction": demo_data["instruction"],
                    "primitive_instructions": demo_data["primitive_instructions"],
                    "task_name": task_name,
                    "demo_id": demo_id,
                    "n_steps": demo_data["n_steps"],
                    "n_move": n_move,
                    "n_grip": n_grip,
                    "split": split_name,
                    "state_format": f"eef_base(3)+eef_ori({ori_dim})+gripper(2)+direction(3)",
                }
                meta_path = save_path.replace(".npz", ".json")
                with open(meta_path, "w") as mf:
                    json.dump(meta, mf, indent=2)

                all_actions.append(demo_data["actions"])
                all_states.append(demo_data["robot_state"])
                total_samples += demo_data["n_steps"]
                total_demos += 1
                move_steps += n_move
                grip_steps += n_grip

    except Exception as e:
        logger.error(f"Failed to process {hdf5_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not all_actions:
        logger.error(f"No data extracted for task {task_name}!")
        return None

    # --- Compute per-task statistics ---
    all_actions_array = np.concatenate(all_actions, axis=0)
    all_states_array = np.concatenate(all_states, axis=0)

    action_stats = compute_stats(all_actions_array)
    action_stats["mode"] = "min_max"
    state_stats = compute_stats(all_states_array)
    action_quantile_stats = {
        "q01": np.percentile(all_actions_array, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions_array, 99, axis=0).tolist(),
    }

    metadata = {
        "action_stats": action_stats,
        "state_stats": state_stats,
        "action_quantile_stats": action_quantile_stats,
        "state_format": f"eef_base(3)+eef_ori({ori_dim})+gripper(2)+direction(3)",
        "action_dim": int(all_actions_array.shape[1]),
        "state_dim": int(all_states_array.shape[1]),
        "total_demos": total_demos,
        "total_samples": total_samples,
        "move_steps": move_steps,
        "grip_steps": grip_steps,
        "grip_window": grip_window,
        "val_ratio": val_ratio,
        "task_name": task_name,
        "language_instruction": language_instruction,
        "source": "libero_object",
    }

    metadata_path = os.path.join(task_dir, "metadata.json")
    with open(metadata_path, "w") as mf:
        json.dump(metadata, mf, indent=2)

    # --- Print task summary ---
    move_pct = move_steps / max(total_samples, 1) * 100
    logger.info(
        f"  {task_name}: {total_demos} demos, {total_samples} steps "
        f"(move: {move_pct:.0f}%, grip: {100 - move_pct:.0f}%)"
    )

    # --- Verify direction norms ---
    move_mask = all_states_array[:, 9:12].any(axis=1)
    if move_mask.any():
        dir_norms = np.linalg.norm(all_states_array[move_mask, 9:12], axis=1)
        logger.info(
            f"    Direction norms (move steps): mean={dir_norms.mean():.4f}, "
            f"min={dir_norms.min():.4f}, max={dir_norms.max():.4f}"
        )

    return {
        "task_name": task_name,
        "demos": total_demos,
        "steps": total_samples,
        "move_steps": move_steps,
        "grip_steps": grip_steps,
        "action_stats": action_stats,
        "state_stats": state_stats,
    }


def process_all(
    data_root: str,
    output_dir: str,
    task_filter: Optional[str] = None,
    grip_window: int = 5,
    val_ratio: float = 0.1,
    max_demos: Optional[int] = None,
    ori_dim: int = 4,
) -> None:
    """Process all (or filtered) LIBERO-Object tasks."""
    datasets = find_all_datasets(data_root, task_filter)
    if not datasets:
        logger.error(f"No datasets found in {data_root}" +
                     (f" (filter: {task_filter})" if task_filter else ""))
        return

    logger.info(f"Found {len(datasets)} task(s) to process")
    os.makedirs(output_dir, exist_ok=True)

    all_task_summaries = {}
    global_actions = []
    global_states = []

    for ds_idx, (task_name, lang, hdf5_path) in enumerate(datasets):
        logger.info(f"[{ds_idx + 1}/{len(datasets)}] Processing: {task_name} ({hdf5_path})")

        summary = process_task(
            task_name=task_name,
            language_instruction=lang,
            hdf5_path=hdf5_path,
            output_base=output_dir,
            grip_window=grip_window,
            val_ratio=val_ratio,
            max_demos=max_demos,
            ori_dim=ori_dim,
        )
        if summary is not None:
            all_task_summaries[task_name] = {
                "demos": summary["demos"],
                "steps": summary["steps"],
                "move_steps": summary["move_steps"],
                "grip_steps": summary["grip_steps"],
            }

    # --- Global summary ---
    total_demos = sum(s["demos"] for s in all_task_summaries.values())
    total_steps = sum(s["steps"] for s in all_task_summaries.values())

    logger.info("=" * 60)
    logger.info("LIBERO-Object preprocessing complete!")
    logger.info(f"  Output dir: {output_dir}")
    ori_label = "eef_quat(4)" if ori_dim == 4 else f"eef_ori({ori_dim})"
    logger.info(f"  State format: eef_base(3) + {ori_label} + gripper(2) + direction(3)")
    logger.info(f"  Tasks processed: {len(all_task_summaries)}")
    logger.info(f"  Total demos: {total_demos}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info("")
    logger.info("Per-task statistics:")
    for task, stats in sorted(all_task_summaries.items()):
        move_pct = stats["move_steps"] / max(stats["steps"], 1) * 100
        logger.info(
            f"  {task:25s}: {stats['demos']:3d} demos, {stats['steps']:5d} steps "
            f"(move: {move_pct:.0f}%, grip: {100 - move_pct:.0f}%)"
        )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LIBERO-Object preprocessor for GROOT N1.5 LoRA training"
    )
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to LIBERO-Object multicam HDF5 directory",
    )
    parser.add_argument(
        "--output", type=str, default="data_libero",
        help="Output base directory (per-task subdirs will be created)",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Process single task (short name, e.g. 'alphabet_soup')",
    )
    parser.add_argument("--grip-window", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--ori-dim", type=int, default=4,
                        help="Orientation dimensions: 4=quat (12D), 3=ori (11D for Long)")
    args = parser.parse_args()

    process_all(
        data_root=args.data_root,
        output_dir=args.output,
        task_filter=args.task,
        grip_window=args.grip_window,
        val_ratio=args.val_ratio,
        max_demos=args.max_demos,
        ori_dim=args.ori_dim,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
