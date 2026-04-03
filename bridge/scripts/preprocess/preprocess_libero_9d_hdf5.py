#!/usr/bin/env python3
"""Generic LIBERO 9D preprocessor — reads HDF5 directly (no env replay needed).

State 9D: eef_pos(3) + eef_quat(4) + gripper(2)
Works with any LIBERO suite: libero_object, libero_spatial, libero_goal.

Usage:
    python preprocess_libero_9d_hdf5.py --data-root .../libero_spatial --output data_libero_spatial_9d
    python preprocess_libero_9d_hdf5.py --data-root .../libero_goal --output data_libero_goal_9d
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


def extract_task_name(filename: str) -> Tuple[str, str]:
    """Extract short task name and language instruction from HDF5 filename."""
    stem = filename.replace("_multicam_demo.hdf5", "").replace("_multicam_demo", "")
    lang = stem.replace("_", " ")

    # libero_object
    m = re.match(r"pick_up_the_(.+)_and_place_it_in_the_basket", stem)
    if m:
        return m.group(1), lang

    # libero_spatial
    m = re.match(r"pick_up_the_black_bowl_(.+)_and_place_it_on_the_plate", stem)
    if m:
        return m.group(1), lang

    # libero_10 (long-horizon): KITCHEN_SCENE3_turn_on_the_stove_...
    m = re.match(r"(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_(.+)", stem)
    if m:
        return m.group(1), lang

    # libero_goal / fallback
    return stem, lang


def find_all_datasets(data_root: str, task_filter: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Find all HDF5 files in data_root."""
    root = Path(data_root)
    results = []
    for hdf5_path in sorted(root.glob("*.hdf5")):
        task_short, lang = extract_task_name(hdf5_path.name)
        if task_filter and task_short != task_filter:
            continue
        results.append((task_short, lang, str(hdf5_path)))
    return results


def compute_stats(data: np.ndarray) -> Dict:
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.clip(np.std(data, axis=0), 1e-6, None).tolist(),
    }


def process_task(
    task_name: str,
    language_instruction: str,
    hdf5_path: str,
    output_base: str,
    val_ratio: float = 0.1,
    max_demos: Optional[int] = None,
) -> Optional[Dict]:
    """Process a single task: extract 9D state from HDF5 directly."""
    task_dir = os.path.join(output_base, task_name)
    train_dir = os.path.join(task_dir, "train")
    val_dir = os.path.join(task_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_actions = []
    all_states = []
    total_demos = 0

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
                demo = f[f"data/{demo_key}"]
                obs = demo["obs"]

                actions = demo["actions"][:].astype(np.float32)  # (T, 7)
                n_steps = actions.shape[0]
                if n_steps < 10:
                    logger.warning(f"  Skip {demo_key}: too short ({n_steps})")
                    continue

                # Images: 180° rotation (LIBERO convention)
                agentview = obs["agentview_rgb"][:]  # (T, 224, 224, 3)
                images = agentview[:, ::-1, ::-1].copy()

                # 9D state: eef_pos(3) + eef_quat(4) + gripper(2)
                ee_states = obs["ee_states"][:].astype(np.float32)  # (T, 7)
                eef_pos = ee_states[:, :3]   # (T, 3)
                eef_quat = ee_states[:, 3:7]  # (T, 4)
                gripper = obs["gripper_states"][:].astype(np.float32)[:, :2]  # (T, 2)

                robot_state = np.concatenate([eef_pos, eef_quat, gripper], axis=1)  # (T, 9)

                is_val = i in val_indices
                split_dir = val_dir if is_val else train_dir

                demo_id = int(demo_key.split("_")[1])
                npz_name = f"{task_name}_demo{demo_id:03d}.npz"
                np.savez_compressed(
                    os.path.join(split_dir, npz_name),
                    images=images,
                    robot_state=robot_state,
                    actions=actions,
                    phases=np.zeros(n_steps, dtype=np.int32),
                )

                meta = {
                    "task_name": task_name,
                    "instruction": language_instruction,
                    "demo_id": demo_id,
                    "n_steps": n_steps,
                    "state_dim": 9,
                    "state_format": "eef_pos(3)+eef_quat(4)+gripper(2)",
                    "primitive_instructions": [language_instruction] * n_steps,
                    "split": "val" if is_val else "train",
                }
                with open(os.path.join(split_dir, npz_name.replace(".npz", ".json")), "w") as mf:
                    json.dump(meta, mf, indent=2)

                all_actions.append(actions)
                all_states.append(robot_state)
                total_demos += 1

    except Exception as e:
        logger.error(f"Failed to process {hdf5_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not all_actions:
        logger.error(f"No data extracted for {task_name}")
        return None

    all_actions_arr = np.concatenate(all_actions, axis=0)
    all_states_arr = np.concatenate(all_states, axis=0)
    total_steps = all_actions_arr.shape[0]

    # Save stats
    stats = {
        "action_stats": {
            **compute_stats(all_actions_arr),
            "mode": "min_max",
        },
        "state_stats": compute_stats(all_states_arr),
    }
    with open(os.path.join(task_dir, "data_stats.json"), "w") as sf:
        json.dump(stats, sf, indent=2)

    # Save metadata
    task_meta = {
        "task_name": task_name,
        "instruction": language_instruction,
        "n_demos": total_demos,
        "total_steps": total_steps,
        "state_dim": 9,
        "action_dim": 7,
        "action_stats": stats["action_stats"],
        "state_stats": stats["state_stats"],
    }
    with open(os.path.join(task_dir, "metadata.json"), "w") as mf:
        json.dump(task_meta, mf, indent=2)

    logger.info(f"  {task_name}: {total_demos} demos, {total_steps} steps")
    return {"task_name": task_name, "demos": total_demos, "steps": total_steps}


def main():
    parser = argparse.ArgumentParser(description="Generic LIBERO 9D preprocessor (HDF5-only)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to LIBERO multicam HDF5 directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output base directory")
    parser.add_argument("--task", type=str, default=None,
                        help="Process single task (short name)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-demos", type=int, default=None)
    args = parser.parse_args()

    datasets = find_all_datasets(args.data_root, args.task)
    if not datasets:
        logger.error(f"No datasets found in {args.data_root}")
        return 1

    logger.info(f"Found {len(datasets)} task(s) → {args.output} (9D state)")
    os.makedirs(args.output, exist_ok=True)

    for idx, (task_name, lang, hdf5_path) in enumerate(datasets):
        logger.info(f"[{idx+1}/{len(datasets)}] {task_name}: {hdf5_path}")
        process_task(task_name, lang, hdf5_path, args.output,
                     val_ratio=args.val_ratio, max_demos=args.max_demos)

    logger.info("Done!")


if __name__ == "__main__":
    sys.exit(main() or 0)
