#!/usr/bin/env python3
"""LIBERO-Object 9D proprioception-only preprocessor.

State 9D: eef_pos(3) + eef_quat(4) + gripper(2)
object_pos(3)와 basket_pos(3)는 포함하지 않음 (순수 proprioception만 사용).
No phase labels, no direction vectors. Full continuous trajectories.

Usage:
    MUJOCO_GL=egl PYTHONPATH=~/SGRPO/libero_repo:. python preprocess_libero_9d.py
    MUJOCO_GL=egl PYTHONPATH=~/SGRPO/libero_repo:. python preprocess_libero_9d.py --task alphabet_soup
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

# Patch torch.load for PyTorch 2.6
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")

DATA_ROOT = os.environ.get("LIBERO_DATA_ROOT", os.path.expanduser("~/SGRPO/data/libero/datasets_multicam/libero_object"))
OUTPUT_BASE = "data_libero_9d"  # 9D 전용 출력 디렉토리

# Task name → (benchmark_task_id, obj_obs_prefix, basket_obs_prefix)
TASK_INFO = {
    "alphabet_soup":      (0, "alphabet_soup_1",      "basket_1"),
    "bbq_sauce":          (1, "bbq_sauce_1",          "basket_1"),
    "butter":             (2, "butter_1",             "basket_1"),
    "chocolate_pudding":  (3, "chocolate_pudding_1",  "basket_1"),
    "cream_cheese":       (4, "cream_cheese_1",       "basket_1"),
    "ketchup":            (5, "ketchup_1",            "basket_1"),
    "milk":               (6, "milk_1",               "basket_1"),
    "orange_juice":       (7, "orange_juice_1",       "basket_1"),
    "salad_dressing":     (8, "salad_dressing_1",     "basket_1"),
    "tomato_sauce":       (9, "tomato_sauce_1",       "basket_1"),
}


def find_hdf5(task_short: str) -> Optional[str]:
    """Find HDF5 file for a task."""
    for p in Path(DATA_ROOT).glob("*.hdf5"):
        name = p.stem.replace("_multicam_demo", "").replace("_", " ")
        if task_short.replace("_", " ") in name:
            return str(p)
    return None


def extract_task_instruction(filename: str) -> str:
    stem = Path(filename).stem.replace("_multicam_demo", "")
    return stem.replace("_", " ")


def process_task(task_short: str):
    """Process a single task: replay demos and extract 9D state (proprioception only)."""
    sys.path.insert(0, os.path.expanduser("~/SGRPO/libero_repo"))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    task_id, obj_prefix, basket_prefix = TASK_INFO[task_short]
    hdf5_path = find_hdf5(task_short)
    if not hdf5_path:
        logger.error(f"HDF5 not found for {task_short}")
        return

    logger.info(f"Processing {task_short}: {hdf5_path}")
    instruction = extract_task_instruction(os.path.basename(hdf5_path))

    # Setup env for replay
    benchmark = get_benchmark("libero_object")()
    task = benchmark.tasks[task_id]
    env_args = {
        "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        "camera_heights": 224, "camera_widths": 224,
        "has_renderer": False, "has_offscreen_renderer": True,
        "ignore_done": True, "use_camera_obs": True,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "reward_shaping": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)

    output_dir = Path(OUTPUT_BASE) / task_short / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(hdf5_path, "r")
    demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")],
                        key=lambda x: int(x.split("_")[1]))

    all_actions = []
    all_states = []
    n_saved = 0

    for demo_idx, demo_key in enumerate(demo_keys):
        demo = f["data"][demo_key]
        states_arr = demo["states"][:]
        actions_arr = demo["actions"][:].astype(np.float32)
        n_steps = actions_arr.shape[0]

        if n_steps < 10:
            logger.warning(f"  Skip {demo_key}: too short ({n_steps})")
            continue

        # Replay: set init state, then step through
        try:
            obs = env.set_init_state(states_arr[0])
        except Exception as e:
            logger.warning(f"  Skip {demo_key}: set_init_state failed: {e}")
            continue

        images_list = []
        robot_states_list = []

        for t in range(n_steps):
            # Extract current obs
            img = obs["agentview_image"]
            img = img[::-1, ::-1].copy()  # 180° rotation to match training convention (HDF5 원본)

            eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
            eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
            gripper = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]

            # 9D state (proprioception only): eef_pos(3) + eef_quat(4) + gripper(2)
            # object_pos(3)와 basket_pos(3)는 제외 — 15D e2e 대비 6D 감소
            state_9d = np.concatenate([eef_pos, eef_quat, gripper])
            robot_states_list.append(state_9d)
            images_list.append(img)

            # Step env with recorded action
            obs, _, _, _ = env.step(actions_arr[t].tolist())

        images = np.stack(images_list)           # (T, 224, 224, 3)
        robot_state = np.stack(robot_states_list)  # (T, 9)

        # Save NPZ
        npz_name = f"{task_short}_demo{demo_idx:03d}.npz"
        np.savez_compressed(
            output_dir / npz_name,
            images=images,
            robot_state=robot_state,
            actions=actions_arr,
            phases=np.zeros(n_steps, dtype=np.int32),
        )

        # Save metadata JSON
        meta = {
            "task_name": task_short,
            "instruction": instruction,  # full task instruction (NOT primitive "move"/"grip")
            "demo_id": demo_idx,
            "n_steps": n_steps,
            "state_dim": 9,
            "state_format": "eef_pos(3)+eef_quat(4)+gripper(2)",
            "primitive_instructions": [instruction] * n_steps,
        }
        with open(output_dir / f"{task_short}_demo{demo_idx:03d}.json", "w") as mf:
            json.dump(meta, mf, indent=2)

        all_actions.append(actions_arr)
        all_states.append(robot_state)
        n_saved += 1

        if demo_idx % 10 == 0:
            logger.info(f"  {demo_key}: {n_steps} steps, eef_pos={eef_pos.round(3)}")

    f.close()
    env.close()

    if not all_actions:
        logger.error(f"  No demos saved for {task_short}")
        return

    # Compute stats
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    stats = {
        "action_stats": {
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.clip(np.std(all_actions, axis=0), 1e-6, None).tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
            "mode": "min_max",
        },
        "state_stats": {
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.clip(np.std(all_states, axis=0), 1e-6, None).tolist(),
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
        },
    }
    stats_path = Path(OUTPUT_BASE) / task_short / "data_stats.json"
    with open(stats_path, "w") as sf:
        json.dump(stats, sf, indent=2)

    # Copy extended metadata
    src_ext = Path("data_libero") / task_short / "metadata_extended.json"
    dst_ext = Path(OUTPUT_BASE) / task_short / "metadata_extended.json"
    if src_ext.exists():
        import shutil
        shutil.copy2(src_ext, dst_ext)

    # Write task metadata
    task_meta = {
        "task_name": task_short,
        "instruction": instruction,
        "n_demos": n_saved,
        "state_dim": 9,   # 9D: proprioception only (eef_pos+quat+gripper)
        "action_dim": 7,
    }
    with open(Path(OUTPUT_BASE) / task_short / "metadata.json", "w") as tmf:
        json.dump(task_meta, tmf, indent=2)

    logger.info(f"  {task_short}: {n_saved} demos saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASK_INFO.keys())

    logger.info(f"Processing {len(tasks)} tasks → {OUTPUT_BASE} (state_dim=9, proprioception only)")
    for task in tasks:
        process_task(task)
    logger.info("Done!")


if __name__ == "__main__":
    main()
