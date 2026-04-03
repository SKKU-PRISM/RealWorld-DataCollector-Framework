#!/usr/bin/env python3
"""Preprocess SO101 LeRobot v3 datasets for GROOT LoRA training (방향 B).

State 11D: ee_pos(3) + ee_quat(4) + gripper(1) + goal_direction(3)
Action 7D: delta_pos(3) + delta_rot_rpy(3) + gripper_binary(1)

원본 데이터의 skill.goal_position.robot_xyzrpy 를 사용하여
매 스텝마다 실제 목표 방향(goal_direction)을 계산합니다.

Usage:
    python preprocess_so101.py \
        --data-dir data_hf/SO101-single-pick_redblock_place_bluedish \
        --output-dir data_so101/pick_redblock_place_bluedish \
        --image-size 224

    # 전체 SO101 데이터셋:
    python preprocess_so101.py --all --image-size 224
"""

import argparse
import json
import glob
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# SO101 datasets
# ---------------------------------------------------------------------------

SO101_DATASETS = {
    "pick_redblock_place_bluedish": {
        "dir": "SO101-single-pick_redblock_place_bluedish",
        "instruction": "pick red block and place on blue dish",
    },
    "distribute_chocolatepies": {
        "dir": "SO101-single-distribute_chocolatepies",
        "instruction": "distribute chocolate pies",
    },
    "place_yellowblock_between_chocolatepies": {
        "dir": "SO101-single-place_yellowblock_between_chocolatepies",
        "instruction": "place yellow block between chocolate pies",
    },
    "arrange_YRPblock_top2bottom": {
        "dir": "SO101-single-arrange_YRPblock_top2bottom",
        "instruction": "arrange YRP blocks top to bottom",
    },
    "stack_yellow_redblock_30epi": {
        "dir": "SO101-single-stack_yellow_redblock_30epi",
        "instruction": "stack yellow block on red block",
    },
    "stack_red_yellow_purple_blocks": {
        "dir": "SO101-single-stack_red_yellow_purple_blocks",
        "instruction": "stack red yellow purple blocks",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """RPY (N, 3) → quaternion (N, 4) [x, y, z, w] (scipy convention)."""
    return Rotation.from_euler("xyz", rpy).as_quat().astype(np.float32)


def compute_goal_direction(ee_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """매 스텝마다 goal_direction = normalize(goal_xyz - ee_xyz).

    Args:
        ee_pos: (N, 3) current EEF position
        goal_pos: (N, 3) skill goal position (robot_xyzrpy[:3])

    Returns:
        (N, 3) normalized direction vectors
    """
    diff = goal_pos - ee_pos
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (diff / norms).astype(np.float32)


def compute_delta_actions(ee_pos: np.ndarray, ee_rpy: np.ndarray,
                          gripper: np.ndarray) -> np.ndarray:
    """7D delta actions: delta_pos(3) + delta_rot_rpy(3) + gripper(1).

    Args:
        ee_pos: (N, 3)
        ee_rpy: (N, 3)
        gripper: (N,) gripper_binary (0 or 1)

    Returns:
        (N, 7) actions. Last step has zero delta.
    """
    n = len(ee_pos)
    actions = np.zeros((n, 7), dtype=np.float32)

    # Delta position
    actions[:-1, :3] = ee_pos[1:] - ee_pos[:-1]

    # Delta rotation (with angle wrapping)
    delta_rpy = ee_rpy[1:] - ee_rpy[:-1]
    delta_rpy = (delta_rpy + np.pi) % (2 * np.pi) - np.pi
    actions[:-1, 3:6] = delta_rpy

    # Gripper binary (1.0 = open, 0.0 = closed)
    actions[:, 6] = gripper

    return actions


def decode_video_frames(video_path: str, image_size: int) -> np.ndarray:
    """mp4 → (N, H, W, 3) uint8 RGB."""
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size))
        frames.append(img)
    container.close()
    return np.array(frames, dtype=np.uint8)


def load_parquet_data(data_dir: Path) -> pd.DataFrame:
    """Load all parquet chunks into a single DataFrame."""
    pfiles = sorted(glob.glob(str(data_dir / "data" / "chunk-*" / "*.parquet")))
    if not pfiles:
        raise FileNotFoundError(f"No parquet files in {data_dir / 'data'}")
    tables = [pq.read_table(f) for f in pfiles]
    import pyarrow as pa
    return pa.concat_tables(tables).to_pandas()


def load_video_frames(data_dir: Path, camera: str, image_size: int) -> np.ndarray:
    """Load and decode all video chunks for a camera."""
    video_dir = data_dir / "videos" / camera
    video_files = sorted(glob.glob(str(video_dir / "chunk-*" / "*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No video files in {video_dir}")
    all_frames = []
    for vf in video_files:
        frames = decode_video_frames(vf, image_size)
        all_frames.append(frames)
    return np.concatenate(all_frames, axis=0)


def compute_stats(all_actions: list, all_states: list) -> dict:
    """Compute normalization statistics."""
    actions = np.concatenate(all_actions, axis=0)
    states = np.concatenate(all_states, axis=0)

    def stats_for(arr):
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    return {
        "action_stats": {**stats_for(actions), "mode": "min_max"},
        "state_stats": stats_for(states),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_dataset(data_dir: Path, output_dir: Path, image_size: int,
                    task_name: str, default_instruction: str):
    """하나의 SO101 데이터셋 전처리."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {task_name}")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Load metadata
    info_path = data_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    n_episodes = info.get("total_episodes", 0)
    print(f"  FPS: {fps}, Episodes: {n_episodes}")

    # Load parquet data
    df = load_parquet_data(data_dir)
    print(f"  Total rows: {len(df)}")

    # Load video frames
    print(f"  Loading realsense video...")
    realsense_frames = load_video_frames(data_dir, "observation.images.realsense", image_size)
    print(f"  Realsense frames: {realsense_frames.shape}")

    has_wrist = (data_dir / "videos" / "observation.images.innomaker").exists()
    if has_wrist:
        print(f"  Loading innomaker (wrist) video...")
        wrist_frames = load_video_frames(data_dir, "observation.images.innomaker", image_size)
        print(f"  Wrist frames: {wrist_frames.shape}")
    else:
        wrist_frames = None

    # Process episodes
    episodes = sorted(df["episode_index"].unique())
    all_actions = []
    all_states = []

    for ep_idx, ep_id in enumerate(episodes):
        ep_df = df[df["episode_index"] == ep_id].sort_values("frame_index").reset_index(drop=True)
        n_frames = len(ep_df)
        demo_num = ep_idx + 1

        # --- EEF position and rotation ---
        ee_raw = np.array(ep_df["observation.ee_pos.robot_xyzrpy"].tolist(), dtype=np.float32)
        ee_pos = ee_raw[:, :3]
        ee_rpy = ee_raw[:, 3:6]
        ee_quat = rpy_to_quat(ee_rpy)  # (N, 4) xyzw

        # --- Gripper binary ---
        gripper = np.array(ep_df["observation.gripper_binary"].tolist(), dtype=np.float32).flatten()

        # --- Goal direction (from skill.goal_position) ---
        goal_raw = np.array(ep_df["skill.goal_position.robot_xyzrpy"].tolist(), dtype=np.float32)
        goal_xyz = goal_raw[:, :3]
        goal_direction = compute_goal_direction(ee_pos, goal_xyz)

        # --- State 11D ---
        robot_state = np.concatenate([
            ee_pos,          # 0-2:  eef xyz
            ee_quat,         # 3-6:  eef quaternion (xyzw)
            gripper.reshape(-1, 1),  # 7: gripper binary
            goal_direction,  # 8-10: normalized goal direction
        ], axis=-1).astype(np.float32)

        # --- Actions 7D ---
        actions = compute_delta_actions(ee_pos, ee_rpy, gripper)

        # --- Skill annotations ---
        skills = ep_df["skill.natural_language"].tolist()
        skill_types = ep_df["skill.type"].tolist()

        # --- Images ---
        global_indices = ep_df["index"].values
        images = realsense_frames[global_indices]
        if wrist_frames is not None:
            wrist_imgs = wrist_frames[global_indices]
        else:
            wrist_imgs = images.copy()

        # --- Save npz ---
        npz_path = output_dir / f"{task_name}_demo{demo_num}.npz"
        np.savez_compressed(
            npz_path,
            images=images,
            wrist_images=wrist_imgs,
            robot_state=robot_state,
            actions=actions,
        )

        # --- Save json metadata ---
        # Collect skill segments
        skill_segments = []
        prev_skill = None
        for i in range(n_frames):
            if skill_types[i] != prev_skill:
                skill_segments.append({
                    "start_step": i,
                    "type": skill_types[i],
                    "instruction": skills[i],
                    "goal_ee": goal_raw[i, :3].tolist(),
                    "goal_gripper": float(ep_df.iloc[i]["skill.goal_position.gripper"]),
                })
                prev_skill = skill_types[i]

        meta = {
            "instruction": default_instruction,
            "task_name": task_name,
            "demo_id": demo_num,
            "n_steps": n_frames,
            "split": "train",
            "state_format": "ee_pos(3)+ee_quat_xyzw(4)+gripper(1)+goal_direction(3)",
            "action_format": "delta_pos(3)+delta_rpy(3)+gripper_binary(1)",
            "state_dim": 11,
            "action_dim": 7,
            "skill_segments": skill_segments,
        }
        json_path = output_dir / f"{task_name}_demo{demo_num}.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_actions.append(actions)
        all_states.append(robot_state)

        grip_changes = np.sum(np.abs(np.diff(gripper)) > 0.5)
        print(f"  demo{demo_num:3d}: {n_frames} steps, "
              f"{len(skill_segments)} skills, "
              f"grip_changes={grip_changes}")

    # --- Global stats ---
    stats = compute_stats(all_actions, all_states)
    stats.update({
        "state_format": "ee_pos(3)+ee_quat_xyzw(4)+gripper(1)+goal_direction(3)",
        "action_format": "delta_pos(3)+delta_rpy(3)+gripper_binary(1)",
        "state_dim": 11,
        "action_dim": 7,
        "total_demos": len(episodes),
        "total_samples": sum(len(a) for a in all_actions),
        "source_robot": "so101_follower",
        "source_fps": fps,
        "task_name": task_name,
    })

    stats_path = output_dir / "metadata.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Done: {len(episodes)} episodes, "
          f"{stats['total_samples']} samples -> {output_dir}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SO101 datasets for GROOT training (방향 B: Cartesian + skill goal)"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="단일 데이터셋 경로")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="출력 경로")
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="move")
    parser.add_argument("--all", action="store_true",
                        help="모든 SO101 데이터셋 처리")
    parser.add_argument("--data-root", type=str, default="data_hf",
                        help="HF 데이터셋 루트 (--all 사용 시)")
    parser.add_argument("--output-root", type=str, default="data_so101",
                        help="출력 루트 (--all 사용 시)")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if args.all:
        data_root = Path(args.data_root)
        output_root = Path(args.output_root)
        all_stats = {}

        for task_name, cfg in SO101_DATASETS.items():
            data_dir = data_root / cfg["dir"]
            if not data_dir.exists():
                print(f"SKIP: {data_dir} not found")
                continue
            output_dir = output_root / task_name
            stats = process_dataset(
                data_dir, output_dir, args.image_size,
                task_name, cfg["instruction"],
            )
            all_stats[task_name] = {
                "n_demos": stats["total_demos"],
                "n_samples": stats["total_samples"],
            }

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, s in all_stats.items():
            print(f"  {name:50s} {s['n_demos']:3d} demos  {s['n_samples']:6d} samples")
        total_demos = sum(s["n_demos"] for s in all_stats.values())
        total_samples = sum(s["n_samples"] for s in all_stats.values())
        print(f"  {'TOTAL':50s} {total_demos:3d} demos  {total_samples:6d} samples")

    else:
        if not args.data_dir:
            parser.error("--data-dir required (or use --all)")
        task_name = args.task_name or Path(args.data_dir).name
        output_dir = Path(args.output_dir or f"data_so101/{task_name}")
        process_dataset(
            Path(args.data_dir), output_dir, args.image_size,
            task_name, args.instruction,
        )


if __name__ == "__main__":
    main()
