#!/usr/bin/env python3
"""Preprocess SO101 datasets for GROOT LoRA — Joint-space variant.

State 9D: joint_angles(6) + goal_xyz(3)
Action 6D: joint target positions (absolute, degrees)

원본 데이터의 observation.state (6D joint degrees)와
skill.goal_position.robot_xyzrpy[:3] (goal xyz)를 사용합니다.
Action은 원본의 action (6D joint target degrees) 그대로 사용.

Usage:
    python preprocess_so101_joint.py --all --image-size 224
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


def decode_video_frames(video_path: str, image_size: int) -> np.ndarray:
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
    pfiles = sorted(glob.glob(str(data_dir / "data" / "chunk-*" / "*.parquet")))
    if not pfiles:
        raise FileNotFoundError(f"No parquet files in {data_dir / 'data'}")
    import pyarrow as pa
    tables = [pq.read_table(f) for f in pfiles]
    return pa.concat_tables(tables).to_pandas()


def load_video_frames(data_dir: Path, camera: str, image_size: int) -> np.ndarray:
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


def process_dataset(data_dir: Path, output_dir: Path, image_size: int,
                    task_name: str, default_instruction: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing (joint): {task_name}")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    info_path = data_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 30)

    df = load_parquet_data(data_dir)
    print(f"  Total rows: {len(df)}")

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

    episodes = sorted(df["episode_index"].unique())
    all_actions = []
    all_states = []

    for ep_idx, ep_id in enumerate(episodes):
        ep_df = df[df["episode_index"] == ep_id].sort_values("frame_index").reset_index(drop=True)
        n_frames = len(ep_df)
        demo_num = ep_idx + 1

        # Joint angles (6D, degrees)
        joint_state = np.array(ep_df["observation.state"].tolist(), dtype=np.float32)

        # Goal xyz from skill.goal_position.robot_xyzrpy[:3]
        goal_raw = np.array(ep_df["skill.goal_position.robot_xyzrpy"].tolist(), dtype=np.float32)
        goal_xyz = goal_raw[:, :3]

        # State 9D: joint(6) + goal_xyz(3)
        robot_state = np.concatenate([joint_state, goal_xyz], axis=-1).astype(np.float32)

        # Action 6D: joint target positions (degrees) — use as-is
        actions = np.array(ep_df["action"].tolist(), dtype=np.float32)

        # Skill annotations
        skill_types = ep_df["skill.type"].tolist()
        skills_nl = ep_df["skill.natural_language"].tolist()

        # Images
        global_indices = ep_df["index"].values
        images = realsense_frames[global_indices]
        if wrist_frames is not None:
            wrist_imgs = wrist_frames[global_indices]
        else:
            wrist_imgs = images.copy()

        # Save npz
        npz_path = output_dir / f"{task_name}_demo{demo_num}.npz"
        np.savez_compressed(
            npz_path,
            images=images,
            wrist_images=wrist_imgs,
            robot_state=robot_state,
            actions=actions,
        )

        # Save json metadata
        skill_segments = []
        prev_skill = None
        for i in range(n_frames):
            if skill_types[i] != prev_skill:
                skill_segments.append({
                    "start_step": i,
                    "type": skill_types[i],
                    "instruction": skills_nl[i],
                    "goal_xyz": goal_raw[i, :3].tolist(),
                    "goal_gripper": float(ep_df.iloc[i]["skill.goal_position.gripper"]),
                })
                prev_skill = skill_types[i]

        meta = {
            "instruction": default_instruction,
            "task_name": task_name,
            "demo_id": demo_num,
            "n_steps": n_frames,
            "split": "train",
            "state_format": "joint_deg(6)+goal_xyz(3)",
            "action_format": "joint_target_deg(6)",
            "state_dim": 9,
            "action_dim": 6,
            "skill_segments": skill_segments,
        }
        json_path = output_dir / f"{task_name}_demo{demo_num}.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_actions.append(actions)
        all_states.append(robot_state)

        print(f"  demo{demo_num:3d}: {n_frames} steps, {len(skill_segments)} skills")

    stats = compute_stats(all_actions, all_states)
    stats.update({
        "state_format": "joint_deg(6)+goal_xyz(3)",
        "action_format": "joint_target_deg(6)",
        "state_dim": 9,
        "action_dim": 6,
        "total_demos": len(episodes),
        "total_samples": sum(len(a) for a in all_actions),
        "source_robot": "so101_follower",
        "source_fps": fps,
        "task_name": task_name,
    })

    stats_path = output_dir / "metadata.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # metadata_extended for training
    ext = {
        "state_stats": stats["state_stats"],
        "action_stats": stats["action_stats"],
        "action_quantile_stats": {
            "q01": stats["action_stats"]["q01"],
            "q99": stats["action_stats"]["q99"],
        },
    }
    with open(output_dir / "metadata_extended.json", "w") as f:
        json.dump(ext, f, indent=2)

    print(f"\n  Done: {len(episodes)} episodes, {stats['total_samples']} samples -> {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SO101 datasets — Joint-space (9D state, 6D action)"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="move")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=str, default="data_hf")
    parser.add_argument("--output-root", type=str, default="data_so101_joint")
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
            stats = process_dataset(data_dir, output_dir, args.image_size, task_name, cfg["instruction"])
            all_stats[task_name] = {"n_demos": stats["total_demos"], "n_samples": stats["total_samples"]}

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
        output_dir = Path(args.output_dir or f"data_so101_joint/{task_name}")
        process_dataset(Path(args.data_dir), output_dir, args.image_size, task_name, args.instruction)


if __name__ == "__main__":
    main()
