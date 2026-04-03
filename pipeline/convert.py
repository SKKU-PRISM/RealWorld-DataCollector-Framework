#!/usr/bin/env python3
"""LeRobot v3.0 (AutoDataCollector) -> RoboBridge NPZ 변환기.

AutoDataCollector가 수집한 LeRobot v3.0 데이터셋을 RoboBridge의 VLA 학습에
사용할 수 있는 NPZ + JSON 포맷으로 변환합니다.

데이터 흐름:
    collector/ (LeRobot v3.0 Parquet + MP4)
        -> pipeline/convert.py (이 파일)
            -> data/{task_name}/ (NPZ + JSON, bridge/ 학습 스크립트 호환)

Usage:
    # 단일 데이터셋 변환
    python -m pipeline.convert \
        --data-dir ~/.cache/huggingface/lerobot/local/user/pick_redblock \
        --output-dir data/pick_redblock_place_bluedish \
        --task-name pick_redblock_place_bluedish \
        --instruction "pick red block and place on blue dish"

    # 설정 파일로 일괄 변환
    python -m pipeline.convert --config configs/tasks.yaml

    # 변환 후 바로 학습
    python -m pipeline.convert --config configs/tasks.yaml && \
    python bridge/scripts/train/train_lora_movegrip.py \
        --config bridge/configs/groot_so101_move.yaml \
        --data-dir data/pick_redblock_place_bluedish
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers (bridge/scripts/preprocess/preprocess_so101.py 기반)
# ---------------------------------------------------------------------------

def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """RPY (N, 3) -> quaternion (N, 4) [x, y, z, w] (scipy convention)."""
    return Rotation.from_euler("xyz", rpy).as_quat().astype(np.float32)


def compute_goal_direction(ee_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """매 스텝마다 goal_direction = normalize(goal_xyz - ee_xyz)."""
    diff = goal_pos - ee_pos
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (diff / norms).astype(np.float32)


def compute_delta_actions(ee_pos: np.ndarray, ee_rpy: np.ndarray,
                          gripper: np.ndarray) -> np.ndarray:
    """7D delta actions: delta_pos(3) + delta_rot_rpy(3) + gripper(1)."""
    n = len(ee_pos)
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:-1, :3] = ee_pos[1:] - ee_pos[:-1]
    delta_rpy = ee_rpy[1:] - ee_rpy[:-1]
    delta_rpy = (delta_rpy + np.pi) % (2 * np.pi) - np.pi
    actions[:-1, 3:6] = delta_rpy
    actions[:, 6] = gripper
    return actions


def decode_video_frames(video_path: str, image_size: int) -> np.ndarray:
    """MP4 -> (N, H, W, 3) uint8 RGB."""
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
    """모든 parquet chunk를 하나의 DataFrame으로 로드."""
    pfiles = sorted(glob.glob(str(data_dir / "data" / "chunk-*" / "*.parquet")))
    if not pfiles:
        raise FileNotFoundError(f"No parquet files in {data_dir / 'data'}")
    tables = [pq.read_table(f) for f in pfiles]
    return pa.concat_tables(tables).to_pandas()


def load_video_frames(data_dir: Path, camera: str, image_size: int) -> np.ndarray:
    """카메라별 비디오 프레임 로드 및 디코딩."""
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
    """정규화 통계 계산."""
    actions_cat = np.concatenate(all_actions, axis=0)
    states_cat = np.concatenate(all_states, axis=0)

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
        "action_stats": {**stats_for(actions_cat), "mode": "min_max"},
        "state_stats": stats_for(states_cat),
    }


# ---------------------------------------------------------------------------
# 필수 컬럼 검증
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "observation.ee_pos.robot_xyzrpy",
    "observation.gripper_binary",
    "skill.goal_position.robot_xyzrpy",
    "skill.natural_language",
    "skill.type",
    "skill.goal_position.gripper",
    "episode_index",
    "frame_index",
    "index",
]


def validate_columns(df: pd.DataFrame, data_dir: Path) -> None:
    """변환에 필요한 컬럼이 모두 존재하는지 검증."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"데이터셋에 필수 컬럼이 없습니다: {missing}\n"
            f"  경로: {data_dir}\n"
            f"  존재하는 컬럼: {list(df.columns)}\n\n"
            f"AutoDataCollector의 recording_config.yaml에서 다음 설정을 확인하세요:\n"
            f"  observation_features:\n"
            f"    ee_pos.robot_xyzrpy: true\n"
            f"    gripper_binary: true\n"
            f"  skill_features:\n"
            f"    skill.goal_position.robot_xyzrpy: true\n"
            f"    skill.natural_language: true\n"
            f"    skill.type: true\n"
            f"    skill.goal_position.gripper: true"
        )


# ---------------------------------------------------------------------------
# 메인 변환 로직
# ---------------------------------------------------------------------------

def convert_dataset(
    data_dir: Path,
    output_dir: Path,
    task_name: str,
    instruction: str,
    image_size: int = 224,
    top_camera: str = "observation.images.realsense",
    wrist_camera: Optional[str] = "observation.images.innomaker",
) -> dict:
    """하나의 LeRobot v3.0 데이터셋을 RoboBridge NPZ 포맷으로 변환.

    Args:
        data_dir: LeRobot 데이터셋 경로 (meta/, data/, videos/ 포함)
        output_dir: NPZ + JSON 출력 경로
        task_name: 태스크 이름 (파일명에 사용)
        instruction: 자연어 태스크 설명
        image_size: 이미지 리사이즈 크기 (default: 224)
        top_camera: 상부 카메라 키 (default: observation.images.realsense)
        wrist_camera: 손목 카메라 키 (None이면 상부 카메라 복사)

    Returns:
        변환 통계 dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[convert] LeRobot -> RoboBridge NPZ")
    print(f"  Task:   {task_name}")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # 메타데이터 로드
    info_path = data_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    n_episodes = info.get("total_episodes", 0)
    print(f"  FPS: {fps}, Episodes: {n_episodes}")

    # Parquet 로드 및 검증
    df = load_parquet_data(data_dir)
    validate_columns(df, data_dir)
    print(f"  Total rows: {len(df)}")

    # 비디오 프레임 로드
    print(f"  Loading top camera video ({top_camera})...")
    top_frames = load_video_frames(data_dir, top_camera, image_size)
    print(f"  Top camera frames: {top_frames.shape}")

    wrist_frames = None
    if wrist_camera and (data_dir / "videos" / wrist_camera).exists():
        print(f"  Loading wrist camera video ({wrist_camera})...")
        wrist_frames = load_video_frames(data_dir, wrist_camera, image_size)
        print(f"  Wrist camera frames: {wrist_frames.shape}")

    # 에피소드별 처리
    episodes = sorted(df["episode_index"].unique())
    all_actions = []
    all_states = []

    for ep_idx, ep_id in enumerate(episodes):
        ep_df = df[df["episode_index"] == ep_id].sort_values("frame_index").reset_index(drop=True)
        n_frames = len(ep_df)
        demo_num = ep_idx + 1

        # EEF position and rotation
        ee_raw = np.array(ep_df["observation.ee_pos.robot_xyzrpy"].tolist(), dtype=np.float32)
        ee_pos = ee_raw[:, :3]
        ee_rpy = ee_raw[:, 3:6]
        ee_quat = rpy_to_quat(ee_rpy)

        # Gripper binary
        gripper = np.array(ep_df["observation.gripper_binary"].tolist(), dtype=np.float32).flatten()

        # Goal direction
        goal_raw = np.array(ep_df["skill.goal_position.robot_xyzrpy"].tolist(), dtype=np.float32)
        goal_xyz = goal_raw[:, :3]
        goal_direction = compute_goal_direction(ee_pos, goal_xyz)

        # State 11D: ee_pos(3) + ee_quat(4) + gripper(1) + goal_direction(3)
        robot_state = np.concatenate([
            ee_pos,
            ee_quat,
            gripper.reshape(-1, 1),
            goal_direction,
        ], axis=-1).astype(np.float32)

        # Actions 7D: delta_pos(3) + delta_rpy(3) + gripper(1)
        actions = compute_delta_actions(ee_pos, ee_rpy, gripper)

        # Skill annotations
        skills = ep_df["skill.natural_language"].tolist()
        skill_types = ep_df["skill.type"].tolist()

        # Images
        global_indices = ep_df["index"].values
        images = top_frames[global_indices]
        if wrist_frames is not None:
            wrist_imgs = wrist_frames[global_indices]
        else:
            wrist_imgs = images.copy()

        # Save NPZ
        npz_path = output_dir / f"{task_name}_demo{demo_num}.npz"
        np.savez_compressed(
            npz_path,
            images=images,
            wrist_images=wrist_imgs,
            robot_state=robot_state,
            actions=actions,
        )

        # Save JSON metadata
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
            "instruction": instruction,
            "task_name": task_name,
            "demo_id": demo_num,
            "n_steps": n_frames,
            "split": "train",
            "state_format": "ee_pos(3)+ee_quat_xyzw(4)+gripper(1)+goal_direction(3)",
            "action_format": "delta_pos(3)+delta_rpy(3)+gripper_binary(1)",
            "state_dim": 11,
            "action_dim": 7,
            "skill_segments": skill_segments,
            "source": "AutoDataCollector",
            "source_fps": fps,
        }
        json_path = output_dir / f"{task_name}_demo{demo_num}.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_actions.append(actions)
        all_states.append(robot_state)

        grip_changes = np.sum(np.abs(np.diff(gripper)) > 0.5)
        print(f"  demo{demo_num:3d}: {n_frames} steps, "
              f"{len(skill_segments)} skills, grip_changes={grip_changes}")

    # Global stats + metadata.json
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
        "source_system": "AutoDataCollector",
    })

    stats_path = output_dir / "metadata.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Done: {len(episodes)} demos, "
          f"{stats['total_samples']} samples -> {output_dir}")

    return stats


# ---------------------------------------------------------------------------
# 설정 파일 기반 일괄 변환
# ---------------------------------------------------------------------------

def convert_from_config(config_path: str, image_size: int = 224) -> None:
    """YAML 설정 파일에 정의된 태스크들을 일괄 변환.

    configs/tasks.yaml 포맷:
        lerobot_root: ~/.cache/huggingface/lerobot/local
        output_root: data
        tasks:
          - name: pick_redblock_place_bluedish
            dataset: user/pick_redblock
            instruction: "pick red block and place on blue dish"
          - name: stack_blocks
            dataset: user/stack_blocks
            instruction: "stack yellow block on red block"
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lerobot_root = Path(cfg.get("lerobot_root", "~/.cache/huggingface/lerobot/local")).expanduser()
    output_root = Path(cfg.get("output_root", "data"))
    tasks = cfg.get("tasks", [])

    all_stats = {}
    for task_cfg in tasks:
        name = task_cfg["name"]
        dataset = task_cfg["dataset"]
        instruction = task_cfg.get("instruction", name)
        top_cam = task_cfg.get("top_camera", "observation.images.realsense")
        wrist_cam = task_cfg.get("wrist_camera", "observation.images.innomaker")

        data_dir = lerobot_root / dataset
        if not data_dir.exists():
            print(f"SKIP: {data_dir} not found")
            continue

        output_dir = output_root / name
        stats = convert_dataset(
            data_dir=data_dir,
            output_dir=output_dir,
            task_name=name,
            instruction=instruction,
            image_size=image_size,
            top_camera=top_cam,
            wrist_camera=wrist_cam,
        )
        all_stats[name] = {
            "n_demos": stats["total_demos"],
            "n_samples": stats["total_samples"],
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, s in all_stats.items():
        print(f"  {name:50s} {s['n_demos']:3d} demos  {s['n_samples']:6d} samples")
    if all_stats:
        total_demos = sum(s["n_demos"] for s in all_stats.values())
        total_samples = sum(s["n_samples"] for s in all_stats.values())
        print(f"  {'TOTAL':50s} {total_demos:3d} demos  {total_samples:6d} samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AutoDataCollector (LeRobot v3.0) -> RoboBridge (NPZ) 변환기"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="단일 LeRobot 데이터셋 경로")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="NPZ 출력 경로")
    parser.add_argument("--task-name", type=str, default=None,
                        help="태스크 이름")
    parser.add_argument("--instruction", type=str, default="move",
                        help="자연어 태스크 설명")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 설정 파일 경로 (일괄 변환)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="이미지 리사이즈 크기 (default: 224)")
    parser.add_argument("--top-camera", type=str,
                        default="observation.images.realsense",
                        help="상부 카메라 키")
    parser.add_argument("--wrist-camera", type=str,
                        default="observation.images.innomaker",
                        help="손목 카메라 키 (없으면 상부 카메라 복사)")
    args = parser.parse_args()

    if args.config:
        convert_from_config(args.config, args.image_size)
    elif args.data_dir:
        task_name = args.task_name or Path(args.data_dir).name
        output_dir = Path(args.output_dir or f"data/{task_name}")
        convert_dataset(
            data_dir=Path(args.data_dir),
            output_dir=output_dir,
            task_name=task_name,
            instruction=args.instruction,
            image_size=args.image_size,
            top_camera=args.top_camera,
            wrist_camera=args.wrist_camera,
        )
    else:
        parser.error("--data-dir 또는 --config 중 하나를 지정하세요")


if __name__ == "__main__":
    main()
