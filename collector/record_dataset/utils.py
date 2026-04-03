"""
Utility functions for LeRobot Dataset Recording
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np


def generate_repo_id(
    username: str = "local",
    task_name: str = "cap_dataset",
    timestamp: bool = True,
) -> str:
    """
    데이터셋 repo_id 생성

    Args:
        username: HuggingFace 사용자명 또는 "local"
        task_name: 태스크 이름
        timestamp: 타임스탬프 추가 여부

    Returns:
        repo_id (예: "local/cap_pick_and_place_20240101_120000")
    """
    base = f"{username}/{task_name}"
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"
    return base


def normalize_joint_positions(
    positions: np.ndarray,
    source_range: tuple = (-100.0, 100.0),
    target_range: tuple = (-1.0, 1.0),
) -> np.ndarray:
    """
    조인트 위치 정규화 범위 변환

    Args:
        positions: 입력 조인트 위치
        source_range: 입력 범위 (min, max)
        target_range: 출력 범위 (min, max)

    Returns:
        변환된 조인트 위치
    """
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range

    # Normalize to [0, 1]
    normalized = (positions - src_min) / (src_max - src_min)
    # Scale to target range
    return normalized * (tgt_max - tgt_min) + tgt_min


def validate_frame_data(
    observation: np.ndarray,
    action: np.ndarray,
    image: np.ndarray,
    num_joints: int = 6,
) -> Dict[str, Any]:
    """
    프레임 데이터 유효성 검사

    Args:
        observation: 관측값
        action: 액션
        image: 이미지
        num_joints: 조인트 수

    Returns:
        검증 결과 딕셔너리
    """
    errors = []
    warnings = []

    # Observation 검사
    if observation.shape != (num_joints,):
        errors.append(f"observation shape mismatch: expected ({num_joints},), got {observation.shape}")

    if not np.isfinite(observation).all():
        errors.append("observation contains NaN or Inf values")

    # Action 검사
    if action.shape != (num_joints,):
        errors.append(f"action shape mismatch: expected ({num_joints},), got {action.shape}")

    if not np.isfinite(action).all():
        errors.append("action contains NaN or Inf values")

    # Image 검사
    if len(image.shape) != 3:
        errors.append(f"image must be 3D (H, W, C), got {len(image.shape)}D")
    elif image.shape[2] not in [1, 3, 4]:
        warnings.append(f"unexpected image channels: {image.shape[2]}")

    if image.dtype != np.uint8:
        warnings.append(f"image dtype should be uint8, got {image.dtype}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def estimate_dataset_size(
    num_episodes: int,
    frames_per_episode: int,
    image_shape: tuple = (480, 640, 3),
    num_joints: int = 6,
    video_compression_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    예상 데이터셋 크기 계산

    Args:
        num_episodes: 에피소드 수
        frames_per_episode: 에피소드당 프레임 수
        image_shape: 이미지 shape (H, W, C)
        num_joints: 조인트 수
        video_compression_ratio: 비디오 압축률 (원본 대비)

    Returns:
        크기 추정 딕셔너리 (MB 단위)
    """
    total_frames = num_episodes * frames_per_episode

    # Parquet data (state + action)
    # float32 = 4 bytes per value
    parquet_size_bytes = total_frames * num_joints * 2 * 4  # observation + action
    parquet_size_mb = parquet_size_bytes / (1024 * 1024)

    # Video data
    h, w, c = image_shape
    raw_image_bytes = h * w * c
    raw_video_bytes = total_frames * raw_image_bytes
    compressed_video_bytes = raw_video_bytes * video_compression_ratio
    video_size_mb = compressed_video_bytes / (1024 * 1024)

    # Metadata (rough estimate)
    metadata_size_mb = 0.1 * num_episodes

    total_mb = parquet_size_mb + video_size_mb + metadata_size_mb

    return {
        "parquet_mb": parquet_size_mb,
        "video_mb": video_size_mb,
        "metadata_mb": metadata_size_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }


def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_recording_summary(stats: Dict[str, Any]) -> None:
    """레코딩 요약 출력"""
    print("\n" + "=" * 60)
    print("Recording Summary")
    print("=" * 60)
    print(f"  Repository: {stats.get('repo_id', 'N/A')}")
    print(f"  Total Episodes: {stats.get('total_episodes', 0)}")
    print(f"  Total Frames: {stats.get('total_frames', 0)}")
    print(f"  FPS: {stats.get('fps', 'N/A')}")
    print(f"  Root Path: {stats.get('root', 'N/A')}")

    if stats.get('total_frames', 0) > 0 and stats.get('fps', 0) > 0:
        duration = stats['total_frames'] / stats['fps']
        print(f"  Total Duration: {format_duration(duration)}")

    print("=" * 60 + "\n")


class DummyCamera:
    """
    테스트용 더미 카메라

    실제 카메라 없이 레코딩 파이프라인을 테스트할 때 사용합니다.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self._frame_count = 0

    def get_frames(self) -> tuple:
        """더미 프레임 반환 (color, depth)"""
        # 프레임 번호를 이미지에 인코딩 (디버깅용)
        color = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        # 간단한 패턴 생성
        x = self._frame_count % self.width
        color[:, x:x+5, :] = 255  # 수직 흰색 선

        self._frame_count += 1

        # depth는 None 반환
        return color, None

    def start(self) -> None:
        """카메라 시작 (더미)"""
        self._frame_count = 0

    def stop(self) -> None:
        """카메라 정지 (더미)"""
        pass


def check_lerobot_installation() -> Dict[str, Any]:
    """
    lerobot 설치 상태 확인

    Returns:
        설치 상태 정보
    """
    result = {
        "installed": False,
        "version": None,
        "path": None,
        "error": None,
    }

    try:
        # Local lerobot path
        lerobot_path = Path(__file__).parent.parent / "lerobot" / "src"
        if lerobot_path.exists():
            sys.path.insert(0, str(lerobot_path))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot import __version__

        result["installed"] = True
        result["version"] = __version__
        result["path"] = str(lerobot_path)

    except ImportError as e:
        result["error"] = str(e)

    return result
