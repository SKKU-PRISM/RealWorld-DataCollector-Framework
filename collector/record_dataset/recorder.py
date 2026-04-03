"""
DatasetRecorder: LeRobot Dataset v3.0 레코딩 관리 클래스

Forward 실행 중 로봇 상태/액션/이미지를 캡처하여 LeRobot 데이터셋으로 저장합니다.
멀티 카메라 지원: pipeline_config/recording_config.yaml에서 동적으로 카메라 설정 로드

Usage:
    # 단일 카메라 (레거시)
    recorder = DatasetRecorder(repo_id="user/my_dataset", fps=30)
    recorder.record_frame(observation, action, image)

    # 멀티 카메라 (새 방식)
    recorder = DatasetRecorder(repo_id="user/my_dataset", fps=30, config_yaml="path/to/config.yaml")
    recorder.record_frame_multi(observation, action, {"realsense": img1, "innomaker": img2})
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np

# Add lerobot to path
LEROBOT_PATH = Path(__file__).parent.parent / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

from .config import (
    DATASET_FEATURES,
    DEFAULT_FPS,
    ROBOT_TYPE,
    NUM_JOINTS,
    load_cameras_from_yaml,
    load_skill_features_from_yaml,
    load_observation_features_from_yaml,
    build_features_from_yaml,
    CameraConfigRecord,
)


class DatasetRecorder:
    """
    LeRobot Dataset v3.0 레코딩 관리 클래스

    LeRobotDataset API를 래핑하여 간편한 레코딩 인터페이스를 제공합니다.
    멀티 카메라 지원: config_yaml에서 동적으로 카메라 설정 로드

    Args:
        repo_id: 데이터셋 저장 경로 (예: "user/cap_pick_and_place")
        fps: 레코딩 프레임레이트 (기본: 30)
        robot_type: 로봇 타입 (기본: "so101")
        root: 로컬 저장 경로 (None이면 HF_LEROBOT_HOME 사용)
        use_videos: 비디오로 저장할지 여부 (기본: True)
        image_writer_threads: 비동기 이미지 쓰기 스레드 수 (기본: 4)
        config_yaml: 카메라 설정 YAML 경로 (None이면 기본 recording_config.yaml)
        features: 수동으로 features 지정 (None이면 YAML에서 동적 생성)

    Example:
        # 멀티 카메라 (YAML에서 자동 로드)
        >>> recorder = DatasetRecorder("user/my_dataset", fps=30)
        >>> recorder.start_episode("pick up the cube")
        >>> recorder.record_frame_multi(obs, action, {"realsense": img1, "innomaker": img2})
        >>> recorder.end_episode()
        >>> recorder.finalize()
    """

    def __init__(
        self,
        repo_id: str,
        fps: int = DEFAULT_FPS,
        robot_type: str = ROBOT_TYPE,
        root: Optional[str] = None,
        use_videos: bool = True,
        image_writer_threads: int = 4,
        config_yaml: Optional[str] = None,
        features: Optional[Dict] = None,
        resume: bool = False,
        num_robots: int = None,
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.root = Path(root) if root else None
        self.use_videos = use_videos
        self.image_writer_threads = image_writer_threads
        self.config_yaml = config_yaml
        self.resume = resume

        # Features & 카메라 설정: features가 전달되면 그 기준으로, 아니면 YAML에서 동적 생성
        if features is not None:
            self.features = features
            # features에서 카메라 목록 추출 (observation.images.* 키)
            self.camera_names = [
                k.replace("observation.images.", "")
                for k in features if k.startswith("observation.images.")
            ]
            self.camera_configs = [
                cam for cam in load_cameras_from_yaml(config_yaml, num_robots=num_robots)
                if cam.enabled and cam.feature_name in self.camera_names
            ]
            self.enabled_cameras = self.camera_configs
        else:
            self.camera_configs: List[CameraConfigRecord] = load_cameras_from_yaml(config_yaml, num_robots=num_robots)
            self.enabled_cameras = [cam for cam in self.camera_configs if cam.enabled]
            self.camera_names = [cam.feature_name for cam in self.enabled_cameras]
            self.features = build_features_from_yaml(config_yaml, num_robots=num_robots)

        # Skill feature enabled 설정 로드
        self._skill_enabled = load_skill_features_from_yaml(config_yaml)

        # Observation feature enabled 설정 로드
        self._obs_enabled = load_observation_features_from_yaml(config_yaml)

        # Subtask feature enabled 설정 로드
        from record_dataset.config import load_subtask_features_from_yaml
        self._subtask_enabled = load_subtask_features_from_yaml(config_yaml)

        print(f"[DatasetRecorder] Camera features: {[cam.to_feature_key() for cam in self.enabled_cameras]}")
        enabled_skills = [k for k, v in self._skill_enabled.items() if v]
        print(f"[DatasetRecorder] Skill features: {enabled_skills}")
        enabled_obs = [k for k, v in self._obs_enabled.items() if v]
        if enabled_obs:
            print(f"[DatasetRecorder] Observation features: {enabled_obs}")

        # State tracking
        self._dataset = None
        self._is_recording = False
        self._current_task = None
        self._episode_count = 0
        self._frame_count = 0
        self._total_frames = 0

        # Initialize dataset
        self._init_dataset()

    def _init_dataset(self):
        """LeRobotDataset 초기화 (resume=True면 기존 데이터셋에 append)"""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

            dataset_path = self.root if self.root else HF_LEROBOT_HOME / self.repo_id

            if self.resume and dataset_path.exists():
                # Resume: 기존 데이터셋을 열어서 에피소드 append
                print(f"[DatasetRecorder] Resume mode: opening existing dataset")
                print(f"  Path: {dataset_path}")
                self._dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=self.root,
                )
                self._episode_count = self._dataset.num_episodes
                print(f"[DatasetRecorder] Resumed: {self._episode_count} existing episodes")
            elif dataset_path.exists():
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Dataset already exists!\n"
                    f"========================================\n"
                    f"Path: {dataset_path}\n"
                    f"\n"
                    f"To continue, either:\n"
                    f"  1. Delete the existing dataset:\n"
                    f"     rm -rf {dataset_path}\n"
                    f"  2. Use a different repo_id\n"
                    f"========================================"
                )
            else:
                # 새 데이터셋 생성
                print(f"[DatasetRecorder] Creating dataset: {self.repo_id}")
                print(f"  FPS: {self.fps}")
                print(f"  Robot type: {self.robot_type}")
                print(f"  Features: {list(self.features.keys())}")

                self._dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    fps=self.fps,
                    robot_type=self.robot_type,
                    features=self.features,
                    root=self.root,
                    use_videos=self.use_videos,
                    image_writer_threads=self.image_writer_threads,
                )

            print(f"[DatasetRecorder] Dataset at: {self._dataset.root}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import lerobot. Make sure lerobot is installed or "
                f"available in {LEROBOT_PATH}. Error: {e}"
            )
        except AssertionError:
            # Re-raise AssertionError to terminate the pipeline
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to create dataset: {e}")

    @property
    def dataset(self):
        """내부 LeRobotDataset 인스턴스 접근"""
        return self._dataset

    @property
    def is_recording(self) -> bool:
        """현재 에피소드 레코딩 중인지 여부"""
        return self._is_recording

    @property
    def episode_count(self) -> int:
        """저장된 에피소드 수"""
        return self._episode_count

    @property
    def current_frame_count(self) -> int:
        """현재 에피소드의 프레임 수"""
        return self._frame_count

    @property
    def total_frames(self) -> int:
        """전체 프레임 수"""
        return self._total_frames

    def start_episode(self, task: str) -> None:
        """
        새 에피소드 레코딩 시작

        Args:
            task: 태스크 설명 (자연어)

        Raises:
            RuntimeError: 이미 레코딩 중인 경우
        """
        if self._is_recording:
            raise RuntimeError(
                "Already recording an episode. Call end_episode() first."
            )

        self._current_task = task
        self._is_recording = True
        self._frame_count = 0

        print(f"\n[DatasetRecorder] Started episode {self._episode_count}")
        print(f"  Task: {task}")

    def record_frame(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        image: np.ndarray,
    ) -> None:
        """
        단일 프레임 레코딩 (레거시 호환 - 단일 카메라)

        첫 번째 활성화된 카메라로 이미지를 저장합니다.

        Args:
            observation: 현재 로봇 상태 (shape: (6,), normalized)
            action: 타겟 액션 (shape: (6,), normalized)
            image: 카메라 이미지 (shape: (H, W, 3), RGB, uint8)
        """
        if not self.enabled_cameras:
            raise RuntimeError("No cameras configured")

        # 첫 번째 카메라 이름으로 이미지 전달
        images = {self.enabled_cameras[0].name: image}
        self.record_frame_multi(observation, action, images)

    def record_frame_multi(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        images: Dict[str, np.ndarray],
        skill_label: Optional[str] = None,
        observation_extras: Optional[Dict[str, np.ndarray]] = None,
        subtask_info: Optional[Dict] = None,
    ) -> None:
        """
        멀티 카메라 프레임 레코딩 + 스킬 라벨 + observation extras

        Args:
            observation: 현재 로봇 상태 (shape: (6,), normalized)
            action: 타겟 액션 (shape: (6,), normalized)
            images: 카메라별 이미지 딕셔너리
                    {camera_name: image} 형태
                    예: {"realsense": img1, "innomaker": img2}
            skill_label: 현재 실행 중인 스킬의 자연어 설명 (optional)
                        예: "move to blue dish", "pick yellow dice"
            observation_extras: FK 기반 observation features (optional)
                    {feature_key: np.ndarray} 형태
                    예: {"observation.ee_pos.robot_xyzrpy": array([x,y,z,r,p,y])}

        Note:
            - timestamp는 LeRobotDataset이 frame index와 FPS 기반으로 자동 계산
            - 모든 활성화된 카메라의 이미지가 필요합니다

        Raises:
            RuntimeError: 레코딩 중이 아닌 경우
            ValueError: 잘못된 데이터 shape 또는 누락된 카메라 이미지
        """
        if not self._is_recording:
            raise RuntimeError(
                "Not recording. Call start_episode() first."
            )

        # Validate shapes
        if observation.shape != (NUM_JOINTS,):
            raise ValueError(
                f"observation shape must be ({NUM_JOINTS},), got {observation.shape}"
            )
        if action.shape != (NUM_JOINTS,):
            raise ValueError(
                f"action shape must be ({NUM_JOINTS},), got {action.shape}"
            )

        # Ensure correct dtypes
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        # Build frame dict
        frame = {
            "observation.state": observation,
            "action": action,
            "task": self._current_task,
        }

        # Skill-level subgoal info (RecordingContext에서 가져옴, enabled인 것만 추가)
        try:
            from .context import RecordingContext
            skill_info = RecordingContext.get_skill_info(current_state=observation)
            skill_data = {
                "skill.natural_language": skill_info["label"],
                "skill.type": skill_info["type"],
                "skill.verification_question": skill_info["verification_question"],
                "skill.progress": np.array([skill_info["progress"]], dtype=np.float32),
                "skill.goal_position.joint": skill_info["goal_joint"],
                "skill.goal_position.robot_xyzrpy": skill_info["goal_robot_xyzrpy"],
                "skill.goal_position.gripper": np.array([skill_info["goal_gripper"]], dtype=np.float32),
            }
        except ImportError:
            skill_data = {
                "skill.natural_language": skill_label if skill_label else "",
                "skill.type": "",
                "skill.verification_question": "",
                "skill.progress": np.array([0.0], dtype=np.float32),
                "skill.goal_position.joint": np.zeros(6, dtype=np.float32),
                "skill.goal_position.robot_xyzrpy": np.zeros(6, dtype=np.float32),
                "skill.goal_position.gripper": np.array([0.0], dtype=np.float32),
            }

        # enabled인 skill feature만 frame에 추가
        for key, value in skill_data.items():
            if self._skill_enabled.get(key, True):
                frame[key] = value

        # Observation extras (FK 기반 EE 자세 등, enabled인 것만 추가)
        if observation_extras:
            for key, value in observation_extras.items():
                if self._obs_enabled.get(key, False):
                    frame[key] = value

        # Sub-task info (상위 계층 라벨, 없으면 기본값으로 기록)
        if self._subtask_enabled.get("subtask.natural_language", False):
            frame["subtask.natural_language"] = subtask_info.get("natural_language", "") if subtask_info else ""
        if self._subtask_enabled.get("subtask.object_name", False):
            frame["subtask.object_name"] = subtask_info.get("object_name", "") if subtask_info else ""
        if self._subtask_enabled.get("subtask.target_position", False):
            frame["subtask.target_position"] = np.asarray(
                subtask_info.get("target_position", [0.0, 0.0, 0.0]) if subtask_info else [0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        # 각 카메라 이미지 추가
        for cam in self.enabled_cameras:
            cam_name = cam.feature_name  # "top", "left_wrist", etc. (matches camera_manager key)
            feature_key = cam.to_feature_key()  # "observation.images.{feature_name}"

            if cam_name in images:
                img = images[cam_name]
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(
                        f"Image for '{cam_name}' must be (H, W, 3) RGB, got shape {img.shape}"
                    )
                frame[feature_key] = np.asarray(img, dtype=np.uint8)
            else:
                # 이미지가 없으면 더미 이미지 사용 (경고 출력)
                print(f"[DatasetRecorder] Warning: No image for camera '{cam_name}', using dummy")
                frame[feature_key] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

        # Add frame to dataset
        self._dataset.add_frame(frame)
        self._frame_count += 1

    def add_frame(self, frame: Dict[str, Any]) -> None:
        """Add a pre-built frame dict directly to the dataset.

        Used by MultiArmRecorder which builds its own 12-axis frames.
        Skips shape validation since multi-arm uses different shapes.
        """
        if not self._is_recording:
            raise RuntimeError("Not recording. Call start_episode() first.")
        self._dataset.add_frame(frame)
        self._frame_count += 1

    def end_episode(self, discard: bool = False) -> Dict[str, Any]:
        """
        현재 에피소드 종료 및 저장

        Args:
            discard: True면 에피소드 저장하지 않고 버림

        Returns:
            에피소드 정보 딕셔너리

        Raises:
            RuntimeError: 레코딩 중이 아닌 경우
        """
        if not self._is_recording:
            raise RuntimeError("Not recording any episode.")

        episode_info = {
            "episode_index": self._episode_count,
            "task": self._current_task,
            "num_frames": self._frame_count,
            "discarded": discard,
        }

        if discard:
            print(f"[DatasetRecorder] Discarding episode {self._episode_count}")
            self._dataset.clear_episode_buffer()
        else:
            if self._frame_count > 0:
                print(f"[DatasetRecorder] Saving episode {self._episode_count}")
                print(f"  Frames: {self._frame_count}")
                self._dataset.save_episode()
                self._episode_count += 1
                self._total_frames += self._frame_count
            else:
                print(f"[DatasetRecorder] Skipping empty episode")

        self._is_recording = False
        self._current_task = None
        self._frame_count = 0

        return episode_info

    def finalize(self) -> None:
        """
        데이터셋 완료 처리 (필수!)

        Parquet writer를 닫고 메타데이터를 완성합니다.
        이 함수를 호출하지 않으면 데이터셋이 손상될 수 있습니다.
        """
        if self._is_recording:
            print("[DatasetRecorder] Warning: Ending incomplete episode before finalize")
            self.end_episode(discard=True)

        print(f"\n[DatasetRecorder] Finalizing dataset...")
        print(f"  Total episodes: {self._episode_count}")
        print(f"  Total frames: {self._total_frames}")

        self._dataset.finalize()
        print(f"[DatasetRecorder] Dataset finalized at: {self._dataset.root}")

    def push_to_hub(
        self,
        tags: Optional[list] = None,
        private: bool = False,
        license: str = "apache-2.0",
    ) -> None:
        """
        데이터셋을 HuggingFace Hub에 업로드

        Args:
            tags: 데이터셋 태그 리스트
            private: 비공개 데이터셋 여부
            license: 라이선스 (기본: apache-2.0)
        """
        if self._is_recording:
            raise RuntimeError(
                "Cannot push while recording. Call end_episode() and finalize() first."
            )

        default_tags = ["LeRobot", "robotics", "so101", "code-as-policies"]
        all_tags = list(set(default_tags + (tags or [])))

        print(f"\n[DatasetRecorder] Pushing to Hub: {self.repo_id}")
        self._dataset.push_to_hub(
            tags=all_tags,
            private=private,
            license=license,
        )
        print(f"[DatasetRecorder] Uploaded successfully!")
        print(f"  URL: https://huggingface.co/datasets/{self.repo_id}")

    def get_stats(self) -> Dict[str, Any]:
        """현재 레코딩 통계 반환"""
        return {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "total_episodes": self._episode_count,
            "total_frames": self._total_frames,
            "is_recording": self._is_recording,
            "current_episode_frames": self._frame_count if self._is_recording else 0,
            "root": str(self._dataset.root) if self._dataset else None,
        }

    def __repr__(self) -> str:
        status = "recording" if self._is_recording else "idle"
        return (
            f"DatasetRecorder(\n"
            f"  repo_id='{self.repo_id}',\n"
            f"  fps={self.fps},\n"
            f"  episodes={self._episode_count},\n"
            f"  frames={self._total_frames},\n"
            f"  status='{status}'\n"
            f")"
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto finalize"""
        if self._is_recording:
            self.end_episode(discard=exc_type is not None)
        self.finalize()
        return False
