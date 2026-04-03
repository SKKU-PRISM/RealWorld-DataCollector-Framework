"""
LeRobot Dataset Recording Pipeline

Forward 과정에서 발생하는 로봇 움직임을 LeRobot Dataset v3.0 포맷으로 레코딩합니다.

멀티 카메라 지원:
- pipeline_config/recording_config.yaml에서 카메라 설정 동적 로드
- Intel RealSense + Innomaker 등 여러 카메라 동시 지원
- async_read 방식으로 LeRobot 공식 구현과 동일한 캡처

Usage:
    # 멀티 카메라 (권장)
    from record_dataset import create_recording_setup

    camera_manager, recorder, callback = create_recording_setup(
        repo_id="local/my_dataset",
        config_yaml="pipeline_config/recording_config.yaml",
    )

    camera_manager.connect_all()
    recorder.start_episode(task="pick up the object")
    # control loop with callback.on_step(state, action)
    recorder.end_episode()
    camera_manager.disconnect_all()
    recorder.finalize()

    # 레거시 단일 카메라
    from record_dataset import DatasetRecorder, SkillsRecordingWrapper

    recorder = DatasetRecorder(repo_id="user/my_dataset", fps=30)
    wrapped_skills = SkillsRecordingWrapper(skills, recorder, camera)

    recorder.start_episode(task="pick up the object")
    wrapped_skills.move_to_position([0.2, 0.0, 0.05])
    recorder.end_episode()
    recorder.finalize()
"""

from .recorder import DatasetRecorder
from .callback import RecordingCallback, SimpleRecordingCallback, create_recording_setup
from .skills_wrapper import SkillsRecordingWrapper
from .context import RecordingContext
from .config import (
    DATASET_FEATURES,
    DEFAULT_FPS,
    ROBOT_TYPE,
    load_cameras_from_yaml,
    create_camera_manager_from_config,
    build_features_from_yaml,
)

__all__ = [
    # Core classes
    "DatasetRecorder",
    "RecordingCallback",
    "SimpleRecordingCallback",
    "RecordingContext",
    "SkillsRecordingWrapper",
    # Factory function
    "create_recording_setup",
    # Config utilities
    "DATASET_FEATURES",
    "DEFAULT_FPS",
    "ROBOT_TYPE",
    "load_cameras_from_yaml",
    "create_camera_manager_from_config",
    "build_features_from_yaml",
]
