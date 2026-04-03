"""
SkillsRecordingWrapper: LeRobotSkills를 래핑하여 레코딩 기능 추가

기존 LeRobotSkills API를 유지하면서 실행 중 데이터를 자동으로 레코딩합니다.

두 가지 레코딩 모드:
1. Threaded Mode: 별도 스레드에서 고정 FPS로 연속 캡처 (기본)
2. Sync Mode: 각 스킬 호출 시작/종료 시점에만 캡처

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from record_dataset import DatasetRecorder, SkillsRecordingWrapper

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
    skills.connect()

    recorder = DatasetRecorder("user/my_dataset", fps=30)
    camera = RealSenseD435()
    camera.start()

    wrapped = SkillsRecordingWrapper(skills, recorder, camera)

    recorder.start_episode("pick up the cube")
    wrapped.move_to_initial_state()
    wrapped.move_to_position([0.2, 0.0, 0.05])
    wrapped.gripper_close()
    recorder.end_episode()
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Any, List, Union, Callable
from functools import wraps

import numpy as np

# Add src to path for LeRobotSkills
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .recorder import DatasetRecorder
from .callback import RecordingCallback
from .config import DEFAULT_FPS, NUM_JOINTS


class SkillsRecordingWrapper:
    """
    LeRobotSkills 래퍼 - 레코딩 기능 추가

    기존 LeRobotSkills의 모든 메서드를 프록시하면서,
    실행 중 로봇 상태/액션/이미지를 자동으로 레코딩합니다.

    Args:
        skills: LeRobotSkills 인스턴스
        recorder: DatasetRecorder 인스턴스
        camera: 카메라 인스턴스 (get_frames() 메서드 필요)
        recording_fps: 레코딩 FPS (기본: 30)
        threaded: True면 별도 스레드에서 연속 캡처

    Example:
        >>> skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
        >>> skills.connect()
        >>> recorder = DatasetRecorder("user/dataset", fps=30)
        >>> wrapped = SkillsRecordingWrapper(skills, recorder, camera)
        >>> recorder.start_episode("pick up cube")
        >>> wrapped.move_to_position([0.2, 0, 0.05])
        >>> recorder.end_episode()
    """

    def __init__(
        self,
        skills: Any,  # LeRobotSkills
        recorder: DatasetRecorder,
        camera: Any,
        recording_fps: int = DEFAULT_FPS,
        threaded: bool = True,
    ):
        self._skills = skills
        self._recorder = recorder
        self._camera = camera
        self._recording_fps = recording_fps
        self._threaded = threaded

        # Threading
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_recording = threading.Event()
        self._is_executing = threading.Event()

        # Last recorded values (for action tracking)
        self._last_target_action: Optional[np.ndarray] = None
        self._current_gripper_pos: float = 0.0

        # Statistics
        self._total_recorded_frames = 0

    # =========================================================================
    # Recording Control
    # =========================================================================

    def _start_recording_thread(self) -> None:
        """레코딩 스레드 시작"""
        if not self._threaded:
            return

        if self._recording_thread is not None and self._recording_thread.is_alive():
            return

        self._stop_recording.clear()
        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True,
        )
        self._recording_thread.start()

    def _stop_recording_thread(self) -> None:
        """레코딩 스레드 종료"""
        if self._recording_thread is None:
            return

        self._stop_recording.set()
        if self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        self._recording_thread = None

    def _recording_loop(self) -> None:
        """
        레코딩 스레드 메인 루프

        고정 FPS로 연속 캡처합니다.
        _is_executing이 설정된 동안에만 레코딩합니다.
        """
        interval = 1.0 / self._recording_fps
        last_time = time.time()

        while not self._stop_recording.is_set():
            current_time = time.time()
            elapsed = current_time - last_time

            # FPS 유지
            if elapsed < interval:
                time.sleep(interval - elapsed)
                continue

            last_time = current_time

            # 실행 중이고 레코딩 중일 때만 캡처
            if self._is_executing.is_set() and self._recorder.is_recording:
                self._capture_frame()

    def _capture_frame(self) -> bool:
        """현재 프레임 캡처 및 레코딩"""
        try:
            # 현재 로봇 상태 읽기
            if not hasattr(self._skills, 'robot') or self._skills.robot is None:
                return False

            pos_all = self._skills.robot.read_positions(normalize=True)
            current_state = pos_all  # 6 joints (5 arm + 1 gripper)

            # 액션: 마지막 목표 또는 현재 상태
            if self._last_target_action is not None:
                action = self._last_target_action.copy()
            else:
                action = current_state.copy()

            # 그리퍼 상태 업데이트
            if len(current_state) >= 6:
                self._current_gripper_pos = current_state[5]

            # 카메라 이미지
            image = self._capture_image()
            if image is None:
                return False

            # 레코딩
            self._recorder.record_frame(
                observation=current_state,
                action=action,
                image=image,
            )
            self._total_recorded_frames += 1
            return True

        except Exception as e:
            print(f"[SkillsRecordingWrapper] Capture error: {e}")
            return False

    def _capture_image(self) -> Optional[np.ndarray]:
        """카메라 이미지 캡처"""
        if self._camera is None:
            # 더미 이미지 반환
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            if hasattr(self._camera, 'get_frames'):
                color, _ = self._camera.get_frames()
                return color
            elif hasattr(self._camera, 'read'):
                ret, frame = self._camera.read()
                if ret:
                    import cv2
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"[SkillsRecordingWrapper] Camera error: {e}")
            return None

    # =========================================================================
    # Execution Wrapper Decorator
    # =========================================================================

    def _wrap_execution(self, method_name: str):
        """스킬 메서드를 래핑하여 레코딩 활성화"""
        original_method = getattr(self._skills, method_name)

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            # 레코딩 스레드 시작 (필요시)
            if self._recorder.is_recording:
                self._start_recording_thread()
                self._is_executing.set()

            try:
                result = original_method(*args, **kwargs)
                return result
            finally:
                # 실행 종료
                self._is_executing.clear()
                # 약간의 settle time 동안 추가 프레임 캡처
                time.sleep(0.05)

        return wrapped

    # =========================================================================
    # Proxied Skill Methods
    # =========================================================================

    def move_to_initial_state(self, duration: Optional[float] = None) -> bool:
        """초기 상태로 이동"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            # 초기 상태를 타겟 액션으로 설정
            if self._skills.initial_state is not None:
                full_action = np.concatenate([
                    self._skills.initial_state,
                    [self._skills.initial_state_gripper]
                ])
                self._last_target_action = full_action.astype(np.float32)

            return self._skills.move_to_initial_state(duration)
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    def move_to_position(
        self,
        position: Union[List[float], np.ndarray],
        duration: Optional[float] = None,
        maintain_wrist_roll: bool = True,
        maintain_pitch: bool = False,
    ) -> bool:
        """지정 위치로 이동"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            result = self._skills.move_to_position(
                position=position,
                duration=duration,
                maintain_wrist_roll=maintain_wrist_roll,
                maintain_pitch=maintain_pitch,
            )

            # 이동 후 현재 상태를 타겟으로 업데이트
            if hasattr(self._skills, 'robot') and self._skills.robot:
                pos_all = self._skills.robot.read_positions(normalize=True)
                self._last_target_action = pos_all.astype(np.float32)

            return result
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    def gripper_open(self, duration: float = 1.0) -> None:
        """그리퍼 열기 (recording은 _set_gripper_position 내부에서 처리)"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            self._skills.gripper_open(duration=duration)

            # 그리퍼 액션 업데이트
            if self._last_target_action is not None and len(self._last_target_action) >= 6:
                self._last_target_action[5] = self._skills.gripper_open_pos
            self._current_gripper_pos = self._skills.gripper_open_pos
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    def gripper_close(self, duration: float = 1.0) -> None:
        """그리퍼 닫기 (recording은 _set_gripper_position 내부에서 처리)"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            self._skills.gripper_close(duration=duration)

            # 그리퍼 액션 업데이트
            if self._last_target_action is not None and len(self._last_target_action) >= 6:
                self._last_target_action[5] = self._skills.gripper_close_pos
            self._current_gripper_pos = self._skills.gripper_close_pos
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    def rotate_90degree(self, direction: int = 1, duration: float = 2.0) -> bool:
        """그리퍼 90도 회전"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            result = self._skills.rotate_90degree(direction=direction, duration=duration)

            # 회전 후 상태 업데이트
            if hasattr(self._skills, 'robot') and self._skills.robot:
                pos_all = self._skills.robot.read_positions(normalize=True)
                self._last_target_action = pos_all.astype(np.float32)

            return result
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    def move_to_free_state(self, duration: Optional[float] = None) -> bool:
        """프리 상태로 이동"""
        if self._recorder.is_recording:
            self._start_recording_thread()
            self._is_executing.set()

        try:
            if self._skills.free_state is not None:
                full_action = np.concatenate([
                    self._skills.free_state,
                    [self._skills.free_state_gripper]
                ])
                self._last_target_action = full_action.astype(np.float32)

            return self._skills.move_to_free_state(duration)
        finally:
            self._is_executing.clear()
            time.sleep(0.05)

    # =========================================================================
    # Pass-through Properties and Methods
    # =========================================================================

    @property
    def skills(self):
        """원본 LeRobotSkills 접근"""
        return self._skills

    @property
    def recorder(self):
        """DatasetRecorder 접근"""
        return self._recorder

    @property
    def is_connected(self) -> bool:
        return self._skills.is_connected

    def connect(self) -> bool:
        return self._skills.connect()

    def disconnect(self) -> None:
        self._stop_recording_thread()
        self._skills.disconnect()

    def get_current_ee_position(self) -> np.ndarray:
        return self._skills.get_current_ee_position()

    # =========================================================================
    # Attribute Proxy
    # =========================================================================

    def __getattr__(self, name: str):
        """정의되지 않은 속성/메서드는 원본 skills로 전달"""
        return getattr(self._skills, name)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_recording_stats(self) -> dict:
        """레코딩 통계"""
        return {
            "total_recorded_frames": self._total_recorded_frames,
            "recording_fps": self._recording_fps,
            "threaded_mode": self._threaded,
            "is_executing": self._is_executing.is_set(),
            "recorder_stats": self._recorder.get_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"SkillsRecordingWrapper(\n"
            f"  skills={type(self._skills).__name__},\n"
            f"  recorder={self._recorder.repo_id},\n"
            f"  fps={self._recording_fps},\n"
            f"  recorded_frames={self._total_recorded_frames}\n"
            f")"
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_recording_thread()
        return False


def create_recording_skills(
    robot_config: str = "robot_configs/robot/so101_robot3.yaml",
    repo_id: str = "local/cap_dataset",
    fps: int = DEFAULT_FPS,
    camera: Any = None,
    **skills_kwargs,
) -> tuple:
    """
    레코딩 가능한 Skills 인스턴스 생성 헬퍼

    Args:
        robot_config: 로봇 설정 파일 경로
        repo_id: 데이터셋 저장 경로
        fps: 레코딩 FPS
        camera: 카메라 인스턴스
        **skills_kwargs: LeRobotSkills 추가 인자

    Returns:
        (wrapped_skills, recorder) 튜플
    """
    from skills.skills_lerobot import LeRobotSkills

    # Skills 초기화
    skills = LeRobotSkills(robot_config=robot_config, **skills_kwargs)

    # Recorder 초기화
    recorder = DatasetRecorder(repo_id=repo_id, fps=fps)

    # Wrapper 생성
    wrapped = SkillsRecordingWrapper(
        skills=skills,
        recorder=recorder,
        camera=camera,
        recording_fps=fps,
    )

    return wrapped, recorder
