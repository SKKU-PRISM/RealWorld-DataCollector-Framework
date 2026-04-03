"""
RecordingCallback: 제어 루프에 삽입될 레코딩 콜백

Skills의 _execute_planned_trajectory() 등의 제어 루프에서
매 스텝마다 호출되어 데이터를 캡처합니다.

FPS 동기화: 50Hz 제어 → 30Hz 레코딩

통합 카메라 지원:
- 공식 LeRobot 방식과 동일하게 MultiCameraManager만 사용
- 카메라 수와 관계없이 동일한 코드 경로
- async_read_all()로 모든 카메라에서 동시 캡처
"""

import time
from typing import Optional, Any, Dict

import numpy as np

from .config import DEFAULT_FPS, NUM_JOINTS

# Legacy: CONTROL_HZ removed from config, default to DEFAULT_FPS
CONTROL_HZ = DEFAULT_FPS

# cameras 모듈에서 MultiCameraManager import
try:
    import sys
    from pathlib import Path
    cameras_path = Path(__file__).parent.parent / "cameras"
    if str(cameras_path.parent) not in sys.path:
        sys.path.insert(0, str(cameras_path.parent))
    from cameras import MultiCameraManager
    HAS_MULTI_CAMERA = True
except ImportError:
    HAS_MULTI_CAMERA = False
    MultiCameraManager = None


class RecordingCallback:
    """
    제어 루프용 레코딩 콜백 (통합 방식)

    Skills의 내부 제어 루프에서 매 스텝마다 호출되어
    상태/액션/이미지를 캡처하고 FPS 동기화를 수행합니다.

    통합 카메라 지원 (공식 LeRobot 방식):
    - MultiCameraManager만 사용 (단일/다중 카메라 구분 없음)
    - 카메라 수와 관계없이 동일한 코드 경로

    Args:
        recorder: DatasetRecorder 인스턴스
        camera_manager: MultiCameraManager 인스턴스
        target_fps: 목표 레코딩 FPS (기본: 30)
        control_hz: 제어 루프 주파수 (기본: 50)

    Example:
        from record_dataset.config import create_camera_manager_from_config

        camera_manager = create_camera_manager_from_config()
        camera_manager.connect_all()
        callback = RecordingCallback(recorder, camera_manager, target_fps=30)

        while executing:
            state = robot.read_positions()
            action = trajectory.get_target()

            if callback.should_record():
                callback.on_step(state, action)

            robot.write_positions(action)
            time.sleep(0.02)  # 50Hz
    """

    def __init__(
        self,
        recorder: "DatasetRecorder",
        camera_manager: Any,
        target_fps: int = DEFAULT_FPS,
        control_hz: int = CONTROL_HZ,
    ):
        # Lazy import to avoid circular dependency
        from .recorder import DatasetRecorder

        self.recorder = recorder
        self.camera_manager = camera_manager
        self.target_fps = target_fps
        self.control_hz = control_hz

        # FPS 동기화 계산
        self.frame_skip_ratio = control_hz / target_fps
        self._step_counter = 0
        self._last_record_step = -1
        self._accumulated_time = 0.0

        # 타이밍
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None

        # 통계
        self._recorded_frames = 0
        self._skipped_frames = 0
        self._camera_errors = 0

        if self.camera_manager is not None and hasattr(self.camera_manager, 'camera_names'):
            print(f"[RecordingCallback] Cameras: {self.camera_manager.camera_names}")
        else:
            print("[RecordingCallback] No camera manager configured (dummy mode)")

    def reset(self) -> None:
        """콜백 상태 리셋 (새 에피소드 시작 시)"""
        self._step_counter = 0
        self._last_record_step = -1
        self._accumulated_time = 0.0
        self._start_time = time.time()
        self._last_step_time = self._start_time
        self._recorded_frames = 0
        self._skipped_frames = 0

    def should_record(self) -> bool:
        """
        현재 스텝에서 레코딩해야 하는지 판단

        FPS 동기화를 위해 프레임을 스킵합니다.
        50Hz 제어 → 30Hz 레코딩: 약 1.67 스텝마다 레코딩

        Returns:
            레코딩 여부
        """
        # 아직 레코딩 중이 아니면 False
        if not self.recorder.is_recording:
            return False

        # FPS 동기화: accumulated steps 기반
        target_frame = int(self._step_counter / self.frame_skip_ratio)
        should = target_frame > self._last_record_step

        if not should:
            self._skipped_frames += 1

        return should

    def on_step(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        gripper_state: Optional[float] = None,
        gripper_action: Optional[float] = None,
    ) -> bool:
        """
        제어 스텝에서 호출 - 데이터 레코딩 (통합 방식)

        Args:
            current_state: 현재 관절 위치 (5 arm joints, normalized)
            action: 타겟 관절 위치 (5 arm joints, normalized)
            gripper_state: 현재 그리퍼 위치 (normalized, optional)
            gripper_action: 타겟 그리퍼 위치 (normalized, optional)

        Returns:
            레코딩 성공 여부
        """
        if not self.recorder.is_recording:
            return False

        try:
            # 상태 및 액션 구성 (6 joints: 5 arm + 1 gripper)
            observation = self._build_observation(current_state, gripper_state)
            full_action = self._build_action(action, gripper_action)

            # 통합 방식: MultiCameraManager로 모든 카메라에서 캡처
            images = self._capture_images()
            if not images:
                self._camera_errors += 1
                return False

            # 멀티 카메라 레코딩 (통합)
            self.recorder.record_frame_multi(
                observation=observation,
                action=full_action,
                images=images,
            )

            # 상태 업데이트
            self._last_record_step = int(self._step_counter / self.frame_skip_ratio)
            self._recorded_frames += 1
            self._last_step_time = time.time()

            return True

        except Exception as e:
            print(f"[RecordingCallback] Error recording frame: {e}")
            return False

        finally:
            self._step_counter += 1

    def step_without_record(self) -> None:
        """레코딩 없이 스텝 카운터만 증가"""
        self._step_counter += 1
        self._skipped_frames += 1

    def _capture_images(self) -> Optional[Dict[str, np.ndarray]]:
        """
        MultiCameraManager에서 모든 이미지 캡처 (통합 방식)

        공식 LeRobot 방식과 동일하게 async_read_all()을 사용하여
        모든 카메라에서 최신 프레임을 비동기로 가져옴.

        Returns:
            Dict[str, np.ndarray]: {camera_name: image} 형태
            예: {"realsense": img1, "innomaker": img2}
        """
        if self.camera_manager is None:
            # 카메라 없이 테스트용 더미 이미지
            return {"dummy": np.zeros((480, 640, 3), dtype=np.uint8)}

        try:
            # MultiCameraManager.async_read_all() -> {name: image}
            images = self.camera_manager.async_read_all()

            if not images:
                print("[RecordingCallback] No images captured from cameras")
                return None

            return images

        except TimeoutError as e:
            print(f"[RecordingCallback] Camera timeout: {e}")
            return None
        except Exception as e:
            print(f"[RecordingCallback] Camera error: {e}")
            return None

    def _build_observation(
        self,
        arm_state: np.ndarray,
        gripper_state: Optional[float] = None,
    ) -> np.ndarray:
        """
        관측값 구성 (6 joints)

        Args:
            arm_state: 5 arm joint positions (normalized)
            gripper_state: 1 gripper position (normalized)

        Returns:
            shape (6,) observation
        """
        arm_state = np.asarray(arm_state, dtype=np.float32)

        if arm_state.shape == (NUM_JOINTS,):
            # 이미 6 joints 포함
            return arm_state
        elif arm_state.shape == (5,):
            # 5 arm joints -> 6 joints with gripper
            gripper = gripper_state if gripper_state is not None else 0.0
            return np.concatenate([arm_state, [gripper]], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected arm_state shape: {arm_state.shape}")

    def _build_action(
        self,
        arm_action: np.ndarray,
        gripper_action: Optional[float] = None,
    ) -> np.ndarray:
        """
        액션 구성 (6 joints)

        Args:
            arm_action: 5 arm joint targets (normalized)
            gripper_action: 1 gripper target (normalized)

        Returns:
            shape (6,) action
        """
        arm_action = np.asarray(arm_action, dtype=np.float32)

        if arm_action.shape == (NUM_JOINTS,):
            return arm_action
        elif arm_action.shape == (5,):
            gripper = gripper_action if gripper_action is not None else 0.0
            return np.concatenate([arm_action, [gripper]], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected arm_action shape: {arm_action.shape}")

    def get_stats(self) -> dict:
        """레코딩 통계 반환"""
        stats = {
            "recorded_frames": self._recorded_frames,
            "skipped_frames": self._skipped_frames,
            "total_steps": self._step_counter,
            "camera_errors": self._camera_errors,
            "target_fps": self.target_fps,
            "control_hz": self.control_hz,
            "frame_skip_ratio": self.frame_skip_ratio,
            "effective_fps": (
                self._recorded_frames / (time.time() - self._start_time)
                if self._start_time and self._recorded_frames > 0
                else 0
            ),
        }

        if self.camera_manager is not None and hasattr(self.camera_manager, 'camera_names'):
            stats["cameras"] = self.camera_manager.camera_names

        return stats

    def __repr__(self) -> str:
        stats = self.get_stats()
        camera_info = f"cameras={stats.get('cameras', [])}"
        return (
            f"RecordingCallback(\n"
            f"  recorded={stats['recorded_frames']},\n"
            f"  skipped={stats['skipped_frames']},\n"
            f"  target_fps={stats['target_fps']},\n"
            f"  effective_fps={stats['effective_fps']:.1f},\n"
            f"  {camera_info}\n"
            f")"
        )


class SimpleRecordingCallback:
    """
    단순화된 레코딩 콜백 (FPS 동기화 없음)

    모든 스텝을 레코딩합니다. 테스트용 또는 제어 루프 FPS가
    이미 목표 FPS와 일치하는 경우에 사용합니다.

    통합 카메라 지원:
    - MultiCameraManager만 사용 (단일/다중 카메라 구분 없음)
    """

    def __init__(
        self,
        recorder: "DatasetRecorder",
        camera_manager: Any,
    ):
        # Lazy import to avoid circular dependency
        from .recorder import DatasetRecorder

        self.recorder = recorder
        self.camera_manager = camera_manager
        self._start_time: Optional[float] = None
        self._frame_count = 0

    def reset(self) -> None:
        self._start_time = time.time()
        self._frame_count = 0

    def on_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> bool:
        """매 스텝 레코딩 (통합 방식)"""
        if not self.recorder.is_recording:
            return False

        try:
            if self._start_time is None:
                self._start_time = time.time()

            # 통합 방식: MultiCameraManager로 모든 카메라에서 캡처
            images = {}
            if self.camera_manager is not None:
                try:
                    images = self.camera_manager.async_read_all()
                except:
                    pass

            if not images:
                # 더미 이미지 (모든 활성화된 카메라에 대해)
                for cam in self.recorder.enabled_cameras:
                    images[cam.name] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

            self.recorder.record_frame_multi(
                observation=observation,
                action=action,
                images=images,
            )

            self._frame_count += 1
            return True

        except Exception as e:
            print(f"[SimpleRecordingCallback] Error: {e}")
            return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_recording_setup(
    repo_id: str,
    config_yaml: Optional[str] = None,
    fps: int = DEFAULT_FPS,
    control_hz: int = CONTROL_HZ,
) -> tuple:
    """
    YAML 설정에서 레코딩 시스템 전체 생성 (통합 방식)

    pipeline_config/recording_config.yaml에서 카메라 설정을 읽어
    MultiCameraManager, DatasetRecorder, RecordingCallback을 생성합니다.

    Args:
        repo_id: 데이터셋 저장 경로
        config_yaml: YAML 설정 파일 경로 (None이면 기본 경로)
        fps: 레코딩 FPS
        control_hz: 제어 루프 주파수

    Returns:
        tuple: (camera_manager, recorder, callback)

    Example:
        camera_manager, recorder, callback = create_recording_setup(
            repo_id="local/my_dataset",
            config_yaml="pipeline_config/recording_config.yaml",
        )

        camera_manager.connect_all()
        recorder.start_episode("pick and place task")

        while executing:
            if callback.should_record():
                callback.on_step(state, action)
            # control loop...

        recorder.end_episode()
        camera_manager.disconnect_all()
        recorder.finalize()
    """
    from .recorder import DatasetRecorder
    from .config import create_camera_manager_from_config

    # 1. 카메라 매니저 생성 (YAML에서 동적 로드)
    print(f"[RecordingSetup] Loading cameras from config...")
    camera_manager = create_camera_manager_from_config(config_yaml)

    # 2. 레코더 생성 (YAML에서 features 동적 생성)
    print(f"[RecordingSetup] Creating dataset recorder: {repo_id}")
    recorder = DatasetRecorder(
        repo_id=repo_id,
        fps=fps,
        config_yaml=config_yaml,
    )

    # 3. 콜백 생성 (통합 방식: camera_manager 사용)
    print(f"[RecordingSetup] Creating recording callback")
    callback = RecordingCallback(
        recorder=recorder,
        camera_manager=camera_manager,
        target_fps=fps,
        control_hz=control_hz,
    )

    print(f"[RecordingSetup] Setup complete!")
    print(f"  Cameras: {camera_manager.camera_names if hasattr(camera_manager, 'camera_names') else 'N/A'}")
    print(f"  FPS: {fps}, Control Hz: {control_hz}")

    return camera_manager, recorder, callback
