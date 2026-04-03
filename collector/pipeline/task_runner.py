"""
TaskRunner — VLM 생성 코드를 로봇에서 실행하는 공통 런너.

역할:
1. skills + positions를 실행 환경(exec_globals)에 주입
2. LLM이 생성한 execute_task() / execute_reset_task() 호출
3. 실행 중 state/action 레코딩 (서브클래스에서 방식 결정)

구조:
    TaskRunner (base)
    ├── SingleArmTaskRunner  — RecordingContext 기반
    └── DualArmTaskRunner    — MultiArmRecorder 기반
"""

import time
import numpy as np
from typing import Dict, Optional


class TaskRunner:
    """VLM 생성 코드를 로봇에서 실행 + 레코딩하는 공통 런너."""

    def __init__(
        self,
        skills,
        recorder=None,
        camera_manager=None,
        camera=None,
        recording_fps: int = 30,
    ):
        """
        Args:
            skills: LeRobotSkills 또는 MultiArmSkills 인스턴스
            recorder: DatasetRecorder 인스턴스 (None이면 레코딩 안 함)
            camera_manager: MultiCameraManager (레코딩용)
            camera: 단일 카메라 fallback (싱글암)
            recording_fps: 레코딩 FPS
        """
        self.skills = skills
        self.recorder = recorder
        self.camera_manager = camera_manager
        self.camera = camera
        self.recording_fps = recording_fps

    def build_exec_globals(self, positions: Dict, extra_globals: Dict = None) -> Dict:
        """실행 환경 구성 — 싱글/멀티 100% 동일."""
        exec_globals = {
            "__name__": "__generated__",
            "skills": self.skills,
            "positions": positions,
        }
        if extra_globals:
            exec_globals.update(extra_globals)
        return exec_globals

    def execute(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """코드 실행 — 싱글/멀티 100% 동일.

        레코딩 여부에 따라 자동 분기.
        """
        try:
            exec_globals = self.build_exec_globals(positions, extra_globals)

            if self.recorder is not None:
                return self._execute_with_recording(code, exec_globals)
            else:
                return self._execute_bare(code, exec_globals)

        except AssertionError as e:
            error_msg = str(e)
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                print(f"\n[TaskRunner] FATAL: Robot connection failed - {e}")
                raise
            print(f"[TaskRunner] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"[TaskRunner] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_bare(self, code: str, exec_globals: Dict) -> bool:
        """레코딩 없이 코드만 실행 — 싱글/멀티 100% 동일."""
        exec(code, exec_globals)
        self._call_entry_function(exec_globals)
        return True

    def _execute_with_recording(self, code: str, exec_globals: Dict) -> bool:
        """레코딩 포함 실행 — 서브클래스에서 override."""
        raise NotImplementedError(
            "TaskRunner._execute_with_recording must be overridden by subclass"
        )

    @staticmethod
    def _call_entry_function(exec_globals: Dict):
        """exec_globals에서 execute_task 또는 execute_reset_task를 호출."""
        if "execute_task" in exec_globals:
            exec_globals["execute_task"]()
        elif "execute_reset_task" in exec_globals:
            exec_globals["execute_reset_task"]()


class SingleArmTaskRunner(TaskRunner):
    """싱글암: RecordingContext 기반 레코딩."""

    def _execute_with_recording(self, code: str, exec_globals: Dict) -> bool:
        """RecordingContext를 사용한 레코딩 포함 실행.

        RecordingContext를 설정하면 LeRobotSkills가 connect() 시
        자동으로 콜백을 획득하여 제어 루프 내에서 (state, action) 쌍을 캡처.
        """
        try:
            from record_dataset.context import RecordingContext

            recording_camera = self.camera_manager if self.camera_manager else self.camera

            active_recorder = (
                RecordingContext._recorder
                if RecordingContext._recorder
                else self.recorder
            )
            RecordingContext.setup(
                recorder=active_recorder,
                camera_manager=recording_camera,
                target_fps=self.recording_fps,
                control_hz=50,
                use_async_capture=True,
            )
            RecordingContext.reset_episode()
            # Re-acquire callback for pre-created skills (created before setup)
            if self.skills is not None:
                self.skills.recording_callback = RecordingContext.get_callback()
            print(f"[Recording] Context setup complete")

            exec(code, exec_globals)
            self._call_entry_function(exec_globals)

            stats = RecordingContext.get_stats()
            print(
                f"[Recording] Recorded {stats['recorded_frames']} frames, "
                f"effective FPS: {stats['effective_fps']:.1f}"
            )

            return True

        except ImportError as e:
            print(f"[Recording] Warning: RecordingContext not available: {e}")
            print(f"[Recording] Falling back to non-recording execution")
            exec(code, exec_globals)
            self._call_entry_function(exec_globals)
            return True

        finally:
            try:
                from record_dataset.context import RecordingContext
                RecordingContext.clear()
            except Exception:
                pass


class DualArmTaskRunner(TaskRunner):
    """멀티암: MultiArmRecorder 기반 12축 레코딩."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_arm_recorder = None
        self._last_mar_stats: Optional[Dict] = None

    def _execute_with_recording(self, code: str, exec_globals: Dict) -> bool:
        """MultiArmRecorder를 사용한 12축 레코딩 포함 실행.

        1. MultiArmRecorder 생성 + 각 arm에 action 콜백 주입
        2. connect() 완료 후 recorder.start()
        3. exec → 스킬 실행
        4. recorder.stop()
        """
        from record_dataset.multi_arm_recorder import MultiArmRecorder

        multi_arm = self.skills  # MultiArmSkills 인스턴스

        try:
            mar = MultiArmRecorder(
                multi_arm=multi_arm,
                recorder=self.recorder,
                camera_manager=self.camera_manager,
                target_fps=self.recording_fps,
            )
            self._multi_arm_recorder = mar

            # Per-arm 콜백 주입 (state + action)
            def make_callback(set_state_fn, set_action_fn):
                def callback(state, action):
                    set_state_fn(state)
                    set_action_fn(action)
                return callback

            multi_arm.left_arm.recording_callback = make_callback(
                mar.set_left_state, mar.set_left_action
            )
            multi_arm.right_arm.recording_callback = make_callback(
                mar.set_right_state, mar.set_right_action
            )
            multi_arm.left_arm.skill_info_callback = mar.set_left_skill_info
            multi_arm.right_arm.skill_info_callback = mar.set_right_skill_info

            # exec(code) → execute_task() 정의만 (호출은 아래에서)
            exec(code, exec_globals)

            # connect() 완료 후 recorder 시작 (monkey-patch)
            original_connect = multi_arm.connect

            def connect_then_record():
                result = original_connect()
                # 초기 action을 현재 state로 설정 (None fallback 방지)
                if multi_arm.left_arm.robot:
                    left_state = multi_arm.left_arm.robot.read_positions()
                    if left_state is not None:
                        mar.set_left_action(np.asarray(left_state, dtype=np.float32))
                if multi_arm.right_arm.robot:
                    right_state = multi_arm.right_arm.robot.read_positions()
                    if right_state is not None:
                        mar.set_right_action(np.asarray(right_state, dtype=np.float32))
                mar._recording_start_wall = time.time()
                mar.start()
                print(
                    f"[Recording] MultiArmRecorder started "
                    f"(12-axis, 50Hz → {self.recording_fps}fps)"
                )
                return result

            multi_arm.connect = connect_then_record

            self._call_entry_function(exec_globals)

            # 정리
            multi_arm.connect = original_connect
            multi_arm.left_arm.skill_info_callback = None
            multi_arm.right_arm.skill_info_callback = None

            mar._recording_stop_wall = time.time()
            mar.stop()

            # 통계
            stats = mar.get_stats()
            wall_duration = mar._recording_stop_wall - getattr(
                mar, "_recording_start_wall", mar._recording_stop_wall
            )
            expected_frames = int(wall_duration * self.recording_fps)
            actual_ratio = stats["recorded_frames"] / max(expected_frames, 1)
            print(
                f"[Recording] Recorded {stats['recorded_frames']} frames "
                f"(wall={wall_duration:.1f}s, expected={expected_frames}, "
                f"ratio={actual_ratio:.2f})"
            )
            self._last_mar_stats = {
                **stats,
                "wall_duration_s": round(wall_duration, 1),
                "expected_frames": expected_frames,
                "ratio": round(actual_ratio, 2),
            }

            return True

        except AssertionError as e:
            error_msg = str(e)
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                raise
            print(f"[DualArmTaskRunner] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"[DualArmTaskRunner] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if self._multi_arm_recorder and self._multi_arm_recorder.is_running:
                self._multi_arm_recorder.stop()
            self._multi_arm_recorder = None
