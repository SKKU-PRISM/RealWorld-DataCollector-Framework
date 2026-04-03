#!/usr/bin/env python3
"""
Forward + Reset Integrated Pipeline

Forward Execution → Judge (Evaluation) → Reset Execution 통합 파이프라인

Usage:
    # 전체 파이프라인 실행
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" "blue box"

    # Reset 건너뛰기
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --skip-reset

    # 결과 저장
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --save results/

    # LeRobot 데이터셋 레코딩 모드
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --record --dataset-repo-id "user/my_dataset"
"""

import argparse
import io
import json
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class TeeLogger:
    """
    터미널 출력을 파일과 stdout 모두에 기록하는 로거.

    Usage:
        logger = TeeLogger("output.txt")
        logger.start()
        print("Hello, World!")  # stdout과 파일 모두에 기록
        logger.stop()
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.original_stdout = None
        self.buffer = io.StringIO()

    def write(self, text):
        """stdout과 버퍼 모두에 기록"""
        if self.original_stdout:
            self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        """버퍼 플러시"""
        if self.original_stdout:
            self.original_stdout.flush()

    def start(self):
        """로깅 시작"""
        self.original_stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self

    def stop(self):
        """로깅 종료 및 파일 저장"""
        if self.original_stdout:
            sys.stdout = self.original_stdout

        # 로그 내용 정리 (ANSI 색상 코드 제거)
        log_content = self.buffer.getvalue()

        # ANSI 이스케이프 시퀀스 제거
        import re
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_content = ansi_escape.sub('', log_content)

        # 적당한 위치에 줄바꿈 추가 (가독성 향상)
        clean_content = self._format_log(clean_content)

        # 파일 저장
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(clean_content)

        self.original_stdout = None
        return self.filepath

    def _format_log(self, content: str) -> str:
        """로그 내용 포맷팅 (가독성 향상)"""
        # 주요 섹션 앞에 빈 줄 추가
        formatted = content

        # 섹션 구분자 패턴들
        section_patterns = [
            r'(\[PHASE \d\])',
            r'(\[Step \d+/\d+\])',
            r'(={50,})',
            r'(-{40,})',
            r'(Moving to)',
            r'(Gripper:)',
            r'(PICK AND PLACE)',
            r'(EXECUTE PICK)',
            r'(EXECUTE PLACE)',
            r'(ROTATE 90)',
            r'(Error:)',
            r'(WARNING:)',
        ]

        return formatted

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))


from pipeline.base_pipeline import BasePipeline


class ForwardAndResetPipeline(BasePipeline):
    """Forward → Judge → Reset 통합 파이프라인"""

    def __init__(
        self,
        robot_id: int = 3,
        llm_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o",
        judge_timeout_ms: int = 5000,
        num_random_seeds: int = 1,
        verbose: bool = True,
        # Recording options
        record_dataset: bool = False,
        dataset_repo_id: Optional[str] = None,
        recording_fps: int = 30,
        resume_recording: bool = False,
        # Multi-turn options
        multi_turn: bool = False,
        cad_image_dirs: List[str] = None,
        side_view_image: str = None,
        codegen_model: str = None,
        reset_instruction: str = None,
        skip_turn_test: bool = False,
        detect_model: str = None,
        resetspace: str = None,
    ):
        """
        초기화

        Args:
            robot_id: 로봇 번호 (2 또는 3)
            llm_model: 코드 생성용 LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
            judge_model: Judge VLM 모델
            judge_timeout_ms: Judge UI 타임아웃 (밀리초)
            num_random_seeds: 배치 수 (1=초기 위치 유지, N>1=N종류 랜덤 배치)
            verbose: 상세 출력 여부
            record_dataset: LeRobot 데이터셋 레코딩 활성화
            dataset_repo_id: 데이터셋 저장 경로 (예: "user/my_dataset")
            recording_fps: 레코딩 FPS (기본: 30)
            multi_turn: True면 crop-then-point 멀티턴 LLM 코드 생성 사용
            cad_image_dirs: CAD 참조 이미지 디렉토리 리스트 (옵션)
        """
        self.robot_id = robot_id
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.judge_timeout_ms = judge_timeout_ms
        self.num_random_seeds = num_random_seeds
        self.verbose = verbose

        # Multi-turn options
        self.multi_turn = multi_turn
        self.cad_image_dirs = cad_image_dirs or []
        self.side_view_image = side_view_image
        self.codegen_model = codegen_model
        self.reset_instruction = reset_instruction or "move objects to certain position"
        self.skip_turn_test = skip_turn_test
        self.detect_model = detect_model
        self.resetspace = resetspace
        self.multi_turn_info: Dict = {}
        self.reset_multi_turn_info: Dict = {}

        # Recording options
        self.record_dataset = record_dataset
        self.dataset_repo_id = dataset_repo_id
        self.recording_fps = recording_fps
        self.resume_recording = resume_recording
        self.dataset_recorder = None
        self.recording_skills_wrapper = None
        self.camera_manager = None  # MultiCameraManager for recording

        # 상태 저장
        self.camera = None
        self.initial_image: Optional[np.ndarray] = None
        self.final_image: Optional[np.ndarray] = None
        self.detection_image: Optional[np.ndarray] = None
        self.detected_positions: Dict = {}
        self.extended_detections: Dict = {}
        self.generated_spec: Dict = {}
        self.generated_code: str = ""
        self.instruction: str = ""

        # 이미지 해상도 저장 (Judge용)
        self.initial_image_resolution: Optional[Tuple[int, int]] = None  # (width, height)
        self.final_image_resolution: Optional[Tuple[int, int]] = None

        # Reset recording
        self.reset_dataset_recorder = None

        # Reset 관련 상태
        self.reset_initial_image: Optional[np.ndarray] = None
        self.reset_final_image: Optional[np.ndarray] = None
        self.reset_initial_resolution: Optional[Tuple[int, int]] = None
        self.reset_final_resolution: Optional[Tuple[int, int]] = None
        self.reset_code: str = ""

        # Forward initial image path (for reset multi-turn)
        self.forward_initial_image_path: Optional[str] = None

        # Context 저장용
        self.execution_context: Dict = {}

        # Episode tracking for logging
        self.current_episode: int = 1
        self.total_episodes: int = 1
        self.current_phase: str = "Forward"  # "Forward" or "Reset"

        # First episode positions for "original" reset mode
        # Stores the first episode's detection result to avoid cumulative drift
        self.first_episode_positions: Optional[Dict] = None

        # 과거 모든 배치 위치 누적 (랜덤 위치 생성 시 겹침 방지)
        self._all_previous_seed_positions = []

        # Code reuse cache: 성공한 코드를 캐싱하여 이후 에피소드에서 재사용
        self.cached_forward_code: Optional[str] = None
        self.cached_forward_keys: List[str] = []  # 캐싱 코드가 참조하는 position keys
        self.cached_reset_code: Optional[str] = None
        self.cached_reset_keys: List[str] = []

        # 레코딩 모드 초기화 (resume 모드는 cleanup 후 초기화)
        if self.record_dataset and not self.resume_recording:
            self._init_recording()

    @staticmethod
    def _extract_position_keys(code: str) -> List[str]:
        """캐싱된 코드에서 positions['xxx'] 패턴의 key 추출"""
        import re
        pattern = r'positions\[["\'](.+?)["\']\]'
        return list(dict.fromkeys(re.findall(pattern, code)))

    def _can_reuse_code(self, cached_code: Optional[str], cached_keys: List[str],
                        new_positions: Dict) -> bool:
        """캐싱된 코드를 재사용할 수 있는지 확인 (key 일치 검사)

        cached_keys가 비어있으면 코드가 positions를 참조하지 않는 것이므로
        (좌표 하드코딩 등) 무조건 재사용 가능.
        """
        if cached_code is None:
            return False
        if not cached_keys:
            return True
        return set(cached_keys) <= set(new_positions.keys())

    def _create_skills(self):
        """LeRobotSkills 인스턴스 생성 (exec_globals 주입용).

        LLM 코드가 직접 import/생성하지 않고, 파이프라인이 미리 생성하여 주입.
        """
        from skills.skills_lerobot import LeRobotSkills

        robot_config = f"robot_configs/robot/so101_robot{self.robot_id}.yaml"
        kwargs = {
            "robot_config": robot_config,
            "frame": "base_link",
        }
        if self.detect_model:
            kwargs["detect_model"] = self.detect_model

        self._skills = LeRobotSkills(**kwargs)

        # 공유 카메라 주입 (detect_objects에서 사용)
        if self.camera:
            self._skills.camera = self.camera

        return self._skills

    def _get_pipeline_camera(self):
        """PipelineCamera lazy init."""
        if not hasattr(self, '_pipeline_camera') or self._pipeline_camera is None:
            from pipeline.camera_session import PipelineCamera
            self._pipeline_camera = PipelineCamera(
                camera_manager=self.camera_manager, verbose=self.verbose)
        return self._pipeline_camera

    def initialize_camera(self) -> bool:
        """카메라 초기화 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera_manager = self.camera_manager  # 동기화
        result = pc.initialize()
        self.camera = pc.camera
        return result

    def shutdown_camera(self) -> None:
        """카메라 종료 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera = self.camera  # 동기화
        pc.shutdown()
        self.camera = pc.camera  # None으로 반영

    def _init_recording(self) -> None:
        """LeRobot 데이터셋 레코딩 초기화 (멀티 카메라 지원)"""
        if not self.record_dataset:
            return

        if not self.dataset_repo_id:
            # 기본 repo_id 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_repo_id = f"local/cap_dataset_{timestamp}"

        try:
            from record_dataset import DatasetRecorder
            from record_dataset.config import (
                create_camera_manager_from_config,
                load_cameras_from_yaml,
                build_features_from_yaml,
            )

            print(f"\n[Recording] Initializing multi-camera dataset recorder...")
            print(f"  Repo ID: {self.dataset_repo_id}")
            print(f"  FPS: {self.recording_fps}")

            # 1. 카메라 매니저 초기화 (YAML에서 동적 로드, 싱글암: left_arm만)
            print(f"[Recording] Loading camera configuration...")
            self.camera_manager = create_camera_manager_from_config(num_robots=1)

            # 2. 카메라 연결
            print(f"[Recording] Connecting cameras...")
            self.camera_manager.connect_all()
            print(f"[Recording] Cameras connected: {self.camera_manager.camera_names}")

            # 3. 카메라 연결 검증
            cameras = load_cameras_from_yaml(num_robots=1)
            enabled_cameras = [cam for cam in cameras if cam.enabled]
            connected_names = set(self.camera_manager.camera_names)
            expected_names = {cam.feature_name for cam in enabled_cameras}
            missing = expected_names - connected_names
            if missing:
                missing_details = []
                for cam in enabled_cameras:
                    if cam.feature_name in missing:
                        device = cam.get_device_path() or cam.serial_number or "unknown"
                        missing_details.append(f"  - {cam.feature_name} ({cam.type}, device={device})")
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Camera connection failed!\n"
                    f"========================================\n"
                    f"The following cameras are enabled in recording_config.yaml\n"
                    f"but failed to connect:\n"
                    + "\n".join(missing_details) + "\n"
                    f"\n"
                    f"To fix, either:\n"
                    f"  1. Connect the camera hardware and verify device path\n"
                    f"     (run: v4l2-ctl --list-devices)\n"
                    f"  2. Set 'enabled: false' for unavailable cameras in\n"
                    f"     pipeline_config/recording_config.yaml\n"
                    f"========================================"
                )

            # 4. Features 빌드 (연결된 카메라 기준, 멀티암과 동일 패턴)
            features = build_features_from_yaml(num_robots=1)

            # 5. 기존 dataset 존재 여부 미리 체크 (forward + reset)
            from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
            reset_repo_id = self.dataset_repo_id + "_reset"
            existing = []
            for rid in [self.dataset_repo_id, reset_repo_id]:
                ds_path = HF_LEROBOT_HOME / rid
                if ds_path.exists() and not self.resume_recording:
                    existing.append(str(ds_path))
            if existing:
                paths_str = "\n".join(f"     rm -rf {p}" for p in existing)
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Dataset already exists!\n"
                    f"========================================\n"
                    f"Paths:\n" + "\n".join(f"  - {p}" for p in existing) + "\n"
                    f"\n"
                    f"To continue, either:\n"
                    f"  1. Delete the existing datasets:\n"
                    f"{paths_str}\n"
                    f"  2. Use a different repo_id\n"
                    f"========================================"
                )

            # 6. 레코더 초기화 (빌드된 features 전달, 멀티암과 동일 패턴)
            self.dataset_recorder = DatasetRecorder(
                repo_id=self.dataset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
                features=features,
            )
            print(f"[Recording] Recorder initialized successfully")
            print(f"[Recording] Features: {list(self.dataset_recorder.features.keys())}")

            # 7. Reset 레코더 초기화 (별도 dataset, 동일 features)
            print(f"\n[Recording] Initializing reset dataset recorder...")
            print(f"  Reset Repo ID: {reset_repo_id}")
            self.reset_dataset_recorder = DatasetRecorder(
                repo_id=reset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
                features=features,
            )
            print(f"[Recording] Reset recorder initialized")

            # Signal handler: Ctrl+C 시 finalize() 호출하여 데이터셋 보존
            self._install_recording_signal_handler()

        except ImportError as e:
            print(f"[Recording] Warning: Failed to import record_dataset: {e}")
            print(f"[Recording] Dataset recording will be disabled")
            self.record_dataset = False
            self.dataset_recorder = None
            self.camera_manager = None
        except AssertionError:
            # Dataset already exists - 파이프라인 완전 종료
            raise
        except Exception as e:
            print(f"[Recording] Warning: Failed to initialize recorder: {e}")
            import traceback
            traceback.print_exc()
            self.record_dataset = False
            self.dataset_recorder = None
            self.camera_manager = None

    def _start_reset_episode_recording(self, target_positions: Dict) -> None:
        """Reset 에피소드 레코딩 시작 (별도 dataset, recorder 교체)"""
        try:
            from record_dataset.context import RecordingContext

            # Reset recorder로 episode 시작
            self.reset_dataset_recorder.start_episode(task=self.reset_instruction)
            print(f"[Reset Recording] Episode started: {self.reset_instruction}")

            # RecordingContext의 recorder를 reset recorder로 교체
            self._saved_forward_recorder = RecordingContext._recorder
            RecordingContext._recorder = self.reset_dataset_recorder
        except Exception as e:
            print(f"[Reset Recording] Warning: Failed to start: {e}")

    def _end_reset_episode_recording(self, discard: bool = False) -> None:
        """Reset 에피소드 레코딩 종료 (forward recorder로 복원)"""
        try:
            from record_dataset.context import RecordingContext

            # Reset episode 종료
            info = self.reset_dataset_recorder.end_episode(discard=discard)
            if not discard:
                print(f"[Reset Recording] Episode saved: {info.get('num_frames', 0)} frames")
            else:
                print(f"[Reset Recording] Episode discarded")

            # Forward recorder로 복원
            RecordingContext._recorder = self._saved_forward_recorder
            RecordingContext.clear_subtask()
        except Exception as e:
            print(f"[Reset Recording] Warning: Failed to end: {e}")

    def _finalize_recording(self) -> None:
        """데이터셋 레코딩 완료 및 카메라 연결 해제"""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.finalize()
                print(f"\n[Recording] Forward dataset finalized at: {self.dataset_recorder._dataset.root}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to finalize forward dataset: {e}")

        if self.reset_dataset_recorder:
            try:
                self.reset_dataset_recorder.finalize()
                print(f"[Recording] Reset dataset finalized at: {self.reset_dataset_recorder._dataset.root}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to finalize reset dataset: {e}")

        # 멀티 카메라 연결 해제
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
                print(f"[Recording] Camera manager disconnected")
            except Exception as e:
                print(f"[Recording] Warning: Failed to disconnect cameras: {e}")
            self.camera_manager = None


    def capture_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 캡처 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera_manager = self.camera_manager  # 동기화
        pc.camera = self.camera
        return pc.capture_frame()

    def run_detection(
        self,
        queries: list,
        timeout: float = 10.0,
        visualize: bool = False,
    ) -> Dict:
        """객체 검출 수행 (통합된 run_realtime_detection 사용)

        Recording 카메라가 있으면 공유하여 리소스 충돌 방지.
        """
        from run_detect import run_realtime_detection

        # 카메라 공유 (리소스 충돌 방지)
        # 우선순위: 1. self.camera (initialize_camera로 생성된 것)
        #          2. camera_manager의 카메라
        #          3. None (run_realtime_detection에서 자체 생성)
        external_camera = None
        if self.camera is not None:
            external_camera = self.camera
            print("  [Detection] Using pipeline camera")
        elif self.camera_manager and self.camera_manager.is_connected:
            try:
                external_camera = self._get_pipeline_camera().get_realsense()
                print("  [Detection] Using shared camera from recording")
            except KeyError:
                print("  [Detection] No realsense camera in manager, using internal camera")

        extended_results, last_frame, vis_image = run_realtime_detection(
            queries=queries,
            timeout=timeout,
            unit="m",
            return_last_frame=True,
            return_extended=True,
            visualize=visualize,
            robot_id=self.robot_id,
            external_camera=external_camera,
        )

        if last_frame is not None:
            self.initial_image = last_frame

        if vis_image is not None:
            self.detection_image = vis_image

        # Store extended info for later use
        # Note: Workspace filtering is done at detection level (run_detect.py)
        self.extended_detections = extended_results

        # Return extended format: {name: {"position": [x,y,z], ...}}
        return extended_results

    def generate_forward_code(
        self,
        instruction: str,
        positions: Dict,
        image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
        canonical_point_labels: Dict[str, List[str]] = None,
    ) -> str:
        """Forward 코드 생성

        Args:
            instruction: 자연어 목표
            positions: Extended format {name: {"position": [x,y,z], ...}}
            image_path: 초기 이미지 경로 (multi-turn 모드에서 필수)
            skip_codegen: True면 검출만 수행, 코드 생성 스킵 (multi-turn 전용)
            canonical_labels: 코드 재사용 시 강제할 라벨 목록 (multi-turn 전용)
            canonical_point_labels: 코드 재사용 시 강제할 point 라벨 {obj: [labels]}

        Returns:
            생성된 Python 코드 문자열 (skip_codegen=True면 빈 문자열)
        """
        if self.multi_turn:
            return self._generate_forward_code_multi_turn(
                instruction, positions, image_path,
                skip_codegen=skip_codegen,
                canonical_labels=canonical_labels,
                canonical_point_labels=canonical_point_labels,
            )
        else:
            return self._generate_forward_code_single(instruction, positions)

    def _generate_forward_code_single(self, instruction: str, positions: Dict) -> str:
        """Single-turn 코드 생성 (기존 방식)"""
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen

        not_found = [name for name, info in positions.items() if info is None]
        if not_found:
            raise ValueError(f"Required objects not detected: {not_found}")

        code, _ = lerobot_code_gen(
            instruction=instruction,
            object_positions=positions,
            use_detection=False,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
        )

        return code

    def _generate_forward_code_multi_turn(
        self,
        instruction: str,
        positions: Dict,
        image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
        canonical_point_labels: Dict[str, List[str]] = None,
    ) -> str:
        """Multi-turn 코드 생성 (4-turn LLM 대화)

        LLM이 직접 이미지를 보고 장면 이해 → bbox 검출 → grasp point → 코드 생성.
        positions는 fallback으로 사용 (LLM pixel→world 변환 실패 시).

        Args:
            skip_codegen: True면 T0~T2(검출)만 수행하고 코드 생성(T3) 스킵
            canonical_labels: 코드 재사용 시 T1에서 강제할 라벨 목록
            canonical_point_labels: 코드 재사용 시 T2에서 강제할 point 라벨 {obj: [labels]}
        """
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen_multi_turn

        if image_path is None:
            print("[MultiTurn] WARNING: No image_path provided, falling back to single-turn")
            return self._generate_forward_code_single(instruction, positions)

        # depth 기반 3D 좌표 변환을 위해 카메라 전달
        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self._get_pipeline_camera().get_realsense()
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        code, mt_positions, mt_info = lerobot_code_gen_multi_turn(
            instruction=instruction,
            image_path=image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            fallback_positions=positions,
            camera=active_camera,
            cad_image_dirs=self.cad_image_dirs,
            side_view_image=self.side_view_image,
            codegen_model=self.codegen_model,
            skip_codegen=skip_codegen,
            canonical_labels=canonical_labels,
            canonical_point_labels=canonical_point_labels,
            skip_turn_test=self.skip_turn_test,
        )

        # multi-turn 정보 저장
        self.multi_turn_info = mt_info

        # multi-turn으로 생성된 positions로 업데이트
        # (world 좌표 변환 성공한 것만)
        for name, info in mt_positions.items():
            if not info.get("_needs_world_coords", False):
                self.detected_positions[name] = info

        return code

    def _extract_skill_sequence(self, exec_globals: Dict = None):
        """LeRobotSkills._last_instance에서 스킬 시퀀스를 추출"""
        try:
            from skills.skills_lerobot import LeRobotSkills
            if LeRobotSkills._last_instance and hasattr(LeRobotSkills._last_instance, 'skill_sequence'):
                self._last_skill_sequence = LeRobotSkills._last_instance.skill_sequence
        except Exception:
            pass

    def _extract_skill_sequence_from_skills(self, skills):
        """skills 인스턴스에서 스킬 시퀀스를 추출"""
        try:
            if hasattr(skills, 'skill_sequence'):
                self._last_skill_sequence = skills.skill_sequence
        except Exception:
            pass

    @staticmethod
    def _patch_code_block(code: str, block_name: str, new_values: Dict) -> str:
        """코드 내 지정된 dict 블록의 좌표를 치환.

        예: _patch_code_block(code, "target_positions", {...})
            → target_positions = { ... } 블록 내 좌표만 교체
        """
        import re
        block_match = re.search(rf'({block_name}\s*=\s*\{{)(.*?)(\}})', code, re.DOTALL)
        if not block_match:
            return code

        block_body = block_match.group(2)
        for name, info in new_values.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos is None or len(pos) < 3:
                continue
            pattern = rf'("{name}":\s*\[)[^\]]+(\])'
            replacement = rf'\g<1>{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}\2'
            block_body = re.sub(pattern, replacement, block_body)

        return code[:block_match.start()] + block_match.group(1) + block_body + block_match.group(3) + code[block_match.end():]

    @staticmethod
    def _patch_reset_code_targets(code: str, new_targets: Dict) -> str:
        """Reset 코드 내 target_positions 좌표 치환 (하위호환)."""
        return ForwardAndResetPipeline._patch_code_block(code, "target_positions", new_targets)

    def dry_run_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """생성된 코드를 IK 검증만으로 가상 실행 (로봇 연결 없음).

        LeRobotSkills를 DryRunSkills로 교체하여 실행.
        모든 move/pick/place가 IK 체크로 대체됨.
        """
        from skills.dry_run_skills import DryRunSkills

        try:
            dry_skills = DryRunSkills(
                robot_config=f"robot_configs/robot/so101_robot{self.robot_id}.yaml",
                frame="base_link",
            )
            exec_globals = {
                "__name__": "__generated__",
                "skills": dry_skills,
                "positions": positions,
            }
            if extra_globals:
                exec_globals.update(extra_globals)
            exec(code, exec_globals)
            if "execute_task" in exec_globals:
                exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                exec_globals["execute_reset_task"]()

            # DryRunSkills 인스턴스에서 결과 확인
            from skills.dry_run_skills import DryRunSkills as _DRS
            if _DRS._last_instance is not None:
                dry = _DRS._last_instance
                if not dry.success:
                    print(f"  [DryRun] IK failures: {dry.ik_failures}")
                return dry.success
            return True

        except Exception as e:
            print(f"  [DryRun] Code execution error: {e}")
            return False

    def _get_task_runner(self):
        """TaskRunner 인스턴스 반환 (lazy init)."""
        if not hasattr(self, '_task_runner') or self._task_runner is None:
            from pipeline.task_runner import SingleArmTaskRunner
            skills = self._create_skills()
            self._task_runner = SingleArmTaskRunner(
                skills=skills,
                recorder=self.dataset_recorder if self.record_dataset else None,
                camera_manager=self.camera_manager,
                camera=self.camera,
                recording_fps=self.recording_fps,
            )
        return self._task_runner

    def execute_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """생성된 코드 실행 (TaskRunner에 위임)"""
        self._last_skill_sequence = []
        runner = self._get_task_runner()
        # 카메라가 나중에 초기화될 수 있으므로 동기화
        if self.camera and runner.skills:
            runner.skills.camera = self.camera
        success = runner.execute(code, positions, extra_globals)
        # 스킬 시퀀스 추출
        self._extract_skill_sequence_from_skills(runner.skills)
        return success

    def capture_final_image(self) -> Optional[np.ndarray]:
        """최종 이미지 캡처 (해상도도 함께 저장)"""
        # camera_manager가 사용 가능하면 self.camera 초기화 불필요
        if not (self.camera_manager and self.camera_manager.is_connected):
            # camera_manager가 없으면 self.camera 필요
            if self.camera is None:
                if not self.initialize_camera():
                    return None
                time.sleep(0.5)

        self.final_image = self.capture_frame()
        if self.final_image is not None:
            # 해상도 저장 (width, height)
            self.final_image_resolution = (self.final_image.shape[1], self.final_image.shape[0])
        return self.final_image


    def show_judge_ui(
        self,
        instruction: str,
        prediction: str,
        reasoning: str,
        positions: Dict,
    ) -> Optional[np.ndarray]:
        """Judge 결과 UI 표시 (타임아웃 적용)"""
        if self.initial_image is None or self.final_image is None:
            return None

        from judge import show_judge_result

        result_image = show_judge_result(
            initial_image=self.initial_image,
            final_image=self.final_image,
            instruction=instruction,
            prediction=prediction,
            reasoning=reasoning,
            object_positions=positions,
            wait_key=True,
            timeout_ms=self.judge_timeout_ms,
        )

        return result_image

    def show_reset_judge_ui(
        self,
        reset_mode: str,
        prediction: str,
        reasoning: str,
        current_positions: Dict,
        target_positions: Dict,
    ) -> Optional[np.ndarray]:
        """Reset Judge 결과 UI 표시 (타임아웃 적용)"""
        if self.reset_initial_image is None or self.reset_final_image is None:
            return None

        from judge import show_reset_judge_result

        result_image = show_reset_judge_result(
            initial_image=self.reset_initial_image,
            final_image=self.reset_final_image,
            reset_mode=reset_mode,
            prediction=prediction,
            reasoning=reasoning,
            current_positions=current_positions,
            target_positions=target_positions,
            wait_key=True,
            timeout_ms=self.judge_timeout_ms,
        )

        return result_image

    def generate_reset_code(
        self,
        original_instruction: str,
        original_positions: Dict,
        forward_spec: Dict = None,
        forward_code: str = None,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        current_positions: Dict = None,
        current_state_image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """Reset 코드 생성

        multi_turn=True일 때 VLM multi-turn 파이프라인 사용,
        False일 때 기존 Grounding DINO + 단일 LLM 방식 사용.

        Args:
            current_positions: 미리 감지된 현재 위치 (multi-robot 공유 감지용)
                              제공되면 내부 detection을 건너뜀
            current_state_image_path: 현재 상태 이미지 경로 (multi-turn 모드용)
            skip_codegen: True면 검출만 수행, 코드 생성 스킵 (multi-turn 전용)
            canonical_labels: 코드 재사용 시 강제할 라벨 목록

        Returns:
            Tuple[str, Dict, Dict, Dict]: (코드, 원래 위치, 현재 위치, 타겟 위치)
        """
        if self.multi_turn:
            return self._generate_reset_code_multi_turn(
                original_instruction=original_instruction,
                original_positions=original_positions,
                current_state_image_path=current_state_image_path,
                skip_codegen=skip_codegen,
                canonical_labels=canonical_labels,
            )
        else:
            return self._generate_reset_code_single(
                original_instruction=original_instruction,
                original_positions=original_positions,
                forward_spec=forward_spec,
                forward_code=forward_code,
                detection_timeout=detection_timeout,
                visualize_detection=visualize_detection,
                current_positions=current_positions,
            )

    def _generate_reset_code_single(
        self,
        original_instruction: str,
        original_positions: Dict,
        forward_spec: Dict = None,
        forward_code: str = None,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        current_positions: Dict = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """기존 single-turn 방식 reset 코드 생성"""
        from code_gen_lerobot.reset_execution import lerobot_reset_code_gen

        # Recording용 카메라가 있으면 공유 (current_positions가 없을 때만)
        external_camera = None
        if current_positions is None and self.camera_manager and self.camera_manager.is_connected:
            try:
                external_camera = self._get_pipeline_camera().get_realsense()
                print("  [Reset Detection] Using shared camera from recording")
            except KeyError:
                pass

        reset_code, orig_pos, current_pos, target_pos = lerobot_reset_code_gen(
            original_instruction=original_instruction,
            original_positions=original_positions,
            forward_spec=forward_spec,
            forward_code=forward_code,
            external_camera=external_camera,
            detection_timeout=detection_timeout,
            visualize_detection=visualize_detection,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            reset_mode="original",
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            current_positions=current_positions,
            resetspace=self.resetspace,
        )

        return reset_code, orig_pos, current_pos, target_pos

    def _generate_reset_code_multi_turn(
        self,
        original_instruction: str,
        original_positions: Dict,
        current_state_image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """VLM multi-turn 방식 reset 코드 생성

        Forward와 동일한 crop-then-point 파이프라인으로
        현재 물체 위치를 VLM이 직접 검출하고 reset 코드 생성.
        """
        from code_gen_lerobot.reset_execution import lerobot_reset_code_gen_multi_turn

        if current_state_image_path is None or self.forward_initial_image_path is None:
            print("  [Reset MultiTurn] WARNING: Missing images, falling back to single-turn")
            return self._generate_reset_code_single(
                original_instruction=original_instruction,
                original_positions=original_positions,
            )

        # depth 기반 3D 좌표 변환용 카메라
        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self._get_pipeline_camera().get_realsense()
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        reset_code, current_pos, target_pos, grippable, obstacles, reset_mt_info = lerobot_reset_code_gen_multi_turn(
            original_instruction=original_instruction,
            original_positions=original_positions,
            current_state_image_path=current_state_image_path,
            initial_state_image_path=self.forward_initial_image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            reset_mode="original",
            camera=active_camera,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            codegen_model=self.codegen_model,
            skip_codegen=skip_codegen,
            canonical_labels=canonical_labels,
            resetspace=self.resetspace,
        )

        self.reset_multi_turn_info = reset_mt_info

        return reset_code, original_positions, current_pos, target_pos

    def run(
        self,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        use_timestamp_subdir: bool = True,
        skip_reset: bool = False,
        reset_target_positions: Optional[Dict] = None,
        pre_reset_callback=None,
        post_judge_callback=None,
    ) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            instruction: 자연어 명령어
            objects: 검출할 객체 리스트
            detection_timeout: 검출 타임아웃
            visualize_detection: 검출 시각화 여부
            save_dir: 결과 저장 디렉토리
            skip_reset: Reset 단계 건너뛰기

        Returns:
            파이프라인 결과 딕셔너리
        """
        self.instruction = instruction

        # 결과 저장 디렉토리 설정
        if save_dir is None:
            save_dir = "results"

        if use_timestamp_subdir:
            # 단일 실행: timestamp 서브디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = str(Path(save_dir) / timestamp)
        else:
            # multi-episode 모드: save_dir을 직접 사용
            result_dir = str(Path(save_dir))

        Path(result_dir).mkdir(parents=True, exist_ok=True)

        # Forward/Reset 별도 디렉토리
        forward_dir = str(Path(result_dir) / "forward")
        reset_dir = str(Path(result_dir) / "reset")
        Path(forward_dir).mkdir(parents=True, exist_ok=True)
        Path(reset_dir).mkdir(parents=True, exist_ok=True)

        # 로거 초기화
        forward_logger = TeeLogger(str(Path(forward_dir) / "forward_log.txt"))
        reset_logger = TeeLogger(str(Path(reset_dir) / "reset_log.txt"))

        result = {
            'forward': {
                'positions': {},
                'code': '',
                'execution_success': False,
            },
            'judge': {
                'prediction': 'UNCERTAIN',
                'reasoning': '',
            },
            'reset': {
                'mode': 'original',
                'current_positions': {},
                'target_positions': {},
                'code': '',
                'execution_success': False,
            },
            'reset_judge': {
                'prediction': 'UNCERTAIN',
                'reasoning': '',
                'reset_mode': "original",
            },
            'saved_files': {},
        }

        # 색상 코드
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # Set phase for logging
        self.current_phase = "Forward"
        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"

        print("\n" + CYAN + "=" * 70 + RESET)
        print(CYAN + BOLD + f"[{ep_str}] Forward + Reset Pipeline".center(70) + RESET)
        print(CYAN + "=" * 70 + RESET)

        try:
            # Forward 로깅 시작
            forward_logger.start()

            # ================================================================
            # PHASE 1: FORWARD EXECUTION
            # ================================================================
            print(f"\n{GREEN}{BOLD}" + self._log("FORWARD EXECUTION") + f"{RESET}")
            print(GREEN + "-" * 70 + RESET)

            if self.multi_turn:
                # ============================================================
                # Multi-turn: Detection 스킵, 이미지만 캡처하여 VLM에 전달
                # VLM이 Turn 1에서 직접 물체를 식별함
                # ============================================================

                # Step 1: 이미지 캡처 (Detection 없이)
                print(f"\n{YELLOW}" + self._log("Capturing image for VLM (no detection)...", step="Step 1/6") + f"{RESET}")

                # 카메라 초기화
                if not self.camera and not (self.camera_manager and self.camera_manager.is_connected):
                    if not self.initialize_camera():
                        print(f"{RED}[Error] Camera initialization failed{RESET}")
                        return result

                self.initial_image = self.capture_frame()

                # 캡처 실패 시 카메라 재초기화 후 재시도
                if self.initial_image is None:
                    print(f"  {YELLOW}[Warning] Capture failed, reinitializing camera...{RESET}")
                    pc = self._get_pipeline_camera()
                    pc.camera_manager = self.camera_manager
                    if pc.force_recovery():
                        self.camera = pc.camera
                        self.camera_manager = pc.camera_manager
                        time.sleep(0.5)
                        self.initial_image = self.capture_frame()

                if self.initial_image is None:
                    print(f"{RED}[Error] Failed to capture image{RESET}")
                    return result

                self.initial_image_resolution = (self.initial_image.shape[1], self.initial_image.shape[0])
                print(f"  Image captured ({self.initial_image_resolution[0]}x{self.initial_image_resolution[1]})")

                # [즉시 저장] Initial 이미지
                initial_path = Path(forward_dir) / "initial_state.jpg"
                cv2.imwrite(str(initial_path), self.initial_image)
                self.forward_initial_image_path = str(initial_path)
                print(f"  Image saved: {initial_path}")

                # Detection 없이 빈 positions
                self.detected_positions = {}
                result['forward']['positions'] = self.detected_positions

                # Step 2: (스킵 — Step 1에서 이미 캡처됨)

                # Step 3: Forward 코드 생성 (multi-turn)
                # 코드 재사용: 캐싱된 코드가 있으면 T0~T2(검출)만 수행, T3(코드생성) 스킵
                use_cached = self.cached_forward_code is not None
                if use_cached:
                    print(f"\n{YELLOW}" + self._log(f"Detection only (reusing cached code, T3 skipped)...", step="Step 3/6") + f"{RESET}")
                    # T0~T2만 수행 (positions 갱신) — point 라벨도 강제
                    self.generate_forward_code(
                        instruction, self.detected_positions,
                        image_path=str(initial_path),
                        skip_codegen=True,
                        canonical_labels=self.cached_forward_keys,
                        canonical_point_labels=getattr(self, '_cached_point_labels', None),
                    )
                    # key 일치 확인
                    if self._can_reuse_code(self.cached_forward_code, self.cached_forward_keys, self.detected_positions):
                        self.generated_code = self.cached_forward_code
                        print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET}")
                    else:
                        missing = set(self.cached_forward_keys) - set(self.detected_positions.keys())
                        print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET}")
                        self.cached_forward_code = None
                        self.cached_forward_keys = []
                        self.generated_code = self.generate_forward_code(
                            instruction, self.detected_positions,
                            image_path=str(initial_path),
                        )
                else:
                    print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, multi-turn)...", step="Step 3/6") + f"{RESET}")
                    self.generated_code = self.generate_forward_code(
                        instruction, self.detected_positions,
                        image_path=str(initial_path),
                    )

            else:
                # ============================================================
                # Single-turn: 기존 Grounding DINO Detection → 코드 생성
                # ============================================================

                # Step 1: 객체 검출
                # Note: Recording 카메라가 있으면 run_detection에서 자동 공유
                print(f"\n{YELLOW}" + self._log(f"Detecting objects: {objects}", step="Step 1/6") + f"{RESET}")
                if visualize_detection:
                    print("  (Visualization mode)")
                else:
                    if not self.initialize_camera():
                        print(f"{RED}[Error] Camera initialization failed{RESET}")
                        return result

                self.detected_positions = self.run_detection(
                    queries=objects,
                    timeout=detection_timeout,
                    visualize=visualize_detection,
                )
                result['forward']['positions'] = self.detected_positions

                # [즉시 저장] Detection 이미지
                if self.detection_image is not None:
                    detection_path = Path(forward_dir) / "detection_result.jpg"
                    cv2.imwrite(str(detection_path), self.detection_image)  # Already BGR
                    print(f"  Detection image saved: {detection_path}")

                # 검출 결과 검증 1: 객체 미발견 체크
                not_found = [k for k, v in self.detected_positions.items() if v is None]
                if not_found:
                    print(f"{RED}[Error] Objects not detected: {not_found}{RESET}")
                    return result

                # 첫 에피소드의 검출 위치 저장 (original reset mode용)
                if self.first_episode_positions is None:
                    import copy
                    self.first_episode_positions = copy.deepcopy(self.detected_positions)
                    print(f"  {GREEN}[First Episode] Initial positions saved for 'original' reset mode{RESET}")

                # 검출 결과 검증 2: Workspace 범위 체크
                print(f"\n{YELLOW}" + self._log("Checking workspace bounds...", tag="Validation") + f"{RESET}")
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from lerobot_cap.workspace import BaseWorkspace

                workspace = BaseWorkspace()
                print(f"  Workspace: reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

                critical_error = False
                for obj_name, obj_info in self.detected_positions.items():
                    if obj_info is None:
                        continue
                    pos = obj_info.get("position") if isinstance(obj_info, dict) else obj_info
                    if pos is None:
                        continue
                    position_m = np.array([pos[0], pos[1], pos[2]])

                    if not workspace.is_reachable(position_m):
                        print(f"{RED}[CRITICAL] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m{RESET}")
                        print(f"{RED}  Outside reach limits: [{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m{RESET}")
                        critical_error = True
                    else:
                        print(f"  {GREEN}✓ '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m - OK{RESET}")

                if critical_error:
                    print(f"\n{RED}[Error] Critical workspace violation detected. Terminating pipeline.{RESET}")
                    return result

                # Step 2: Initial 이미지 캡처
                print(f"\n{YELLOW}" + self._log("Capturing initial state...", step="Step 2/6") + f"{RESET}")
                if self.initial_image is None:
                    self.initial_image = self.capture_frame()
                if self.initial_image is not None:
                    # 해상도 저장 (Judge용)
                    self.initial_image_resolution = (self.initial_image.shape[1], self.initial_image.shape[0])
                    print(f"  Initial image captured ({self.initial_image_resolution[0]}x{self.initial_image_resolution[1]})")
                    # [즉시 저장] Initial 이미지
                    initial_path = Path(forward_dir) / "initial_state.jpg"
                    cv2.imwrite(str(initial_path), self.initial_image)  # Already BGR
                    self.forward_initial_image_path = str(initial_path)
                    print(f"  Initial image saved: {initial_path}")

                # Step 3: Forward 코드 생성 (single-turn)
                # 코드 재사용: 캐싱된 코드가 있고 key 일치하면 스킵
                if self._can_reuse_code(self.cached_forward_code, self.cached_forward_keys, self.detected_positions):
                    self.generated_code = self.cached_forward_code
                    print(f"\n{YELLOW}" + self._log(f"Reusing cached code (single-turn, T3 skipped)...", step="Step 3/6") + f"{RESET}")
                    print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET}")
                else:
                    if self.cached_forward_code is not None:
                        missing = set(self.cached_forward_keys) - set(self.detected_positions.keys())
                        print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET}")
                        self.cached_forward_code = None
                        self.cached_forward_keys = []
                    print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, single-turn)...", step="Step 3/6") + f"{RESET}")
                    self.generated_code = self.generate_forward_code(
                        instruction,
                        self.detected_positions,
                    )
            result['forward']['code'] = self.generated_code
            result['forward']['positions'] = self.detected_positions

            # 첫 에피소드의 검출 위치 저장 (original reset mode용)
            # multi-turn에서도 detected_positions가 갱신된 후 저장
            if self.first_episode_positions is None and self.detected_positions:
                import copy
                self.first_episode_positions = copy.deepcopy(self.detected_positions)
                GREEN_TMP = "\033[92m"
                RESET_TMP = "\033[0m"
                print(f"  {GREEN_TMP}[First Episode] Initial positions saved for 'original' reset mode{RESET_TMP}")
                for name, info in self.detected_positions.items():
                    if isinstance(info, dict) and "position" in info:
                        pos = info["position"]
                        print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

                # seed_01_setup 즉시 저장 (forward detection 직후)
                session_dir_path = Path(save_dir).parent if not use_timestamp_subdir else Path(result_dir)
                seed1_dir = session_dir_path / "seed_01_setup"
                seed1_dir.mkdir(parents=True, exist_ok=True)
                with open(str(seed1_dir / "seed_positions.json"), 'w') as f:
                    json.dump({"positions": self.first_episode_positions}, f, indent=2, default=str)
                print(f"  [SeedGen] seed_01 (initial) saved: {seed1_dir / 'seed_positions.json'}")

            # [즉시 저장] Generated code
            code_path = Path(forward_dir) / "generated_code.py"
            code_path.write_text(self.generated_code)
            print(f"  Generated code saved: {code_path}")

            # [즉시 저장] Multi-turn info + visualizations (if available)
            if self.multi_turn and self.multi_turn_info:
                from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
                save_multi_turn_info(forward_dir, self.multi_turn_info, phase="forward")
                save_turn_visualizations(forward_dir, self.multi_turn_info, self.initial_image, phase="forward")

            print("\n" + "-" * 40)
            print("Generated Forward Code (preview):")
            print("-" * 40)
            code_preview = self.generated_code[:600]
            if len(self.generated_code) > 600:
                code_preview += "\n... (truncated)"
            print(code_preview)
            print("-" * 40)

            # Step 4: Code Verification (LLM 기반 코드 검증)
            # 캐시된 코드를 재사용하는 경우 검증 스킵 (이미 이전에 검증됨)
            code_was_cached = (self.cached_forward_code is not None
                               and self.generated_code == self.cached_forward_code)
            if code_was_cached:
                print(f"\n{YELLOW}" + self._log("Skipping verification (cached code, already verified)...", step="Step 4/6", tag="Verify") + f"{RESET}")
            else:
                print(f"\n{YELLOW}" + self._log(f"Verifying generated code via LLM ({self.llm_model})...", step="Step 4/6", tag="Verify") + f"{RESET}")
                from verification import verify_generated_code

                max_verification_retries = 2
                for verify_attempt in range(1, max_verification_retries + 1):
                    passed, reason = verify_generated_code(
                        instruction=instruction,
                        generated_code=self.generated_code,
                        object_positions=self.detected_positions,
                        llm_model=self.llm_model,
                    )

                    if passed:
                        print(f"  {GREEN}[Verify] PASS{RESET}")
                        break
                    else:
                        print(f"  {RED}[Verify] FAIL (attempt {verify_attempt}/{max_verification_retries}): {reason}{RESET}")

                        if verify_attempt < max_verification_retries:
                            # 코드 재생성
                            print(f"  {YELLOW}[Verify] Regenerating code...{RESET}")
                            if self.multi_turn:
                                self.generated_code = self.generate_forward_code(
                                    instruction, self.detected_positions,
                                    image_path=self.forward_initial_image_path,
                                )
                            else:
                                self.generated_code = self.generate_forward_code(
                                    instruction, self.detected_positions,
                                )
                            result['forward']['code'] = self.generated_code

                            # 재생성된 코드 저장
                            code_path = Path(forward_dir) / "generated_code.py"
                            code_path.write_text(self.generated_code)
                            print(f"  {YELLOW}[Verify] Regenerated code saved: {code_path}{RESET}")
                        else:
                            # 최대 재시도 도달 — 현재 코드로 진행
                            print(f"  {YELLOW}[Verify] Max retries reached, proceeding with current code{RESET}")

            # Step 5: Forward 코드 실행
            print(f"\n{YELLOW}" + self._log(f"Executing forward code on Robot {self.robot_id}...", step="Step 5/6") + f"{RESET}")

            # 레코딩 모드: 에피소드 시작
            if self.record_dataset:
                self._start_episode_recording(task=instruction)

            import builtins
            builtins._current_execution_dir = forward_dir
            builtins._scene_summary = self.multi_turn_info.get("turn0_response", "") if self.multi_turn_info else ""

            forward_success = self.execute_code(self.generated_code, self.detected_positions)
            result['forward']['execution_success'] = forward_success

            if forward_success:
                print(f"  {GREEN}Forward execution SUCCESS{RESET}")
            else:
                print(f"  {RED}Forward execution FAILED{RESET}")

            # VLM pixel move 시각화 (pixel 좌표로 직접 이동한 경우)
            try:
                from skills.skills_lerobot import LeRobotSkills
                if (LeRobotSkills._last_instance
                        and hasattr(LeRobotSkills._last_instance, 'pixel_move_log')
                        and LeRobotSkills._last_instance.pixel_move_log):
                    grasp_img_path = str(Path(forward_dir) / "turn2_grasp_points.jpg")
                    if os.path.isfile(grasp_img_path):
                        self._visualize_pixel_moves(
                            grasp_img_path,
                            LeRobotSkills._last_instance.pixel_move_log,
                            str(Path(forward_dir) / "pixel_moves_overlay.jpg"),
                        )
            except Exception as e:
                print(f"  Warning: pixel move visualization failed: {e}")

            # Update llm_cost with detect_objects token usage
            try:
                runner = self._get_task_runner()
                from pipeline.save_logs import update_llm_cost_with_detect_usage
                update_llm_cost_with_detect_usage(forward_dir, runner.skills)
            except Exception as e:
                print(f"  Warning: detect_objects cost merge failed: {e}")

            # Step 5: Context 저장
            print(f"\n{YELLOW}" + self._log("Saving execution context...", step="Step 6/6") + f"{RESET}")
            from pipeline.save_logs import save_execution_context as _save_ec
            _save_ec(forward_dir, instruction, self.detected_positions,
                     self.generated_code, forward_success, robot_id=self.robot_id)

            # ================================================================
            # PHASE 2: JUDGE (EVALUATION)
            # ================================================================
            print(f"\n{MAGENTA}{BOLD}" + self._log("JUDGE (Forward Evaluation)") + f"{RESET}")
            print(MAGENTA + "-" * 70 + RESET)

            # Step 1: Final 이미지 캡처
            print(f"\n{YELLOW}" + self._log("Capturing final state...", step="Step 1/3", tag="Judge") + f"{RESET}")
            time.sleep(1.0)
            self.capture_final_image()
            if self.final_image is not None:
                print("  Final image captured")
                final_path = Path(forward_dir) / "final_state.jpg"
                cv2.imwrite(str(final_path), self.final_image)
                print(f"  Final image saved: {final_path}")

            # Save batch_info early (before judge) so resume can detect this episode
            batch_idx = getattr(self, '_current_batch_index', 0)
            slot = getattr(self, '_current_slot', 0)
            episode_root = str(Path(forward_dir).parent)
            batch_info = {
                "batch_seed_index": batch_idx + 1,
                "slot": slot,
                "judge": "PENDING",
            }
            bi_path = Path(episode_root) / "batch_info.json"
            bi_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bi_path, 'w') as f:
                json.dump(batch_info, f, indent=2)

            # Step 2: Judge 실행
            judge_prediction = "UNCERTAIN"
            print(f"\n{YELLOW}" + self._log(f"Running VLM Judge ({self.judge_model})...", step="Step 2/3", tag="Judge") + f"{RESET}")
            if self.initial_image is not None and self.final_image is not None:
                image_resolution = self.final_image_resolution or self.initial_image_resolution
                judge_result = self._run_forward_judge(
                    instruction=instruction,
                    initial_image=self.initial_image,
                    final_image=self.final_image,
                    object_positions=self.detected_positions,
                    executed_code=self.generated_code,
                    image_resolution=image_resolution,
                )
                result['judge'] = judge_result

                judge_prediction = judge_result.get('prediction', 'UNCERTAIN')
                reasoning = judge_result.get('reasoning', '')

                pred_color = GREEN if judge_prediction == "TRUE" else RED if judge_prediction == "FALSE" else YELLOW
                print(f"  Prediction: {pred_color}{judge_prediction}{RESET}")
                print(f"  Reasoning: {reasoning[:100]}...")
                # Judge 비용을 llm_cost.json에 추가
                try:
                    from judge.vlm import _call_gemini_vlm
                    judge_usage = getattr(_call_gemini_vlm, '_last_usage', None)
                    if judge_usage:
                        cost_path = Path(forward_dir) / "llm_cost.json"
                        if cost_path.exists():
                            with open(cost_path) as f:
                                cost_data = json.load(f)
                        else:
                            cost_data = {"phase": "forward"}
                        cost_data["judge"] = {
                            "model": judge_usage.get("model", self.judge_model),
                            "inference_time_s": judge_usage.get("inference_time_s", 0),
                            "input_tokens": judge_usage.get("in", 0),
                            "output_tokens": judge_usage.get("out", 0),
                            "total_tokens": judge_usage.get("total", 0),
                        }
                        # total에 judge 비용 합산
                        if "total" in cost_data:
                            cost_data["total"]["inference_time_s"] = round(
                                cost_data["total"]["inference_time_s"] + judge_usage.get("inference_time_s", 0), 2)
                            cost_data["total"]["input_tokens"] += judge_usage.get("in", 0)
                            cost_data["total"]["output_tokens"] += judge_usage.get("out", 0)
                            cost_data["total"]["total_tokens"] += judge_usage.get("total", 0)
                        with open(cost_path, 'w') as f:
                            json.dump(cost_data, f, indent=2)
                except Exception:
                    pass
            else:
                print(f"  {YELLOW}Skipped (missing images){RESET}")

            # 레코딩 모드: Judge 결과에 따라 에피소드 저장/폐기
            if self.record_dataset:
                should_discard = judge_prediction != "TRUE"
                self._end_episode_recording(discard=should_discard)

                if not should_discard:
                    # Skill recording 시각화 저장 (성공한 에피소드만)
                    try:
                        from record_dataset.visualize_skills import generate_skill_visualizations
                        dataset = self.dataset_recorder._dataset
                        dataset._ensure_hf_dataset_loaded()
                        episode_df = dataset.hf_dataset.to_pandas()
                        saved_viz = generate_skill_visualizations(
                            dataframe=episode_df,
                            save_dir=forward_dir,
                            episode_index=self.dataset_recorder.episode_count - 1,
                        )
                        if saved_viz:
                            print(f"  Skill visualizations saved: {len(saved_viz)} files")
                    except Exception as e:
                        import traceback
                        print(f"  Warning: Skill visualization failed: {e}")
                        traceback.print_exc()

                # Step 3: Judge UI 표시 (타임아웃 적용)
                print(f"\n{YELLOW}" + self._log(f"Displaying result ({self.judge_timeout_ms/1000:.1f}s timeout)...", step="Step 3/3", tag="Judge") + f"{RESET}")
                if self.initial_image is not None and self.final_image is not None:
                    from judge import save_judge_log

                    result_image = self.show_judge_ui(
                        instruction=instruction,
                        prediction=result['judge'].get('prediction', 'UNCERTAIN'),
                        reasoning=result['judge'].get('reasoning', ''),
                        positions=self.detected_positions,
                    )

                    # Judge 로그 저장 (forward 폴더에)
                    result['saved_files'] = save_judge_log(
                        save_dir=forward_dir,
                        initial_image=self.initial_image,
                        final_image=self.final_image,
                        instruction=instruction,
                        prediction=result['judge'].get('prediction', 'UNCERTAIN'),
                        reasoning=result['judge'].get('reasoning', ''),
                        object_positions=self.detected_positions,
                        executed_code=self.generated_code,
                        result_image=result_image,
                        detection_image=self.detection_image,
                    )
            # 코드 캐시 갱신: 실행 성공 + Judge!=FALSE이면 캐싱
            # 한 번이라도 TRUE가 나온 코드는 유지 (Judge=FALSE로 무효화하지 않음)
            judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
            should_cache = forward_success and judge_pred != 'FALSE'
            if should_cache:
                if self.cached_forward_code is None:
                    self.cached_forward_code = self.generated_code
                    self.cached_forward_keys = self._extract_position_keys(self.generated_code)
                    # point labels 캐시: {obj: [label1, label2, ...]}
                    self._cached_point_labels = {}
                    for name, info in self.detected_positions.items():
                        if isinstance(info, dict) and "points" in info:
                            self._cached_point_labels[name] = list(info["points"].keys())
                    print(f"  {GREEN}[CodeReuse] Forward code cached (keys: {self.cached_forward_keys}){RESET}")
                    if self._cached_point_labels:
                        print(f"  {GREEN}[CodeReuse] Point labels cached: {self._cached_point_labels}{RESET}")
            elif not forward_success:
                # 실행 자체가 실패한 경우만 캐시 무효화 (코드 자체의 문제)
                # Judge=FALSE는 detection/환경 문제일 수 있으므로 이전 성공 코드 유지
                if self.cached_forward_code is not None:
                    print(f"  {YELLOW}[CodeReuse] Cache invalidated (execution failed){RESET}")
                self.cached_forward_code = None
                self.cached_forward_keys = []
                self._cached_point_labels = None
            elif judge_pred == 'FALSE' and self.cached_forward_code is not None:
                print(f"  {YELLOW}[CodeReuse] Judge=FALSE but keeping cached code (previously validated){RESET}")

            # Forward 로깅 종료
            forward_log_path = forward_logger.stop()
            print(f"\n  Forward log saved to: {forward_log_path}")

            # Post-judge 콜백 (batch_info 저장 등, reset 전에 실행)
            if post_judge_callback is not None:
                post_judge_callback(result)

            # ================================================================
            # PHASE 3: RESET EXECUTION
            # ================================================================
            if not skip_reset:
                # Switch phase for logging
                self.current_phase = "Reset"

                # Reset 로깅 시작
                reset_logger.start()

                print(f"\n{CYAN}{BOLD}" + self._log("RESET EXECUTION") + f"{RESET}")
                print(CYAN + "-" * 70 + RESET)

                # 카메라 종료 (Forward에서 사용하던 별도 카메라)
                self.shutdown_camera()

                # Reset target 결정 (콜백 전 — 현재 seed 사용, 콜백 후 갱신)
                if reset_target_positions is not None:
                    reset_original_positions = reset_target_positions
                    print(f"  Reset target: seed positions")
                elif self.first_episode_positions is not None:
                    reset_original_positions = self.first_episode_positions
                    print(f"  Reset target: first episode positions")
                else:
                    reset_original_positions = self.detected_positions
                    print(f"  Reset target: current detected positions")

                # 디버그: reset target 상세 출력
                for _name, _info in reset_original_positions.items():
                    _pos = _info.get("position") if isinstance(_info, dict) else _info
                    if _pos and len(_pos) >= 3:
                        print(f"    {_name}: [{_pos[0]:.4f}, {_pos[1]:.4f}, {_pos[2]:.4f}]")

                # Step 1 & 2: Reset 코드 생성
                multi_turn_str = "multi-turn VLM" if self.multi_turn else "single-turn"
                print(f"\n{YELLOW}" + self._log(f"Generating reset code ({multi_turn_str})...", step="Step 1/4") + f"{RESET}")
                try:

                    # Multi-turn 모드: 코드 생성 전에 current_state 이미지 캡처
                    reset_current_state_image_path = None
                    if self.multi_turn:
                        print(f"  [MultiTurn] Capturing current state for VLM...")
                        # Forward 후 카메라가 shutdown된 상태이므로 강제 재초기화
                        if not self.initialize_camera():
                            print(f"  {RED}Failed to initialize camera for reset VLM{RESET}")
                        time.sleep(0.3)
                        reset_current_frame = self.capture_frame()
                        if reset_current_frame is not None:
                            reset_current_state_image_path = str(Path(reset_dir) / "current_state.jpg")
                            cv2.imwrite(reset_current_state_image_path, reset_current_frame)
                            print(f"  Current state captured: {reset_current_state_image_path}")
                            # 이 이미지를 reset_initial_image로도 사용 (Judge용)
                            self.reset_initial_image = reset_current_frame
                            self.reset_initial_resolution = (reset_current_frame.shape[1], reset_current_frame.shape[0])

                    # Reset 코드 재사용 또는 새로 생성
                    if self.cached_reset_code is not None:
                        # 캐싱된 dict 참조 코드 재사용: 검출(T0~T2)만 수행
                        print(f"  {GREEN}[CodeReuse] Using cached reset code{RESET}")
                        # 검출로 current_positions 획득
                        self.generate_forward_code(
                            instruction, {},
                            image_path=reset_current_state_image_path,
                            skip_codegen=True,
                            canonical_labels=list(reset_original_positions.keys()),
                        )
                        current_positions = self.detected_positions
                        target_positions = reset_original_positions
                        reset_code = self.cached_reset_code
                        # 검출 결과를 reset_multi_turn_info에도 저장 (시각화/로그용)
                        self.reset_multi_turn_info = getattr(self, 'multi_turn_info', None)
                    else:
                        # LLM으로 새로 생성
                        reset_code, _, current_positions, _ = self.generate_reset_code(
                            original_instruction=instruction,
                            original_positions=reset_original_positions,
                            forward_spec=self.generated_spec,
                            forward_code=self.generated_code,
                            detection_timeout=detection_timeout,
                            visualize_detection=visualize_detection,
                            current_state_image_path=reset_current_state_image_path,
                        )
                        # target은 generate_reset_code의 반환값이 아닌
                        # 콜백으로 교체된 reset_original_positions를 직접 사용
                        target_positions = reset_original_positions

                    # 검출 완료 후 콜백: 실제 current_positions로 seed 생성
                    if pre_reset_callback is not None:
                        new_target = pre_reset_callback(current_positions=current_positions)
                        if new_target is not None:
                            reset_original_positions = new_target
                            target_positions = new_target

                    result['reset']['current_positions'] = current_positions
                    result['reset']['target_positions'] = target_positions
                    result['reset']['code'] = reset_code

                    # [즉시 저장] Reset generated code
                    reset_code_path = Path(reset_dir) / "generated_code.py"
                    reset_code_path.write_text(reset_code)
                    print(f"  Reset code saved: {reset_code_path}")

                    # [즉시 저장] Reset positions (current & target)
                    reset_positions_path = Path(reset_dir) / "positions.json"
                    reset_positions_data = {
                        "current_positions": current_positions,
                        "target_positions": target_positions,
                        "reset_mode": "original",
                    }
                    with open(reset_positions_path, 'w') as f:
                        json.dump(reset_positions_data, f, indent=2, default=str)
                    print(f"  Reset positions saved: {reset_positions_path}")

                    # LLM 비용 통계 저장 (reset)
                    try:
                        from code_gen_lerobot.reset_execution.code_gen import lerobot_reset_code_gen_multi_turn
                        reset_cost = getattr(lerobot_reset_code_gen_multi_turn, '_last_llm_cost', None)
                        if reset_cost:
                            cost_path = Path(reset_dir) / "llm_cost.json"
                            with open(cost_path, 'w') as f:
                                json.dump({"phase": "reset", **reset_cost}, f, indent=2)
                            print(f"  LLM cost saved: {cost_path}")
                    except Exception:
                        pass

                    # [즉시 저장] Reset multi-turn VLM 데이터
                    reset_mt = getattr(self, 'reset_multi_turn_info', None)
                    if reset_mt:
                        from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
                        save_multi_turn_info(reset_dir, reset_mt, phase="reset")
                        save_turn_visualizations(reset_dir, reset_mt, self.reset_initial_image, phase="reset")

                    print("\n" + "-" * 40)
                    print("Generated Reset Code (preview):")
                    print("-" * 40)
                    reset_preview = reset_code[:600]
                    if len(reset_code) > 600:
                        reset_preview += "\n... (truncated)"
                    print(reset_preview)
                    print("-" * 40)

                    # Step 2: Reset 초기 이미지 캡처
                    # multi-turn 모드에서는 이미 current_state로 캡처됨
                    if self.reset_initial_image is not None and self.multi_turn:
                        print(f"\n{YELLOW}" + self._log("Reset initial image already captured (multi-turn)", step="Step 2/4") + f"{RESET}")
                        # [즉시 저장] Reset initial 이미지 = forward 시작 전 이미지 (되돌려야 할 상태)
                        reset_initial_path = Path(reset_dir) / "initial_state.jpg"
                        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
                            import shutil
                            shutil.copy2(self.forward_initial_image_path, str(reset_initial_path))
                        else:
                            cv2.imwrite(str(reset_initial_path), self.reset_initial_image)
                        print(f"  Reset initial image saved: {reset_initial_path}")
                    else:
                        print(f"\n{YELLOW}" + self._log("Capturing reset initial state...", step="Step 2/4") + f"{RESET}")
                        # camera_manager가 있으면 재사용, 없으면 새로 생성
                        if not (self.camera_manager and self.camera_manager.is_connected):
                            if not self.initialize_camera():
                                print(f"  {RED}Failed to initialize camera{RESET}")
                        time.sleep(0.3)
                        self.reset_initial_image = self.capture_frame()
                        if self.reset_initial_image is not None:
                            # 해상도 저장 (Judge용)
                            self.reset_initial_resolution = (self.reset_initial_image.shape[1], self.reset_initial_image.shape[0])
                            print(f"  {GREEN}Reset initial image captured ({self.reset_initial_resolution[0]}x{self.reset_initial_resolution[1]}){RESET}")
                            # [즉시 저장] Reset initial 이미지
                            reset_initial_path = Path(reset_dir) / "initial_state.jpg"
                            cv2.imwrite(str(reset_initial_path), self.reset_initial_image)  # Already BGR
                            print(f"  Reset initial image saved: {reset_initial_path}")
                        else:
                            print(f"  {YELLOW}Warning: Failed to capture reset initial image{RESET}")

                    # Step 3: Reset 코드 실행 (recording 포함)
                    print(f"\n{YELLOW}" + self._log("Executing reset code...", step="Step 3/4") + f"{RESET}")
                    self.reset_code = reset_code

                    # Reset recording: forward recorder → reset recorder로 교체
                    if self.record_dataset and self.reset_dataset_recorder:
                        self._start_reset_episode_recording(target_positions)

                    import builtins
                    builtins._current_execution_dir = reset_dir
                    reset_mt = getattr(self, 'reset_multi_turn_info', None)
                    builtins._scene_summary = reset_mt.get("turn0_response", "") if reset_mt else ""

                    reset_success = self.execute_code(reset_code, current_positions, extra_globals={
                        "current_positions": current_positions,
                        "target_positions": target_positions,
                    })
                    result['reset']['execution_success'] = reset_success

                    # Reset recording 종료
                    if self.record_dataset and self.reset_dataset_recorder:
                        reset_judge_pred = result.get('reset_judge', {}).get('prediction', 'TRUE')
                        should_discard = not reset_success or reset_judge_pred == "FALSE"
                        self._end_reset_episode_recording(discard=should_discard)

                    if reset_success:
                        print(f"  {GREEN}Reset execution SUCCESS{RESET}")
                    else:
                        print(f"  {RED}Reset execution FAILED{RESET}")

                    # Step 4: Reset 최종 이미지 캡처
                    print(f"\n{YELLOW}" + self._log("Capturing reset final state...", step="Step 4/4") + f"{RESET}")
                    time.sleep(0.5)  # 로봇 정지 대기
                    # camera_manager가 있으면 재사용, 없으면 새로 생성
                    if not (self.camera_manager and self.camera_manager.is_connected):
                        if self.camera is None:
                            self.initialize_camera()
                            time.sleep(0.3)
                    self.reset_final_image = self.capture_frame()
                    if self.reset_final_image is not None:
                        # 해상도 저장 (Judge용)
                        self.reset_final_resolution = (self.reset_final_image.shape[1], self.reset_final_image.shape[0])
                        print(f"  {GREEN}Reset final image captured ({self.reset_final_resolution[0]}x{self.reset_final_resolution[1]}){RESET}")
                        # [즉시 저장] Reset final 이미지
                        reset_final_path = Path(reset_dir) / "final_state.jpg"
                        cv2.imwrite(str(reset_final_path), self.reset_final_image)  # Already BGR
                        print(f"  Reset final image saved: {reset_final_path}")
                    else:
                        print(f"  {YELLOW}Warning: Failed to capture reset final image{RESET}")

                    # Step 5: Reset Judge 실행
                    print(f"\n{CYAN}" + self._log("Evaluating reset result...", tag="Judge") + f"{RESET}")
                    reset_image_resolution = self.reset_final_resolution or self.reset_initial_resolution
                    reset_judge_result = self._run_reset_judge(
                        reset_mode="original",
                        current_positions=current_positions,
                        target_positions=target_positions,
                        initial_image=self.reset_initial_image,
                        final_image=self.reset_final_image,
                        executed_code=reset_code,
                        original_instruction=instruction,
                        image_resolution=reset_image_resolution,
                    )
                    result['reset_judge'] = reset_judge_result

                    rj_pred = reset_judge_result.get('prediction', 'UNCERTAIN')
                    pred_color = GREEN if rj_pred == "TRUE" else RED if rj_pred == "FALSE" else YELLOW
                    print(f"  Prediction: {pred_color}{rj_pred}{RESET}")
                    rj_reasoning = reset_judge_result.get('reasoning', '')
                    if rj_reasoning:
                        reasoning_preview = rj_reasoning[:200]
                        if len(rj_reasoning) > 200:
                            reasoning_preview += "..."
                        print(f"  Reasoning: {reasoning_preview}")

                    # Reset Judge UI 표시
                    if self.reset_initial_image is not None and self.reset_final_image is not None:
                        self.show_reset_judge_ui(
                            reset_mode="original",
                            prediction=rj_pred,
                            reasoning=rj_reasoning,
                            current_positions=current_positions,
                            target_positions=target_positions,
                        )

                    # Reset judge 로그 저장
                    reset_log = {
                        'reset_mode': "original",
                        'current_positions': current_positions,
                        'target_positions': target_positions,
                        'prediction': rj_pred,
                        'reasoning': rj_reasoning,
                        'execution_success': result['reset']['execution_success'],
                    }
                    reset_log_path = Path(reset_dir) / "reset_judge_result.json"
                    with open(reset_log_path, 'w') as f:
                        json.dump(reset_log, f, indent=2, default=str)
                    print(f"  Reset judge result saved to: {reset_log_path}")

                    # Reset 코드를 캐싱 (dict 참조 방식 + 로컬 재정의 없음)
                    reset_code_text = result['reset'].get('code', '')
                    if result['reset']['execution_success'] and reset_code_text:
                        # 주석(#)이 아닌 실제 코드 라인에서 dict 참조 확인
                        code_lines = [ln.strip() for ln in reset_code_text.splitlines()
                                      if ln.strip() and not ln.strip().startswith('#')]
                        has_cur_ref = any('current_positions[' in ln for ln in code_lines)
                        has_tgt_ref = any('target_positions[' in ln for ln in code_lines)
                        # 로컬 재정의가 있으면 캐시 거부 (exec_globals shadow 방지)
                        has_cur_redef = any(ln.startswith('current_positions') and '=' in ln and '{' in ln
                                           for ln in code_lines)
                        has_tgt_redef = any(ln.startswith('target_positions') and '=' in ln and '{' in ln
                                           for ln in code_lines)
                        if has_cur_ref and has_tgt_ref and not has_cur_redef and not has_tgt_redef:
                            self.cached_reset_code = reset_code_text
                        elif has_cur_redef or has_tgt_redef:
                            print(f"  {YELLOW}[Cache] Reset code has hardcoded position defs — not caching{RESET}")

                except Exception as e:
                    print(f"{RED}[Error] Reset failed: {e}{RESET}")
                    import traceback
                    traceback.print_exc()

                # Reset 로깅 종료
                reset_txt_log_path = reset_logger.stop()
                print(f"\n  Reset log saved to: {reset_txt_log_path}")
            else:
                print(f"\n{YELLOW}[PHASE 3] RESET EXECUTION - Skipped{RESET}")

            # ================================================================
            # SUMMARY
            # ================================================================
            self._print_summary(result, skip_reset)

            return result

        finally:
            self.shutdown_camera()

    def _visualize_turn1(
        self,
        image: np.ndarray,
        turn1_parsed,
        save_path: str,
        turn1_raw: str = "",
    ) -> None:
        """Turn 1 bbox 시각화 — test6 스타일 (녹색 bbox + 라벨)"""
        obj_list = None
        if isinstance(turn1_parsed, list):
            obj_list = turn1_parsed
        elif isinstance(turn1_parsed, dict) and "objects" in turn1_parsed:
            obj_list = turn1_parsed["objects"]

        if not obj_list:
            print(f"  [Visualize] Turn 1: No objects to draw")
            return

        img_h, img_w = image.shape[:2]

        for obj in obj_list:
            label = obj.get("label") or obj.get("name", "?")
            box = obj.get("box_2d") or obj.get("bbox_pixel")
            if box and len(box) == 4:
                ymin, xmin, ymax, xmax = box
                x1 = int(xmin * img_w / 1000)
                y1 = int(ymin * img_h / 1000)
                x2 = int(xmax * img_w / 1000)
                y2 = int(ymax * img_h / 1000)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1 + 2, y1 - 4), font, 0.5, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn 1 visualization saved: {save_path}")

    def _visualize_turn2(
        self,
        image: np.ndarray,
        turn1_parsed,
        turn2_parsed,
        save_path: str,
    ) -> None:
        """Turn 2 결과 시각화: bbox + grasp/interaction 포인트 (test6 스타일)

        Args:
            image: BGR 이미지 (copy)
            turn1_parsed: Turn 1 parsed 데이터 (bbox 리스트)
            turn2_parsed: Turn 2 parsed 데이터 ({"grasp_points": [...]})
            save_path: 저장 경로
        """
        img_h, img_w = image.shape[:2]

        # Turn 1 bbox 그리기
        obj_list = None
        if isinstance(turn1_parsed, list):
            obj_list = turn1_parsed
        elif isinstance(turn1_parsed, dict) and "objects" in turn1_parsed:
            obj_list = turn1_parsed["objects"]

        if obj_list:
            for obj in obj_list:
                box = obj.get("box_2d") or obj.get("bbox_pixel")
                label = obj.get("label", "")
                if box and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * img_w / 1000)
                    y1 = int(ymin * img_h / 1000)
                    x2 = int(xmax * img_w / 1000)
                    y2 = int(ymax * img_h / 1000)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Critical points 그리기 (grasp: 녹색 원, interaction: 빨간 X)
        ROLE_COLORS = {
            "grasp": (0, 255, 0),       # green
            "pick": (0, 255, 0),         # green (하위호환)
            "interaction": (0, 0, 255),  # red
            "place": (255, 0, 0),        # blue (하위호환)
        }

        grasp_points = []
        if isinstance(turn2_parsed, dict) and "grasp_points" in turn2_parsed:
            grasp_points = turn2_parsed["grasp_points"]

        if not grasp_points:
            print(f"  [Visualize] Turn 2: No points to draw")
            return

        for i, gp in enumerate(grasp_points):
            name = gp.get("object_name", "unknown")
            sub_label = gp.get("label", "")
            role = gp.get("role", "grasp")
            pixel = gp.get("point_pixel")

            if not pixel or len(pixel) != 2:
                continue

            px = int(pixel[1] * img_w / 1000)
            py = int(pixel[0] * img_h / 1000)
            color = ROLE_COLORS.get(role, (255, 255, 255))

            # 마커: grasp → green dot, interaction → red dot
            cv2.circle(image, (px, py), 3, color, -1)
            cv2.circle(image, (px, py), 3, (0, 0, 0), 1)

            # 라벨
            marker_label = f"{name}: {sub_label}" if sub_label else name
            (tw, th), _ = cv2.getTextSize(marker_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # 짝수/홀수로 텍스트 위치 교차 (겹침 방지)
            if i % 2 == 0:
                text_x = min(px + 15, img_w - tw - 5)
                text_y = max(py - 8, th + 5)
            else:
                text_x = max(px - tw - 15, 2)
                text_y = min(py + 12, img_h - 5)

            cv2.rectangle(image, (text_x - 2, text_y - th - 4),
                          (text_x + tw + 2, text_y + 4), color, -1)
            cv2.putText(image, marker_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn 2 visualization saved: {save_path}")

    def _visualize_pixel_moves(
        self,
        base_image_path: str,
        pixel_move_log: list,
        save_path: str,
    ) -> None:
        """VLM이 지정한 pixel 위치들을 grasp_points 이미지 위에 시각화.

        Args:
            base_image_path: turn2_grasp_points.jpg 경로 (이미 bbox+grasp가 그려진 이미지)
            pixel_move_log: LeRobotSkills.pixel_move_log 리스트
                           [{"pixel": [u, v], "target_name": str, "skill_description": str}, ...]
            save_path: 저장 경로
        """
        if not pixel_move_log:
            return

        image = cv2.imread(base_image_path)
        if image is None:
            print(f"  [Visualize] Cannot read base image: {base_image_path}")
            return

        img_h, img_w = image.shape[:2]

        for i, entry in enumerate(pixel_move_log):
            px, py = entry["pixel"]
            name = entry.get("target_name", "")
            desc = entry.get("skill_description", "")

            # 마커: 시안 다이아몬드 (기존 grasp point와 구별)
            color = (255, 255, 0)  # cyan (BGR)
            pts = np.array([
                [px, py - 6], [px + 6, py], [px, py + 6], [px - 6, py]
            ], dtype=np.int32)
            cv2.polylines(image, [pts], True, color, 2)
            cv2.circle(image, (px, py), 2, color, -1)

            # 라벨
            label = name if name else desc
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                text_x = min(px + 10, img_w - tw - 5)
                text_y = max(py - 5, th + 5)
                cv2.rectangle(image, (text_x - 2, text_y - th - 2),
                              (text_x + tw + 2, text_y + 2), color, -1)
                cv2.putText(image, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Pixel move visualization saved: {save_path}")

    def _visualize_turn_test(
        self,
        image: np.ndarray,
        turn2_parsed,
        waypoints: list,
        save_path: str,
    ) -> None:
        """Turn Test 결과 시각화: interaction points + waypoint trajectory

        Args:
            image: BGR 이미지 (copy)
            turn2_parsed: Turn 2 parsed 데이터 (interaction point 표시용)
            waypoints: turn_test_waypoints 리스트 [{"py", "px", "label", ...}, ...]
            save_path: 저장 경로
        """
        img_h, img_w = image.shape[:2]

        WAYPOINT_COLOR = (255, 165, 0)  # orange (BGR)
        LINE_COLOR = (255, 200, 100)    # light blue-ish line
        INTERACTION_COLOR = (0, 0, 255) # red

        # Draw interaction points from Turn 2 (for reference)
        if isinstance(turn2_parsed, dict) and "grasp_points" in turn2_parsed:
            for gp in turn2_parsed["grasp_points"]:
                if gp.get("role") != "interaction":
                    continue
                pixel = gp.get("point_pixel")
                if not pixel or len(pixel) != 2:
                    continue
                ipx = int(pixel[1] * img_w / 1000)
                ipy = int(pixel[0] * img_h / 1000)
                cv2.circle(image, (ipx, ipy), 5, INTERACTION_COLOR, -1)
                cv2.circle(image, (ipx, ipy), 5, (0, 0, 0), 1)

        if not waypoints:
            print(f"  [Visualize] Turn Test: No waypoints to draw")
            return

        # Collect pixel coords for line drawing
        wp_pixels = []
        for wp in waypoints:
            wy = wp.get("py", 0)
            wx = wp.get("px", 0)
            wp_pixels.append((wx, wy))

        # Draw lines between consecutive waypoints
        for i in range(len(wp_pixels) - 1):
            cv2.line(image, wp_pixels[i], wp_pixels[i + 1], LINE_COLOR, 2)

        # Draw waypoint dots and labels
        for i, (wp, (wx, wy)) in enumerate(zip(waypoints, wp_pixels)):
            label = wp.get("label", f"wp{i}")

            cv2.circle(image, (wx, wy), 4, WAYPOINT_COLOR, -1)
            cv2.circle(image, (wx, wy), 4, (0, 0, 0), 1)

            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            text_x = min(wx + 8, img_w - tw - 5)
            text_y = max(wy - 6, th + 5)
            cv2.rectangle(image, (text_x - 2, text_y - th - 2),
                          (text_x + tw + 2, text_y + 2), WAYPOINT_COLOR, -1)
            cv2.putText(image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn Test visualization saved: {save_path}")

    def _update_results(self, all_results: Dict, result: Dict, episode_num: int, skip_reset: bool) -> None:
        """에피소드 결과를 all_results에 추가"""
        all_results['episodes'].append({
            'episode': episode_num, 'result': result, 'success': True, 'error': None,
        })
        judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
        if result['forward']['execution_success']:
            all_results['summary']['forward_success'] += 1
        if judge_pred == 'TRUE':
            all_results['summary']['forward_judge_true'] += 1
        elif judge_pred == 'FALSE':
            all_results['summary']['forward_judge_false'] += 1
        if not skip_reset:
            if result['reset']['execution_success']:
                all_results['summary']['reset_success'] += 1
            rj_pred = result['reset_judge'].get('prediction', 'UNCERTAIN')
            if rj_pred == 'TRUE':
                all_results['summary']['reset_judge_true'] += 1
            elif rj_pred == 'FALSE':
                all_results['summary']['reset_judge_false'] += 1

    def _print_summary(self, result: Dict, skip_reset: bool) -> None:
        """결과 요약 출력"""
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print("\n" + CYAN + "=" * 70 + RESET)
        print(CYAN + BOLD + "PIPELINE COMPLETE".center(70) + RESET)
        print(CYAN + "=" * 70 + RESET)

        forward_status = result['forward']['execution_success']
        forward_color = GREEN if forward_status else RED
        print(f"  Forward: {forward_color}{'SUCCESS' if forward_status else 'FAILED'}{RESET}")

        judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
        judge_color = GREEN if judge_pred == "TRUE" else RED if judge_pred == "FALSE" else YELLOW
        print(f"  Judge:   {judge_color}{judge_pred}{RESET}")

        if not skip_reset:
            reset_status = result['reset']['execution_success']
            reset_color = GREEN if reset_status else RED
            print(f"  Reset:   {reset_color}{'SUCCESS' if reset_status else 'FAILED'}{RESET}")

            rj_pred = result['reset_judge'].get('prediction', 'UNCERTAIN')
            rj_color = GREEN if rj_pred == "TRUE" else RED if rj_pred == "FALSE" else YELLOW
            print(f"  Reset Judge: {rj_color}{rj_pred}{RESET}")
        else:
            print(f"  Reset:   {YELLOW}SKIPPED{RESET}")

        print(CYAN + "=" * 70 + RESET)

    def _generate_seed_positions(self, session_dir: str, seed_index: int, current_positions: Dict = None) -> Optional[Dict]:
        """
        새 seed 위치 생성: 랜덤 위치 생성 → IK dry_run 검증.

        first_episode_positions를 기반으로 랜덤 위치를 생성합니다.
        first_episode_positions는 변경하지 않습니다.

        Args:
            session_dir: 세션 디렉토리
            seed_index: 시드 인덱스
            current_positions: 실제 검출된 현재 위치 (dry-run용).
                              None이면 candidate를 current로 사용 (fallback).

        Returns:
            성공 시 새 positions dict, 실패 시 None
        """
        from code_gen_lerobot.reset_execution.workspace import (
            generate_random_positions, classify_objects, ResetWorkspace,
        )

        if self.first_episode_positions is None:
            print("  [SeedGen] No first_episode_positions, cannot generate")
            return None

        save_dir = str(Path(session_dir) / f"seed_{seed_index+1:02d}_setup")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        grippable, obstacles = classify_objects(self.first_episode_positions)

        # pix2robot 로드
        pix2robot = None
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            calib_path = Path(__file__).parent / "robot_configs" / "pix2robot_matrices" / f"robot{self.robot_id}_pix2robot_data.npz"
            if calib_path.exists():
                pix2robot = Pix2RobotCalibrator(robot_id=self.robot_id)
                if not pix2robot.load(str(calib_path)):
                    pix2robot = None
        except Exception:
            pass

        # Workspace
        kin_engine = None
        try:
            from lerobot_cap.kinematics.engine import KinematicsEngine
            urdf_path = Path(__file__).parent / "assets" / "urdf" / f"so101_robot{self.robot_id}.urdf"
            if urdf_path.exists():
                kin_engine = KinematicsEngine(str(urdf_path))
        except Exception:
            pass
        workspace = ResetWorkspace(kinematics_engine=kin_engine)

        # 과거 시드 위치: 같은 객체끼리만 겹침 비교하도록 _pseed 키 사용
        # first_episode_positions도 과거 seed (seed_1)이므로 포함
        all_initial = {}
        for name, info in self.first_episode_positions.items():
            all_initial[f"{name}_pseed_init"] = info
        for i, prev_positions in enumerate(self._all_previous_seed_positions):
            for name, info in prev_positions.items():
                all_initial[f"{name}_pseed{i}"] = info

        # Free state EE 주변 제외 영역 계산 (충돌 방지)
        FREE_STATE_EXCLUSION_RADIUS = 0.08  # 8cm
        exclusion_zones = []
        try:
            from lerobot_cap.kinematics import load_calibration_limits as _load_cl
            free_state_path = Path(__file__).parent / "robot_configs" / "free_state" / f"robot{self.robot_id}_free_state.json"
            if free_state_path.exists():
                with open(free_state_path) as f:
                    free_norm = np.array(json.load(f)["initial_state_normalized"])
                _cl = _load_cl(
                    str(Path(__file__).parent / "robot_configs" / "motor_calibration" / "so101" / f"robot{self.robot_id}_calibration.json"),
                    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                )
                free_rad = _cl.normalized_to_radians(free_norm)
                free_ee = kin_engine.get_ee_position(free_rad)
                exclusion_zones.append({
                    "center": [float(free_ee[0]), float(free_ee[1])],
                    "radius": FREE_STATE_EXCLUSION_RADIUS,
                })
                print(f"  [SeedGen] Free state exclusion: center=[{free_ee[0]:.3f}, {free_ee[1]:.3f}], radius={FREE_STATE_EXCLUSION_RADIUS}m")
        except Exception as e:
            print(f"  [SeedGen] Warning: Could not compute free state exclusion: {e}")

        # 랜덤 위치 생성 + dry_run 검증 (최대 10회 재시도)
        reset_code = self.cached_reset_code
        accepted_positions = None

        for attempt in range(10):
            random_targets = generate_random_positions(
                grippable_objects=grippable,
                obstacle_objects=obstacles,
                initial_positions=all_initial,
                workspace=workspace,
                pix2robot=pix2robot,
                current_positions=current_positions,
                exclusion_zones=exclusion_zones,
                resetspace=self.resetspace,
            )
            if not random_targets:
                print(f"  [SeedGen] Attempt {attempt+1}: position generation failed, retrying...")
                continue

            candidate = self._build_batch_positions(random_targets, obstacles, pix2robot=pix2robot)

            # dry_run 검증: 캐싱된 reset 코드에 실제 current_positions + candidate target 주입
            if reset_code is not None:
                print(f"  [SeedGen] Attempt {attempt+1}: dry-run validating...")
                dry_run_current = current_positions if current_positions else candidate
                if self.dry_run_code(reset_code, candidate, extra_globals={
                    "current_positions": dry_run_current,
                    "target_positions": candidate,
                }):
                    print(f"  [SeedGen] Attempt {attempt+1}: dry-run PASSED")
                    accepted_positions = candidate
                    break
                else:
                    print(f"  [SeedGen] Attempt {attempt+1}: dry-run FAILED, retrying...")
            else:
                # reset 코드 없으면 기본 IK 검증만으로 채택
                accepted_positions = candidate
                break
        else:
            print(f"  [SeedGen] All 10 attempts failed")
            return None

        new_positions = accepted_positions

        print(f"  [SeedGen] seed_{seed_index+1} positions:")
        for name, info in new_positions.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos and len(pos) >= 3:
                print(f"    {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        # 과거 시드 위치에 추가 (다음 시드 생성 시 겹침 방지) — grippable만
        from code_gen_lerobot.reset_execution.workspace import is_grippable as _is_grip
        self._all_previous_seed_positions.append(
            {name: info for name, info in new_positions.items()
             if isinstance(info, dict) and _is_grip(info.get("bbox_px"))}
        )

        # 시각화: workspace + 과거 시드 bbox + 새 시드 bbox
        self._visualize_seed_positions(
            save_dir=save_dir,
            new_positions=new_positions,
            seed_index=seed_index,
            pix2robot=pix2robot,
        )

        # 로그 저장
        with open(str(Path(save_dir) / "seed_positions.json"), 'w') as f:
            json.dump({"seed_index": seed_index, "positions": new_positions}, f, indent=2, default=str)

        return new_positions

    def _visualize_seed_positions(
        self,
        save_dir: str,
        new_positions: Dict,
        seed_index: int,
        pix2robot=None,
    ):
        """
        시드 위치 시각화: workspace 위에 과거/현재 시드 bbox를 그림.

        - workspace 도넛 마스크 + 가장자리 마진
        - 과거 시드 (초기 포함): 회색 계열 bbox (seed 번호 라벨)
        - 장애물: 빨간색 bbox
        - 새 시드: 초록색 bbox (굵게)
        """
        import cv2
        from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image, _get_bbox_px, is_grippable as _is_grippable

        # 초기 이미지 로드
        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
            base_img = cv2.imread(self.forward_initial_image_path)
        else:
            base_img = np.zeros((480, 640, 3), dtype=np.uint8) + 60

        # workspace 시각화 베이스
        result = draw_workspace_on_image(base_img, robot_id=self.robot_id, pix2robot_calibrator=pix2robot, resetspace=self.resetspace)

        def _draw_bbox(img, center_px, bbox_px, color, thickness, label=""):
            hw, hh = bbox_px[0] // 2, bbox_px[1] // 2
            cu, cv = int(center_px[0]), int(center_px[1])
            cv2.rectangle(img, (cu - hw, cv - hh), (cu + hw, cv + hh), color, thickness)
            if label:
                cv2.putText(img, label, (cu - hw, cv - hh - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # 장애물 (빨간색)
        if self.first_episode_positions:
            for name, info in self.first_episode_positions.items():
                if not isinstance(info, dict):
                    continue
                if not _is_grippable(info.get("bbox_px")):
                    pos = info.get("position")
                    bbox = _get_bbox_px(info)
                    if pos and pix2robot:
                        try:
                            px = pix2robot.robot_to_pixel(pos[0], pos[1])
                            _draw_bbox(result, px, bbox, (0, 0, 255), 2, f"obs:{name}")
                        except Exception:
                            pass

        # 과거 시드 — 모두 회색 계열 (초기 위치 = s0)
        PAST_COLORS = [
            (150, 150, 150),  # 회색
            (180, 130, 180),  # 보라
            (130, 180, 180),  # 청록
            (180, 180, 130),  # 올리브
            (130, 130, 180),  # 남색
        ]
        for i, prev in enumerate(self._all_previous_seed_positions):
            color = PAST_COLORS[i % len(PAST_COLORS)]
            for name, info in prev.items():
                if not isinstance(info, dict):
                    continue
                pos = info.get("position")
                bbox = _get_bbox_px(info)
                if pos and pix2robot:
                    try:
                        px = pix2robot.robot_to_pixel(pos[0], pos[1])
                        _draw_bbox(result, px, bbox, color, 1, f"s{i+1}:{name}")
                    except Exception:
                        pass

        # 새 시드 (초록색, 굵게) — grippable만
        for name, info in new_positions.items():
            if not isinstance(info, dict):
                continue
            if not _is_grippable(info.get("bbox_px")):
                continue
            pos = info.get("position")
            bbox = _get_bbox_px(info)
            if pos and pix2robot:
                try:
                    px = pix2robot.robot_to_pixel(pos[0], pos[1])
                    _draw_bbox(result, px, bbox, (0, 255, 0), 3, f"NEW s{seed_index+1}:{name}")
                except Exception:
                    pass

        # 범례
        img_h = result.shape[0]
        cv2.putText(result, f"Seed {seed_index+1} | Past: {len(self._all_previous_seed_positions)} seeds",
                    (10, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = str(Path(save_dir) / "seed_visualization.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [SeedGen] Visualization saved: {out_path}")

        # ── 객체 종류별 포인트 시각화 ──
        self._visualize_seed_points(save_dir, new_positions, seed_index, pix2robot)

    def _visualize_seed_points(
        self,
        save_dir: str,
        new_positions: Dict,
        seed_index: int,
        pix2robot=None,
    ):
        """
        객체 종류별 색상으로 모든 시드의 위치를 점으로 시각화.

        각 객체에 고유 색상을 할당하고, 과거 시드 + 현재 시드의
        위치를 점(원)으로 표시. 현재 시드는 크게, 과거는 작게.
        """
        import cv2
        from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image, is_grippable as _is_grippable

        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
            base_img = cv2.imread(self.forward_initial_image_path)
        else:
            base_img = np.zeros((480, 640, 3), dtype=np.uint8) + 60

        result = draw_workspace_on_image(base_img, robot_id=self.robot_id, pix2robot_calibrator=pix2robot, resetspace=self.resetspace)

        # 객체 이름 수집 (grippable만)
        all_obj_names = set()
        if self.first_episode_positions:
            for name, info in self.first_episode_positions.items():
                if isinstance(info, dict) and _is_grippable(info.get("bbox_px")):
                    all_obj_names.add(name)
        for name in new_positions:
            if isinstance(new_positions[name], dict) and _is_grippable(new_positions[name].get("bbox_px")):
                all_obj_names.add(name)
        all_obj_names = sorted(all_obj_names)

        # 객체별 고유 색상 (BGR)
        OBJ_COLORS = [
            (0, 0, 255),    # 빨강
            (255, 0, 0),    # 파랑
            (0, 200, 0),    # 초록
            (0, 200, 255),  # 노랑
            (255, 0, 255),  # 마젠타
            (255, 200, 0),  # 시안
            (0, 128, 255),  # 주황
            (200, 0, 128),  # 보라
        ]
        obj_color_map = {name: OBJ_COLORS[i % len(OBJ_COLORS)] for i, name in enumerate(all_obj_names)}

        # 과거 시드 (작은 점 + 시드 번호)
        for seed_i, prev in enumerate(self._all_previous_seed_positions):
            for name, info in prev.items():
                if name not in obj_color_map or not isinstance(info, dict):
                    continue
                pos = info.get("position")
                if pos and pix2robot:
                    try:
                        px, py = pix2robot.robot_to_pixel(pos[0], pos[1])
                        px, py = int(px), int(py)
                        color = obj_color_map[name]
                        cv2.circle(result, (px, py), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(result, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(result, f"s{seed_i+1}", (px + 7, py + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
                    except Exception:
                        pass

        # 현재 시드 (큰 점 + 라벨)
        for name, info in new_positions.items():
            if name not in obj_color_map or not isinstance(info, dict):
                continue
            if not _is_grippable(info.get("bbox_px")):
                continue
            pos = info.get("position")
            if pos and pix2robot:
                try:
                    px, py = pix2robot.robot_to_pixel(pos[0], pos[1])
                    px, py = int(px), int(py)
                    color = obj_color_map[name]
                    cv2.circle(result, (px, py), 9, color, -1, cv2.LINE_AA)
                    cv2.circle(result, (px, py), 9, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(result, f"s{seed_index+1}", (px + 11, py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                except Exception:
                    pass

        # 범례: 객체별 색상
        img_h = result.shape[0]
        y_offset = img_h - 15 * len(all_obj_names) - 10
        for i, name in enumerate(all_obj_names):
            color = obj_color_map[name]
            y = y_offset + i * 15
            cv2.circle(result, (15, y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(result, name, (25, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        cv2.putText(result, f"Seed {seed_index+1} | {len(self._all_previous_seed_positions)} past seeds",
                    (10, img_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = str(Path(save_dir) / "seed_visualization_point.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [SeedGen] Point visualization saved: {out_path}")

    def _build_batch_positions(self, random_targets: Dict, obstacles: Dict, pix2robot=None) -> Dict:
        """랜덤 타겟과 obstacle을 합쳐 positions dict 구성. pixel 필드도 갱신."""
        new_positions = {}
        for name, pos in random_targets.items():
            orig = self.first_episode_positions.get(name, {})
            if isinstance(orig, dict):
                new_positions[name] = {**orig, "position": pos}
                # pixel 필드를 새 position에 맞게 갱신
                if pix2robot is not None:
                    try:
                        new_positions[name]["pixel"] = list(pix2robot.robot_to_pixel(pos[0], pos[1]))
                    except Exception:
                        pass
                if "points" in orig:
                    new_positions[name]["points"] = {
                        pt_name: pos for pt_name in orig["points"]
                    }
            else:
                new_positions[name] = pos

        for name, info in obstacles.items():
            new_positions[name] = self.first_episode_positions.get(name, info)

        return new_positions

    def _load_resume_state(self, session_dir: str) -> Tuple[List[List[bool]], List[Optional[Dict]], List[bool]]:
        """이전 세션에서 상태 복원 (batch_info.json 기반).

        Returns:
            (batch_slots, seed_positions, batch_attempted):
                batch_slots[i] = [True/False, ...] 배치 i의 각 slot 성공 여부
                seed_positions[i] = 배치 i의 seed 위치 (없으면 None)
                batch_attempted[i] = True/False 배치 i가 한 번이라도 시도되었는지
        """
        import copy
        episodes_per_seed = max(1, self.total_episodes // self.num_random_seeds)
        batch_slots = [[False] * episodes_per_seed for _ in range(self.num_random_seeds)]
        batch_attempted = [False] * self.num_random_seeds
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # 1. first_episode_positions 복원 (ep_01 execution_context)
        ctx_path = Path(session_dir) / "episode_01" / "forward" / "execution_context.json"
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            self.first_episode_positions = ctx.get("object_positions", {})
            seed_positions[0] = copy.deepcopy(self.first_episode_positions)
            # forward_initial_image_path 복원 (시각화용)
            initial_img = Path(session_dir) / "episode_01" / "forward" / "initial_state.jpg"
            if initial_img.exists():
                self.forward_initial_image_path = str(initial_img)
            print(f"  [Resume] first_episode_positions restored from {ctx_path}")

        # 2. seed positions 복원
        for i in range(1, self.num_random_seeds):
            sp_path = Path(session_dir) / f"seed_{i+1:02d}_setup" / "seed_positions.json"
            if sp_path.exists():
                with open(sp_path) as f:
                    sp = json.load(f)
                seed_positions[i] = sp.get("positions", None)
                print(f"  [Resume] seed_{i+1} restored from {sp_path}")

        # 3. batch_info.json 기반으로 slot별 성공 여부 파악
        for ep_dir in sorted(Path(session_dir).glob("episode_*")):
            batch_info_path = ep_dir / "batch_info.json"
            if not batch_info_path.exists():
                continue
            with open(batch_info_path) as f:
                bi = json.load(f)
            # batch_seed_index (1-based) 또는 legacy batch_index (0-based) 호환
            if "batch_seed_index" in bi:
                batch_idx = bi["batch_seed_index"] - 1  # 1-based → 0-based
            else:
                batch_idx = bi.get("batch_index", -1)  # legacy 호환
            slot = bi.get("slot", -1)
            judge_pred = bi.get("judge", "")
            if 0 <= batch_idx < self.num_random_seeds and 0 <= slot < episodes_per_seed:
                batch_attempted[batch_idx] = True
                if judge_pred == "TRUE":
                    batch_slots[batch_idx][slot] = True

        # 4. cached_reset_code 복원 (이전 세션의 reset 코드를 찾아서 캐시)
        for ep_dir in sorted(Path(session_dir).glob("episode_*"), reverse=True):
            reset_code_path = ep_dir / "reset" / "generated_code.py"
            if reset_code_path.exists():
                code = reset_code_path.read_text().strip()
                if code:
                    self.cached_reset_code = code
                    self.cached_reset_keys = self._extract_position_keys(code)
                    print(f"  [Resume] cached_reset_code restored from {reset_code_path}")
                    break

        # 5. cached_forward_code 복원 (이전 세션의 forward 코드를 찾아서 캐시)
        for ep_dir in sorted(Path(session_dir).glob("episode_*"), reverse=True):
            fwd_code_path = ep_dir / "forward" / "generated_code.py"
            judge_path = ep_dir / "forward" / "judge_result.json"
            if fwd_code_path.exists() and judge_path.exists():
                with open(judge_path) as f:
                    jr = json.load(f)
                pred = jr.get("judge_result", {}).get("prediction", jr.get("prediction", ""))
                if pred == "TRUE":
                    code = fwd_code_path.read_text().strip()
                    if code:
                        self.cached_forward_code = code
                        self.cached_forward_keys = self._extract_position_keys(code)
                        # point labels 캐시 복원
                        ctx_path = ep_dir / "forward" / "execution_context.json"
                        if ctx_path.exists():
                            with open(ctx_path) as f:
                                ctx = json.load(f)
                            self._cached_point_labels = {}
                            for name, info in ctx.get("object_positions", {}).items():
                                if isinstance(info, dict) and "points" in info:
                                    self._cached_point_labels[name] = list(info["points"].keys())
                        print(f"  [Resume] cached_forward_code restored from {fwd_code_path}")
                        break

        # 6. 복원된 seed positions를 _all_previous_seed_positions에 등록
        for i, sp in enumerate(seed_positions):
            if sp is not None:
                self._all_previous_seed_positions.append(sp)

        # 요약 출력
        for i in range(self.num_random_seeds):
            done = sum(batch_slots[i])
            status = "DONE" if done >= episodes_per_seed else f"{done}/{episodes_per_seed}"
            seed_str = "loaded" if seed_positions[i] else "to generate"
            attempted_str = "attempted" if batch_attempted[i] else "new"
            print(f"    Batch {i+1}: {status} ({seed_str}, {attempted_str})")
        print(f"  [Resume] Previous seeds registered: {len(self._all_previous_seed_positions)}")
        return batch_slots, seed_positions, batch_attempted

    def _restore_to_seed(
        self,
        target_positions: Dict,
        instruction: str,
        detection_timeout: float = 10.0,
    ) -> bool:
        """물체를 seed 위치로 복원 (레코딩 없음).

        현재 물체 위치를 검출하고, target_positions로 이동하는 Reset 코드를 생성/실행.
        """
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET_C = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET_C}")
        print(f"{CYAN}{BOLD}  RESTORE TO SEED POSITION (no recording){RESET_C}")
        print(f"{CYAN}{BOLD}{'=' * 60}{RESET_C}")

        for name, info in target_positions.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos and len(pos) >= 3:
                print(f"  Target: {name} → [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        try:
            # 카메라로 현재 상태 캡처
            if not self.camera and not (self.camera_manager and self.camera_manager.is_connected):
                self.initialize_camera()
            time.sleep(0.5)
            current_frame = self.capture_frame()
            if current_frame is None:
                print(f"  {RED}Failed to capture frame{RESET_C}")
                return False

            import tempfile
            tmp_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(tmp_path, current_frame)
            if self.multi_turn and not self.forward_initial_image_path:
                self.forward_initial_image_path = tmp_path

            # Reset 코드 재사용 또는 새로 생성
            if self.cached_reset_code is not None:
                print(f"  {GREEN}[CodeReuse] Using cached reset code{RESET_C}")
                # 검출로 current_positions 획득 (forward 함수 재사용, detection만 수행)
                print(f"\n{CYAN}  [Restore] Running detection to find current positions...{RESET_C}")
                self.generate_forward_code(
                    instruction, {},
                    image_path=tmp_path,
                    skip_codegen=True,
                    canonical_labels=list(target_positions.keys()),
                )
                print(f"{CYAN}  [Restore] Detection complete{RESET_C}")
                current_pos = self.detected_positions
                reset_code = self.cached_reset_code
            else:
                print(f"  Generating reset code via LLM...")
                reset_code, _, current_pos, _ = self.generate_reset_code(
                    original_instruction=instruction,
                    original_positions=target_positions,
                    current_state_image_path=tmp_path,
                )

            # Reset 코드 실행 (레코딩 임시 비활성화)
            self.shutdown_camera()
            saved_record = self.record_dataset
            self.record_dataset = False
            try:
                success = self.execute_code(reset_code, {}, extra_globals={
                    "current_positions": current_pos,
                    "target_positions": target_positions,
                })
            finally:
                self.record_dataset = saved_record

            if success:
                print(f"  {GREEN}Restore to seed: SUCCESS{RESET_C}")
            else:
                print(f"  {RED}Restore to seed: FAILED{RESET_C}")

            return success

        except Exception as e:
            print(f"  {RED}Restore to seed error: {e}{RESET_C}")
            import traceback
            traceback.print_exc()
            return False

    # ================================================================
    # 공통 헬퍼
    # ================================================================

    def _init_session(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        save_dir: Optional[str],
        session_dir: Optional[str] = None,
    ) -> Tuple[str, int, Dict]:
        """세션 초기화: 디렉토리, 배치 계산, 결과 딕셔너리 생성.

        Returns:
            (session_dir, episodes_per_seed, all_results)
        """
        if save_dir is None:
            save_dir = "results"

        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = str(Path(save_dir) / f"session_{timestamp}")
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        self.total_episodes = num_episodes
        episodes_per_seed = max(1, num_episodes // self.num_random_seeds)

        # 세션 설정 저장 (최초 생성 시만, resume 시 덮어쓰지 않음)
        config_path = Path(session_dir) / "session_config.json"
        if not config_path.exists():
            session_config = {
                "num_episodes": num_episodes,
                "num_random_seeds": self.num_random_seeds,
                "episodes_per_seed": episodes_per_seed,
                "instruction": instruction,
                "objects": objects,
            }
            with open(config_path, 'w') as f:
                json.dump(session_config, f, indent=2)
            print(f"  [Session] Config saved: {config_path}")

        all_results = {
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'session_dir': session_dir,
            'episodes': [],
            'summary': {
                'forward_success': 0,
                'forward_judge_true': 0,
                'forward_judge_false': 0,
                'reset_success': 0,
                'reset_judge_true': 0,
                'reset_judge_false': 0,
            },
        }

        return session_dir, episodes_per_seed, all_results

    def _finalize_session(
        self,
        all_results: Dict,
        session_dir: str,
        num_episodes: int,
        instruction: str,
        objects: list,
        skip_reset: bool,
    ) -> None:
        """세션 마무리: 요약 출력, JSON 저장, 레코딩 finalize."""
        self._print_final_summary(all_results, skip_reset)

        summary_path = Path(session_dir) / "session_summary.json"
        summary_data = {
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'summary': all_results['summary'],
            'episodes': [
                {
                    'episode': ep['episode'],
                    'success': ep['success'],
                    'error': ep['error'],
                    'forward_success': ep['result']['forward']['execution_success'] if ep['result'] else False,
                    'forward_judge': ep['result']['judge'].get('prediction', 'N/A') if ep['result'] else 'N/A',
                    'reset_success': ep['result']['reset']['execution_success'] if ep['result'] and not skip_reset else 'N/A',
                    'reset_judge': ep['result']['reset_judge'].get('prediction', 'N/A') if ep['result'] and not skip_reset else 'N/A',
                }
                for ep in all_results['episodes']
            ],
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Session Summary] Saved to: {summary_path}")

        # 최종 시드 시각화 (new_seed 없이 — 모든 시드 분포 확인용)
        if self._all_previous_seed_positions and self.num_random_seeds > 1:
            try:
                from pix2robot_calibrator import Pix2RobotCalibrator
                pix2robot = None
                calib_path = Path(__file__).parent / "robot_configs" / "pix2robot_matrices" / f"robot{self.robot_id}_pix2robot_data.npz"
                if calib_path.exists():
                    pix2robot = Pix2RobotCalibrator(robot_id=self.robot_id)
                    if not pix2robot.load(str(calib_path)):
                        pix2robot = None

                last_seed_idx = len(self._all_previous_seed_positions) - 1
                final_save_dir = str(Path(session_dir) / f"seed_{last_seed_idx+1:02d}_setup")
                Path(final_save_dir).mkdir(parents=True, exist_ok=True)
                # new_positions를 빈 dict로 → 초록 bbox 없음
                self._visualize_seed_positions(
                    save_dir=final_save_dir,
                    new_positions={},
                    seed_index=last_seed_idx,
                    pix2robot=pix2robot,
                )
                print(f"  [Session] Final seed visualization saved")
            except Exception as e:
                print(f"  [Session] Warning: Final seed visualization failed: {e}")

        if self.record_dataset:
            self._finalize_recording()


    # ================================================================
    # 새 세션 모드
    # ================================================================

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
    ) -> Dict:
        """새 세션: 모든 에피소드를 순차 실행. 실패해도 멈추지 않고 계속 진행.

        배치 마지막 에피소드의 reset에서 다음 seed를 생성하고 그 위치로 reset하여
        다음 배치의 시작 상태를 준비한다.

        흐름 (6ep, 3seed, eps_per_seed=2):
            EP01: forward(seed0) → reset(seed0)
            EP02: forward(seed0) → seed_gen(1) → reset(seed1)   ← 배치 마지막
            EP03: forward(seed1) → reset(seed1)
            EP04: forward(seed1) → seed_gen(2) → reset(seed2)   ← 배치 마지막
            EP05: forward(seed2) → reset(seed2)
            EP06: forward(seed2) → reset(seed2)                  ← 전체 마지막
        """
        import copy

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        session_dir, episodes_per_seed, all_results = self._init_session(
            num_episodes, instruction, objects, save_dir,
        )
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # 헤더 출력
        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  MULTI-EPISODE SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        # 에피소드 루프
        for episode_idx in range(num_episodes):
            episode_num = episode_idx + 1
            batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
            is_batch_last = (episode_idx % episodes_per_seed == episodes_per_seed - 1) or (episode_idx == num_episodes - 1)
            next_batch_index = batch_index + 1

            self.current_episode = episode_num
            self._current_batch_index = batch_index
            self._current_slot = episode_idx % episodes_per_seed

            print("\n" + CYAN + "=" * 70 + RESET)
            print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1})  ".center(70) + RESET)
            print(CYAN + "=" * 70 + RESET)

            episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")

            # Reset target: 기본은 현재 seed, 배치 전환 시 콜백으로 갱신
            reset_target = seed_positions[batch_index]

            # 배치 마지막이면: Forward 후 다음 seed 생성 → Reset target 갱신 콜백
            # 단, 전체 마지막 에피소드이거나 마지막 배치이면 seed gen 생략
            pre_reset_cb = None
            if is_batch_last and next_batch_index < self.num_random_seeds:
                _next_idx = next_batch_index
                _sp = seed_positions
                _sd = session_dir
                def _make_next_seed(next_idx=_next_idx, sp=_sp, sd=_sd, current_positions=None):
                    if sp[0] is None and self.first_episode_positions is not None:
                        sp[0] = copy.deepcopy(self.first_episode_positions)
                        self._all_previous_seed_positions.append(sp[0])
                    if sp[next_idx] is None:
                        print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{next_idx+1}...{RESET}")
                        sp[next_idx] = self._generate_seed_positions(sd, next_idx, current_positions=current_positions)
                    return sp[next_idx]
                pre_reset_cb = _make_next_seed

            try:
                _slot = episode_idx % episodes_per_seed
                _ep_dir = episode_dir
                _bi = batch_index
                def _post_judge(res):
                    jp = res['judge'].get('prediction', 'UNCERTAIN')
                    from pipeline.save_logs import save_batch_info as _sbi
                    _sbi(_ep_dir, _bi, _slot, jp)

                result = self.run(
                    instruction=instruction, objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir, use_timestamp_subdir=False,
                    skip_reset=skip_reset, reset_target_positions=reset_target,
                    pre_reset_callback=pre_reset_cb,
                    post_judge_callback=_post_judge,
                )

                # first_episode_positions → seed_positions[0] 초기 설정
                # (seed_01_setup 폴더/파일은 run() 내부에서 이미 저장됨)
                if seed_positions[0] is None and self.first_episode_positions is not None:
                    seed_positions[0] = copy.deepcopy(self.first_episode_positions)
                    self._all_previous_seed_positions.append(seed_positions[0])

                self._update_results(all_results, result, episode_num, skip_reset)

            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                import traceback; traceback.print_exc()
                all_results['episodes'].append({'episode': episode_num, 'result': None, 'success': False, 'error': str(e)})

            time.sleep(2)

        self._finalize_session(all_results, session_dir, num_episodes, instruction, objects, skip_reset)
        return all_results

    # ================================================================
    # Resume 모드
    # ================================================================

    def resume_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        resume_session_dir: str,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
    ) -> Dict:
        """Resume: 이전 세션의 미완료 배치만 골라서 재시도.

        1. 완료된 배치 → 스킵
        2. 시도했지만 실패한 배치 → _restore_to_seed 후 미완료 slot 재시도 (최대 3회)
        3. 미시도 배치 → _restore_to_seed 후 순차 실행 (새 세션과 동일 방식)

        흐름 예시 (이전 세션 6ep, 3seed):
            Batch 0: slot0=TRUE, slot1=TRUE  → 스킵
            Batch 1: slot0=TRUE, slot1=FALSE → restore(seed1) → slot1 재시도
            Batch 2: 미시도                   → restore(seed2) → EP05, EP06 순차 실행
        """
        import copy

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        session_dir = str(resume_session_dir)
        session_dir, episodes_per_seed, all_results = self._init_session(
            num_episodes, instruction, objects, save_dir, session_dir=session_dir,
        )
        batch_slots, seed_positions, batch_attempted = self._load_resume_state(session_dir)

        # 헤더 출력
        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  RESUME SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Resume from: {session_dir}")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        # --------------------------------------------------------
        # 데이터셋 정리: 재취득 대상 에피소드를 데이터셋에서 제거
        # --------------------------------------------------------
        if self.record_dataset:
            if self.dataset_repo_id:
                try:
                    from record_dataset.cleanup import cleanup_dataset_for_resume
                    cleanup_stats = cleanup_dataset_for_resume(
                        session_dir=session_dir,
                        repo_id=self.dataset_repo_id,
                    )
                    print(f"\n[Cleanup] Result: {cleanup_stats['dataset_episodes_before']} → {cleanup_stats['dataset_episodes_after']} episodes")
                    if cleanup_stats['deleted_indices']:
                        print(f"[Cleanup] Deleted dataset indices: {cleanup_stats['deleted_indices']}")
                except Exception as e:
                    print(f"\n{YELLOW}[Cleanup] Warning: Dataset cleanup failed: {e}{RESET}")
                    import traceback; traceback.print_exc()

            # cleanup 후 레코딩 초기화 (resume=True로 append 모드)
            self._init_recording()

        # --------------------------------------------------------
        # 통합 에피소드 루프: 성공 slot 스킵, 실패/미시도 slot 실행
        # --------------------------------------------------------

        def _find_next_incomplete(from_batch):
            """from_batch 이후 첫 미완료 배치 인덱스 반환"""
            for i in range(from_batch + 1, self.num_random_seeds):
                if any(not done for done in batch_slots[i]):
                    return i
            return None

        # 첫 미완료 배치 찾기 → _restore_to_seed 1회
        first_incomplete = None
        for i in range(self.num_random_seeds):
            if any(not done for done in batch_slots[i]):
                first_incomplete = i
                break

        if first_incomplete is None:
            # 설정 불일치 감지: session_config.json 또는 실제 데이터에서 원래 설정 확인
            config_path = Path(session_dir) / "session_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    orig = json.load(f)
                orig_eps = orig.get("num_episodes", num_episodes)
                orig_seeds = orig.get("num_random_seeds", self.num_random_seeds)
                if orig_eps != num_episodes or orig_seeds != self.num_random_seeds:
                    print(f"\n{RED}{BOLD}  [Config Mismatch] 원래 세션 설정: NUM_EPISODES={orig_eps}, NUM_RANDOM_SEEDS={orig_seeds}{RESET}")
                    print(f"{RED}  현재 설정: NUM_EPISODES={num_episodes}, NUM_RANDOM_SEEDS={self.num_random_seeds}{RESET}")
                    print(f"{RED}  → run_forward_and_reset.sh에서 NUM_EPISODES={orig_eps}, NUM_RANDOM_SEEDS={orig_seeds}로 맞춰주세요.{RESET}")
            else:
                # session_config.json 없는 이전 세션: 실제 데이터에서 추정
                actual_episodes = len(list(Path(session_dir).glob("episode_*")))
                if actual_episodes > num_episodes:
                    print(f"\n{RED}{BOLD}  [Config Mismatch] Session has {actual_episodes} episodes but NUM_EPISODES={num_episodes}, NUM_RANDOM_SEEDS={self.num_random_seeds}{RESET}")
                    print(f"{RED}  → run_forward_and_reset.sh의 NUM_EPISODES와 NUM_RANDOM_SEEDS를 원래 세션 설정으로 맞춰주세요.{RESET}")
            print(f"\n{GREEN}  All batches complete, nothing to resume{RESET}")
        else:
            # seed 확보 + restore (Batch 0/seed 1은 initial positions이므로 restore 스킵)
            if first_incomplete > 0:
                if seed_positions[first_incomplete] is None:
                    print(f"\n{MAGENTA}{BOLD}  Generating seed_{first_incomplete+1}...{RESET}")
                    seed_positions[first_incomplete] = self._generate_seed_positions(session_dir, first_incomplete)
                if seed_positions[first_incomplete] is not None:
                    print(f"\n{CYAN}{BOLD}  Restoring to seed_{first_incomplete+1}...{RESET}")
                    self._restore_to_seed(seed_positions[first_incomplete], instruction, detection_timeout)

            # 에피소드 루프
            for episode_idx in range(num_episodes):
                batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
                slot = episode_idx % episodes_per_seed
                episode_num = episode_idx + 1

                # 이미 성공한 slot → 스킵
                if batch_slots[batch_index][slot]:
                    continue

                self.current_episode = episode_num
                self._current_batch_index = batch_index
                self._current_slot = slot

                print("\n" + CYAN + "=" * 70 + RESET)
                print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1}, Slot {slot})  ".center(70) + RESET)
                print(CYAN + "=" * 70 + RESET)

                episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")

                # Reset target 결정: 배치 내 남은 미완료 slot이 있는지 확인
                remaining_in_batch = [s for s in range(slot + 1, episodes_per_seed)
                                      if not batch_slots[batch_index][s]]
                if not remaining_in_batch:
                    # 배치 마지막 실행 slot → 다음 미완료 배치 탐색
                    next_incomplete = _find_next_incomplete(batch_index)
                    if next_incomplete is not None:
                        if seed_positions[next_incomplete] is None:
                            print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{next_incomplete+1}...{RESET}")
                            seed_positions[next_incomplete] = self._generate_seed_positions(session_dir, next_incomplete)
                        reset_target = seed_positions[next_incomplete] if seed_positions[next_incomplete] else seed_positions[batch_index]
                    else:
                        reset_target = seed_positions[batch_index]  # 정리
                else:
                    reset_target = seed_positions[batch_index]  # 같은 배치 유지

                try:
                    _slot_r = slot
                    _ep_dir_r = episode_dir
                    _bi_r = batch_index
                    def _post_judge_resume(res):
                        jp = res['judge'].get('prediction', 'UNCERTAIN')
                        from pipeline.save_logs import save_batch_info as _sbi_r
                        _sbi_r(_ep_dir_r, _bi_r, _slot_r, jp)

                    result = self.run(
                        instruction=instruction, objects=objects,
                        detection_timeout=detection_timeout,
                        visualize_detection=visualize_detection,
                        save_dir=episode_dir, use_timestamp_subdir=False,
                        skip_reset=skip_reset, reset_target_positions=reset_target,
                        post_judge_callback=_post_judge_resume,
                    )

                    judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                    if judge_pred == 'TRUE':
                        batch_slots[batch_index][slot] = True
                    self._update_results(all_results, result, episode_num, skip_reset)

                except Exception as e:
                    print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                    import traceback; traceback.print_exc()
                    all_results['episodes'].append({'episode': episode_num, 'result': None, 'success': False, 'error': str(e)})
                time.sleep(2)

        self._finalize_session(all_results, session_dir, num_episodes, instruction, objects, skip_reset)
        return all_results

    def _print_final_summary(self, all_results: Dict, skip_reset: bool) -> None:
        """전체 에피소드 최종 요약 출력"""
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        n = all_results['num_episodes']
        s = all_results['summary']

        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  FINAL SUMMARY ({n} Episodes)  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)

        # Forward 결과
        fwd_rate = s['forward_success'] / n * 100 if n > 0 else 0
        fwd_color = GREEN if fwd_rate >= 80 else YELLOW if fwd_rate >= 50 else RED
        print(f"  Forward Success:    {fwd_color}{s['forward_success']}/{n} ({fwd_rate:.1f}%){RESET}")

        # Forward Judge 결과
        judge_true_rate = s['forward_judge_true'] / n * 100 if n > 0 else 0
        judge_color = GREEN if judge_true_rate >= 80 else YELLOW if judge_true_rate >= 50 else RED
        print(f"  Forward Judge TRUE: {judge_color}{s['forward_judge_true']}/{n} ({judge_true_rate:.1f}%){RESET}")

        if not skip_reset:
            # Reset 결과
            reset_rate = s['reset_success'] / n * 100 if n > 0 else 0
            reset_color = GREEN if reset_rate >= 80 else YELLOW if reset_rate >= 50 else RED
            print(f"  Reset Success:      {reset_color}{s['reset_success']}/{n} ({reset_rate:.1f}%){RESET}")

            # Reset Judge 결과
            rj_true_rate = s['reset_judge_true'] / n * 100 if n > 0 else 0
            rj_color = GREEN if rj_true_rate >= 80 else YELLOW if rj_true_rate >= 50 else RED
            print(f"  Reset Judge TRUE:   {rj_color}{s['reset_judge_true']}/{n} ({rj_true_rate:.1f}%){RESET}")

        print(MAGENTA + "=" * 70 + RESET)


def main():
    parser = argparse.ArgumentParser(
        description="Forward + Reset Integrated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수 인자
    parser.add_argument(
        "--instruction", "-i",
        type=str,
        required=True,
        help="Natural language instruction"
    )

    parser.add_argument(
        "--objects", "-o",
        type=str,
        nargs="+",
        default=[],
        help="Objects to detect (only used in single-turn mode)"
    )

    # 선택 인자
    parser.add_argument(
        "--robot", "-r",
        type=int,
        nargs="+",
        default=[3],
        help="Robot ID(s). Single: --robot 2, Dual: --robot 2 3"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for code generation (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge VLM model (default: gpt-4o)"
    )

    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=10.0,
        help="Detection timeout in seconds (default: 10)"
    )

    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=5.0,
        help="Judge UI timeout in seconds (default: 5.0)"
    )

    parser.add_argument(
        "--visualize-detection",
        action="store_true",
        help="Show real-time detection visualization (single-turn mode only)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous session directory (e.g., results/session_20260319_174942)"
    )

    parser.add_argument(
        "--save", "-s",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )

    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip reset execution phase"
    )

    parser.add_argument(
        "--num-random-seeds",
        type=int,
        default=1,
        help="Number of random position batches (1=keep initial positions, N>1=N different random layouts)"
    )

    parser.add_argument(
        "--use-server",
        action="store_true",
        help="Use vLLM server for LLM/VLM inference instead of API"
    )

    # CodeGen LLM 서버 설정
    parser.add_argument(
        "--codegen-server-url",
        type=str,
        default="http://localhost:8001/v1",
        help="CodeGen LLM server URL (default: http://localhost:8001/v1)"
    )
    parser.add_argument(
        "--codegen-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="CodeGen LLM model name (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    )

    # Judge VLM 서버 설정
    parser.add_argument(
        "--judge-server-url",
        type=str,
        default="http://localhost:8002/v1",
        help="Judge VLM server URL (default: http://localhost:8002/v1)"
    )
    parser.add_argument(
        "--judge-server-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Judge VLM model name (default: Qwen/Qwen2-VL-2B-Instruct)"
    )

    parser.add_argument(
        "--num-episodes", "-n",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )

    # LeRobot 데이터셋 레코딩 옵션
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable LeRobot dataset recording during forward execution"
    )

    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="LeRobot dataset repository ID (e.g., 'user/my_dataset'). If not specified, auto-generated."
    )

    parser.add_argument(
        "--recording-fps",
        type=int,
        default=30,
        help="Recording FPS for LeRobot dataset (default: 30)"
    )

    # Multi-turn 옵션
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Use crop-then-point multi-turn LLM code generation (requires Gemini model)"
    )

    parser.add_argument(
        "--cad-image-dirs",
        type=str,
        nargs="*",
        default=None,
        help="CAD reference image directories for Turn 0 scene understanding"
    )

    parser.add_argument(
        "--codegen-session2-model",
        type=str,
        default=None,
        help="Model for code generation Session 2 (context handoff). If not set, uses --llm model."
    )

    parser.add_argument(
        "--detect-model",
        type=str,
        default=None,
        help="VLM model for detect_objects skill (default: uses gemini-3.1-flash-lite-preview)"
    )

    parser.add_argument(
        "--side-view-image",
        type=str,
        default=None,
        help="Side-view image path for Turn Test waypoint trajectory prediction"
    )

    parser.add_argument(
        "--reset-instruction",
        type=str,
        default=None,
        help="Reset task instruction for recording. Default: 'move objects to certain position'"
    )

    parser.add_argument(
        "--skip-turn-test",
        action="store_true",
        help="Skip Turn Test (Waypoint Trajectory Prediction) in multi-turn code generation"
    )

    parser.add_argument(
        "--resetspace-per-robot",
        type=str,
        nargs="+",
        default=None,
        help="Per-robot reset quadrant (all, top-left, top-right, bottom-left, bottom-right). "
             "Order matches --robot order. Default: 'all' for each robot."
    )

    args = parser.parse_args()

    # 서버 모드 설정 (환경변수로 전달)
    if args.use_server:
        import os
        os.environ["USE_LLM_SERVER"] = "1"
        os.environ["USE_VLM_SERVER"] = "1"  # Judge용 VLM 서버 모드 활성화
        # CodeGen LLM 서버 설정
        os.environ["VLLM_SERVER_URL"] = args.codegen_server_url
        os.environ["VLLM_MODEL_NAME"] = args.codegen_model
        # Judge VLM 서버 설정
        os.environ["JUDGE_SERVER_URL"] = args.judge_server_url
        os.environ["JUDGE_MODEL_NAME"] = args.judge_server_model

    # 파이프라인 실행: 로봇 수에 따라 분기
    robot_ids = args.robot  # list of int (nargs="+")

    if len(robot_ids) == 1:
        # ── Single-arm: 기존 ForwardAndResetPipeline ──
        # resetspace: single-arm이면 첫 번째 값 사용
        _rs = args.resetspace_per_robot
        single_resetspace = _rs[0] if _rs else None

        pipeline = ForwardAndResetPipeline(
            robot_id=robot_ids[0],
            llm_model=args.llm,
            judge_model=args.judge_model,
            judge_timeout_ms=int(args.judge_timeout * 1000),
            num_random_seeds=args.num_random_seeds,
            verbose=True,
            record_dataset=args.record,
            dataset_repo_id=args.dataset_repo_id,
            resume_recording=bool(args.resume),
            multi_turn=args.multi_turn,
            cad_image_dirs=args.cad_image_dirs,
            side_view_image=args.side_view_image,
            recording_fps=args.recording_fps,
            codegen_model=args.codegen_session2_model,
            reset_instruction=args.reset_instruction,
            skip_turn_test=args.skip_turn_test,
            detect_model=args.detect_model,
            resetspace=single_resetspace,
        )
    else:
        # ── Multi-arm: UnifiedMultiArmPipeline ──
        from unified_multi_arm import UnifiedMultiArmPipeline
        # resetspace: multi-arm이면 로봇 순서대로 매핑
        _rs = args.resetspace_per_robot or ["all"] * len(robot_ids)
        # 부족하면 마지막 값으로 채움
        while len(_rs) < len(robot_ids):
            _rs.append(_rs[-1] if _rs else "all")
        resetspace_per_robot = dict(zip(robot_ids, _rs))

        pipeline = UnifiedMultiArmPipeline(
            robot_ids=robot_ids,
            llm_model=args.llm,
            judge_model=args.judge_model,
            judge_timeout_ms=int(args.judge_timeout * 1000),
            num_random_seeds=args.num_random_seeds,
            verbose=True,
            record_dataset=args.record,
            dataset_repo_id=args.dataset_repo_id,
            resume_recording=bool(args.resume),
            multi_turn=args.multi_turn,
            cad_image_dirs=args.cad_image_dirs,
            side_view_image=args.side_view_image,
            recording_fps=args.recording_fps,
            codegen_model=args.codegen_session2_model,
            reset_instruction=args.reset_instruction,
            skip_turn_test=args.skip_turn_test,
            detect_model=args.detect_model,
            resetspace_per_robot=resetspace_per_robot,
        )

    # 에피소드 실행: resume 모드와 새 세션 모드 분기
    if args.resume:
        all_results = pipeline.resume_multiple_episodes(
            num_episodes=args.num_episodes,
            instruction=args.instruction,
            objects=args.objects,
            resume_session_dir=args.resume,
            detection_timeout=args.timeout,
            visualize_detection=args.visualize_detection,
            save_dir=args.save,
            skip_reset=args.skip_reset,
        )
    else:
        all_results = pipeline.run_multiple_episodes(
            num_episodes=args.num_episodes,
            instruction=args.instruction,
            objects=args.objects,
            detection_timeout=args.timeout,
            visualize_detection=args.visualize_detection,
            save_dir=args.save,
            skip_reset=args.skip_reset,
        )

    # 종료 코드 결정 (성공률 기반)
    n = all_results['num_episodes']
    s = all_results['summary']
    success_rate = s['forward_judge_true'] / n if n > 0 else 0

    if success_rate >= 0.5:  # 50% 이상 성공
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
