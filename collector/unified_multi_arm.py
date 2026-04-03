#!/usr/bin/env python3
"""
UnifiedMultiArmPipeline — Bi-arm orchestrator

Same interface as ForwardAndResetPipeline but controls two arms:
  1. MultiArmSkills (left_arm + right_arm)
  2. Multi-turn VLM code generation (code_gen_lerobot/multi_arm/)
  3. MultiArmRecorder (ALOHA 12-axis recording)
  4. Judge (1 evaluation per episode — whole-task success/failure)
  5. Multi-arm reset code generation + execution

Detection is handled by multi-turn VLM pipeline (Turn 0~2), not Grounding DINO.

Usage (from execution_forward_and_reset.py):
    if len(robot_ids) > 1:
        pipeline = UnifiedMultiArmPipeline(robot_ids=[2, 3], ...)
        pipeline.run_multiple_episodes(...)
"""

import copy
import json
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skills.multi_arm_skills import MultiArmSkills


# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET_COLOR = "\033[0m"
BOLD = "\033[1m"


from pipeline.base_pipeline import BasePipeline


class UnifiedMultiArmPipeline(BasePipeline):
    """
    Unified bi-arm pipeline with the same interface as ForwardAndResetPipeline.

    Entry point routing:
        ROBOT_IDS=(2)   → ForwardAndResetPipeline (single-arm, existing)
        ROBOT_IDS=(2 3) → UnifiedMultiArmPipeline (this class)
    """

    def __init__(
        self,
        robot_ids: List[int],
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
        resetspace_per_robot: Dict[int, str] = None,
    ):
        assert len(robot_ids) >= 2, f"Multi-arm requires >= 2 robots, got {robot_ids}"
        self.robot_ids = robot_ids
        self.left_id = robot_ids[0]
        self.right_id = robot_ids[1]
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.judge_timeout_ms = judge_timeout_ms
        self.num_random_seeds = num_random_seeds
        self.verbose = verbose

        # Multi-turn
        self.multi_turn = multi_turn
        self.cad_image_dirs = cad_image_dirs or []
        self.side_view_image = side_view_image
        self.codegen_model = codegen_model
        self.reset_instruction = reset_instruction or "move objects to their original positions"
        self.skip_turn_test = skip_turn_test
        self.detect_model = detect_model
        # Per-robot reset quadrant: {robot_id: "all"|"top-left"|...}
        self.resetspace_per_robot = resetspace_per_robot or {rid: "all" for rid in robot_ids}
        self.multi_turn_info: Dict = {}
        self.reset_multi_turn_info: Dict = {}

        # Recording
        self.record_dataset = record_dataset
        self.dataset_repo_id = dataset_repo_id
        self.recording_fps = recording_fps
        self.resume_recording = resume_recording

        # State
        self.camera = None
        self.camera_manager = None
        self.multi_arm: Optional[MultiArmSkills] = None
        self.dataset_recorder = None

        # Episode tracking
        self.current_episode = 1
        self.total_episodes = 1
        self.current_phase = "Forward"
        self.instruction = ""

        # Positions tracking
        self.detected_positions: Dict = {}
        self.first_episode_positions: Optional[Dict] = None
        self._all_previous_seed_positions = []

        # Code cache
        self.cached_forward_code: Optional[str] = None
        self.cached_forward_keys: List[str] = []
        self.cached_reset_code: Optional[str] = None
        self.cached_reset_keys: List[str] = []

        # 레코딩 모드 초기화 (resume 모드는 run_multiple_episodes에서 초기화)
        if self.record_dataset and not self.resume_recording:
            self._init_recording()


    # ─────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────

    def _init_multi_arm(self) -> bool:
        """Initialize MultiArmSkills with both robot configs.

        Note: connect()는 여기서 하지 않음.
        LLM 생성 코드가 skills.connect()를 직접 호출.
        RecordingContext가 활성화되어 있으면 connect() 시 양팔 모두 자동으로 콜백 획득.
        """
        left_config = f"robot_configs/robot/so101_robot{self.left_id}.yaml"
        right_config = f"robot_configs/robot/so101_robot{self.right_id}.yaml"

        detect_kwargs = {}
        if self.detect_model:
            detect_kwargs["detect_model"] = self.detect_model

        self.multi_arm = MultiArmSkills(
            left_config=left_config,
            right_config=right_config,
            frame="base_link",
            verbose=self.verbose,
            **detect_kwargs,
        )
        return True  # connect()는 LLM 코드에서 호출

    def _init_camera(self) -> bool:
        """Initialize camera via PipelineCamera."""
        if not hasattr(self, '_pipeline_camera') or self._pipeline_camera is None:
            from pipeline.camera_session import PipelineCamera
            self._pipeline_camera = PipelineCamera(
                camera_manager=self.camera_manager, verbose=self.verbose)
        result = self._pipeline_camera.initialize()
        self.camera = self._pipeline_camera.camera
        return result

    def _init_recording(self):
        """Initialize recording with MultiArmRecorder (12-axis ALOHA format)."""
        if not self.record_dataset:
            return

        if not self.dataset_repo_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_repo_id = f"local/multi_arm_{timestamp}"

        try:
            from record_dataset.recorder import DatasetRecorder
            from record_dataset.config import (
                create_camera_manager_from_config,
                build_multi_arm_features,
                load_cameras_from_yaml,
            )
            from record_dataset.multi_arm_recorder import MultiArmRecorder

            print(f"\n[Recording] Initializing multi-arm dataset recorder...")
            print(f"  Repo ID: {self.dataset_repo_id}")

            # 1. 기존 dataset 존재 여부 미리 체크 (forward + reset)
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

            # 2. Camera manager (YAML에서 동적 로드)
            if self.camera_manager is None:
                self.camera_manager = create_camera_manager_from_config(num_robots=len(self.robot_ids))
                self.camera_manager.connect_all()
                print(f"[Recording] Cameras connected: {self.camera_manager.camera_names}")

            # 3. YAML에 enabled된 카메라가 모두 연결되었는지 검증
            cameras = load_cameras_from_yaml(num_robots=len(self.robot_ids))
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

            # 4. Dataset recorder with 12-axis multi-arm features
            multi_arm_features = build_multi_arm_features(cameras=enabled_cameras)

            self.dataset_recorder = DatasetRecorder(
                repo_id=self.dataset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
                features=multi_arm_features,
            )
            print(f"[Recording] Recorder initialized (12-axis ALOHA format)")

            # 5. MultiArmRecorder will be created after skills.connect()
            # (needs multi_arm instance which is created later)
            self._multi_arm_recorder: Optional['MultiArmRecorder'] = None

            # 6. Signal handler: Ctrl+C 시 finalize() 호출하여 meta/episodes/ 보존
            self._install_recording_signal_handler()

        except AssertionError:
            raise
        except Exception as e:
            print(f"[Recording] Init failed: {e}")
            import traceback
            traceback.print_exc()
            self.record_dataset = False
            self.dataset_recorder = None


    def _finalize_recording(self) -> None:
        """데이터셋 finalize (signal handler, _finalize_session 양쪽에서 호출)."""
        if self.dataset_recorder:
            try:
                self.dataset_recorder.finalize()
                print(f"[Recording] Dataset finalized: {self.dataset_recorder.repo_id}")
            except Exception as e:
                print(f"[Recording] Finalize error: {e}")
                import traceback
                traceback.print_exc()


    # ─────────────────────────────────────────────
    # Code Generation
    # ─────────────────────────────────────────────

    def _generate_forward_code(
        self,
        instruction: str,
        image_path: str,
        skip_codegen: bool = False,
        canonical_labels: list = None,
        canonical_point_labels: dict = None,
    ) -> str:
        """Generate forward code via multi-turn VLM.

        Args:
            skip_codegen: True면 T0~T2(검출)만 수행, T3(코드생성) 스킵 (코드 재활용 시).
            canonical_labels: 이전 에피소드에서 검출된 물체 라벨 (강제 사용).
            canonical_point_labels: 이전 에피소드에서 사용된 포인트 라벨.
        """
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen_multi_turn

        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("top")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        code, mt_positions, mt_info = lerobot_code_gen_multi_turn(
            instruction=instruction,
            image_path=image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_ids[0],
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            fallback_positions={},
            camera=active_camera,
            cad_image_dirs=self.cad_image_dirs,
            side_view_image=self.side_view_image,
            codegen_model=self.codegen_model,
            skip_turn_test=self.skip_turn_test,
            robot_ids=self.robot_ids,
            skip_codegen=skip_codegen,
            canonical_labels=canonical_labels,
            canonical_point_labels=canonical_point_labels,
        )

        # multi-turn 정보 저장
        self.multi_turn_info = mt_info

        # Build dual-arm positions: re-run pixel→world for each robot's calibration
        from code_gen_lerobot.code_gen_with_skill import _points_to_positions
        all_points = mt_info.get("all_points", [])
        valid_objects = mt_info.get("detected_objects", [])

        self.detected_positions = {
            "left_arm": _points_to_positions(all_points, robot_id=self.left_id, camera=active_camera, valid_objects=valid_objects),
            "right_arm": _points_to_positions(all_points, robot_id=self.right_id, camera=active_camera, valid_objects=valid_objects),
        }

        return code

    def _generate_reset_code(
        self,
        original_instruction: str,
        original_positions: Dict,
        current_state_image_path: str,
        initial_state_image_path: str,
    ) -> str:
        """Generate reset code via multi-turn VLM pipeline (same as forward).

        Turn 0~2 perception is reused from single-arm pipeline.
        CodeGen stage branches to multi-arm reset prompt.
        """
        from code_gen_lerobot.reset_execution.code_gen import lerobot_reset_code_gen_multi_turn

        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("top")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        # Multi-arm original_positions is {"left_arm": {obj: ...}, "right_arm": {obj: ...}}.
        # Reset perception only needs object labels (keys) — pick one arm's dict.
        # Codegen needs per-arm coordinates — pass original dual structure separately.
        is_dual = "left_arm" in original_positions or "right_arm" in original_positions
        if is_dual:
            one_arm_positions = original_positions.get("left_arm") or original_positions.get("right_arm") or {}
        else:
            one_arm_positions = original_positions

        reset_code, current_pos, target_pos, grippable, obstacles, reset_mt_info = lerobot_reset_code_gen_multi_turn(
            original_instruction=original_instruction,
            original_positions=one_arm_positions,
            original_positions_dual=original_positions if is_dual else None,
            current_state_image_path=current_state_image_path,
            initial_state_image_path=initial_state_image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_ids[0],
            camera=active_camera,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            codegen_model=self.codegen_model,
            robot_ids=self.robot_ids,
            resetspace=self.resetspace_per_robot,
        )

        self.reset_multi_turn_info = reset_mt_info
        self._reset_current_positions = current_pos
        self._reset_target_positions = target_pos
        return reset_code

    # ─────────────────────────────────────────────
    # Seed Position Generation (multi-arm)
    # ─────────────────────────────────────────────

    def _generate_seed_positions(self, session_dir: str, seed_index: int, current_positions: Dict = None) -> Optional[Dict]:
        """
        새 seed 위치 생성 (멀티암): per-arm 구조 랜덤 위치 생성.

        각 물체의 "home arm"을 pixel 좌표 기반으로 결정하고,
        해당 arm의 resetspace 안에서 랜덤 위치를 생성한 뒤,
        pixel 경유로 양 팔 좌표계에 동기화합니다.

        Args:
            session_dir: 세션 디렉토리
            seed_index: 시드 인덱스
            current_positions: forward 후 실제 검출 위치 (per-arm dict)

        Returns:
            성공 시 per-arm positions dict, 실패 시 None
        """
        from code_gen_lerobot.reset_execution.workspace import (
            generate_random_positions, classify_objects, ResetWorkspace,
        )

        if self.first_episode_positions is None:
            print("  [SeedGen] No first_episode_positions, cannot generate")
            return None

        save_dir = str(Path(session_dir) / f"seed_{seed_index+1:02d}_setup")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # ── per-arm pix2robot 로드 ──
        pix2robot_map = {}  # {robot_id: Pix2RobotCalibrator}
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            for rid in self.robot_ids:
                calib_path = Path(__file__).parent / "robot_configs" / "pix2robot_matrices" / f"robot{rid}_pix2robot_data.npz"
                if calib_path.exists():
                    p2r = Pix2RobotCalibrator(robot_id=rid)
                    if p2r.load(str(calib_path)):
                        pix2robot_map[rid] = p2r
        except Exception:
            pass

        # ── per-arm workspace + kinematics ──
        workspace_map = {}  # {robot_id: ResetWorkspace}
        kin_map = {}        # {robot_id: KinematicsEngine}
        try:
            from lerobot_cap.kinematics.engine import KinematicsEngine
            for rid in self.robot_ids:
                urdf_path = Path(__file__).parent / "assets" / "urdf" / f"so101_robot{rid}.urdf"
                if urdf_path.exists():
                    kin = KinematicsEngine(str(urdf_path))
                    kin_map[rid] = kin
                    workspace_map[rid] = ResetWorkspace(kinematics_engine=kin)
        except Exception:
            pass

        # ── per-arm free state exclusion zones ──
        FREE_STATE_EXCLUSION_RADIUS = 0.08
        exclusion_map = {}  # {robot_id: [zone, ...]}
        for rid in self.robot_ids:
            exclusion_map[rid] = []
            # 각 팔의 free state EE 위치를 해당 팔의 exclusion zone으로 추가
            for excl_rid in self.robot_ids:
                try:
                    from lerobot_cap.kinematics import load_calibration_limits as _load_cl
                    free_state_path = Path(__file__).parent / "robot_configs" / "free_state" / f"robot{excl_rid}_free_state.json"
                    calib_path = Path(__file__).parent / "robot_configs" / "motor_calibration" / "so101" / f"robot{excl_rid}_calibration.json"
                    if free_state_path.exists() and calib_path.exists() and excl_rid in kin_map:
                        with open(free_state_path) as f:
                            free_norm = np.array(json.load(f)["initial_state_normalized"])
                        _cl = _load_cl(str(calib_path),
                            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
                        free_rad = _cl.normalized_to_radians(free_norm)
                        free_ee = kin_map[excl_rid].get_ee_position(free_rad)
                        # free_ee는 excl_rid 좌표계 → rid 좌표계로 변환 (pixel 경유)
                        if excl_rid != rid and excl_rid in pix2robot_map and rid in pix2robot_map:
                            px = pix2robot_map[excl_rid].robot_to_pixel(free_ee[0], free_ee[1])
                            rx, ry, _ = pix2robot_map[rid].pixel_to_robot(int(px[0]), int(px[1]))
                            center = [float(rx), float(ry)]
                        else:
                            center = [float(free_ee[0]), float(free_ee[1])]
                        exclusion_map[rid].append({"center": center, "radius": FREE_STATE_EXCLUSION_RADIUS})
                except Exception:
                    pass

        # ── 물체 분류 (한쪽 arm dict에서 추출) ──
        one_arm_pos = (self.first_episode_positions.get("left_arm")
                       or self.first_episode_positions.get("right_arm") or {})
        grippable, obstacles = classify_objects(one_arm_pos)

        # ── 물체별 home arm 결정 (pixel x 좌표 기준) ──
        IMAGE_WIDTH = 640
        arm_keys = {self.left_id: "left_arm", self.right_id: "right_arm"}

        def _get_home_arm(obj_name):
            """pixel x 좌표 기준으로 home arm 결정."""
            for arm_key in ["left_arm", "right_arm"]:
                info = self.first_episode_positions.get(arm_key, {}).get(obj_name)
                if info and "pixel" in info:
                    px_x = info["pixel"][0]  # pixel [u, v] → u가 x
                    return self.left_id if px_x < IMAGE_WIDTH // 2 else self.right_id
            return self.left_id  # fallback

        # ── 과거 시드 위치 (겹침 방지) ──
        all_initial = {}
        for name in one_arm_pos:
            all_initial[f"{name}_pseed_init"] = one_arm_pos[name]
        for i, prev in enumerate(self._all_previous_seed_positions):
            one_arm_prev = prev.get("left_arm") or prev.get("right_arm") or prev
            for name, info in one_arm_prev.items():
                all_initial[f"{name}_pseed{i}"] = info

        # ── arm별로 랜덤 위치 생성 (최대 10회 재시도) ──
        accepted = None
        for attempt in range(10):
            per_arm_targets = {}  # {arm_key: {obj_name: [x, y, z]}}

            for arm_key, rid in [("left_arm", self.left_id), ("right_arm", self.right_id)]:
                # 이 arm의 resetspace에 속하는 물체 필터
                my_grippable = {name: info for name, info in grippable.items()
                                if _get_home_arm(name) == rid}
                if not my_grippable:
                    per_arm_targets[arm_key] = {}
                    continue

                ws = workspace_map.get(rid)
                p2r = pix2robot_map.get(rid)
                rs = self.resetspace_per_robot.get(rid, "all")
                excl = exclusion_map.get(rid, [])

                targets = generate_random_positions(
                    grippable_objects=my_grippable,
                    obstacle_objects=obstacles,
                    initial_positions=all_initial,
                    workspace=ws,
                    pix2robot=p2r,
                    current_positions=None,
                    exclusion_zones=excl,
                    resetspace=rs,
                )
                per_arm_targets[arm_key] = targets or {}

            # 유효성 검사: 모든 grippable 물체에 위치가 생성되었는지
            generated_names = set()
            for targets in per_arm_targets.values():
                generated_names.update(targets.keys())
            if generated_names >= set(grippable.keys()):
                accepted = per_arm_targets
                print(f"  [SeedGen] Attempt {attempt+1}: positions generated for {len(generated_names)} objects")
                break
            else:
                missing = set(grippable.keys()) - generated_names
                print(f"  [SeedGen] Attempt {attempt+1}: missing {missing}, retrying...")
        else:
            print(f"  [SeedGen] All 10 attempts failed")
            return None

        # ── per-arm positions dict 조합 (pixel 경유 양팔 좌표 동기화) ──
        candidate = self._build_batch_positions_multi_arm(accepted, obstacles, pix2robot_map)

        # ── dry_run 검증: 캐시된 reset 코드로 IK 도달 가능 여부 확인 ──
        reset_code = self.cached_reset_code
        if reset_code is not None:
            print(f"  [SeedGen] Dry-run validating with cached reset code...")
            dry_current = current_positions if current_positions else candidate
            try:
                dry_ok = self.execute_code(reset_code, {}, extra_globals={
                    "current_positions": dry_current,
                    "target_positions": candidate,
                })
                if not dry_ok:
                    print(f"  [SeedGen] Dry-run FAILED, using positions anyway (best effort)")
                else:
                    print(f"  [SeedGen] Dry-run PASSED")
            except Exception as e:
                print(f"  [SeedGen] Dry-run error: {e}, using positions anyway")

        result = candidate

        # 로그
        print(f"  [SeedGen] seed_{seed_index+1} positions:")
        for arm_key in ["left_arm", "right_arm"]:
            for name, info in result.get(arm_key, {}).items():
                pos = info.get("position") if isinstance(info, dict) else None
                if pos:
                    print(f"    {arm_key}/{name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        # ── seed_positions.json 저장 ──
        try:
            seed_json = {
                "seed_index": seed_index,
                "positions": self._make_serializable(result),
            }
            with open(Path(save_dir) / "seed_positions.json", 'w') as f:
                json.dump(seed_json, f, indent=2, default=str)
            print(f"  [SeedGen] Saved: {save_dir}/seed_positions.json")
        except Exception as e:
            print(f"  [SeedGen] Warning: Failed to save seed JSON: {e}")

        # ── seed 시각화 (per-arm workspace overlay) ──
        self._visualize_seed_positions_multi_arm(save_dir, result, seed_index, pix2robot_map)

        # 과거 시드에 추가 (다음 시드 겹침 방지)
        self._all_previous_seed_positions.append(copy.deepcopy(result))

        return result

    def _build_batch_positions_multi_arm(
        self,
        per_arm_targets: Dict[str, Dict],
        obstacles: Dict,
        pix2robot_map: Dict,
    ) -> Dict:
        """
        arm별 랜덤 타겟을 per-arm positions dict로 조합.
        한쪽 arm에서 생성된 위치를 pixel 경유로 상대 arm 좌표계에도 동기화.

        Args:
            per_arm_targets: {"left_arm": {obj: [x,y,z]}, "right_arm": {obj: [x,y,z]}}
            obstacles: 비grippable 물체 dict
            pix2robot_map: {robot_id: Pix2RobotCalibrator}

        Returns:
            {"left_arm": {obj: {"position":..., "points":..., "pixel":...}},
             "right_arm": {obj: {"position":..., "points":..., "pixel":...}}}
        """
        arm_to_rid = {"left_arm": self.left_id, "right_arm": self.right_id}
        other_arm = {"left_arm": "right_arm", "right_arm": "left_arm"}
        result = {"left_arm": {}, "right_arm": {}}

        # 1. 각 arm의 grippable 물체: 생성된 위치 + pixel 경유로 상대 arm 좌표 생성
        for arm_key, targets in per_arm_targets.items():
            rid = arm_to_rid[arm_key]
            p2r = pix2robot_map.get(rid)
            opp_arm_key = other_arm[arm_key]
            opp_rid = arm_to_rid[opp_arm_key]
            opp_p2r = pix2robot_map.get(opp_rid)

            for name, pos in targets.items():
                pos = list(pos) if not isinstance(pos, list) else pos
                orig = (self.first_episode_positions.get(arm_key, {}).get(name)
                        or self.first_episode_positions.get(opp_arm_key, {}).get(name) or {})
                bbox_px = orig.get("bbox_px", (30, 30))

                # pixel 좌표 계산
                pixel = None
                if p2r is not None:
                    try:
                        pixel = list(p2r.robot_to_pixel(pos[0], pos[1]))
                    except Exception:
                        pass

                # home arm entry
                entry = {
                    "position": pos,
                    "points": {pt_name: pos for pt_name in orig.get("points", {"grasp center": None})},
                    "bbox_px": bbox_px,
                }
                if pixel:
                    entry["pixel"] = pixel
                result[arm_key][name] = entry

                # 상대 arm entry (pixel 경유 좌표 변환)
                if pixel and opp_p2r is not None:
                    try:
                        ox, oy, oz = opp_p2r.pixel_to_robot(int(pixel[0]), int(pixel[1]))
                        opp_pos = [ox, oy, pos[2]]  # z는 동일 (물체 높이)
                        opp_entry = {
                            "position": opp_pos,
                            "points": {pt_name: opp_pos for pt_name in orig.get("points", {"grasp center": None})},
                            "pixel": pixel,
                            "bbox_px": bbox_px,
                        }
                        result[opp_arm_key][name] = opp_entry
                    except Exception:
                        pass

        # 2. obstacle 물체: first_episode_positions에서 그대로 복사
        for name in obstacles:
            for arm_key in ["left_arm", "right_arm"]:
                orig = self.first_episode_positions.get(arm_key, {}).get(name)
                if orig and name not in result[arm_key]:
                    result[arm_key][name] = copy.deepcopy(orig)

        return result

    def _visualize_seed_positions_multi_arm(
        self,
        save_dir: str,
        positions: Dict,
        seed_index: int,
        pix2robot_map: Dict,
    ):
        """
        Seed 위치를 per-arm workspace 이미지에 시각화.

        Args:
            save_dir: 저장 디렉토리
            positions: per-arm positions dict
            seed_index: 시드 인덱스
            pix2robot_map: {robot_id: Pix2RobotCalibrator}
        """
        try:
            import cv2
            from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image

            # forward 초기 이미지 로드 (가장 최근 에피소드)
            base_image = None
            for ep_idx in range(self.current_episode, 0, -1):
                img_path = Path(save_dir).parent / f"episode_{ep_idx:02d}" / "forward" / "initial_state.jpg"
                if img_path.exists():
                    base_image = cv2.imread(str(img_path))
                    break
            if base_image is None:
                return

            arm_to_rid = {"left_arm": self.left_id, "right_arm": self.right_id}

            for arm_key in ["left_arm", "right_arm"]:
                rid = arm_to_rid[arm_key]
                rs = self.resetspace_per_robot.get(rid, "all")
                img = draw_workspace_on_image(base_image.copy(), robot_id=rid, resetspace=rs)

                # seed 위치에 bbox 그리기
                arm_positions = positions.get(arm_key, {})
                for name, info in arm_positions.items():
                    pixel = info.get("pixel")
                    bbox_px = info.get("bbox_px", (30, 30))
                    if pixel is None:
                        continue
                    u, v = int(pixel[0]), int(pixel[1])
                    bw, bh = int(bbox_px[0] // 2), int(bbox_px[1] // 2)
                    # 새 seed: 초록 박스
                    cv2.rectangle(img, (u - bw, v - bh), (u + bw, v + bh), (0, 255, 0), 2)
                    cv2.putText(img, name, (u - bw, v - bh - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 과거 시드: 회색 박스
                for prev in self._all_previous_seed_positions:
                    prev_arm = prev.get(arm_key, prev) if isinstance(prev, dict) and "left_arm" in prev else prev
                    if isinstance(prev_arm, dict):
                        for name, info in prev_arm.items():
                            pixel = info.get("pixel") if isinstance(info, dict) else None
                            if pixel is None:
                                continue
                            u, v = int(pixel[0]), int(pixel[1])
                            cv2.circle(img, (u, v), 4, (128, 128, 128), -1)

                out_path = str(Path(save_dir) / f"seed_{seed_index+1:02d}_{arm_key}.jpg")
                cv2.imwrite(out_path, img)

            print(f"  [SeedGen] Visualization saved: {save_dir}/seed_{seed_index+1:02d}_*.jpg")
        except Exception as e:
            print(f"  [SeedGen] Visualization warning: {e}")

        # ── 포인트 시각화 추가 ──
        self._visualize_seed_points_multi_arm(save_dir, positions, seed_index, pix2robot_map)

    def _visualize_seed_points_multi_arm(
        self,
        save_dir: str,
        positions: Dict,
        seed_index: int,
        pix2robot_map: Dict,
    ):
        """
        객체 종류별 색상으로 per-arm 시드 위치를 점으로 시각화.

        싱글암 _visualize_seed_points()와 동일한 방식:
        - 각 객체에 고유 색상 할당
        - 과거 시드: 작은 점 + 시드 번호
        - 현재 시드: 큰 점 + 시드 번호
        """
        try:
            import cv2
            from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image

            # forward 초기 이미지 로드
            base_image = None
            for ep_idx in range(self.current_episode, 0, -1):
                img_path = Path(save_dir).parent / f"episode_{ep_idx:02d}" / "forward" / "initial_state.jpg"
                if img_path.exists():
                    base_image = cv2.imread(str(img_path))
                    break
            if base_image is None:
                return

            arm_to_rid = {"left_arm": self.left_id, "right_arm": self.right_id}

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

            for arm_key in ["left_arm", "right_arm"]:
                rid = arm_to_rid[arm_key]
                rs = self.resetspace_per_robot.get(rid, "all")
                img = draw_workspace_on_image(base_image.copy(), robot_id=rid, resetspace=rs)

                # 이 팔의 모든 객체 이름 수집
                all_obj_names = set()
                arm_positions = positions.get(arm_key, {})
                for name, info in arm_positions.items():
                    if isinstance(info, dict):
                        all_obj_names.add(name)
                for prev in self._all_previous_seed_positions:
                    prev_arm = prev.get(arm_key, prev) if isinstance(prev, dict) and "left_arm" in prev else prev
                    if isinstance(prev_arm, dict):
                        for name, info in prev_arm.items():
                            if isinstance(info, dict):
                                all_obj_names.add(name)
                all_obj_names = sorted(all_obj_names)

                obj_color_map = {name: OBJ_COLORS[i % len(OBJ_COLORS)] for i, name in enumerate(all_obj_names)}

                # 과거 시드 (작은 점 + 시드 번호)
                for seed_i, prev in enumerate(self._all_previous_seed_positions):
                    prev_arm = prev.get(arm_key, prev) if isinstance(prev, dict) and "left_arm" in prev else prev
                    if not isinstance(prev_arm, dict):
                        continue
                    for name, info in prev_arm.items():
                        if name not in obj_color_map or not isinstance(info, dict):
                            continue
                        pixel = info.get("pixel")
                        if pixel is None:
                            continue
                        u, v = int(pixel[0]), int(pixel[1])
                        color = obj_color_map[name]
                        cv2.circle(img, (u, v), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(img, (u, v), 5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(img, f"s{seed_i+1}", (u + 7, v + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

                # 현재 시드 (큰 점 + 시드 번호)
                for name, info in arm_positions.items():
                    if name not in obj_color_map or not isinstance(info, dict):
                        continue
                    pixel = info.get("pixel")
                    if pixel is None:
                        continue
                    u, v = int(pixel[0]), int(pixel[1])
                    color = obj_color_map[name]
                    cv2.circle(img, (u, v), 9, color, -1, cv2.LINE_AA)
                    cv2.circle(img, (u, v), 9, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, f"s{seed_index+1}", (u + 11, v + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

                # 범례: 객체별 색상
                img_h = img.shape[0]
                y_offset = img_h - 15 * len(all_obj_names) - 20
                for i, name in enumerate(all_obj_names):
                    color = obj_color_map[name]
                    y = y_offset + i * 15
                    cv2.circle(img, (15, y), 5, color, -1, cv2.LINE_AA)
                    cv2.putText(img, name, (25, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

                cv2.putText(img, f"Seed {seed_index+1} | {len(self._all_previous_seed_positions)} past seeds",
                            (10, img_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                out_path = str(Path(save_dir) / f"seed_{seed_index+1:02d}_{arm_key}_point.jpg")
                cv2.imwrite(out_path, img)

            print(f"  [SeedGen] Point visualization saved: {save_dir}/seed_{seed_index+1:02d}_*_point.jpg")
        except Exception as e:
            print(f"  [SeedGen] Point visualization warning: {e}")

    # ─────────────────────────────────────────────
    # Code Execution
    # ─────────────────────────────────────────────

    def _get_task_runner(self):
        """DualArmTaskRunner 인스턴스 반환 (lazy init)."""
        if not hasattr(self, '_task_runner') or self._task_runner is None:
            from pipeline.task_runner import DualArmTaskRunner
            self._task_runner = DualArmTaskRunner(
                skills=self.multi_arm,
                recorder=self.dataset_recorder if self.record_dataset else None,
                camera_manager=self.camera_manager,
                recording_fps=self.recording_fps,
            )
        return self._task_runner

    def execute_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """Execute LLM-generated code (TaskRunner에 위임)."""
        runner = self._get_task_runner()
        success = runner.execute(code, positions, extra_globals)
        # DualArmTaskRunner의 레코딩 통계를 파이프라인에 전달
        self._last_mar_stats = getattr(runner, '_last_mar_stats', None)
        return success


    # ─────────────────────────────────────────────
    # Judge
    # ─────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # Multi-turn artifact saving
    # ─────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # Capture helpers
    # ─────────────────────────────────────────────

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a color frame via PipelineCamera."""
        if hasattr(self, '_pipeline_camera') and self._pipeline_camera:
            return self._pipeline_camera.capture_frame()
        if self.camera:
            try:
                color, _ = self.camera.get_frames()
                return color
            except Exception as e:
                print(f"[MultiArm] Frame capture failed: {e}")
        return None

    # ─────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────

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
        Run full bi-arm pipeline for one episode.

        Same interface as ForwardAndResetPipeline.run().

        Flow:
            1. Initialize arms + camera + recording
            2. Capture image → multi-turn VLM code generation (detection inside VLM)
            3. Execute code with both arms
            4. Judge evaluation
            5. Reset (optional)
        """
        self.instruction = instruction

        # Result dir
        if save_dir is None:
            save_dir = "results"
        if use_timestamp_subdir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = str(Path(save_dir) / timestamp)
        else:
            result_dir = str(Path(save_dir))
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        forward_dir = str(Path(result_dir) / "forward")
        reset_dir = str(Path(result_dir) / "reset")
        Path(forward_dir).mkdir(parents=True, exist_ok=True)
        Path(reset_dir).mkdir(parents=True, exist_ok=True)

        result = {
            'forward': {'positions': {}, 'code': '', 'execution_success': False},
            'judge': {'prediction': 'UNCERTAIN', 'reasoning': ''},
            'reset': {'mode': 'original', 'current_positions': {}, 'target_positions': {},
                      'code': '', 'execution_success': False},
            'reset_judge': {'prediction': 'UNCERTAIN', 'reasoning': '', 'reset_mode': 'original'},
            'saved_files': {},
        }

        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"
        print(f"\n{CYAN}{'='*70}{RESET_COLOR}")
        print(f"{CYAN}{BOLD}[{ep_str}] Multi-Arm Forward + Reset Pipeline{RESET_COLOR}")
        print(f"{CYAN}{'='*70}{RESET_COLOR}")

        try:
            # ══════════════════════════════════════
            # PHASE 0: INITIALIZATION
            # ══════════════════════════════════════
            self.current_phase = "Init"

            # Initialize arms (connect는 LLM 코드에서)
            if self.multi_arm is None:
                print(f"\n{YELLOW}" + self._log("Initializing both arms...") + f"{RESET_COLOR}")
                if not self._init_multi_arm():
                    print(f"{RED}[Error] Failed to initialize robot arms{RESET_COLOR}")
                    return result

            # Initialize recording first (camera_manager 생성)
            # → 그 다음 camera는 camera_manager에서 가져옴 (리소스 충돌 방지)
            if self.record_dataset and self.dataset_recorder is None:
                self._init_recording()

            # Initialize camera (camera_manager가 있으면 그것 사용)
            if self.camera is None:
                print(f"{YELLOW}" + self._log("Initializing camera...") + f"{RESET_COLOR}")
                if not self._init_camera():
                    print(f"{RED}[Error] Camera initialization failed{RESET_COLOR}")
                    return result

            # Camera를 양팔 skills에 주입 (detect_objects에서 사용)
            if self.multi_arm is not None and self.camera is not None:
                self.multi_arm.left_arm.camera = self.camera
                self.multi_arm.right_arm.camera = self.camera

            # ══════════════════════════════════════
            # PHASE 1: FORWARD EXECUTION
            # ══════════════════════════════════════
            self.current_phase = "Forward"
            print(f"\n{GREEN}{BOLD}" + self._log("FORWARD EXECUTION") + f"{RESET_COLOR}")
            print(f"{GREEN}{'-'*70}{RESET_COLOR}")

            # Step 1: Capture initial image
            print(f"\n{YELLOW}" + self._log("Capturing initial image...", step="Step 1/5") + f"{RESET_COLOR}")
            initial_image = self._capture_frame()
            if initial_image is not None:
                cv2.imwrite(str(Path(forward_dir) / "initial_state.jpg"), initial_image)
                print(f"  Image captured ({initial_image.shape[1]}x{initial_image.shape[0]})")

            # Step 2: Code generation (with reuse if cached code exists)
            image_path = str(Path(forward_dir) / "initial_state.jpg")

            if self.cached_forward_code is not None:
                # Reuse: T0~T2 only (detection), skip T3 (codegen)
                print(f"\n{YELLOW}" + self._log(f"Detection only (reusing cached code)...", step="Step 2/5") + f"{RESET_COLOR}")
                self._generate_forward_code(
                    instruction, image_path,
                    skip_codegen=True,
                    canonical_labels=self.cached_forward_keys,
                    canonical_point_labels=getattr(self, '_cached_point_labels', None),
                )
                # Verify keys match
                detected_keys = set()
                for arm_data in self.detected_positions.values():
                    if isinstance(arm_data, dict):
                        detected_keys.update(arm_data.keys())
                if set(self.cached_forward_keys) <= detected_keys:
                    code = self.cached_forward_code
                    print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET_COLOR}")
                else:
                    missing = set(self.cached_forward_keys) - detected_keys
                    print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET_COLOR}")
                    self.cached_forward_code = None
                    self.cached_forward_keys = []
                    code = self._generate_forward_code(instruction, image_path)
            else:
                print(f"\n{YELLOW}" + self._log(f"Generating forward code via multi-turn VLM...", step="Step 2/5") + f"{RESET_COLOR}")
                code = self._generate_forward_code(instruction, image_path)

            result['forward']['code'] = code
            result['forward']['positions'] = self.detected_positions

            # Store first episode positions + seed_01 즉시 저장
            if self.first_episode_positions is None and self.detected_positions:
                self.first_episode_positions = copy.deepcopy(self.detected_positions)
                # seed_01_setup 즉시 생성 (세션 디렉토리에)
                try:
                    _session_dir = str(Path(forward_dir).parent.parent)
                    _seed01_dir = Path(_session_dir) / "seed_01_setup"
                    _seed01_dir.mkdir(parents=True, exist_ok=True)
                    with open(_seed01_dir / "seed_positions.json", 'w') as f:
                        json.dump({
                            "seed_index": 0,
                            "positions": self._make_serializable(self.first_episode_positions),
                        }, f, indent=2, default=str)
                    print(f"  [Seed] seed_01_setup saved: {_seed01_dir}/seed_positions.json")
                except Exception as e:
                    print(f"  [Seed] Warning: Failed to save seed_01: {e}")

            # Save generated code
            with open(Path(forward_dir) / "generated_code.py", 'w') as f:
                f.write(code)

            # Save multi-turn info (turn logs, LLM cost, crop images, visualizations)
            from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
            save_multi_turn_info(forward_dir, self.multi_turn_info, phase="forward")
            save_turn_visualizations(forward_dir, self.multi_turn_info, initial_image, phase="forward")

            # Step 3: Code Verification (LLM 기반 코드 검증)
            code_was_cached = (self.cached_forward_code is not None
                               and code == self.cached_forward_code)
            if code_was_cached:
                print(f"\n{YELLOW}" + self._log("Skipping verification (cached code, already verified)...", step="Step 3/5", tag="Verify") + f"{RESET_COLOR}")
            else:
                print(f"\n{YELLOW}" + self._log(f"Verifying generated code via LLM ({self.llm_model})...", step="Step 3/5", tag="Verify") + f"{RESET_COLOR}")
                from verification import verify_generated_code

                max_verification_retries = 2
                for verify_attempt in range(1, max_verification_retries + 1):
                    passed, reason = verify_generated_code(
                        instruction=instruction,
                        generated_code=code,
                        object_positions=self.detected_positions,
                        llm_model=self.llm_model,
                    )

                    if passed:
                        print(f"  {GREEN}[Verify] PASS{RESET_COLOR}")
                        break
                    else:
                        print(f"  {RED}[Verify] FAIL (attempt {verify_attempt}/{max_verification_retries}): {reason}{RESET_COLOR}")

                        if verify_attempt < max_verification_retries:
                            print(f"  {YELLOW}[Verify] Regenerating code...{RESET_COLOR}")
                            code = self._generate_forward_code(instruction, image_path)
                            result['forward']['code'] = code

                            # 재생성된 코드 저장
                            with open(Path(forward_dir) / "generated_code.py", 'w') as f:
                                f.write(code)
                            print(f"  {YELLOW}[Verify] Regenerated code saved{RESET_COLOR}")
                        else:
                            print(f"  {YELLOW}[Verify] Max retries reached, proceeding with current code{RESET_COLOR}")

            # Step 4: Execute code
            print(f"\n{YELLOW}" + self._log("Executing forward code...", step="Step 4/5") + f"{RESET_COLOR}")

            # Set execution dir for skill_detect_results logging
            import builtins as _builtins
            _builtins._current_execution_dir = forward_dir

            # Start episode recording (RecordingContext 기반 — single-arm과 동일)
            if self.record_dataset and self.dataset_recorder:
                self._start_episode_recording(instruction)

            # Turn 2 라벨을 추출하여 detect_objects 재검출 시 재사용
            point_labels = {}
            if self.detected_positions:
                for arm_key in ["left_arm", "right_arm"]:
                    arm_pos = self.detected_positions.get(arm_key, {})
                    for obj_name, obj_info in arm_pos.items():
                        if obj_info and "points" in obj_info and obj_name not in point_labels:
                            point_labels[obj_name] = list(obj_info["points"].keys())
            if point_labels:
                self.multi_arm._point_labels = point_labels

            execution_success = self.execute_code(code, self.detected_positions)
            result['forward']['execution_success'] = execution_success

            # End episode recording
            if self.record_dataset and self.dataset_recorder:
                self._end_episode_recording()

            # Save pixel_moves_overlay
            try:
                for arm in [self.multi_arm.left_arm, self.multi_arm.right_arm]:
                    if hasattr(arm, 'pixel_move_log') and arm.pixel_move_log:
                        grasp_img_path = str(Path(forward_dir) / "turn2_grasp_points.jpg")
                        if os.path.isfile(grasp_img_path):
                            from execution_forward_and_reset import ForwardAndResetPipeline
                            _dummy = object.__new__(ForwardAndResetPipeline)
                            _dummy._visualize_pixel_moves(
                                grasp_img_path, arm.pixel_move_log,
                                str(Path(forward_dir) / "pixel_moves_overlay.jpg"),
                            )
                        break
            except Exception as e:
                print(f"  [Warning] pixel move visualization failed: {e}")

            # Update llm_cost.json with detect_objects token usage
            from pipeline.save_logs import update_llm_cost_with_detect_usage
            update_llm_cost_with_detect_usage(forward_dir, self.multi_arm)

            # Save execution_context.json
            from pipeline.save_logs import save_execution_context
            save_execution_context(
                forward_dir, instruction,
                self._make_serializable(self.detected_positions or {}),
                code, execution_success, robot_ids=self.robot_ids,
            )

            # Capture final image
            time.sleep(1.0)
            final_image = self._capture_frame()
            if final_image is not None:
                cv2.imwrite(str(Path(forward_dir) / "final_state.jpg"), final_image)

            # Save batch_info early (before judge) so resume can detect this episode
            # Judge result will be updated after judge completes
            episode_root = str(Path(forward_dir).parent)
            from pipeline.save_logs import save_batch_info as _save_bi
            _save_bi(episode_root, getattr(self, '_current_batch_index', 0),
                     getattr(self, '_current_slot', 0), "PENDING")

            # Step 4: Judge evaluation
            print(f"\n{YELLOW}" + self._log("Running judge evaluation...", step="Step 5/5") + f"{RESET_COLOR}")
            if initial_image is not None and final_image is not None:
                judge_result = self._run_forward_judge(
                    instruction, initial_image, final_image,
                    object_positions=self.detected_positions or {},
                    executed_code=code,
                )
                result['judge'] = judge_result
                prediction = judge_result.get('prediction', 'UNCERTAIN')
                color = GREEN if prediction == 'TRUE' else RED
                print(f"  Judge: {color}{prediction}{RESET_COLOR}")

            # Post-judge callback
            if post_judge_callback:
                post_judge_callback(result)

            # Cache successful code + object keys + point labels
            judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
            should_cache = execution_success and judge_pred != 'FALSE'
            if should_cache and self.cached_forward_code is None:
                self.cached_forward_code = code
                obj_keys = set()
                point_labels = {}
                if self.detected_positions:
                    for arm_key in ["left_arm", "right_arm"]:
                        arm_pos = self.detected_positions.get(arm_key, {})
                        for obj_name, obj_info in arm_pos.items():
                            obj_keys.add(obj_name)
                            if obj_info and "points" in obj_info and obj_name not in point_labels:
                                point_labels[obj_name] = list(obj_info["points"].keys())
                self.cached_forward_keys = sorted(obj_keys)
                self._cached_point_labels = point_labels if point_labels else None
                print(f"  {GREEN}[CodeReuse] Forward code cached (keys: {self.cached_forward_keys}){RESET_COLOR}")
            elif not execution_success:
                if self.cached_forward_code is not None:
                    print(f"  {YELLOW}[CodeReuse] Cache invalidated (execution failed){RESET_COLOR}")
                self.cached_forward_code = None
                self.cached_forward_keys = []

            # Save forward result + judge
            with open(Path(forward_dir) / "result.json", 'w') as f:
                json.dump(self._make_serializable(result['forward']), f, indent=2)
            from pipeline.save_logs import save_judge_result
            save_judge_result(forward_dir, result['judge'], initial_image, final_image, instruction)

            # Save forward_log.txt (콘솔 로그 요약 + 레코딩 디버그)
            mar_stats = getattr(self, '_last_mar_stats', {})
            fwd_log_lines = [
                f"Instruction: {instruction}",
                f"Robot IDs: {self.robot_ids}",
                f"Execution success: {execution_success}",
                f"Judge: {result['judge'].get('prediction', 'UNCERTAIN')}",
                f"Judge reasoning: {result['judge'].get('reasoning', '')}",
                f"",
                f"=== Recording Debug ===",
                f"# ratio: recorded/expected frames. 1.0=perfect, <1.0=frame drops (fast playback)",
                f"# effective_hz: actual recording loop speed. Should be ~50. Lower=loop starved",
                f"# overruns: loop iterations where work exceeded 20ms budget",
                f"# loop_ms: per-iteration work time (excl sleep). >20ms means loop can't keep 50Hz",
                f"# record_ms: per-frame _record_frame() time (camera capture + frame build + dataset write)",
                f"record_dataset: {self.record_dataset}",
                f"dataset_recorder: {'YES' if self.dataset_recorder else 'NO'}",
                f"resume_recording: {self.resume_recording}",
                f"episode_num: {self.current_episode}",
                f"cached_forward_code: {'YES' if self.cached_forward_code else 'NO'}",
                f"multi_arm_recorder_stats: {mar_stats}",
            ]
            (Path(forward_dir) / "forward_log.txt").write_text("\n".join(fwd_log_lines), encoding="utf-8")

            # Update batch_info.json with judge result (early save was PENDING)
            judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
            episode_root = str(Path(forward_dir).parent)
            _save_bi(episode_root, getattr(self, '_current_batch_index', 0),
                     getattr(self, '_current_slot', 0), judge_pred)

            # ══════════════════════════════════════
            # PHASE 2: RESET EXECUTION
            # ══════════════════════════════════════
            if not skip_reset:
                self.current_phase = "Reset"
                print(f"\n{GREEN}{BOLD}" + self._log("RESET EXECUTION") + f"{RESET_COLOR}")
                print(f"{GREEN}{'-'*70}{RESET_COLOR}")

                # Pre-reset callback (seed transition)
                if pre_reset_callback:
                    reset_target = pre_reset_callback(current_positions=self.detected_positions)
                    if reset_target:
                        reset_target_positions = reset_target

                # Target positions: use provided or first episode positions
                target_positions = reset_target_positions or self.first_episode_positions or {}

                result['reset']['target_positions'] = target_positions

                # Capture current scene image for reset (= initial state for reset)
                print(f"\n{YELLOW}" + self._log("Capturing reset scene...") + f"{RESET_COLOR}")
                reset_image = self._capture_frame()
                if reset_image is not None:
                    cv2.imwrite(str(Path(reset_dir) / "current_state.jpg"), reset_image)
                    print(f"  Reset scene captured")
                # initial_state = forward 시작 전 이미지 (되돌려야 할 상태)
                import shutil
                forward_initial = Path(forward_dir) / "initial_state.jpg"
                if forward_initial.exists():
                    shutil.copy2(str(forward_initial), str(Path(reset_dir) / "initial_state.jpg"))
                reset_image_path = str(Path(reset_dir) / "current_state.jpg")

                # Save reset positions
                reset_positions_save = {
                    "current_positions": self._make_serializable(self.detected_positions or {}),
                    "target_positions": self._make_serializable(target_positions),
                }
                with open(Path(reset_dir) / "positions.json", 'w') as f:
                    json.dump(reset_positions_save, f, indent=2, default=str)

                # Generate reset code via multi-turn VLM pipeline
                print(f"\n{YELLOW}" + self._log("Generating reset code (multi-turn VLM)...") + f"{RESET_COLOR}")
                initial_image_path = str(Path(forward_dir) / "initial_state.jpg")
                reset_code = self._generate_reset_code(
                    original_instruction=instruction,
                    original_positions=target_positions,
                    current_state_image_path=reset_image_path,
                    initial_state_image_path=initial_image_path,
                )
                result['reset']['code'] = reset_code

                with open(Path(reset_dir) / "generated_code.py", 'w') as f:
                    f.write(reset_code)

                # Save reset multi-turn artifacts (turn logs, visualizations, crop images)
                from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
                save_multi_turn_info(reset_dir, self.reset_multi_turn_info, phase="reset")
                save_turn_visualizations(reset_dir, self.reset_multi_turn_info, reset_image, phase="reset")

                # Execute reset
                print(f"\n{YELLOW}" + self._log("Executing reset code...") + f"{RESET_COLOR}")

                # Set execution dir for skill_detect_results logging
                _builtins._current_execution_dir = reset_dir

                if self.record_dataset and self.dataset_recorder:
                    self._start_episode_recording(self.reset_instruction)

                # current_positions / target_positions를 globals로 주입 (싱글암 패턴과 동일)
                reset_current = getattr(self, '_reset_current_positions', {})
                reset_target = target_positions
                reset_success = self.execute_code(reset_code, {}, extra_globals={
                    "current_positions": reset_current,
                    "target_positions": reset_target,
                })
                result['reset']['execution_success'] = reset_success

                if self.record_dataset and self.dataset_recorder:
                    self._end_episode_recording()

                # Capture reset final image
                time.sleep(1.0)
                reset_final_image = self._capture_frame()
                if reset_final_image is not None:
                    cv2.imwrite(str(Path(reset_dir) / "final_state.jpg"), reset_final_image)

                # Save reset result + judge
                with open(Path(reset_dir) / "result.json", 'w') as f:
                    json.dump(self._make_serializable(result['reset']), f, indent=2)

                # Reset judge
                reset_judge_result = self._run_reset_judge(
                    reset_mode="original",
                    current_positions=self._make_serializable(self.detected_positions or {}),
                    target_positions=self._make_serializable(target_positions),
                    initial_image=reset_image,
                    final_image=reset_final_image,
                    executed_code=reset_code,
                    original_instruction=instruction,
                )
                result['reset_judge'] = reset_judge_result
                if reset_judge_result.get('prediction') != 'UNCERTAIN':
                    with open(Path(reset_dir) / "reset_judge_result.json", 'w') as f:
                        json.dump(reset_judge_result, f, indent=2, default=str)

                # Save reset_log.txt
                reset_log_lines = [
                    f"Instruction: {instruction}",
                    f"Reset success: {reset_success}",
                    f"Reset judge: {result.get('reset_judge', {}).get('prediction', 'UNCERTAIN')}",
                ]
                (Path(reset_dir) / "reset_log.txt").write_text("\n".join(reset_log_lines), encoding="utf-8")

        except AssertionError:
            raise
        except Exception as e:
            print(f"\n{RED}[MultiArm] Pipeline error: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()

        return result

    # ─────────────────────────────────────────────
    # Multi-episode execution
    # ─────────────────────────────────────────────

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
        """
        Run multiple episodes sequentially.
        Same interface as ForwardAndResetPipeline.run_multiple_episodes().
        """
        self.total_episodes = num_episodes
        self.instruction = instruction

        # Create session directory
        if save_dir is None:
            save_dir = "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(save_dir) / f"session_{timestamp}")
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        # Results accumulator
        all_results = {
            'session_dir': session_dir,
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'robot_ids': self.robot_ids,
            'episodes': [],
            'summary': {
                'forward_success': 0,
                'forward_judge_true': 0,
                'reset_success': 0,
                'reset_judge_true': 0,
            },
        }

        episodes_per_seed = max(1, num_episodes // max(1, self.num_random_seeds))
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        print(f"\n{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"{MAGENTA}{BOLD}  MULTI-ARM SESSION: {num_episodes} Episodes (robots {self.robot_ids})  {RESET_COLOR}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Save Dir: {session_dir}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")

        for episode_idx in range(num_episodes):
            episode_num = episode_idx + 1
            batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
            slot = episode_idx % episodes_per_seed
            is_batch_last = (slot == episodes_per_seed - 1)
            next_batch_index = batch_index + 1
            self.current_episode = episode_num
            self._current_batch_index = batch_index
            self._current_slot = slot

            print(f"\n{CYAN}{'='*70}{RESET_COLOR}")
            print(f"{CYAN}{BOLD}  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1})  {RESET_COLOR}")
            print(f"{CYAN}{'='*70}{RESET_COLOR}")

            episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")
            reset_target = seed_positions[batch_index]

            # 배치 마지막이면: Forward 후 다음 seed 생성 → Reset target 갱신 콜백
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
                        print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{next_idx+1}...{RESET_COLOR}")
                        sp[next_idx] = self._generate_seed_positions(sd, next_idx, current_positions=current_positions)
                    return sp[next_idx]
                pre_reset_cb = _make_next_seed

            try:
                result = self.run(
                    instruction=instruction,
                    objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir,
                    use_timestamp_subdir=False,
                    skip_reset=skip_reset,
                    reset_target_positions=reset_target,
                    pre_reset_callback=pre_reset_cb,
                )

                # Save batch_info.json
                judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                from pipeline.save_logs import save_batch_info
                save_batch_info(episode_dir, batch_index, slot, judge_pred)

                # Store first episode positions for seed[0]
                if seed_positions[0] is None and self.first_episode_positions is not None:
                    seed_positions[0] = copy.deepcopy(self.first_episode_positions)
                    self._all_previous_seed_positions.append(seed_positions[0])

                # Update summary
                s = all_results['summary']
                if result['forward']['execution_success']:
                    s['forward_success'] += 1
                if result['judge'].get('prediction') == 'TRUE':
                    s['forward_judge_true'] += 1
                if result['reset']['execution_success']:
                    s['reset_success'] += 1
                if result.get('reset_judge', {}).get('prediction') == 'TRUE':
                    s['reset_judge_true'] += 1

                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': result,
                    'success': result['forward']['execution_success'],
                })

            except AssertionError:
                raise
            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET_COLOR}")
                import traceback
                traceback.print_exc()
                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': None,
                    'success': False,
                    'error': str(e),
                })

            time.sleep(2)

        # Finalize
        self._finalize_session(all_results, session_dir)
        return all_results


    def _load_resume_state(self, session_dir: str):
        """이전 세션에서 상태 복원 (batch_info.json 기반).

        Returns:
            (batch_slots, seed_positions): slot 성공 여부 + seed positions
        """
        episodes_per_seed = max(1, self.total_episodes // max(1, self.num_random_seeds))
        batch_slots = [[False] * episodes_per_seed for _ in range(self.num_random_seeds)]
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # 1. first_episode_positions 복원
        ctx_path = Path(session_dir) / "episode_01" / "forward" / "execution_context.json"
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            self.first_episode_positions = ctx.get("object_positions", {})
            seed_positions[0] = copy.deepcopy(self.first_episode_positions)
            print(f"  [Resume] first_episode_positions restored")

        # 2. batch_info.json 기반으로 slot별 성공 여부 파악
        for ep_dir in sorted(Path(session_dir).glob("episode_*")):
            bi_path = ep_dir / "batch_info.json"
            if not bi_path.exists():
                continue
            with open(bi_path) as f:
                bi = json.load(f)
            batch_idx = bi.get("batch_seed_index", 1) - 1
            slot = bi.get("slot", -1)
            judge = bi.get("judge", "")
            if 0 <= batch_idx < self.num_random_seeds and 0 <= slot < episodes_per_seed:
                if judge == "TRUE":
                    batch_slots[batch_idx][slot] = True

        # 3. cached_forward_code 복원
        for ep_dir in sorted(Path(session_dir).glob("episode_*"), reverse=True):
            fwd_code = ep_dir / "forward" / "generated_code.py"
            judge_json = ep_dir / "forward" / "judge_result.json"
            if fwd_code.exists() and judge_json.exists():
                with open(judge_json) as f:
                    jr = json.load(f)
                pred = jr.get("prediction", jr.get("judge_result", {}).get("prediction", ""))
                if pred == "TRUE":
                    self.cached_forward_code = fwd_code.read_text().strip()
                    # Extract object keys from execution_context
                    ec_path = ep_dir / "forward" / "execution_context.json"
                    if ec_path.exists():
                        with open(ec_path) as f:
                            ec = json.load(f)
                        obj_keys = set()
                        point_labels = {}
                        for arm_key in ["left_arm", "right_arm"]:
                            arm_pos = ec.get("object_positions", {}).get(arm_key, {})
                            for obj_name, obj_info in arm_pos.items():
                                obj_keys.add(obj_name)
                                if isinstance(obj_info, dict) and "points" in obj_info:
                                    point_labels[obj_name] = list(obj_info["points"].keys())
                        self.cached_forward_keys = sorted(obj_keys)
                        self._cached_point_labels = point_labels if point_labels else None
                    print(f"  [Resume] cached_forward_code restored")
                    break

        # 요약 출력
        for i in range(self.num_random_seeds):
            done = sum(batch_slots[i])
            print(f"    Batch {i+1}: {done}/{episodes_per_seed}")

        return batch_slots, seed_positions

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
        """Resume: 이전 세션의 미완료 에피소드만 재시도."""
        self.total_episodes = num_episodes
        self.instruction = instruction

        session_dir = str(resume_session_dir)
        episodes_per_seed = max(1, num_episodes // max(1, self.num_random_seeds))

        # Load previous state
        print(f"\n{YELLOW}[Resume] Loading previous session: {session_dir}{RESET_COLOR}")
        batch_slots, seed_positions = self._load_resume_state(session_dir)

        # Check if all done
        all_done = all(all(slots) for slots in batch_slots)
        if all_done:
            print(f"\n{GREEN}  All episodes complete, nothing to resume{RESET_COLOR}")
            return {'session_dir': session_dir, 'episodes': [], 'summary': {}}

        # Cleanup failed episodes from dataset, then initialize recording (resume/append)
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
                    print(f"\n{YELLOW}[Cleanup] Warning: Dataset cleanup failed: {e}{RESET_COLOR}")
                    import traceback; traceback.print_exc()

            if self.dataset_recorder is None:
                self.resume_recording = True
                self._init_recording()

        # Results
        all_results = {
            'session_dir': session_dir,
            'num_episodes': num_episodes,
            'instruction': instruction,
            'robot_ids': self.robot_ids,
            'episodes': [],
            'summary': {'forward_success': 0, 'forward_judge_true': 0,
                        'reset_success': 0, 'reset_judge_true': 0},
        }

        print(f"\n{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"{MAGENTA}{BOLD}  RESUME MULTI-ARM SESSION: {num_episodes} Episodes  {RESET_COLOR}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")

        for episode_idx in range(num_episodes):
            episode_num = episode_idx + 1
            batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
            slot = episode_idx % episodes_per_seed
            self.current_episode = episode_num
            self._current_batch_index = batch_index
            self._current_slot = slot

            # Skip completed slots
            if batch_slots[batch_index][slot]:
                print(f"  [{episode_num:02d}/{num_episodes:02d}] Batch {batch_index+1} Slot {slot} — already TRUE, skipping")
                continue

            print(f"\n{CYAN}{'='*70}{RESET_COLOR}")
            print(f"{CYAN}{BOLD}  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1}, Slot {slot})  {RESET_COLOR}")
            print(f"{CYAN}{'='*70}{RESET_COLOR}")

            episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")
            reset_target = seed_positions[batch_index]

            try:
                result = self.run(
                    instruction=instruction,
                    objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir,
                    use_timestamp_subdir=False,
                    skip_reset=skip_reset,
                    reset_target_positions=reset_target,
                )

                judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                from pipeline.save_logs import save_batch_info
                save_batch_info(episode_dir, batch_index, slot, judge_pred)

                if seed_positions[0] is None and self.first_episode_positions is not None:
                    seed_positions[0] = copy.deepcopy(self.first_episode_positions)

                s = all_results['summary']
                if result['forward']['execution_success']:
                    s['forward_success'] += 1
                if judge_pred == 'TRUE':
                    s['forward_judge_true'] += 1
                if result['reset']['execution_success']:
                    s['reset_success'] += 1
                if result.get('reset_judge', {}).get('prediction') == 'TRUE':
                    s['reset_judge_true'] += 1

                all_results['episodes'].append({
                    'episode': episode_num, 'result': result, 'success': result['forward']['execution_success'],
                })

            except AssertionError:
                raise
            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}] Error: {e}{RESET_COLOR}")
                import traceback
                traceback.print_exc()
                all_results['episodes'].append({'episode': episode_num, 'success': False, 'error': str(e)})

            time.sleep(2)

        self._finalize_session(all_results, session_dir)
        return all_results

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    @staticmethod
    def _extract_position_keys(code: str) -> List[str]:
        """Extract positions['xxx'] pattern keys from code."""
        import re
        pattern = r'positions\[["\'](.+?)["\']\]'
        return list(dict.fromkeys(re.findall(pattern, code)))

    def _finalize_session(self, all_results: Dict, session_dir: str):
        """Print summary and save session results."""
        n = all_results['num_episodes']
        s = all_results['summary']

        print(f"\n{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"{MAGENTA}{BOLD}  SESSION SUMMARY  {RESET_COLOR}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"  Episodes: {n}")
        print(f"  Robot IDs: {self.robot_ids}")
        if n > 0:
            rate = s['forward_judge_true'] / n * 100
            color = GREEN if rate >= 80 else YELLOW if rate >= 50 else RED
            print(f"  Forward Judge TRUE: {color}{s['forward_judge_true']}/{n} ({rate:.1f}%){RESET_COLOR}")
            print(f"  Forward Success: {s['forward_success']}/{n}")
            print(f"  Reset Success: {s['reset_success']}/{n}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")

        # Save summary
        summary_path = Path(session_dir) / "session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self._make_serializable(all_results), f, indent=2)
        print(f"  Results saved: {summary_path}")

        # Finalize recording
        self._finalize_recording()

        # Camera cleanup: camera_manager가 소유한 카메라는 stop하지 않음
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
            except Exception:
                pass
            self.camera_manager = None
            self.camera = None  # camera_manager가 관리하므로 참조만 해제
        elif self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

    @staticmethod
    def _make_serializable(obj):
        """Make result dict JSON-serializable."""
        if isinstance(obj, dict):
            return {k: UnifiedMultiArmPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [UnifiedMultiArmPipeline._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, bytes):
            return "<bytes>"
        return obj
