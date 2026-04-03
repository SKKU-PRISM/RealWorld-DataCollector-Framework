"""
DryRunSkills — LeRobotSkills의 가상 롤아웃 대체 클래스.

실제 로봇 연결 없이 IK 검증만 수행.
생성된 코드를 그대로 실행하되, 모든 동작을 IK 체크로 대체.

사용:
    from skills.dry_run_skills import DryRunSkills as LeRobotSkills
    exec(code, {"LeRobotSkills": DryRunSkills, "positions": positions})
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from lerobot_cap.kinematics.engine import KinematicsEngine


class DryRunSkills:
    """LeRobotSkills와 동일한 인터페이스, IK 검증만 수행."""

    _last_instance = None

    def __init__(self, robot_config: str = "", frame: str = "base_link", **kwargs):
        DryRunSkills._last_instance = self
        self.frame = frame
        self.pick_offset = kwargs.get("pick_offset", 0.02)
        self.kinematics = None
        self._saved_pitch = None
        self._pick_z = None
        self.ik_failures = []  # 실패 기록
        self._all_ok = True

        # robot_config에서 URDF 경로 추출
        try:
            import yaml
            config_path = Path(robot_config)
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                urdf_path = config.get("kinematics", {}).get("urdf_path", "")
                if Path(urdf_path).exists():
                    self.kinematics = KinematicsEngine(
                        str(urdf_path),
                        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                    )
        except Exception:
            pass

    def connect(self):
        return True

    def disconnect(self):
        pass

    def move_to_initial_state(self, **kwargs):
        return True

    def move_to_free_state(self, **kwargs):
        return True

    def gripper_open(self, **kwargs):
        pass

    def gripper_close(self, **kwargs):
        pass

    @property
    def success(self) -> bool:
        return self._all_ok

    def move_to_position(
        self,
        position,
        duration=None,
        maintain_wrist_roll=True,
        maintain_pitch=False,
        target_pitch=None,
        target_name=None,
        skill_description=None,
        verification_question=None,
    ) -> bool:
        if self.kinematics is None:
            return True

        pos = np.array(position)

        if target_pitch is not None:
            # pitch 제약 IK
            _, success = self.kinematics.inverse_kinematics_position_with_pitch(
                pos, target_pitch=target_pitch, max_iterations=50,
            )
        else:
            # position-only IK
            _, success = self.kinematics.inverse_kinematics_position_only(
                pos, max_iterations=50,
            )

        if not success:
            self.ik_failures.append({
                "position": pos.tolist(),
                "pitch": float(np.degrees(target_pitch)) if target_pitch else None,
                "target": target_name,
            })
            self._all_ok = False

        return success

    def execute_pick_object(
        self,
        object_position,
        object_name=None,
        skill_description=None,
        verification_question=None,
    ) -> bool:
        object_position = np.array(object_position)
        object_height = object_position[2]

        MIN_PICK_Z = 0.005  # match skills_lerobot.py
        pick_z = max(object_height - self.pick_offset, MIN_PICK_Z)
        pick_position = np.array([object_position[0], object_position[1], pick_z])

        # IK로 pick → pitch 획득
        if self.kinematics is not None:
            # position-only IK로 풀어서 pitch 추출
            joints, success = self.kinematics.inverse_kinematics_position_only(
                pick_position, max_iterations=50,
            )
            if success:
                self._saved_pitch = self.kinematics.get_gripper_pitch(joints)
                self._pick_z = pick_z
            else:
                self.ik_failures.append({
                    "action": "pick",
                    "position": pick_position.tolist(),
                    "object": object_name,
                })
                self._all_ok = False
                return False

        return True

    def execute_place_object(
        self,
        place_position,
        is_table=True,
        gripper_open_ratio=1.0,
        target_name=None,
        skill_description=None,
        verification_question=None,
    ) -> bool:
        place_position = np.array(place_position)
        target_surface_height = 0.0 if is_table else place_position[2]

        MIN_PLACE_Z = 0.005  # Minimum place height (0.5cm) — match skills_lerobot.py
        if is_table:
            place_z = max(place_position[2] - self.pick_offset, MIN_PLACE_Z)
        else:
            pick_z = self._pick_z or self.pick_offset
            place_z = max(target_surface_height + pick_z, MIN_PLACE_Z)

        final_position = np.array([place_position[0], place_position[1], place_z])

        # 저장된 pitch로 place IK
        return self.move_to_position(
            final_position,
            target_pitch=self._saved_pitch,
            target_name=target_name,
        )

    def set_subtask(self, object_name=None, current_position=None, target_position=None):
        """Sub-task label (no-op in dry run)."""
        pass

    def clear_subtask(self):
        """Clear sub-task label (no-op in dry run)."""
        pass

    def move_to_pixel(self, pixel, **kwargs) -> bool:
        """Pixel-based move (no-op in dry run — no pix2robot available)."""
        return True

    def execute_place_at_pixel(self, pixel, **kwargs) -> bool:
        """Pixel-based place (no-op in dry run)."""
        return True

    def rotate_90degree(self, direction=1, **kwargs) -> bool:
        """Rotate gripper (no-op in dry run)."""
        return True

    def detect_objects(self, queries, **kwargs):
        return {q: None for q in queries}
