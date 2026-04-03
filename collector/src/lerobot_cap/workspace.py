"""
Base Workspace - Kinematics 기반 Workspace

모든 태스크별 workspace는 이것을 상속하여 추가 제약을 건다.

계층 구조:
    BaseWorkspace (Kinematics 기반)
    ├── ResetWorkspace (reset_execution/workspace.py)
    ├── ForwardWorkspace (forward_execution/workspace.py)
    └── ...

좌표계:
    - Base_link frame: 로봇 기준 좌표계 (reach limits 적용)
"""

import math
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot_cap.kinematics.engine import KinematicsEngine


class BaseWorkspace:
    """
    Kinematics 기반 기본 Workspace.

    Reach limits (원형 도달 범위) 기반 제약만 적용.

    Attributes:
        min_reach: 최소 도달 거리 (m, base_link frame)
        max_reach: 최대 도달 거리 (m, base_link frame)
        z_floor: 바닥 높이 (m)
    """

    def __init__(
        self,
        kinematics_engine: Optional["KinematicsEngine"] = None,
        min_reach: float = 0.22,
        max_reach: float = 0.407,
        z_floor: float = -0.02,  # 캘리브레이션 오차 허용 (-2cm)
        **kwargs,
    ):
        """
        Initialize BaseWorkspace.

        Args:
            kinematics_engine: KinematicsEngine 인스턴스 (reach limits 자동 로드)
            min_reach: 최소 도달 거리 (kinematics_engine 없을 때 사용)
            max_reach: 최대 도달 거리 (kinematics_engine 없을 때 사용)
            z_floor: 바닥 높이
        """
        if kinematics_engine is not None:
            # KinematicsEngine의 reach를 기본 안전 범위로 clamp
            self.min_reach = max(kinematics_engine.min_reach, min_reach)
            self.max_reach = min(kinematics_engine.max_reach, max_reach)
            self._kinematics = kinematics_engine
        else:
            self.min_reach = min_reach
            self.max_reach = max_reach
            self._kinematics = None

        self.z_floor = z_floor

    def is_reachable(
        self,
        position: np.ndarray,
        margin: float = 0.01,  # 1cm 안전 마진
    ) -> bool:
        """
        도달 가능 여부 검사 (base_link frame).

        Args:
            position: [x, y, z] 위치 (base_link frame, meters)
            margin: 안전 여유 (meters)

        Returns:
            True if reachable
        """
        return self._check_reach_base(position, margin)

    def _check_reach_base(
        self,
        position_base: np.ndarray,
        margin: float = 0.01,  # 1cm 안전 마진
    ) -> bool:
        """
        Base_link frame에서 reach limits 검사.

        Args:
            position_base: [x, y, z] 위치 (base_link frame, meters)
            margin: 안전 여유 (meters)

        Returns:
            True if within reach
        """
        # 항상 workspace의 clamp된 reach 범위 사용
        x, y, z = position_base

        # Z floor check
        if z < self.z_floor:
            return False

        # Horizontal distance (XY plane)
        horizontal_distance = math.sqrt(x * x + y * y)

        # 3D distance
        distance_3d = math.sqrt(x * x + y * y + z * z)

        # Check reach limits
        max_horizontal = self.max_reach - margin
        min_horizontal = self.min_reach + margin

        is_within_horizontal = min_horizontal <= horizontal_distance <= max_horizontal
        is_within_3d = distance_3d <= (self.max_reach - margin)

        return is_within_horizontal and is_within_3d

    def get_reach_limits(self) -> tuple:
        """
        Reach 범위 반환.

        Returns:
            (min_reach, max_reach) in meters
        """
        return (self.min_reach, self.max_reach)

    def __repr__(self) -> str:
        return (f"BaseWorkspace(min_reach={self.min_reach:.3f}, "
                f"max_reach={self.max_reach:.3f})")


# Singleton instance for convenience (initialized lazily)
_base_workspace: Optional[BaseWorkspace] = None


def get_base_workspace(
    kinematics_engine: Optional["KinematicsEngine"] = None,
) -> BaseWorkspace:
    """
    BaseWorkspace 싱글톤 인스턴스 반환.

    Args:
        kinematics_engine: KinematicsEngine (첫 호출 시 필요)

    Returns:
        BaseWorkspace instance
    """
    global _base_workspace

    if _base_workspace is None:
        _base_workspace = BaseWorkspace(kinematics_engine)

    return _base_workspace
