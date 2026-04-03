"""
Forward Workspace - Forward 태스크용 추가 제약

BaseWorkspace를 상속하고 pick & place 작업에 필요한 제약 추가.
"""

import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot_cap.workspace import BaseWorkspace

if TYPE_CHECKING:
    from lerobot_cap.kinematics.engine import KinematicsEngine


class ForwardWorkspace(BaseWorkspace):
    """
    Forward 태스크 전용 Workspace.

    BaseWorkspace를 상속하고 추가 제약:
    - Z 범위 제한 (테이블 ~ pitch 유지 가능 높이)
    """

    def __init__(
        self,
        kinematics_engine: Optional["KinematicsEngine"] = None,
        z_min_world: float = 0.01,
        z_max_world: float = 0.18,
    ):
        """
        Initialize ForwardWorkspace.

        Args:
            kinematics_engine: KinematicsEngine 인스턴스
            z_min_world: 최소 Z 높이 (테이블 표면, world frame)
            z_max_world: 최대 Z 높이 (pitch 유지 가능 범위, world frame)
        """
        super().__init__(kinematics_engine)

        # Forward 전용 제약 (world frame)
        self.z_min_world = z_min_world
        self.z_max_world = z_max_world

    def is_valid(self, position_world: np.ndarray) -> bool:
        """
        Forward용 유효성 검사.

        1. 기본 검사 (Kinematics reach)
        2. Forward 추가 제약 (Z 범위)

        Args:
            position_world: [x, y, z] 위치 (world frame)

        Returns:
            True if valid for forward
        """
        # 1. 기본 검사 (BaseWorkspace)
        if not self.is_reachable(position_world):
            return False

        # 2. Forward 추가 제약
        z = position_world[2]
        return self.z_min_world <= z <= self.z_max_world

    def __repr__(self) -> str:
        return f"ForwardWorkspace(z_world=[{self.z_min_world:.2f}, {self.z_max_world:.2f}])"


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("ForwardWorkspace Test")
    print("=" * 60)

    # Create workspace
    ws = ForwardWorkspace()
    print(f"Workspace: {ws}")
    print(f"Base reach: {ws.get_reach_limits()}")
    print()

    # Test positions
    test_positions = [
        [0.15, 0.0, 0.10],   # Valid
        [0.15, 0.0, 0.0],   # Below z_min
        [0.15, 0.0, 0.25],   # Above z_max
        [0.50, 0.0, 0.10],   # Out of reach
        [0.02, 0.0, 0.10],   # Below min reach
    ]

    for pos in test_positions:
        reachable = ws.is_reachable(np.array(pos))
        valid = ws.is_valid(np.array(pos))
        print(f"  {pos} -> reachable={reachable}, valid={valid}")
