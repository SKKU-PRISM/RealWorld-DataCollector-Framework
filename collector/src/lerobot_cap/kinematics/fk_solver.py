"""
FK Solver

High-level forward kinematics interface.
"""

from typing import Tuple
import numpy as np

from lerobot_cap.kinematics.engine import KinematicsEngine


class FKSolver:
    """
    High-level FK solver wrapper.

    Args:
        kinematics: KinematicsEngine instance
    """

    def __init__(self, kinematics: KinematicsEngine):
        self.kinematics = kinematics

    def solve(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for given joint positions.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            (position [3], rotation_matrix [3x3])
        """
        return self.kinematics.forward_kinematics(joint_positions)

    def get_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get only the position.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            [x, y, z] in meters
        """
        return self.kinematics.get_ee_position(joint_positions)

    def get_transform(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get 4x4 homogeneous transformation matrix.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            4x4 transformation matrix
        """
        return self.kinematics.get_ee_pose(joint_positions)
