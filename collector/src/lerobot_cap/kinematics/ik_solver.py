"""
IK Solver

High-level inverse kinematics interface.
"""

from typing import Optional, Tuple
import numpy as np

from lerobot_cap.kinematics.engine import KinematicsEngine


class IKSolver:
    """
    High-level IK solver wrapper.

    Args:
        kinematics: KinematicsEngine instance
    """

    def __init__(self, kinematics: KinematicsEngine):
        self.kinematics = kinematics

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        position_only: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for target pose.

        Args:
            target_position: [x, y, z] in meters
            target_orientation: 3x3 rotation matrix (optional)
            initial_guess: Initial joint positions (optional)
            position_only: If True, ignore orientation

        Returns:
            (joint_positions, success)
        """
        if position_only or target_orientation is None:
            return self.kinematics.inverse_kinematics_position_only(
                target_position, initial_guess
            )
        else:
            return self.kinematics.inverse_kinematics(
                target_position, target_orientation, initial_guess
            )

    def solve_with_current_joints(
        self,
        target_position: np.ndarray,
        current_joints: np.ndarray,
        position_only: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve IK using current joints as initial guess (closed-loop mode).

        Args:
            target_position: [x, y, z] in meters
            current_joints: Current joint positions
            position_only: If True, ignore orientation

        Returns:
            (joint_positions, success)
        """
        return self.solve(target_position, initial_guess=current_joints, position_only=position_only)
