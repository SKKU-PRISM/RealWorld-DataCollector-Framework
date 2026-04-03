"""
Safety System

Joint limits, workspace bounds, and safety checks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


@dataclass
class SafetyLimits:
    """Safety limit configuration."""
    # Joint limits (normalized -100 to +100 or radians)
    joint_min: np.ndarray
    joint_max: np.ndarray

    # Velocity limits
    max_velocity: float = 50.0  # normalized units/s

    # Workspace bounds (meters) - DEPRECATED
    # Use BaseWorkspace from src/lerobot_cap/workspace.py instead
    # These are kept for backwards compatibility only
    workspace_min: np.ndarray = None
    workspace_max: np.ndarray = None

    def __post_init__(self):
        if self.workspace_min is None:
            self.workspace_min = np.array([-0.5, -0.5, 0.0])
        if self.workspace_max is None:
            self.workspace_max = np.array([0.5, 0.5, 0.6])


class SafetySystem:
    """
    Safety checks for robot control.

    Validates commands before execution to prevent damage.

    Args:
        num_joints: Number of joints
        limits: Safety limits configuration
    """

    def __init__(
        self,
        num_joints: int = 6,
        limits: Optional[SafetyLimits] = None,
    ):
        self.num_joints = num_joints

        if limits is None:
            # Default limits for normalized coordinates (-100 to +100)
            self.limits = SafetyLimits(
                joint_min=np.full(num_joints, -100.0),
                joint_max=np.full(num_joints, 100.0),
            )
        else:
            self.limits = limits

        # Track previous command for velocity checking
        self.last_command: Optional[np.ndarray] = None
        self.last_command_time: float = 0.0

    def check_joint_limits(self, positions: np.ndarray) -> Tuple[bool, str]:
        """
        Check if positions are within joint limits.

        Args:
            positions: Joint positions to check

        Returns:
            (is_safe, error_message)
        """
        for i, pos in enumerate(positions):
            if pos < self.limits.joint_min[i]:
                return False, f"Joint {i} below minimum: {pos:.2f} < {self.limits.joint_min[i]:.2f}"
            if pos > self.limits.joint_max[i]:
                return False, f"Joint {i} above maximum: {pos:.2f} > {self.limits.joint_max[i]:.2f}"

        return True, ""

    def check_workspace(self, ee_position: np.ndarray) -> Tuple[bool, str]:
        """
        Check if EE position is within workspace bounds.

        Args:
            ee_position: End-effector position [x, y, z]

        Returns:
            (is_safe, error_message)
        """
        for i, (pos, axis) in enumerate(zip(ee_position, ['x', 'y', 'z'])):
            if pos < self.limits.workspace_min[i]:
                return False, f"EE {axis} below minimum: {pos:.3f} < {self.limits.workspace_min[i]:.3f}"
            if pos > self.limits.workspace_max[i]:
                return False, f"EE {axis} above maximum: {pos:.3f} > {self.limits.workspace_max[i]:.3f}"

        return True, ""

    def check_velocity(
        self,
        positions: np.ndarray,
        current_time: float,
    ) -> Tuple[bool, str]:
        """
        Check if velocity is within limits.

        Args:
            positions: Target joint positions
            current_time: Current time in seconds

        Returns:
            (is_safe, error_message)
        """
        if self.last_command is None:
            return True, ""

        dt = current_time - self.last_command_time
        if dt <= 0:
            return True, ""

        velocity = np.abs(positions - self.last_command) / dt
        max_vel = np.max(velocity)

        if max_vel > self.limits.max_velocity:
            return False, f"Velocity too high: {max_vel:.2f} > {self.limits.max_velocity:.2f}"

        return True, ""

    def clamp_to_limits(self, positions: np.ndarray) -> np.ndarray:
        """
        Clamp positions to joint limits.

        Args:
            positions: Joint positions

        Returns:
            Clamped positions
        """
        return np.clip(positions, self.limits.joint_min, self.limits.joint_max)

    def validate_command(
        self,
        positions: np.ndarray,
        ee_position: Optional[np.ndarray] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Full validation of a command.

        Args:
            positions: Joint positions
            ee_position: End-effector position (optional)
            current_time: Current time for velocity check (optional)

        Returns:
            (is_safe, error_message)
        """
        # Joint limits
        is_safe, msg = self.check_joint_limits(positions)
        if not is_safe:
            return False, msg

        # Workspace bounds
        if ee_position is not None:
            is_safe, msg = self.check_workspace(ee_position)
            if not is_safe:
                return False, msg

        # Velocity limits
        if current_time is not None:
            is_safe, msg = self.check_velocity(positions, current_time)
            if not is_safe:
                return False, msg

        return True, ""

    def update_last_command(self, positions: np.ndarray, time: float):
        """Update the last command for velocity tracking."""
        self.last_command = positions.copy()
        self.last_command_time = time

    def reset(self):
        """Reset safety system state."""
        self.last_command = None
        self.last_command_time = 0.0
