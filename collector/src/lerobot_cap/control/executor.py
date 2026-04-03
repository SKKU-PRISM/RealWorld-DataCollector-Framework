"""
Trajectory Executor

Executes trajectories on real robot hardware.
"""

import time
from typing import Optional, Callable
import numpy as np

from lerobot_cap.hardware.base import BaseController
from lerobot_cap.planning.trajectory import Trajectory
from lerobot_cap.control.safety import SafetySystem


class TrajectoryExecutor:
    """
    Executes trajectories on robot hardware.

    Handles timing, safety checks, and progress callbacks.

    Args:
        robot: Robot controller instance
        safety: Safety system (optional)
        check_interval: How often to check for stop conditions (seconds)
    """

    def __init__(
        self,
        robot: BaseController,
        safety: Optional[SafetySystem] = None,
        check_interval: float = 0.01,
    ):
        self.robot = robot
        self.safety = safety or SafetySystem(robot.num_joints)
        self.check_interval = check_interval

        self._stop_requested = False

    def execute(
        self,
        trajectory: Trajectory,
        blocking: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Execute a trajectory.

        Args:
            trajectory: Trajectory to execute
            blocking: If True, wait for completion
            progress_callback: Called with progress (0.0 to 1.0)

        Returns:
            True if completed successfully
        """
        self._stop_requested = False

        if blocking:
            return self._execute_blocking(trajectory, progress_callback)
        else:
            # TODO: Implement non-blocking execution with threading
            return self._execute_blocking(trajectory, progress_callback)

    def _execute_blocking(
        self,
        trajectory: Trajectory,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Execute trajectory with blocking."""
        start_time = time.time()
        duration = trajectory.duration

        print(f"Executing trajectory: {trajectory.num_points} points, {duration:.2f}s")

        while not self._stop_requested:
            elapsed = time.time() - start_time

            if elapsed >= duration:
                # Final point
                final_positions = trajectory.joint_positions[-1]

                # Safety check
                is_safe, msg = self.safety.check_joint_limits(final_positions)
                if not is_safe:
                    print(f"Safety violation: {msg}")
                    return False

                # Clamp and send
                clamped = self.safety.clamp_to_limits(final_positions)
                self._send_to_robot(clamped)

                if progress_callback:
                    progress_callback(1.0)

                print("Trajectory completed")
                return True

            # Get state at current time
            positions = trajectory.get_state_at_time(elapsed)

            # Safety check
            is_safe, msg = self.safety.check_joint_limits(positions)
            if not is_safe:
                print(f"Safety violation at t={elapsed:.2f}s: {msg}")
                return False

            # Clamp and send
            clamped = self.safety.clamp_to_limits(positions)
            self._send_to_robot(clamped)

            # Update safety tracking
            self.safety.update_last_command(clamped, time.time())

            # Progress callback
            if progress_callback:
                progress = elapsed / duration
                progress_callback(progress)

            # Sleep for control rate
            time.sleep(self.check_interval)

        print("Trajectory execution stopped")
        return False

    def _send_to_robot(self, positions: np.ndarray):
        """Send positions to robot (convert radians to normalized if needed)."""
        # Assuming positions are in radians, convert to normalized (-100 to +100)
        # This assumes ±π radians maps to ±100
        normalized = positions / np.pi * 100.0
        self.robot.write_positions(normalized, normalize=True)

    def stop(self):
        """Request stop of current execution."""
        self._stop_requested = True

    def execute_single_position(
        self,
        target_positions: np.ndarray,
        duration: float = 0.0,
    ) -> bool:
        """
        Move to a single target position.

        Args:
            target_positions: Target joint positions (radians)
            duration: Time to reach target (0 = immediate)

        Returns:
            True if successful
        """
        # Safety check
        is_safe, msg = self.safety.check_joint_limits(target_positions)
        if not is_safe:
            print(f"Safety violation: {msg}")
            return False

        clamped = self.safety.clamp_to_limits(target_positions)

        if duration <= 0:
            # Immediate
            self._send_to_robot(clamped)
            return True

        # Create simple trajectory
        current = self.robot.read_positions(normalize=True) / 100.0 * np.pi
        trajectory = Trajectory(
            joint_positions=np.array([current, clamped]),
            timestamps=np.array([0.0, duration]),
        )

        return self.execute(trajectory)


class JointPositionController:
    """
    Simple joint position controller for direct commands.

    Args:
        robot: Robot controller
        safety: Safety system (optional)
    """

    def __init__(
        self,
        robot: BaseController,
        safety: Optional[SafetySystem] = None,
    ):
        self.robot = robot
        self.safety = safety or SafetySystem(robot.num_joints)

    def set_positions(self, positions: np.ndarray, normalized: bool = True) -> bool:
        """
        Set joint positions directly.

        Args:
            positions: Target positions (normalized or radians)
            normalized: If True, positions are in -100 to +100 range

        Returns:
            True if command was sent
        """
        if not normalized:
            # Convert radians to normalized
            positions = positions / np.pi * 100.0

        # Safety check
        is_safe, msg = self.safety.check_joint_limits(positions)
        if not is_safe:
            print(f"Safety violation: {msg}")
            return False

        clamped = self.safety.clamp_to_limits(positions)
        self.robot.write_positions(clamped, normalize=True)
        return True

    def get_positions(self, normalized: bool = True) -> np.ndarray:
        """
        Get current joint positions.

        Args:
            normalized: If True, return -100 to +100 range

        Returns:
            Current joint positions
        """
        positions = self.robot.read_positions(normalize=True)
        if normalized:
            return positions
        else:
            return positions / 100.0 * np.pi
