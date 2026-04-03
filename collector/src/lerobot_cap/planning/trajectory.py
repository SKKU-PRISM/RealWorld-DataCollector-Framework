"""
Trajectory Planner

Plans trajectories from current state to target EE pose.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from lerobot_cap.kinematics import KinematicsEngine
from lerobot_cap.planning.interpolation import (
    smooth_linear_interpolation,
    s_curve_interpolation,
    cubic_interpolation,
    time_parameterize_trajectory,
)

if TYPE_CHECKING:
    from lerobot_cap.kinematics.calibration_limits import CalibrationJointLimits


@dataclass
class Trajectory:
    """
    Robot trajectory data structure.

    Attributes:
        joint_positions: Joint positions over time (num_points, num_joints)
        timestamps: Time for each point (seconds)
        ee_positions: End-effector positions (optional)
        ik_converged: Whether IK solver converged successfully
        expected_position_error: Expected position error in meters (if IK computed)
        target_position: Target position that was requested (optional)
        joints_within_limits: Whether all target joints are within limits
        joint_limit_violations: List of (joint_index, joint_name, value, limit) for violations
    """
    joint_positions: np.ndarray
    timestamps: np.ndarray
    ee_positions: Optional[np.ndarray] = None
    ik_converged: bool = True
    expected_position_error: Optional[float] = None
    target_position: Optional[np.ndarray] = None
    joints_within_limits: bool = True
    joint_limit_violations: Optional[List[Tuple[int, str, float, float]]] = None

    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def num_points(self) -> int:
        """Number of trajectory points."""
        return len(self.timestamps)

    def get_state_at_time(self, t: float) -> np.ndarray:
        """
        Get interpolated joint state at given time.

        Args:
            t: Time in seconds

        Returns:
            Joint positions at time t
        """
        if t <= self.timestamps[0]:
            return self.joint_positions[0]
        if t >= self.timestamps[-1]:
            return self.joint_positions[-1]

        # Find surrounding points
        idx = np.searchsorted(self.timestamps, t)
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        q0, q1 = self.joint_positions[idx - 1], self.joint_positions[idx]

        # Linear interpolation
        alpha = (t - t0) / (t1 - t0)
        return q0 + alpha * (q1 - q0)


class TrajectoryPlanner:
    """
    Plans trajectories to target end-effector poses.

    Args:
        kinematics: KinematicsEngine instance
        max_velocity: Maximum joint velocity (rad/s)
        max_acceleration: Maximum joint acceleration (rad/s^2)
        interpolation_points: Number of interpolation points
        calibration_limits: Optional CalibrationJointLimits for actual robot limits
        use_s_curve: Use S-curve interpolation (jerk-limited) instead of smoothstep
        s_curve_accel_ratio: Fraction of trajectory time for acceleration/deceleration
    """

    def __init__(
        self,
        kinematics: KinematicsEngine,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0,
        interpolation_points: int = 50,
        calibration_limits: Optional["CalibrationJointLimits"] = None,
        use_s_curve: bool = False,
        s_curve_accel_ratio: float = 0.3,
    ):
        self.kinematics = kinematics
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.interpolation_points = interpolation_points
        self.calibration_limits = calibration_limits
        self.use_s_curve = use_s_curve
        self.s_curve_accel_ratio = s_curve_accel_ratio

    def _interpolate(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Select interpolation method based on settings."""
        if self.use_s_curve:
            return s_curve_interpolation(
                start, end, self.interpolation_points, self.s_curve_accel_ratio
            )
        else:
            return smooth_linear_interpolation(
                start, end, self.interpolation_points
            )

    def plan_to_position(
        self,
        target_position: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
    ) -> Trajectory:
        """
        Plan trajectory to target EE position.

        Args:
            target_position: Target [x, y, z] in meters
            current_joints: Current joint positions (radians)
            duration: Trajectory duration (auto-compute if None)

        Returns:
            Trajectory object
        """
        # Get current joints if not provided
        if current_joints is None:
            current_joints = np.zeros(self.kinematics.num_joints)

        # Solve IK for target
        target_joints, success = self.kinematics.inverse_kinematics_position_only(
            target_position,
            initial_guess=current_joints,
        )

        # Calculate expected position error
        final_ee_position = self.kinematics.get_ee_position(target_joints)
        position_error = np.linalg.norm(target_position - final_ee_position)

        # Check joint limits (use calibration limits if available, otherwise URDF limits)
        if self.calibration_limits is not None:
            # Use calibration-based limits (actual robot physical limits)
            joints_within_limits, joint_limit_violations = self.calibration_limits.is_within_limits(target_joints)
        else:
            # Use URDF-based limits (theoretical design limits)
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_names = self.kinematics.joint_names

            joints_within_limits = True
            joint_limit_violations = []

            for i, (val, lower, upper, name) in enumerate(zip(target_joints, joint_lower, joint_upper, joint_names)):
                if val < lower:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, lower))
                elif val > upper:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, upper))

        # Interpolate in joint space
        joint_trajectory = self._interpolate(current_joints, target_joints)

        # Time parameterization
        if duration is not None:
            timestamps = np.linspace(0, duration, self.interpolation_points)
        else:
            timestamps, _ = time_parameterize_trajectory(
                joint_trajectory,
                self.max_velocity,
                self.max_acceleration,
            )

        # Compute EE positions
        ee_positions = np.array([
            self.kinematics.get_ee_position(q)
            for q in joint_trajectory
        ])

        return Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
            ee_positions=ee_positions,
            ik_converged=success,
            expected_position_error=position_error,
            target_position=target_position,
            joints_within_limits=joints_within_limits,
            joint_limit_violations=joint_limit_violations if joint_limit_violations else None,
        )

    def plan_to_position_multi(
        self,
        target_position: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        num_random_samples: int = 10,
        verbose: bool = False,
        fixed_joints: Optional[List[int]] = None,
        target_pitch: Optional[float] = None,
    ) -> Tuple[Trajectory, dict]:
        """
        Plan trajectory to target EE position using multi-solution IK.

        Tries multiple initial guesses to find valid solutions within joint limits,
        then selects the best one closest to current joints.

        Args:
            target_position: Target [x, y, z] in meters
            current_joints: Current joint positions (radians)
            duration: Trajectory duration (auto-compute if None)
            num_random_samples: Number of random IK attempts
            verbose: Print IK search progress
            fixed_joints: List of joint indices to keep fixed (0-indexed)
                         e.g., [4] to fix wrist_roll
            target_pitch: Target gripper pitch in radians. If specified, uses
                         pitch-constrained IK to maintain gripper orientation.

        Returns:
            Tuple of (Trajectory object, ik_info dict)
        """
        # Get current joints if not provided
        if current_joints is None:
            current_joints = np.zeros(self.kinematics.num_joints)

        # Get custom limits from calibration if available
        custom_limits = None
        if self.calibration_limits is not None:
            custom_limits = (
                self.calibration_limits.lower_limits_radians,
                self.calibration_limits.upper_limits_radians,
            )

        # Solve IK with multi-solution search
        if verbose:
            print(f"  Multi-IK search for target {target_position}...")

        target_joints, success, ik_info = self.kinematics.inverse_kinematics_multi(
            target_position,
            current_joints=current_joints,
            custom_limits=custom_limits,
            num_random_samples=num_random_samples,
            verbose=verbose,
            fixed_joints=fixed_joints,
            target_pitch=target_pitch,
        )

        # Calculate expected position error
        final_ee_position = self.kinematics.get_ee_position(target_joints)
        position_error = np.linalg.norm(target_position - final_ee_position)

        # Check joint limits
        if self.calibration_limits is not None:
            joints_within_limits, joint_limit_violations = self.calibration_limits.is_within_limits(target_joints)
        else:
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_names = self.kinematics.joint_names

            joints_within_limits = True
            joint_limit_violations = []

            for i, (val, lower, upper, name) in enumerate(zip(target_joints, joint_lower, joint_upper, joint_names)):
                if val < lower:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, lower))
                elif val > upper:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, upper))

        # Interpolate in joint space
        joint_trajectory = self._interpolate(current_joints, target_joints)

        # Time parameterization
        if duration is not None:
            timestamps = np.linspace(0, duration, self.interpolation_points)
        else:
            timestamps, _ = time_parameterize_trajectory(
                joint_trajectory,
                self.max_velocity,
                self.max_acceleration,
            )

        # Compute EE positions
        ee_positions = np.array([
            self.kinematics.get_ee_position(q)
            for q in joint_trajectory
        ])

        trajectory = Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
            ee_positions=ee_positions,
            ik_converged=success,
            expected_position_error=position_error,
            target_position=target_position,
            joints_within_limits=joints_within_limits,
            joint_limit_violations=joint_limit_violations if joint_limit_violations else None,
        )

        return trajectory, ik_info

    def plan_to_pose_multi(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        num_random_samples: int = 10,
        verbose: bool = False,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-1,
    ) -> Tuple[Trajectory, dict]:
        """
        Plan trajectory to target EE pose using multi-solution IK with orientation.

        Tries multiple initial guesses to find valid solutions that satisfy
        both position AND orientation constraints.

        Args:
            target_position: Target [x, y, z] in meters
            target_orientation: Target rotation matrix (3x3)
            current_joints: Current joint positions (radians)
            duration: Trajectory duration (auto-compute if None)
            num_random_samples: Number of random IK attempts
            verbose: Print IK search progress
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)

        Returns:
            Tuple of (Trajectory object, ik_info dict)
        """
        # Get current joints if not provided
        if current_joints is None:
            current_joints = np.zeros(self.kinematics.num_joints)

        # Get custom limits from calibration if available
        custom_limits = None
        if self.calibration_limits is not None:
            custom_limits = (
                self.calibration_limits.lower_limits_radians,
                self.calibration_limits.upper_limits_radians,
            )

        # Solve IK with multi-solution search (with orientation)
        if verbose:
            print(f"  Multi-IK (with orientation) search for target {target_position}...")

        target_joints, success, ik_info = self.kinematics.inverse_kinematics_with_orientation_multi(
            target_position,
            target_orientation=target_orientation,
            current_joints=current_joints,
            custom_limits=custom_limits,
            num_random_samples=num_random_samples,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            verbose=verbose,
        )

        # Calculate expected position error
        final_ee_position = self.kinematics.get_ee_position(target_joints)
        position_error = np.linalg.norm(target_position - final_ee_position)

        # Check joint limits
        if self.calibration_limits is not None:
            joints_within_limits, joint_limit_violations = self.calibration_limits.is_within_limits(target_joints)
        else:
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_names = self.kinematics.joint_names

            joints_within_limits = True
            joint_limit_violations = []

            for i, (val, lower, upper, name) in enumerate(zip(target_joints, joint_lower, joint_upper, joint_names)):
                if val < lower:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, lower))
                elif val > upper:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, upper))

        # Interpolate in joint space
        joint_trajectory = self._interpolate(current_joints, target_joints)

        # Time parameterization
        if duration is not None:
            timestamps = np.linspace(0, duration, self.interpolation_points)
        else:
            timestamps, _ = time_parameterize_trajectory(
                joint_trajectory,
                self.max_velocity,
                self.max_acceleration,
            )

        # Compute EE positions
        ee_positions = np.array([
            self.kinematics.get_ee_position(q)
            for q in joint_trajectory
        ])

        trajectory = Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
            ee_positions=ee_positions,
            ik_converged=success,
            expected_position_error=position_error,
            target_position=target_position,
            joints_within_limits=joints_within_limits,
            joint_limit_violations=joint_limit_violations if joint_limit_violations else None,
        )

        return trajectory, ik_info

    def plan_to_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        orientation_tolerance: float = 0.3,
    ) -> Trajectory:
        """
        Plan trajectory to target EE pose (position + orientation).

        Args:
            target_position: Target [x, y, z] in meters
            target_orientation: Target rotation matrix (3x3)
            current_joints: Current joint positions
            duration: Trajectory duration
            orientation_tolerance: Orientation error tolerance in radians

        Returns:
            Trajectory object
        """
        if current_joints is None:
            current_joints = np.zeros(self.kinematics.num_joints)

        target_joints, success = self.kinematics.inverse_kinematics(
            target_position,
            target_orientation,
            initial_guess=current_joints,
            orientation_tolerance=orientation_tolerance,
        )

        # Calculate expected position error
        final_ee_position = self.kinematics.get_ee_position(target_joints)
        position_error = np.linalg.norm(target_position - final_ee_position)

        # Check joint limits (use calibration limits if available, otherwise URDF limits)
        if self.calibration_limits is not None:
            # Use calibration-based limits (actual robot physical limits)
            joints_within_limits, joint_limit_violations = self.calibration_limits.is_within_limits(target_joints)
        else:
            # Use URDF-based limits (theoretical design limits)
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_names = self.kinematics.joint_names

            joints_within_limits = True
            joint_limit_violations = []

            for i, (val, lower, upper, name) in enumerate(zip(target_joints, joint_lower, joint_upper, joint_names)):
                if val < lower:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, lower))
                elif val > upper:
                    joints_within_limits = False
                    joint_limit_violations.append((i, name, val, upper))

        joint_trajectory = self._interpolate(current_joints, target_joints)

        if duration is not None:
            timestamps = np.linspace(0, duration, self.interpolation_points)
        else:
            timestamps, _ = time_parameterize_trajectory(
                joint_trajectory,
                self.max_velocity,
                self.max_acceleration,
            )

        return Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
            ik_converged=success,
            expected_position_error=position_error,
            target_position=target_position,
            joints_within_limits=joints_within_limits,
            joint_limit_violations=joint_limit_violations if joint_limit_violations else None,
        )

    def plan_to_joints(
        self,
        target_joints: np.ndarray,
        current_joints: np.ndarray,
        duration: Optional[float] = None,
    ) -> Trajectory:
        """
        Plan trajectory directly in joint space.

        Args:
            target_joints: Target joint positions (radians)
            current_joints: Current joint positions
            duration: Trajectory duration

        Returns:
            Trajectory object
        """
        joint_trajectory = self._interpolate(current_joints, target_joints)

        if duration is not None:
            timestamps = np.linspace(0, duration, self.interpolation_points)
        else:
            timestamps, _ = time_parameterize_trajectory(
                joint_trajectory,
                self.max_velocity,
                self.max_acceleration,
            )

        return Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
        )

    def plan_waypoints(
        self,
        waypoint_positions: List[np.ndarray],
        current_joints: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """
        Plan trajectory through multiple EE waypoints.

        Args:
            waypoint_positions: List of [x, y, z] positions
            current_joints: Current joint positions

        Returns:
            Trajectory object
        """
        if current_joints is None:
            current_joints = np.zeros(self.kinematics.num_joints)

        # Solve IK for each waypoint
        joint_waypoints = [current_joints]
        prev_joints = current_joints

        for target_pos in waypoint_positions:
            target_joints, success = self.kinematics.inverse_kinematics_position_only(
                target_pos,
                initial_guess=prev_joints,
            )
            joint_waypoints.append(target_joints)
            prev_joints = target_joints

        # Cubic interpolation through waypoints
        joint_trajectory = cubic_interpolation(np.array(joint_waypoints))

        # Time parameterization
        timestamps, _ = time_parameterize_trajectory(
            joint_trajectory,
            self.max_velocity,
            self.max_acceleration,
        )

        return Trajectory(
            joint_positions=joint_trajectory,
            timestamps=timestamps,
        )
