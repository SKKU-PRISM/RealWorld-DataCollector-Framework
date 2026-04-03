"""
Trajectory Interpolation Methods

Linear, cubic, and SLERP interpolation for smooth trajectories.
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def linear_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int = 50,
) -> np.ndarray:
    """
    Linear interpolation between two configurations.

    Args:
        start: Start configuration
        end: End configuration
        num_points: Number of interpolation points

    Returns:
        Array of shape (num_points, len(start))
    """
    t = np.linspace(0, 1, num_points)
    return np.outer(1 - t, start) + np.outer(t, end)


def cubic_interpolation(
    waypoints: np.ndarray,
    num_points_per_segment: int = 20,
) -> np.ndarray:
    """
    Cubic spline interpolation through waypoints.

    Args:
        waypoints: Array of waypoints (num_waypoints, num_joints)
        num_points_per_segment: Points per segment

    Returns:
        Smoothly interpolated trajectory
    """
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        raise ValueError("Need at least 2 waypoints")

    # Create parameter t
    t_waypoints = np.linspace(0, 1, num_waypoints)
    t_interp = np.linspace(0, 1, (num_waypoints - 1) * num_points_per_segment + 1)

    # Interpolate each joint
    trajectory = np.zeros((len(t_interp), waypoints.shape[1]))
    for j in range(waypoints.shape[1]):
        cs = CubicSpline(t_waypoints, waypoints[:, j], bc_type='clamped')
        trajectory[:, j] = cs(t_interp)

    return trajectory


def smoothstep(t: float) -> float:
    """Smoothstep function for smooth acceleration/deceleration."""
    return 3 * t**2 - 2 * t**3


def smootherstep(t: float) -> float:
    """Smoother step function (quintic)."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def smooth_linear_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int = 50,
    smooth_type: str = "smoothstep",
) -> np.ndarray:
    """
    Linear interpolation with smooth acceleration/deceleration.

    Args:
        start: Start configuration
        end: End configuration
        num_points: Number of points
        smooth_type: "smoothstep" or "smootherstep"

    Returns:
        Smoothly interpolated trajectory
    """
    t_linear = np.linspace(0, 1, num_points)

    if smooth_type == "smootherstep":
        t_smooth = np.array([smootherstep(t) for t in t_linear])
    else:
        t_smooth = np.array([smoothstep(t) for t in t_linear])

    return np.outer(1 - t_smooth, start) + np.outer(t_smooth, end)


def slerp_interpolation(
    start_rotation: np.ndarray,
    end_rotation: np.ndarray,
    num_points: int = 50,
) -> List[np.ndarray]:
    """
    Spherical linear interpolation for rotations.

    Args:
        start_rotation: Start rotation matrix (3x3)
        end_rotation: End rotation matrix (3x3)
        num_points: Number of points

    Returns:
        List of rotation matrices
    """
    rotations = Rotation.from_matrix([start_rotation, end_rotation])
    slerp = Slerp([0, 1], rotations)

    t = np.linspace(0, 1, num_points)
    interpolated = slerp(t)

    return [r.as_matrix() for r in interpolated]


def interpolate_pose(
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    end_position: np.ndarray,
    end_rotation: np.ndarray,
    num_points: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Interpolate full 6-DOF pose (position + orientation).

    Uses smooth linear for position and SLERP for rotation.

    Args:
        start_position: Start position [3]
        start_rotation: Start rotation matrix [3x3]
        end_position: End position [3]
        end_rotation: End rotation matrix [3x3]
        num_points: Number of points

    Returns:
        Tuple of (positions [num_points, 3], rotations [num_points, 3x3])
    """
    positions = smooth_linear_interpolation(start_position, end_position, num_points)
    rotations = slerp_interpolation(start_rotation, end_rotation, num_points)

    return positions, rotations


def s_curve_profile(t: float, accel_ratio: float = 0.3) -> float:
    """
    S-curve velocity profile for smooth acceleration/deceleration.

    Creates a 3-phase profile:
    1. Acceleration phase (0 to accel_ratio): smooth ramp up
    2. Constant velocity phase (accel_ratio to 1-accel_ratio)
    3. Deceleration phase (1-accel_ratio to 1): smooth ramp down

    The acceleration/deceleration phases use smooth S-curves (no jerk discontinuity).

    Args:
        t: Normalized time [0, 1]
        accel_ratio: Fraction of time spent accelerating (and decelerating)

    Returns:
        Position along trajectory [0, 1]
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0

    decel_start = 1.0 - accel_ratio

    if t < accel_ratio:
        # Acceleration phase - use smoothstep for S-curve
        phase_t = t / accel_ratio
        # Smoothstep gives smooth acceleration profile
        smooth_t = 3 * phase_t**2 - 2 * phase_t**3
        # Scale to end at the velocity plateau position
        return smooth_t * accel_ratio * 0.5

    elif t < decel_start:
        # Constant velocity phase
        # At end of accel: position = accel_ratio * 0.5
        # Need to cover: 1 - 2 * (accel_ratio * 0.5) = 1 - accel_ratio in const_vel phase
        const_vel_duration = decel_start - accel_ratio
        const_vel_distance = 1.0 - accel_ratio
        velocity = const_vel_distance / const_vel_duration
        time_in_phase = t - accel_ratio
        return accel_ratio * 0.5 + velocity * time_in_phase

    else:
        # Deceleration phase - mirror of acceleration
        phase_t = (t - decel_start) / accel_ratio
        smooth_t = 3 * phase_t**2 - 2 * phase_t**3
        # Position at start of decel phase
        start_pos = 1.0 - accel_ratio * 0.5
        return start_pos + smooth_t * accel_ratio * 0.5


def s_curve_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int = 50,
    accel_ratio: float = 0.3,
) -> np.ndarray:
    """
    S-curve interpolation with jerk-limited velocity profile.

    Args:
        start: Start configuration
        end: End configuration
        num_points: Number of points
        accel_ratio: Fraction of time for acceleration/deceleration

    Returns:
        Smoothly interpolated trajectory with S-curve velocity profile
    """
    t_linear = np.linspace(0, 1, num_points)
    t_scurve = np.array([s_curve_profile(t, accel_ratio) for t in t_linear])

    return np.outer(1 - t_scurve, start) + np.outer(t_scurve, end)


def time_parameterize_trajectory(
    trajectory: np.ndarray,
    max_velocity: float,
    max_acceleration: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add time parameterization to trajectory.

    Args:
        trajectory: Joint trajectory (num_points, num_joints)
        max_velocity: Maximum joint velocity
        max_acceleration: Maximum joint acceleration

    Returns:
        Tuple of (timestamps, trajectory)
    """
    num_points = len(trajectory)
    if num_points < 2:
        return np.array([0.0]), trajectory

    # Compute segment lengths in configuration space
    segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Compute time for each segment based on velocity limit
    segment_times = segment_lengths / max_velocity

    # Ensure minimum time for acceleration
    min_segment_time = np.sqrt(2 * segment_lengths.max() / max_acceleration)
    segment_times = np.maximum(segment_times, min_segment_time)

    # Cumulative time
    timestamps = np.zeros(num_points)
    timestamps[1:] = np.cumsum(segment_times)

    return timestamps, trajectory
