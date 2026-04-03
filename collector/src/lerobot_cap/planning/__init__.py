"""
Planning Layer

Motion planning and trajectory generation.
Uses OMPL for path planning (optional) and custom trajectory interpolation.
"""

from lerobot_cap.planning.trajectory import TrajectoryPlanner, Trajectory
from lerobot_cap.planning.interpolation import (
    linear_interpolation,
    cubic_interpolation,
    slerp_interpolation,
)

__all__ = [
    "TrajectoryPlanner",
    "Trajectory",
    "linear_interpolation",
    "cubic_interpolation",
    "slerp_interpolation",
]
