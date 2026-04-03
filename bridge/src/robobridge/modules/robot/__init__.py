"""
Robot Module

Interface for robot control and state management.
"""

from .robot import Robot
from .types import (
    ExecutionResult,
    GripperCommand,
    RobotConfig,
    RobotState,
    RobotStateData,
    TrajectoryPoint,
)

__all__ = [
    "Robot",
    "ExecutionResult",
    "GripperCommand",
    "RobotConfig",
    "RobotState",
    "RobotStateData",
    "TrajectoryPoint",
]
