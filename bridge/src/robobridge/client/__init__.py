"""
RoboBridge Client - User-friendly API for robot control

Provides a simple, high-level API for interacting with robots.

Example:
    from robobridge import RoboBridge

    robot = RoboBridge.initialize()
    robot.execute("Pick up the red cup")
    robot.pick("red_cup")
    robot.shutdown()
"""

from .client import RoboBridge, RoboBridgeClient
from .types import ExecutionResult, RobotState, DetectedObject, ModelInfo

__all__ = [
    "RoboBridge",
    "RoboBridgeClient",
    "ExecutionResult",
    "RobotState",
    "DetectedObject",
    "ModelInfo",
]
