"""
LeRobot CaP Distillation

Unified robot control framework with IK, motion planning, and learning.
"""

__version__ = "0.1.0"

from lerobot_cap.hardware import FeetechController, DynamixelController, CameraController
from lerobot_cap.kinematics import KinematicsEngine
from lerobot_cap.planning import TrajectoryPlanner
from lerobot_cap.control import TrajectoryExecutor, SafetySystem

__all__ = [
    "FeetechController",
    "DynamixelController",
    "CameraController",
    "KinematicsEngine",
    "TrajectoryPlanner",
    "TrajectoryExecutor",
    "SafetySystem",
]
