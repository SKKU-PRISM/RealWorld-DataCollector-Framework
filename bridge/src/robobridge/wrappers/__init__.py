"""
Custom Model Wrappers for RoboBridge Modules

These wrappers allow you to integrate your own custom models into RoboBridge
by simply implementing the inference method and matching the input/output format.

Available wrappers:
- CustomPerception: For object detection models
- CustomPlanner: For high-level planning (LLM) models
- CustomController: For low-level planning (VLA) models
- CustomRobot: For robot hardware interfaces
- CustomMonitor: For feedback/monitoring (VLM) models
"""

from .custom_perception import CustomPerception
from .custom_planner import CustomPlanner
from .custom_controller import CustomController
from .custom_robot import CustomRobot
from .custom_monitor import CustomMonitor

__all__ = [
    "CustomPerception",
    "CustomPlanner",
    "CustomController",
    "CustomRobot",
    "CustomMonitor",
]
