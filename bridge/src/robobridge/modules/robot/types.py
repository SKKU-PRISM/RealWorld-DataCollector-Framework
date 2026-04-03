"""
Robot module types and data classes.

Defines robot state, execution results, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RobotExecutionState(Enum):
    """Robot execution states."""

    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    ESTOP = "estop"


@dataclass
class RobotState:
    """Robot state information (simplified for wrapper interface)."""

    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    ee_pose: Optional[Dict[str, Any]] = None
    is_moving: bool = False
    timestamp: float = 0.0


@dataclass
class ExecutionResult:
    """Result of command execution."""

    success: bool
    command_id: str = ""
    state: str = "completed"
    actual_positions: Optional[List[float]] = None
    execution_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "success": self.success,
            "state": self.state,
            "actual_positions": self.actual_positions,
            "execution_time_s": self.execution_time_s,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            success=data.get("success", False),
            command_id=data.get("command_id", ""),
            state=data.get("state", "completed"),
            actual_positions=data.get("actual_positions"),
            execution_time_s=data.get("execution_time_s", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RobotStateData:
    """Current robot state data."""

    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_torques: List[float] = field(default_factory=list)
    ee_pose: Optional[Dict[str, Any]] = None  # position + orientation
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    gripper_qpos: List[float] = field(default_factory=list)  # raw gripper joint positions (2D)
    robot_mode: str = "idle"
    errors: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_torques": self.joint_torques,
            "ee_pose": self.ee_pose,
            "gripper_width": self.gripper_width,
            "gripper_force": self.gripper_force,
            "gripper_qpos": self.gripper_qpos,
            "robot_mode": self.robot_mode,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


@dataclass
class TrajectoryPoint:
    """Single trajectory point."""

    positions: List[float]  # Joint positions
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None
    time_from_start: float = 0.0


@dataclass
class GripperCommand:
    """Gripper command."""

    action: str  # "open", "close", "move"
    width: float = 0.08  # Target width in meters
    speed: float = 0.1  # Speed in m/s
    force: float = 40.0  # Grasp force in N


@dataclass
class RobotConfig:
    """Robot interface specific configuration."""

    robot_type: str = "franka"
    rate_hz: float = 100.0
    timeout_s: float = 15.0
    units: str = "SI"
    frame_convention: str = "base"
    estop_policy: str = "stop_and_report"
    # Topics
    command_topic: str = "/planning/low_level_cmd"
    result_topic: str = "/robot/execution_result"
    state_topic: str = "/robot/state"
