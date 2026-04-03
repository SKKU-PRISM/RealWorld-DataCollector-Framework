"""
Controller module types and data classes.

Defines trajectory points, commands, and configuration for motion control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    positions: List[float]  # Joint positions or Cartesian pose
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None
    rotations: Optional[List[float]] = None  # Euler angles [roll, pitch, yaw] in degrees
    time_from_start: float = 0.0

    def to_dict(self) -> dict:
        result = {
            "positions": self.positions,
            "velocities": self.velocities,
            "accelerations": self.accelerations,
            "time_from_start": self.time_from_start,
        }
        if self.rotations is not None:
            result["rotations"] = self.rotations
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryPoint":
        return cls(
            positions=data.get("positions", []),
            velocities=data.get("velocities"),
            accelerations=data.get("accelerations"),
            rotations=data.get("rotations"),
            time_from_start=data.get("time_from_start", 0.0),
        )


@dataclass
class Command:
    """Low-level command for robot execution."""

    command_id: str
    command_type: str  # trajectory, waypoints, joint_targets, vel_cmd
    points: List[TrajectoryPoint] = field(default_factory=list)
    frame_id: str = "base"
    gripper_command: Optional[Dict[str, Any]] = None  # open, close, width
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "frame_id": self.frame_id,
            "points": [p.to_dict() for p in self.points],
            "gripper_command": self.gripper_command,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Command":
        return cls(
            command_id=data.get("command_id", ""),
            command_type=data.get("command_type", ""),
            points=[TrajectoryPoint.from_dict(p) for p in data.get("points", [])],
            frame_id=data.get("frame_id", "base"),
            gripper_command=data.get("gripper_command"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ControllerConfig:
    """Controller module specific configuration."""

    backend: str = "vla"
    temperature: float = 0.5
    action_space: str = "trajectory"  # trajectory, waypoints, joint_targets, vel_cmd
    control_rate_hz: float = 20.0
    horizon_steps: int = 10
    frame_convention: str = "base"
    safety_limits: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_joint_vel": 1.5,
            "max_ee_vel": 0.3,
            "workspace": [-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
        }
    )
    # Topics
    plan_topic: str = "/planning/high_level_plan"  # Legacy
    primitive_plan_topic: str = "/planning/primitive_plan"
    rgb_topic: str = "/camera/rgb"
    depth_topic: str = "/camera/depth"
    robot_state_topic: str = "/robot/state"
    output_topic: str = "/planning/low_level_cmd"
