"""
Planner module types and data structures.

VLM-based single-stage planning:
Planner: Natural language + images -> Primitive actions (move, grip)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


# =============================================================================
# Position Type
# =============================================================================


@dataclass
class Position3D:
    """3D position in robot base frame."""

    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_dict(cls, data: Dict) -> "Position3D":
        return cls(x=data["x"], y=data["y"], z=data["z"])


# =============================================================================
# Rotation Type
# =============================================================================


@dataclass
class Rotation3D:
    """3D rotation in euler angles (degrees)."""

    roll: float
    pitch: float
    yaw: float

    def to_dict(self) -> dict:
        return {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw}

    def to_list(self) -> List[float]:
        return [self.roll, self.pitch, self.yaw]

    @classmethod
    def from_dict(cls, data: Dict) -> "Rotation3D":
        return cls(
            roll=data.get("roll", 0.0),
            pitch=data.get("pitch", 0.0),
            yaw=data.get("yaw", 0.0),
        )


# =============================================================================
# High-Level Action Type
# =============================================================================


@dataclass
class HighLevelAction:
    """
    High-level action (pick, place, etc.).

    Used as parent reference for primitive actions.
    """

    action_id: int
    action_type: str  # pick, place, push, pull, open, close, etc.
    target_object: str
    target_location: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target_object": self.target_object,
            "target_location": self.target_location,
            "parameters": self.parameters,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HighLevelAction":
        return cls(
            action_id=data.get("action_id", 0),
            action_type=data.get("action_type", ""),
            target_object=data.get("target_object", ""),
            target_location=data.get("target_location"),
            parameters=data.get("parameters", {}),
            status=data.get("status", "pending"),
        )


# =============================================================================
# Primitive Action Types (Planner output)
# =============================================================================


PrimitiveType = Literal["move", "grip"]


@dataclass
class PrimitiveAction:
    """
    Low-level primitive action from Planner.

    Two types of primitives:
    - move: Arm end-effector movement to target position and orientation
    - grip: Gripper open/close control
    """

    primitive_id: int
    primitive_type: PrimitiveType  # "move", "grip"

    # For move: target position (robot-base frame)
    target_position: Optional[Position3D] = None

    # For move: target rotation (euler degrees)
    target_rotation: Optional[Rotation3D] = None

    # For grip: gripper width (0.0 = fully closed, 0.08 = fully open)
    grip_width: Optional[float] = None

    # Primitive instruction for VLA training/inference
    # e.g., "move to (0.53, 0.01, 0.77, -161.6, -5.7, 72.8)" or "grip close"
    instruction: str = ""

    # Reference to parent high-level action
    parent_action_id: int = 0

    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> dict:
        result = {
            "primitive_id": self.primitive_id,
            "primitive_type": self.primitive_type,
            "parent_action_id": self.parent_action_id,
            "grip_width": self.grip_width,
            "instruction": self.instruction,
            "parameters": self.parameters,
            "status": self.status,
        }
        if self.target_position:
            result["target_position"] = self.target_position.to_dict()
        if self.target_rotation:
            result["target_rotation"] = self.target_rotation.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "PrimitiveAction":
        tp = data.get("target_position")
        tr = data.get("target_rotation")
        return cls(
            primitive_id=data.get("primitive_id", 0),
            primitive_type=data.get("primitive_type", "move"),
            target_position=Position3D.from_dict(tp) if tp else None,
            target_rotation=Rotation3D.from_dict(tr) if tr else None,
            grip_width=data.get("grip_width"),
            instruction=data.get("instruction", ""),
            parent_action_id=data.get("parent_action_id", 0),
            parameters=data.get("parameters", {}),
            status=data.get("status", "pending"),
        )


@dataclass
class PrimitivePlan:
    """
    Primitive action sequence for a single high-level action.

    Generated by Planner. This is what gets sent to the Controller.
    """

    plan_id: str
    parent_action: HighLevelAction
    primitives: List[PrimitiveAction]
    status: str = "pending"
    current_primitive_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "parent_action": self.parent_action.to_dict(),
            "status": self.status,
            "current_primitive_index": self.current_primitive_index,
            "primitives": [p.to_dict() for p in self.primitives],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PrimitivePlan":
        return cls(
            plan_id=data.get("plan_id", ""),
            parent_action=HighLevelAction.from_dict(data.get("parent_action", {})),
            primitives=[
                PrimitiveAction.from_dict(p) for p in data.get("primitives", [])
            ],
            status=data.get("status", "pending"),
            current_primitive_index=data.get("current_primitive_index", 0),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Object Info Type (from Perception)
# =============================================================================


@dataclass
class ObjectInfo:
    """
    Object information from perception for planning.

    Contains position, bbox, and role info.
    """

    name: str
    position: Position3D
    bbox_3d: Optional[Dict[str, List[float]]] = None  # {"min": [x,y,z], "max": [x,y,z]}
    role: Optional[str] = None  # "target", "obstacle", "place_target", etc.
    size: Optional[Position3D] = None
    orientation: Optional[Dict[str, float]] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
        }
        if self.bbox_3d:
            result["bbox_3d"] = self.bbox_3d
        if self.role:
            result["role"] = self.role
        if self.size:
            result["size"] = self.size.to_dict()
        if self.orientation:
            result["orientation"] = self.orientation
        return result

    @classmethod
    def from_detection(cls, detection: Dict) -> "ObjectInfo":
        """Create ObjectInfo from perception Detection dict."""
        pose = detection.get("pose", {})
        position_data = pose.get("position", {"x": 0, "y": 0, "z": 0.5})
        metadata = detection.get("metadata", {})

        return cls(
            name=detection.get("name", "unknown"),
            position=Position3D(
                x=position_data.get("x", 0),
                y=position_data.get("y", 0),
                z=position_data.get("z", 0.5),
            ),
            bbox_3d=metadata.get("bbox_3d"),
            role=metadata.get("role"),
            orientation=pose.get("orientation"),
            confidence=detection.get("confidence", 1.0),
        )
