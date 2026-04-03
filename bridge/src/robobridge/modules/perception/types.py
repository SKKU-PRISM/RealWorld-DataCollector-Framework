"""
Perception module types and data structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    """Single object detection result."""

    name: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized
    mask: Optional[Any] = None
    pose: Optional[Dict[str, Any]] = None  # position + orientation quaternion
    frame_id: str = ""
    metadata: Optional[Dict[str, Any]] = None  # Additional info (e.g., fixture type)

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "name": self.name,
            "confidence": self.confidence,
            "frame_id": self.frame_id,
        }
        if self.bbox:
            result["bbox"] = self.bbox
        if self.pose:
            result["pose"] = self.pose
        if self.mask is not None:
            result["has_mask"] = True
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(
            name=data.get("name", "unknown"),
            confidence=data.get("confidence", 0.0),
            bbox=data.get("bbox"),
            pose=data.get("pose"),
            frame_id=data.get("frame_id", ""),
            metadata=data.get("metadata"),
        )


@dataclass
class PerceptionConfig:
    """Perception module specific configuration."""

    image_size: int = 640
    conf_threshold: float = 0.25
    nms_threshold: float = 0.50
    max_dets: int = 50
    pose_format: str = "pose_quat"  # bbox, mask, pose_quat
    frame_id: str = "camera_color_optical_frame"
    # Topics
    rgb_topic: str = "/camera/rgb"
    depth_topic: Optional[str] = "/camera/depth"
    object_list_topic: Optional[str] = "/perception/object_list"
    output_topic: str = "/perception/objects"


@dataclass
class PerceptionResult:
    """Complete perception result for a frame."""

    timestamp: float
    frame_id: str
    detections: List[Detection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
            "metadata": self.metadata,
        }
