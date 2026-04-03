"""
Client data types for RoboBridge.
"""

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time_s: float = 0.0


@dataclass
class RobotState:
    """Robot state information."""

    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    ee_pose: Optional[Dict[str, Any]] = None
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    is_moving: bool = False
    timestamp: float = 0.0


@dataclass
class DetectedObject:
    """Detected object information."""

    name: str
    confidence: float
    position: Optional[Dict[str, float]] = None
    orientation: Optional[Dict[str, float]] = None
    bbox: Optional[List[float]] = None


@dataclass
class ModelInfo:
    """Model configuration information."""

    provider: str
    model: str
    device: Optional[str] = None
    api_base: Optional[str] = None


# =============================================================================
# Async Perception Types
# =============================================================================


@dataclass
class PerceptionState:
    """Snapshot of perception for divergence comparison."""

    objects: Dict[str, np.ndarray]  # object_name -> [x, y, z]
    detections: List[Any]  # Raw detection dicts for replanning
    timestamp: float = 0.0


class PerceptionBuffer:
    """
    Thread-safe single-result buffer (paper's B).

    Stores only the latest PerceptionState, not a full history.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[PerceptionState] = None

    def update(self, state: PerceptionState) -> None:
        """Thread-safe write of latest perception state."""
        with self._lock:
            self._state = state

    def get(self) -> Optional[PerceptionState]:
        """Thread-safe read of latest perception state."""
        with self._lock:
            return self._state

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._state = None


def scene_divergence(
    p_a: PerceptionState,
    p_b: PerceptionState,
    lambda_weight: float = 0.5,
) -> float:
    """
    Compute scene divergence between two perception states.

    Δ(Pa, Pb) = max(
        max_{i ∈ Oa ∩ Ob} ||pa_i - pb_i||_2,
        λ * |Oa △ Ob|
    )

    Args:
        p_a: Reference perception state (used for planning)
        p_b: Latest perception state
        lambda_weight: Weight for set symmetric difference term

    Returns:
        Divergence score (meters). 0 means identical scenes.
    """
    set_a = set(p_a.objects.keys())
    set_b = set(p_b.objects.keys())

    # Max positional divergence over shared objects
    shared = set_a & set_b
    max_pos_div = 0.0
    for name in shared:
        dist = float(np.linalg.norm(p_a.objects[name] - p_b.objects[name]))
        if dist > max_pos_div:
            max_pos_div = dist

    # Symmetric difference penalty (objects appeared/disappeared)
    sym_diff = len(set_a ^ set_b)
    set_penalty = lambda_weight * sym_diff

    return max(max_pos_div, set_penalty)
