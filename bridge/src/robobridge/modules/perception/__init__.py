"""
Perception Module

Detects objects in images and estimates poses.
Supports various providers: HuggingFace, OpenAI, custom models, etc.
"""

from .perception import Perception
from .types import Detection, PerceptionConfig, PerceptionResult

__all__ = [
    "Perception",
    "Detection",
    "PerceptionConfig",
    "PerceptionResult",
]
