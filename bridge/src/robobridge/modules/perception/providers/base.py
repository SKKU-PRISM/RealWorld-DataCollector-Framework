"""
Base class for perception providers.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..types import Detection


class BasePerceptionProvider(ABC):
    """Abstract base class for perception providers."""

    def __init__(
        self,
        model: str,
        device: str = "cuda:0",
        **kwargs,
    ):
        self.model = model
        self.device = device
        self._extra_config = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def detect(
        self,
        rgb: Any,
        depth: Optional[Any] = None,
        object_list: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Run object detection.

        Args:
            rgb: RGB image
            depth: Optional depth image
            object_list: Optional list of target objects

        Returns:
            List of Detection
        """
        pass

    def unload_model(self) -> None:
        """Unload the model (optional)."""
        pass
