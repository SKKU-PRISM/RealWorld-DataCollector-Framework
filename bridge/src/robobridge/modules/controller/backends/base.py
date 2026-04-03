"""
Base controller backend class.

Abstract interface for motion planning backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from robobridge.modules.controller.types import Command, TrajectoryPoint


class ControllerBackend(ABC):
    """
    Abstract base class for controller backends.

    Implement this to create custom motion planning backends.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backend.

        Args:
            config: Backend configuration dictionary
        """
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load models, connect to services, etc.)."""
        pass

    @abstractmethod
    def plan(
        self,
        step: Dict[str, Any],
        rgb: Optional[Any] = None,
        depth: Optional[Any] = None,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Command]:
        """
        Generate a command for the given step.

        Args:
            step: High-level plan step
            rgb: RGB image observation
            depth: Depth image observation
            robot_state: Current robot state

        Returns:
            Command or None if planning failed
        """
        pass

    def check_safety(self, command: Command) -> bool:
        """
        Check if command satisfies safety constraints.

        Args:
            command: Command to check

        Returns:
            True if safe, False otherwise
        """
        return True

    def shutdown(self) -> None:
        """Cleanup backend resources."""
        pass
