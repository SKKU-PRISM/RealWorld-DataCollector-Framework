"""
Base monitor provider class.

Abstract interface for VLM-based monitoring providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from robobridge.modules.monitor.types import FeedbackResult


class MonitorProvider(ABC):
    """
    Abstract base class for monitor providers.

    Implement this to create custom VLM-based monitoring providers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider.

        Args:
            config: Provider configuration dictionary
        """
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider (load models, connect to APIs, etc.)."""
        pass

    @abstractmethod
    def observe(
        self,
        rgb: Any,
        plan: Optional[Dict[str, Any]] = None,
        current_step: Optional[Dict[str, Any]] = None,
    ) -> FeedbackResult:
        """
        Perform observation and return result.

        Args:
            rgb: RGB image observation
            plan: Current high-level plan
            current_step: Current step being executed

        Returns:
            FeedbackResult with success and confidence
        """
        pass

    def shutdown(self) -> None:
        """Cleanup provider resources."""
        pass
