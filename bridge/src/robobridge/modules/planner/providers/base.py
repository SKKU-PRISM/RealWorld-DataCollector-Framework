"""
Base class for planner providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import Plan


class BasePlannerProvider(ABC):
    """Abstract base class for planner providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra_config = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load/initialize the model."""
        pass

    @abstractmethod
    def plan(
        self,
        instruction: str,
        world_state: Optional[Dict] = None,
        skill_list: Optional[List[str]] = None,
    ) -> Plan:
        """
        Generate a plan from instruction.

        Args:
            instruction: Natural language instruction
            world_state: Current world state
            skill_list: Available skills

        Returns:
            Plan
        """
        pass

    def unload_model(self) -> None:
        """Unload the model (optional)."""
        pass
