"""
Abstract base class for VLA model backends.

All VLA backends (OpenVLA, LeRobot, HF VLM) implement this interface.
Supports model loading, LoRA adapter management, and inference.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import VLAInput, VLAModelConfig, VLAOutput

logger = logging.getLogger(__name__)


class BaseVLA(ABC):
    """
    Abstract base for VLA model backends.

    Lifecycle:
        1. __init__(config) - store config
        2. load_model() - load base model weights
        3. load_lora_adapter(path, name) - load LoRA adapter(s)
        4. set_active_adapter(name) - switch active adapter (< 10ms)
        5. predict(input) - run inference with active adapter
    """

    def __init__(self, config: VLAModelConfig):
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._loaded_adapters: Dict[str, str] = {}  # name -> path
        self._active_adapter: Optional[str] = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the base VLA model (without LoRA adapters).

        Should respect config.quantize_4bit and config.device.
        """

    @abstractmethod
    def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a LoRA adapter into the model.

        Multiple adapters can be loaded simultaneously.
        """

    @abstractmethod
    def set_active_adapter(self, adapter_name: str) -> None:
        """Switch to a specific LoRA adapter.

        Must be fast (< 10ms) for pre-loaded adapters.
        """

    @abstractmethod
    def predict(self, vla_input: VLAInput) -> VLAOutput:
        """Run inference with the currently active adapter.

        Args:
            vla_input: Images, robot state, instruction, primitive type.

        Returns:
            VLAOutput with predicted action (action_dim,).
        """

    def reset_policy(self) -> None:
        """Reset policy internal state (e.g., action queues). Override if needed."""
        pass

    def get_loaded_adapters(self) -> List[str]:
        """Return names of currently loaded adapters."""
        return list(self._loaded_adapters.keys())

    def get_active_adapter(self) -> Optional[str]:
        """Return name of currently active adapter."""
        return self._active_adapter

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._processor = None
        self._loaded_adapters.clear()
        self._active_adapter = None
        logger.info("VLA model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if base model is loaded."""
        return self._model is not None
