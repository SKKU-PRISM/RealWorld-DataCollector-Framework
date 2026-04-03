"""
LoRA adapter manager for VLA models.

Handles loading, caching, and fast swapping of LoRA adapters
based on primitive type (move/grip) and optional task name.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .base_vla import BaseVLA
from .types import LoRAAdapterConfig, VLAInput, VLAOutput

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manages LoRA adapter loading and fast swapping.

    Loads base model once, pre-loads all adapters into memory.
    Dispatches to correct adapter based on (primitive_type, task_name).

    Adapter resolution order:
        1. Task-specific adapter: "{task_name}_{primitive_type}" (e.g. "PnPCounterToCab_move")
        2. General adapter: "{primitive_type}" (e.g. "move")
        3. Error if neither found.
    """

    def __init__(
        self,
        vla_backend: BaseVLA,
        adapter_configs: Dict[str, LoRAAdapterConfig],
    ):
        """
        Args:
            vla_backend: Initialized BaseVLA backend instance.
            adapter_configs: Map of adapter_name -> LoRAAdapterConfig.
        """
        self._backend = vla_backend
        self._adapter_configs = adapter_configs
        self._current_adapter: Optional[str] = None

        # Build lookup: (task_name, primitive_type) -> adapter_name
        self._lookup: Dict[tuple, str] = {}
        for name, cfg in adapter_configs.items():
            for ptype in cfg.primitive_types:
                key = (cfg.task_name, ptype)  # task_name=None for general
                self._lookup[key] = name

    def initialize(self) -> None:
        """Load base model and all LoRA adapters."""
        if not self._backend.is_loaded:
            logger.info("Loading base VLA model...")
            self._backend.load_model()

        for name, cfg in self._adapter_configs.items():
            logger.info(f"Loading LoRA adapter: {name} from {cfg.path}")
            self._backend.load_lora_adapter(cfg.path, name)

        logger.info(
            f"LoRA manager initialized with {len(self._adapter_configs)} adapters: "
            f"{list(self._adapter_configs.keys())}"
        )

    def resolve_adapter(
        self,
        primitive_type: str,
        task_name: Optional[str] = None,
    ) -> str:
        """Resolve which adapter to use.

        Resolution order:
            1. (task_name, primitive_type) - task-specific
            2. (None, primitive_type) - general fallback

        Raises:
            ValueError: If no adapter found.
        """
        # Try task-specific first
        if task_name:
            key = (task_name, primitive_type)
            if key in self._lookup:
                return self._lookup[key]

        # Fallback to general
        key = (None, primitive_type)
        if key in self._lookup:
            return self._lookup[key]

        # Fallback: if grip adapter missing, use move adapter
        if primitive_type == "grip":
            fallback_key = (None, "move")
            if task_name:
                task_fallback = (task_name, "move")
                if task_fallback in self._lookup:
                    logger.warning(f"No grip adapter found, falling back to task move adapter")
                    return self._lookup[task_fallback]
            if fallback_key in self._lookup:
                logger.warning(f"No grip adapter found, falling back to move adapter")
                return self._lookup[fallback_key]

        available = list(self._adapter_configs.keys())
        raise ValueError(
            f"No adapter for primitive_type='{primitive_type}', "
            f"task_name='{task_name}'. Available: {available}"
        )

    def predict(
        self,
        vla_input: VLAInput,
        task_name: Optional[str] = None,
    ) -> VLAOutput:
        """Select adapter and run VLA inference.

        Args:
            vla_input: Model input (images, state, instruction, primitive_type).
            task_name: Optional task name for task-specific adapter resolution.

        Returns:
            VLAOutput with predicted action.
        """
        # Select adapter based on primitive type (move or grip)
        ptype = vla_input.primitive_type or "move"
        adapter_name = self.resolve_adapter(ptype, task_name)

        if self._current_adapter != adapter_name:
            t0 = time.perf_counter()
            self._backend.set_active_adapter(adapter_name)
            swap_ms = (time.perf_counter() - t0) * 1000
            logger.debug(f"Adapter swap '{self._current_adapter}' -> '{adapter_name}': {swap_ms:.1f}ms")
            self._current_adapter = adapter_name

        return self._backend.predict(vla_input)

    def reset_policy(self) -> None:
        """Reset policy state (clear action queues). Call at episode start."""
        if hasattr(self._backend, "reset_policy"):
            self._backend.reset_policy()

    def get_loaded_adapters(self) -> List[str]:
        """Return names of loaded adapters."""
        return self._backend.get_loaded_adapters()

    def get_current_adapter(self) -> Optional[str]:
        """Return name of currently active adapter."""
        return self._current_adapter

    def unload(self) -> None:
        """Release all resources."""
        self._backend.unload()
        self._current_adapter = None
        logger.info("LoRA manager unloaded")
