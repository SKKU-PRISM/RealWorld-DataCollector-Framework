"""
VLA model registry.

Register and retrieve VLA backends by name.
Backends are registered via the @register_vla decorator.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base_vla import BaseVLA

logger = logging.getLogger(__name__)

_VLA_REGISTRY: Dict[str, Type[BaseVLA]] = {}


def register_vla(name: str):
    """Decorator to register a VLA model backend.

    Usage:
        @register_vla("openvla")
        class OpenVLABackend(BaseVLA):
            ...
    """

    def wrapper(cls: Type[BaseVLA]) -> Type[BaseVLA]:
        key = name.lower()
        if key in _VLA_REGISTRY:
            logger.warning(f"Overwriting VLA backend: {key}")
        _VLA_REGISTRY[key] = cls
        return cls

    return wrapper


def get_vla_backend(name: str) -> Type[BaseVLA]:
    """Get VLA backend class by name.

    Args:
        name: Backend name (case-insensitive).

    Returns:
        BaseVLA subclass.

    Raises:
        ValueError: If backend not found.
    """
    key = name.lower()

    # Lazy-import backends to trigger registration
    if not _VLA_REGISTRY:
        _import_backends()

    if key not in _VLA_REGISTRY:
        # Try lazy import again for specific backend
        _import_backends()

    if key not in _VLA_REGISTRY:
        available = list(_VLA_REGISTRY.keys())
        raise ValueError(f"Unknown VLA backend '{name}'. Available: {available}")

    return _VLA_REGISTRY[key]


def list_vla_backends() -> List[str]:
    """List all registered VLA backend names."""
    if not _VLA_REGISTRY:
        _import_backends()
    return list(_VLA_REGISTRY.keys())


def _import_backends() -> None:
    """Import all backend modules to trigger @register_vla decorators."""
    import importlib

    backend_modules = [
        "robobridge.modules.controller.vla.openvla_backend",
        "robobridge.modules.controller.vla.lerobot_backend",
        "robobridge.modules.controller.vla.hf_vlm_backend",
    ]

    for module_name in backend_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.debug(f"Optional VLA backend not available: {module_name} ({e})")
