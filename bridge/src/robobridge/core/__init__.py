"""
RoboBridge Core

Core components for RoboBridge orchestration and communication.
"""

from .robobridge import RoboBridge, RoboBridgeError
from .adapters import (
    Adapter,
    AdapterManager,
    InProcLink,
    SocketLink,
    ModuleLink,
    TraceContext,
    build_qos_profile,
)

__all__ = [
    # Core
    "RoboBridge",
    "RoboBridgeError",
    # Adapters
    "Adapter",
    "AdapterManager",
    "InProcLink",
    "SocketLink",
    "ModuleLink",
    "TraceContext",
    "build_qos_profile",
]
