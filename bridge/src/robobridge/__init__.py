"""
RoboBridge - General Robot Manipulation Framework

A modular, LangChain-based robot manipulation framework that enables
natural language task planning and execution.

Quick Start:
    from robobridge import RoboBridge

    robot = RoboBridge.initialize()
    robot.execute("Pick up the red cup")
    robot.shutdown()

Custom Model Integration:
    from robobridge import CustomPerception, CustomPlanner

    class MyDetector(CustomPerception):
        def load_model(self):
            self._model = load_my_model()

        def detect(self, rgb, depth=None, object_list=None):
            return [Detection(name="cup", confidence=0.95, ...)]
"""

__version__ = "0.1.0"

# Core components
from .core import (
    RoboBridge as RoboBridgeCore,
    RoboBridgeError,
    Adapter,
    AdapterManager,
    InProcLink,
    SocketLink,
    ModuleLink,
    TraceContext,
    build_qos_profile,
)

# Modules
from .modules import (
    BaseModule,
    ModuleConfig,
    Perception,
    Planner,
    Controller,
    Robot,
    Monitor,
)

# Custom wrappers for user models
from .wrappers import (
    CustomPerception,
    CustomPlanner,
    CustomController,
    CustomRobot,
    CustomMonitor,
)

# Configuration
from .config import (
    load_config,
    LoadedConfig,
    AdapterConfig,
    ConfigError,
    DEFAULT_QOS_PROFILES,
    DEFAULT_SOCKET_DEFAULTS,
    DEFAULT_MODULE_CONFIGS,
)

# Utilities
from .utils import (
    load_custom_class,
    parse_custom_model_path,
)


# Lazy import for client to avoid circular imports
def __getattr__(name: str):
    if name == "RoboBridge":
        from .client import RoboBridge

        return RoboBridge
    if name == "RoboBridgeClient":
        from .client import RoboBridgeClient

        return RoboBridgeClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "RoboBridge",
    "RoboBridgeClient",
    "RoboBridgeCore",
    "RoboBridgeError",
    "Adapter",
    "AdapterManager",
    "InProcLink",
    "SocketLink",
    "ModuleLink",
    "TraceContext",
    "build_qos_profile",
    "BaseModule",
    "ModuleConfig",
    "Perception",
    "Planner",
    "Controller",
    "Robot",
    "Monitor",
    "CustomPerception",
    "CustomPlanner",
    "CustomController",
    "CustomRobot",
    "CustomMonitor",
    "load_config",
    "LoadedConfig",
    "AdapterConfig",
    "ConfigError",
    "DEFAULT_QOS_PROFILES",
    "DEFAULT_SOCKET_DEFAULTS",
    "DEFAULT_MODULE_CONFIGS",
    "load_custom_class",
    "parse_custom_model_path",
]
