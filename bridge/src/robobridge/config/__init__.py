"""
RoboBridge Configuration

Configuration loading and management.
"""

from .loader import (
    AdapterConfig,
    ConfigError,
    LoadedConfig,
    load_config,
    get_message_class,
)
from .defaults import (
    DEFAULT_QOS_PROFILES,
    DEFAULT_SOCKET_DEFAULTS,
    DEFAULT_MODULE_CONFIGS,
)

__all__ = [
    # Loader
    "AdapterConfig",
    "ConfigError",
    "LoadedConfig",
    "load_config",
    "get_message_class",
    # Defaults
    "DEFAULT_QOS_PROFILES",
    "DEFAULT_SOCKET_DEFAULTS",
    "DEFAULT_MODULE_CONFIGS",
]
