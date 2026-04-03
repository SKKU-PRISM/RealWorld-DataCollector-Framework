"""
Configuration loader for RoboBridge.

Loads adapter configurations from a Python config file.
"""

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .defaults import DEFAULT_QOS_PROFILES, DEFAULT_SOCKET_DEFAULTS, DEFAULT_MODULE_CONFIGS

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


@dataclass
class AdapterConfig:
    """Validated adapter configuration."""

    role: str
    link_mode: str
    bind_host: Optional[str] = None
    bind_port: Optional[int] = None
    auth_token: Optional[str] = None
    pub_topics: List[str] = field(default_factory=list)
    sub_topics: List[str] = field(default_factory=list)
    topic_types: Dict[str, str] = field(default_factory=dict)
    qos: Dict[str, str] = field(default_factory=dict)


@dataclass
class LoadedConfig:
    """Complete loaded configuration."""

    adapters: Dict[str, AdapterConfig]
    qos_profiles: Dict[str, Dict[str, Any]]
    socket_defaults: Dict[str, Any]
    config_path: str
    module_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Required keys for each adapter
REQUIRED_ADAPTER_KEYS = {"link_mode", "pub_topics", "sub_topics", "topic_types"}
SOCKET_REQUIRED_KEYS = {"bind_host", "bind_port"}
VALID_LINK_MODES = {"socket", "in_proc", "direct"}


def load_config(config_path: str) -> LoadedConfig:
    """
    Load and validate configuration from a Python file.

    Args:
        config_path: Path to the config.py file

    Returns:
        LoadedConfig with validated configuration

    Raises:
        ConfigError: If configuration is invalid or cannot be loaded
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    if not path.suffix == ".py":
        raise ConfigError(f"Config file must be a .py file: {path}")

    # Load the module
    try:
        spec = importlib.util.spec_from_file_location("robobridge_config", path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Failed to create module spec for: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise ConfigError(f"Failed to load config module: {e}")

    # Extract configuration
    if not hasattr(module, "ADAPTERS"):
        raise ConfigError("Config must define ADAPTERS dictionary")

    raw_adapters = getattr(module, "ADAPTERS")
    if not isinstance(raw_adapters, dict):
        raise ConfigError("ADAPTERS must be a dictionary")

    # Load optional configurations with defaults
    qos_profiles = getattr(module, "QOS_PROFILES", DEFAULT_QOS_PROFILES)
    socket_defaults = getattr(module, "SOCKET_DEFAULTS", DEFAULT_SOCKET_DEFAULTS)
    module_configs = getattr(module, "MODULE_CONFIGS", DEFAULT_MODULE_CONFIGS)

    # Validate and convert adapters
    adapters = {}
    for role, config in raw_adapters.items():
        adapters[role] = _validate_adapter(role, config, qos_profiles)

    logger.info(f"Loaded configuration from {path} with {len(adapters)} adapters")

    return LoadedConfig(
        adapters=adapters,
        qos_profiles=qos_profiles,
        socket_defaults={**DEFAULT_SOCKET_DEFAULTS, **socket_defaults},
        config_path=str(path),
        module_configs={**DEFAULT_MODULE_CONFIGS, **module_configs},
    )


def _validate_adapter(
    role: str,
    config: Dict[str, Any],
    qos_profiles: Dict[str, Dict],
) -> AdapterConfig:
    """Validate and create AdapterConfig from raw config dict."""

    # Check required keys
    missing = REQUIRED_ADAPTER_KEYS - set(config.keys())
    if missing:
        raise ConfigError(f"Adapter '{role}' missing required keys: {missing}")

    # Validate link_mode
    link_mode = config["link_mode"]
    if link_mode not in VALID_LINK_MODES:
        raise ConfigError(
            f"Adapter '{role}' has invalid link_mode '{link_mode}'. Valid modes: {VALID_LINK_MODES}"
        )

    # Socket mode requires host/port
    if link_mode == "socket":
        socket_missing = SOCKET_REQUIRED_KEYS - set(config.keys())
        if socket_missing:
            raise ConfigError(f"Adapter '{role}' with link_mode='socket' missing: {socket_missing}")

    # Validate topic_types covers all topics
    all_topics = set(config["pub_topics"]) | set(config["sub_topics"])
    typed_topics = set(config["topic_types"].keys())
    untyped = all_topics - typed_topics
    if untyped:
        raise ConfigError(f"Adapter '{role}' has topics without types: {untyped}")

    # Validate QoS references (if provided)
    if "qos" in config:
        for topic, profile_name in config["qos"].items():
            if profile_name not in qos_profiles:
                logger.warning(
                    f"Adapter '{role}' topic '{topic}' references unknown "
                    f"QoS profile '{profile_name}'"
                )

    # Validate topic_types format
    for topic, msg_type in config["topic_types"].items():
        if not _is_valid_msg_type_format(msg_type):
            logger.warning(
                f"Adapter '{role}' topic '{topic}' has unusual message type format: '{msg_type}'"
            )

    return AdapterConfig(
        role=role,
        link_mode=link_mode,
        bind_host=config.get("bind_host"),
        bind_port=config.get("bind_port"),
        auth_token=config.get("auth_token"),
        pub_topics=list(config["pub_topics"]),
        sub_topics=list(config["sub_topics"]),
        topic_types=dict(config["topic_types"]),
        qos=dict(config.get("qos", {})),
    )


def _is_valid_msg_type_format(msg_type: str) -> bool:
    """Check if message type follows ROS2 naming convention."""
    # Format: package/MessageType (e.g., std_msgs/String, sensor_msgs/Image)
    if "/" not in msg_type:
        return False
    parts = msg_type.split("/")
    if len(parts) != 2:
        return False
    return len(parts[0]) > 0 and len(parts[1]) > 0


def _get_valid_message_types() -> Set[str]:
    """Get set of known valid ROS2 message types."""
    # Common message types - can be extended
    return {
        "std_msgs/String",
        "std_msgs/Bool",
        "std_msgs/Int32",
        "std_msgs/Float64",
        "sensor_msgs/Image",
        "sensor_msgs/PointCloud2",
        "sensor_msgs/JointState",
        "geometry_msgs/Pose",
        "geometry_msgs/PoseStamped",
        "geometry_msgs/Twist",
        "nav_msgs/Odometry",
    }


def get_message_class(msg_type: str):
    """
    Get ROS2 message class from type string.

    Args:
        msg_type: Message type string (e.g., "sensor_msgs/Image")

    Returns:
        ROS2 message class

    Raises:
        ConfigError: If message type cannot be resolved
    """
    try:
        parts = msg_type.split("/")
        if len(parts) != 2:
            raise ConfigError(f"Invalid message type format: {msg_type}")

        package, class_name = parts

        # Handle msg submodule convention
        module_name = f"{package}.msg"
        module = importlib.import_module(module_name)

        if not hasattr(module, class_name):
            raise ConfigError(f"Message class '{class_name}' not found in {module_name}")

        return getattr(module, class_name)

    except ImportError as e:
        raise ConfigError(f"Failed to import message type '{msg_type}': {e}")
