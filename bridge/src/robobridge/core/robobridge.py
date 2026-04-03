"""
RoboBridge Core - Robot AI Framework Orchestrator

Main orchestrator class that manages adapters and ROS2 runtime.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .adapters import AdapterManager, InProcLink

if TYPE_CHECKING:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from robobridge.config import LoadedConfig

logger = logging.getLogger(__name__)


class RoboBridgeError(Exception):
    """Base exception for RoboBridge errors."""

    pass


class RoboBridge:
    """
    RoboBridge Core - Main Orchestrator

    Provides:
    - ROS2 runtime management
    - Built-in adapters for module communication
    - Configuration-based setup
    - Tracing and logging

    Usage:
        bridge = RoboBridge(
            ros_domain_id=0,
            ros_namespace="/robobridge",
            adapters_config_path="./config.py",
            trace=True,
            log_dir="./logs/robobridge"
        )
        bridge.start()
        # ... run your application ...
        bridge.shutdown()
    """

    def __init__(
        self,
        ros_domain_id: int = 0,
        ros_namespace: str = "/robobridge",
        adapters_config_path: str = "./config.py",
        trace: bool = False,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize RoboBridge.

        Args:
            ros_domain_id: ROS2 DDS domain ID for discovery
            ros_namespace: ROS2 topic namespace prefix
            adapters_config_path: Path to config.py file
            trace: Enable tracing metadata in messages
            log_dir: Directory for logs (optional)
        """
        self.ros_domain_id = ros_domain_id
        self.ros_namespace = ros_namespace
        self.adapters_config_path = adapters_config_path
        self.trace_enabled = trace
        self.log_dir = log_dir

        self._config: Optional[LoadedConfig] = None
        self._node: Optional[Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._adapter_manager: Optional[AdapterManager] = None
        self._executor_thread: Optional[threading.Thread] = None
        self._initialized = False
        self._running = False
        self._shutdown_event = threading.Event()

        # Setup logging
        self._setup_logging()

        # Load configuration
        self._load_configuration()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = logging.INFO

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup file handler if log_dir specified
        handlers = [console_handler]
        if self.log_dir:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(log_path / f"robobridge_{timestamp}.log")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # Configure root logger for robobridge package
        robobridge_logger = logging.getLogger("robobridge")
        robobridge_logger.setLevel(log_level)
        for handler in handlers:
            robobridge_logger.addHandler(handler)

        logger.info("Logging initialized")

    def _load_configuration(self) -> None:
        """Load adapter configuration from config file."""
        try:
            from robobridge.config import load_config

            config_path = Path(self.adapters_config_path).resolve()
            self._config = load_config(str(config_path))
            logger.info(f"Configuration loaded from {config_path}")
            logger.info(f"Loaded {len(self._config.adapters)} adapter configurations")
        except Exception as e:
            raise RoboBridgeError(f"Failed to load configuration: {e}")

    def initialize(self) -> None:
        """
        Initialize ROS2 runtime and adapters.

        Call this before start() to set up all components.
        """
        if self._initialized:
            logger.warning("RoboBridge already initialized")
            return

        # Set ROS2 domain ID
        os.environ["ROS_DOMAIN_ID"] = str(self.ros_domain_id)

        # Initialize ROS2
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
            logger.info(f"ROS2 initialized with domain ID {self.ros_domain_id}")
        except Exception as e:
            raise RoboBridgeError(f"Failed to initialize ROS2: {e}")

        # Create node
        node_name = "robobridge_orchestrator"
        if self.ros_namespace:
            node_name = self.ros_namespace.strip("/").replace("/", "_") + "_orchestrator"

        try:
            import rclpy

            self._node = rclpy.create_node(node_name, namespace=self.ros_namespace)
            logger.info(f"Created ROS2 node: {node_name}")
        except Exception as e:
            raise RoboBridgeError(f"Failed to create ROS2 node: {e}")

        # Create executor
        from rclpy.executors import MultiThreadedExecutor

        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)

        # Initialize adapter manager
        if self._config is None:
            raise RoboBridgeError("Configuration not loaded")

        self._adapter_manager = AdapterManager(
            node=self._node,
            adapter_configs=self._config.adapters,
            qos_profiles=self._config.qos_profiles,
            socket_defaults=self._config.socket_defaults,
            trace_enabled=self.trace_enabled,
        )

        try:
            self._adapter_manager.initialize_all()
        except Exception as e:
            raise RoboBridgeError(f"Failed to initialize adapters: {e}")

        self._initialized = True
        logger.info("RoboBridge initialization complete")

    def start(self) -> None:
        """
        Start RoboBridge runtime.

        Starts ROS2 executor and all adapters.
        """
        if not self._initialized:
            self.initialize()

        if self._running:
            logger.warning("RoboBridge already running")
            return

        if self._adapter_manager is None:
            raise RoboBridgeError("Adapter manager not initialized")

        # Start adapters
        self._adapter_manager.start_all()

        # Start executor in background thread
        self._executor_thread = threading.Thread(
            target=self._run_executor,
            daemon=True,
            name="RoboBridge-Executor",
        )
        self._executor_thread.start()

        self._running = True
        logger.info("RoboBridge started")

    def _run_executor(self) -> None:
        """Run ROS2 executor loop."""
        try:
            while not self._shutdown_event.is_set():
                if self._executor:
                    self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.error(f"Executor error: {e}")

    def shutdown(self) -> None:
        """
        Shutdown RoboBridge cleanly.

        Stops adapters, executor, and ROS2 runtime.
        """
        if not self._running:
            logger.warning("RoboBridge not running")
            return

        logger.info("Shutting down RoboBridge...")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop adapters
        if self._adapter_manager:
            self._adapter_manager.stop_all()

        # Wait for executor thread
        if self._executor_thread:
            self._executor_thread.join(timeout=5.0)

        # Cleanup ROS2
        if self._executor:
            self._executor.shutdown()

        if self._node:
            self._node.destroy_node()

        try:
            import rclpy

            rclpy.shutdown()
        except Exception:
            pass

        self._running = False
        self._initialized = False
        logger.info("RoboBridge shutdown complete")

    def spin(self) -> None:
        """
        Block and spin until shutdown.

        Handles SIGINT/SIGTERM for clean shutdown.
        """
        if not self._running:
            self.start()

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("RoboBridge spinning... Press Ctrl+C to exit")

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.shutdown()

    def get_adapter_link(self, role: str) -> Optional[InProcLink]:
        """
        Get in-proc link for a module role.

        Use this for in-process module registration.

        Args:
            role: Adapter role name (e.g., "perception")

        Returns:
            InProcLink if available, None otherwise
        """
        if self._adapter_manager:
            return self._adapter_manager.get_in_proc_link(role)
        return None

    def publish(self, topic: str, data: Any) -> None:
        """
        Publish data to a ROS2 topic directly.

        Args:
            topic: Topic name
            data: Data to publish (will be wrapped in String if dict)
        """
        if not self._node or not self._config or not self._adapter_manager:
            raise RoboBridgeError("RoboBridge not initialized")

        # Find which adapter handles this topic
        for role, config in self._config.adapters.items():
            if topic in config.pub_topics:
                adapter = self._adapter_manager.get_adapter(role)
                if adapter and topic in adapter._publishers:
                    import json

                    from std_msgs.msg import String

                    msg = String()
                    if isinstance(data, dict):
                        msg.data = json.dumps(data)
                    else:
                        msg.data = str(data)
                    adapter._publishers[topic].publish(msg)
                    return

        logger.warning(f"No adapter found for topic: {topic}")

    @property
    def is_running(self) -> bool:
        """Check if RoboBridge is running."""
        return self._running

    @property
    def is_initialized(self) -> bool:
        """Check if RoboBridge is initialized."""
        return self._initialized

    @property
    def config(self) -> Optional[LoadedConfig]:
        """Get loaded configuration."""
        return self._config

    @property
    def node(self) -> Optional[Node]:
        """Get ROS2 node."""
        return self._node

    def __enter__(self) -> "RoboBridge":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
