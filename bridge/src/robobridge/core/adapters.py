"""
RoboBridge Adapters Runtime

Provides ROS2 pub/sub functionality with socket/in-proc module bridging.
Adapters are built into RoboBridge and handle communication between:
- ROS2 topics (inter-adapter communication)
- Modules via socket or in-proc (module-adapter communication)
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import select
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .protocols import LenJsonProtocol, ProtocolError

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.qos import QoSProfile

    from robobridge.config import AdapterConfig

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Tracing metadata for debugging."""

    request_id: Optional[str] = None
    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "request_id": self.request_id,
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "timestamp": self.timestamp or time.time(),
            }.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TraceContext":
        return cls(
            request_id=d.get("request_id"),
            episode_id=d.get("episode_id"),
            step_id=d.get("step_id"),
            timestamp=d.get("timestamp"),
        )


def build_qos_profile(profile_config: Dict[str, Any]) -> "QoSProfile":
    """Build ROS2 QoSProfile from configuration dictionary."""
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    reliability_map = {
        "reliable": QoSReliabilityPolicy.RELIABLE,
        "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
    }
    durability_map = {
        "volatile": QoSDurabilityPolicy.VOLATILE,
        "transient_local": QoSDurabilityPolicy.TRANSIENT_LOCAL,
    }

    reliability = reliability_map.get(
        profile_config.get("reliability", "reliable"),
        QoSReliabilityPolicy.RELIABLE,
    )
    durability = durability_map.get(
        profile_config.get("durability", "volatile"),
        QoSDurabilityPolicy.VOLATILE,
    )
    depth = profile_config.get("depth", 10)

    return QoSProfile(
        reliability=reliability,
        durability=durability,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class ModuleLink(ABC):
    """Abstract base class for module communication links."""

    @abstractmethod
    def start(self) -> None:
        """Start the link."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the link."""
        pass

    @abstractmethod
    def send_to_module(
        self, topic: str, payload: Any, trace: Optional[TraceContext] = None
    ) -> None:
        """Send message to connected module."""
        pass

    @abstractmethod
    def set_publish_callback(
        self, callback: Callable[[str, Any, Optional[TraceContext]], None]
    ) -> None:
        """Set callback for when module wants to publish."""
        pass


class InProcLink(ModuleLink):
    """In-process module link using queues."""

    def __init__(self, role: str):
        self.role = role
        self._to_module_queue: queue.Queue = queue.Queue()
        self._from_module_queue: queue.Queue = queue.Queue()
        self._publish_callback: Optional[Callable] = None
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_from_module,
            daemon=True,
            name=f"InProcLink-{self.role}",
        )
        self._worker_thread.start()
        logger.info(f"InProcLink started for {self.role}")

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        logger.info(f"InProcLink stopped for {self.role}")

    def send_to_module(
        self, topic: str, payload: Any, trace: Optional[TraceContext] = None
    ) -> None:
        msg: dict[str, Any] = {"type": "sub", "topic": topic, "payload": payload}
        if trace:
            msg["trace"] = trace.to_dict()
        self._to_module_queue.put(msg)

    def set_publish_callback(self, callback: Callable) -> None:
        self._publish_callback = callback

    def _process_from_module(self) -> None:
        while self._running:
            try:
                msg = self._from_module_queue.get(timeout=0.1)
                if self._publish_callback and msg.get("type") == "pub":
                    trace = None
                    if "trace" in msg:
                        trace = TraceContext.from_dict(msg["trace"])
                    self._publish_callback(msg["topic"], msg["payload"], trace)
            except queue.Empty:
                continue

    # Methods for module to use
    def get_message(self, timeout: float = 1.0) -> Optional[dict]:
        """Get message from adapter (for module to call)."""
        try:
            return self._to_module_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish(self, topic: str, payload: Any, trace: Optional[dict] = None) -> None:
        """Publish message through adapter (for module to call)."""
        msg: dict[str, Any] = {"type": "pub", "topic": topic, "payload": payload}
        if trace:
            msg["trace"] = trace
        self._from_module_queue.put(msg)


class SocketLink(ModuleLink):
    """Socket-based module link."""

    def __init__(
        self,
        role: str,
        bind_host: str,
        bind_port: int,
        auth_token: Optional[str] = None,
        max_connections: int = 8,
        recv_timeout_s: float = 3.0,
        send_timeout_s: float = 3.0,
        max_payload_bytes: int = 10485760,
    ):
        self.role = role
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.auth_token = auth_token
        self.max_connections = max_connections

        self.protocol = LenJsonProtocol(
            recv_timeout_s=recv_timeout_s,
            send_timeout_s=send_timeout_s,
            max_payload_bytes=max_payload_bytes,
            auth_token=auth_token,
        )

        self._server_socket: Optional[socket.socket] = None
        self._clients: Dict[int, socket.socket] = {}  # fd -> socket
        self._clients_lock = threading.Lock()
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._publish_callback: Optional[Callable] = None

    def start(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.bind_host, self.bind_port))
        self._server_socket.listen(self.max_connections)
        self._server_socket.setblocking(False)

        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name=f"SocketLink-Accept-{self.role}",
        )
        self._accept_thread.start()

        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"SocketLink-Recv-{self.role}",
        )
        self._recv_thread.start()

        logger.info(f"SocketLink started for {self.role} on {self.bind_host}:{self.bind_port}")

    def stop(self) -> None:
        self._running = False

        # Close all client sockets
        with self._clients_lock:
            for sock in self._clients.values():
                try:
                    sock.close()
                except Exception:
                    pass
            self._clients.clear()

        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        # Wait for threads
        if self._accept_thread:
            self._accept_thread.join(timeout=2.0)
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

        logger.info(f"SocketLink stopped for {self.role}")

    def send_to_module(
        self, topic: str, payload: Any, trace: Optional[TraceContext] = None
    ) -> None:
        msg = self.protocol.create_sub_message(
            topic,
            payload,
            trace.to_dict() if trace else None,
        )

        with self._clients_lock:
            dead_fds = []
            for fd, sock in self._clients.items():
                try:
                    self.protocol.send(sock, msg)
                except ProtocolError as e:
                    logger.warning(f"Failed to send to client {fd}: {e}")
                    dead_fds.append(fd)

            for fd in dead_fds:
                self._remove_client(fd)

    def set_publish_callback(self, callback: Callable) -> None:
        self._publish_callback = callback

    def _accept_loop(self) -> None:
        while self._running:
            try:
                if not self._server_socket:
                    time.sleep(0.1)
                    continue

                readable, _, _ = select.select([self._server_socket], [], [], 0.5)
                if not readable:
                    continue

                client_sock, addr = self._server_socket.accept()
                client_sock.setblocking(True)
                logger.info(f"New connection from {addr} for {self.role}")

                # Authenticate if required
                if self.auth_token:
                    if not self.protocol.authenticate_client(client_sock):
                        logger.warning(f"Authentication failed for {addr}")
                        client_sock.close()
                        continue

                with self._clients_lock:
                    if len(self._clients) >= self.max_connections:
                        logger.warning(f"Max connections reached for {self.role}")
                        client_sock.close()
                        continue
                    self._clients[client_sock.fileno()] = client_sock

            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")

    def _recv_loop(self) -> None:
        while self._running:
            with self._clients_lock:
                if not self._clients:
                    time.sleep(0.1)
                    continue
                sockets = list(self._clients.values())

            try:
                readable, _, _ = select.select(sockets, [], [], 0.5)
            except (ValueError, OSError):
                # Socket was closed
                continue

            for sock in readable:
                try:
                    msg = self.protocol.recv(sock)
                    self._handle_message(msg)
                except ProtocolError as e:
                    logger.warning(f"Receive error: {e}")
                    with self._clients_lock:
                        self._remove_client(sock.fileno())
                except OSError:
                    # Socket was closed during shutdown
                    if not self._running:
                        break
                    with self._clients_lock:
                        self._remove_client(sock.fileno())

    def _handle_message(self, msg: dict) -> None:
        if msg.get("type") == "pub" and self._publish_callback:
            trace = None
            if "trace" in msg:
                trace = TraceContext.from_dict(msg["trace"])
            self._publish_callback(msg.get("topic"), msg.get("payload"), trace)

    def _remove_client(self, fd: int) -> None:
        if fd in self._clients:
            try:
                self._clients[fd].close()
            except Exception:
                pass
            del self._clients[fd]
            logger.info(f"Removed client {fd} from {self.role}")


class Adapter:
    """
    Single adapter runtime for a module role.

    Handles:
    - ROS2 publishers for pub_topics
    - ROS2 subscribers for sub_topics
    - Module link (socket or in-proc)
    """

    def __init__(
        self,
        node: "Node",
        config: "AdapterConfig",
        qos_profiles: Dict[str, Dict[str, Any]],
        socket_defaults: Dict[str, Any],
        trace_enabled: bool = False,
    ):
        self.node = node
        self.config = config
        self.role = config.role
        self.qos_profiles = qos_profiles
        self.socket_defaults = socket_defaults
        self.trace_enabled = trace_enabled

        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}
        self._link: Optional[ModuleLink] = None
        self._msg_classes: Dict[str, type] = {}

    def initialize(self) -> None:
        """Initialize ROS2 pubs/subs and module link."""
        self._resolve_message_classes()
        self._create_publishers()
        self._create_subscribers()
        self._create_module_link()
        logger.info(f"Adapter '{self.role}' initialized")

    def start(self) -> None:
        """Start the module link."""
        if self._link:
            self._link.start()
            self._link.set_publish_callback(self._on_module_publish)

    def stop(self) -> None:
        """Stop the module link."""
        if self._link:
            self._link.stop()

    def _resolve_message_classes(self) -> None:
        """Resolve ROS2 message classes for all topics."""
        from robobridge.config import get_message_class

        for topic, msg_type in self.config.topic_types.items():
            try:
                self._msg_classes[topic] = get_message_class(msg_type)
            except Exception as e:
                logger.error(f"Failed to resolve message class for {topic}: {e}")
                raise

    def _get_qos_for_topic(self, topic: str) -> "QoSProfile":
        """Get QoS profile for a topic."""
        profile_name = self.config.qos.get(topic)
        if profile_name and profile_name in self.qos_profiles:
            return build_qos_profile(self.qos_profiles[profile_name])
        return build_qos_profile({"reliability": "reliable", "durability": "volatile", "depth": 10})

    def _create_publishers(self) -> None:
        """Create ROS2 publishers for pub_topics."""
        for topic in self.config.pub_topics:
            msg_class = self._msg_classes[topic]
            qos = self._get_qos_for_topic(topic)
            self._publishers[topic] = self.node.create_publisher(msg_class, topic, qos)
            logger.debug(f"Created publisher for {topic}")

    def _create_subscribers(self) -> None:
        """Create ROS2 subscribers for sub_topics."""
        for topic in self.config.sub_topics:
            msg_class = self._msg_classes[topic]
            qos = self._get_qos_for_topic(topic)

            # Create callback closure
            def make_callback(t):
                return lambda msg: self._on_ros_message(t, msg)

            self._subscribers[topic] = self.node.create_subscription(
                msg_class,
                topic,
                make_callback(topic),
                qos,
            )
            logger.debug(f"Created subscriber for {topic}")

    def _create_module_link(self) -> None:
        """Create module communication link based on config."""
        link_mode = self.config.link_mode

        if link_mode == "socket":
            if self.config.bind_host is None or self.config.bind_port is None:
                raise ValueError(
                    f"Adapter '{self.role}' with link_mode='socket' requires bind_host and bind_port"
                )
            self._link = SocketLink(
                role=self.role,
                bind_host=self.config.bind_host,
                bind_port=self.config.bind_port,
                auth_token=self.config.auth_token,
                max_connections=self.socket_defaults.get("max_connections", 8),
                recv_timeout_s=self.socket_defaults.get("recv_timeout_s", 3.0),
                send_timeout_s=self.socket_defaults.get("send_timeout_s", 3.0),
                max_payload_bytes=self.socket_defaults.get("max_payload_bytes", 10485760),
            )
        elif link_mode == "in_proc":
            self._link = InProcLink(role=self.role)
        elif link_mode == "direct":
            # Direct mode: no adapter link, module process() called directly
            self._link = None
        else:
            raise ValueError(f"Unknown link_mode: {link_mode}")

    def _on_ros_message(self, topic: str, msg: Any) -> None:
        """Handle incoming ROS2 message, forward to module."""
        try:
            payload = self._serialize_ros_message(topic, msg)
            trace = TraceContext(timestamp=time.time()) if self.trace_enabled else None
            if self._link:
                self._link.send_to_module(topic, payload, trace)
        except Exception as e:
            logger.error(f"Error handling ROS message on {topic}: {e}")

    def _on_module_publish(self, topic: str, payload: Any, trace: Optional[TraceContext]) -> None:
        """Handle publish request from module."""
        if topic not in self._publishers:
            logger.warning(f"Module {self.role} tried to publish to unknown topic: {topic}")
            return

        try:
            msg = self._deserialize_to_ros_message(topic, payload)
            self._publishers[topic].publish(msg)

            if self.trace_enabled and trace:
                logger.debug(f"Published to {topic} (request_id={trace.request_id})")

        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")

    def _serialize_ros_message(self, topic: str, msg: Any) -> Any:
        """Serialize ROS message to JSON-compatible format."""
        msg_type = self.config.topic_types.get(topic, "")

        if msg_type == "std_msgs/String":
            return {"data": msg.data}
        elif msg_type == "sensor_msgs/Image":
            # Encode image data as base64 for JSON transport
            return {
                "header": {
                    "stamp": {
                        "sec": msg.header.stamp.sec,
                        "nanosec": msg.header.stamp.nanosec,
                    },
                    "frame_id": msg.header.frame_id,
                },
                "height": msg.height,
                "width": msg.width,
                "encoding": msg.encoding,
                "is_bigendian": msg.is_bigendian,
                "step": msg.step,
                "data": base64.b64encode(bytes(msg.data)).decode("utf-8"),
            }
        else:
            # Generic fallback: try to convert using slots
            return self._ros_msg_to_dict(msg)

    def _deserialize_to_ros_message(self, topic: str, payload: Any) -> Any:
        """Deserialize JSON payload to ROS message."""
        msg_class = self._msg_classes[topic]
        msg_type = self.config.topic_types.get(topic, "")

        if msg_type == "std_msgs/String":
            msg = msg_class()
            msg.data = payload.get("data", "") if isinstance(payload, dict) else str(payload)
            return msg
        elif msg_type == "sensor_msgs/Image":
            msg = msg_class()
            if isinstance(payload, dict):
                msg.height = payload.get("height", 0)
                msg.width = payload.get("width", 0)
                msg.encoding = payload.get("encoding", "")
                msg.is_bigendian = payload.get("is_bigendian", 0)
                msg.step = payload.get("step", 0)
                data = payload.get("data", "")
                if isinstance(data, str):
                    msg.data = list(base64.b64decode(data))
                else:
                    msg.data = list(data)
                if "header" in payload:
                    h = payload["header"]
                    msg.header.frame_id = h.get("frame_id", "")
                    if "stamp" in h:
                        msg.header.stamp.sec = h["stamp"].get("sec", 0)
                        msg.header.stamp.nanosec = h["stamp"].get("nanosec", 0)
            return msg
        else:
            # Generic fallback
            return self._dict_to_ros_msg(msg_class, payload)

    def _ros_msg_to_dict(self, msg: Any) -> dict:
        """Convert ROS message to dictionary."""
        result = {}
        for slot in msg.__slots__:
            attr_name = slot.lstrip("_")
            value = getattr(msg, attr_name)
            if hasattr(value, "__slots__"):
                result[attr_name] = self._ros_msg_to_dict(value)
            elif isinstance(value, (list, tuple)):
                result[attr_name] = list(value)
            else:
                result[attr_name] = value
        return result

    def _dict_to_ros_msg(self, msg_class: type, data: dict) -> Any:
        """Convert dictionary to ROS message."""
        msg = msg_class()
        if not isinstance(data, dict):
            return msg

        for key, value in data.items():
            if hasattr(msg, key):
                attr = getattr(msg, key)
                if hasattr(attr, "__slots__") and isinstance(value, dict):
                    setattr(msg, key, self._dict_to_ros_msg(type(attr), value))
                else:
                    setattr(msg, key, value)
        return msg

    def get_in_proc_link(self) -> Optional[InProcLink]:
        """Get in-proc link for direct module access."""
        if isinstance(self._link, InProcLink):
            return self._link
        return None


class AdapterManager:
    """Manages all adapters for RoboBridge."""

    def __init__(
        self,
        node: "Node",
        adapter_configs: Dict[str, "AdapterConfig"],
        qos_profiles: Dict[str, Dict[str, Any]],
        socket_defaults: Dict[str, Any],
        trace_enabled: bool = False,
    ):
        self.node = node
        self.adapter_configs = adapter_configs
        self.qos_profiles = qos_profiles
        self.socket_defaults = socket_defaults
        self.trace_enabled = trace_enabled

        self._adapters: Dict[str, Adapter] = {}

    def initialize_all(self) -> None:
        """Initialize all adapters."""
        for role, config in self.adapter_configs.items():
            adapter = Adapter(
                node=self.node,
                config=config,
                qos_profiles=self.qos_profiles,
                socket_defaults=self.socket_defaults,
                trace_enabled=self.trace_enabled,
            )
            adapter.initialize()
            self._adapters[role] = adapter

        logger.info(f"Initialized {len(self._adapters)} adapters")

    def start_all(self) -> None:
        """Start all adapters."""
        for adapter in self._adapters.values():
            adapter.start()
        logger.info("All adapters started")

    def stop_all(self) -> None:
        """Stop all adapters."""
        for adapter in self._adapters.values():
            adapter.stop()
        logger.info("All adapters stopped")

    def get_adapter(self, role: str) -> Optional[Adapter]:
        """Get adapter by role name."""
        return self._adapters.get(role)

    def get_in_proc_link(self, role: str) -> Optional[InProcLink]:
        """Get in-proc link for a role (for direct module registration)."""
        adapter = self._adapters.get(role)
        if adapter:
            return adapter.get_in_proc_link()
        return None
