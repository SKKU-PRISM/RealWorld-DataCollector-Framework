"""
Base module class for RoboBridge modules.

Provides common functionality for socket/in-proc/http communication with adapters.
"""

import base64
import io
import logging
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from robobridge.core.protocols import LenJsonProtocol, ProtocolError

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Common module configuration."""

    provider: str
    model: str
    device: str = "cpu"
    api_key: Optional[str] = None
    link_mode: str = "direct"
    adapter_endpoint: Optional[Tuple[str, int]] = None
    adapter_protocol: str = "len_json"
    auth_token: Optional[str] = None
    timeout_s: float = 3.0
    max_retries: int = 2


class BaseModule(ABC):
    """
    Base class for RoboBridge modules.

    Handles:
    - Connection to adapter (socket, in-proc, or direct)
    - Message sending/receiving
    - Reconnection logic

    link_mode options:
    - "direct": No network, call process() directly (default, for testing/simulation)
    - "socket": TCP socket communication with adapter
    - "in_proc": In-process queue communication with adapter
    """

    def __init__(
        self,
        provider: str,
        model: str,
        device: str = "cpu",
        api_key: Optional[str] = None,
        link_mode: str = "direct",
        adapter_endpoint: Optional[Tuple[str, int]] = None,
        adapter_protocol: str = "len_json",
        auth_token: Optional[str] = None,
        timeout_s: float = 3.0,
        max_retries: int = 2,
        **kwargs,
    ):
        self.config = ModuleConfig(
            provider=provider,
            model=model,
            device=device,
            api_key=api_key,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            adapter_protocol=adapter_protocol,
            auth_token=auth_token,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        self._socket: Optional[socket.socket] = None
        self._protocol: Optional[LenJsonProtocol] = None
        self._connected = False
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._topic_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # In-proc link reference (set externally when using in_proc mode)
        self._in_proc_link: Optional[Any] = None

        # Store extra kwargs for subclasses
        self._extra_config = kwargs

    def set_in_proc_link(self, link: Any) -> None:
        """
        Set in-proc link for queue-based communication.

        Called by RoboBridge core when using in_proc mode.

        Args:
            link: InProcLink instance
        """
        self._in_proc_link = link

    def connect(self) -> bool:
        """Connect to adapter."""
        if self.config.link_mode != "socket":
            logger.info("Non-socket mode, skipping connect")
            return True

        if not self.config.adapter_endpoint:
            logger.error("No adapter endpoint configured")
            return False

        host, port = self.config.adapter_endpoint

        for attempt in range(self.config.max_retries + 1):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.config.timeout_s)
                self._socket.connect((host, port))

                self._protocol = LenJsonProtocol(
                    recv_timeout_s=self.config.timeout_s,
                    send_timeout_s=self.config.timeout_s,
                    auth_token=self.config.auth_token,
                )

                # Authenticate if token provided
                if self.config.auth_token:
                    if not self._protocol.authenticate_with_server(
                        self._socket, self.config.auth_token
                    ):
                        logger.error("Authentication failed")
                        self._socket.close()
                        continue

                self._connected = True
                logger.info(f"Connected to adapter at {host}:{port}")
                return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception:
                        pass
                time.sleep(0.5)

        logger.error(f"Failed to connect after {self.config.max_retries + 1} attempts")
        return False

    def disconnect(self) -> None:
        """Disconnect from adapter."""
        self._connected = False

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        logger.info("Disconnected from adapter")

    def start(self) -> None:
        """Start receiving messages."""
        if not self._connected and self.config.link_mode == "socket":
            if not self.connect():
                raise RuntimeError("Failed to connect to adapter")

        self._running = True

        if self.config.link_mode == "socket":
            self._recv_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True,
                name=f"{self.__class__.__name__}-Recv",
            )
            self._recv_thread.start()

        logger.info(f"{self.__class__.__name__} started")

    def stop(self) -> None:
        """Stop receiving messages."""
        self._running = False

        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

        self.disconnect()
        logger.info(f"{self.__class__.__name__} stopped")

    def publish(self, topic: str, payload: Any, trace: Optional[dict] = None) -> bool:
        """
        Publish message to adapter.

        Args:
            topic: Topic to publish to
            payload: Message payload
            trace: Optional trace metadata

        Returns:
            True if successful
        """
        if self.config.link_mode == "direct":
            logger.debug("Publish skipped in direct mode")
            return False

        msg: dict[str, Any] = {
            "type": "pub",
            "topic": topic,
            "payload": payload,
        }
        if trace:
            msg["trace"] = trace

        # In-proc mode: use queue
        if self.config.link_mode == "in_proc":
            if self._in_proc_link is None:
                logger.error("In-proc link not set")
                return False
            try:
                self._in_proc_link._from_module_queue.put(msg)
                return True
            except Exception as e:
                logger.error(f"Failed to publish to {topic} via in_proc: {e}")
                return False

        # Socket mode: use TCP
        if not self._connected or not self._socket or not self._protocol:
            logger.error("Not connected to adapter")
            return False

        try:
            with self._lock:
                self._protocol.send(self._socket, msg)
            return True
        except ProtocolError as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False

    def subscribe(self, topic: str, handler: Callable[[Any, Optional[dict]], None]) -> None:
        """
        Register handler for topic messages.

        Args:
            topic: Topic to subscribe to
            handler: Callback function(payload, trace)
        """
        self._topic_handlers[topic] = handler
        logger.debug(f"Subscribed to {topic}")

    def get_message(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get next message from queue.

        Args:
            timeout: Wait timeout in seconds

        Returns:
            Message dict or None if timeout
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _receive_loop(self) -> None:
        """Background receive loop."""
        while self._running:
            try:
                if not self._socket or not self._protocol:
                    time.sleep(0.1)
                    continue

                msg = self._protocol.recv(self._socket)
                self._handle_message(msg)

            except ProtocolError as e:
                if self._running:
                    logger.warning(f"Receive error: {e}")
                    time.sleep(1.0)
                    self.connect()
            except OSError:
                if not self._running:
                    break

    def _handle_message(self, msg: dict) -> None:
        """Handle received message."""
        msg_type = msg.get("type")
        topic = msg.get("topic")
        payload = msg.get("payload")
        trace = msg.get("trace")

        if msg_type == "sub" and topic:
            # Check for registered handler
            if topic in self._topic_handlers:
                try:
                    self._topic_handlers[topic](payload, trace)
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}")

            # Also put in queue for polling
            self._message_queue.put(msg)

    # =========================================================================
    # HTTP Remote Execution Utilities
    # =========================================================================

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """Encode numpy image to JPEG base64 string for HTTP transfer."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _decode_image(b64_str: str) -> np.ndarray:
        """Decode JPEG base64 string to numpy array."""
        from PIL import Image as PILImage

        img_bytes = base64.b64decode(b64_str)
        return np.array(PILImage.open(io.BytesIO(img_bytes)))

    def _http_post(self, endpoint: str, payload: dict, timeout: float = None) -> dict:
        """Send POST request to HTTP module server.

        Args:
            endpoint: API endpoint path (e.g., "process")
            payload: JSON-serializable dict
            timeout: Request timeout in seconds

        Returns:
            Response JSON as dict
        """
        import requests as _requests

        if not self.config.adapter_endpoint:
            raise RuntimeError("adapter_endpoint not configured for HTTP mode")

        host, port = self.config.adapter_endpoint
        url = f"http://{host}:{port}/{endpoint}"
        resp = _requests.post(
            url, json=payload, timeout=timeout or self.config.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method - implemented by subclasses.
        """
        pass

    def __enter__(self) -> "BaseModule":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
