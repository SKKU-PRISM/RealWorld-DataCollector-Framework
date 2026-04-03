"""
Length-prefixed JSON protocol implementation.

Protocol format:
- 4-byte big-endian length prefix
- JSON payload bytes (UTF-8 encoded)

Packet format (Module <-> Adapter):
- Module -> Adapter (publish): {"type":"pub","topic":"/x","payload":{...},"trace":{...}}
- Adapter -> Module (subscribe): {"type":"sub","topic":"/y","payload":{...},"trace":{...}}
"""

import json
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class ProtocolConfig:
    """Configuration for length-prefixed JSON protocol."""

    recv_timeout_s: float = 3.0
    send_timeout_s: float = 3.0
    max_payload_bytes: int = 10485760  # 10MB


class ProtocolError(Exception):
    """Base exception for protocol errors."""

    pass


class PayloadTooLargeError(ProtocolError):
    """Raised when payload exceeds max size."""

    pass


class TimeoutError(ProtocolError):
    """Raised when operation times out."""

    pass


class ConnectionClosedError(ProtocolError):
    """Raised when connection is closed unexpectedly."""

    pass


class AuthenticationError(ProtocolError):
    """Raised when authentication fails."""

    pass


def send_message(
    sock: socket.socket,
    data: dict,
    config: Optional[ProtocolConfig] = None,
) -> None:
    """
    Send a length-prefixed JSON message.

    Args:
        sock: Socket to send on
        data: Dictionary to send as JSON
        config: Protocol configuration
    """
    if config is None:
        config = ProtocolConfig()

    try:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"Failed to serialize message: {e}")

    if len(payload) > config.max_payload_bytes:
        raise PayloadTooLargeError(
            f"Payload size {len(payload)} exceeds max {config.max_payload_bytes}"
        )

    length_prefix = struct.pack(">I", len(payload))

    sock.settimeout(config.send_timeout_s)
    try:
        sock.sendall(length_prefix + payload)
    except socket.timeout:
        raise TimeoutError("Send operation timed out")
    except socket.error as e:
        raise ProtocolError(f"Send failed: {e}")


def recv_message(
    sock: socket.socket,
    config: Optional[ProtocolConfig] = None,
) -> dict:
    """
    Receive a length-prefixed JSON message.

    Args:
        sock: Socket to receive from
        config: Protocol configuration

    Returns:
        Deserialized JSON as dictionary
    """
    if config is None:
        config = ProtocolConfig()

    sock.settimeout(config.recv_timeout_s)

    # Read length prefix (4 bytes)
    try:
        length_data = _recv_exactly(sock, 4)
    except socket.timeout:
        raise TimeoutError("Receive operation timed out waiting for length prefix")

    if len(length_data) == 0:
        raise ConnectionClosedError("Connection closed by peer")

    if len(length_data) < 4:
        raise ProtocolError(f"Incomplete length prefix: got {len(length_data)} bytes")

    payload_length = struct.unpack(">I", length_data)[0]

    if payload_length > config.max_payload_bytes:
        raise PayloadTooLargeError(
            f"Incoming payload size {payload_length} exceeds max {config.max_payload_bytes}"
        )

    # Read payload
    try:
        payload_data = _recv_exactly(sock, payload_length)
    except socket.timeout:
        raise TimeoutError("Receive operation timed out waiting for payload")

    if len(payload_data) < payload_length:
        raise ProtocolError(
            f"Incomplete payload: expected {payload_length}, got {len(payload_data)}"
        )

    try:
        return json.loads(payload_data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"Failed to deserialize message: {e}")


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return data  # Connection closed
        data += chunk
    return data


def encode_message(data: dict) -> bytes:
    """
    Encode a dictionary to length-prefixed JSON bytes.

    This is a convenience function for encoding without a socket.
    Useful for testing or serializing messages to bytes.

    Args:
        data: Dictionary to encode

    Returns:
        Length-prefixed JSON bytes
    """
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    length_prefix = struct.pack(">I", len(payload))
    return length_prefix + payload


def decode_message(data: bytes) -> dict:
    """
    Decode length-prefixed JSON bytes to a dictionary.

    This is a convenience function for decoding without a socket.
    Useful for testing or deserializing messages from bytes.

    Args:
        data: Length-prefixed JSON bytes

    Returns:
        Decoded dictionary
    """
    if len(data) < 4:
        raise ProtocolError("Data too short for length prefix")

    payload_length = struct.unpack(">I", data[:4])[0]
    payload_data = data[4:]

    if len(payload_data) < payload_length:
        raise ProtocolError(
            f"Incomplete payload: expected {payload_length}, got {len(payload_data)}"
        )

    return json.loads(payload_data[:payload_length].decode("utf-8"))


class LenJsonProtocol:
    """
    High-level wrapper for length-prefixed JSON protocol.

    Provides connection management, authentication, and message handling.
    """

    def __init__(
        self,
        recv_timeout_s: float = 3.0,
        send_timeout_s: float = 3.0,
        max_payload_bytes: int = 10485760,
        auth_token: Optional[str] = None,
    ):
        self.config = ProtocolConfig(
            recv_timeout_s=recv_timeout_s,
            send_timeout_s=send_timeout_s,
            max_payload_bytes=max_payload_bytes,
        )
        self.auth_token = auth_token
        self._lock = threading.Lock()

    def send(self, sock: socket.socket, data: dict) -> None:
        """Thread-safe send."""
        with self._lock:
            send_message(sock, data, self.config)

    def recv(self, sock: socket.socket) -> dict:
        """Thread-safe receive."""
        return recv_message(sock, self.config)

    def authenticate_client(self, sock: socket.socket) -> bool:
        """
        Authenticate incoming client connection.

        Expects first message to be: {"type": "auth", "token": "<token>"}
        Responds with: {"type": "auth_result", "success": true/false}

        Returns:
            True if authentication succeeded or no auth required
        """
        if not self.auth_token:
            return True

        try:
            auth_msg = self.recv(sock)

            if auth_msg.get("type") != "auth":
                self.send(
                    sock,
                    {
                        "type": "auth_result",
                        "success": False,
                        "error": "Expected auth message",
                    },
                )
                return False

            if auth_msg.get("token") != self.auth_token:
                self.send(
                    sock,
                    {
                        "type": "auth_result",
                        "success": False,
                        "error": "Invalid token",
                    },
                )
                return False

            self.send(sock, {"type": "auth_result", "success": True})
            return True

        except ProtocolError:
            return False

    def authenticate_with_server(self, sock: socket.socket, token: str) -> bool:
        """
        Authenticate with server (client side).

        Args:
            sock: Connected socket
            token: Authentication token

        Returns:
            True if authentication succeeded
        """
        try:
            self.send(sock, {"type": "auth", "token": token})
            response = self.recv(sock)
            return bool(response.get("type") == "auth_result" and response.get("success", False))
        except ProtocolError:
            return False

    def create_pub_message(
        self,
        topic: str,
        payload: Any,
        trace: Optional[dict] = None,
    ) -> dict:
        """Create a publish message."""
        msg: dict[str, Any] = {
            "type": "pub",
            "topic": topic,
            "payload": payload,
        }
        if trace:
            msg["trace"] = trace
        return msg

    def create_sub_message(
        self,
        topic: str,
        payload: Any,
        trace: Optional[dict] = None,
    ) -> dict:
        """Create a subscribe (forward) message."""
        msg: dict[str, Any] = {
            "type": "sub",
            "topic": topic,
            "payload": payload,
        }
        if trace:
            msg["trace"] = trace
        return msg
