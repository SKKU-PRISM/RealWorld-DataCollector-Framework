"""
Custom Robot Interface Wrapper

Wrap your own robot SDK (Franka FCI, UR RTDE, etc.) to use with RoboBridge.
Simply inherit this class and implement the required methods.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule
from robobridge.modules.robot.types import ExecutionResult, RobotState

logger = logging.getLogger(__name__)


class CustomRobot(BaseModule):
    """
    Base wrapper for custom robot interfaces.

    To use your own robot SDK:
    1. Inherit this class
    2. Implement `connect()` to establish connection to your robot
    3. Implement `execute()` to send commands to your robot
    4. Implement `get_state()` to read current robot state
    5. Optionally implement `disconnect()` and `stop()` for cleanup

    Example (Franka FCI):
        class FrankaInterface(CustomRobot):
            def connect(self):
                import frankx
                self._robot = frankx.Robot(self.robot_ip)
                self._gripper = frankx.Gripper(self.robot_ip)

            def execute(self, command):
                points = command.get("points", [])
                for point in points:
                    motion = frankx.JointMotion(point["positions"])
                    self._robot.move(motion)
                return ExecutionResult(success=True, command_id=command.get("command_id", ""))

            def get_state(self):
                state = self._robot.read_once()
                return RobotState(
                    joint_positions=list(state.q),
                    gripper_width=self._gripper.width()
                )

    Input Topics:
        - /planning/low_level_cmd: Trajectory commands from Controller

    Output Topics:
        - /robot/execution_result: Execution result
        - /robot/state: Current robot state (published periodically)
    """

    def __init__(
        self,
        robot_ip: str = "",
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51001),
        auth_token: Optional[str] = None,
        command_topic: str = "/planning/low_level_cmd",
        result_topic: str = "/robot/execution_result",
        state_topic: str = "/robot/state",
        state_publish_rate_hz: float = 50.0,
        command_timeout_s: float = 30.0,
        **kwargs,
    ):
        super().__init__(
            provider="custom",
            model="robot_interface",
            device="robot",
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            auth_token=auth_token,
            **kwargs,
        )

        self.robot_ip = robot_ip
        self.command_topic = command_topic
        self.result_topic = result_topic
        self.state_topic = state_topic
        self.state_publish_rate_hz = state_publish_rate_hz
        self.command_timeout_s = command_timeout_s

        self._state_publish_thread: Optional[threading.Thread] = None

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to your robot."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from robot."""
        pass

    @abstractmethod
    def execute(self, command: Dict[str, Any]) -> ExecutionResult:
        """
        Execute robot command.

        Args:
            command: Command dict with keys like:
                - command_id: str
                - command_type: str
                - points: List of trajectory points
                - gripper_command: Optional gripper action

        Returns:
            ExecutionResult with success status
        """
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop robot motion."""
        pass

    def start(self) -> None:
        """Start interface with robot connection."""
        logger.info(f"Connecting to robot at: {self.robot_ip}")
        self.connect()
        logger.info("Robot connected successfully")

        super().start()

        self.subscribe(self.command_topic, self._on_command)

        self._state_publish_thread = threading.Thread(
            target=self._state_publish_loop,
            daemon=True,
            name="CustomRobot-StatePublish",
        )
        self._state_publish_thread.start()

    def shutdown(self) -> None:
        """Stop interface and disconnect from robot."""
        self._running = False

        self.stop()

        if self._state_publish_thread:
            self._state_publish_thread.join(timeout=2.0)

        self.disconnect()
        super().stop()

    def _state_publish_loop(self) -> None:
        """Background loop to publish robot state."""
        rate = 1.0 / min(self.state_publish_rate_hz, 100.0)

        while self._running:
            try:
                state = self.get_state()
                self.publish(
                    self.state_topic,
                    {
                        "data": json.dumps(
                            {
                                "joint_positions": state.joint_positions,
                                "joint_velocities": state.joint_velocities,
                                "gripper_width": state.gripper_width,
                                "gripper_force": state.gripper_force,
                                "is_moving": state.is_moving,
                                "timestamp": state.timestamp,
                            }
                        )
                    },
                )
            except Exception as e:
                logger.error(f"State publish error: {e}")

            time.sleep(rate)

    def _on_command(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle incoming command from Controller."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    command = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    command = payload
            else:
                command = payload
        elif isinstance(payload, str):
            try:
                command = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                command = {"raw": payload}
        else:
            command = {"data": payload}

        logger.info(f"Received command: {command.get('command_id', 'unknown')}")

        result = self.execute(command)

        self.publish(
            self.result_topic,
            {
                "data": json.dumps(
                    {
                        "success": result.success,
                        "command_id": result.command_id,
                        "state": result.state,
                        "execution_time_s": result.execution_time_s,
                    }
                )
            },
            trace,
        )

    def process(self, command: Dict) -> ExecutionResult:
        """Required by BaseModule - execute command directly."""
        return self.execute(command)
