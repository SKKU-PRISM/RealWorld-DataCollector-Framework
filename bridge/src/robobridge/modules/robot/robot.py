"""
Robot Interface Module

Receives low-level commands from the controller and forwards them to the robot.
This module handles the communication layer - actual robot control is implemented
in CustomRobot wrappers or backend classes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule

from .types import (
    ExecutionResult,
    GripperCommand,
    RobotConfig,
    RobotExecutionState,
    RobotStateData,
    TrajectoryPoint,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Robot(BaseModule):
    """
    Robot Interface Module

    Handles communication between the Controller and the custom robot interface.
    This module is responsible for:
    - Receiving commands from /planning/low_level_cmd topic
    - Forwarding commands to the configured custom robot interface
    - Publishing execution results to /robot/execution_result topic
    - Publishing robot state to /robot/state topic

    For actual robot control, use a CustomRobot wrapper that implements
    the connection to your specific robot SDK (Franka FCI, UR RTDE, etc.)

    There are two ways to use this module:

    1. Simulation mode (no real robot):
       Uses the built-in simulation for testing without hardware.

       robot = Robot(
           custom_interface="simulation",
           link_mode="socket",
           adapter_endpoint=("127.0.0.1", 51001)
       )

    2. With CustomRobot:
       Pass your own interface instance that inherits from CustomRobot.

       class MyFrankaInterface(CustomRobot):
           def connect(self):
               self._robot = frankx.Robot(self.robot_ip)
           # ... implement other methods

       my_interface = MyFrankaInterface(robot_ip="<ROBOT_IP>")
       robot = Robot(
           custom_interface=my_interface,
           link_mode="socket",
           adapter_endpoint=("127.0.0.1", 51001)
       )

    Args:
        custom_interface: CustomRobot instance or "simulation" for sim mode
        robot_type: Robot type identifier (for metadata)
        rate_hz: State publishing frequency
        timeout_s: Execution timeout
        command_topic: Command input topic
        result_topic: Result output topic
        state_topic: State output topic
        link_mode: Connection mode
        adapter_endpoint: (host, port) for socket mode
        auth_token: Authentication token
    """

    def __init__(
        self,
        custom_interface: Any = "simulation",  # CustomRobot instance or "simulation"
        robot_type: str = "franka",
        rate_hz: float = 100.0,
        timeout_s: float = 15.0,
        units: str = "SI",
        frame_convention: str = "base",
        estop_policy: str = "stop_and_report",
        link_mode: str = "direct",
        adapter_endpoint: Optional[Tuple[str, int]] = None,
        adapter_protocol: str = "len_json",
        auth_token: Optional[str] = None,
        command_topic: str = "/planning/low_level_cmd",
        result_topic: str = "/robot/execution_result",
        state_topic: str = "/robot/state",
        max_retries: int = 1,
        **kwargs,
    ):
        super().__init__(
            provider="robot_interface",
            model=robot_type,
            device="robot",
            api_key=None,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            adapter_protocol=adapter_protocol,
            auth_token=auth_token,
            timeout_s=timeout_s,
            max_retries=max_retries,
            **kwargs,
        )

        self.robot_config = RobotConfig(
            robot_type=robot_type,
            rate_hz=rate_hz,
            timeout_s=timeout_s,
            units=units,
            frame_convention=frame_convention,
            estop_policy=estop_policy,
            command_topic=command_topic,
            result_topic=result_topic,
            state_topic=state_topic,
        )

        # CustomRobot instance or "simulation"
        self._custom_interface = custom_interface
        self._use_simulation = custom_interface == "simulation"

        # Simulation state (only used when custom_interface="simulation")
        self._sim_state = RobotStateData(
            joint_positions=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            joint_velocities=[0.0] * 7,
            joint_torques=[0.0] * 7,
            ee_pose={
                "position": {"x": 0.5, "y": 0.0, "z": 0.4},
                "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
            },
            gripper_width=0.08,
            robot_mode="idle",
            timestamp=time.time(),
        )

        self._execution_state = RobotExecutionState.IDLE
        self._state_lock = threading.Lock()
        self._state_publish_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start robot interface."""
        if self._use_simulation:
            logger.info("Starting Robot in simulation mode")
        else:
            logger.info(
                f"Starting Robot with custom interface: {type(self._custom_interface).__name__}"
            )
            # Start the custom interface (connects to robot)
            if hasattr(self._custom_interface, "start"):
                self._custom_interface.start()
            elif hasattr(self._custom_interface, "connect"):
                self._custom_interface.connect()

        super().start()

        # Register topic handlers
        self.subscribe(self.robot_config.command_topic, self._on_command)

        # Start state publishing thread
        self._state_publish_thread = threading.Thread(
            target=self._state_publish_loop,
            daemon=True,
            name="Robot-StatePublish",
        )
        self._state_publish_thread.start()

    def stop(self) -> None:
        """Stop robot interface."""
        self._running = False

        # Stop robot if executing
        if self._execution_state == RobotExecutionState.EXECUTING:
            self._stop_robot()

        if self._state_publish_thread:
            self._state_publish_thread.join(timeout=2.0)

        # Stop custom interface
        if not self._use_simulation and self._custom_interface:
            if hasattr(self._custom_interface, "stop"):
                self._custom_interface.stop()
            elif hasattr(self._custom_interface, "disconnect"):
                self._custom_interface.disconnect()

        super().stop()

    def _state_publish_loop(self) -> None:
        """Background loop to publish robot state."""
        rate = 1.0 / min(self.robot_config.rate_hz, 50.0)

        while self._running:
            try:
                state_dict = self._get_current_state().to_dict()

                self.publish(self.robot_config.state_topic, {"data": json.dumps(state_dict)})

            except Exception as e:
                logger.error(f"State publish error: {e}")

            time.sleep(rate)

    def _get_current_state(self) -> RobotStateData:
        """Get current robot state from custom_interface or simulation."""
        if self._use_simulation:
            with self._state_lock:
                self._sim_state.timestamp = time.time()
                self._sim_state.robot_mode = self._execution_state.value
                return RobotStateData(
                    joint_positions=self._sim_state.joint_positions.copy(),
                    joint_velocities=self._sim_state.joint_velocities.copy(),
                    joint_torques=self._sim_state.joint_torques.copy(),
                    ee_pose=(self._sim_state.ee_pose.copy() if self._sim_state.ee_pose else None),
                    gripper_width=self._sim_state.gripper_width,
                    gripper_force=self._sim_state.gripper_force,
                    robot_mode=self._sim_state.robot_mode,
                    errors=self._sim_state.errors.copy(),
                    timestamp=self._sim_state.timestamp,
                )
        else:
            # Get state from custom_interface
            custom_interface_state = self._custom_interface.get_robot_state()
            custom_interface_state.robot_mode = self._execution_state.value
            return custom_interface_state

    def _on_command(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle incoming command."""
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

        # Execute command
        result = self.process(command)

        # Publish result
        self.publish(
            self.robot_config.result_topic,
            {"data": json.dumps(result.to_dict())},
            trace,
        )

    def process(self, command: Dict) -> ExecutionResult:
        """
        Execute robot command.

        Args:
            command: Low-level command dictionary

        Returns:
            ExecutionResult
        """
        # HTTP remote mode
        if self.config.link_mode == "http":
            result = self._http_post("process", command, timeout=15.0)
            return ExecutionResult.from_dict(result)

        command_id = command.get("command_id", "unknown")
        command_type = command.get("command_type", "trajectory")
        points = command.get("points", [])
        gripper_cmd = command.get("gripper_command")

        start_time = time.time()

        try:
            self._execution_state = RobotExecutionState.EXECUTING

            if self._use_simulation:
                # Simulation mode
                result = self._execute_simulation(command_id, command_type, points, gripper_cmd)
            else:
                # Forward to custom_interface
                result = self._custom_interface.process(command)

            self._execution_state = RobotExecutionState.IDLE
            result.execution_time_s = time.time() - start_time
            return result

        except Exception as e:
            self._execution_state = RobotExecutionState.ERROR
            logger.error(f"Execution error: {e}")

            return ExecutionResult(
                command_id=command_id,
                success=False,
                state=self._execution_state.value,
                execution_time_s=time.time() - start_time,
            )

    def _execute_simulation(
        self,
        command_id: str,
        command_type: str,
        points: List[Dict],
        gripper_cmd: Optional[Dict],
    ) -> ExecutionResult:
        """Execute command in simulation mode."""
        # Simulate trajectory
        for point in points:
            positions = point.get("positions", [])
            time_from_start = point.get("time_from_start", 0.0)

            time.sleep(min(time_from_start, 0.5))

            with self._state_lock:
                if positions:
                    self._sim_state.joint_positions = list(positions)

        # Simulate gripper
        if gripper_cmd:
            action = gripper_cmd.get("action", "")
            width = gripper_cmd.get("width", 0.08)
            force = gripper_cmd.get("force", 40.0)

            time.sleep(0.5)

            with self._state_lock:
                if action == "close":
                    self._sim_state.gripper_width = 0.0
                    self._sim_state.gripper_force = force
                elif action == "open":
                    self._sim_state.gripper_width = width
                    self._sim_state.gripper_force = 0.0
                else:
                    self._sim_state.gripper_width = width

        with self._state_lock:
            actual_positions = self._sim_state.joint_positions.copy()

        return ExecutionResult(
            command_id=command_id,
            success=True,
            state=RobotExecutionState.IDLE.value,
            actual_positions=actual_positions,
            metadata={"command_type": command_type, "simulation": True},
        )

    def _stop_robot(self) -> None:
        """Emergency stop the robot."""
        logger.warning("Stopping robot")
        self._execution_state = RobotExecutionState.IDLE

        if self._use_simulation:
            with self._state_lock:
                self._sim_state.joint_velocities = [0.0] * 7
        elif hasattr(self._custom_interface, "stop_robot"):
            self._custom_interface.stop_robot()
        elif hasattr(self._custom_interface, "stop"):
            self._custom_interface.stop()

    def get_state(self) -> RobotStateData:
        """Get current robot state."""
        return self._get_current_state()

    def is_executing(self) -> bool:
        """Check if robot is currently executing a command."""
        return self._execution_state == RobotExecutionState.EXECUTING

    def has_error(self) -> bool:
        """Check if robot has an error."""
        return self._execution_state == RobotExecutionState.ERROR
