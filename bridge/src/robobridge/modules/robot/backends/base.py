"""
Base robot backend class.

Abstract interface for robot hardware backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from robobridge.modules.robot.types import (
    ExecutionResult,
    GripperCommand,
    RobotStateData,
    TrajectoryPoint,
)


class RobotBackend(ABC):
    """
    Abstract base class for robot backends.

    Implement this to create custom robot hardware interfaces.

    Example (Franka FCI):
        class FrankaBackend(RobotBackend):
            def connect(self):
                import frankx
                self._robot = frankx.Robot(self.robot_ip)
                self._gripper = frankx.Gripper(self.robot_ip)

            def execute_trajectory(self, points):
                for point in points:
                    motion = frankx.JointMotion(point.positions)
                    self._robot.move(motion)
                return True

            def execute_gripper(self, command):
                if command.action == "close":
                    self._gripper.grasp(command.width, command.speed, command.force)
                else:
                    self._gripper.move(command.width, command.speed)
                return True

            def get_robot_state(self):
                state = self._robot.read_once()
                return RobotStateData(
                    joint_positions=list(state.q),
                    joint_velocities=list(state.dq),
                    ee_pose={"position": ..., "orientation": ...}
                )

    Example (UR RTDE):
        class URBackend(RobotBackend):
            def connect(self):
                import rtde_control
                import rtde_receive
                self._control = rtde_control.RTDEControlInterface(self.robot_ip)
                self._receive = rtde_receive.RTDEReceiveInterface(self.robot_ip)

            def execute_trajectory(self, points):
                path = [[*p.positions, p.velocities[0], p.accelerations[0], 0]
                        for p in points]
                self._control.moveJ(path)
                return True

            def get_robot_state(self):
                return RobotStateData(
                    joint_positions=self._receive.getActualQ(),
                    joint_velocities=self._receive.getActualQd()
                )
    """

    def __init__(self, robot_ip: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backend.

        Args:
            robot_ip: Robot IP address
            config: Backend configuration dictionary
        """
        self.robot_ip = robot_ip
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the robot.

        This method is called once during initialization.
        Store your robot interface objects as instance attributes.
        """
        pass

    @abstractmethod
    def execute_trajectory(self, points: List[TrajectoryPoint]) -> bool:
        """
        Execute trajectory on the robot.

        Args:
            points: List of TrajectoryPoint with:
                - positions: Joint positions (radians)
                - velocities: Optional joint velocities
                - accelerations: Optional joint accelerations
                - time_from_start: Time from trajectory start

        Returns:
            True if execution succeeded, False otherwise
        """
        pass

    @abstractmethod
    def execute_gripper(self, command: GripperCommand) -> bool:
        """
        Execute gripper command.

        Args:
            command: GripperCommand with:
                - action: "open", "close", or "move"
                - width: Target gripper width in meters
                - speed: Movement speed in m/s
                - force: Grasp force in N (for close action)

        Returns:
            True if execution succeeded, False otherwise
        """
        pass

    @abstractmethod
    def get_robot_state(self) -> RobotStateData:
        """
        Get current robot state from hardware.

        Returns:
            RobotStateData with current joint positions, velocities, etc.
        """
        pass

    def disconnect(self) -> None:
        """
        Disconnect from robot. Override if cleanup is needed.
        """
        pass

    def stop_robot(self) -> None:
        """
        Emergency stop the robot. Override for your robot's stop command.
        """
        pass

    def process(self, command: Dict[str, Any]) -> ExecutionResult:
        """
        Process a robot command.

        Args:
            command: Command dictionary with points and gripper_command

        Returns:
            ExecutionResult
        """
        import time

        command_id = command.get("command_id", "unknown")
        command_type = command.get("command_type", "joint")
        raw_points = command.get("points", [])
        gripper_cmd = command.get("gripper_command")

        start_time = time.time()

        try:
            # Execute trajectory
            if raw_points:
                points = [
                    TrajectoryPoint(
                        positions=p.get("positions", []),
                        velocities=p.get("velocities"),
                        accelerations=p.get("accelerations"),
                        time_from_start=p.get("time_from_start", 0.0),
                    )
                    for p in raw_points
                ]

                success = self.execute_trajectory(points)
                if not success:
                    return ExecutionResult(
                        command_id=command_id,
                        success=False,
                        state="error",
                        execution_time_s=time.time() - start_time,
                    )

            # Execute gripper command
            if gripper_cmd:
                cmd = GripperCommand(
                    action=gripper_cmd.get("action", ""),
                    width=gripper_cmd.get("width", 0.08),
                    speed=gripper_cmd.get("speed", 0.1),
                    force=gripper_cmd.get("force", 40.0),
                )

                success = self.execute_gripper(cmd)
                if not success:
                    return ExecutionResult(
                        command_id=command_id,
                        success=False,
                        state="error",
                        execution_time_s=time.time() - start_time,
                    )

            # Get final state
            current_state = self.get_robot_state()

            return ExecutionResult(
                command_id=command_id,
                success=True,
                state="idle",
                actual_positions=current_state.joint_positions,
                execution_time_s=time.time() - start_time,
                metadata={"command_type": command_type},
            )

        except Exception as e:
            return ExecutionResult(
                command_id=command_id,
                success=False,
                state="error",
                execution_time_s=time.time() - start_time,
            )
