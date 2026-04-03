"""
RoboCasa ROS2 Bridge Backend.

Bridges RoboCasa simulation with ROS2/MoveIt for motion planning.
Publishes joint states and subscribes to trajectory commands.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .robocasa import RoboCasaBackend, RoboCasaConfig, RoboCasaObservation

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


@dataclass
class RoboCasaROS2Config:
    robocasa: RoboCasaConfig
    node_name: str = "robocasa_bridge"
    joint_state_topic: str = "/joint_states"
    trajectory_topic: str = "/joint_trajectory"
    publish_rate_hz: float = 50.0


class RoboCasaROS2Backend(RoboCasaBackend):
    """
    RoboCasa backend with ROS2 bridge for MoveIt integration.
    
    Publishes /joint_states for MoveIt state monitoring.
    Subscribes to /joint_trajectory for MoveIt commands.
    """

    JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]

    def __init__(
        self,
        robot_ip: str = "simulation",
        config: Optional[RoboCasaROS2Config] = None,
    ):
        ros2_config = config or RoboCasaROS2Config(robocasa=RoboCasaConfig())
        super().__init__(robot_ip=robot_ip, config=ros2_config.robocasa)
        
        self.ros2_config = ros2_config
        self._node: Optional[Any] = None
        self._joint_state_pub = None
        self._trajectory_sub = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False
        self._pending_trajectory: Optional[List[np.ndarray]] = None
        self._trajectory_lock = threading.Lock()

    def connect(self) -> None:
        super().connect()
        
        if not ROS2_AVAILABLE:
            raise ImportError("rclpy not available. Install ROS2.")

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(self.ros2_config.node_name)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        self._joint_state_pub = self._node.create_publisher(
            JointState, self.ros2_config.joint_state_topic, qos
        )
        
        self._trajectory_sub = self._node.create_subscription(
            JointTrajectory,
            self.ros2_config.trajectory_topic,
            self._on_trajectory,
            qos,
        )

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

    def disconnect(self) -> None:
        self._running = False
        
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, '_publish_thread') and self._publish_thread:
            self._publish_thread.join(timeout=2.0)

        if self._node:
            self._node.destroy_node()
            self._node = None

        super().disconnect()

    def _spin_loop(self) -> None:
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)

    def _publish_loop(self) -> None:
        rate = 1.0 / self.ros2_config.publish_rate_hz
        
        while self._running:
            self._publish_joint_state()
            self._execute_pending_trajectory()
            time.sleep(rate)

    def _publish_joint_state(self) -> None:
        if self._joint_state_pub is None or self._last_obs is None:
            return

        state = self.get_robot_state()
        
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES
        
        positions = list(state.joint_positions) if state.joint_positions else [0.0] * 7
        
        gripper_pos = state.gripper_width / 2 if state.gripper_width else 0.04
        positions.extend([gripper_pos, gripper_pos])
        
        msg.position = positions
        msg.velocity = list(state.joint_velocities) + [0.0, 0.0] if state.joint_velocities else [0.0] * 9
        msg.effort = [0.0] * 9

        self._joint_state_pub.publish(msg)

    def _on_trajectory(self, msg: Any) -> None:
        trajectory = []
        for point in msg.points:
            positions = np.array(point.positions[:7])
            trajectory.append(positions)
        
        with self._trajectory_lock:
            self._pending_trajectory = trajectory

    def _execute_pending_trajectory(self) -> None:
        with self._trajectory_lock:
            if self._pending_trajectory is None:
                return
            trajectory = self._pending_trajectory
            self._pending_trajectory = None

        for target_positions in trajectory:
            current = self.get_robot_state()
            current_pos = np.array(current.joint_positions[:7]) if current.joint_positions else np.zeros(7)
            
            delta = target_positions - current_pos
            delta = np.clip(delta, -0.1, 0.1)
            
            action = np.zeros(self.action_dim)
            action[:7] = delta
            
            self.step(action)

    def get_ros2_node(self) -> Optional[Any]:
        return self._node
