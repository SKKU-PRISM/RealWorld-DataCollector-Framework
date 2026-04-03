"""
Base Controller Interface

Abstract base class for all robot controllers.
Provides unified interface for hardware abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for robot controllers.

    All hardware controllers (Feetech, Dynamixel, etc.) should inherit from this class
    and implement the required methods.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 1000000,
        motor_ids: Optional[List[int]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.motor_ids = motor_ids or []
        self.is_connected = False

        # Position tracking
        self.current_positions = np.zeros(len(self.motor_ids), dtype=np.float32)
        self.target_positions = np.zeros(len(self.motor_ids), dtype=np.float32)

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the robot hardware.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the robot hardware."""
        pass

    @abstractmethod
    def enable_torque(self, motor_ids: Optional[List[int]] = None):
        """
        Enable motor torque.

        Args:
            motor_ids: List of motor IDs to enable. None = all motors
        """
        pass

    @abstractmethod
    def disable_torque(self, motor_ids: Optional[List[int]] = None):
        """
        Disable motor torque (motors can be moved by hand).

        Args:
            motor_ids: List of motor IDs to disable. None = all motors
        """
        pass

    @abstractmethod
    def read_positions(self, normalize: bool = True) -> np.ndarray:
        """
        Read current motor positions.

        Args:
            normalize: If True, return normalized values (-100 to +100)
                      If False, return raw values

        Returns:
            Array of joint positions
        """
        pass

    @abstractmethod
    def write_positions(self, positions: np.ndarray, normalize: bool = True):
        """
        Write target positions to motors.

        Args:
            positions: Target position array
            normalize: If True, positions are normalized (-100 to +100)
                      If False, positions are raw values
        """
        pass

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation (LeRobot compatible).

        Returns:
            {"observation.state": positions}
        """
        positions = self.read_positions()
        return {"observation.state": positions}

    def send_action(self, action: Dict[str, np.ndarray]):
        """
        Send action to robot (LeRobot compatible).

        Args:
            action: {"action": positions}
        """
        if "action" in action:
            self.write_positions(action["action"])
        else:
            raise ValueError("Action dict missing 'action' key")

    def get_joint_positions_radians(self) -> np.ndarray:
        """
        Get current joint positions in radians.

        Useful for kinematics computations.

        Returns:
            Joint positions in radians
        """
        normalized = self.read_positions(normalize=True)
        # Convert normalized (-100 to +100) to radians
        # Assuming ±100 maps to approximately ±π radians
        return normalized / 100.0 * np.pi

    def set_joint_positions_radians(self, positions_rad: np.ndarray):
        """
        Set joint positions from radians.

        Args:
            positions_rad: Joint positions in radians
        """
        # Convert radians to normalized (-100 to +100)
        normalized = positions_rad / np.pi * 100.0
        self.write_positions(normalized, normalize=True)

    @property
    def num_joints(self) -> int:
        """Number of joints/motors."""
        return len(self.motor_ids)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

    def __del__(self):
        """Destructor."""
        if hasattr(self, 'is_connected') and self.is_connected:
            self.disconnect()
