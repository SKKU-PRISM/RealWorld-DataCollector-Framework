"""
State Manager

Manages robot state, history, and observations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import time


@dataclass
class RobotState:
    """Single robot state snapshot."""
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: Optional[np.ndarray] = None
    ee_position: Optional[np.ndarray] = None
    ee_orientation: Optional[np.ndarray] = None


class StateManager:
    """
    Manages robot state and observation history.

    Useful for learning and trajectory recording.

    Args:
        history_length: Number of states to keep in history
        record_ee: Whether to record end-effector state
    """

    def __init__(
        self,
        history_length: int = 100,
        record_ee: bool = False,
    ):
        self.history_length = history_length
        self.record_ee = record_ee

        self._history: deque = deque(maxlen=history_length)
        self._current_state: Optional[RobotState] = None

        # For velocity estimation
        self._last_positions: Optional[np.ndarray] = None
        self._last_time: float = 0.0

    def update(
        self,
        joint_positions: np.ndarray,
        ee_position: Optional[np.ndarray] = None,
        ee_orientation: Optional[np.ndarray] = None,
    ):
        """
        Update state with new observation.

        Args:
            joint_positions: Current joint positions
            ee_position: End-effector position (optional)
            ee_orientation: End-effector orientation (optional)
        """
        current_time = time.time()

        # Estimate velocity
        velocities = None
        if self._last_positions is not None:
            dt = current_time - self._last_time
            if dt > 0:
                velocities = (joint_positions - self._last_positions) / dt

        state = RobotState(
            timestamp=current_time,
            joint_positions=joint_positions.copy(),
            joint_velocities=velocities,
            ee_position=ee_position.copy() if ee_position is not None else None,
            ee_orientation=ee_orientation.copy() if ee_orientation is not None else None,
        )

        self._current_state = state
        self._history.append(state)

        self._last_positions = joint_positions.copy()
        self._last_time = current_time

    @property
    def current_state(self) -> Optional[RobotState]:
        """Get current state."""
        return self._current_state

    @property
    def history(self) -> List[RobotState]:
        """Get state history."""
        return list(self._history)

    def get_observation(self, history_len: int = 1) -> Dict[str, np.ndarray]:
        """
        Get observation in LeRobot-compatible format.

        Args:
            history_len: Number of historical states to include

        Returns:
            Dictionary with observation arrays
        """
        if len(self._history) == 0:
            return {}

        # Get recent states
        states = list(self._history)[-history_len:]

        # Stack joint positions
        positions = np.array([s.joint_positions for s in states])

        obs = {
            "observation.state": positions.flatten() if history_len > 1 else positions[0],
        }

        # Add EE position if available
        if states[-1].ee_position is not None:
            ee_positions = np.array([
                s.ee_position for s in states if s.ee_position is not None
            ])
            obs["observation.ee_position"] = ee_positions.flatten() if history_len > 1 else ee_positions[0]

        return obs

    def clear(self):
        """Clear history."""
        self._history.clear()
        self._current_state = None
        self._last_positions = None
        self._last_time = 0.0

    def get_trajectory(self) -> np.ndarray:
        """
        Get recorded trajectory from history.

        Returns:
            Array of shape (num_states, num_joints)
        """
        if len(self._history) == 0:
            return np.array([])

        return np.array([s.joint_positions for s in self._history])

    def get_timestamps(self) -> np.ndarray:
        """Get timestamps for recorded history."""
        if len(self._history) == 0:
            return np.array([])

        timestamps = np.array([s.timestamp for s in self._history])
        return timestamps - timestamps[0]  # Relative to start
