"""
RoboCasa Robot Wrapper for RoboBridge.

High-level wrapper that integrates RoboCasa with RoboBridge pub/sub system.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robobridge.modules.base import BaseModule
from robobridge.modules.robot.types import ExecutionResult, RobotState
from robobridge.modules.robot.backends.robocasa import (
    RoboCasaBackend,
    RoboCasaConfig,
    RoboCasaObservation,
)

logger = logging.getLogger(__name__)


class RoboCasaRobot(BaseModule):
    """
    High-level RoboCasa robot wrapper with pub/sub integration.
    
    Wraps RoboCasaBackend and integrates with RoboBridge messaging system.
    Publishes robot state and observations, subscribes to commands.
    """

    def __init__(
        self,
        robocasa_config: Optional[RoboCasaConfig] = None,
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51001),
        command_topic: str = "/planning/low_level_cmd",
        result_topic: str = "/robot/execution_result",
        state_topic: str = "/robot/state",
        image_topic: str = "/camera/rgb",
        state_publish_rate_hz: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            provider="robocasa",
            model="simulation",
            device="cpu",
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            **kwargs,
        )
        
        self.robocasa_config = robocasa_config or RoboCasaConfig()
        self.command_topic = command_topic
        self.result_topic = result_topic
        self.state_topic = state_topic
        self.image_topic = image_topic
        self.state_publish_rate_hz = state_publish_rate_hz
        
        self._backend: Optional[RoboCasaBackend] = None
        self._publish_thread: Optional[threading.Thread] = None

    @property
    def backend(self) -> Optional[RoboCasaBackend]:
        return self._backend

    @property
    def task_language(self) -> str:
        return self._backend.task_language if self._backend else ""

    def start(self) -> None:
        self._backend = RoboCasaBackend(config=self.robocasa_config)
        self._backend.connect()
        
        logger.info(f"RoboCasa connected. Task: {self.task_language}")
        
        super().start()
        
        self.subscribe(self.command_topic, self._on_command)
        
        self._publish_thread = threading.Thread(
            target=self._publish_loop, daemon=True
        )
        self._publish_thread.start()

    def shutdown(self) -> None:
        self._running = False
        
        if self._publish_thread:
            self._publish_thread.join(timeout=2.0)
        
        if self._backend:
            self._backend.disconnect()
            self._backend = None
        
        super().stop()

    def reset(self) -> RoboCasaObservation:
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.reset()

    def step(self, action: np.ndarray) -> Tuple[RoboCasaObservation, float, bool, Dict]:
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.step(action)

    def get_observation(self) -> RoboCasaObservation:
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend.get_observation()

    def _publish_loop(self) -> None:
        rate = 1.0 / self.state_publish_rate_hz
        
        while self._running:
            try:
                if self._backend:
                    state = self._backend.get_robot_state()
                    self.publish(
                        self.state_topic,
                        {"data": json.dumps({
                            "joint_positions": state.joint_positions,
                            "joint_velocities": state.joint_velocities,
                            "gripper_width": state.gripper_width,
                        })}
                    )
                    
                    obs = self._backend.get_observation()
                    if obs.rgb is not None:
                        self.publish(
                            self.image_topic,
                            {"data": obs.rgb.tobytes(), "shape": list(obs.rgb.shape)}
                        )
            except Exception as e:
                logger.error(f"Publish error: {e}")
            
            time.sleep(rate)

    def _on_command(self, payload: Any, trace: Optional[dict]) -> None:
        if isinstance(payload, dict) and "data" in payload:
            try:
                command = json.loads(payload["data"])
            except (json.JSONDecodeError, TypeError):
                command = payload
        else:
            command = payload if isinstance(payload, dict) else {"raw": payload}

        result = self.process(command)
        
        self.publish(
            self.result_topic,
            {"data": json.dumps({
                "success": result.success,
                "command_id": result.command_id,
                "state": result.state,
            })},
            trace,
        )

    def process(self, command: Dict) -> ExecutionResult:
        if self._backend is None:
            return ExecutionResult(
                command_id=command.get("command_id", ""),
                success=False,
                state="error",
            )
        
        return self._backend.process(command)
