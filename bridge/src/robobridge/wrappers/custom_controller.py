"""
Custom Controller Wrapper (Low-Level Planner)

Wrap your own VLA/motion planning model to use with RoboBridge.
Simply inherit this class and implement the `load_model` and `generate_trajectory` methods.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    positions: List[float]  # Joint positions (7-DoF) or Cartesian pose (6/7-DoF)
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None
    time_from_start: float = 0.0


@dataclass
class Command:
    """Low-level command for robot execution."""

    command_id: str
    command_type: str  # "trajectory", "waypoints", "joint_targets", "cartesian"
    points: List[TrajectoryPoint]
    frame_id: str = "base"
    gripper_command: Optional[Dict] = None  # {"action": "open/close", "width": float}

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "frame_id": self.frame_id,
            "points": [
                {
                    "positions": p.positions,
                    "velocities": p.velocities,
                    "accelerations": p.accelerations,
                    "time_from_start": p.time_from_start,
                }
                for p in self.points
            ],
            "gripper_command": self.gripper_command,
        }


class CustomController(BaseModule):
    """
    Base wrapper for custom low-level planning / VLA models.

    To use your own model:
    1. Inherit this class
    2. Implement `load_model()` to load your model
    3. Implement `generate_trajectory()` to generate robot commands

    Example:
        class MyVLA(CustomController):
            def load_model(self):
                self.model = load_vla_model(self.model_path)
                self.model.to(self.device)

            def generate_trajectory(self, step, rgb=None, depth=None, robot_state=None):
                action = self.model.predict(rgb, step["skill"])
                return Command(
                    command_id=f"cmd_{time.time()}",
                    command_type="trajectory",
                    points=[TrajectoryPoint(positions=action.tolist())]
                )

    Input Topics:
        - /planning/high_level_plan: High-level plan with steps
        - /camera/rgb: RGB image (for VLA models)
        - /camera/depth: Depth image (optional)
        - /robot/state: Current robot state

    Output Topics:
        - /planning/low_level_cmd: Robot command
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        # Connection settings
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51003),
        auth_token: Optional[str] = None,
        # Topic settings
        plan_topic: str = "/planning/high_level_plan",
        rgb_topic: str = "/camera/rgb",
        depth_topic: str = "/camera/depth",
        robot_state_topic: str = "/robot/state",
        output_topic: str = "/planning/low_level_cmd",
        # Planning settings
        control_rate_hz: float = 20.0,
        horizon_steps: int = 10,
        action_space: str = "trajectory",
        **kwargs,
    ):
        super().__init__(
            provider="custom",
            model=model_path,
            device=device,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            auth_token=auth_token,
            **kwargs,
        )

        self.model_path = model_path
        self.device = device
        self.control_rate_hz = control_rate_hz
        self.horizon_steps = horizon_steps
        self.action_space = action_space

        # Topics
        self.plan_topic = plan_topic
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.robot_state_topic = robot_state_topic
        self.output_topic = output_topic

        # State
        self._model: Any = None
        self._current_plan: Optional[Dict] = None
        self._current_step_idx: int = 0
        self._latest_rgb: Any = None
        self._latest_depth: Any = None
        self._robot_state: Dict = {}
        self._command_counter = 0

    @abstractmethod
    def load_model(self) -> None:
        """
        Load your custom VLA/motion planning model.

        This method is called once during initialization.
        Store your model in self._model or any attribute you prefer.

        Example:
            def load_model(self):
                import torch
                self._model = torch.load(self.model_path)
                self._model.to(self.device)
                self._model.eval()
        """
        pass

    @abstractmethod
    def generate_trajectory(
        self,
        step: Dict,
        rgb: Optional[Any] = None,
        depth: Optional[Any] = None,
        robot_state: Optional[Dict] = None,
    ) -> Command:
        """
        Generate robot trajectory/command for a high-level step.

        Args:
            step: High-level plan step
                Format: {
                    "step_id": int,
                    "skill": str,  # e.g., "pick", "place"
                    "target_object": str,
                    "target_location": str,
                    "parameters": dict
                }
            rgb: Current RGB image observation
            depth: Current depth image (optional)
            robot_state: Current robot state
                Format: {
                    "joint_positions": [7 floats],
                    "joint_velocities": [7 floats],
                    "ee_pose": {"position": {x,y,z}, "orientation": {x,y,z,w}}
                }

        Returns:
            Command object containing trajectory and gripper command

        Example:
            def generate_trajectory(self, step, rgb=None, depth=None, robot_state=None):
                image = self._preprocess_image(rgb)
                state = robot_state.get("joint_positions", [0]*7)

                with torch.no_grad():
                    actions = self._model.predict(image, state, step["skill"])

                dt = 1.0 / self.control_rate_hz
                points = [
                    TrajectoryPoint(
                        positions=actions[i].tolist(),
                        time_from_start=i * dt
                    )
                    for i in range(len(actions))
                ]

                gripper = None
                if step["skill"] == "pick":
                    gripper = {"action": "close", "width": 0.0}

                self._command_counter += 1
                return Command(
                    command_id=f"cmd_{self._command_counter}",
                    command_type=self.action_space,
                    points=points,
                    gripper_command=gripper
                )
        """
        pass

    def start(self) -> None:
        """Start controller with model initialization."""
        logger.info(f"Loading custom controller model from: {self.model_path}")
        self.load_model()
        logger.info("Custom controller model loaded successfully")

        super().start()

        # Register topic handlers
        self.subscribe(self.plan_topic, self._on_plan)
        self.subscribe(self.rgb_topic, self._on_rgb)
        self.subscribe(self.depth_topic, self._on_depth)
        self.subscribe(self.robot_state_topic, self._on_robot_state)

    def _on_plan(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle high-level plan message."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    self._current_plan = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    self._current_plan = payload
            else:
                self._current_plan = payload
        elif isinstance(payload, str):
            try:
                self._current_plan = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                self._current_plan = {"raw": payload}
        else:
            self._current_plan = None
            return

        self._current_step_idx = self._current_plan.get("current_step_index", 0)
        logger.info(f"Received plan: {self._current_plan.get('plan_id', 'unknown')}")

        # Execute current step
        self._execute_current_step(trace)

    def _on_rgb(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle RGB image message."""
        self._latest_rgb = payload

    def _on_depth(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle depth image message."""
        self._latest_depth = payload

    def _on_robot_state(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle robot state message."""
        if isinstance(payload, dict):
            self._robot_state = payload
        elif isinstance(payload, str):
            try:
                self._robot_state = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                self._robot_state = {}

    def _execute_current_step(self, trace: Optional[dict] = None) -> None:
        """Execute current step of the plan."""
        if not self._current_plan:
            return

        steps = self._current_plan.get("steps", [])
        if self._current_step_idx >= len(steps):
            logger.info("All plan steps completed")
            return

        step = steps[self._current_step_idx]
        logger.info(f"Executing step {step.get('step_id')}: {step.get('skill')}")

        try:
            # Generate command using custom model
            command = self.generate_trajectory(
                step=step,
                rgb=self._latest_rgb,
                depth=self._latest_depth,
                robot_state=self._robot_state,
            )

            # Publish command
            self.publish(self.output_topic, {"data": json.dumps(command.to_dict())}, trace)
            logger.info(f"Published command: {command.command_id}")

        except Exception as e:
            logger.error(f"Trajectory generation error: {e}")

    def process(self, *args, **kwargs) -> Any:
        """Required by BaseModule - use generate_trajectory() for planning."""
        return self.generate_trajectory(*args, **kwargs)
