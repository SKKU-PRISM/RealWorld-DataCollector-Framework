"""
RoboCasa Bridge - Full Pipeline Integration.

Orchestrates Perception -> Planner -> MoveIt Controller -> RoboCasa via ROS2.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from robobridge.modules.robot.backends.robocasa import RoboCasaConfig
from robobridge.modules.robot.backends.robocasa_ros2 import (
    RoboCasaROS2Backend,
    RoboCasaROS2Config,
)
from robobridge.modules.controller.backends.moveit import MoveItBackend, MoveItConfig

logger = logging.getLogger(__name__)


@dataclass
class RoboCasaBridgeConfig:
    robocasa: RoboCasaConfig = field(default_factory=RoboCasaConfig)
    moveit: MoveItConfig = field(default_factory=MoveItConfig)
    ros2_node_name: str = "robocasa_bridge"
    perception_provider: str = "hf"
    perception_model: str = "microsoft/Florence-2-base"
    perception_device: str = "cpu"
    planner_provider: str = "openai"
    planner_model: str = "gpt-4.1"
    planner_api_key: Optional[str] = None
    use_monitor: bool = False


@dataclass
class ExecutionResult:
    success: bool
    total_reward: float = 0.0
    total_steps: int = 0


class RoboCasaBridge:
    """
    Full pipeline orchestrator for RoboCasa with MoveIt motion planning.
    
    Architecture:
        Perception -> Planner -> MoveIt -> ROS2 Bridge -> RoboCasa
    """

    def __init__(
        self,
        config: Optional[RoboCasaBridgeConfig] = None,
        planner_provider: str = "openai",
        planner_model: str = "gpt-4o",
        planner_api_key: Optional[str] = None,
    ):
        if config:
            self.config = config
        else:
            self.config = RoboCasaBridgeConfig(
                planner_provider=planner_provider,
                planner_model=planner_model,
                planner_api_key=planner_api_key,
            )

        self._robocasa: Optional[RoboCasaROS2Backend] = None
        self._moveit: Optional[MoveItBackend] = None
        self._perception = None
        self._planner = None
        self._initialized = False

    @property
    def task_language(self) -> str:
        return self._robocasa.task_language if self._robocasa else ""

    def initialize(self) -> None:
        ros2_config = RoboCasaROS2Config(
            robocasa=self.config.robocasa,
            node_name=self.config.ros2_node_name,
        )
        self._robocasa = RoboCasaROS2Backend(config=ros2_config)
        self._robocasa.connect()

        self._moveit = MoveItBackend(config=self.config.moveit)
        self._moveit.initialize()

        self._init_perception()
        self._init_planner()

        self._initialized = True
        logger.info(f"RoboCasaBridge initialized. Task: {self.task_language}")

    def _init_perception(self) -> None:
        try:
            from robobridge.modules.perception import Perception
            self._perception = Perception(
                provider=self.config.perception_provider,
                model=self.config.perception_model,
                device=self.config.perception_device,
            )
        except Exception as e:
            logger.warning(f"Perception init failed: {e}")

    def _init_planner(self) -> None:
        try:
            from robobridge.modules.planner import Planner
            planner_kwargs = {
                "provider": self.config.planner_provider,
                "model": self.config.planner_model,
            }
            if self.config.planner_api_key:
                planner_kwargs["api_key"] = self.config.planner_api_key
            self._planner = Planner(**planner_kwargs)
        except Exception as e:
            logger.warning(f"Planner init failed: {e}")

    def shutdown(self) -> None:
        if self._moveit:
            self._moveit.shutdown()
            self._moveit = None
        if self._robocasa:
            self._robocasa.disconnect()
            self._robocasa = None
        self._initialized = False

    def execute(self, instruction: Optional[str] = None) -> ExecutionResult:
        if not self._initialized:
            return ExecutionResult(success=False)

        task_instruction = instruction or self.task_language

        obs = self._robocasa.get_observation()
        detections = self._run_perception(obs)
        plan = self._run_planner(task_instruction, detections)

        if not plan:
            return ExecutionResult(success=False)

        return self._execute_plan(plan)

    def _run_perception(self, obs: Any) -> List[Dict]:
        if self._perception is None or obs.rgb is None:
            return []
        try:
            detections = self._perception.process(rgb=obs.rgb, depth=obs.depth)
            return [
                {"name": d.name, "position": d.pose.get("position", [0, 0, 0]) if d.pose else [0, 0, 0]}
                for d in detections
            ]
        except Exception:
            return []

    def _run_planner(self, instruction: str, detections: List[Dict]) -> List[Any]:
        if self._planner is None:
            return []
        try:
            world_state = {"detections": detections}
            primitive_plans = self._planner.process_full(instruction, world_state)
            return primitive_plans
        except Exception as e:
            logger.warning(f"Planner failed: {e}")
            return []

    def _execute_plan(self, primitive_plans: List[Any]) -> ExecutionResult:
        total_steps = 0
        
        for primitive_plan in primitive_plans:
            parent_action = primitive_plan.parent_action
            logger.info(f"Executing action: {parent_action.action_type} on {parent_action.target_object}")
            
            for primitive in primitive_plan.primitives:
                primitive_dict = primitive.to_dict()
                logger.info(f"  Primitive: {primitive_dict['primitive_type']}")
                
                success = self._moveit.execute_primitive(primitive_dict)
                
                if not success:
                    return ExecutionResult(
                        success=False,
                        total_steps=total_steps,
                    )
                
                total_steps += 1

        return ExecutionResult(
            success=True,
            total_steps=total_steps,
        )
