"""
Custom Planner Wrapper

Wrap your own LLM/planning model to use with RoboBridge.
Simply inherit this class and implement the `load_model` and `plan` methods.
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
class PlanStep:
    """Single step in a plan."""

    step_id: int
    skill: str
    target_object: str
    target_location: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_effects: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "skill": self.skill,
            "target_object": self.target_object,
            "target_location": self.target_location,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
        }


@dataclass
class Plan:
    """High-level plan."""

    plan_id: str
    instruction: str
    steps: List[PlanStep]
    current_step_index: int = 0
    status: str = "success"

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "instruction": self.instruction,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
        }


class CustomPlanner(BaseModule):
    """
    Base wrapper for custom high-level planning (LLM) models.

    To use your own model:
    1. Inherit this class
    2. Implement `load_model()` to load your model
    3. Implement `plan()` to generate plans

    Example:
        class MyLLMPlanner(CustomPlanner):
            def load_model(self):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self._model.to(self.device)

            def plan(self, instruction, object_poses=None, context=None):
                prompt = f"Task: {instruction}\\nGenerate a step-by-step plan:\\n"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._model.generate(**inputs, max_new_tokens=512)
                response = self.tokenizer.decode(outputs[0])
                steps = self._parse_response(response)
                return Plan(
                    plan_id=f"plan_{self._plan_counter}",
                    instruction=instruction,
                    steps=[PlanStep(step_id=i, skill=s["skill"], target_object=s["object"])
                           for i, s in enumerate(steps)]
                )

    Input Topics:
        - /robobridge/instruction: Natural language instruction
        - /perception/objects: Detected object poses

    Output Topics:
        - /planning/high_level_plan: Generated plan
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        # Connection settings
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51002),
        auth_token: Optional[str] = None,
        # Topic settings
        instruction_topic: str = "/robobridge/instruction",
        object_poses_topic: str = "/perception/objects",
        output_topic: str = "/planning/high_level_plan",
        # Planning settings
        temperature: float = 0.7,
        max_tokens: int = 1500,
        skill_list: Optional[List[str]] = None,
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
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.skill_list = skill_list or ["pick", "place", "push", "pull", "open", "close"]

        # Topics
        self.instruction_topic = instruction_topic
        self.object_poses_topic = object_poses_topic
        self.output_topic = output_topic

        # State
        self._model: Any = None
        self._latest_object_poses: Optional[Dict] = None
        self._plan_counter = 0

    @abstractmethod
    def load_model(self) -> None:
        """
        Load your custom planning model.

        This method is called once during initialization.
        Store your model in self._model or any attribute you prefer.

        Example:
            def load_model(self):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self._model.to(self.device)
        """
        pass

    @abstractmethod
    def plan(
        self,
        instruction: str,
        object_poses: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Plan:
        """
        Generate a plan for the given instruction.

        Args:
            instruction: Natural language instruction
            object_poses: Detected object poses from perception
            context: Additional context (e.g., scene info, history)

        Returns:
            Plan object with steps

        Example:
            def plan(self, instruction, object_poses=None, context=None):
                prompt = self._build_prompt(instruction, object_poses)
                response = self._generate(prompt)
                steps = self._parse_response(response)
                self._plan_counter += 1
                return Plan(
                    plan_id=f"plan_{self._plan_counter}",
                    instruction=instruction,
                    steps=steps
                )
        """
        pass

    def start(self) -> None:
        """Start planner with model initialization."""
        logger.info(f"Loading custom planner model from: {self.model_path}")
        self.load_model()
        logger.info("Custom planner model loaded successfully")

        super().start()

        # Register topic handlers
        self.subscribe(self.instruction_topic, self._on_instruction)
        self.subscribe(self.object_poses_topic, self._on_object_poses)

    def _on_instruction(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle instruction message."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    data = json.loads(payload["data"])
                    instruction = data.get("instruction", "")
                except (json.JSONDecodeError, TypeError):
                    instruction = payload.get("instruction", "")
            else:
                instruction = payload.get("instruction", "")
        elif isinstance(payload, str):
            instruction = payload
        else:
            instruction = str(payload)

        if not instruction:
            return

        logger.info(f"Received instruction: {instruction}")

        try:
            plan = self.plan(
                instruction=instruction,
                object_poses=self._latest_object_poses,
            )

            # Publish plan
            self.publish(self.output_topic, {"data": json.dumps(plan.to_dict())}, trace)
            logger.info(f"Published plan: {plan.plan_id}")

        except Exception as e:
            logger.error(f"Planning error: {e}")

    def _on_object_poses(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle object poses message."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    self._latest_object_poses = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    self._latest_object_poses = payload
            else:
                self._latest_object_poses = payload

    def process(self, *args, **kwargs) -> Any:
        """Required by BaseModule - use plan() for planning."""
        return self.plan(*args, **kwargs)
