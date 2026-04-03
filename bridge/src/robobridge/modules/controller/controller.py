"""
Controller Module

Converts high-level skill plans into executable robot trajectories/commands.
Supports VLA models, MoveIt, motion primitives, and custom backends.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule

from .types import Command, ControllerConfig, TrajectoryPoint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Controller(BaseModule):
    """
    Controller Module (Low-Level Planner)

    Converts high-level plans into executable robot commands.

    Supported backends:
    - "vla": Vision-Language-Action models (pi0, rt-2, octo, etc.)
    - "moveit": MoveIt motion planning
    - "primitives": Skill primitives with parameterized motion
    - "custom": Custom wrapper class

    Args:
        provider: Model provider
        model: Model identifier
        backend: Motion planning backend
        device: Compute device
        api_key: API key
        temperature: Sampling temperature (for VLA)
        action_space: Output command type
        control_rate_hz: Control loop frequency
        horizon_steps: Prediction horizon
        frame_convention: Coordinate frame
        safety_limits: Motion safety constraints
        link_mode: Connection mode
        adapter_endpoint: (host, port) for socket mode
        auth_token: Authentication token
        plan_topic: High-level plan input topic
        rgb_topic: RGB image input topic
        depth_topic: Depth image input topic
        robot_state_topic: Robot state input topic
        output_topic: Command output topic
        timeout_s: Operation timeout
        max_retries: Max retry attempts
    """

    def __init__(
        self,
        provider: str,
        model: str,
        backend: str = "vla",
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        action_space: str = "trajectory",
        control_rate_hz: float = 20.0,
        horizon_steps: int = 10,
        frame_convention: str = "base",
        safety_limits: Optional[Dict[str, Any]] = None,
        link_mode: str = "direct",
        adapter_endpoint: Optional[Tuple[str, int]] = None,
        adapter_protocol: str = "len_json",
        auth_token: Optional[str] = None,
        plan_topic: str = "/planning/high_level_plan",
        primitive_plan_topic: str = "/planning/primitive_plan",
        rgb_topic: str = "/camera/rgb",
        depth_topic: str = "/camera/depth",
        robot_state_topic: str = "/robot/state",
        output_topic: str = "/planning/low_level_cmd",
        timeout_s: float = 6.0,
        max_retries: int = 1,
        **kwargs,
    ):
        super().__init__(
            provider=provider,
            model=model,
            device=device,
            api_key=api_key,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            adapter_protocol=adapter_protocol,
            auth_token=auth_token,
            timeout_s=timeout_s,
            max_retries=max_retries,
            **kwargs,
        )

        default_limits = {
            "max_joint_vel": 1.5,
            "max_ee_vel": 0.3,
            "workspace": [-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
        }

        self.controller_config = ControllerConfig(
            backend=backend,
            temperature=temperature,
            action_space=action_space,
            control_rate_hz=control_rate_hz,
            horizon_steps=horizon_steps,
            frame_convention=frame_convention,
            safety_limits=safety_limits or default_limits,
            plan_topic=plan_topic,
            primitive_plan_topic=primitive_plan_topic,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            robot_state_topic=robot_state_topic,
            output_topic=output_topic,
        )

        self._model: Any = None
        self._processor: Any = None
        self._custom_wrapper: Any = None
        self._vla_lora: Any = None  # VLALoRAController instance
        self._current_plan: Optional[Dict] = None
        self._current_primitive_plan: Optional[Dict] = None
        self._current_step_idx: int = 0
        self._current_primitive_idx: int = 0
        self._latest_rgb: Any = None
        self._latest_depth: Any = None
        self._robot_state: Dict = {}
        self._command_counter = 0

    def initialize_model(self) -> None:
        """Initialize the motion planning backend."""
        backend = self.controller_config.backend.lower()

        if backend == "vla_lora":
            self._init_vla_lora()
        elif backend == "vla":
            self._init_vla_model()
        elif backend == "moveit":
            self._init_moveit()
        elif backend == "primitives":
            self._init_primitives()
        elif backend == "custom":
            self._init_custom_wrapper()
        else:
            logger.warning(f"Unknown backend: {backend}, using stub mode")

        logger.info(f"Initialized {backend} backend with model: {self.config.model}")

    def _init_custom_wrapper(self) -> None:
        """Initialize custom wrapper from model path."""
        from robobridge.utils import load_custom_class
        from robobridge.wrappers import CustomController

        try:
            custom_cls = load_custom_class(self.config.model, CustomController)
            self._custom_wrapper = custom_cls(
                model_path=self.config.model,
                device=self.config.device,
                control_rate_hz=self.controller_config.control_rate_hz,
                horizon_steps=self.controller_config.horizon_steps,
                action_space=self.controller_config.action_space,
                link_mode=self.config.link_mode,
                adapter_endpoint=self.config.adapter_endpoint,
                auth_token=self.config.auth_token,
            )
            self._custom_wrapper.load_model()
            logger.info(f"Loaded custom wrapper from: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to load custom wrapper: {e}")
            raise

    def _init_vla_model(self) -> None:
        """Initialize VLA model."""
        try:
            model_name = self.config.model.lower()

            if "pi0" in model_name:
                self._init_pi0()
            elif "rt" in model_name:
                self._init_rt()
            elif "octo" in model_name:
                self._init_octo()
            else:
                logger.warning(f"Unknown VLA model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize VLA model: {e}")

    def _init_pi0(self) -> None:
        """Initialize Pi0 model."""
        logger.info("Initializing Pi0 model (stub)")

    def _init_rt(self) -> None:
        """Initialize RT-X model."""
        logger.info("Initializing RT model (stub)")

    def _init_octo(self) -> None:
        """Initialize Octo model."""
        logger.info("Initializing Octo model (stub)")

    def _init_vla_lora(self) -> None:
        """Initialize VLA LoRA controller with adapter switching."""
        from .ik_solver import create_ik_solver
        from .vla_lora_controller import VLALoRAController

        vla_config = getattr(self, "_vla_config", None)
        if vla_config is None:
            # Try to get from kwargs or use defaults
            vla_config = {
                "backend": self.config.model.split("/")[0] if "/" not in self.config.model else "openvla",
                "model_name": self.config.model,
                "device": self.config.device,
                "quantize_4bit": True,
            }

        # Create IK solver from vla_config
        ik_config = vla_config.get("ik_solver", {})
        ik_type = ik_config.get("type", "passthrough")
        ik_kwargs = {k: v for k, v in ik_config.items() if k != "type"}
        ik_solver = create_ik_solver(ik_type, **ik_kwargs)

        use_absolute = vla_config.get("use_absolute_targets", False)

        self._vla_lora = VLALoRAController(
            vla_config=vla_config,
            control_rate_hz=self.controller_config.control_rate_hz,
            frame_convention=self.controller_config.frame_convention,
            ik_solver=ik_solver,
            use_absolute_targets=use_absolute,
        )
        self._vla_lora.initialize()
        logger.info(
            f"VLA LoRA controller initialized: ik={ik_type}, "
            f"absolute_targets={use_absolute}"
        )

    def update_observation(self, rgb=None, robot_state=None):
        """Update latest observation for VLA inference.

        Args:
            rgb: RGB image (numpy array)
            robot_state: Robot state dict with ee_pose, gripper_qpos, etc.
        """
        if rgb is not None:
            self._latest_rgb = rgb
        if robot_state is not None:
            self._robot_state = robot_state

    def execute_primitive(self, primitive: Dict, instruction: str = "") -> Any:
        """
        Convert a primitive action to a Command.

        Controller only decides *what* to do (generates a Command).
        Execution is always handled by the Robot module.

        Args:
            primitive: Primitive dict with primitive_type, target_position, etc.
            instruction: Full natural language task instruction for VLA models.

        Returns:
            Command for Robot to execute
        """
        return self.process_primitive(
            primitive=primitive,
            rgb=self._latest_rgb,
            depth=self._latest_depth,
            robot_state=self._robot_state,
            instruction=instruction,
        )

    def reset_policy(self) -> None:
        """Reset VLA policy state for new episode."""
        if self._vla_lora is not None:
            self._vla_lora.reset_policy()

    def soft_reset_policy(self) -> None:
        """Partial reset: clear chunk buffer but keep EMA for smooth transitions."""
        if self._vla_lora is not None:
            self._vla_lora.soft_reset_policy()

    def _init_moveit(self) -> None:
        """Initialize MoveIt interface."""
        try:
            logger.info("MoveIt backend selected - will use ROS2 MoveIt interface")
        except Exception as e:
            logger.error(f"Failed to initialize MoveIt: {e}")

    def _init_primitives(self) -> None:
        """Initialize motion primitives."""
        logger.info("Using motion primitives backend")

    def start(self) -> None:
        """Start controller with backend initialization."""
        self.initialize_model()
        super().start()

        self.subscribe(self.controller_config.plan_topic, self._on_plan)
        self.subscribe(self.controller_config.primitive_plan_topic, self._on_primitive_plan)
        self.subscribe(self.controller_config.rgb_topic, self._on_rgb)
        self.subscribe(self.controller_config.depth_topic, self._on_depth)
        self.subscribe(self.controller_config.robot_state_topic, self._on_robot_state)

    def _on_plan(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle high-level plan message."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    plan = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    plan = payload
            else:
                plan = payload
        elif isinstance(payload, str):
            try:
                plan = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                plan = {"raw": payload}
        else:
            plan = {"data": payload}

        self._current_plan = plan
        self._current_step_idx = int(plan.get("current_step_index", 0))

        logger.info(f"Received plan: {plan.get('plan_id', 'unknown')}")

        # Start executing plan
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
                self._robot_state = {"raw": payload}

    def _on_primitive_plan(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle primitive plan message from PrimitivePlanner."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    prim_plan = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    prim_plan = payload
            else:
                prim_plan = payload
        elif isinstance(payload, str):
            try:
                prim_plan = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                prim_plan = {"raw": payload}
        else:
            prim_plan = {"data": payload}

        self._current_primitive_plan = prim_plan
        self._current_primitive_idx = int(prim_plan.get("current_primitive_index", 0))

        logger.info(f"Received primitive plan: {prim_plan.get('plan_id', 'unknown')}")

        self._execute_primitive_plan(prim_plan, trace)

    def _execute_primitive_plan(self, prim_plan: Dict, trace: Optional[dict] = None) -> None:
        """Execute all primitives in a primitive plan."""
        primitives = prim_plan.get("primitives", [])
        
        for i, primitive in enumerate(primitives):
            logger.info(f"Executing primitive {i}: {primitive.get('primitive_type')}")
            
            command = self.process_primitive(
                primitive=primitive,
                rgb=self._latest_rgb,
                depth=self._latest_depth,
                robot_state=self._robot_state,
            )

            if command:
                self.publish(
                    self.controller_config.output_topic,
                    {"data": json.dumps(command.to_dict())},
                    trace,
                )
                logger.info(f"Published command: {command.command_id}")

    def process_primitive(
        self,
        primitive: Dict,
        rgb: Optional[Any] = None,
        depth: Optional[Any] = None,
        robot_state: Optional[Dict] = None,
        instruction: str = "",
    ) -> Optional[Command]:
        """
        Convert primitive action to low-level command.

        Args:
            primitive: Primitive action (go, move, grip)
            rgb: RGB image observation
            depth: Depth image observation
            robot_state: Current robot state
            instruction: Full natural language task instruction for VLA models.

        Returns:
            Command or None
        """
        # HTTP remote mode
        if self.config.link_mode == "http":
            payload = {"primitive": primitive}
            if rgb is not None:
                payload["rgb_b64"] = self._encode_image(rgb)
            if robot_state is not None:
                payload["robot_state"] = robot_state
            result = self._http_post("process_primitive", payload, timeout=15.0)
            if result is None:
                return None
            return Command.from_dict(result)

        # Delegate to VLA LoRA controller if active
        if self._vla_lora is not None:
            return self._vla_lora.process_primitive(
                primitive=primitive,
                rgb=rgb,
                robot_state=robot_state,
                instruction=instruction,
            )

        prim_type = primitive.get("primitive_type", "")

        if prim_type == "go":
            return self._process_go_primitive(primitive, robot_state)
        elif prim_type == "move":
            return self._process_move_primitive(primitive, robot_state)
        elif prim_type == "grip":
            return self._process_grip_primitive(primitive)
        else:
            logger.warning(f"Unknown primitive type: {prim_type}")
            return None

    def _process_go_primitive(
        self,
        primitive: Dict,
        robot_state: Optional[Dict],
    ) -> Optional[Command]:
        """Generate command for mobile base movement."""
        target_pos = primitive.get("target_position", {})
        approach_dir = primitive.get("approach_direction", {})
        
        self._command_counter += 1
        return Command(
            command_id=f"cmd_{self._command_counter:04d}",
            command_type="base_trajectory",
            points=[
                TrajectoryPoint(
                    positions=[
                        target_pos.get("x", 0.0),
                        target_pos.get("y", 0.0),
                        0.0,
                    ],
                    time_from_start=2.0,
                )
            ],
            frame_id="map",
            metadata={
                "primitive_type": "go",
                "approach_direction": approach_dir,
                "safety_distance": primitive.get("safety_distance", 0.3),
            },
        )

    def _process_move_primitive(
        self,
        primitive: Dict,
        robot_state: Optional[Dict],
    ) -> Optional[Command]:
        """Generate command for arm end-effector movement."""
        target_pos = primitive.get("target_position", {})
        target_rot = primitive.get("target_rotation", {})
        approach_dir = primitive.get("approach_direction", {})

        rotations = None
        if target_rot:
            rotations = [
                target_rot.get("roll", 0.0),
                target_rot.get("pitch", 0.0),
                target_rot.get("yaw", 0.0),
            ]

        self._command_counter += 1
        return Command(
            command_id=f"cmd_{self._command_counter:04d}",
            command_type="cartesian_trajectory",
            points=[
                TrajectoryPoint(
                    positions=[
                        target_pos.get("x", 0.0),
                        target_pos.get("y", 0.0),
                        target_pos.get("z", 0.0),
                    ],
                    rotations=rotations,
                    time_from_start=1.5,
                )
            ],
            frame_id=self.controller_config.frame_convention,
            metadata={
                "primitive_type": "move",
                "approach_direction": approach_dir,
            },
        )

    def _process_grip_primitive(self, primitive: Dict) -> Optional[Command]:
        """Generate command for gripper control."""
        grip_width = primitive.get("grip_width", 0.0)
        
        self._command_counter += 1
        return Command(
            command_id=f"cmd_{self._command_counter:04d}",
            command_type="gripper",
            points=[],
            frame_id=self.controller_config.frame_convention,
            gripper_command={
                "action": "close" if grip_width < 0.01 else "open",
                "width": grip_width,
            },
            metadata={
                "primitive_type": "grip",
            },
        )

    def _execute_current_step(self, trace: Optional[dict] = None) -> None:
        """Execute current step of the plan (legacy)."""
        if not self._current_plan:
            return

        steps = self._current_plan.get("steps", [])
        if self._current_step_idx >= len(steps):
            logger.info("All plan steps completed")
            return

        step = steps[self._current_step_idx]
        logger.info(f"Executing step {step.get('step_id')}: {step.get('skill')}")

        # Generate low-level command
        command = self.process(
            step=step,
            rgb=self._latest_rgb,
            depth=self._latest_depth,
            robot_state=self._robot_state,
        )

        if command:
            # Publish command
            self.publish(
                self.controller_config.output_topic,
                {"data": json.dumps(command.to_dict())},
                trace,
            )
            logger.info(f"Published command: {command.command_id}")

    def process(
        self,
        step: Dict,
        rgb: Optional[Any] = None,
        depth: Optional[Any] = None,
        robot_state: Optional[Dict] = None,
    ) -> Optional[Command]:
        """
        Convert high-level step to low-level command.

        Args:
            step: High-level plan step
            rgb: RGB image observation
            depth: Depth image observation
            robot_state: Current robot state

        Returns:
            Command or None
        """
        backend = self.controller_config.backend.lower()

        if self._custom_wrapper is not None:
            # Use custom wrapper's generate_trajectory method
            custom_cmd = self._custom_wrapper.generate_trajectory(
                step=step, rgb=rgb, depth=depth, robot_state=robot_state
            )
            # Convert custom Command to our Command if needed
            if hasattr(custom_cmd, "to_dict"):
                cmd_dict = custom_cmd.to_dict()
                return Command(
                    command_id=cmd_dict.get("command_id", f"cmd_{self._command_counter}"),
                    command_type=cmd_dict.get("command_type", self.controller_config.action_space),
                    points=[
                        TrajectoryPoint(
                            positions=p.get("positions", []),
                            velocities=p.get("velocities"),
                            accelerations=p.get("accelerations"),
                            time_from_start=p.get("time_from_start", 0.0),
                        )
                        for p in cmd_dict.get("points", [])
                    ],
                    frame_id=cmd_dict.get("frame_id", self.controller_config.frame_convention),
                    gripper_command=cmd_dict.get("gripper_command"),
                )
            return custom_cmd
        elif backend == "vla" and self._model:
            return self._plan_vla(step, rgb, depth, robot_state)
        elif backend == "moveit":
            return self._plan_moveit(step, robot_state)
        elif backend == "primitives":
            return self._plan_primitives(step, robot_state)
        else:
            return self._plan_stub(step)

    def _plan_vla(
        self,
        step: Dict,
        rgb: Optional[Any],
        depth: Optional[Any],
        robot_state: Optional[Dict],
    ) -> Optional[Command]:
        """Generate trajectory using VLA model."""
        try:
            logger.debug("Running VLA inference (stub)")

            points = []
            dt = 1.0 / self.controller_config.control_rate_hz

            for i in range(self.controller_config.horizon_steps):
                point = TrajectoryPoint(
                    positions=[0.0] * 7,  # 7-DoF arm
                    velocities=[0.0] * 7,
                    time_from_start=i * dt,
                )
                points.append(point)

            self._command_counter += 1
            command = Command(
                command_id=f"cmd_{self._command_counter:04d}",
                command_type=self.controller_config.action_space,
                points=points,
                frame_id=self.controller_config.frame_convention,
                gripper_command=self._get_gripper_command(step),
                metadata={
                    "step_id": step.get("step_id"),
                    "skill": step.get("skill"),
                    "model": self.config.model,
                },
            )

            return command

        except Exception as e:
            logger.error(f"VLA planning error: {e}")
            return None

    def _plan_moveit(self, step: Dict, robot_state: Optional[Dict]) -> Optional[Command]:
        """Generate trajectory using MoveIt."""
        try:
            skill = step.get("skill", "")
            target_object = step.get("target_object", "")

            logger.info(f"MoveIt planning for {skill} -> {target_object}")

            self._command_counter += 1
            command = Command(
                command_id=f"cmd_{self._command_counter:04d}",
                command_type="trajectory",
                points=[],  # Would be filled by MoveIt
                frame_id=self.controller_config.frame_convention,
                gripper_command=self._get_gripper_command(step),
                metadata={
                    "step_id": step.get("step_id"),
                    "skill": skill,
                    "planner": "moveit",
                },
            )

            return command

        except Exception as e:
            logger.error(f"MoveIt planning error: {e}")
            return None

    def _plan_primitives(self, step: Dict, robot_state: Optional[Dict]) -> Optional[Command]:
        """Generate trajectory using motion primitives."""
        skill = step.get("skill", "")
        target_object = step.get("target_object", "")
        target_location = step.get("target_location")
        params = step.get("parameters", {})

        # Map skill to primitive
        primitive_map = {
            "pick": self._primitive_pick,
            "place": self._primitive_place,
            "push": self._primitive_push,
            "pull": self._primitive_pull,
            "open": self._primitive_open,
            "close": self._primitive_close,
            "move_to": self._primitive_move_to,
        }

        primitive_fn = primitive_map.get(skill, self._primitive_default)
        points, gripper_cmd = primitive_fn(target_object, target_location, params, robot_state)

        self._command_counter += 1
        command = Command(
            command_id=f"cmd_{self._command_counter:04d}",
            command_type="waypoints",
            points=points,
            frame_id=self.controller_config.frame_convention,
            gripper_command=gripper_cmd,
            metadata={
                "step_id": step.get("step_id"),
                "skill": skill,
                "planner": "primitives",
            },
        )

        return command

    def _primitive_pick(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate pick motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.0, -0.3, 0.3, -0.8, 0.0, 0.8, 0.0], time_from_start=1.0),
            TrajectoryPoint(positions=[0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0], time_from_start=2.0),
        ]
        gripper = {"action": "close", "width": 0.0, "force": 40.0}
        return points, gripper

    def _primitive_place(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate place motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.3, -0.3, 0.3, -0.8, 0.0, 0.8, 0.0], time_from_start=1.0),
            TrajectoryPoint(positions=[0.3, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=2.0),
        ]
        gripper = {"action": "open", "width": 0.08}
        return points, gripper

    def _primitive_push(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate push motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.2, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=1.5),
        ]
        return points, None

    def _primitive_pull(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate pull motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.2, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=1.5),
        ]
        gripper = {"action": "close", "width": 0.0, "force": 20.0}
        return points, gripper

    def _primitive_open(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate open (e.g., drawer) motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[-0.3, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=2.0),
        ]
        gripper = {"action": "close", "width": 0.0, "force": 20.0}
        return points, gripper

    def _primitive_close(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate close (e.g., drawer) motion primitive."""
        points = [
            TrajectoryPoint(positions=[-0.3, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=2.0),
        ]
        return points, None

    def _primitive_move_to(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Generate move-to motion primitive."""
        points = [
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0], time_from_start=2.0),
        ]
        return points, None

    def _primitive_default(
        self,
        target: str,
        location: Optional[str],
        params: Dict,
        state: Optional[Dict],
    ) -> Tuple[List[TrajectoryPoint], Optional[Dict]]:
        """Default motion primitive."""
        return [], None

    def _plan_stub(self, step: Dict) -> Optional[Command]:
        """Generate stub command for testing."""
        self._command_counter += 1

        points = [
            TrajectoryPoint(positions=[0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0], time_from_start=0.0),
            TrajectoryPoint(positions=[0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0], time_from_start=2.0),
        ]

        return Command(
            command_id=f"cmd_{self._command_counter:04d}",
            command_type=self.controller_config.action_space,
            points=points,
            frame_id=self.controller_config.frame_convention,
            gripper_command=self._get_gripper_command(step),
            metadata={
                "step_id": step.get("step_id"),
                "skill": step.get("skill"),
                "planner": "stub",
            },
        )

    def _get_gripper_command(self, step: Dict) -> Optional[Dict]:
        """Determine gripper command from step."""
        skill = step.get("skill", "")

        gripper_map = {
            "pick": {"action": "close", "width": 0.0, "force": 40.0},
            "grasp": {"action": "close", "width": 0.0, "force": 40.0},
            "place": {"action": "open", "width": 0.08},
            "release": {"action": "open", "width": 0.08},
        }

        return gripper_map.get(skill)

    def check_safety(self, command: Command) -> bool:
        """Check if command satisfies safety constraints."""
        limits = self.controller_config.safety_limits

        for point in command.points:
            # Check joint velocity limits
            if point.velocities:
                max_vel = max(abs(v) for v in point.velocities)
                if max_vel > limits.get("max_joint_vel", float("inf")):
                    logger.warning(f"Joint velocity {max_vel} exceeds limit")
                    return False

        return True
