"""
RoboBridge Client - User-friendly API for robot control

Provides a simple, LangChain-style API for interacting with robots.

Example:
    from robobridge import RoboBridge

    robot = RoboBridge.initialize()
    robot.execute("Pick up the red cup and place it on the table")
    robot.pick("red_cup", grasp_width=0.05)
    robot.shutdown()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .types import (
    ExecutionResult,
    RobotState,
    DetectedObject,
    ModelInfo,
    PerceptionState,
    PerceptionBuffer,
    scene_divergence,
)
from ..modules.controller.types import Command
import numpy as np

logger = logging.getLogger(__name__)


class RoboBridgeClient:
    """
    User-friendly API client for RoboBridge.

    Provides simple methods for robot control, perception, and task execution.

    Example:
        robot = RoboBridgeClient.initialize(config_path="./config.py")

        # Natural language execution
        robot.execute("Pick up the red cup and place it on the table")

        # Direct skill calls
        robot.pick("red_cup", grasp_width=0.05)
        robot.move(position=[0.4, 0.2, 0.3])
        robot.place("red_cup", position=[0.4, 0.2, 0.05])

        # Model configuration
        robot.set_model("planner", provider="anthropic", model="claude-3-sonnet")

        robot.shutdown()
    """

    _instance: Optional["RoboBridgeClient"] = None

    def __init__(
        self,
        config_path: Optional[str] = None,
        robot_ip: Optional[str] = None,
        simulation: bool = True,
        modules: Optional[List[str]] = None,
        auto_start: bool = True,
        link_mode: str = "direct",
        perception: Optional[Dict[str, Any]] = None,
        planner: Optional[Dict[str, Any]] = None,
        controller: Optional[Dict[str, Any]] = None,
        monitor: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RoboBridge client.

        Args:
            config_path: Path to config.py file
            robot_ip: Robot IP address (for real robot)
            simulation: Use simulation mode
            modules: List of modules to enable (None = all)
            auto_start: Automatically start all modules
            link_mode: Communication mode ("direct" or "socket")
            perception: Perception config {"provider": "...", "model": "..."}
            planner: Planner config {"provider": "...", "model": "..."}
            controller: Controller config (optional)
            monitor: Monitor config (optional)
        """
        self.config_path = config_path or "./config.py"
        self.robot_ip = robot_ip
        self.simulation = simulation
        self.link_mode = link_mode
        self.enabled_modules = modules or [
            "perception",
            "planner",
            "controller",
            "robot",
            "monitor",
        ]

        # Module config overrides
        self._module_overrides = {
            "perception": perception or {},
            "planner": planner or {},
            "controller": controller or {},
            "monitor": monitor or {},
        }

        # Internal state
        self._core = None
        self._modules: Dict[str, Any] = {}
        self._running = False
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

        # Cached state
        self._latest_robot_state: Optional[RobotState] = None
        self._latest_objects: List[DetectedObject] = []
        self._current_plan: Optional[Dict] = None

        # Environment state (for simulation)
        self._env = None
        self._obs = None
        self._ep_meta = None

        # Control state
        self._paused = False
        self._velocity_scale = 1.0

        # Template plan config (training-data-aligned plans)
        self._template_plan_config = None
        self._current_task_name = None

        # Result waiting
        self._pending_results: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}

        if auto_start:
            self._start()

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        robot_ip: Optional[str] = None,
        simulation: bool = True,
        modules: Optional[List[str]] = None,
        link_mode: str = "direct",
        perception: Optional[Dict[str, Any]] = None,
        planner: Optional[Dict[str, Any]] = None,
        controller: Optional[Dict[str, Any]] = None,
        monitor: Optional[Dict[str, Any]] = None,
    ) -> "RoboBridgeClient":
        """
        Initialize and return RoboBridge client instance.

        Args:
            config_path: Path to config.py file
            robot_ip: Robot IP address (for real robot)
            simulation: Use simulation mode (default: True)
            modules: List of modules to enable (None = all)
            link_mode: Communication mode ("direct" or "socket")
            perception: Perception config {"provider": "...", "model": "..."}
            planner: Planner config {"provider": "...", "model": "..."}
            controller: Controller config (optional)
            monitor: Monitor config (optional)

        Returns:
            RoboBridgeClient instance

        Example (direct mode - no ROS):
            bridge = RoboBridge.initialize(
                perception={"provider": "robocasa_gt"},
                planner={"provider": "openai", "model": "gpt-4o"},
            )

        Example (socket mode - with ROS):
            bridge = RoboBridge.initialize(
                link_mode="socket",
                config_path="./config.py",
            )
        """
        if cls._instance is not None:
            logger.warning("RoboBridge already initialized, returning existing instance")
            return cls._instance

        cls._instance = cls(
            config_path=config_path,
            robot_ip=robot_ip,
            simulation=simulation,
            modules=modules,
            auto_start=True,
            link_mode=link_mode,
            perception=perception,
            planner=planner,
            controller=controller,
            monitor=monitor,
        )
        return cls._instance

    def _start(self) -> None:
        """Start RoboBridge core and modules."""
        logger.info("Starting RoboBridge...")

        if self.link_mode == "direct":
            # Direct mode: no ROS, just initialize modules directly
            self._init_modules()
        else:
            # Socket/in_proc mode: use ROS core
            from ..core import RoboBridge as RoboBridgeCore
            self._core = RoboBridgeCore(
                ros_domain_id=0,
                ros_namespace="/robobridge",
                adapters_config_path=self.config_path,
                trace=True,
            )
            self._core.start()
            self._init_modules()
            self._setup_subscriptions()

        self._running = True
        logger.info(f"RoboBridge initialized (link_mode={self.link_mode})")

    def _init_modules(self) -> None:
        """Initialize enabled modules."""
        import os
        from ..modules import Perception, Planner, Controller, Robot, Monitor

        # Get config from file or use overrides
        file_configs = {}
        if self.link_mode != "direct":
            from ..config import load_config
            config = load_config(self.config_path)
            file_configs = config.module_configs

        def get_config(module_name: str) -> dict:
            """Merge file config with overrides, overrides take precedence."""
            base = file_configs.get(module_name, {})
            override = self._module_overrides.get(module_name, {})
            return {**base, **override}

        def get_adapter_params(module_name: str) -> dict:
            """Get adapter params based on link_mode.

            Supports per-module HTTP override: if the module config dict
            contains link_mode="http" and adapter_endpoint, use those
            instead of the global link_mode.
            """
            # Per-module HTTP override (e.g., remote VLA controller)
            override = self._module_overrides.get(module_name, {})
            if override.get("link_mode") == "http":
                endpoint = override.get("adapter_endpoint", ("127.0.0.1", 8000))
                if isinstance(endpoint, list):
                    endpoint = tuple(endpoint)
                return {
                    "link_mode": "http",
                    "adapter_endpoint": endpoint,
                }

            if self.link_mode == "direct":
                return {"link_mode": "direct"}

            from ..config import load_config
            config = load_config(self.config_path)
            adapter = config.adapters.get(module_name)
            if adapter is None:
                return {
                    "link_mode": "socket",
                    "adapter_endpoint": ("127.0.0.1", 51000),
                    "adapter_protocol": "len_json",
                    "auth_token": None,
                }
            return {
                "link_mode": adapter.link_mode,
                "adapter_endpoint": (adapter.bind_host, adapter.bind_port),
                "adapter_protocol": "len_json",
                "auth_token": adapter.auth_token,
            }

        # Perception
        if "perception" in self.enabled_modules:
            cfg = get_config("perception")
            adapter = get_adapter_params("perception")
            self._modules["perception"] = Perception(
                provider=cfg.get("provider", "stub"),
                model=cfg.get("model", ""),
                device=cfg.get("device", "cpu"),
                api_key=cfg.get("api_key"),
                image_size=cfg.get("image_size", 640),
                conf_threshold=cfg.get("conf_threshold", 0.25),
                nms_threshold=cfg.get("nms_threshold", 0.5),
                max_dets=cfg.get("max_dets", 50),
                pose_format=cfg.get("pose_format", "pose_quat"),
                frame_id=cfg.get("frame_id", "robot_base"),
                timeout_s=cfg.get("timeout_s", 3.0),
                max_retries=cfg.get("max_retries", 2),
                **adapter,
            )
            if adapter.get("link_mode") != "http":
                self._modules["perception"].initialize_model()
            logger.info(f"Perception initialized: {cfg.get('provider', 'stub')}/{cfg.get('model', '')} (link={adapter.get('link_mode', 'direct')})")

        # Planner
        if "planner" in self.enabled_modules:
            cfg = get_config("planner")
            adapter = get_adapter_params("planner")
            self._modules["planner"] = Planner(
                provider=cfg.get("provider", "openai"),
                model=cfg.get("model", "gpt-4o"),
                api_key=cfg.get("api_key") or os.environ.get("OPENAI_API_KEY"),
                api_base=cfg.get("api_base"),
                temperature=cfg.get("temperature", 0.3),
                max_tokens=cfg.get("max_tokens", 2000),
                timeout_s=cfg.get("timeout_s", 12.0),
                max_retries=cfg.get("max_retries", 2),
                **adapter,
            )
            if adapter.get("link_mode") != "http":
                self._modules["planner"].initialize_client()
            logger.info(f"Planner initialized: {cfg.get('provider', 'openai')}/{cfg.get('model', 'gpt-4o')} (link={adapter.get('link_mode', 'direct')})")

        # Controller
        if "controller" in self.enabled_modules:
            cfg = get_config("controller")
            adapter = get_adapter_params("controller")
            self._modules["controller"] = Controller(
                provider=cfg.get("provider", "primitives"),
                backend=cfg.get("backend", "primitives"),
                model=cfg.get("model", ""),
                device=cfg.get("device", "cpu"),
                api_key=cfg.get("api_key"),
                temperature=cfg.get("temperature", 0.5),
                action_space=cfg.get("action_space", "trajectory"),
                control_rate_hz=cfg.get("control_rate_hz", 20),
                horizon_steps=cfg.get("horizon_steps", 10),
                frame_convention=cfg.get("frame_convention", "base"),
                safety_limits=cfg.get("safety_limits", {}),
                timeout_s=cfg.get("timeout_s", 6.0),
                max_retries=cfg.get("max_retries", 1),
                **adapter,
            )
            logger.info(f"Controller initialized: {cfg.get('provider', 'primitives')}")

        # Robot
        if "robot" in self.enabled_modules:
            cfg = get_config("robot")
            adapter = get_adapter_params("robot")
            custom_interface = cfg.get("custom_interface", "simulation" if self.simulation else None)
            if not self.simulation and self.robot_ip and not custom_interface:
                logger.warning(
                    "Real robot mode requires CustomRobot. Use set_robot_interface() to configure."
                )

            self._modules["robot"] = Robot(
                custom_interface=custom_interface,
                robot_type=cfg.get("robot_type", "franka"),
                rate_hz=cfg.get("rate_hz", 100),
                timeout_s=cfg.get("timeout_s", 15.0),
                units=cfg.get("units", "SI"),
                frame_convention=cfg.get("frame_convention", "base"),
                estop_policy=cfg.get("estop_policy", "stop_and_report"),
                max_retries=cfg.get("max_retries", 1),
                **adapter,
            )
            logger.info(f"Robot initialized: {cfg.get('robot_type', 'franka')}")

        # Monitor
        if "monitor" in self.enabled_modules:
            cfg = get_config("monitor")
            adapter = get_adapter_params("monitor")
            self._modules["monitor"] = Monitor(
                provider=cfg.get("provider", "stub"),
                model=cfg.get("model", ""),
                api_key=cfg.get("api_key"),
                api_base=cfg.get("api_base"),
                temperature=cfg.get("temperature", 0.0),
                max_tokens=cfg.get("max_tokens", 200),
                image_size=cfg.get("image_size", 224),
                observation_rate_hz=cfg.get("observation_rate_hz", 10.0),
                failure_confidence_threshold=cfg.get("failure_confidence_threshold", 0.7),
                stop_on_consecutive_failures=cfg.get("stop_on_consecutive_failures", 3),
                only_publish_on_failure=cfg.get("only_publish_on_failure", True),
                enable_continuous_mode=cfg.get("enable_continuous_mode", True),
                timeout_s=cfg.get("timeout_s", 4.0),
                max_retries=cfg.get("max_retries", 1),
                **adapter,
            )
            self._modules["monitor"].initialize_client()
            logger.info(f"Monitor initialized: {cfg.get('provider', 'stub')}")

        # Set in_proc links if using in_proc mode with RoboBridge core
        if self.link_mode == "in_proc" and self._core is not None:
            for module_name, module in self._modules.items():
                link = self._core.get_adapter_link(module_name)
                if link is not None:
                    module.set_in_proc_link(link)
                    logger.debug(f"Set in_proc link for {module_name}")

    def _setup_subscriptions(self) -> None:
        """Setup internal subscriptions for state updates."""
        # These would subscribe to relevant topics to update internal state
        pass

    # =========================================================================
    # Environment Connection (for simulation)
    # =========================================================================

    @staticmethod
    def _filter_reachable(detections, arm_reach=1.5):
        """Filter detections to objects within arm reach of robot base.

        Args:
            detections: List of detection dicts
            arm_reach: Maximum reachable distance in meters (base frame)

        Returns:
            Dict of name -> ObjectInfo for reachable objects
        """
        from ..modules.planner import ObjectInfo

        objects = {}
        filtered = 0
        for det in detections:
            obj_info = ObjectInfo.from_detection(det)
            dist = (obj_info.position.x**2 + obj_info.position.y**2 + obj_info.position.z**2) ** 0.5
            if dist <= arm_reach:
                objects[obj_info.name] = obj_info
            else:
                filtered += 1
        if filtered > 0:
            logger.info(f"[REACH] Filtered {filtered} objects beyond {arm_reach}m ({len(objects)} reachable)")
        return objects

    def connect_env(
        self,
        env: Any,
        obs: Dict[str, Any],
        ep_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Connect simulation environment for ground-truth perception.

        Args:
            env: Simulation environment (RoboCasa, MuJoCo, etc.)
            obs: Current observation from environment
            ep_meta: Episode metadata

        Example:
            env = make_robocasa_env()
            obs = env.reset()
            bridge.connect_env(env, obs, ep_meta)
        """
        self._env = env
        self._obs = obs
        self._ep_meta = ep_meta

        # Update perception with environment state
        if "perception" in self._modules:
            self._modules["perception"].set_environment_state(obs, ep_meta, env)

        # Controller only generates Commands; Robot handles execution

        logger.info("Environment connected")

        # Reset VLA policy state for new episode (clears action queue)
        controller = self._modules.get("controller")
        if controller is not None and hasattr(controller, "reset_policy"):
            controller.reset_policy()
        self._env_step_count = 0

    def update_obs(self, obs: Dict[str, Any]) -> None:
        """
        Update current observation.

        Args:
            obs: New observation from environment
        """
        self._obs = obs
        if "perception" in self._modules:
            self._modules["perception"].set_environment_state(obs, self._ep_meta, self._env)
        # Controller receives robot state via topic subscription

    def _generate_template_plan(self, task_name, detections):
        """Generate plan from per-task template config (training-data aligned).

        Uses direction_target from training data for VLA direction state,
        and perception target for convergence detection.
        """
        config = self._template_plan_config.get(task_name)
        if not config:
            return []

        # Find perception target (role=target) for convergence
        perception_target = None
        for det in detections:
            meta = det.get("metadata", {})
            if meta.get("role") == "target":
                pos = det.get("pose", {}).get("position", {})
                if pos:
                    perception_target = {
                        "x": float(pos.get("x", 0)),
                        "y": float(pos.get("y", 0)),
                        "z": float(pos.get("z", 0)),
                    }
                    break

        if perception_target is None:
            logger.warning(f"[TEMPLATE PLAN] No target detection found for {task_name}")
            # Fallback: use first move segment's direction_target as convergence target
            for p in config["primitives"]:
                if p["type"] == "move" and "direction_target" in p:
                    dt = p["direction_target"]
                    perception_target = {"x": dt[0], "y": dt[1], "z": dt[2]}
                    break
            if perception_target is None:
                return []

        # Build primitives
        class _TPrim:
            def __init__(self, d):
                self._d = d
                self.instruction = f"{d.get('primitive_type', '?')}"
            def to_dict(self):
                return self._d

        class _TAction:
            def __init__(self, task_name):
                self.action_type = "template"
                self.target_object = task_name
                self.target_location = None

        class _TPlan:
            def __init__(self, prims):
                self.primitives = prims
                self.parent_action = _TAction(task_name)

        primitives = []
        for prim_cfg in config["primitives"]:
            prim_type = prim_cfg["type"]
            d = {"primitive_type": prim_type}
            if prim_type == "move":
                d["target_position"] = perception_target  # for convergence
                # Priority 1: direction_vector (training motion direction, position-independent)
                dv = prim_cfg.get("direction_vector")
                if dv:
                    d["direction_vector"] = {
                        "x": dv[0],
                        "y": dv[1],  # None → controller computes from perception Y
                        "z": dv[2],
                    }
                else:
                    # Priority 2: direction_target (absolute endpoint, Y from perception)
                    dt = prim_cfg.get("direction_target")
                    if dt:
                        dir_y = perception_target["y"] if perception_target else dt[1]
                        d["direction_target"] = {"x": dt[0], "y": dir_y, "z": dt[2]}
            elif prim_type == "grip":
                if "grip_width" in prim_cfg:
                    d["grip_width"] = prim_cfg["grip_width"]
            primitives.append(_TPrim(d))

        plan = _TPlan(primitives)
        prim_types = [p["type"] for p in config["primitives"]]
        logger.info(
            f"[TEMPLATE PLAN] {task_name}: {prim_types}, "
            f"perception_target=[{perception_target['x']:.3f}, "
            f"{perception_target['y']:.3f}, {perception_target['z']:.3f}]"
        )
        return [plan]

    def step(
        self,
        obs: Dict[str, Any],
        instruction: str,
        execute: bool = False,
        frame_callback: Optional[Callable] = None,
        divergence_threshold: float = 0.05,
        lambda_weight: float = 0.5,
        enable_async_perception: bool = True,
        max_steps: int = 400,
        move_timeout_replan: bool = False,
        diverge_margin: float = 0.40,
        stuck_window: int = 100,
        move_max_steps: int = 250,
        replan_cooldown: int = 150,
        trend_window: int = 30,
    ) -> Dict[str, Any]:
        """
        Run one step of the pipeline: perception -> planning -> (optional) execution.

        When execute=True and enable_async_perception=True, a background perception
        thread continuously updates a buffer. After each primitive, scene divergence
        is checked against the planning-time perception state. If divergence exceeds
        the threshold τ, selective replanning regenerates primitives while preserving
        high-level actions.

        Args:
            obs: Current observation from environment
            instruction: Task instruction
            execute: If True, execute the generated plans in the environment
            frame_callback: Optional callback called after each simulation step (for video recording)
            divergence_threshold: τ — max allowed scene change (meters) before replanning
            lambda_weight: λ — weight for object set difference in divergence formula
            enable_async_perception: Enable background perception thread and divergence check

        Returns:
            Dict containing:
                - "plans": List of primitive plans
                - "detections": List of detected objects
                - "done": Whether all plans are complete
                - "execution_results": List of execution results (if execute=True)
                - "success": Overall execution success (if execute=True)
                - "divergence_replans": Count of divergence-triggered replans (if execute=True)

        Example:
            # Planning only
            result = bridge.step(obs, instruction)

            # Planning + Execution with async perception
            result = bridge.step(obs, instruction, execute=True,
                                 divergence_threshold=0.05)
        """
        self.update_obs(obs)

        # Perception
        detections = []
        if "perception" in self._modules:
            det_objects = self._modules["perception"].process(
                rgb=obs.get("robot0_robotview_image"),
            )
            detections = [d.to_dict() for d in det_objects]

        # Planning — template plan (training-data aligned) takes priority
        plans = []
        if (self._template_plan_config and self._current_task_name
                and self._current_task_name in self._template_plan_config):
            plans = self._generate_template_plan(self._current_task_name, detections)

        if not plans and "planner" in self._modules:
            from ..modules.planner import ObjectInfo

            objects = self._filter_reachable(detections)

            # Images for planner
            images = {
                "robotview": obs.get("robot0_robotview_image"),
                "eye_in_hand": obs.get("robot0_eye_in_hand_image"),
            }

            # Pass robot eef_pos to planner for coordinate-aware planning
            eef_pos = obs.get("robot0_eef_pos")
            plans = self._modules["planner"].process(
                instruction=instruction,
                images=images,
                objects=objects,
                eef_pos=eef_pos,
            ) or []

        result = {
            "plans": plans,
            "detections": detections,
            "done": len(plans) == 0,
        }

        # Execution (optional)
        if execute and plans and self._env is not None:
            self._task_instruction = instruction
            controller = self._modules.get("controller")
            if controller is None:
                logger.error("Controller module not available for execution")
                return result

            # Store frame callback for use in _execute_command_in_env
            self._frame_callback = frame_callback

            # Check if monitor is available and start continuous monitoring
            monitor = self._modules.get("monitor")
            use_monitor = monitor is not None
            # Monitor is used for on-demand VLM failure analysis only
            # (no continuous monitoring thread — heuristics handle detection)

            # --- Async perception setup ---
            use_async = (
                enable_async_perception
                and "perception" in self._modules
                and detections
            )
            buffer = None
            stop_event = None
            perception_thread = None
            p_used = None
            divergence_replans = 0

            if use_async:
                p_used = self._extract_perception_state(detections)
                buffer = PerceptionBuffer()
                stop_event = threading.Event()
                perception_thread = threading.Thread(
                    target=self._perception_worker,
                    args=(buffer, stop_event),
                    daemon=True,
                )
                perception_thread.start()
                logger.debug("Async perception thread started")

            # Execute plans with monitor feedback and replanning
            exec_results = []
            overall_success = True
            failure_info = None
            replan_count = 0
            current_plans = plans
            total_steps_used = 0
            eef_path = []  # current attempt trajectory
            failed_attempts = []  # list of (fail_type, key_points) tuples

            # Convergence detection parameters
            GRIP_MIN_STEPS = 3
            GRIP_MAX_STEPS = 10  # grip only needs to open/close gripper
            MOVE_MIN_STEPS = 10  # minimum steps before checks
            POS_ERROR_THRESHOLD = 0.04  # 4cm — reached target
            STUCK_VELOCITY_THRESH = 0.0005  # 0.5mm/step — relaxed for push tasks
            STUCK_WINDOW = stuck_window  # give VLA plenty of time before declaring stuck
            DIVERGE_MARGIN = diverge_margin  # generous tolerance for push/approach tasks
            MOVE_MAX_STEPS = move_max_steps  # allow more steps per move for better coverage
            REPLAN_COOLDOWN = replan_cooldown  # steps before STUCK/DIVERGE checks activate after replan
            TREND_WINDOW = trend_window  # window size for trend-based DIVERGE detection

            # Replan cooldown and trend-based DIVERGE tracking
            steps_since_replan = 0
            dist_history = []

            # Record initial EEF position for return-to-start on replan
            _initial_eef_pos = None
            _best_eef_pos = None  # tracks closest-to-target position
            if self._obs is not None and self._obs.get("robot0_eef_pos") is not None:
                _initial_eef_pos = self._obs["robot0_eef_pos"].copy()

            try:
                while current_plans:
                    plan_failed = False
                    divergence_replan = False

                    for plan_idx, plan in enumerate(current_plans):
                        action = plan.parent_action

                        # Pre-scan: find next move target for each primitive
                        # so grip primitives can use it for delta_base
                        prim_list = [p.to_dict() for p in plan.primitives]
                        for i, pd in enumerate(prim_list):
                            if pd.get("primitive_type") == "grip" and not pd.get("target_position"):
                                # Find next move primitive's target
                                for j in range(i + 1, len(prim_list)):
                                    next_tp = prim_list[j].get("target_position")
                                    if next_tp:
                                        pd["target_position"] = next_tp
                                        break

                        # Find ultimate target (last move target in the plan)
                        # Direction model was trained with vectors pointing to demo's FINAL position
                        ultimate_target = None
                        for p in reversed(prim_list):
                            if p.get("primitive_type") == "move" and p.get("target_position"):
                                ultimate_target = p["target_position"]
                                break

                        # Set ultimate_target on all primitives
                        if ultimate_target:
                            for p in prim_list:
                                p["ultimate_target"] = ultimate_target

                        for prim_dict in prim_list:
                            prim_type = prim_dict.get("primitive_type", "move")

                            # Reset between primitives:
                            # ROBOBRIDGE_FULL_RESET=1 → full reset
                            # default → soft reset (keep EMA)
                            if os.environ.get("ROBOBRIDGE_FULL_RESET"):
                                controller.reset_policy()
                            else:
                                controller.soft_reset_policy()

                            # Debug: log primitive details including target_position
                            tp = prim_dict.get("target_position")
                            gw = prim_dict.get("grip_width")
                            logger.info(
                                f"Executing primitive: type={prim_type}, "
                                f"target_pos={tp}, grip_width={gw}"
                            )

                            # Progress tracking
                            best_dist = None
                            stuck_count = 0
                            prev_eef_pos = None
                            min_steps = GRIP_MIN_STEPS if prim_type == "grip" else MOVE_MIN_STEPS
                            prim_step = 0
                            move_retry_count = 0  # A: progressive retry counter
                            prim_move_budget = MOVE_MAX_STEPS  # per-primitive budget (can extend)

                            prim_done = False
                            task_succeeded = False
                            move_budget_ok = True

                            while not prim_done and total_steps_used < max_steps and move_budget_ok:
                                # Feed current observation to controller for VLA inference
                                if self._obs is not None:
                                    rgb_dict = None
                                    for cam_key in [
                                        "robot0_agentview_left_image",
                                        "robot0_robotview_image",
                                        "robot0_eye_in_hand_image",
                                    ]:
                                        if cam_key in self._obs and self._obs[cam_key] is not None:
                                            cam_name = cam_key.replace("_image", "")
                                            rgb_dict = {cam_name: self._obs[cam_key]}
                                            break
                                    controller.update_observation(
                                        rgb=rgb_dict,
                                        robot_state=self._get_robot_state_from_obs(self._obs),
                                    )

                                # Use primitive_type as instruction to match
                                # move/grip adapter training (trained with "move"/"grip")
                                prim_instruction = prim_type
                                cmd_or_result = controller.execute_primitive(prim_dict, instruction=prim_instruction)

                                # Controller returns Command; route through Robot module
                                if isinstance(cmd_or_result, Command):
                                    robot = self._modules.get("robot")
                                    if robot is not None:
                                        exec_result = robot.process(cmd_or_result.to_dict())
                                    else:
                                        exec_result = self._execute_command_in_env(cmd_or_result)
                                else:
                                    exec_result = cmd_or_result

                                exec_results.append(exec_result)
                                prim_step += 1
                                total_steps_used += 1

                                if not exec_result.success:
                                    plan_failed = True
                                    failure_info = {
                                        "type": "execution_failure",
                                        "action": action.action_type,
                                        "primitive": prim_dict.get("primitive_type"),
                                        "message": getattr(exec_result, "message", exec_result.state),
                                    }
                                    break

                                # Refresh observation from env after step
                                self._refresh_obs()

                                # Record EEF path (every 5 steps to keep compact)
                                if self._obs is not None and prim_step % 5 == 0:
                                    _p = self._obs.get("robot0_eef_pos")
                                    if _p is not None:
                                        eef_path.append(tuple(np.round(_p, 3)))

                                # Grip primitive: complete after max steps
                                if prim_type == "grip" and prim_step >= GRIP_MAX_STEPS:
                                    logger.info(
                                        f"[GRIP DONE] completed after {prim_step} steps"
                                    )
                                    prim_done = True

                                # Move primitive: enforce per-primitive budget
                                if prim_type == "move" and prim_step >= prim_move_budget:
                                    _timeout_dist = cur_dist if 'cur_dist' in dir() and cur_dist is not None else -1
                                    if move_timeout_replan:
                                        # B: Distance-gated skip — close enough, proceed
                                        if _timeout_dist >= 0 and _timeout_dist < 0.05:
                                            logger.info(
                                                f"[MOVE CLOSE ENOUGH] dist={_timeout_dist:.4f} < 0.05, "
                                                f"proceeding to next primitive"
                                            )
                                            prim_done = True
                                            move_budget_ok = False
                                        # A: Progressive retry — reset VLA, retry once
                                        # If close (<0.2m), grant proximity extension instead of hard retry
                                        elif _timeout_dist >= 0 and _timeout_dist < 0.2 and move_retry_count < 2:
                                            move_retry_count += 1
                                            _ext_steps = MOVE_MAX_STEPS // 2  # 50% extra budget
                                            logger.info(
                                                f"[PROXIMITY EXTEND] dist={_timeout_dist:.4f} < 0.2m, "
                                                f"extending budget by {_ext_steps} steps "
                                                f"(attempt {move_retry_count})"
                                            )
                                            # Don't reset VLA — keep momentum
                                            prim_move_budget += _ext_steps
                                            move_budget_ok = True
                                            continue
                                        elif move_retry_count < 1:
                                            move_retry_count += 1
                                            logger.info(
                                                f"[MOVE RETRY] dist={_timeout_dist:.4f}, resetting VLA "
                                                f"state and retrying (attempt {move_retry_count})"
                                            )
                                            controller.reset_policy()
                                            prim_step = 0
                                            stuck_count = 0
                                            prev_eef_pos = None
                                            best_dist = None
                                            dist_history = []
                                            steps_since_replan = 0
                                            move_budget_ok = True
                                            continue  # restart primitive while loop
                                        else:
                                            logger.info(
                                                f"[MOVE TIMEOUT] move primitive reached {prim_step} steps "
                                                f"(retry exhausted), triggering replan"
                                            )
                                            plan_failed = True
                                            failure_info = {"type": "move_timeout", "distance": _timeout_dist}
                                            prim_done = True
                                            move_budget_ok = False
                                    else:
                                        logger.info(
                                            f"[MOVE BUDGET] move primitive reached {prim_step} steps "
                                            f"(MOVE_MAX_STEPS={MOVE_MAX_STEPS}), treating as converged"
                                        )
                                        prim_done = True
                                        move_budget_ok = False

                                # Update move_budget_ok for while-condition check
                                if not prim_done:
                                    move_budget_ok = (prim_type != "move") or (prim_step < prim_move_budget)

                                # Task success check (every step, before STUCK/DIVERGE)
                                if self._env is not None:
                                    try:
                                        if self._env._check_success():
                                            logger.info(
                                                f"Task succeeded at step {total_steps_used} "
                                                f"(during {prim_type} primitive)"
                                            )
                                            prim_done = True
                                            task_succeeded = True
                                    except Exception:
                                        pass

                                # Progress detection (skip if already succeeded)
                                if not prim_done and prim_step >= min_steps and self._obs is not None:
                                    cur_pos = self._obs.get("robot0_eef_pos")
                                    if cur_pos is not None and tp is not None:
                                        target_base = np.array([tp["x"], tp["y"], tp["z"]])
                                        base_pos = self._obs.get("robot0_base_pos")
                                        base_quat = self._obs.get("robot0_base_quat")
                                        if base_pos is not None and base_quat is not None:
                                            from scipy.spatial.transform import Rotation as _Rot
                                            _rot_inv = _Rot.from_quat(base_quat).inv()
                                            eef_base = _rot_inv.apply(cur_pos - base_pos)
                                        else:
                                            eef_base = cur_pos

                                        cur_dist = float(np.linalg.norm(eef_base - target_base))

                                        # For push tasks (close drawer), use tighter threshold
                                        # to avoid premature termination — keep pushing past target
                                        task_instruction = instruction.lower() if instruction else ""
                                        is_push_task = any(
                                            kw in task_instruction
                                            for kw in ["close", "push"]
                                        )
                                        effective_pos_threshold = 0.02 if is_push_task else POS_ERROR_THRESHOLD

                                        # Reached target?
                                        if cur_dist < effective_pos_threshold:
                                            logger.info(
                                                f"[REACHED] {prim_type} reached target after {prim_step} steps "
                                                f"(dist={cur_dist:.4f}, threshold={effective_pos_threshold:.3f})"
                                            )
                                            prim_done = True

                                        if not prim_done:
                                            # Track best distance and position
                                            if best_dist is None or cur_dist < best_dist:
                                                best_dist = cur_dist
                                                _best_eef_pos = cur_pos.copy()

                                            # Always track distance history for trend detection
                                            dist_history.append(cur_dist)

                                            if use_monitor and steps_since_replan >= REPLAN_COOLDOWN:
                                                # STUCK: EEF not moving physically
                                                if prev_eef_pos is not None:
                                                    movement = float(np.linalg.norm(cur_pos - prev_eef_pos))
                                                    if movement < STUCK_VELOCITY_THRESH:
                                                        stuck_count += 1
                                                    else:
                                                        stuck_count = 0
                                                    if stuck_count >= STUCK_WINDOW:
                                                        logger.warning(
                                                            f"[STUCK] {prim_type} physically blocked for "
                                                            f"{stuck_count} steps (dist={cur_dist:.3f})"
                                                        )
                                                        plan_failed = True
                                                        failure_info = {"type": "stuck", "distance": cur_dist}
                                                        prim_done = True

                                                # DIVERGE: trend-based (replaces instant best_dist check)
                                                if not prim_done and len(dist_history) >= TREND_WINDOW * 2:
                                                    recent_avg = float(np.mean(dist_history[-TREND_WINDOW:]))
                                                    older_avg = float(np.mean(dist_history[-TREND_WINDOW*2:-TREND_WINDOW]))
                                                    if recent_avg > older_avg + DIVERGE_MARGIN:
                                                        logger.warning(
                                                            f"[DIVERGE-TREND] {prim_type} trending away: "
                                                            f"recent={recent_avg:.3f} vs older={older_avg:.3f} "
                                                            f"(margin={DIVERGE_MARGIN:.2f})"
                                                        )
                                                        plan_failed = True
                                                        failure_info = {"type": "diverge", "distance": cur_dist, "best": best_dist}
                                                        prim_done = True
                                            # No stuck_count accumulation during cooldown (conservative)

                                            steps_since_replan += 1

                                    # Update prev position for velocity calc
                                    if cur_pos is not None:
                                        prev_eef_pos = cur_pos.copy()

                                # Update perception env state for async thread
                                if use_async and self._obs is not None:
                                    self._modules["perception"].set_environment_state(
                                        self._obs, self._ep_meta, self._env
                                    )

                                if prim_done:
                                    break

                            # Early exit if task already succeeded, budget exhausted, or plan failed
                            if plan_failed or task_succeeded or total_steps_used >= max_steps:
                                if task_succeeded:
                                    overall_success = True
                                break

                            # Monitor check after each primitive (not every step)
                            # v4: Disabled per-primitive VLM failure check.
                            # Reason: high false positive rate triggers unnecessary replans.
                            # The GT _check_success() already handles early success detection.
                            # VLM failure analysis (path D) is kept for move_timeout recovery.
                            if False and not plan_failed and use_monitor:
                                current_obs = self._obs
                                if current_obs is not None:
                                    rgb = current_obs.get("robot0_robotview_image")

                                    plan_context = {
                                        "skill": action.action_type,
                                        "target_object": action.target_object,
                                        "primitive": prim_dict.get("primitive_type"),
                                        "instruction": instruction,
                                    }

                                    feedback = monitor.process(rgb=rgb, plan=plan_context)

                                    if feedback and not feedback.success:
                                        if feedback.confidence >= monitor.monitor_config.failure_confidence_threshold:
                                            if "robot" in self._modules:
                                                self._modules["robot"]._stop_robot()
                                                logger.info("Robot stopped due to high-confidence failure")

                                            analysis = monitor._analyze_failure(rgb, plan_context)
                                            recovery_target = analysis.recovery_target

                                            logger.warning(
                                                f"Monitor detected failure: confidence={feedback.confidence:.2f}, "
                                                f"recovery_target={recovery_target}"
                                            )
                                            plan_failed = True
                                            failure_info = {
                                                "type": "monitor_failure",
                                                "confidence": feedback.confidence,
                                                "action": action.action_type,
                                                "primitive": prim_dict.get("primitive_type"),
                                                "recovery_target": recovery_target,
                                            }

                            # --- Divergence check after each primitive ---
                            if not plan_failed and use_async and buffer is not None and p_used is not None:
                                p_latest = buffer.get()
                                if p_latest is not None:
                                    delta = scene_divergence(p_used, p_latest, lambda_weight)
                                    if delta > divergence_threshold:
                                        logger.info(
                                            f"Scene divergence {delta:.4f} > τ={divergence_threshold}, "
                                            f"triggering selective replan"
                                        )
                                        remaining_actions = [
                                            p.parent_action for p in current_plans[plan_idx:]
                                        ]

                                        updated_objects = self._filter_reachable(p_latest.detections)

                                        current_obs = self._obs
                                        current_images = {
                                            "robotview": current_obs.get("robot0_robotview_image") if current_obs else None,
                                            "eye_in_hand": current_obs.get("robot0_eye_in_hand_image") if current_obs else None,
                                        }

                                        new_plans = self._modules["planner"].replan_primitives(
                                            actions=remaining_actions,
                                            instruction=instruction,
                                            images=current_images,
                                            objects=updated_objects,
                                        )

                                        if new_plans:
                                            p_used = p_latest
                                            current_plans = new_plans
                                            divergence_replans += 1
                                            divergence_replan = True
                                            logger.info(
                                                f"Selective replan #{divergence_replans}: "
                                                f"{len(new_plans)} plans regenerated"
                                            )
                                            break  # Break primitive loop, restart with new plans

                        if plan_failed or divergence_replan or task_succeeded or total_steps_used >= max_steps:
                            break

                    # Task already succeeded — exit
                    if task_succeeded:
                        overall_success = True
                        break

                    # Budget exhausted
                    if total_steps_used >= max_steps:
                        break

                    # If divergence triggered replan, restart execution with new plans
                    if divergence_replan:
                        continue

                    if plan_failed:
                        replan_count += 1
                        steps_since_replan = 0  # reset cooldown
                        dist_history = []       # reset history
                        best_dist = None        # reset best distance

                        # VLM failure analysis: get reason + recovery_target
                        vlm_reason = ""
                        recovery_target = failure_info.get("recovery_target") if failure_info else None
                        if use_monitor and self._obs is not None:
                            _rgb = None
                            for cam_key in ["robot0_agentview_left_image", "agentview_image", "robot0_robotview_image"]:
                                if cam_key in self._obs and self._obs[cam_key] is not None:
                                    _rgb = self._obs[cam_key]
                                    break
                            if _rgb is not None:
                                monitor._current_plan = {
                                    "skill": action.action_type if action else "unknown",
                                    "target_object": action.target_object if action else "unknown",
                                    "instruction": instruction,
                                }
                                try:
                                    analysis = monitor._analyze_failure(_rgb, monitor._current_plan)
                                    if analysis.recovery_target:
                                        recovery_target = analysis.recovery_target
                                    vlm_reason = analysis.metadata.get("reason", "")
                                    logger.info(
                                        f"[VLM-ANALYSIS] recovery={recovery_target}, "
                                        f"reason={vlm_reason}"
                                    )
                                except Exception as e:
                                    logger.warning(f"VLM failure analysis failed: {e}")

                        fail_type = failure_info.get("type", "unknown") if failure_info else "unknown"
                        logger.info(
                            f"Recovery (attempt {replan_count}), "
                            f"heuristic={fail_type}, vlm_reason={vlm_reason}, "
                            f"strategy: {recovery_target or 'full_replan'}"
                        )

                        current_obs = self._obs

                        # Skip controller retry — it never works (same VLA, same plan).
                        # v2: For STUCK/DIVERGE, escalate to planning (not perception)
                        #     to avoid costly re-perception when position barely changed.
                        #     For other failures, escalate to perception as before.
                        if recovery_target in ("retry", "controller"):
                            if fail_type in ("stuck", "diverge"):
                                logger.info(
                                    f"Skipping {recovery_target} retry for {fail_type}, "
                                    f"escalating to planning (not perception)"
                                )
                                recovery_target = "planning"
                            else:
                                logger.info(
                                    f"Skipping {recovery_target} retry, "
                                    f"escalating to full replan"
                                )
                                recovery_target = "perception"

                        # Move back to best position (closest to target) before replanning
                        # v5: Return-to-position is CRITICAL for move_timeout replans.
                        # Only skip for stuck/diverge (robot truly blocked, position won't help).
                        _skip_return = fail_type in ("stuck", "diverge")
                        _return_target = _best_eef_pos if _best_eef_pos is not None else _initial_eef_pos
                        if _return_target is not None and not _skip_return:
                            label = "best" if _best_eef_pos is not None else "start"
                            logger.info(f"Returning to {label} position...")
                            cb = getattr(self, "_frame_callback", None)
                            return_steps = self._return_to_position(
                                _return_target, max_steps=100,
                                frame_callback=cb,
                            )
                            if return_steps == -1:
                                # Task succeeded during return movement
                                overall_success = True
                                task_succeeded = True
                                break
                            total_steps_used += return_steps
                            controller.reset_policy()
                            # Update obs and perception after returning
                            current_obs = self._obs
                        elif _skip_return:
                            logger.info(
                                f"[v2] Skipping return-to-position for {fail_type}, "
                                f"replanning from current position"
                            )

                        # C: Save failed path waypoints for replan context
                        failed_eef_path = list(eef_path) if eef_path else []
                        eef_path = []
                        failure_reason = ""
                        if failure_info:
                            fail_type = failure_info.get("type", "")
                            _eef_cur = self._obs.get("robot0_eef_pos") if self._obs else None
                            # Convert world frame → base frame for replan prompt
                            # Planner expects base frame coords (0~1m range)
                            _base_pos = self._obs.get("robot0_base_pos") if self._obs else None
                            _base_quat = self._obs.get("robot0_base_quat") if self._obs else None
                            _replan_rot_inv = None
                            if _base_pos is not None and _base_quat is not None:
                                from scipy.spatial.transform import Rotation as _Rot
                                _replan_rot_inv = _Rot.from_quat(_base_quat).inv()
                            def _to_base(world_pos):
                                """Convert world frame position to base frame."""
                                if _replan_rot_inv is not None and _base_pos is not None:
                                    return _replan_rot_inv.apply(world_pos - _base_pos)
                                return world_pos
                            if _eef_cur is not None:
                                _eef_base = _to_base(_eef_cur)
                            else:
                                _eef_base = None
                            if fail_type == "stuck":
                                if _eef_base is not None:
                                    failure_reason = (
                                        f"Robot got stuck near position "
                                        f"({round(float(_eef_base[0]),3)}, "
                                        f"{round(float(_eef_base[1]),3)}, "
                                        f"{round(float(_eef_base[2]),3)}). "
                                    )
                                else:
                                    failure_reason = "Robot got stuck. "
                            elif fail_type == "move_timeout":
                                if _eef_base is not None:
                                    failure_reason = (
                                        f"Robot timed out at position "
                                        f"({round(float(_eef_base[0]),3)}, "
                                        f"{round(float(_eef_base[1]),3)}, "
                                        f"{round(float(_eef_base[2]),3)}). "
                                    )
                                else:
                                    failure_reason = "Robot timed out reaching target. "
                            elif fail_type == "diverge":
                                failure_reason = "Robot diverged from target. "
                            elif fail_type == "monitor_failure":
                                failure_reason = "Execution monitor detected failure. "
                        # Append failed waypoints (converted to base frame) to help planner
                        if failed_eef_path:
                            _sampled = failed_eef_path[::max(1, len(failed_eef_path) // 5)][-5:]
                            _sampled_base = [_to_base(np.array(p)) for p in _sampled]
                            _wp_str = ", ".join(
                                f"({round(float(p[0]),3)},{round(float(p[1]),3)},{round(float(p[2]),3)})" for p in _sampled_base
                            )
                            failure_reason += f"Failed path: [{_wp_str}]. Try a different approach angle."
                        replan_instruction = (
                            f"{instruction}. {failure_reason}"
                            if failure_reason
                            else instruction
                        )
                        logger.info(f"[REPLAN PROMPT] {replan_instruction}")

                        if recovery_target == "planning":
                            # Planning: skip perception, replan with existing detections
                            logger.info("Replanning (keeping current perception)")
                            if "planner" in self._modules and current_obs is not None:
                                objects = self._filter_reachable(detections)

                                images = {
                                    "robotview": current_obs.get("robot0_robotview_image"),
                                    "eye_in_hand": current_obs.get("robot0_eye_in_hand_image"),
                                }

                                _eef = current_obs.get("robot0_eef_pos")
                                current_plans = self._modules["planner"].process(
                                    instruction=replan_instruction,
                                    images=images,
                                    objects=objects,
                                    eef_pos=_eef,
                                ) or []

                                if not current_plans:
                                    logger.warning("Replanning returned empty plan")
                                    overall_success = False
                                    break
                            else:
                                overall_success = False
                                break

                        else:
                            # "perception" or None fallback: full replan from perception
                            logger.info("Full replan from perception")
                            if "perception" in self._modules and current_obs is not None:
                                self._modules["perception"].set_environment_state(
                                    current_obs, self._ep_meta, self._env
                                )
                                det_objects = self._modules["perception"].process(
                                    rgb=current_obs.get("robot0_robotview_image"),
                                )
                                detections = [d.to_dict() for d in det_objects]

                                # Update p_used for divergence tracking
                                if use_async:
                                    p_used = self._extract_perception_state(detections)

                            if "planner" in self._modules and current_obs is not None:
                                from ..modules.planner import ObjectInfo

                                objects = self._filter_reachable(detections)

                                images = {
                                    "robotview": current_obs.get("robot0_robotview_image"),
                                    "eye_in_hand": current_obs.get("robot0_eye_in_hand_image"),
                                }

                                _eef = current_obs.get("robot0_eef_pos")
                                current_plans = self._modules["planner"].process(
                                    instruction=replan_instruction,
                                    images=images,
                                    objects=objects,
                                    eef_pos=_eef,
                                ) or []

                                if not current_plans:
                                    logger.warning("Replanning returned empty plan")
                                    overall_success = False
                                    break
                            else:
                                overall_success = False
                                break
                    else:
                        # All plans executed successfully
                        overall_success = True
                        break

            finally:
                # --- Stop async perception thread ---
                if stop_event is not None:
                    stop_event.set()
                if perception_thread is not None:
                    perception_thread.join(timeout=2.0)
                    logger.debug("Async perception thread stopped")

            result["execution_results"] = exec_results
            result["total_steps_used"] = total_steps_used  # includes return-to-start
            result["success"] = overall_success
            result["replan_count"] = replan_count
            result["divergence_replans"] = divergence_replans
            if failure_info and not overall_success:
                result["failure_info"] = failure_info
            result["final_obs"] = self._obs

        return result

    # =========================================================================
    # Async Perception Helpers
    # =========================================================================

    def _perception_worker(
        self,
        buffer: PerceptionBuffer,
        stop_event: threading.Event,
    ) -> None:
        """
        Perception thread: continuously process latest obs into buffer.

        Runs autonomously at inference speed, independent of the execution
        thread. Buffer B stores only the newest result (single-result buffer).
        """
        while not stop_event.is_set():
            obs = self._obs
            if obs is None:
                # No observation yet; brief sleep to avoid busy-wait
                stop_event.wait(timeout=0.1)
                continue
            rgb = obs.get("robot0_robotview_image")
            if rgb is None:
                stop_event.wait(timeout=0.1)
                continue
            try:
                det_objects = self._modules["perception"].process(rgb=rgb)
                detections = [d.to_dict() for d in det_objects]
                state = self._extract_perception_state(detections)
                buffer.update(state)
            except Exception as e:
                logger.debug(f"Perception worker error: {e}")
                stop_event.wait(timeout=0.1)

    def _extract_perception_state(self, detections: list) -> PerceptionState:
        """Convert detection list to PerceptionState (object -> 3D position map)."""
        objects = {}
        for det in detections:
            pose = det.get("pose", {})
            pos = pose.get("position", {})
            objects[det["name"]] = np.array([
                pos.get("x", 0.0),
                pos.get("y", 0.0),
                pos.get("z", 0.0),
            ])
        return PerceptionState(
            objects=objects,
            detections=detections,
            timestamp=time.time(),
        )

    def _refresh_obs(self) -> None:
        """Get fresh observation from environment after primitive execution."""
        if self._env is not None:
            try:
                obs = self._env._get_observations()
                self._obs = obs
            except Exception:
                pass  # Keep existing obs

    def _get_robot_state_from_obs(self, obs: Dict) -> Dict:
        """Extract robot state dict from RoboCasa observation for VLA inference."""
        result = {
            "ee_pose": {
                "position": {
                    "x": float(obs["robot0_eef_pos"][0]),
                    "y": float(obs["robot0_eef_pos"][1]),
                    "z": float(obs["robot0_eef_pos"][2]),
                },
                "orientation": {
                    "x": float(obs["robot0_eef_quat"][0]),
                    "y": float(obs["robot0_eef_quat"][1]),
                    "z": float(obs["robot0_eef_quat"][2]),
                    "w": float(obs["robot0_eef_quat"][3]),
                },
            },
            "gripper_qpos": list(obs.get("robot0_gripper_qpos", [0.04, 0.04])),
        }
        # Robot base pose for coordinate frame conversion (robot-base → world)
        if "robot0_base_pos" in obs:
            result["base_pos"] = [float(v) for v in obs["robot0_base_pos"]]
        if "robot0_base_quat" in obs:
            result["base_quat"] = [float(v) for v in obs["robot0_base_quat"]]
        # GT base-frame EEF pos (matches HDF5 training data, avoids manual transform drift)
        if "robot0_base_to_eef_pos" in obs:
            result["robot0_base_to_eef_pos"] = [float(v) for v in obs["robot0_base_to_eef_pos"]]
        return result

    def _execute_command_in_env(self, command: Command) -> Any:
        """Execute Command directly in env when Robot module is not available."""
        from ..modules.robot.types import ExecutionResult as RobotExecutionResult

        if self._env is None:
            return RobotExecutionResult(
                success=False, command_id=command.command_id, state="error",
            )

        env_action_dim = self._env.action_dim if hasattr(self._env, 'action_dim') else 7
        cb = getattr(self, "_frame_callback", None)

        points = command.points if command.points else [None]
        for point in points:
            action = np.zeros(env_action_dim)
            if point is not None:
                positions = point.positions
                if positions:
                    action[:min(3, len(positions))] = np.clip(positions[:3], -1.0, 1.0)
                if point.rotations:
                    action[3:min(6, 3 + len(point.rotations))] = np.clip(point.rotations[:3], -1.0, 1.0)

            # Gripper: RoboCasa PandaMobile 12D layout:
            #   [0:6] arm, [6] gripper, [7:11] base(zeros), [11] gripper copy
            if command.gripper_command:
                # Use raw gripper value from VLA if available (matches direct mode)
                raw_grip = command.gripper_command.get("raw_value")
                if raw_grip is not None:
                    gripper_val = np.clip(float(raw_grip), -1.0, 1.0)
                else:
                    gripper_val = -1.0 if command.gripper_command.get("action") == "open" else 1.0
                if env_action_dim >= 12:
                    action[6] = gripper_val
                    action[11] = gripper_val
                else:
                    action[env_action_dim - 1] = gripper_val

            # Debug: log env action occasionally
            _env_step_count = getattr(self, '_env_step_count', 0) + 1
            self._env_step_count = _env_step_count
            if _env_step_count <= 3 or _env_step_count % 50 == 0:
                logger.info(
                    f"ENV action [step {_env_step_count}]: "
                    f"pos={[round(float(v), 4) for v in action[:3]]}, "
                    f"rot={[round(float(v), 4) for v in action[3:6]]}, "
                    f"grip={round(float(action[6]), 4) if len(action) >= 7 else 'N/A'}"
                )
            obs, reward, done, info = self._env.step(action)
            self._obs = obs

            # Stop if env signals episode is done (horizon reached)
            if done:
                return RobotExecutionResult(
                    success=True, command_id=command.command_id, state="done",
                )

            if cb is not None:
                cb(obs)

        return RobotExecutionResult(
            success=True, command_id=command.command_id, state="idle",
        )

    # =========================================================================
    # Return-to-position helper (used before replanning)
    # =========================================================================

    def _return_to_position(
        self, target_pos: np.ndarray, max_steps: int = 30,
        threshold: float = 0.01, gain: float = 5.0,
        frame_callback=None,
    ) -> int:
        """Move robot EEF back to target_pos using simple P-control.

        Uses OSC_POSE delta actions in robot base frame.
        target_pos is in world frame; converted to base frame for OSC_POSE.
        Returns number of env steps used.  Returns -1 if task succeeded during return.
        """
        steps_used = 0
        for _ in range(max_steps):
            self._refresh_obs()
            if self._obs is None:
                break
            cur_pos = self._obs.get("robot0_eef_pos")
            if cur_pos is None:
                break

            err_world = target_pos - cur_pos
            dist = float(np.linalg.norm(err_world))
            if dist < threshold:
                break

            # Convert world-frame error to base frame for OSC_POSE
            base_quat = self._obs.get("robot0_base_quat")
            if base_quat is not None:
                from scipy.spatial.transform import Rotation as _Rot
                rot_inv = _Rot.from_quat(base_quat).inv()
                err_base = rot_inv.apply(err_world)
            else:
                err_base = err_world

            # P-control: scale error, clip to [-1, 1] for OSC_POSE
            delta = np.clip(err_base * gain, -1.0, 1.0)

            env_action_dim = self._env.action_dim if hasattr(self._env, 'action_dim') else 7
            action = np.zeros(env_action_dim)
            action[:3] = delta
            # Keep gripper unchanged (no rotation, no gripper change)

            obs, reward, done, info = self._env.step(action)
            self._obs = obs
            steps_used += 1

            if frame_callback is not None:
                frame_callback(obs)

            # Check task success during return movement
            if self._env is not None:
                try:
                    if self._env._check_success():
                        logger.info(
                            f"Task succeeded during return-to-start at step {steps_used}"
                        )
                        return -1  # signal success
                except Exception:
                    pass

            if done:
                break

        if steps_used > 0:
            logger.info(
                f"Return-to-start: {steps_used} steps, "
                f"final dist={float(np.linalg.norm(target_pos - self._obs.get('robot0_eef_pos', target_pos))):.4f}m"
            )
        return steps_used

    # =========================================================================
    # Shutdown
    # =========================================================================

    def shutdown(self, wait_for_completion: bool = True, timeout_s: float = 10.0) -> None:
        """
        Shutdown RoboBridge and all modules.

        Args:
            wait_for_completion: Wait for current execution to complete
            timeout_s: Timeout for waiting
        """
        if not self._running:
            return

        logger.info("Shutting down RoboBridge...")

        if wait_for_completion and self.is_busy():
            self._wait_for_idle(timeout_s)

        # Stop all modules
        for name, module in self._modules.items():
            try:
                module.stop()
                logger.info(f"Module {name} stopped")
            except Exception as e:
                logger.error(f"Error stopping module {name}: {e}")

        # Stop core
        if self._core:
            self._core.shutdown()

        self._running = False
        RoboBridgeClient._instance = None
        logger.info("RoboBridge shutdown complete")

    def reset(
        self,
        home_position: Optional[List[float]] = None,
        open_gripper: bool = True,
    ) -> ExecutionResult:
        """
        Reset robot to home position.

        Args:
            home_position: Custom home joint positions (optional)
            open_gripper: Open gripper after reset

        Returns:
            ExecutionResult
        """
        default_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        positions = home_position or default_home

        result = self.move(joint_positions=positions)
        if result.success and open_gripper:
            result = self.open_gripper()

        return result

    # =========================================================================
    # Natural Language Commands
    # =========================================================================

    def execute(
        self,
        instruction: str,
        wait: bool = True,
        timeout_s: float = 60.0,
        retry_on_failure: bool = True,
    ) -> ExecutionResult:
        """
        Execute a natural language instruction.

        Args:
            instruction: Natural language command (e.g., "Pick up the red cup")
            wait: Wait for execution to complete
            timeout_s: Timeout for execution
            retry_on_failure: Retry with re-planning on failure

        Returns:
            ExecutionResult with success status and message

        Example:
            result = robot.execute("Pick up the red cup and place it on the table")
            if result.success:
                print("Task completed!")
        """
        if not self._running:
            return ExecutionResult(success=False, message="RoboBridge not running")

        start_time = time.time()
        command_id = str(uuid.uuid4())[:8]

        logger.info(f"Executing instruction: {instruction}")

        # Publish instruction to planner
        self._publish_instruction(instruction, command_id)

        if not wait:
            return ExecutionResult(
                success=True,
                message="Instruction sent",
                data={"command_id": command_id},
            )

        # Wait for completion
        result = self._wait_for_execution(command_id, timeout_s)
        result.execution_time_s = time.time() - start_time

        # Retry on failure if enabled
        if not result.success and retry_on_failure:
            logger.info("Execution failed, retrying...")
            result = self._wait_for_execution(command_id, timeout_s)
            result.execution_time_s = time.time() - start_time

        return result

    def plan(self, instruction: str, return_steps: bool = True) -> ExecutionResult:
        """
        Generate a plan without executing it.

        Args:
            instruction: Natural language instruction
            return_steps: Include step details in result

        Returns:
            ExecutionResult with plan in data field

        Example:
            result = robot.plan("Pick up the red cup")
            for step in result.data["steps"]:
                print(f"{step['skill']}: {step['target']}")
        """
        if not self._running:
            return ExecutionResult(success=False, message="RoboBridge not running")

        if "planner" not in self._modules:
            return ExecutionResult(
                success=False,
                message="Planner module not available",
            )

        try:
            world_state = None
            if self._latest_objects:
                world_state = {
                    "detections": [
                        {
                            "name": obj.name,
                            "confidence": obj.confidence,
                            "position": obj.position,
                        }
                        for obj in self._latest_objects
                    ]
                }

            # Convert world_state detections to ObjectInfo for planner
            objects = {}
            if world_state and "detections" in world_state:
                objects = self._filter_reachable(world_state["detections"])

            plan = self._modules["planner"].process(
                instruction=instruction,
                objects=objects,
            )

            if plan is None:
                return ExecutionResult(
                    success=False,
                    message="Failed to generate plan",
                    data={"steps": []},
                )

            plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
            steps = plan_dict.get("steps", [])
            self._current_plan = plan_dict

            return ExecutionResult(
                success=True,
                message=f"Plan generated with {len(steps)} steps",
                data={
                    "plan_id": plan_dict.get("plan_id", ""),
                    "instruction": instruction,
                    "steps": steps if return_steps else [],
                    "step_count": len(steps),
                },
            )

        except Exception as e:
            logger.error(f"Planning error: {e}")
            return ExecutionResult(
                success=False,
                message=f"Planning failed: {e}",
                data={"steps": []},
            )

    def _publish_instruction(self, instruction: str, command_id: str) -> None:
        """Publish instruction to the system."""
        if self._core:
            self._core.publish(
                "/robobridge/instruction",
                {
                    "data": json.dumps(
                        {
                            "instruction": instruction,
                            "command_id": command_id,
                        }
                    )
                },
            )

    def _wait_for_execution(self, command_id: str, timeout_s: float) -> ExecutionResult:
        """Wait for execution to complete."""
        start = time.time()
        while time.time() - start < timeout_s:
            if not self.is_busy():
                return ExecutionResult(success=True, message="Execution complete")
            time.sleep(0.1)

        return ExecutionResult(success=False, message="Execution timeout")

    def _wait_for_idle(self, timeout_s: float) -> bool:
        """Wait for robot to become idle."""
        start = time.time()
        while time.time() - start < timeout_s:
            if not self.is_busy():
                return True
            time.sleep(0.1)
        return False

    # =========================================================================
    # Direct Skill Calls
    # =========================================================================

    def pick(
        self,
        object_name: str,
        grasp_width: float = 0.08,
        force: float = 40.0,
        approach_height: float = 0.1,
    ) -> ExecutionResult:
        """
        Pick up an object.

        Args:
            object_name: Name of object to pick
            grasp_width: Gripper width for grasping (meters)
            force: Grasp force (N)
            approach_height: Height above object for approach (meters)

        Returns:
            ExecutionResult

        Example:
            robot.pick("red_cup", grasp_width=0.05, force=30)
        """
        logger.info(f"Picking object: {object_name}")

        # Get object pose
        obj_pose = self.get_object_pose(object_name)
        if obj_pose is None:
            return ExecutionResult(
                success=False,
                message=f"Object '{object_name}' not found",
            )

        # Execute pick sequence
        # 1. Move above object
        approach_pos = obj_pose.copy()
        approach_pos["z"] = approach_pos.get("z", 0) + approach_height
        self.move(position=approach_pos)

        # 2. Open gripper
        self.open_gripper(width=grasp_width)

        # 3. Move down to object
        self.move(position=obj_pose)

        # 4. Close gripper
        result = self.close_gripper(width=grasp_width, force=force)

        # 5. Lift
        self.move(position=approach_pos)

        return result

    def place(
        self,
        object_name: str,
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        release_width: float = 0.08,
    ) -> ExecutionResult:
        """
        Place an object at a position.

        Args:
            object_name: Name of object being placed
            position: Target [x, y, z] position (meters)
            orientation: Target orientation [x, y, z, w] quaternion
            release_width: Gripper width for release (meters)

        Returns:
            ExecutionResult

        Example:
            robot.place("red_cup", position=[0.4, 0.2, 0.05])
        """
        logger.info(f"Placing object: {object_name} at {position}")

        if position is None:
            return ExecutionResult(
                success=False,
                message="Position required for place",
            )

        pos_dict = {"x": position[0], "y": position[1], "z": position[2]}

        # Move above target
        approach_pos = pos_dict.copy()
        approach_pos["z"] = approach_pos["z"] + 0.1
        self.move(position=approach_pos)

        # Move to target
        self.move(position=pos_dict)

        # Release
        result = self.open_gripper(width=release_width)

        # Lift
        self.move(position=approach_pos)

        return result

    def move(
        self,
        position: Optional[Union[List[float], Dict[str, float]]] = None,
        orientation: Optional[List[float]] = None,
        velocity: float = 0.1,
        frame: str = "base",
        joint_positions: Optional[List[float]] = None,
    ) -> ExecutionResult:
        """
        Move robot to a position.

        Args:
            position: Target [x, y, z] or {"x": x, "y": y, "z": z} (meters)
            orientation: Target orientation [x, y, z, w] quaternion
            velocity: Movement velocity (m/s)
            frame: Reference frame ("base" or "ee")
            joint_positions: Direct joint positions (radians), overrides position

        Returns:
            ExecutionResult

        Example:
            robot.move(position=[0.4, 0.2, 0.3])
            robot.move(joint_positions=[0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        """
        logger.info(f"Moving to position: {position}, joints: {joint_positions}")

        if joint_positions is not None:
            # Direct joint move
            command = {
                "command_type": "joint",
                "points": [{"positions": joint_positions, "time_from_start": 1.0}],
            }
        elif position is not None:
            # Cartesian move
            if isinstance(position, list):
                pos = {"x": position[0], "y": position[1], "z": position[2]}
            else:
                pos = position

            command = {
                "command_type": "cartesian",
                "target_pose": {
                    "position": pos,
                    "orientation": orientation or {"x": 1, "y": 0, "z": 0, "w": 0},
                },
                "velocity": velocity,
            }
        else:
            return ExecutionResult(
                success=False,
                message="Either position or joint_positions required",
            )

        return self._execute_robot_command(command)

    def open_gripper(self, width: float = 0.08, speed: float = 0.1) -> ExecutionResult:
        """
        Open the gripper.

        Args:
            width: Target opening width (meters)
            speed: Opening speed (m/s)

        Returns:
            ExecutionResult

        Example:
            robot.open_gripper(width=0.08)
        """
        logger.info(f"Opening gripper to width: {width}")

        command = {
            "gripper_command": {
                "action": "open",
                "width": width,
                "speed": speed,
            }
        }
        return self._execute_robot_command(command)

    def close_gripper(
        self,
        width: float = 0.0,
        force: float = 40.0,
        speed: float = 0.1,
    ) -> ExecutionResult:
        """
        Close the gripper.

        Args:
            width: Target closing width (meters)
            force: Grasp force (N)
            speed: Closing speed (m/s)

        Returns:
            ExecutionResult

        Example:
            robot.close_gripper(force=30)
        """
        logger.info(f"Closing gripper with force: {force}")

        command = {
            "gripper_command": {
                "action": "close",
                "width": width,
                "force": force,
                "speed": speed,
            }
        }
        return self._execute_robot_command(command)

    def _execute_robot_command(self, command: Dict) -> ExecutionResult:
        """Execute a robot command through robot interface."""
        if "robot" not in self._modules:
            return ExecutionResult(
                success=False,
                message="Robot module not available",
            )

        command["command_id"] = str(uuid.uuid4())[:8]

        try:
            result = self._modules["robot"].process(command)
            return ExecutionResult(
                success=result.success,
                message="Success" if result.success else "Failed",
                data=result.to_dict() if hasattr(result, "to_dict") else {},
                execution_time_s=getattr(result, "execution_time_s", 0.0),
            )
        except Exception as e:
            return ExecutionResult(success=False, message=str(e))

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_robot_state(
        self,
        include_gripper: bool = True,
        include_torques: bool = False,
    ) -> RobotState:
        """
        Get current robot state.

        Args:
            include_gripper: Include gripper state
            include_torques: Include joint torques

        Returns:
            RobotState object

        Example:
            state = robot.get_robot_state()
            print(f"Joints: {state.joint_positions}")
            print(f"Gripper: {state.gripper_width}m")
        """
        if "robot" not in self._modules:
            return RobotState()

        try:
            state = self._modules["robot"].get_state()
            return RobotState(
                joint_positions=state.joint_positions,
                joint_velocities=state.joint_velocities if include_torques else [],
                ee_pose=state.ee_pose,
                gripper_width=state.gripper_width if include_gripper else 0,
                gripper_force=state.gripper_force if include_gripper else 0,
                is_moving=state.robot_mode == "executing",
                timestamp=state.timestamp,
            )
        except Exception as e:
            logger.error(f"Failed to get robot state: {e}")
            return RobotState()

    def get_ee_pose(self, frame: str = "base") -> Optional[Dict[str, Any]]:
        """
        Get end-effector pose.

        Args:
            frame: Reference frame ("base")

        Returns:
            Dict with "position" and "orientation" or None
        """
        state = self.get_robot_state()
        return state.ee_pose

    def get_joint_positions(self) -> List[float]:
        """
        Get current joint positions.

        Returns:
            List of joint positions in radians
        """
        state = self.get_robot_state()
        return state.joint_positions

    def get_detected_objects(
        self,
        object_names: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> List[DetectedObject]:
        """
        Get detected objects.

        Args:
            object_names: Filter by object names (optional)
            refresh: Force new detection

        Returns:
            List of DetectedObject
        """
        if refresh:
            self.detect_objects(object_names)

        if object_names:
            return [o for o in self._latest_objects if o.name in object_names]
        return self._latest_objects

    def get_object_pose(
        self,
        object_name: str,
        frame: str = "base",
    ) -> Optional[Dict[str, float]]:
        """
        Get pose of a specific object.

        Args:
            object_name: Name of object
            frame: Reference frame

        Returns:
            Position dict {"x", "y", "z"} or None if not found
        """
        objects = self.get_detected_objects()
        for obj in objects:
            if obj.name == object_name and obj.position:
                return obj.position
        return None

    def is_holding(self, object_name: Optional[str] = None) -> bool:
        """
        Check if robot is holding an object.

        Args:
            object_name: Specific object to check (optional)

        Returns:
            True if holding object
        """
        state = self.get_robot_state()
        # Simple heuristic: gripper is closed and has force
        return state.gripper_width < 0.01 and state.gripper_force > 5.0

    def is_busy(self) -> bool:
        """
        Check if robot is currently executing.

        Returns:
            True if executing
        """
        if "robot" not in self._modules:
            return False
        return self._modules["robot"].is_executing()

    # =========================================================================
    # Camera/Perception
    # =========================================================================

    def capture_image(
        self,
        camera: str = "default",
        include_depth: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture image from camera.

        Args:
            camera: Camera identifier
            include_depth: Include depth image

        Returns:
            Dict with "rgb" and optionally "depth" keys
        """
        result: Dict[str, Any] = {"rgb": None, "depth": None}

        if "perception" in self._modules:
            perception = self._modules["perception"]
            if hasattr(perception, "_latest_rgb"):
                result["rgb"] = perception._latest_rgb
            if include_depth and hasattr(perception, "_latest_depth"):
                result["depth"] = perception._latest_depth

        return result

    def detect_objects(
        self,
        object_list: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> List[DetectedObject]:
        """
        Run object detection.

        Args:
            object_list: List of objects to detect (optional)
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of DetectedObject
        """
        if "perception" not in self._modules:
            return []

        try:
            images = self.capture_image(include_depth=True)
            if images["rgb"] is None:
                return []

            detections = self._modules["perception"].process(
                rgb=images["rgb"],
                depth=images.get("depth"),
                object_list=object_list,
            )

            results = []
            for det in detections:
                if det.confidence >= confidence_threshold:
                    results.append(
                        DetectedObject(
                            name=det.name,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            position=det.pose.get("position") if det.pose else None,
                            orientation=det.pose.get("orientation") if det.pose else None,
                        )
                    )

            self._latest_objects = results
            return results

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def wait_for_object(
        self,
        object_name: str,
        timeout_s: float = 30.0,
    ) -> Optional[DetectedObject]:
        """
        Wait for an object to appear.

        Args:
            object_name: Name of object to wait for
            timeout_s: Timeout in seconds

        Returns:
            DetectedObject if found, None if timeout
        """
        start = time.time()
        while time.time() - start < timeout_s:
            objects = self.detect_objects([object_name])
            for obj in objects:
                if obj.name == object_name:
                    return obj
            time.sleep(0.5)
        return None

    # =========================================================================
    # Safety/Control
    # =========================================================================

    def stop(self, deceleration: float = 1.0) -> ExecutionResult:
        """
        Stop robot immediately.

        Args:
            deceleration: Deceleration rate

        Returns:
            ExecutionResult
        """
        logger.warning("Stopping robot")
        if "robot" in self._modules:
            self._modules["robot"]._stop_robot()
        return ExecutionResult(success=True, message="Robot stopped")

    def pause(self) -> ExecutionResult:
        """
        Pause current execution.

        Returns:
            ExecutionResult
        """
        logger.info("Pausing execution")
        with self._lock:
            self._paused = True

        if "robot" in self._modules and hasattr(self._modules["robot"], "pause"):
            try:
                self._modules["robot"].pause()
            except Exception as e:
                logger.warning(f"Robot pause failed: {e}")

        return ExecutionResult(success=True, message="Paused")

    def resume(self) -> ExecutionResult:
        """
        Resume paused execution.

        Returns:
            ExecutionResult
        """
        logger.info("Resuming execution")
        with self._lock:
            self._paused = False

        if "robot" in self._modules and hasattr(self._modules["robot"], "resume"):
            try:
                self._modules["robot"].resume()
            except Exception as e:
                logger.warning(f"Robot resume failed: {e}")

        return ExecutionResult(success=True, message="Resumed")

    def set_speed(self, velocity_scale: float) -> ExecutionResult:
        """
        Set robot speed scale.

        Args:
            velocity_scale: Speed scale 0.0-1.0

        Returns:
            ExecutionResult
        """
        if not 0.0 <= velocity_scale <= 1.0:
            return ExecutionResult(
                success=False,
                message="velocity_scale must be between 0.0 and 1.0",
            )

        logger.info(f"Setting speed scale to {velocity_scale}")
        self._velocity_scale = velocity_scale

        if "robot" in self._modules and hasattr(self._modules["robot"], "set_speed_scale"):
            try:
                self._modules["robot"].set_speed_scale(velocity_scale)
            except Exception as e:
                logger.warning(f"Robot set_speed failed: {e}")

        return ExecutionResult(success=True, message=f"Speed set to {velocity_scale}")

    # =========================================================================
    # Events/Callbacks
    # =========================================================================

    def on(self, event: str, callback: Callable) -> None:
        """
        Register event callback.

        Args:
            event: Event name ("error", "object_detected", "task_complete", etc.)
            callback: Callback function

        Example:
            robot.on("error", lambda e: print(f"Error: {e}"))
            robot.on("object_detected", lambda obj: print(f"Found: {obj.name}"))
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(callback)

    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout_s: float = 30.0,
    ) -> bool:
        """
        Wait until condition is met.

        Args:
            condition: Callable that returns True when condition is met
            timeout_s: Timeout in seconds

        Returns:
            True if condition met, False if timeout

        Example:
            robot.wait_until(lambda: not robot.is_busy(), timeout_s=10)
        """
        start = time.time()
        while time.time() - start < timeout_s:
            if condition():
                return True
            time.sleep(0.1)
        return False

    def _emit_event(self, event: str, data: Any = None) -> None:
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    # =========================================================================
    # Model Configuration
    # =========================================================================

    def set_model(
        self,
        module: str,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Change model for a specific module.

        Args:
            module: Module name ("perception", "planner", "controller", "monitor")
            provider: Model provider ("openai", "anthropic", "google", "hf", "custom")
            model: Model name or path
            api_key: API key (optional)
            device: Device for local models ("cuda:0", "cpu")
            api_base: API base URL (for ollama, etc.)
            **kwargs: Additional model-specific parameters

        Returns:
            ExecutionResult

        Example:
            robot.set_model(
                module="planner",
                provider="anthropic",
                model="claude-3-sonnet",
                api_key="sk-..."
            )

            robot.set_model(
                module="perception",
                provider="custom",
                model="/path/to/detector.py:MyDetector",
                device="cuda:0"
            )
        """
        valid_modules = ["perception", "planner", "controller", "monitor"]

        if module not in valid_modules:
            return ExecutionResult(
                success=False,
                message=f"Invalid module. Valid: {valid_modules}",
            )

        if module not in self._modules:
            return ExecutionResult(
                success=False,
                message=f"Module {module} not loaded",
            )

        logger.info(f"Changing {module} model to {provider}/{model}")

        # Stop current module
        try:
            self._modules[module].stop()
        except Exception as e:
            logger.warning(f"Error stopping module: {e}")

        # Reinitialize with new config would require recreating the module
        # For now, store the config and recreate on next use

        return ExecutionResult(
            success=True,
            message=f"Model changed to {provider}/{model}. Restart module to apply.",
        )

    def get_model(self, module: str) -> Optional[ModelInfo]:
        """
        Get current model configuration for a module.

        Args:
            module: Module name

        Returns:
            ModelInfo or None

        Example:
            info = robot.get_model("planner")
            print(f"Provider: {info.provider}, Model: {info.model}")
        """
        if module not in self._modules:
            return None

        mod = self._modules[module]
        return ModelInfo(
            provider=getattr(mod, "provider", "unknown"),
            model=getattr(mod, "model", "unknown"),
            device=getattr(mod, "device", None),
            api_base=getattr(mod, "api_base", None),
        )

    def list_modules(self) -> List[str]:
        """
        List available modules.

        Returns:
            List of module names

        Example:
            modules = robot.list_modules()
            # ["perception", "planner", "controller", "robot", "monitor"]
        """
        return list(self._modules.keys())

    # =========================================================================
    # Robot Interface Configuration
    # =========================================================================

    def set_robot_interface(self, custom_interface: Any) -> ExecutionResult:
        """
        Set custom robot interface for real robot control.

        Args:
            custom_interface: CustomRobot instance

        Returns:
            ExecutionResult

        Example:
            from robobridge import CustomRobot

            class MyFrankaInterface(CustomRobot):
                def connect(self):
                    self._robot = frankx.Robot(self.robot_ip)
                # ... other methods

            robot.set_robot_interface(MyFrankaInterface(robot_ip="172.16.0.2"))
        """
        if "robot" not in self._modules:
            return ExecutionResult(
                success=False,
                message="Robot module not loaded",
            )

        # Stop current
        self._modules["robot"].stop()

        # Create new with custom interface
        from ..config import load_config
        from ..modules import Robot

        config = load_config(self.config_path)
        cfg = config.module_configs.get("robot", config.module_configs.get("robot_interface", {}))
        adapter = config.adapters.get("robot")

        host = adapter.bind_host if adapter and adapter.bind_host else "127.0.0.1"
        port = adapter.bind_port if adapter and adapter.bind_port else 51004
        endpoint = (host, port)
        self._modules["robot"] = Robot(
            custom_interface=custom_interface,
            robot_type=cfg.get("robot_type", "franka"),
            rate_hz=cfg.get("rate_hz", 100),
            timeout_s=cfg.get("timeout_s", 15.0),
            link_mode=adapter.link_mode if adapter else "socket",
            adapter_endpoint=endpoint,
        )
        self._modules["robot"].start()

        self.simulation = False
        return ExecutionResult(
            success=True,
            message="Robot interface configured",
        )


# Alias for convenience
RoboBridge = RoboBridgeClient
