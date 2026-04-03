"""
RoboCasa Backend for RoboBridge.

Low-level interface for RoboCasa simulation environment.
RoboCasa is a MuJoCo-based simulation framework for household robot tasks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import robocasa
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.controllers import load_part_controller_config
    ROBOCASA_AVAILABLE = True
except ImportError:
    ROBOCASA_AVAILABLE = False

from robobridge.modules.robot.backends.base import RobotBackend
from robobridge.modules.robot.types import (
    ExecutionResult,
    GripperCommand,
    RobotStateData,
    TrajectoryPoint,
)


@dataclass
class RoboCasaConfig:
    """Configuration for RoboCasa environment."""
    env_name: str = "PnPCounterToCab"
    robots: str = "PandaOmron"
    translucent_robot: bool = False
    seed: int = 0
    render_onscreen: bool = False
    render_camera: str = "robot0_agentview_left"
    camera_heights: int = 256
    camera_widths: int = 256
    controller: str = "BASIC"
    
    move_speed: float = 0.4
    position_threshold_m: float = 0.01
    max_steps_per_move: int = 300
    gripper_action_steps: int = 50
    gripper_action_idx: int = 6


@dataclass
class RoboCasaObservation:
    """Observation from RoboCasa environment."""
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    robot_state: Optional[np.ndarray] = None
    gripper_state: Optional[np.ndarray] = None
    ee_pos: Optional[np.ndarray] = None
    ee_quat: Optional[np.ndarray] = None
    raw_obs: Dict[str, Any] = field(default_factory=dict)


class RoboCasaBackend(RobotBackend):
    """
    RoboCasa simulation backend.

    Provides interface between RoboBridge and RoboCasa/MuJoCo simulation.

    Example:
        config = RoboCasaConfig(env_name="PnPCounterToCab", render_onscreen=True)
        backend = RoboCasaBackend(config=config)
        backend.connect()
        
        obs = backend.get_observation()
        action = np.zeros(backend.action_dim)
        obs, reward, done, info = backend.step(action)
        
        backend.disconnect()
    """

    def __init__(
        self,
        robot_ip: str = "simulation",
        config: Optional[RoboCasaConfig] = None,
    ):
        super().__init__(robot_ip=robot_ip, config=config.__dict__ if config else {})
        self.robocasa_config = config or RoboCasaConfig()
        self._env = None
        self._last_obs = None
        self._task_language = ""
        
        self._robot_base_pos: Optional[np.ndarray] = None
        self._robot_base_rot: Optional[Rotation] = None
        self._step_count: int = 0

    @property
    def env(self):
        return self._env

    @property
    def task_language(self) -> str:
        return self._task_language

    @property
    def action_dim(self) -> int:
        if self._env is None:
            return 0
        return self._env.action_spec[0].shape[0]

    def connect(self) -> None:
        if not ROBOCASA_AVAILABLE:
            raise ImportError(
                "robocasa is not installed. Install with: "
                "git clone https://github.com/robocasa/robocasa && cd robocasa && pip install -e ."
            )

        # Build controller config
        controller_config = load_composite_controller_config(
            controller=None,
            robot=self.robocasa_config.robots,
        )

        # Override arm controller to IK_POSE when configured
        # Must replace the entire body part config (not just the type field)
        # because OSC_POSE params (input_max, kp, etc.) conflict with IK controller.
        # Preserve the nested "gripper" config from the original body part.
        # NOTE: IK_POSE only supports Panda, Sawyer, Baxter, GR1FixedLowerBody.
        if self.robocasa_config.controller == "IK_POSE":
            _ik_supported = {"Panda", "Sawyer", "Baxter", "GR1FixedLowerBody"}
            if self.robocasa_config.robots not in _ik_supported:
                raise ValueError(
                    f"IK_POSE does not support robot '{self.robocasa_config.robots}'. "
                    f"Supported: {_ik_supported}. Use OSC_POSE or switch to a supported robot."
                )
            ik_config = load_part_controller_config(default_controller="IK_POSE")
            for part_name, cfg in controller_config.get("body_parts", {}).items():
                if cfg.get("type") == "OSC_POSE":
                    gripper_cfg = cfg.get("gripper")
                    new_cfg = dict(ik_config)
                    if gripper_cfg:
                        new_cfg["gripper"] = gripper_cfg
                    controller_config["body_parts"][part_name] = new_cfg

        self._env = robosuite.make(
            self.robocasa_config.env_name,
            robots=self.robocasa_config.robots,
            controller_configs=controller_config,
            has_renderer=self.robocasa_config.render_onscreen,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            render_camera=self.robocasa_config.render_camera,
            camera_names=[self.robocasa_config.render_camera, "robot0_eye_in_hand"],
            camera_heights=self.robocasa_config.camera_heights,
            camera_widths=self.robocasa_config.camera_widths,
            seed=self.robocasa_config.seed,
            translucent_robot=self.robocasa_config.translucent_robot,
        )

        self._last_obs = self._env.reset()
        
        ep_meta = self._env.get_ep_meta()
        self._task_language = ep_meta.get("lang", "")
        
        self._setup_robot_frame()

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _setup_robot_frame(self) -> None:
        if self._env is None or not SCIPY_AVAILABLE:
            return
        
        robot = self._env.robots[0]
        base_body_id = self._env.sim.model.body_name2id(robot.robot_model.root_body)
        self._robot_base_pos = self._env.sim.data.body_xpos[base_body_id].copy()
        
        base_quat_wxyz = self._env.sim.data.body_xquat[base_body_id]
        quat_xyzw = [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        self._robot_base_rot = Rotation.from_quat(quat_xyzw)

    def _world_to_robot_frame(self, world_vec: np.ndarray) -> np.ndarray:
        if self._robot_base_rot is None:
            return world_vec
        return self._robot_base_rot.inv().apply(world_vec)

    def reset(self) -> RoboCasaObservation:
        if self._env is None:
            raise RuntimeError("Environment not connected")
        
        self._last_obs = self._env.reset()
        ep_meta = self._env.get_ep_meta()
        self._task_language = ep_meta.get("lang", "")
        
        return self._parse_observation(self._last_obs)

    def step(self, action: np.ndarray) -> Tuple[RoboCasaObservation, float, bool, Dict]:
        if self._env is None:
            raise RuntimeError("Environment not connected")
        
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        
        return self._parse_observation(obs), reward, done, info

    def render(self) -> Optional[np.ndarray]:
        if self._env is None:
            return None
        
        if self.robocasa_config.render_onscreen:
            self._env.render()
            return None
        else:
            return self._env.sim.render(
                camera_name=self.robocasa_config.render_camera,
                width=self.robocasa_config.camera_widths,
                height=self.robocasa_config.camera_heights,
            )

    def get_observation(self) -> RoboCasaObservation:
        if self._last_obs is None:
            raise RuntimeError("No observation available. Call reset() first.")
        return self._parse_observation(self._last_obs)

    def _parse_observation(self, obs: Dict[str, Any]) -> RoboCasaObservation:
        camera_name = self.robocasa_config.render_camera
        rgb_key = f"{camera_name}_image"
        depth_key = f"{camera_name}_depth"

        rgb = obs.get(rgb_key)
        if rgb is None:
            for key in obs:
                if key.endswith("_image"):
                    rgb = obs[key]
                    break

        depth = obs.get(depth_key)

        robot_state = obs.get("robot0_proprio-state")
        gripper_state = obs.get("robot0_gripper_qpos")
        ee_pos = obs.get("robot0_eef_pos")
        ee_quat = obs.get("robot0_eef_quat")

        return RoboCasaObservation(
            rgb=rgb,
            depth=depth,
            robot_state=robot_state,
            gripper_state=gripper_state,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            raw_obs=obs,
        )

    def get_robot_state(self) -> RobotStateData:
        obs = self.get_observation()
        
        joint_positions = []
        joint_velocities = []
        
        if obs.robot_state is not None:
            n_joints = min(7, len(obs.robot_state) // 2)
            joint_positions = list(obs.robot_state[:n_joints])
            joint_velocities = list(obs.robot_state[n_joints:2*n_joints])

        gripper_width = 0.08
        if obs.gripper_state is not None and len(obs.gripper_state) >= 2:
            gripper_width = float(obs.gripper_state[0] + obs.gripper_state[1])

        ee_pose = {}
        if obs.ee_pos is not None:
            ee_pose["position"] = list(obs.ee_pos)
        if obs.ee_quat is not None:
            ee_pose["orientation"] = list(obs.ee_quat)

        gripper_qpos = []
        if obs.gripper_state is not None:
            gripper_qpos = list(obs.gripper_state)

        return RobotStateData(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            gripper_width=gripper_width,
            gripper_qpos=gripper_qpos,
            ee_pose=ee_pose if ee_pose else None,
        )

    def execute_trajectory(self, points: List[TrajectoryPoint]) -> bool:
        if self._env is None:
            return False

        for point in points:
            action = np.zeros(self.action_dim)
            
            if point.positions:
                n = min(len(point.positions), 7)
                action[:n] = point.positions[:n]

            self.step(action)

        return True

    def execute_gripper(self, command: GripperCommand) -> bool:
        if self._env is None:
            return False

        action = np.zeros(self.action_dim)

        gripper_idx = self.robocasa_config.gripper_action_idx
        if command.action == "close":
            gripper_val = 1.0
        elif command.action == "open":
            gripper_val = -1.0
        else:
            gripper_val = (command.width / 0.08) * 2 - 1
        action[gripper_idx] = gripper_val

        for _ in range(self.robocasa_config.gripper_action_steps):
            self.step(action)

        return True

    def stop_robot(self) -> None:
        if self._env is not None:
            action = np.zeros(self.action_dim)
            self.step(action)

    def move_to_position(
        self, 
        target_pos: np.ndarray, 
        maintain_orientation: bool = True,
        verbose: bool = False
    ) -> bool:
        """
        Move end-effector to target position using constant velocity control.
        
        This uses constant-velocity control instead of proportional control,
        which ensures sufficient movement commands even near the target.
        
        Args:
            target_pos: Target position in world frame [x, y, z]
            maintain_orientation: Keep current orientation during move
            verbose: Print debug info
            
        Returns:
            True if target reached within threshold
        """
        if self._env is None:
            return False
        
        target_pos = np.array(target_pos)
        
        for step_i in range(self.robocasa_config.max_steps_per_move):
            current_pos = self._last_obs["robot0_eef_pos"]
            error = target_pos - current_pos
            dist = np.linalg.norm(error)
            
            if verbose and step_i % 20 == 0:
                print(f"  Step {step_i}: pos={current_pos}, dist={dist:.4f}m")
                if step_i == 0:
                    print(f"    Robot base pos: {self._robot_base_pos}")
                    euler = self._robot_base_rot.as_euler('xyz', degrees=True) if self._robot_base_rot else [0,0,0]
                    print(f"    Robot base rot (deg): {euler}")
            
            if dist < self.robocasa_config.position_threshold_m:
                if verbose:
                    print(f"  Reached target in {step_i} steps, final dist={dist:.4f}m")
                return True
            
            # Constant velocity control: direction * speed instead of error * gain
            # Key insight from human demo analysis - maintains sufficient command near target
            direction = error / dist
            speed = self.robocasa_config.move_speed
            
            delta_world = direction * speed
            delta_robot = self._world_to_robot_frame(delta_world)
            
            action = np.zeros(self.action_dim)
            action[0:3] = delta_robot
            
            if verbose and step_i % 40 == 0:
                print(f"    delta_world={delta_world}, delta_robot={delta_robot}")
            
            self.step(action)
            self._step_count += 1
        
        if verbose:
            final_pos = self._last_obs["robot0_eef_pos"]
            final_dist = np.linalg.norm(target_pos - final_pos)
            print(f"  Max steps reached, final dist={final_dist:.4f}m")
        
        return False

    def set_gripper(self, open_gripper: bool, verbose: bool = False) -> bool:
        """
        Open or close the gripper.
        
        Args:
            open_gripper: True to open, False to close
            verbose: Print debug info
            
        Returns:
            True on success
        """
        if self._env is None:
            return False
        
        gripper_idx = self.robocasa_config.gripper_action_idx
        gripper_value = -1.0 if open_gripper else 1.0
        
        if verbose:
            action_name = "open" if open_gripper else "close"
            print(f"  Gripper {action_name} for {self.robocasa_config.gripper_action_steps} steps")
        
        for _ in range(self.robocasa_config.gripper_action_steps):
            action = np.zeros(self.action_dim)
            action[gripper_idx] = gripper_value
            self.step(action)
            self._step_count += 1
        
        return True

    def execute_primitive(
        self, 
        primitive_type: str, 
        target_pos: Optional[np.ndarray] = None,
        gripper_open: Optional[bool] = None,
        verbose: bool = False
    ) -> bool:
        """
        Execute a single primitive action.
        
        Primitive types:
        - "move": Move to target position
        - "grip": Set gripper state (open/close)
        - "go": Same as move (alias)
        
        Args:
            primitive_type: Type of primitive ("move", "grip", "go")
            target_pos: Target position for move primitives [x, y, z]
            gripper_open: Gripper state for grip primitive (True=open, False=close)
            verbose: Print debug info
            
        Returns:
            True on success
        """
        if verbose:
            print(f"Executing primitive: {primitive_type}")
        
        if primitive_type in ("move", "go"):
            if target_pos is None:
                print("  Error: target_pos required for move primitive")
                return False
            return self.move_to_position(target_pos, verbose=verbose)
        
        elif primitive_type == "grip":
            if gripper_open is None:
                print("  Error: gripper_open required for grip primitive")
                return False
            return self.set_gripper(gripper_open, verbose=verbose)
        
        else:
            print(f"  Unknown primitive type: {primitive_type}")
            return False

    def process(self, command: Dict[str, Any]) -> ExecutionResult:
        """Process a Command dict from Controller.

        Handles VLA-style cartesian_delta commands by building a single
        action vector and calling env.step(). For IK_POSE environments,
        robosuite internally converts the EE delta to joint angles.

        Args:
            command: Command dict with command_type, points, gripper_command.

        Returns:
            ExecutionResult
        """
        import time as _time

        command_type = command.get("command_type", "")
        command_id = command.get("command_id", "unknown")
        start_time = _time.time()

        try:
            if command_type == "cartesian_delta":
                # VLA output: 1-step EE delta
                points = command.get("points", [])
                if points and self._env is not None:
                    point = points[0]
                    action = np.zeros(self.action_dim)
                    positions = point.get("positions", [0, 0, 0])
                    action[0:min(3, len(positions))] = positions[:3]
                    rotations = point.get("rotations")
                    if rotations:
                        action[3:min(6, 3 + len(rotations))] = rotations[:3]
                    # Gripper: maintain current state or apply command
                    gripper_cmd = command.get("gripper_command")
                    if gripper_cmd:
                        action_val = -1.0 if gripper_cmd.get("action") == "open" else 1.0
                        action[self.robocasa_config.gripper_action_idx] = action_val
                    self.step(action)

                state = self.get_robot_state()
                return ExecutionResult(
                    command_id=command_id,
                    success=True,
                    state="idle",
                    actual_positions=state.joint_positions,
                    execution_time_s=_time.time() - start_time,
                    metadata={"command_type": command_type},
                )

            elif command_type == "joint_targets":
                # IK-solved joint positions: send directly as joint trajectory
                points = command.get("points", [])
                if points and self._env is not None:
                    point = points[0]
                    positions = point.get("positions", [])
                    action = np.zeros(self.action_dim)
                    n = min(len(positions), 7)
                    action[:n] = positions[:n]
                    # Apply gripper command if present
                    gripper_cmd = command.get("gripper_command")
                    if gripper_cmd:
                        action_val = -1.0 if gripper_cmd.get("action") == "open" else 1.0
                        action[self.robocasa_config.gripper_action_idx] = action_val
                    self.step(action)

                state = self.get_robot_state()
                return ExecutionResult(
                    command_id=command_id,
                    success=True,
                    state="idle",
                    actual_positions=state.joint_positions,
                    execution_time_s=_time.time() - start_time,
                    metadata={"command_type": command_type},
                )

            elif command_type == "cartesian_absolute":
                # Absolute EEF target from passthrough IK: use move_to_position
                points = command.get("points", [])
                if points and self._env is not None:
                    point = points[0]
                    positions = point.get("positions", [0, 0, 0])
                    target_pos = np.array(positions[:3])
                    self.move_to_position(target_pos)

                state = self.get_robot_state()
                return ExecutionResult(
                    command_id=command_id,
                    success=True,
                    state="idle",
                    actual_positions=state.joint_positions,
                    execution_time_s=_time.time() - start_time,
                    metadata={"command_type": command_type},
                )

            elif command_type == "gripper":
                gripper_cmd = command.get("gripper_command", {})
                gc = GripperCommand(
                    action=gripper_cmd.get("action", "open"),
                    width=gripper_cmd.get("width", 0.08),
                    speed=gripper_cmd.get("speed", 0.1),
                    force=gripper_cmd.get("force", 40.0),
                )
                success = self.execute_gripper(gc)
                state = self.get_robot_state()
                return ExecutionResult(
                    command_id=command_id,
                    success=success,
                    state="idle",
                    actual_positions=state.joint_positions,
                    execution_time_s=_time.time() - start_time,
                    metadata={"command_type": command_type},
                )

            else:
                # Fallback to base class (joint trajectory, etc.)
                return super().process(command)

        except Exception as e:
            return ExecutionResult(
                command_id=command_id,
                success=False,
                state="error",
                execution_time_s=_time.time() - start_time,
            )

    def start(self) -> None:
        """Start backend (Robot module compatibility)."""
        self.connect()

    def stop(self) -> None:
        """Stop backend (Robot module compatibility)."""
        self.stop_robot()
        self.disconnect()

    def get_available_tasks(self) -> List[str]:
        if not ROBOCASA_AVAILABLE:
            return []

        from robocasa.utils.env_utils import get_all_env_names
        return get_all_env_names()
