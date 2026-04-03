"""
VLA LoRA Controller.

Bridges the VLA LoRA system with the existing Controller module.
Receives primitives from the planner, selects the appropriate LoRA adapter
(move or grip), runs VLA inference, and returns Command objects.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .types import Command, TrajectoryPoint

if TYPE_CHECKING:
    from .ik_solver import IKSolver

logger = logging.getLogger(__name__)


class VLALoRAController:
    """
    VLA controller with LoRA adapter switching.

    Planner primitive_type -> LoRA adapter selection -> VLA inference -> Command.

    Supports task-specific adapters with general fallback:
        1. Try "{task_name}_{primitive_type}" adapter
        2. Fall back to "{primitive_type}" adapter
    """

    def __init__(
        self,
        vla_config: Dict[str, Any],
        control_rate_hz: float = 20.0,
        frame_convention: str = "base",
        ik_solver: Optional[IKSolver] = None,
        use_absolute_targets: bool = False,
    ):
        """
        Args:
            vla_config: VLA configuration dict with keys:
                - backend: VLA backend name (openvla, smolvla, etc.)
                - model_name: HuggingFace model ID
                - move_adapter_path: Path to move LoRA adapter
                - grip_adapter_path: Path to grip LoRA adapter
                - task_adapters: Optional dict of {task_name: {move: path, grip: path}}
                - quantize_4bit: Whether to use 4-bit quantization
                - action_stats_path: Path to action normalization stats
            control_rate_hz: Control loop frequency.
            frame_convention: Coordinate frame for commands.
            ik_solver: Optional IK solver for converting EEF targets to joint angles.
            use_absolute_targets: If True, convert VLA delta output to absolute
                EEF target, solve IK, and emit joint_targets commands.
                If False (default), emit cartesian_delta commands as before.
        """
        self._vla_config = vla_config
        self._control_rate_hz = control_rate_hz
        self._frame_convention = frame_convention
        self._ik_solver = ik_solver
        self._use_absolute_targets = use_absolute_targets
        self._lora_manager = None
        self._action_tokenizer = None
        self._state_stats: Optional[Dict] = None
        self._command_counter = 0
        self._current_task: Optional[str] = None
        self._remote_url: Optional[str] = vla_config.get("remote_url")
        self._http_session = None
        self._action_scale: float = float(vla_config.get("action_scale", 1.0))
        self._action_ema_alpha: float = float(vla_config.get("action_ema_alpha", 0.3))
        self._prev_action: Optional[np.ndarray] = None
        self._use_direction_normalization: bool = False
        self._zero_direction: bool = False
        self._use_ultimate_target: bool = vla_config.get("use_ultimate_target", True)

        # Note: action chunking is managed by the backend (LeRobotBackend)
        # which maintains its own chunk buffer and stride internally.

    def initialize(self) -> None:
        """Load VLA model and LoRA adapters (or set up HTTP remote mode)."""
        cfg = self._vla_config

        # Load action tokenizer (needed for both local and remote mode)
        from robobridge.modules.controller.vla.action_tokenizer import ActionTokenizer

        stats_path = cfg.get("action_stats_path")
        if stats_path:
            self._action_tokenizer = ActionTokenizer.from_file(stats_path)
            # Allow config to override auto-detected mode (e.g. GROOT needs min_max)
            norm_type_override = cfg.get("norm_type")
            if norm_type_override:
                self._action_tokenizer.mode = norm_type_override
            logger.info(f"Action tokenizer loaded (mode={self._action_tokenizer.mode})")
        else:
            logger.warning(
                "action_stats_path not set in vla_config. "
                "VLA output will NOT be denormalized — actions will be wrong. "
                "Set action_stats_path to the metadata.json from preprocessing."
            )

        # Load state normalization stats
        state_stats_path = cfg.get("state_stats_path")
        if not state_stats_path:
            # Auto-detect: look for data_stats.json near adapter paths
            for adapter_path in [cfg.get("move_adapter_path"), cfg.get("grip_adapter_path")]:
                if not adapter_path:
                    continue
                for candidate in [
                    os.path.join(adapter_path, "data_stats.json"),
                    os.path.join(os.path.dirname(adapter_path), "data_stats.json"),
                ]:
                    if os.path.exists(candidate):
                        state_stats_path = candidate
                        break
                if state_stats_path:
                    break

        if state_stats_path:
            with open(state_stats_path) as f:
                data_stats = json.load(f)
            self._state_stats = data_stats.get("state_stats")
            # Detect PI0.5 tensor-state mode
            self._pi05_tensor_state = data_stats.get("pi05_tensor_state", False)
            if self._pi05_tensor_state:
                logger.info("PI0.5 tensor-state mode detected from data_stats.json")
            # Detect GROOT full fine-tuning mode
            self._groot_full_ft = data_stats.get("groot_full_ft", False)
            if self._groot_full_ft:
                logger.info("GROOT full fine-tuning mode detected from data_stats.json")
            # Detect direction-normalized state format
            state_fmt = data_stats.get("state_format", "")
            self._use_direction_normalization = "direction" in state_fmt
            if self._use_direction_normalization:
                logger.info("Direction normalization enabled (unit vector delta)")
            self._zero_direction = "zero_direction" in state_fmt
            if self._zero_direction:
                logger.info("Zero direction mode: state direction always [0,0,0]")
            if self._state_stats:
                logger.info(
                    f"State stats loaded (dim={len(self._state_stats['min'])}) "
                    f"from {state_stats_path}"
                )
        else:
            logger.warning(
                "state_stats not found — state normalization disabled. "
                "Set state_stats_path or place data_stats.json near adapter."
            )

        # Remote mode: GPU server handles model inference
        if self._remote_url:
            import requests
            self._http_session = requests.Session()
            # Verify connection
            try:
                resp = self._http_session.get(f"{self._remote_url}/health", timeout=5)
                resp.raise_for_status()
                logger.info(f"VLA remote mode: connected to {self._remote_url}")
                info = self._http_session.get(f"{self._remote_url}/info", timeout=5).json()
                logger.info(f"  Remote model: {info.get('backend')}/{info.get('model')}")
                logger.info(f"  Remote adapters: {info.get('adapters')}")
            except Exception as e:
                logger.warning(f"VLA remote server not reachable: {e} (will retry on first request)")
            return

        # Local mode: load model + LoRA adapters on this machine
        from robobridge.modules.controller.vla import (
            LoRAManager,
            get_vla_backend,
        )
        from robobridge.modules.controller.vla.types import (
            LoRAAdapterConfig,
            VLAModelConfig,
        )

        backend_name = cfg.get("backend", "openvla")

        # Read lora_rank/alpha from adapter config.json (e.g. GROOT uses 64/128)
        lora_rank = 16
        lora_alpha = 32
        for path in [cfg.get("move_adapter_path"), cfg.get("grip_adapter_path")]:
            if path:
                adapter_cfg_path = os.path.join(path, "config.json")
                if os.path.exists(adapter_cfg_path):
                    with open(adapter_cfg_path) as f:
                        acfg = json.load(f)
                    if "lora_rank" in acfg:
                        lora_rank = acfg["lora_rank"]
                        lora_alpha = acfg.get("lora_alpha", lora_rank * 2)
                        logger.info(f"Read lora config from adapter: rank={lora_rank}, alpha={lora_alpha}")
                        break

        # For GROOT full-FT: load from checkpoint path directly (no base model + adapters)
        groot_full_ft = getattr(self, '_groot_full_ft', False)
        if groot_full_ft and cfg.get("move_adapter_path"):
            model_name = cfg["move_adapter_path"]
            logger.info(f"GROOT full-FT: loading model from checkpoint {model_name}")
        else:
            model_name = cfg["model_name"]

        # Create model config
        model_config = VLAModelConfig(
            backend=backend_name,
            model_name=model_name,
            device=cfg.get("device", "cuda:0"),
            quantize_4bit=cfg.get("quantize_4bit", False),
            action_dim=cfg.get("action_dim", 7),
            lora_rank=0 if groot_full_ft else lora_rank,
            lora_alpha=0 if groot_full_ft else lora_alpha,
            chunk_stride=cfg.get("chunk_stride", 8),
        )

        # Create VLA backend
        backend_cls = get_vla_backend(backend_name)
        vla_backend = backend_cls(model_config)

        # Set tensor-state mode before load_model() so the correct policy class is used
        if getattr(self, '_pi05_tensor_state', False):
            vla_backend._use_tensor_state = True
        # Set full-FT flag so backend skips adapter loading/switching
        if groot_full_ft:
            vla_backend._is_full_ft = True

        # Build adapter configs
        adapter_configs: Dict[str, LoRAAdapterConfig] = {}

        if groot_full_ft:
            # Full-FT: register dummy adapters (no actual weight loading)
            # LoRAManager still needs entries for resolve_adapter() to work
            adapter_configs["move"] = LoRAAdapterConfig(
                name="move",
                path=cfg.get("move_adapter_path", ""),
                primitive_types=["move"],
                task_name=None,
            )
            adapter_configs["grip"] = LoRAAdapterConfig(
                name="grip",
                path=cfg.get("move_adapter_path", ""),  # same model for grip
                primitive_types=["grip"],
                task_name=None,
            )
        else:
            # General adapters
            if cfg.get("move_adapter_path"):
                adapter_configs["move"] = LoRAAdapterConfig(
                    name="move",
                    path=cfg["move_adapter_path"],
                    primitive_types=["move"],
                    task_name=None,
                )
            if cfg.get("grip_adapter_path"):
                adapter_configs["grip"] = LoRAAdapterConfig(
                    name="grip",
                    path=cfg["grip_adapter_path"],
                    primitive_types=["grip"],
                    task_name=None,
                )

        # Task-specific adapters
        task_adapters = cfg.get("task_adapters", {})
        for task_name, paths in task_adapters.items():
            if "move" in paths:
                name = f"{task_name}_move"
                adapter_configs[name] = LoRAAdapterConfig(
                    name=name,
                    path=paths["move"],
                    primitive_types=["move"],
                    task_name=task_name,
                )
            if "grip" in paths:
                name = f"{task_name}_grip"
                adapter_configs[name] = LoRAAdapterConfig(
                    name=name,
                    path=paths["grip"],
                    primitive_types=["grip"],
                    task_name=task_name,
                )

        # Create LoRA manager
        self._lora_manager = LoRAManager(vla_backend, adapter_configs)
        self._lora_manager.initialize()

        # Pass state stats to backend (PI0.5 needs quantile stats for state-in-prompt)
        if self._state_stats and hasattr(vla_backend, 'set_state_stats'):
            vla_backend.set_state_stats(self._state_stats)

        logger.info(
            f"VLA LoRA Controller initialized: backend={backend_name}, "
            f"adapters={list(adapter_configs.keys())}"
        )

    def set_task(self, task_name: Optional[str]) -> None:
        """Set current task for task-specific adapter selection."""
        self._current_task = task_name
        logger.info(f"Task set to: {task_name}")

    def reset_policy(self) -> None:
        """Reset VLA policy state (clear backend chunk buffer). Call at primitive start."""
        if self._lora_manager is not None:
            self._lora_manager.reset_policy()
        self._prev_action = None

    def soft_reset_policy(self) -> None:
        """Partial reset: clear chunk buffer but keep EMA for smooth transitions."""
        if self._lora_manager is not None:
            # Clear chunk buffer and index, but don't call policy.reset()
            self._lora_manager._chunk_buffer = None
            self._lora_manager._chunk_idx = 0
        # Keep self._prev_action for EMA continuity

    def process_primitive(
        self,
        primitive: Dict,
        rgb: Optional[Dict[str, np.ndarray]] = None,
        robot_state: Optional[Dict] = None,
        instruction: str = "",
    ) -> Optional[Command]:
        """Process a single primitive using VLA with LoRA.

        Args:
            primitive: Primitive action dict with keys:
                - primitive_type: "move" or "grip"
                - target_position: {x, y, z} (for move)
                - target_rotation: {roll, pitch, yaw} (for move, optional)
                - grip_width: float (for grip)
                - instruction: str (primitive-level instruction)
            rgb: Camera images dict.
            robot_state: Current robot state dict.
            instruction: Natural language task instruction (fallback).

        Returns:
            Command object for robot execution.
        """
        primitive_type = primitive.get("primitive_type", "move")

        # Build state vector from robot_state dict, with target position appended
        state_vector = self._build_state_vector(robot_state or {}, primitive)

        # Predict single action (backend handles chunking internally)
        action = self._predict(state_vector, rgb, primitive_type, instruction)

        # Denormalize action
        raw_action = action.copy()
        if self._action_tokenizer:
            # Clip raw VLA output before denormalization
            if self._action_tokenizer.mode == "zscore":
                # zscore: allow up to ±5 std deviations (covers 99.99994%)
                raw_action = np.clip(raw_action, -5.0, 5.0)
            else:
                # min_max / quantile / discrete: output should be in [-1, 1]
                raw_action = np.clip(raw_action, -1.0, 1.0)
            action = self._action_tokenizer.recover_action(raw_action)

        # [v5] SmolVLA approach corrections: rotation clipping + direction alignment
        is_smolvla = (
            self._lora_manager is not None
            and getattr(self._lora_manager._backend, '_policy_type', None) == 'smolvla'
        )
        if is_smolvla and primitive_type == "move":
            target_pos_dict = primitive.get("target_position")
            if target_pos_dict and isinstance(target_pos_dict, dict):
                from scipy.spatial.transform import Rotation as Rot
                ee_pose = (robot_state or {}).get("ee_pose", {})
                ee_p = ee_pose.get("position", {})
                eef_w = np.array([ee_p.get("x", 0), ee_p.get("y", 0), ee_p.get("z", 0)])
                bp = (robot_state or {}).get("base_pos")
                bq = (robot_state or {}).get("base_quat")
                if bp is not None and bq is not None:
                    eef_rb = Rot.from_quat(bq).inv().apply(eef_w - np.array(bp))
                else:
                    eef_rb = eef_w
                tgt_rb = np.array([target_pos_dict["x"], target_pos_dict["y"], target_pos_dict["z"]])
                delta = tgt_rb - eef_rb
                dist = float(np.linalg.norm(delta))

                # (1) Rotation clipping during approach (dist > 0.15m)
                #     Prevents large rotations from destabilizing trajectory before reaching target
                if dist > 0.15:
                    action[3:6] = np.clip(action[3:6], -0.05, 0.05)

                # (2) Direction alignment: correct position when diverging from target
                #     Fixes action mean bias (e.g. TurnOffSinkFaucet pos_y=+0.108 bias)
                if dist > 0.05:
                    tgt_dir = delta / (dist + 1e-8)
                    pos_mag = float(np.linalg.norm(action[:3]))
                    if pos_mag > 1e-6:
                        alignment = float(np.dot(action[:3] / pos_mag, tgt_dir))
                        if alignment < 0.3:
                            # Blend 50% towards target direction, keeping original magnitude
                            action[:3] = 0.5 * tgt_dir * pos_mag + 0.5 * action[:3]

        # Apply action scale (reduce overshooting for OSC_POSE)
        if self._action_scale != 1.0:
            action[:6] *= self._action_scale

        # EMA smoothing: reduce jerk at chunk boundaries
        if self._action_ema_alpha > 0 and self._prev_action is not None:
            alpha = self._action_ema_alpha
            action[:6] = alpha * self._prev_action[:6] + (1 - alpha) * action[:6]
        self._prev_action = action.copy()

        # Debug: log action values every 50 steps
        self._command_counter_debug = getattr(self, '_command_counter_debug', 0) + 1
        if self._command_counter_debug <= 5 or self._command_counter_debug % 50 == 0:
            logger.info(
                f"VLA action [step {self._command_counter_debug}]: "
                f"raw={[round(float(v), 4) for v in raw_action[:7]]}, "
                f"denorm={[round(float(v), 4) for v in action[:7]]}, "
                f"mode={self._action_tokenizer.mode if self._action_tokenizer else 'none'}"
            )

        return self._action_to_command(action, primitive_type, primitive, robot_state or {})

    def _predict_remote(
        self,
        state_vector: np.ndarray,
        rgb: Optional[Dict[str, np.ndarray]],
        primitive_type: str,
    ) -> np.ndarray:
        """Send raw inputs to remote GPU server, return raw action.

        Encodes image as base64 JPEG, sends state + adapter name.
        GPU server runs model forward pass and returns raw 7D action.
        """
        # Encode image as base64 JPEG
        image_b64 = None
        if rgb:
            # Use first available camera image
            image = next(iter(rgb.values()))
            if image is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(image)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        payload = {
            "state": state_vector.tolist(),
            "adapter": primitive_type,
            "instruction": primitive_type,
        }
        if image_b64:
            payload["image_b64"] = image_b64

        resp = self._http_session.post(
            f"{self._remote_url}/predict",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Remote VLA error: {data['error']}")

        return np.array(data["action"], dtype=np.float32)

    def _predict(
        self,
        state_vector: np.ndarray,
        rgb: Optional[Dict[str, np.ndarray]],
        primitive_type: str,
        instruction: str,
    ) -> np.ndarray:
        """Predict a single action. Backend handles chunking internally.

        Returns:
            np.ndarray of shape (action_dim,).
        """
        if self._remote_url:
            return self._predict_remote(state_vector, rgb, primitive_type)

        # Local mode
        from robobridge.modules.controller.vla.types import VLAInput

        state_mask = None
        if self._state_stats:
            state_vector, state_mask = self._normalize_and_pad_state(state_vector)

        vla_input = VLAInput(
            images=rgb or {},
            robot_state=state_vector,
            instruction=instruction if instruction else primitive_type,
            primitive_type=primitive_type,
            state_mask=state_mask,
        )

        vla_output = self._lora_manager.predict(vla_input, task_name=self._current_task)
        action = vla_output.action

        # Backend returns single action (7,) from its internal chunk buffer
        return action.flatten()

    def _build_state_vector(self, robot_state: Dict, primitive: Dict) -> np.ndarray:
        """Convert robot state dict to flat numpy vector with delta_base target.

        Output: 12D = eef_base(3) + eef_quat(4) + gripper(2) + delta_base(3)

        eef_base  = R_inv @ (eef_world - base_pos)  — robot-centric proprioception
        delta_base = target_rb - eef_base            — robot-centric goal direction

        For move primitives, delta_base = planner target (robot-base) - eef_base.
        For grip primitives, delta_base = [0, 0, 0] (stay in place).
        """
        from scipy.spatial.transform import Rotation as Rot

        # End-effector position in world frame
        ee_pose = robot_state.get("ee_pose", {})
        ee_pos = ee_pose.get("position", {"x": 0, "y": 0, "z": 0})
        eef_world = np.array([ee_pos.get("x", 0), ee_pos.get("y", 0), ee_pos.get("z", 0)], dtype=np.float64)

        # Base pose for frame transformation
        base_pos = robot_state.get("base_pos")
        base_quat = robot_state.get("base_quat")  # xyzw format

        # Prefer GT base-frame EEF pos from obs (matches HDF5 training data exactly).
        # Fall back to manual computation from world-frame pos if not available.
        gt_eef_base = robot_state.get("robot0_base_to_eef_pos")
        _log_eef_source = not getattr(self, '_eef_source_logged', False)
        if gt_eef_base is not None:
            eef_base = np.array(gt_eef_base, dtype=np.float32)
            if _log_eef_source:
                logger.info("eef_base: using GT robot0_base_to_eef_pos from obs (train-eval consistent)")
                self._eef_source_logged = True
            if base_pos is not None and base_quat is not None:
                base_pos = np.array(base_pos, dtype=np.float64)
                rot_inv = Rot.from_quat(base_quat).inv()
            else:
                rot_inv = None
        elif base_pos is not None and base_quat is not None:
            base_pos = np.array(base_pos, dtype=np.float64)
            rot_inv = Rot.from_quat(base_quat).inv()
            eef_base = rot_inv.apply(eef_world - base_pos).astype(np.float32)
            if _log_eef_source:
                logger.info("eef_base: using manual R_inv@(eef_world-base_pos) (robot0_base_to_eef_pos not in obs)")
                self._eef_source_logged = True
        else:
            eef_base = eef_world.astype(np.float32)
            rot_inv = None
            if _log_eef_source:
                logger.info("eef_base: using eef_world directly (no base_pos/base_quat in obs)")
                self._eef_source_logged = True

        parts = []

        # eef_base (3D) — robot-centric proprioception
        parts.append(eef_base.tolist())

        # End-effector orientation (4D quaternion) — convert to base frame
        ee_ori = ee_pose.get("orientation", {"x": 0, "y": 0, "z": 0, "w": 1})
        quat_world = np.array([ee_ori.get("x", 0), ee_ori.get("y", 0), ee_ori.get("z", 0), ee_ori.get("w", 1)])
        if rot_inv is not None:
            quat_base = (rot_inv * Rot.from_quat(quat_world)).as_quat()
        else:
            quat_base = quat_world
        parts.append(quat_base.tolist())

        # Gripper (2D)
        gripper = robot_state.get("gripper_qpos", [0.04, 0.04])
        parts.append(list(gripper))

        # delta_base (3D) — robot-centric goal direction
        prim_type = primitive.get("primitive_type", "move")
        target_pos = primitive.get("target_position")

        # DIAGNOSTIC: use fixed training-mean direction to test model quality
        _DIAG_FIXED_DIRECTION = getattr(self, '_diag_fixed_direction', None)

        if getattr(self, '_zero_direction', False):
            # Zero direction mode: always [0,0,0] (model relies on vision only)
            parts.append([0.0, 0.0, 0.0])
            # Still need to set _current_target_world for convergence detection
            if target_pos and isinstance(target_pos, dict):
                target_rb = np.array([target_pos.get("x", 0), target_pos.get("y", 0), target_pos.get("z", 0)], dtype=np.float32)
                tp_world = self._robot_base_to_world(target_rb.tolist(), robot_state)
                self._current_target_world = np.array(tp_world, dtype=np.float64)
            else:
                self._current_target_world = eef_world.copy()
        elif prim_type == "grip":
            # Use next move target for delta_base if available (client pre-populates target_position)
            # This lets VLA maintain direction awareness during grip primitives
            if target_pos and isinstance(target_pos, dict):
                dir_rb = np.array([target_pos.get("x", 0), target_pos.get("y", 0), target_pos.get("z", 0)], dtype=np.float32)
                delta_base = dir_rb - eef_base
                if self._use_direction_normalization:
                    delta_norm = np.linalg.norm(delta_base)
                    if delta_norm > 1e-6:
                        delta_base = delta_base / delta_norm
                parts.append(delta_base.tolist())
            else:
                parts.append([0.0, 0.0, 0.0])
            self._current_target_world = eef_world.copy()
        elif _DIAG_FIXED_DIRECTION is not None:
            # Diagnostic: use fixed direction vector (training mean)
            parts.append(list(_DIAG_FIXED_DIRECTION))
        elif target_pos and isinstance(target_pos, dict):
            # Priority 1: direction_vector — use directly as delta (no target subtraction)
            # This encodes the training-data motion direction independent of absolute position
            dir_vec = primitive.get("direction_vector")
            if dir_vec and isinstance(dir_vec, dict):
                dvx = float(dir_vec.get("x", 0))
                dvy = dir_vec.get("y")  # None → compute from perception target
                dvz = float(dir_vec.get("z", 0))
                if dvy is None:
                    # Y direction adapts to actual fixture position in this kitchen
                    dvy = float(target_pos.get("y", 0)) - eef_base[1]
                else:
                    dvy = float(dvy)
                delta_base = np.array([dvx, dvy, dvz], dtype=np.float32)
            else:
                # Priority 2: direction_target override (absolute position)
                dir_override = primitive.get("direction_target")
                if dir_override and isinstance(dir_override, dict):
                    dir_target = dir_override
                elif self._use_ultimate_target:
                    ult_target = primitive.get("ultimate_target")
                    dir_target = ult_target if (ult_target and isinstance(ult_target, dict)) else target_pos
                else:
                    dir_target = target_pos
                dir_rb = np.array([dir_target.get("x", 0), dir_target.get("y", 0), dir_target.get("z", 0)], dtype=np.float32)
                delta_base = dir_rb - eef_base

            # Direction normalization
            if self._use_direction_normalization:
                delta_norm = np.linalg.norm(delta_base)
                if delta_norm > 1e-6:
                    delta_base = delta_base / delta_norm
            parts.append(delta_base.tolist())

            # Convergence target from target_position (perception)
            target_rb = np.array([target_pos.get("x", 0), target_pos.get("y", 0), target_pos.get("z", 0)], dtype=np.float32)
            tp_world = self._robot_base_to_world(target_rb.tolist(), robot_state)
            self._current_target_world = np.array(tp_world, dtype=np.float64)
        else:
            # no target: delta = 0
            parts.append([0.0, 0.0, 0.0])
            self._current_target_world = eef_world.copy()

        flat = []
        for part in parts:
            flat.extend(part)

        state = np.array(flat, dtype=np.float32)

        # Debug: log state vector components once per primitive reset
        dbg_count = getattr(self, '_state_dbg_count', 0) + 1
        self._state_dbg_count = dbg_count
        if dbg_count <= 3 or dbg_count % 100 == 0:
            logger.info(
                f"State12D [step {dbg_count}]: "
                f"eef_base={[round(v, 3) for v in state[:3]]}, "
                f"delta_base={[round(v, 3) for v in state[9:12]]}, "
                f"base_pos={robot_state.get('base_pos', 'N/A')}"
            )

        return state

    @staticmethod
    def _robot_base_to_world(pos_robot_base: list, robot_state: Dict) -> list:
        """Convert position from robot-base frame to world frame.

        Inverse of perception's _world_to_robot_base:
            robot_base = R_inv @ (world - base_pos)
        So:
            world = R @ robot_base + base_pos
        """
        base_pos = robot_state.get("base_pos")
        base_quat = robot_state.get("base_quat")

        if base_pos is None or base_quat is None:
            # No base pose available — assume pos is already in world frame
            return list(pos_robot_base)

        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat(base_quat)  # xyzw format
        pos_rb = np.array(pos_robot_base, dtype=np.float64)
        pos_world = rot.apply(pos_rb) + np.array(base_pos, dtype=np.float64)
        return pos_world.tolist()

    def _normalize_and_pad_state(
        self, state: np.ndarray,
    ) -> tuple:
        """Min-max normalize state to [-1, 1], pad to 64D for GROOT only.

        GROOT requires 64D padded state + state_mask for its DiT architecture.
        SmolVLA/Pi0.5 handle their own padding internally (max_state_dim=32/64),
        so we just normalize and return the original dimensionality.

        Returns:
            (state_norm_or_padded, state_mask_or_None)
        """
        s_min = np.array(self._state_stats["min"], dtype=np.float32)
        s_max = np.array(self._state_stats["max"], dtype=np.float32)

        r = s_max - s_min + 1e-8
        state_norm = np.clip(2.0 * (state - s_min) / r - 1.0, -1.0, 1.0)

        # GROOT needs 64D padding + state_mask; others handle padding internally
        is_groot = (
            self._lora_manager is not None
            and hasattr(self._lora_manager, '_backend')
            and getattr(self._lora_manager._backend, '_policy_type', None) == 'groot'
        )

        if is_groot:
            state_padded = np.zeros(64, dtype=np.float32)
            state_padded[:len(state_norm)] = state_norm
            state_mask = np.zeros(64, dtype=bool)
            state_mask[:len(state_norm)] = True
            return state_padded, state_mask

        # PI0.5: state normalization is handled by backend (discretize in prompt)
        # Pass raw 10D state (drop gripper) — no min-max normalization needed
        is_pi05 = (
            self._lora_manager is not None
            and getattr(self._lora_manager._backend, '_policy_type', None) in ('pi05', 'pi0.5')
        )
        if is_pi05:
            # Drop gripper indices [7,8] from 12D → 10D (matches training)
            if len(state) == 12:
                raw_state = state[[0,1,2,3,4,5,6,9,10,11]]
            else:
                raw_state = state
            return raw_state.astype(np.float32), None

        # SmolVLA: use mean/std normalization + drop gripper (12D → 10D) to match training
        is_smolvla = (
            self._lora_manager is not None
            and getattr(self._lora_manager._backend, '_policy_type', None) == 'smolvla'
        )
        if is_smolvla:
            # Drop gripper indices [7,8] from 12D → 10D (matches training)
            if len(state) == 12:
                state_10d = state[[0,1,2,3,4,5,6,9,10,11]]
            else:
                state_10d = state
            # Mean/std normalization (matches SmolVLAChunkDataset training)
            s_mean = np.array(self._state_stats["mean"], dtype=np.float32)
            s_std = np.array(self._state_stats["std"], dtype=np.float32)
            if len(s_mean) == 12:
                s_mean = s_mean[[0,1,2,3,4,5,6,9,10,11]]
                s_std = s_std[[0,1,2,3,4,5,6,9,10,11]]
            state_norm_ms = (state_10d - s_mean) / (s_std + 1e-8)
            return state_norm_ms.astype(np.float32), None

        # Other backends: return min-max normalized state without extra padding
        return state_norm.astype(np.float32), None

    def _action_to_command(
        self,
        action: np.ndarray,
        primitive_type: str,
        primitive: Dict,
        robot_state: Optional[Dict] = None,
    ) -> Command:
        """Convert 7D action to Command object.

        action = [rel_pos_x, rel_pos_y, rel_pos_z,
                  rel_rot_x, rel_rot_y, rel_rot_z,
                  gripper]

        When use_absolute_targets is True and an IK solver is available,
        the delta is converted to an absolute EEF target, IK is solved,
        and a joint_targets command is emitted. On IK failure, falls back
        to the original cartesian_delta command.
        """
        self._command_counter += 1

        # NOTE: HDF5 rel_pos/rel_rot are already in robot base frame, and
        # OSC_POSE also operates in base frame. No frame rotation needed.

        if primitive_type == "move":
            # Try IK path: delta → absolute EEF → joint_targets
            if self._use_absolute_targets and self._ik_solver is not None:
                ik_cmd = self._try_ik_command(action, robot_state or {})
                if ik_cmd is not None:
                    return ik_cmd
                logger.debug("IK failed, falling back to cartesian_delta")

            # Default: cartesian velocity command (Eq.8: u_t = [δx; δφ] / Δt)
            dt = 1.0 / self._control_rate_hz
            velocity_pos = (action[:3] / dt).tolist()
            velocity_rot = (action[3:6] / dt).tolist()
            point = TrajectoryPoint(
                positions=action[:3].tolist(),  # rel_pos delta (for reference)
                rotations=action[3:6].tolist(),  # rel_rot delta (for reference)
                velocities=velocity_pos + velocity_rot,  # Cartesian velocity [v_x,v_y,v_z,ω_x,ω_y,ω_z]
                time_from_start=dt,
            )
            # Include gripper value from VLA output (action[6]) to maintain
            # gripper state during move, matching direct mode behavior
            gripper_cmd = None
            if len(action) > 6:
                gripper_value = float(action[6])
                gripper_cmd = {
                    "action": "close" if gripper_value > 0 else "open",
                    "width": 0.0 if gripper_value > 0 else 0.08,
                    "raw_value": gripper_value,
                }
            return Command(
                command_id=f"vla_cmd_{self._command_counter:04d}",
                command_type="cartesian_delta",
                points=[point],
                frame_id=self._frame_convention,
                gripper_command=gripper_cmd,
                metadata={
                    "primitive_type": "move",
                    "vla_adapter": self._lora_manager.get_current_adapter(),
                },
            )

        elif primitive_type == "grip":
            # Use VLA-predicted gripper action
            gripper_value = float(action[6]) if len(action) > 6 else 0.0
            grip_action = "close" if gripper_value > 0 else "open"
            grip_width = 0.0 if gripper_value > 0 else 0.08

            # Also include position delta for grip primitives (VLA may output arm motion)
            point = TrajectoryPoint(
                positions=action[:3].tolist(),
                rotations=action[3:6].tolist(),
                time_from_start=1.0 / self._control_rate_hz,
            )

            return Command(
                command_id=f"vla_cmd_{self._command_counter:04d}",
                command_type="gripper",
                points=[point],
                frame_id=self._frame_convention,
                gripper_command={
                    "action": grip_action,
                    "width": grip_width,
                    "raw_value": gripper_value,
                },
                metadata={
                    "primitive_type": "grip",
                    "gripper_raw": gripper_value,
                    "vla_adapter": self._lora_manager.get_current_adapter(),
                },
            )

        else:
            logger.warning(f"Unknown primitive type: {primitive_type}")
            return Command(
                command_id=f"vla_cmd_{self._command_counter:04d}",
                command_type="noop",
                points=[],
                frame_id=self._frame_convention,
            )

    @staticmethod
    def _compose_orientation(
        current_quat: np.ndarray,
        delta_euler: np.ndarray,
    ) -> np.ndarray:
        """Compose current quaternion with euler delta rotation.

        Implements: q_new = q_current * q_delta  (Hamilton product)

        Args:
            current_quat: Current orientation as [x, y, z, w] quaternion.
            delta_euler: Rotation delta as [δroll, δpitch, δyaw] in radians.

        Returns:
            New orientation as [x, y, z, w] quaternion (normalized).
        """
        from scipy.spatial.transform import Rotation

        # Skip composition if delta is negligible (< 0.001 rad ≈ 0.06°)
        if np.linalg.norm(delta_euler) < 1e-3:
            return current_quat

        # scipy Rotation uses [x, y, z, w] scalar-last convention
        r_current = Rotation.from_quat(current_quat)
        r_delta = Rotation.from_euler("xyz", delta_euler)

        # q_new = q_current * q_delta (apply delta in body frame)
        r_new = r_current * r_delta

        return r_new.as_quat()  # [x, y, z, w]

    def _try_ik_command(
        self,
        action: np.ndarray,
        robot_state: Dict,
    ) -> Optional[Command]:
        """Attempt to convert VLA delta action to joint_targets via IK.

        1. Read current EE pose from robot_state
        2. Add position delta and compose rotation delta to get absolute target
        3. Solve IK
        4. Return joint_targets Command, or None on failure

        Returns:
            Command with command_type="joint_targets", or None if IK fails.
        """
        from .ik_solver import EEFTarget

        # Extract current EE pose
        ee_pose = robot_state.get("ee_pose", {})
        ee_pos_dict = ee_pose.get("position", {})
        current_pos = np.array([
            ee_pos_dict.get("x", 0.0),
            ee_pos_dict.get("y", 0.0),
            ee_pos_dict.get("z", 0.0),
        ])

        ee_ori_dict = ee_pose.get("orientation", {})
        current_ori = np.array([
            ee_ori_dict.get("x", 0.0),
            ee_ori_dict.get("y", 0.0),
            ee_ori_dict.get("z", 0.0),
            ee_ori_dict.get("w", 1.0),
        ])

        # Position: delta → absolute (Eq.9: x_{t+1} = x_t + δx)
        delta_pos = action[:3]
        abs_pos = current_pos + delta_pos

        # Orientation: pass current orientation to IK (Eq.9: q* = IK(x_{t+1}, q_t))
        # Rotation delta δφ is applied only in the velocity path (Eq.8),
        # not in the IK path.
        target = EEFTarget(
            position=abs_pos,
            orientation=current_ori,
            gripper_state=float(action[6]) if len(action) > 6 else None,
        )

        # Current joint positions for IK seed
        current_joints = None
        joint_pos_list = robot_state.get("joint_positions")
        if joint_pos_list is not None:
            current_joints = np.array(joint_pos_list, dtype=np.float64)

        result = self._ik_solver.solve(
            target=target,
            current_joints=current_joints,
            current_ee_pose=ee_pose,
        )

        if not result.success:
            return None

        # PassthroughIKSolver returns joint_positions=None → env handles IK
        if result.joint_positions is None:
            # Emit cartesian_absolute so the env can do its own IK
            dt = 1.0 / self._control_rate_hz
            point = TrajectoryPoint(
                positions=abs_pos.tolist(),
                rotations=current_ori.tolist(),
                velocities=None,
                time_from_start=dt,
            )
            return Command(
                command_id=f"vla_cmd_{self._command_counter:04d}",
                command_type="cartesian_absolute",
                points=[point],
                frame_id=self._frame_convention,
                gripper_command=None,
                metadata={
                    "primitive_type": "move",
                    "ik_solver": "passthrough",
                    "vla_adapter": self._lora_manager.get_current_adapter(),
                },
            )

        # Real IK solution → joint_targets command
        dt = 1.0 / self._control_rate_hz
        point = TrajectoryPoint(
            positions=result.joint_positions.tolist(),
            velocities=None,
            time_from_start=dt,
        )
        return Command(
            command_id=f"vla_cmd_{self._command_counter:04d}",
            command_type="joint_targets",
            points=[point],
            frame_id=self._frame_convention,
            gripper_command=None,
            metadata={
                "primitive_type": "move",
                "ik_solver": type(self._ik_solver).__name__,
                "ik_residual": result.residual,
                "vla_adapter": self._lora_manager.get_current_adapter(),
            },
        )

    def unload(self) -> None:
        """Release resources."""
        if self._lora_manager:
            self._lora_manager.unload()
