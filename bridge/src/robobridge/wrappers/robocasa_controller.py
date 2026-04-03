"""
RoboCasa Controller Wrapper.

Executes primitive actions (move, grip) in RoboCasa simulation environment.
Abstracts low-level control details like action dimensions and gripper mapping.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExecutionResult:
    """Result of primitive execution."""
    success: bool
    steps_taken: int
    final_distance: float = 0.0
    grasp_detected: bool = False
    message: str = ""


class RoboCasaController:
    """
    Controller for executing primitives in RoboCasa environment.

    Handles:
    - Move: Delta position control with IK controller
    - Grip: Gripper open/close with position hold

    All coordinates are in robot-base frame.
    """

    def __init__(
        self,
        env: Any,
        grip_open_width: float = 0.08,
        position_threshold: float = 0.01,
        position_tolerance: float = 0.08,
        max_move_steps: int = 200,
        grip_steps_close: int = 50,
        grip_steps_open: int = 30,
        settle_steps: int = 30,
        position_gain: float = 2.0,
    ):
        """
        Initialize RoboCasa controller.

        Args:
            env: RoboCasa environment instance
            grip_open_width: Gripper width for open state (default: 0.08m)
            position_threshold: Distance threshold for reaching target (default: 1cm)
            position_tolerance: Maximum acceptable final distance (default: 8cm)
            max_move_steps: Maximum steps for move primitive
            grip_steps_close: Steps for closing gripper
            grip_steps_open: Steps for opening gripper
            settle_steps: Steps to wait after releasing object
            position_gain: Gain for position control
        """
        self.env = env
        self.grip_open_width = grip_open_width
        self.position_threshold = position_threshold
        self.position_tolerance = position_tolerance
        self.max_move_steps = max_move_steps
        self.grip_steps_close = grip_steps_close
        self.grip_steps_open = grip_steps_open
        self.settle_steps = settle_steps
        self.position_gain = position_gain

        # Detect action dimension and gripper index from environment
        self.action_dim = env.action_spec[0].shape[0]
        self.gripper_idx = 6  # Standard for Panda arm

        # Current observation (updated after each step)
        self._obs: Optional[Dict] = None

        # Step counter
        self.total_steps = 0

        # Optional callback for frame recording
        self._frame_callback: Optional[Callable] = None

    def set_observation(self, obs: Dict) -> None:
        """Update current observation."""
        self._obs = obs

    def set_frame_callback(self, callback: Callable[[], None]) -> None:
        """Set callback function called after each simulation step."""
        self._frame_callback = callback

    def _step(self, action: np.ndarray) -> Dict:
        """Execute one simulation step and update observation."""
        self._obs, _, _, _ = self.env.step(action)
        self.total_steps += 1

        if self._frame_callback:
            self._frame_callback()

        return self._obs

    def _get_eef_pos(self) -> np.ndarray:
        """Get current end-effector position in robot-base frame."""
        return self._obs["robot0_base_to_eef_pos"].copy()

    def _get_gripper_opening(self) -> float:
        """Get current gripper opening width."""
        qpos = self._obs["robot0_gripper_qpos"]
        return abs(qpos[0]) + abs(qpos[1])

    def _is_gripper_closed(self) -> bool:
        """Check if gripper is closed (holding something)."""
        return self._get_gripper_opening() < 0.079

    def execute_primitive(self, primitive: Dict) -> ExecutionResult:
        """
        Execute a single primitive action.

        Args:
            primitive: Primitive dict with keys:
                - primitive_type: "move", "grip", or "go"
                - target_position: {x, y, z} for move
                - target_rotation: {roll, pitch, yaw} for move (optional, degrees)
                - grip_width: float for grip

        Returns:
            ExecutionResult with success status and details
        """
        ptype = primitive.get("primitive_type", "")

        if ptype == "grip":
            return self.execute_grip(primitive.get("grip_width", 0.5))
        elif ptype == "move":
            target = primitive.get("target_position", {})
            if not target:
                return ExecutionResult(False, 0, message="No target position")
            pos = np.array([target.get("x", 0), target.get("y", 0), target.get("z", 0)])
            target_rot = primitive.get("target_rotation")
            rotation = None
            if target_rot:
                rotation = np.array([
                    target_rot.get("roll", 0.0),
                    target_rot.get("pitch", 0.0),
                    target_rot.get("yaw", 0.0),
                ])
            return self.execute_move(pos, rotation=rotation)
        elif ptype == "go":
            # Mobile base movement - not implemented
            return ExecutionResult(True, 0, message="GO skipped (no mobile base)")
        else:
            return ExecutionResult(False, 0, message=f"Unknown primitive: {ptype}")

    def execute_move(
        self,
        target: np.ndarray,
        rotation: Optional[np.ndarray] = None,
    ) -> ExecutionResult:
        """
        Execute move primitive to target position and optional rotation.

        Uses delta position control with IK controller.
        Maintains gripper state during movement.

        Args:
            target: Target position [x, y, z] in robot-base frame
            rotation: Target rotation [roll, pitch, yaw] in degrees (optional).
                      Converted to axis-angle delta for action[3:6].

        Returns:
            ExecutionResult
        """
        if self._obs is None:
            return ExecutionResult(False, 0, message="No observation set")

        steps_taken = 0

        for i in range(self.max_move_steps):
            current_pos = self._get_eef_pos()
            error = target - current_pos
            dist = np.linalg.norm(error)

            if dist < self.position_threshold:
                return ExecutionResult(
                    success=True,
                    steps_taken=steps_taken,
                    final_distance=dist,
                    message=f"Reached target in {steps_taken} steps"
                )

            # Calculate delta command
            delta = error * self.position_gain
            delta = np.clip(delta, -1.0, 1.0)

            # Build action
            action = np.zeros(self.action_dim)
            action[0:3] = delta

            # Apply rotation if provided (degrees -> scaled for controller)
            if rotation is not None:
                rot_scaled = np.deg2rad(rotation) * self.position_gain
                rot_scaled = np.clip(rot_scaled, -1.0, 1.0)
                action[3:6] = rot_scaled

            # Maintain gripper state
            if self._is_gripper_closed():
                action[self.gripper_idx] = 1.0  # Keep closed
            else:
                action[self.gripper_idx] = -1.0  # Keep open

            self._step(action)
            steps_taken += 1

        # Check final distance
        final_dist = np.linalg.norm(target - self._get_eef_pos())
        success = final_dist < self.position_tolerance

        return ExecutionResult(
            success=success,
            steps_taken=steps_taken,
            final_distance=final_dist,
            message=f"Move completed (dist={final_dist:.4f})"
        )

    def execute_grip(self, grip_width: float) -> ExecutionResult:
        """
        Execute grip primitive.

        Gripper mapping:
        - grip_width = 0.08 (open) -> action = -1
        - grip_width = 0.00 (closed) -> action = +1

        Args:
            grip_width: Target gripper width (0=closed, 0.08=open)

        Returns:
            ExecutionResult with grasp_detected if closing
        """
        if self._obs is None:
            return ExecutionResult(False, 0, message="No observation set")

        is_closing = grip_width < 0.01

        # Map grip_width to action: 0.08->-1, 0.0->+1
        gripper_action = 1.0 - (grip_width / 0.04)
        gripper_action = np.clip(gripper_action, -1.0, 1.0)

        # Hold current position during grip
        target_pos = self._get_eef_pos()

        num_steps = self.grip_steps_close if is_closing else self.grip_steps_open
        steps_taken = 0

        for _ in range(num_steps):
            # Position hold with active correction
            current_pos = self._get_eef_pos()
            pos_error = target_pos - current_pos
            pos_correction = np.clip(pos_error * self.position_gain, -0.5, 0.5)

            action = np.zeros(self.action_dim)
            action[0:3] = pos_correction
            action[self.gripper_idx] = gripper_action

            self._step(action)
            steps_taken += 1

        # Check grasp result
        grasp_detected = False
        if is_closing:
            gripper_opening = self._get_gripper_opening()
            grasp_detected = gripper_opening > 0.01
        else:
            # Wait for object to settle after release
            for _ in range(self.settle_steps):
                current_pos = self._get_eef_pos()
                pos_error = target_pos - current_pos
                pos_correction = np.clip(pos_error * self.position_gain, -0.5, 0.5)

                action = np.zeros(self.action_dim)
                action[0:3] = pos_correction
                action[self.gripper_idx] = -1.0  # Keep open

                self._step(action)
                steps_taken += 1

        return ExecutionResult(
            success=True,
            steps_taken=steps_taken,
            grasp_detected=grasp_detected,
            message="GRASP DETECTED" if grasp_detected else "Grip completed"
        )

    def execute_plan(
        self,
        primitive_plans: List[Any],
        skip_actions: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[bool, List[ExecutionResult]]:
        """
        Execute a list of primitive plans.

        Args:
            primitive_plans: List of PrimitivePlan objects
            skip_actions: List of (action_type, target_keyword) tuples to skip

        Returns:
            (overall_success, list_of_results)
        """
        skip_actions = skip_actions or []
        results = []
        overall_success = True

        for plan in primitive_plans:
            action = plan.parent_action

            # Check if this action should be skipped
            should_skip = any(
                action.action_type == skip_type and skip_kw.lower() in action.target_object.lower()
                for skip_type, skip_kw in skip_actions
            )

            if should_skip:
                continue

            # Execute all primitives in this plan
            for prim in plan.primitives:
                prim_dict = prim.to_dict()
                result = self.execute_primitive(prim_dict)
                results.append(result)

                if not result.success:
                    overall_success = False
                    break

            if not overall_success:
                break

        return overall_success, results
