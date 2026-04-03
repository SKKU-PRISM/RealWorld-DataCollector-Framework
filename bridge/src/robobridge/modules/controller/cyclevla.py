"""
CycleVLA inference pipeline for GROOT N1.5.

Implements CycleVLA paper (arxiv 2601.02295) features:
- MBR (Minimum Bayes Risk) decoding for action selection
- Subtask state machine with progress/stop signals
- VLM failure prediction at subtask transitions
- Subtask backtracking on failure detection
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SubtaskState(Enum):
    EXECUTING = "executing"
    TRANSITIONING = "transitioning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubtaskCheckpoint:
    """Snapshot at subtask start for backtracking."""
    eef_pos: np.ndarray  # (3,) end-effector position
    gripper_state: float  # gripper value
    subtask_idx: int
    step_count: int = 0


@dataclass
class CycleVLAConfig:
    """Configuration for CycleVLA inference."""
    mbr_samples: int = 8
    stop_threshold: float = 0.5  # stop signal threshold for transition check
    stop_complete_threshold: float = 0.8  # stop signal for subtask completion
    progress_threshold: float = 0.9  # progress threshold
    max_backtracks: int = 2  # max retries per subtask
    use_vlm_predictor: bool = True  # whether to use VLM failure prediction
    vlm_provider: str = "bedrock"
    vlm_model: str = "eu.anthropic.claude-sonnet-4-20250514"
    action_dim: int = 9


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""
    success: bool
    subtask_idx: int
    steps_executed: int
    backtrack_count: int = 0
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MBRDecoder:
    """Minimum Bayes Risk decoder for action chunk selection.

    Samples N action chunks with different random seeds,
    computes pairwise L2 distances, selects consensus trajectory.
    """

    def __init__(self, n_samples: int = 8, action_dim: int = 9):
        self.n_samples = n_samples
        self.action_dim = action_dim

    def decode(self, policy, batch: dict) -> np.ndarray:
        """Sample N trajectories and return consensus via MBR.

        Args:
            policy: GROOT policy with predict_action_chunk() method
            batch: Input batch dict for policy

        Returns:
            Best trajectory as numpy array (chunk_size, action_dim)
        """
        trajectories = []

        # Save RNG state
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        try:
            for i in range(self.n_samples):
                # Set deterministic seed for this sample
                torch.manual_seed(42 + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + i)

                # Get full action chunk from policy (bfloat16 autocast like pipeline)
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        action_chunk = policy.predict_action_chunk(batch)  # (B, T, D)

                # Convert to float32 before numpy (matches pipeline)
                traj = action_chunk[0, :, :self.action_dim].float().cpu().numpy()
                trajectories.append(traj)

                # Clear internal state (matches pipeline's predict())
                if hasattr(policy, 'reset'):
                    policy.reset()
        finally:
            # Restore RNG state
            torch.random.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

        # Compute pairwise L2 distances
        n = len(trajectories)
        costs = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    costs[i] += np.linalg.norm(
                        trajectories[i].flatten() - trajectories[j].flatten()
                    )

        best_idx = int(np.argmin(costs))

        logger.debug(
            f"MBR decode: {n} samples, costs range [{costs.min():.3f}, {costs.max():.3f}], "
            f"selected idx={best_idx}"
        )

        return trajectories[best_idx]


class VLMFailurePredictor:
    """VLM-based failure prediction at subtask transitions.

    Queries a VLM to determine if the current subtask is likely to
    succeed or fail based on visual observation.
    """

    def __init__(self, provider: str = "bedrock", model: str = "eu.anthropic.claude-sonnet-4-20250514"):
        self.provider = provider
        self.model = model
        self._client = None

    def initialize(self):
        """Initialize VLM client."""
        # Use the same VLM infrastructure as Monitor module
        try:
            if self.provider == "bedrock":
                import boto3
                self._client = boto3.client("bedrock-runtime", region_name="eu-west-1")
            else:
                logger.warning(f"VLM provider {self.provider} not fully implemented, using dummy predictor")
        except Exception as e:
            logger.warning(f"Failed to initialize VLM client: {e}, using dummy predictor")

    def predict_failure(
        self,
        image: np.ndarray,
        task_description: str,
        subtask_idx: int,
        subtask_total: int,
        progress: float,
    ) -> Tuple[bool, float]:
        """Predict whether current subtask will fail.

        Args:
            image: Current RGB observation (H, W, 3)
            task_description: Natural language task description
            subtask_idx: Current subtask index
            subtask_total: Total number of subtasks
            progress: Current progress within subtask

        Returns:
            (is_failing, confidence) tuple
        """
        if self._client is None:
            # Dummy predictor: always predict success
            return False, 0.5

        try:
            import base64
            import json
            import re
            from io import BytesIO

            from PIL import Image

            # Encode image
            pil_img = Image.fromarray(image)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            prompt = (
                f"You are a robot manipulation failure predictor. "
                f"Task: {task_description}. "
                f"Current subtask: {subtask_idx + 1}/{subtask_total} (progress: {progress:.0%}). "
                f"Based on the image, is this subtask likely to FAIL? "
                f'Respond with JSON: {{"failing": true/false, "confidence": 0.0-1.0, "reason": "brief"}}'
            )

            if self.provider == "bedrock":
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "temperature": 0.0,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                })

                response = self._client.invoke_model(
                    modelId=self.model,
                    body=body,
                )
                result = json.loads(response["body"].read())
                text = result["content"][0]["text"]

                # Parse JSON from response
                json_match = re.search(r'\{[^}]+\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return parsed.get("failing", False), parsed.get("confidence", 0.5)

            return False, 0.5

        except Exception as e:
            logger.warning(f"VLM failure prediction error: {e}")
            return False, 0.5


class SubtaskStateMachine:
    """Manages subtask state transitions based on VLA output signals.

    Tracks progress and stop signals from 9D VLA output to determine
    when to transition between subtasks.
    """

    def __init__(self, config: CycleVLAConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset state machine."""
        self.current_subtask_idx = 0
        self.state = SubtaskState.EXECUTING
        self._progress_history: List[float] = []
        self._stop_history: List[float] = []
        self._step_count = 0

    def update(self, stop_signal: float, progress: float) -> SubtaskState:
        """Update state based on new VLA output signals.

        Args:
            stop_signal: VLA dim 7 output (0->1 ramp)
            progress: VLA dim 8 output (0->1 within subtask)

        Returns:
            Current SubtaskState
        """
        self._step_count += 1
        self._progress_history.append(progress)
        self._stop_history.append(stop_signal)

        # Use exponential moving average for stability
        window = min(5, len(self._stop_history))
        avg_stop = np.mean(self._stop_history[-window:])
        avg_progress = np.mean(self._progress_history[-window:])

        if avg_stop >= self.config.stop_complete_threshold:
            self.state = SubtaskState.COMPLETED
        elif avg_stop >= self.config.stop_threshold:
            self.state = SubtaskState.TRANSITIONING
        else:
            self.state = SubtaskState.EXECUTING

        return self.state

    def advance_subtask(self):
        """Move to next subtask."""
        self.current_subtask_idx += 1
        self.state = SubtaskState.EXECUTING
        self._progress_history.clear()
        self._stop_history.clear()
        self._step_count = 0

    def mark_failed(self):
        """Mark current subtask as failed."""
        self.state = SubtaskState.FAILED


class CycleVLARunner:
    """Main CycleVLA execution runner.

    Wraps VLALoRAController to add CycleVLA features:
    - MBR decoding for robust action selection
    - Subtask state machine with progress tracking
    - VLM failure prediction at transitions
    - Backtracking on failure

    Usage in eval loop::

        runner = CycleVLARunner(config, controller, monitor)
        runner.initialize()

        while not done:
            action, meta = runner.step(obs, env)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        config: CycleVLAConfig,
        controller=None,  # VLALoRAController instance
        vlm_predictor: Optional[VLMFailurePredictor] = None,
    ):
        self.config = config
        self.controller = controller
        self.mbr_decoder = MBRDecoder(
            n_samples=config.mbr_samples,
            action_dim=config.action_dim,
        )
        self.state_machine = SubtaskStateMachine(config)
        self.vlm_predictor = vlm_predictor or VLMFailurePredictor(
            provider=config.vlm_provider,
            model=config.vlm_model,
        )

        # Backtracking state
        self._checkpoints: List[SubtaskCheckpoint] = []
        self._backtrack_counts: Dict[int, int] = {}  # subtask_idx -> count

        # Chunk buffer for MBR
        self._chunk_buffer: Optional[np.ndarray] = None
        self._chunk_step: int = 0
        self._chunk_stride: int = 4  # re-infer every N steps

        # Episode state
        self._total_steps: int = 0
        self._task_description: str = ""
        self._n_subtasks: int = 2  # default, updated per task

    def initialize(self):
        """Initialize all sub-components."""
        if self.config.use_vlm_predictor:
            self.vlm_predictor.initialize()
        logger.info(
            f"CycleVLA runner initialized: MBR={self.config.mbr_samples} samples, "
            f"stop_thresh={self.config.stop_threshold}, max_backtracks={self.config.max_backtracks}"
        )

    def reset_episode(self, task_description: str = "", n_subtasks: int = 2):
        """Reset for new episode.

        Args:
            task_description: Natural language task description
            n_subtasks: Expected number of subtasks for this task
        """
        self.state_machine.reset()
        self._checkpoints.clear()
        self._backtrack_counts.clear()
        self._chunk_buffer = None
        self._chunk_step = 0
        self._total_steps = 0
        self._task_description = task_description
        self._n_subtasks = n_subtasks

        if self.controller and hasattr(self.controller, '_lora_manager'):
            self.controller._lora_manager.reset_policy()

        if hasattr(self, '_last_policy') and self._last_policy is not None:
            if hasattr(self._last_policy, 'reset'):
                self._last_policy.reset()
        self._last_policy = None

    def save_checkpoint(self, eef_pos: np.ndarray, gripper_state: float):
        """Save checkpoint at subtask start."""
        checkpoint = SubtaskCheckpoint(
            eef_pos=eef_pos.copy(),
            gripper_state=gripper_state,
            subtask_idx=self.state_machine.current_subtask_idx,
        )
        self._checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint saved for subtask {checkpoint.subtask_idx} at pos={eef_pos}")

    def step(
        self,
        obs: Dict[str, Any],
        policy=None,  # GROOT policy object
        batch: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute one CycleVLA step.

        This is the main entry point called from the eval loop.

        Args:
            obs: Current observation dict with 'image', 'robot_state', etc.
            policy: GROOT policy for MBR (if None, uses controller)
            batch: Pre-built batch dict for policy

        Returns:
            (action_7d, metadata) where action_7d is the 7D robot action
            and metadata contains stop_signal, progress, subtask_state, etc.
        """
        self._total_steps += 1

        # Determine if we need a new chunk (MBR or regular)
        need_new_chunk = (
            self._chunk_buffer is None
            or self._chunk_step >= self._chunk_stride
        )

        if need_new_chunk and policy is not None and batch is not None:
            # MBR decode: get consensus action chunk
            self._last_policy = policy  # save for reset_episode cleanup
            action_chunk_9d = self.mbr_decoder.decode(policy, batch)  # (T, 9)
            self._chunk_buffer = action_chunk_9d
            self._chunk_step = 0

        # Get current action from chunk buffer
        if self._chunk_buffer is not None and self._chunk_step < len(self._chunk_buffer):
            action_9d = self._chunk_buffer[self._chunk_step]
            self._chunk_step += 1
        else:
            # Fallback: zero action
            action_9d = np.zeros(self.config.action_dim)
            logger.warning("No action chunk available, using zero action")

        # Extract signals
        action_7d = action_9d[:7]
        stop_signal = float((action_9d[7] + 1.0) / 2.0) if len(action_9d) > 7 else 0.0  # [-1,1] -> [0,1]
        progress = float((action_9d[8] + 1.0) / 2.0) if len(action_9d) > 8 else 0.0      # [-1,1] -> [0,1]

        # Clamp signals to [0, 1]
        stop_signal = np.clip(stop_signal, 0.0, 1.0)
        progress = np.clip(progress, 0.0, 1.0)

        # Update state machine
        state = self.state_machine.update(stop_signal, progress)

        metadata = {
            "stop_signal": stop_signal,
            "progress": progress,
            "subtask_state": state.value,
            "subtask_idx": self.state_machine.current_subtask_idx,
            "total_steps": self._total_steps,
            "mbr_active": policy is not None,
        }

        # Handle state transitions
        if state == SubtaskState.TRANSITIONING:
            metadata["transition_pending"] = True

        elif state == SubtaskState.COMPLETED:
            subtask_idx = self.state_machine.current_subtask_idx
            logger.info(
                f"Subtask {subtask_idx} completed at step {self._total_steps} "
                f"(stop={stop_signal:.2f}, progress={progress:.2f})"
            )

            # Check VLM before advancing
            if self.config.use_vlm_predictor and "image" in obs:
                is_failing, confidence = self.vlm_predictor.predict_failure(
                    image=obs["image"],
                    task_description=self._task_description,
                    subtask_idx=subtask_idx,
                    subtask_total=self._n_subtasks,
                    progress=progress,
                )
                metadata["vlm_failing"] = is_failing
                metadata["vlm_confidence"] = confidence

                if is_failing and confidence > 0.7:
                    self.state_machine.mark_failed()
                    metadata["subtask_state"] = SubtaskState.FAILED.value
                    state = SubtaskState.FAILED

            if state == SubtaskState.COMPLETED:
                self.state_machine.advance_subtask()
                self._chunk_buffer = None  # Force new chunk for next subtask
                metadata["subtask_advanced"] = True

        if state == SubtaskState.FAILED:
            subtask_idx = self.state_machine.current_subtask_idx
            bt_count = self._backtrack_counts.get(subtask_idx, 0)

            if bt_count < self.config.max_backtracks:
                self._backtrack_counts[subtask_idx] = bt_count + 1
                metadata["backtrack_requested"] = True
                metadata["backtrack_count"] = bt_count + 1
                logger.info(f"Subtask {subtask_idx} failed, backtrack #{bt_count + 1}")

                # Reset state machine for retry
                self.state_machine.state = SubtaskState.EXECUTING
                self.state_machine._progress_history.clear()
                self.state_machine._stop_history.clear()
                self._chunk_buffer = None
            else:
                metadata["max_backtracks_reached"] = True
                logger.warning(
                    f"Subtask {subtask_idx} exceeded max backtracks ({self.config.max_backtracks})"
                )
                # Advance anyway to avoid infinite loop
                self.state_machine.advance_subtask()
                self._chunk_buffer = None

        return action_7d, metadata

    def get_backtrack_target(self) -> Optional[SubtaskCheckpoint]:
        """Get the checkpoint to backtrack to.

        Returns:
            SubtaskCheckpoint for current subtask start, or None
        """
        subtask_idx = self.state_machine.current_subtask_idx
        for cp in reversed(self._checkpoints):
            if cp.subtask_idx == subtask_idx:
                return cp
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return {
            "total_steps": self._total_steps,
            "current_subtask": self.state_machine.current_subtask_idx,
            "current_state": self.state_machine.state.value,
            "backtrack_counts": dict(self._backtrack_counts),
            "n_checkpoints": len(self._checkpoints),
        }


# ===========================================================================
# Hybrid CycleVLA: budget-based transitions + VLM failure checks
# ===========================================================================


@dataclass
class HybridCycleVLAConfig:
    """Configuration for Hybrid CycleVLA inference."""
    mbr_samples: int = 8
    action_dim: int = 7  # 7D: pos3 + rot3 + grip1
    max_backtracks: int = 2
    vlm_check_interval: int = 50  # VLM failure check every N steps
    vlm_provider: str = "bedrock"
    vlm_model: str = "eu.anthropic.claude-sonnet-4-6-v1"


class HybridSubtaskStateMachine:
    """Budget-based subtask state machine for Hybrid CycleVLA.

    Instead of relying on VLA stop/progress signals (which don't work with 7D),
    uses steps_budget from template plan config for transitions.
    Optionally queries VLM for early completion or failure detection.
    """

    def __init__(self, primitives: List[Dict[str, Any]]):
        """
        Args:
            primitives: List of primitive dicts from task_plan_config.json.
                        Each has 'type', 'steps_budget', and optionally
                        'direction_vector' or 'direction_target'.
        """
        self.primitives = primitives
        self.current_idx = 0
        self._step_in_primitive = 0
        self.state = SubtaskState.EXECUTING

    @property
    def current_primitive(self) -> Optional[Dict[str, Any]]:
        if self.current_idx < len(self.primitives):
            return self.primitives[self.current_idx]
        return None

    @property
    def current_budget(self) -> int:
        p = self.current_primitive
        return p["steps_budget"] if p else 0

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.primitives)

    def step(self) -> SubtaskState:
        """Advance one step. Returns current state."""
        if self.is_done:
            self.state = SubtaskState.COMPLETED
            return self.state

        self._step_in_primitive += 1

        if self._step_in_primitive >= self.current_budget:
            self.state = SubtaskState.TRANSITIONING
        else:
            self.state = SubtaskState.EXECUTING

        return self.state

    def advance(self):
        """Move to next primitive."""
        self.current_idx += 1
        self._step_in_primitive = 0
        self.state = SubtaskState.EXECUTING if not self.is_done else SubtaskState.COMPLETED
        logger.info(
            f"HybridFSM: advanced to primitive {self.current_idx}/{len(self.primitives)}"
        )

    def mark_failed(self):
        self.state = SubtaskState.FAILED

    def reset_current(self):
        """Reset current primitive for retry (backtrack)."""
        self._step_in_primitive = 0
        self.state = SubtaskState.EXECUTING

    def get_progress(self) -> float:
        """Progress within current primitive (0-1)."""
        budget = self.current_budget
        if budget <= 0:
            return 1.0
        return min(1.0, self._step_in_primitive / budget)
