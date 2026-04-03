"""
Custom Monitor Wrapper (Feedback)

Wrap your own VLM/feedback model to use with RoboBridge.
Runs continuously at a configurable rate, outputting ONLY:
- success: bool (binary success/fail)
- confidence: float (0.0-1.0)

On failure detection:
1. Robot is stopped immediately via /robot/stop topic
2. Failure signal is published for recovery handling

Recovery decisions (where to restart) are made externally by user or
Planner - NOT by this module.

Simply inherit this class and implement the `load_model` and `observe` methods.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from robobridge.modules.base import BaseModule

logger = logging.getLogger(__name__)


@dataclass
class FeedbackResult:
    """
    Simplified feedback observation result.

    Only contains success/fail and confidence.
    No decision-making - that's handled externally.
    """

    success: bool  # Binary: True = success, False = failure
    confidence: float  # 0.0 - 1.0
    consecutive_failures: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "consecutive_failures": self.consecutive_failures,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class CustomMonitor(BaseModule):
    """
    Base wrapper for custom feedback/VLM models with continuous monitoring.

    Simplified to output ONLY:
    - success: bool (binary success/fail)
    - confidence: float (0.0-1.0)

    On failure:
    1. Robot is stopped immediately via /robot/stop topic
    2. Failure signal is published

    Recovery decisions are handled externally - NOT by this module.

    To use your own model:
    1. Inherit this class
    2. Implement `load_model()` to load your model
    3. Implement `observe()` for rapid binary classification

    Example:
        class MyFeedback(CustomMonitor):
            def load_model(self):
                self.model = load_vlm(self.model_path)
                self.model.to(self.device)

            def observe(self, rgb, plan=None, current_step=None):
                output = self.model.quick_check(rgb)
                return FeedbackResult(
                    success=output["ok"],
                    confidence=output["confidence"]
                )

    Input Topics:
        - /camera/rgb: RGB image for continuous visual monitoring
        - /planning/high_level_plan: Current plan (for context)

    Output Topics:
        - /feedback/failure_signal: Published when failure detected
        - /robot/stop: Published immediately on failure to halt robot
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        # Connection settings
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51005),
        auth_token: Optional[str] = None,
        # Topic settings
        rgb_topic: str = "/camera/rgb",
        plan_topic: str = "/planning/high_level_plan",
        output_topic: str = "/feedback/failure_signal",
        robot_stop_topic: str = "/robot/stop",
        # Continuous monitoring settings
        observation_rate_hz: float = 10.0,
        only_publish_on_failure: bool = True,
        failure_confidence_threshold: float = 0.7,
        stop_on_consecutive_failures: int = 2,
        # Model settings
        image_size: int = 224,
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
        self.image_size = image_size

        # Topics
        self.rgb_topic = rgb_topic
        self.plan_topic = plan_topic
        self.output_topic = output_topic
        self.robot_stop_topic = robot_stop_topic

        # Continuous monitoring settings
        self.observation_rate_hz = observation_rate_hz
        self.only_publish_on_failure = only_publish_on_failure
        self.failure_confidence_threshold = failure_confidence_threshold
        self.stop_on_consecutive_failures = stop_on_consecutive_failures

        # State
        self._model: Any = None
        self._latest_rgb: Any = None
        self._current_plan: Optional[Dict] = None
        self._current_step_idx: int = 0

        # Monitoring state
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active: bool = False
        self._monitoring_lock = threading.Lock()
        self._consecutive_failures: int = 0
        self._observation_count: int = 0
        self._failure_count: int = 0
        self._last_observation_time: float = 0.0

    @abstractmethod
    def load_model(self) -> None:
        """
        Load your custom feedback/VLM model.

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
    def observe(
        self,
        rgb: Any,
        plan: Optional[Dict] = None,
        current_step: Optional[Dict] = None,
    ) -> FeedbackResult:
        """
        Perform rapid binary observation (success/fail).

        This method is called continuously at `observation_rate_hz`.
        Should be optimized for speed - focus on binary classification only.

        IMPORTANT: Only return success and confidence. Do NOT include
        decision-making logic (retry/replan/stop) - that's handled externally.

        Args:
            rgb: Current RGB image observation
            plan: Current high-level plan (for context)
            current_step: Current step being executed (if known)
                Format: {"skill": str, "target_object": str}

        Returns:
            FeedbackResult with ONLY:
                - success: Whether execution looks successful (True/False)
                - confidence: Confidence in the observation (0.0-1.0)

        Example:
            def observe(self, rgb, plan=None, current_step=None):
                img = self._fast_preprocess(rgb)
                skill = current_step.get("skill", "") if current_step else ""

                with torch.no_grad():
                    output = self._model.quick_classify(img, skill)

                success = output["success_prob"] > 0.5
                return FeedbackResult(
                    success=success,
                    confidence=output["success_prob"] if success else 1 - output["success_prob"]
                )
        """
        pass

    def start(self) -> None:
        """Start monitor module with continuous monitoring."""
        logger.info(f"Loading custom monitor model from: {self.model_path}")
        self.load_model()
        logger.info("Custom monitor model loaded successfully")

        super().start()

        # Register topic handlers
        self.subscribe(self.rgb_topic, self._on_rgb)
        self.subscribe(self.plan_topic, self._on_plan)

        # Start continuous monitoring
        self._start_monitoring()
        logger.info(f"Started continuous monitoring at {self.observation_rate_hz} Hz")

    def stop(self) -> None:
        """Stop monitor module and monitoring thread."""
        self._stop_monitoring()
        super().stop()

    def _start_monitoring(self) -> None:
        """Start the continuous monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="CustomMonitorLoop",
            daemon=True,
        )
        self._monitoring_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop the continuous monitoring thread."""
        self._monitoring_active = False
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None
        logger.info("Monitoring thread stopped")

    def _monitoring_loop(self) -> None:
        """Main continuous monitoring loop - runs independently."""
        interval = 1.0 / self.observation_rate_hz

        while self._monitoring_active:
            loop_start = time.time()

            try:
                # Get current state (thread-safe)
                with self._monitoring_lock:
                    rgb = self._latest_rgb
                    plan = self._current_plan

                # Skip if no image available
                if rgb is None:
                    time.sleep(interval)
                    continue

                # Get current step info
                current_step = self._get_current_step(plan)

                # Perform observation using custom model
                result = self.observe(rgb=rgb, plan=plan, current_step=current_step)
                self._observation_count += 1

                # Log observation
                logger.debug(
                    f"Observation #{self._observation_count}: "
                    f"success={result.success}, confidence={result.confidence:.2f}"
                )

                # Handle failure
                if not result.success:
                    self._failure_count += 1
                    self._consecutive_failures += 1
                    result.consecutive_failures = self._consecutive_failures

                    # Check confidence threshold
                    if result.confidence >= self.failure_confidence_threshold:
                        # IMMEDIATELY stop the robot
                        self._publish_robot_stop()

                        # Publish failure signal
                        self._publish_failure(result)

                        logger.warning(
                            f"FAILURE DETECTED - Robot stopped! "
                            f"(confidence: {result.confidence:.2f}, "
                            f"consecutive: {result.consecutive_failures})"
                        )
                else:
                    # Reset consecutive failures on success
                    self._consecutive_failures = 0

                    # Optionally publish success observations
                    if not self.only_publish_on_failure:
                        self._publish_observation(result)

                self._last_observation_time = time.time()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Maintain observation rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _get_current_step(self, plan: Optional[Dict]) -> Optional[Dict]:
        """Get current step info from plan (simplified)."""
        if not plan:
            return None

        steps = plan.get("steps", [])
        current_idx = plan.get("current_step_index", self._current_step_idx)

        if 0 <= current_idx < len(steps):
            step = steps[current_idx]
            return {
                "skill": step.get("skill", "unknown"),
                "target_object": step.get("target_object", "unknown"),
            }
        return None

    def _publish_robot_stop(self) -> None:
        """
        Publish immediate stop signal to halt robot motion.

        This is called IMMEDIATELY when a failure is detected.
        """
        stop_payload = {
            "command": "stop",
            "reason": "failure_detected",
            "timestamp": time.time(),
            "consecutive_failures": self._consecutive_failures,
        }
        self.publish(self.robot_stop_topic, stop_payload, None)
        logger.warning("ROBOT STOP signal published!")

    def _publish_failure(self, result: FeedbackResult) -> None:
        """Publish failure signal."""
        payload = {
            "data": json.dumps(result.to_dict()),
            "timestamp": time.time(),
            "observation_count": self._observation_count,
            "failure_count": self._failure_count,
        }
        self.publish(self.output_topic, payload, None)
        logger.info(f"Published failure signal (confidence: {result.confidence:.2f})")

    def _publish_observation(self, result: FeedbackResult) -> None:
        """Publish observation (success or failure)."""
        payload = {
            "data": json.dumps(result.to_dict()),
            "timestamp": time.time(),
            "observation_count": self._observation_count,
        }
        self.publish(self.output_topic, payload, None)

    def _on_rgb(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle RGB image message."""
        with self._monitoring_lock:
            self._latest_rgb = payload

    def _on_plan(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle plan message."""
        with self._monitoring_lock:
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
                    self._current_plan = None

            self._current_step_idx = 0
            self._consecutive_failures = 0

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "observation_count": self._observation_count,
            "failure_count": self._failure_count,
            "consecutive_failures": self._consecutive_failures,
            "monitoring_active": self._monitoring_active,
            "observation_rate_hz": self.observation_rate_hz,
            "last_observation_time": self._last_observation_time,
        }

    def reset_stats(self) -> None:
        """Reset monitoring statistics."""
        with self._monitoring_lock:
            self._observation_count = 0
            self._failure_count = 0
            self._consecutive_failures = 0

    def process(self, *args, **kwargs) -> Any:
        """Required by BaseModule - use observe() for continuous monitoring."""
        return self.observe(*args, **kwargs)
