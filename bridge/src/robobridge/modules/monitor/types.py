"""
Monitor module types and data classes.

Defines feedback results and configuration for execution monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
    recovery_target: Optional[str] = None  # "perception", "planning", "controller", "retry"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "consecutive_failures": self.consecutive_failures,
            "recovery_target": self.recovery_target,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackResult":
        return cls(
            success=data.get("success", False),
            confidence=data.get("confidence", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            recovery_target=data.get("recovery_target"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MonitorConfig:
    """Monitor module specific configuration."""

    temperature: float = 0.0
    max_tokens: int = 200
    image_size: int = 224
    output_format: str = "json"
    base_prompt: str = ""
    # Input topics
    rgb_topic: str = "/camera/rgb"
    plan_topic: str = "/planning/high_level_plan"
    exec_result_topic: str = "/robot/execution_result"
    # Output topics
    output_topic: str = "/feedback/failure_signal"
    robot_stop_topic: str = "/robot/stop"
    # Continuous monitoring settings
    observation_rate_hz: float = 10.0  # How fast to observe (10 Hz default)
    only_publish_on_failure: bool = True  # Only send signals on failure
    failure_confidence_threshold: float = 0.7  # Min confidence to trigger failure
    enable_continuous_mode: bool = True  # Enable continuous monitoring
    stop_on_consecutive_failures: int = 2  # Auto-stop after N consecutive failures


@dataclass
class MonitoringStats:
    """Monitoring statistics."""

    observation_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    monitoring_active: bool = False
    observation_rate_hz: float = 10.0
    last_observation_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "observation_count": self.observation_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "monitoring_active": self.monitoring_active,
            "observation_rate_hz": self.observation_rate_hz,
            "last_observation_time": self.last_observation_time,
        }
