"""
Adaptive Direction Compensation for SO-101 Robot

Compensates for backlash and gravity effects based on:
1. Movement direction (up vs down)
2. Target z position (lower positions need stronger compensation)
3. Direction change detection with preload
4. Gravity LUT (angle-dependent static offset)

Calibrated values from mechanical diagnostics:
- shoulder_lift: backlash ~1.91 units, gravity asymmetry ~1.77 units
- elbow_flex: backlash ~2.86 units, gravity asymmetry ~2.59 units
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

# ============================================================================
# Backlash Parameters (per joint)
# ============================================================================

# Deadband: the "dead zone" width where gear play causes no movement
DEADBAND = {
    0: 0.0,    # shoulder_pan
    1: 1.91,   # shoulder_lift - measured backlash
    2: 2.86,   # elbow_flex - measured backlash
    3: 0.0,    # wrist_flex
    4: 0.0,    # wrist_roll
}

# Direction-dependent offsets (positive direction vs negative direction)
# offset_pos: correction when moving in positive direction
# offset_neg: correction when moving in negative direction
OFFSET_POS = {
    0: 0.0,
    1: -0.92,  # shoulder_lift: moving positive (down) tends to overshoot
    2: 1.48,   # elbow_flex: moving positive (extending) undershoots
    3: 0.0,
    4: 0.0,
}

OFFSET_NEG = {
    0: 0.0,
    1: 0.92,   # shoulder_lift: moving negative (up) undershoots
    2: -1.48,  # elbow_flex: moving negative (folding) overshoots
    3: 0.0,
    4: 0.0,
}

# Preload gain: extra push on direction change to overcome deadband
PRELOAD_GAIN = 0.5  # Fraction of deadband to apply as preload

# Base compensation values (legacy, for backward compatibility)
BASE_COMPENSATION = {
    0: 0.0,    # shoulder_pan - no compensation needed
    1: 1.84,   # shoulder_lift - measured difference
    2: 2.95,   # elbow_flex - measured difference
    3: 0.0,    # wrist_flex - no compensation needed
    4: 0.0,    # wrist_roll - no compensation needed
}

# ============================================================================
# Adaptive Factor (z-based)
# ============================================================================

Z_ADAPTIVE_FACTORS = {
    "low": 2.0,      # z < 0.12m
    "medium": 1.75,  # 0.12m <= z < 0.15m
    "high": 1.5,     # z >= 0.15m
}


def get_adaptive_factor(target_z: float) -> float:
    """
    Get compensation factor based on target z height.

    Args:
        target_z: Target z position in meters

    Returns:
        Compensation factor (higher for lower z positions)
    """
    if target_z < 0.12:
        return Z_ADAPTIVE_FACTORS["low"]
    elif target_z < 0.15:
        return Z_ADAPTIVE_FACTORS["medium"]
    else:
        return Z_ADAPTIVE_FACTORS["high"]


def apply_direction_compensation(
    current_norm: np.ndarray,
    target_norm: np.ndarray,
    compensation_factor: float = 1.0,
    base_compensation: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Apply direction-aware compensation for backlash + gravity (legacy).

    Args:
        current_norm: Current joint positions (normalized -100 to +100)
        target_norm: Target joint positions (normalized -100 to +100)
        compensation_factor: Multiplier for compensation (1.0 = base, 2.0 = double)
        base_compensation: Override base compensation values (optional)

    Returns:
        Compensated target positions (normalized)
    """
    if base_compensation is None:
        base_compensation = BASE_COMPENSATION

    compensated = target_norm.copy()

    for joint_idx in [1, 2]:  # Only compensate shoulder_lift and elbow_flex
        delta = target_norm[joint_idx] - current_norm[joint_idx]
        comp = base_compensation[joint_idx] * compensation_factor

        if joint_idx == 1:  # shoulder_lift
            # More negative = arm up, more positive = arm down
            if delta < -1.0:  # Moving up significantly
                compensated[joint_idx] -= comp * 0.5
            elif delta > 1.0:  # Moving down significantly
                compensated[joint_idx] -= comp * 0.5

        elif joint_idx == 2:  # elbow_flex
            # More positive = arm extended, more negative = arm folded
            if delta > 1.0:  # Moving positive (extending)
                compensated[joint_idx] += comp * 0.5
            elif delta < -1.0:  # Moving negative (folding)
                compensated[joint_idx] -= comp * 0.5

    return compensated


def apply_backlash_compensation_with_preload(
    current_norm: np.ndarray,
    target_norm: np.ndarray,
    prev_direction: np.ndarray,
    compensation_factor: float = 1.0,
    base_compensation: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply legacy direction compensation + preload on direction change.

    This combines:
    1. Legacy direction compensation (proven to work)
    2. Extra preload when direction changes (to overcome deadband)

    Args:
        current_norm: Current joint positions (normalized)
        target_norm: Target joint positions (normalized)
        prev_direction: Previous movement direction per joint (-1, 0, +1)
        compensation_factor: Multiplier for compensation
        base_compensation: Override base compensation values (optional)

    Returns:
        Tuple of (compensated positions, new direction)
    """
    # First apply legacy compensation
    compensated = apply_direction_compensation(
        current_norm, target_norm, compensation_factor, base_compensation
    )

    new_direction = prev_direction.copy()

    for joint_idx in [1, 2]:  # shoulder_lift and elbow_flex
        delta = target_norm[joint_idx] - current_norm[joint_idx]

        # Determine current direction
        if abs(delta) < 0.5:  # Dead zone - no significant movement
            current_dir = 0
        else:
            current_dir = 1 if delta > 0 else -1

        # Direction change preload (additive to legacy compensation)
        if prev_direction[joint_idx] != 0 and current_dir != 0:
            if prev_direction[joint_idx] != current_dir:
                # Direction changed - apply extra preload
                deadband = DEADBAND[joint_idx] * compensation_factor
                preload = PRELOAD_GAIN * deadband * current_dir
                compensated[joint_idx] += preload

        # Update direction state
        if current_dir != 0:
            new_direction[joint_idx] = current_dir

    return compensated, new_direction


# ============================================================================
# Gravity LUT Compensation
# ============================================================================

class GravityLUT:
    """
    Gravity compensation lookup table.

    Loads calibration data and provides interpolated corrections
    based on joint angles.
    """

    def __init__(self, lut_path: Optional[str] = None):
        """
        Initialize gravity LUT.

        Args:
            lut_path: Path to gravity LUT JSON file
        """
        self.lut_data = {}
        self.enabled = False

        if lut_path and Path(lut_path).exists():
            self.load(lut_path)

    def load(self, lut_path: str):
        """Load LUT from JSON file."""
        with open(lut_path, 'r') as f:
            data = json.load(f)

        self.lut_data = {}
        for joint_name, lut_info in data.get("lut", {}).items():
            angles = np.array(lut_info["angles"])
            corrections = np.array(lut_info["corrections"])
            self.lut_data[joint_name] = {
                "angles": angles,
                "corrections": corrections,
            }

        self.enabled = len(self.lut_data) > 0

    def get_correction(self, joint_name: str, angle: float) -> float:
        """
        Get gravity correction for a joint at a given angle.

        Uses linear interpolation between calibration points.

        Args:
            joint_name: Name of the joint
            angle: Current angle (normalized)

        Returns:
            Correction to add (normalized units)
        """
        if not self.enabled or joint_name not in self.lut_data:
            return 0.0

        lut = self.lut_data[joint_name]
        angles = lut["angles"]
        corrections = lut["corrections"]

        # Linear interpolation
        return float(np.interp(angle, angles, corrections))

    def apply(self, target_norm: np.ndarray, joint_names: list) -> np.ndarray:
        """
        Apply gravity correction to target positions.

        Args:
            target_norm: Target joint positions (normalized)
            joint_names: List of joint names

        Returns:
            Corrected target positions
        """
        if not self.enabled:
            return target_norm

        corrected = target_norm.copy()
        for i, joint_name in enumerate(joint_names):
            if i < len(target_norm):
                correction = self.get_correction(joint_name, target_norm[i])
                corrected[i] += correction

        return corrected


def apply_adaptive_compensation(
    current_norm: np.ndarray,
    target_norm: np.ndarray,
    target_z: float,
    override_factor: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply z-adaptive direction compensation.

    Args:
        current_norm: Current joint positions (normalized)
        target_norm: Target joint positions (normalized)
        target_z: Target z position in meters
        override_factor: Override automatic factor selection (optional)

    Returns:
        Tuple of (compensated target positions, factor used)
    """
    if override_factor is not None:
        factor = override_factor
    else:
        factor = get_adaptive_factor(target_z)

    compensated = apply_direction_compensation(
        current_norm, target_norm, factor
    )

    return compensated, factor


# ============================================================================
# Gravity Sag Pre-Compensation (Cartesian z offset before IK)
# ============================================================================

class GravitySagCompensator:
    """Pre-compensate IK target z for gravity-induced arm sag.

    At extended reach + high z positions, gravity causes the arm to droop
    below the commanded height.  This compensator increases the IK target z
    by the predicted sag amount so the actual end-effector lands at the
    desired position after gravity pulls it down.

    Model:
        sag = gain * reach^reach_power * max(0, z - z_deadzone)

    The parameters can be tuned per-robot via the compensation config JSON
    under the ``gravity_sag`` key.
    """

    def __init__(
        self,
        gain: float = 1.5,
        reach_power: float = 2.0,
        z_deadzone: float = 0.05,
        max_offset: float = 0.05,
        enabled: bool = True,
    ):
        self.gain = gain
        self.reach_power = reach_power
        self.z_deadzone = z_deadzone
        self.max_offset = max_offset
        self.enabled = enabled

    @classmethod
    def from_config(cls, config: dict) -> "GravitySagCompensator":
        """Create from ``gravity_sag`` section of compensation config."""
        return cls(
            gain=config.get("gain", 1.5),
            reach_power=config.get("reach_power", 2.0),
            z_deadzone=config.get("z_deadzone", 0.05),
            max_offset=config.get("max_offset", 0.05),
            enabled=config.get("enabled", True),
        )

    def compute_offset(self, target_position: np.ndarray) -> float:
        """Compute z offset (meters, >= 0) for a given base_link target.

        Args:
            target_position: [x, y, z] in base_link frame (meters)

        Returns:
            Positive z offset to *add* to the IK target z.
        """
        if not self.enabled:
            return 0.0

        x, y, z = float(target_position[0]), float(target_position[1]), float(target_position[2])
        reach = np.sqrt(x ** 2 + y ** 2)

        z_factor = max(0.0, z - self.z_deadzone)
        if z_factor <= 0.0:
            return 0.0

        offset = self.gain * (reach ** self.reach_power) * z_factor
        return min(offset, self.max_offset)

    def get_info(self) -> dict:
        return {
            "enabled": self.enabled,
            "gain": self.gain,
            "reach_power": self.reach_power,
            "z_deadzone": self.z_deadzone,
            "max_offset": self.max_offset,
        }


class AdaptiveCompensator:
    """
    Stateful adaptive compensator for trajectory execution.

    Supports multiple compensation layers:
    1. Direction compensation (backlash + gravity direction effects)
    2. Direction change preload (optional)
    3. Gravity LUT (optional, angle-dependent static offset)

    Usage (from config file - recommended):
        compensator = AdaptiveCompensator.from_config(
            config_path="robot_configs/motor_calibration/so101/robot3_compensation.json",
            target_z=0.12
        )

    Usage (legacy):
        compensator = AdaptiveCompensator(
            target_z=0.12,
            use_preload=False,
            gravity_lut_path="robot_configs/motor_calibration/gravity_lut.json"
        )

        for waypoint in trajectory:
            actual_pos = robot.read_positions()
            compensated = compensator.compensate(actual_pos, waypoint)
            robot.write_positions(compensated)
    """

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

    def __init__(
        self,
        target_z: float,
        override_factor: Optional[float] = None,
        base_compensation: Optional[Dict[int, float]] = None,
        use_preload: bool = False,
        gravity_lut_path: Optional[str] = None,
        z_adaptive_config: Optional[Dict] = None,
    ):
        """
        Initialize compensator.

        Args:
            target_z: Target z position for adaptive factor selection
            override_factor: Override automatic factor (optional)
            base_compensation: Override base compensation values (optional)
            use_preload: Enable direction change preload (default: False)
            gravity_lut_path: Path to gravity LUT JSON file (optional)
            z_adaptive_config: Z-adaptive factor config (thresholds and factors)
        """
        self.target_z = target_z
        self.base_compensation = base_compensation or BASE_COMPENSATION
        self.use_preload = use_preload
        self.z_adaptive_config = z_adaptive_config

        if override_factor is not None:
            self.factor = override_factor
        else:
            self.factor = self._get_adaptive_factor(target_z)

        # State for direction tracking
        self.prev_direction = np.zeros(5)  # -1, 0, +1 per joint
        self.prev_target = None

        # Gravity LUT
        self.gravity_lut = None
        if gravity_lut_path:
            self.gravity_lut = GravityLUT(gravity_lut_path)

    def compensate(
        self,
        current_norm: np.ndarray,
        target_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Apply compensation to target positions.

        Applies in order:
        1. Direction compensation (backlash + gravity direction)
        2. Direction change preload (if enabled)
        3. Gravity LUT correction (if enabled)

        Args:
            current_norm: Current joint positions (normalized)
            target_norm: Target joint positions (normalized)

        Returns:
            Compensated target positions
        """
        # Step 1 & 2: Direction compensation (+ optional preload)
        if self.use_preload:
            compensated, self.prev_direction = apply_backlash_compensation_with_preload(
                current_norm,
                target_norm,
                self.prev_direction,
                self.factor,
                self.base_compensation,
            )
            self.prev_target = target_norm.copy()
        else:
            compensated = apply_direction_compensation(
                current_norm,
                target_norm,
                self.factor,
                self.base_compensation,
            )

        # Step 3: Gravity LUT correction
        if self.gravity_lut is not None and self.gravity_lut.enabled:
            compensated = self.gravity_lut.apply(compensated, self.JOINT_NAMES)

        return compensated

    def reset(self):
        """Reset direction tracking state."""
        self.prev_direction = np.zeros(5)
        self.prev_target = None

    def get_factor(self) -> float:
        """Get current compensation factor."""
        return self.factor

    def _get_adaptive_factor(self, target_z: float) -> float:
        """
        Get compensation factor based on target z height.

        Uses instance z_adaptive_config if set, otherwise global defaults.
        """
        if self.z_adaptive_config is not None:
            low_thresh = self.z_adaptive_config.get("low_z_threshold", 0.12)
            mid_thresh = self.z_adaptive_config.get("mid_z_threshold", 0.15)
            low_factor = self.z_adaptive_config.get("low_z_factor", 2.0)
            mid_factor = self.z_adaptive_config.get("mid_z_factor", 1.75)
            high_factor = self.z_adaptive_config.get("high_z_factor", 1.5)

            if target_z < low_thresh:
                return low_factor
            elif target_z < mid_thresh:
                return mid_factor
            else:
                return high_factor
        else:
            return get_adaptive_factor(target_z)

    def get_info(self) -> dict:
        """Get compensator configuration info."""
        info = {
            "target_z": self.target_z,
            "factor": self.factor,
            "use_preload": self.use_preload,
            "base_compensation": self.base_compensation,
            "shoulder_lift_comp": self.base_compensation[1] * self.factor,
            "elbow_flex_comp": self.base_compensation[2] * self.factor,
            "deadband_shoulder": DEADBAND[1] * self.factor,
            "deadband_elbow": DEADBAND[2] * self.factor,
            "gravity_lut_enabled": self.gravity_lut is not None and self.gravity_lut.enabled,
        }
        if self.z_adaptive_config is not None:
            info["z_adaptive_config"] = self.z_adaptive_config
        return info

    @classmethod
    def from_config(
        cls,
        config_path: str,
        target_z: float,
        override_factor: Optional[float] = None,
        use_preload: bool = False,
    ) -> "AdaptiveCompensator":
        """
        Create compensator from a robot-specific config file.

        Args:
            config_path: Path to compensation config JSON file
            target_z: Target z position for adaptive factor selection
            override_factor: Override automatic factor (optional)
            use_preload: Enable direction change preload (default: False)

        Returns:
            Configured AdaptiveCompensator instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Convert base_compensation from joint names to indices
        base_comp_raw = config.get("base_compensation", {})
        base_compensation = {}
        for joint_name, value in base_comp_raw.items():
            if joint_name in cls.JOINT_NAME_TO_IDX:
                idx = cls.JOINT_NAME_TO_IDX[joint_name]
                base_compensation[idx] = value

        # Fill missing joints with 0
        for i in range(5):
            if i not in base_compensation:
                base_compensation[i] = 0.0

        # Z-adaptive config
        z_adaptive_config = config.get("z_adaptive", None)

        # Create GravityLUT from embedded config
        gravity_lut = None
        if "gravity_lut" in config:
            gravity_lut = GravityLUT()
            gravity_lut.lut_data = {}
            lut_config = config["gravity_lut"].get("lut", {})
            for joint_name, lut_info in lut_config.items():
                angles = np.array(lut_info.get("angles", []))
                corrections = np.array(lut_info.get("corrections", []))
                if len(angles) > 0 and len(corrections) > 0:
                    gravity_lut.lut_data[joint_name] = {
                        "angles": angles,
                        "corrections": corrections,
                    }
            gravity_lut.enabled = len(gravity_lut.lut_data) > 0

        # Create instance
        instance = cls(
            target_z=target_z,
            override_factor=override_factor,
            base_compensation=base_compensation,
            use_preload=use_preload,
            gravity_lut_path=None,
            z_adaptive_config=z_adaptive_config,
        )

        # Set gravity LUT directly
        instance.gravity_lut = gravity_lut
        instance.config_path = config_path
        instance.robot_id = config.get("robot_id", "unknown")

        # Gravity sag pre-compensator (Cartesian z offset before IK)
        sag_config = config.get("gravity_sag", None)
        if sag_config is not None:
            instance.gravity_sag = GravitySagCompensator.from_config(sag_config)
        else:
            instance.gravity_sag = None

        return instance
