"""
Calibration-based Joint Limits

This module provides joint limits based on actual robot calibration data,
instead of theoretical URDF limits.

The calibration data defines:
- range_min: Minimum encoder position (maps to -100 normalized)
- range_max: Maximum encoder position (maps to +100 normalized)

These represent the ACTUAL physical limits of the specific robot,
which may differ from the theoretical URDF limits.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationJointLimits:
    """
    Joint limits derived from calibration data.

    Attributes:
        joint_names: List of joint names
        normalized_min: Minimum normalized values (typically -100 for all)
        normalized_max: Maximum normalized values (typically +100 for all)
        range_degrees: Angular range in degrees for each joint
        range_radians: Angular range in radians for each joint
    """
    joint_names: List[str]
    normalized_min: np.ndarray  # Always -100
    normalized_max: np.ndarray  # Always +100
    range_degrees: np.ndarray   # Total angular range per joint
    range_radians: np.ndarray   # Total angular range per joint

    # For conversion between normalized and radians
    center_radians: np.ndarray  # Center position in radians (always 0 for URDF)
    half_range_radians: np.ndarray  # Half of total range in radians

    # Homing offset correction
    # offset_normalized: The normalized value where URDF's 0 radians is located
    # If homing_offset == center, offset_normalized = 0
    # If homing_offset < center, offset_normalized < 0 (URDF 0° is on the negative side)
    offset_normalized: np.ndarray = None  # Offset in normalized units

    def __post_init__(self):
        """Initialize offset_normalized if not provided."""
        if self.offset_normalized is None:
            self.offset_normalized = np.zeros(len(self.joint_names))

    def normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized values (-100 to +100) to URDF radians.

        With homing offset correction:
        - normalized at offset_normalized → radians 0 (URDF reference pose)
        - normalized 0 → radians at some offset from URDF 0

        Formula: radians = ((normalized - offset_normalized) / 100) * half_range
        """
        normalized = np.asarray(normalized)
        # Apply homing offset correction
        corrected_normalized = normalized - self.offset_normalized
        return (corrected_normalized / 100.0) * self.half_range_radians

    def radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert URDF radians to normalized values (-100 to +100).

        With homing offset correction:
        - radians 0 (URDF reference pose) → normalized at offset_normalized

        Formula: normalized = (radians / half_range) * 100 + offset_normalized
        """
        radians = np.asarray(radians)
        # Apply homing offset correction
        base_normalized = (radians / self.half_range_radians) * 100.0
        return base_normalized + self.offset_normalized

    def is_within_limits(self, radians: np.ndarray) -> Tuple[bool, List[Tuple[int, str, float, float]]]:
        """
        Check if joint positions are within calibration limits.

        Args:
            radians: Joint positions in radians

        Returns:
            Tuple of (all_within_limits, list of violations)
            Each violation is (joint_index, joint_name, value_rad, limit_rad)
        """
        violations = []

        for i, (val, half_range, name) in enumerate(zip(radians, self.half_range_radians, self.joint_names)):
            lower = -half_range
            upper = half_range

            if val < lower:
                violations.append((i, name, val, lower))
            elif val > upper:
                violations.append((i, name, val, upper))

        return len(violations) == 0, violations

    @property
    def lower_limits_radians(self) -> np.ndarray:
        """Get lower limits in radians."""
        return -self.half_range_radians

    @property
    def upper_limits_radians(self) -> np.ndarray:
        """Get upper limits in radians."""
        return self.half_range_radians


def load_calibration_limits(
    calibration_file: str,
    joint_names: List[str] = None,
    use_homing_offset: bool = True,
) -> CalibrationJointLimits:
    """
    Load joint limits from calibration file.

    Args:
        calibration_file: Path to calibration JSON file
        joint_names: List of joint names (default: standard SO-101 joints)
        use_homing_offset: If True, apply homing offset correction for URDF alignment

    Returns:
        CalibrationJointLimits object
    """
    if joint_names is None:
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

    # Load calibration data
    # Supports both formats:
    #   1) AutoDataCollector: {"motor_1": {"motor_id": 1, ...}, "motor_2": ...}
    #   2) LeRobot official:  {"shoulder_pan": {"id": 1, ...}, "shoulder_lift": ...}
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)

    # Build motor lookup by joint name or motor index
    # Try joint_name keys first (LeRobot format), then motor_N keys (AutoDataCollector format)
    def _find_motor(calib_data, joint_name, motor_index):
        # LeRobot format: key is joint name
        if joint_name in calib_data:
            return calib_data[joint_name]
        # AutoDataCollector format: key is motor_N
        mkey = f'motor_{motor_index + 1}'
        if mkey in calib_data:
            return calib_data[mkey]
        raise ValueError(f"Motor for '{joint_name}' (motor_{motor_index+1}) not found in calibration file")

    range_degrees = []
    range_radians = []
    offset_normalized_list = []

    for i, jname in enumerate(joint_names):
        motor = _find_motor(calib_data, jname, i)
        range_min = motor['range_min']
        range_max = motor['range_max']
        homing_offset = motor.get('homing_offset', 0)
        drive_mode = motor.get('drive_mode', 0)

        # Convert encoder range to degrees (4096 steps = 360 degrees)
        encoder_range = range_max - range_min
        degrees = encoder_range * 360.0 / 4096.0
        radians = np.radians(degrees)

        range_degrees.append(degrees)
        range_radians.append(radians)

        # Calculate offset_normalized
        # URDF 0° corresponds to encoder value 2048 (HALF_TURN) after homing calibration
        # The motor's homing_offset correction makes: corrected_pos = raw_pos + homing_offset
        # After calibration, when robot is at URDF 0°, encoder reads 2048
        #
        # We need to find what normalized value corresponds to encoder 2048
        HALF_TURN = 2048  # URDF 0° encoder value after homing

        if use_homing_offset and encoder_range > 0:
            # Convert URDF 0° encoder value (2048) to normalized
            # normalized = ((encoder - range_min) / encoder_range) * 200 - 100
            urdf_zero_normalized = ((HALF_TURN - range_min) / encoder_range) * 200 - 100

            # Apply drive_mode (direction inversion)
            if drive_mode == 1:
                urdf_zero_normalized = -urdf_zero_normalized

            # offset_normalized is where URDF 0° is in normalized space
            offset_normalized_list.append(urdf_zero_normalized)
        else:
            offset_normalized_list.append(0.0)

    range_degrees = np.array(range_degrees)
    range_radians = np.array(range_radians)
    half_range_radians = range_radians / 2.0
    offset_normalized = np.array(offset_normalized_list)

    if use_homing_offset:
        print(f"  Homing offset correction enabled:")
        for i, name in enumerate(joint_names):
            print(f"    {name}: offset = {offset_normalized[i]:.1f} (normalized)")

    return CalibrationJointLimits(
        joint_names=joint_names,
        normalized_min=np.full(len(joint_names), -100.0),
        normalized_max=np.full(len(joint_names), 100.0),
        range_degrees=range_degrees,
        range_radians=range_radians,
        center_radians=np.zeros(len(joint_names)),
        half_range_radians=half_range_radians,
        offset_normalized=offset_normalized,
    )


def compare_limits(
    calibration_file: str,
    urdf_lower: np.ndarray,
    urdf_upper: np.ndarray,
    joint_names: List[str] = None,
) -> Dict:
    """
    Compare calibration limits with URDF limits.

    Args:
        calibration_file: Path to calibration JSON file
        urdf_lower: URDF lower limits in radians
        urdf_upper: URDF upper limits in radians
        joint_names: List of joint names

    Returns:
        Dictionary with comparison results
    """
    calib_limits = load_calibration_limits(calibration_file, joint_names)

    if joint_names is None:
        joint_names = calib_limits.joint_names

    results = {
        'joints': [],
        'summary': {
            'calibration_narrower': 0,
            'urdf_narrower': 0,
            'similar': 0,
        }
    }

    for i, name in enumerate(joint_names):
        urdf_range = urdf_upper[i] - urdf_lower[i]
        calib_range = calib_limits.range_radians[i]

        ratio = calib_range / urdf_range if urdf_range > 0 else 1.0

        joint_result = {
            'name': name,
            'index': i + 1,
            'urdf_range_deg': np.degrees(urdf_range),
            'calib_range_deg': np.degrees(calib_range),
            'ratio': ratio,
            'difference_deg': np.degrees(calib_range - urdf_range),
        }

        if ratio < 0.9:
            joint_result['status'] = 'calibration_narrower'
            results['summary']['calibration_narrower'] += 1
        elif ratio > 1.1:
            joint_result['status'] = 'urdf_narrower'
            results['summary']['urdf_narrower'] += 1
        else:
            joint_result['status'] = 'similar'
            results['summary']['similar'] += 1

        results['joints'].append(joint_result)

    return results


def print_limits_comparison(
    calibration_file: str,
    urdf_lower: np.ndarray,
    urdf_upper: np.ndarray,
    joint_names: List[str] = None,
):
    """Print a formatted comparison of calibration vs URDF limits."""
    results = compare_limits(calibration_file, urdf_lower, urdf_upper, joint_names)

    print("=" * 70)
    print("캘리브레이션 한계 vs URDF 한계 비교")
    print("=" * 70)
    print()

    for joint in results['joints']:
        print(f"[Joint {joint['index']}] {joint['name']}")
        print(f"  URDF 범위:         {joint['urdf_range_deg']:6.1f}°")
        print(f"  캘리브레이션 범위: {joint['calib_range_deg']:6.1f}°")
        print(f"  비율 (calib/urdf): {joint['ratio']:.2f}")

        if joint['status'] == 'calibration_narrower':
            pct = (1 - joint['ratio']) * 100
            print(f"  ⚠️  캘리브레이션이 {pct:.0f}% 좁음 - URDF 한계 도달 불가!")
        elif joint['status'] == 'urdf_narrower':
            pct = (joint['ratio'] - 1) * 100
            print(f"  ℹ️  캘리브레이션이 {pct:.0f}% 넓음")
        else:
            print(f"  ✓  범위가 유사함")
        print()

    print("=" * 70)
    print(f"요약: 캘리브레이션이 좁음={results['summary']['calibration_narrower']}, "
          f"URDF가 좁음={results['summary']['urdf_narrower']}, "
          f"유사={results['summary']['similar']}")
    print("=" * 70)
