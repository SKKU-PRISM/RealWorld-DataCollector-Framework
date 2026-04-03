"""
Motor Calibration Data Structures

Store and load calibration data (LeRobot-compatible)
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MotorCalibration:
    """
    Motor calibration data (LeRobot-compatible)

    Attributes:
        motor_id: Motor ID
        model: Motor model name (e.g., "sts3215")
        drive_mode: Drive mode (0=normal, 1=inverted)
        homing_offset: Homing offset written to motor EEPROM
        range_min: Minimum position (Present_Position)
        range_max: Maximum position (Present_Position)
    """
    motor_id: int
    model: str
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 4095

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'MotorCalibration':
        """Create from dictionary"""
        return cls(**data)

    def clamp_position(self, position: int) -> int:
        """Clamp position within calibration range"""
        return max(self.range_min, min(position, self.range_max))

    def is_within_range(self, position: int) -> bool:
        """Check if position is within calibration range"""
        return self.range_min <= position <= self.range_max

    @property
    def range_center(self) -> float:
        """Get center of calibration range"""
        return (self.range_min + self.range_max) / 2

    @property
    def range_size(self) -> int:
        """Get size of calibration range"""
        return self.range_max - self.range_min


class CalibrationManager:
    """Calibration file manager"""

    def __init__(self, calibration_dir: Optional[Path] = None):
        """
        Args:
            calibration_dir: Calibration storage directory
        """
        self.calibration_dir = Path(calibration_dir) if calibration_dir else None

    def save(self, calibrations: Dict[str, MotorCalibration], filename: str = "calibration.json"):
        """
        Save calibration to JSON file

        Args:
            calibrations: Motor name -> MotorCalibration dictionary
            filename: Filename to save
        """
        if not self.calibration_dir:
            raise ValueError("calibration_dir not set")

        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        calib_dict = {
            name: calib.to_dict()
            for name, calib in calibrations.items()
        }

        filepath = self.calibration_dir / filename
        with open(filepath, "w") as f:
            json.dump(calib_dict, f, indent=2)

        print(f"Calibration saved: {filepath}")

    def load(self, filename: str = "calibration.json") -> Dict[str, MotorCalibration]:
        """
        Load calibration from JSON file

        Args:
            filename: Filename to load

        Returns:
            Motor name -> MotorCalibration dictionary
        """
        if not self.calibration_dir:
            raise ValueError("calibration_dir not set")

        filepath = self.calibration_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        with open(filepath, "r") as f:
            calib_dict = json.load(f)

        calibrations = {
            name: MotorCalibration.from_dict(data)
            for name, data in calib_dict.items()
        }

        print(f"Calibration loaded: {filepath} ({len(calibrations)} motors)")
        return calibrations

    def exists(self, filename: str = "calibration.json") -> bool:
        """Check if calibration file exists"""
        if not self.calibration_dir:
            return False
        return (self.calibration_dir / filename).exists()


def create_default_calibration(
    motor_id: int,
    model: str = "sts3215",
    drive_mode: int = 0,
    homing_offset: int = 0,
    range_min: int = 0,
    range_max: int = 4095
) -> MotorCalibration:
    """Create default calibration"""
    return MotorCalibration(
        motor_id=motor_id,
        model=model,
        drive_mode=drive_mode,
        homing_offset=homing_offset,
        range_min=range_min,
        range_max=range_max,
    )


def load_calibration(calibration_dir: Path, filename: str = "calibration.json") -> Dict[str, MotorCalibration]:
    """Load calibration (convenience function)"""
    manager = CalibrationManager(calibration_dir)
    return manager.load(filename)


def save_calibration(
    calibrations: Dict[str, MotorCalibration],
    calibration_dir: Path,
    filename: str = "calibration.json"
):
    """Save calibration (convenience function)"""
    manager = CalibrationManager(calibration_dir)
    manager.save(calibrations, filename)
