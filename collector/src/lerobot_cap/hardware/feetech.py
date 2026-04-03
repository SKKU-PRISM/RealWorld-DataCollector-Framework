"""
Feetech Motor Controller

Controls Feetech STS3215 servo motors (used in SO-100, SO-101 robots).
Ported from lerobot_ros2 project.
"""

import time
from typing import List, Dict, Optional
import numpy as np

try:
    import scservo_sdk as scs
except ImportError:
    raise ImportError(
        "scservo_sdk is not installed.\n"
        "Install: pip install feetech-servo-sdk"
    )

from lerobot_cap.hardware.base import BaseController
from lerobot_cap.hardware.calibration import MotorCalibration
from lerobot_cap.hardware.find_port import find_port, get_available_ports


class FeetechController(BaseController):
    """
    Controller for Feetech STS3215 servo motors.

    Args:
        port: Serial port (e.g., '/dev/ttyACM0'). Auto-detect if None.
        baudrate: Communication speed (Default: 1000000)
        motor_ids: List of motor IDs (e.g., [1, 2, 3, 4, 5, 6])
        calibration: Motor calibration data (motor_id -> MotorCalibration)
    """

    # Control table addresses for STS3215
    ADDR_MIN_POSITION_LIMIT = 9   # 2 bytes
    ADDR_MAX_POSITION_LIMIT = 11  # 2 bytes
    ADDR_MAX_TORQUE = 16          # 2 bytes
    ADDR_HOMING_OFFSET = 31       # 2 bytes (signed)
    ADDR_OPERATING_MODE = 33
    ADDR_TORQUE_ENABLE = 40
    ADDR_ACCELERATION = 41
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_TIME = 44
    ADDR_GOAL_VELOCITY = 46
    ADDR_TORQUE_LIMIT = 48
    ADDR_LOCK = 55
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58
    ADDR_PRESENT_LOAD = 60

    PROTOCOL_VERSION = 0  # SCS Protocol
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 1000000,
        motor_ids: Optional[List[int]] = None,
        calibration: Optional[Dict[int, MotorCalibration]] = None,
    ):
        # Auto-detect port if not specified
        if port is None:
            print("Auto-detecting port...")
            port = find_port(baudrate)
            if port is None:
                available = get_available_ports()
                if available:
                    print(f"Available ports: {', '.join(available)}")
                    port = available[0]
                    print(f"Using first port: {port}")
                else:
                    raise ValueError("No USB serial ports found")
            else:
                print(f"Port detected: {port}")

        super().__init__(port, baudrate, motor_ids or [])

        self.calibration = calibration or {}

        # SDK objects
        self.port_handler = None
        self.packet_handler = None

    def connect(self) -> bool:
        """Connect to motors."""
        try:
            self.port_handler = scs.PortHandler(self.port)
            self.packet_handler = scs.PacketHandler(self.PROTOCOL_VERSION)

            if not self.port_handler.openPort():
                raise Exception(f"Failed to open port: {self.port}")

            if not self.port_handler.setBaudRate(self.baudrate):
                raise Exception(f"Failed to set baudrate: {self.baudrate}")

            print(f"Port connected: {self.port} @ {self.baudrate} bps")

            # Check each motor
            found_motors = []
            for motor_id in self.motor_ids:
                model_number, comm_result, error = self.packet_handler.ping(
                    self.port_handler, motor_id
                )
                if comm_result == scs.COMM_SUCCESS:
                    print(f"  Motor {motor_id} found (model: {model_number})")
                    found_motors.append(motor_id)
                else:
                    print(f"  Motor {motor_id} not responding")

            if not found_motors:
                raise Exception("No motors found")

            self.is_connected = True
            self.motor_ids = found_motors

            # Read initial positions
            if self.calibration:
                self.current_positions = self.read_positions(normalize=True)
            else:
                self.current_positions = self._read_positions_raw().astype(np.float32)

            self.target_positions = self.current_positions.copy()

            print(f"{len(found_motors)} motors initialized")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from motors."""
        if not self.is_connected:
            return

        try:
            for motor_id in self.motor_ids:
                self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)

            if self.port_handler:
                self.port_handler.closePort()

            self.is_connected = False
            print("Motors disconnected")

        except Exception as e:
            print(f"Error during disconnect: {e}")

    def enable_torque(self, motor_ids: Optional[List[int]] = None):
        """Enable motor torque."""
        if not self.is_connected:
            raise Exception("Motors not connected")

        ids_to_enable = motor_ids if motor_ids is not None else self.motor_ids

        # Set Goal_Position to current position before enabling
        print("Initializing Goal_Position before enabling torque...")
        for motor_id in ids_to_enable:
            try:
                current_pos = self._read_2byte(motor_id, self.ADDR_PRESENT_POSITION)
                self._write_2byte(motor_id, self.ADDR_GOAL_POSITION, current_pos)
                time.sleep(0.05)
            except Exception as e:
                print(f"  Motor {motor_id} initialization failed: {e}")

        time.sleep(0.1)

        # Set default movement parameters
        DEFAULT_VELOCITY = 1500
        DEFAULT_ACCELERATION = 50

        for motor_id in ids_to_enable:
            self._write_2byte(motor_id, self.ADDR_GOAL_VELOCITY, DEFAULT_VELOCITY)
            self._write_1byte(motor_id, self.ADDR_ACCELERATION, DEFAULT_ACCELERATION)

        time.sleep(0.05)

        # Enable torque
        for motor_id in ids_to_enable:
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            self._write_1byte(motor_id, self.ADDR_LOCK, 1)

        print(f"Torque enabled: {ids_to_enable}")

    def disable_torque(self, motor_ids: Optional[List[int]] = None):
        """Disable motor torque."""
        if not self.is_connected:
            raise Exception("Motors not connected")

        ids_to_disable = motor_ids if motor_ids is not None else self.motor_ids

        for motor_id in ids_to_disable:
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            self._write_1byte(motor_id, self.ADDR_LOCK, 0)

        time.sleep(0.1)
        print(f"Torque disabled: {ids_to_disable}")

    def read_positions(self, normalize: bool = True) -> np.ndarray:
        """Read current motor positions."""
        if not self.is_connected:
            return self.current_positions

        positions_raw = self._read_positions_raw()

        if normalize:
            normalized = self._normalize(positions_raw)
            self.current_positions = normalized
            return normalized.copy()
        else:
            return positions_raw.astype(np.float32)

    def write_positions(self, positions: np.ndarray, normalize: bool = True):
        """Write target positions to motors."""
        if not self.is_connected:
            return

        if len(positions) != len(self.motor_ids):
            raise ValueError(f"Position count mismatch: {len(positions)} != {len(self.motor_ids)}")

        self.target_positions = positions.copy()

        if normalize:
            positions_raw = self._unnormalize(positions)
        else:
            positions_raw = positions.astype(np.int32)

        for i, motor_id in enumerate(self.motor_ids):
            position_raw = max(0, min(4095, int(positions_raw[i])))
            self._write_2byte(motor_id, self.ADDR_GOAL_POSITION, position_raw)

    def _read_positions_raw(self) -> np.ndarray:
        """Read raw position values."""
        positions = np.zeros(len(self.motor_ids), dtype=np.int32)
        for i, motor_id in enumerate(self.motor_ids):
            try:
                positions[i] = self._read_2byte(motor_id, self.ADDR_PRESENT_POSITION)
            except:
                pass
        return positions

    def _normalize(self, positions_raw: np.ndarray) -> np.ndarray:
        """Convert raw encoder values to normalized (-100 to +100).

        Uses range_min/range_max from calibration to define the physical limits.
        Maps: range_min → -100, range_max → +100, center → 0
        """
        if not self.calibration:
            raise RuntimeError("Calibration not loaded")

        normalized = np.zeros(len(positions_raw), dtype=np.float32)

        for i, (motor_id, pos_raw) in enumerate(zip(self.motor_ids, positions_raw)):
            if motor_id not in self.calibration:
                raise ValueError(f"No calibration for motor {motor_id}")

            calib = self.calibration[motor_id]
            min_val, max_val = calib.range_min, calib.range_max

            if min_val == max_val:
                raise ValueError(f"Motor {motor_id}: min and max are identical")

            bounded_val = min(max_val, max(min_val, pos_raw))
            normalized[i] = (((bounded_val - min_val) / (max_val - min_val)) * 200) - 100

            if calib.drive_mode == 1:
                normalized[i] = -normalized[i]

        return normalized

    def _unnormalize(self, positions_norm: np.ndarray) -> np.ndarray:
        """Convert normalized (-100 to +100) to raw encoder values.

        Inverse of _normalize():
        Maps: -100 → range_min, 0 → center, +100 → range_max
        """
        if not self.calibration:
            raise RuntimeError("Calibration not loaded")

        raw = np.zeros(len(positions_norm), dtype=np.int32)

        for i, (motor_id, pos_norm) in enumerate(zip(self.motor_ids, positions_norm)):
            if motor_id not in self.calibration:
                raise ValueError(f"No calibration for motor {motor_id}")

            calib = self.calibration[motor_id]
            min_val, max_val = calib.range_min, calib.range_max

            bounded_norm = min(100.0, max(-100.0, pos_norm))
            if calib.drive_mode == 1:
                bounded_norm = -bounded_norm

            raw_val = int(((bounded_norm + 100) / 200) * (max_val - min_val) + min_val)
            raw[i] = max(min_val, min(max_val, raw_val))

        return raw

    def _read_1byte(self, motor_id: int, address: int, retries: int = 3) -> int:
        """Read 1 byte with retry."""
        for attempt in range(retries):
            data, result, error = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, address
            )
            if result == scs.COMM_SUCCESS:
                return data
            if attempt < retries - 1:
                time.sleep(0.02 * (attempt + 1))
        raise Exception(f"Read failed (motor {motor_id}, address {address})")

    def _read_2byte(self, motor_id: int, address: int, retries: int = 3) -> int:
        """Read 2 bytes with retry."""
        for attempt in range(retries):
            data, result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, address
            )
            if result == scs.COMM_SUCCESS:
                return data
            if attempt < retries - 1:
                time.sleep(0.02 * (attempt + 1))
        raise Exception(f"Read failed (motor {motor_id}, address {address})")

    def _write_1byte(self, motor_id: int, address: int, value: int, retries: int = 3):
        """Write 1 byte with retry."""
        for attempt in range(retries):
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value
            )
            if result == scs.COMM_SUCCESS:
                return
            if attempt < retries - 1:
                time.sleep(0.02 * (attempt + 1))
        print(f"Write1B failed (motor {motor_id}, addr {address})")

    def _write_2byte(self, motor_id: int, address: int, value: int, retries: int = 3):
        """Write 2 bytes with retry."""
        for attempt in range(retries):
            result, error = self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id, address, value
            )
            if result == scs.COMM_SUCCESS:
                return
            if attempt < retries - 1:
                time.sleep(0.02 * (attempt + 1))
        print(f"Write2B failed (motor {motor_id}, addr {address})")

    def set_torque_limit(self, limit: int = 500, motor_ids: Optional[List[int]] = None):
        """Set torque limit (0-1000, lower = more compliant)."""
        if not self.is_connected:
            raise Exception("Motors not connected")

        ids_to_set = motor_ids if motor_ids is not None else self.motor_ids
        limit = max(0, min(1000, limit))

        for motor_id in ids_to_set:
            self._write_2byte(motor_id, self.ADDR_TORQUE_LIMIT, limit)

        print(f"Torque limit set to {limit}/1000: {ids_to_set}")

    def read_present_load(self) -> np.ndarray:
        """Read current load from motors (-1000 to 1000)."""
        if not self.is_connected:
            return np.zeros(len(self.motor_ids), dtype=np.float32)

        loads = np.zeros(len(self.motor_ids), dtype=np.float32)
        for i, motor_id in enumerate(self.motor_ids):
            try:
                raw = self._read_2byte(motor_id, self.ADDR_PRESENT_LOAD)
                if raw & 0x400:
                    loads[i] = -(raw & 0x3FF)
                else:
                    loads[i] = raw & 0x3FF
            except:
                pass
        return loads
