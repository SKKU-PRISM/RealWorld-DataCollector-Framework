"""
USB Serial Port Finder for Robot Hardware

Cross-platform utility to find USB serial ports for Dynamixel/Feetech motors.
"""

import glob
import platform
from typing import List, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    raise ImportError("pyserial is required. Install: pip install pyserial")


def get_available_ports() -> List[str]:
    """
    Get list of available serial ports.

    Returns:
        List of port names (e.g., ['/dev/ttyUSB0', 'COM3'])
    """
    ports = []
    system = platform.system()

    if system == "Linux":
        ports.extend(glob.glob('/dev/ttyUSB*'))
        ports.extend(glob.glob('/dev/ttyACM*'))
        ports.extend(glob.glob('/dev/serial/by-id/*'))
    elif system == "Darwin":  # macOS
        ports.extend(glob.glob('/dev/tty.usb*'))
        ports.extend(glob.glob('/dev/cu.usb*'))
    elif system == "Windows":
        for i in range(256):
            try:
                port = f'COM{i}'
                s = serial.Serial(port)
                s.close()
                ports.append(port)
            except (OSError, serial.SerialException):
                pass

    # Also use pyserial's built-in detector
    detected_ports = serial.tools.list_ports.comports()
    for port_info in detected_ports:
        if port_info.device not in ports:
            ports.append(port_info.device)

    return sorted(ports)


def get_port_info(port: str) -> dict:
    """Get detailed information about a serial port."""
    try:
        port_info = None
        for p in serial.tools.list_ports.comports():
            if p.device == port:
                port_info = p
                break

        if port_info:
            return {
                'device': port_info.device,
                'description': port_info.description,
                'hwid': port_info.hwid,
                'vid': port_info.vid,
                'pid': port_info.pid,
                'serial_number': port_info.serial_number,
                'manufacturer': port_info.manufacturer,
                'product': port_info.product,
            }
        else:
            return {'device': port, 'description': 'Unknown', 'hwid': 'Unknown'}
    except Exception as e:
        return {'device': port, 'error': str(e)}


def test_port_connection(port: str, baudrate: int = 1000000, timeout: float = 0.5) -> bool:
    """Test if a port can be opened successfully."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        ser.close()
        return True
    except Exception:
        return False


def find_port(baudrate: int = 1000000) -> Optional[str]:
    """
    Attempt to find the motor port automatically.

    Args:
        baudrate: Baud rate to test (default: 1000000)

    Returns:
        Port name if found, None otherwise
    """
    ports = get_available_ports()

    for port in ports:
        if test_port_connection(port, baudrate):
            info = get_port_info(port)
            description = (info.get('description') or '').lower()
            manufacturer = (info.get('manufacturer') or '').lower()

            # Common USB-to-serial chips
            if any(keyword in description or keyword in manufacturer for keyword in
                   ['ft232', 'ftdi', 'cp210', 'silicon labs', 'ch340', 'prolific']):
                return port

    # If no specific match, return first available port
    for port in ports:
        if test_port_connection(port, baudrate):
            return port

    return None


# Alias for compatibility
find_dynamixel_port = find_port


def print_port_info():
    """Print available ports information."""
    ports = get_available_ports()

    if not ports:
        print("No serial ports found.")
        return

    print(f"\nFound {len(ports)} serial port(s):")
    print("-" * 80)

    for port in ports:
        info = get_port_info(port)
        accessible = test_port_connection(port)
        status = "OK" if accessible else "BLOCKED"

        print(f"  {port}: {info.get('description', 'Unknown')} [{status}]")

    print("-" * 80)


if __name__ == "__main__":
    print_port_info()
    port = find_port()
    if port:
        print(f"\nRecommended port: {port}")
    else:
        print("\nNo suitable port found.")
