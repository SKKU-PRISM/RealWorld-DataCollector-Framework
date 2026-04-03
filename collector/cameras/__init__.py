"""
Camera modules for LeRobot dataset recording.

Supports:
- Intel RealSense D435 (RGB + Depth)
- Innomaker U20CAM (OpenCV-based USB camera)
"""

from .camera_config import CameraConfig, RealSenseCameraConfig, OpenCVCameraConfig
from .realsense_camera import RealSenseCamera
from .opencv_camera import OpenCVCamera
from .multi_camera import MultiCameraManager

__all__ = [
    "CameraConfig",
    "RealSenseCameraConfig",
    "OpenCVCameraConfig",
    "RealSenseCamera",
    "OpenCVCamera",
    "MultiCameraManager",
]
