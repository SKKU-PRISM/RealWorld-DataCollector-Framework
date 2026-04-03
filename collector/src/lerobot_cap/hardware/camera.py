"""
Camera Controller

Supports OpenCV and RealSense cameras.
"""

from typing import Dict, List, Optional, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class CameraController:
    """
    Multi-camera controller supporting OpenCV and RealSense cameras.

    Args:
        camera_configs: Dictionary of camera configurations
            {
                "cam_0": {"type": "opencv", "index": 0, "width": 640, "height": 480},
                "cam_1": {"type": "realsense", "serial": "123456", "width": 640, "height": 480},
            }
    """

    def __init__(self, camera_configs: Optional[Dict] = None):
        self.camera_configs = camera_configs or {}
        self.cameras: Dict[str, Union[cv2.VideoCapture, 'rs.pipeline']] = {}
        self.is_connected = False

    def connect(self) -> bool:
        """Connect to all configured cameras."""
        if cv2 is None:
            print("Warning: OpenCV not installed")

        try:
            for name, config in self.camera_configs.items():
                cam_type = config.get("type", "opencv")

                if cam_type == "opencv":
                    self._connect_opencv(name, config)
                elif cam_type == "realsense":
                    self._connect_realsense(name, config)
                else:
                    print(f"Unknown camera type: {cam_type}")

            self.is_connected = len(self.cameras) > 0
            print(f"Connected to {len(self.cameras)} camera(s)")
            return self.is_connected

        except Exception as e:
            print(f"Camera connection failed: {e}")
            return False

    def _connect_opencv(self, name: str, config: dict):
        """Connect OpenCV camera."""
        if cv2 is None:
            raise ImportError("OpenCV not installed")

        index = config.get("index", 0)
        width = config.get("width", 640)
        height = config.get("height", 480)
        fps = config.get("fps", 30)

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise Exception(f"Failed to open camera {index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        self.cameras[name] = cap
        print(f"  Camera '{name}' (OpenCV index {index}): {width}x{height}@{fps}fps")

    def _connect_realsense(self, name: str, config: dict):
        """Connect RealSense camera."""
        if rs is None:
            raise ImportError("pyrealsense2 not installed")

        serial = config.get("serial")
        width = config.get("width", 640)
        height = config.get("height", 480)
        fps = config.get("fps", 30)

        pipeline = rs.pipeline()
        rs_config = rs.config()

        if serial:
            rs_config.enable_device(serial)

        rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        pipeline.start(rs_config)

        self.cameras[name] = pipeline
        print(f"  Camera '{name}' (RealSense): {width}x{height}@{fps}fps")

    def disconnect(self):
        """Disconnect all cameras."""
        for name, camera in self.cameras.items():
            try:
                if isinstance(camera, cv2.VideoCapture):
                    camera.release()
                elif rs and hasattr(camera, 'stop'):
                    camera.stop()
            except:
                pass

        self.cameras.clear()
        self.is_connected = False
        print("Cameras disconnected")

    def read_images(self) -> Dict[str, np.ndarray]:
        """
        Read images from all cameras.

        Returns:
            Dictionary of camera_name -> image (RGB format)
        """
        images = {}

        for name, camera in self.cameras.items():
            try:
                if isinstance(camera, cv2.VideoCapture):
                    ret, frame = camera.read()
                    if ret:
                        # Convert BGR to RGB
                        images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif rs and hasattr(camera, 'wait_for_frames'):
                    frames = camera.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        images[name] = np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"Error reading camera {name}: {e}")

        return images

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get observation in LeRobot format.

        Returns:
            {"observation.images.{camera_name}": image}
        """
        images = self.read_images()
        return {
            f"observation.images.{name}": image
            for name, image in images.items()
        }

    @property
    def camera_names(self) -> List[str]:
        """List of connected camera names."""
        return list(self.cameras.keys())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __del__(self):
        if hasattr(self, 'is_connected') and self.is_connected:
            self.disconnect()
