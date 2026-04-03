"""
Camera Configuration Classes

카메라 설정을 정의하는 dataclass들
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


class CameraType(Enum):
    """카메라 타입"""
    REALSENSE = "realsense"
    OPENCV = "opencv"


class ColorMode(Enum):
    """색상 모드"""
    RGB = "rgb"
    BGR = "bgr"


@dataclass
class CameraConfig:
    """기본 카메라 설정"""
    name: str  # 카메라 식별자 (예: "front", "wrist")
    width: int = 640
    height: int = 480
    fps: int = 30
    color_mode: ColorMode = ColorMode.RGB
    enabled: bool = True

    def to_feature_key(self) -> str:
        """LeRobot 데이터셋 feature 키 생성"""
        return f"observation.images.{self.name}"


@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Intel RealSense 카메라 설정

    pyrealsense2 SDK가 자동으로 카메라를 감지합니다.
    여러 대 연결 시 serial_number로 특정 카메라를 지정할 수 있습니다.
    """
    camera_type: CameraType = field(default=CameraType.REALSENSE, init=False)
    serial_number: Optional[str] = None  # 특정 카메라 지정 (여러 대 연결 시)
    enable_depth: bool = False  # RGB만 사용할지 Depth도 포함할지

    def __post_init__(self):
        if not self.name:
            self.name = "realsense"


@dataclass
class OpenCVCameraConfig(CameraConfig):
    """OpenCV 기반 USB 카메라 설정 (Innomaker U20CAM 등)"""
    camera_type: CameraType = field(default=CameraType.OPENCV, init=False)
    device_path: str = "/dev/video7"  # Linux 장치 경로
    device_index: Optional[int] = None  # 또는 인덱스 사용
    fourcc: Optional[str] = "MJPG"  # 코덱 (MJPG가 일반적으로 더 빠름)
    warmup_frames: int = 30  # 연결 후 버릴 프레임 수

    def __post_init__(self):
        if not self.name:
            self.name = "usb_cam"

    @property
    def index_or_path(self):
        """device_index가 있으면 인덱스 반환, 없으면 device_path 반환"""
        if self.device_index is not None:
            return self.device_index
        return self.device_path


@dataclass
class MultiCameraConfig:
    """멀티 카메라 설정"""
    cameras: List[CameraConfig] = field(default_factory=list)
    sync_mode: bool = False  # 동기화 모드 (모든 카메라에서 동시에 캡처)

    def get_camera(self, name: str) -> Optional[CameraConfig]:
        """이름으로 카메라 설정 찾기"""
        for cam in self.cameras:
            if cam.name == name:
                return cam
        return None

    def get_enabled_cameras(self) -> List[CameraConfig]:
        """활성화된 카메라만 반환"""
        return [cam for cam in self.cameras if cam.enabled]

    def get_feature_keys(self) -> List[str]:
        """LeRobot 데이터셋 feature 키 목록"""
        return [cam.to_feature_key() for cam in self.get_enabled_cameras()]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MultiCameraConfig":
        """YAML 파일에서 로드"""
        import yaml

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        cameras = []
        for cam_data in data.get("cameras", []):
            cam_type = cam_data.pop("type", "opencv")

            if cam_type == "realsense":
                cameras.append(RealSenseCameraConfig(**cam_data))
            else:
                cameras.append(OpenCVCameraConfig(**cam_data))

        return cls(
            cameras=cameras,
            sync_mode=data.get("sync_mode", False),
        )


# 기본 설정 (Intel RealSense + Innomaker U20CAM)
DEFAULT_CAMERA_CONFIGS = MultiCameraConfig(
    cameras=[
        RealSenseCameraConfig(
            name="realsense",
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        ),
        OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video7",
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        ),
    ],
    sync_mode=False,
)
