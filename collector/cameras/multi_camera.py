"""
Multi-Camera Manager

여러 카메라를 동시에 관리
LeRobot 공식 구현과 동일한 async_read 방식 사용
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from .camera_config import (
    CameraConfig,
    RealSenseCameraConfig,
    OpenCVCameraConfig,
    MultiCameraConfig,
)
from .realsense_camera import RealSenseCamera
from .opencv_camera import OpenCVCamera


class MultiCameraManager:
    """
    멀티 카메라 매니저

    LeRobot 공식 구현과 동일한 방식:
    - 각 카메라는 백그라운드 스레드에서 연속 캡처
    - get_observation()에서 async_read()로 최신 프레임 반환

    Usage:
        from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

        configs = [
            RealSenseCameraConfig(name="realsense", width=640, height=480),
            OpenCVCameraConfig(name="innomaker", device_path="/dev/video7"),
        ]

        manager = MultiCameraManager(configs)
        manager.connect_all()

        # LeRobot 방식: async_read로 최신 프레임
        images = manager.get_observation()
        # {"observation.images.realsense": np.array, "observation.images.innomaker": np.array}

        manager.disconnect_all()
    """

    def __init__(
        self,
        configs: Optional[List[CameraConfig]] = None,
        multi_config: Optional[MultiCameraConfig] = None,
    ):
        """
        초기화

        Args:
            configs: 카메라 설정 리스트
            multi_config: MultiCameraConfig 객체 (configs 대신 사용 가능)
        """
        if multi_config is not None:
            self.configs = multi_config.get_enabled_cameras()
        else:
            self.configs = configs or []

        self.cameras: Dict[str, Any] = {}
        self._is_connected = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MultiCameraManager":
        """YAML 파일에서 설정 로드하여 생성"""
        multi_config = MultiCameraConfig.from_yaml(yaml_path)
        return cls(multi_config=multi_config)

    @property
    def camera_names(self) -> List[str]:
        """연결된 카메라 이름 목록"""
        return list(self.cameras.keys())

    @property
    def is_connected(self) -> bool:
        """모든 카메라 연결 상태"""
        return self._is_connected and len(self.cameras) > 0

    def get_camera(self, name: str) -> Any:
        """
        특정 카메라 인스턴스 반환

        Detection과 Recording에서 카메라를 공유할 때 사용.

        Args:
            name: 카메라 이름 (예: "realsense")

        Returns:
            카메라 인스턴스 (RealSenseCamera 또는 OpenCVCamera)

        Raises:
            KeyError: 카메라를 찾을 수 없는 경우
        """
        if name not in self.cameras:
            raise KeyError(f"Camera '{name}' not found. Available: {list(self.cameras.keys())}")
        return self.cameras[name]

    def connect_all(self, warmup: bool = True) -> bool:
        """
        모든 카메라 연결

        Args:
            warmup: warmup 수행 여부

        Returns:
            bool: 성공 여부
        """
        if self._is_connected:
            print("[MultiCamera] Already connected")
            return True

        print(f"\n[MultiCamera] Connecting {len(self.configs)} camera(s)...")

        success_count = 0

        for config in self.configs:
            try:
                camera = self._create_camera(config)
                camera.connect(warmup=warmup)
                self.cameras[config.name] = camera
                success_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to connect {config.name}: {e}")

        self._is_connected = success_count > 0

        print(f"[MultiCamera] Connected {success_count}/{len(self.configs)} camera(s)")
        return success_count == len(self.configs)

    def _create_camera(self, config: CameraConfig):
        """설정에 맞는 카메라 객체 생성"""
        if isinstance(config, RealSenseCameraConfig):
            return RealSenseCamera(config)
        elif isinstance(config, OpenCVCameraConfig):
            return OpenCVCamera(config)
        else:
            raise ValueError(f"Unknown camera config type: {type(config)}")

    def disconnect_all(self) -> None:
        """모든 카메라 연결 해제"""
        print(f"\n[MultiCamera] Disconnecting {len(self.cameras)} camera(s)...")

        for name, camera in self.cameras.items():
            try:
                camera.disconnect()
            except Exception as e:
                print(f"  [ERROR] Failed to disconnect {name}: {e}")

        self.cameras.clear()
        self._is_connected = False
        print("[MultiCamera] All cameras disconnected")

    # =========================================================================
    # LeRobot 공식 구현과 동일한 방식
    # =========================================================================

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        LeRobot 공식 구현과 동일한 방식으로 observation 반환

        각 카메라에서 async_read()로 최신 프레임을 가져옴.
        백그라운드 스레드가 연속 캡처 중이므로 즉시 반환됨.

        Returns:
            Dict[str, np.ndarray]: {observation.images.{name}: image}
        """
        if not self._is_connected:
            raise RuntimeError("[MultiCamera] Not connected")

        obs_dict = {}

        for cam_name, camera in self.cameras.items():
            start = time.perf_counter()
            obs_dict[f"observation.images.{cam_name}"] = camera.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            # 디버그 로그 (필요시 활성화)
            # print(f"  {cam_name}: {dt_ms:.1f}ms")

        return obs_dict

    def async_read_all(self) -> Dict[str, np.ndarray]:
        """
        모든 카메라에서 async_read로 이미지 읽기 (병렬)

        Returns:
            Dict[str, np.ndarray]: {camera_name: image}
        """
        if not self._is_connected:
            raise RuntimeError("[MultiCamera] Not connected")

        if len(self.cameras) <= 1:
            images = {}
            for name, camera in self.cameras.items():
                try:
                    images[name] = camera.async_read()
                except Exception as e:
                    print(f"  [ERROR] async_read {name}: {e}")
            return images

        # 2대 이상: 병렬 캡처
        from concurrent.futures import ThreadPoolExecutor, as_completed

        images = {}
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            futures = {
                executor.submit(camera.async_read): name
                for name, camera in self.cameras.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    images[name] = future.result()
                except Exception as e:
                    print(f"  [ERROR] async_read {name}: {e}")

        return images

    def read_all(self) -> Dict[str, np.ndarray]:
        """
        모든 카메라에서 동기 read로 이미지 읽기 (블로킹)

        Returns:
            Dict[str, np.ndarray]: {camera_name: image}
        """
        if not self._is_connected:
            raise RuntimeError("[MultiCamera] Not connected")

        images = {}

        for name, camera in self.cameras.items():
            try:
                images[name] = camera.read()
            except Exception as e:
                print(f"  [ERROR] read {name}: {e}")

        return images

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def read(self, camera_name: str) -> np.ndarray:
        """
        특정 카메라에서 이미지 읽기 (동기)

        Args:
            camera_name: 카메라 이름

        Returns:
            np.ndarray: RGB 이미지
        """
        if camera_name not in self.cameras:
            raise KeyError(f"Camera '{camera_name}' not found")

        return self.cameras[camera_name].read()

    def async_read(self, camera_name: str) -> np.ndarray:
        """
        특정 카메라에서 async_read

        Args:
            camera_name: 카메라 이름

        Returns:
            np.ndarray: RGB 이미지
        """
        if camera_name not in self.cameras:
            raise KeyError(f"Camera '{camera_name}' not found")

        return self.cameras[camera_name].async_read()

    def get_feature_shapes(self) -> Dict[str, tuple]:
        """
        LeRobot 데이터셋 feature shapes 반환

        Returns:
            Dict[str, tuple]: {feature_key: (height, width, channels)}
        """
        shapes = {}
        for config in self.configs:
            if config.enabled:
                key = config.to_feature_key()
                shapes[key] = (config.height, config.width, 3)
        return shapes

    def get_info(self) -> Dict[str, Any]:
        """모든 카메라 정보"""
        return {
            name: camera.get_info()
            for name, camera in self.cameras.items()
        }

    def __enter__(self):
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_all()

    def __len__(self):
        return len(self.cameras)


def create_default_cameras() -> MultiCameraManager:
    """
    기본 카메라 설정으로 매니저 생성

    Intel RealSense + Innomaker U20CAM
    """
    from .camera_config import DEFAULT_CAMERA_CONFIGS
    return MultiCameraManager(multi_config=DEFAULT_CAMERA_CONFIGS)
