"""
Coordinate Transform Module
픽셀 좌표 → 카메라 좌표 → 월드 좌표 변환
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class CoordinateTransformer:
    """좌표 변환 통합 클래스"""

    def __init__(self, calibration_file: str = None):
        """
        초기화

        Args:
            calibration_file: 캘리브레이션 파일 경로 (.npz)
        """
        self.homography_matrix: Optional[np.ndarray] = None
        self.transform_matrix_3d: Optional[np.ndarray] = None
        self.z_offset: float = 0.0

        self.camera_intrinsics: Optional[dict] = None

        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)

    def load_calibration(self, filepath: str) -> bool:
        """캘리브레이션 데이터 로드"""
        try:
            data = np.load(filepath, allow_pickle=True)
            self.homography_matrix = data['homography_matrix']
            self.z_offset = float(data['z_offset'])

            if 'transform_matrix_3d' in data:
                self.transform_matrix_3d = data['transform_matrix_3d']
                print("[Transform] 3D transform matrix loaded")

            if 'intrinsics' in data:
                intrinsics_arr = data['intrinsics']
                self.camera_intrinsics = {
                    'fx': float(intrinsics_arr[0]),
                    'fy': float(intrinsics_arr[1]),
                    'cx': float(intrinsics_arr[2]),
                    'cy': float(intrinsics_arr[3])
                }

            print(f"[Transform] Loaded calibration from {filepath}")
            return True
        except Exception as e:
            print(f"[Transform] Failed to load calibration: {e}")
            return False

    def set_camera_intrinsics(self, intrinsics: dict) -> None:
        """카메라 내부 파라미터 설정"""
        self.camera_intrinsics = intrinsics

    def pixel_to_world_2d(self, u: int, v: int) -> Tuple[float, float, float]:
        """
        픽셀 좌표 → 월드 좌표 (2D, 평면 가정)

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표

        Returns:
            (x, y, z) 월드 좌표 (cm)
        """
        if self.homography_matrix is None:
            raise RuntimeError("Calibration not loaded")

        pixel = np.array([[[u, v]]], dtype=np.float32)
        world_2d = cv2.perspectiveTransform(pixel, self.homography_matrix)

        x = float(world_2d[0, 0, 0])
        y = float(world_2d[0, 0, 1])
        z = self.z_offset

        return x, y, z

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        월드 좌표 → 픽셀 좌표 (역변환)

        Args:
            x: 월드 x 좌표 (cm)
            y: 월드 y 좌표 (cm)

        Returns:
            (u, v) 픽셀 좌표
        """
        if self.homography_matrix is None:
            raise RuntimeError("Calibration not loaded")

        # Inverse homography matrix
        H_inv = np.linalg.inv(self.homography_matrix)

        # World coords → pixel coords
        world_point = np.array([[[x, y]]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(world_point, H_inv)

        u = int(round(pixel[0, 0, 0]))
        v = int(round(pixel[0, 0, 1]))

        return u, v

    def pixel_to_camera_3d(self, u: int, v: int, depth_m: float) -> Tuple[float, float, float]:
        """
        픽셀 좌표 + 깊이 → 카메라 3D 좌표

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_m: 깊이 (meters)

        Returns:
            (X, Y, Z) 카메라 좌표 (meters)
        """
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics not set")

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m

        return X, Y, Z

    def camera_to_world_3d(self, cam_x: float, cam_y: float, cam_z: float) -> Tuple[float, float, float]:
        """
        카메라 3D 좌표 → 월드 3D 좌표

        Args:
            cam_x, cam_y, cam_z: 카메라 좌표 (meters)

        Returns:
            (x, y, z) 월드 좌표 (cm)
        """
        if self.transform_matrix_3d is not None:
            # 완전한 3D 변환
            point = np.array([cam_x, cam_y, cam_z, 1.0])
            world_point = self.transform_matrix_3d @ point
            # Z축 부호 반전: 캘리브레이션이 평면(z=0)에서 수행되어 z방향이 뒤집힘
            return float(world_point[0]), float(world_point[1]), float(-world_point[2])
        else:
            # 2D 변환 사용 (Z는 근사)
            # 카메라 좌표를 픽셀로 역변환 후 homography 적용
            raise NotImplementedError("Use pixel_to_world_2d for 2D-only calibration")

    def pixel_depth_to_world(self, u: int, v: int, depth_m: float) -> Tuple[float, float, float]:
        """
        픽셀 좌표 + 깊이 → 월드 좌표 (통합 함수)

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_m: 깊이 (meters)

        Returns:
            (x, y, z) 월드 좌표 (cm)
        """
        if self.transform_matrix_3d is not None and self.camera_intrinsics is not None:
            # 3D 변환 경로
            # transform_matrix_3d는 cm 단위로 계산되었으므로, 카메라 좌표도 cm으로 변환
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']

            depth_cm = depth_m * 100.0
            cam_x = (u - cx) * depth_cm / fx
            cam_y = (v - cy) * depth_cm / fy
            cam_z = depth_cm

            return self.camera_to_world_3d(cam_x, cam_y, cam_z)
        else:
            # 2D 변환 경로 (fallback)
            return self.pixel_to_world_2d(u, v)

    @property
    def is_ready(self) -> bool:
        """변환 준비 완료 여부"""
        return self.homography_matrix is not None


class ObjectLocalizer:
    """객체 위치 추정 통합 클래스"""

    def __init__(self, camera, calibrator_or_file):
        """
        초기화

        Args:
            camera: RealSenseD435 인스턴스
            calibrator_or_file: GridCalibrator 인스턴스 또는 캘리브레이션 파일 경로
        """
        self.camera = camera
        self.transformer = CoordinateTransformer()

        # 캘리브레이션 로드
        if isinstance(calibrator_or_file, str):
            self.transformer.load_calibration(calibrator_or_file)
        else:
            # GridCalibrator 인스턴스에서 직접 복사
            self.transformer.homography_matrix = calibrator_or_file.homography_matrix
            self.transformer.z_offset = calibrator_or_file.z_offset
            if hasattr(calibrator_or_file, 'transform_matrix'):
                self.transformer.transform_matrix_3d = calibrator_or_file.transform_matrix

    def localize_pixel(self, u: int, v: int, depth_image: np.ndarray = None) -> Tuple[float, float, float]:
        """
        픽셀 좌표의 월드 좌표 계산

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_image: 깊이 이미지 (optional)

        Returns:
            (x, y, z) 월드 좌표 (cm)
        """
        if depth_image is not None:
            # 깊이값 추출
            depth_m = self.camera.get_depth_at_pixel(u, v, depth_image)
            if depth_m > 0:
                return self.transformer.pixel_depth_to_world(u, v, depth_m)

        # 깊이 없이 2D 변환
        return self.transformer.pixel_to_world_2d(u, v)

    def localize_detection(self, detection, depth_image: np.ndarray = None) -> dict:
        """
        Detection 객체의 월드 좌표 계산

        Args:
            detection: Detection 객체 (from grounding_detector)
            depth_image: 깊이 이미지

        Returns:
            {
                'label': str,
                'confidence': float,
                'pixel': (u, v),
                'world_coords': (x, y, z),
                'depth_m': float (if available)
            }
        """
        cx, cy = detection.center
        world_coords = self.localize_pixel(cx, cy, depth_image)

        result = {
            'label': detection.label,
            'confidence': detection.confidence,
            'pixel': (cx, cy),
            'world_coords': world_coords
        }

        if depth_image is not None:
            depth_m = self.camera.get_depth_at_pixel(cx, cy, depth_image)
            result['depth_m'] = depth_m

        return result


# cv2 import (pixel_to_world_2d에서 사용)
import cv2


# 테스트
if __name__ == "__main__":
    print("Coordinate Transform Test")

    # 예시 캘리브레이션 데이터 생성
    test_homography = np.array([
        [0.1, 0.0, -32.0],
        [0.0, 0.1, -24.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    transformer = CoordinateTransformer()
    transformer.homography_matrix = test_homography
    transformer.z_offset = 0.0

    # 테스트
    test_pixels = [(320, 240), (0, 0), (640, 480)]
    for u, v in test_pixels:
        world = transformer.pixel_to_world_2d(u, v)
        print(f"Pixel ({u}, {v}) -> World ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) cm")
