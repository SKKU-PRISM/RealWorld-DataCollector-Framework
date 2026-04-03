"""
RealSense D435 Camera Module
RGB + Depth 스트림을 동기화하여 제공
"""

import pyrealsense2 as rs
import numpy as np
from typing import Tuple, Optional


class RealSenseD435:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        D435 카메라 초기화

        Args:
            width: 이미지 너비
            height: 이미지 높이
            fps: 프레임 레이트
        """
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.intrinsics = None

        self._is_running = False

    def _hardware_reset(self) -> None:
        """RealSense 하드웨어 리셋 후 재초기화."""
        import time
        print("[Camera] Hardware reset...")
        try:
            self.pipeline.stop()
        except Exception:
            pass
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            devs[0].hardware_reset()
        time.sleep(5)
        self.pipeline = rs.pipeline()
        self.config = rs.config()

    def start(self) -> None:
        """카메라 스트림 시작 (실패 시 자동 리셋 후 재시도)."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                self._start_pipeline()
                return
            except RuntimeError as e:
                if "Frame didn't arrive" in str(e) or "wait_for_frames" in str(e):
                    if attempt < max_retries:
                        print(f"[Camera] Frame timeout, resetting device (retry {attempt + 1}/{max_retries})...")
                        self._hardware_reset()
                    else:
                        raise
                else:
                    raise

    def _start_pipeline(self) -> None:
        """카메라 파이프라인 시작 (내부 구현)."""
        # RGB와 Depth 스트림 설정
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        # 파이프라인 시작
        profile = self.pipeline.start(self.config)

        # Depth를 RGB에 정렬
        self.align = rs.align(rs.stream.color)

        # Warmup — 여기서 Frame timeout이 발생할 수 있음
        self.pipeline.wait_for_frames(timeout_ms=5000)

        # 카메라 내부 파라미터 가져오기
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self._is_running = True
        print(f"[Camera] Started: {self.width}x{self.height} @ {self.fps}fps")

    def stop(self) -> None:
        """카메라 스트림 중지"""
        if self._is_running:
            self.pipeline.stop()
            self._is_running = False
            print("[Camera] Stopped")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RGB와 정렬된 Depth 프레임 가져오기

        Returns:
            (color_image, depth_image) 튜플
            - color_image: BGR 이미지 (H, W, 3)
            - depth_image: 깊이 이미지 (H, W), 단위: mm
        """
        if not self._is_running:
            return None, None

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def get_depth_at_pixel(self, u: int, v: int, depth_image: np.ndarray, patch_size: int = 3) -> float:
        """
        특정 픽셀 주변 영역의 median 깊이값 반환 (미터 단위)

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_image: 깊이 이미지
            patch_size: 주변 패치 크기 (patch_size x patch_size, 홀수)

        Returns:
            깊이값 (meters), median 기반으로 노이즈에 강건
        """
        if not (0 <= u < self.width and 0 <= v < self.height):
            return 0.0

        half = patch_size // 2
        v_min = max(0, v - half)
        v_max = min(self.height, v + half + 1)
        u_min = max(0, u - half)
        u_max = min(self.width, u + half + 1)

        patch = depth_image[v_min:v_max, u_min:u_max]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0

        depth_mm = float(np.median(valid))
        return depth_mm * 0.001  # mm to meters

    def pixel_to_camera_coords(self, u: int, v: int, depth_m: float) -> Tuple[float, float, float]:
        """
        픽셀 좌표 + 깊이 → 카메라 3D 좌표

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_m: 깊이 (meters)

        Returns:
            (X, Y, Z) 카메라 좌표 (meters)
        """
        if self.intrinsics is None:
            raise RuntimeError("Camera not started. Call start() first.")

        # 카메라 내부 파라미터
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.ppx
        cy = self.intrinsics.ppy

        # 역투영 (픽셀 → 3D)
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m

        return X, Y, Z

    def get_intrinsics(self) -> dict:
        """카메라 내부 파라미터 반환"""
        if self.intrinsics is None:
            raise RuntimeError("Camera not started. Call start() first.")

        return {
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'cx': self.intrinsics.ppx,
            'cy': self.intrinsics.ppy,
            'width': self.intrinsics.width,
            'height': self.intrinsics.height
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 간단한 테스트
if __name__ == "__main__":
    import cv2

    with RealSenseD435() as camera:
        print("Camera intrinsics:", camera.get_intrinsics())

        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            # 깊이 시각화
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 나란히 표시
            images = np.hstack((color, depth_colormap))
            cv2.imshow('RealSense D435', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
