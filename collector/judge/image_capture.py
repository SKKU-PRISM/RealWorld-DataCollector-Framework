"""
Image Capture Utility for Judge Module

카메라에서 이미지를 캡처하고 저장하는 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import base64
import sys

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ImageCapture:
    """카메라 이미지 캡처 클래스"""

    def __init__(self):
        """초기화"""
        self.camera = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """카메라 초기화"""
        if self._is_initialized:
            return True

        try:
            sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))
            from object_detection.camera import RealSenseD435

            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            self._is_initialized = True
            print("[ImageCapture] Camera initialized")
            return True

        except Exception as e:
            print(f"[ImageCapture] Failed to initialize camera: {e}")
            return False

    def shutdown(self) -> None:
        """카메라 종료"""
        if self.camera:
            self.camera.stop()
            self._is_initialized = False
            print("[ImageCapture] Camera shutdown")

    def capture(self) -> Optional[np.ndarray]:
        """
        현재 프레임 캡처

        Returns:
            BGR 이미지 numpy array 또는 None
        """
        if not self._is_initialized:
            print("[ImageCapture] Error: Camera not initialized")
            return None

        try:
            color, _ = self.camera.get_frames()
            return color
        except Exception as e:
            print(f"[ImageCapture] Capture failed: {e}")
            return None

    def capture_and_save(self, save_path: str) -> Optional[str]:
        """
        프레임 캡처 후 파일로 저장

        Args:
            save_path: 저장할 파일 경로

        Returns:
            저장된 파일 경로 또는 None
        """
        image = self.capture()
        if image is None:
            return None

        try:
            cv2.imwrite(save_path, image)
            print(f"[ImageCapture] Image saved: {save_path}")
            return save_path
        except Exception as e:
            print(f"[ImageCapture] Save failed: {e}")
            return None

    def __enter__(self):
        """Context manager 진입"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.shutdown()
        return False


def capture_frame(camera=None) -> Optional[np.ndarray]:
    """
    단일 프레임 캡처 (편의 함수)

    Args:
        camera: 기존 카메라 인스턴스 (없으면 새로 생성)

    Returns:
        BGR 이미지 numpy array 또는 None
    """
    if camera is not None:
        # 기존 카메라 사용
        try:
            color, _ = camera.get_frames()
            return color
        except Exception as e:
            print(f"[capture_frame] Failed: {e}")
            return None

    # 새 카메라 인스턴스 생성
    with ImageCapture() as cap:
        return cap.capture()


def image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """
    이미지를 base64 문자열로 변환

    Args:
        image: BGR 이미지 numpy array
        format: 이미지 포맷 ("jpeg" 또는 "png")

    Returns:
        base64 인코딩된 문자열
    """
    if format.lower() == "jpeg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ext = ".jpg"
    else:
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        ext = ".png"

    _, buffer = cv2.imencode(ext, image, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
    """
    base64 문자열을 이미지로 변환

    Args:
        base64_str: base64 인코딩된 문자열

    Returns:
        BGR 이미지 numpy array 또는 None
    """
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[base64_to_image] Failed: {e}")
        return None


def save_image(image: np.ndarray, save_dir: str = None, prefix: str = "capture") -> str:
    """
    이미지를 타임스탬프 기반 파일명으로 저장

    Args:
        image: BGR 이미지 numpy array
        save_dir: 저장 디렉토리 (기본: /tmp)
        prefix: 파일명 접두사

    Returns:
        저장된 파일 경로
    """
    if save_dir is None:
        save_dir = "/tmp"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = Path(save_dir) / filename

    cv2.imwrite(str(filepath), image)
    return str(filepath)


# 테스트
if __name__ == "__main__":
    print("Image Capture Test")
    print("=" * 60)

    with ImageCapture() as cap:
        print("\nCapturing frame...")
        image = cap.capture()

        if image is not None:
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")

            # 저장 테스트
            save_path = save_image(image, prefix="test")
            print(f"  Saved to: {save_path}")

            # base64 변환 테스트
            b64 = image_to_base64(image)
            print(f"  Base64 length: {len(b64)}")

            # 미리보기
            cv2.imshow("Captured Frame", image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("  Failed to capture image")
