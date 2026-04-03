"""
Async Camera Capture

제어 루프를 블로킹하지 않는 진정한 비동기 카메라 캡처.

백그라운드 스레드에서 연속적으로 카메라 이미지를 캡처하고,
제어 루프에서는 최신 이미지만 즉시 가져갑니다 (블로킹 없음).

Usage:
    from record_dataset.async_camera import AsyncCameraCapture

    async_capture = AsyncCameraCapture(camera_manager)
    async_capture.start()

    # 제어 루프에서
    images = async_capture.get_latest_images()  # 즉시 반환 (블로킹 없음!)

    async_capture.stop()
"""

import time
import numpy as np
from typing import Dict, Optional, Any
from threading import Thread, Lock, Event


class AsyncCameraCapture:
    """
    비동기 카메라 캡처 스레드

    백그라운드에서 연속적으로 카메라 이미지를 캡처하여 버퍼에 저장.
    제어 루프는 버퍼에서 최신 이미지만 즉시 가져가므로 블로킹 없음.

    Args:
        camera_manager: MultiCameraManager 인스턴스
        capture_fps: 카메라 캡처 FPS (기본: 60Hz, 제어 루프보다 빠르게)
        buffer_size: 이미지 버퍼 크기 (기본: 2, 최신 2개만 유지)
    """

    def __init__(
        self,
        camera_manager: Any,
        capture_fps: int = 60,
        buffer_size: int = 2,
    ):
        self.camera_manager = camera_manager
        self.capture_fps = capture_fps
        self.buffer_size = buffer_size

        # 최신 이미지 버퍼 (thread-safe)
        self._latest_images: Optional[Dict[str, np.ndarray]] = None
        self._latest_timestamp: float = 0.0
        self._image_lock = Lock()

        # 스레드 제어
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._is_running = False

        # 통계
        self._total_captures = 0
        self._capture_errors = 0
        self._start_time = 0.0

    def start(self) -> None:
        """백그라운드 캡처 스레드 시작"""
        if self._is_running:
            print("[AsyncCamera] Already running")
            return

        self._stop_event.clear()
        self._is_running = True
        self._start_time = time.time()
        self._total_captures = 0
        self._capture_errors = 0

        self._capture_thread = Thread(
            target=self._capture_loop,
            name="AsyncCameraCapture",
            daemon=True,
        )
        self._capture_thread.start()

        print(f"[AsyncCamera] Started (target: {self.capture_fps} fps)")

    def stop(self) -> None:
        """백그라운드 캡처 스레드 정지"""
        if not self._is_running:
            return

        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        self._is_running = False

        # 통계 출력
        elapsed = time.time() - self._start_time
        actual_fps = self._total_captures / elapsed if elapsed > 0 else 0

        print(f"[AsyncCamera] Stopped")
        print(f"  Total captures: {self._total_captures}")
        print(f"  Capture errors: {self._capture_errors}")
        print(f"  Actual capture FPS: {actual_fps:.1f}")

    def _capture_loop(self) -> None:
        """
        백그라운드 캡처 루프

        지속적으로 카메라에서 이미지를 캡처하여 버퍼에 저장.
        제어 루프보다 빠르게 실행되어야 함 (60fps vs 50Hz).
        """
        interval = 1.0 / self.capture_fps
        next_capture = time.time()

        while not self._stop_event.is_set():
            try:
                # 카메라에서 이미지 캡처
                start = time.time()
                images = self.camera_manager.async_read_all()
                capture_time = (time.time() - start) * 1000  # ms

                if images:
                    # 최신 이미지 업데이트 (thread-safe)
                    with self._image_lock:
                        self._latest_images = images
                        self._latest_timestamp = time.time()

                    self._total_captures += 1

                    # 디버그 (처음 10개만)
                    if self._total_captures <= 10:
                        print(f"[AsyncCamera] Capture #{self._total_captures}: "
                              f"{capture_time:.1f}ms, cameras: {list(images.keys())}")

            except Exception as e:
                self._capture_errors += 1
                if self._capture_errors <= 5:  # 처음 5개만 출력
                    print(f"[AsyncCamera] Capture error: {e}")

            # 일정한 간격 유지
            next_capture += interval
            sleep_time = next_capture - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 지연 발생 시 다음 캡처 시간 재조정
                next_capture = time.time()

    def get_latest_images(self) -> Optional[Dict[str, np.ndarray]]:
        """
        최신 이미지 가져오기 (블로킹 없음!)

        백그라운드 스레드가 캡처한 최신 이미지를 즉시 반환.
        이미지 복사본을 반환하여 thread-safe 보장.

        Returns:
            Dict[str, np.ndarray]: {camera_name: image} 또는 None
        """
        with self._image_lock:
            if self._latest_images is None:
                return None

            # 이미지 복사 (thread-safe)
            # numpy array는 참조이므로 복사 필요
            images_copy = {
                name: img.copy()
                for name, img in self._latest_images.items()
            }

            return images_copy

    def get_latest_timestamp(self) -> float:
        """최신 이미지의 타임스탬프 반환"""
        with self._image_lock:
            return self._latest_timestamp

    def get_capture_stats(self) -> Dict:
        """캡처 통계 반환"""
        elapsed = time.time() - self._start_time
        actual_fps = self._total_captures / elapsed if elapsed > 0 else 0

        return {
            "is_running": self._is_running,
            "target_fps": self.capture_fps,
            "actual_fps": actual_fps,
            "total_captures": self._total_captures,
            "capture_errors": self._capture_errors,
            "latest_timestamp": self._latest_timestamp,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
