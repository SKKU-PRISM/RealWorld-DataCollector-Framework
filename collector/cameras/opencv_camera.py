"""
OpenCV Camera Wrapper

Innomaker U20CAM 등 일반 USB 카메라를 위한 래퍼 클래스
LeRobot 공식 구현과 동일한 async_read() 지원
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from threading import Thread, Event, Lock

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

from .camera_config import OpenCVCameraConfig, ColorMode


class OpenCVCamera:
    """
    OpenCV 기반 USB 카메라 래퍼

    Innomaker U20CAM 등 일반 USB 카메라용.
    LeRobot 공식 구현과 동일한 백그라운드 스레드 + async_read() 방식.

    Usage:
        config = OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video7",
            width=640,
            height=480,
        )
        camera = OpenCVCamera(config)
        camera.connect()

        # 동기 읽기 (블로킹)
        image = camera.read()

        # 비동기 읽기 (백그라운드 스레드에서 최신 프레임)
        image = camera.async_read()

        camera.disconnect()
    """

    def __init__(self, config: OpenCVCameraConfig):
        if not HAS_OPENCV:
            raise ImportError(
                "OpenCV not installed. Install with: pip install opencv-python"
            )

        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False

        # 백그라운드 스레드 관련 (LeRobot 공식 구현과 동일)
        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event: Event = Event()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.cap is not None and self.cap.isOpened()

    MAX_CONNECT_RETRIES = 3
    CONNECT_RETRY_DELAY = 2.0  # seconds

    def connect(self, warmup: bool = True) -> None:
        """카메라 연결 (USB 불안정 시 자동 재시도)"""
        if self._is_connected:
            print(f"[OpenCV:{self.name}] Already connected")
            return

        # 장치 열기 (V4L2 백엔드 사용 - Qt 스레드 문제 방지)
        # USB 버스 불안정(RealSense reset 등)으로 간헐적 실패 → 재시도
        for attempt in range(self.MAX_CONNECT_RETRIES):
            self.cap = cv2.VideoCapture(self.config.index_or_path, cv2.CAP_V4L2)
            if self.cap.isOpened():
                break
            if attempt < self.MAX_CONNECT_RETRIES - 1:
                print(f"[OpenCV:{self.name}] Open failed, retrying ({attempt+1}/{self.MAX_CONNECT_RETRIES})...")
                self.cap.release()
                time.sleep(self.CONNECT_RETRY_DELAY)

        if not self.cap.isOpened():
            raise ConnectionError(
                f"[OpenCV:{self.name}] Failed to open {self.config.index_or_path} "
                f"after {self.MAX_CONNECT_RETRIES} attempts"
            )

        # 설정 적용
        self._configure()

        self._is_connected = True

        # Warmup (처음 몇 프레임 버리기)
        if warmup:
            for _ in range(self.config.warmup_frames):
                self.cap.read()

        # 실제 설정 확인
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[OpenCV:{self.name}] Connected: {actual_w}x{actual_h}@{actual_fps:.1f}fps")
        print(f"  Device: {self.config.index_or_path}")

    def _configure(self) -> None:
        """카메라 설정 적용"""
        if not self.cap:
            return

        # FOURCC 설정 (먼저 설정해야 해상도/FPS 설정이 적용됨)
        if self.config.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        # FPS 설정
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # 버퍼 크기 최소화 (지연 줄이기)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def disconnect(self) -> None:
        """카메라 연결 해제"""
        # 백그라운드 스레드 정지
        if self.thread is not None:
            self._stop_read_thread()

        if self.cap:
            self.cap.release()
            self.cap = None

        self._is_connected = False
        print(f"[OpenCV:{self.name}] Disconnected")

    def read(self) -> np.ndarray:
        """
        동기 읽기 - 카메라에서 직접 프레임 캡처 (블로킹)

        Returns:
            np.ndarray: RGB 이미지 (H, W, 3), dtype=uint8
        """
        if not self.is_connected:
            raise RuntimeError(f"[OpenCV:{self.name}] Not connected")

        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"[OpenCV:{self.name}] Failed to read frame")

        # BGR -> RGB 변환 (OpenCV는 기본 BGR)
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    # =========================================================================
    # 비동기 읽기 (LeRobot 공식 구현과 동일)
    # =========================================================================

    _MAX_READ_RETRIES = 3
    _READ_RETRY_DELAY = 0.5  # seconds

    def _reconnect_cap(self) -> bool:
        """VideoCapture 재연결 시도. 성공 시 True."""
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.config.index_or_path, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self._configure()
                # warmup: 처음 몇 프레임 버리기
                for _ in range(min(self.config.warmup_frames, 3)):
                    self.cap.read()
                return True
            else:
                self.cap.release()
                self.cap = None
                return False
        except Exception:
            return False

    def _read_loop(self) -> None:
        """
        백그라운드 스레드에서 실행되는 연속 캡처 루프

        LeRobot 공식 구현과 동일:
        1. 프레임 캡처
        2. latest_frame에 저장 (thread-safe)
        3. new_frame_event 설정

        cap.read() 실패 시 VideoCapture 재연결을 시도하여
        일시적 USB 단절로부터 자동 복구합니다.
        """
        if self.stop_event is None:
            raise RuntimeError(f"[OpenCV:{self.name}] stop_event not initialized")

        consecutive_failures = 0

        while not self.stop_event.is_set():
            try:
                frame = self.read()

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

                consecutive_failures = 0

            except RuntimeError:
                consecutive_failures += 1
                if consecutive_failures <= self._MAX_READ_RETRIES:
                    print(
                        f"[OpenCV:{self.name}] cap.read() failed, "
                        f"reconnecting ({consecutive_failures}/{self._MAX_READ_RETRIES})..."
                    )
                    time.sleep(self._READ_RETRY_DELAY)
                    if self._reconnect_cap():
                        print(f"[OpenCV:{self.name}] Reconnected successfully")
                        continue
                # 재연결 실패 or 최대 재시도 초과
                print(f"[OpenCV:{self.name}] Giving up after {consecutive_failures} retries")
                break
            except Exception as e:
                print(f"[OpenCV:{self.name}] Background read error: {e}")
                time.sleep(0.05)

    def _start_read_thread(self) -> None:
        """백그라운드 읽기 스레드 시작"""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            name=f"OpenCV_{self.name}_read_loop",
            daemon=True,
        )
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """백그라운드 읽기 스레드 정지"""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        비동기 읽기 - 백그라운드 스레드에서 캡처된 최신 프레임 반환

        LeRobot 공식 구현과 동일한 방식:
        - 백그라운드 스레드가 연속으로 프레임 캡처
        - async_read()는 가장 최신 프레임을 즉시 반환

        Args:
            timeout_ms: 프레임 대기 최대 시간 (밀리초)

        Returns:
            np.ndarray: RGB 이미지 (H, W, 3), dtype=uint8

        Raises:
            RuntimeError: 연결되지 않은 경우
            TimeoutError: timeout 내에 프레임을 받지 못한 경우
        """
        if not self.is_connected:
            raise RuntimeError(f"[OpenCV:{self.name}] Not connected")

        # 백그라운드 스레드가 없으면 시작
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # 새 프레임 대기
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"[OpenCV:{self.name}] Timeout waiting for frame ({timeout_ms}ms). "
                f"Thread alive: {thread_alive}"
            )

        # 최신 프레임 반환
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"[OpenCV:{self.name}] No frame available")

        return frame

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """카메라 정보 반환"""
        info = {
            "name": self.name,
            "type": "OpenCV",
            "device": self.config.index_or_path,
            "connected": self._is_connected,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "async_thread_alive": self.thread is not None and self.thread.is_alive(),
        }

        if self.cap and self.cap.isOpened():
            info["actual_width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["actual_height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["actual_fps"] = self.cap.get(cv2.CAP_PROP_FPS)

            # FOURCC
            fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
            info["fourcc"] = fourcc_str

        return info

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        연결된 OpenCV 카메라 목록

        Linux에서 /dev/video* 스캔
        """
        if not HAS_OPENCV:
            return []

        import platform

        cameras = []

        if platform.system() == "Linux":
            # Linux: /dev/video* 스캔 (V4L2 백엔드 사용)
            video_devices = sorted(Path("/dev").glob("video*"))

            for device in video_devices:
                device_path = str(device)
                cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

                if cap.isOpened():
                    cameras.append({
                        "name": f"OpenCV @ {device_path}",
                        "type": "OpenCV",
                        "device": device_path,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                    })
                    cap.release()
        else:
            # Windows/macOS: 인덱스 스캔
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append({
                        "name": f"OpenCV @ index {i}",
                        "type": "OpenCV",
                        "device": i,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                    })
                    cap.release()

        return cameras

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
