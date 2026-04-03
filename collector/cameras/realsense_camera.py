"""
Intel RealSense Camera Wrapper

RealSense D435 카메라를 위한 래퍼 클래스
LeRobot 공식 구현과 동일한 async_read() 지원
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from threading import Thread, Event, Lock

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    rs = None

from .camera_config import RealSenseCameraConfig, ColorMode


class RealSenseCamera:
    """
    Intel RealSense D435 카메라 래퍼

    RGB 이미지 캡처를 위해 최적화됨.
    LeRobot 공식 구현과 동일한 백그라운드 스레드 + async_read() 방식.

    Usage:
        config = RealSenseCameraConfig(name="front", width=640, height=480)
        camera = RealSenseCamera(config)
        camera.connect()

        # 동기 읽기 (블로킹)
        image = camera.read()

        # 비동기 읽기 (백그라운드 스레드에서 최신 프레임)
        image = camera.async_read()

        camera.disconnect()
    """

    def __init__(self, config: RealSenseCameraConfig):
        if not HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )

        self.config = config
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.intrinsics: Optional[rs.intrinsics] = None
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
        return self._is_connected

    def _hardware_reset(self) -> None:
        """RealSense 하드웨어 리셋 후 재초기화."""
        import time
        print(f"[RealSense:{self.name}] Hardware reset...")
        try:
            self.pipeline.stop()
        except Exception:
            pass
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            devs[0].hardware_reset()
        time.sleep(5)
        self._is_connected = False

    def connect(self, warmup: bool = True) -> None:
        """카메라 연결 및 스트림 시작 (실패 시 자동 리셋 후 재시도).

        카메라 식별:
        - serial_number가 지정된 경우 해당 카메라 연결
        - serial_number가 없으면 첫 번째 발견된 RealSense 사용
        """
        if self._is_connected:
            print(f"[RealSense:{self.name}] Already connected")
            return

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                self._connect_pipeline(warmup)
                return
            except RuntimeError as e:
                if "Frame didn't arrive" in str(e) or "wait_for_frames" in str(e):
                    if attempt < max_retries:
                        print(f"[RealSense:{self.name}] Frame timeout, resetting device (retry {attempt + 1}/{max_retries})...")
                        self._hardware_reset()
                    else:
                        raise
                else:
                    raise

    def _connect_pipeline(self, warmup: bool = True) -> None:
        """카메라 파이프라인 연결 (내부 구현)."""
        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        # 특정 카메라 지정 (serial_number로만 식별)
        if self.config.serial_number:
            rs_config.enable_device(self.config.serial_number)
            print(f"[RealSense:{self.name}] Using device serial: {self.config.serial_number}")

        # RGB 스트림 설정
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8 if self.config.color_mode == ColorMode.RGB else rs.format.bgr8,
            self.config.fps,
        )

        # Depth 스트림 (옵션)
        if self.config.enable_depth:
            rs_config.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )

        # 파이프라인 시작
        profile = self.pipeline.start(rs_config)

        # Depth 정렬 설정
        if self.config.enable_depth:
            self.align = rs.align(rs.stream.color)

        # Intrinsics 저장
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self._is_connected = True

        # Warmup (처음 몇 프레임 버리기 — 여기서 Frame timeout 발생 가능)
        if warmup:
            for _ in range(30):
                self.pipeline.wait_for_frames()

        print(f"[RealSense:{self.name}] Connected: {self.config.width}x{self.config.height}@{self.config.fps}fps")

    def disconnect(self) -> None:
        """카메라 연결 해제"""
        # 백그라운드 스레드 정지
        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline and self._is_connected:
            self.pipeline.stop()
            self._is_connected = False
            print(f"[RealSense:{self.name}] Disconnected")

    def read(self) -> np.ndarray:
        """
        동기 읽기 - 카메라에서 직접 프레임 캡처 (블로킹)

        Returns:
            np.ndarray: RGB 이미지 (H, W, 3), dtype=uint8
        """
        if not self._is_connected:
            raise RuntimeError(f"[RealSense:{self.name}] Not connected")

        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"[RealSense:{self.name}] Failed to get color frame")

        return np.asanyarray(color_frame.get_data())

    # =========================================================================
    # 비동기 읽기 (LeRobot 공식 구현과 동일)
    # =========================================================================

    def _read_loop(self) -> None:
        """
        백그라운드 스레드에서 실행되는 연속 캡처 루프

        LeRobot 공식 구현과 동일:
        1. 프레임 캡처
        2. latest_frame에 저장 (thread-safe)
        3. new_frame_event 설정
        """
        if self.stop_event is None:
            raise RuntimeError(f"[RealSense:{self.name}] stop_event not initialized")

        while not self.stop_event.is_set():
            try:
                frame = self.read()

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

            except RuntimeError:
                # 연결 끊김
                break
            except Exception as e:
                print(f"[RealSense:{self.name}] Background read error: {e}")

    def _start_read_thread(self) -> None:
        """백그라운드 읽기 스레드 시작"""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            name=f"RealSense_{self.name}_read_loop",
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
        if not self._is_connected:
            raise RuntimeError(f"[RealSense:{self.name}] Not connected")

        # 백그라운드 스레드가 없으면 시작
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # 새 프레임 대기
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"[RealSense:{self.name}] Timeout waiting for frame ({timeout_ms}ms). "
                f"Thread alive: {thread_alive}"
            )

        # 최신 프레임 반환
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"[RealSense:{self.name}] No frame available")

        return frame

    # =========================================================================
    # Depth 관련
    # =========================================================================

    def read_with_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        RGB + Depth 이미지 읽기

        Returns:
            Tuple[np.ndarray, np.ndarray]: (RGB 이미지, Depth 이미지)
        """
        if not self._is_connected:
            raise RuntimeError(f"[RealSense:{self.name}] Not connected")

        if not self.config.enable_depth:
            raise RuntimeError(f"[RealSense:{self.name}] Depth not enabled")

        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError(f"[RealSense:{self.name}] Failed to get frames")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return color, depth

    # =========================================================================
    # Detection 호환 인터페이스 (object_detection/camera/realsense.py 호환)
    # =========================================================================

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detection 모듈 호환용: RGB + Depth 프레임 가져오기

        object_detection/camera/realsense.py의 RealSenseD435와 동일한 인터페이스.
        Recording 카메라를 Detection에서 공유할 수 있게 함.

        Returns:
            (color_image, depth_image) 튜플
            - color_image: BGR 이미지 (H, W, 3) - Detection은 BGR 사용
            - depth_image: 깊이 이미지 (H, W), 단위: mm
        """
        if not self._is_connected:
            return None, None

        if not self.config.enable_depth:
            # Depth 없으면 color만 반환
            color = self.read()
            # RGB to BGR 변환 (detection은 BGR 사용)
            import cv2
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            return color_bgr, None

        try:
            color, depth = self.read_with_depth()
            # RGB to BGR 변환
            import cv2
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            return color_bgr, depth
        except Exception:
            return None, None

    def get_depth_at_pixel(self, u: int, v: int, depth_image: np.ndarray, patch_size: int = 3) -> float:
        """
        Detection 모듈 호환용: 특정 픽셀 주변 영역의 median 깊이값 반환 (미터 단위)

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_image: 깊이 이미지
            patch_size: 주변 패치 크기 (patch_size x patch_size, 홀수)

        Returns:
            깊이값 (meters), median 기반으로 노이즈에 강건
        """
        if depth_image is None:
            return 0.0
        h, w = depth_image.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return 0.0

        half = patch_size // 2
        v_min = max(0, int(v) - half)
        v_max = min(h, int(v) + half + 1)
        u_min = max(0, int(u) - half)
        u_max = min(w, int(u) + half + 1)

        patch = depth_image[v_min:v_max, u_min:u_max]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0

        depth_mm = float(np.median(valid))
        return depth_mm * 0.001  # mm to meters

    def get_intrinsics(self) -> Dict[str, Any]:
        """
        Detection 모듈 호환용: 카메라 내부 파라미터 반환

        Returns:
            dict: fx, fy, cx, cy, width, height
        """
        if self.intrinsics is None:
            raise RuntimeError(f"[RealSense:{self.name}] Camera not connected")

        return {
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'cx': self.intrinsics.ppx,
            'cy': self.intrinsics.ppy,
            'width': self.intrinsics.width,
            'height': self.intrinsics.height,
        }

    def start(self) -> None:
        """Detection 모듈 호환용: connect()의 별칭"""
        self.connect(warmup=True)

    def stop(self) -> None:
        """Detection 모듈 호환용: disconnect()의 별칭"""
        self.disconnect()

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """카메라 정보 반환"""
        info = {
            "name": self.name,
            "type": "RealSense",
            "connected": self._is_connected,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "enable_depth": self.config.enable_depth,
            "async_thread_alive": self.thread is not None and self.thread.is_alive(),
        }

        if self.intrinsics:
            info["intrinsics"] = {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.ppx,
                "cy": self.intrinsics.ppy,
            }

        return info

    @staticmethod
    def find_cameras() -> list:
        """연결된 RealSense 카메라 목록"""
        if not HAS_REALSENSE:
            return []

        ctx = rs.context()
        devices = ctx.query_devices()

        cameras = []
        for dev in devices:
            cameras.append({
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "type": "RealSense",
            })

        return cameras

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
