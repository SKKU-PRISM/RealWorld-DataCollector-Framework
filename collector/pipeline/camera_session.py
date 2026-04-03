"""
camera_session — 파이프라인 카메라 세션 관리.

싱글/멀티암 공용. 카메라 초기화, 캡처, 종료, 복구를 통합.
cameras/ 패키지(하드웨어 드라이버)는 건드리지 않고,
파이프라인 레벨의 카메라 조작만 담당.
"""

import time
import numpy as np
from typing import Optional


class PipelineCamera:
    """파이프라인에서 카메라를 사용하는 공통 인터페이스.

    camera_manager(레코딩용)와 standalone camera를 통합 관리.
    """

    # RealSense 카메라 이름 후보 (config에 따라 다름)
    REALSENSE_NAMES = ["top", "realsense"]

    def __init__(self, camera_manager=None, verbose: bool = True):
        """
        Args:
            camera_manager: MultiCameraManager (레코딩 시 생성됨, 없으면 None)
            verbose: 로그 출력 여부
        """
        self.camera_manager = camera_manager
        self.camera = None  # 현재 활성 카메라 (RealSense 인스턴스)
        self.verbose = verbose

    def get_realsense(self):
        """camera_manager에서 RealSense 카메라를 가져오기 (이름 호환).

        Returns:
            카메라 인스턴스 또는 None
        """
        if not self.camera_manager:
            return None
        for cam_name in self.REALSENSE_NAMES:
            try:
                return self.camera_manager.get_camera(cam_name)
            except KeyError:
                continue
        return None

    def initialize(self) -> bool:
        """카메라 초기화.

        1순위: camera_manager에서 RealSense 가져오기
        2순위: 직접 RealSense 연결 (fallback)
        """
        # 1순위: camera_manager
        if self.camera_manager:
            if not self.camera_manager.is_connected:
                try:
                    self.camera_manager.connect_all()
                except Exception as e:
                    print(f"[Camera] Manager reconnect failed: {e}")
            if self.camera_manager.is_connected:
                cam = self.get_realsense()
                if cam is not None:
                    self.camera = cam
                    if self.verbose:
                        print(f"[Camera] Initialized (from camera_manager)")
                    return True

        # 2순위: 직접 연결
        try:
            from object_detection.camera import RealSenseD435
            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            for _ in range(30):
                self.camera.get_frames()
            if self.verbose:
                print("[Camera] Initialized (direct)")
            return True
        except Exception as e:
            print(f"[Camera] Initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """카메라 종료.

        camera_manager 소유 카메라는 참조만 해제 (manager가 관리).
        standalone 카메라는 stop() 호출.
        """
        if not self.camera:
            return

        # camera_manager 소유 여부 확인
        if self.camera_manager:
            cm_cam = self.get_realsense()
            if self.camera is cm_cam:
                self.camera = None
                if self.verbose:
                    print("[Camera] Reference cleared (managed by camera_manager)")
                return

        # standalone 카메라 종료
        try:
            self.camera.stop()
        except Exception:
            pass
        self.camera = None
        if self.verbose:
            print("[Camera] Shutdown")

    def capture_frame(self) -> Optional[np.ndarray]:
        """프레임 캡처.

        1순위: camera_manager의 RealSense
        2순위: self.camera (standalone)
        """
        try:
            # 1순위: camera_manager
            if self.camera_manager and self.camera_manager.is_connected:
                cam = self.get_realsense()
                if cam is not None:
                    try:
                        color, _ = cam.get_frames()
                        return color
                    except Exception:
                        pass

            # 2순위: self.camera
            if self.camera is not None:
                color, _ = self.camera.get_frames()
                return color

            return None
        except Exception as e:
            print(f"[Camera] Frame capture failed: {e}")
            return None

    def force_recovery(self) -> bool:
        """카메라 강제 복구 (disconnect → reconnect).

        USB 에러 등으로 카메라가 비정상 상태일 때 사용.
        Returns:
            True if recovery succeeded
        """
        # 1. 카메라 강제 stop
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

        # 2. camera_manager disconnect → reconnect
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
            except Exception:
                pass
            time.sleep(2.0)
            try:
                self.camera_manager.connect_all()
                print(f"[Camera] Recovery: camera_manager reconnected")
            except Exception as e:
                print(f"[Camera] Recovery: reconnect failed: {e}")
                self.camera_manager = None
                return False
        else:
            time.sleep(2.0)

        # 3. 카메라 재초기화
        return self.initialize()
