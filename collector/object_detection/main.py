#!/usr/bin/env python3
"""
Object Detection System
자연어로 객체를 찾아 픽셀 좌표 및 depth를 반환

사용법:
    python main.py --query "red cup"    # 객체 찾기
    python main.py --interactive        # 대화형 모드
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple

from camera import RealSenseD435
from detection import GroundingDINODetector, Detection


class ObjectLocalizationSystem:
    """자연어 기반 객체 탐지 시스템"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        시스템 초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)

        # 컴포넌트 초기화
        self.camera: Optional[RealSenseD435] = None
        self.detector: Optional[GroundingDINODetector] = None

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        config_full_path = Path(__file__).parent / config_path
        if config_full_path.exists():
            with open(config_full_path) as f:
                return yaml.safe_load(f)
        else:
            # 기본 설정
            return {
                'camera': {'width': 640, 'height': 480, 'fps': 30},
                'detection': {'box_threshold': 0.25, 'text_threshold': 0.25, 'device': 'cuda'}
            }

    def initialize(self, load_detector: bool = True) -> None:
        """시스템 초기화"""
        print("[System] Initializing...")

        # 카메라 초기화
        cam_cfg = self.config['camera']
        self.camera = RealSenseD435(
            width=cam_cfg['width'],
            height=cam_cfg['height'],
            fps=cam_cfg['fps']
        )
        self.camera.start()
        print("[System] Camera initialized")

        # 탐지기 초기화 (선택적)
        if load_detector:
            det_cfg = self.config['detection']
            self.detector = GroundingDINODetector(
                box_threshold=det_cfg['box_threshold'],
                text_threshold=det_cfg['text_threshold'],
                device=det_cfg['device']
            )
            self.detector.load_model()
            print("[System] Detector initialized")

    def shutdown(self) -> None:
        """시스템 종료"""
        if self.camera:
            self.camera.stop()
        print("[System] Shutdown complete")

    def find_object(self, query: str) -> Optional[dict]:
        """
        자연어 쿼리로 객체 찾기

        Args:
            query: 객체 설명 (예: "red cup", "yellow ball")

        Returns:
            {
                'label': str,
                'confidence': float,
                'pixel': (u, v),
                'depth_m': float
            }
            또는 못 찾으면 None
        """
        if self.detector is None:
            print("[System] Error: Detector not initialized")
            return None

        # 이미지 캡처
        color, depth = self.camera.get_frames()
        if color is None:
            return None

        # 객체 탐지
        detection = self.detector.find_best_match(color, query)
        if detection is None:
            print(f"[System] Object not found: '{query}'")
            return None

        cx, cy = detection.center
        depth_m = self.camera.get_depth_at_pixel(cx, cy, depth)

        result = {
            'label': detection.label,
            'confidence': detection.confidence,
            'pixel': (cx, cy),
            'depth_m': depth_m
        }

        return result

    def run_interactive(self) -> None:
        """대화형 모드 실행"""
        print("\n" + "="*60)
        print("Interactive Object Localization")
        print("="*60)
        print("Commands:")
        print("  - Type object name to find (e.g., 'red cup')")
        print("  - 'v' : Toggle visualization")
        print("  - 'q' : Quit")
        print("="*60 + "\n")

        show_visualization = True
        current_query = ""

        while True:
            # 프레임 가져오기
            color, depth = self.camera.get_frames()
            if color is None:
                continue

            display = color.copy()

            # 현재 쿼리가 있으면 탐지 수행
            if current_query and self.detector:
                vis_image, detections = self.detector.detect_and_draw(color, current_query)

                if detections:
                    for det in detections:
                        cx, cy = det.center
                        depth_m = self.camera.get_depth_at_pixel(cx, cy, depth)

                        coord_text = f"Pixel: ({cx}, {cy})"
                        cv2.putText(vis_image, coord_text,
                                   (det.bbox[0], det.bbox[3] + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        depth_text = f"Depth: {depth_m*100:.1f} cm"
                        cv2.putText(vis_image, depth_text,
                                   (det.bbox[0], det.bbox[3] + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    display = vis_image

            # 상태 표시
            status = f"Query: '{current_query}'" if current_query else "Enter object name in terminal"
            cv2.putText(display, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if show_visualization:
                cv2.imshow("Object Localization", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('v'):
                show_visualization = not show_visualization
                if not show_visualization:
                    cv2.destroyAllWindows()
            elif key == ord('n'):
                # 새 쿼리 입력
                print("\nEnter object to find: ", end="", flush=True)
                current_query = input().strip()
                if current_query:
                    result = self.find_object(current_query)
                    if result:
                        print(f"\n[Result] Found '{result['label']}'")
                        print(f"  Confidence: {result['confidence']:.2f}")
                        print(f"  Pixel: {result['pixel']}")
                        print(f"  Depth: {result['depth_m']*100:.2f} cm")

        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Object Detection System")
    parser.add_argument('--query', type=str, help='Object to find (e.g., "red cup")')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # 시스템 초기화
    system = ObjectLocalizationSystem(args.config)

    try:
        # 단일 쿼리 모드
        if args.query:
            system.initialize(load_detector=True)
            result = system.find_object(args.query)

            if result:
                print(f"\n{'='*50}")
                print(f"Object: {result['label']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Pixel: {result['pixel']}")
                print(f"Depth: {result['depth_m']*100:.2f} cm")
                print(f"{'='*50}\n")
            else:
                print(f"\nObject '{args.query}' not found\n")

        # 대화형 모드
        elif args.interactive:
            system.initialize(load_detector=True)
            system.run_interactive()

        # 기본: 대화형 모드
        else:
            print("Usage:")
            print("  python main.py --query 'object' # Find object")
            print("  python main.py --interactive    # Interactive mode")

    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
