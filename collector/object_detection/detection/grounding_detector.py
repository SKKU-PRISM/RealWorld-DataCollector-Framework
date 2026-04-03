"""
Grounding DINO based Object Detection Module
자연어 쿼리로 객체를 탐지하고 바운딩박스 반환
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    """탐지 결과"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    center: Tuple[int, int]  # (cx, cy) in pixels
    bbox_size_px: Tuple[int, int] = None  # (width, height) in pixels
    bbox_size_m: Tuple[float, float] = None  # (width, height) in meters (depth 기반)

    def __post_init__(self):
        """bbox에서 자동으로 pixel 크기 계산"""
        if self.bbox_size_px is None and self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            self.bbox_size_px = (x2 - x1, y2 - y1)


class GroundingDINODetector:
    """Grounding DINO 기반 자연어 객체 탐지기"""

    def __init__(self,
                 model_id: str = "IDEA-Research/grounding-dino-base",
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.25,
                 device: str = None):
        """
        초기화

        Args:
            model_id: HuggingFace 모델 ID
            box_threshold: 박스 신뢰도 임계값
            text_threshold: 텍스트 매칭 임계값
            device: cuda 또는 cpu
        """
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self) -> None:
        """모델 로드"""
        if self._is_loaded:
            return

        print(f"[Detection] Loading Grounding DINO on {self.device}...")

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            print("[Detection] Model loaded successfully")

        except ImportError:
            print("[Detection] Error: transformers library not found")
            print("  Install with: pip install transformers")
            raise

    def detect(self, image: np.ndarray, text_query: str) -> List[Detection]:
        """
        자연어 쿼리로 객체 탐지

        Args:
            image: BGR 이미지 (OpenCV 형식)
            text_query: 찾을 객체 설명 (예: "red cup", "yellow ball")

        Returns:
            Detection 객체 리스트
        """
        if not self._is_loaded:
            self.load_model()

        # BGR to RGB
        image_rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        # 전처리
        inputs = self.processor(
            images=pil_image,
            text=text_query,
            return_tensors="pt"
        ).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 후처리 (새 API)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            threshold=self.box_threshold,
        )[0]

        detections = []
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        # threshold 필터링
        mask = scores > self.box_threshold
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()
        labels = [labels[i] for i in range(len(mask)) if mask[i]]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detection = Detection(
                label=label,
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                center=(cx, cy)
            )
            detections.append(detection)

        return detections

    def detect_and_draw(self, image: np.ndarray, text_query: str) -> Tuple[np.ndarray, List[Detection]]:
        """
        탐지 수행 및 결과 시각화

        Args:
            image: BGR 이미지
            text_query: 검색 쿼리

        Returns:
            (시각화된 이미지, Detection 리스트)
        """
        import cv2

        detections = self.detect(image, text_query)
        vis_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center

            # 바운딩박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 중심점
            cv2.circle(vis_image, (cx, cy), 5, (0, 0, 255), -1)

            # 라벨
            label_text = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(vis_image, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image, detections

    def find_best_match(self, image: np.ndarray, text_query: str) -> Optional[Detection]:
        """
        가장 신뢰도가 높은 탐지 결과 반환

        Args:
            image: BGR 이미지
            text_query: 검색 쿼리

        Returns:
            가장 높은 신뢰도의 Detection 또는 None
        """
        detections = self.detect(image, text_query)

        if not detections:
            return None

        return max(detections, key=lambda d: d.confidence)


class OWLv2Detector:
    """OWL-ViT v2 기반 대안 탐지기 (Grounding DINO 대안)"""

    def __init__(self,
                 model_id: str = "google/owlv2-base-patch16-ensemble",
                 score_threshold: float = 0.1,
                 device: str = None):
        """
        초기화

        Args:
            model_id: HuggingFace 모델 ID
            score_threshold: 신뢰도 임계값
            device: cuda 또는 cpu
        """
        self.model_id = model_id
        self.score_threshold = score_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self) -> None:
        """모델 로드"""
        if self._is_loaded:
            return

        print(f"[Detection] Loading OWL-ViT v2 on {self.device}...")

        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

            self.processor = Owlv2Processor.from_pretrained(self.model_id)
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            print("[Detection] Model loaded successfully")

        except ImportError:
            print("[Detection] Error: transformers library not found")
            raise

    def detect(self, image: np.ndarray, text_queries: List[str]) -> List[Detection]:
        """
        텍스트 쿼리 리스트로 객체 탐지

        Args:
            image: BGR 이미지
            text_queries: 찾을 객체 리스트 (예: ["red cup", "yellow ball"])

        Returns:
            Detection 객체 리스트
        """
        if not self._is_loaded:
            self.load_model()

        # BGR to RGB
        image_rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        # 전처리
        inputs = self.processor(
            text=[text_queries],
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 후처리
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detection = Detection(
                label=text_queries[label_idx],
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                center=(cx, cy)
            )
            detections.append(detection)

        return detections


# 편의를 위한 팩토리 함수
def create_detector(detector_type: str = "grounding_dino", **kwargs):
    """
    탐지기 생성

    Args:
        detector_type: "grounding_dino" 또는 "owlv2"
        **kwargs: 탐지기별 추가 인자

    Returns:
        탐지기 인스턴스
    """
    if detector_type == "grounding_dino":
        return GroundingDINODetector(**kwargs)
    elif detector_type == "owlv2":
        return OWLv2Detector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


# 테스트
if __name__ == "__main__":
    import cv2
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from camera.realsense import RealSenseD435

    print("Grounding DINO Detection Test")
    print("Enter object name to detect (e.g., 'red cup', 'yellow ball')")
    print("Press 'q' to quit")

    detector = GroundingDINODetector()
    detector.load_model()

    with RealSenseD435() as camera:
        current_query = "object"

        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            # 탐지 수행
            vis_image, detections = detector.detect_and_draw(color, current_query)

            # 결과 표시
            cv2.putText(vis_image, f"Query: {current_query}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Found: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Detection", vis_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # 새 쿼리 입력
                cv2.destroyWindow("Detection")
                current_query = input("Enter new query: ")

    cv2.destroyAllWindows()
