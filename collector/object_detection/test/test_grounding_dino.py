#!/usr/bin/env python3
"""
Grounding DINO 테스트 스크립트
카메라 없이 모델 로딩 및 추론 테스트

사용법:
    python test/test_grounding_dino.py                    # 샘플 이미지로 테스트
    python test/test_grounding_dino.py --camera           # 카메라로 테스트
    python test/test_grounding_dino.py --image path.jpg   # 특정 이미지로 테스트
"""

import sys
from pathlib import Path

# 부모 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np
import torch


def test_model_loading():
    """모델 로딩 테스트"""
    print("\n" + "="*50)
    print("1. Model Loading Test")
    print("="*50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        print("  Install with: pip install transformers")
        return False

    print("\nLoading Grounding DINO model...")
    try:
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"✓ Model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def test_inference_with_sample():
    """샘플 이미지로 추론 테스트"""
    print("\n" + "="*50)
    print("2. Inference Test (Sample Image)")
    print("="*50)

    result = test_model_loading()
    if result is None:
        return False

    processor, model, device = result

    # 테스트용 더미 이미지 생성 (컬러풀한 도형들)
    print("\nCreating test image with shapes...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 빨간 원
    cv2.circle(image, (160, 240), 60, (0, 0, 255), -1)
    # 파란 사각형
    cv2.rectangle(image, (350, 180), (470, 300), (255, 0, 0), -1)
    # 초록 삼각형
    pts = np.array([[540, 350], [590, 430], (490, 430)], np.int32)
    cv2.fillPoly(image, [pts], (0, 255, 0))

    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from PIL import Image
    pil_image = Image.fromarray(image_rgb)

    # 테스트 쿼리들
    test_queries = ["red circle", "blue rectangle", "green triangle", "object"]

    print("\nRunning inference...")
    for query in test_queries:
        try:
            inputs = processor(
                images=pil_image,
                text=query,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # 새 버전 API 사용
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                target_sizes=target_sizes,
                threshold=0.2,
            )[0]

            boxes = results["boxes"]
            scores = results["scores"]

            # threshold 필터링
            mask = scores > 0.2
            boxes = boxes[mask]
            scores = scores[mask]

            print(f"  Query: '{query}' -> Found {len(boxes)} objects")
            for i, (box, score) in enumerate(zip(boxes, scores)):
                print(f"    [{i}] confidence: {score:.3f}, bbox: {box.tolist()}")

        except Exception as e:
            print(f"  Query: '{query}' -> Error: {e}")

    # 결과 이미지 저장
    output_path = Path(__file__).parent / "test_output.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"\n✓ Test image saved to: {output_path}")

    return True


def test_with_camera():
    """카메라로 실시간 테스트"""
    print("\n" + "="*50)
    print("3. Camera Test")
    print("="*50)

    result = test_model_loading()
    if result is None:
        return False

    processor, model, device = result

    try:
        from camera import RealSenseD435
        print("✓ RealSense module imported")
    except ImportError:
        print("✗ Failed to import RealSense module")
        return False

    from PIL import Image

    print("\nStarting camera...")
    print("Press 'n' to enter new query")
    print("Press 'q' to quit\n")

    current_query = "object"

    with RealSenseD435() as camera:
        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            # 탐지 수행
            image_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inputs = processor(
                images=pil_image,
                text=current_query,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                target_sizes=target_sizes,
                threshold=0.25,
            )[0]

            # 결과 시각화
            display = color.copy()
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

            # threshold 필터링
            mask = scores > 0.25
            boxes = boxes[mask].cpu().numpy()
            scores = scores[mask].cpu().numpy()
            labels = [labels[i] for i in range(len(mask)) if mask[i]]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)
                cv2.putText(display, f"{label}: {score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display, f"Query: {current_query}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Found: {len(boxes)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Grounding DINO Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                cv2.destroyAllWindows()
                current_query = input("Enter query: ").strip()
                if not current_query:
                    current_query = "object"

    cv2.destroyAllWindows()
    return True


def test_with_image(image_path: str):
    """특정 이미지로 테스트"""
    print("\n" + "="*50)
    print("4. Image File Test")
    print("="*50)

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return False

    result = test_model_loading()
    if result is None:
        return False

    processor, model, device = result

    from PIL import Image

    print(f"\nLoading image: {image_path}")
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    print("Enter queries (empty to quit):")
    while True:
        query = input("Query: ").strip()
        if not query:
            break

        inputs = processor(
            images=pil_image,
            text=query,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            threshold=0.25,
        )[0]

        display = image.copy()
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        # threshold 필터링
        mask = scores > 0.25
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()
        labels = [labels[i] for i in range(len(mask)) if mask[i]]

        print(f"Found {len(boxes)} objects:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = map(int, box)
            print(f"  [{i}] {label}: {score:.3f} at ({x1}, {y1}, {x2}, {y2})")
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{label}: {score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Result", display)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO Test")
    parser.add_argument('--camera', action='store_true', help='Test with camera')
    parser.add_argument('--image', type=str, help='Test with specific image')
    args = parser.parse_args()

    print("\n" + "#"*50)
    print("# Grounding DINO Test Script")
    print("#"*50)

    if args.camera:
        test_with_camera()
    elif args.image:
        test_with_image(args.image)
    else:
        # 기본: 샘플 이미지로 테스트
        success = test_inference_with_sample()
        if success:
            print("\n" + "="*50)
            print("✓ All tests passed!")
            print("="*50)
            print("\nNext steps:")
            print("  python test/test_grounding_dino.py --camera  # 카메라 테스트")
        else:
            print("\n✗ Tests failed")


if __name__ == "__main__":
    main()
