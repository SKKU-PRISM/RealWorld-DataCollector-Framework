"""
Judge Result Visualization Module

Judge 결과를 시각화하는 UI 및 이미지 저장 기능
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
import textwrap


def create_result_image(
    initial_image: np.ndarray,
    final_image: np.ndarray,
    instruction: str,
    prediction: str,
    reasoning: str,
    object_positions: Dict = None,
) -> np.ndarray:
    """
    Judge 결과 시각화 이미지 생성

    Args:
        initial_image: 초기 상태 이미지 (BGR)
        final_image: 최종 상태 이미지 (BGR)
        instruction: 목표 명령어
        prediction: 예측 결과 (TRUE/FALSE/UNCERTAIN)
        reasoning: 판단 근거
        object_positions: 객체 위치 딕셔너리

    Returns:
        시각화 이미지 (BGR numpy array)
    """
    # 이미지 크기 설정
    img_width = 480
    img_height = 360
    padding = 20
    text_area_height = 300

    # 전체 캔버스 크기
    canvas_width = img_width * 2 + padding * 3
    canvas_height = img_height + text_area_height + padding * 3

    # 캔버스 생성 (어두운 회색 배경)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # BGR dark gray

    # 이미지 리사이즈
    initial_resized = cv2.resize(initial_image, (img_width, img_height))
    final_resized = cv2.resize(final_image, (img_width, img_height))

    # 이미지 배치
    y_start = padding
    x_left = padding
    x_right = img_width + padding * 2

    canvas[y_start:y_start + img_height, x_left:x_left + img_width] = initial_resized
    canvas[y_start:y_start + img_height, x_right:x_right + img_width] = final_resized

    # 이미지 테두리
    cv2.rectangle(canvas, (x_left - 2, y_start - 2),
                  (x_left + img_width + 2, y_start + img_height + 2), (100, 100, 100), 2)
    cv2.rectangle(canvas, (x_right - 2, y_start - 2),
                  (x_right + img_width + 2, y_start + img_height + 2), (100, 100, 100), 2)

    # 이미지 라벨
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Initial State", (x_left, y_start - 5),
                font, 0.7, (200, 200, 200), 2)
    cv2.putText(canvas, "Final State", (x_right, y_start - 5),
                font, 0.7, (200, 200, 200), 2)

    # 텍스트 영역 시작
    text_y_start = y_start + img_height + padding + 10
    text_x = padding

    # 구분선
    cv2.line(canvas, (padding, text_y_start - 5),
             (canvas_width - padding, text_y_start - 5), (100, 100, 100), 1)

    # Judge Result 헤더
    cv2.putText(canvas, "Judge Result", (text_x, text_y_start + 20),
                font, 0.8, (255, 255, 255), 2)

    # Instruction
    text_y = text_y_start + 50
    instruction_display = instruction if len(instruction) < 80 else instruction[:77] + "..."
    cv2.putText(canvas, f"Instruction: {instruction_display}", (text_x, text_y),
                font, 0.5, (180, 180, 180), 1)

    # Prediction (색상 코딩)
    text_y += 30
    pred_color = (0, 255, 0) if prediction == "TRUE" else \
                 (0, 0, 255) if prediction == "FALSE" else \
                 (0, 255, 255)  # UNCERTAIN = yellow
    cv2.putText(canvas, "Prediction: ", (text_x, text_y),
                font, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, prediction, (text_x + 100, text_y),
                font, 0.7, pred_color, 2)

    # 구분선
    text_y += 15
    cv2.line(canvas, (text_x, text_y), (canvas_width - padding, text_y), (80, 80, 80), 1)

    # Reasoning
    text_y += 25
    cv2.putText(canvas, "Reasoning:", (text_x, text_y),
                font, 0.55, (200, 200, 200), 1)

    # Reasoning 텍스트 줄바꿈 처리
    text_y += 20
    max_chars_per_line = 90
    wrapped_lines = []
    for line in reasoning.split('\n'):
        if len(line) <= max_chars_per_line:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, max_chars_per_line))

    # 최대 표시 줄 수 제한
    max_lines = 8
    for i, line in enumerate(wrapped_lines[:max_lines]):
        cv2.putText(canvas, line.strip(), (text_x + 10, text_y + i * 18),
                    font, 0.45, (170, 170, 170), 1)

    if len(wrapped_lines) > max_lines:
        cv2.putText(canvas, "... (truncated)", (text_x + 10, text_y + max_lines * 18),
                    font, 0.4, (120, 120, 120), 1)

    return canvas


def show_judge_result(
    initial_image: np.ndarray,
    final_image: np.ndarray,
    instruction: str,
    prediction: str,
    reasoning: str,
    object_positions: Dict = None,
    wait_key: bool = True,
    timeout_ms: int = 0,
) -> np.ndarray:
    """
    Judge 결과 UI 표시

    Args:
        initial_image: 초기 상태 이미지
        final_image: 최종 상태 이미지
        instruction: 목표 명령어
        prediction: 예측 결과
        reasoning: 판단 근거
        object_positions: 객체 위치
        wait_key: True면 키 입력 대기
        timeout_ms: UI 타임아웃 (밀리초). 0이면 무한 대기

    Returns:
        시각화 이미지
    """
    result_image = create_result_image(
        initial_image=initial_image,
        final_image=final_image,
        instruction=instruction,
        prediction=prediction,
        reasoning=reasoning,
        object_positions=object_positions,
    )

    window_name = "Judge Result - Press any key to close"
    cv2.imshow(window_name, result_image)

    if wait_key:
        if timeout_ms > 0:
            print(f"\n[Judge UI] Window will close in {timeout_ms/1000:.1f}s (or press any key)...")
            cv2.waitKey(timeout_ms)
        else:
            print("\n[Judge UI] Press any key to close the window...")
            cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return result_image


def save_judge_log(
    save_dir: str,
    initial_image: np.ndarray,
    final_image: np.ndarray,
    instruction: str,
    prediction: str,
    reasoning: str,
    object_positions: Dict,
    executed_code: str,
    result_image: np.ndarray = None,
    detection_image: np.ndarray = None,
) -> Dict[str, str]:
    """
    Judge 결과 일괄 저장

    Args:
        save_dir: 저장 디렉토리
        initial_image: 초기 상태 이미지
        final_image: 최종 상태 이미지
        instruction: 목표 명령어
        prediction: 예측 결과
        reasoning: 판단 근거
        object_positions: 객체 위치
        executed_code: 실행된 코드
        result_image: 시각화 이미지 (없으면 생성)
        detection_image: 검출 결과 시각화 이미지 (옵션)

    Returns:
        저장된 파일 경로 딕셔너리
    """
    # 저장 디렉토리 생성 (forward 폴더에 직접 저장)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. 초기 상태 이미지 저장
    initial_path = save_path / "initial_state.jpg"
    cv2.imwrite(str(initial_path), initial_image)
    saved_files['initial_image'] = str(initial_path)

    # 2. 최종 상태 이미지 저장
    final_path = save_path / "final_state.jpg"
    cv2.imwrite(str(final_path), final_image)
    saved_files['final_image'] = str(final_path)

    # 3. 검출 결과 이미지 저장 (있는 경우)
    if detection_image is not None:
        detection_path = save_path / "detection_result.jpg"
        cv2.imwrite(str(detection_path), detection_image)
        saved_files['detection_image'] = str(detection_path)

    # 5. 시각화 이미지 저장
    if result_image is None:
        result_image = create_result_image(
            initial_image=initial_image,
            final_image=final_image,
            instruction=instruction,
            prediction=prediction,
            reasoning=reasoning,
            object_positions=object_positions,
        )
    result_path = save_path / "judge_result.jpg"
    cv2.imwrite(str(result_path), result_image)
    saved_files['result_image'] = str(result_path)

    # 6. 생성된 코드 저장
    code_path = save_path / "generated_code.py"
    code_path.write_text(executed_code)
    saved_files['generated_code'] = str(code_path)

    # 7. JSON 로그 저장
    # positions를 JSON 직렬화 가능하게 변환
    def serialize_value(v):
        """값을 JSON 직렬화 가능하게 변환 (numpy 스칼라 타입 포함)"""
        if v is None:
            return None
        if isinstance(v, dict):
            # Extended format: {"position": [x,y,z], "pixel_coords": (cx,cy), ...}
            return {k2: serialize_value(v2) for k2, v2 in v.items()}
        if isinstance(v, (list, tuple)):
            return [serialize_value(item) for item in v]  # 재귀 적용
        if isinstance(v, np.ndarray):
            return v.tolist()
        # numpy 스칼라 타입 처리 (np.bool_, np.int64, np.float64 등)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    positions_serializable = {}
    for k, v in object_positions.items():
        if k.startswith("_"):  # 메타데이터 키 (_image_resolution 등)
            positions_serializable[k] = serialize_value(v)
        else:
            positions_serializable[k] = serialize_value(v)

    files_info = {
        "initial_image": "initial_state.jpg",
        "final_image": "final_state.jpg",
        "result_visualization": "judge_result.jpg",
        "generated_code": "generated_code.py",
    }
    if detection_image is not None:
        files_info["detection_result"] = "detection_result.jpg"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "instruction": instruction,
        "object_positions": positions_serializable,
        "judge_result": {
            "prediction": prediction,
            "reasoning": reasoning,
        },
        "files": files_info
    }

    log_path = save_path / "judge_result.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    saved_files['judge_result'] = str(log_path)

    print(f"\n[Judge Log] Saved to: {save_path}")
    for key, path in saved_files.items():
        print(f"  - {key}: {Path(path).name}")

    return saved_files


def create_reset_result_image(
    initial_image: np.ndarray,
    final_image: np.ndarray,
    reset_mode: str,
    prediction: str,
    reasoning: str,
    current_positions: Dict = None,
    target_positions: Dict = None,
) -> np.ndarray:
    """
    Reset Judge 결과 시각화 이미지 생성

    Args:
        initial_image: Reset 전 이미지 (BGR)
        final_image: Reset 후 이미지 (BGR)
        reset_mode: "original" 또는 "random"
        prediction: 예측 결과 (TRUE/FALSE/UNCERTAIN)
        reasoning: 판단 근거
        current_positions: Reset 전 물체 위치
        target_positions: 목표 위치

    Returns:
        시각화 이미지 (BGR numpy array)
    """
    # 이미지 크기 설정
    img_width = 480
    img_height = 360
    padding = 20
    text_area_height = 300

    # 전체 캔버스 크기
    canvas_width = img_width * 2 + padding * 3
    canvas_height = img_height + text_area_height + padding * 3

    # 캔버스 생성 (어두운 회색 배경)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # BGR dark gray

    # 이미지 리사이즈
    initial_resized = cv2.resize(initial_image, (img_width, img_height))
    final_resized = cv2.resize(final_image, (img_width, img_height))

    # 이미지 배치
    y_start = padding
    x_left = padding
    x_right = img_width + padding * 2

    canvas[y_start:y_start + img_height, x_left:x_left + img_width] = initial_resized
    canvas[y_start:y_start + img_height, x_right:x_right + img_width] = final_resized

    # 이미지 테두리
    cv2.rectangle(canvas, (x_left - 2, y_start - 2),
                  (x_left + img_width + 2, y_start + img_height + 2), (100, 100, 100), 2)
    cv2.rectangle(canvas, (x_right - 2, y_start - 2),
                  (x_right + img_width + 2, y_start + img_height + 2), (100, 100, 100), 2)

    # 이미지 라벨
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Before Reset", (x_left, y_start - 5),
                font, 0.7, (200, 200, 200), 2)
    cv2.putText(canvas, "After Reset", (x_right, y_start - 5),
                font, 0.7, (200, 200, 200), 2)

    # 텍스트 영역 시작
    text_y_start = y_start + img_height + padding + 10
    text_x = padding

    # 구분선
    cv2.line(canvas, (padding, text_y_start - 5),
             (canvas_width - padding, text_y_start - 5), (100, 100, 100), 1)

    # Reset Judge Result 헤더
    mode_str = "RESTORE TO ORIGINAL" if reset_mode == "original" else "RANDOM SHUFFLE"
    cv2.putText(canvas, f"Reset Judge Result ({mode_str})", (text_x, text_y_start + 20),
                font, 0.7, (255, 255, 255), 2)

    # Prediction (색상 코딩)
    text_y = text_y_start + 55
    pred_color = (0, 255, 0) if prediction == "TRUE" else \
                 (0, 0, 255) if prediction == "FALSE" else \
                 (0, 255, 255)  # UNCERTAIN = yellow
    cv2.putText(canvas, "Prediction: ", (text_x, text_y),
                font, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, prediction, (text_x + 100, text_y),
                font, 0.7, pred_color, 2)

    # 구분선
    text_y += 15
    cv2.line(canvas, (text_x, text_y), (canvas_width - padding, text_y), (80, 80, 80), 1)

    # Reasoning
    text_y += 25
    cv2.putText(canvas, "Reasoning:", (text_x, text_y),
                font, 0.55, (200, 200, 200), 1)

    # Reasoning 텍스트 줄바꿈 처리
    text_y += 20
    max_chars_per_line = 90
    wrapped_lines = []
    for line in reasoning.split('\n'):
        if len(line) <= max_chars_per_line:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, max_chars_per_line))

    # 최대 표시 줄 수 제한
    max_lines = 8
    for i, line in enumerate(wrapped_lines[:max_lines]):
        cv2.putText(canvas, line.strip(), (text_x + 10, text_y + i * 18),
                    font, 0.45, (170, 170, 170), 1)

    if len(wrapped_lines) > max_lines:
        cv2.putText(canvas, "... (truncated)", (text_x + 10, text_y + max_lines * 18),
                    font, 0.4, (120, 120, 120), 1)

    return canvas


def show_reset_judge_result(
    initial_image: np.ndarray,
    final_image: np.ndarray,
    reset_mode: str,
    prediction: str,
    reasoning: str,
    current_positions: Dict = None,
    target_positions: Dict = None,
    wait_key: bool = True,
    timeout_ms: int = 0,
) -> np.ndarray:
    """
    Reset Judge 결과 UI 표시

    Args:
        initial_image: Reset 전 이미지
        final_image: Reset 후 이미지
        reset_mode: "original" 또는 "random"
        prediction: 예측 결과
        reasoning: 판단 근거
        current_positions: Reset 전 물체 위치
        target_positions: 목표 위치
        wait_key: True면 키 입력 대기
        timeout_ms: UI 타임아웃 (밀리초). 0이면 무한 대기

    Returns:
        시각화 이미지
    """
    result_image = create_reset_result_image(
        initial_image=initial_image,
        final_image=final_image,
        reset_mode=reset_mode,
        prediction=prediction,
        reasoning=reasoning,
        current_positions=current_positions,
        target_positions=target_positions,
    )

    mode_str = "Original" if reset_mode == "original" else "Random"
    window_name = f"Reset Judge ({mode_str}) - Press any key to close"
    cv2.imshow(window_name, result_image)

    if wait_key:
        if timeout_ms > 0:
            print(f"\n[Reset Judge UI] Window will close in {timeout_ms/1000:.1f}s (or press any key)...")
            cv2.waitKey(timeout_ms)
        else:
            print("\n[Reset Judge UI] Press any key to close the window...")
            cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return result_image


# 테스트
if __name__ == "__main__":
    print("Judge Visualization Test")
    print("=" * 60)

    # 테스트용 더미 이미지 생성
    dummy_initial = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_initial[:] = (60, 60, 60)
    cv2.putText(dummy_initial, "Initial State", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.circle(dummy_initial, (320, 300), 30, (0, 255, 0), -1)  # green block
    cv2.rectangle(dummy_initial, (400, 350), (500, 400), (255, 100, 0), -1)  # blue dish

    dummy_final = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_final[:] = (60, 60, 60)
    cv2.putText(dummy_final, "Final State", (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.circle(dummy_final, (450, 375), 30, (0, 255, 0), -1)  # green block moved
    cv2.rectangle(dummy_final, (400, 350), (500, 400), (255, 100, 0), -1)  # blue dish

    # 테스트 데이터
    test_instruction = "pick up the green block and place it on the blue dish"
    test_prediction = "TRUE"
    test_reasoning = """- Goal: pick up the green block and place it on the blue dish.
- Visual comparison:
  - Initial image: green block sits near the center of the workspace; blue dish is at the bottom-right, empty.
  - Final image: green block has been moved and is now positioned on top of the blue dish.
- The robot successfully picked up the green block and placed it on the blue dish as instructed.
- Task completed successfully."""

    test_positions = {
        "green block": [0.1500, 0.0500, 0.0200],
        "blue dish": [0.2000, -0.0300, 0.0100],
    }

    # UI 표시
    result_img = show_judge_result(
        initial_image=dummy_initial,
        final_image=dummy_final,
        instruction=test_instruction,
        prediction=test_prediction,
        reasoning=test_reasoning,
        object_positions=test_positions,
    )

    # 저장 테스트
    save_judge_log(
        save_dir="/tmp/judge_test",
        initial_image=dummy_initial,
        final_image=dummy_final,
        instruction=test_instruction,
        prediction=test_prediction,
        reasoning=test_reasoning,
        object_positions=test_positions,
        executed_code="# Test code\nprint('Hello')",
        result_image=result_img,
    )
