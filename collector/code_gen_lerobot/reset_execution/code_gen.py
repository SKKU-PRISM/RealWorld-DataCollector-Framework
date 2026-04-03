"""
Reset Execution Code Generator

Generates executable Python code to reverse/undo forward task executions.
Uses object detection to find current positions and context for original positions.

Reset Modes:
- "original": Restore objects to their initial positions (default)
- "random": Shuffle objects to random positions within workspace

Multi-turn mode:
- Uses VLM crop-then-point pipeline (same as forward)
- Grounding DINO 의존 제거, VLM 기반 검출
"""

import re
from typing import Dict, List, Optional, Tuple

from ..llm import llm_response
from .prompt import (
    lerobot_reset_code_gen_prompt,
    turn1_reset_bbox_detection_prompt,
)
from .turn0_prompt import turn0_reset_scene_understanding_prompt
from .workspace import (
    classify_objects,
    generate_random_positions,
    ResetWorkspace,
    draw_workspace_on_image,
)


def lerobot_reset_code_gen(
    original_instruction: str,
    original_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
    forward_code: str = None,
    object_queries: List[str] = None,
    detection_timeout: float = 10.0,
    visualize_detection: bool = False,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
    external_camera=None,
    # Episode tracking for logging
    current_episode: int = 1,
    total_episodes: int = 1,
    # Pre-detected positions (for multi-robot shared detection)
    current_positions: Dict[str, Dict] = None,
    # Reset quadrant constraint
    resetspace: str = None,
) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    LeRobot SO-101용 리셋 코드 생성

    Detection을 사용하여 현재 물체 위치를 파악하고,
    Forward execution context를 참조하여 초기 환경으로 되돌리거나
    랜덤 위치로 shuffle하는 코드를 생성합니다.

    Args:
        original_instruction: 원래 태스크 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        original_positions: 객체들의 원래 위치 (forward 실행 전 위치, context에서 로드)
        forward_spec: Forward execution에서 생성된 spec (step-by-step plan)
        forward_code: Forward execution에서 생성된 Python 코드
        object_queries: Detection할 객체 리스트 (None이면 original_positions의 키 사용)
        detection_timeout: Detection 타임아웃 (초)
        visualize_detection: Detection 결과 시각화 여부
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호 (2 또는 3)
        reset_mode: "original" (초기 위치로 복귀) 또는 "random" (랜덤 위치로 shuffle)
        random_seed: 랜덤 위치 생성용 seed (재현성 보장)
        external_camera: 외부에서 전달받은 카메라 인스턴스 (공유 모드)
                        - Recording 카메라와 공유하여 리소스 충돌 방지
        current_positions: 미리 감지된 현재 위치 (multi-robot 공유 감지용)
                          - 제공되면 내부 detection을 건너뛰고 이 위치 사용
                          - extended format: {name: {"position": [...], ...}}

    Returns:
        Tuple[str, Dict, Dict, Dict]: (리셋 코드, 원래 위치, 현재 위치, 타겟 위치)
        - 타겟 위치: reset_mode에 따라 original_positions 또는 random positions

    Example:
        # Original mode (restore)
        reset_code, orig, curr, target = lerobot_reset_code_gen(
            original_instruction="pick red cup and place on blue box",
            original_positions={"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]},
            reset_mode="original",
        )

        # Random mode (shuffle)
        reset_code, orig, curr, target = lerobot_reset_code_gen(
            original_instruction="pick red cup and place on blue box",
            original_positions={"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]},
            reset_mode="random",
            random_seed=42,
        )
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET_COLOR = "\033[0m"
    line_width = 60

    # Episode prefix for logging
    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    def _log(msg: str, step: str = None) -> str:
        prefix = f"[Reset][{ep_str}]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    mode_str = "RANDOM RESET" if reset_mode == "random" else "RESET TO ORIGINAL"

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(MAGENTA + f"LeRobot {mode_str}".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)

    # 1) Context 검증
    print(f"{YELLOW}" + _log("Validating execution context", step="1/5") + f"{RESET_COLOR}")

    if not original_instruction:
        raise ValueError("original_instruction is required for reset code generation")

    if not original_positions or len(original_positions) == 0:
        raise ValueError("original_positions is required for reset code generation")

    print(f"  Original task: {original_instruction[:50]}...")
    print(f"  Reset mode: {reset_mode}")
    print(f"  Objects: {list(original_positions.keys())}")

    # 원래 위치 출력 (extended format 지원)
    print(f"\n  Initial positions (forward 시작 시):")
    for name, info in original_positions.items():
        if info is None:
            continue
        elif isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # 2) Detection으로 현재 위치 획득 (확장 정보 포함)
    # current_positions가 제공되면 detection 건너뛰기 (multi-robot 공유 감지 모드)
    if current_positions is not None:
        print(f"\n{YELLOW}" + _log("Using pre-detected positions (shared detection)", step="2/5") + f"{RESET_COLOR}")
        extended_detections = current_positions
    else:
        print(f"\n{YELLOW}" + _log("Detecting current object positions", step="2/5") + f"{RESET_COLOR}")

        # object_queries가 없으면 original_positions의 키 사용
        if object_queries is None:
            object_queries = list(original_positions.keys())

        print(f"  Detecting: {object_queries}")

        try:
            from run_detect import run_realtime_detection
            # 확장 정보 요청 (bbox_size_m, grippable 포함)
            # external_camera가 있으면 공유하여 리소스 충돌 방지
            extended_detections = run_realtime_detection(
                queries=object_queries,
                timeout=detection_timeout,
                unit="m",
                visualize=visualize_detection,
                return_extended=True,
                robot_id=robot_id,
                external_camera=external_camera,
            )
        except ImportError as e:
            print(f"{RED}[Error] Could not import detection module: {e}{RESET_COLOR}")
            raise ValueError("Detection module not available")
        except Exception as e:
            print(f"{RED}[Error] Detection failed: {e}{RESET_COLOR}")
            raise ValueError(f"Detection failed: {e}")

    # Detection 결과 출력
    found_count = sum(1 for v in extended_detections.values() if v is not None)
    print(f"  Found: {found_count}/{len(extended_detections)} objects")

    print(f"\n  Current positions (detected):")
    for name, info in extended_detections.items():
        if info:
            pos = info["position"]
            bbox = info.get("bbox_px")
            grippable = info.get("grippable", True)
            bbox_str = f"[{bbox[0]}x{bbox[1]}px]" if bbox else "[?x?]"
            grip_str = "grippable" if grippable else "OBSTACLE"
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] {bbox_str} ({grip_str})")
        else:
            print(f"    x {name}: NOT FOUND")

    # Workspace 범위 검증
    print(f"\n{YELLOW}[Validation] Checking workspace bounds...{RESET_COLOR}")
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from lerobot_cap.workspace import BaseWorkspace
    workspace = BaseWorkspace()
    print(f"  Workspace: reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

    for obj_name, obj_info in extended_detections.items():
        if obj_info is None:
            continue
        pos = obj_info.get("position")
        if pos is None:
            continue
        position_m = np.array([pos[0], pos[1], pos[2]])
        if not workspace.is_reachable(position_m):
            warning_msg = (
                f"[WARNING] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m "
                f"is OUTSIDE workspace!"
            )
            print(f"{RED}{warning_msg}{RESET_COLOR}")
            print(f"{RED}  Workspace: reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m{RESET_COLOR}")
            assert False, f"Object '{obj_name}' is outside workspace bounds"
        else:
            print(f"  {LIGHT_GREEN}✓ '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m - OK{RESET_COLOR}")

    # 3) 객체 분류 (grippable vs obstacle)
    print(f"\n{YELLOW}" + _log("Classifying objects", step="3/5") + f"{RESET_COLOR}")

    grippable_objects, obstacle_objects = classify_objects(extended_detections)

    print(f"  Grippable (will move): {list(grippable_objects.keys())}")
    print(f"  Obstacles (fixed): {list(obstacle_objects.keys())}")

    # grippable 객체 중 검출 실패 확인
    grippable_not_found = [
        name for name in original_positions.keys()
        if name not in grippable_objects and name not in obstacle_objects
    ]
    if grippable_not_found:
        print(f"\n{RED}[Error] Required objects not detected: {grippable_not_found}{RESET_COLOR}")
        raise ValueError(f"Cannot generate reset code: objects not detected - {grippable_not_found}")

    # 4) 타겟 위치 결정
    print(f"\n{YELLOW}" + _log(f"Determining target positions ({reset_mode} mode)", step="4/5") + f"{RESET_COLOR}")

    if reset_mode == "random":
        # 랜덤 위치 생성
        target_positions = generate_random_positions(
            grippable_objects=grippable_objects,
            obstacle_objects=obstacle_objects,
            initial_positions=original_positions,
            seed=random_seed,
            resetspace=resetspace,
        )
        print(f"  Random target positions generated:")
    else:
        # original 모드: grippable 객체만 초기 위치로
        target_positions = {
            name: original_positions[name]
            for name in grippable_objects.keys()
            if name in original_positions and original_positions[name] is not None
        }
        print(f"  Restoring to initial positions:")

    for name, info in target_positions.items():
        if isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # 현재 위치 추출 (단순 형식으로 변환)
    current_positions = {
        name: info["position"]
        for name, info in grippable_objects.items()
        if name in target_positions
    }

    # 5) 프롬프트 생성 및 LLM 호출
    print(f"\n{YELLOW}" + _log(f"Generating reset code via LLM ({llm_model})", step="5/5") + f"{RESET_COLOR}")

    prompt = lerobot_reset_code_gen_prompt(
        original_instruction=original_instruction,
        target_positions=target_positions,
        current_positions=current_positions,
        forward_spec=forward_spec,
        forward_code=forward_code,
        robot_id=robot_id,
        is_random_reset=(reset_mode == "random"),
    )

    response = llm_response(llm_model, prompt, check_time=True)

    # 코드 추출
    code = extract_code_from_response(response)

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(LIGHT_GREEN + f"{mode_str} code generation completed.".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)

    return code, original_positions, current_positions, target_positions


def lerobot_reset_code_gen_from_context(
    context_path: str,
    detection_timeout: float = 10.0,
    visualize_detection: bool = False,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    저장된 execution context 파일에서 리셋 코드 생성

    Args:
        context_path: execution_context.json 파일 경로
        detection_timeout: Detection 타임아웃 (초)
        visualize_detection: Detection 결과 시각화 여부
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호
        reset_mode: "original" (초기 위치로 복귀) 또는 "random" (랜덤 위치로 shuffle)
        random_seed: 랜덤 위치 생성용 seed

    Returns:
        Tuple[str, Dict, Dict, Dict]: (리셋 코드, 원래 위치, 현재 위치, 타겟 위치)

    Example:
        reset_code, orig_pos, curr_pos, target_pos = lerobot_reset_code_gen_from_context(
            context_path="outputs/execution_context.json",
            reset_mode="random",
        )
    """
    import json

    # Context 파일 로드
    with open(context_path, "r", encoding="utf-8") as f:
        context = json.load(f)

    # 필수 필드 확인
    required_fields = ["instruction", "object_positions"]
    for field in required_fields:
        if field not in context:
            raise ValueError(f"Context file missing required field: {field}")

    return lerobot_reset_code_gen(
        original_instruction=context["instruction"],
        original_positions=context["object_positions"],
        forward_spec=context.get("generated_spec"),
        forward_code=context.get("generated_code"),
        detection_timeout=detection_timeout,
        visualize_detection=visualize_detection,
        llm_model=llm_model,
        robot_id=robot_id,
        reset_mode=reset_mode,
        random_seed=random_seed,
    )


def extract_code_from_response(response: str) -> str:
    """
    LLM 응답에서 Python 코드 블록을 추출

    Args:
        response: LLM 응답 문자열

    Returns:
        추출된 Python 코드
    """
    # 코드 블록 패턴 (```python ... ``` 또는 ``` ... ```)
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # 가장 긴 코드 블록 반환 (보통 메인 코드)
        return max(matches, key=len).strip()

    # 코드 블록이 없으면 전체 응답 반환 (plain text 형식일 수 있음)
    # "from skills" 또는 "def execute"로 시작하는 부분 찾기
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    # 그래도 없으면 전체 반환
    return response.strip()


# ============================================================
# Multi-Turn Reset Code Generation
# ============================================================

def _compute_bbox_size_m(
    valid_objects: list,
    img_w: int,
    img_h: int,
    coord_transformer,
) -> Dict[str, Tuple[float, float]]:
    """
    VLM Turn 1의 bbox(0-1000 스케일) → 미터 단위 크기.

    각 bbox의 좌상단/우하단을 pixel → world_2d 변환 후
    width_m = |world_x2 - world_x1|, height_m = |world_y2 - world_y1|

    Args:
        valid_objects: Turn 1 bbox 결과 리스트 [{"box_2d": [ymin,xmin,ymax,xmax], "label": str}, ...]
        img_w: 이미지 폭 (pixels)
        img_h: 이미지 높이 (pixels)
        coord_transformer: CoordinateTransformer 인스턴스

    Returns:
        {label: (width_m, height_m)}
    """
    sizes = {}
    for obj in valid_objects:
        label = obj.get("label", "")
        box = obj.get("box_2d", [])
        if len(box) != 4 or not label:
            continue

        ymin, xmin, ymax, xmax = box

        # 0-1000 → pixel
        px1 = int(xmin * img_w / 1000)
        py1 = int(ymin * img_h / 1000)
        px2 = int(xmax * img_w / 1000)
        py2 = int(ymax * img_h / 1000)

        try:
            # Pix2Robot 직접 변환 (coord_transformer가 Pix2RobotCalibrator인 경우)
            if hasattr(coord_transformer, 'pixel_to_robot'):
                r1 = coord_transformer.pixel_to_robot(px1, py1)
                r2 = coord_transformer.pixel_to_robot(px2, py2)
                width_m = abs(r2[0] - r1[0])
                height_m = abs(r2[1] - r1[1])
            else:
                # Fallback: 기존 pixel → world (cm) → meters
                wx1, wy1, _ = coord_transformer.pixel_to_world_2d(px1, py1)
                wx2, wy2, _ = coord_transformer.pixel_to_world_2d(px2, py2)
                width_m = abs(wx2 - wx1) / 100.0
                height_m = abs(wy2 - wy1) / 100.0
            sizes[label] = (width_m, height_m)
        except Exception as e:
            print(f"  [BBoxSize] Failed for '{label}': {e}")
            sizes[label] = (0.03, 0.03)  # fallback 3cm

    return sizes


def _match_labels(
    detected_labels: list,
    original_labels: list,
) -> Dict[str, str]:
    """
    Reset VLM 검출 라벨과 Forward original_positions 라벨 간 매칭.

    VLM은 매번 다른 라벨을 생성할 수 있으므로 (예: "chocolate pie" vs "chocolate pie 1")
    substring/fuzzy 매칭으로 대응.

    Args:
        detected_labels: Reset VLM에서 검출된 라벨 리스트
        original_labels: original_positions의 키 리스트

    Returns:
        {detected_label: original_label} 매핑
    """
    def _normalize(label: str) -> str:
        """라벨 정규화: 언더스코어→공백, 소문자, strip"""
        return label.lower().strip().replace("_", " ")

    mapping = {}
    used_originals = set()

    # Pass 0: 정규화 후 정확한 매칭 (언더스코어 vs 공백 차이 해결)
    for dl in detected_labels:
        for ol in original_labels:
            if ol in used_originals:
                continue
            if _normalize(dl) == _normalize(ol):
                mapping[dl] = ol
                used_originals.add(ol)
                break

    # Pass 1: 정확한 매칭 (원본 문자열)
    for dl in detected_labels:
        if dl in mapping:
            continue
        if dl in original_labels and dl not in used_originals:
            mapping[dl] = dl
            used_originals.add(dl)

    # Pass 2: 정규화 후 substring 매칭
    remaining_detected = [dl for dl in detected_labels if dl not in mapping]
    remaining_original = [ol for ol in original_labels if ol not in used_originals]

    for dl in remaining_detected:
        best_match = None
        best_len = 0
        dl_norm = _normalize(dl)
        for ol in remaining_original:
            ol_norm = _normalize(ol)
            # original 라벨이 detected 라벨에 포함
            if ol_norm in dl_norm and len(ol_norm) > best_len:
                best_match = ol
                best_len = len(ol_norm)
            # detected 라벨이 original 라벨에 포함
            elif dl_norm in ol_norm and len(dl_norm) > best_len:
                best_match = ol
                best_len = len(dl_norm)
        if best_match:
            mapping[dl] = best_match
            remaining_original.remove(best_match)

    # Pass 3: 단어 기반 매칭 (정규화 후)
    remaining_detected = [dl for dl in detected_labels if dl not in mapping]
    for dl in remaining_detected:
        dl_words = set(_normalize(dl).split())
        best_match = None
        best_overlap = 0
        for ol in remaining_original:
            ol_words = set(_normalize(ol).split())
            overlap = len(dl_words & ol_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ol
        if best_match and best_overlap > 0:
            mapping[dl] = best_match
            remaining_original.remove(best_match)

    return mapping


def lerobot_reset_code_gen_multi_turn(
    original_instruction: str,
    original_positions: Dict,
    current_state_image_path: str,
    initial_state_image_path: str,
    llm_model: str = "gemini-2.0-flash",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
    camera=None,
    coord_transformer=None,
    workspace=None,
    current_episode: int = 1,
    total_episodes: int = 1,
    codegen_model: str = None,
    skip_codegen: bool = False,
    canonical_labels: List[str] = None,
    robot_ids: List[int] = None,
    original_positions_dual: Dict = None,
    resetspace: str = None,
) -> Tuple[str, Dict, Dict, Dict, Dict, Dict]:
    """
    VLM Multi-Turn Reset 코드 생성 파이프라인.

    Forward와 동일한 crop-then-point 방식으로 현재 물체 위치를 검출하고,
    reset 코드를 생성합니다. Grounding DINO 의존 없이 VLM만으로 동작.

    Turn 0: Scene Understanding (current + initial 이미지)
    Turn 1: BBox Detection (forward turn1_prompt 재사용)
    Turn 2~N: Crop → Critical Point (forward turn2_prompt 재사용)
    CodeGen: Reset 코드 생성

    Args:
        original_instruction: 원래 forward 태스크 명령
        original_positions: Forward 전 초기 위치 (extended format)
        current_state_image_path: Forward 후 현재 이미지 경로
        initial_state_image_path: Forward 전 초기 이미지 경로
        llm_model: LLM 모델
        robot_id: 로봇 번호
        reset_mode: "original" | "random"
        random_seed: 랜덤 위치 생성용 seed
        camera: RealSense camera (depth용)
        coord_transformer: Pix2RobotCalibrator 또는 호환 좌표 변환기
        workspace: ResetWorkspace 인스턴스
        current_episode: 현재 에피소드 번호
        total_episodes: 총 에피소드 수

    Returns:
        (generated_code, current_positions, target_positions,
         grippable_objects, obstacle_objects)
    """
    import cv2
    import tempfile
    import json
    import numpy as np
    from pathlib import Path

    from ..llm_utils.gemini import gemini_chat_start, gemini_chat_send
    from ..forward_execution.turn2_prompt import turn2_crop_pointing_prompt
    from ..code_gen_with_skill import _points_to_positions, _parse_json_from_response
    from ..forward_execution.system_prompt import PERCEPTION_SYSTEM_PROMPT

    CROP_PADDING = 25  # bbox 패딩 (0-1000 스케일)

    # ANSI colors
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET_COLOR = "\033[0m"
    line_width = 60

    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    mode_str = "RANDOM RESET" if reset_mode == "random" else "RESET TO ORIGINAL"

    def _log(msg, step=None):
        prefix = f"[Reset][{ep_str}][MultiTurn]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(MAGENTA + f"LeRobot {mode_str} (Multi-Turn VLM)".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)
    print(f"  Model: {llm_model}")

    # 토큰/시간 통계 누적
    _usage_stats = {"total_inference_time": 0.0, "total_in": 0, "total_out": 0, "total_think": 0, "total_tokens": 0}
    _turn_costs = []
    def _accumulate_usage(turn_name=""):
        u = gemini_chat_send._last_usage
        if u:
            _usage_stats["total_inference_time"] += u.get("inference_time", 0)
            _usage_stats["total_in"] += u.get("in", 0)
            _usage_stats["total_out"] += u.get("out", 0)
            _usage_stats["total_think"] += u.get("think", 0)
            _usage_stats["total_tokens"] += u.get("total", 0)
            _turn_costs.append({
                "turn": turn_name,
                "inference_time_s": round(u.get("inference_time", 0), 2),
                "input_tokens": u.get("in", 0),
                "output_tokens": u.get("out", 0),
                "thinking_tokens": u.get("think", 0),
                "total_tokens": u.get("total", 0),
            })
    print(f"  Current image: {current_state_image_path}")
    print(f"  Initial image: {initial_state_image_path}")

    # ── Step 1: Workspace 준비 ──
    print(f"\n{YELLOW}" + _log("Preparing workspace bounds", step="Setup") + f"{RESET_COLOR}")

    if workspace is None:
        # KinematicsEngine 로드하여 IK 검증 가능한 workspace 생성
        _kin_engine = None
        try:
            from lerobot_cap.kinematics.engine import KinematicsEngine
            urdf_path = Path(__file__).parent.parent.parent / "assets" / "urdf" / f"so101_robot{robot_id}.urdf"
            if urdf_path.exists():
                _kin_engine = KinematicsEngine(str(urdf_path))
        except Exception as e:
            print(f"  [Workspace] KinematicsEngine not available: {e}")
        workspace = ResetWorkspace(kinematics_engine=_kin_engine)

    print(f"  Workspace: {workspace}")

    # Pix2Robot 캘리브레이션 로드 (우선) → fallback: CoordinateTransformer
    if coord_transformer is None:
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            pix2robot_path = Path(__file__).parent.parent.parent / "robot_configs" / "pix2robot_matrices" / f"robot{robot_id}_pix2robot_data.npz"
            if pix2robot_path.exists():
                _p2r = Pix2RobotCalibrator(robot_id=robot_id)
                if _p2r.load(str(pix2robot_path)):
                    coord_transformer = _p2r
                    print(f"  Pix2Robot calibration loaded ({len(_p2r.pixel_points)} points)")
        except Exception:
            pass

    # Workspace 시각화 이미지 생성 (pix2robot 기반)
    reset_dir = Path(current_state_image_path).parent
    annotated_image_path = None
    current_img = cv2.imread(current_state_image_path)
    if current_img is not None:
        # resetspace가 dict이면 (multi-arm) 해당 robot_id의 값 추출
        _rs = resetspace.get(robot_id, "all") if isinstance(resetspace, dict) else resetspace
        annotated = draw_workspace_on_image(current_img, robot_id=robot_id, resetspace=_rs)
        annotated_image_path = str(reset_dir / "workspace_annotated.jpg")
        cv2.imwrite(annotated_image_path, annotated)
        print(f"  Workspace annotated image: {annotated_image_path}")

    # ── Step 2: Gemini chat 시작 ──
    print(f"\n{YELLOW}" + _log("Starting Gemini chat", step="Chat") + f"{RESET_COLOR}")
    if robot_ids and len(robot_ids) >= 2:
        from ..multi_arm.forward_execution.system_prompt import MULTI_ARM_PERCEPTION_SYSTEM_PROMPT
        perception_sp = MULTI_ARM_PERCEPTION_SYSTEM_PROMPT
    else:
        perception_sp = PERCEPTION_SYSTEM_PROMPT
    chat, gen_config = gemini_chat_start(llm_model, system_prompt=perception_sp)

    # ── Turn 0: Scene Understanding ──
    print(f"\n{YELLOW}" + _log("Turn 0 — Scene Understanding", step="Turn0") + f"{RESET_COLOR}")
    original_labels = list(original_positions.keys()) if original_positions else []

    # Multi-arm: use multi-arm reset Turn 0 prompt
    if robot_ids and len(robot_ids) >= 2:
        from ..multi_arm.reset_execution.turn0_prompt import turn0_reset_scene_understanding_prompt as multi_arm_turn0
        turn0_text = multi_arm_turn0(
            original_instruction=original_instruction,
            reset_mode=reset_mode,
            original_object_labels=original_labels,
        )
    else:
        turn0_text = turn0_reset_scene_understanding_prompt(
            original_instruction=original_instruction,
            reset_mode=reset_mode,
            original_object_labels=original_labels,
        )
    if original_labels:
        print(f"  Original labels provided to VLM: {original_labels}")
    # Image 1 = annotated current state (or original if no transformer)
    turn0_image = annotated_image_path or current_state_image_path
    turn0_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn0_text,
            "image_path": turn0_image,
            "image_paths": [initial_state_image_path],
        },
        turn_label="Turn 0 (Reset)")
    _accumulate_usage("Turn 0")
    print(f"  {turn0_resp[:300]}{'...' if len(turn0_resp) > 300 else ''}")

    # ── Turn 1: BBox Detection (Reset 전용 — Forward 라벨 강제 사용) ──
    print(f"\n{YELLOW}" + _log("Turn 1 — BBox Detection", step="Turn1") + f"{RESET_COLOR}")
    # canonical_labels가 있으면 (코드 재사용 모드) 그것을 우선 사용
    enforced_labels = canonical_labels if canonical_labels else (original_labels if original_labels else None)
    turn1_text = turn1_reset_bbox_detection_prompt(
        original_object_labels=enforced_labels,
    )
    if enforced_labels:
        print(f"  Enforcing labels in Turn 1: {enforced_labels}")
    turn1_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn1_text,
            "image_path": current_state_image_path,  # 원본 이미지 (시각화 없음)
        },
        turn_label="Turn 1 (Reset)")
    _accumulate_usage("Turn 1")
    print(f"  {turn1_resp[:300]}{'...' if len(turn1_resp) > 300 else ''}")

    # Parse bboxes
    turn1_data = _parse_json_from_response(turn1_resp)
    if isinstance(turn1_data, list):
        obj_list = turn1_data
    elif isinstance(turn1_data, dict):
        obj_list = turn1_data.get("objects", turn1_data.get("detected_objects", []))
    else:
        obj_list = []

    valid_objects = []
    for obj in obj_list:
        box = obj.get("box_2d") or obj.get("bbox") or []
        if len(box) == 4 and obj.get("label"):
            obj["box_2d"] = box
            valid_objects.append(obj)
            print(f"    [{obj['label']}] bbox={box}")

    assert valid_objects, "No valid bboxes detected from Turn 1"

    # 이미지 로드
    full_img = cv2.imread(current_state_image_path)
    assert full_img is not None, f"Cannot read image: {current_state_image_path}"
    img_h, img_w = full_img.shape[:2]

    # ── Turn 2+: Crop-then-Point ──
    all_points = []
    crop_responses = []
    crop_dir = str(reset_dir / "crops")
    Path(crop_dir).mkdir(parents=True, exist_ok=True)

    for i, obj in enumerate(valid_objects):
        label = obj["label"]
        ymin, xmin, ymax, xmax = obj["box_2d"]

        print(f"\n{YELLOW}" + _log(f"Crop — {label}", step=f"Crop{i}") + f"{RESET_COLOR}")

        # Padding + clamp (0-1000 스케일)
        ymin_p = max(0, ymin - CROP_PADDING)
        xmin_p = max(0, xmin - CROP_PADDING)
        ymax_p = min(1000, ymax + CROP_PADDING)
        xmax_p = min(1000, xmax + CROP_PADDING)

        # 0-1000 → pixel
        crop_x1 = int(xmin_p * img_w / 1000)
        crop_y1 = int(ymin_p * img_h / 1000)
        crop_x2 = int(xmax_p * img_w / 1000)
        crop_y2 = int(ymax_p * img_h / 1000)

        crop_img = full_img[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w = crop_img.shape[:2]
        print(f"    bbox=[{ymin},{xmin},{ymax},{xmax}] → crop ({crop_w}x{crop_h})")

        safe_label = label.replace(" ", "_").replace("/", "_")
        crop_path = f"{crop_dir}/crop_{safe_label}.jpg"
        cv2.imwrite(crop_path, crop_img)

        # Send crop + pointing prompt
        resp = gemini_chat_send(chat, gen_config,
            {
                "text": turn2_crop_pointing_prompt(label, has_side_view=False),
                "image_path": crop_path,
            },
            turn_label=f"Crop: {label}")
        _accumulate_usage(f"Crop: {label}")
        crop_responses.append({"label": label, "response": resp})
        print(f"    {resp[:200]}{'...' if len(resp) > 200 else ''}")

        # Parse critical_points
        parsed = _parse_json_from_response(resp)
        if not parsed:
            print(f"    [Warning] Failed to parse points for '{label}'")
            continue

        if "critical_points" in parsed:
            oh_points = parsed["critical_points"]
        elif "overhead_critical_points" in parsed:
            oh_points = parsed["overhead_critical_points"]
        else:
            print(f"    [Warning] No recognized point keys for '{label}'")
            continue

        for pt in oh_points:
            point_2d = pt.get("point_2d", [])
            if len(point_2d) != 2:
                continue
            norm_y, norm_x = point_2d
            crop_px = int(norm_x * crop_w / 1000)
            crop_py = int(norm_y * crop_h / 1000)
            px = crop_x1 + crop_px
            py = crop_y1 + crop_py

            entry = {
                "object_label": label,
                "label": pt.get("label", ""),
                "role": pt.get("role", "interaction"),
                "reasoning": pt.get("reasoning", ""),
                "point_2d": point_2d,
                "px": px, "py": py,
                "crop_px": crop_px, "crop_py": crop_py,
            }
            print(f"    [{pt.get('role','?')}] ({norm_y},{norm_x}) → full({px},{py})")
            all_points.append(entry)

    print(f"\n  Total: {len(all_points)} points across {len(valid_objects)} objects")

    # ── Step 6: 좌표 변환 & 분류 ──
    print(f"\n{YELLOW}" + _log("Building positions (pixel → world)", step="Positions") + f"{RESET_COLOR}")
    current_positions = _points_to_positions(all_points, robot_id=robot_id, camera=camera)

    # BBox 픽셀 크기 추출 + grippable 판정
    bbox_px_map = {}
    for obj in valid_objects:
        label = obj.get("label", "")
        box = obj.get("box_2d", [])
        if len(box) == 4 and label:
            ymin, xmin, ymax, xmax = box
            w_px = int((xmax - xmin) * img_w / 1000)
            h_px = int((ymax - ymin) * img_h / 1000)
            bbox_px_map[label] = (max(w_px, 10), max(h_px, 10))

    from .workspace import is_grippable
    for label, info in current_positions.items():
        info["bbox_px"] = bbox_px_map.get(label, (30, 30))
        info["grippable"] = is_grippable(info["bbox_px"])

    # Classify objects
    grippable_objects, obstacle_objects = classify_objects(current_positions)
    print(f"  Grippable: {list(grippable_objects.keys())}")
    print(f"  Obstacles: {list(obstacle_objects.keys())}")

    # ── Step 7: 라벨 매칭 + Target 위치 결정 ──
    print(f"\n{YELLOW}" + _log("Matching labels (reset VLM → original)", step="LabelMatch") + f"{RESET_COLOR}")

    detected_labels = list(grippable_objects.keys()) + list(obstacle_objects.keys())
    original_labels = list(original_positions.keys())
    label_map = _match_labels(detected_labels, original_labels)

    for dl, ol in label_map.items():
        match_type = "exact" if dl == ol else "fuzzy"
        print(f"    {dl} → {ol} ({match_type})")

    unmatched = [dl for dl in detected_labels if dl not in label_map]
    if unmatched:
        print(f"  {RED}[Warning] Unmatched labels: {unmatched}{RESET_COLOR}")

    print(f"\n{YELLOW}" + _log(f"Determining target positions ({reset_mode} mode)", step="Target") + f"{RESET_COLOR}")

    if reset_mode == "random":
        # random 모드: label_map을 사용하여 initial_positions 키를 매핑
        mapped_initial = {}
        for dl, ol in label_map.items():
            if ol in original_positions:
                mapped_initial[dl] = original_positions[ol]

        target_positions = generate_random_positions(
            grippable_objects=grippable_objects,
            obstacle_objects=obstacle_objects,
            initial_positions=mapped_initial if mapped_initial else original_positions,
            workspace=workspace,
            pix2robot=coord_transformer if hasattr(coord_transformer, 'robot_to_pixel') else None,
            seed=random_seed,
            resetspace=resetspace,
        )
        print(f"  Random target positions generated:")
    else:
        # original mode: label_map을 사용하여 grippable → original 매핑
        target_positions = {}
        for name in grippable_objects.keys():
            original_key = label_map.get(name, name)
            if original_key in original_positions and original_positions[original_key] is not None:
                target_positions[name] = original_positions[original_key]
                print(f"    {name} → target from '{original_key}'")

        if not target_positions:
            print(f"  {RED}[Error] No target positions matched!{RESET_COLOR}")
            print(f"  Grippable labels: {list(grippable_objects.keys())}")
            print(f"  Original labels: {list(original_positions.keys())}")
            print(f"  Label map: {label_map}")
            raise ValueError(
                f"Reset target matching failed: grippable={list(grippable_objects.keys())} "
                f"vs original={list(original_positions.keys())}. Label map: {label_map}"
            )

        print(f"  Restoring to initial positions:")

    for name, info in target_positions.items():
        if isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # ── skip_codegen 모드: T0~T2 검출만 수행, 코드 생성 스킵 ──
    if skip_codegen:
        print(f"\n{LIGHT_GREEN}" + _log("Code generation SKIPPED (reusing cached code)", step="CodeGen") + f"{RESET_COLOR}")
        code = ""
    else:
        # ── Context Summary Turn (Session 1 마지막) ──
        from .prompt import reset_context_summary_prompt, codegen_reset_with_context_prompt
        print(f"\n{YELLOW}" + _log("Context Summary (handoff)", step="Summary") + f"{RESET_COLOR}")
        summary_resp = gemini_chat_send(chat, gen_config,
            {"text": reset_context_summary_prompt()},
            turn_label="Context Summary (Reset)")
        _accumulate_usage("Context Summary")
        print(f"  Summary: {summary_resp[:200]}{'...' if len(summary_resp) > 200 else ''}")

        # ── Code Generation (새 Session 2) ──
        session2_model = codegen_model or llm_model
        print(f"\n{YELLOW}" + _log(f"Code Generation (new session: {session2_model})", step="CodeGen") + f"{RESET_COLOR}")

        # Multi-arm: use multi-arm reset system prompt + user prompt + workspace images
        if robot_ids and len(robot_ids) >= 2:
            from ..multi_arm.reset_execution.system_prompt import MULTI_ARM_CODEGEN_RESET_SYSTEM_PROMPT
            from ..multi_arm.reset_execution.turn3_prompt import multi_arm_turn3_reset_codegen_prompt
            from ..code_gen_with_skill import _points_to_positions

            codegen_chat, codegen_config = gemini_chat_start(session2_model, system_prompt=MULTI_ARM_CODEGEN_RESET_SYSTEM_PROMPT, thinking_budget=5000)

            # Build dual-arm current positions (pixel → per-arm robot frame)
            # Only include grippable objects (exclude obstacles like large containers)
            grippable_names = set(grippable_objects.keys())
            current_dual = {}
            for arm_key, rid in [("left_arm", robot_ids[0]), ("right_arm", robot_ids[1])]:
                arm_pos = _points_to_positions(all_points, robot_id=rid, camera=camera, valid_objects=valid_objects)
                current_dual[arm_key] = {k: v for k, v in arm_pos.items() if k in grippable_names}

            # Build dual-arm target positions from original per-arm data.
            # target_positions is flat {obj: info}, but codegen needs per-arm coordinates.
            # Use original_positions_dual (per-arm) with label_map to remap keys.
            if original_positions_dual:
                target_dual = {}
                for arm_key in ["left_arm", "right_arm"]:
                    arm_orig = original_positions_dual.get(arm_key, {})
                    arm_target = {}
                    for detected_name in target_positions:
                        original_key = label_map.get(detected_name, detected_name)
                        if original_key in arm_orig:
                            arm_target[detected_name] = arm_orig[original_key]
                    target_dual[arm_key] = arm_target
            else:
                # Fallback: same flat positions for both arms
                target_dual = target_positions

            codegen_text = multi_arm_turn3_reset_codegen_prompt(
                current_positions=current_dual,
                target_positions=target_dual,
                robot_ids=robot_ids,
                instruction=original_instruction,
                context_summary=summary_resp,
                all_points=all_points,
            )

            # Generate per-arm workspace images
            codegen_workspace_images = []
            current_img_for_ws = cv2.imread(current_state_image_path)
            if current_img_for_ws is not None:
                arm_labels = ["left_arm", "right_arm"]
                # resetspace를 로봇별로 분리 (dict이면 robot_id로, 아니면 동일 값 적용)
                resetspace_map = {}
                if isinstance(resetspace, dict):
                    resetspace_map = resetspace
                else:
                    for rid in robot_ids:
                        resetspace_map[rid] = resetspace
                for i, rid in enumerate(robot_ids):
                    arm_rs = resetspace_map.get(rid, None)
                    arm_annotated = draw_workspace_on_image(current_img_for_ws.copy(), robot_id=rid, resetspace=arm_rs)
                    arm_path = str(reset_dir / f"workspace_{arm_labels[i]}_robot{rid}.jpg")
                    cv2.imwrite(arm_path, arm_annotated)
                    codegen_workspace_images.append(arm_path)
                    print(f"  Workspace image ({arm_labels[i]}/robot{rid}): {arm_path}")

            codegen_resp = gemini_chat_send(codegen_chat, codegen_config,
                {"text": codegen_text, "image_paths": codegen_workspace_images},
                turn_label="CodeGen (Reset Multi-Arm)")

            # Override return values with per-arm dicts so runtime globals match the generated code
            current_positions = current_dual
            target_positions = target_dual
        else:
            from ..forward_execution.system_prompt import CODEGEN_SYSTEM_PROMPT
            codegen_chat, codegen_config = gemini_chat_start(session2_model, system_prompt=CODEGEN_SYSTEM_PROMPT, thinking_budget=5000)
            codegen_resp = gemini_chat_send(codegen_chat, codegen_config,
                {"text": codegen_reset_with_context_prompt(
                    context_summary=summary_resp,
                    target_positions=target_positions,
                    current_positions={
                        name: grippable_objects[name]
                        for name in grippable_objects
                        if name in target_positions
                    },
                    robot_id=robot_id,
                    is_random_reset=(reset_mode == "random"),
                    all_points=all_points,
                )},
                turn_label="CodeGen (Reset)")
        _accumulate_usage("CodeGen")

        code = extract_code_from_response(codegen_resp)
        assert code, "Failed to extract code from CodeGen response"

    # Summary
    print(GRAY + "=" * line_width + RESET_COLOR)
    print(LIGHT_GREEN + f"{mode_str} multi-turn code gen completed.".center(line_width) + RESET_COLOR)
    s = _usage_stats
    print(f"  Total: {s['total_inference_time']:.1f}s, in={s['total_in']}, out={s['total_out']}, think={s['total_think']}, tokens={s['total_tokens']}")
    print(GRAY + "=" * line_width + RESET_COLOR)

    # llm_cost를 모듈 변수로 저장 (호출부에서 접근 가능)
    # perception/codegen 분류
    perception_turns = [t for t in _turn_costs if t["turn"] != "CodeGen"]
    codegen_turns = [t for t in _turn_costs if t["turn"] == "CodeGen"]
    def _sum(ts, key): return sum(t.get(key, 0) for t in ts)
    lerobot_reset_code_gen_multi_turn._last_llm_cost = {
        "perception": {
            "model": llm_model,
            "inference_time_s": round(_sum(perception_turns, "inference_time_s"), 2),
            "input_tokens": _sum(perception_turns, "input_tokens"),
            "output_tokens": _sum(perception_turns, "output_tokens"),
            "thinking_tokens": _sum(perception_turns, "thinking_tokens"),
            "total_tokens": _sum(perception_turns, "total_tokens"),
            "turns": perception_turns,
        },
        "codegen": {
            "model": codegen_model or llm_model,
            "inference_time_s": round(_sum(codegen_turns, "inference_time_s"), 2),
            "input_tokens": _sum(codegen_turns, "input_tokens"),
            "output_tokens": _sum(codegen_turns, "output_tokens"),
            "thinking_tokens": _sum(codegen_turns, "thinking_tokens"),
            "total_tokens": _sum(codegen_turns, "total_tokens"),
            "turns": codegen_turns,
        },
        "total": {
            "inference_time_s": round(_usage_stats["total_inference_time"], 2),
            "input_tokens": _usage_stats["total_in"],
            "output_tokens": _usage_stats["total_out"],
            "thinking_tokens": _usage_stats["total_think"],
            "total_tokens": _usage_stats["total_tokens"],
        },
    }

    # Multi-turn info 수집 (forward와 동일한 구조)
    reset_multi_turn_info = {
        "turn0_response": turn0_resp,
        "turn1_response": turn1_resp,
        "turn1_parsed": valid_objects,
        "all_points": all_points,
        "crop_responses": crop_responses,
        "detected_objects": valid_objects,
        "crop_dir": crop_dir,
    }

    return code, current_positions, target_positions, grippable_objects, obstacle_objects, reset_multi_turn_info
