from .llm import llm_response
from .forward_execution.user_prompt import lerobot_code_gen_prompt, turn3_code_gen_prompt, context_summary_prompt, codegen_with_context_prompt
from .forward_execution.turn0_prompt import turn0_scene_understanding_prompt
from .forward_execution.turn1_prompt import turn1_detect_task_relevant_objects_prompt
from .forward_execution.turn2_prompt import turn2_crop_pointing_prompt
from .forward_execution.turn_test_prompt import turn_test_waypoint_trajectory_prompt
from .forward_execution.system_prompt import PERCEPTION_SYSTEM_PROMPT, CODEGEN_SYSTEM_PROMPT

import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def lerobot_code_gen(
    instruction: str,
    object_queries: List[str] = None,
    object_positions: Dict = None,
    use_detection: bool = True,
    detection_timeout: float = 10.0,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    visualize_detection: bool = False,
    image_path: str = None,
    # Episode tracking for logging
    current_episode: int = 1,
    total_episodes: int = 1,
) -> str:
    """
    LeRobot SO-101용 스킬 기반 코드 생성 (단순화 버전)

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_queries: 디텍션 ON시 찾을 객체 리스트 (예: ["red cup", "blue box"])
        object_positions: 디텍션 OFF시 직접 전달할 위치 딕셔너리
                         Extended format: {"name": {"position": [x,y,z], ...}}
                         Legacy format: {"name": [x,y,z]}
        use_detection: True면 object_detection으로 위치 획득, False면 object_positions 사용
        detection_timeout: 디텍션 타임아웃 (초)
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        실행 가능한 Python 코드 문자열

    Raises:
        ValueError: use_detection=True인데 object_queries가 없거나,
                   use_detection=False인데 object_positions가 없는 경우

    Example:
        # 디텍션 ON (카메라로 객체 위치 자동 획득)
        code = lerobot_code_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_queries=["red cup", "blue box"],
            use_detection=True,
        )

        # 디텍션 OFF (위치 직접 전달 - extended format)
        code = lerobot_code_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_positions={
                "red cup": {"position": [0.15, 0.05, 0.02]},
                "blue box": {"position": [0.20, -0.05, 0.03]},
            },
            use_detection=False,
        )
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    # Episode prefix for logging
    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    def _log(msg: str, step: str = None) -> str:
        prefix = f"[Forward][{ep_str}]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Code Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 1) 객체 위치 획득
    if use_detection:
        if object_queries is None or len(object_queries) == 0:
            raise ValueError("use_detection=True requires object_queries (non-empty list)")

        print(f"{YELLOW}" + _log(f"Detecting objects: {object_queries}", step="1/3") + f"{RESET}")

        try:
            from run_detect import run_realtime_detection
            positions = run_realtime_detection(
                queries=object_queries,
                timeout=detection_timeout,
                unit="m",
                visualize=visualize_detection,
            )
        except ImportError as e:
            print(f"[Error] Could not import object_positions module: {e}")
            print("[Fallback] Using empty positions")
            positions = {q: None for q in object_queries}
        except Exception as e:
            print(f"[Error] Detection failed: {e}")
            positions = {q: None for q in object_queries}

    else:
        if object_positions is None or len(object_positions) == 0:
            raise ValueError("use_detection=False requires object_positions (non-empty dict)")

        print(f"{YELLOW}" + _log("Using provided positions", step="1/3") + f"{RESET}")
        positions = object_positions

    # Normalize to extended format if legacy format is used
    # Legacy: {"name": [x,y,z]} → Extended: {"name": {"position": [x,y,z]}}
    normalized_positions = {}
    for name, info in positions.items():
        if info is None:
            normalized_positions[name] = None
        elif isinstance(info, dict) and "position" in info:
            # Already extended format
            normalized_positions[name] = info
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            # Legacy format - convert to extended
            normalized_positions[name] = {
                "position": list(info[:3]),
            }
        else:
            normalized_positions[name] = None

    positions = normalized_positions

    # 검출 결과 출력
    found_count = sum(1 for v in positions.values() if v is not None)
    print(f"  Found: {found_count}/{len(positions)} objects")
    for name, info in positions.items():
        if info:
            pos = info["position"]
            bbox = info.get("bbox_size_m")
            grippable = info.get("grippable", True)

            # 기본 정보
            base_info = f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"

            # bbox 크기 정보
            if bbox:
                width_cm = bbox[0] * 100
                height_cm = bbox[1] * 100
                size_info = f"size=[{width_cm:.1f}x{height_cm:.1f}]cm"
            else:
                size_info = "size=N/A"

            # grippable 정보
            grip_info = f"grippable={grippable}"

            print(f"    ✓ {name}: {base_info}, {size_info}, {grip_info}")
        else:
            print(f"    ✗ {name}: NOT FOUND")

    # 검출 결과 검증 - 모든 객체가 검출되어야 함
    not_found = [name for name, info in positions.items() if info is None]
    if not_found:
        print("\n" + "="*60)
        print("ERROR: Required objects not detected!")
        print("="*60)
        print(f"  Missing objects: {not_found}")
        print(f"  Detected objects: {[name for name, info in positions.items() if info is not None]}")
        print("="*60)
        assert False, f"Cannot generate code: objects not detected - {not_found}"

    # 2) 프롬프트 생성
    print(f"\n{YELLOW}" + _log("Generating prompt", step="2/3") + f"{RESET}")
    prompt = lerobot_code_gen_prompt(
        instruction=instruction,
        object_positions=positions,
        robot_id=robot_id,
    )

    # 3) LLM 호출 (단일턴 모드: codegen 프롬프트 사용)
    system_prompt = CODEGEN_SYSTEM_PROMPT
    if image_path:
        print(f"\n{YELLOW}" + _log(f"Calling LLM ({llm_model}) with image: {image_path}", step="3/3") + f"{RESET}")
    else:
        print(f"\n{YELLOW}" + _log(f"Calling LLM ({llm_model})", step="3/3") + f"{RESET}")
    response = llm_response(
        llm_model, prompt,
        check_time=True,
        system_prompt=system_prompt,
        image_path=image_path,
    )

    # LLM 응답 검증
    assert response is not None, "LLM response is None - check server connection or API key"

    # 4) 코드 추출
    code = extract_code_from_response(response)

    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + "Code generation completed successfully.".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 코드와 positions 함께 반환
    return code, positions


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
    # "from skills" 또는 "def execute_task"로 시작하는 부분 찾기
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute_task', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    # 그래도 없으면 전체 반환
    return response.strip()


def _parse_json_from_response(response: str) -> Optional[Dict]:
    """LLM 응답에서 JSON 블록을 추출하여 파싱"""
    # ```json ... ``` 블록 찾기
    json_block_pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(json_block_pattern, response, re.DOTALL)
    if matches:
        for m in matches:
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue

    # { ... } 직접 찾기
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _points_to_positions(
    all_points: list,
    robot_id: int = 3,
    camera=None,
    valid_objects: list = None,
) -> Dict:
    """
    Crop-then-point 결과에서 object별 모든 point를 pixel→world 변환.

    각 object에서 "grasp" role 포인트를 기본 position으로 선택하고,
    모든 point를 "points" 딕셔너리에 포함시킴.

    Args:
        all_points: [{"object_label", "label", "role", "px", "py"}, ...]
        robot_id: 로봇 번호
        camera: depth 카메라 (3D 변환용)
        valid_objects: Turn 1 bbox 결과 [{"box_2d": [...], "label": str}, ...]

    Returns:
        {object_label: {"position": [x,y,z], "pixel": [px,py], "bbox_px": (w,h),
                         "points": {"label": [x,y,z], ...}}}
    """
    # Object별로 모든 point 수집 + grasp point 선택
    points_by_object = {}  # {obj: [pt, ...]}
    grasp_by_object = {}   # {obj: pt} — 기본 position용
    for pt in all_points:
        obj = pt["object_label"]
        if obj not in points_by_object:
            points_by_object[obj] = []
        points_by_object[obj].append(pt)

        if obj not in grasp_by_object:
            grasp_by_object[obj] = pt  # 첫 번째 point (fallback)
        elif pt.get("role") == "grasp" and grasp_by_object[obj].get("role") != "grasp":
            grasp_by_object[obj] = pt  # grasp 우선

    if not grasp_by_object:
        return {}

    # Pix2Robot 캘리브레이션 로드 (pixel → robot 직접 변환)
    pix2robot = None
    try:
        from pix2robot_calibrator import Pix2RobotCalibrator
        calib_path = Path(__file__).parent.parent / "robot_configs" / "pix2robot_matrices" / f"robot{robot_id}_pix2robot_data.npz"
        if calib_path.exists():
            pix2robot = Pix2RobotCalibrator(robot_id=robot_id)
            if pix2robot.load(str(calib_path)):
                print(f"  [CropPoint] Pix2Robot calibration loaded ({len(pix2robot.pixel_points)} points)")
            else:
                pix2robot = None
    except Exception as e:
        print(f"  [CropPoint] Pix2Robot not available: {e}")

    # Depth 프레임 (pix2robot 높이 추정용)
    depth_frame = None
    if camera is not None:
        try:
            _, depth_frame = camera.get_frames()
        except Exception:
            pass

    # pixel→robot 변환 헬퍼
    Z_DEFAULT = 0.02  # z 이상 시 대체값 (2cm)

    def _pixel_to_robot(px, py):
        if pix2robot is not None:
            try:
                obj_depth = None
                if depth_frame is not None and camera is not None:
                    obj_depth = camera.get_depth_at_pixel(px, py, depth_frame)
                    if obj_depth <= 0.05:
                        obj_depth = None
                pos = pix2robot.pixel_to_robot(px, py, depth_m=obj_depth)
                return pos, True
            except Exception as e:
                print(f"    pix2robot failed ({px},{py}): {e}")
        return [0.0, 0.0, 0.03], False

    # bbox_px 맵 구축 (valid_objects에서 추출)
    bbox_px_map = {}
    if valid_objects:
        # 이미지 크기 기본값 (640x480)
        img_w, img_h = 640, 480
        for obj in valid_objects:
            label = obj.get("label", "")
            box = obj.get("box_2d", [])
            if len(box) == 4 and label:
                ymin, xmin, ymax, xmax = box
                w_px = int((xmax - xmin) * img_w / 1000)
                h_px = int((ymax - ymin) * img_h / 1000)
                bbox_px_map[label] = (max(w_px, 10), max(h_px, 10))

    # positions 구성
    positions = {}
    for obj_label, grasp_pt in grasp_by_object.items():
        # 기본 position (grasp point)
        gpx, gpy = grasp_pt["px"], grasp_pt["py"]
        world_pos, success = _pixel_to_robot(gpx, gpy)

        positions[obj_label] = {
            "position": world_pos,
            "pixel": [gpx, gpy],
            "bbox_px": bbox_px_map.get(obj_label, (30, 30)),
        }
        if not success:
            positions[obj_label]["_needs_world_coords"] = True

        # 모든 point를 points 딕셔너리에 추가
        obj_points = {}
        for pt in points_by_object.get(obj_label, []):
            pt_label = pt.get("label", "unknown")
            pt_px, pt_py = pt["px"], pt["py"]
            pt_world, pt_success = _pixel_to_robot(pt_px, pt_py)
            if pt_success:
                obj_points[pt_label] = pt_world
        if obj_points:
            positions[obj_label]["points"] = obj_points

        print(f"    {obj_label}: pos={world_pos} [OK]")
        if obj_points:
            for lbl, pos in obj_points.items():
                print(f"      - {lbl}: {pos}")

    return positions


def lerobot_code_gen_multi_turn(
    instruction: str,
    image_path: str,
    llm_model: str = "gemini-2.0-flash",
    robot_id: int = 3,
    current_episode: int = 1,
    total_episodes: int = 1,
    fallback_positions: Dict = None,
    camera=None,
    cad_image_dirs: List[str] = None,
    side_view_image: str = None,
    codegen_model: str = None,
    skip_codegen: bool = False,
    canonical_labels: List[str] = None,
    canonical_point_labels: Dict[str, List[str]] = None,
    skip_turn_test: bool = False,
    robot_ids: List[int] = None,
) -> Tuple[str, Dict, Dict]:
    """
    Crop-then-Point 멀티턴 LLM 코드 생성 파이프라인

    Turn 0: 이미지 (+ CAD) + instruction → 장면 이해 (reasoning only)
    Turn 1: bbox 검출 (JSON)
    Turn 2~N: 물체별 crop → critical point pointing
    Turn N+1: 코드 생성

    Args:
        instruction: 자연어 태스크 명령
        image_path: 초기 오버헤드 카메라 이미지 경로
        llm_model: LLM 모델 (Gemini 모델 필요)
        robot_id: 로봇 번호 (2 또는 3)
        current_episode: 현재 에피소드 번호
        total_episodes: 총 에피소드 수
        fallback_positions: Grounding DINO fallback 용 positions
        camera: RealSenseD435 카메라 (depth 기반 3D 좌표 변환용)
        cad_image_dirs: CAD 참조 이미지 디렉토리 리스트 (옵션)

    Returns:
        Tuple[str, Dict, Dict]:
            - 실행 가능한 Python 코드
            - positions dict (extended format)
            - multi_turn_info: 각 턴별 응답 등 메타 정보
    """
    import cv2
    import glob as glob_mod
    import tempfile
    from .llm_utils.gemini import gemini_chat_start, gemini_chat_send

    CROP_PADDING = 25  # bbox 패딩 (0-1000 스케일)

    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    ep_str = f"{current_episode:02d}/{total_episodes:02d}"

    def _log(msg, step=None):
        prefix = f"[Forward][{ep_str}][CropPoint]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Crop-then-Point Code Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)
    print(f"  Model: {llm_model}")
    print(f"  Image: {image_path}")

    # 토큰/시간 통계 누적
    _usage_stats = {"total_inference_time": 0.0, "total_in": 0, "total_out": 0, "total_think": 0, "total_tokens": 0}
    _turn_costs = []  # 턴별 비용 기록
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

    def _build_llm_cost(turns, stats, perception_model, codegen_model):
        """턴별 비용을 perception/codegen으로 분류하여 구조화"""
        perception_turns = [t for t in turns if t["turn"] != "Code Gen"]
        codegen_turns = [t for t in turns if t["turn"] == "Code Gen"]
        def _sum(ts, key): return sum(t.get(key, 0) for t in ts)
        return {
            "perception": {
                "model": perception_model,
                "inference_time_s": round(_sum(perception_turns, "inference_time_s"), 2),
                "input_tokens": _sum(perception_turns, "input_tokens"),
                "output_tokens": _sum(perception_turns, "output_tokens"),
                "thinking_tokens": _sum(perception_turns, "thinking_tokens"),
                "total_tokens": _sum(perception_turns, "total_tokens"),
                "turns": perception_turns,
            },
            "codegen": {
                "model": codegen_model,
                "inference_time_s": round(_sum(codegen_turns, "inference_time_s"), 2),
                "input_tokens": _sum(codegen_turns, "input_tokens"),
                "output_tokens": _sum(codegen_turns, "output_tokens"),
                "thinking_tokens": _sum(codegen_turns, "thinking_tokens"),
                "total_tokens": _sum(codegen_turns, "total_tokens"),
                "turns": codegen_turns,
            },
            "total": {
                "inference_time_s": round(stats["total_inference_time"], 2),
                "input_tokens": stats["total_in"],
                "output_tokens": stats["total_out"],
                "thinking_tokens": stats["total_think"],
                "total_tokens": stats["total_tokens"],
            },
        }

    # CAD 이미지 수집 (Step 6)
    cad_paths = []
    if cad_image_dirs:
        for d in cad_image_dirs:
            cad_paths.extend(sorted(glob_mod.glob(f"{d}/*.jpg")))
            cad_paths.extend(sorted(glob_mod.glob(f"{d}/*.png")))
        print(f"  CAD images: {len(cad_paths)} files from {len(cad_image_dirs)} dirs")

    # Chat session 시작 (Session 1: Perception)
    if robot_ids and len(robot_ids) >= 2:
        from .multi_arm.forward_execution.system_prompt import MULTI_ARM_PERCEPTION_SYSTEM_PROMPT
        perception_prompt = MULTI_ARM_PERCEPTION_SYSTEM_PROMPT
    else:
        perception_prompt = PERCEPTION_SYSTEM_PROMPT
    chat, gen_config = gemini_chat_start(llm_model, system_prompt=perception_prompt)

    has_cad = bool(cad_paths)

    # ── Turn 0: Scene Understanding (이미지 + CAD) — test6 방식 ──
    print(f"\n{YELLOW}" + _log("Turn 0 — Scene Understanding", step="Turn0") + f"{RESET}")
    turn0_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn0_scene_understanding_prompt(instruction, has_cad=has_cad),
            "image_path": image_path,
            "image_paths": cad_paths,
        },
        turn_label="Turn 0")
    _accumulate_usage("Turn 0")
    print(f"  {turn0_resp[:300]}{'...' if len(turn0_resp) > 300 else ''}")

    # ── Turn 1: BBox Detection (이미지 재전송) — test6 방식 ──
    has_side_view = side_view_image and os.path.isfile(side_view_image)
    print(f"\n{YELLOW}" + _log("Turn 1 — BBox Detection", step="Turn1") + f"{RESET}")
    if has_side_view:
        print(f"    Dual-view mode: overhead + side-view ({side_view_image})")

    turn1_text = turn1_detect_task_relevant_objects_prompt(has_side_view=has_side_view)
    if canonical_labels:
        label_list = ", ".join(f'"{l}"' for l in canonical_labels)
        turn1_text += f"\n\n**IMPORTANT: You MUST use exactly these labels: [{label_list}]. Do NOT rename or paraphrase them.**"
        print(f"    [CodeReuse] Enforcing canonical labels: {canonical_labels}")
    turn1_msg = {
        "text": turn1_text,
        "image_path": image_path,
    }
    if has_side_view:
        turn1_msg["image_paths"] = [side_view_image]
    turn1_resp = gemini_chat_send(chat, gen_config, turn1_msg, turn_label="Turn 1")
    _accumulate_usage("Turn 1")
    print(f"  {turn1_resp[:300]}{'...' if len(turn1_resp) > 300 else ''}")

    # Parse bboxes (dual-view or single-view)
    turn1_data = _parse_json_from_response(turn1_resp)

    sv_bbox_by_label = {}  # label → sideview bbox (for Turn 2 crop)
    turn1_sideview_parsed = []  # for visualization

    if has_side_view and isinstance(turn1_data, dict) and "overhead" in turn1_data:
        # Dual-view response: {"overhead": [...], "sideview": [...]}
        oh_list = turn1_data.get("overhead", [])
        sv_list = turn1_data.get("sideview", [])
        obj_list = oh_list  # overhead objects are the primary list

        # Build sideview bbox lookup by label
        for sv_obj in sv_list:
            sv_box = sv_obj.get("box_2d") or sv_obj.get("bbox") or []
            sv_label = sv_obj.get("label", "")
            if len(sv_box) == 4 and sv_label:
                sv_bbox_by_label[sv_label] = sv_box
                turn1_sideview_parsed.append({"box_2d": sv_box, "label": sv_label})
                print(f"    [sideview] [{sv_label}] bbox={sv_box}")
    else:
        # Single-view response: list or {"objects": [...]}
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
            print(f"    [overhead] [{obj['label']}] bbox={box}")

    assert valid_objects, "No valid bboxes detected from Turn 1"

    # 이미지 로드
    full_img = cv2.imread(image_path)
    assert full_img is not None, f"Cannot read image: {image_path}"
    img_h, img_w = full_img.shape[:2]

    # Side-view 이미지 로드 (for cropping in Turn 2)
    sv_full_img = None
    sv_img_h, sv_img_w = 0, 0
    if has_side_view:
        sv_full_img = cv2.imread(side_view_image)
        if sv_full_img is not None:
            sv_img_h, sv_img_w = sv_full_img.shape[:2]
            print(f"    Side-view image loaded: {sv_img_w}x{sv_img_h}")
        else:
            print(f"    [Warning] Cannot read side-view image: {side_view_image}")
            has_side_view = False

    # ── Turn 2+: Crop-then-Point ──
    all_points = []
    crop_responses = []
    crop_dir = tempfile.mkdtemp(prefix="crop_")

    for i, obj in enumerate(valid_objects):
        label = obj["label"]
        ymin, xmin, ymax, xmax = obj["box_2d"]

        print(f"\n{YELLOW}" + _log(f"Crop — {label}", step=f"Crop{i}") + f"{RESET}")

        # ── Overhead crop ──
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
        print(f"    [overhead] bbox=[{ymin},{xmin},{ymax},{xmax}] → crop ({crop_w}x{crop_h})")

        # Save overhead crop
        safe_label = label.replace(" ", "_").replace("/", "_")
        crop_path = f"{crop_dir}/crop_{safe_label}.jpg"
        cv2.imwrite(crop_path, crop_img)

        # ── Side-view crop (if available) ──
        obj_has_sv = has_side_view and label in sv_bbox_by_label and sv_full_img is not None
        sv_crop_path = None
        sv_crop_x1 = sv_crop_y1 = sv_crop_w = sv_crop_h = 0

        if obj_has_sv:
            sv_ymin, sv_xmin, sv_ymax, sv_xmax = sv_bbox_by_label[label]
            sv_ymin_p = max(0, sv_ymin - CROP_PADDING)
            sv_xmin_p = max(0, sv_xmin - CROP_PADDING)
            sv_ymax_p = min(1000, sv_ymax + CROP_PADDING)
            sv_xmax_p = min(1000, sv_xmax + CROP_PADDING)

            sv_crop_x1 = int(sv_xmin_p * sv_img_w / 1000)
            sv_crop_y1 = int(sv_ymin_p * sv_img_h / 1000)
            sv_crop_x2 = int(sv_xmax_p * sv_img_w / 1000)
            sv_crop_y2 = int(sv_ymax_p * sv_img_h / 1000)

            sv_crop_img = sv_full_img[sv_crop_y1:sv_crop_y2, sv_crop_x1:sv_crop_x2]
            sv_crop_h, sv_crop_w = sv_crop_img.shape[:2]
            print(f"    [sideview] bbox=[{sv_ymin},{sv_xmin},{sv_ymax},{sv_xmax}] → crop ({sv_crop_w}x{sv_crop_h})")

            sv_crop_path = f"{crop_dir}/crop_sv_{safe_label}.jpg"
            cv2.imwrite(sv_crop_path, sv_crop_img)

        # Send crop(s) + pointing prompt
        turn2_msg = {
            "text": turn2_crop_pointing_prompt(label, has_side_view=obj_has_sv, canonical_point_labels=canonical_point_labels),
            "image_path": crop_path,
        }
        if obj_has_sv and sv_crop_path:
            turn2_msg["image_paths"] = [sv_crop_path]

        resp = gemini_chat_send(chat, gen_config, turn2_msg,
            turn_label=f"Crop: {label}")
        _accumulate_usage(f"Crop: {label}")
        crop_responses.append({"label": label, "response": resp})
        print(f"    {resp[:200]}{'...' if len(resp) > 200 else ''}")

        # Parse critical_points (dual or single view)
        parsed = _parse_json_from_response(resp)
        if not parsed:
            print(f"    [Warning] Failed to parse points for '{label}'")
            continue

        # Determine which key(s) hold the overhead points
        if obj_has_sv and "overhead_critical_points" in parsed:
            oh_points = parsed["overhead_critical_points"]
            sv_points = parsed.get("sideview_critical_points", [])
            # Build label→sv_point lookup for matching
            sv_by_label = {}
            for sv_pt in sv_points:
                sv_lbl = sv_pt.get("label", "")
                if sv_lbl:
                    sv_by_label[sv_lbl] = sv_pt
        elif "critical_points" in parsed:
            oh_points = parsed["critical_points"]
            sv_by_label = {}
        else:
            print(f"    [Warning] No recognized point keys for '{label}'")
            continue

        # canonical point labels가 있으면 VLM 출력 라벨을 강제 교체
        canonical_pts = canonical_point_labels.get(label, []) if canonical_point_labels else []

        for pt_idx, pt in enumerate(oh_points):
            point_2d = pt.get("point_2d", [])
            if len(point_2d) != 2:
                continue
            norm_y, norm_x = point_2d
            crop_px = int(norm_x * crop_w / 1000)
            crop_py = int(norm_y * crop_h / 1000)
            px = crop_x1 + crop_px
            py = crop_y1 + crop_py

            # VLM 라벨을 canonical으로 강제 교체 (인덱스 매칭)
            pt_label = pt.get("label", "")
            if canonical_pts and pt_idx < len(canonical_pts) and pt_label != canonical_pts[pt_idx]:
                print(f"    [LabelFix] '{pt_label}' → '{canonical_pts[pt_idx]}'")
                pt_label = canonical_pts[pt_idx]

            entry = {
                "object_label": label,
                "label": pt_label,
                "role": pt.get("role", "interaction"),
                "reasoning": pt.get("reasoning", ""),
                "point_2d": point_2d,
                "px": px, "py": py,
                "crop_px": crop_px, "crop_py": crop_py,
            }

            # Match side-view point by label (pt_label은 위에서 canonical 교체 완료)
            sv_match = sv_by_label.get(pt_label)
            if sv_match:
                sv_pt_2d = sv_match.get("point_2d", [])
                if len(sv_pt_2d) == 2:
                    sv_norm_y, sv_norm_x = sv_pt_2d
                    sv_cpx = int(sv_norm_x * sv_crop_w / 1000)
                    sv_cpy = int(sv_norm_y * sv_crop_h / 1000)
                    sv_px_full = sv_crop_x1 + sv_cpx
                    sv_py_full = sv_crop_y1 + sv_cpy
                    entry["sv_point_2d"] = sv_pt_2d
                    entry["sv_crop_px"] = sv_cpx
                    entry["sv_crop_py"] = sv_cpy
                    entry["sv_px"] = sv_px_full
                    entry["sv_py"] = sv_py_full
                    print(f"    [{pt.get('role','?')}] OH({norm_y},{norm_x})→full({px},{py}) "
                          f"| SV({sv_norm_y},{sv_norm_x})→full({sv_px_full},{sv_py_full})")
                else:
                    print(f"    [{pt.get('role','?')}] OH({norm_y},{norm_x})→full({px},{py}) | SV: invalid point_2d")
            else:
                print(f"    [{pt.get('role','?')}] ({norm_y},{norm_x}) "
                      f"→ crop({crop_px},{crop_py}) → full({px},{py})")

            all_points.append(entry)

    print(f"\n  Total: {len(all_points)} points across {len(valid_objects)} objects")
    sv_count = sum(1 for pt in all_points if pt.get("sv_px") is not None)
    if sv_count:
        print(f"  Side-view points matched: {sv_count}/{len(all_points)}")

    # Positions 구성 (pixel → world)
    print(f"\n{YELLOW}" + _log("Building positions...", step="Positions") + f"{RESET}")
    positions = _points_to_positions(all_points, robot_id=robot_id, camera=camera, valid_objects=valid_objects)

    # Fallback
    if fallback_positions:
        for name, info in positions.items():
            if info.get("_needs_world_coords") and name in fallback_positions:
                fb = fallback_positions[name]
                if fb is not None:
                    fb_pos = fb["position"] if isinstance(fb, dict) and "position" in fb else fb
                    info["position"] = list(fb_pos[:3])
                    info.pop("_needs_world_coords", None)

    for name, info in positions.items():
        pos = info["position"]
        status = "NEED WORLD" if info.get("_needs_world_coords") else "OK"
        print(f"    {name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] [{status}]")

    # ── Turn Test: Waypoint Trajectory Prediction ──
    turn_test_overhead_waypoints = []
    turn_test_sideview_waypoints = []
    turn_test_resp = ""

    if len(all_points) >= 2 and not skip_turn_test:
        print(f"\n{YELLOW}" + _log("Waypoint Trajectory", step="TurnTest") + f"{RESET}")
        print(f"    All detected points: {len(all_points)}")
        for pt in all_points:
            print(f"      - [{pt['role']}] {pt['object_label']}: {pt['label']} @ ({pt['py']}, {pt['px']})")
        print(f"    Phase (from instruction): {instruction}")
        if has_side_view:
            print(f"    Side-view image: {side_view_image}")

        turn_msg = {
            "text": turn_test_waypoint_trajectory_prompt(
                instruction=instruction,
                phase=instruction,
                all_points=all_points,
                has_side_view=has_side_view,
            ),
        }
        if has_side_view:
            turn_msg["image_path"] = side_view_image

        turn_test_resp = gemini_chat_send(chat, gen_config, turn_msg,
            turn_label="Waypoint Trajectory")
        _accumulate_usage("Waypoint Trajectory")
        print(f"    {turn_test_resp[:300]}{'...' if len(turn_test_resp) > 300 else ''}")

        parsed_test = _parse_json_from_response(turn_test_resp)
        if parsed_test:
            # Log selected points
            selected = parsed_test.get("selected_points", [])
            if selected:
                print(f"    Selected points: {selected}")

            # Parse overhead waypoints
            oh_wps = parsed_test.get("overhead_waypoints", [])
            for wp in oh_wps:
                point_2d = wp.get("point_2d", [])
                if len(point_2d) != 2:
                    continue
                wy, wx = point_2d
                turn_test_overhead_waypoints.append({
                    "label": wp.get("label", ""),
                    "reasoning": wp.get("reasoning", ""),
                    "point_2d": point_2d,
                    "py": wy, "px": wx,
                })
                print(f"    [overhead] ({wy}, {wx}) — {wp.get('label', '')}")
            print(f"    Overhead waypoints: {len(turn_test_overhead_waypoints)}")

            # Parse side-view waypoints
            sv_wps = parsed_test.get("sideview_waypoints", [])
            for wp in sv_wps:
                point_2d = wp.get("point_2d", [])
                if len(point_2d) != 2:
                    continue
                wy, wx = point_2d
                turn_test_sideview_waypoints.append({
                    "label": wp.get("label", ""),
                    "reasoning": wp.get("reasoning", ""),
                    "point_2d": point_2d,
                    "py": wy, "px": wx,
                })
                print(f"    [sideview] ({wy}, {wx}) — {wp.get('label', '')}")
            print(f"    Side-view waypoints: {len(turn_test_sideview_waypoints)}")
        else:
            print(f"    [Warning] Failed to parse waypoints from response")
    else:
        print(f"\n{YELLOW}" + _log("Waypoint Trajectory — skipped (< 2 points)", step="TurnTest") + f"{RESET}")

    # ── skip_codegen 모드: T0~T2 검출만 수행, 코드 생성 스킵 ──
    if skip_codegen:
        print(f"\n{LIGHT_GREEN}" + _log("Code generation SKIPPED (reusing cached code)", step="CodeGen") + f"{RESET}")
        code = ""
        codegen_resp = ""
        summary_resp = ""
    else:
        # ── Context Summary Turn (Session 1 마지막) ──
        print(f"\n{YELLOW}" + _log("Context Summary (handoff)", step="Summary") + f"{RESET}")
        summary_resp = gemini_chat_send(chat, gen_config,
            {"text": context_summary_prompt()},
            turn_label="Context Summary")
        _accumulate_usage("Context Summary")
        print(f"  Summary: {summary_resp[:200]}{'...' if len(summary_resp) > 200 else ''}")

        # ── Code Generation (새 Session 2) ──
        session2_model = codegen_model or llm_model
        print(f"\n{YELLOW}" + _log(f"Code Generation (new session: {session2_model})", step="CodeGen") + f"{RESET}")

        # Generate workspace-annotated image for codegen
        codegen_image = image_path
        codegen_extra_images = []  # multi-arm: per-arm workspace images
        try:
            from .reset_execution.workspace import draw_workspace_on_image
            raw_img = cv2.imread(image_path)
            if raw_img is not None:
                if robot_ids and len(robot_ids) >= 2:
                    # Multi-arm: separate workspace image per arm
                    arm_labels = ["left_arm", "right_arm"]
                    for i, rid in enumerate(robot_ids):
                        arm_annotated = draw_workspace_on_image(raw_img.copy(), robot_id=rid)
                        arm_path = str(Path(image_path).parent / f"workspace_{arm_labels[i]}_robot{rid}.jpg")
                        cv2.imwrite(arm_path, arm_annotated)
                        codegen_extra_images.append(arm_path)
                        print(f"  Workspace image ({arm_labels[i]}/robot{rid}): {arm_path}")
                else:
                    annotated = draw_workspace_on_image(raw_img, robot_id=robot_id)
                    annotated_path = str(Path(image_path).parent / "workspace_annotated_codegen.jpg")
                    cv2.imwrite(annotated_path, annotated)
                    codegen_image = annotated_path
                    print(f"  Workspace annotated image: {annotated_path}")
        except Exception as e:
            print(f"  Warning: workspace annotation failed: {e}")

        # Multi-arm: use multi-arm system prompt and user prompt for Turn 3
        if robot_ids and len(robot_ids) >= 2:
            from .multi_arm.forward_execution.system_prompt import MULTI_ARM_CODEGEN_SYSTEM_PROMPT
            codegen_system_prompt = MULTI_ARM_CODEGEN_SYSTEM_PROMPT
        else:
            codegen_system_prompt = CODEGEN_SYSTEM_PROMPT

        codegen_chat, codegen_config = gemini_chat_start(session2_model, system_prompt=codegen_system_prompt, thinking_budget=10000)

        if robot_ids and len(robot_ids) >= 2:
            from .multi_arm.forward_execution.turn3_prompt import multi_arm_turn3_codegen_prompt
            codegen_text = multi_arm_turn3_codegen_prompt(
                instruction=instruction,
                robot_ids=robot_ids,
                all_points=all_points,
                context_summary=summary_resp,
                positions=positions,
            )
        else:
            codegen_text = codegen_with_context_prompt(
                instruction=instruction, robot_id=robot_id,
                all_points=all_points, context_summary=summary_resp,
                positions=positions)

        codegen_msg = {
            "text": codegen_text,
            "image_path": codegen_image if not codegen_extra_images else None,
            "image_paths": codegen_extra_images if codegen_extra_images else [],
        }
        codegen_resp = gemini_chat_send(codegen_chat, codegen_config,
            codegen_msg,
            turn_label="Code Gen")
        _accumulate_usage("Code Gen")
        code = extract_code_from_response(codegen_resp)
        assert code, "Failed to extract code from Code Gen response"

    # multi_turn_info (하위호환: execution_forward_and_reset.py가 사용하는 키 유지)
    turn2_compat = {
        "grasp_points": [
            {
                "object_name": pt["object_label"],
                "label": pt.get("label", ""),
                "role": pt["role"],
                "point_pixel": [
                    int(pt["py"] * 1000 / img_h),
                    int(pt["px"] * 1000 / img_w),
                ],
            }
            for pt in all_points
        ],
    }

    # Side-view grasp points (for visualization)
    if has_side_view and sv_full_img is not None:
        turn2_compat["sv_grasp_points"] = [
            {
                "object_name": pt["object_label"],
                "label": pt.get("label", ""),
                "role": pt["role"],
                "point_pixel": [
                    int(pt["sv_py"] * 1000 / sv_img_h),
                    int(pt["sv_px"] * 1000 / sv_img_w),
                ],
            }
            for pt in all_points
            if pt.get("sv_px") is not None
        ]

    multi_turn_info = {
        "turn0_response": turn0_resp,
        "turn1_response": turn1_resp,
        "turn2_response": "\n".join(cr["response"] for cr in crop_responses),
        "turn3_response": codegen_resp,
        "turn1_parsed": valid_objects,
        "turn2_parsed": turn2_compat,
        # Side-view Turn 1 data (for visualization)
        "turn1_sideview_parsed": turn1_sideview_parsed if turn1_sideview_parsed else None,
        # 신규 필드
        "detected_objects": valid_objects,
        "crop_responses": crop_responses,
        "all_points": all_points,
        "crop_dir": crop_dir,
        "turn_test_response": turn_test_resp,
        "turn_test_overhead_waypoints": turn_test_overhead_waypoints,
        "turn_test_sideview_waypoints": turn_test_sideview_waypoints,
        "side_view_image": side_view_image if has_side_view else None,
        "context_summary": summary_resp,
        "llm_cost": _build_llm_cost(_turn_costs, _usage_stats, llm_model, codegen_model or llm_model),
    }

    n_turns = 2 + len(valid_objects) + 1
    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + f"Crop-then-Point completed ({n_turns} turns).".center(line_width) + RESET)
    s = _usage_stats
    print(f"  Total: {s['total_inference_time']:.1f}s, in={s['total_in']}, out={s['total_out']}, think={s['total_think']}, tokens={s['total_tokens']}")
    print(GRAY + "=" * line_width + RESET)

    return code, positions, multi_turn_info