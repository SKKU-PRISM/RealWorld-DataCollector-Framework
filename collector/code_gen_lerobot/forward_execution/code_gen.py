"""
Forward Execution Code Generator

Generates executable Python code for forward task execution.
"""

import re
from typing import Dict, List, Optional

from ..llm import llm_response
from .user_prompt import lerobot_code_gen_prompt


def lerobot_code_gen(
    instruction: str,
    object_queries: List[str] = None,
    object_positions: Dict[str, List[float]] = None,
    use_detection: bool = True,
    detection_timeout: float = 10.0,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    visualize_detection: bool = False,
) -> str:
    """
    LeRobot SO-101용 스킬 기반 코드 생성 (단순화 버전)

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_queries: 디텍션 ON시 찾을 객체 리스트 (예: ["red cup", "blue box"])
        object_positions: 디텍션 OFF시 직접 전달할 위치 딕셔너리
                         (예: {"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]})
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

        # 디텍션 OFF (위치 직접 전달)
        code = lerobot_code_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_positions={
                "red cup": [0.15, 0.05, 0.02],
                "blue box": [0.20, -0.05, 0.03],
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

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Code Generation (Forward)".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 1) 객체 위치 획득
    if use_detection:
        if object_queries is None or len(object_queries) == 0:
            raise ValueError("use_detection=True requires object_queries (non-empty list)")

        print(f"{YELLOW}[1/3] Detecting objects: {object_queries}{RESET}")

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

        print(f"{YELLOW}[1/3] Using provided positions{RESET}")
        positions = object_positions

    # 검출 결과 출력
    found_count = sum(1 for v in positions.values() if v is not None)
    print(f"  Found: {found_count}/{len(positions)} objects")
    for name, pos in positions.items():
        if pos:
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        else:
            print(f"    x {name}: NOT FOUND")

    # 검출 결과 검증 - 모든 객체가 검출되어야 함
    not_found = [name for name, pos in positions.items() if pos is None]
    if not_found:
        print("\n" + "="*60)
        print("ERROR: Required objects not detected!")
        print("="*60)
        print(f"  Missing objects: {not_found}")
        print(f"  Detected objects: {[name for name, pos in positions.items() if pos is not None]}")
        print("="*60)
        assert False, f"Cannot generate code: objects not detected - {not_found}"

    # 2) 프롬프트 생성
    print(f"\n{YELLOW}[2/3] Generating prompt{RESET}")
    prompt = lerobot_code_gen_prompt(
        instruction=instruction,
        object_positions=positions,
        robot_id=robot_id,
    )

    # 3) LLM 호출
    print(f"\n{YELLOW}[3/3] Calling LLM ({llm_model}){RESET}")
    response = llm_response(llm_model, prompt, check_time=True)

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
