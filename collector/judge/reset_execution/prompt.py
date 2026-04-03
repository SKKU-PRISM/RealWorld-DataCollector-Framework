"""
Reset Judge Prompt Template

Reset 실행 완료 여부를 판단하는 VLM 프롬프트 템플릿
- Original Mode: 원래 위치로 복원 여부 판단
- Random Mode: 랜덤 위치로 이동 여부 판단

Chain-of-Thought 방식으로 단계별 추론을 유도합니다.
"""

from typing import Dict, List, Optional, Tuple, Union


# ============================================================
# System Prompts (Chain-of-Thought)
# ============================================================

RESET_SYSTEM_PROMPT_ORIGINAL = """You are a Reset Task Judge for robotic manipulation tasks.

Your role is to evaluate whether a robot successfully RESTORED objects to their ORIGINAL positions.

## Evaluation Process

You MUST complete all 5 steps and mark each step as PASS or FAIL.

**Step 1: Identify Objects Before Reset**
- Use the provided pixel coordinates to locate each object in Image 1
- Verify the objects are at their "current" positions (after forward task)
- Mark PASS if all objects are correctly identified

**Step 2: Identify Objects After Reset**
- Find the same objects in Image 2 (after reset)
- Note their new approximate pixel positions
- Mark PASS if all objects are visible and identifiable

**Step 3: Verify Position Restoration (Per Object)**
- For EACH object, check if it moved toward its original position
- Compare the final position with the target original position
- Mark PASS if object is at or near its original position

**Step 4: Check Scene Restoration**
- Compare the overall scene in final image with the expected original state
- Verify no objects are dropped or misplaced
- Mark PASS if the scene looks properly restored

**Step 5: Final Judgment**
- Count PASS/FAIL results from steps 1-4
- TRUE: All objects successfully restored (Steps 3 and 4 PASS)
- FALSE: Any object failed to restore
- UNCERTAIN: Cannot determine (e.g., object not visible)

Be fair in evaluation. Focus on whether objects returned to their original positions.
"""

RESET_SYSTEM_PROMPT_RANDOM = """You are a Reset Task Judge for robotic manipulation tasks.

Your role is to evaluate whether a robot successfully MOVED objects to their designated RANDOM positions.

## Evaluation Process

You MUST complete all 5 steps and mark each step as PASS or FAIL.

**Step 1: Identify Objects Before Reset**
- Use the provided pixel coordinates to locate each object in Image 1
- Verify the objects are at their "current" positions
- Mark PASS if all objects are correctly identified

**Step 2: Identify Objects After Reset**
- Find the same objects in Image 2 (after reset)
- Note their new approximate pixel positions
- Mark PASS if all objects are visible and identifiable

**Step 3: Verify Target Position Reached (Per Object)**
- For EACH object, check if it moved to its random target position
- Compare the final position with the target position
- Mark PASS if object is at or near its target position

**Step 4: Check Placement Quality**
- Verify all objects are properly placed (not dropped, stable)
- Check objects are within the workspace
- Mark PASS if all objects are properly placed

**Step 5: Final Judgment**
- Count PASS/FAIL results from steps 1-4
- TRUE: All objects successfully moved to targets (Steps 3 and 4 PASS)
- FALSE: Any object failed to reach target
- UNCERTAIN: Cannot determine (e.g., object not visible)

Be lenient on exact positioning - approximate placement is acceptable.
"""


# ============================================================
# User Prompt Templates (Chain-of-Thought)
# ============================================================

RESET_USER_PROMPT_ORIGINAL = """## Reset Evaluation: RESTORE TO ORIGINAL

### 1. Goal
Restore objects to their **original positions** (before the forward task was executed).

### 2. Original Task (for context)
{original_instruction}

### 3. Image Information
- **Image Resolution**: {image_width} x {image_height} pixels
- **Image 1 (Before Reset)**: Objects at current positions (after forward task)
- **Image 2 (After Reset)**: Objects should be at original positions

### 4. Object Positions
{position_table}

### 5. Executed Reset Code
```python
{executed_code}
```

---

## Your Evaluation

Analyze each step and provide your findings in the format below.

### Output Format (You MUST follow this exact format)

**STEP 1 - Identify Objects Before Reset**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 1 Result: [PASS/FAIL] - [brief reason]

**STEP 2 - Identify Objects After Reset**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 2 Result: [PASS/FAIL] - [brief reason]

**STEP 3 - Verify Position Restoration (Per Object)**
- Object: [object name]
  - Current (before reset): ([x1], [y1])
  - Target (original): approximately ([x2], [y2])
  - Final (after reset): ([x3], [y3])
  - Restored: [YES/NO]
- Step 3 Result: [PASS/FAIL] - [Did all objects return to original positions?]

**STEP 4 - Check Scene Restoration**
- Scene similarity to original: [HIGH/MEDIUM/LOW]
- Objects dropped or misplaced: [YES/NO]
- Step 4 Result: [PASS/FAIL] - [brief reason]

**STEP 5 - Final Judgment**
- Steps passed: [list which steps passed]
- Steps failed: [list which steps failed, if any]
- Objects restored: [X/Y objects]

PREDICTION: [TRUE/FALSE/UNCERTAIN]

REASONING: [One sentence summary based on the step results]
"""


RESET_USER_PROMPT_RANDOM = """## Reset Evaluation: RANDOM SHUFFLE

### 1. Goal
Move objects to new **random target positions** within the workspace.

### 2. Image Information
- **Image Resolution**: {image_width} x {image_height} pixels
- **Image 1 (Before Reset)**: Objects at current positions
- **Image 2 (After Reset)**: Objects should be at random target positions

### 3. Object Positions
{position_table}

### 4. Executed Reset Code
```python
{executed_code}
```

---

## Your Evaluation

Analyze each step and provide your findings in the format below.

### Output Format (You MUST follow this exact format)

**STEP 1 - Identify Objects Before Reset**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 1 Result: [PASS/FAIL] - [brief reason]

**STEP 2 - Identify Objects After Reset**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 2 Result: [PASS/FAIL] - [brief reason]

**STEP 3 - Verify Target Position Reached (Per Object)**
- Object: [object name]
  - Current (before reset): ([x1], [y1])
  - Target (random): approximately ([x2], [y2])
  - Final (after reset): ([x3], [y3])
  - Reached target: [YES/NO]
- Step 3 Result: [PASS/FAIL] - [Did all objects reach their targets?]

**STEP 4 - Check Placement Quality**
- All objects visible: [YES/NO]
- Objects properly placed (not dropped): [YES/NO]
- Step 4 Result: [PASS/FAIL] - [brief reason]

**STEP 5 - Final Judgment**
- Steps passed: [list which steps passed]
- Steps failed: [list which steps failed, if any]
- Objects at target: [X/Y objects]

PREDICTION: [TRUE/FALSE/UNCERTAIN]

REASONING: [One sentence summary based on the step results]
"""


# ============================================================
# Helper Functions
# ============================================================

def _format_position(pos: Union[List[float], Tuple[float, float, float], Dict, None]) -> str:
    """위치를 문자열로 포맷 (world coordinates)"""
    if pos is None:
        return "Not detected"

    # Extended format (딕셔너리) 처리
    if isinstance(pos, dict):
        if "position" in pos:
            pos = pos["position"]
        else:
            return "Invalid format"

    return f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"


def _format_pixel_coords(pos: Union[Dict, None]) -> str:
    """픽셀 좌표를 문자열로 포맷"""
    if pos is None:
        return "N/A"

    if isinstance(pos, dict):
        pixel_coords = pos.get("pixel_coords")
        if pixel_coords:
            return f"({pixel_coords[0]}, {pixel_coords[1]})"

    return "N/A"


def _build_position_table(
    current_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    target_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    target_label: str = "Target",
) -> str:
    """위치 변화 테이블 생성 (픽셀 좌표 포함)"""
    lines = []

    # 메타데이터 키 제외
    current_objects = {k: v for k, v in current_positions.items() if not k.startswith("_")}
    target_objects = {k: v for k, v in target_positions.items() if not k.startswith("_")}
    all_objects = set(current_objects.keys()) | set(target_objects.keys())

    for obj_name in sorted(all_objects):
        current = current_objects.get(obj_name)
        target = target_objects.get(obj_name)

        current_world = _format_position(current)
        current_pixel = _format_pixel_coords(current)
        target_world = _format_position(target)
        target_pixel = _format_pixel_coords(target)

        lines.append(f"**{obj_name}**:")
        lines.append(f"  - Current: {current_world} (World) | Pixel: {current_pixel}")
        lines.append(f"  - {target_label}: {target_world} (World) | Pixel: {target_pixel}")
        lines.append("")

    return "\n".join(lines) if lines else "No objects detected"


# ============================================================
# Main Functions
# ============================================================

def build_reset_judge_prompt(
    reset_mode: str,
    current_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    target_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    executed_code: str,
    original_instruction: str = None,
    image_resolution: Tuple[int, int] = None,
) -> str:
    """
    Reset Judge 프롬프트 생성 (Chain-of-Thought 구조)

    Args:
        reset_mode: "original" 또는 "random"
        current_positions: Reset 전 물체 위치 (Forward 완료 후 상태)
            Extended format: {name: {"position": [x,y,z], "pixel_coords": (cx,cy), ...}}
        target_positions: 목표 위치
        executed_code: 실행된 Reset 코드
        original_instruction: 원래 태스크 명령어 (original 모드에서만 사용)
        image_resolution: 이미지 해상도 (width, height)

    Returns:
        포맷된 프롬프트 문자열
    """
    # 이미지 해상도 (기본값)
    if image_resolution is None:
        # current_positions에서 메타데이터 확인
        if "_image_resolution" in current_positions:
            image_resolution = current_positions["_image_resolution"]
        else:
            image_resolution = (640, 480)

    if reset_mode == "original":
        return _build_original_mode_prompt(
            current_positions=current_positions,
            target_positions=target_positions,
            executed_code=executed_code,
            original_instruction=original_instruction or "Not provided",
            image_resolution=image_resolution,
        )
    else:
        return _build_random_mode_prompt(
            current_positions=current_positions,
            target_positions=target_positions,
            executed_code=executed_code,
            image_resolution=image_resolution,
        )


def _build_original_mode_prompt(
    current_positions: Dict,
    target_positions: Dict,
    executed_code: str,
    original_instruction: str,
    image_resolution: Tuple[int, int],
) -> str:
    """Original 모드 프롬프트 생성"""
    position_table = _build_position_table(
        current_positions=current_positions,
        target_positions=target_positions,
        target_label="Original",
    )

    return RESET_USER_PROMPT_ORIGINAL.format(
        original_instruction=original_instruction,
        image_width=image_resolution[0],
        image_height=image_resolution[1],
        position_table=position_table,
        executed_code=executed_code,
    )


def _build_random_mode_prompt(
    current_positions: Dict,
    target_positions: Dict,
    executed_code: str,
    image_resolution: Tuple[int, int],
) -> str:
    """Random 모드 프롬프트 생성"""
    position_table = _build_position_table(
        current_positions=current_positions,
        target_positions=target_positions,
        target_label="Random Target",
    )

    return RESET_USER_PROMPT_RANDOM.format(
        image_width=image_resolution[0],
        image_height=image_resolution[1],
        position_table=position_table,
        executed_code=executed_code,
    )


def get_reset_system_prompt(reset_mode: str) -> str:
    """모드에 따른 시스템 프롬프트 반환"""
    if reset_mode == "original":
        return RESET_SYSTEM_PROMPT_ORIGINAL
    else:
        return RESET_SYSTEM_PROMPT_RANDOM


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    # 테스트 데이터 (extended format with pixel coords)
    test_current = {
        "_image_resolution": (640, 480),
        "green block": {
            "position": [0.28, -0.12, 0.05],
            "pixel_coords": (420, 300),
            "bbox_pixels": (390, 270, 450, 330),
        },
        "blue dish": {
            "position": [0.28, -0.12, 0.05],
            "pixel_coords": (420, 310),
            "bbox_pixels": (380, 280, 460, 340),
        },
    }
    test_target_original = {
        "green block": {
            "position": [0.11, 0.10, 0.05],
            "pixel_coords": (280, 320),
        },
        "blue dish": {
            "position": [0.27, -0.16, 0.05],
            "pixel_coords": (450, 290),
        },
    }
    test_target_random = {
        "green block": {
            "position": [0.15, 0.00, 0.05],
            "pixel_coords": (320, 300),
        },
        "blue dish": {
            "position": [0.20, 0.05, 0.05],
            "pixel_coords": (380, 280),
        },
    }
    test_code = """
skills.move_to_position([0.28, -0.12, 0.15])
skills.gripper_close()
skills.move_to_position([0.11, 0.10, 0.05])
skills.gripper_open()
"""
    test_instruction = "pick up the green block and place it on the blue dish"

    print("=" * 70)
    print("ORIGINAL MODE (Chain-of-Thought)")
    print("=" * 70)
    print("\n[System Prompt]")
    print(get_reset_system_prompt("original"))
    print("\n[User Prompt]")
    print(build_reset_judge_prompt(
        reset_mode="original",
        current_positions=test_current,
        target_positions=test_target_original,
        executed_code=test_code,
        original_instruction=test_instruction,
    ))

    print("\n" + "=" * 70)
    print("RANDOM MODE (Chain-of-Thought)")
    print("=" * 70)
    print("\n[System Prompt]")
    print(get_reset_system_prompt("random"))
    print("\n[User Prompt]")
    print(build_reset_judge_prompt(
        reset_mode="random",
        current_positions=test_current,
        target_positions=test_target_random,
        executed_code=test_code,
    ))
