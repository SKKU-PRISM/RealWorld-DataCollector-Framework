"""
Judge Prompt Template

태스크 완료 여부를 판단하는 VLM 프롬프트 템플릿
Chain-of-Thought 방식으로 단계별 추론을 유도합니다.
"""

from typing import Dict, List, Optional, Tuple, Union


JUDGE_SYSTEM_PROMPT = """You are a Task Completion Judge for robotic manipulation tasks.

You will evaluate whether a robot successfully completed a given task by comparing the initial and final images of the workspace.

## Evaluation Process

You MUST complete all 5 steps and mark each step as PASS or FAIL.

**Step 1: Identify Objects in Initial Image**
- Locate all task-relevant objects in the initial image
- Note each object's position and appearance
- Mark PASS if all objects are correctly identified

**Step 2: Identify Objects in Final Image**
- Find the same objects in the final image
- Note their new positions and any changes
- Mark PASS if all objects are visible and identifiable

**Step 3: Analyze Changes**
- Compare initial vs final state for ALL task-relevant objects
- Describe what moved, what stayed, and how the scene changed
- Mark PASS if the observed changes are consistent with an attempt to complete the task

**Step 4: Verify Goal Achievement**
- Re-read the goal instruction carefully
- Check whether the final state satisfies the instruction
- Consider ALL aspects of the instruction (order, arrangement, placement, etc.)
- Mark PASS if the final state matches what the instruction requires

**Step 5: Final Judgment**
- TRUE: Step 4 PASS (the task goal is achieved in the final image)
- FALSE: Step 4 FAIL (the task goal is clearly not achieved)
- UNCERTAIN: Cannot determine (e.g., objects not visible, ambiguous result)

Be fair in evaluation. Focus on whether the core task objective was achieved, not on minor imperfections.
"""


JUDGE_USER_PROMPT_TEMPLATE = """## Task Evaluation Request

### 1. Goal Instruction
{instruction}

### 2. Image Information
- **Image 1 (Initial State)**: Before task execution
- **Image 2 (Final State)**: After task execution

### 3. Task-Relevant Objects
{object_names_text}

---

## Your Evaluation

Analyze each step and provide your findings in the format below.

### Output Format (You MUST follow this exact format)

**STEP 1 - Identify Objects in Initial Image**
- Object: [object name]
  - Position: [approximate location in the image]
  - Description: [what you see]
- [Repeat for each object]
- Step 1 Result: [PASS/FAIL] - [brief reason]

**STEP 2 - Identify Objects in Final Image**
- Object: [object name]
  - Position: [approximate location in the image]
  - Description: [what you see]
- [Repeat for each object]
- Step 2 Result: [PASS/FAIL] - [brief reason]

**STEP 3 - Analyze Changes**
- Object: [object name]
  - Change: [moved from A to B / stayed in place / etc.]
- [Repeat for each object]
- Step 3 Result: [PASS/FAIL] - [Are the changes consistent with the task?]

**STEP 4 - Verify Goal Achievement**
- [Evaluate each aspect of the instruction]
- Step 4 Result: [PASS/FAIL] - [brief reason]

**STEP 5 - Final Judgment**
- Steps passed: [list which steps passed]
- Steps failed: [list which steps failed, if any]

PREDICTION: [TRUE/FALSE/UNCERTAIN]

REASONING: [One sentence summary of why you made this prediction based on the step results]
"""


def build_judge_prompt(
    instruction: str,
    object_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    executed_code: str,
    image_resolution: Tuple[int, int] = None,
) -> str:
    """
    Judge 프롬프트 생성 (Chain-of-Thought 구조)

    Args:
        instruction: 목표 명령어
        object_positions: 객체 위치 딕셔너리
            Extended format: {
                name: {
                    "position": [x, y, z],
                    "pixel_coords": (cx, cy),
                    "bbox_pixels": (x1, y1, x2, y2),
                    ...
                }
            }
        executed_code: 실행된 Python 코드
        image_resolution: 이미지 해상도 (width, height)

    Returns:
        포맷된 프롬프트 문자열
    """
    # 객체 이름 목록 생성 (좌표는 제외 — Judge는 이미지로 판단)
    object_names = [
        f"- {name}" for name in object_positions.keys()
        if not name.startswith("_") and object_positions[name] is not None
    ]
    object_names_text = "\n".join(object_names) if object_names else "No objects detected"

    # 프롬프트 생성
    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        object_names_text=object_names_text,
    )

    return prompt


def get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    return JUDGE_SYSTEM_PROMPT


# 테스트
if __name__ == "__main__":
    # 테스트 데이터 (extended format with pixel coords)
    test_instruction = "pick up the green block and place it on the blue dish"
    test_positions = {
        "_image_resolution": (640, 480),
        "green block": {
            "position": [0.1500, 0.0500, 0.0200],
            "pixel_coords": (280, 320),
            "bbox_pixels": (250, 290, 310, 350),
            "confidence": 0.92,
        },
        "blue dish": {
            "position": [0.2000, -0.0300, 0.0100],
            "pixel_coords": (400, 280),
            "bbox_pixels": (350, 230, 450, 330),
            "confidence": 0.88,
        },
    }
    test_code = """
from skills.skills_lerobot import LeRobotSkills

skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
skills.connect()

# Pick up green block
skills.execute_pick_object([0.15, 0.05, 0.02])

# Place on blue dish
skills.execute_place_object([0.20, -0.03, 0.03])

skills.disconnect()
"""

    prompt = build_judge_prompt(
        test_instruction,
        test_positions,
        test_code,
        image_resolution=(640, 480),
    )

    print("=" * 70)
    print("System Prompt:")
    print("=" * 70)
    print(get_system_prompt())
    print("\n" + "=" * 70)
    print("User Prompt:")
    print("=" * 70)
    print(prompt)
