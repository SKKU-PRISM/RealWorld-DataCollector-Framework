from typing import Dict, List, Optional


def lerobot_spec_gen_prompt(
    instruction: str,
    object_positions: Dict[str, List[float]],
) -> str:
    """
    LeRobot SO-101용 태스크 스펙 생성 프롬프트

    자연어 목표를 구조화된 스펙(단계별 액션)으로 변환합니다.
    이 스펙은 이후 code_gen에서 실제 Python 코드로 변환됩니다.

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_positions: 객체별 위치 딕셔너리 {name: [x, y, z]}

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 객체 위치 포맷팅
    positions_str = "\n".join([
        f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
        for name, pos in object_positions.items()
        if pos is not None
    ])

    # 검출 실패한 객체
    not_found = [name for name, pos in object_positions.items() if pos is None]
    not_found_str = ", ".join(not_found) if not_found else "None"
    # assert not_found is not None f"{not_found_str} 객체의 위치가 제공되지 않았습니다."

    # 객체 이름 리스트
    object_names = list(object_positions.keys())

    prompt = f"""
You are an AI assistant that generates structured task specifications for a LeRobot SO-101 robot arm.
Your task is to analyze the goal and break it down into a sequence of skill actions.

### **Input Details:**

1. **Goal (Natural Language)**:
   What the robot must accomplish.
   ```
   {instruction}
   ```

2. **Detected Objects and Positions (World Frame, meters)**:
   ```
    {positions_str}
   ```

3. **Available Primitive Skills**:
   | Skill | Description | Parameters |
   |-------|-------------|------------|
   | `move_to_position` | Move end-effector to object position (fixed orientation, no rotation) | object: str |
   | `gripper_open` | Open the gripper | - |
   | `gripper_close` | Close the gripper | - |
   | `move_to_initial_state` | Move to home position | - |
   | `move_to_free_state` | Move to safe parking position | - |
   | `rotate_90degree` | Rotate gripper 90° | direction: "cw" or "ccw" |

   **Note**: `move_to_position` maintains current wrist_roll during movement.
   The gripper will NOT rotate during position changes (keeps orientation from rotate_90degree).

   **Note**: SO-101 has asymmetric gripper (fixed finger at +Y, moving at -Y).

4. **Available Object Names**:
   {object_names}

### **Task**

1. Analyze the goal and determine the required sequence of primitive actions
2. Break down the task into individual primitive skill steps
3. Map each step to an available primitive skill with appropriate parameters
4. Use ONLY the detected object names as parameters


### **Output Format**

Generate a JSON specification using **primitive skills only**.

Specification: {{"required_skills": ["<skill1>", "<skill2>", ...], "steps": [{{"step": 1, "action": "<skill_name>", "object": "<object_name>"}}, ...]}}

### **Examples**
Primitive Skill Composition Patterns: 
1. Use these example patterns to compose complex actions from primitives.
2. When you start New apisode, always begin with `move_to_initial_state` and end with `move_to_initial_state` and `move_to_free_state`.

**Example 1**: "Pick up the red cup and place it on the blue box"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "gripper_open", "gripper_close", "move_to_position", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "gripper_open"}},
    {{"step": 3, "action": "move_to_position", "object": "red cup"}},
    {{"step": 4, "action": "gripper_close"}},
    {{"step": 5, "action": "move_to_position", "target": "blue box"}},
    {{"step": 6, "action": "gripper_open"}},
    {{"step": 7, "action": "move_to_initial_state"}},
    {{"step": 8, "action": "move_to_free_state"}}
  ]
}}
```

**Example 2**: "Move the yellow dice to the drawer"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "gripper_open", "gripper_close", "move_to_position", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "gripper_open"}},
    {{"step": 3, "action": "move_to_position", "object": "yellow dice"}},
    {{"step": 4, "action": "gripper_close"}},
    {{"step": 5, "action": "move_to_position", "target": "drawer"}},
    {{"step": 6, "action": "gripper_open"}},
    {{"step": 7, "action": "move_to_initial_state"}},
    {{"step": 8, "action": "move_to_free_state"}}
  ]
}}
```

**Example 3**: "Pick up red dice and green dice, stack them on the plate"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "gripper_open", "gripper_close", "move_to_position", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "gripper_open"}},
    {{"step": 3, "action": "move_to_position", "object": "red dice"}},
    {{"step": 4, "action": "gripper_close"}},
    {{"step": 5, "action": "move_to_position", "target": "plate"}},
    {{"step": 6, "action": "gripper_open"}},
    {{"step": 7, "action": "move_to_position", "object": "green dice"}},
    {{"step": 8, "action": "gripper_close"}},
    {{"step": 9, "action": "move_to_position", "target": "plate"}},
    {{"step": 10, "action": "gripper_open"}},
    {{"step": 11, "action": "move_to_initial_state"}},
    {{"step": 12, "action": "move_to_free_state"}}
  ]
}}
```

### **Guidelines**

1. Use ONLY primitive skills (no execute_pick, execute_place, execute_pick_and_place)
2. Follow the PICK and PLACE patterns for grasping and releasing objects
3. Always start with `move_to_initial_state`
4. Always end with `move_to_initial_state` and `move_to_free_state`
5. Use ONLY the object names from the detected objects list
7. If an object is not found, skip actions involving that object
8. Always ensure logical ordering (pick before place)
9. Do not use code blocks in output - provide plain text specification

### **Generate Specification:**
"""

    return prompt