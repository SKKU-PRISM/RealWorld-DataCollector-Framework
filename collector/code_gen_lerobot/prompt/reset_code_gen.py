from pathlib import Path
from typing import Dict, List, Optional


def _get_frame_for_robot(robot_id: int) -> str:
    """pix2robot 캘리브레이션이 있으면 base_link, 없으면 world."""
    pix2robot_path = (
        Path(__file__).parent.parent.parent
        / "robot_configs" / "pix2robot_matrices"
        / f"robot{robot_id}_pix2robot_data.npz"
    )
    return "base_link" if pix2robot_path.exists() else "world"


def lerobot_code_gen_prompt(
    instruction: str,
    object_positions: Dict[str, List[float]],
    spec: str = None,
    robot_id: int = 3,
) -> str:
    """
    LeRobot SO-101용 코드 생성 프롬프트

    Franka realworld_code_gen_prompt 양식을 기반으로 LeRobot에 맞게 단순화

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_positions: 객체별 위치 딕셔너리 {name: [x, y, z]}
        spec: 코드 생성 가이드라인/스펙 (선택)
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 로봇 설정 파일 경로
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)

    # 객체 위치 포맷팅
    positions_str = "\n".join([
        f'      "{name}": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}],'
        for name, pos in object_positions.items()
        if pos is not None
    ])

    # 검출 실패한 객체
    not_found = [name for name, pos in object_positions.items() if pos is None]
    not_found_str = ", ".join(not_found) if not_found else "None"

    prompt = f"""You are an AI assistant tasked with generating executable Python code that controls a LeRobot SO-101 robot arm.
    The goal is to ensure the robot can achieve the specified objective by executing a sequence of **primitive** skill actions.

### **Input Details:**

    1. **Goal (Natural Language Description)**:
       The goal specifies what the robot must accomplish, described in plain language.

       Provided Goal Description:
       ```
       {instruction}
       ```

    2. **Object Positions (World Coordinate Frame, unit: meters)**:
       The detected target object positions in the environment.
       ```python
       positions = {{
      {positions_str}
       }}
       ```
       Objects not found: {not_found_str}

    3. **Specification (Code Generation Guidelines)**:
       The specification outlines the exact requirements and constraints for generating the executable Python code.
       ```
       {spec}
       ```

    4. **Available Primitive Skills**:
       The LeRobotSkills class provides these **primitive** methods only:

       | Method | Description | Parameters |
       |--------|-------------|------------|
       | `connect()` | Connect to robot | - |
       | `disconnect()` | Disconnect from robot | - |
       | `gripper_open()` | Open gripper | - |
       | `gripper_close()` | Close gripper | - |
       | `move_to_initial_state()` | Move to initial/home position | - |
       | `move_to_free_state()` | Move to safe parking position | - |
       | `move_to_position(position, maintain_pitch=False)` | Move end-effector to [x,y,z] with fixed orientation | position: List[float], maintain_pitch: bool |
       | `rotate_90degree(direction)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW) |

       **Note**: `move_to_position` maintains current wrist_roll during movement.
       The gripper will NOT rotate during position changes (keeps orientation from rotate_90degree).

       **Maintain Pitch**: Use `maintain_pitch=True` when moving while holding an object (between gripper_close and gripper_open).

    5. **Primitive Skill Composition Patterns**:
       Use these patterns to compose pick and place actions from primitives:

       ```
       # PICK pattern (to grasp an object at position):
       approach_height = 0.08  # 8cm above object

       # positions is a dict: {{name: {{"position": [x,y,z], ...}}}}
       pick_obj = positions["object_name"]
       pick_pos = pick_obj["position"]
       pick_approach = [pick_pos[0], pick_pos[1], pick_pos[2] + approach_height]

       skills.gripper_open()
       skills.move_to_position(pick_approach)    # approach
       skills.move_to_position(pick_pos)          # descend
       skills.gripper_close()                      # grasp object
       skills.move_to_position(pick_approach, maintain_pitch=True)  # lift object (maintain pitch)

       # PLACE pattern (to release object at target position):
       # Use maintain_pitch=True while holding object
       place_obj = positions["target_name"]
       place_pos = place_obj["position"]
       place_approach = [place_pos[0], place_pos[1], place_pos[2] + approach_height]

       skills.move_to_position(place_approach, maintain_pitch=True)  # approach (holding object)
       skills.move_to_position(place_pos, maintain_pitch=True)       # descend to target
       skills.gripper_open()                                          # release object
       skills.move_to_position(place_approach)                        # retract upward
       ```

    6. **Executable Code Skeleton**:
       A template showing how to structure the code using **primitives only**.

       ```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Execute the robot task based on the goal using primitive skills.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="{frame}",
    )
    skills.connect()

    try:
        # === Setup ===
        approach_height = 0.08  # 8cm above objects

        # Move to initial state
        skills.move_to_initial_state()

        # === Object Positions ===
        # positions is a dict: {{name: {{"position": [x,y,z], ...}}}}
        pick_obj = positions["object_name"]
        pick_pos = pick_obj["position"]

        place_obj = positions["target_name"]
        place_pos = place_obj["position"]

        # === PICK sequence ===
        pick_approach = [pick_pos[0], pick_pos[1], pick_pos[2] + approach_height]
        skills.gripper_open()
        skills.move_to_position(pick_approach)
        skills.move_to_position(pick_pos)
        skills.gripper_close()
        skills.move_to_position(pick_approach, maintain_pitch=True)

        # === PLACE sequence (maintain_pitch while holding object) ===
        place_approach = [place_pos[0], place_pos[1], place_pos[2] + approach_height]
        skills.move_to_position(place_approach, maintain_pitch=True)
        skills.move_to_position(place_pos, maintain_pitch=True)
        skills.gripper_open()
        skills.move_to_position(place_approach)

        # === Cleanup ===
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
       ```

    7. **Available Primitive Skill Names**:
       ```
       ['connect', 'disconnect', 'gripper_open', 'gripper_close',
        'move_to_initial_state', 'move_to_free_state', 'move_to_position',
        'rotate_90degree']
       ```

### **Task Requirements:**

1. **Complete the Executable Code:**
   - Fill in the primitive skill sequence to accomplish the goal
   - Use ONLY the primitive skills listed above (no execute_pick, execute_place, etc.)
   - Ensure the code is executable as-is

2. **Guidelines for Implementation:**
   - Use ONLY primitive skills from LeRobotSkills
   - Compose PICK and PLACE sequences manually using the patterns above
   - Always start with `move_to_initial_state()`
   - Always end with `move_to_initial_state()` and `move_to_free_state()`
   - Use `approach_height = 0.08` (8cm) for approach/retract movements
   - Use the exact object positions provided (do not invent positions)
   - Always include try/finally for proper cleanup
   - Call `connect()` at start and `disconnect()` at end

3. **Output Format:**
   - Do not use code blocks in your final answer
   - Provide the generated code in plain text format
   - Include all imports and the complete execute_task() function

**Generate the complete executable Python code:**
"""

    return prompt