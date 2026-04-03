"""
shake — 흔들기 스킬 (modular)

물체를 잡은 상태에서 빠르게 왕복 운동하여 흔드는 core shake skill.
병 흔들기, 스프레이 흔들기, 내용물 섞기 등에 사용.

**전제 조건**: 물체가 이미 잡혀 있어야 함 (LLM 생성 코드에서 pick 처리).
이 스킬은 오직 shake_height로 들어올리기 → 왕복 진동 → 센터 복귀만 수행.

동작 원리:
    물체를 잡은 채 shake_height로 들어올린 후
    shake_axis 방향으로 shake_amplitude만큼 빠르게 왕복 진동.

동작 시퀀스:
    1. move_to_position  — shake_height로 들어올리기
    2. move_to_position × N — 왕복 진동 (num_shakes회)
    3. move_to_position  — 센터로 복귀 (shake_height)

    shake 동작 (측면에서 본 모습):
        shake_height에서 좌우(또는 상하) 왕복
        ←amp→   ←amp→
        ●───────●───────●───────●  ...
         (shake 1) (shake 2)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.shake import shake

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    bottle_pos = [0.20, 0.0, 0.03]

    # LLM generated code handles approach & grasp:
    move_approach_position(skills, object_position=bottle_pos, approach_height=0.10, ...)
    skills.execute_pick_object(bottle_pos, ...)
    skills.move_to_position([bottle_pos[0], bottle_pos[1], 0.15], ...)  # lift after pick

    # Core shake action (lifts to shake_height internally, shakes, returns to center):
    shake(skills, position=bottle_pos, shake_axis="y", shake_amplitude=0.03, num_shakes=6, shake_height=0.10, ...)

    # LLM generated code handles place & release:
    skills.move_to_position([bottle_pos[0], bottle_pos[1], bottle_pos[2]], ...)  # lower to place
    skills.gripper_open(...)
    skills.move_to_position([bottle_pos[0], bottle_pos[1], 0.10], ...)  # retreat

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np


def shake(
    skills,
    position: Union[List[float], np.ndarray],
    shake_axis: Union[List[float], str] = "y",
    shake_amplitude: float = 0.03,
    num_shakes: int = 6,
    shake_height: float = 0.10,
    shake_speed: Optional[float] = None,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    물체를 잡은 상태에서 빠르게 왕복 흔들기 (core shake action).

    전제 조건: 물체가 이미 잡혀 있어야 함 (LLM 코드에서 pick 처리).
    shake_height로 들어올린 후, shake_axis 방향으로 shake_amplitude만큼
    num_shakes회 왕복. 흔들기 완료 후 센터(shake_height)에서 대기.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 물체 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - x, y는 흔들기 중심 좌표로 사용
                  - 예: [0.20, 0.0, 0.03] → 병 위치

        shake_axis: 흔드는 방향. default="y"
                    - "x": x축 방향 왕복 (앞뒤)
                    - "y": y축 방향 왕복 (좌우)
                    - "z": z축 방향 왕복 (위아래)
                    - [dx, dy]: 임의 xy 방향 벡터 (자동 정규화)
                    - 예: [1, 1] → 대각선 방향

        shake_amplitude: 흔들기 진폭 (단위: meters). default=0.03 (3cm)
                         - 중심에서 한쪽 방향 최대 변위
                         - 전체 왕복 거리 = 2 × amplitude
                         - 0.02: 작은 흔들기 (소스통)
                         - 0.03: 보통 흔들기 (병)
                         - 0.05: 큰 흔들기 (페인트통)

        num_shakes: 편도 횟수 (단위: 회). default=6
                    - 1: 편도 1회 (center → +amp)
                    - 2: 1왕복 (center → +amp → -amp)
                    - 6: 3왕복 (보통)
                    - 짝수 권장 (원래 위치로 복귀)

        shake_height: 흔들기 시 EE 높이 (단위: meters). default=0.10 (10cm)
                      - 물체를 들어올린 높이에서 흔들기
                      - 테이블 위 충분한 높이로 설정

        shake_speed: 편도 1회 이동 시간 (단위: 초). default=None
                     - None: 0.3초 (빠른 왕복)
                     - 작을수록 빠르게 흔듬
                     - 0.2: 매우 빠르게
                     - 0.3: 보통
                     - 0.5: 천천히

        duration: lift 이동 시간 (단위: 초). default=None

        object_name: 흔들 대상 이름 (선택). default=None
                     - 예: "bottle", "spray can"

        skill_description: 스킬 동작 설명 (선택). default=None

    Returns:
        bool: True면 흔들기 성공, False면 실패
    """
    pos = np.array(position, dtype=float)

    # shake_axis 파싱
    if isinstance(shake_axis, str):
        if shake_axis == "x":
            axis_vec = np.array([1.0, 0.0, 0.0])
        elif shake_axis == "y":
            axis_vec = np.array([0.0, 1.0, 0.0])
        elif shake_axis == "z":
            axis_vec = np.array([0.0, 0.0, 1.0])
        else:
            skills._log(f"ERROR: Unknown shake_axis string: {shake_axis}")
            return False
    else:
        # [dx, dy] 벡터 → 3D로 확장
        ax = np.array(shake_axis, dtype=float)
        if len(ax) == 2:
            axis_vec = np.array([ax[0], ax[1], 0.0])
        else:
            axis_vec = ax

    # 정규화
    ax_norm = np.linalg.norm(axis_vec)
    if ax_norm < 1e-6:
        skills._log("ERROR: shake_axis is zero vector")
        return False
    axis_vec = axis_vec / ax_norm

    # shake 속도
    if shake_speed is None:
        shake_speed = 0.3

    desc_prefix = f"shake {object_name}" if object_name else "shake"

    axis_name = (
        shake_axis
        if isinstance(shake_axis, str)
        else f"[{axis_vec[0]:.1f},{axis_vec[1]:.1f},{axis_vec[2]:.1f}]"
    )

    skills._log(f"\n{'='*60}")
    skills._log(f"[shake] {desc_prefix}")
    skills._log(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Axis: {axis_name}, Amplitude: {shake_amplitude*100:.1f}cm")
    skills._log(f"  Shakes: {num_shakes}, Speed: {shake_speed:.2f}s/stroke")
    skills._log(f"  Shake height: {shake_height*100:.1f}cm")
    skills._log(f"{'='*60}")

    # 흔들기 중심 위치
    shake_center = np.array([pos[0], pos[1], shake_height])

    # Step 1. Lift to shake height
    skills._log(f"\n[Step 1] Lift to shake height ({shake_height*100:.1f}cm)")
    if not skills.move_to_position(
        position=shake_center.tolist(),
        duration=duration,

        target_name=object_name,
        skill_description=f"{desc_prefix}: lift to shake height",
    ):
        skills._log("ERROR: Failed to lift to shake height")
        return False

    # Step 2. Shake: 왕복 진동
    skills._log(f"\n[Step 2] Shaking ({num_shakes} strokes)")
    all_success = True

    for i in range(num_shakes):
        stroke_num = i + 1
        # 홀수: +방향, 짝수: -방향
        if i % 2 == 0:
            target = shake_center + axis_vec * shake_amplitude
            direction = "+"
        else:
            target = shake_center - axis_vec * shake_amplitude
            direction = "-"

        skills._log(f"  Shake {stroke_num}/{num_shakes} ({direction})")
        success = skills.move_to_position(
            position=target.tolist(),
            duration=shake_speed,
    
            target_name=object_name,
            skill_description=(
                skill_description
                or f"{desc_prefix}: shake {stroke_num} ({direction})"
            ),
        )
        if not success:
            all_success = False

    # Step 3. Return to center
    skills._log("\n[Step 3] Return to center")
    skills.move_to_position(
        position=shake_center.tolist(),
        duration=shake_speed,

        target_name=object_name,
        skill_description=f"{desc_prefix}: return to center",
    )

    skills._log(f"\n[shake] Complete (all_success={all_success})")
    return all_success


if __name__ == "__main__":
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        exit(1)

    try:
        skills.move_to_initial_state()

        bottle_pos = [0.20, 0.0, 0.03]

        # 1. Approach & pick (simulates LLM-generated code)
        print("\n=== Pre-shake: approach & pick (LLM code) ===")
        skills.gripper_open(skill_description="test: open gripper")
        skills.move_to_position(
            position=[bottle_pos[0], bottle_pos[1], 0.12],
            target_name="test bottle",
            skill_description="test: approach above",
        )
        skills.execute_pick_object(
            bottle_pos,
            target_name="test bottle",
            skill_description="test: pick bottle",
        )
        skills.move_to_position(
            position=[bottle_pos[0], bottle_pos[1], 0.15],
            target_name="test bottle",
            skill_description="test: lift after pick",
        )

        # 2. Core shake action
        print("\n=== Test: Shake along y-axis (6 strokes) ===")
        success = shake(
            skills,
            position=bottle_pos,
            shake_axis="y",
            shake_amplitude=0.03,
            num_shakes=6,
            shake_height=0.10,
            object_name="test bottle",
            skill_description="test: shake bottle",
        )
        print(f"Shake result: {'success' if success else 'failed'}")

        # 3. Place & release (simulates LLM-generated code)
        print("\n=== Post-shake: place & release (LLM code) ===")
        skills.move_to_position(
            position=[bottle_pos[0], bottle_pos[1], bottle_pos[2]],
            target_name="test bottle",
            skill_description="test: lower to place",
        )
        skills.gripper_open(skill_description="test: release bottle")
        skills.move_to_position(
            position=[bottle_pos[0], bottle_pos[1], 0.10],
            target_name="test bottle",
            skill_description="test: retreat",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
