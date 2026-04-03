"""
push_object — 밀기 스킬

물체에 접촉하여 직선으로 미는 스킬.
내부적으로 move_linear를 핵심 동작으로 사용.

전제 조건 (LLM 코드에서 외부 처리):
    - gripper가 이미 닫혀 있어야 함 (gripper_close)
    - EE가 이미 start 위 approach_height에 있어야 함 (move_to_position)

동작 시퀀스:
    1. move_to_position  — pre-contact로 하강 (start에서 push 반대방향으로 run_up만큼 뒤)
    2. move_linear       — pre-contact → end 직선 (run-up + 밀기를 한 번에)
    3. move_to_position  — end에서 push 방향으로 run_up만큼 이탈 (물체에서 벗어남)
    4. move_to_position  — clear 위치에서 approach_height로 상승 (복귀)

    (사전 조건: gripper_close + move_to_position approach)

                    approach_height (외부)
    ─────●                                          ●
         │                                          │ (Step 4: 상승)
         │                                          │
         ●────→──●─────────────────────────→──●──→──●  push_height
    pre-contact  start(interaction)           end  clear
    (Step 1:하강) (Step 2: move_linear)      (Step 3: 이탈)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.push_object import push_object

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    start_pos = [0.20, -0.05, 0.03]
    end_pos = [0.20, 0.05, 0.03]

    # 1. 외부: gripper 닫기
    skills.gripper_close(skill_description="push red block: close gripper")

    # 2. 외부: start 위 approach 위치로 이동
    skills.move_to_position([start_pos[0], start_pos[1], 0.20],
                            skill_description="push red block: approach above start")

    # 3. Push (하강 + run-up + 밀기 + 복귀 포함)
    push_object(
        skills,
        start_position=start_pos,
        end_position=end_pos,
        push_height=0.01,
        object_name="red block",
    )

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def push_object(
    skills,
    start_position: Union[List[float], np.ndarray],
    end_position: Union[List[float], np.ndarray],
    push_height: float = 0.01,
    run_up_distance: float = 0.04,
    approach_height: float = 0.20,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    물체를 접촉하여 직선으로 밀기.

    EE가 이미 start 위 approach_height에 있고 gripper가 닫혀 있다고 가정.
    pre-contact로 하강 → run-up + 밀기 직선 이동 → approach_height로 복귀.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        start_position: 물체 접촉 위치 [x, y, z] (단위: meters)
                        - interaction point (예: 물체 왼쪽 가장자리)
                        - z값은 물체 높이 참조용 (실제 밀기 높이는 push_height)

        end_position: 밀기 끝 위치 [x, y, z] (단위: meters)
                      - xy 평면에서 밀기 방향과 거리를 결정
                      - z값은 사용되지 않음 (push_height가 우선)

        push_height: 밀기 시 EE 높이 (단위: meters). default=0.01 (1cm)
                     - 물체 높이의 1/3~1/2 권장 (예: 3cm 물체 → push_height=0.01)

        run_up_distance: pre-contact offset 거리 (단위: meters). default=0.03 (3cm)
                         - start에서 push 반대 방향으로 이 거리만큼 뒤로 뺀 위치에서 하강
                         - run-up 구간을 통해 물체 접촉 전 안정적 이동 확보

        approach_height: 복귀 높이 (단위: meters). default=0.20 (20cm)
                         - Step 3에서 end 위 이 높이로 상승

        duration: 밀기 구간 이동 시간 (단위: 초). default=None
                  - None: 거리 기반 자동 계산 (5cm/s, 최소 2초)

        object_name: 밀 대상 물체 이름 (선택). default=None

        skill_description: 스킬 동작 설명 (선택). default=None

    Returns:
        bool: True면 밀기 동작 성공, False면 실패
    """
    start_pos = np.array(start_position, dtype=float)
    end_pos = np.array(end_position, dtype=float)

    # push_height 최소값 보장 (IK 안정성)
    MIN_PUSH_HEIGHT = 0.01  # 1cm
    if push_height < MIN_PUSH_HEIGHT:
        skills._log(f"WARNING: push_height {push_height*100:.1f}mm < {MIN_PUSH_HEIGHT*100:.0f}mm, "
                     f"clamping to {MIN_PUSH_HEIGHT*100:.0f}mm")
        push_height = MIN_PUSH_HEIGHT

    # push 방향 벡터 계산
    push_vec = end_pos[:2] - start_pos[:2]
    push_distance = np.linalg.norm(push_vec)
    if push_distance < 1e-6:
        skills._log("ERROR: start and end positions are the same")
        return False
    push_dir = push_vec / push_distance

    # pre-contact: start에서 push 반대방향으로 run_up_distance만큼 뒤로
    pre_contact = start_pos[:2] - push_dir * run_up_distance

    desc_prefix = f"push {object_name}" if object_name else "push object"

    skills._log(f"\n{'='*60}")
    skills._log(f"[push_object] {desc_prefix}")
    skills._log(f"  Start (contact): [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    skills._log(f"  End:             [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    skills._log(f"  Pre-contact:     [{pre_contact[0]:.3f}, {pre_contact[1]:.3f}]")
    skills._log(f"  Push height: {push_height*100:.1f}cm, run-up: {run_up_distance*100:.1f}cm, distance: {push_distance*100:.1f}cm")
    skills._log(f"{'='*60}")

    # Step 1: pre-contact 위치로 하강
    skills._log("\n[Step 1] Descend to pre-contact position")
    if not skills.move_to_position(
        position=[pre_contact[0], pre_contact[1], push_height],

        maintain_pitch=False,
        target_name=object_name,
        skill_description=f"{desc_prefix}: descend to pre-contact",
    ):
        skills._log("ERROR: Failed to descend to pre-contact position")
        return False

    # Step 2: pre-contact → end 직선 밀기 (run-up + push 한 번에)
    # 기본 속도: 3cm/s (duration 미지정 시)
    total_linear_dist = np.linalg.norm(end_pos[:2] - pre_contact)
    if duration is None:
        duration = max(total_linear_dist / 0.05, 2.0)  # 5cm/s, 최소 2초

    skills._log("\n[Step 2] Linear push (pre-contact → end)")
    push_success = move_linear(
        skills,
        start=[pre_contact[0], pre_contact[1], push_height],
        end=[end_pos[0], end_pos[1], push_height],
        duration=duration,

        maintain_pitch=True,
        target_name=object_name,
        skill_description=skill_description or f"{desc_prefix}: pushing",
    )

    if not push_success:
        skills._log("WARNING: Push move_linear did not fully converge")

    # Step 3: push 반대 방향으로 run_up만큼 되돌아가서 물체에서 이탈
    retract_pos = end_pos[:2] - push_dir * run_up_distance
    skills._log(f"\n[Step 3] Retract from object (opposite push direction -{run_up_distance*100:.0f}cm)")
    skills.move_to_position(
        position=[retract_pos[0], retract_pos[1], push_height],

        maintain_pitch=False,
        target_name=object_name,
        skill_description=f"{desc_prefix}: retract away from object",
    )

    # Step 4: approach_height로 상승 (복귀)
    skills._log(f"\n[Step 4] Retreat to approach height ({approach_height*100:.0f}cm)")
    skills.move_to_position(
        position=[retract_pos[0], retract_pos[1], approach_height],

        maintain_pitch=False,
        target_name=object_name,
        skill_description=f"{desc_prefix}: retreat after push",
    )

    skills._log(f"\n[push_object] Complete (push_success={push_success})")
    return push_success


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

        start_pos = [0.20, -0.05, 0.03]
        end_pos = [0.20, 0.05, 0.03]

        # 외부: gripper 닫기
        skills.gripper_close(skill_description="test: close gripper for push")

        # 외부: start 위 approach 위치로 이동
        skills.move_to_position(
            [start_pos[0], start_pos[1], 0.20],
            skill_description="test: approach above start",
        )

        # Push (하강 + run-up + 밀기 + 복귀 포함)
        success = push_object(
            skills,
            start_position=start_pos,
            end_position=end_pos,
            push_height=0.01,
            object_name="test block",
            skill_description="test: push block along y-axis",
        )
        print(f"Result: {'success' if success else 'failed'}")

        skills.move_to_free_state()
    finally:
        skills.disconnect()
