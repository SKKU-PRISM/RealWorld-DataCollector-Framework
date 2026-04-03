"""
press — 누르기 스킬 (모듈형)

특정 위치를 수직으로 눌렀다가 복귀하는 composite skill.
버튼, 스위치, 터치패드 등 접촉 후 일정 깊이를 더 누르는 태스크에 사용.

동작 원리:
    EE가 이미 press 대상 위에 위치한 상태에서 호출.
    접촉면(contact_height)까지는 일반 속도로 하강,
    접촉 후 press_depth만큼 천천히 추가 하강하여 "누름" 효과.
    hold_time 동안 유지 후 contact_height로 복귀.

    접근(approach) 및 후퇴(retreat)는 LLM 생성 코드에서 처리하므로,
    이 함수는 핵심 press 동작만 수행.

동작 시퀀스:
    1. move_to_position  — contact_height로 하강 (접촉면까지, 일반 속도)
    2. move_to_position  — press_depth만큼 추가 하강 (천천히 누름, 토크 제한)
    3. hold              — hold_time 동안 누른 상태 유지
    4. torque restore    — 토크 복원 (try/finally로 보장)
    5. move_to_position  — contact_height로 복귀 (접촉면 이탈)

    (LLM 코드가 approach)
    │
    ● ─ ─ contact_height ─ ─ ─ ─ ─ ●  (1.하강 / 5.복귀)
    │                                │
    │  (2.누름, 천천히)    (5.복귀)  │
    │                                │
    ● ─ ─ press_point ─ ─ ─ ─ ─ ─ ─●
         (3.hold_time 유지)
    │
    (LLM 코드가 retreat)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.press import press

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # LLM generated code handles approach:
    skills.gripper_close(...)
    move_approach_position(skills, position=button_pos, approach_height=0.10, ...)

    # Core press action:
    press(skills, position=button_pos, contact_height=0.03, press_depth=0.01, ...)

    # LLM generated code handles retreat:
    skills.move_to_position([x, y, 0.10], ...)

    skills.disconnect()
"""

import time
from typing import List, Optional, Union

import numpy as np


def press(
    skills,
    position: Union[List[float], np.ndarray],
    press_depth: float = 0.01,
    contact_height: float = 0.02,
    press_duration: float = 0.5,
    hold_time: float = 0.3,
    max_press_torque: int = 400,
    duration: Optional[float] = None,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    특정 위치를 수직으로 눌렀다가 복귀 (핵심 press 동작만 수행).

    EE가 이미 press 대상 위에 위치한 상태에서 호출.
    2단계 하강: contact_height까지 일반 하강 → press_depth만큼 추가 하강(천천히).
    hold_time 유지 후 contact_height로 복귀.

    접근(gripper_close, move_approach_position)과 후퇴(lift)는
    LLM 생성 코드에서 처리.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 누를 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - xy: 버튼/스위치의 수평 좌표
                  - z: 참조용 (실제 높이는 contact_height로 제어)
                  - 예: [0.20, 0.05, 0.03] → 버튼 위치

        press_depth: 접촉면에서 추가로 누르는 깊이 (단위: meters). default=0.01 (1cm)
                     - contact_height에서 아래로 추가 하강하는 거리
                     - 최저 z = contact_height - press_depth
                     - 버튼: 0.003~0.01 (3~10mm, 가볍게)
                     - 스위치: 0.01~0.02 (10~20mm, 확실히)
                     - 너무 크면 테이블/물체 충돌 위험

        contact_height: 접촉면 예상 높이 (단위: meters). default=0.02 (2cm)
                        - 버튼/스위치 표면의 z좌표 (테이블 기준)
                        - 여기까지는 일반 속도로 하강
                        - 여기서부터 press_depth만큼 천천히 추가 하강
                        - 예: 테이블 위 2cm 높이의 버튼 → contact_height=0.02

        press_duration: 누르는 구간 소요 시간 (단위: 초). default=0.5
                        - contact_height → 최저점 구간의 이동 시간
                        - 접근/복귀 속도와 독립적으로 제어
                        - 0.3: 빠른 탭 (버튼)
                        - 0.5: 보통 press
                        - 1.0+: 부드러운 천천히 누름 (민감한 스위치)

        hold_time: 눌린 상태 유지 시간 (단위: 초). default=0.3
                   - 최저점에서 대기하는 시간
                   - 0.0: 누르자마자 즉시 복귀 (탭)
                   - 0.3: 짧은 press (일반 버튼)
                   - 2.0+: long press (전원 버튼 등)

        max_press_torque: 누르기 구간 토크 제한 (0~1000). default=400
                          - 서보 모터의 최대 출력을 제한하여 과도한 힘 방지
                          - 1000: 제한 없음 (기본 서보 최대 토크)
                          - 400: 보통 press (버튼, 스위치)
                          - 200: 약하게 press (깨지기 쉬운 대상)
                          - 누르기(Step 2) 시작 전에 적용, 복귀(Step 5) 전에 해제

        duration: 하강/복귀 이동 시간 (단위: 초). default=None
                  - contact_height까지 하강 구간의 이동 시간
                  - None: skills 내부 기본값(movement_duration) 사용
                  - 누르는 속도(press_duration)와 별도 제어

        target_name: 누를 대상 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "power button", "switch"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - 레코딩 시 스킬 라벨에 사용
                           - None이면 자동 생성: "press {target_name}"

    Returns:
        bool: True면 press 동작 성공, False면 하강 중 실패
    """
    pos = np.array(position, dtype=float)

    press_z = contact_height - press_depth
    if press_z < 0:
        press_z = 0.001  # 최소 1mm (테이블 충돌 방지)

    desc_prefix = f"press {target_name}" if target_name else "press"

    skills._log(f"\n{'='*60}")
    skills._log(f"[press] {desc_prefix}")
    skills._log(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Contact height: {contact_height*100:.1f}cm")
    skills._log(f"  Press depth: {press_depth*100:.1f}cm → press z: {press_z*100:.1f}cm")
    skills._log(f"  Press duration: {press_duration:.1f}s, Hold: {hold_time:.1f}s")
    skills._log(f"  Max press torque: {max_press_torque}/1000")
    skills._log(f"{'='*60}")

    # 1. Descend to contact height (일반 속도, pitch 자유)
    skills._log("\n[Step 1] Descend to contact height")
    if not skills.move_to_position(
        position=[pos[0], pos[1], contact_height],
        duration=duration,

        maintain_pitch=False,
        target_name=target_name,
        skill_description=f"{desc_prefix}: descend to contact",
    ):
        skills._log("ERROR: Failed to descend to contact height")
        return False

    # 2~4: 토크 제한 구간 (try/finally로 반드시 복원)
    try:
        # 2. Press down (천천히 추가 하강) — 토크 제한 적용
        skills._log(f"\n[Step 2] Press down {press_depth*100:.1f}cm (torque limit: {max_press_torque}/1000)")
        skills.robot.set_torque_limit(max_press_torque)
        press_success = skills.move_to_position(
            position=[pos[0], pos[1], press_z],
            duration=press_duration,
    
            maintain_pitch=True,
            target_name=target_name,
            skill_description=skill_description or f"{desc_prefix}: pressing down",
        )

        if not press_success:
            skills._log("WARNING: Press did not fully converge")

        # 3. Hold (눌린 상태 유지)
        if hold_time > 0:
            skills._log(f"\n[Step 3] Hold for {hold_time:.1f}s")
            time.sleep(hold_time)

    finally:
        # 4. 토크 복원 (에러 발생해도 반드시 실행)
        skills.robot.set_torque_limit(1000)
        skills._log("\n[Step 4] Torque restored to 1000/1000")

    # 5. Retract to contact height (접촉면 이탈)
    skills._log("\n[Step 5] Retract to contact height")
    skills.move_to_position(
        position=[pos[0], pos[1], contact_height],
        duration=press_duration,

        maintain_pitch=True,
        target_name=target_name,
        skill_description=f"{desc_prefix}: retract from surface",
    )

    skills._log(f"\n[press] Complete (press_success={press_success})")
    return press_success


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

        button_pos = [0.20, 0.0, 0.03]
        approach_height = 0.10

        # LLM generated code handles approach:
        print("\n=== Approach (simulating LLM code) ===")
        skills.gripper_close(skill_description="press test: close gripper")
        skills.move_to_position(
            position=[button_pos[0], button_pos[1], approach_height],
            target_name="test button",
            skill_description="press test: approach above",
        )

        # Core press action:
        print("\n=== Test: Press button ===")
        success = press(
            skills,
            position=button_pos,
            press_depth=0.01,
            contact_height=0.03,
            hold_time=0.5,
            target_name="test button",
            skill_description="test: press button",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # LLM generated code handles retreat:
        print("\n=== Retreat (simulating LLM code) ===")
        skills.move_to_position(
            position=[button_pos[0], button_pos[1], approach_height],
            target_name="test button",
            skill_description="press test: lift after press",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
