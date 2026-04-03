"""
insert — 삽입 스킬

구멍/슬롯에 수직으로 삽입하는 core insert skill.
peg-in-hole, 배터리 삽입, 플러그 꽂기 등에 사용.

Assumes:
    - 물체를 이미 잡은 상태 (LLM 생성 코드에서 pick 완료)
    - EE가 삽입 지점 위에 위치 (LLM 생성 코드에서 approach 완료)

동작 원리:
    2단계 하강으로 정밀 삽입.
    align_height까지 일반 속도로 하강하여 위치 정렬,
    이후 insert_depth만큼 천천히 추가 하강하여 삽입.
    삽입 시 토크 제한으로 과도한 힘 방지.

동작 시퀀스:
    1. move_to_position  — align_height로 하강 (정렬 위치)
    2. move_to_position  — insert_depth만큼 추가 하강 (천천히, 토크 제한)
    3. hold              — 삽입 안착 대기
    4. torque restore    — 토크 복원
    5. gripper_open      — 물체 놓기 (release_after=True일 때)

    (LLM code: pick & approach)
    │
    ● ─ ─ align_height ─ ─ ─ ─ ─ ─ ─  (1.정렬)
    │
    │  (2.삽입, 천천히+토크제한)
    │
    ● ─ ─ insert_point ─ ─ ─ ─ ─ ─ ─
         (3.hold, 4.torque restore)
         (5.gripper open)
    │
    (LLM code: retreat)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.insert import insert

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # LLM generated code handles pick & approach:
    move_approach_position(skills, object_position=peg_pos, ...)
    skills.execute_pick_object(peg_pos, ...)
    skills.move_to_position([peg_x, peg_y, 0.15], ...)  # lift
    move_approach_position(skills, object_position=hole_pos, ...)

    # Core insert action:
    insert(skills, position=hole_pos, insert_depth=0.02, align_height=0.04, ...)

    # LLM generated code handles retreat:
    skills.move_to_position([x, y, 0.10], ...)

    # 예시: 정밀 삽입 (느리게, 낮은 토크)
    insert(
        skills,
        position=[0.20, 0.05, 0.03],
        insert_depth=0.015,
        insert_duration=2.0,
        max_insert_torque=300,
        target_name="battery slot",
    )

    # 예시: 삽입 후 물체를 계속 잡은 상태 유지
    insert(
        skills,
        position=[0.20, 0.05, 0.03],
        insert_depth=0.02,
        release_after=False,
        target_name="screw hole",
    )

    skills.disconnect()
"""

import time
from typing import List, Optional, Union

import numpy as np


def insert(
    skills,
    position: Union[List[float], np.ndarray],
    insert_depth: float = 0.02,
    align_height: float = 0.04,
    insert_duration: float = 1.0,
    max_insert_torque: int = 400,
    hold_time: float = 0.3,
    duration: Optional[float] = None,
    release_after: bool = True,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    구멍/슬롯에 수직으로 삽입 (core action only).

    EE가 이미 삽입 지점 위에 위치한 상태에서 호출.
    2단계 하강: align_height까지 정렬 → insert_depth만큼 천천히 삽입.
    삽입 구간에서 토크 제한 적용하여 과도한 힘 방지.

    Pick, approach, retreat은 LLM 생성 코드에서 처리.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 삽입할 구멍/슬롯 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - xy: 구멍의 수평 좌표
                  - z: 참조용 (실제 높이는 align_height, insert_depth로 제어)
                  - 예: [0.20, 0.05, 0.03] → 구멍 위치

        insert_depth: 삽입 깊이 (단위: meters). default=0.02 (2cm)
                      - align_height에서 추가로 하강하는 거리
                      - 최저 z = align_height - insert_depth
                      - peg-in-hole: 0.01~0.03 (1~3cm)
                      - 배터리 삽입: 0.01~0.02 (1~2cm)
                      - 너무 크면 테이블/물체 충돌 위험

        align_height: 정렬 높이 (단위: meters). default=0.04 (4cm)
                      - 구멍 바로 위에서 위치 정렬하는 높이
                      - 여기까지는 일반 속도로 하강
                      - 여기서부터 천천히 삽입 시작
                      - 구멍 표면보다 약간 위로 설정

        insert_duration: 삽입 구간 이동 시간 (단위: 초). default=1.0
                         - align_height → 최저점 구간의 이동 시간
                         - 0.5: 빠른 삽입 (큰 구멍, 여유 있는 공차)
                         - 1.0: 보통 삽입
                         - 2.0+: 정밀 삽입 (좁은 공차)

        max_insert_torque: 삽입 구간 토크 제한 (0~1000). default=400
                           - 서보 최대 출력을 제한하여 과도한 힘 방지
                           - 1000: 제한 없음
                           - 400: 보통 삽입
                           - 200: 약하게 (깨지기 쉬운 부품)
                           - 삽입 시작 전 적용, 삽입 후 해제

        hold_time: 삽입 후 유지 시간 (단위: 초). default=0.3
                   - 최저점에서 대기 (삽입 안착 대기)
                   - 0.0: 즉시 놓기
                   - 0.3~0.5: 안착 확인

        duration: 정렬 하강 이동 시간 (단위: 초). default=None
                  - None: skills 내부 기본값 사용
                  - 삽입 속도(insert_duration)와 별도 제어

        release_after: 삽입 후 gripper 열기 여부. default=True
                       - True: 삽입 후 물체 놓기 (peg-in-hole)
                       - False: 삽입 후 계속 잡기 (나사 조이기 전 단계)

        target_name: 삽입 대상 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "hole", "battery slot", "USB port"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - None이면 자동 생성: "insert into {target_name}"

    Returns:
        bool: True면 삽입 성공, False면 정렬/삽입 중 실패
    """
    pos = np.array(position, dtype=float)

    insert_z = align_height - insert_depth
    if insert_z < 0:
        insert_z = 0.001  # 최소 1mm

    desc_prefix = f"insert into {target_name}" if target_name else "insert"

    skills._log(f"\n{'='*60}")
    skills._log(f"[insert] {desc_prefix}")
    skills._log(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Align height: {align_height*100:.1f}cm")
    skills._log(f"  Insert depth: {insert_depth*100:.1f}cm → insert z: {insert_z*100:.1f}cm")
    skills._log(f"  Insert duration: {insert_duration:.1f}s")
    skills._log(f"  Max insert torque: {max_insert_torque}/1000")
    skills._log(f"{'='*60}")

    # 1. Descend to align height (정렬)
    skills._log(f"\n[Step 1] Descend to align height ({align_height*100:.1f}cm)")
    if not skills.move_to_position(
        position=[pos[0], pos[1], align_height],
        duration=duration,

        maintain_pitch=True,
        target_name=target_name,
        skill_description=f"{desc_prefix}: align above hole",
    ):
        skills._log("ERROR: Failed to descend to align height")
        return False

    # 2~3: 토크 제한 구간 (try/finally로 반드시 복원)
    try:
        # 2. Insert (천천히 삽입, 토크 제한)
        skills._log(
            f"\n[Step 2] Insert {insert_depth*100:.1f}cm "
            f"(torque limit: {max_insert_torque}/1000)"
        )
        skills.robot.set_torque_limit(max_insert_torque)
        insert_success = skills.move_to_position(
            position=[pos[0], pos[1], insert_z],
            duration=insert_duration,
    
            maintain_pitch=True,
            target_name=target_name,
            skill_description=skill_description or f"{desc_prefix}: inserting",
        )

        if not insert_success:
            skills._log("WARNING: Insert did not fully converge")

        # 3. Hold (삽입 안착 대기)
        if hold_time > 0:
            skills._log(f"\n[Step 3] Hold for {hold_time:.1f}s")
            time.sleep(hold_time)

    finally:
        # 4. Torque restore (에러 발생해도 반드시 실행)
        skills.robot.set_torque_limit(1000)
        skills._log("\n[Step 4] Torque restored to 1000/1000")

    # 5. Release (선택)
    if release_after:
        skills._log("\n[Step 5] Open gripper (release)")
        skills.gripper_open(
            skill_description=f"{desc_prefix}: release after insert",
        )
    else:
        skills._log("\n[Step 5] Skipped (release_after=False)")

    skills._log(f"\n[insert] Complete (insert_success={insert_success})")
    return insert_success


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

        # 테스트: EE가 이미 삽입 지점 위에 위치한 상태 시뮬레이션
        # 1) gripper close로 잡은 상태 만들기
        skills.gripper_close()

        # 2) 삽입 지점 위로 이동 (LLM 코드가 하는 역할)
        hole_pos = [0.20, 0.0, 0.03]
        skills.move_to_position(
            position=[hole_pos[0], hole_pos[1], 0.10],
            target_name="test hole",
            skill_description="test: approach above hole",
        )

        # 3) Core insert action
        print("\n=== Test: Insert into hole ===")
        success = insert(
            skills,
            position=hole_pos,
            insert_depth=0.02,
            align_height=0.04,
            max_insert_torque=400,
            target_name="test hole",
            skill_description="test: insert peg into hole",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # 4) Retreat (LLM 코드가 하는 역할)
        skills.move_to_position(
            position=[hole_pos[0], hole_pos[1], 0.10],
            target_name="test hole",
            skill_description="test: retreat after insert",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
