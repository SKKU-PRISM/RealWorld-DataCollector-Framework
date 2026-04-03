"""
strike — 치기/두드리기 스킬 (core action only)

빠르게 내려쳤다가 즉시 복귀하는 composite skill.
못 박기, 두드리기, 탭핑 등에 사용.
press와 유사하지만 속도가 빠르고 유지 시간이 없음.

이 함수는 EE가 이미 타격 대상 위에 위치하고 gripper가 닫혀 있다고 가정합니다.
접근(approach), gripper 닫기, 이탈(retreat)은 LLM 생성 코드에서 처리합니다.

동작 시퀀스:
    1. move_to_position   — wind_up_height로 이동 (타격 준비)
    2. move_to_position   — strike_height까지 빠르게 하강 (타격!)
    3. move_to_position   — wind_up_height로 즉시 복귀
    (2~3 반복 × num_strikes)

    wind_up_height ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●  (1.준비)
    │                                        │
    │  (2.타격! 빠르게)        (3.즉시복귀) │
    │                                        │
    ● ─ ─ strike_height ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●
         (접촉 순간)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.strike import strike

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # LLM generated code handles approach:
    skills.gripper_close(...)  # holding hammer or closed fist
    move_approach_position(skills, object_position=nail_pos, approach_height=0.10, ...)

    # Core strike action:
    strike(skills, position=nail_pos, strike_height=0.01, wind_up_height=0.06, num_strikes=3, ...)

    # LLM generated code handles retreat:
    skills.move_to_position([x, y, 0.10], ...)

    skills.disconnect()
"""

import time
from typing import List, Optional, Union

import numpy as np


def strike(
    skills,
    position: Union[List[float], np.ndarray],
    strike_height: float = 0.01,
    wind_up_height: float = 0.06,
    strike_duration: float = 0.3,
    retract_duration: float = 0.3,
    max_strike_torque: int = 600,
    num_strikes: int = 1,
    strike_interval: float = 0.3,
    duration: Optional[float] = None,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    빠르게 내려쳐서 타격 후 즉시 복귀 (core action only).

    EE가 이미 타격 대상 위에 위치하고 gripper가 닫혀 있다고 가정합니다.
    wind_up_height에서 strike_height까지 빠르게 하강, 즉시 복귀.
    num_strikes회 반복 가능.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 타격할 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - xy: 타격 대상의 수평 좌표
                  - z: 참조용 (실제 높이는 strike_height)
                  - 예: [0.20, 0.0, 0.03] → 못 머리 위치

        strike_height: 타격 최저점 높이 (단위: meters). default=0.01 (1cm)
                       - EE가 도달하는 최저 높이
                       - 대상 표면 높이와 맞추거나 약간 아래로 설정
                       - 너무 낮으면 테이블 충돌 위험

        wind_up_height: 타격 준비 높이 (단위: meters). default=0.06 (6cm)
                        - 타격 전 EE를 올려놓는 높이
                        - strike_height와의 차이 = 타격 스트로크
                        - 클수록 더 강한 타격 (가속 거리 증가)
                        - 작을수록 부드러운 타격

        strike_duration: 타격 하강 시간 (단위: 초). default=0.3
                         - wind_up → strike_height 이동 시간
                         - 작을수록 빠른(강한) 타격
                         - 0.2: 빠른 타격
                         - 0.3: 보통 타격
                         - 0.5: 느린 타격

        retract_duration: 타격 후 복귀 시간 (단위: 초). default=0.3
                          - strike_height → wind_up 복귀 시간
                          - 빠를수록 깔끔한 타격 (반동 방지)

        max_strike_torque: 타격 구간 토크 제한 (0~1000). default=600
                           - press보다 높게 설정 (타격은 더 큰 힘 필요)
                           - 1000: 제한 없음 (최대 힘)
                           - 600: 보통 타격
                           - 400: 약한 타격
                           - 타격 하강 시 적용, 복귀 전 해제

        num_strikes: 타격 횟수 (단위: 회). default=1
                     - 1: 단일 타격
                     - 3~5: 반복 타격 (못 박기)

        strike_interval: 타격 간 대기 시간 (단위: 초). default=0.3
                         - 연속 타격 시 각 타격 사이의 대기
                         - wind_up 복귀 후 다음 타격 전 대기

        duration: wind-up 이동 시간 (단위: 초). default=None

        target_name: 타격 대상 이름 (선택). default=None
                     - 예: "nail", "button", "xylophone key"

        skill_description: 스킬 동작 설명 (선택). default=None

    Returns:
        bool: True면 모든 타격 성공, False면 실패
    """
    pos = np.array(position, dtype=float)

    desc_prefix = f"strike {target_name}" if target_name else "strike"

    skills._log(f"\n{'='*60}")
    skills._log(f"[strike] {desc_prefix}")
    skills._log(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Strike height: {strike_height*100:.1f}cm")
    skills._log(f"  Wind-up height: {wind_up_height*100:.1f}cm")
    skills._log(f"  Stroke: {(wind_up_height - strike_height)*100:.1f}cm")
    skills._log(f"  Strike duration: {strike_duration:.2f}s")
    skills._log(f"  Num strikes: {num_strikes}")
    skills._log(f"  Max strike torque: {max_strike_torque}/1000")
    skills._log(f"{'='*60}")

    # 1. Move to wind-up height
    skills._log(f"\n[Step 1] Move to wind-up height ({wind_up_height*100:.1f}cm)")
    if not skills.move_to_position(
        position=[pos[0], pos[1], wind_up_height],
        duration=duration,
        maintain_pitch=True,
        target_name=target_name,
        skill_description=f"{desc_prefix}: wind up",
    ):
        skills._log("ERROR: Failed to reach wind-up position")
        return False

    # 2~3. Strike loop
    all_success = True

    for i in range(num_strikes):
        strike_num = i + 1
        skills._log(f"\n[Step 2.{strike_num}] Strike {strike_num}/{num_strikes}")

        # 타격 (토크 제한 적용)
        try:
            skills.robot.set_torque_limit(max_strike_torque)
            skills._log(f"  Torque limited to {max_strike_torque}/1000")

            strike_success = skills.move_to_position(
                position=[pos[0], pos[1], strike_height],
                duration=strike_duration,

                maintain_pitch=True,
                target_name=target_name,
                skill_description=(
                    skill_description
                    or f"{desc_prefix}: strike {strike_num}"
                ),
            )
            if not strike_success:
                all_success = False

        finally:
            # 토크 복원
            skills.robot.set_torque_limit(1000)

        # 3. 즉시 복귀
        skills._log(f"  [Step 3.{strike_num}] Retract to wind-up")
        skills.move_to_position(
            position=[pos[0], pos[1], wind_up_height],
            duration=retract_duration,
            maintain_pitch=True,
            target_name=target_name,
            skill_description=f"{desc_prefix}: retract {strike_num}",
        )

        # 타격 간 대기 (마지막 타격 제외)
        if i < num_strikes - 1 and strike_interval > 0:
            skills._log(f"  Wait {strike_interval:.1f}s")
            time.sleep(strike_interval)

    skills._log(f"\n[strike] Complete (all_success={all_success})")
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

        # Simulate LLM-generated approach code
        print("\n=== Pre-strike: gripper close + approach (LLM code) ===")
        skills.gripper_close(skill_description="test: close gripper for hammer")
        skills.move_to_position(
            position=[0.20, 0.0, 0.10],
            target_name="test target",
            skill_description="test: approach above target",
        )

        # Core strike action
        print("\n=== Test: Strike 3 times ===")
        success = strike(
            skills,
            position=[0.20, 0.0, 0.03],
            strike_height=0.02,
            wind_up_height=0.06,
            num_strikes=3,
            max_strike_torque=500,
            target_name="test target",
            skill_description="test: strike target",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # Simulate LLM-generated retreat code
        print("\n=== Post-strike: retreat (LLM code) ===")
        skills.move_to_position(
            position=[0.20, 0.0, 0.10],
            target_name="test target",
            skill_description="test: retreat after strike",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
