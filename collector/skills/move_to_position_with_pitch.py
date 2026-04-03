"""
move_to_position_with_pitch — pitch 명시 이동 스킬

기존 move_to_position의 target_pitch를 도(degree) 단위로 직관적으로 노출.
LLM이 gripper 자세를 명시적으로 지정할 수 있게 함.

Pitch 정의:
    - Robot base_link 좌표계 기준, gripper Z축(pointing direction)과 수평면 사이의 각도.
    - SO-101 로봇은 base가 테이블 위에 수직 고정이므로 world frame과 실질적으로 동일.

Pitch 규약:
    -90도: gripper가 수직 아래를 향함 (테이블 방향, 물체를 위에서 잡을 때)
      0도: gripper가 수평 (앞을 향함)
    +90도: gripper가 수직 위를 향함

사용 예시:
    - 물체를 위에서 집기:     pitch_deg=-90 (수직 하강)
    - 물체를 기울여 붓기:     pitch_deg=-45 (45도 기울임)
    - 물체를 수평으로 밀기:   pitch_deg=0   (수평)
    - 물체를 들어올린 후:     pitch_deg=-30 (약간 아래 기울임)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.move_to_position_with_pitch import move_to_position_with_pitch

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # 예시 1: gripper를 60도 아래로 기울여서 이동
    move_to_position_with_pitch(
        skills,
        position=[0.2, 0.1, 0.10],
        pitch_deg=-60,
    )

    # 예시 2: gripper를 수평으로 유지하며 이동
    move_to_position_with_pitch(
        skills,
        position=[0.2, 0.0, 0.15],
        pitch_deg=0,
        target_name="cup",
        skill_description="approach cup horizontally",
    )

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np


def move_to_position_with_pitch(
    skills,
    position: Union[List[float], np.ndarray],
    pitch_deg: float,
    duration: Optional[float] = None,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    지정한 pitch 각도로 목표 위치에 이동.

    내부적으로 skills.move_to_position(target_pitch=...) 을 호출하며,
    pitch 각도를 도(degree) 단위로 받아 radian 변환 후 전달.
    IK solver가 해당 pitch를 만족하는 joint 해를 찾으며,
    정확한 pitch가 불가능하면 ±10도 범위 내에서 가장 가까운 해를 사용.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 목표 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - 예: [0.20, 0.05, 0.10] → x=20cm 전방, y=5cm 좌측, z=10cm 높이

        pitch_deg: gripper pitch 각도 (단위: 도)
                   - -90: gripper가 수직 아래 (테이블을 향함, pick 동작)
                   -   0: gripper가 수평 (앞을 향함, push/pour 동작)
                   - +90: gripper가 수직 위
                   - IK로 정확한 각도 달성이 불가하면 ±10도 내에서 fallback

        duration: 이동 시간 (단위: 초). default=None
                  - None이면 skills 내부 기본값(movement_duration) 사용
                  - 작을수록 빠르게 이동, 클수록 느리고 안정적

        target_name: 대상 물체 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "red block", "blue dish"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - 레코딩 시 스킬 라벨에 사용
                           - 예: "tilt gripper to pour water"

    Returns:
        bool: True면 목표 위치 도달 성공, False면 IK 실패 또는 도달 불가
    """
    pitch_rad = np.radians(pitch_deg)

    return skills.move_to_position(
        position=position,
        duration=duration,
        target_pitch=pitch_rad,
        target_name=target_name,
        skill_description=skill_description,
    )


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

        # 테스트: gripper를 수평(0도)으로 이동
        success = move_to_position_with_pitch(
            skills,
            position=[0.20, 0.0, 0.10],
            pitch_deg=0,
            skill_description="test: move with pitch 0 deg",
        )
        print(f"Result: {'success' if success else 'failed'}")

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
