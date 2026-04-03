"""
pull — 당기기 스킬

현재 위치에서 직선으로 당기는 core skill.
서랍 열기, 물체 끌어당기기 등에 사용.
내부적으로 move_linear를 핵심 동작으로 사용.

전제 조건:
    - LLM 생성 코드가 이미 접근(approach), 하강, gripper_close를 완료한 상태
    - EE가 이미 pull 시작 위치(물체를 잡은 상태)에 있음

동작 시퀀스:
    1. move_linear — pull_direction 방향으로 pull_distance만큼 직선 당기기

    current position (grasped)
    ●─────────────────────→●
    pull start  (1.당기기)  pull end

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.pull import pull

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    handle_pos = [0.25, 0.0, 0.03]

    # LLM generated code handles approach & grasp:
    move_approach_position(skills, object_position=handle_pos, approach_height=0.10, ...)
    skills.execute_pick_object(handle_pos, ...)
    # Stay at grasp height (no lift needed for pull)

    # Core pull action:
    pull(skills, position=handle_pos, pull_direction=[-1, 0], pull_distance=0.05, ...)

    # LLM generated code handles release & retreat:
    skills.gripper_open(...)
    skills.move_to_position([x, y, 0.10], ...)

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def pull(
    skills,
    position: Union[List[float], np.ndarray],
    pull_direction: Union[List[float], np.ndarray],
    pull_distance: float = 0.05,
    pull_height: Optional[float] = None,
    pull_duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    현재 위치에서 직선으로 당기기.

    EE가 이미 물체를 잡은 상태에서 pull_direction 방향으로 pull_distance만큼
    직선 이동. 접근/잡기/놓기는 LLM 생성 코드에서 처리.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        position: 당기기 시작 위치 [x, y, z] (단위: meters)
                  - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                  - 물체 또는 손잡이의 중심 좌표 (이미 잡고 있는 위치)
                  - z값은 pull_height 미지정 시 당기는 높이로 사용
                  - 예: [0.25, 0.0, 0.03] → 서랍 손잡이 위치

        pull_direction: 당기는 방향 벡터 [dx, dy] (단위: 없음, 정규화 불필요)
                        - xy 평면에서의 당기는 방향
                        - 자동으로 단위 벡터로 정규화됨
                        - [-1, 0]: 로봇 쪽으로 (서랍 열기)
                        - [0, 1]: 오른쪽으로
                        - [1, 1]: 대각선 방향
                        - 예: [-1, 0] → x축 음의 방향 (로봇 쪽으로 당기기)

        pull_distance: 당기는 거리 (단위: meters). default=0.05 (5cm)
                       - pull_direction 방향으로 이동할 직선 거리
                       - 서랍: 0.05~0.15 (5~15cm)
                       - 물체 이동: 0.03~0.10 (3~10cm)
                       - 너무 크면 workspace 벗어남 주의

        pull_height: 당기는 높이 (단위: meters). default=None
                     - None: position의 z값 사용
                     - 별도 지정 시 position의 z 무시, 이 높이에서 당김
                     - 서랍 손잡이: 0.03~0.05 (3~5cm)
                     - 테이블 위 물체: 0.01~0.03 (1~3cm)

        pull_duration: 당기기 이동 시간 (단위: 초). default=None
                       - None: 거리 기반 자동 계산 (5cm/s, 최소 2초)
                       - 서랍: 2~3초 (천천히, 안정적으로)
                       - 빠른 당기기: 1~2초

        object_name: 당길 대상 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "drawer handle", "red block"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - 레코딩 시 스킬 라벨에 사용
                           - None이면 자동 생성: "pull {object_name}: pulling"

    Returns:
        bool: True면 당기기 동작 성공, False면 실패
    """
    pos = np.array(position, dtype=float)

    # pull_direction 정규화
    pull_dir = np.array(pull_direction[:2], dtype=float)
    dir_norm = np.linalg.norm(pull_dir)
    if dir_norm < 1e-6:
        skills._log("ERROR: pull_direction is zero vector")
        return False
    pull_dir = pull_dir / dir_norm

    # 당기는 높이 결정
    p_height = pull_height if pull_height is not None else pos[2]

    # 당기기 시작점/끝점 계산
    pull_start = [pos[0], pos[1], p_height]
    pull_end = [
        pos[0] + pull_dir[0] * pull_distance,
        pos[1] + pull_dir[1] * pull_distance,
        p_height,
    ]

    # 당기기 duration 자동 계산
    if pull_duration is None:
        pull_duration = max(pull_distance / 0.05, 2.0)  # 5cm/s, 최소 2초

    desc_prefix = f"pull {object_name}" if object_name else "pull"

    skills._log(f"\n{'='*60}")
    skills._log(f"[pull] {desc_prefix}")
    skills._log(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Pull dir: [{pull_dir[0]:.2f}, {pull_dir[1]:.2f}]")
    skills._log(f"  Pull distance: {pull_distance*100:.1f}cm")
    skills._log(f"  Pull height: {p_height*100:.1f}cm")
    skills._log(f"  Pull start: [{pull_start[0]:.3f}, {pull_start[1]:.3f}, {pull_start[2]:.3f}]")
    skills._log(f"  Pull end: [{pull_end[0]:.3f}, {pull_end[1]:.3f}, {pull_end[2]:.3f}]")
    skills._log(f"{'='*60}")

    # 1. Pull: 직선 당기기 (핵심)
    skills._log(f"\n[Step 1] Pull {pull_distance*100:.1f}cm")
    pull_success = move_linear(
        skills,
        start=pull_start,
        end=pull_end,
        duration=pull_duration,

        maintain_pitch=True,
        target_name=object_name,
        skill_description=skill_description or f"{desc_prefix}: pulling",
    )

    if not pull_success:
        skills._log("WARNING: Pull move_linear did not fully converge")

    skills._log(f"\n[pull] Complete (pull_success={pull_success})")
    return pull_success


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

        # 테스트: x=0.22 위치의 물체를 로봇 쪽으로 5cm 당기기
        # (실제로는 LLM 코드가 approach & grasp를 먼저 수행)
        print("\n=== Test: Pull object toward robot ===")

        # Simulate LLM-generated approach & grasp
        test_pos = [0.22, 0.0, 0.02]
        skills.gripper_open()
        skills.move_to_position(position=[test_pos[0], test_pos[1], 0.10])
        skills.move_to_position(position=test_pos, maintain_pitch=True)
        skills.gripper_close()

        # Core pull action
        success = pull(
            skills,
            position=test_pos,
            pull_direction=[-1, 0],
            pull_distance=0.05,
            object_name="test object",
            skill_description="test: pull object toward robot",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # Simulate LLM-generated release & retreat
        skills.gripper_open()
        skills.move_to_position(position=[test_pos[0], test_pos[1], 0.10])

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
