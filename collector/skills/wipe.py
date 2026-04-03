"""
wipe — 닦기 스킬 (core action only)

테이블 표면을 접촉하며 왕복 직선 이동으로 닦는 core skill.
내부적으로 move_linear를 핵심 동작으로 사용.

전제 조건 (LLM 코드에서 외부 처리):
    - gripper가 이미 닫혀 있어야 함 (gripper_close)
    - EE가 이미 start 위 공중에 있어야 함 (move_approach_position)

동작 시퀀스:
    1. move_to_position  — start의 wipe_height로 하강 (표면 접촉)
    2. move_linear × N   — start ↔ end 왕복 (편도 × num_strokes)

    (사전 조건: gripper_close + move_approach_position)

    num_strokes 동작:
        1회: start ──→ end                  (편도)
        2회: start ──→ end ──→ start        (1왕복)
        3회: start ──→ end ──→ start ──→ end (1.5왕복)

    stroke_length 적용 시:
        start ────────────── end (전체 10cm)
        start ──── end'          (stroke_length=5cm, 방향 유지)
              ←5cm→

                    approach_height (외부 처리)
    ─────●                              ●─────  (외부 처리)
         │  (외부)              (외부)  │
         │                              │
         ●───↔───↔───↔──────────────→──●   wipe_height
         start   (1.하강→2.왕복 닦기)  end

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.move_approach_position import move_approach_position
    from skills.wipe import wipe

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    start_pos = [0.20, -0.05, 0.0]
    end_pos = [0.20, 0.05, 0.0]

    # 1. 외부: gripper 닫기
    skills.gripper_close(skill_description="wipe table: close gripper")

    # 2. 외부: start 위 approach 위치로 이동
    move_approach_position(skills, object_position=start_pos, approach_height=0.10,
                           object_name="table surface")

    # 3. Core wipe
    wipe(
        skills,
        start_position=start_pos,
        end_position=end_pos,
        num_strokes=3,
        object_name="table surface",
    )

    # 4. 외부: 이탈 (상승)
    skills.move_to_position([end_pos[0], end_pos[1], 0.10],
                            skill_description="wipe table: lift after wiping")

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def wipe(
    skills,
    start_position: Union[List[float], np.ndarray],
    end_position: Union[List[float], np.ndarray],
    num_strokes: int = 3,
    stroke_length: Optional[float] = None,
    wipe_height: float = 0.01,
    stroke_duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    표면을 접촉하며 왕복 닦기 (core action).

    EE가 이미 start 위 공중에 있고 gripper가 닫혀 있다고 가정.
    wipe_height로 하강한 뒤 start ↔ end 사이를 num_strokes회 왕복.
    stroke_length 지정 시 start-end 방향은 유지하되 실제 왕복 거리를 제한.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        start_position: 닦기 시작 위치 [x, y, z] (단위: meters)
                        - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
                        - 닦기 왕복의 한쪽 끝점
                        - z값은 참조용 (실제 높이는 wipe_height)

        end_position: 닦기 끝 위치 [x, y, z] (단위: meters)
                      - start와 동일한 좌표계
                      - 닦기 왕복의 반대쪽 끝점 (방향과 최대 거리 결정)
                      - stroke_length 미지정 시 이 점까지 전체 구간 닦기

        num_strokes: 편도 횟수 (단위: 회). default=3
                     - 1: 편도 1회 (start → end)
                     - 2: 1왕복 (start → end → start)
                     - 3: 1.5왕복 (start → end → start → end)
                     - 홀수: end에서 끝남, 짝수: start에서 끝남

        stroke_length: 편도 직선 거리 (단위: meters). default=None
                       - None: start-end 전체 거리 사용
                       - < 전체 거리: start-end 방향 유지, end 위치를 단축
                       - >= 전체 거리: 전체 거리 그대로 사용 (무시)
                       - 예: start-end=10cm, stroke_length=0.05 → 5cm만 왕복

        wipe_height: 닦기 시 EE 높이 (단위: meters). default=0.01 (1cm)
                     - 테이블 표면 기준 EE가 유지할 높이
                     - 너무 낮으면 테이블 충돌, 너무 높으면 접촉 불가
                     - 0.005~0.02 범위 권장

        stroke_duration: 편도 1회 이동 시간 (단위: 초). default=None
                         - None: 거리 기반 자동 계산 (8cm/s, 최소 1.5초)
                         - 모든 stroke에 동일하게 적용
                         - 작을수록 빠르게 닦음 (단, 추종 오차 증가)

        object_name: 닦기 대상 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "table surface", "spill area"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - 레코딩 시 스킬 라벨에 사용
                           - None이면 자동 생성: "wipe {object_name}: stroke N (forward/return)"

    Returns:
        bool: True면 모든 stroke 성공, False면 하나 이상의 stroke 실패
    """
    start_pos = np.array(start_position, dtype=float)
    end_pos = np.array(end_position, dtype=float)

    full_distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
    desc_prefix = f"wipe {object_name}" if object_name else "wipe"

    # stroke_length로 실제 end 위치 조정
    if stroke_length is not None and stroke_length < full_distance:
        direction = (end_pos - start_pos)
        direction_norm = direction / np.linalg.norm(direction[:2])
        end_pos = start_pos + direction_norm * stroke_length
        wipe_distance = stroke_length
    else:
        wipe_distance = full_distance

    skills._log(f"\n{'='*60}")
    skills._log(f"[wipe] {desc_prefix}")
    skills._log(f"  Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    skills._log(f"  End:   [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    skills._log(f"  Strokes: {num_strokes}, stroke distance: {wipe_distance*100:.1f}cm")
    if stroke_length is not None:
        skills._log(f"  (stroke_length={stroke_length*100:.1f}cm, full={full_distance*100:.1f}cm)")
    skills._log(f"  Wipe height: {wipe_height*100:.1f}cm")
    skills._log(f"{'='*60}")

    # 편도 duration 계산 (없으면 거리 기반)
    if stroke_duration is None:
        stroke_duration = max(wipe_distance / 0.08, 1.5)  # 8cm/s, 최소 1.5초

    # 1. Descend to wipe height
    skills._log("\n[Step 1] Descend to wipe height")
    if not skills.move_to_position(
        position=[start_pos[0], start_pos[1], wipe_height],

        maintain_pitch=True,
        target_name=object_name,
        skill_description=f"{desc_prefix}: descend to surface",
    ):
        skills._log("ERROR: Failed to descend to wipe height")
        return False

    # 2. Wipe strokes
    wipe_start = [start_pos[0], start_pos[1], wipe_height]
    wipe_end = [end_pos[0], end_pos[1], wipe_height]

    all_success = True
    for i in range(num_strokes):
        stroke_num = i + 1
        if i % 2 == 0:
            # Forward stroke: start → end
            s_from, s_to = wipe_start, wipe_end
            direction = "forward"
        else:
            # Return stroke: end → start
            s_from, s_to = wipe_end, wipe_start
            direction = "return"

        skills._log(f"\n[Step 2.{stroke_num}] Stroke {stroke_num}/{num_strokes} ({direction})")
        success = move_linear(
            skills,
            start=s_from,
            end=s_to,
            duration=stroke_duration,
    
            maintain_pitch=True,
            target_name=object_name,
            skill_description=(
                skill_description or f"{desc_prefix}: stroke {stroke_num} ({direction})"
            ),
        )
        if not success:
            skills._log(f"WARNING: Stroke {stroke_num} did not fully converge")
            all_success = False

    skills._log(f"\n[wipe] Complete (all_success={all_success})")
    return all_success


if __name__ == "__main__":
    from skills.move_approach_position import move_approach_position
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

        start_pos = [0.20, -0.05, 0.0]
        end_pos = [0.20, 0.05, 0.0]

        # 외부: gripper 닫기
        skills.gripper_close(skill_description="test: close gripper for wipe")

        # 외부: start 위 approach 위치로 이동
        move_approach_position(
            skills,
            object_position=start_pos,
            approach_height=0.10,
            object_name="table surface",
        )

        # Core wipe: y축 방향 3회 왕복 닦기 (1.5왕복)
        success = wipe(
            skills,
            start_position=start_pos,
            end_position=end_pos,
            num_strokes=3,
            wipe_height=0.01,
            object_name="table surface",
            skill_description="test: wipe table surface",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # 외부: 이탈 (상승)
        skills.move_to_position(
            [end_pos[0], end_pos[1], 0.10],
            skill_description="test: lift after wiping",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
