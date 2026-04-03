"""
rotate_wrist — 임의 각도 wrist 회전 스킬

기존 rotate_90degree의 확장. 90도 고정이 아닌 임의 각도 회전 지원.
상대 모드 (현재에서 N도 회전)와 절대 모드 (절대 각도로 이동) 지원.

동작 원리:
    1. 현재 wrist_roll joint 값 읽기
    2. 목표 각도 계산 (상대 or 절대)
    3. joint limit 체크 및 clamp
    4. joint-space 보간으로 회전 실행 (arm 다른 축은 고정)

Wrist Roll 규약:
    - wrist_roll은 joint index 4 (0-indexed, 5-DOF arm의 마지막 축)
    - +값: 시계 방향 (CW, 위에서 봤을 때)
    - -값: 반시계 방향 (CCW)
    - 회전 시 arm의 다른 4개 축은 현재 위치 유지

사용 시나리오:
    - 볼트 조이기:    angle_deg=180 (CW 반복)
    - 병뚜껑 열기:    angle_deg=-90 (CCW)
    - 물체 방향 조정: angle_deg=45
    - 중립 복귀:      angle_deg=0, absolute=True

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.rotate_wrist import rotate_wrist

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # 예시 1: 시계방향 45도 상대 회전
    rotate_wrist(skills, angle_deg=45)

    # 예시 2: 반시계 120도 회전 (느리게)
    rotate_wrist(skills, angle_deg=-120, duration=3.0)

    # 예시 3: 절대 0도(중립)로 복귀
    rotate_wrist(skills, angle_deg=0, absolute=True)

    skills.disconnect()
"""

from typing import Optional

import numpy as np


def rotate_wrist(
    skills,
    angle_deg: float,
    absolute: bool = False,
    duration: Optional[float] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    wrist_roll을 임의 각도로 회전.

    상대 모드(기본)에서는 현재 각도 기준으로 회전,
    절대 모드에서는 wrist_roll의 절대 각도로 이동.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        angle_deg: 회전 각도 (단위: 도)
                   - absolute=False (상대 모드, 기본):
                       현재 wrist_roll에서 추가 회전할 각도
                       +값: 시계 방향 (CW), -값: 반시계 방향 (CCW)
                       예: 45 → 현재에서 CW 45도, -90 → 현재에서 CCW 90도
                   - absolute=True (절대 모드):
                       wrist_roll의 목표 절대 각도
                       0: 중립 위치, 90: 절대 90도 위치
                       예: 0 → 중립 복귀, 90 → 절대 90도로 이동

        absolute: 절대/상대 모드 선택. default=False
                  - False: 현재 각도 기준 상대 회전 (가장 직관적)
                           "현재에서 45도 더 돌려"
                  - True:  wrist_roll 절대 각도 지정
                           "정확히 0도(중립)로 가"
                  - 볼트 조이기처럼 반복 회전: absolute=False
                  - 특정 자세 복원: absolute=True

        duration: 이동 시간 (단위: 초). default=None
                  - None: 회전 각도에 비례하여 자동 계산
                          (45도당 ~1초, 최소 1초, 최대 4초)
                  - 작을수록 빠르게 회전 (빠른 동작, 볼트 조이기)
                  - 클수록 느리게 회전 (정밀 조립, 안전한 동작)

        skill_description: 스킬 동작 설명 (선택). default=None
                           - None: 자동 생성 ("rotate wrist 45.0° CW" 등)
                           - 레코딩 시 스킬 라벨에 사용
                           - 예: "tighten bolt", "open bottle cap"

    Returns:
        bool: True면 회전 성공, False면 joint limit 초과로 회전 불가
    """
    # 1. 현재 상태 읽기
    current_arm_norm, current_joints, current_ee = skills._get_current_state()
    wrist_roll_idx = 4
    current_wrist_deg = np.degrees(current_joints[wrist_roll_idx])

    # 2. 목표 각도 계산
    if absolute:
        target_wrist_rad = np.radians(angle_deg)
        actual_rotation_deg = angle_deg - current_wrist_deg
    else:
        target_wrist_rad = current_joints[wrist_roll_idx] + np.radians(angle_deg)
        actual_rotation_deg = angle_deg

    target_wrist_deg = np.degrees(target_wrist_rad)

    # 방향 문자열
    if actual_rotation_deg > 0:
        dir_str = "CW"
    elif actual_rotation_deg < 0:
        dir_str = "CCW"
    else:
        dir_str = "no rotation"

    skills._log(f"\n{'='*60}")
    skills._log(f"ROTATE WRIST ({abs(actual_rotation_deg):.1f}° {dir_str})")
    skills._log(f"  Mode: {'absolute' if absolute else 'relative'}")
    skills._log(f"  Current: {current_wrist_deg:.1f}°")
    skills._log(f"  Target:  {target_wrist_deg:.1f}°")
    skills._log(f"{'='*60}")

    # 3. Joint limit 체크
    if skills.calibration_limits:
        lower = skills.calibration_limits.lower_limits_radians[wrist_roll_idx]
        upper = skills.calibration_limits.upper_limits_radians[wrist_roll_idx]

        if target_wrist_rad < lower or target_wrist_rad > upper:
            clamped_rad = np.clip(target_wrist_rad, lower, upper)
            skills._log(
                f"  WARNING: Target {target_wrist_deg:.1f}° exceeds limits "
                f"[{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]"
            )
            skills._log(f"  Clamped to: {np.degrees(clamped_rad):.1f}°")
            target_wrist_rad = clamped_rad
            target_wrist_deg = np.degrees(target_wrist_rad)
            actual_rotation_deg = target_wrist_deg - current_wrist_deg

    # 4. 회전할 필요 없으면 조기 종료
    if abs(actual_rotation_deg) < 0.5:
        skills._log("  Already at target angle, skipping rotation")
        return True

    # 5. Duration 자동 계산
    if duration is None:
        duration = max(min(abs(actual_rotation_deg) / 45.0, 4.0), 1.0)

    skills._log(f"  Duration: {duration:.1f}s")

    # 6. 목표 joint 생성 (wrist_roll만 변경, 나머지 유지)
    target_joints = current_joints.copy()
    target_joints[wrist_roll_idx] = target_wrist_rad

    # 7. 스킬 레코딩 설정
    label = skill_description or f"rotate wrist {abs(actual_rotation_deg):.1f}° {dir_str}"

    skills._set_skill_recording(
        label=label,
        skill_type="rotate",
        goal_joint_5=target_joints,
        goal_gripper=skills.current_gripper_pos,
    )

    # 8. Joint-space 보간 실행
    start_normalized = current_arm_norm
    end_normalized = skills._radians_to_normalized(target_joints)

    try:
        success = skills._execute_move_to_known_pose(
            start_normalized=start_normalized,
            end_normalized=end_normalized,
            duration=duration,
            description=f"Rotating wrist {abs(actual_rotation_deg):.1f}° {dir_str}",
        )

        # 최종 오차 계산
        _, final_rad, final_ee = skills._get_current_state()
        final_wrist_deg = np.degrees(final_rad[wrist_roll_idx])
        wrist_error = abs(target_wrist_deg - final_wrist_deg)

        skills.last_error = skills._calculate_error(
            target_position=current_ee,
            actual_position=final_ee,
            target_wrist_roll_rad=target_wrist_rad,
            actual_wrist_roll_rad=final_rad[wrist_roll_idx],
        )
        skills._print_error(skills.last_error, f"Rotate {abs(actual_rotation_deg):.1f}°")

        skills._log(f"  Final wrist_roll: {final_wrist_deg:.1f}° (error: {wrist_error:.1f}°)")
        skills._log("ROTATE WRIST: Complete")
        return success
    finally:
        skills._clear_skill_recording()


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

        # 테스트 1: 상대 모드 — 시계방향 45도
        print("\n=== Test 1: Relative CW 45° ===")
        rotate_wrist(skills, angle_deg=45, skill_description="test: CW 45 deg")

        # 테스트 2: 상대 모드 — 반시계 90도
        print("\n=== Test 2: Relative CCW 90° ===")
        rotate_wrist(skills, angle_deg=-90, skill_description="test: CCW 90 deg")

        # 테스트 3: 절대 모드 — 중립 복귀
        print("\n=== Test 3: Absolute 0° (neutral) ===")
        rotate_wrist(skills, angle_deg=0, absolute=True, skill_description="test: return to neutral")

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
