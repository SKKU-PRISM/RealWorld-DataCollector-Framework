"""
move_linear — Cartesian 직선 경로 이동 스킬

기존 move_to_position은 joint-space 보간이라 EE가 곡선을 그림.
move_linear는 Cartesian 공간에서 직선 보간 → waypoint별 IK → EE 직선 경로 보장.

동작 원리:
    1. start-end 사이를 num_waypoints개의 Cartesian 직선 점으로 분할
    2. 각 waypoint마다 IK를 풀어 joint trajectory 생성
    3. cosine smoothing으로 시작/끝 가감속 적용
    4. 50Hz 제어 루프로 waypoint 간 시간 보간하며 실행
    5. 마지막 waypoint 도달 후 Hold phase로 위치 유지

접촉 동작 (push, sweep, fold, wipe 등)에서 EE 직선 궤적이 필요할 때 사용.
push_object, wipe 등 상위 스킬의 핵심 빌딩블록.

move_to_position vs move_linear:
    - move_to_position: joint-space 보간 → EE가 곡선 경로 → 빠름, 일반 이동용
    - move_linear:      Cartesian 보간 → EE가 직선 경로 → 접촉/정밀 동작용

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.move_linear import move_linear

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    # 예시 1: y축 방향 직선 이동
    move_linear(
        skills,
        start=[0.20, -0.05, 0.02],
        end=[0.20, 0.05, 0.02],
        skill_description="sweep left to right",
    )

    # 예시 2: 대각선 직선 이동 (느리게)
    move_linear(
        skills,
        start=[0.15, -0.03, 0.05],
        end=[0.25, 0.03, 0.05],
        duration=5.0,              # 5초에 걸쳐 천천히
        num_waypoints=30,          # 더 촘촘한 waypoint
    )

    skills.disconnect()
"""

import time
from typing import List, Optional, Union

import numpy as np


def _cartesian_ik_trajectory(
    skills,
    waypoints_base: np.ndarray,
    current_joints: np.ndarray,
    maintain_pitch: bool,
):
    """
    Cartesian waypoints → per-waypoint IK → joint trajectory.

    Args:
        skills: LeRobotSkills 인스턴스
        waypoints_base: (N, 3) base_link frame 좌표
        current_joints: 현재 joint positions (radians, 5축)
        maintain_pitch: pitch 유지 여부

    Returns:
        joint_trajectory: (N, 5) radians
    """
    kinematics = skills.planner.kinematics

    # Get current pitch for constraint
    target_pitch = None
    if maintain_pitch:
        target_pitch = kinematics.get_gripper_pitch(current_joints)

    # Fixed wrist_roll
    fixed_joints = [4]

    # Calibration limits
    custom_limits = None
    if skills.planner.calibration_limits is not None:
        custom_limits = (
            skills.planner.calibration_limits.lower_limits_radians,
            skills.planner.calibration_limits.upper_limits_radians,
        )

    joint_trajectory = []
    prev_joints = current_joints

    for i, wp in enumerate(waypoints_base):
        target_joints, success, ik_info = kinematics.inverse_kinematics_multi(
            wp,
            current_joints=prev_joints,
            custom_limits=custom_limits,
            num_random_samples=10,
            verbose=False,
            fixed_joints=fixed_joints,
            target_pitch=target_pitch,
        )

        if not success or ik_info["num_valid"] == 0:
            raise RuntimeError(
                f"[move_linear] IK failed at waypoint {i+1}/{len(waypoints_base)}: "
                f"[{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]"
            )

        joint_trajectory.append(target_joints)
        prev_joints = target_joints

    return np.array(joint_trajectory)


def move_linear(
    skills,
    start: Union[List[float], np.ndarray],
    end: Union[List[float], np.ndarray],
    duration: Optional[float] = None,
    maintain_pitch: bool = True,
    num_waypoints: int = 20,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    Cartesian 직선 경로로 start에서 end까지 이동.

    joint-space 보간이 아닌 Cartesian 보간 → waypoint별 IK로 EE 직선 경로 보장.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        start: 시작 위치 [x, y, z] (단위: meters)
               - skills 초기화 시 지정한 frame 좌표계 기준 (보통 "world")
               - 직선 경로의 시작점
               - 예: [0.20, -0.05, 0.02] → x=20cm, y=-5cm, z=2cm

        end: 끝 위치 [x, y, z] (단위: meters)
             - start와 동일한 좌표계
             - 직선 경로의 끝점
             - start와 end 사이를 직선으로 이동

        duration: 이동 시간 (단위: 초). default=None
                  - None: 거리 기반 자동 계산 (5cm/s 기준, 최소 2초)
                  - 작을수록 빠르게 이동 (단, 너무 빠르면 추종 오차 증가)
                  - 접촉 동작에서는 2~5초 권장

        maintain_pitch: 이동 중 pitch 유지 여부. default=True
                        - True: 이동 시작 시점의 gripper pitch를 전 구간 유지
                        - False: IK solver가 자유롭게 pitch 결정
                        - 접촉 동작(push, wipe)에서는 True 권장

        num_waypoints: Cartesian 직선 분할 수. default=20
                       - 직선을 몇 개의 중간점으로 나눌지 결정
                       - 많을수록 직선에 가깝지만 IK 계산 시간 증가
                       - 10cm 이하: 15~20, 10cm 이상: 20~30 권장
                       - 각 waypoint마다 IK를 풀기 때문에 너무 크면 초기화 느림

        target_name: 대상 물체 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용
                     - 예: "table surface", "red block"

        skill_description: 스킬 동작 설명 (선택). default=None
                           - 레코딩 시 스킬 라벨에 사용
                           - 예: "sweep left to right", "push block forward"

    Returns:
        bool: True면 목표 위치 도달 성공 (5mm 이내 또는 15mm 이내 수렴)
              False면 IK 실패, 도달 불가, 또는 workspace 밖
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    # Duration: 거리 기반 자동 계산 (없으면)
    distance = np.linalg.norm(end - start)
    if duration is None:
        duration = max(distance / 0.05, 2.0)  # 5cm/s 기준, 최소 2초

    skills._log(f"\n[move_linear] {start} -> {end}")
    skills._log(f"  Distance: {distance*100:.1f}cm, Duration: {duration:.1f}s")
    skills._log(f"  Waypoints: {num_waypoints}, maintain_pitch: {maintain_pitch}")

    # 1. Cartesian waypoints 생성 (직선 보간)
    alphas = np.linspace(0, 1, num_waypoints)
    # Cosine smoothing for start/end deceleration
    smooth_alphas = (1 - np.cos(alphas * np.pi)) / 2
    waypoints_frame = np.array([start + a * (end - start) for a in smooth_alphas])

    # 2. World → base_link 변환
    waypoints_base = np.array([
        skills._transform_pos_world2robot(wp) for wp in waypoints_frame
    ])

    # 3. Reachability check (first and last)
    kinematics = skills.planner.kinematics

    for label_check, wp in [("start", waypoints_base[0]), ("end", waypoints_base[-1])]:
        if not kinematics.is_position_reachable(wp):
            skills._log(
                f"ERROR: {label_check} position [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}] "
                f"outside reachable workspace"
            )
            return False

    # 4. Current state
    current_arm_norm, current_joints, _ = skills._get_current_state()

    # 5. Per-waypoint IK
    try:
        joint_trajectory = _cartesian_ik_trajectory(
            skills, waypoints_base, current_joints,
            maintain_pitch=maintain_pitch,
        )
    except RuntimeError as e:
        skills._log(f"ERROR: {e}")
        return False

    # 6. Set skill recording
    label = skill_description or (
        f"linear move to {target_name}" if target_name else "linear move"
    )

    try:
        from record_dataset.context import RecordingContext
        has_recording = True
    except ImportError:
        has_recording = False

    if has_recording and RecordingContext.is_active():
        goal_joint_rad = joint_trajectory[-1]
        skills._set_skill_recording(
            label=label,
            skill_type="move_linear",
            goal_joint_5=goal_joint_rad,
            goal_gripper=skills.current_gripper_pos,
            kinematics=kinematics,
        )

    # 7. Execute trajectory (50Hz control loop)
    try:
        success = _execute_linear_trajectory(
            skills,
            joint_trajectory=joint_trajectory,
            waypoints_base=waypoints_base,
            duration=duration,
            target_position=waypoints_base[-1],
            kinematics=kinematics,
        )
    finally:
        skills._clear_skill_recording()

    return success


def _execute_linear_trajectory(
    skills,
    joint_trajectory: np.ndarray,
    waypoints_base: np.ndarray,
    duration: float,
    target_position: np.ndarray,
    kinematics,
) -> bool:
    """
    Per-waypoint joint trajectory를 시간에 맞춰 실행.

    기존 _execute_trajectory와 유사하지만, 다수 waypoint를 순차 보간.
    """
    num_waypoints = len(joint_trajectory)
    timestamps = np.linspace(0, duration, num_waypoints)

    POSITION_TOLERANCE = 0.005  # 5mm
    MAX_TOTAL_TIME = duration + 2.0
    SETTLE_TIME = 0.2

    start_time = time.time()
    target_reached = False
    reach_time = None

    while True:
        elapsed = time.time() - start_time

        # Read current state
        actual_norm, actual_rad, current_ee = skills._get_current_state(kinematics)
        position_error = np.linalg.norm(target_position - current_ee)

        if elapsed < duration:
            # Trajectory phase: 시간에 맞는 waypoint 보간
            # Find surrounding waypoints
            idx = np.searchsorted(timestamps, elapsed)
            idx = min(idx, num_waypoints - 1)

            if idx == 0:
                arm_positions_rad = joint_trajectory[0]
            else:
                t0, t1 = timestamps[idx - 1], timestamps[idx]
                q0, q1 = joint_trajectory[idx - 1], joint_trajectory[idx]
                alpha = (elapsed - t0) / (t1 - t0) if t1 > t0 else 1.0
                arm_positions_rad = q0 + alpha * (q1 - q0)

            arm_normalized = skills._radians_to_normalized(arm_positions_rad)
            phase = "Traj"
        else:
            # Hold phase: 마지막 waypoint 유지
            arm_normalized = skills._radians_to_normalized(joint_trajectory[-1])
            phase = "Hold"

        # Apply compensation
        if skills.use_compensation and skills.compensator:
            arm_normalized = skills.compensator.compensate(actual_norm, arm_normalized)

        # Send command
        arm_normalized = np.clip(arm_normalized, -99.0, 99.0)
        full_normalized = np.concatenate([arm_normalized, [skills.current_gripper_pos]])
        skills.robot.write_positions(full_normalized, normalize=True)

        # Recording callback (Traj phase only)
        if skills.recording_callback is not None and phase == "Traj":
            try:
                state_full = skills.robot.read_positions(normalize=True)
                skills.recording_callback(state_full.copy(), full_normalized.copy())
            except Exception:
                pass

        # Progress display
        if skills.verbose:
            progress = min(elapsed / duration, 1.0)
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"\r  [{bar}] {phase} err:{position_error*1000:6.1f}mm",
                end="", flush=True,
            )

        # Check target reached
        if position_error < POSITION_TOLERANCE:
            if reach_time is None:
                reach_time = time.time()
            elif time.time() - reach_time > SETTLE_TIME:
                target_reached = True
                break
        else:
            reach_time = None

        # Timeout
        if elapsed > MAX_TOTAL_TIME:
            if skills.verbose:
                print(f"\n  Timeout after {MAX_TOTAL_TIME:.1f}s")
            break

        time.sleep(0.02)  # 50Hz

    if skills.verbose:
        if target_reached:
            print(f"\r  [{'=' * 30}] Done (err: {position_error*1000:.1f}mm)    ")
        else:
            print(f"\r  [{'=' * 30}] Timeout (err: {position_error*1000:.1f}mm)")

    # Final error
    _, final_rad, final_ee = skills._get_current_state(kinematics)
    skills.last_error = skills._calculate_error(
        target_position=target_position,
        actual_position=final_ee,
    )
    skills._print_error(skills.last_error, "move_linear")

    return target_reached or position_error < POSITION_TOLERANCE * 3


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

        # 테스트: y축 방향 직선 이동 (10cm 직선)
        success = move_linear(
            skills,
            start=[0.20, -0.05, 0.10],
            end=[0.20, 0.05, 0.10],
            duration=3.0,
            skill_description="test: linear move along y-axis",
        )
        print(f"Result: {'success' if success else 'failed'}")

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
