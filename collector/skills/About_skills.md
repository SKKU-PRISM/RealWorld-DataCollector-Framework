  ---
  [2] skills.py 구조 분석

  전체 아키텍처

  PrimitiveSkill (Franka용 스킬 클래스)
  ├── 의존성 모듈
  │   ├── MoveGroupControl (MoveIt 기반 로봇 제어)
  │   ├── GripperInterface (Panda 그리퍼 제어)
  │   └── ROS2 utilities (rclpy 초기화)
  ├── 유틸리티 함수
  │   ├── interpolate_pose() - 선형 보간
  │   ├── interpolate_pose_quaternion() - Quaternion SLERP 보간
  │   ├── _clamp_min_z() - Z축 최소값 제한
  │   └── _execute_waypoints() - waypoint 실행
  ├── Pose 설정 함수
  │   ├── setPose0/1() - Euler 기반
  │   ├── setPose0/1Quaternion() - Quaternion 기반
  │   └── setTargetPose/Quaternion()
  ├── Pose 읽기 함수
  │   ├── getPose() - Euler 반환
  │   └── getPoseQuaternion() - Quaternion 반환
  └── Primitive Skills (기본 동작)
      ├── execute_go() - 단순 이동
      ├── execute_pick() - 물체 집기
      ├── execute_place() - 물체 놓기
      ├── execute_pick_and_place() - 집고 놓기
      ├── execute_push() - 밀기
      ├── execute_pull() - 당기기
      ├── execute_sweep() - 쓸기
      ├── execute_rotate() - 회전
      └── execute_gripper() - 그리퍼 제어

  핵심 컴포넌트 상세

  1. 생성자 파라미터 (__init__)

  | 파라미터              | 기본값 | 설명                    |
  |-----------------------|--------|-------------------------|
  | gripper_offset        | 0.05m  | 그리퍼 접근 높이 오프셋 |
  | intermediate_z_stop   | 0.6m   | 안전 높이 (중간 경유점) |
  | intermediate_distance | 0.09m  | 측면 접근 거리          |
  | speed                 | 0.05   | 이동 속도               |
  | waypoint_density      | 5      | 보간 밀도               |

  2. 이동 방식 비교

  Euler 기반 (기존)
  # setPose0(): roll+pi/4 보정 적용
  self.pose0 = [x, y, z, roll + pi/4, pitch, yaw]

  # execute_go(): MoveIt Cartesian path 또는 joint space planning
  move_group.follow_cartesian_path([self.target_pose])

  Quaternion 기반 (개선)
  # setTargetPoseQuaternion(): singularity 회피
  self.target_pose_quat = {
      'position': (x, y, z),
      'quaternion': (qx, qy, qz, qw)
  }

  # interpolate_pose_quaternion(): SLERP 보간
  theta = np.arccos(dot)
  quat = (sin((1-t)*theta)*q0 + sin(t*theta)*q1) / sin(theta)

  3. Primitive Skill 동작 패턴

  모든 skill은 동일한 패턴을 따릅니다:

  1. 현재 위치 읽기 (_get_current_pose)
        ↓
  2. 중간 경유점 계산 (intermediate_pose)
     - z = intermediate_z_stop (안전 높이)
     - x/y = target ± intermediate_distance
        ↓
  3. Waypoint 보간 (interpolate_pose)
        ↓
  4. Z축 클램핑 (_clamp_min_z)
        ↓
  5. 경로 실행 (_execute_waypoints)
     - Cartesian path 시도
     - 실패시 pose goal fallback
        ↓
  6. 그리퍼 동작 (gripper.grasp)
        ↓
  7. Ready pose 복귀 (선택적)

  4. 각 Skill 특성

  | Skill          | 설명      | 특이사항                              |
  |----------------|-----------|---------------------------------------|
  | execute_pick   | 물체 집기 | gripper_offset만큼 위에서 접근        |
  | execute_place  | 물체 놓기 | target 높이까지 하강                  |
  | execute_push   | 밀기      | axis 방향으로 distance만큼 밀기       |
  | execute_pull   | 당기기    | 그리퍼로 잡고 axis 방향 반대로 당기기 |
  | execute_sweep  | 쓸기      | axis 방향으로 sweep_width만큼 왕복    |
  | execute_rotate | 회전      | 마지막 관절 90° 회전                  |

  ---
  [3] LeRobot 변환을 위한 필요 작업

  대체해야 할 컴포넌트

  | Franka (현재)             | LeRobot (변환 후)                     |
  |---------------------------|---------------------------------------|
  | MoveGroupControl (MoveIt) | TrajectoryPlanner + FeetechController |
  | GripperInterface (Panda)  | SO-101 그리퍼 제어 (motor_id 6)       |
  | rclpy / ROS2 노드         | 불필요 (직접 제어)                    |
  | Quaternion/Euler pose     | Position-only (orientation 고정)      |
  | Cartesian path planning   | Joint-space trajectory                |

  변환 작업 목록

  A. 하드웨어 인터페이스 변환

  # Franka (ROS2/MoveIt)
  self.moveit_control = MoveGroupControl(speed, node=self.node)
  self.gripper = GripperInterface(node=self.node)

  # LeRobot (Direct)
  self.controller = FeetechController(...)  # 모터 제어
  self.kinematics = KinematicsEngine(...)   # FK/IK
  self.planner = TrajectoryPlanner(...)     # 궤적 계획

  B. 좌표계 및 Pose 처리

  # Franka: 6DOF Pose (x, y, z, roll, pitch, yaw)
  self.target_pose = [x, y, z, roll + pi/4, pitch, yaw]

  # LeRobot: 3DOF Position (orientation 고정)
  self.target_position = np.array([x, y, z])
  # SO-101은 5DOF이므로 orientation 자유도 제한됨

  C. 경로 실행 방식

  # Franka: MoveIt Cartesian/Joint planning
  move_group.follow_cartesian_path(waypoints)
  move_group.go_to_pose_goal(x, y, z, r, p, y)

  # LeRobot: Trajectory + Direct motor control
  trajectory = self.planner.plan_to_position(target_pos, current_joints)
  for joints in trajectory.joint_positions:
      normalized = calibration_limits.radians_to_normalized(joints)
      controller.write_positions(normalized)
      time.sleep(dt)

  D. 그리퍼 제어

  # Franka: 별도 인터페이스
  self.gripper.grasp(width=0.01, force=5)

  # LeRobot: 6번 모터 직접 제어
  gripper_position = ...  # normalized value
  controller.write_positions([..., gripper_position])

  추가로 필요한 구현

  1. 안전 높이 조정
    - Franka: intermediate_z_stop = 0.6m
    - SO-101: 작업공간에 맞게 조정 필요 (약 0.15~0.25m)
  3. 속도/가속도 파라미터
    - Franka: MoveIt이 자동 계산
    - LeRobot: TrajectoryPlanner의 max_velocity, max_acceleration 설정
  4. 에러 처리
    - IK 실패시 처리 로직
    - 관절 한계 위반 처리

---

## [4] LeRobot Skills 구현 (skills_lerobot.py)

### 파일 구조

```
skills/
├── skills.py           # Franka 원본 (참고용)
├── skills_lerobot.py   # LeRobot 구현 (SO-101용)
└── About_skills.md     # 이 문서
```

### LeRobotSkills 클래스 구조

```
LeRobotSkills
├── 의존성 모듈 (LeRobot 전용)
│   ├── FeetechController    - Feetech 모터 직접 제어
│   ├── KinematicsEngine     - Pinocchio FK/IK
│   ├── TrajectoryPlanner    - 궤적 계획
│   └── AdaptiveCompensator  - 보상 (백래시/중력)
├── 유틸리티 함수
│   ├── _normalized_to_radians() - 정규화 → 라디안
│   ├── _radians_to_normalized() - 라디안 → 정규화
│   ├── _transform_position()    - 좌표계 변환
│   ├── _apply_end_deceleration() - 종단 감속
│   ├── _get_current_state()     - 현재 상태 읽기
│   ├── _calculate_error()       - 오차 계산 (x,y,z,r,p,y 축)
│   └── _print_error()           - 오차 출력 (mm 단위)
├── 그리퍼 제어
│   ├── gripper_open()  - 그리퍼 열기
│   └── gripper_close() - 그리퍼 닫기
├── 이동 함수
│   ├── move_to_initial_state() - 초기 상태로 이동
│   ├── move_to_position()      - 목표 위치로 이동
│   ├── move_to_free_state()    - Free 상태로 이동 (안전 주차)
│   └── execute_go()            - 단순 이동 (alias)
└── Primitive Skills
    ├── execute_pick()          - 물체 집기
    ├── execute_place()         - 물체 놓기
    ├── execute_pick_and_place() - 집고 놓기 전체 시퀀스
    └── rotate_90degree()       - 그리퍼 90도 회전
```

### 사용법

```python
from skills.skills_lerobot import LeRobotSkills

# 초기화
skills = LeRobotSkills(
    robot_config="robot_configs/robot/so101_robot3.yaml",
    frame="base_link",            # 좌표계 (base_link)
    gripper_open_pos=85.0,        # 그리퍼 열림 위치
    gripper_close_pos=-70.0,      # 그리퍼 닫힘 위치
    movement_duration=3.0,        # 이동 시간
    use_compensation=True,        # 보상 활성화
    use_deceleration=True,        # 종단 감속 활성화
    gripper_y_offset=0.015,       # 비대칭 그리퍼 Y offset (15mm)
)

# 연결
skills.connect()

# Pick and Place 실행
skills.execute_pick_and_place(
    pick_position=[0.15, 0.05, 0.02],   # world 좌표
    place_position=[0.15, -0.05, 0.02],
    approach_height=0.05,                # 접근 높이 (위에서 내려오기)
)

# 개별 동작
skills.move_to_initial_state()
skills.gripper_open()
skills.move_to_position([0.2, 0.0, 0.1])
skills.gripper_close()

# 그리퍼 90도 회전
skills.rotate_90degree(direction=1)   # 시계 방향
skills.rotate_90degree(direction=-1)  # 반시계 방향

# Free 상태로 이동 (안전 주차)
skills.move_to_free_state()

# 연결 해제
skills.disconnect()

# 오차 확인
print(f"Last error: {skills.last_error}")
```

### 실행 스크립트

```bash
# Pick and Place 데모
./run_pick_and_place.sh

# 환경 변수로 위치 지정
PICK_X=0.2 PICK_Y=0.05 PICK_Z=0.02 \
PLACE_X=0.2 PLACE_Y=-0.05 PLACE_Z=0.02 \
./run_pick_and_place.sh

# 커맨드라인 옵션
./run_pick_and_place.sh \
    --pick-x 0.15 --pick-y 0.05 --pick-z 0.02 \
    --place-x 0.15 --place-y -0.05 --place-z 0.02 \
    --approach-height 0.05 \
    --duration 3.0
```

### Pick and Place 동작 순서

```
[Step 0] Initial State로 이동
    ↓
[Step 1] Execute Pick
    1-1. 그리퍼 열기
    1-2. Pick 위치 위로 이동 (approach_height)
    1-3. Pick 위치로 하강
    1-4. 그리퍼 닫기
    1-5. 들어올리기
    ↓
[Step 2] Execute Place
    2-1. Place 위치 위로 이동
    2-2. Place 위치로 하강
    2-3. 그리퍼 열기
    2-4. 들어올리기
    ↓
[Step 3] Initial State로 복귀
```

### Franka vs LeRobot 비교

| 항목 | Franka (skills.py) | LeRobot (skills_lerobot.py) |
|------|-------------------|----------------------------|
| 하드웨어 | MoveIt + ROS2 | FeetechController 직접 제어 |
| 그리퍼 | GripperInterface | Motor 6 직접 제어 |
| 좌표계 | 6DOF (position + orientation) | 6DOF (position + fixed orientation) |
| IK | MoveIt IK | Pinocchio + Multi-IK with orientation |
| 궤적 | Cartesian path | Joint-space trajectory |
| 보상 | MoveIt 자동 처리 | AdaptiveCompensator |
| 좌표 변환 | TF2 | Pix2Robot (direct calibration) |

### move_to_position 동작 방식

**v2.0 업데이트**: `move_to_position`은 이제 wrist_roll을 0 rad로 고정합니다.

5-DOF 로봇(SO-101)에서는 full 6-DOF orientation 제어가 불가능합니다.
대신 wrist_roll 관절을 고정하여 그리퍼 회전을 방지합니다.

```python
# 이전 (position-only IK)
# - IK가 자유롭게 wrist_roll을 결정
# - 이동 중 그리퍼가 ~90도 회전할 수 있음

# 현재 (wrist_roll 고정)
# - 현재 wrist_roll 값 유지
# - 이동 중 그리퍼 회전 없음

# 예시: rotate_90degree 후 move_to_position
skills.rotate_90degree(direction=1)       # wrist_roll = 90°
skills.move_to_position([0.2, 0.0, 0.1])  # wrist_roll = 90° 유지
skills.move_to_position([0.3, 0.0, 0.1])  # wrist_roll = 90° 유지
```

**파라미터**:
- `position`: 목표 위치 [x, y, z] (meters)
- `duration`: 이동 시간 (기본값: 3.0초)
- `maintain_wrist_roll`: True면 현재 wrist_roll 값 유지 (기본값: True)
- `gripper_offset`: 비대칭 그리퍼 offset 값 (meters, 기본값: 0.0)
- `maintain_pitch`: True면 현재 pitch 값 유지 (기본값: False)

### Asymmetric Gripper Offset (v2.2)

SO-101은 비대칭 그리퍼 구조입니다:
- **고정 핑거**: 로봇 프레임 +Y 방향
- **가동 핑거**: 로봇 프레임 -Y 방향

물체를 집을 때 검출된 위치 그대로 접근하면 고정 핑거와 충돌합니다.
`gripper_offset=value`를 사용하면 가동 핑거 방향(-Y)으로 지정된 거리만큼 offset이 적용됩니다.

**v2.2 업데이트**: offset 값은 물체 크기에 따라 동적으로 계산됩니다.
- detection 시 bbox 크기를 기반으로 `gripper_offset` 값이 자동 계산됨
- 계산 공식: `offset = bbox_height / 2 + 1cm` (이미지 y축 폭 기준)
- 범위: 1cm ~ 4cm (안전 제한)

```python
# positions 딕셔너리 구조
# {name: {"position": [x,y,z], "gripper_offset": float, ...}}

pick_obj = positions["object_name"]
pick_pos = pick_obj["position"]
pick_offset = pick_obj["gripper_offset"]  # 물체 크기 기반 동적 값

# Pick 시 gripper offset 적용
skills.move_to_position(pick_pos, gripper_offset=pick_offset)  # 충돌 방지

# Place 시에는 offset 불필요
skills.move_to_position(place_pos)  # offset 없이 이동

# 물체를 들고 이동할 때는 maintain_pitch=True 사용
skills.move_to_position(place_approach, maintain_pitch=True)
```

**Offset 변환**:
- offset은 로봇 프레임에서 정의됨 (meters 단위)
- world 좌표계로 변환 시 회전만 적용 (offset은 방향 벡터이므로)
- `offset_world = R_world_from_base @ offset_robot`