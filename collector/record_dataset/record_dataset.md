# 데이터셋 Recording 상세 문서

## 1. 전체 Recording 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Recording System Flow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  run_forward_and_reset.sh                                               │
│         ↓                                                               │
│  execution_forward_and_reset.py                                         │
│         ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ForwardAndResetPipeline._init_recording()                        │   │
│  │   ├─ DatasetRecorder 생성                                        │   │
│  │   └─ MultiCameraManager 연결 (realsense, innomaker)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ _execute_code_with_recording()                                   │   │
│  │   ├─ RecordingContext.setup()    ← 전역 컨텍스트 설정            │   │
│  │   ├─ AsyncCameraCapture.start()  ← 비동기 카메라 캡처 시작 (60Hz)│   │
│  │   └─ exec(generated_code)        ← 생성된 코드 실행              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LeRobotSkills (자동으로 콜백 획득)                               │   │
│  │   └─ 제어 루프 내에서 recording_callback() 호출                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 각 데이터별 Recording 방식

### 2.1 State (observation.state)

| 항목 | 내용 |
|------|------|
| **데이터** | 로봇 관절 위치 6개 (5 arm + 1 gripper) |
| **취득 방식** | `self.robot.read_positions(normalize=True)` |
| **취득 시점** | Action을 로봇에 write한 **직후** |
| **소스** | 로봇 모터 엔코더 (실제 위치) |
| **Shape** | `(6,)` float32 |
| **범위** | -100.0 ~ +100.0 (normalized) |

**코드 위치** (`skills/skills_lerobot.py:771-773`):
```python
actual_norm = self.robot.read_positions(normalize=True)[:5]
current_state_full = np.concatenate([actual_norm, [self.current_gripper_pos]])
```

### 2.2 Action

| 항목 | 내용 |
|------|------|
| **데이터** | 타겟 관절 위치 6개 (5 arm + 1 gripper) |
| **취득 방식** | 트라젝토리 보간 계산 결과 복사 |
| **취득 시점** | 로봇에 write하는 **동시** |
| **소스** | 트라젝토리 플래너 계산 결과 |
| **Shape** | `(6,)` float32 |
| **범위** | -100.0 ~ +100.0 (normalized) |

**코드 위치** (`skills/skills_lerobot.py:774`):
```python
action_full = full_normalized.copy()
```

### 2.3 Images (observation.images.*)

| 항목 | 내용 |
|------|------|
| **데이터** | RGB 이미지 (realsense, innomaker) |
| **취득 방식** | AsyncCameraCapture에서 최신 이미지 획득 |
| **취득 시점** | recording_callback() 호출 시점 |
| **동기화** | 비동기 (0~16ms 지연 가능) |
| **Shape** | `(480, 640, 3)` uint8 |

**코드 위치** (`record_dataset/context.py:331-340`):
```python
# 비동기 캡처 사용 (블로킹 없음!)
if cls._async_capture is not None:
    images = cls._async_capture.get_latest_images()  # 즉시 반환
```

---

## 3. Recording 타이밍 상세

### 3.1 제어 루프 구조 (50Hz)

```python
# skills/skills_lerobot.py - _execute_planned_trajectory()

while True:
    # 1. 현재 상태 읽기
    actual_norm, actual_rad, current_ee = self._get_current_state()

    # 2. 트라젝토리 보간 (Traj phase에서만)
    if elapsed < duration:
        arm_positions_rad = trajectory.get_state_at_time(elapsed)
        phase = "Traj"
    else:
        phase = "Hold"  # ← 이 phase에서는 레코딩 스킵!

    # 3. 로봇에 명령 전송
    self.robot.write_positions(full_normalized, normalize=True)

    # 4. Recording Callback (Traj phase에서만!)
    if self.recording_callback is not None and phase == "Traj":
        current_state_full = np.concatenate([actual_norm, [gripper_pos]])
        action_full = full_normalized.copy()
        self.recording_callback(current_state_full, action_full)

    time.sleep(0.02)  # 50Hz
```

### 3.2 FPS 동기화 (50Hz → 30fps)

```
제어 루프: 50Hz (20ms 간격)
레코딩 FPS: 30fps (33.3ms 간격)
스킵 비율: 50/30 ≈ 1.67

→ 약 5 control steps마다 3 frames 레코딩
```

**코드 위치** (`record_dataset/context.py:239-240`):
```python
target_frame = int(cls._step_counter / cls._frame_skip_ratio)
return target_frame > cls._last_record_step
```

---

## 4. 비동기 카메라 캡처

### 4.1 구조

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   Background Thread (60Hz)  │     │   Main Thread (50Hz)        │
├─────────────────────────────┤     ├─────────────────────────────┤
│                             │     │                             │
│  while not stop:            │     │  # 제어 루프               │
│    images = camera.read()   │     │  robot.write(action)        │
│    with lock:               │     │                             │
│      _latest_images = images│←────│  imgs = get_latest_images() │
│    sleep(1/60)              │     │  record(state, action, imgs)│
│                             │     │                             │
└─────────────────────────────┘     └─────────────────────────────┘
```

### 4.2 장점

| 기존 (동기) | 현재 (비동기) |
|------------|--------------|
| 카메라 읽기 ~25ms 블로킹 | 블로킹 없음 (<1ms) |
| 제어 루프 ~25Hz | 제어 루프 50Hz 유지 |

---

## 5. 데이터 흐름 타임라인

```
T=0ms    [Control Step 0]
         ├─ robot.write(action_0)      → Action 기록
         ├─ robot.read() → state_0     → State 기록
         ├─ get_latest_images()        → Images 기록 (비동기)
         └─ should_record()? YES       → Frame 0 저장

T=20ms   [Control Step 1]
         ├─ robot.write(action_1)
         ├─ robot.read() → state_1
         └─ should_record()? NO        → 스킵 (FPS 동기화)

T=40ms   [Control Step 2]
         ├─ robot.write(action_2)
         ├─ robot.read() → state_2
         ├─ get_latest_images()
         └─ should_record()? YES       → Frame 1 저장

...반복...
```

---

## 6. 레코딩되는/안되는 구간

| 구간 | 레코딩 여부 | 이유 |
|------|------------|------|
| **Traj Phase** (이동 중) | ✅ 레코딩 | 유효한 학습 데이터 |
| **Hold Phase** (settling) | ❌ 스킵 | 정지 상태, 불필요 |
| **IK 계산 중** | ❌ 스킵 | 제어 루프 밖 |
| **그리퍼 동작** | ✅ 레코딩 | arm 정지해도 gripper 동작 |
| **스킬 함수 사이** | ❌ 스킵 | 제어 루프 밖 |

---

## 7. 최종 저장 형식

```
~/.cache/huggingface/lerobot/local/{repo_id}/
├── data/
│   └── chunk-000/
│       └── file-000.parquet    ← state, action, timestamps
├── videos/
│   ├── observation.images.realsense/
│   │   └── chunk-000/
│   │       └── episode_000000.mp4
│   └── observation.images.innomaker/
│       └── chunk-000/
│           └── episode_000000.mp4
└── meta/
    ├── info.json               ← 데이터셋 메타정보
    ├── stats.json              ← 통계
    └── tasks.json              ← 태스크 목록
```

---

## 8. 관련 코드 파일 요약

| 파일 | 역할 |
|------|------|
| `record_dataset/recorder.py` | DatasetRecorder - LeRobot 데이터셋 생성/저장 |
| `record_dataset/context.py` | RecordingContext - 전역 컨텍스트, FPS 동기화 |
| `record_dataset/async_camera.py` | AsyncCameraCapture - 비동기 카메라 캡처 |
| `skills/skills_lerobot.py` | LeRobotSkills - 제어 루프에서 콜백 호출 |
| `execution_forward_and_reset.py` | 파이프라인 - 레코딩 초기화/종료 관리 |

---

## 9. Recording 데이터 예시

### 9.1 단일 프레임

```python
frame = {
    "observation.state": np.array([12.5, -45.2, 67.8, -23.1, 5.6, 80.0], dtype=np.float32),
    "action": np.array([13.2, -44.8, 68.5, -22.5, 5.8, 80.0], dtype=np.float32),
    "observation.images.realsense": image_realsense,  # (480, 640, 3) uint8
    "observation.images.innomaker": image_innomaker,  # (480, 640, 3) uint8
    "task": "pick up the red block and place it on the blue dish",
}
```

### 9.2 관절 순서

| Index | 관절 이름 | 설명 |
|-------|----------|------|
| 0 | shoulder_pan.pos | 어깨 회전 |
| 1 | shoulder_lift.pos | 어깨 상하 |
| 2 | elbow_flex.pos | 팔꿈치 굽힘 |
| 3 | wrist_flex.pos | 손목 굽힘 |
| 4 | wrist_roll.pos | 손목 회전 |
| 5 | gripper.pos | 그리퍼 개폐 |
