# RoboBridge - 범용 로봇 조작 프레임워크

## 프로젝트 목적

RoboBridge는 **범용 로봇 조작 프레임워크(General Robot Manipulation Framework)**를 목표로 합니다.

핵심 목표:
- **자연어 기반 로봇 제어**: 사람이 말하듯이 로봇에게 명령 ("빨간 컵을 집어서 테이블에 놓아줘")
- **하드웨어 독립성**: Franka, UR, 시뮬레이션 등 다양한 로봇에서 동일한 코드로 동작
- **모듈 교체 가능**: 인식, 계획, 제어 모듈을 독립적으로 교체 가능
- **멀티 프로바이더 지원**: OpenAI, Anthropic, Google, Ollama, 로컬 HuggingFace 모델 모두 지원

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     자연어 입력 (Natural Language)                │
│                "빨간 컵을 집어서 테이블 위에 놓아줘"                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RoboBridge Core                          │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Perception │  │  Planner   │  │ Controller │  │   Robot    │ │
│  │  (인식)    │─▶│  (계획)    │─▶│  (제어)    │─▶│ (하드웨어) │ │
│  │ Florence-2 │  │ GPT-4/Qwen │  │ VLA/기본   │  │ Franka/Sim │ │
│  └────────────┘  └─────┬──────┘  └────────────┘  └──────┬─────┘ │
│         │              │                                │       │
│         │              │  ┌────────────────────────┐    │       │
│         │              │  │      Recovery Flow     │    │       │
│         │              │  │ ┌──────────────────┐   │    │       │
│         │              └──┼─│ 실패 시 재계획    │◀──┼────┘       │
│         │                 │ │ (Replan on Fail) │   │            │
│         │                 │ └──────────────────┘   │            │
│         │                 └────────────────────────┘            │
│         │                          ▲                            │
│         │         ┌────────────────┴───────────────┐            │
│         │         │          Monitor               │            │
│         └────────▶│  (실시간 실행 모니터링 + 피드백) │◀───────────┤
│                   │  - 10Hz 연속 관찰              │            │
│                   │  - 실패 감지 → 로봇 정지       │            │
│                   │  - 피드백 → Planner 재계획     │            │
│                   └────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 실행 흐름 (Execution Flow)

```
1. 정상 흐름 (Forward Flow)
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │Perception│───▶│ Planner  │───▶│Controller│───▶│  Robot   │
   │ 물체인식 │    │ 계획생성 │    │ 궤적생성 │    │ 명령실행 │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘

2. 피드백 흐름 (Feedback Flow) - Monitor가 실패 감지 시
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Robot   │───▶│ Monitor  │───▶│ Planner  │
   │ 실행상태 │    │ 실패감지 │    │ 재계획   │
   └──────────┘    └──────────┘    └──────────┘
                         │
                         ▼
                   ┌──────────┐
                   │  Robot   │
                   │ 즉시정지 │
                   └──────────┘

3. 복구 전략 (Recovery Strategy)
   Monitor 실패 감지 → 복구 대상 결정:
   
   ├─ "perception" → Perception부터 다시 (물체 위치 재확인)
   ├─ "planning"   → Planner부터 다시 (새로운 계획 수립)  
   ├─ "controller" → Controller부터 다시 (궤적 재생성)
   └─ "retry"      → 현재 스텝만 재시도
```

---

## 핵심 모듈 설명

### 1. Perception (인식 모듈)
**역할**: 카메라 이미지에서 물체를 감지하고 위치를 추정

| 항목 | 설명 |
|------|------|
| 기본 모델 | Microsoft Florence-2-base |
| 입력 | RGB 이미지, (선택) Depth 이미지, 타겟 물체 리스트 |
| 출력 | Detection 리스트 (물체명, 신뢰도, bbox, 3D pose) |
| 지원 프로바이더 | HuggingFace, OpenAI Vision, Custom |

```python
# 사용 예시
detections = perception.process(rgb=image, object_list=["red_cup", "table"])
# [Detection(name="red_cup", confidence=0.95, bbox=[0.1,0.2,0.3,0.4], pose={...})]
```

### 2. Planner (계획 모듈) - 이중 루프 LLM 아키텍처
**역할**: 자연어 명령을 실행 가능한 primitive actions로 변환

| 항목 | 설명 |
|------|------|
| 기본 모델 | Qwen2.5-1.5B-Instruct (로컬) 또는 GPT-4o (API) |
| 입력 | 자연어 명령, 인식된 물체 정보 (좌표 포함) |
| 출력 | PrimitivePlan (go, move, grip 시퀀스) |
| 지원 프로바이더 | OpenAI, Anthropic, Google, Ollama, HuggingFace |

**이중 루프 구조:**
```
Stage 1: ActionPlanner (LLM)
  자연어 → High-level actions (pick, place, push, pull, open, close)

Stage 2: PrimitivePlanner (LLM)
  High-level action + 물체 좌표 → Primitives (go, move, grip)
```

```python
# 사용 예시
primitive_plans = planner.process_full("빨간 컵을 테이블에 놓아줘", world_state)
# [
#   PrimitivePlan(parent_action="pick", primitives=[grip(open), move(above), move(grasp), grip(close), move(lift)]),
#   PrimitivePlan(parent_action="place", primitives=[move(above), move(place), grip(open), move(retreat)])
# ]
```

### 3. Controller (제어 모듈)
**역할**: 고수준 스킬 명령을 저수준 로봇 궤적으로 변환

| 항목 | 설명 |
|------|------|
| 기본 백엔드 | Motion Primitives (기본 동작) |
| 입력 | Plan의 각 스텝 (skill + target) |
| 출력 | Command (궤적 포인트들, 그리퍼 명령) |
| 지원 백엔드 | Primitives, VLA (Vision-Language-Action), MoveIt |

### 4. Robot (로봇 인터페이스)
**역할**: 실제 로봇 하드웨어 또는 시뮬레이터 제어

| 항목 | 설명 |
|------|------|
| 기본 모드 | Simulation (시뮬레이션) |
| 입력 | Command (궤적, 그리퍼 명령) |
| 출력 | ExecutionResult (성공/실패, 실행 시간) |
| 지원 로봇 | Franka Emika, Universal Robots, 시뮬레이션 |

### 5. Monitor (모니터 모듈)
**역할**: 실행 중 실패 감지, 로봇 정지, Planner에 피드백 전달

| 항목 | 설명 |
|------|------|
| 기본 모델 | Qwen2.5-VL-3B-Instruct (로컬) 또는 GPT-4o Vision |
| 입력 | RGB 이미지, 현재 계획 상태, 로봇 실행 결과 |
| 출력 | FeedbackResult (success/fail, confidence, recovery_target) |
| 동작 방식 | 10Hz로 지속 모니터링, 실패 시 즉시 로봇 정지 + 재계획 요청 |

**Monitor 동작 흐름:**
```
1. 실시간 관찰 (10Hz)
   - 카메라 이미지 + 현재 스텝 정보로 성공/실패 판단
   
2. 실패 감지 시:
   a) /robot/stop 토픽으로 즉시 정지 명령
   b) /feedback/failure_signal 토픽으로 실패 정보 발행
   c) Planner가 실패 신호 수신 → 재계획 수행
   
3. 재계획 대상 결정:
   - 물체 위치 잘못됨 → "perception" (인식부터 다시)
   - 계획 자체 문제 → "planning" (계획부터 다시)
   - 궤적만 문제 → "controller" (제어부터 다시)
```

```python
# Monitor 출력 예시
FeedbackResult(
    success=False,
    confidence=0.85,
    recovery_target="planning",  # 어디서부터 다시 시작할지
    consecutive_failures=2,
    metadata={"reason": "object_not_grasped"}
)
```

---

## 디렉토리 구조

```
./
├── src/robobridge/
│   ├── __init__.py              # 메인 export (RoboBridge, wrappers)
│   ├── core/
│   │   ├── robobridge.py        # RoboBridge 오케스트레이터 (ROS2 + 어댑터 관리)
│   │   ├── adapters.py          # 모듈 간 통신 어댑터
│   │   └── protocols/
│   │       └── len_json.py      # 길이 prefix JSON 프로토콜
│   ├── config/
│   │   ├── loader.py            # 설정 파일 로더
│   │   └── defaults.py          # 기본 모듈 설정값
│   ├── modules/
│   │   ├── base.py              # BaseModule 추상 클래스
│   │   ├── perception/          # 인식 모듈 (Florence-2)
│   │   ├── planner/             # 계획 모듈 (LangChain + LLM)
│   │   ├── controller/          # 제어 모듈 (궤적 생성)
│   │   ├── robot/               # 로봇 인터페이스
│   │   └── monitor/             # 실행 모니터링 (VLM)
│   ├── wrappers/
│   │   ├── custom_perception.py # 커스텀 인식 모델 래퍼
│   │   ├── custom_planner.py    # 커스텀 계획 모델 래퍼
│   │   ├── custom_controller.py # 커스텀 제어기 래퍼
│   │   ├── custom_robot.py      # 커스텀 로봇 인터페이스 래퍼
│   │   └── custom_monitor.py    # 커스텀 모니터 래퍼
│   ├── client/
│   │   ├── client.py            # RoboBridgeClient (고수준 API)
│   │   └── types.py             # 클라이언트용 타입 정의
│   └── cli/
│       └── main.py              # CLI 엔트리포인트
├── examples/
│   └── config.py                # 예제 설정 파일
├── tests/
│   ├── unit/                    # 단위 테스트 (56개)
│   └── integration/             # 통합 테스트 (14개)
├── docs/                        # MkDocs 문서
├── pyproject.toml
└── README.md
```

---

## 기본 모델 설정

| 모듈 | 기본 모델 | 파라미터 | 용도 |
|------|-----------|----------|------|
| Perception | `microsoft/Florence-2-base` | ~230M | 물체 인식 + 캡션 |
| Planner | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 작업 계획 |
| Monitor | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | 실행 모니터링 |

대체 가능한 소형 모델:
- **LLM**: `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B-Instruct`
- **VLM**: `Qwen/Qwen2-VL-2B-Instruct`, `vikhyatk/moondream2`

---

## 사용 방법

### 기본 사용법

```python
from robobridge import RoboBridge

# 초기화 (시뮬레이션 모드)
robot = RoboBridge.initialize()

# 자연어 명령 실행
result = robot.execute("빨간 컵을 집어서 테이블에 놓아줘")

# 직접 제어
robot.pick("red_cup", grasp_width=0.05)
robot.place("red_cup", position=[0.4, 0.2, 0.05])

# 종료
robot.shutdown()
```

### 커스텀 모델 사용

```python
from robobridge.wrappers import CustomPerception, Detection

class MyDetector(CustomPerception):
    def load_model(self):
        self._model = load_my_yolo_model()
    
    def detect(self, rgb, depth=None, object_list=None):
        results = self._model(rgb)
        return [Detection(name=r.name, confidence=r.conf, bbox=r.bbox) 
                for r in results]

# 사용
robot = RoboBridge.initialize()
robot.set_model("perception", provider="custom", model="path/to/detector.py:MyDetector")
```

---

## 테스트 실행

```bash
cd .
PYTHONPATH=src python3 -m pytest tests/ -v
# 결과: 70 passed
```

---

## 향후 개발 방향

1. **실제 로봇 연동**: Franka Emika, UR5e 테스트
2. **VLA 모델 통합**: OpenVLA, RT-2 등 Vision-Language-Action 모델
3. **MoveIt 통합**: 경로 계획 및 충돌 회피
4. **Multi-robot 지원**: 여러 로봇 동시 제어
5. **Learning from Demonstration**: 시연 학습 기능

---

## 핵심 설계 원칙

1. **모듈 독립성**: 각 모듈은 독립적으로 교체 가능
2. **프로토콜 통일**: 모든 모듈 간 통신은 JSON 기반 메시지
3. **프로바이더 추상화**: LangChain으로 다양한 LLM/VLM 통합
4. **래퍼 패턴**: 커스텀 모델을 쉽게 통합할 수 있는 추상 클래스 제공
5. **타입 안전성**: dataclass 기반 타입 정의로 데이터 흐름 명확화
