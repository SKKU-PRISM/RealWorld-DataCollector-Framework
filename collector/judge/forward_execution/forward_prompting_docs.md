# Forward Judge Prompting Documentation

Forward Task의 완료 여부를 판단하는 VLM Judge의 프롬프팅 구현 문서입니다.

## 1. 개요

### 1.1 목적
- Pick-and-Place 태스크가 성공적으로 완료되었는지 VLM을 통해 자동 판단
- Chain-of-Thought 방식으로 단계별 추론을 유도하여 판단 정확도 향상

### 1.2 파일 위치
```
judge/forward_execution/
├── prompt.py    # 프롬프트 템플릿 및 생성 함수
└── judge.py     # TaskJudge 클래스 (VLM 호출 및 응답 파싱)
```

---

## 2. 프롬프트 구조

### 2.1 System Prompt

VLM의 역할과 평가 프로세스를 정의합니다.

```
JUDGE_SYSTEM_PROMPT
├── 역할 정의: "Task Completion Judge for robotic manipulation tasks"
├── 평가 프로세스 (6단계)
│   ├── Step 1: Identify Objects in Initial Image
│   ├── Step 2: Identify Objects in Final Image
│   ├── Step 3: Analyze Movement
│   ├── Step 4: Verify Goal Achievement
│   ├── Step 5: Check Spatial Relationships
│   └── Step 6: Final Judgment
└── 판단 기준
    ├── TRUE: Steps 3, 4, 5 모두 PASS
    ├── FALSE: Steps 3, 4, 5 중 하나라도 FAIL
    └── UNCERTAIN: 판단 불가 (객체 미식별 등)
```

### 2.2 User Prompt Template

실제 태스크 정보를 포함하는 프롬프트입니다.

```
JUDGE_USER_PROMPT_TEMPLATE
├── 1. Goal Instruction
│   └── 자연어 목표 명령어 (예: "pick up the red cup and place it on the carrier")
│
├── 2. Image Information
│   ├── Image Resolution: {width} x {height} pixels
│   ├── Image 1 (Initial State): 태스크 실행 전
│   └── Image 2 (Final State): 태스크 실행 후
│
├── 3. Object Positions in Initial Image
│   └── 각 객체별 정보:
│       ├── World position: [x, y, z] meters
│       ├── Pixel center: (cx, cy)
│       └── Bounding box: (x1, y1) to (x2, y2)
│
├── 4. Executed Python Code
│   └── 실행된 로봇 제어 코드
│
└── 5. Output Format (6단계 구조화된 응답 요구)
```

---

## 3. Chain-of-Thought 평가 단계

### Step 1: Identify Objects in Initial Image
**목적**: 초기 이미지에서 객체 식별 및 위치 확인

**입력 정보**:
- Initial Image (Image 1)
- 각 객체의 픽셀 좌표 (pixel_coords, bbox_pixels)

**VLM 수행 작업**:
1. 제공된 픽셀 좌표로 이동
2. 해당 위치에 있는 객체 확인
3. 객체 설명 작성

**출력 형식**:
```
**STEP 1 - Identify Objects in Initial Image**
- Object: red cup
  - Found at pixel: (320, 280)
  - Description: A small red cup visible in the left portion of the image
- Step 1 Result: PASS - All objects correctly identified
```

### Step 2: Identify Objects in Final Image
**목적**: 최종 이미지에서 객체 위치 파악

**입력 정보**:
- Final Image (Image 2)
- Step 1에서 식별한 객체 목록

**VLM 수행 작업**:
1. 최종 이미지에서 동일 객체 탐색
2. 새로운 위치 추정 (픽셀 좌표)
3. 객체 상태 설명

**출력 형식**:
```
**STEP 2 - Identify Objects in Final Image**
- Object: red cup
  - Found at pixel: (445, 295)
  - Description: Red cup now visible on top of the carrier
- Step 2 Result: PASS - All objects visible in final image
```

### Step 3: Analyze Movement
**목적**: 객체 이동 여부 및 방향 분석

**입력 정보**:
- Step 1의 초기 위치
- Step 2의 최종 위치

**VLM 수행 작업**:
1. 초기 vs 최종 위치 비교
2. 이동 거리 및 방향 파악
3. Pick 객체가 이동했는지 확인

**출력 형식**:
```
**STEP 3 - Analyze Movement**
- Object: red cup
  - Initial: (320, 280) → Final: (445, 295)
  - Movement: moved, Direction: right
- Step 3 Result: PASS - Red cup moved from left to right as expected
```

### Step 4: Verify Goal Achievement
**목적**: 목표 달성 여부 확인

**입력 정보**:
- Goal Instruction
- Executed Code
- 객체 이동 정보

**VLM 수행 작업**:
1. 명령어에서 의도 파악
2. Pick 객체가 픽업되었는지 확인
3. 목적지에 도달했는지 확인

**출력 형식**:
```
**STEP 4 - Verify Goal Achievement**
- Goal: "pick up the red cup and place it on the carrier"
- Pick object picked up: YES
- Reached destination: YES
- Step 4 Result: PASS - Red cup successfully moved to carrier
```

### Step 5: Check Spatial Relationships
**목적**: 최종 공간 관계 검증

**입력 정보**:
- Final Image
- Goal Instruction

**VLM 수행 작업**:
1. 명령어에서 요구하는 관계 추출
2. 최종 이미지에서 실제 관계 확인
3. 일치 여부 판단

**출력 형식**:
```
**STEP 5 - Check Spatial Relationships**
- Expected relationship: "red cup should be on carrier"
- Observed relationship: Red cup is positioned on top of the carrier
- Step 5 Result: PASS - Spatial relationship matches instruction
```

### Step 6: Final Judgment
**목적**: 종합 판단

**판단 로직**:
```
if Steps 3, 4, 5 all PASS:
    PREDICTION = TRUE
elif any of Steps 3, 4, 5 FAIL:
    PREDICTION = FALSE
else:
    PREDICTION = UNCERTAIN
```

**출력 형식**:
```
**STEP 6 - Final Judgment**
- Steps passed: 1, 2, 3, 4, 5
- Steps failed: none

PREDICTION: TRUE

REASONING: All 5 steps passed. The red cup was successfully picked up and placed on the carrier.
```

---

## 4. 데이터 흐름

### 4.1 입력 데이터 구조

```python
# object_positions (Extended Format)
{
    "_image_resolution": (640, 480),  # 메타데이터
    "red cup": {
        "position": [0.15, 0.05, 0.02],      # World 좌표 (meters)
        "pixel_coords": (320, 280),           # 중심점 픽셀 좌표
        "bbox_pixels": (280, 240, 360, 320),  # Bounding box (x1, y1, x2, y2)
        "confidence": 0.92,                   # 검출 신뢰도
        "gripper_offset": 0.045,              # 그리퍼 오프셋
    },
    "carrier": {
        "position": [0.20, -0.03, 0.01],
        "pixel_coords": (450, 300),
        "bbox_pixels": (400, 250, 500, 350),
        "confidence": 0.88,
    },
}
```

### 4.2 프롬프트 생성 함수

```python
def build_judge_prompt(
    instruction: str,                    # 목표 명령어
    object_positions: Dict,              # 객체 위치 (Extended Format)
    executed_code: str,                  # 실행된 코드
    image_resolution: Tuple[int, int],   # 이미지 해상도
) -> str:
    """
    1. image_resolution 추출 (object_positions 또는 파라미터)
    2. 객체별 위치 텍스트 생성 (World + Pixel 좌표)
    3. JUDGE_USER_PROMPT_TEMPLATE에 포맷팅
    4. 완성된 프롬프트 반환
    """
```

---

## 5. VLM 호출 및 응답 처리

### 5.1 VLM 호출 (judge.py)

```python
class TaskJudge:
    def judge(
        self,
        instruction: str,
        initial_image: np.ndarray,
        final_image: np.ndarray,
        object_positions: Dict,
        executed_code: str,
        image_resolution: Tuple[int, int] = None,
    ) -> Dict:
        """
        1. 이미지 해상도 추출
        2. 프롬프트 생성 (build_judge_prompt)
        3. 이미지 base64 인코딩
        4. VLM API 호출 (OpenAI 또는 vLLM 서버)
        5. 응답 파싱
        6. 결과 반환
        """
```

### 5.2 응답 파싱

```python
def _parse_response(self, response: str) -> Dict:
    """
    정규식 패턴으로 추출:
    - PREDICTION: TRUE/FALSE/UNCERTAIN
    - REASONING: 판단 근거
    """
```

---

## 6. 출력 결과

```python
{
    'prediction': 'TRUE',           # TRUE / FALSE / UNCERTAIN
    'reasoning': '...',             # 판단 근거
    'raw_response': '...',          # VLM 원본 응답
    'success': True,                # API 호출 성공 여부
    'error': None,                  # 에러 메시지 (있는 경우)
}
```

---

## 7. 핵심 설계 원칙

### 7.1 Chain-of-Thought
- 단계별 추론을 유도하여 VLM의 판단 과정을 명시적으로 만듦
- 각 단계에 PASS/FAIL을 부여하여 어디서 실패했는지 추적 가능

### 7.2 Pixel Coordinates
- 이미지 내 정확한 위치 참조를 위해 픽셀 좌표 제공
- VLM이 객체를 정확히 식별하도록 도움

### 7.3 Structured Output
- 고정된 출력 형식을 요구하여 파싱 용이성 확보
- Step별 결과를 명시적으로 요구

### 7.4 Core Steps Focus
- Steps 1, 2는 객체 식별 (보조)
- Steps 3, 4, 5가 핵심 판단 기준
- 핵심 단계 모두 통과해야 TRUE
