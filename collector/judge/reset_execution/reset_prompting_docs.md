# Reset Judge Prompting Documentation

Reset Task의 완료 여부를 판단하는 VLM Judge의 프롬프팅 구현 문서입니다.

## 1. 개요

### 1.1 목적
- Reset 태스크가 성공적으로 완료되었는지 VLM을 통해 자동 판단
- 두 가지 모드 지원:
  - **Original Mode**: 물체를 원래 위치로 복원했는지 판단
  - **Random Mode**: 물체를 랜덤 목표 위치로 이동했는지 판단
- Chain-of-Thought 방식으로 단계별 추론을 유도하여 판단 정확도 향상

### 1.2 파일 위치
```
judge/reset_execution/
├── prompt.py    # 프롬프트 템플릿 및 생성 함수
└── judge.py     # ResetJudge 클래스 (VLM 호출 및 응답 파싱)
```

---

## 2. 프롬프트 구조

### 2.1 System Prompt (Original Mode)

원래 위치로 복원하는 모드의 시스템 프롬프트입니다.

```
RESET_SYSTEM_PROMPT_ORIGINAL
├── 역할 정의: "Reset Task Judge for robotic manipulation tasks"
├── 목표: "evaluate whether a robot successfully RESTORED objects to ORIGINAL positions"
├── 평가 프로세스 (5단계)
│   ├── Step 1: Identify Objects Before Reset
│   ├── Step 2: Identify Objects After Reset
│   ├── Step 3: Verify Position Restoration (Per Object)
│   ├── Step 4: Check Scene Restoration
│   └── Step 5: Final Judgment
└── 판단 기준
    ├── TRUE: Steps 3, 4 모두 PASS (All objects restored)
    ├── FALSE: Any object failed to restore
    └── UNCERTAIN: 판단 불가 (객체 미식별 등)
```

### 2.2 System Prompt (Random Mode)

랜덤 위치로 이동하는 모드의 시스템 프롬프트입니다.

```
RESET_SYSTEM_PROMPT_RANDOM
├── 역할 정의: "Reset Task Judge for robotic manipulation tasks"
├── 목표: "evaluate whether a robot successfully MOVED objects to designated RANDOM positions"
├── 평가 프로세스 (5단계)
│   ├── Step 1: Identify Objects Before Reset
│   ├── Step 2: Identify Objects After Reset
│   ├── Step 3: Verify Target Position Reached (Per Object)
│   ├── Step 4: Check Placement Quality
│   └── Step 5: Final Judgment
└── 판단 기준 (관대한 평가)
    ├── TRUE: All objects at or near targets (Steps 3, 4 PASS)
    ├── FALSE: Any object failed to reach target
    └── UNCERTAIN: 판단 불가
```

### 2.3 User Prompt Template (Original Mode)

```
RESET_USER_PROMPT_ORIGINAL
├── 1. Goal
│   └── "Restore objects to their original positions"
│
├── 2. Original Task (for context)
│   └── {original_instruction} (Forward 태스크 명령어)
│
├── 3. Image Information
│   ├── Image Resolution: {width} x {height} pixels
│   ├── Image 1 (Before Reset): Objects at current positions
│   └── Image 2 (After Reset): Objects should be at original positions
│
├── 4. Object Positions (테이블 형식)
│   └── 각 객체별:
│       ├── Current: [x, y, z] (World) | Pixel: (cx, cy)
│       └── Original: [x, y, z] (World) | Pixel: (cx, cy)
│
├── 5. Executed Reset Code
│   └── 실행된 코드
│
└── 6. Output Format (5단계 구조화된 응답)
```

### 2.4 User Prompt Template (Random Mode)

```
RESET_USER_PROMPT_RANDOM
├── 1. Goal
│   └── "Move objects to new random target positions"
│
├── 2. Image Information
│   ├── Image Resolution: {width} x {height} pixels
│   ├── Image 1 (Before Reset): Objects at current positions
│   └── Image 2 (After Reset): Objects should be at random target positions
│
├── 3. Object Positions (테이블 형식)
│   └── 각 객체별:
│       ├── Current: [x, y, z] (World) | Pixel: (cx, cy)
│       └── Random Target: [x, y, z] (World) | Pixel: (cx, cy)
│
├── 4. Executed Reset Code
│   └── 실행된 코드
│
└── 5. Output Format (5단계 구조화된 응답)
```

---

## 3. Chain-of-Thought 평가 단계

### 3.1 Original Mode 평가 단계

#### Step 1: Identify Objects Before Reset
**목적**: Reset 전 이미지(Forward 완료 후 상태)에서 객체 식별

**입력 정보**:
- Image 1 (Before Reset)
- 각 객체의 Current 픽셀 좌표

**VLM 수행 작업**:
1. 제공된 픽셀 좌표로 이동
2. 해당 위치에 있는 객체 확인
3. 객체 설명 작성

**출력 형식**:
```
**STEP 1 - Identify Objects Before Reset**
- Object: green block
  - Found at pixel: (420, 300)
  - Description: A green block visible on the blue dish
- Step 1 Result: PASS - All objects correctly identified
```

#### Step 2: Identify Objects After Reset
**목적**: Reset 후 이미지에서 객체 위치 파악

**입력 정보**:
- Image 2 (After Reset)
- Step 1에서 식별한 객체 목록

**VLM 수행 작업**:
1. 최종 이미지에서 동일 객체 탐색
2. 새로운 위치 추정 (픽셀 좌표)
3. 객체 상태 설명

**출력 형식**:
```
**STEP 2 - Identify Objects After Reset**
- Object: green block
  - Found at pixel: (280, 320)
  - Description: Green block now visible at original position
- Step 2 Result: PASS - All objects visible in final image
```

#### Step 3: Verify Position Restoration (Per Object)
**목적**: 각 객체가 원래 위치로 복원되었는지 확인

**입력 정보**:
- Current positions (before reset)
- Target original positions
- 이미지 상의 실제 위치

**VLM 수행 작업**:
1. 각 객체별로 복원 상태 확인
2. 초기/목표/최종 위치 비교
3. 복원 여부 판단

**출력 형식**:
```
**STEP 3 - Verify Position Restoration (Per Object)**
- Object: green block
  - Current (before reset): (420, 300)
  - Target (original): approximately (280, 320)
  - Final (after reset): (285, 318)
  - Restored: YES
- Step 3 Result: PASS - All objects returned to original positions
```

#### Step 4: Check Scene Restoration
**목적**: 전체 장면이 올바르게 복원되었는지 확인

**입력 정보**:
- Final Image
- Original positions

**VLM 수행 작업**:
1. 전체 장면과 원래 상태 비교
2. 물체 낙하/오배치 확인
3. 장면 복원 품질 평가

**출력 형식**:
```
**STEP 4 - Check Scene Restoration**
- Scene similarity to original: HIGH
- Objects dropped or misplaced: NO
- Step 4 Result: PASS - Scene properly restored
```

#### Step 5: Final Judgment
**목적**: 종합 판단

**판단 로직**:
```
if Steps 3 and 4 both PASS:
    PREDICTION = TRUE
elif any of Steps 3, 4 FAIL:
    PREDICTION = FALSE
else:
    PREDICTION = UNCERTAIN
```

**출력 형식**:
```
**STEP 5 - Final Judgment**
- Steps passed: 1, 2, 3, 4
- Steps failed: none
- Objects restored: 2/2 objects

PREDICTION: TRUE

REASONING: All objects successfully restored to original positions.
```

---

### 3.2 Random Mode 평가 단계

#### Step 1-2: 동일 (객체 식별)

Original Mode와 동일한 방식으로 객체를 식별합니다.

#### Step 3: Verify Target Position Reached (Per Object)
**목적**: 각 객체가 랜덤 목표 위치에 도달했는지 확인

**출력 형식**:
```
**STEP 3 - Verify Target Position Reached (Per Object)**
- Object: green block
  - Current (before reset): (420, 300)
  - Target (random): approximately (320, 300)
  - Final (after reset): (325, 298)
  - Reached target: YES
- Step 3 Result: PASS - All objects reached their targets
```

#### Step 4: Check Placement Quality
**목적**: 물체 배치 품질 확인 (낙하, 안정성 등)

**출력 형식**:
```
**STEP 4 - Check Placement Quality**
- All objects visible: YES
- Objects properly placed (not dropped): YES
- Step 4 Result: PASS - All objects properly placed
```

#### Step 5: Final Judgment (동일)

---

## 4. 데이터 흐름

### 4.1 입력 데이터 구조

```python
# current_positions (Extended Format - Forward 완료 후 상태)
{
    "_image_resolution": (640, 480),  # 메타데이터
    "green block": {
        "position": [0.28, -0.12, 0.05],     # World 좌표 (meters)
        "pixel_coords": (420, 300),           # 중심점 픽셀 좌표
        "bbox_pixels": (390, 270, 450, 330),  # Bounding box
    },
    "blue dish": {
        "position": [0.28, -0.12, 0.05],
        "pixel_coords": (420, 310),
        "bbox_pixels": (380, 280, 460, 340),
    },
}

# target_positions (Original Mode: 원래 위치 / Random Mode: 랜덤 목표)
{
    "green block": {
        "position": [0.11, 0.10, 0.05],
        "pixel_coords": (280, 320),
    },
    "blue dish": {
        "position": [0.27, -0.16, 0.05],
        "pixel_coords": (450, 290),
    },
}
```

### 4.2 위치 테이블 생성

`_build_position_table()` 함수가 위치 정보를 테이블 형식으로 변환합니다:

```
**green block**:
  - Current: [0.280, -0.120, 0.050] (World) | Pixel: (420, 300)
  - Original: [0.110, 0.100, 0.050] (World) | Pixel: (280, 320)

**blue dish**:
  - Current: [0.280, -0.120, 0.050] (World) | Pixel: (420, 310)
  - Original: [0.270, -0.160, 0.050] (World) | Pixel: (450, 290)
```

### 4.3 프롬프트 생성 함수

```python
def build_reset_judge_prompt(
    reset_mode: str,                    # "original" 또는 "random"
    current_positions: Dict,            # Reset 전 위치 (Extended Format)
    target_positions: Dict,             # 목표 위치
    executed_code: str,                 # 실행된 Reset 코드
    original_instruction: str = None,   # Forward 태스크 명령어 (original 모드)
    image_resolution: Tuple[int, int] = None,
) -> str:
    """
    1. reset_mode에 따라 적절한 템플릿 선택
    2. image_resolution 추출 (current_positions 또는 파라미터)
    3. 위치 테이블 생성 (_build_position_table)
    4. 템플릿에 포맷팅하여 반환
    """
```

---

## 5. VLM 호출 및 응답 처리

### 5.1 VLM 호출 (judge.py)

```python
class ResetJudge:
    def judge(
        self,
        reset_mode: str,
        current_positions: Dict,
        target_positions: Dict,
        initial_image: np.ndarray,    # Reset 전 이미지
        final_image: np.ndarray,      # Reset 후 이미지
        executed_code: str,
        original_instruction: str = None,
        image_resolution: Tuple[int, int] = None,
    ) -> Dict:
        """
        1. 이미지 해상도 추출
        2. System prompt 선택 (get_reset_system_prompt)
        3. User prompt 생성 (build_reset_judge_prompt)
        4. 이미지 base64 인코딩
        5. VLM API 호출 (OpenAI 또는 vLLM 서버)
        6. 응답 파싱
        7. 결과 반환
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
    'reset_mode': 'original',       # 사용된 Reset 모드
}
```

---

## 7. Original vs Random 모드 차이점

| 항목 | Original Mode | Random Mode |
|------|--------------|-------------|
| **목적** | 원래 위치로 복원 | 랜덤 위치로 이동 |
| **목표 위치** | Forward 전 초기 위치 | 무작위 생성된 위치 |
| **평가 기준** | 복원 정확도 (엄격) | 도달 여부 (관대) |
| **Step 3** | Position Restoration | Target Position Reached |
| **Step 4** | Scene Restoration | Placement Quality |
| **Original Instruction** | 제공됨 (컨텍스트용) | 제공 안함 |

---

## 8. 핵심 설계 원칙

### 8.1 Chain-of-Thought
- 단계별 추론을 유도하여 VLM의 판단 과정을 명시적으로 만듦
- 각 단계에 PASS/FAIL을 부여하여 어디서 실패했는지 추적 가능

### 8.2 Pixel Coordinates
- 이미지 내 정확한 위치 참조를 위해 픽셀 좌표 제공
- World 좌표와 함께 제공하여 다양한 관점에서 검증

### 8.3 Dual Position Information
- Current + Target 위치를 함께 제공
- VLM이 이동 전후 비교를 정확히 수행할 수 있음

### 8.4 Mode-Specific Evaluation
- Original: 엄격한 복원 검사 (장면 유사도 포함)
- Random: 관대한 도달 검사 (대략적 위치 허용)

### 8.5 Structured Output
- 고정된 출력 형식을 요구하여 파싱 용이성 확보
- Step별 PASS/FAIL 결과를 명시적으로 요구
