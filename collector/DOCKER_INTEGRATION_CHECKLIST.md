# Docker 통합 체크리스트

> AutoDataCollector를 Docker 이미지로 패키징하여 제출하기 위한 작업 목록
> 
> 기준 문서: `참고자료_Docker_File_제출_가이드라인_및_목록_예시.md`

---

## 1. 필수 파일 생성

### 1-1. Dockerfile

```dockerfile
# 예시 — 실제 빌드 테스트 후 확정
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git curl libgl1-mesa-glv libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pinocchio — pip install pin이 안 되면 아래로 대체
# RUN conda install -c conda-forge pinocchio -y

COPY . /app/

RUN chmod +x run_agent.sh

# API Key는 빌드 시 넣지 않음 — 실행 시 -e로 주입
ENV GOOGLE_API_KEY=""
ENV OPENAI_API_KEY=""

ENTRYPOINT ["./run_agent.sh"]
```

**확인 사항:**
- [ ] `pip install pin` (Pinocchio)이 Docker에서 빌드되는지 테스트
  - 안 되면 conda-forge 사용 또는 시스템 패키지로 설치
- [ ] `COPY . /app/` 시 불필요한 파일 제외 → `.dockerignore` 작성 필요
  - 제외 대상: `.git/`, `results/`, `outputs/`, `__pycache__/`, `.claude/`
- [ ] 가이드라인은 `COPY src/ ./src/`이지만, 현재 구조는 루트에 실행 파일이 있으므로 `COPY . /app/`이 현실적

### 1-2. run_agent.sh

심사위원이 실행할 단일 진입점. 가이드라인 형식:

```bash
#!/bin/bash
INPUT_FILE=${1:-"./data/input_sample.json"}
OUTPUT_FILE=${2:-"./results/output.json"}

echo "Starting AutoDataCollector Agent..."
python execution_forward_and_reset.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"
```

**확인 사항:**
- [ ] 심사 시 입력/출력 형식 확인 (JSON? 이미지?)
- [ ] 어떤 Python 스크립트가 진입점인지 결정
  - 단일 암: `execution_forward_and_reset.py`
  - 멀티 암: `unified_multi_arm.py`
  - 셸 래퍼: `run_forward_and_reset.sh`
- [ ] CLI 인자 매핑 (심사 입력 → 파이프라인 인자)

### 1-3. .dockerignore

```
.git/
.claude/
results/
outputs/
__pycache__/
*.pyc
.env
openai_api_key.json
google_aistudio_key.json
```

---

## 2. 환경변수 정리

심사위원에게 안내해야 할 환경변수 목록:

| 환경변수 | 용도 | 필수 | 기본값 |
|----------|------|:----:|--------|
| `GOOGLE_API_KEY` | Gemini API (코드 생성, 물체 감지) | **필수** | 없음 |
| `OPENAI_API_KEY` | OpenAI API (Judge 평가) | 선택 | 없음 |
| `DEEPSEEK_API_KEY` | DeepSeek API | 선택 | 없음 |
| `VLLM_SERVER_URL` | 코드생성 LLM 서버 주소 | 선택 | `http://localhost:8001/v1` |
| `JUDGE_SERVER_URL` | Judge VLM 서버 주소 | 선택 | `http://localhost:8002/v1` |

**실행 예시:**
```bash
docker run --rm --gpus all \
    -e GOOGLE_API_KEY="your-key" \
    -e OPENAI_API_KEY="your-key" \
    team_name_agent
```

---

## 3. 코드 수정 필요 항목

### 3-1. llama.py 서버 URL 환경변수 fallback (1건)

**파일:** `code_gen_lerobot/llm_utils/llama.py:15`

현재:
```python
url = "http://localhost:8000/v1/completions"
```

수정:
```python
url = os.getenv("LLAMA_SERVER_URL", "http://localhost:8000/v1/completions")
```

다른 서버 URL들은 이미 env fallback이 있음:
- `llm.py:28` — `VLLM_SERVER_URL` ✓
- `vlm.py:25` — `JUDGE_SERVER_URL` / `VLM_SERVER_URL` ✓

### 3-2. SSH 서버 IP (파이프라인에서 사용 시)

**파일:** `pipeline_config/free_api_config.yaml`

```yaml
server_host: "<SERVER_HOST>"    # ← 환경변수로 주입
ssh_user: "<SSH_USER>"          # ← 환경변수로 주입
```

심사 환경에서 이 서버에 접근 불가능할 수 있음.
- API 모드로만 실행하면 문제 없음 (SSH 서버 불필요)
- `paid_api_config.yaml` 사용 시 SSH 관련 설정 없으므로 안전

---

## 4. Docker 빌드 시 확인할 의존성

| 패키지 | 설치 방법 | 주의 |
|--------|----------|------|
| `pin` (Pinocchio) | `pip install pin` 또는 `conda install -c conda-forge pinocchio` | C++ 빌드 필요, Docker에서 시간 오래 걸릴 수 있음 |
| `pyrealsense2` | `pip install pyrealsense2` | Docker 내에서 실제 카메라 없이 import만 되면 OK |
| `feetech-servo-sdk` | `pip install feetech-servo-sdk` | 시리얼 포트 없어도 import는 됨 |
| `lerobot` | 번들 포함 (`lerobot/` 디렉토리) | pip install 아님, sys.path로 참조 |

---

## 5. 로컬 테스트 명령어

```bash
# 1. 이미지 빌드
docker build -t autodatacollector .

# 2. 빌드 성공 후 import 테스트
docker run --rm autodatacollector python -c "
import torch; print(f'torch: {torch.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import cv2; print(f'opencv: {cv2.__version__}')
from google import genai; print('google-genai: OK')
from openai import OpenAI; print('openai: OK')
"

# 3. 실제 실행 테스트
docker run --rm --gpus all \
    -e GOOGLE_API_KEY="your-key" \
    autodatacollector

# 4. 디버깅 (셸 접속)
docker run --rm -it --gpus all autodatacollector /bin/bash
```

---

## 6. 현재 코드 상태 (참고)

통합 전 수정 완료 항목:

| 항목 | 상태 |
|------|:----:|
| 절대경로 (`/home/...`) 제거 | ✅ |
| 존재하지 않는 파일 참조 복구 | ✅ |
| YAML 상대경로 변환 + Python resolve | ✅ |
| API 키 환경변수 우선 읽기 | ✅ |
| 디펜던시 누락/버전충돌 수정 | ✅ |
| requirements.txt 생성 | ✅ |
| lerobot 번들 포함 | ✅ |
| README 업데이트 | ✅ |

Docker 내에서 그대로 동작하는 항목 (수정 불필요):

| 항목 | 이유 |
|------|------|
| CWD 의존 경로 30건+ | `WORKDIR /app` = 프로젝트 루트 |
| `sys.path` 해킹 14건 | 내부 디렉토리 구조 유지됨 |
| PROJECT_ROOT 가정 6건 | 파일 위치 변경 없음 |
