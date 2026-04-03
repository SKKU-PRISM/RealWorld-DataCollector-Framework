#!/bin/bash
# Forward + Reset Integrated Pipeline Runner
# Forward Execution → Judge (Evaluation) → Reset Execution 통합 파이프라인
#
# Config 파일들:
#   - pipeline_config/paid_api_config.yaml     : 유료 API 설정 (USE_SERVER=false)
#   - pipeline_config/free_api_config.yaml     : vLLM 서버 설정 (USE_SERVER=true)
#   - pipeline_config/recording_config.yaml    : 레코딩 설정 (RECORD_DATASET=true)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# 핵심 설정 (Essential Configuration) / 워크스페이스 명시 / 에피소드 갯수 명시
# ============================================================

# # Grasping:                                                                  
# (1, 완료) pick up the red block and place it on the blue plate
# (2, 완료) distribute chocolate pies to each plate                           
# (3) -

# # Arrangement:                                                               
# (1, 완료) place the yellow block between chocolate pies             
# (2, 완료) arrange yellow, red, and purple blocks in a line from left to right
# (3, 완료) stack the blocks in the order of red and yellow
# (3, 완료) stack the blocks in the order of red and yellow, purple

# # Non-grasping:
# (1, 성공) turn on the microphone by pressing the power button
# (2, 성공) Push the bowl of cereal 5cm from left to right
# (3, 성공) Open the trash can lid

# # Deformable:
# (1, 완료) fold the towel
# (2) sweep the floor with a towel
# (3) bend the microphone gooseneck leftward

# # Articulated:
# (1) open the drawers
# (2) close the drawers
# (3) beat the red block with a hammer

# # Insertion/Assembly:
# (1) assemble the battery pack
# (2) peg-in-hole
# (3) clean the desk

# # Rotation:
# (1) tighten the bolt
# (2) open the bottle
# (3) mix the tea

# # Contact-rich:
# (1) wipe the dish with a sponge
# (2) sweep the floor with a brush
# (3) shake the bottle

# INSTRUCTION="make sandwich using the ingredients on the table"
# INSTRUCTION="pick up the red block and place it on the blue dish"
# INSTRUCTION="fold the green towel"
# INSTRUCTION="pick up the brown peg and insert it into the hole of the gray structure"
# INSTRUCTION = "Pick up the banana and place it in the bowl. 
# You may need to handover the banana from one arm to the other if the initial arm picking the banana cannot reach the bowl. 
# After picking the banana with one arm, you can handover the banana by first placing it carefully on the table surface and then using the other arm to pick it up. 
# The placing position must be on the table, as far as possible from other objects but absolutely within the reachable table area of the other arm. 
# Make sure to move the picking arm out of the way before the receiving arm moves towards grasping the object."

# INSTRUCTION="Assemble the green hinge and red hinge.
# You need to carefully assemble the green hinge's male part to red hinge's hole part.
# since the green hinge's male part is upward, you need to rotate it downward first before assembling."

# [필수] 로봇 번호 배열 — 순서가 arm 그룹을 결정 (최대 4대):
#   ROBOT_IDS[0] → left_arm
#   ROBOT_IDS[1] → right_arm
#   ROBOT_IDS[2] → top_arm
#   ROBOT_IDS[3] → bottom_arm
# shared 카메라는 항상 포함. 제공된 ID 수만큼만 arm 그룹 활성화.
# 예: (0)       → shared + left_arm
#     (2 3)     → shared + left_arm(robot2) + right_arm(robot3)
#     (1 2 3 4) → shared + left_arm + right_arm + top_arm + bottom_arm
ROBOT_IDS=(2 3)

## Task_instruction 
# INSTRUCTION="stack red block at center, then place yellow block on top of red block"

### [single arm task]
## pick and place
# INSTRUCTION="pick up the red block and place it on the blue dish"
# RESET_INSTRUCTION=""

## stack red and yellow
# INSTRUCTION="stack the blocks in the order of red and yellow"
# RESET_INSTRUCTION=""

## stack RYP blocks
# INSTRUCTION="stack the blocks in the order of red and yellow, purple."
# RESET_INSTRUCTION=""

## distribute chocolate pies to each plate
# INSTRUCTION="distribute chocolate pies to each plate."
# RESET_INSTRUCTION=""

### [dual arm task]
## towel folding
# INSTRUCTION="move the yellow block from top-left edge to bottom-right edge"
# RESET_INSTRUCTION="move the yellow block from bottom-right edge to top-left edge"

## move
INSTRUCTION="move the yellow block from top-left area to bottom-right edge"
RESET_INSTRUCTION="move the yellow block from bottom-right edge to top-left area"

## hand over the sponge
# INSTRUCTION="move the yellow block from top-left edge to bottom-right edge"
# RESET_INSTRUCTION="move the yellow block from bottom-right edge to top-left edge"

## Reset_instruction(Empty is default: "move objects to certain position")

# [필수] 에피소드 반복 횟수
NUM_EPISODES=30
NUM_RANDOM_SEEDS=15 # 배치 수 (1=초기 위치 유지, N>1=N종류 랜덤 배치, 에피소드를 N등분)

# [선택] 로봇별 reset 공간 제약 (all, top-left, top-right, bottom-left, bottom-right)
# 로봇 순서대로 지정. 예: 단일 (top-left), 듀얼 (top-left top-right)
# all: 워크스페이스 전역, top-left 등: 테이블 4분면 중 해당 영역 ∩ 로봇 도달 범위
RESETSPACE_PER_ROBOT=(top-left bottom-right) 

# [필수] 결과 저장 경로
SAVE_DIR="./results"

# [선택] Turn Test (Waypoint Trajectory) 스킵 여부
SKIP_TURN_TEST=true

# 서버 추론 사용 여부 (true: vLLM 서버, false: 유료 API)
USE_SERVER=false

# Reset execution 설정
EXECUTE_RESET=true # Reset 실행 여부

# Dataset Recording 설정
RECORD_DATASET=true

# Resume 설정 (이전 세션 이어받기)
# 비어있으면 새 세션, 경로 지정 시 이전 세션 이어받기
RESUME_SESSION=""
# RESUME_SESSION="./results/session_20260319_174942"

# ============================================================
# Multi-turn LLM 코드 생성 설정
# true: crop-then-point 멀티턴 (LLM이 이미지 보고 검출→crop pointing→코드 생성)
# false: single-turn (Grounding DINO 검출 후 LLM 코드 생성)
# ============================================================
MULTI_TURN=true

## CAD 참조 이미지 디렉토리 (비어있으면 CAD 없이 실행)
# 예: CAD_IMAGE_DIRS=("/path/to/cad_male" "/path/to/cad_female")
CAD_IMAGE_DIRS=()
# CAD_IMAGE_DIRS=("pipeline_config/cad_images/brown_peg", "pipeline_config/cad_images/gray_structure_with_hole")

## Side-view 이미지 경로 (Turn Test waypoint trajectory 예측용, 비어있으면 overhead만 사용)
SIDE_VIEW_IMAGE=""
# SIDE_VIEW_IMAGE="pipeline_config/side_view_images/side_view.jpg"

# ============================================================
# Config 파일 로드 함수
# ============================================================

CONFIG_DIR="$SCRIPT_DIR/pipeline_config"

load_paid_api_config() {
    local config_file="$CONFIG_DIR/paid_api_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
        echo "[Config] Loaded: paid_api_config.yaml"
    else
        echo "[Config] Warning: paid_api_config.yaml not found, using defaults"
        CODEGEN_LLM_MODEL="gpt-4o-mini"
        JUDGE_VLM_MODEL="gpt-4o"
        JUDGE_TIMEOUT=5.0
    fi
}

load_free_api_config() {
    local config_file="$CONFIG_DIR/free_api_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
        echo "[Config] Loaded: free_api_config.yaml"
    else
        echo "[Config] Warning: free_api_config.yaml not found, using defaults"
        CODEGEN_SERVER_HOST="localhost"
        CODEGEN_SERVER_PORT=8001
        CODEGEN_SSH_PORT=22
        CODEGEN_SSH_USER="user"
        CODEGEN_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
        JUDGE_SERVER_HOST="localhost"
        JUDGE_SERVER_PORT=8002
        JUDGE_SSH_PORT=22
        JUDGE_SSH_USER="user"
        JUDGE_MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
        JUDGE_TIMEOUT=3.0
    fi
}

load_recording_config() {
    local config_file="$CONFIG_DIR/recording_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
        echo "[Config] Loaded: recording_config.yaml"
    else
        echo "[Config] Warning: recording_config.yaml not found, using defaults"
        DATASET_REPO_ID=""
        RECORDING_FPS=30
    fi
}

# ============================================================
# Config 로드
# ============================================================

echo "========================================"
echo "Loading Configuration Files..."
echo "========================================"

# API config 로드 (USE_SERVER에 따라 선택)
if [ "$USE_SERVER" = true ]; then
    load_free_api_config
    LLM_MODEL="$CODEGEN_MODEL_NAME"
    JUDGE_MODEL="$JUDGE_MODEL_NAME"
else
    load_paid_api_config
    LLM_MODEL="$CODEGEN_LLM_MODEL"
    JUDGE_MODEL="$JUDGE_VLM_MODEL"
    CODEGEN_SESSION2_MODEL="${CODEGEN_SESSION2_MODEL:-}"
fi

# Recording config 로드 (RECORD_DATASET=true일 때만)
if [ "$RECORD_DATASET" = true ]; then
    load_recording_config
fi

echo ""

# ============================================================
# SSH 터널 설정 (USE_SERVER=true 시)
# ============================================================

TUNNEL_PIDS=()

cleanup_tunnels() {
    for pid in "${TUNNEL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[SSH Tunnel] Closing tunnel (PID: $pid)"
            kill "$pid" 2>/dev/null
        fi
    done
}
trap cleanup_tunnels EXIT

if [ "$USE_SERVER" = true ]; then
    echo "[Server] Setting up vLLM server connections..."

    # CodeGen 서버 연결 설정
    if [ "$CODEGEN_SSH_PORT" = "22" ]; then
        CODEGEN_SERVER_URL="http://${CODEGEN_SERVER_HOST}:${CODEGEN_SERVER_PORT}/v1"
        echo "[CodeGen] Direct connection: ${CODEGEN_SERVER_URL}"
    else
        LOCAL_CODEGEN_PORT=$CODEGEN_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT}.*${CODEGEN_SERVER_HOST}" > /dev/null; then
            echo "[CodeGen] Creating SSH tunnel (localhost:${LOCAL_CODEGEN_PORT} → ${CODEGEN_SERVER_HOST}:${CODEGEN_SERVER_PORT})..."
            ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT} \
                -p ${CODEGEN_SSH_PORT} \
                ${CODEGEN_SSH_USER}@${CODEGEN_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        else
            echo "[CodeGen] SSH tunnel already exists"
        fi
        CODEGEN_SERVER_URL="http://localhost:${LOCAL_CODEGEN_PORT}/v1"
    fi

    # Judge 서버 연결 설정
    if [ "$JUDGE_SSH_PORT" = "22" ]; then
        JUDGE_SERVER_URL="http://${JUDGE_SERVER_HOST}:${JUDGE_SERVER_PORT}/v1"
        echo "[Judge] Direct connection: ${JUDGE_SERVER_URL}"
    else
        LOCAL_JUDGE_PORT=$JUDGE_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT}.*${JUDGE_SERVER_HOST}" > /dev/null; then
            echo "[Judge] Creating SSH tunnel (localhost:${LOCAL_JUDGE_PORT} → ${JUDGE_SERVER_HOST}:${JUDGE_SERVER_PORT})..."
            ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT} \
                -p ${JUDGE_SSH_PORT} \
                ${JUDGE_SSH_USER}@${JUDGE_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        else
            echo "[Judge] SSH tunnel already exists"
        fi
        JUDGE_SERVER_URL="http://localhost:${LOCAL_JUDGE_PORT}/v1"
    fi

    echo ""
    echo "[Server] Checking connections..."

    if curl -s --connect-timeout 5 "${CODEGEN_SERVER_URL}/models" > /dev/null 2>&1; then
        echo "[Server] CodeGen LLM server OK (${CODEGEN_SERVER_URL})"
    else
        echo "[Server] WARNING: CodeGen LLM server not responding (${CODEGEN_SERVER_URL})"
    fi

    if curl -s --connect-timeout 5 "${JUDGE_SERVER_URL}/models" > /dev/null 2>&1; then
        echo "[Server] Judge VLM server OK (${JUDGE_SERVER_URL})"
    else
        echo "[Server] WARNING: Judge VLM server not responding (${JUDGE_SERVER_URL})"
    fi
    echo ""
fi

# ============================================================
# 설정 출력
# ============================================================

echo "========================================"
echo "Forward + Reset Pipeline"
echo "========================================"
echo "Instruction: $INSTRUCTION"
echo "Robot IDs: ${ROBOT_IDS[*]}"
echo "Num Episodes: $NUM_EPISODES"
echo "Save Dir: $SAVE_DIR"
echo ""
echo "--- Feature Toggles ---"
echo "Execute Reset: $EXECUTE_RESET"
echo "Random Seeds: $NUM_RANDOM_SEEDS"
echo "Use Server: $USE_SERVER"
echo "Record Dataset: $RECORD_DATASET"
echo "Multi-Turn: $MULTI_TURN"
echo ""
echo "--- Model Settings ---"
echo "LLM Model (Session 1): $LLM_MODEL"
if [ -n "$CODEGEN_SESSION2_MODEL" ]; then
    echo "CodeGen Model (Session 2): $CODEGEN_SESSION2_MODEL"
fi
echo "Judge Model: $JUDGE_MODEL"
echo "Judge Timeout: ${JUDGE_TIMEOUT}s"
if [ "$USE_SERVER" = true ]; then
    echo "  CodeGen URL: $CODEGEN_SERVER_URL"
    echo "  Judge URL: $JUDGE_SERVER_URL"
fi
if [ "$RECORD_DATASET" = true ]; then
    echo ""
    echo "--- Recording Settings ---"
    if [ -n "$DATASET_REPO_ID" ]; then
        echo "  Dataset Repo ID: $DATASET_REPO_ID"
    else
        echo "  Dataset Repo ID: (auto-generated)"
    fi
    echo "  Recording FPS: $RECORDING_FPS"
fi
echo "========================================"
echo ""

# ============================================================
# 실행 인자 구성
# ============================================================

EXTRA_ARGS=""

if [ "$EXECUTE_RESET" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-reset"
fi

if [ "$USE_SERVER" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-server"
    EXTRA_ARGS="$EXTRA_ARGS --codegen-server-url $CODEGEN_SERVER_URL"
    EXTRA_ARGS="$EXTRA_ARGS --codegen-model $CODEGEN_MODEL_NAME"
    EXTRA_ARGS="$EXTRA_ARGS --judge-server-url $JUDGE_SERVER_URL"
    EXTRA_ARGS="$EXTRA_ARGS --judge-server-model $JUDGE_MODEL_NAME"
fi

if [ "$RECORD_DATASET" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --record"
    if [ -n "$DATASET_REPO_ID" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --dataset-repo-id $DATASET_REPO_ID"
    fi
    EXTRA_ARGS="$EXTRA_ARGS --recording-fps $RECORDING_FPS"
fi

if [ "$MULTI_TURN" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --multi-turn"
fi

if [ ${#CAD_IMAGE_DIRS[@]} -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --cad-image-dirs ${CAD_IMAGE_DIRS[@]}"
fi

if [ -n "$SIDE_VIEW_IMAGE" ] && [ -f "$SIDE_VIEW_IMAGE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --side-view-image $SIDE_VIEW_IMAGE"
fi

if [ -n "$RESUME_SESSION" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME_SESSION"
fi

if [ ${#RESETSPACE_PER_ROBOT[@]} -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resetspace-per-robot ${RESETSPACE_PER_ROBOT[@]}"
fi

# ============================================================
# 파이프라인 실행
# ============================================================

CODEGEN_S2_ARG=""
if [ -n "$CODEGEN_SESSION2_MODEL" ]; then
    CODEGEN_S2_ARG="--codegen-session2-model $CODEGEN_SESSION2_MODEL"
fi

python execution_forward_and_reset.py \
    --instruction "$INSTRUCTION" \
    --robot ${ROBOT_IDS[@]} \
    --llm "$LLM_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --num-random-seeds "$NUM_RANDOM_SEEDS" \
    ${RESET_INSTRUCTION:+--reset-instruction "$RESET_INSTRUCTION"} \
    $( [ "$SKIP_TURN_TEST" = "true" ] && echo "--skip-turn-test" ) \
    --save "$SAVE_DIR" \
    --num-episodes "$NUM_EPISODES" \
    $CODEGEN_S2_ARG \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline completed with errors (exit code: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
