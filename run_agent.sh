#!/bin/bash
# =============================================================================
# ACS 통합 실행 스크립트
#   STAGE 1: 데이터 수집  (AutoDataCollector — collector/)
#   STAGE 2: VLA 학습     (RoboBridge LoRA — bridge/)
#
# 사용법:
#   ./run_agent.sh                                # 전체 (수집→학습)
#   ./run_agent.sh --stage collect                # 데이터 수집만
#   ./run_agent.sh --stage train                  # VLA LoRA 학습만
#   ./run_agent.sh --stage collect,train          # 수집+학습
#
# 환경변수 (API 키 — 이미지 내 하드코딩 금지):
#   GOOGLE_API_KEY    (필수) Gemini API 키
#   OPENAI_API_KEY    (선택) OpenAI API 키
#   DEEPSEEK_API_KEY  (선택) DeepSeek API 키
#
# 실행 예시:
#   docker run --gpus all --rm \
#       -e GOOGLE_API_KEY="your-key" \
#       -v $(pwd)/outputs:/app/outputs \
#       acs ./run_agent.sh --stage train
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# 인자 파싱
# =============================================================================

STAGES="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)   STAGES="$2"; shift 2 ;;
        --stage=*) STAGES="${1#--stage=}"; shift ;;
        *)         break ;;
    esac
done

if [ "$STAGES" = "all" ]; then
    STAGES="collect,train"
fi

# =============================================================================
# 데이터 수집 설정 (collector)
# =============================================================================
INSTRUCTION="${INSTRUCTION:-pick up the red block and place it on the blue dish}"
RESET_INSTRUCTION="${RESET_INSTRUCTION:-}"
ROBOT_IDS="${ROBOT_IDS:-2 3}"
NUM_EPISODES="${NUM_EPISODES:-30}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-15}"
COLLECT_USE_SERVER="${COLLECT_USE_SERVER:-false}"
RECORD_DATASET="${RECORD_DATASET:-true}"
MULTI_TURN="${MULTI_TURN:-true}"
SKIP_TURN_TEST="${SKIP_TURN_TEST:-true}"

# =============================================================================
# VLA 학습 설정 (bridge — train_lora.py)
# =============================================================================
TRAIN_CONFIG="${TRAIN_CONFIG:-}"                          # YAML 설정 파일 (예: configs/multitask_groot.yaml)
TRAIN_MODEL_BACKEND="${TRAIN_MODEL_BACKEND:-groot_n1.5}"  # groot_n1.5 | smolvla | pi05 | openvla
TRAIN_MODEL_NAME="${TRAIN_MODEL_NAME:-nvidia/GR00T-N1.5-3B}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-16}"
TRAIN_LORA_RANK="${TRAIN_LORA_RANK:-64}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
TRAIN_SCHEDULER="${TRAIN_SCHEDULER:-cosine}"
TRAIN_HDF5_DIR="${TRAIN_HDF5_DIR:-}"                     # HDF5 직접 모드 (전처리 스킵)
TRAIN_PROCESSED_DIR="${TRAIN_PROCESSED_DIR:-}"            # NPZ 전처리 데이터
TRAIN_TASK="${TRAIN_TASK:-}"                              # 특정 태스크만 학습
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./outputs/vla_adapters}"

# =============================================================================
# 공통
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ACS: AutoDataCollector + RoboBridge 통합 파이프라인     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Stages : $STAGES"
echo "║  시작   : $(date)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# STAGE 1: 데이터 수집 (AutoDataCollector)
# =============================================================================

if [[ "$STAGES" == *"collect"* ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STAGE 1: 데이터 수집 (AutoDataCollector)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Instruction : $INSTRUCTION"
    echo "  Robot IDs   : $ROBOT_IDS"
    echo "  Episodes    : $NUM_EPISODES"
    echo "  Multi-turn  : $MULTI_TURN"
    echo ""

    # API 키 검증
    if [ -z "$GOOGLE_API_KEY" ] && [ "$COLLECT_USE_SERVER" = "false" ]; then
        echo "[ERROR] GOOGLE_API_KEY 환경변수가 필요합니다."
        echo "  docker run -e GOOGLE_API_KEY='your-key' ..."
        exit 1
    fi

    cd "$SCRIPT_DIR/collector"
    export PYTHONPATH="$SCRIPT_DIR/collector/lerobot/src:${PYTHONPATH:-}"

    COLLECT_ARGS="--instruction \"$INSTRUCTION\""
    COLLECT_ARGS="$COLLECT_ARGS --robot $ROBOT_IDS"
    COLLECT_ARGS="$COLLECT_ARGS --num-episodes $NUM_EPISODES"
    COLLECT_ARGS="$COLLECT_ARGS --num-random-seeds $NUM_RANDOM_SEEDS"
    COLLECT_ARGS="$COLLECT_ARGS --save $OUTPUT_DIR/collect_$TIMESTAMP"

    [ "$RECORD_DATASET" = "true" ]  && COLLECT_ARGS="$COLLECT_ARGS --record"
    [ "$MULTI_TURN" = "true" ]      && COLLECT_ARGS="$COLLECT_ARGS --multi-turn"
    [ "$SKIP_TURN_TEST" = "true" ]  && COLLECT_ARGS="$COLLECT_ARGS --skip-turn-test"
    [ -n "$RESET_INSTRUCTION" ]     && COLLECT_ARGS="$COLLECT_ARGS --reset-instruction \"$RESET_INSTRUCTION\""

    # API 설정
    if [ "$COLLECT_USE_SERVER" = "false" ]; then
        CONFIG_FILE="$SCRIPT_DIR/collector/pipeline_config/paid_api_config.yaml"
        if [ -f "$CONFIG_FILE" ]; then
            LLM_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['codegen_llm_model'])")
            JUDGE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['judge_vlm_model'])")
            JUDGE_TIMEOUT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['judge_timeout'])")
        else
            LLM_MODEL="gemini-3.1-flash-lite-preview"
            JUDGE_MODEL="gemini-2.5-flash"
            JUDGE_TIMEOUT="0.5"
        fi
    else
        COLLECT_ARGS="$COLLECT_ARGS --use-server"
        LLM_MODEL="${CODEGEN_MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
        JUDGE_MODEL="${JUDGE_MODEL_NAME:-nvidia/Cosmos-Reason1-7B}"
        JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-0.5}"
    fi

    echo "[Collect] Starting..."
    eval python execution_forward_and_reset.py \
        $COLLECT_ARGS \
        --llm "$LLM_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --judge-timeout "$JUDGE_TIMEOUT" \
    || echo "[Collect] WARNING: finished with errors"

    echo "[Collect] Done."
    cd "$SCRIPT_DIR"
    echo ""
fi

# =============================================================================
# STAGE 2: VLA LoRA 학습 (RoboBridge — train_lora.py)
# =============================================================================

if [[ "$STAGES" == *"train"* ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STAGE 2: VLA LoRA 학습 (RoboBridge)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TRAIN_SCRIPT="$SCRIPT_DIR/bridge/multitask_training_package/train_lora.py"
    TRAIN_ARGS=""

    # 설정 파일 결정
    if [ -n "$TRAIN_CONFIG" ]; then
        TRAIN_ARGS="--config $TRAIN_CONFIG"
    else
        # 모델 백엔드에 따른 기본 설정 파일 선택
        case "$TRAIN_MODEL_BACKEND" in
            groot_n1.5|groot) DEFAULT_CONFIG="$SCRIPT_DIR/bridge/configs/base.yaml" ;;
            smolvla)          DEFAULT_CONFIG="$SCRIPT_DIR/bridge/multitask_training_package/configs/multitask_smolvla.yaml" ;;
            pi05)             DEFAULT_CONFIG="$SCRIPT_DIR/bridge/multitask_training_package/configs/multitask_pi05.yaml" ;;
            openvla)          DEFAULT_CONFIG="$SCRIPT_DIR/bridge/multitask_training_package/configs/multitask_openvla.yaml" ;;
            *)                DEFAULT_CONFIG="$SCRIPT_DIR/bridge/configs/base.yaml" ;;
        esac

        if [ -f "$DEFAULT_CONFIG" ]; then
            TRAIN_ARGS="--config $DEFAULT_CONFIG"
        else
            echo "[Train] WARNING: Config file not found: $DEFAULT_CONFIG"
        fi
    fi

    # CLI 오버라이드
    TRAIN_ARGS="$TRAIN_ARGS --lr $TRAIN_LR"
    TRAIN_ARGS="$TRAIN_ARGS --epochs $TRAIN_EPOCHS"
    TRAIN_ARGS="$TRAIN_ARGS --batch-size $TRAIN_BATCH_SIZE"
    TRAIN_ARGS="$TRAIN_ARGS --grad-accum $TRAIN_GRAD_ACCUM"
    TRAIN_ARGS="$TRAIN_ARGS --lora-rank $TRAIN_LORA_RANK"
    TRAIN_ARGS="$TRAIN_ARGS --scheduler-type $TRAIN_SCHEDULER"
    TRAIN_ARGS="$TRAIN_ARGS --output-base-dir $TRAIN_OUTPUT_DIR"

    [ -n "$TRAIN_MAX_STEPS" ]    && TRAIN_ARGS="$TRAIN_ARGS --max-steps $TRAIN_MAX_STEPS"
    [ -n "$TRAIN_HDF5_DIR" ]     && TRAIN_ARGS="$TRAIN_ARGS --hdf5-dir $TRAIN_HDF5_DIR"
    [ -n "$TRAIN_PROCESSED_DIR" ] && TRAIN_ARGS="$TRAIN_ARGS --processed-dir $TRAIN_PROCESSED_DIR"
    [ -n "$TRAIN_TASK" ]         && TRAIN_ARGS="$TRAIN_ARGS --task $TRAIN_TASK"

    echo "  Backend     : $TRAIN_MODEL_BACKEND"
    echo "  Model       : $TRAIN_MODEL_NAME"
    echo "  LoRA rank   : $TRAIN_LORA_RANK"
    echo "  LR          : $TRAIN_LR"
    echo "  Epochs      : $TRAIN_EPOCHS"
    echo "  Batch       : ${TRAIN_BATCH_SIZE}x${TRAIN_GRAD_ACCUM} (effective=$(( TRAIN_BATCH_SIZE * TRAIN_GRAD_ACCUM )))"
    echo "  Output      : $TRAIN_OUTPUT_DIR"
    echo ""

    echo "[Train] Starting VLA LoRA training..."
    cd "$SCRIPT_DIR/bridge/multitask_training_package"

    python "$TRAIN_SCRIPT" $TRAIN_ARGS \
    || { echo "[Train] ERROR: Training failed!"; exit 1; }

    echo "[Train] Done. Adapters saved to: $TRAIN_OUTPUT_DIR"
    cd "$SCRIPT_DIR"
    echo ""
fi

# =============================================================================
# 완료
# =============================================================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ACS 파이프라인 완료                                     ║"
echo "║  종료: $(date)"
echo "║  결과: $OUTPUT_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
