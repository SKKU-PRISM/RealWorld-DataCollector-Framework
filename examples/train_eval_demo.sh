#!/bin/bash
# =============================================================================
# ACS Demo: CloseDrawer 예시 데이터로 학습 → 평가 테스트
#
# 사용법:
#   ./examples/run_demo.sh              # 학습 + 평가 (dry-run)
#   ./examples/run_demo.sh --train      # 학습만
#   ./examples/run_demo.sh --eval       # 평가만 (dry-run)
#
# 필요 환경: robobridge conda 환경 (torch, transformers, lerobot, etc.)
#   conda activate robobridge
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE="${1:-all}"
DEMO_DATA="$SCRIPT_DIR/demo_data"
DEMO_OUTPUT="$SCRIPT_DIR/demo_output"
TRAIN_STEPS=20
BATCH_SIZE=1
GRAD_ACCUM=1
LORA_RANK=8

# =============================================================================
# Auto-download example data from HuggingFace if not present
# =============================================================================

if [ ! -f "$DEMO_DATA/metadata.json" ]; then
    echo "[Setup] Example data not found. Downloading from HuggingFace..."
    mkdir -p "$DEMO_DATA"

    # Download small example data (~13MB) from acs-example-data
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download skkuprism/acs-example-data \
            --repo-type dataset \
            --local-dir "$DEMO_DATA"
    elif command -v python &> /dev/null; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'skkuprism/acs-example-data',
    repo_type='dataset',
    local_dir='$DEMO_DATA',
)
print('Download complete.')
"
    else
        echo "[Setup] ERROR: huggingface-cli or python required to download data."
        echo "  pip install huggingface-hub"
        exit 1
    fi

    echo "[Setup] Example data downloaded to: $DEMO_DATA"
    echo ""
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ACS Demo: CloseDrawer 학습 + 평가                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data    : $DEMO_DATA"
echo "║  Output  : $DEMO_OUTPUT"
echo "║  Steps   : $TRAIN_STEPS"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# STEP 1: 학습 (GROOT N1.5 LoRA, train_lora_movegrip.py)
# =============================================================================

if [ "$MODE" = "all" ] || [ "$MODE" = "--train" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STEP 1: VLA LoRA 학습 (CloseDrawer)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TRAIN_SCRIPT="$PROJECT_DIR/bridge/scripts/train/train_lora_movegrip.py"
    CONFIG="$PROJECT_DIR/bridge/multitask_training_package/configs/models/groot.yaml"

    echo "  Script      : train_lora_movegrip.py"
    echo "  Config      : $CONFIG"
    echo "  Data dir    : $DEMO_DATA"
    echo "  Steps       : $TRAIN_STEPS"
    echo "  LoRA rank   : $LORA_RANK"
    echo "  Batch       : ${BATCH_SIZE}x${GRAD_ACCUM}"
    echo ""

    cd "$PROJECT_DIR/bridge/scripts/train"

    python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --processed-dir "$DEMO_DATA" \
        --output-base-dir "$DEMO_OUTPUT/adapters" \
        --max-steps "$TRAIN_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --lora-rank "$LORA_RANK" \
        --lr 1e-4 \
        --epochs 1 \
        --task CloseDrawer \
    || { echo "[Train] ERROR: Training failed!"; exit 1; }

    echo "[Train] Done. Output: $DEMO_OUTPUT/adapters"
    cd "$PROJECT_DIR"
    echo ""
fi

# =============================================================================
# STEP 2: 평가 (SO-101 dry-run, 로봇 불필요)
# =============================================================================

if [ "$MODE" = "all" ] || [ "$MODE" = "--eval" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STEP 2: RoboCasa 시뮬레이션 평가"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EVAL_SCRIPT="$PROJECT_DIR/bridge/scripts/eval/eval_vla_robocasa.py"

    # 학습된 어댑터 탐지
    ADAPTER_DIR="$DEMO_OUTPUT/adapters"
    MOVE_ADAPTER=$(find "$ADAPTER_DIR" -name "checkpoint-best" -type d 2>/dev/null | head -1)
    if [ -z "$MOVE_ADAPTER" ]; then
        MOVE_ADAPTER=$(find "$ADAPTER_DIR" -name "checkpoint-final" -type d 2>/dev/null | head -1)
    fi

    if [ -z "$MOVE_ADAPTER" ]; then
        echo "[Eval] WARNING: No trained adapter found at $ADAPTER_DIR"
        echo "[Eval] Run with --train first."
        exit 1
    fi

    STATS_FILE="$DEMO_DATA/metadata.json"

    echo "  Move adapter: $MOVE_ADAPTER"
    echo "  Stats file  : $STATS_FILE"
    echo "  Task        : CloseDrawer"
    echo "  Episodes    : 2"
    echo ""

    python "$EVAL_SCRIPT" \
        --model groot \
        --move-adapter "$MOVE_ADAPTER" \
        --tasks CloseDrawer \
        --num-episodes 2 \
        --output-dir "$DEMO_OUTPUT/eval_results" \
        --action-stats "$STATS_FILE" \
    || echo "[Eval] WARNING: Evaluation finished with errors"

    echo ""
    echo "[Eval] Done. Results: $DEMO_OUTPUT/eval_results"
    echo ""
fi

# =============================================================================
# 완료
# =============================================================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Demo 완료                                               ║"
echo "║  학습 결과 : $DEMO_OUTPUT/adapters"
echo "║  평가 결과 : $DEMO_OUTPUT/eval_results"
echo "╚══════════════════════════════════════════════════════════╝"
