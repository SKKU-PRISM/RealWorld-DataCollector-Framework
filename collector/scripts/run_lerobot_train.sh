#!/bin/bash
# LeRobot Training Script for CaP Dataset
#
# Usage:
#   ./scripts/run_lerobot_train.sh cap_dataset_20260106_212506
#   ./scripts/run_lerobot_train.sh cap_dataset_20260106_212506 --steps 10000
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ============================================================
# Configuration
# ============================================================

# Dataset name (required argument)
DATASET_NAME="${1:-}"
if [ -z "$DATASET_NAME" ]; then
    echo "Usage: $0 <dataset_name> [additional_args...]"
    echo ""
    echo "Example:"
    echo "  $0 cap_dataset_20260106_212506"
    echo "  $0 cap_dataset_20260106_212506 --steps 10000 --batch_size 16"
    echo ""
    echo "Available datasets:"
    ls -1 ~/.cache/huggingface/lerobot/local/ 2>/dev/null | grep "^cap_" || echo "  (no CaP datasets found)"
    exit 1
fi
shift  # Remove dataset name from arguments

# Dataset paths
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/local/$DATASET_NAME"
DATASET_REPO_ID="local/$DATASET_NAME"

# Verify dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "[ERROR] Dataset not found: $DATASET_ROOT"
    exit 1
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[ERROR] Invalid dataset (missing meta/info.json): $DATASET_ROOT"
    exit 1
fi

# Default training parameters
POLICY_TYPE="diffusion"
STEPS=50000
BATCH_SIZE=32
LOG_FREQ=100
SAVE_FREQ=5000
EVAL_FREQ=0  # Disable eval for real robot data
NUM_WORKERS=4
OUTPUT_DIR="outputs/train_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"

# ============================================================
# Display configuration
# ============================================================

echo "========================================"
echo "LeRobot Training Pipeline"
echo "========================================"
echo ""
echo "Dataset:"
echo "  Name: $DATASET_NAME"
echo "  Path: $DATASET_ROOT"
echo "  Repo ID: $DATASET_REPO_ID"
echo ""

# Show dataset info
echo "Dataset Info:"
python3 -c "
import json
with open('$DATASET_ROOT/meta/info.json') as f:
    info = json.load(f)
print(f'  Episodes: {info.get(\"total_episodes\", \"?\")}')
print(f'  Frames: {info.get(\"total_frames\", \"?\")}')
print(f'  FPS: {info.get(\"fps\", \"?\")}')
"
echo ""

echo "Training Config:"
echo "  Policy: $POLICY_TYPE"
echo "  Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Additional args: $@"
echo "========================================"
echo ""

# ============================================================
# Run training
# ============================================================

# Set Python path
export PYTHONPATH="$PROJECT_DIR/lerobot/src:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training with lerobot_train.py
echo "[Training] Starting..."
echo ""

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type="$POLICY_TYPE" \
    --policy.push_to_hub=false \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --eval_freq="$EVAL_FREQ" \
    --num_workers="$NUM_WORKERS" \
    --output_dir="$OUTPUT_DIR" \
    "$@"

echo ""
echo "========================================"
echo "[Training] Complete!"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
