#!/bin/bash
# =============================================================================
# ACS 통합 파이프라인: 데이터 수집 -> 변환 -> 학습
# =============================================================================
#
# 사용법:
#   # 1단계: 데이터 수집 (AutoDataCollector)
#   ./scripts/collect_and_train.sh collect "pick red block and place on blue dish"
#
#   # 2단계: 변환 (LeRobot -> NPZ)
#   ./scripts/collect_and_train.sh convert
#
#   # 3단계: 학습 (RoboBridge VLA LoRA)
#   ./scripts/collect_and_train.sh train pick_redblock_place_bluedish
#
#   # 전체 파이프라인 (변환 + 학습)
#   ./scripts/collect_and_train.sh pipeline pick_redblock_place_bluedish
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[ACS]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[ACS]${NC} $*"; }
log_error() { echo -e "${RED}[ACS]${NC} $*"; }

# ----- 수집 (AutoDataCollector) -----
cmd_collect() {
    local instruction="${1:-pick red block and place on blue dish}"
    log_info "데이터 수집 시작: '$instruction'"
    log_info "AutoDataCollector 실행..."

    cd collector
    python execution_forward_and_reset.py \
        -i "$instruction" \
        --record \
        --dataset-repo-id "user/$(echo "$instruction" | tr ' ' '_' | head -c 50)"
    cd "$ROOT_DIR"

    log_info "데이터 수집 완료"
}

# ----- 변환 (LeRobot -> NPZ) -----
cmd_convert() {
    local config="${1:-configs/tasks.yaml}"
    log_info "데이터 변환 시작 (config: $config)"

    python -m pipeline.convert --config "$config"

    log_info "변환 완료. 결과: data/ 디렉토리 확인"
}

# ----- 학습 (RoboBridge VLA LoRA) -----
cmd_train() {
    local task_name="${1:?태스크 이름을 지정하세요}"
    local train_config="${2:-bridge/configs/base.yaml}"

    log_info "학습 시작: $task_name"
    log_info "  데이터: data/$task_name"
    log_info "  설정: $train_config"

    if [ ! -d "data/$task_name" ]; then
        log_error "데이터 디렉토리가 없습니다: data/$task_name"
        log_error "먼저 변환을 실행하세요: $0 convert"
        exit 1
    fi

    python bridge/scripts/train/train_lora_movegrip.py \
        --config "$train_config" \
        --task "$task_name" \
        --data-dir "data/$task_name"

    log_info "학습 완료"
}

# ----- 평가 -----
cmd_eval() {
    local task_name="${1:?태스크 이름을 지정하세요}"
    local mode="${2:-direct}"
    log_info "평가 시작: $task_name (mode: $mode)"

    python bridge/scripts/eval/eval_so101.py \
        --tasks "$task_name" \
        --mode "$mode" \
        --num-episodes 5

    log_info "평가 완료"
}

# ----- 전체 파이프라인 (변환 + 학습) -----
cmd_pipeline() {
    local task_name="${1:?태스크 이름을 지정하세요}"
    log_info "=== 전체 파이프라인 시작: $task_name ==="

    cmd_convert
    cmd_train "$task_name"

    log_info "=== 파이프라인 완료 ==="
}

# ----- 도움말 -----
cmd_help() {
    cat << 'EOF'
ACS 통합 파이프라인 (AutoDataCollector + RoboBridge)

사용법: ./scripts/collect_and_train.sh <command> [args...]

Commands:
  collect <instruction>       AutoDataCollector로 데이터 수집
  convert [config.yaml]       LeRobot -> RoboBridge NPZ 변환
  train <task_name> [config]  RoboBridge VLA LoRA 학습
  eval <task_name> [mode]     학습된 모델 평가
  pipeline <task_name>        변환 + 학습 한번에 실행
  help                        이 도움말 표시

전체 워크플로우:
  1. collector/ 에서 데이터 수집 (AutoDataCollector 직접 실행)
  2. ./scripts/collect_and_train.sh convert
  3. ./scripts/collect_and_train.sh train pick_redblock_place_bluedish
EOF
}

# ----- 메인 -----
case "${1:-help}" in
    collect)  shift; cmd_collect "$@" ;;
    convert)  shift; cmd_convert "$@" ;;
    train)    shift; cmd_train "$@" ;;
    eval)     shift; cmd_eval "$@" ;;
    pipeline) shift; cmd_pipeline "$@" ;;
    help|*)   cmd_help ;;
esac
