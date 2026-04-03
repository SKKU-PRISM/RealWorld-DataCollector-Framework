#!/bin/bash
#
# Judge VLM Server Runner (Qwen2-VL)
#
# vLLM OpenAI-compatible API 서버를 실행합니다. (멀티모달 VLM)
# Judge 태스크 평가용 Vision-Language Model 서빙.
#
# Usage:
#   ./run_judge_server.sh
#   ./run_judge_server.sh --background
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 설정 (Configuration)
MODEL="Qnvidia/Cosmos-Reason1-7B"  # 2B 권장, 7B는 더 정확
HOST="0.0.0.0"
PORT=8002  # LLM 서버(8001)와 분리
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
MAX_IMAGES=2  # Judge는 initial + final 이미지 2장


# 실행 (Execution)
echo "========================================"
echo "Judge VLM Server (vLLM)"
echo "========================================"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Images: $MAX_IMAGES"
echo "========================================"
echo ""
echo "API Endpoint: http://$HOST:$PORT/v1"
echo ""

# 백그라운드 실행 옵션
if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    echo "Starting server in background..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
        > "$PROJECT_ROOT/server_inference/judge_server.log" 2>&1 &

    echo "Server PID: $!"
    echo "Log file: $PROJECT_ROOT/server_inference/judge_server.log"
    echo ""
    echo "To check status: curl http://localhost:$PORT/v1/models"
    echo "To stop server: kill $!"
else
    echo "Press Ctrl+C to stop the server"
    echo ""
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}"
fi
