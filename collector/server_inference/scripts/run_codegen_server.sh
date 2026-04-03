#!/bin/bash
#
# Code Generation LLM Server Runner (Qwen2.5-Coder)
#
# vLLM OpenAI-compatible API 서버를 실행합니다. (텍스트 LLM)
# 코드 생성용 Language Model 서빙.
#
# Usage:
#   ./run_codegen_server.sh
#   ./run_codegen_server.sh --background
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 설정 (Configuration)
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
HOST="0.0.0.0"
PORT=8001
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9


# 실행 (Execution)
echo "========================================"
echo "Code Generation LLM Server (vLLM)"
echo "========================================"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
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
        > "$PROJECT_ROOT/server_inference/codegen_server.log" 2>&1 &

    echo "Server PID: $!"
    echo "Log file: $PROJECT_ROOT/server_inference/codegen_server.log"
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
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
fi
