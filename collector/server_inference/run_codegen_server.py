#!/usr/bin/env python3
"""
Code Generation LLM Server Runner (Qwen2.5-Coder)

vLLM OpenAI-compatible API 서버를 실행합니다. (텍스트 LLM)
코드 생성용 Language Model 서빙.

Usage:
    python run_codegen_server.py
    python run_codegen_server.py --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8001
    python run_codegen_server.py --help
"""

import argparse
import os
import subprocess
import sys


# 기본 설정
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_MAX_MODEL_LEN = 4096  # 메모리 절약을 위해 제한
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def run_vllm_server(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
):
    """
    vLLM OpenAI-compatible API 서버 실행

    Args:
        model: HuggingFace 모델명
        host: 서버 호스트
        port: 서버 포트
        max_model_len: 최대 컨텍스트 길이
        gpu_memory_utilization: GPU 메모리 사용률
    """
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    print("=" * 60)
    print("Code Generation LLM Server Starting (vLLM)")
    print("=" * 60)
    print(f"  Model: {model}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Max Model Len: {max_model_len}")
    print(f"  GPU Memory Utilization: {gpu_memory_utilization}")
    print("=" * 60)
    print(f"\nAPI Endpoint: http://{host}:{port}/v1")
    print("Press Ctrl+C to stop the server\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Code Generation LLM Server (vLLM OpenAI-compatible API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_codegen_server.py
    python run_codegen_server.py --model Qwen/Qwen2.5-Coder-7B-Instruct
    python run_codegen_server.py --port 8002
    python run_codegen_server.py --max-model-len 8192
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum context length (default: {DEFAULT_MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization (default: {DEFAULT_GPU_MEMORY_UTILIZATION})"
    )

    args = parser.parse_args()

    run_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
