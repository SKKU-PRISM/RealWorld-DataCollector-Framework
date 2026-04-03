#!/usr/bin/env python3
"""
Judge VLM Server Runner (Qwen2-VL)

vLLM OpenAI-compatible API 서버를 실행합니다. (멀티모달 VLM)
Judge 태스크 평가용 Vision-Language Model 서빙.

Usage:
    python run_judge_server.py
    python run_judge_server.py --model Qwen/Qwen2-VL-7B-Instruct --port 8002
    python run_judge_server.py --help
"""

import argparse
import subprocess
import sys


# 기본 설정
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # 2B는 가벼움, 7B는 더 정확
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8002  # LLM 서버(8001)와 분리
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MAX_IMAGES = 2  # Judge는 initial + final 이미지 2장


def run_judge_server(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_images: int = DEFAULT_MAX_IMAGES,
):
    """
    vLLM VLM 서버 실행 (Judge용)

    Args:
        model: HuggingFace 모델명 (VLM)
        host: 서버 호스트
        port: 서버 포트
        max_model_len: 최대 컨텍스트 길이
        gpu_memory_utilization: GPU 메모리 사용률
        max_images: 요청당 최대 이미지 수
    """
    import json
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",  # VLM 모델 필수
        "--limit-mm-per-prompt", json.dumps({"image": max_images}),
    ]

    print("=" * 60)
    print("Judge VLM Server Starting (vLLM)")
    print("=" * 60)
    print(f"  Model: {model}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Max Model Len: {max_model_len}")
    print(f"  GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"  Max Images per Request: {max_images}")
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
        description="Judge VLM Server (vLLM OpenAI-compatible API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 기본 실행 (Qwen2-VL-2B, 포트 8002)
    python run_judge_server.py

    # 7B 모델 사용 (더 정확하지만 느림)
    python run_judge_server.py --model Qwen/Qwen2-VL-7B-Instruct

    # 포트 변경
    python run_judge_server.py --port 8003

Supported VLM Models:
    - Qwen/Qwen2-VL-2B-Instruct  (권장, 가벼움)
    - Qwen/Qwen2-VL-7B-Instruct  (더 정확)
    - Qwen/Qwen3-VL-2B-Instruct  (최신)
    - llava-hf/llava-1.5-7b-hf
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace VLM model name (default: {DEFAULT_MODEL})"
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
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f"Max images per request (default: {DEFAULT_MAX_IMAGES})"
    )

    args = parser.parse_args()

    run_judge_server(
        model=args.model,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
