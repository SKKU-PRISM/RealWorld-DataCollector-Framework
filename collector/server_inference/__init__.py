"""
Server Inference Package

vLLM 기반 LLM/VLM 원격 추론 모듈

서버 실행:
    # Code Generation LLM (텍스트)
    python -m server_inference.run_codegen_server
    ./server_inference/scripts/run_codegen_server.sh

    # Judge VLM (멀티모달)
    python -m server_inference.run_judge_server
    ./server_inference/scripts/run_judge_server.sh

클라이언트 사용:
    # LLM (코드 생성)
    code_gen_lerobot/llm.py에서 USE_LLM_SERVER=1 환경변수로 활성화

    # VLM (Judge)
    judge/vlm.py에서 USE_VLM_SERVER=1 환경변수로 활성화

환경변수:
    USE_LLM_SERVER=1     LLM 서버 모드 활성화
    VLLM_SERVER_URL      LLM 서버 URL (기본: http://localhost:8001/v1)
    VLLM_MODEL_NAME      LLM 모델명 (기본: Qwen/Qwen2.5-Coder-7B-Instruct)

    USE_VLM_SERVER=1     VLM 서버 모드 활성화
    VLM_SERVER_URL       VLM 서버 URL (기본: http://localhost:8002/v1)
    VLM_MODEL_NAME       VLM 모델명 (기본: Qwen/Qwen2-VL-2B-Instruct)
"""

__all__ = []
