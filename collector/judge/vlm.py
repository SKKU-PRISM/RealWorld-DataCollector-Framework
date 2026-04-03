"""
VLM Response Module for Judge

vLLM 서버 또는 OpenAI API를 통한 VLM(Vision-Language Model) 호출.
code_gen_lerobot/llm.py와 동일한 패턴으로 구현.

서버 모드:
- USE_VLM_SERVER=1 환경변수 또는 use_server=True 파라미터로 활성화
- VLM_SERVER_URL로 서버 주소 설정 가능 (기본: http://localhost:8002/v1)
- VLM_MODEL_NAME로 모델명 설정 가능 (기본: Qwen/Qwen2-VL-2B-Instruct)

API 모드:
- OpenAI API 사용 (gpt-4o, gpt-4o-mini 등)
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional


# VLM 서버 설정 (환경변수로 override 가능)
# JUDGE_SERVER_URL을 우선 사용, 없으면 VLM_SERVER_URL 사용
VLM_SERVER_URL = os.getenv("JUDGE_SERVER_URL") or os.getenv("VLM_SERVER_URL", "http://localhost:8002/v1")
VLM_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME") or os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")


def _load_api_keys():
    """API 키 로드 (환경변수 우선 → JSON 파일 fallback)"""
    keys = {}
    key_file = Path(__file__).parent.parent / "openai_api_key.json"
    if key_file.exists():
        with open(key_file, "r") as f:
            keys = json.load(f)
    if os.getenv("OPENAI_API_KEY"):
        keys["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    return keys


def _should_use_server(use_server: bool = None) -> bool:
    """서버 사용 여부 결정 (파라미터 > 환경변수)"""
    if use_server is not None:
        return use_server
    # USE_VLM_SERVER 또는 USE_LLM_SERVER 환경변수 확인
    use_vlm = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
    use_llm = os.getenv("USE_LLM_SERVER", "").lower() in ("1", "true", "yes")
    return use_vlm or use_llm


def _call_vlm_server(
    prompt: str,
    images_b64: List[str],
    max_tokens: int = 1000,
    temperature: float = 0.0,
    check_time: bool = True,
) -> Optional[str]:
    """
    vLLM VLM 서버 호출 (OpenAI-compatible API)

    Args:
        prompt: 텍스트 프롬프트
        images_b64: base64 인코딩된 이미지 리스트
        max_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        check_time: 시간 출력 여부

    Returns:
        생성된 텍스트 또는 실패 시 None
    """
    from openai import OpenAI
    import httpx

    # 함수 호출 시점에 환경변수 읽기 (모듈 로드 시점이 아닌)
    server_url = os.getenv("JUDGE_SERVER_URL") or os.getenv("VLM_SERVER_URL", "http://localhost:8002/v1")
    model_name = os.getenv("JUDGE_MODEL_NAME") or os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")

    # vLLM 서버는 모델 로딩/이미지 처리에 시간이 걸릴 수 있음
    client = OpenAI(
        base_url=server_url,
        api_key="not-needed",  # vLLM 로컬 서버는 API 키 불필요
        timeout=httpx.Timeout(120.0, connect=10.0),  # 응답 120초, 연결 10초
    )

    # 멀티모달 메시지 구성
    content = [{"type": "text", "text": prompt}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if check_time:
            elapsed = time.time() - start_time
            print(f"[VLM Server] Model: {model_name}, Response time: {elapsed:.2f}s")

        return response.choices[0].message.content

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[VLM Server] Error ({elapsed:.2f}s): {e}")
        return None


def _call_openai_vlm(
    prompt: str,
    images_b64: List[str],
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    check_time: bool = True,
) -> Optional[str]:
    """
    OpenAI VLM API 호출 (유료)

    Args:
        prompt: 텍스트 프롬프트
        images_b64: base64 인코딩된 이미지 리스트
        model: 모델명 (gpt-4o, gpt-4o-mini 등)
        max_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        check_time: 시간 출력 여부

    Returns:
        생성된 텍스트 또는 실패 시 None
    """
    from openai import OpenAI

    api_keys = _load_api_keys()
    client = OpenAI(api_key=api_keys.get("openai_api_key"))

    # 멀티모달 메시지 구성
    content = [{"type": "text", "text": prompt}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high",
            }
        })

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )

        if check_time:
            elapsed = time.time() - start_time
            print(f"[OpenAI VLM] Model: {model}, Response time: {elapsed:.2f}s")

        return response.choices[0].message.content

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[OpenAI VLM] Error ({elapsed:.2f}s): {e}")
        return None


def _is_gemini_model(model: str) -> bool:
    """Gemini 모델 여부 판단"""
    return "gemini" in model.lower()


def _call_gemini_vlm(
    prompt: str,
    images_b64: List[str],
    model: str = "gemini-2.5-flash",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    check_time: bool = True,
) -> Optional[str]:
    """
    Google AI Studio Gemini VLM 호출 (base64 이미지 지원)

    Args:
        prompt: 텍스트 프롬프트
        images_b64: base64 인코딩된 이미지 리스트
        model: Gemini 모델명
        max_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        check_time: 시간 출력 여부

    Returns:
        생성된 텍스트 또는 실패 시 None
    """
    import base64 as b64_mod

    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError, ServerError
    except ImportError as e:
        print(f"[Gemini VLM] Failed to import google-genai SDK: {e}")
        return None

    try:
        from code_gen_lerobot.llm_utils.gemini import _get_client
        client = _get_client()
    except ImportError:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)

    # 컨텐츠 구성: 텍스트 + base64 이미지들
    contents = [prompt]
    for img_b64 in images_b64:
        img_bytes = b64_mod.b64decode(img_b64)
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()

    try:
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                break
            except (ClientError, ServerError) as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                    delay = 30 * (2 ** attempt)
                    print(f"  [Gemini VLM] Rate limit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        elapsed = time.time() - start_time

        # usage 추출 + 저장
        usage_dict = {"model": model, "inference_time_s": round(elapsed, 2), "in": 0, "out": 0, "total": 0}
        try:
            usage = response.usage_metadata
            if usage:
                usage_dict["in"] = getattr(usage, 'prompt_token_count', 0) or 0
                usage_dict["out"] = getattr(usage, 'candidates_token_count', 0) or 0
                usage_dict["total"] = getattr(usage, 'total_token_count', 0) or 0
        except Exception:
            pass
        _call_gemini_vlm._last_usage = usage_dict

        if check_time:
            parts = [f"{k}={v}" for k, v in usage_dict.items() if k not in ("model", "inference_time_s") and v > 0]
            token_str = f" ({', '.join(parts)})" if parts else ""
            print(f"[Gemini VLM] Model: {model}, Response time: {elapsed:.2f}s{token_str}")

        return response.text

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Gemini VLM] Error ({elapsed:.2f}s): {e}")
        return None


def vlm_response(
    prompt: str,
    images_b64: List[str],
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    check_time: bool = True,
    use_server: bool = None,
) -> Optional[str]:
    """
    VLM 응답 생성 (메인 인터페이스)

    모델명에 따라 자동 라우팅:
    - "gemini" 포함 → Vertex AI Gemini
    - use_server=True → vLLM 서버
    - 그 외 → OpenAI API
    """
    # vLLM 서버 사용 여부 결정
    if _should_use_server(use_server):
        return _call_vlm_server(
            prompt=prompt,
            images_b64=images_b64,
            max_tokens=max_tokens,
            temperature=temperature,
            check_time=check_time,
        )

    # Gemini 모델이면 Vertex AI 사용
    if _is_gemini_model(model):
        return _call_gemini_vlm(
            prompt=prompt,
            images_b64=images_b64,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            check_time=check_time,
        )

    # 기존 OpenAI API 호출
    return _call_openai_vlm(
        prompt=prompt,
        images_b64=images_b64,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        check_time=check_time,
    )


# 테스트
if __name__ == "__main__":
    print("VLM Response Module Test")
    print("=" * 60)

    # 테스트용 더미 이미지 (1x1 빨간 픽셀)
    import base64
    dummy_image = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
        b'\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()

    print(f"VLM_SERVER_URL: {VLM_SERVER_URL}")
    print(f"VLM_MODEL_NAME: {VLM_MODEL_NAME}")
    print(f"USE_VLM_SERVER: {os.getenv('USE_VLM_SERVER', 'not set')}")
    print()

    # 서버 모드 테스트
    if _should_use_server():
        print("Testing server mode...")
        response = vlm_response(
            prompt="Describe this image briefly.",
            images_b64=[dummy_image],
            use_server=True,
        )
        print(f"Response: {response}")
    else:
        print("Server mode not enabled. Set USE_VLM_SERVER=1 to test.")
        print("Testing API mode...")
        response = vlm_response(
            prompt="Describe this image briefly.",
            images_b64=[dummy_image],
            model="gpt-4o-mini",
            use_server=False,
        )
        print(f"Response: {response}")
