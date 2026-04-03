import os
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# Google AI Studio API Key (환경변수 → JSON 파일 순으로 로드)
def _load_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    key_file = os.path.join(os.path.dirname(__file__), "..", "..", "google_aistudio_key.json")
    if os.path.exists(key_file):
        import json
        with open(key_file) as f:
            return json.load(f).get("api_key", "")
    return ""

GOOGLE_API_KEY = _load_api_key()

MAX_RETRIES = 10
RETRY_DELAY = 10  # seconds (fixed interval)
GEMINI3_DEFAULT_THINKING_BUDGET = 0  # Gemini 3 thinking 비활성화 (0 = no thinking)

# Singleton client
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Google AI Studio client 싱글톤."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _make_gen_config(model: str, temperature: float = 0.0, **kwargs) -> types.GenerateContentConfig:
    """GenerateContentConfig 생성. Gemini 3 모델은 thinking budget 자동 적용."""
    config_kwargs = {"temperature": temperature}

    if "gemini-3" in model.lower():
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=GEMINI3_DEFAULT_THINKING_BUDGET
        )

    # stop_sequences 처리
    if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
        config_kwargs["stop_sequences"] = kwargs["stop_sequences"]

    return types.GenerateContentConfig(**config_kwargs)


def _load_image(image_path: str):
    """이미지 파일을 로드하여 Part로 변환."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 확장자로 MIME 타입 결정
    ext = image_path.lower().rsplit(".", 1)[-1] if "." in image_path else "jpg"
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    mime_type = mime_map.get(ext, "image/jpeg")

    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def _build_contents(turn: Dict) -> list:
    """턴 dict에서 API contents 리스트 구성.

    지원하는 키:
        text: 프롬프트 텍스트 (필수)
        image_path: 단일 이미지 경로 (옵션)
        image_paths: 복수 이미지 경로 리스트 (옵션)
    """
    contents = [turn["text"]]
    if turn.get("image_path"):
        contents.append(_load_image(turn["image_path"]))
    for img_path in turn.get("image_paths", []):
        contents.append(_load_image(img_path))
    return contents


def _send_with_retry(chat, contents, config, max_retries=MAX_RETRIES):
    """Rate limit / Service Unavailable 시 재시도."""
    from google.genai.errors import ClientError, ServerError

    for attempt in range(max_retries + 1):
        try:
            return chat.send_message(contents, config=config)
        except (ClientError, ServerError) as e:
            err_str = str(e)
            if attempt == max_retries:
                raise
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                err_type = "Rate limit (429)"
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                err_type = "503 Unavailable"
            elif "404" in err_str:
                err_type = "404 NotFound"
            else:
                raise  # 재시도 불가능한 에러
            print(f"  [{err_type}] Waiting {RETRY_DELAY}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(RETRY_DELAY)


# ============================================================
# Dynamic chat session API
# ============================================================

def gemini_chat_start(
    model: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    thinking_budget: int = None,
) -> Tuple:
    """Chat 세션 시작. (chat, gen_config) 튜플 반환.

    Args:
        model: Gemini 모델 이름
        system_prompt: 시스템 프롬프트
        temperature: 샘플링 온도
        thinking_budget: thinking 토큰 제한 (Gemini 3 전용).

    Returns:
        (chat_session, generation_config) 튜플
    """
    client = _get_client()

    config = _make_gen_config(model, temperature=temperature)

    # thinking_budget 파라미터 우선
    if "gemini-3" in model.lower() and thinking_budget is not None:
        config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

    if "gemini-3" in model.lower():
        budget = config.thinking_config.thinking_budget if config.thinking_config else "N/A"
        print(f"[GEMINI] Thinking budget: {budget} tokens")

    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
            thinking_config=config.thinking_config if hasattr(config, 'thinking_config') and config.thinking_config else None,
        ),
    )

    return chat, config


def gemini_chat_send(
    chat,
    gen_config,
    turn: Dict,
    check_time: bool = True,
    turn_label: str = None,
) -> str:
    """기존 chat 세션에 턴 1개 전송.

    Args:
        chat: gemini_chat_start()에서 반환된 chat 객체
        gen_config: gemini_chat_start()에서 반환된 config (unused, 호환용)
        turn: {"text": str, "image_path": str|None, "image_paths": list|None}
        check_time: 시간 출력 여부
        turn_label: 로그에 표시할 턴 라벨

    Returns:
        LLM 응답 텍스트
    """
    start_time = time.time()

    contents = _build_contents(turn)
    resp = _send_with_retry(chat, contents, None)

    elapsed = time.time() - start_time

    # usage 추출
    usage_dict = {"inference_time": elapsed, "in": 0, "out": 0, "think": 0, "total": 0}
    try:
        usage = resp.usage_metadata
        if usage:
            usage_dict["in"] = getattr(usage, 'prompt_token_count', 0) or 0
            usage_dict["out"] = getattr(usage, 'candidates_token_count', 0) or 0
            usage_dict["think"] = getattr(usage, 'thoughts_token_count', 0) or getattr(usage, 'thinking_token_count', 0) or 0
            usage_dict["total"] = getattr(usage, 'total_token_count', 0) or 0
    except Exception:
        pass

    if check_time:
        n_images = (1 if turn.get("image_path") else 0) + len(turn.get("image_paths", []))
        img_str = f" + {n_images} image(s)" if n_images > 0 else ""
        label = f" [{turn_label}]" if turn_label else ""
        parts = [f"{k}={v}" for k, v in usage_dict.items() if k != "inference_time" and v > 0]
        token_str = f" ({', '.join(parts)})" if parts else ""
        print(f"[GEMINI/Chat]{label}{img_str}: {elapsed:.2f}s{token_str}")

    gemini_chat_send._last_usage = usage_dict
    return resp.text

# 초기화
gemini_chat_send._last_usage = None


# ============================================================
# 단일 호출 API (하위호환)
# ============================================================

def gemini_response(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    timeout: float = 120.0,
    return_usage: bool = False,
):
    client = _get_client()

    start_time = time.time()

    # contents 구성
    contents = [prompt]
    if image_path:
        contents.append(_load_image(image_path))

    # config 구성
    config_kwargs = {"temperature": temperature}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if stop_sequences:
        filtered = [s for s in stop_sequences if s]
        if filtered:
            config_kwargs["stop_sequences"] = filtered
    if "gemini-3" in model.lower():
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=GEMINI3_DEFAULT_THINKING_BUDGET
        )

    config = types.GenerateContentConfig(**config_kwargs)

    # Retry with backoff
    from google.genai.errors import ClientError, ServerError
    import concurrent.futures

    MAX_RETRIES_LOCAL = 5
    for attempt in range(MAX_RETRIES_LOCAL):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            try:
                response = future.result(timeout=timeout)
                break
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f"[GEMINI] {model} did not respond within {timeout}s (elapsed: {elapsed:.1f}s)"
                )
            except (ClientError, ServerError) as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 2 ** attempt * 5
                    print(f"[GEMINI] Rate limit (429). Retrying in {wait_time}s... (attempt {attempt+1}/{MAX_RETRIES_LOCAL})")
                    time.sleep(wait_time)
                    if attempt == MAX_RETRIES_LOCAL - 1:
                        raise
                else:
                    raise

    elapsed = time.time() - start_time
    if check_time:
        has_image = " + image" if image_path else ""
        has_system = " + system_prompt" if system_prompt else ""
        print(f"[GEMINI/AI Studio] Model: {model}{has_system}{has_image}, Response time: {elapsed:.2f}s")

    if return_usage:
        usage = {}
        meta = getattr(response, 'usage_metadata', None)
        if meta:
            usage = {
                "input_tokens": getattr(meta, 'prompt_token_count', 0) or 0,
                "output_tokens": getattr(meta, 'candidates_token_count', 0) or 0,
                "total_tokens": getattr(meta, 'total_token_count', 0) or 0,
                "inference_time_s": round(elapsed, 2),
            }
        return response.text, usage

    return response.text


def gemini_chat(
    model: str,
    system_prompt: str,
    turns: List[Dict],
    temperature: float = 0.0,
    check_time: bool = True,
) -> List[str]:
    """멀티턴 Gemini chat session (하위호환)"""
    start_time = time.time()

    chat, config = gemini_chat_start(model, system_prompt=system_prompt, temperature=temperature)

    responses = []
    for i, turn in enumerate(turns):
        resp_text = gemini_chat_send(chat, config, turn, check_time=check_time,
                                      turn_label=f"Turn {i+1}/{len(turns)}")
        responses.append(resp_text)

    if check_time:
        total = time.time() - start_time
        print(f"[GEMINI/Chat] Total ({len(turns)} turns): {total:.2f}s")

    return responses
