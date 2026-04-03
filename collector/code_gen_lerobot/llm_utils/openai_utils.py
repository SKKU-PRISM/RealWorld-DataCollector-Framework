import json
import os
import time
from pathlib import Path
from openai import OpenAI


# API 키 로드 (환경변수 우선 → JSON 파일 fallback)
def load_api_keys():
    keys = {}
    key_file = Path(__file__).parent.parent.parent / "openai_api_key.json"
    if key_file.exists():
        with open(key_file, "r") as f:
            keys = json.load(f)
    # 환경변수가 있으면 파일보다 우선
    if os.getenv("OPENAI_API_KEY"):
        keys["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("DEEPSEEK_API_KEY"):
        keys["deepseek_api_key"] = os.getenv("DEEPSEEK_API_KEY")
    return keys

_api_keys = load_api_keys()
openai_client = OpenAI(api_key=_api_keys.get("openai_api_key"))
deepseek_client = OpenAI(
    api_key=_api_keys.get("deepseek_api_key"),
    base_url="https://api.deepseek.com/v1"
)

def chatgpt_response(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    stop_sequences: list[str] = None,
    check_time: bool = True,
    source: str = "openai"
) -> str | None:
    client = openai_client if source == "openai" else deepseek_client
    stop_sequences = stop_sequences or ['']

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if check_time:
            elapsed = time.time() - start_time
            print(f"[{source.upper()}] Model: {model}, Response time: {elapsed:.2f}s")
        return response.choices[0].message.content

    except Exception as e:
        print(f"{source.upper()} API Error:", str(e))
        return None
