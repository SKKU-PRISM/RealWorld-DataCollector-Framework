import json
import time
from pathlib import Path
from openai import OpenAI


# JSON 파일에서 API 키 읽기
def load_api_keys():
    key_file = Path(__file__).parent.parent.parent / "openai_api_key.json"
    if key_file.exists():
        with open(key_file, "r") as f:
            return json.load(f)
    return {}

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
