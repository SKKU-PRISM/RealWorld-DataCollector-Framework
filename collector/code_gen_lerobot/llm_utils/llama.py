import requests
import time


def llama_response(
    prompt: str,
    model: str = "meta-llama/Llama-3.2-3B",
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
) -> str:
    start_time = time.time()
    stop_sequences = stop_sequences or ['']

    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": temperature,
        "stop": stop_sequences
    }

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0].get("text", "")
            else:
                output = result
        else:
            print("요청 실패: 상태 코드", response.status_code)
            print("응답 메시지:", response.text)
            output = None
    except Exception as e:
        print("예외 발생:", str(e))
        output = None

    if check_time:
        elapsed = time.time() - start_time
        print(f"[LLAMA] Model: {model}, Response time: {elapsed:.2f}s")

    return output


if __name__ == "__main__":
    test_prompt = "Once upon a time,"
    result = llama_response(test_prompt)
    if result:
        print("생성된 텍스트 응답:")
        print(result)
