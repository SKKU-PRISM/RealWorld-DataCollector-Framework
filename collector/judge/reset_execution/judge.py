"""
Reset Task Judge Module

VLM을 활용하여 Reset 태스크 완료 여부를 판단
- Original Mode: 물체가 원래 위치로 복원되었는지 판단
- Random Mode: 물체가 랜덤 목표 위치로 이동되었는지 판단
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .prompt import build_reset_judge_prompt, get_reset_system_prompt
from ..image_capture import image_to_base64


# API 키 로드
def _load_api_keys():
    # judge/reset_execution/judge.py → reset_execution/ → judge/ → root/
    key_file = Path(__file__).parent.parent.parent / "openai_api_key.json"
    if key_file.exists():
        with open(key_file, "r") as f:
            return json.load(f)
    return {}


class ResetJudge:
    """Reset 태스크 완료 여부 판단 클래스"""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        verbose: bool = True,
        use_server: bool = False,
    ):
        """
        초기화

        Args:
            model: 사용할 VLM 모델 (gpt-4o, gpt-4o-mini 등)
            temperature: 생성 온도 (0.0 = 결정적)
            verbose: 상세 출력 여부
            use_server: True면 SSH 서버로 VLM 추론
        """
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.use_server = use_server

        self.is_gemini = "gemini" in model.lower()

        if use_server or self.is_gemini:
            self.client = None
            if verbose:
                if use_server:
                    print("[ResetJudge] Using SSH server for VLM inference")
                else:
                    print(f"[ResetJudge] Using Google AI Studio Gemini ({model})")
        else:
            try:
                from openai import OpenAI
                api_keys = _load_api_keys()
                self.client = OpenAI(api_key=api_keys.get("openai_api_key"))
            except Exception as e:
                print(f"[ResetJudge] Failed to initialize OpenAI client: {e}")
                self.client = None

    def judge(
        self,
        reset_mode: str,
        current_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
        target_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
        initial_image: np.ndarray,
        final_image: np.ndarray,
        executed_code: str,
        original_instruction: str = None,
        image_resolution: Tuple[int, int] = None,
    ) -> Dict:
        """
        Reset 태스크 완료 여부 판단 (Chain-of-Thought)

        Args:
            reset_mode: "original" 또는 "random"
            current_positions: Reset 전 물체 위치 (Forward 완료 후)
                Extended format: {name: {"position": [x,y,z], "pixel_coords": (cx,cy), ...}}
            target_positions: 목표 위치 (원래 위치 또는 랜덤 위치)
            initial_image: Reset 실행 전 이미지 (BGR numpy array)
            final_image: Reset 실행 후 이미지 (BGR numpy array)
            executed_code: 실행된 Reset 코드
            original_instruction: 원래 태스크 명령어 (original 모드에서만 사용)
            image_resolution: 이미지 해상도 (width, height), None이면 자동 추출

        Returns:
            {
                'prediction': 'TRUE' | 'FALSE' | 'UNCERTAIN',
                'reasoning': str,
                'raw_response': str,
                'success': bool,
                'error': str | None,
                'reset_mode': str,
            }
        """
        if not self.use_server and not self.is_gemini and self.client is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'VLM client not initialized',
                'raw_response': '',
                'success': False,
                'error': 'OpenAI client not initialized',
                'reset_mode': reset_mode,
            }

        # 이미지 해상도 추출 (제공되지 않은 경우)
        if image_resolution is None:
            if "_image_resolution" in current_positions:
                image_resolution = current_positions["_image_resolution"]
            else:
                image_resolution = (initial_image.shape[1], initial_image.shape[0])

        # 프롬프트 생성 (Chain-of-Thought 구조)
        user_prompt = build_reset_judge_prompt(
            reset_mode=reset_mode,
            current_positions=current_positions,
            target_positions=target_positions,
            executed_code=executed_code,
            original_instruction=original_instruction,
            image_resolution=image_resolution,
        )
        system_prompt = get_reset_system_prompt(reset_mode)

        # 이미지를 base64로 변환
        initial_b64 = image_to_base64(initial_image)
        final_b64 = image_to_base64(final_image)

        mode_str = "RESTORE TO ORIGINAL" if reset_mode == "original" else "RANDOM SHUFFLE"

        # 메타데이터 키 제외한 객체 수 계산
        object_keys = [k for k in current_positions.keys() if not k.startswith("_")]
        pixel_coords_count = sum(
            1 for k in object_keys
            if isinstance(current_positions.get(k), dict) and current_positions[k].get("pixel_coords")
        )

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Reset Judge - Evaluating Reset ({mode_str}) [Chain-of-Thought]")
            print("=" * 60)
            print(f"  Reset Mode: {reset_mode}")
            print(f"  Model: {self.model}")
            print(f"  Image resolution: {image_resolution[0]} x {image_resolution[1]} pixels")
            print(f"  Objects: {object_keys}")
            print(f"  Objects with pixel coords: {pixel_coords_count}/{len(object_keys)}")

        # VLM API 호출
        try:
            start_time = time.time()
            response = self._call_vlm(system_prompt, user_prompt, initial_b64, final_b64)
            elapsed = time.time() - start_time

            if self.verbose:
                print(f"  Response time: {elapsed:.2f}s")

            # 응답 파싱
            result = self._parse_response(response)
            result['raw_response'] = response
            result['success'] = True
            result['error'] = None
            result['reset_mode'] = reset_mode

            if self.verbose:
                print(f"\n  Prediction: {result['prediction']}")
                print(f"  Reasoning: {result['reasoning'][:200]}...")
                print("=" * 60)

            return result

        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                print(f"  Error: {error_msg}")

            return {
                'prediction': 'UNCERTAIN',
                'reasoning': f'VLM API call failed: {error_msg}',
                'raw_response': '',
                'success': False,
                'error': error_msg,
                'reset_mode': reset_mode,
            }

    def _call_vlm(
        self,
        system_prompt: str,
        user_prompt: str,
        initial_image_b64: str,
        final_image_b64: str,
    ) -> str:
        """VLM API 호출"""
        # vLLM 서버 또는 Gemini 사용 (vlm.py 모듈 활용)
        if self.use_server or self.is_gemini:
            from ..vlm import vlm_response

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = vlm_response(
                prompt=full_prompt,
                images_b64=[initial_image_b64, final_image_b64],
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                use_server=True if self.use_server else None,
            )

            if response is None:
                raise RuntimeError("VLM request failed")

            return response

        # OpenAI API 호출
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{initial_image_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{final_image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=1000,
            reasoning_effort="minimal",  # 추론 토큰 최소화
        )

        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict:
        """VLM 응답 파싱"""
        prediction = "UNCERTAIN"
        reasoning = ""

        # PREDICTION 파싱
        pred_patterns = [
            r"PREDICTION:\s*(TRUE|FALSE|UNCERTAIN)",
            r"Prediction:\s*(TRUE|FALSE|UNCERTAIN)",
            r"\*\*PREDICTION\*\*:\s*(TRUE|FALSE|UNCERTAIN)",
            r"^(TRUE|FALSE|UNCERTAIN)\s*$",
        ]

        for pattern in pred_patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                prediction = match.group(1).upper()
                break

        # REASONING 파싱
        reason_patterns = [
            r"REASONING:\s*(.+?)(?=\n\n|\Z)",
            r"Reasoning:\s*(.+?)(?=\n\n|\Z)",
            r"\*\*REASONING\*\*:\s*(.+?)(?=\n\n|\Z)",
        ]

        for pattern in reason_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                break

        if not reasoning:
            parts = re.split(r"PREDICTION:\s*\w+", response, flags=re.IGNORECASE)
            if len(parts) > 1:
                reasoning = parts[1].strip()
            else:
                reasoning = response

        return {
            'prediction': prediction,
            'reasoning': reasoning,
        }


def judge_reset(
    reset_mode: str,
    current_positions: Dict,
    target_positions: Dict,
    initial_image: np.ndarray,
    final_image: np.ndarray,
    executed_code: str,
    original_instruction: str = None,
    model: str = "gpt-4o",
    use_server: bool = False,
    image_resolution: Tuple[int, int] = None,
) -> Dict:
    """
    Reset 판단 편의 함수 (Chain-of-Thought)

    Args:
        reset_mode: "original" 또는 "random"
        current_positions: Reset 전 물체 위치 (픽셀 좌표 포함 가능)
        target_positions: 목표 위치
        initial_image: Reset 전 이미지
        final_image: Reset 후 이미지
        executed_code: 실행된 Reset 코드
        original_instruction: 원래 태스크 명령어 (original 모드)
        model: VLM 모델
        use_server: SSH 서버 사용 여부
        image_resolution: 이미지 해상도 (width, height)

    Returns:
        판단 결과 딕셔너리
    """
    judge = ResetJudge(model=model, use_server=use_server)
    return judge.judge(
        reset_mode=reset_mode,
        current_positions=current_positions,
        target_positions=target_positions,
        initial_image=initial_image,
        final_image=final_image,
        executed_code=executed_code,
        original_instruction=original_instruction,
        image_resolution=image_resolution,
    )


# 테스트
if __name__ == "__main__":
    import cv2

    print("ResetJudge Test")
    print("=" * 60)

    # 테스트용 더미 이미지 생성
    dummy_initial = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_initial, "Before Reset", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    dummy_final = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_final, "After Reset", (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 테스트 데이터
    test_current = {
        "green block": [0.28, -0.12, 0.05],
        "blue dish": [0.28, -0.12, 0.05],
    }
    test_target = {
        "green block": [0.11, 0.10, 0.05],
        "blue dish": [0.27, -0.16, 0.05],
    }
    test_code = """
skills.move_to_position([0.28, -0.12, 0.15])
skills.gripper_close()
skills.move_to_position([0.11, 0.10, 0.05])
skills.gripper_open()
"""
    test_instruction = "pick up the green block and place it on the blue dish"

    # Judge 테스트 (Original Mode)
    print("\n[Testing Original Mode]")
    judge = ResetJudge(verbose=True)
    result = judge.judge(
        reset_mode="original",
        current_positions=test_current,
        target_positions=test_target,
        initial_image=dummy_initial,
        final_image=dummy_final,
        executed_code=test_code,
        original_instruction=test_instruction,
    )

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(f"  Prediction: {result['prediction']}")
    print(f"  Reasoning: {result['reasoning'][:200]}...")
    print(f"  Reset Mode: {result['reset_mode']}")
    print(f"  Success: {result['success']}")
    if result['error']:
        print(f"  Error: {result['error']}")
