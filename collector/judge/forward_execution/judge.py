"""
Task Judge Module

VLM을 활용하여 로봇 태스크 완료 여부를 판단

SSH 서버 추론 모드:
- use_server=True로 설정하면 SSH 터널을 통해 원격 서버에서 VLM 추론
- 환경변수 VLM_SERVER_ADDRESS로 서버 주소 설정 가능 (기본: localhost:50052)
"""

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .prompt import build_judge_prompt, get_system_prompt
from ..image_capture import image_to_base64

# Type alias for cleaner type hints
ObjectPositions = Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]]


# API 키 로드 (환경변수 우선 → JSON 파일 fallback)
def _load_api_keys():
    keys = {}
    # judge/forward_execution/judge.py → forward_execution/ → judge/ → root/
    key_file = Path(__file__).parent.parent.parent / "openai_api_key.json"
    if key_file.exists():
        with open(key_file, "r") as f:
            keys = json.load(f)
    if os.getenv("OPENAI_API_KEY"):
        keys["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    return keys


class TaskJudge:
    """태스크 완료 여부 판단 클래스"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        verbose: bool = True,
        use_server: bool = False,
    ):
        """
        초기화

        Args:
            model: 사용할 VLM 모델 (gpt-4o, gpt-4o-mini, gpt-4-turbo 등)
            temperature: 생성 온도 (0.0 = 결정적)
            verbose: 상세 출력 여부
            use_server: True면 SSH 서버로 VLM 추론, False면 기존 API 호출
        """
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.use_server = use_server

        # Gemini 모델은 Google AI Studio 사용 (OpenAI 클라이언트 불필요)
        self.is_gemini = "gemini" in model.lower()

        if use_server or self.is_gemini:
            self.client = None
            if verbose:
                if use_server:
                    print("[TaskJudge] Using SSH server for VLM inference")
                else:
                    print(f"[TaskJudge] Using Google AI Studio Gemini ({model})")
        else:
            # OpenAI 클라이언트 초기화
            try:
                from openai import OpenAI
                api_keys = _load_api_keys()
                self.client = OpenAI(api_key=api_keys.get("openai_api_key"))
            except Exception as e:
                print(f"[TaskJudge] Failed to initialize OpenAI client: {e}")
                self.client = None

    def judge(
        self,
        instruction: str,
        initial_image: np.ndarray,
        final_image: np.ndarray,
        object_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
        executed_code: str,
        image_resolution: Tuple[int, int] = None,
    ) -> Dict:
        """
        태스크 완료 여부 판단

        Args:
            instruction: 목표 명령어
            initial_image: 초기 상태 이미지 (BGR numpy array)
            final_image: 최종 상태 이미지 (BGR numpy array)
            object_positions: 객체 위치 딕셔너리
                Extended format: {name: {"position": [x,y,z], "pixel_coords": (cx,cy), ...}}
            executed_code: 실행된 Python 코드
            image_resolution: 이미지 해상도 (width, height), None이면 자동 추출

        Returns:
            {
                'prediction': 'TRUE' | 'FALSE' | 'UNCERTAIN',
                'reasoning': str,
                'raw_response': str,
                'success': bool,
                'error': str | None,
            }
        """
        if not self.use_server and not self.is_gemini and self.client is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'VLM client not initialized',
                'raw_response': '',
                'success': False,
                'error': 'OpenAI client not initialized',
            }

        # 이미지 해상도 추출 (제공되지 않은 경우)
        if image_resolution is None:
            # object_positions에서 메타데이터 확인
            if "_image_resolution" in object_positions:
                image_resolution = object_positions["_image_resolution"]
            else:
                # 이미지에서 추출
                image_resolution = (initial_image.shape[1], initial_image.shape[0])

        # 프롬프트 생성 (Chain-of-Thought 구조)
        user_prompt = build_judge_prompt(
            instruction=instruction,
            object_positions=object_positions,
            executed_code=executed_code,
            image_resolution=image_resolution,
        )
        system_prompt = get_system_prompt()

        # 이미지를 base64로 변환
        initial_b64 = image_to_base64(initial_image)
        final_b64 = image_to_base64(final_image)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Task Judge - Evaluating Task Completion (Chain-of-Thought)")
            print("=" * 60)
            print(f"  Instruction: {instruction}")
            print(f"  Model: {self.model}")
            print(f"  Image resolution: {image_resolution[0]} x {image_resolution[1]} pixels")
            print(f"  Objects with pixel coords: {sum(1 for k, v in object_positions.items() if isinstance(v, dict) and v.get('pixel_coords'))}")

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
            }

    def _call_vlm(
        self,
        system_prompt: str,
        user_prompt: str,
        initial_image_b64: str,
        final_image_b64: str,
    ) -> str:
        """
        VLM API 호출

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            initial_image_b64: 초기 이미지 base64
            final_image_b64: 최종 이미지 base64

        Returns:
            VLM 응답 문자열
        """
        # vLLM 서버 또는 Gemini 사용 (vlm.py 모듈 활용)
        if self.use_server or self.is_gemini:
            from ..vlm import vlm_response

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = vlm_response(
                prompt=full_prompt,
                images_b64=[initial_image_b64, final_image_b64],
                model=self.model,
                max_tokens=10000,
                temperature=self.temperature,
                use_server=True if self.use_server else None,
            )

            if response is None:
                raise RuntimeError("VLM request failed")

            return response

        # 기존 OpenAI API 호출
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
        """
        VLM 응답 파싱

        Args:
            response: VLM 응답 문자열

        Returns:
            {'prediction': str, 'reasoning': str}
        """
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

        # reasoning이 비어있으면 전체 응답에서 PREDICTION 이후 부분 사용
        if not reasoning:
            # PREDICTION 라인 이후의 모든 텍스트
            parts = re.split(r"PREDICTION:\s*\w+", response, flags=re.IGNORECASE)
            if len(parts) > 1:
                reasoning = parts[1].strip()
            else:
                reasoning = response

        return {
            'prediction': prediction,
            'reasoning': reasoning,
        }


def judge_task(
    instruction: str,
    initial_image: np.ndarray,
    final_image: np.ndarray,
    object_positions: Dict,
    executed_code: str,
    model: str = "gpt-4o",
    use_server: bool = False,
    image_resolution: Tuple[int, int] = None,
) -> Dict:
    """
    태스크 판단 편의 함수

    Args:
        instruction: 목표 명령어
        initial_image: 초기 상태 이미지
        final_image: 최종 상태 이미지
        object_positions: 객체 위치 딕셔너리 (픽셀 좌표 포함 가능)
        executed_code: 실행된 Python 코드
        model: VLM 모델
        use_server: True면 SSH 서버로 VLM 추론
        image_resolution: 이미지 해상도 (width, height)

    Returns:
        판단 결과 딕셔너리
    """
    judge = TaskJudge(model=model, use_server=use_server)
    return judge.judge(
        instruction=instruction,
        initial_image=initial_image,
        final_image=final_image,
        object_positions=object_positions,
        executed_code=executed_code,
        image_resolution=image_resolution,
    )


# 테스트
if __name__ == "__main__":
    import cv2

    print("TaskJudge Test")
    print("=" * 60)

    # 테스트용 더미 이미지 생성
    dummy_initial = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_initial, "Initial State", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    dummy_final = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_final, "Final State", (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 테스트 데이터
    test_instruction = "pick up the green block and place it on the blue dish"
    test_positions = {
        "green block": [0.1500, 0.0500, 0.0200],
        "blue dish": [0.2000, -0.0300, 0.0100],
    }
    test_code = """
from skills.skills_lerobot import LeRobotSkills
skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
skills.connect()
skills.execute_pick(position=[0.15, 0.05, 0.02])
skills.execute_place(position=[0.20, -0.03, 0.03])
skills.disconnect()
"""

    # Judge 테스트
    judge = TaskJudge(verbose=True)
    result = judge.judge(
        instruction=test_instruction,
        initial_image=dummy_initial,
        final_image=dummy_final,
        object_positions=test_positions,
        executed_code=test_code,
    )

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(f"  Prediction: {result['prediction']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Success: {result['success']}")
    if result['error']:
        print(f"  Error: {result['error']}")
