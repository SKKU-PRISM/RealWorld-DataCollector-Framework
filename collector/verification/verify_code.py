"""
Code Verification via LLM

생성된 로봇 코드를 LLM으로 검증하여 PASS/FAIL 결과를 반환합니다.
"""

import re
from typing import Dict, Tuple

from .prompt import VERIFICATION_SYSTEM_PROMPT, build_verification_prompt


def verify_generated_code(
    instruction: str,
    generated_code: str,
    object_positions: Dict,
    llm_model: str,
) -> Tuple[bool, str]:
    """생성된 코드를 LLM으로 검증.

    Args:
        instruction: 태스크 지시사항
        generated_code: 검증할 코드
        object_positions: 물체 위치 정보
        llm_model: 검증에 사용할 LLM 모델명

    Returns:
        (passed, reason) 튜플
        - passed: True면 PASS, False면 FAIL
        - reason: FAIL 시 사유, PASS 시 빈 문자열
    """
    from code_gen_lerobot.llm import llm_response

    prompt = build_verification_prompt(
        instruction=instruction,
        generated_code=generated_code,
        object_positions=object_positions,
    )

    response = llm_response(
        model=llm_model,
        prompt=prompt,
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        check_time=True,
    )

    if response is None:
        # LLM 호출 실패 시 PASS로 처리 (실행을 막지 않음)
        return True, "LLM verification call failed, proceeding anyway"

    return _parse_verification_response(response)


def _parse_verification_response(response: str) -> Tuple[bool, str]:
    """LLM 응답을 파싱하여 (passed, reason) 반환.

    Expected formats:
        "PASS"
        "FAIL: <reason>"
    """
    response = response.strip()

    # PASS 체크
    if response.upper().startswith("PASS"):
        return True, ""

    # FAIL 체크
    fail_match = re.match(r"FAIL\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if fail_match:
        reason = fail_match.group(1).strip()
        return False, reason

    # 예상하지 못한 포맷 — FAIL 키워드가 포함되어 있으면 FAIL로 처리
    if "FAIL" in response.upper():
        return False, response[:200]

    # 그 외 — PASS로 처리 (모호한 경우 실행을 막지 않음)
    return True, f"Ambiguous response: {response[:100]}"
