"""
Verification Prompt Template

생성된 로봇 코드가 태스크 지시사항과 물체 위치 정보에 맞게
올바르게 생성되었는지 LLM에게 검증을 요청하는 프롬프트.
"""

VERIFICATION_SYSTEM_PROMPT = """You are a lenient robot code reviewer.
Your job is to catch only CRITICAL, show-stopping bugs in generated robot arm code.

Be very permissive. PASS the code unless there is an obvious, fatal error that would guaranteed cause the task to fail or damage the robot.

You must output EXACTLY one of two responses:
- PASS
- FAIL: <brief reason>

Do NOT output anything else."""


def build_verification_prompt(
    instruction: str,
    generated_code: str,
    object_positions: dict,
    skill_api_summary: str = "",
) -> str:
    """Verification 프롬프트 생성.

    Args:
        instruction: 태스크 자연어 지시사항
        generated_code: 검증할 생성된 Python 코드
        object_positions: 검출된 물체 위치 정보 dict
        skill_api_summary: 사용 가능한 스킬 API 요약 (선택)

    Returns:
        LLM에 보낼 프롬프트 문자열
    """
    # 물체 위치 정보를 읽기 쉬운 포맷으로 변환
    positions_text = _format_positions(object_positions)

    prompt = f"""## Task Instruction
{instruction}

## Detected Object Positions
{positions_text}

## Generated Code
```python
{generated_code}
```

## Rules — ONLY fail for these critical errors:
1. Completely wrong object targeted (e.g., instruction says "red block" but code picks "blue block")
2. Fatal operation order (e.g., place called before any pick)
3. Code references objects/keys that do not exist in the positions at all
4. Code would obviously crash (syntax error, undefined variable used, etc.)

## What to IGNORE (these are NOT reasons to fail):
- Minor style or structure differences
- Missing connect/disconnect or init/free calls (handled by the framework)
- Suboptimal approach heights, durations, or parameters
- Extra or redundant movements
- Coordinate offsets or small numerical adjustments
- Any concern about "could be better" — if it would roughly work, PASS it

When in doubt, PASS.

Respond with EXACTLY one line: PASS or FAIL: <reason>"""

    return prompt


def _format_positions(positions: dict) -> str:
    """물체 위치 dict를 사람이 읽기 쉬운 텍스트로 변환.

    싱글암: {"object_name": {"position": [x,y,z], ...}}
    멀티암: {"left_arm": {"object_name": {...}}, "right_arm": {"object_name": {...}}}
    """
    if not positions:
        return "(no positions detected)"

    # 멀티암 포맷 감지: top-level key가 arm 이름인 경우
    arm_keys = {"left_arm", "right_arm", "top_arm", "bottom_arm"}
    if arm_keys & set(positions.keys()):
        lines = []
        for arm_name, arm_positions in positions.items():
            if arm_name not in arm_keys or not isinstance(arm_positions, dict):
                continue
            lines.append(f"[{arm_name}]")
            lines.append(_format_single_arm_positions(arm_positions))
        return "\n".join(lines)

    return _format_single_arm_positions(positions)


def _format_single_arm_positions(positions: dict) -> str:
    """단일 arm의 물체 위치를 포맷."""
    lines = []
    for name, info in positions.items():
        if info is None:
            lines.append(f"- {name}: (not detected)")
            continue

        if isinstance(info, dict):
            pos = info.get("position")
            if pos:
                line = f"- {name}: position=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                if "points" in info and info["points"]:
                    point_names = list(info["points"].keys())
                    line += f", interaction_points={point_names}"
                lines.append(line)
            else:
                lines.append(f"- {name}: {info}")
        else:
            lines.append(f"- {name}: {info}")

    return "\n".join(lines)
