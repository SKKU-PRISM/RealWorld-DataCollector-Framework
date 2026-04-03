"""
Post-processing: Skill-level natural language label generation

실행 완료 후 스킬 시퀀스를 LLM에 전달하여
각 스킬에 대한 의미적 자연어 라벨을 생성합니다.
"""

import json
import os
from typing import Dict, List, Optional


def _build_labeling_prompt(instruction: str, skill_sequence: List[Dict]) -> str:
    """LLM에 전달할 스킬 라벨링 프롬프트 생성"""

    # 스킬 시퀀스를 텍스트로 변환
    seq_lines = []
    for i, skill in enumerate(skill_sequence):
        parts = [f"[{i}] type={skill['type']}"]
        if skill.get("target_name"):
            parts.append(f"target={skill['target_name']}")
        if skill.get("position"):
            pos = skill["position"]
            parts.append(f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        seq_lines.append(", ".join(parts))

    seq_text = "\n".join(seq_lines)

    prompt = f"""Given a robot manipulation task and the sequence of skills executed, generate a natural language description for each skill step.

**Task instruction**: "{instruction}"

**Executed skill sequence**:
{seq_text}

For each skill step, write a concise natural language sentence (5-15 words) that describes:
- WHY the robot is performing this action in context of the overall task
- Include the object name and spatial context (e.g., "above", "onto")

Output as JSON array with the same length as the skill sequence:
[
  "description for step 0",
  "description for step 1",
  ...
]

Output ONLY the JSON array, no other text."""

    return prompt


def generate_skill_labels(
    instruction: str,
    skill_sequence: List[Dict],
    llm_model: str = "gemini-2.0-flash",
    save_path: Optional[str] = None,
) -> List[str]:
    """실행된 스킬 시퀀스에 대해 자연어 라벨을 생성합니다.

    Args:
        instruction: 원래 태스크 지시문
        skill_sequence: LeRobotSkills.skill_sequence (스킬 실행 기록)
        llm_model: 라벨 생성에 사용할 LLM 모델
        save_path: 결과 저장 경로 (JSON)

    Returns:
        각 스킬에 대한 자연어 라벨 리스트
    """
    if not skill_sequence:
        print("[SkillLabeler] No skill sequence to label")
        return []

    print(f"\n[SkillLabeler] Generating labels for {len(skill_sequence)} skills...")

    prompt = _build_labeling_prompt(instruction, skill_sequence)

    try:
        from code_gen_lerobot.llm_utils.gemini import gemini_response
        response = gemini_response(
            prompt=prompt,
            model=llm_model,
            temperature=0.0,
            check_time=True,
        )

        # JSON 파싱
        # ```json ... ``` 블록 제거
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # 첫 줄 제거
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        labels = json.loads(text)

        if len(labels) != len(skill_sequence):
            print(f"[SkillLabeler] Warning: label count ({len(labels)}) != skill count ({len(skill_sequence)})")
            # 부족하면 auto label로 채우고, 넘치면 잘라냄
            while len(labels) < len(skill_sequence):
                labels.append(skill_sequence[len(labels)].get("label", ""))
            labels = labels[:len(skill_sequence)]

        # 결과 출력
        for i, (skill, label) in enumerate(zip(skill_sequence, labels)):
            print(f"  [{i}] {skill['type']}: {label}")

        # 저장
        if save_path:
            result = {
                "instruction": instruction,
                "labels": labels,
                "skill_sequence": skill_sequence,
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[SkillLabeler] Saved to {save_path}")

        return labels

    except Exception as e:
        print(f"[SkillLabeler] Error: {e}")
        # 폴백: 기존 auto label 사용
        return [s.get("label", "") for s in skill_sequence]
