

def turn0_scene_understanding_prompt(instruction: str, has_cad: bool = False) -> str:
    """Turn 0: Scene understanding prompt.

    Args:
        instruction: natural language task description.
        has_cad: CAD reference multi-view images are available or not.
    """
    if has_cad:
        image_desc = """\
The attached images include:
1. An **overhead camera image** of the workspace (first image).
2. **CAD reference images** of the task-relevant parts from multiple angles."""
    else:
        image_desc = """\
The attached image is an **overhead camera image** of the workspace."""

    return f"""\
{image_desc}

Task: {instruction}

Analyze the scene and describe:
1. **Object identification**: What task-relevant objects are visible? Describe each object's color, shape, and approximate size.
2. **Spatial layout**: Where is each object located on the table? (e.g., center, left side, near the edge)
3. **Object relationships**: How are the objects spatially related to each other?

**Important**:
- This is a scene understanding step ONLY.
- Do NOT generate any code, numeric coordinates, or step-by-step execution plans.
- Focus purely on visual observation and spatial description in natural language."""


# ──────────────────────────────────────────────
# [ARCHIVED] 기존 turn0 prompt
# "how each should be grasped, and how they should be manipulated"가
# system prompt의 절차 지시와 결합하여 Turn 0에서 코드까지 생성하는 문제가 있었음.
# ──────────────────────────────────────────────
# def turn0_scene_understanding_prompt(instruction: str, has_cad: bool = False) -> str:
#     if has_cad:
#         return f"""\
# The attached images include:
# 1. An **overhead camera image** of the workspace (first image).
# 2. **CAD reference images** of the task-relevant parts from multiple angles.
#
# {instruction}
# Tell me in detail which objects are relevant, how each should be grasped, and how they should be manipulated to complete the task."""
#     else:
#         return f"""\
# The attached image is an **overhead camera image** of the workspace.
#
# {instruction}
# Tell me in detail which objects are relevant, how each should be grasped, and how they should be manipulated to complete the task."""
