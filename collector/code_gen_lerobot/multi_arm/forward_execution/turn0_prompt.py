"""
Multi-Arm Turn 0: Scene Understanding Prompt

Multi-arm specific — describes both arms' positions and analyzes
which objects are in which arm's reach.
"""


def multi_arm_turn0_scene_understanding_prompt(
    instruction: str,
    has_cad: bool = False,
) -> str:
    """Turn 0: Multi-arm scene understanding prompt.

    Unlike single-arm Turn 0, this version asks the VLM to identify
    which arm should handle which objects based on spatial layout.

    Args:
        instruction: Natural language task description.
        has_cad: Whether CAD reference images are attached.
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

This is a **bi-arm** robot setup:
- **Left arm** is visible at the middle-left edge of the image.
- **Right arm** is visible at the middle-right edge of the image.
- The center of the table is reachable by both arms (overlap zone).

Task: {instruction}

Analyze the scene and describe:
1. **Object identification**: What task-relevant objects are visible? Describe each object's color, shape, and approximate size.
2. **Spatial layout**: Where is each object located on the table? (left side, right side, center)
3. **Arm assignment**: For each object, which arm can reach it?
   - Objects on the left half → left arm
   - Objects on the right half → right arm
   - Objects in the center → both arms (note this)
4. **Object relationships**: How are the objects spatially related to each other?
5. **Collision awareness**: Are any objects close to the other arm's position? Note potential collision risks.

**Important**:
- This is a scene understanding step ONLY.
- Do NOT generate any code, numeric coordinates, or step-by-step execution plans.
- Focus purely on visual observation, spatial description, and arm assignment."""
