"""
Multi-Arm Reset Turn 0: Scene Understanding Prompt

Compares current state (after forward task) vs initial state (before forward task)
to understand what needs to be reset. Includes bi-arm context.
"""

from typing import List, Tuple, Optional


def turn0_reset_scene_understanding_prompt(
    original_instruction: str,
    reset_mode: str,
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    original_object_labels: List[str] = None,
) -> str:
    """
    Turn 0: Reset 장면 이해 프롬프트 (VLM multi-turn 용).

    Image 1 = 현재 상태 (workspace 시각화 포함)
    Image 2 = 초기 상태 (forward 실행 전)

    Args:
        original_instruction: 원래 forward 태스크 명령
        reset_mode: "original" | "random"
        workspace_bounds: (legacy, 미사용)
        original_object_labels: Forward에서 검출된 원래 물체 라벨 리스트
    """
    workspace_desc = """\
In Image 1, the robot's reachable workspace is visually marked:
- The BRIGHT area shows the robot's reachable donut-shaped workspace (between min and max reach from the robot base)
- The DARKENED areas are UNREACHABLE by the robot (too close to robot base, or too far away)
- CYAN DOTTED ARCS show the inner (min reach) and outer (max reach) boundaries
- GREEN RECTANGLE shows the camera FOV safe margin (30px inset) — objects must stay within this rectangle
All object placements MUST be within the intersection of the bright donut area AND the green rectangle."""

    if reset_mode == "original":
        mode_desc = f"""\
**RESET MODE: ORIGINAL (restore to initial state)**
- Image 2 shows the INITIAL state (before "{original_instruction}" was executed).
- Your goal is to identify which objects were moved by the forward task, and determine how to return them to their initial positions shown in Image 2.
- Compare Image 1 (current) vs Image 2 (initial) to find which objects have changed position."""
    else:
        mode_desc = f"""\
**RESET MODE: RANDOM (shuffle to new positions)**
- Image 2 shows the INITIAL state (before "{original_instruction}" was executed).
- Your goal is to identify which objects were manipulated during the forward task.
- These objects will be moved to NEW random positions (computed programmatically).
- Compare Image 1 (current) vs Image 2 (initial) to identify which objects were manipulated."""

    # 원래 물체 라벨 안내
    if original_object_labels:
        labels_str = ", ".join(f'"{l}"' for l in original_object_labels)
        label_guidance = f"""
**IMPORTANT — Object Labels**:
During the forward task, the following objects were detected: {labels_str}.
When referring to objects in your analysis, use these EXACT labels to maintain consistency.
If the same physical object type appears multiple times (e.g., two cups), use the labels above
and map them to the objects you see in the current image."""
    else:
        label_guidance = ""

    return f"""\
You are given **two images**:
1. **Image 1 (Current state)** — the workspace AFTER the forward task "{original_instruction}" was executed.
2. **Image 2 (Initial state)** — the workspace BEFORE the forward task was executed.

This is a **bi-arm** robot setup:
- **Left arm** is visible at the middle-left edge of the image.
- **Right arm** is visible at the middle-right edge of the image.
- The center of the table is reachable by both arms (overlap zone).

{workspace_desc}

{mode_desc}
{label_guidance}

Analyze the scene and describe:
1. **Object identification**: What objects are visible in both images? Describe each object's color, shape, and approximate size.
2. **Change analysis**: Compare Image 1 vs Image 2. Which objects changed position/state? Where were they before, and where are they now?
3. **Arm assignment**: For each object that needs to be moved, which arm(s) should handle it?
   - Objects on the left half → left arm
   - Objects on the right half → right arm
   - Objects in the center or large deformable objects (towel, cloth) → **both arms**
4. **Reset plan**: Which objects need to be moved for the reset? In what order should they be moved? For deformable objects like towels, consider whether both arms are needed to unfold/restore the shape.

**Important**:
- This is a scene understanding step ONLY.
- Do NOT generate any code, numeric coordinates, or step-by-step execution plans.
- Focus purely on visual observation, spatial analysis, and arm assignment."""
