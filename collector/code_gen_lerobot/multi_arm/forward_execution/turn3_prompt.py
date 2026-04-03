"""
Multi-Arm Forward Execution User Prompts

Turn 3 code generation prompt for bi-arm setup.
Turn 0~2 are reused from single-arm forward_execution.

Pattern: Pipeline pre-creates `skills = MultiArmSkills(...)` and injects via exec_globals.
LLM code calls `skills.connect()` / `skills.disconnect()` and uses `skills.move_to_position(...)`, `skills.pick_object(...)`, etc.
"""

from typing import Dict, List, Optional

from ..multi_skill_api_doc import MULTI_ARM_API_DOC


def multi_arm_turn3_codegen_prompt(
    instruction: str,
    robot_ids: List[int] = None,
    all_points: list = None,
    context_summary: str = "",
    positions: dict = None,
) -> str:
    """
    Turn 3: Multi-arm code generation prompt.

    Args:
        instruction: Natural language task description.
        robot_ids: Robot IDs [left_id, right_id] (e.g., [2, 3]).
        all_points: Turn 2 detected critical points.
        context_summary: Scene context summary from Turn 0~2.
        positions: Detected object positions (single-arm format, for key listing).

    Returns:
        Turn 3 user prompt string.
    """
    if robot_ids is None:
        robot_ids = [2, 3]

    # Format detected points
    points_desc = ""
    if all_points:
        from collections import defaultdict
        by_object = defaultdict(list)
        for pt in all_points:
            by_object[pt["object_label"]].append(pt)

        lines = []
        for obj, pts in by_object.items():
            lines.append(f"  {obj}:")
            for pt in pts:
                label = pt.get("label", "unknown")
                role = pt.get("role", "unknown")
                reasoning = pt.get("reasoning", "")
                lines.append(f'    - "{label}" [{role}]: {reasoning}')
        points_desc = "\n".join(lines)

    points_section = ""
    if points_desc:
        points_section = f"""
**Detected Critical Points** (from scene analysis):
Choose the most appropriate point for the task.
Access via `pos_left["object"]["points"]["label"]` or `pos_right["object"]["points"]["label"]`.
{points_desc}
"""

    # positions key listing (object names only)
    positions_keys = ""
    if positions:
        key_lines = []
        for name, info in positions.items():
            if isinstance(info, dict) and "position" in info:
                pts = info.get("points", {})
                pt_labels = ", ".join(f'"{k}"' for k in pts.keys()) if pts else ""
                key_lines.append(f'  - "{name}": position, points: [{pt_labels}]')
        positions_keys = "\n".join(key_lines)

    context_section = ""
    if context_summary:
        context_section = f"""### Scene Context (from prior analysis session)

{context_summary}

"""

    prompt = f"""{context_section}### Your Job (Turn 3 — Multi-Arm Code Generation)

Generate executable Python code to complete the task using **two** SO-101 robot arms.
Use the scene understanding, detected objects, and grasp/place points from our previous conversation turns.

**Task**: "{instruction}"

**Robot Setup**:
- Two SO-101 robot arms: left arm (robot{robot_ids[0]}) handles the left side and center, right arm (robot{robot_ids[1]}) handles the right side and center.
- A pre-created `skills` object (MultiArmSkills) is provided as a global variable.
- Call `skills.connect()` at the start and `skills.disconnect()` in a finally block.

**Workspace Images** (provided as attached images):
- **Image 1**: left_arm (robot{robot_ids[0]}) workspace. **Image 2**: right_arm (robot{robot_ids[1]}) workspace.
- Each image shows a **green rectangle** (table boundary) and a **cyan arc** (that arm's reachable range).
- **BRIGHT area** (inside the cyan arc) = reachable zone. **DARK area** (outside the cyan arc) = physically unreachable.
- CRITICAL: The robot CANNOT move to, pick from, or place at ANY position in the darkened area.
  Even if the task says "edge" or "corner", you MUST choose a pixel coordinate that falls inside the bright area (cyan arc).
  If the literal target (e.g. "bottom-right edge") is in the dark area, pick the closest reachable point inside the cyan arc instead.
- Assign each object to the arm whose bright area covers that object.
- If an object is reachable by both arms, prefer the arm closer to it.
- Reach range: [0.22, 0.41]m from each arm's base.

{points_section}
**The `positions` dictionary** will be provided at runtime with this structure:
```python
positions = {{
    "left_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in left arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}
        }},
        ...
    }},
    "right_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in right arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}
        }},
        ...
    }},
}}
```

**How to use positions**:
```python
# First, split positions by arm at the top of execute_task()
pos_left = positions["left_arm"]
pos_right = positions["right_arm"]

# Then access object positions for each arm
left_pick = pos_left["object_name"]["position"]    # use with skills.left_arm.*
right_pick = pos_right["object_name"]["position"]  # use with skills.right_arm.*
```

Available object keys:
{positions_keys}

- **CRITICAL**: Use ONLY exact key names from the `positions` dictionary.
- **CRITICAL**: Do NOT redefine or hardcode the `positions` dictionary in your code.
- **CRITICAL**: Do NOT hardcode any coordinate values.

{MULTI_ARM_API_DOC}

**Skill Composition Patterns**:

```python
# BOTH ARMS (parallel operation)
pos_left = positions["left_arm"]
pos_right = positions["right_arm"]
approach_height = 0.20

# Open both grippers
skills.gripper_control(left_arm="open", right_arm="open",
    left_skill_description="Open left gripper", left_verification_question="Is left gripper open?",
    right_skill_description="Open right gripper", right_verification_question="Is right gripper open?")

# Move both arms to approach positions
left_pick = pos_left["left_object"]["position"]
right_pick = pos_right["right_object"]["position"]
skills.move_to_position(
    left_arm=[left_pick[0], left_pick[1], approach_height],
    right_arm=[right_pick[0], right_pick[1], approach_height],
    left_skill_description="Move left arm above left_object",
    left_verification_question="Is left gripper above left_object?",
    right_skill_description="Move right arm above right_object",
    right_verification_question="Is right gripper above right_object?")

# Pick both (descend + grip)
skills.pick_object(left_arm=left_pick, right_arm=right_pick,
    left_object_name="left_object", right_object_name="right_object",
    left_skill_description="Pick left_object", left_verification_question="Is left_object grasped?",
    right_skill_description="Pick right_object", right_verification_question="Is right_object grasped?")

# Retract both (lift back to approach height — MANDATORY after pick)
skills.move_to_position(
    left_arm=[left_pick[0], left_pick[1], approach_height],
    right_arm=[right_pick[0], right_pick[1], approach_height],
    left_skill_description="Retract left arm after pick",
    left_verification_question="Is left arm lifted to approach height?",
    right_skill_description="Retract right arm after pick",
    right_verification_question="Is right arm lifted to approach height?")

# ONE ARM ONLY (other arm holds position with "wait")
# Pattern: open → approach → pick → retract → move → place → retract
skills.move_to_position(
    left_arm="wait",
    right_arm=[right_pick[0], right_pick[1], approach_height],
    right_skill_description="Move right arm above object",
    right_verification_question="Is right arm above object?")

skills.pick_object(left_arm="wait", right_arm=right_pick,
    right_object_name="right_object",
    right_skill_description="Pick right_object", right_verification_question="Is right_object grasped?")

# Retract (MANDATORY after pick)
skills.move_to_position(
    left_arm="wait",
    right_arm=[right_pick[0], right_pick[1], approach_height],
    right_skill_description="Retract right arm after pick",
    right_verification_question="Is right arm lifted to approach height?")

# PIXEL-BASED PLACEMENT (for locations NOT in the positions dict, e.g., empty spot on table)
# Specify [y, x] in normalized 0–1000 coordinates from the top-view image.
#
# ⚠ MANDATORY — Pixel Coordinate Reachability Check:
#   (a) Identify which arm will execute this action (left_arm or right_arm).
#   (b) Look at THAT arm's workspace image (Image 1 = left arm, Image 2 = right arm).
#   (c) Find the [y, x] point on the image. Is it inside the BRIGHT area (cyan arc)?
#   (d) If YES → use it. If NO → shift to the nearest point inside the bright area.
#   (e) Add a comment explaining the coordinate choice.
#   NEVER guess extreme values like [900, 900] — always verify on the workspace image.
target_pixel = [y, x]  # ← verified on Image 1: inside left arm's bright area
skills.move_to_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_skill_description="Move left arm above target location",
    left_verification_question="Is left arm above target location?")

skills.place_at_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_is_table=True,
    left_skill_description="Place object at target location",
    left_verification_question="Is object placed at target location?")

skills.move_to_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_skill_description="Retract from target location",
    left_verification_question="Is left arm clear of target location?")

# SUBTASK + RE-DETECTION PATTERN (MANDATORY for multi-object tasks)
# Each pick-place of one object = one subtask.
# After each subtask, re-detect all objects to update positions.

# Subtask 1: pick → retract → move → place → retract
skills.set_subtask("pick A with left arm and place at center of workspace")
skills.gripper_control(left_arm="open", right_arm="wait", ...)
skills.move_to_position(left_arm=[...approach above object...], right_arm="wait", ...)  # approach
skills.pick_object(left_arm=pos_left["A"]["position"], right_arm="wait", ...)           # descend + grip
skills.move_to_position(left_arm=[...approach above object...], right_arm="wait", ...)  # retract (MANDATORY)
skills.move_to_pixel(left_arm=target_pixel, right_arm="wait", ...)                      # move to target
skills.place_at_pixel(left_arm=target_pixel, right_arm="wait", ...)                     # descend + release
skills.move_to_pixel(left_arm=target_pixel, right_arm="wait", ...)                      # retract
skills.clear_subtask()

# Re-detection (MANDATORY between subtasks)
# detect_objects may return None for an object if detection fails — always check before using.
skills.move_to_initial_state()  # clear arms from camera view
updated = skills.detect_objects(["A", "B"])
# Update both arm position dicts (skip if detection failed)
if "left_arm" in updated:
    pos_left.update(updated["left_arm"])
if "right_arm" in updated:
    pos_right.update(updated["right_arm"])
# IMPORTANT: Re-assign ALL local variables from the updated dicts
# (pos_left.update() replaces dict entries, but previously extracted variables still reference OLD values)
b_grasp = pos_right["B"]["points"]["grasp center"]  # must re-extract after update

# Subtask 2
skills.set_subtask("pick B with right arm and place on top of A")
# ... pick and place B using updated b_grasp ...
skills.clear_subtask()

# ═══════════════════════════════════════════════════════════
# BIMANUAL PATTERNS (both arms hold the SAME object)
# Use bimanual_* skills instead of pick_object/place_object/move_to_position.
# bimanual_* skills run a single synchronized control loop.
# ═══════════════════════════════════════════════════════════

# FOLD pattern (arc trajectory — for folding towel, cloth, paper)
# bimanual_pick/place only do descend+grip / descend+release (like execute_pick/place_object).
# open, approach, lift, retract are YOUR responsibility (same as independent pick/place pattern).
skills.set_subtask("fold towel — pick top edge, fold to bottom edge")

# 1. Open + approach (before contact — independent is fine)
skills.gripper_control(left_arm="open", right_arm="open",
    left_skill_description="Open left gripper", left_verification_question="Is left open?",
    right_skill_description="Open right gripper", right_verification_question="Is right open?")
skills.move_to_position(
    left_arm=[top_left[0], top_left[1], approach_height],
    right_arm=[top_right[0], top_right[1], approach_height],
    left_skill_description="Approach left grasp", left_verification_question="Is left above grasp?",
    right_skill_description="Approach right grasp", right_verification_question="Is right above grasp?")

# 2. Pick (descend + grip — contact starts, synchronized)
skills.bimanual_pick_object(left_arm=top_left, right_arm=top_right, object_name="towel")

# 3. Fold arc (contact, synchronized — arc automatically lifts and descends)
# Do NOT add a separate bimanual_move(lift) before fold — the arc handles it.
skills.bimanual_fold(
    left_start=top_left, right_start=top_right,
    left_end=bottom_left, right_end=bottom_right, arc_height=0.20,
    left_skill_description="Fold left to bottom", left_verification_question="Is left folded?",
    right_skill_description="Fold right to bottom", right_verification_question="Is right folded?")

# 5. Place (descend + release — contact ends, synchronized)
skills.bimanual_place_object(left_arm=bottom_left, right_arm=bottom_right, object_name="towel")

# 6. Retract (after contact — independent is fine)
skills.move_to_position(
    left_arm=[bottom_left[0], bottom_left[1], approach_height],
    right_arm=[bottom_right[0], bottom_right[1], approach_height],
    left_skill_description="Retract left", left_verification_question="Is left clear?",
    right_skill_description="Retract right", right_verification_question="Is right clear?")

skills.clear_subtask()

# CARRY pattern (straight-line — for carrying large/heavy objects)
skills.set_subtask("carry large box to the right side")

# Open + approach (before contact)
skills.gripper_control(left_arm="open", right_arm="open",
    left_skill_description="Open left", left_verification_question="Is left open?",
    right_skill_description="Open right", right_verification_question="Is right open?")
skills.move_to_position(
    left_arm=[box_left[0], box_left[1], approach_height],
    right_arm=[box_right[0], box_right[1], approach_height],
    left_skill_description="Approach box left", left_verification_question="Is left above box?",
    right_skill_description="Approach box right", right_verification_question="Is right above box?")

# Pick (contact starts)
skills.bimanual_pick_object(left_arm=box_left, right_arm=box_right, object_name="box")

# Lift + carry (contact, synchronized)
skills.bimanual_move(
    left_arm=[box_left[0], box_left[1], approach_height],
    right_arm=[box_right[0], box_right[1], approach_height],
    left_skill_description="Lift box", left_verification_question="Is box lifted?",
    right_skill_description="Lift box", right_verification_question="Is box lifted?")
skills.bimanual_move(
    left_arm=[target_left[0], target_left[1], approach_height],
    right_arm=[target_right[0], target_right[1], approach_height],
    left_skill_description="Carry box to target", left_verification_question="Is box above target?",
    right_skill_description="Carry box to target", right_verification_question="Is box above target?")

# Place (contact ends)
skills.bimanual_place_object(left_arm=target_left, right_arm=target_right, object_name="box")

# Retract (after contact)
skills.move_to_position(
    left_arm=[target_left[0], target_left[1], approach_height],
    right_arm=[target_right[0], target_right[1], approach_height],
    left_skill_description="Retract left", left_verification_question="Is left clear?",
    right_skill_description="Retract right", right_verification_question="Is right clear?")

skills.clear_subtask()
```

**Code Skeleton**:

```python
def execute_task():    # NO ARGUMENTS — positions/skills are pre-injected globals
    '''Execute the bi-arm robot task.'''
    skills.connect()

    try:
        pos_left = positions["left_arm"]
        pos_right = positions["right_arm"]
        approach_height = 0.20

        skills.move_to_initial_state()

        # Subtask 1: set_subtask → open → approach → pick → retract → move → place → retract → clear_subtask
        # Re-detection: move_to_initial_state → detect_objects → update positions
        # Subtask 2: set_subtask → open → approach → pick → retract → move → place → retract → clear_subtask
        # ... repeat for each object ...

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

**Task Requirements**:
1. Assign objects to the appropriate arm based on workspace images (bright area = reachable).
2. Use `skills.move_to_position()` / `skills.pick_object()` / `skills.place_object()` / `skills.gripper_control()` for ALL operations. Pass `left_arm="wait"` or `right_arm="wait"` for the arm that should hold position.
   - For locations NOT in the `positions` dict (e.g., empty spot on table), use `skills.move_to_pixel()` / `skills.place_at_pixel()` with [y, x] in normalized 0–1000 coordinates.
   - **MANDATORY (pixel reachability check)**: Before writing ANY hardcoded pixel coordinate:
     (a) Identify the executing arm. (b) Check THAT arm's workspace image (Image 1 = left, Image 2 = right).
     (c) Verify [y, x] is inside the BRIGHT area (cyan arc). (d) If in dark area → shift inward to nearest bright point.
     (e) Add a code comment with reasoning. NEVER use extreme values like [900, 900].
   - **Pick pattern**: open → approach (move to approach_height above object) → pick_object (descend + grip) → **retract (move back to approach_height — MANDATORY)** → move to next target.
   - **Place pattern**: move_to_pixel (approach above target) → place_at_pixel (descend + release) → move_to_pixel (retract).
   - NEVER skip the retract step after pick or place. Without retract, the arm drags the object across the table.
3. **Subtask pattern**: Wrap each logical unit of work with `set_subtask()` before and `clear_subtask()` after.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, ALL grippers must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. If an arm is holding an object, the subtask is not yet complete — do NOT call `clear_subtask()` until all grippers have released.
4. **Re-detection (MANDATORY)**: After each subtask (after `clear_subtask()`), call `skills.move_to_initial_state()` to clear arms from camera view, then `skills.detect_objects([...all object names...])` to update positions. Skip re-detection only after the very last subtask.
   - **CRITICAL**: After `pos_left.update()` / `pos_right.update()`, you MUST **re-assign ALL local variables** that were extracted from the positions dict (e.g., `grasp_pt = pos_left["obj"]["points"]["grasp center"]`). The `update()` call replaces dict entries, but previously extracted variables still reference the OLD values.
5. Always start with `skills.move_to_initial_state()`, end with `skills.move_to_initial_state()` then `skills.move_to_free_state()`.
6. Use `approach_height = 0.20` for approach/retreat movements.
7. ALWAYS pass `left_skill_description`/`right_skill_description` and `left_verification_question`/`right_verification_question` for every arm that is NOT `"wait"`.
8. Always include try/finally with `skills.disconnect()` for cleanup.
   - **CRITICAL**: `def execute_task()` must have **NO arguments**. `skills` and `positions` are pre-injected global variables. Do NOT pass them as function parameters.
9. **Bimanual skills** (both arms interact with the **same object**):
   - Use `bimanual_pick_object` / `bimanual_place_object` / `bimanual_move` / `bimanual_fold` instead of `pick_object` / `place_object` / `move_to_position`.
   - **Folding** (towel, cloth, paper): `bimanual_pick_object` → `bimanual_fold` → `bimanual_place_object` (NO separate lift step — the arc handles it)
   - **Carrying** (large/heavy object): `bimanual_pick_object` → `bimanual_move` → `bimanual_place_object`
   - Do NOT use independent `pick_object` / `place_object` / `move_to_position` for shared-object manipulation (no synchronization).

**Generate the complete executable Python code:**
"""
    return prompt
