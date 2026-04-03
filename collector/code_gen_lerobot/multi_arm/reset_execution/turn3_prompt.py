"""
Multi-Arm Reset Turn 3: Code Generation Prompt

Pattern: `skills` object (MultiArmSkills) is pre-created and injected.
LLM code calls skills.connect() / skills.disconnect() and uses
skills.move_to_position(), skills.pick_object(), skills.place_object(), etc.

Reuses MULTI_ARM_API_DOC from forward_execution.
"""

from typing import Dict, List, Optional

from ..multi_skill_api_doc import MULTI_ARM_API_DOC


def _format_per_arm_positions(positions: Dict, label: str) -> str:
    """Per-arm position dict를 프롬프트용 문자열로 포맷팅.

    Args:
        positions: {"left_arm": {obj: {"position": [x,y,z]}}, "right_arm": {...}}
                   또는 flat dict {obj: {"position": [x,y,z]}}
        label: 변수명 (e.g., "current_positions", "target_positions")

    Returns:
        포맷팅된 Python dict 문자열
    """
    def _get_pos(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    def _format_obj(info):
        """Format single object entry with position + points."""
        pos = _get_pos(info)
        if pos is None:
            return None
        pts = info.get("points", {}) if isinstance(info, dict) else {}
        if pts:
            pt_strs = []
            for pt_label, pt_pos in pts.items():
                if isinstance(pt_pos, (list, tuple)) and len(pt_pos) >= 3:
                    pt_strs.append(f'"{pt_label}": [{pt_pos[0]:.4f}, {pt_pos[1]:.4f}, {pt_pos[2]:.4f}]')
            pts_str = ", ".join(pt_strs)
            return f'{{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "points": {{{pts_str}}}}}'
        else:
            return f'{{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}}'

    # per-arm 구조 감지
    if "left_arm" in positions or "right_arm" in positions:
        lines = [f"{label} = {{"]
        for arm_key in ["left_arm", "right_arm"]:
            arm_data = positions.get(arm_key, {})
            lines.append(f'    "{arm_key}": {{')
            for name, info in arm_data.items():
                obj_str = _format_obj(info)
                if obj_str is not None:
                    lines.append(f'        "{name}": {obj_str},')
            lines.append("    },")
        lines.append("}")
        return "\n".join(lines)
    else:
        # flat dict fallback
        lines = [f"{label} = {{"]
        for name, info in positions.items():
            obj_str = _format_obj(info)
            if obj_str is not None:
                lines.append(f'    "{name}": {obj_str},')
        lines.append("}")
        return "\n".join(lines)


def multi_arm_turn3_reset_codegen_prompt(
    current_positions: Dict = None,
    target_positions: Dict = None,
    robot_ids: List[int] = None,
    instruction: str = "move objects to their original positions",
    context_summary: str = "",
    all_points: list = None,
) -> str:
    """Multi-arm reset 코드 생성 프롬프트 (bi-arm API).

    Args:
        current_positions: Per-arm current positions
            {"left_arm": {obj: {"position": [...]}}, "right_arm": {...}}
        target_positions: Per-arm target positions (same structure)
        robot_ids: [left_id, right_id] e.g., [2, 3]
        instruction: Reset task description
        context_summary: VLM scene summary from Turn 0~2
    """
    if robot_ids is None:
        robot_ids = [2, 3]
    if current_positions is None:
        current_positions = {}
    if target_positions is None:
        target_positions = {}

    current_str = _format_per_arm_positions(current_positions, "current_positions")
    target_str = _format_per_arm_positions(target_positions, "target_positions")

    # Build points section (same as forward Turn 3)
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
Access via `cur_left["object"]["points"]["label"]` or `cur_right["object"]["points"]["label"]`.
{points_desc}
"""

    context_section = ""
    if context_summary:
        context_section = f"""### Scene Context (from prior analysis session)

{context_summary}

"""

    return f"""\
{context_section}### Task: Reset Environment
{instruction}
Move each object from its current position to its target position so the next task episode can begin.

**Robot Setup**:
- Two SO-101 robot arms: left arm (robot{robot_ids[0]}) handles the left side and center, right arm (robot{robot_ids[1]}) handles the right side and center.
- A pre-created `skills` object (MultiArmSkills) is provided as a global variable.
- Call `skills.connect()` at the start and `skills.disconnect()` in a finally block.

**Workspace Images** (provided as attached images):
- **Image 1**: left_arm (robot{robot_ids[0]}) workspace. **Image 2**: right_arm (robot{robot_ids[1]}) workspace.
- Each image shows a **green rectangle** (table boundary) and a **cyan arc** (that arm's reachable range).
- **BRIGHT area** (inside the cyan arc) = reachable zone. **DARK area** (outside the cyan arc) = physically unreachable.
- CRITICAL: The robot CANNOT move to, pick from, or place at ANY position in the darkened area.
  Positions outside the cyan arc are physically unreachable by that arm, even if they appear in the positions dict.

**Arm Selection Rule for Reset** (CRITICAL):
- For each object, check which arm can reach BOTH the current position AND the target position.
- If the current and target are in different arms' workspaces, you MUST use a handover:
  1. Arm A picks from current → places at a handover point in the center (reachable by both arms).
  2. Arm B picks from the handover point → places at the target.
  Use `move_to_pixel` / `place_at_pixel` with a center pixel (e.g., [500, 500]) for the handover point.
- If one arm can reach both current and target, use that single arm (no handover needed).
- NEVER attempt to place at a position outside the executing arm's cyan arc — it will fail with IK error.

{points_section}
### Environment States
Object position z-coordinate = object height (table surface is z=0).

The `current_positions` and `target_positions` dictionaries are provided at runtime as global variables with this structure:
```python
positions = {{
    "left_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in left arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}  # detected critical points
        }},
        ...
    }},
    "right_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in right arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}  # detected critical points
        }},
        ...
    }},
}}
```

Actual detected values:
```python
{current_str}

{target_str}
```

**How to use positions**:
```python
# Split positions by arm at the top of execute_reset_task()
cur_left = current_positions["left_arm"]
cur_right = current_positions["right_arm"]
tgt_left = target_positions["left_arm"]
tgt_right = target_positions["right_arm"]

# Access object positions for each arm
left_cur = cur_left["object_name"]["position"]    # default grasp point
left_pts = cur_left["object_name"]["points"]      # all detected critical points
left_grasp = cur_left["object_name"]["points"]["grasp center"]  # specific point

right_cur = cur_right["object_name"]["position"]
right_pts = cur_right["object_name"]["points"]
```
- `position`: default grasp point for this object.
- `points`: all detected critical points. Choose the best point for the task.
- **CRITICAL**: Use ONLY the exact key names from the `positions`/`current_positions`/`target_positions` dictionaries. Do NOT invent new key names.
- **CRITICAL**: Do NOT redefine or hardcode the dictionaries. They are global variables.
- **CRITICAL**: Do NOT hardcode any coordinate values or compute offsets manually — always use values from the dict.

{MULTI_ARM_API_DOC}

### Skill Composition Patterns (MUST follow exactly)

```python
cur_left = current_positions["left_arm"]
cur_right = current_positions["right_arm"]
tgt_left = target_positions["left_arm"]
tgt_right = target_positions["right_arm"]
approach_height = 0.20

# === STEP 1: Move 1st object to target (no re-detection needed) ===
# Determine which arm to use based on workspace images.
# Example: object is on the left side → use left_arm.
skills.set_subtask("move object_name to target with left arm")

cur = cur_left["object_name"]["position"]
tgt = tgt_left["object_name"]["position"]

# Pick: open gripper → approach → pick → lift
skills.gripper_control(left_arm="open", right_arm="wait",
    left_skill_description="Open left gripper for object_name",
    left_verification_question="Is left gripper open?")
skills.move_to_position(
    left_arm=[cur[0], cur[1], approach_height], right_arm="wait",
    left_skill_description="Move left arm above object_name",
    left_verification_question="Is left arm above object_name?")
skills.pick_object(left_arm=cur, right_arm="wait",
    left_object_name="object_name",
    left_skill_description="Pick up object_name",
    left_verification_question="Is object_name grasped?")
skills.move_to_position(
    left_arm=[cur[0], cur[1], approach_height], right_arm="wait",
    left_skill_description="Lift object_name",
    left_verification_question="Is object_name lifted?")

# Place: approach target → place → retract
skills.move_to_position(
    left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Move object_name above target",
    left_verification_question="Is object_name above target?")
skills.place_object(left_arm=tgt, right_arm="wait",
    left_is_table=True,
    left_skill_description="Place object_name at target",
    left_verification_question="Is object_name placed at target?")
skills.move_to_position(
    left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Retract from object_name target",
    left_verification_question="Is left arm clear of object_name?")

skills.clear_subtask()

# === STEP 2: Move 2nd object (re-detect first) ===
skills.set_subtask("move next_object to target with right arm")

# Re-detection: clear arms → detect → update positions
skills.move_to_initial_state()
updated = skills.detect_objects(["object_name", "next_object"])
if "left_arm" in updated:
    cur_left.update(updated["left_arm"])
if "right_arm" in updated:
    cur_right.update(updated["right_arm"])

cur = cur_right["next_object"]["position"]
tgt = tgt_right["next_object"]["position"]

# Pick with right arm (same pattern, swap left_arm="wait" / right_arm=...)
skills.gripper_control(left_arm="wait", right_arm="open",
    right_skill_description="Open right gripper for next_object",
    right_verification_question="Is right gripper open?")
skills.move_to_position(
    left_arm="wait", right_arm=[cur[0], cur[1], approach_height],
    right_skill_description="Move right arm above next_object",
    right_verification_question="Is right arm above next_object?")
skills.pick_object(left_arm="wait", right_arm=cur,
    right_object_name="next_object",
    right_skill_description="Pick up next_object",
    right_verification_question="Is next_object grasped?")
# ... (same place pattern with right_arm) ...

skills.clear_subtask()

# === STEP 3+: repeat set_subtask → move_to_initial → detect → update → pick → place → clear_subtask ===

# === HANDOVER PATTERN (when current and target are in different arms' workspaces) ===
# Example: object is currently in right arm's area, target is in left arm's area.
#
# ⚠ MANDATORY — Pixel Coordinate Reachability Check (for handover_pixel and any other hardcoded pixel):
#   (a) Identify which arm will execute the move_to_pixel / place_at_pixel.
#   (b) Look at THAT arm's workspace image (Image 1 = left arm, Image 2 = right arm).
#   (c) Verify [y, x] is inside the BRIGHT area (cyan arc) of THAT arm.
#       For handover points, the coordinate must be in the bright area of BOTH arms.
#   (d) If the point is in the dark area → shift to the nearest bright region.
#   (e) Add a comment explaining why this coordinate was chosen.
#   NEVER guess extreme values like [900, 900].
handover_pixel = [500, 500]  # center of table — verified: inside bright area of BOTH arms

# Phase 1: Right arm picks from current → places at center handover point
skills.set_subtask("right arm picks object and places at center for handover")
cur = cur_right["object_name"]["points"]["grasp center"]

skills.gripper_control(left_arm="wait", right_arm="open",
    right_skill_description="Open right gripper", right_verification_question="Is right open?")
skills.move_to_position(left_arm="wait", right_arm=[cur[0], cur[1], approach_height],
    right_skill_description="Approach object", right_verification_question="Is right above object?")
skills.pick_object(left_arm="wait", right_arm=cur, right_object_name="object_name",
    right_skill_description="Pick object", right_verification_question="Is object grasped?")
skills.move_to_position(left_arm="wait", right_arm=[cur[0], cur[1], approach_height],
    right_skill_description="Retract after pick", right_verification_question="Is right lifted?")
skills.move_to_pixel(left_arm="wait", right_arm=handover_pixel,
    right_skill_description="Move to handover point", right_verification_question="Is right above center?")
skills.place_at_pixel(left_arm="wait", right_arm=handover_pixel, right_is_table=True,
    right_skill_description="Place at handover", right_verification_question="Is object at center?")
skills.move_to_pixel(left_arm="wait", right_arm=handover_pixel,
    right_skill_description="Retract after place", right_verification_question="Is right clear?")
skills.clear_subtask()

# Re-detect after handover
skills.move_to_initial_state()
updated = skills.detect_objects(["object_name"])
if "left_arm" in updated:
    cur_left.update(updated["left_arm"])
if "right_arm" in updated:
    cur_right.update(updated["right_arm"])

# Phase 2: Left arm picks from center → places at target
skills.set_subtask("left arm picks object from center and places at target")
cur_updated = cur_left["object_name"]["points"]["grasp center"]
tgt = tgt_left["object_name"]["points"]["grasp center"]

skills.gripper_control(left_arm="open", right_arm="wait",
    left_skill_description="Open left gripper", left_verification_question="Is left open?")
skills.move_to_position(left_arm=[cur_updated[0], cur_updated[1], approach_height], right_arm="wait",
    left_skill_description="Approach object at center", left_verification_question="Is left above object?")
skills.pick_object(left_arm=cur_updated, right_arm="wait", left_object_name="object_name",
    left_skill_description="Pick object from center", left_verification_question="Is object grasped?")
skills.move_to_position(left_arm=[cur_updated[0], cur_updated[1], approach_height], right_arm="wait",
    left_skill_description="Retract after pick", left_verification_question="Is left lifted?")
skills.move_to_position(left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Move to target", left_verification_question="Is left above target?")
skills.place_object(left_arm=tgt, right_arm="wait", left_is_table=True,
    left_skill_description="Place at target", left_verification_question="Is object at target?")
skills.move_to_position(left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Retract after place", left_verification_question="Is left clear?")
skills.clear_subtask()
```

### Unstacking (Disassembling a Stack)

When objects are stacked, you MUST unstack from **top to bottom**.
- The object with the highest z is topmost — always pick it first.
- After picking each stacked object, re-detect to get updated z-heights.

### Unfolding (Restoring a folded towel/cloth to flat state)

When a towel or cloth is folded and needs to be unfolded back to its original flat state,
use the bimanual UNFOLD pattern — the reverse of FOLD.

```python
# UNFOLD pattern: grasp the folded edge → arc trajectory to unfold → place flat
skills.set_subtask("unfold towel — grasp folded edge, pull back to flat")

# Get grasp points from detected critical points (the folded/free edge)
left_grasp = cur_left["towel"]["points"]["left grasp point"]
right_grasp = cur_right["towel"]["points"]["right grasp point"]

# Target: where the edge should end up when unfolded (from target_positions)
left_target = tgt_left["towel"]["points"]["left grasp point"]
right_target = tgt_right["towel"]["points"]["right grasp point"]

# 1. Open + approach
skills.gripper_control(left_arm="open", right_arm="open",
    left_skill_description="Open left gripper", left_verification_question="Is left open?",
    right_skill_description="Open right gripper", right_verification_question="Is right open?")
skills.move_to_position(
    left_arm=[left_grasp[0], left_grasp[1], approach_height],
    right_arm=[right_grasp[0], right_grasp[1], approach_height],
    left_skill_description="Approach left grasp", left_verification_question="Is left above grasp?",
    right_skill_description="Approach right grasp", right_verification_question="Is right above grasp?")

# 2. Pick (descend + grip — synchronized)
skills.bimanual_pick_object(left_arm=left_grasp, right_arm=right_grasp, object_name="towel")

# 3. Unfold arc (reverse fold — arc lifts and descends to target, synchronized)
skills.bimanual_fold(
    left_start=left_grasp, right_start=right_grasp,
    left_end=left_target, right_end=right_target, arc_height=0.20,
    left_skill_description="Unfold left to target", left_verification_question="Is left unfolded?",
    right_skill_description="Unfold right to target", right_verification_question="Is right unfolded?")

# 4. Place (descend + release — synchronized)
skills.bimanual_place_object(left_arm=left_target, right_arm=right_target, object_name="towel")

# 5. Retract
skills.move_to_position(
    left_arm=[left_target[0], left_target[1], approach_height],
    right_arm=[right_target[0], right_target[1], approach_height],
    left_skill_description="Retract left", left_verification_question="Is left clear?",
    right_skill_description="Retract right", right_verification_question="Is right clear?")

skills.clear_subtask()
```

### Code Skeleton

```python
def execute_reset_task():
    '''Move objects from current to target positions using two arms.'''
    skills.connect()

    try:
        cur_left = current_positions["left_arm"]
        cur_right = current_positions["right_arm"]
        tgt_left = target_positions["left_arm"]
        tgt_right = target_positions["right_arm"]
        approach_height = 0.20

        skills.move_to_initial_state()

        # Subtask 1: set_subtask → pick → place → clear_subtask
        # Re-detection: move_to_initial_state → detect_objects → update cur_left/cur_right
        # Subtask 2: set_subtask → pick → place → clear_subtask
        # ... repeat for each object ...

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
```

### Task Requirements
1. **Arm selection by reachability** (CRITICAL): For each object, check which arm can reach BOTH the current AND target positions (bright area in workspace images).
   - If one arm reaches both → use that arm directly.
   - If current and target are in different arms' workspaces → use the HANDOVER pattern (Arm A picks → places at center → Arm B picks from center → places at target).
   - NEVER attempt to move/place at a position outside the executing arm's cyan arc.
   - **MANDATORY (pixel reachability check)**: Before writing ANY hardcoded pixel coordinate:
     (a) Identify the executing arm. (b) Check THAT arm's workspace image (Image 1 = left, Image 2 = right).
     (c) Verify [y, x] is inside the BRIGHT area (cyan arc). For handover points, verify BOTH arms' images.
     (d) If in dark area → shift inward to nearest bright point. (e) Add a code comment with reasoning.
     NEVER use extreme values like [900, 900].
2. Use `skills.move_to_position()` / `skills.pick_object()` / `skills.place_object()` / `skills.gripper_control()` for ALL operations. Pass `left_arm="wait"` or `right_arm="wait"` for the arm that should hold position.
   - **Pick pattern**: open → approach (move to approach_height above object) → pick_object (descend + grip) → **retract (move back to approach_height — MANDATORY)** → move to next target.
   - **Place pattern**: move_to_position (approach above target) → place_object (descend + release) → retract (move back to approach_height).
   - NEVER skip the retract step after pick or place.
3. **Subtask pattern**: Wrap each logical unit of work with `set_subtask()` before and `clear_subtask()` after.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, ALL grippers must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. If an arm is holding an object, the subtask is not yet complete — do NOT call `clear_subtask()` until all grippers have released.
4. **Re-detection (MANDATORY)**: After each subtask (after `clear_subtask()`), call `skills.move_to_initial_state()` to clear arms from camera view, then `skills.detect_objects([...all object names...])` to update positions. Skip re-detection only after the very last subtask. The 1st object does NOT need re-detection.
   - **CRITICAL**: After `cur_left.update()` / `cur_right.update()`, you MUST **re-assign ALL local variables** extracted from the positions dict. `update()` replaces dict entries, but previously extracted variables still reference the OLD values.
5. Always start with `skills.move_to_initial_state()`, end with `skills.move_to_initial_state()` then `skills.move_to_free_state()`.
6. Use `approach_height = 0.20` for approach/retreat movements.
7. ALWAYS pass `left_skill_description`/`right_skill_description` and `left_verification_question`/`right_verification_question` for every arm that is NOT `"wait"`.
8. Always include try/finally with `skills.disconnect()` for cleanup.
9. **Unstacking**: If objects are stacked, always unstack from top to bottom (highest z first).
10. **Bimanual move**: When both arms hold the **same object** and must move together (e.g., unfolding, stretching), use `skills.bimanual_move()` instead of `skills.move_to_position()`. This guarantees synchronized progress.
11. **Unfolding** (towel, cloth): Use the UNFOLD pattern above — `bimanual_pick_object` → `bimanual_fold` (reverse direction) → `bimanual_place_object`. Grasp the folded/free edge and arc it back to the original flat position. Use detected `"points"` for grasp locations — do NOT compute offsets manually.

### Output Format
- Provide complete executable Python code
- Do not include markdown code blocks

**Generate the complete executable RESET code:**"""
