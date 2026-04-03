"""
Turn 2+ Prompt — Crop-then-Point (per object)

Predicts grasp/interaction points from per-object crop images.
(1) grasp points: optimal gripper grasp locations regarding success of the task.
(2) interaction points: None-grasping locations, only functional sub-parts for task (pin, hole, slot, etc.) that are critical for task execution.
Crops are generated from bboxes detected in Turn 1, and this prompt is called once per object.
"""


def turn2_crop_pointing_prompt(object_label: str, has_side_view: bool = False, canonical_point_labels: dict = None) -> str:
    """
    Turn 2: exact location for each critical manipulation point on the cropped image of the target object.

    Args:
        object_label: label of the object detected in Turn 1 (e.g., "red block")
        has_side_view: If True, two crop images are provided (overhead + side-view)
                       and the output includes points for both views.
        canonical_point_labels: {object_label: ["grasp center", "plate center", ...]}
                               이전 에피소드에서 사용된 point 라벨. 제공되면 동일 라벨 강제.

    Returns:
        prompt string corpus for turn 2
    """
    if has_side_view:
        prompt = f"""
Now I am showing you **two cropped close-up images** of the object "{object_label}":
1. **Crop 1 (Overhead view)** — cropped from the overhead camera image.
2. **Crop 2 (Side view)** — cropped from the side camera image.

Based on your analysis above, identify critical points on this object in **both** views.

**Point types**:
1. **grasp** — optimal gripper grasp location for successful task execution.
2. **interaction** — non-grasping functional sub-part location critical for task execution (e.g., pin, hole, slot, rim, edge).

For each point, provide:
1. **point_2d**: The point location as `[y, x]` — 2 integers, each normalized to **0–1000** (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of that cropped image).
2. **label**: A short, descriptive name (e.g., "grasp center", "pin tip", "hole opening"). **Use the same label for the same physical point across both views.**
3. **role**: `"grasp"` or `"interaction"`.
4. **reasoning**: Why this point matters, referencing your previous analysis.

### Output Format
Return a JSON block:
```json
{{
  "overhead_critical_points": [
    {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
  ],
  "sideview_critical_points": [
    {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
  ]
}}
```

**Important**:
- Look carefully at each cropped image and provide accurate coordinates.
- Coordinates are normalized 0–1000 relative to each respective cropped image.
- **Labels must be consistent** between overhead and sideview — the same physical point must have the same label in both views.
- Both `overhead_critical_points` and `sideview_critical_points` must contain the **same set of points** (same labels, same roles), just with different coordinates for each view.
""".strip()
    else:
        prompt = f"""
Now I am showing you a **cropped close-up image** of the object "{object_label}" from the overhead camera.

Based on your analysis above, identify critical points on this object.

**Point types**:
1. **grasp** — optimal gripper grasp location for successful task execution.
2. **interaction** — non-grasping functional sub-part location critical for task execution (e.g., pin, hole, slot, rim, edge).

For each point, provide:
1. **point_2d**: The point location as `[y, x]` — 2 integers, each normalized to **0–1000** (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image).
2. **label**: A short, descriptive name (e.g., "grasp center", "pin tip", "hole opening").
3. **role**: `"grasp"` or `"interaction"`.
4. **reasoning**: Why this point matters, referencing your previous analysis.

### Output Format
Return a JSON block:
```json
{{
  "critical_points": [
    {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
  ]
}}
```

**Important**:
- Look carefully at the cropped image and provide accurate coordinates.
- Coordinates are normalized 0–1000 relative to this cropped image.
- **Grasp stability**: Choose the grasp point that maximizes gripper contact and grip stability. Prefer the geometric center of the widest graspable surface. Avoid edges, corners, or thin protrusions where the gripper may slip.
- **Deformable objects** (towel, cloth, paper): NEVER place grasp points at exact corners or extreme edges — the gripper will slip off. Instead, place grasp points **at least 75 pixels inward from the edge** (in the 0–1000 normalized coordinate space of the cropped image) so the gripper can firmly pinch the material with sufficient contact area.
- **Task awareness**: Consider what the robot needs to do with this object. If the object will be stacked, placed precisely, or inserted, choose a grasp point that allows stable holding during the entire manipulation sequence.
""".strip()

    # canonical point labels가 있으면 강제 추가
    if canonical_point_labels and object_label in canonical_point_labels:
        labels = canonical_point_labels[object_label]
        labels_str = ", ".join(f'"{l}"' for l in labels)
        prompt += f"\n\n**CRITICAL: You MUST use exactly these point labels for this object: [{labels_str}]. Do NOT rename, paraphrase, or add/remove labels.**"

    return prompt


# ──────────────────────────────────────────────
# [ARCHIVED] 기존 turn2 prompt
# "identify each critical manipulation point"가 모호하여
# point types 정의를 output format보다 위로 올려 명시함.
# ──────────────────────────────────────────────
# def turn2_crop_pointing_prompt(object_label: str) -> str:
#     return f"""
# Now I am showing you a **cropped close-up image** of the object "{object_label}" from the overhead camera.
#
# Based on your analysis above, identify each critical manipulation point on this object and provide their **precise locations** in the image.
#
# For each point, provide:
# 1. **point_2d**: The point location as `[y, x]` — 2 integers, each normalized to **0–1000** (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image).
# 2. **label**: A short, descriptive name (e.g., "grasp center", "pin tip", "hole opening").
# 3. **role**: One of:
#    - `"grasp"` — optimal gripper grasp location.
#    - `"interaction"` — functional sub-part for task (pin, hole, slot, etc.).
# 4. **reasoning**: Why this point matters, referencing your previous analysis.
#
# ### Output Format
# Return a JSON block:
# ```json
# {{
#   "critical_points": [
#     {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
#     {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
#   ]
# }}
# ```
#
# **Important**:
# - Look carefully at the cropped image and provide accurate coordinates.
# - Coordinates are normalized 0–1000 relative to this cropped image.
# """.strip()
