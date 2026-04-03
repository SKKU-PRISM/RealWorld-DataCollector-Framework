"""
Turn Test Prompt — Waypoint Trajectory Prediction

Given all detected points (grasp + interaction) from Turn 2 and the current subtask (phase),
the VLM selects the relevant points and predicts a waypoint trajectory between them.
"""


def turn_test_waypoint_trajectory_prompt(
    instruction: str,
    phase: str,
    all_points: list[dict],
    has_side_view: bool = False,
) -> str:
    """
    Turn Test: predict waypoint trajectory for the current phase.

    The VLM receives the full list of detected points and autonomously selects
    which points are relevant for the given phase, then predicts waypoints between them.

    Args:
        instruction: the full task instruction
        phase: current subtask / phase description (e.g., "insert the brown peg into the gray structure")
        all_points: all detected points from Turn 2 (grasp + interaction), each with:
            - "object_label", "label", "role", "px", "py"
        has_side_view: True if a side-view image is provided

    Returns:
        prompt string for waypoint trajectory prediction
    """
    # Build all points description (include side-view coords when available)
    points_desc = ""
    any_sv = any(pt.get("sv_py") is not None for pt in all_points)
    for i, pt in enumerate(all_points):
        obj = pt.get("object_label", "unknown")
        label = pt.get("label", "")
        role = pt.get("role", "unknown")
        py = pt.get("py", 0)
        px = pt.get("px", 0)
        if any_sv and pt.get("sv_py") is not None:
            sv_py = pt.get("sv_py", 0)
            sv_px = pt.get("sv_px", 0)
            points_desc += (
                f"  {i + 1}. [{role}] **{label}** (on {obj}) "
                f"— overhead: (y={py}, x={px}), sideview: (y={sv_py}, x={sv_px})\n"
            )
        else:
            points_desc += f"  {i + 1}. [{role}] **{label}** (on {obj}) — pixel: (y={py}, x={px})\n"

    # View-specific sections
    if has_side_view:
        view_intro = (
            "You are given two images:\n"
            "1. **Overhead view** (top-down, already seen in previous turns)\n"
            "2. **Side view** (attached, 640x480 pixels) — shows the workspace from the side, "
            "revealing object heights and vertical structure.\n\n"
            "Predict waypoint trajectories on **both** views."
        )
        output_format = """\
### Output Format
Return a JSON block with **exactly 10 waypoints** per view:
```json
{{
  "selected_points": ["<label of selected point 1>", "<label of selected point 2>"],
  "overhead_waypoints": [
    {{"point_2d": [y, x], "label": "wp1: <description>", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "wp2: <description>", "reasoning": "..."}},
    ...
    {{"point_2d": [y, x], "label": "wp10: <description>", "reasoning": "..."}}
  ],
  "sideview_waypoints": [
    {{"point_2d": [y, x], "label": "wp1: <description>", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "wp2: <description>", "reasoning": "..."}},
    ...
    {{"point_2d": [y, x], "label": "wp10: <description>", "reasoning": "..."}}
  ]
}}
```

**Critical — 1:1 correspondence**:
- Both `overhead_waypoints` and `sideview_waypoints` must have **exactly the same number** of waypoints.
- Waypoint at index *i* in both lists must represent the **same 3D point** projected onto each view.
  - Overhead waypoint *i* gives the (x, y) horizontal position.
  - Side-view waypoint *i* gives the height (z) at that same point.
- Use matching labels (wp1, wp2, ..., wpN) to make the correspondence explicit.

**Other notes**:
- `selected_points`: labels of the points you chose as start/end for this phase.
- `overhead_waypoints`: `[y, x]` in **pixel coordinates** of the overhead image (640x480).
- `sideview_waypoints`: `[y, x]` in **pixel coordinates** of the side-view image (640x480).
- **Anchor point constraint**: The sideview coordinates listed above for each detected point were precisely measured via crop-then-point. The **first** and **last** sideview waypoints (start/end) must use the exact sideview coordinates of the corresponding selected points as anchors. Only intermediate waypoints should be newly estimated."""
    else:
        view_intro = (
            "Based on the overhead camera image you have already seen:"
        )
        output_format = """\
### Output Format
Return a JSON block with **exactly 10 waypoints**:
```json
{{
  "selected_points": ["<label of selected point 1>", "<label of selected point 2>"],
  "overhead_waypoints": [
    {{"point_2d": [y, x], "label": "wp1: start — <description>", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "wp2: <description>", "reasoning": "..."}},
    ...
    {{"point_2d": [y, x], "label": "wp10: end — <description>", "reasoning": "..."}}
  ]
}}
```

**Important**:
- `selected_points`: labels of the points you chose as start/end for this phase.
- Coordinates are `[y, x]` in **pixel coordinates** of the overhead image (640x480).
- The trajectory represents the top-down 2D projection of the end-effector path."""

    return f"""
{view_intro}

**Task**: {instruction}
**Current phase**: {phase}

**All detected points from scene analysis**:
{points_desc}
From the points above, select the ones relevant to the current phase, then predict the **waypoint trajectory** the robot end-effector should follow to complete this phase.

**Guidelines**:
- First, identify which points are the start and end for this phase.
- The first waypoint should be at or near the selected start point; the last at or near the selected end point.
- Add intermediate waypoints where the path changes direction or where precision matters.
- Waypoints should be ordered sequentially along the movement path.
- Consider obstacle avoidance and smooth movement.
- Provide **exactly 10 waypoints** per view.

{output_format}
""".strip()
