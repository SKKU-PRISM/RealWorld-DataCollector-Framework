"""
stir -- circular stirring skill

Executes a circular stirring motion around a given center point.
Uses move_linear to approximate a circle with straight-line segments.
Suitable for stirring tea, mixing paint, blending ingredients in a bowl, etc.

This skill assumes the EE is already positioned above the stir center
(e.g., via move_approach_position in LLM-generated code). It handles only
the core stirring action:

    1. move_to_position  -- descend to circle start point at stir_height
    2. move_linear x N   -- circular path segments (num_rotations turns)

    Circular path (top-down view):
        *---*
       /     \\
      *       *    <-- segments_per_rotation line segments approximate a circle
       \\     /       radius = stir_radius
        *---*         center = center position

Approach above the center and lifting after stirring are handled by the
LLM-generated code that calls this skill.

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.stir import stir
    from skills.move_approach_position import move_approach_position

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml", frame="world")
    skills.connect()

    cup_center = [0.20, 0.0, 0.03]

    # LLM generated code handles approach:
    skills.gripper_close(...)  # holding spoon
    move_approach_position(skills, object_position=cup_center, approach_height=0.10, ...)

    # Core stir action:
    stir(skills, center=cup_center, stir_radius=0.02, num_rotations=3, stir_height=0.02, ...)

    # LLM generated code handles retreat:
    skills.move_to_position([cup_center[0], cup_center[1], 0.10], ...)

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def stir(
    skills,
    center: Union[List[float], np.ndarray],
    stir_radius: float = 0.02,
    num_rotations: int = 3,
    stir_height: float = 0.02,
    rotation_duration: Optional[float] = None,
    segments_per_rotation: int = 8,
    clockwise: bool = True,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    Execute a circular stirring motion.

    Traces a circle of radius stir_radius around center for num_rotations
    turns. The circle is approximated by segments_per_rotation straight-line
    segments per turn.

    The caller is responsible for positioning the EE above the center
    before calling this function and for lifting the EE afterward.

    Args:
        skills: LeRobotSkills instance (already connected).

        center: Stir center position [x, y, z] (meters).
                - In the coordinate frame specified at skills init (usually "world").
                - Typically the center of the container (cup, bowl).
                - The z value is for reference only; actual height is stir_height.
                - Example: [0.20, 0.0, 0.03]

        stir_radius: Radius of the circular path (meters). default=0.02 (2 cm).
                     - Tea cup: 0.01--0.02 (1--2 cm)
                     - Large bowl: 0.03--0.05 (3--5 cm)
                     - Keep smaller than the container radius to avoid the wall.

        num_rotations: Number of full turns. default=3.
                       - 1: single turn
                       - 3: normal stir
                       - 5+: thorough mixing

        stir_height: EE height during stirring (meters). default=0.02 (2 cm).
                     - Height at which the EE moves inside the container.
                     - Set high enough to avoid scraping the bottom.
                     - Set low enough to contact the contents.

        rotation_duration: Time for one full turn (seconds). default=None.
                           - None: auto-calculated from circumference (3 cm/s, min 2 s).
                           - 2.0: fast stir
                           - 4.0: slow stir
                           - Segment time = rotation_duration / segments_per_rotation.

        segments_per_rotation: Line segments per turn. default=8.
                               - 6: hexagonal (coarse)
                               - 8: octagonal (normal)
                               - 12: dodecagonal (smooth, slower)

        clockwise: Rotation direction (top-down view). default=True.
                   - True: clockwise
                   - False: counter-clockwise

        duration: Duration for the initial descent move (seconds). default=None.
                  - None: use skills internal default.

        object_name: Name of the target container (optional). default=None.
                     - Used for subgoal labels during recording.
                     - Example: "tea cup", "mixing bowl"

        skill_description: Skill description string (optional). default=None.
                           - None: auto-generated from object_name.

    Returns:
        bool: True if all segments succeeded, False if any segment failed.
    """
    c = np.array(center, dtype=float)

    desc_prefix = f"stir {object_name}" if object_name else "stir"

    # Circumference and auto duration
    circumference = 2 * np.pi * stir_radius
    if rotation_duration is None:
        rotation_duration = max(circumference / 0.03, 2.0)  # 3 cm/s, min 2 s

    segment_duration = rotation_duration / segments_per_rotation

    skills._log(f"\n{'='*60}")
    skills._log(f"[stir] {desc_prefix}")
    skills._log(f"  Center: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
    skills._log(f"  Radius: {stir_radius*100:.1f}cm")
    skills._log(f"  Rotations: {num_rotations}, Direction: {'CW' if clockwise else 'CCW'}")
    skills._log(f"  Segments/rotation: {segments_per_rotation}")
    skills._log(
        f"  Rotation duration: {rotation_duration:.1f}s "
        f"(segment: {segment_duration:.2f}s)"
    )
    skills._log(f"  Stir height: {stir_height*100:.1f}cm")
    skills._log(f"{'='*60}")

    # --- Step 1: Descend to circle start point at stir_height ---
    start_angle = 0.0
    start_x = c[0] + stir_radius * np.cos(start_angle)
    start_y = c[1] + stir_radius * np.sin(start_angle)

    skills._log("\n[Step 1] Move to circle start at stir height")
    if not skills.move_to_position(
        position=[start_x, start_y, stir_height],
        duration=duration,

        maintain_pitch=True,
        target_name=object_name,
        skill_description=f"{desc_prefix}: descend to stir height",
    ):
        skills._log("ERROR: Failed to reach circle start at stir height")
        return False

    # --- Step 2: Circular stir segments ---
    all_success = True
    total_segments = num_rotations * segments_per_rotation
    direction = -1 if clockwise else 1  # CW: decreasing angle, CCW: increasing angle

    for seg in range(total_segments):
        rotation_num = seg // segments_per_rotation + 1
        seg_in_rotation = seg % segments_per_rotation + 1

        angle_start = direction * (2 * np.pi * seg / segments_per_rotation)
        angle_end = direction * (2 * np.pi * (seg + 1) / segments_per_rotation)

        seg_start = [
            c[0] + stir_radius * np.cos(angle_start),
            c[1] + stir_radius * np.sin(angle_start),
            stir_height,
        ]
        seg_end = [
            c[0] + stir_radius * np.cos(angle_end),
            c[1] + stir_radius * np.sin(angle_end),
            stir_height,
        ]

        skills._log(
            f"\n[Step 2.{seg+1}] Rotation {rotation_num}/{num_rotations}, "
            f"segment {seg_in_rotation}/{segments_per_rotation}"
        )
        success = move_linear(
            skills,
            start=seg_start,
            end=seg_end,
            duration=segment_duration,
    
            maintain_pitch=True,
            target_name=object_name,
            skill_description=(
                skill_description
                or f"{desc_prefix}: rotation {rotation_num} seg {seg_in_rotation}"
            ),
        )
        if not success:
            skills._log(f"WARNING: Segment {seg+1} did not fully converge")
            all_success = False

    skills._log(f"\n[stir] Complete (all_success={all_success})")
    return all_success


if __name__ == "__main__":
    from skills.move_approach_position import move_approach_position
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        exit(1)

    try:
        skills.move_to_initial_state()

        # Close gripper (assume spoon is held)
        skills.gripper_close()

        cup_center = [0.20, 0.0, 0.03]

        # Approach above center (handled by LLM code in production)
        print("\n=== Approach above center ===")
        move_approach_position(
            skills,
            object_position=cup_center,
            approach_height=0.10,
            object_name="test cup",
        )

        # Core stir action
        print("\n=== Test: Stir in circle ===")
        success = stir(
            skills,
            center=cup_center,
            stir_radius=0.03,
            num_rotations=2,
            stir_height=0.02,
            segments_per_rotation=8,
            object_name="test cup",
            skill_description="test: stir in cup",
        )
        print(f"Result: {'success' if success else 'failed'}")

        # Lift after stirring (handled by LLM code in production)
        print("\n=== Lift after stirring ===")
        skills.move_to_position(
            position=[cup_center[0], cup_center[1], 0.10],
            target_name="test cup",
            skill_description="test: lift after stir",
        )

        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
