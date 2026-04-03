"""
Robot API Skills Documentation (Gemini Robotics class definition style)

LeRobotSkills 클래스의 API 문서를 LLM 프롬프트용으로 제공합니다.
실제 구현은 skills/skills_lerobot.py에 있으며,
이 파일은 LLM이 코드 생성 시 참조할 API 명세만 포함합니다.
"""

ROBOT_API_DOC = '''class LeRobotSkills:
    """Interface for controlling the LeRobot SO-101 single robot arm.
    The robot has a 5-DOF arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
    with an asymmetric two-finger gripper (left finger is fixed, right finger is actuated).
    The gripper can open up to 0.07m (7cm) and approaches objects from directly above (top-down grasp).
    All positions are specified in the world coordinate frame in meters.

    IMPORTANT: Every skill method accepts two optional string parameters for dataset recording:
        skill_description (str): Concise sentence describing the action and purpose.
            Example: "Move gripper above chocolate_pie_1 to prepare for picking"
        verification_question (str): Yes/No question to visually verify the action's outcome.
            Example: "Is the gripper positioned above chocolate_pie_1?"
    You MUST always pass both parameters for every skill call.
    """

    def connect(self) -> bool:
        """Connects to robot hardware and initializes kinematics. Returns True if successful."""

    def disconnect(self):
        """Disconnects from robot hardware. Must be called in a finally block for cleanup."""

    def gripper_open(self, duration: float = 1.5, ratio: float = 1.0):
        """Opens the gripper to the specified ratio.

        Args:
            duration: Movement duration in seconds.
            ratio: Open ratio where 0.0 = fully closed and 1.0 = fully open.
                Use 0.7 for partial open during place operations for controlled release.
        """

    def gripper_close(self, duration: float = 1.5):
        """Closes the gripper to grasp an object.
        The gripper closes with sufficient force to hold objects up to ~500g.

        Args:
            duration: Movement duration in seconds.
        """

    def move_to_initial_state(self) -> bool:
        """Moves the arm to its home position. Call at the start of every task."""

    def move_to_free_state(self) -> bool:
        """Moves the arm to a safe parking position. Call as the very last skill after task completion."""

    def move_to_position(self, position: list[float], duration: float = None,
                         target_name: str = None) -> bool:
        """Moves the end-effector to the given XYZ position in world coordinates.
        Use this for approach movements (moving above an object before pick/place),
        retreat movements (lifting after pick/place), and transit movements between objects.

        The world coordinate frame origin is at the rear-center of the workspace table
        on the table surface:
            Positive x: towards front of the table
            Negative x: towards back of the table
            Positive y: towards right
            Negative y: towards left
            Positive z: up, towards ceiling (z=0 is table surface)

        Args:
            position: Target position [x, y, z] in meters in world frame.
            duration: Movement duration in seconds. Uses default if None.
            target_name: Name of the target object for subgoal labeling in dataset recording.
                Example: "yellow dice", "blue dish".

        Returns:
            True if movement successful, False if position is outside reachable workspace.
        """

    def rotate_90degree(self, direction: int = 1, duration: float = 2.0) -> bool:
        """Rotates the gripper (wrist_roll joint) by 90 degrees in place.
        The arm position remains the same; only the gripper orientation changes.
        Use this when an object needs to be reoriented after picking.

        Args:
            direction: 1 for clockwise rotation, -1 for counter-clockwise rotation.
            duration: Movement duration in seconds.

        Returns:
            True if rotation successful.
        """

    def execute_pick_object(self, object_position: list[float],
                            object_name: str = None) -> bool:
        """Executes a pick (grasp) action at the given object position.
        Must be called AFTER moving to the approach position above the object.
        The robot descends to the grasp height (2.5cm offset from object top),
        closes the gripper to grasp the object, and internally saves the current
        pitch angle for the subsequent place operation.

        IMPORTANT: Pass the object position as-is from the positions dictionary.
        The function internally calculates the grasp height (2.5cm below object top).
        Do NOT subtract any offset from z — just pass pick_pos directly.

        Args:
            object_position: Object position [x, y, z] in meters. Pass as-is from
                positions dictionary. The function internally handles the grasp offset.
            object_name: Name of the object being picked for subgoal labeling.
                Example: "yellow dice", "red cup".

        Returns:
            True if pick successful (gripper closed around object).
        """

    def execute_place_object(self, place_position: list[float],
                             is_table: bool = True,
                             gripper_open_ratio: float = 1.0, target_name: str = None) -> bool:
        """Executes a place (release) action at the given target position.
        Must be called AFTER moving to the approach position above the target.
        The robot descends to the place height with the pitch angle saved during
        the pick operation automatically restored, then opens the gripper to
        release the object.

        IMPORTANT: Pass the target surface position as-is from the positions dictionary.
        The function internally calculates the correct release height using the pick height saved during execute_pick_object.
        - is_table=True: z value is ignored (release height = pick_z above table surface)
        - is_table=False: z value is used as the target surface height (release height = surface_z + pick_z)
        ALWAYS use gripper_open_ratio=0.7 for controlled release.

        Args:
            place_position: Target position [x, y, z] in meters. Pass the target object/surface
                position as-is. The z coordinate is only used when is_table=False.
            is_table: True if placing directly on the table surface (z=0),
                False if placing on top of another object.
            gripper_open_ratio: How much to open the gripper for release (0.0 to 1.0).
                ALWAYS use 0.7 (70% open) for controlled object release.
            target_name: Name of the placement target for subgoal labeling.
                Example: "blue dish", "table".

        Returns:
            True if place successful (object released at target position).
        """

    def execute_press(self, position: list[float], press_depth: float = 0.01,
                      contact_height: float = 0.02, press_duration: float = 0.5,
                      hold_time: float = 0.3, max_press_torque: int = 400,
                      duration: float = None,
                      target_name: str = None) -> bool:
        """Executes a 2-phase press action (normal descent + torque-limited press).
        Must be called AFTER closing the gripper and moving to approach position above target.
        Phase 1: Descend to contact surface at normal speed.
        Phase 2: Press below contact with torque limit for safe force application.

        Args:
            position: Target position [x, y, z] in meters (center of press target).
            press_depth: How far to press below the contact surface in meters (default 0.01 = 1cm).
            contact_height: Height of the contact surface in meters (use object's z value).
            press_duration: Duration for the pressing phase in seconds.
            hold_time: How long to hold at pressed position in seconds (default 0.3).
            max_press_torque: Torque limit during press phase (0-1000, default 400).
            duration: Duration for the descent phase. Uses default if None.
            target_name: Name of the target for subgoal labeling.
                Example: "power button", "microphone".

        Returns:
            True if press action completed successfully.
        """

    def execute_push(self, start_position: list[float], end_position: list[float],
                     push_height: float = 0.01, duration: float = None,
                     object_name: str = None) -> bool:
        """Pushes an object in a straight line using Cartesian linear motion.
        Must be called AFTER closing the gripper and moving to approach position above start.
        Internally handles everything after approach:
          1. Descends to pre-contact position (3cm behind start in opposite push direction)
          2. Moves linearly through start to end (run-up + push in one straight line)
          3. Retreats to approach_height (20cm) after push
        No need for a separate retreat move after calling this method.

        Args:
            start_position: Contact point [x, y, z] in meters — the interaction point
                where the gripper first touches the object (e.g., object's left edge
                for a left-to-right push). The z value is used as reference for object height.
            end_position: Push end position [x, y, z] in meters.
                Determines push direction and distance in the xy plane.
            push_height: Height of the end-effector during the push in meters (default 0.01).
                Set to approximately 1/3 of the object height for good contact.
                Too low risks table collision; too high misses the object.
            duration: Push movement duration in seconds. None for auto-calculation based on distance.
            object_name: Name of the object being pushed for subgoal labeling.
                Example: "bread", "red block".

        Returns:
            True if push completed successfully.
        """

    def detect_objects(self, queries: list[str], timeout: float = 5.0,
                       visualize: bool = False) -> dict:
        """Re-detects objects in real-time using the camera during code execution.
        Returns updated positions for the queried objects.
        Use this after placing an object to get its new position (e.g., for stacking).

        Args:
            queries: List of object names to detect. Example: ["brown block", "green block"].
            timeout: Detection timeout in seconds (default 5.0).
            visualize: Whether to show detection visualization window (default False).

        Returns:
            Dict mapping object name to position info:
            {
                "brown block": {"position": [x, y, z], ...},
                "green block": {"position": [x, y, z], ...},
            }
            Returns None for objects that could not be detected.

        Example:
            # After placing block A, re-detect to get its updated position
            updated = skills.detect_objects(["block A"])
            new_pos = updated["block A"]["position"]
            # Now place block B on top of block A using the updated position
            skills.execute_place_object(new_pos, is_table=False, ...)
        """

    def set_subtask(self, description: str) -> None:
        """Set sub-task label for recording. Groups multiple skills under one higher-level label.
        Call this before each object's pick-place sequence.

        Args:
            description: Short description of the subtask (e.g., "pick red block and place at center")
        """

    def clear_subtask(self) -> None:
        """Clear the current sub-task label. Call after all objects are moved."""

    def detect_objects(self, queries: list) -> dict:
        """Re-detect objects in the current camera view using VLM + depth sensor.

        Returns updated positions including z height (from RealSense depth).
        IMPORTANT: Always call this after set_subtask() to refresh object positions
        before picking/placing. This is critical when objects have been moved
        (e.g., stacked on top of each other, changing their z height).

        Args:
            queries: List of object names to detect, e.g., ["red block", "yellow block"]

        Returns:
            Dict: {"object_name": {"position": [x, y, z], "pixel": (u, v), "bbox_px": (w, h)}}
            Returns None for objects not found.

        Pattern — always follow this sequence when switching to a new object:
            # 1. Set subtask label
            skills.set_subtask("pick purple block and place at target")
            # 2. Move to initial state (clear arm from camera view before detection)
            skills.move_to_initial_state()
            # 3. Re-detect ALL objects to get updated positions
            updated = skills.detect_objects(["red block", "yellow block", "purple block"])
            # 4. Update local variables with fresh positions
            if updated["purple block"]:
                purple_pos = updated["purple block"]["position"]
            if updated["yellow block"]:
                target_pos = updated["yellow block"]["position"]
            # 5. Now pick/place with accurate coordinates
            skills.execute_pick_object(purple_pos, ...)
        """'''
