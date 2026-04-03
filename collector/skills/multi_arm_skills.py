#!/usr/bin/env python3
"""
Multi-Arm Skills API

Provides synchronized bi-arm control by wrapping two LeRobotSkills instances.
Supports parallel execution with ThreadPoolExecutor and "wait" semantics
for single-arm-only moves.

Usage:
    from skills.multi_arm_skills import MultiArmSkills

    multi = MultiArmSkills(
        left_config="robot_configs/robot/so101_robot2.yaml",
        right_config="robot_configs/robot/so101_robot3.yaml",
    )
    multi.connect()

    # Both arms move simultaneously
    multi.move_to_position(left_arm=[0.15, -0.10, 0.20], right_arm=[0.15, 0.10, 0.20])

    # Only right arm moves, left stays
    multi.move_to_position(left_arm="wait", right_arm=[0.15, 0.10, 0.05])

    multi.disconnect()
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from skills.skills_lerobot import LeRobotSkills


# Sentinel value for "no movement" on one arm
WAIT = "wait"


class MultiArmSkills:
    """
    Bi-arm skill controller wrapping two LeRobotSkills instances.

    robot_ids[0] → left_arm, robot_ids[1] → right_arm.

    Design decisions (D-2, D-3 from plan):
    - move_to_position blocks until BOTH arms finish (or timeout).
    - When one arm is "wait", its action = current state (hold position).
    - Skill features are split: left_skill.* / right_skill.* recorded separately.
    """

    def __init__(
        self,
        left_config: str = "robot_configs/robot/so101_robot2.yaml",
        right_config: str = "robot_configs/robot/so101_robot3.yaml",
        frame: str = "world",
        movement_duration: float = 3.0,
        use_compensation: bool = True,
        use_deceleration: bool = True,
        verbose: bool = True,
        pick_offset: float = 0.015,
        recording_callback=None,
        camera=None,
        detect_model: str = None,
    ):
        """
        Args:
            left_config: Robot YAML config path for left arm (robot_ids[0]).
            right_config: Robot YAML config path for right arm (robot_ids[1]).
            recording_callback: Set to None to disable internal per-arm recording.
                Multi-arm recording is handled externally by MultiArmRecorder.
            camera: Shared RealSense camera instance (both arms share one camera).
            detect_model: VLM model for detect_objects skill.
        """
        self.verbose = verbose

        detect_kwargs = {"detect_model": detect_model} if detect_model else {}

        # Create LeRobotSkills instances with recording disabled
        # (MultiArmRecorder handles unified 12-axis recording externally)
        self.left_arm = LeRobotSkills(
            robot_config=left_config,
            frame=frame,
            movement_duration=movement_duration,
            use_compensation=use_compensation,
            use_deceleration=use_deceleration,
            verbose=verbose,
            pick_offset=pick_offset,
            recording_callback=recording_callback,
            camera=camera,
            **detect_kwargs,
        )

        self.right_arm = LeRobotSkills(
            robot_config=right_config,
            frame=frame,
            movement_duration=movement_duration,
            use_compensation=use_compensation,
            use_deceleration=use_deceleration,
            verbose=verbose,
            pick_offset=pick_offset,
            recording_callback=recording_callback,
            camera=camera,
            **detect_kwargs,
        )

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="multi_arm")

    def _log(self, message: str):
        if self.verbose:
            print(f"[MultiArm] {message}")

    # ─────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect both arms in parallel. Returns True if both succeed."""
        self._log("Connecting both arms...")
        # Re-create executor if it was shut down by a previous disconnect()
        if self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="multi_arm")
        futures = {
            self._executor.submit(self.left_arm.connect): "left_arm",
            self._executor.submit(self.right_arm.connect): "right_arm",
        }
        results = {}
        for future in as_completed(futures):
            arm_name = futures[future]
            try:
                results[arm_name] = future.result()
            except Exception as e:
                self._log(f"  {arm_name} connect failed: {e}")
                results[arm_name] = False

        ok = all(results.values())
        if ok:
            self._log("Both arms connected successfully")
        else:
            self._log(f"Connection results: {results}")
        return ok

    def disconnect(self):
        """Disconnect both arms."""
        self._log("Disconnecting both arms...")
        for arm, name in [(self.left_arm, "left_arm"), (self.right_arm, "right_arm")]:
            try:
                arm.disconnect()
            except Exception as e:
                self._log(f"  {name} disconnect error: {e}")
        self._executor.shutdown(wait=False)

    # ─────────────────────────────────────────────
    # Parallel execution helper
    # ─────────────────────────────────────────────

    def _run_both(
        self,
        left_fn,
        right_fn,
        left_args: tuple = (),
        right_args: tuple = (),
        left_kwargs: dict = None,
        right_kwargs: dict = None,
        description: str = "action",
    ) -> Dict[str, bool]:
        """
        Execute left_fn and right_fn in parallel.
        Blocks until both complete. Returns {"left": result, "right": result}.
        """
        left_kwargs = left_kwargs or {}
        right_kwargs = right_kwargs or {}

        futures = {}
        if left_fn is not None:
            futures[self._executor.submit(left_fn, *left_args, **left_kwargs)] = "left"
        if right_fn is not None:
            futures[self._executor.submit(right_fn, *right_args, **right_kwargs)] = "right"

        results = {"left": True, "right": True}
        for future in as_completed(futures):
            side = futures[future]
            try:
                results[side] = future.result()
            except Exception as e:
                self._log(f"  {side}_arm {description} failed: {e}")
                results[side] = False

        return results

    @staticmethod
    def _is_wait(value) -> bool:
        """Check if value is the 'wait' sentinel."""
        return isinstance(value, str) and value.lower() == "wait"

    # ─────────────────────────────────────────────
    # Bi-arm movement skills
    # ─────────────────────────────────────────────

    def move_to_position(
        self,
        left_arm="wait",
        right_arm="wait",
        left_duration: Optional[float] = None,
        right_duration: Optional[float] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Move arms simultaneously. Pass "wait" to skip one arm.

        Args:
            left_arm: [x,y,z] target for left arm, or "wait" to hold.
            right_arm: [x,y,z] target for right arm, or "wait" to hold.

        Returns:
            {"left": bool, "right": bool} success status.
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.move_to_position
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.move_to_position

        skip_msg = []
        if left_fn is None:
            skip_msg.append("left=wait")
        if right_fn is None:
            skip_msg.append("right=wait")
        if skip_msg:
            self._log(f"move_to_position: {', '.join(skip_msg)}")

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "position": left_arm,
                "duration": left_duration,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "position": right_arm,
                "duration": right_duration,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="move_to_position",
        )

    def pick_object(
        self,
        left_arm="wait",
        right_arm="wait",
        left_object_name: Optional[str] = None,
        right_object_name: Optional[str] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Execute pick on arms. Pass "wait" to skip one arm.

        Args:
            left_arm: Object position [x,y,z] for left arm, or "wait".
            right_arm: Object position [x,y,z] for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_pick_object
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_pick_object

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "object_position": left_arm,
                "object_name": left_object_name,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "object_position": right_arm,
                "object_name": right_object_name,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="pick_object",
        )

    def place_object(
        self,
        left_arm="wait",
        right_arm="wait",
        left_is_table: bool = True,
        right_is_table: bool = True,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Execute place on arms. Pass "wait" to skip one arm.

        Args:
            left_arm: Place position [x,y,z] for left arm, or "wait".
            right_arm: Place position [x,y,z] for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_place_object
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_place_object

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "place_position": left_arm,
                "is_table": left_is_table,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "place_position": right_arm,
                "is_table": right_is_table,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="place_object",
        )

    def move_to_pixel(
        self,
        left_arm="wait",
        right_arm="wait",
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Move arms to positions specified by normalized pixel coordinates [y, x] (0–1000).
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] normalized coordinates for left arm, or "wait".
            right_arm: [y, x] normalized coordinates for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.move_to_pixel
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.move_to_pixel

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "pixel": left_arm,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "pixel": right_arm,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="move_to_pixel",
        )

    def place_at_pixel(
        self,
        left_arm="wait",
        right_arm="wait",
        left_is_table: bool = True,
        right_is_table: bool = True,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Place at positions specified by normalized pixel coordinates [y, x] (0–1000).
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] normalized coordinates for left arm, or "wait".
            right_arm: [y, x] normalized coordinates for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_place_at_pixel
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_place_at_pixel

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "pixel": left_arm,
                "is_table": left_is_table,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "pixel": right_arm,
                "is_table": right_is_table,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="place_at_pixel",
        )

    def gripper_control(
        self,
        left_arm: str = "wait",
        right_arm: str = "wait",
        left_duration: float = 1.5,
        right_duration: float = 1.5,
        left_ratio: float = 1.0,
        right_ratio: float = 1.0,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Control grippers. Actions: "open", "close", "wait".

        Args:
            left_arm: "open", "close", or "wait" for left arm.
            right_arm: "open", "close", or "wait" for right arm.
            left_ratio / right_ratio: Open ratio for "open" action (0.0-1.0).
        """
        left_fn = None
        right_fn = None
        left_kwargs = {}
        right_kwargs = {}

        if left_arm.lower() == "open":
            left_fn = self.left_arm.gripper_open
            left_kwargs = {
                "duration": left_duration,
                "ratio": left_ratio,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            }
        elif left_arm.lower() == "close":
            left_fn = self.left_arm.gripper_close
            left_kwargs = {
                "duration": left_duration,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            }

        if right_arm.lower() == "open":
            right_fn = self.right_arm.gripper_open
            right_kwargs = {
                "duration": right_duration,
                "ratio": right_ratio,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            }
        elif right_arm.lower() == "close":
            right_fn = self.right_arm.gripper_close
            right_kwargs = {
                "duration": right_duration,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            }

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs=left_kwargs,
            right_kwargs=right_kwargs,
            description="gripper_control",
        )

    # ─────────────────────────────────────────────
    # Bimanual: synchronized single-loop control
    # ─────────────────────────────────────────────

    # Minimum duration for bimanual moves (deformable objects need slow, gentle motion)
    BIMANUAL_MIN_DURATION = 3.0

    def bimanual_move(
        self,
        left_arm,
        right_arm,
        duration: Optional[float] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
        open_gripper_during_move: bool = False,
    ) -> Dict[str, bool]:
        """
        Move both arms in a single control loop with identical progress.

        Unlike move_to_position (independent threads per arm), this runs ONE
        50Hz loop that queries both trajectories at the same elapsed time,
        guaranteeing identical progress ratio. Use when both arms hold the
        same object (e.g., towel folding, sheet stretching).

        Args:
            left_arm: [x,y,z] target for left arm.
            right_arm: [x,y,z] target for right arm.
            duration: Shared duration (auto-computed from longer arm if None,
                      minimum BIMANUAL_MIN_DURATION seconds).
        """
        import numpy as np
        from lerobot_cap.compensation import AdaptiveCompensator

        la, ra = self.left_arm, self.right_arm
        min_dur = self.BIMANUAL_MIN_DURATION

        # ── 1. Plan trajectories (each arm in own frame) ──
        def _plan(arm, position, dur):
            pos = np.array(position)
            dur = dur or max(arm.movement_duration, min_dur)
            _, current_joints, _ = arm._get_current_state()
            target_bl = arm._transform_pos_world2robot(pos)

            ik_target = target_bl.copy()
            if arm.gravity_sag is not None:
                offset = arm.gravity_sag.compute_offset(target_bl)
                if offset > 0.001:
                    ik_target[2] += offset

            planner = arm.planner
            traj, ik_info = planner.plan_to_position_multi(
                ik_target, current_joints, dur,
                num_random_samples=10, verbose=False,
                fixed_joints=[4],  # maintain wrist_roll
            )
            if not traj.ik_converged:
                raise RuntimeError(f"IK failed for position {position}")

            # Update compensator for this target z
            comp = None
            if arm.compensator:
                comp = AdaptiveCompensator.from_config(
                    config_path=arm.config.get("compensation_file"),
                    target_z=target_bl[2],
                )
                if arm.gravity_sag is not None:
                    comp.gravity_sag = arm.gravity_sag

            return traj, target_bl, comp

        left_traj, left_target, left_comp = _plan(la, left_arm, duration)
        right_traj, right_target, right_comp = _plan(ra, right_arm, duration)

        # ── 2. Synchronize duration (use longer; enforce minimum only when auto) ──
        if duration is not None:
            sync_dur = max(left_traj.duration, right_traj.duration)
        else:
            sync_dur = max(left_traj.duration, right_traj.duration, min_dur)
        self._log(f"[bimanual_move] duration={sync_dur:.2f}s "
                  f"(left={left_traj.duration:.2f}s, right={right_traj.duration:.2f}s)")

        # ── 3. Gripper interpolation setup ──
        if open_gripper_during_move:
            GRIPPER_MAX_RATIO = 0.30
            left_gripper_start = la.current_gripper_pos
            left_gripper_end = la.gripper_close_pos + (la.gripper_open_pos - la.gripper_close_pos) * GRIPPER_MAX_RATIO
            right_gripper_start = ra.current_gripper_pos
            right_gripper_end = ra.gripper_close_pos + (ra.gripper_open_pos - ra.gripper_close_pos) * GRIPPER_MAX_RATIO
            self._log(f"  [bimanual_move] Gripper will open during move")
        else:
            left_gripper_start = left_gripper_end = la.current_gripper_pos
            right_gripper_start = right_gripper_end = ra.current_gripper_pos

        # ── 4. Skill recording ──
        left_label = left_skill_description or "bimanual move (left)"
        right_label = right_skill_description or "bimanual move (right)"
        la._set_skill_recording(
            label=left_label, skill_type="bimanual_move",
            goal_joint_5=left_traj.joint_positions[-1],
            goal_gripper=left_gripper_end,
            position=list(left_arm),
            verification_question=left_verification_question,
        )
        ra._set_skill_recording(
            label=right_label, skill_type="bimanual_move",
            goal_joint_5=right_traj.joint_positions[-1],
            goal_gripper=right_gripper_end,
            position=list(right_arm),
            verification_question=right_verification_question,
        )

        # ── 4. Single control loop ──
        # Serial read is done ONCE per arm per loop iteration to avoid jitter.
        POSITION_TOLERANCE = 0.007
        MAX_TOTAL_TIME = sync_dur + 2.0
        SETTLE_TIME = 0.2

        start_time = time.time()
        left_reached_t = None
        right_reached_t = None
        left_err = right_err = float('inf')

        try:
            while True:
                elapsed = time.time() - start_time

                # ── Read state ONCE per arm ──
                left_actual_norm = la.robot.read_positions(normalize=True)
                right_actual_norm = ra.robot.read_positions(normalize=True)

                # Trajectory query (shared t_normalized → identical progress)
                if elapsed < sync_dur:
                    t_norm = elapsed / sync_dur
                    if la.use_deceleration:
                        warped = la._apply_end_deceleration(t_norm) * sync_dur
                    else:
                        warped = elapsed
                    left_q = left_traj.get_state_at_time(min(warped, left_traj.duration))
                    right_q = right_traj.get_state_at_time(min(warped, right_traj.duration))
                    phase = "Traj"
                else:
                    left_q = left_traj.joint_positions[-1]
                    right_q = right_traj.joint_positions[-1]
                    phase = "Hold"

                # Normalize
                left_norm = la._radians_to_normalized(left_q)
                right_norm = ra._radians_to_normalized(right_q)

                # Compensation (reuse already-read state)
                if la.use_compensation and left_comp:
                    left_norm = left_comp.compensate(left_actual_norm[:5], left_norm)
                if ra.use_compensation and right_comp:
                    right_norm = right_comp.compensate(right_actual_norm[:5], right_norm)

                # Gripper interpolation (smooth open during move)
                if elapsed < sync_dur:
                    grip_alpha = elapsed / sync_dur
                else:
                    grip_alpha = 1.0
                left_grip = left_gripper_start + grip_alpha * (left_gripper_end - left_gripper_start)
                right_grip = right_gripper_start + grip_alpha * (right_gripper_end - right_gripper_start)
                la.current_gripper_pos = left_grip
                ra.current_gripper_pos = right_grip

                # Clip and send
                left_norm = np.clip(left_norm, -99.0, 99.0)
                right_norm = np.clip(right_norm, -99.0, 99.0)
                left_full = np.concatenate([left_norm, [left_grip]])
                right_full = np.concatenate([right_norm, [right_grip]])

                la.robot.write_positions(left_full, normalize=True)
                ra.robot.write_positions(right_full, normalize=True)

                # Recording callbacks (reuse already-read state)
                if la.recording_callback and phase == "Traj":
                    try:
                        la.recording_callback(left_actual_norm.copy(), left_full.copy())
                    except Exception:
                        pass
                if ra.recording_callback and phase == "Traj":
                    try:
                        ra.recording_callback(right_actual_norm.copy(), right_full.copy())
                    except Exception:
                        pass

                # Error check via FK (reuse already-read state)
                if la.kinematics and la.calibration_limits:
                    left_rad = la.calibration_limits.normalized_to_radians(left_actual_norm[:5])
                    left_ee = la.kinematics.get_ee_position(left_rad)
                    left_err = np.linalg.norm(left_target - left_ee)
                if ra.kinematics and ra.calibration_limits:
                    right_rad = ra.calibration_limits.normalized_to_radians(right_actual_norm[:5])
                    right_ee = ra.kinematics.get_ee_position(right_rad)
                    right_err = np.linalg.norm(right_target - right_ee)

                # Progress bar
                if self.verbose:
                    prog = min(elapsed / sync_dur, 1.0)
                    filled = int(30 * prog)
                    bar = "=" * filled + "-" * (30 - filled)
                    print(f"\r  [{bar}] {phase} L:{left_err*1000:5.1f}mm R:{right_err*1000:5.1f}mm", end="", flush=True)

                # Settle check (both arms must settle)
                now = time.time()
                if left_err < POSITION_TOLERANCE:
                    left_reached_t = left_reached_t or now
                else:
                    left_reached_t = None
                if right_err < POSITION_TOLERANCE:
                    right_reached_t = right_reached_t or now
                else:
                    right_reached_t = None

                if (left_reached_t and right_reached_t and
                        now - left_reached_t > SETTLE_TIME and
                        now - right_reached_t > SETTLE_TIME):
                    break

                if elapsed > MAX_TOTAL_TIME:
                    if self.verbose:
                        print(f"\n  Timeout after {MAX_TOTAL_TIME:.1f}s")
                    break

                time.sleep(0.02)  # 50Hz

        finally:
            la._clear_skill_recording()
            ra._clear_skill_recording()

        if self.verbose:
            print(f"\r  [{'=' * 30}] Done (L:{left_err*1000:.1f}mm R:{right_err*1000:.1f}mm)    ")

        left_ok = left_err < POSITION_TOLERANCE * 4
        right_ok = right_err < POSITION_TOLERANCE * 4
        return {"left": left_ok, "right": right_ok}

    # Release height offset: arc ends this far above end_z so place_object
    # can gently descend and release without the folded fabric bunching up.
    FOLD_RELEASE_HEIGHT = 0.05  # 5cm above target

    # Duration per waypoint in bimanual_fold (shorter than BIMANUAL_MIN_DURATION
    # because each waypoint segment is a very short distance)
    FOLD_WAYPOINT_DURATION = 2.0

    # Torque limit for fold compliance (0-1000, lower = more compliant)
    FOLD_TORQUE_LIMIT = 400
    DEFAULT_TORQUE_LIMIT = 1000

    def bimanual_fold(
        self,
        left_start,
        right_start,
        left_end,
        right_end,
        arc_height: float = 0.20,
        num_points: int = 8,
        compliant: bool = True,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Fold motion: move both arms along an arc trajectory from start to end.

        Generates waypoints on a semicircular arc and executes them as a
        sequence of bimanual_move calls. The arc ends slightly above the
        target (FOLD_RELEASE_HEIGHT) to avoid fabric compression — call
        place_object afterwards for the final descent and release.

        When compliant=True, torque limits are lowered during fold so
        the arms yield when fabric becomes taut instead of fighting it.

        Args:
            left_start: [x,y,z] grasp position for left arm.
            right_start: [x,y,z] grasp position for right arm.
            left_end: [x,y,z] fold target position for left arm.
            right_end: [x,y,z] fold target position for right arm.
            arc_height: Peak height of the arc above start z (meters, default 0.20).
            num_points: Number of waypoints along the arc (default 8).
            compliant: Lower torque limit during fold for compliance (default True).
        """
        import numpy as np

        left_start = np.array(left_start, dtype=np.float64)
        right_start = np.array(right_start, dtype=np.float64)
        left_end = np.array(left_end, dtype=np.float64)
        right_end = np.array(right_end, dtype=np.float64)

        base_z = max(left_start[2], right_start[2])
        release_z = max(left_end[2], right_end[2]) + self.FOLD_RELEASE_HEIGHT

        self._log(f"[bimanual_fold] arc_height={arc_height:.2f}m, release_z={release_z:.3f}m, "
                  f"{num_points} waypoints, compliant={compliant}")

        # Lower torque for compliance during fold
        if compliant and self.left_arm.robot and self.right_arm.robot:
            # Arm joints (1-5) get lower torque; gripper (6) keeps full torque to hold object
            arm_motor_ids = list(range(1, 6))  # motors 1-5 (arm only)
            self.left_arm.robot.set_torque_limit(self.FOLD_TORQUE_LIMIT, motor_ids=arm_motor_ids)
            self.right_arm.robot.set_torque_limit(self.FOLD_TORQUE_LIMIT, motor_ids=arm_motor_ids)
            self._log(f"  Compliance ON: arm torque={self.FOLD_TORQUE_LIMIT}/1000, gripper=full")

        left_desc = left_skill_description or "fold (left)"
        right_desc = right_skill_description or "fold (right)"

        # Each arm interpolates independently in its own coordinate frame.
        # Same t ratio guarantees synchronized progress.
        # (Cannot mix coordinates — each arm has its own base_link frame.)

        last_result = {"left": True, "right": True}
        try:
            for i in range(num_points):
                t = (i + 1) / num_points  # 0 → 1
                theta = np.pi * t         # 0 → π

                # x, y: linear interpolation per arm (each in own frame)
                left_xy = left_start[:2] + (left_end[:2] - left_start[:2]) * t
                right_xy = right_start[:2] + (right_end[:2] - right_start[:2]) * t

                # z: sin arc, but floor at release_z (never descend below release height)
                z_arc = base_z + arc_height * np.sin(theta)
                z = max(z_arc, release_z)

                left_wp = [float(left_xy[0]), float(left_xy[1]), float(z)]
                right_wp = [float(right_xy[0]), float(right_xy[1]), float(z)]

                step_desc = f"({i+1}/{num_points})"
                last_result = self.bimanual_move(
                    left_arm=left_wp,
                    right_arm=right_wp,
                    duration=self.FOLD_WAYPOINT_DURATION,
                    left_skill_description=f"{left_desc} {step_desc}",
                    right_skill_description=f"{right_desc} {step_desc}",
                left_verification_question=left_verification_question,
                    right_verification_question=right_verification_question,
                )
        finally:
            # Restore full torque after fold (always, even on error)
            if compliant and self.left_arm.robot and self.right_arm.robot:
                arm_motor_ids = list(range(1, 6))
                self.left_arm.robot.set_torque_limit(self.DEFAULT_TORQUE_LIMIT, motor_ids=arm_motor_ids)
                self.right_arm.robot.set_torque_limit(self.DEFAULT_TORQUE_LIMIT, motor_ids=arm_motor_ids)
                self._log(f"  Compliance OFF: torque restored to {self.DEFAULT_TORQUE_LIMIT}/1000")

        return last_result

    def bimanual_pick_object(
        self,
        left_arm,
        right_arm,
        object_name: Optional[str] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Bimanual pick: synchronized descend + grip. (Called from approach position.)

        Same role as execute_pick_object but for two arms holding the same object.
        Caller must open grippers and move to approach height BEFORE calling this.

        Args:
            left_arm: [x,y,z] grasp position for left arm.
            right_arm: [x,y,z] grasp position for right arm.
            object_name: Name of the object being picked.
        """
        import numpy as np
        left_pos = np.array(left_arm)
        right_pos = np.array(right_arm)
        name = object_name or "object"

        self._log(f"[bimanual_pick_object] {name}")

        # 1. Descend to grasp points (synchronized)
        pick_z_left = max(left_pos[2] - self.left_arm.pick_offset, 0.0)
        pick_z_right = max(right_pos[2] - self.right_arm.pick_offset, 0.0)
        self.bimanual_move(
            left_arm=[left_pos[0], left_pos[1], pick_z_left],
            right_arm=[right_pos[0], right_pos[1], pick_z_right],
            left_skill_description=left_skill_description or f"Descend to grasp {name} (left)",
            left_verification_question=left_verification_question or f"Is left arm at grasp position?",
            right_skill_description=right_skill_description or f"Descend to grasp {name} (right)",
            right_verification_question=right_verification_question or f"Is right arm at grasp position?",
        )

        # 2. Close grippers
        self.gripper_control(
            left_arm="close", right_arm="close",
            left_skill_description=f"Grasp {name} (left)",
            left_verification_question=f"Is {name} grasped by left arm?",
            right_skill_description=f"Grasp {name} (right)",
            right_verification_question=f"Is {name} grasped by right arm?",
        )

        # 3. Save pitch for place
        self.left_arm._saved_pitch = self.left_arm.kinematics.get_gripper_pitch(
            self.left_arm._get_current_state()[1])
        self.right_arm._saved_pitch = self.right_arm.kinematics.get_gripper_pitch(
            self.right_arm._get_current_state()[1])

        self._log(f"[bimanual_pick_object] Complete")
        return {"left": True, "right": True}

    def bimanual_place_object(
        self,
        left_arm,
        right_arm,
        object_name: Optional[str] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Bimanual place: synchronized descend + release. (Called from approach position.)

        Same role as execute_place_object but for two arms holding the same object.
        Caller must move to retract height AFTER calling this.

        Args:
            left_arm: [x,y,z] place position for left arm.
            right_arm: [x,y,z] place position for right arm.
            object_name: Name of the object being placed.
        """
        import numpy as np
        left_pos = np.array(left_arm)
        right_pos = np.array(right_arm)
        name = object_name or "object"

        self._log(f"[bimanual_place_object] {name}")

        # 1. Descend to place points (synchronized)
        place_z_left = max(left_pos[2] - self.left_arm.pick_offset, 0.005)
        place_z_right = max(right_pos[2] - self.right_arm.pick_offset, 0.005)
        self.bimanual_move(
            left_arm=[left_pos[0], left_pos[1], place_z_left],
            right_arm=[right_pos[0], right_pos[1], place_z_right],
            left_skill_description=left_skill_description or f"Lower {name} to place position (left)",
            left_verification_question=left_verification_question or f"Is left side at place position?",
            right_skill_description=right_skill_description or f"Lower {name} to place position (right)",
            right_verification_question=right_verification_question or f"Is right side at place position?",
        )

        # 2. Lift while opening gripper (prevents pushing deformable objects)
        RELEASE_LIFT = 0.03  # 3cm lift during gripper open
        self.bimanual_move(
            left_arm=[left_pos[0], left_pos[1], place_z_left + RELEASE_LIFT],
            right_arm=[right_pos[0], right_pos[1], place_z_right + RELEASE_LIFT],
            open_gripper_during_move=True,
            left_skill_description=f"Release {name} while lifting (left)",
            left_verification_question=f"Is {name} released by left arm?",
            right_skill_description=f"Release {name} while lifting (right)",
            right_verification_question=f"Is {name} released by right arm?",
        )

        # 3. Clear saved state
        self.left_arm._pick_z = None
        self.left_arm._saved_pitch = None
        self.right_arm._pick_z = None
        self.right_arm._saved_pitch = None

        self._log(f"[bimanual_place_object] Complete")
        return {"left": True, "right": True}

    # ─────────────────────────────────────────────
    # Convenience: both arms to known poses
    # ─────────────────────────────────────────────

    def move_to_initial_state(self) -> Dict[str, bool]:
        """Move both arms to their initial (home) positions simultaneously."""
        self._log("Moving both arms to initial state...")
        return self._run_both(
            left_fn=self.left_arm.move_to_initial_state,
            right_fn=self.right_arm.move_to_initial_state,
            description="move_to_initial_state",
        )

    def move_to_free_state(self) -> Dict[str, bool]:
        """Move both arms to their free (parking) positions simultaneously."""
        self._log("Moving both arms to free state...")
        return self._run_both(
            left_fn=self.left_arm.move_to_free_state,
            right_fn=self.right_arm.move_to_free_state,
            description="move_to_free_state",
        )

    # ─────────────────────────────────────────────
    # Subtask labeling (for recording)
    # ─────────────────────────────────────────────

    def set_subtask(self, description: str) -> None:
        """Set sub-task label for recording. Delegates to left_arm's set_subtask."""
        self.left_arm.set_subtask(description)

    def clear_subtask(self) -> None:
        """Clear sub-task label. Delegates to left_arm's clear_subtask."""
        self.left_arm.clear_subtask()

    def detect_objects(self, queries: list, timeout: float = 5.0, point_labels: dict = None) -> dict:
        """Re-detect objects and return positions in both arm frames.

        Uses left_arm to run detection (shared camera), then converts
        pixel coordinates through each arm's pix2robot calibration.

        Args:
            queries: 검출할 객체 이름 리스트
            timeout: 검출 타임아웃
            point_labels: 물체별 포인트 라벨 딕셔너리 (Turn 2 라벨 재사용)
                         {"red block": ["grasp center", "top surface center"], ...}

        Returns:
            {"left_arm": {obj: {"position": [...], ...}, ...},
             "right_arm": {obj: {"position": [...], ...}, ...}}
        """
        # 저장된 point_labels가 있으면 자동 사용 (Turn 2 라벨 재사용)
        if point_labels is None:
            point_labels = getattr(self, '_point_labels', None)

        # Run detection via left_arm (camera + VLM)
        raw = self.left_arm.detect_objects(queries, timeout=timeout, point_labels=point_labels)

        # Build dual-arm result by re-converting pixel coords per arm
        dual = {"left_arm": {}, "right_arm": {}}
        for obj_name, info in raw.items():
            if info is None:
                dual["left_arm"][obj_name] = None
                dual["right_arm"][obj_name] = None
                continue

            pixel = info.get("pixel")
            depth_m = info.get("depth_m")

            pixel_points = info.get("_pixel_points", {})

            for arm_key, arm in [("left_arm", self.left_arm), ("right_arm", self.right_arm)]:
                if arm.pix2robot is not None and pixel is not None:
                    px, py = int(pixel[0]), int(pixel[1])
                    pos = arm.pix2robot.pixel_to_robot(px, py, depth_m=depth_m)
                    # 각 포인트를 해당 포인트의 pixel 좌표로 변환
                    arm_points = {}
                    for pt_label, pt_pos in info.get("points", {}).items():
                        pt_pixel = pixel_points.get(pt_label)
                        if pt_pixel is not None:
                            arm_points[pt_label] = arm.pix2robot.pixel_to_robot(
                                int(pt_pixel[0]), int(pt_pixel[1]), depth_m=depth_m)
                        else:
                            arm_points[pt_label] = arm.pix2robot.pixel_to_robot(px, py, depth_m=depth_m)
                    dual[arm_key][obj_name] = {
                        "position": pos,
                        "points": arm_points,
                        "pixel": pixel,
                        "bbox_px": info.get("bbox_px"),
                    }
                else:
                    dual[arm_key][obj_name] = info.copy()

        return dual

    # ─────────────────────────────────────────────
    # State access helpers (for recording)
    # ─────────────────────────────────────────────

    def read_left_positions(self):
        """Read current left arm joint positions (6-axis normalized)."""
        if self.left_arm.robot:
            return self.left_arm.robot.read_positions()
        return None

    def read_right_positions(self):
        """Read current right arm joint positions (6-axis normalized)."""
        if self.right_arm.robot:
            return self.right_arm.robot.read_positions()
        return None

    def get_left_gripper_pos(self) -> float:
        """Get left arm current gripper target position."""
        return self.left_arm.current_gripper_pos

    def get_right_gripper_pos(self) -> float:
        """Get right arm current gripper target position."""
        return self.right_arm.current_gripper_pos
