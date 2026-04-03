"""
Kinematics Engine

URDF-based forward/inverse kinematics using Pinocchio.
Based on LeRobot's kinematics implementation.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import math

try:
    import pinocchio as pin
except ImportError:
    raise ImportError(
        "Pinocchio is required for kinematics.\n"
        "Install: conda install -c conda-forge pinocchio"
    )


class KinematicsEngine:
    """
    URDF-based kinematics engine using Pinocchio.

    Provides forward kinematics (FK) and inverse kinematics (IK) computations.

    Args:
        urdf_path: Path to URDF file
        end_effector_frame: Name of the end-effector frame in URDF
        joint_names: List of joint names to control (optional, auto-detect if None)
        tcp_offset: TCP offset from end_effector_frame in local frame [x, y, z] meters.
                   If provided, creates a virtual TCP frame and uses it for FK/IK.
    """

    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str = "gripper_frame_link",
        joint_names: Optional[List[str]] = None,
        tcp_offset: Optional[List[float]] = None,
    ):
        self.urdf_path = Path(urdf_path)
        self.end_effector_frame = end_effector_frame
        self.tcp_offset = tcp_offset

        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        # Load URDF model
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        # Get base end-effector frame ID
        if not self.model.existFrame(end_effector_frame):
            available_frames = [f.name for f in self.model.frames]
            raise ValueError(
                f"Frame '{end_effector_frame}' not found in URDF.\n"
                f"Available frames: {available_frames}"
            )
        self.base_ee_frame_id = self.model.getFrameId(end_effector_frame)

        # Add TCP frame if offset is provided
        if tcp_offset is not None:
            self._add_tcp_frame(tcp_offset)
            self.ee_frame_id = self.model.getFrameId("tcp_link")
            self.using_tcp_frame = True
        else:
            self.ee_frame_id = self.base_ee_frame_id
            self.using_tcp_frame = False

        # Get joint names
        if joint_names is None:
            # Auto-detect: exclude universe and fixed joints
            self.joint_names = [
                self.model.names[i]
                for i in range(1, self.model.njoints)
                if self.model.joints[i].nq > 0
            ]
        else:
            self.joint_names = joint_names

        # Get joint indices
        self.joint_ids = []
        for name in self.joint_names:
            if self.model.existJointName(name):
                self.joint_ids.append(self.model.getJointId(name))
            else:
                print(f"Warning: Joint '{name}' not found in URDF")

        self.num_joints = len(self.joint_ids)

        # Neutral configuration
        self.q_neutral = pin.neutral(self.model)

        # Compute reach limits from URDF link lengths
        self._compute_reach_limits()

        print(f"KinematicsEngine initialized:")
        print(f"  URDF: {self.urdf_path}")
        print(f"  Base EE frame: {end_effector_frame}")
        if self.using_tcp_frame:
            print(f"  TCP frame: tcp_link (offset: [{tcp_offset[0]*1000:.1f}, {tcp_offset[1]*1000:.1f}, {tcp_offset[2]*1000:.1f}] mm)")
            print(f"  Active EE frame: tcp_link")
        else:
            print(f"  Active EE frame: {end_effector_frame}")
        print(f"  Joints ({self.num_joints}): {self.joint_names}")
        print(f"  Reach limits: [{self.min_reach:.3f}m, {self.max_reach:.3f}m]")

    def _compute_reach_limits(self) -> None:
        """
        Compute min/max reach limits from URDF link lengths.

        These limits define the geometric workspace boundary based on
        the robot's kinematic chain. Computed once during initialization.

        For SO-101 robot:
        - Upper arm (shoulder to elbow): ~0.113m
        - Lower arm (elbow to wrist): ~0.135m
        - Wrist + gripper: ~0.159m

        max_reach = sum of all link lengths (fully extended)
        min_reach = conservative minimum based on arm folding
        """
        # Extract link lengths from URDF joint origins
        # These are the distances between consecutive joints
        link_lengths = []

        # Parse joint origins to get link lengths
        for joint_name in ['shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']:
            if self.model.existJointName(joint_name):
                joint_id = self.model.getJointId(joint_name)
                # Get joint placement (parent to joint transform)
                placement = self.model.jointPlacements[joint_id]
                # Compute distance from parent
                distance = np.linalg.norm(placement.translation)
                if distance > 0.01:  # Filter out very small offsets
                    link_lengths.append(distance)

        # Add gripper frame offset
        if self.model.existFrame(self.end_effector_frame):
            frame_id = self.model.getFrameId(self.end_effector_frame)
            frame = self.model.frames[frame_id]
            gripper_distance = np.linalg.norm(frame.placement.translation)
            if gripper_distance > 0.01:
                link_lengths.append(gripper_distance)

        # Compute reach limits
        if link_lengths:
            self.max_reach = sum(link_lengths)
            # min_reach: when arm folds back, approximate as 10% of max
            # For safety, use a conservative lower bound
            self.min_reach = 0.05  # 5cm minimum reach
        else:
            # Fallback values for SO-101 based on URDF analysis
            # Upper arm: 0.113m, Lower arm: 0.135m, Wrist+gripper: 0.159m
            self.max_reach = 0.113 + 0.135 + 0.159  # ~0.407m
            self.min_reach = 0.05

        # Store link lengths for debugging
        self._link_lengths = link_lengths

    def _add_tcp_frame(self, tcp_offset: List[float]) -> None:
        """
        Add a virtual TCP (Tool Center Point) frame to the model.

        TCP (Tool Center Point):
        ═══════════════════════
        로봇 공학에서 "실제 작업이 수행되는 지점"을 정의하는 표준 개념.
        - End-effector frame (gripper_frame_link) = 그리퍼 메커니즘의 기하학적 중심
        - TCP frame (tcp_link) = 실제로 물체와 접촉/작업하는 지점

        예시: tcp_offset = [0, 0, -0.04]
        ─────────────────────────────────
            gripper_frame_link
                   │
                   │ -4cm (Z방향)
                   ↓
               tcp_link  ← 물체를 잡는 실제 지점

        Creates a new frame called 'tcp_link' attached to the base end-effector frame
        with the specified offset. This allows IK/FK to target a point offset from
        the original end-effector without modifying the URDF.

        Args:
            tcp_offset: Offset from base_ee_frame in local coordinates [x, y, z] meters
                       e.g., [0, 0, -0.04] means 4cm below gripper_frame_link in its Z direction
        """
        # Get parent frame info
        parent_frame = self.model.frames[self.base_ee_frame_id]

        # Create TCP placement relative to parent frame
        # SE3(rotation, translation) - no rotation, just translation offset
        tcp_placement_local = pin.SE3(
            np.eye(3),  # Same orientation as parent
            np.array(tcp_offset)  # Offset in parent's local frame
        )

        # Compute absolute placement: parent_placement * local_offset
        tcp_placement_absolute = parent_frame.placement * tcp_placement_local

        # Create new frame
        tcp_frame = pin.Frame(
            name="tcp_link",
            parent_joint=parent_frame.parentJoint,
            parent_frame=self.base_ee_frame_id,
            placement=tcp_placement_absolute,
            type=pin.FrameType.OP_FRAME,  # Operational frame
        )

        # Add frame to model
        self.model.addFrame(tcp_frame)

        # Recreate data after model modification
        self.data = self.model.createData()

    def is_position_reachable(self, position: np.ndarray, margin: float = 0.01) -> bool:
        """
        Check if a position is within the robot's geometric reach.

        This is a fast O(1) check based on pre-computed reach limits.
        Does NOT check pitch constraints or joint limits - only geometric reach.

        Args:
            position: Target position [x, y, z] in meters (robot base frame)
            margin: Safety margin in meters (default 1cm)

        Returns:
            True if position is within reachable workspace
        """
        x, y, z = position

        # Distance from robot base to target (in XY plane for horizontal reach)
        # For a vertical articulated arm, horizontal reach matters most
        horizontal_distance = math.sqrt(x*x + y*y)

        # Also consider 3D distance for positions above/below
        distance_3d = math.sqrt(x*x + y*y + z*z)

        # Check horizontal reach (primary constraint)
        max_horizontal = self.max_reach - margin
        min_horizontal = self.min_reach + margin

        # Position is reachable if within horizontal reach ring
        # and 3D distance doesn't exceed max reach
        is_within_horizontal = min_horizontal <= horizontal_distance <= max_horizontal
        is_within_3d = distance_3d <= (self.max_reach - margin)

        return is_within_horizontal and is_within_3d

    def forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics.

        Args:
            joint_positions: Joint positions in radians [num_joints]

        Returns:
            Tuple of (position [3], orientation_matrix [3x3])
        """
        # Create full configuration vector
        q = self._joint_positions_to_q(joint_positions)

        # Compute FK
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get end-effector pose
        ee_placement = self.data.oMf[self.ee_frame_id]

        position = ee_placement.translation.copy()
        rotation = ee_placement.rotation.copy()

        return position, rotation

    def get_ee_pose(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get end-effector pose as 4x4 transformation matrix.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            4x4 homogeneous transformation matrix
        """
        position, rotation = self.forward_kinematics(joint_positions)

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position

        return transform

    def get_ee_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get end-effector position only.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            Position [x, y, z] in meters
        """
        position, _ = self.forward_kinematics(joint_positions)
        return position

    def get_gripper_pitch(self, joint_positions: np.ndarray) -> float:
        """
        Calculate gripper pitch angle in world frame.

        Pitch is the angle between the gripper pointing direction (Z-axis)
        and the horizontal plane.
        - 0° = gripper pointing horizontally
        - -90° (-π/2) = gripper pointing straight down
        - +90° (+π/2) = gripper pointing straight up

        Args:
            joint_positions: Joint positions in radians

        Returns:
            Pitch angle in radians
        """
        _, rotation = self.forward_kinematics(joint_positions)

        # Gripper Z-axis in world frame (pointing direction)
        gripper_z = rotation[:, 2]

        # Pitch = arcsin(z-component of gripper Z-axis)
        # When gripper points down, gripper_z[2] = -1, pitch = -π/2
        pitch = np.arcsin(np.clip(gripper_z[2], -1.0, 1.0))

        return pitch

    def _compute_pitch_jacobian(
        self,
        q: np.ndarray,
        delta: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute the Jacobian of gripper pitch w.r.t. joint velocities.

        Uses numerical differentiation for simplicity and stability.
        Returns Jacobian with model.nv columns to match position Jacobian structure.

        Args:
            q: Pinocchio configuration vector
            delta: Finite difference step size

        Returns:
            1×model.nv Jacobian vector (matching position Jacobian structure)
        """
        joint_positions = self._q_to_joint_positions(q)
        pitch_0 = self.get_gripper_pitch(joint_positions)

        # Create Jacobian with model.nv columns to match position Jacobian
        J_pitch = np.zeros(self.model.nv)

        # Compute derivative for each controllable joint
        for i, joint_id in enumerate(self.joint_ids):
            joint_positions_delta = joint_positions.copy()
            joint_positions_delta[i] += delta
            pitch_i = self.get_gripper_pitch(joint_positions_delta)

            # Map to the correct index in model velocity space
            idx_v = self.model.joints[joint_id].idx_v
            J_pitch[idx_v] = (pitch_i - pitch_0) / delta

        return J_pitch

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
        damping: float = 1e-2,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics using damped least squares.

        Args:
            target_position: Target position [x, y, z] in meters
            target_orientation: Target orientation as 3x3 rotation matrix (optional)
            initial_guess: Initial joint positions (optional, use neutral if None)
            max_iterations: Maximum number of iterations
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)
            damping: Damping factor for DLS

        Returns:
            Tuple of (joint_positions, success)
        """
        # Initial guess
        if initial_guess is not None:
            q = self._joint_positions_to_q(initial_guess)
        else:
            q = self.q_neutral.copy()

        # Target pose
        if target_orientation is not None:
            target_se3 = pin.SE3(target_orientation, target_position)
        else:
            # Position only - use current orientation
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_rotation = self.data.oMf[self.ee_frame_id].rotation
            target_se3 = pin.SE3(current_rotation, target_position)

        # Iterative IK
        for iteration in range(max_iterations):
            # Compute current FK
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_se3 = self.data.oMf[self.ee_frame_id]

            # Compute error in LOCAL frame (consistent with Jacobian)
            # error = log(current^(-1) * target) gives the twist to go from current to target
            error_se3 = current_se3.actInv(target_se3)
            error = pin.log(error_se3).vector

            # Check convergence
            position_error = np.linalg.norm(error[:3])
            orientation_error = np.linalg.norm(error[3:])

            if position_error < position_tolerance:
                if target_orientation is None or orientation_error < orientation_tolerance:
                    return self._q_to_joint_positions(q), True

            # IMPORTANT: computeJointJacobians must be called before getFrameJacobian
            pin.computeJointJacobians(self.model, self.data, q)

            # Compute Jacobian in LOCAL frame (same frame as error)
            pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )
            J = pin.getFrameJacobian(
                self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )

            # Damped least squares
            JJt = J @ J.T
            JJt_damped = JJt + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt_damped, error)

            # Update
            q = pin.integrate(self.model, q, dq)

        # Did not converge
        return self._q_to_joint_positions(q), False

    def inverse_kinematics_position_only(
        self,
        target_position: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        damping: float = 1e-2,
        fixed_joints: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute IK for position only (ignore orientation).

        Faster and more stable for position-only tasks.

        Args:
            target_position: Target position [x, y, z] in meters
            initial_guess: Initial joint positions
            max_iterations: Maximum iterations
            tolerance: Position error tolerance
            damping: Damping factor
            fixed_joints: List of joint indices to keep fixed (0-indexed)
                         e.g., [4] to fix wrist_roll

        Returns:
            Tuple of (joint_positions, success)
        """
        if initial_guess is not None:
            q = self._joint_positions_to_q(initial_guess)
        else:
            q = self.q_neutral.copy()

        # Create mask for active (non-fixed) joints
        num_joints = len(self.joint_ids)
        if fixed_joints is None:
            active_mask = np.ones(num_joints, dtype=bool)
        else:
            active_mask = np.ones(num_joints, dtype=bool)
            for idx in fixed_joints:
                if 0 <= idx < num_joints:
                    active_mask[idx] = False

        active_indices = np.where(active_mask)[0]

        for iteration in range(max_iterations):
            # FK
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_position = self.data.oMf[self.ee_frame_id].translation

            # Position error
            error = target_position - current_position
            error_norm = np.linalg.norm(error)

            if error_norm < tolerance:
                return self._q_to_joint_positions(q), True

            # IMPORTANT: computeJointJacobians must be called before getFrameJacobian
            pin.computeJointJacobians(self.model, self.data, q)

            # Position Jacobian (only translation part)
            pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_full = pin.getFrameJacobian(
                self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J_full[:3, :]  # Position only

            # Use only active joints for IK
            J_active = J[:, active_indices]

            # DLS with active joints only
            JJt = J_active @ J_active.T
            JJt_damped = JJt + damping**2 * np.eye(3)
            dq_active = J_active.T @ np.linalg.solve(JJt_damped, error)

            # Expand to full dq (zeros for fixed joints)
            dq = np.zeros(self.model.nv)
            for i, active_idx in enumerate(active_indices):
                joint_id = self.joint_ids[active_idx]
                idx_v = self.model.joints[joint_id].idx_v
                dq[idx_v] = dq_active[i]

            # Update
            q = pin.integrate(self.model, q, dq)

        return self._q_to_joint_positions(q), False

    def inverse_kinematics_position_with_pitch(
        self,
        target_position: np.ndarray,
        target_pitch: float,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        pitch_tolerance: float = 0.05,  # ~3 degrees
        damping: float = 1e-2,
        fixed_joints: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute IK for position + pitch constraint.

        Solves for joint positions that achieve both:
        1. Target end-effector position [x, y, z]
        2. Target gripper pitch angle

        Uses 4D error: [position_error (3D), pitch_error (1D)]
        With 4×n Jacobian: [position_jacobian (3×n), pitch_jacobian (1×n)]

        Args:
            target_position: Target position [x, y, z] in meters
            target_pitch: Target gripper pitch in radians (-π/2 = pointing down)
            initial_guess: Initial joint positions
            max_iterations: Maximum iterations
            position_tolerance: Position error tolerance (meters)
            pitch_tolerance: Pitch error tolerance (radians)
            damping: Damping factor for DLS
            fixed_joints: List of joint indices to keep fixed (0-indexed)

        Returns:
            Tuple of (joint_positions, success)
        """
        if initial_guess is not None:
            q = self._joint_positions_to_q(initial_guess)
        else:
            q = self.q_neutral.copy()

        # Create mask for active (non-fixed) joints
        num_joints = len(self.joint_ids)
        if fixed_joints is None:
            active_mask = np.ones(num_joints, dtype=bool)
        else:
            active_mask = np.ones(num_joints, dtype=bool)
            for idx in fixed_joints:
                if 0 <= idx < num_joints:
                    active_mask[idx] = False

        active_indices = np.where(active_mask)[0]

        for iteration in range(max_iterations):
            # FK
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_position = self.data.oMf[self.ee_frame_id].translation
            current_pitch = self.get_gripper_pitch(self._q_to_joint_positions(q))

            # 4D error: [position_error (3), pitch_error (1)]
            position_error = target_position - current_position
            pitch_error = target_pitch - current_pitch
            error = np.concatenate([position_error, [pitch_error]])

            # Check convergence
            position_error_norm = np.linalg.norm(position_error)
            pitch_error_abs = abs(pitch_error)

            if position_error_norm < position_tolerance and pitch_error_abs < pitch_tolerance:
                return self._q_to_joint_positions(q), True

            # Compute position Jacobian (3×n)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_full = pin.getFrameJacobian(
                self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_position = J_full[:3, :]  # 3×n

            # Compute pitch Jacobian (1×n) using numerical differentiation
            J_pitch = self._compute_pitch_jacobian(q)  # 1×n (as 1D array)

            # Combine into 4×n Jacobian
            J_combined = np.vstack([J_position, J_pitch.reshape(1, -1)])  # 4×n

            # Use only active joints for IK
            J_active = J_combined[:, active_indices]  # 4×num_active

            # DLS with active joints only
            JJt = J_active @ J_active.T  # 4×4
            JJt_damped = JJt + damping**2 * np.eye(4)
            dq_active = J_active.T @ np.linalg.solve(JJt_damped, error)

            # Expand to full dq (zeros for fixed joints)
            dq = np.zeros(self.model.nv)
            for i, active_idx in enumerate(active_indices):
                joint_id = self.joint_ids[active_idx]
                idx_v = self.model.joints[joint_id].idx_v
                dq[idx_v] = dq_active[i]

            # Update
            q = pin.integrate(self.model, q, dq)

        # Did not converge
        return self._q_to_joint_positions(q), False

    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute the geometric Jacobian at current configuration.

        Args:
            joint_positions: Joint positions in radians

        Returns:
            6 x num_joints Jacobian matrix
        """
        q = self._joint_positions_to_q(joint_positions)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

        pin.computeFrameJacobian(
            self.model, self.data, q,
            self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        return J

    def _joint_positions_to_q(self, joint_positions: np.ndarray) -> np.ndarray:
        """Convert joint positions array to Pinocchio configuration vector."""
        q = self.q_neutral.copy()

        for i, joint_id in enumerate(self.joint_ids):
            if i < len(joint_positions):
                idx_q = self.model.joints[joint_id].idx_q
                q[idx_q] = joint_positions[i]

        return q

    def _q_to_joint_positions(self, q: np.ndarray) -> np.ndarray:
        """Convert Pinocchio configuration vector to joint positions array."""
        joint_positions = np.zeros(len(self.joint_ids))

        for i, joint_id in enumerate(self.joint_ids):
            idx_q = self.model.joints[joint_id].idx_q
            joint_positions[i] = q[idx_q]

        return joint_positions

    @property
    def joint_limits_lower(self) -> np.ndarray:
        """Get lower joint limits."""
        limits = []
        for joint_id in self.joint_ids:
            idx_q = self.model.joints[joint_id].idx_q
            limits.append(self.model.lowerPositionLimit[idx_q])
        return np.array(limits)

    @property
    def joint_limits_upper(self) -> np.ndarray:
        """Get upper joint limits."""
        limits = []
        for joint_id in self.joint_ids:
            idx_q = self.model.joints[joint_id].idx_q
            limits.append(self.model.upperPositionLimit[idx_q])
        return np.array(limits)

    def is_within_joint_limits(
        self,
        joint_positions: np.ndarray,
        margin: float = 0.01,
    ) -> bool:
        """
        Check if joint positions are within URDF limits.

        Args:
            joint_positions: Joint positions in radians
            margin: Safety margin in radians (default 0.01 rad ≈ 0.57°)

        Returns:
            True if all joints are within limits
        """
        lower = self.joint_limits_lower + margin
        upper = self.joint_limits_upper - margin
        return np.all(joint_positions >= lower) and np.all(joint_positions <= upper)

    def inverse_kinematics_with_orientation_multi(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        custom_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        num_random_samples: int = 10,
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-1,
        damping: float = 1e-2,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool, dict]:
        """
        Multi-solution IK solver with orientation constraint.

        Tries multiple initial guesses to find valid solutions that satisfy
        both position AND orientation constraints.

        Args:
            target_position: Target position [x, y, z] in meters
            target_orientation: Target orientation as 3x3 rotation matrix
            current_joints: Current joint positions (used as one guess and for selection)
            custom_limits: Optional (lower, upper) limits to use instead of URDF limits
            num_random_samples: Number of random initial guesses to try
            max_iterations: Maximum iterations per IK solve
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)
            damping: Damping factor for DLS
            verbose: Print debug information

        Returns:
            Tuple of (joint_positions, success, info_dict)
        """
        # Use custom limits or URDF limits
        if custom_limits is not None:
            lower_limits, upper_limits = custom_limits
        else:
            lower_limits = self.joint_limits_lower
            upper_limits = self.joint_limits_upper

        def is_valid(joints: np.ndarray) -> bool:
            """Check if joints are within limits."""
            margin = 0.01  # ~0.57 degrees
            return (np.all(joints >= lower_limits + margin) and
                    np.all(joints <= upper_limits - margin))

        def joint_distance(j1: np.ndarray, j2: np.ndarray) -> float:
            """Compute weighted joint distance."""
            return np.linalg.norm(j1 - j2)

        # Generate initial guesses
        initial_guesses = []

        # 1. Current joints (if provided)
        if current_joints is not None:
            initial_guesses.append(("current", current_joints.copy()))

        # 2. Neutral configuration
        neutral = self._q_to_joint_positions(self.q_neutral)
        initial_guesses.append(("neutral", neutral))

        # 3. Predefined arm configurations
        elbow_up = neutral.copy()
        elbow_up[2] = 0.5
        initial_guesses.append(("elbow_up", elbow_up))

        elbow_down = neutral.copy()
        elbow_down[2] = -0.5
        initial_guesses.append(("elbow_down", elbow_down))

        extended = neutral.copy()
        extended[1] = -0.5
        extended[2] = 0.3
        initial_guesses.append(("extended", extended))

        raised = neutral.copy()
        raised[1] = 0.5
        raised[2] = 0.8
        initial_guesses.append(("raised", raised))

        # 4. Random samples within joint limits
        for i in range(num_random_samples):
            random_joints = np.random.uniform(lower_limits, upper_limits)
            initial_guesses.append((f"random_{i}", random_joints))

        # Try each initial guess
        all_solutions = []
        valid_solutions = []

        for name, guess in initial_guesses:
            # Use full IK with orientation constraint
            solution, success = self.inverse_kinematics(
                target_position,
                target_orientation=target_orientation,
                initial_guess=guess,
                max_iterations=max_iterations,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
                damping=damping,
            )

            if success:
                # Compute actual position error
                actual_pos = self.get_ee_position(solution)
                pos_error = np.linalg.norm(target_position - actual_pos)

                solution_info = {
                    "name": name,
                    "joints": solution,
                    "position_error": pos_error,
                    "is_valid": is_valid(solution),
                }
                all_solutions.append(solution_info)

                if is_valid(solution):
                    valid_solutions.append(solution_info)
                    if verbose:
                        print(f"  [✓] {name}: valid, error={pos_error*1000:.2f}mm")
                else:
                    if verbose:
                        out_of_range = []
                        for i, (j, lo, hi) in enumerate(zip(solution, lower_limits, upper_limits)):
                            if j < lo or j > hi:
                                out_of_range.append(f"j{i}={np.degrees(j):.1f}°")
                        print(f"  [✗] {name}: out of range ({', '.join(out_of_range)})")
            else:
                if verbose:
                    print(f"  [✗] {name}: IK did not converge")

        # Prepare info dict
        info = {
            "num_attempts": len(initial_guesses),
            "num_solutions": len(all_solutions),
            "num_valid": len(valid_solutions),
            "all_solutions": all_solutions,
            "valid_solutions": valid_solutions,
        }

        # No valid solutions found
        if not valid_solutions:
            if all_solutions:
                best_invalid = min(all_solutions, key=lambda s: s["position_error"])
                info["best_invalid"] = best_invalid
                return best_invalid["joints"], False, info
            else:
                return self._q_to_joint_positions(self.q_neutral), False, info

        # Select best valid solution (closest to current joints)
        if current_joints is not None:
            best = min(valid_solutions, key=lambda s: joint_distance(s["joints"], current_joints))
        else:
            best = min(valid_solutions, key=lambda s: s["position_error"])

        info["selected"] = best["name"]

        if verbose:
            print(f"  Selected: {best['name']} (error={best['position_error']*1000:.2f}mm)")

        return best["joints"], True, info

    def inverse_kinematics_multi(
        self,
        target_position: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        custom_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        num_random_samples: int = 10,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        damping: float = 1e-2,
        verbose: bool = False,
        fixed_joints: Optional[List[int]] = None,
        target_pitch: Optional[float] = None,
    ) -> Tuple[np.ndarray, bool, dict]:
        """
        Multi-solution IK solver that tries multiple initial guesses.

        Strategy:
        1. Try multiple initial guesses (current, neutral, random, predefined configs)
        2. Filter solutions that are within joint limits
        3. Select the solution closest to current_joints

        Args:
            target_position: Target position [x, y, z] in meters
            current_joints: Current joint positions (used as one guess and for selection)
            custom_limits: Optional (lower, upper) limits to use instead of URDF limits
            num_random_samples: Number of random initial guesses to try
            max_iterations: Maximum iterations per IK solve
            tolerance: Position error tolerance
            damping: Damping factor for DLS
            verbose: Print debug information
            fixed_joints: List of joint indices to keep fixed (0-indexed)
                         e.g., [4] to fix wrist_roll
            target_pitch: Target gripper pitch in radians. If specified, uses
                         pitch-constrained IK to maintain gripper orientation.

        Returns:
            Tuple of (joint_positions, success, info_dict)
            info_dict contains: num_solutions, num_valid, all_solutions, etc.
        """
        # Use custom limits or URDF limits
        if custom_limits is not None:
            lower_limits, upper_limits = custom_limits
        else:
            lower_limits = self.joint_limits_lower
            upper_limits = self.joint_limits_upper

        def is_valid(joints: np.ndarray) -> bool:
            """Check if joints are within limits."""
            margin = 0.01  # ~0.57 degrees
            return (np.all(joints >= lower_limits + margin) and
                    np.all(joints <= upper_limits - margin))

        def joint_distance(j1: np.ndarray, j2: np.ndarray) -> float:
            """Compute weighted joint distance."""
            return np.linalg.norm(j1 - j2)

        # Generate initial guesses
        initial_guesses = []

        # 1. Current joints (if provided)
        if current_joints is not None:
            initial_guesses.append(("current", current_joints.copy()))

        # 2. Neutral configuration
        neutral = self._q_to_joint_positions(self.q_neutral)
        initial_guesses.append(("neutral", neutral))

        # 3. Predefined arm configurations
        # Elbow up (positive elbow angle)
        elbow_up = neutral.copy()
        elbow_up[2] = 0.5  # elbow_flex positive
        initial_guesses.append(("elbow_up", elbow_up))

        # Elbow down (negative elbow angle)
        elbow_down = neutral.copy()
        elbow_down[2] = -0.5  # elbow_flex negative
        initial_guesses.append(("elbow_down", elbow_down))

        # Arm extended forward
        extended = neutral.copy()
        extended[1] = -0.5  # shoulder_lift slightly down
        extended[2] = 0.3   # elbow slightly bent
        initial_guesses.append(("extended", extended))

        # Arm raised
        raised = neutral.copy()
        raised[1] = 0.5  # shoulder_lift up
        extended[2] = 0.8   # elbow more bent
        initial_guesses.append(("raised", raised))

        # 4. Random samples within joint limits
        for i in range(num_random_samples):
            random_joints = np.random.uniform(lower_limits, upper_limits)
            initial_guesses.append((f"random_{i}", random_joints))

        # Apply fixed joint values from current_joints to ALL initial guesses
        # This ensures all solutions have the same fixed joint values
        if fixed_joints is not None and current_joints is not None:
            for name, guess in initial_guesses:
                for idx in fixed_joints:
                    if 0 <= idx < len(guess):
                        guess[idx] = current_joints[idx]

        # Try each initial guess
        all_solutions = []
        valid_solutions = []

        for name, guess in initial_guesses:
            # Choose IK solver based on whether pitch constraint is specified
            if target_pitch is not None:
                solution, success = self.inverse_kinematics_position_with_pitch(
                    target_position,
                    target_pitch=target_pitch,
                    initial_guess=guess,
                    max_iterations=max_iterations,
                    position_tolerance=tolerance,
                    pitch_tolerance=0.05,  # ~3 degrees
                    damping=damping,
                    fixed_joints=fixed_joints,
                )
            else:
                solution, success = self.inverse_kinematics_position_only(
                    target_position,
                    initial_guess=guess,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    damping=damping,
                    fixed_joints=fixed_joints,
                )

            if success:
                # Compute actual position error
                actual_pos = self.get_ee_position(solution)
                pos_error = np.linalg.norm(target_position - actual_pos)

                # Compute pitch for all solutions
                actual_pitch = self.get_gripper_pitch(solution)

                solution_info = {
                    "name": name,
                    "joints": solution,
                    "position_error": pos_error,
                    "pitch": actual_pitch,
                    "is_valid": is_valid(solution),
                }
                all_solutions.append(solution_info)

                if is_valid(solution):
                    valid_solutions.append(solution_info)
                    if verbose:
                        pitch_deg = np.degrees(actual_pitch)
                        print(f"  [✓] {name}: valid, error={pos_error*1000:.2f}mm, pitch={pitch_deg:.1f}°")
                else:
                    if verbose:
                        # Find which joints are out of range
                        out_of_range = []
                        for i, (j, lo, hi) in enumerate(zip(solution, lower_limits, upper_limits)):
                            if j < lo or j > hi:
                                out_of_range.append(f"j{i}={np.degrees(j):.1f}°")
                        print(f"  [✗] {name}: out of range ({', '.join(out_of_range)})")
            else:
                if verbose:
                    print(f"  [✗] {name}: IK did not converge")

        # Prepare info dict
        info = {
            "num_attempts": len(initial_guesses),
            "num_solutions": len(all_solutions),
            "num_valid": len(valid_solutions),
            "all_solutions": all_solutions,
            "valid_solutions": valid_solutions,
            "target_pitch": target_pitch,
        }

        # No valid solutions found
        if not valid_solutions:
            # Return best invalid solution if any
            if all_solutions:
                best_invalid = min(all_solutions, key=lambda s: s["position_error"])
                info["best_invalid"] = best_invalid
                return best_invalid["joints"], False, info
            else:
                return self._q_to_joint_positions(self.q_neutral), False, info

        # Select best valid solution (closest to current joints)
        if current_joints is not None:
            best = min(valid_solutions, key=lambda s: joint_distance(s["joints"], current_joints))
        else:
            # If no current joints, pick the one with smallest position error
            best = min(valid_solutions, key=lambda s: s["position_error"])

        info["selected"] = best["name"]
        info["selected_pitch"] = best.get("pitch")

        if verbose:
            pitch_str = f", pitch={np.degrees(best.get('pitch', 0)):.1f}°" if best.get("pitch") is not None else ""
            print(f"  Selected: {best['name']} (error={best['position_error']*1000:.2f}mm{pitch_str})")

        return best["joints"], True, info
