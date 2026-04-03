"""
IK Solver abstractions for the Controller module.

Converts EEF targets (position + orientation) into joint angles.
Multiple backends supported:
- PassthroughIKSolver: For environments that handle IK internally (e.g. RoboCasa)
- PinocchioIKSolver: URDF-based CLIK for real robots (Franka, UR)
- MoveItIKSolver: ROS2 MoveIt integration (stub)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EEFTarget:
    """End-effector target for IK solving."""

    position: np.ndarray  # [x, y, z]
    orientation: Optional[np.ndarray] = None  # quaternion [x, y, z, w] or None
    gripper_state: Optional[float] = None  # gripper value (-1 close, 1 open)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=np.float64)


@dataclass
class IKResult:
    """Result from IK solver."""

    success: bool
    joint_positions: Optional[np.ndarray] = None  # target joint angles
    error_message: str = ""
    residual: float = 0.0  # position error norm after solving

    def __post_init__(self):
        if self.joint_positions is not None:
            self.joint_positions = np.asarray(self.joint_positions, dtype=np.float64)


class IKSolver(ABC):
    """Abstract base class for IK solvers."""

    @abstractmethod
    def solve(
        self,
        target: EEFTarget,
        current_joints: Optional[np.ndarray] = None,
        current_ee_pose: Optional[Dict[str, Any]] = None,
    ) -> IKResult:
        """Solve inverse kinematics for the given EEF target.

        Args:
            target: Desired end-effector pose.
            current_joints: Current joint positions (seed for iterative solvers).
            current_ee_pose: Current EE pose dict with 'position' and 'orientation'.

        Returns:
            IKResult with success flag and joint positions.
        """
        pass


class PassthroughIKSolver(IKSolver):
    """Passthrough IK solver for environments that handle IK internally.

    RoboCasa and similar simulators accept cartesian targets and compute
    IK inside env.step(). This solver always returns success with no
    joint positions, signaling that the caller should send the cartesian
    target directly to the environment.
    """

    def solve(
        self,
        target: EEFTarget,
        current_joints: Optional[np.ndarray] = None,
        current_ee_pose: Optional[Dict[str, Any]] = None,
    ) -> IKResult:
        return IKResult(success=True, joint_positions=None)


class PinocchioIKSolver(IKSolver):
    """URDF-based Closed-Loop IK using Pinocchio.

    Uses damped least-squares (Levenberg-Marquardt) for iterative IK.
    Suitable for real robots (Franka Emika Panda, UR5e, etc.).

    Args:
        urdf_path: Path to robot URDF file.
        ee_frame_name: Name of the end-effector frame in URDF.
        max_iter: Maximum CLIK iterations.
        dt: Integration step size.
        damping: Damping factor for DLS.
        position_tol: Convergence tolerance (meters).
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str = "panda_hand",
        max_iter: int = 200,
        dt: float = 0.1,
        damping: float = 1e-6,
        position_tol: float = 1e-3,
    ):
        self._urdf_path = urdf_path
        self._ee_frame_name = ee_frame_name
        self._max_iter = max_iter
        self._dt = dt
        self._damping = damping
        self._position_tol = position_tol
        self._model = None
        self._data = None
        self._ee_frame_id = None

    def _ensure_loaded(self) -> None:
        """Lazy-load pinocchio model from URDF."""
        if self._model is not None:
            return

        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "pinocchio is required for PinocchioIKSolver. "
                "Install with: pip install pin"
            )

        self._model = pin.buildModelFromUrdf(self._urdf_path)
        self._data = self._model.createData()
        self._ee_frame_id = self._model.getFrameId(self._ee_frame_name)

        if self._ee_frame_id >= self._model.nframes:
            raise ValueError(
                f"Frame '{self._ee_frame_name}' not found in URDF. "
                f"Available: {[f.name for f in self._model.frames]}"
            )

        logger.info(
            f"PinocchioIK loaded: {self._urdf_path}, "
            f"frame={self._ee_frame_name}, nq={self._model.nq}"
        )

    def solve(
        self,
        target: EEFTarget,
        current_joints: Optional[np.ndarray] = None,
        current_ee_pose: Optional[Dict[str, Any]] = None,
    ) -> IKResult:
        self._ensure_loaded()

        import pinocchio as pin

        model = self._model
        data = self._data

        # Build target SE3
        target_pos = target.position
        if target.orientation is not None:
            # quaternion xyzw -> pinocchio Quaternion (x, y, z, w)
            quat = pin.Quaternion(
                target.orientation[3],  # w
                target.orientation[0],  # x
                target.orientation[1],  # y
                target.orientation[2],  # z
            )
            oMdes = pin.SE3(quat.matrix(), target_pos)
        else:
            oMdes = pin.SE3(np.eye(3), target_pos)

        # Initial seed
        if current_joints is not None:
            q = np.array(current_joints, dtype=np.float64)
        else:
            q = pin.neutral(model)

        for i in range(self._max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            oMcur = data.oMf[self._ee_frame_id]
            err_se3 = pin.log6(oMcur.actInv(oMdes))
            err = err_se3.vector

            if np.linalg.norm(err[:3]) < self._position_tol:
                return IKResult(
                    success=True,
                    joint_positions=q,
                    residual=float(np.linalg.norm(err[:3])),
                )

            # Damped least-squares
            J = pin.computeFrameJacobian(
                model, data, q, self._ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            JtJ = J.T @ J + self._damping * np.eye(model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)
            q = pin.integrate(model, q, dq * self._dt)

        residual = float(np.linalg.norm(err[:3]))
        logger.warning(
            f"PinocchioIK did not converge after {self._max_iter} iters, "
            f"residual={residual:.6f}m"
        )
        return IKResult(
            success=False,
            joint_positions=q,
            error_message=f"Did not converge (residual={residual:.6f}m)",
            residual=residual,
        )


class MoveItIKSolver(IKSolver):
    """ROS2 MoveIt IK solver stub.

    Placeholder for future MoveIt2 integration via moveit_py or service calls.

    Args:
        move_group: MoveIt planning group name.
        planner_id: Planning algorithm identifier.
    """

    def __init__(
        self,
        move_group: str = "panda_arm",
        planner_id: str = "RRTConnect",
    ):
        self._move_group = move_group
        self._planner_id = planner_id
        logger.info(f"MoveItIKSolver stub: group={move_group}, planner={planner_id}")

    def solve(
        self,
        target: EEFTarget,
        current_joints: Optional[np.ndarray] = None,
        current_ee_pose: Optional[Dict[str, Any]] = None,
    ) -> IKResult:
        logger.warning("MoveItIKSolver is a stub — returning failure")
        return IKResult(
            success=False,
            error_message="MoveIt IK solver not yet implemented",
        )


def create_ik_solver(solver_type: str, **kwargs) -> IKSolver:
    """Factory function to create an IK solver.

    Args:
        solver_type: One of "passthrough", "pinocchio", "moveit".
        **kwargs: Solver-specific configuration.

    Returns:
        IKSolver instance.
    """
    solver_type = solver_type.lower()

    if solver_type == "passthrough":
        return PassthroughIKSolver()
    elif solver_type == "pinocchio":
        return PinocchioIKSolver(**kwargs)
    elif solver_type == "moveit":
        return MoveItIKSolver(**kwargs)
    else:
        logger.warning(f"Unknown IK solver type '{solver_type}', using passthrough")
        return PassthroughIKSolver()
