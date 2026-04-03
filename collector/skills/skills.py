import time
import numpy as np
from pick_and_place_module.eef_control import MoveGroupControl
from pick_and_place_module.grasping import GripperInterface
from pick_and_place_module.orientation import euler_from_quaternion, quaternion_from_euler
from copy import deepcopy
from math import pi
from pick_and_place_module.ros2_utils import ensure_rclpy_initialized, get_node


class PrimitiveSkill:
    def __init__(
        self,
        gripper_offset=0.05,
        intermediate_z_stop=0.6,
        intermediate_distance=0.09,
        speed=0.05,
        push_length=0.02,
        pull_length=0.02,
        sweep_count=3,
        sweep_width=0.03,
        node=None,
        gripper_namespace='auto',
        gripper_wait_timeout=5.0,
    ):
        self.gripper_offset = gripper_offset
        self.intermediate_z_stop = intermediate_z_stop
        self.intermediate_distance = intermediate_distance
        self.speed = speed
        # self.push_length = push_length
        # self.pull_length = pull_length
        self.sweep_count = sweep_count
        self.sweep_width = sweep_width
        self.pose0 = None
        self.pose1 = None
        self.target_pose = None
        self.waypoint_density = 5  # 10 → 5로 줄여서 waypoint 개수 감소 (더 빠름)
        ensure_rclpy_initialized()
        self.node = node or get_node('primitive_skill')
        self._logger = self.node.get_logger()
        self.moveit_control = MoveGroupControl(speed, node=self.node)
        self.gripper = GripperInterface(
            node=self.node,
            namespace=gripper_namespace,
            wait_timeout=gripper_wait_timeout,
        )

    # === Utility Functions ===

    def interpolate_pose(self, start, end, steps):
        """Linear interpolation between start and end points."""
        if steps <= 1:
            return [list(end)]
        start_array = np.array(start)
        end_array = np.array(end)
        array_list = [
            start_array + (end_array - start_array) * i / (steps - 1)
            for i in range(steps)
        ]
        return [array.tolist() for array in array_list]

    def _clamp_min_z(self, waypoints, z_min):
        """Ensure all waypoints have z >= z_min."""
        clamped = []
        for w in waypoints:
            w2 = list(w)
            if w2[2] < z_min:
                w2[2] = z_min
            clamped.append(w2)
        return clamped

    def _execute_waypoints(self, waypoints, label: str) -> bool:
        """Drive the robot through a list of waypoints in a reproducible way."""
        if not waypoints:
            return True

        if self.moveit_control.follow_cartesian_path(waypoints):
            self._logger.info(f"Executed {label} with Cartesian path planning.")
            return True

        self._logger.warning(
            f"Cartesian path planning for '{label}' failed, falling back to pose goals."
        )
        success = True
        for waypoint in waypoints:
            self._logger.info(f"Executing {label} waypoint: {waypoint}")
            success = self.moveit_control.go_to_pose_goal(*waypoint) and success
        return success

    def setPose0(self, x, y, z, roll, pitch, yaw):
        self.pose0 = [x, y, z, roll + pi / 4, pitch, yaw]

    def setPose1(self, x, y, z, roll, pitch, yaw):
        self.pose1 = [x, y, z, roll + pi / 4, pitch, yaw]

    def setTargetPose(self, x, y, z, roll, pitch, yaw):
        self.target_pose = [x, y, z, roll + pi / 4, pitch, yaw]

    # def setPose0(self, x, y, z, roll, pitch, yaw):
    #     # getPose()에서 받은 값을 그대로 사용
    #     self.pose0 = [x, y, z, roll, pitch+np.pi, yaw+0.75*np.pi]

    def setPose0Quaternion(self, x, y, z, qx, qy, qz, qw):
        """Quaternion을 사용해서 pose0 설정"""
        self.pose0_quat = {
            'position': (x, y, z),
            'quaternion': (qx, qy, qz, qw)
        }

    # def setPose1(self, x, y, z, roll, pitch, yaw):
    #     # getPose()에서 받은 값을 그대로 사용
    #     self.pose1 = [x, y, z, roll, pitch+np.pi, yaw+0.75*np.pi]

    def setPose1Quaternion(self, x, y, z, qx, qy, qz, qw):
        """Quaternion을 사용해서 pose1 설정"""
        self.pose1_quat = {
            'position': (x, y, z),
            'quaternion': (qx, qy, qz, qw)
        }

    # def setTargetPose(self, x, y, z, roll, pitch, yaw):
    #     # getPose()에서 받은 값을 그대로 사용
    #     self.target_pose = [x, y, z, roll, pitch+np.pi, yaw+0.75*np.pi]

    def setTargetPoseQuaternion(self, x, y, z, qx, qy, qz, qw):
        """
        Quaternion을 직접 사용해서 목표 자세 설정
        Euler 변환의 singularity 문제를 우회

        Args:
            x, y, z: 위치 (m)
            qx, qy, qz, qw: quaternion (x, y, z, w 순서)
        """
        # Quaternion을 직접 저장 (euler 변환 없음)
        self.target_pose_quat = {
            'position': (x, y, z),
            'quaternion': (qx, qy, qz, qw)
        }

    def execute_go_quaternion(self):
        """
        setTargetPoseQuaternion()으로 설정한 자세로 이동

        2단계 접근:
        1. 먼저 현재 위치에서 목표 회전으로 회전
        2. 회전 유지한 채로 목표 위치로 이동
        """
        if not hasattr(self, 'target_pose_quat'):
            self._logger.error("target_pose_quat not set. Call setTargetPoseQuaternion() first.")
            return False

        target_x, target_y, target_z = self.target_pose_quat['position']
        qx, qy, qz, qw = self.target_pose_quat['quaternion']

        # 현재 위치 가져오기
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            self._logger.error("Failed to get current pose")
            return False

        current_x = current_pose_msg.position.x
        current_y = current_pose_msg.position.y
        current_z = current_pose_msg.position.z

        self._logger.info(f"Step 1: Rotating at current position ({current_x:.4f}, {current_y:.4f}, {current_z:.4f})")
        self._logger.info(f"  Target quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")

        # 1단계: 현재 위치에서 목표 회전으로 회전
        success1 = self.moveit_control.go_to_pose_goal_quaternion(
            current_x, current_y, current_z, qx, qy, qz, qw
        )

        if not success1:
            self._logger.error("Step 1 failed: Could not rotate to target orientation")
            return False

        self._logger.info("Step 1 completed: Rotation successful")

        # 2단계: 회전 유지한 채로 목표 위치로 이동
        self._logger.info(f"Step 2: Moving to target position ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")

        success2 = self.moveit_control.go_to_pose_goal_quaternion(
            target_x, target_y, target_z, qx, qy, qz, qw
        )

        if not success2:
            self._logger.error("Step 2 failed: Could not move to target position")
            return False

        self._logger.info("Step 2 completed: Position reached successfully")
        self._logger.info("execute_go_quaternion completed successfully")

        return True

    def getPoseQuaternion(self):
        """
        현재 자세를 quaternion으로 반환
        setTargetPoseQuaternion()과 함께 사용

        Returns:
            dict: {'position': (x,y,z), 'quaternion': (qx,qy,qz,qw)} 또는 None
        """
        pose_msg = self._get_current_pose()
        if pose_msg is None:
            return None

        p = pose_msg.position
        q = pose_msg.orientation

        # Quaternion 정규화 (w를 양수로)
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        return {
            'position': (p.x, p.y, p.z),
            'quaternion': (qx, qy, qz, qw)
        }

    def go_to_ready_pose(self):
        self.moveit_control.go_to_joint_state(
            0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785
        )
        time.sleep(2)
        print("go to ready pose")

    def _get_current_pose(self):
        pose = self.moveit_control.get_current_pose()
        if pose is None:
            self._logger.error("Unable to retrieve current pose from MoveIt.")
        return pose

    def current_pose(self):
        quaternion_pose = self._get_current_pose()
        if quaternion_pose is None:
            return None
        current_euler_pose = euler_from_quaternion(
            (
                quaternion_pose.orientation.x,
                quaternion_pose.orientation.y,
                quaternion_pose.orientation.z,
                quaternion_pose.orientation.w,
            )
        )
        return current_euler_pose

    def getPose(self):
        pose_msg = self._get_current_pose()
        if pose_msg is None:
            return None
        pose = pose_msg.position
        orientation = pose_msg.orientation

        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        pose0_x, pose0_y, pose0_z = pose.x, pose.y, pose.z
        pose0_roll, pose0_pitch, pose0_yaw = roll, pitch, yaw

        # adjust
        # pose0_yaw = pose0_yaw + np.pi / 4
        # pose0_pitch = pose0_pitch + np.pi
        # pose0_roll = pose0_roll + np.pi
        print(
            f"({pose0_x:.7f}, {pose0_y:.7f}, {pose0_z:.7f}, {pose0_yaw:.7f}, {pose0_pitch:.7f}, {pose0_roll:.7f})"
        )

        return [pose0_x, pose0_y, pose0_z, pose0_yaw, pose0_pitch, pose0_roll]

    # def getPose(self):
    #     """
    #     현재 로봇의 자세를 반환 (원본 euler 각도, 보정 없음)

    #     이제 setTargetPose()가 내부적으로 roll+45도 보정을 하므로,
    #     getPose()는 보정 없이 원본 값을 반환합니다.

    #     Returns:
    #         [x, y, z, roll, pitch, yaw]: 위치와 회전 (라디안)
    #     """
    #     pose_msg = self._get_current_pose()
    #     if pose_msg is None:
    #         return None
    #     p = pose_msg.position
    #     q = pose_msg.orientation

    #     # Quaternion 정규화 (w를 양수로)
    #     qx, qy, qz, qw = q.x, q.y, q.z, q.w
    #     if qw < 0:
    #         qx, qy, qz, qw = -qx, -qy, -qz, -qw

    #     quaternion = [qx, qy, qz, qw]
    #     roll, pitch, yaw = euler_from_quaternion(quaternion)

    #     # 보정 없이 원본 값 반환
    #     print(f"({p.x:.7f}, {p.y:.7f}, {p.z:.7f}, {roll:.7f}, {pitch:.7f}, {yaw:.7f})", flush=True)
    #     return [p.x, p.y, p.z, roll, pitch, yaw]

    # === Primitive Skills ===

    def execute_pick_and_place(self, gripper_force=5, axis=0):
        self.gripper.grasp(0.1, 0)
        time.sleep(1)

        # ---- PICK (to pose0) ----
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.pose0)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.pose0)
        intermediate_pose[2] = self.intermediate_z_stop
        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        destination_pose = deepcopy(self.pose0)
        destination_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            intermediate_pose, destination_pose, self.waypoint_density
        )

        # clamp z to be >= pose0.z
        waypoints = self._clamp_min_z(waypoints, self.pose0[2])

        self._execute_waypoints(waypoints, "pick approach")

        self.gripper.grasp(0.01, gripper_force)
        time.sleep(1)

        # ---- PLACE (to pose1) ----
        current_pose_list = deepcopy(self.pose1)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.pose1)
        intermediate_pose[2] = self.intermediate_z_stop
        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        waypoints += self.interpolate_pose(
            intermediate_pose, self.pose1, self.waypoint_density
        )

        # clamp z to be >= pose1.z
        waypoints = self._clamp_min_z(waypoints, self.pose1[2])

        self._execute_waypoints(waypoints, "place approach")

        self.gripper.grasp(0.1, 0)
        time.sleep(1)
        self.go_to_ready_pose()

    def execute_pick(self, gripper_force=5, axis=0):
        self.gripper.grasp(0.1, 0)
        time.sleep(1)

        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop  # 위쪽 안전 높이로 이동

        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        destination_pose = deepcopy(self.target_pose)
        destination_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            intermediate_pose, destination_pose, self.waypoint_density
        )

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "pick-only approach")
        self.gripper.grasp(0.005, gripper_force)
        time.sleep(1)
        # self.go_to_ready_pose()

    def execute_place(self, axis=0):
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )
        waypoints += self.interpolate_pose(
            intermediate_pose, self.target_pose, self.waypoint_density
        )

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "place-only approach")

        self.gripper.grasp(0.1, 0)
        time.sleep(1)
        self.go_to_ready_pose()

    def execute_push(self, gripper_force=5, axis=0, distance=0.1):
        # NOTE: 원본 target_pose를 바꾸지 않으려면 deepcopy 후 수정하세요.
        if axis < 2:
            self.target_pose[axis] -= distance
        elif axis == 2:
            self.target_pose[axis] += distance

        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            intermediate_pose, approach_pose, self.waypoint_density
        )

        push_pose = deepcopy(self.target_pose)
        if axis < 2:
            push_pose[axis] += distance
        elif axis == 2:
            push_pose[2] -= distance  # self.push_length 대신 distance 사용

        push_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            approach_pose, push_pose, self.waypoint_density
        )

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "push motion")

        self.gripper.grasp(0.1, 0)
        time.sleep(1)
        self.go_to_ready_pose()

    def execute_pull(self, gripper_force=5, axis=0, distance=0.02):
        # self.gripper.grasp(0.01, gripper_force)
        time.sleep(1)

        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints = self.interpolate_pose(
            current_pose_list, approach_pose, self.waypoint_density
        )

        pull_pose = deepcopy(self.target_pose)
        pull_pose[axis] -= distance
        pull_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            approach_pose, pull_pose, self.waypoint_density
        )

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "pull motion")

        time.sleep(1)
        self.gripper.grasp(0.1, 0)
        time.sleep(1)
        self.go_to_ready_pose()

    def execute_sweep(self, axis=0, distance=3):
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        for _ in range(self.sweep_count):
            sweep_positive = deepcopy(self.target_pose)
            sweep_positive[axis] += distance
            sweep_positive[2] += self.gripper_offset + 0.025
            waypoints.append(sweep_positive)

            sweep_negative = deepcopy(self.target_pose)
            sweep_negative[axis] -= distance
            sweep_negative[2] += self.gripper_offset + 0.025
            waypoints.append(sweep_negative)

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "sweep motion")

    def execute_rotate(self, gripper_force=5, axis=0):
        move_group = self.moveit_control
        self.gripper.grasp(0.1, 0)
        time.sleep(1)

        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            return
        current_pose = current_pose_msg.position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
        )

        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(
            current_pose_list, intermediate_pose, self.waypoint_density
        )

        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(
            intermediate_pose, approach_pose, self.waypoint_density
        )

        # clamp z to be >= target z
        waypoints = self._clamp_min_z(waypoints, self.target_pose[2])

        self._execute_waypoints(waypoints, "rotate approach")

        self.gripper.grasp(0.01, gripper_force)
        time.sleep(1)

        current_joint_values = move_group.get_current_joint_states()
        if current_joint_values is None:
            self._logger.error("Failed to read current joint states for rotate motion.")
            return
        current_joint_values[-1] -= pi / 2
        move_group.go_to_joint_state(*current_joint_values)

        target_pose = deepcopy(self.target_pose)
        target_pose[2] += 0.2
        move_group.go_to_pose_goal(
            target_pose[0],
            target_pose[1],
            target_pose[2],
            target_pose[3],
            target_pose[4],
            target_pose[5],
        )

    def execute_go(self):
        move_group = self.moveit_control

        # Try Cartesian path first for straight-line motion
        self._logger.info(f"Attempting Cartesian path to: {self.target_pose}")
        success = move_group.follow_cartesian_path([self.target_pose], max_step=0.01)

        if success:
            self._logger.info("Cartesian path execution succeeded")
        else:
            # Fallback to joint space planning
            self._logger.warn("Cartesian path failed, using joint space planning")
            move_group.go_to_pose_goal(
                self.target_pose[0],
                self.target_pose[1],
                self.target_pose[2],
                self.target_pose[3],
                self.target_pose[4],
                self.target_pose[5],
            )
            self._logger.info(f"Joint space planning completed: {self.target_pose}")

    def execute_gripper(self, width1, force1, speed1=0.08):
        self.gripper.grasp(width1, force1, speed1)

    # ========== Quaternion-based Primitive Skills ==========

    def interpolate_pose_quaternion(self, start_pose_dict, end_pose_dict, steps):
        """
        Quaternion 자세 간 보간

        Args:
            start_pose_dict: {'position': (x,y,z), 'quaternion': (qx,qy,qz,qw)}
            end_pose_dict: {'position': (x,y,z), 'quaternion': (qx,qy,qz,qw)}
            steps: 보간 단계 수

        Returns:
            list of dict: 보간된 자세들
        """
        if steps <= 1:
            return [end_pose_dict]

        start_pos = np.array(start_pose_dict['position'])
        end_pos = np.array(end_pose_dict['position'])

        start_quat = np.array(start_pose_dict['quaternion'])
        end_quat = np.array(end_pose_dict['quaternion'])

        # Quaternion의 부호를 맞춤 (shortest path)
        if np.dot(start_quat, end_quat) < 0:
            end_quat = -end_quat

        waypoints = []
        for i in range(steps):
            t = i / (steps - 1)

            # 위치 선형 보간
            pos = start_pos + (end_pos - start_pos) * t

            # Quaternion SLERP (Spherical Linear Interpolation)
            dot = np.dot(start_quat, end_quat)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            if abs(theta) < 1e-6:
                # 거의 같은 quaternion
                quat = start_quat
            else:
                quat = (np.sin((1-t)*theta) * start_quat + np.sin(t*theta) * end_quat) / np.sin(theta)

            # 정규화
            quat = quat / np.linalg.norm(quat)

            waypoints.append({
                'position': tuple(pos),
                'quaternion': tuple(quat)
            })

        return waypoints

    def _execute_waypoints_quaternion(self, waypoints, description="motion"):
        """
        Quaternion 기반 waypoint들을 Cartesian path로 실행 (관절 연속성 보장)

        Args:
            waypoints: list of {'position': (x,y,z), 'quaternion': (qx,qy,qz,qw)}
            description: 로그용 설명
        """
        self._logger.info(f"Executing {description} with {len(waypoints)} waypoints (quaternion, Cartesian)")

        # Cartesian path로 한번에 실행 (관절 연속성 보장)
        success = self.moveit_control.follow_cartesian_path_quaternion(
            waypoints,
            scale=self.speed,
            max_step=0.2  # Cartesian step 크기 (0.005→0.01로 증가하면 더 빠름)
        )

        if not success:
            self._logger.error(f"{description} failed with Cartesian path")
            return False

        self._logger.info(f"{description} completed successfully")
        return True

    def execute_pull_quaternion(self, gripper_force=5, axis=0, distance=0.02, home=True, gripper= True):
        """
        Quaternion 기반 pull 동작

        Args:
            gripper_force: 그리퍼 힘
            axis: 당길 축 (0=x, 1=y, 2=z)
            distance: 당길 거리 (m)
        """
        if not hasattr(self, 'target_pose_quat'):
            self._logger.error("target_pose_quat not set. Call setTargetPoseQuaternion() first.")
            return False

        time.sleep(1)

        # 현재 위치 가져오기
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            self._logger.error("Failed to get current pose")
            return False

        current_pos = (
            current_pose_msg.position.x,
            current_pose_msg.position.y,
            current_pose_msg.position.z
        )

        # 현재 quaternion (정규화)
        qx, qy, qz, qw = (
            current_pose_msg.orientation.x,
            current_pose_msg.orientation.y,
            current_pose_msg.orientation.z,
            current_pose_msg.orientation.w
        )
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        current_quat = (qx, qy, qz, qw)

        current_pose_dict = {
            'position': current_pos,
            'quaternion': current_quat
        }

        # 목표 자세
        target_x, target_y, target_z = self.target_pose_quat['position']
        target_quat = self.target_pose_quat['quaternion']

        # Approach pose: target 위치 + gripper_offset (z축)
        approach_pos = (target_x, target_y, target_z + self.gripper_offset)
        approach_pose_dict = {
            'position': approach_pos,
            'quaternion': target_quat
        }

        # 1단계: 현재 위치 → approach 위치
        waypoints = self.interpolate_pose_quaternion(
            current_pose_dict, approach_pose_dict, self.waypoint_density
        )

        # Z축 클램핑 (target_z 이상으로만 이동)
        clamped_waypoints = []
        for wp in waypoints:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        # Approach 위치로 이동
        success = self._execute_waypoints_quaternion(clamped_waypoints, "move to approach")
        if not success:
            return False

        # 2단계: target 위치로 하강 (그리퍼 닫기 전)
        target_pose_dict = {
            'position': (target_x, target_y, target_z),
            'quaternion': target_quat
        }

        # Approach에서 target으로 하강 (Cartesian path로 부드럽게)
        waypoints_descend = self.interpolate_pose_quaternion(
            approach_pose_dict, target_pose_dict, self.waypoint_density
        )

        success = self._execute_waypoints_quaternion(waypoints_descend, "descend to target")
        if not success:
            self._logger.error("Failed to descend to target position")
            return False

        # 3단계: 그리퍼 닫기 (물체 잡기)
        # width를 작게 설정하여 그리퍼가 계속 힘을 유지하도록 함
        time.sleep(0.5)
        self.gripper.grasp(0.005, gripper_force, 0.04)  # 0.5cm로 좁게 잡아서 force 유지
        time.sleep(1.5)  # 그리퍼가 완전히 잡을 때까지 대기

        # 4단계: Pull pose로 이동 (target에서 지정된 축 방향으로 distance만큼)
        pull_pos_list = list((target_x, target_y, target_z))
        pull_pos_list[axis] -= distance
        # pull_pos_list[2] += 0.0002  # z축 약간 올림 (필요시 주석 해제)
        pull_pos = tuple(pull_pos_list)

        pull_pose_dict = {
            'position': pull_pos,
            'quaternion': target_quat
        }

        # target → pull waypoints
        waypoints_pull = self.interpolate_pose_quaternion(
            target_pose_dict, pull_pose_dict, self.waypoint_density
        )

        # Z축 클램핑 (target_z 이상으로만 이동)
        clamped_waypoints_pull = []
        for wp in waypoints_pull:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints_pull.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        # Pull 동작 실행
        success = self._execute_waypoints_quaternion(clamped_waypoints_pull, "pull motion")

        if not success:
            return False

        # 5단계: 그리퍼 열기

        if gripper:
            time.sleep(0.5)
            self.gripper.grasp(0.1, 0)
            time.sleep(1)

        # 6단계: 준비 자세로 복귀
        if home:
            self.go_to_ready_pose()

        return True

    def execute_pick_quaternion(self, gripper_force=5, axis=0, distance=0, home=False, gripper=True):
        """
        Quaternion 기반 pick 동작

        Args:
            gripper_force: 그리퍼 힘
            axis: 접근 축 (0=x, 1=y, 2=z) - intermediate pose에서 어느 방향으로 접근할지
            distance: pick 후 들어올릴 거리 (m)
            home: True면 완료 후 ready pose로 복귀
            gripper: True면 물체를 잡기 위해 그리퍼를 닫음
        """
        if not hasattr(self, 'target_pose_quat'):
            self._logger.error("target_pose_quat not set. Call setTargetPoseQuaternion() first.")
            return False

        # 1단계: 그리퍼 열기
        if gripper:
            self.gripper.grasp(0.1, 0)
            time.sleep(1)

        # 현재 위치 가져오기
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            self._logger.error("Failed to get current pose")
            return False

        current_pos = (
            current_pose_msg.position.x,
            current_pose_msg.position.y,
            current_pose_msg.position.z
        )

        # 현재 quaternion (정규화)
        qx, qy, qz, qw = (
            current_pose_msg.orientation.x,
            current_pose_msg.orientation.y,
            current_pose_msg.orientation.z,
            current_pose_msg.orientation.w
        )
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        current_quat = (qx, qy, qz, qw)

        current_pose_dict = {
            'position': current_pos,
            'quaternion': current_quat
        }

        # 목표 자세
        target_x, target_y, target_z = self.target_pose_quat['position']
        target_quat = self.target_pose_quat['quaternion']

        # Intermediate pose: 높은 곳에서 안전하게 접근
        intermediate_pos_list = [target_x, target_y, self.intermediate_z_stop]
        if axis < 2:
            # x 또는 y 축으로 offset 추가 (옆에서 접근)
            intermediate_pos_list[axis] -= self.intermediate_distance
        intermediate_pos = tuple(intermediate_pos_list)

        intermediate_pose_dict = {
            'position': intermediate_pos,
            'quaternion': target_quat
        }

        # 2단계: 현재 위치 → intermediate 위치
        waypoints = self.interpolate_pose_quaternion(
            current_pose_dict, intermediate_pose_dict, self.waypoint_density
        )

        # Z축 클램핑 (target_z 이상으로만 이동)
        clamped_waypoints = []  
        for wp in waypoints:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        # Intermediate 위치로 이동
        success = self._execute_waypoints_quaternion(clamped_waypoints, "move to intermediate")
        if not success:
            return False

        # 3단계: intermediate → target 위치 (gripper_offset만큼 위)로 하강
        approach_pos = (target_x, target_y, target_z + self.gripper_offset)
        approach_pose_dict = {
            'position': approach_pos,
            'quaternion': target_quat
        }

        waypoints_descend = self.interpolate_pose_quaternion(
            intermediate_pose_dict, approach_pose_dict, self.waypoint_density
        )

        # Z축 클램핑
        clamped_waypoints_descend = []
        for wp in waypoints_descend:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints_descend.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        success = self._execute_waypoints_quaternion(clamped_waypoints_descend, "descend to pick")
        if not success:
            self._logger.error("Failed to descend to pick position")
            return False

        # 4단계: 그리퍼 닫기 (물체 잡기)
        if gripper:
            time.sleep(0.5)
            self.gripper.grasp(0.005, gripper_force, 0.04)
            time.sleep(1.5)

        # 5단계: 물체 들어올리기 (distance만큼)
        if distance > 0:
            lift_pos = (target_x, target_y, target_z + self.gripper_offset + distance)
            lift_pose_dict = {
                'position': lift_pos,
                'quaternion': target_quat
            }

            waypoints_lift = self.interpolate_pose_quaternion(
                approach_pose_dict, lift_pose_dict, self.waypoint_density
            )

            success = self._execute_waypoints_quaternion(waypoints_lift, "lift object")
            if not success:
                self._logger.error("Failed to lift object")
                return False

        # 6단계: 준비 자세로 복귀 (선택사항)
        if home:
            self.go_to_ready_pose()

        return True

    def execute_place_quaternion(self, gripper_force=5, axis=0, home=True, gripper=True):
        """
        Quaternion 기반 place 동작

        Args:
            gripper_force: 그리퍼 힘 (사용 안함, 호환성 위해 유지)
            axis: 접근 축 (0=x, 1=y, 2=z) - intermediate pose에서 어느 방향으로 접근할지
            home: True면 완료 후 ready pose로 복귀
            gripper: True면 물체를 놓기 위해 그리퍼를 엶
        """
        if not hasattr(self, 'target_pose_quat'):
            self._logger.error("target_pose_quat not set. Call setTargetPoseQuaternion() first.")
            return False

        time.sleep(1)

        # 현재 위치 가져오기
        current_pose_msg = self._get_current_pose()
        if current_pose_msg is None:
            self._logger.error("Failed to get current pose")
            return False

        current_pos = (
            current_pose_msg.position.x,
            current_pose_msg.position.y,
            current_pose_msg.position.z
        )

        # 현재 quaternion (정규화)
        qx, qy, qz, qw = (
            current_pose_msg.orientation.x,
            current_pose_msg.orientation.y,
            current_pose_msg.orientation.z,
            current_pose_msg.orientation.w
        )
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        current_quat = (qx, qy, qz, qw)

        current_pose_dict = {
            'position': current_pos,
            'quaternion': current_quat
        }

        # 목표 자세
        target_x, target_y, target_z = self.target_pose_quat['position']
        target_quat = self.target_pose_quat['quaternion']

        # Intermediate pose: 높은 곳에서 안전하게 접근
        intermediate_pos_list = [target_x, target_y, self.intermediate_z_stop]
        if axis < 2:
            # x 또는 y 축으로 offset 추가 (옆에서 접근)
            intermediate_pos_list[axis] -= self.intermediate_distance
        intermediate_pos = tuple(intermediate_pos_list)

        intermediate_pose_dict = {
            'position': intermediate_pos,
            'quaternion': target_quat
        }

        # 1단계: 현재 위치 → intermediate 위치
        waypoints = self.interpolate_pose_quaternion(
            current_pose_dict, intermediate_pose_dict, self.waypoint_density
        )

        # Z축 클램핑 (target_z 이상으로만 이동)
        clamped_waypoints = []
        for wp in waypoints:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        # Intermediate 위치로 이동
        success = self._execute_waypoints_quaternion(clamped_waypoints, "move to intermediate")
        if not success:
            return False

        # 2단계: intermediate → target 위치로 하강
        target_pose_dict = {
            'position': (target_x, target_y, target_z),
            'quaternion': target_quat
        }

        waypoints_descend = self.interpolate_pose_quaternion(
            intermediate_pose_dict, target_pose_dict, self.waypoint_density
        )

        # Z축 클램핑
        clamped_waypoints_descend = []
        for wp in waypoints_descend:
            pos = list(wp['position'])
            if pos[2] < target_z:
                pos[2] = target_z
            clamped_waypoints_descend.append({
                'position': tuple(pos),
                'quaternion': wp['quaternion']
            })

        success = self._execute_waypoints_quaternion(clamped_waypoints_descend, "descend to place")
        if not success:
            self._logger.error("Failed to descend to place position")
            return False

        # 3단계: 그리퍼 열기 (물체 놓기)
        if gripper:
            time.sleep(0.5)
            self.gripper.grasp(0.1, 0)
            time.sleep(1)

        # 4단계: 준비 자세로 복귀
        if home:
            self.go_to_ready_pose()

        return True


