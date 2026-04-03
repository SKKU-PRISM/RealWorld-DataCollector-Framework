#!/usr/bin/env python3
"""
SO101 Robot Client — VLM 플래너 + GROOT 추론 서버로 SO101 로봇 제어.

VLM 플래너가 카메라 이미지를 보고 세그먼트별 goal_ee를 생성하고,
각 세그먼트에서 direction = normalize(goal_ee - current_ee)를 실시간 계산하여
GROOT VLA에 11D state를 전달합니다.

데이터 흐름:
    카메라 이미지 + instruction
        ↓ VLM Planner (so101_planner.py)
    skill_segments [{type, goal_ee, goal_gripper, instruction, max_steps}, ...]
        ↓ FSM: 세그먼트 순차 실행
    SO101 ee_pos [x,y,z,r,p,y] (6D) + goal_ee
        ↓ direction = normalize(goal_ee - ee_pos[:3])
    state_11d [ee_pos(3) + quat_xyzw(4) + gripper(1) + direction(3)]
        ↓ HTTP POST /predict (+ camera image)
    action_chunk (16, 7) [delta_pos(3) + delta_rot_rpy(3) + gripper(1)]
        ↓ current_eef + delta → new_eef target
    SO101 IK → joint commands

Usage:
    # VLM 플래너 모드 (권장):
    python so101_client.py --port /dev/ttyACM0 \
        --server http://192.168.1.100:8002 \
        --instruction "pick red block and place on blue dish" \
        --planner-provider bedrock \
        --planner-model anthropic.claude-sonnet-4-20250514-v1:0

    # 고정 direction 모드 (레거시, 단일 세그먼트):
    python so101_client.py --port /dev/ttyACM0 \
        --server http://192.168.1.100:8002 \
        --task pick_redblock_place_bluedish \
        --no-planner
"""

import argparse
import base64
import io
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.spatial.transform import Rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("so101_client")

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

ARM_JOINT_NAMES = MOTOR_NAMES


# ---------------------------------------------------------------------------
# State conversion: SO101 ee_pos(6D) → 11D training format
# ---------------------------------------------------------------------------

def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """RPY (3,) → quaternion (4,) in [x, y, z, w] (matching data_so101 format)."""
    rot = Rotation.from_euler("xyz", rpy)
    return rot.as_quat().astype(np.float32)  # scipy default: [x, y, z, w]


def compute_direction(current_ee: np.ndarray, goal_ee: np.ndarray) -> np.ndarray:
    """goal_ee - current_ee → normalized direction (3D).

    학습 데이터의 goal_direction과 동일한 의미:
    현재 위치에서 목표까지의 정규화된 방향 벡터.
    """
    diff = np.array(goal_ee, dtype=np.float32) - np.array(current_ee[:3], dtype=np.float32)
    norm = np.linalg.norm(diff)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (diff / norm).astype(np.float32)


def build_state_11d(
    ee_pos_6d: np.ndarray,
    gripper_value: float,
    goal_direction: np.ndarray,
) -> np.ndarray:
    """SO101 관측 → 학습 포맷 11D state (data_so101 포맷).

    Args:
        ee_pos_6d: [x, y, z, roll, pitch, yaw] from SO101
        gripper_value: gripper scalar (-100=closed, -50=open, etc.)
        goal_direction: (3,) normalized direction to goal

    Returns:
        (11,) = ee_pos(3) + ee_quat_xyzw(4) + gripper(1) + goal_direction(3)
    """
    ee_pos = ee_pos_6d[:3].astype(np.float32)
    ee_rpy = ee_pos_6d[3:6].astype(np.float64)
    ee_quat = rpy_to_quat_xyzw(ee_rpy)

    gripper_1d = np.array([gripper_value], dtype=np.float32)

    return np.concatenate([ee_pos, ee_quat, gripper_1d, goal_direction]).astype(np.float32)


def apply_delta_action(
    current_ee_6d: np.ndarray,
    delta_action: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """7D delta action을 현재 ee_pos에 적용 → 새 ee_pos(6D) + gripper.

    Args:
        current_ee_6d: [x, y, z, r, p, y] 현재 eef
        delta_action: [dx, dy, dz, dr, dp, dy, gripper_binary] (7D)

    Returns:
        new_ee_6d: [x, y, z, r, p, y] 새 eef 목표
        gripper_cmd: gripper value for robot
    """
    new_pos = current_ee_6d[:3] + delta_action[:3]
    new_rpy = current_ee_6d[3:6] + delta_action[3:6]
    # Angle wrap to [-pi, pi]
    new_rpy = (new_rpy + np.pi) % (2 * np.pi) - np.pi

    # gripper is absolute (not delta): 1.0=open, 0.0=closed (학습 데이터 컨벤션)
    # → 로봇: 0=open, 100=closed
    gripper_cmd = 0.0 if delta_action[6] >= 0.5 else 100.0

    return np.concatenate([new_pos, new_rpy]).astype(np.float32), gripper_cmd


# ---------------------------------------------------------------------------
# Inference Server Client
# ---------------------------------------------------------------------------

class InferenceClient:
    def __init__(self, server_url: str, timeout: float = 30.0):
        self.url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def info(self) -> Dict:
        r = self.session.get(f"{self.url}/info", timeout=5)
        return r.json()

    def reset(self):
        self.session.post(f"{self.url}/reset", timeout=5)

    def predict(
        self,
        state: np.ndarray,
        image: Optional[np.ndarray] = None,
        instruction: str = "move",
    ) -> Tuple[np.ndarray, float]:
        """추론 요청 → (action_chunk (16,7), inference_ms)."""
        payload = {
            "state": state.tolist(),
            "instruction": instruction,
        }
        if image is not None:
            payload["image_b64"] = self._encode_image(image)

        r = self.session.post(f"{self.url}/predict", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Server: {data['error']}")

        chunk = np.array(data["action_chunk"], dtype=np.float32)
        return chunk, data.get("inference_ms", 0.0)

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# SO101 Robot
# ---------------------------------------------------------------------------

class SO101Robot:
    def __init__(self, port: str, camera_index: Optional[int] = 0,
                 image_size: int = 224):
        self.port = port
        self.camera_index = camera_index
        self.image_size = image_size
        self.robot = None
        self.cap = None

    def connect(self):
        # Robot
        try:
            from lerobot.robots.so_follower.so_follower import SOFollower
            from lerobot.robots.so_follower.configuration_so_follower import SOFollowerRobotConfig
            config = SOFollowerRobotConfig(port=self.port)
            self.robot = SOFollower(config)
            self.robot.connect()
            logger.info(f"SO101 connected: {self.port}")
        except ImportError:
            from lerobot.motors.feetech.feetech import FeetechMotorsBus, Motor, MotorNormMode
            motors = {
                "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            }
            self.bus = FeetechMotorsBus(port=self.port, motors=motors)
            self.bus.connect()
            logger.info(f"SO101 connected (direct): {self.port}")

        # Camera
        if self.camera_index is not None:
            import cv2
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {self.camera_index} failed")
            logger.info(f"Camera opened: {self.camera_index}")

    def get_ee_pos(self) -> np.ndarray:
        """현재 EEF 위치 (6D: xyz + rpy)."""
        obs = self.robot.get_observation()
        return np.array(obs["observation.state.ee_pos"], dtype=np.float32)

    def get_gripper_state(self) -> float:
        """그리퍼 상태 → 학습 포맷 (0.0=closed, 1.0=open).

        로봇 반환: 0=open, 100=closed
        학습 데이터: 0=closed, 1=open
        변환: 1.0 - (raw / 100.0)
        """
        obs = self.robot.get_observation()
        raw = float(obs.get("observation.state.gripper_position", 0.0))
        return 1.0 - (raw / 100.0)

    def get_image(self) -> Optional[np.ndarray]:
        """카메라 이미지 → (H, W, 3) uint8 RGB."""
        if self.cap is None:
            return None
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[0] != self.image_size or frame_rgb.shape[1] != self.image_size:
            frame_rgb = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        return frame_rgb

    def send_ee_target(self, ee_target_6d: np.ndarray, gripper_cmd: float):
        """EEF 목표 위치 전송 (IK는 로봇 내부에서 처리)."""
        action = {
            "observation.state.ee_pos": ee_target_6d.tolist(),
            "observation.state.gripper_position": gripper_cmd,
        }
        self.robot.send_action(action)

    def send_joint_action(self, joint_targets: np.ndarray):
        """관절 각도 직접 전송."""
        action = {
            f"{name}.pos": float(joint_targets[i])
            for i, name in enumerate(MOTOR_NAMES)
        }
        self.robot.send_action(action)

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
        if self.robot is not None:
            self.robot.disconnect()
        logger.info("SO101 disconnected")


# ---------------------------------------------------------------------------
# Segment FSM execution
# ---------------------------------------------------------------------------

def run_segment(
    robot: SO101Robot,
    client: InferenceClient,
    segment: Dict,
    chunk_stride: int,
    control_freq: float,
    ema_alpha: float,
    converge_threshold: float = 0.015,
) -> bool:
    """단일 세그먼트 실행.

    Args:
        segment: {type, goal_ee, goal_gripper, instruction, max_steps}
        converge_threshold: goal_ee와의 거리가 이 이하면 수렴 판정 (meters)

    Returns:
        True if converged, False if timed out
    """
    seg_type = segment["type"]
    goal_ee = np.array(segment["goal_ee"], dtype=np.float32)
    goal_gripper = segment.get("goal_gripper", "open")
    instruction = segment.get("instruction", seg_type)
    max_steps = segment.get("max_steps", 150)

    dt = 1.0 / control_freq
    chunk_buf: Optional[np.ndarray] = None
    chunk_idx = 0
    prev_action: Optional[np.ndarray] = None

    # Reset VLA state at segment start
    client.reset()

    logger.info(f"  Segment: {seg_type} | goal_ee={goal_ee.round(4).tolist()} "
                f"grip={goal_gripper} | max={max_steps} | {instruction}")

    # For gripper-only segments, just send gripper command
    if seg_type in ("gripper_open", "gripper_close"):
        gripper_cmd = 0.0 if seg_type == "gripper_open" else 100.0
        ee_pos_6d = robot.get_ee_pos()
        robot.send_ee_target(ee_pos_6d, gripper_cmd)
        time.sleep(max_steps / control_freq)  # Wait for gripper action
        logger.info(f"    Gripper {'opened' if 'open' in seg_type else 'closed'}")
        return True

    for step in range(1, max_steps + 1):
        t0 = time.time()

        # 새 chunk 필요?
        if chunk_buf is None or chunk_idx >= chunk_stride:
            ee_pos_6d = robot.get_ee_pos()
            gripper = robot.get_gripper_state()
            image = robot.get_image()

            # Direction: normalize(goal_ee - current_ee)
            direction = compute_direction(ee_pos_6d[:3], goal_ee)

            # 11D state (data_so101 포맷)
            state = build_state_11d(ee_pos_6d, gripper, direction)

            # 서버 추론
            chunk_buf, inf_ms = client.predict(state, image, instruction)
            chunk_idx = 0

            # 수렴 체크
            dist = np.linalg.norm(ee_pos_6d[:3] - goal_ee)
            if step <= 3 or step % 30 == 0:
                logger.info(
                    f"    step={step:>4d} ee={ee_pos_6d[:3].round(4).tolist()} "
                    f"dir={direction.round(3).tolist()} dist={dist:.4f} inf={inf_ms:.0f}ms"
                )

            if dist < converge_threshold and step > 10:
                logger.info(f"    Converged at step {step} (dist={dist:.4f})")
                return True

        # Chunk에서 delta action 꺼내기
        delta = chunk_buf[chunk_idx].copy()
        chunk_idx += 1

        # EMA 스무딩
        if ema_alpha > 0 and prev_action is not None:
            delta = ema_alpha * prev_action + (1 - ema_alpha) * delta
        prev_action = delta.copy()

        # Delta action 적용
        ee_pos_6d = robot.get_ee_pos()
        new_ee, gripper_cmd = apply_delta_action(ee_pos_6d, delta)

        # 세그먼트의 gripper 의도 적용 (VLA 출력보다 우선)
        if goal_gripper == "close":
            gripper_cmd = 100.0
        elif goal_gripper == "open":
            gripper_cmd = 0.0

        robot.send_ee_target(new_ee, gripper_cmd)

        # 주파수 유지
        elapsed = time.time() - t0
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    logger.info(f"    Timeout at step {max_steps}")
    return False


def run_episode_with_planner(
    robot: SO101Robot,
    client: InferenceClient,
    planner,
    instruction: str,
    chunk_stride: int,
    control_freq: float,
    ema_alpha: float,
    max_attempts: int = 2,
    converge_threshold: float = 0.015,
) -> Dict:
    """VLM 플래너 기반 에피소드 실행.

    Args:
        planner: SO101Planner instance
        instruction: 자연어 태스크 명령
        max_attempts: 실패 시 재계획 최대 횟수

    Returns:
        {"segments_total": N, "segments_completed": M, "success": bool}
    """
    logger.info(f"{'='*60}")
    logger.info(f"Episode: '{instruction}' (planner mode)")
    logger.info(f"{'='*60}")

    for attempt in range(1, max_attempts + 1):
        # 카메라 이미지 캡처
        image = robot.get_image()
        ee_pos = robot.get_ee_pos()

        # VLM 플래너로 세그먼트 생성
        if attempt == 1:
            segments = planner.plan(instruction, image, ee_pos[:3].tolist())
        else:
            failure_ctx = f"Attempt {attempt-1} failed at segment {failed_seg_idx}: {failed_seg_instr}"
            segments = planner.replan(instruction, image, ee_pos[:3].tolist(), failure_ctx)

        if not segments:
            logger.error("Planner returned empty plan")
            continue

        logger.info(f"Attempt {attempt}: {len(segments)} segments")

        # FSM: 세그먼트 순차 실행
        completed = 0
        failed_seg_idx = 0
        failed_seg_instr = ""

        for i, seg in enumerate(segments):
            logger.info(f"\n--- Segment {i+1}/{len(segments)} ---")
            success = run_segment(
                robot, client, seg,
                chunk_stride, control_freq, ema_alpha,
                converge_threshold,
            )
            if success:
                completed += 1
            else:
                # move_free (마지막 복귀)는 타임아웃 허용
                if seg["type"] == "move_free":
                    completed += 1
                else:
                    failed_seg_idx = i
                    failed_seg_instr = seg.get("instruction", seg["type"])
                    logger.warning(f"Segment {i+1} failed: {failed_seg_instr}")
                    # 그래도 다음 세그먼트로 진행 (soft failure)
                    completed += 1

        result = {
            "segments_total": len(segments),
            "segments_completed": completed,
            "attempt": attempt,
            "success": completed == len(segments),
        }
        logger.info(f"\nAttempt {attempt} result: {completed}/{len(segments)} segments completed")

        if result["success"]:
            return result

    return result


def run_episode_fixed_direction(
    robot: SO101Robot,
    client: InferenceClient,
    instruction: str,
    direction: np.ndarray,
    max_steps: int,
    chunk_stride: int,
    control_freq: float,
    ema_alpha: float,
) -> int:
    """고정 direction 모드 에피소드 실행 (레거시 호환)."""
    dt = 1.0 / control_freq
    chunk_buf: Optional[np.ndarray] = None
    chunk_idx = 0
    prev_action: Optional[np.ndarray] = None

    client.reset()
    logger.info(f"Episode start (fixed direction): '{instruction}' | "
                f"dir={direction.round(4).tolist()} stride={chunk_stride}")

    for step in range(1, max_steps + 1):
        t0 = time.time()

        if chunk_buf is None or chunk_idx >= chunk_stride:
            ee_pos_6d = robot.get_ee_pos()
            gripper = robot.get_gripper_state()
            image = robot.get_image()

            state = build_state_11d(ee_pos_6d, gripper, direction)
            chunk_buf, inf_ms = client.predict(state, image, instruction)
            chunk_idx = 0

            if step <= 3 or step % 50 == 0:
                logger.info(
                    f"  step={step} ee={ee_pos_6d[:3].round(4).tolist()} "
                    f"chunk=({chunk_buf.shape[0]},{chunk_buf.shape[1]}) inf={inf_ms:.0f}ms"
                )

        delta = chunk_buf[chunk_idx].copy()
        chunk_idx += 1

        if ema_alpha > 0 and prev_action is not None:
            delta = ema_alpha * prev_action + (1 - ema_alpha) * delta
        prev_action = delta.copy()

        ee_pos_6d = robot.get_ee_pos()
        new_ee, gripper_cmd = apply_delta_action(ee_pos_6d, delta)
        robot.send_ee_target(new_ee, gripper_cmd)

        elapsed = time.time() - t0
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    logger.info(f"Episode done: {max_steps} steps")
    return max_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO101 Client — VLM 플래너 + GROOT 서버로 로봇 제어")

    # Robot
    parser.add_argument("--port", required=True, help="SO101 시리얼 포트")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-camera", action="store_true")

    # Server
    parser.add_argument("--server", required=True, help="GPU 추론 서버 URL")

    # Planner
    parser.add_argument("--planner-provider", default="bedrock",
                        help="VLM 플래너 프로바이더 (bedrock, anthropic, openai, google)")
    parser.add_argument("--planner-model", default="anthropic.claude-sonnet-4-20250514-v1:0",
                        help="VLM 플래너 모델")
    parser.add_argument("--no-planner", action="store_true",
                        help="VLM 플래너 비활성화 (고정 direction 모드)")

    # Task
    parser.add_argument("--instruction", required=True, help="자연어 태스크 명령")
    parser.add_argument("--task", default=None,
                        help="태스크 이름 (--no-planner 시 so101_task_config.json에서 direction 로딩)")
    parser.add_argument("--task-config", default="so101_task_config.json")
    parser.add_argument("--direction", type=float, nargs=3, default=None,
                        help="Direction vector 수동 지정 (--no-planner 전용)")

    # Control
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="고정 direction 모드의 최대 스텝")
    parser.add_argument("--chunk-stride", type=int, default=8)
    parser.add_argument("--control-freq", type=float, default=30.0)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--converge-threshold", type=float, default=0.015,
                        help="세그먼트 수렴 판정 거리 (meters)")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="실패 시 재계획 최대 횟수")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args()

    # Server check
    client = InferenceClient(args.server, timeout=args.timeout)
    logger.info(f"Server: {args.server}")
    if not client.health():
        logger.error("Server not responding. Start so101_server.py first.")
        sys.exit(1)

    server_info = client.info()
    logger.info(f"Server info: {server_info}")

    # Robot
    cam_idx = None if args.no_camera else args.camera_index
    robot = SO101Robot(args.port, cam_idx, args.image_size)

    _running = [True]
    def _sig(s, f):
        _running[0] = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Planner
    planner = None
    if not args.no_planner:
        from so101_planner import SO101Planner
        planner = SO101Planner(
            provider=args.planner_provider,
            model=args.planner_model,
        )
        logger.info(f"Planner: {args.planner_provider}/{args.planner_model}")

    try:
        robot.connect()

        for ep in range(1, args.num_episodes + 1):
            if not _running[0]:
                break
            logger.info(f"\n{'='*60}\nEpisode {ep}/{args.num_episodes}\n{'='*60}")
            if ep > 1:
                input("Enter to continue...")

            if planner is not None:
                # VLM 플래너 모드: 세그먼트별 FSM 실행
                result = run_episode_with_planner(
                    robot, client, planner, args.instruction,
                    args.chunk_stride, args.control_freq, args.ema_alpha,
                    args.max_attempts, args.converge_threshold,
                )
                logger.info(f"Episode {ep} result: {result}")
            else:
                # 고정 direction 모드 (레거시)
                if args.direction is not None:
                    direction = np.array(args.direction, dtype=np.float32)
                elif args.task:
                    with open(args.task_config) as f:
                        task_cfg = json.load(f)
                    if args.task not in task_cfg:
                        logger.error(f"Task '{args.task}' not in {args.task_config}")
                        sys.exit(1)
                    direction = np.array(task_cfg[args.task]["direction"], dtype=np.float32)
                else:
                    direction = np.zeros(3, dtype=np.float32)
                    logger.warning("No direction specified — using [0,0,0]")

                run_episode_fixed_direction(
                    robot, client, args.instruction, direction,
                    args.max_steps, args.chunk_stride,
                    args.control_freq, args.ema_alpha,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
