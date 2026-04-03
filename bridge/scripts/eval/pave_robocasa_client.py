#!/usr/bin/env python3
"""
RoboCasa evaluation client for PAVE GROOT ZMQ server.

Runs n_envs parallel RoboCasa environments and communicates with a ZMQ GROOT
server for batched inference. Supports direct mode (obs->action) and pipeline
mode (VLM planner + direction vectors + monitoring).

Usage:
    MUJOCO_GL=egl python3 pave_robocasa_client.py \
        --tasks all --n-envs 5 --n-episodes 17 \
        --max-horizon 1000 --chunk-stride 4 --ema-alpha 0.6 \
        --server-port 5556 --output-dir eval_results/pave_robocasa_groot_ep50 \
        --seed 42
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.expanduser("~/SGRPO/libero_repo"))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq
import msgpack
import msgpack_numpy as m
m.patch()

import robosuite
from robosuite.wrappers import VisualizationWrapper

ALL_TASKS = [
    "CloseDoubleDoor", "CloseDrawer", "CloseSingleDoor",
    "CoffeePressButton", "CoffeeServeMug", "CoffeeSetupMug",
    "OpenDoubleDoor", "OpenDrawer", "OpenSingleDoor",
    "PnPCabToCounter", "PnPCounterToCab", "PnPCounterToMicrowave",
    "PnPCounterToSink", "PnPCounterToStove", "PnPMicrowaveToCounter",
    "PnPSinkToCounter", "PnPStoveToCounter",
    "TurnOffMicrowave", "TurnOffSinkFaucet", "TurnOffStove",
    "TurnOnMicrowave", "TurnOnSinkFaucet", "TurnOnStove",
    "TurnSinkSpout",
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def create_env(task_name, image_size=224, seed=None):
    """Create a single RoboCasa environment."""
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    env_name = (
        "Kitchen"
        if task_name not in REGISTERED_KITCHEN_ENVS
        else REGISTERED_KITCHEN_ENVS[task_name]
    )

    env = robosuite.make(
        env_name,
        robots="PandaMobile",
        controller_configs=robosuite.load_composite_controller_config(
            controller="BASIC",
            robot="PandaMobile",
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_left"],
        camera_heights=image_size,
        camera_widths=image_size,
        camera_depths=False,
        translucent_robot=False,
        seed=seed,
    )
    return env


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def extract_image(obs, size=224):
    """Extract camera image from robosuite observation."""
    key = "robot0_agentview_left_image"
    img = obs[key]  # (H, W, 3) uint8 BGR
    if img.shape[0] != size or img.shape[1] != size:
        from PIL import Image as PILImage
        img = np.array(PILImage.fromarray(img).resize((size, size)))
    return img


def extract_state_12d(obs):
    """Extract 12D state: eef_base(3) + eef_quat(4) + gripper(2) + direction(3)."""
    eef_pos = obs["robot0_base_to_eef_pos"]       # (3,)
    eef_quat = obs["robot0_base_to_eef_quat"]     # (4,)
    gripper = obs["robot0_gripper_qpos"]           # (2,)
    # direction defaults to zeros (set by pipeline for direction_vector)
    direction = np.zeros(3, dtype=np.float32)
    state = np.concatenate([eef_pos, eef_quat, gripper, direction])
    return state.astype(np.float32)


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def action_7d_to_12d(action_7d):
    """Convert 7D action (pos3+rot3+grip1) to 12D for robosuite.

    RoboCasa PandaMobile layout: [arm(6) + gripper(1) + base(5)]
    arm = [pos(3) + rot(3)], gripper = [-1 or 1], base = zeros
    """
    pos = action_7d[:3]
    rot = action_7d[3:6]
    grip = action_7d[6]
    grip_cmd = -1.0 if grip < 0.5 else 1.0
    action_12d = np.zeros(12)
    action_12d[:3] = pos
    action_12d[3:6] = rot
    action_12d[6] = grip_cmd
    # base stays zeros
    return action_12d


# ---------------------------------------------------------------------------
# ZMQ client
# ---------------------------------------------------------------------------

class VLAClient:
    """ZMQ REQ client for the GROOT inference server."""

    def __init__(self, host="127.0.0.1", port=5556):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt(zmq.RCVTIMEO, 120_000)   # 120 s recv timeout
        self.sock.setsockopt(zmq.SNDTIMEO, 30_000)    # 30 s send timeout

    def ping(self):
        self.sock.send(msgpack.packb({"endpoint": "ping"}))
        resp = msgpack.unpackb(self.sock.recv(), raw=False)
        return resp.get("status") == "ok"

    def get_action_batch(self, images, states, instructions):
        """Send batch of (image, state, instruction) and get action chunks.

        Args:
            images: list of (H, W, 3) uint8 arrays
            states: list of (12,) float32 arrays
            instructions: list of str

        Returns:
            list of (chunk_size, 7) action chunk arrays
        """
        req = {
            "endpoint": "get_action_batch",
            "data": {
                "images": images,
                "states": states,
                "instructions": instructions,
            },
        }
        self.sock.send(msgpack.packb(req, use_bin_type=True))
        resp = msgpack.unpackb(self.sock.recv(), raw=False)
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp["action_chunks"]  # list of (chunk_size, 7) arrays

    def close(self):
        self.sock.close()
        self.ctx.term()


# ---------------------------------------------------------------------------
# Direct-mode evaluation
# ---------------------------------------------------------------------------

def evaluate_direct(
    task_name,
    n_envs,
    n_episodes,
    max_horizon,
    chunk_stride,
    ema_alpha,
    image_size,
    server_host,
    server_port,
    seed,
):
    """Run direct-mode evaluation with parallel envs.

    The speedup comes from batched inference on the GPU server, not from
    parallel env stepping (robosuite is not thread-safe).
    """
    client = VLAClient(server_host, server_port)
    assert client.ping(), "Server not responding"

    # Create environments
    envs = [create_env(task_name, image_size, seed=seed) for _ in range(n_envs)]
    print(f"[{task_name}] Created {n_envs} envs, running {n_episodes} episodes "
          f"(horizon={max_horizon}, stride={chunk_stride}, ema={ema_alpha})")

    all_results = []
    episodes_done = 0

    while episodes_done < n_episodes:
        batch_n = min(n_envs, n_episodes - episodes_done)

        # Reset envs
        obs_list = [envs[i].reset() for i in range(batch_n)]

        # Per-env state
        action_chunks = [None] * batch_n
        chunk_idxs = [0] * batch_n
        active = [True] * batch_n
        successes = [False] * batch_n
        steps = [0] * batch_n
        prev_actions = [None] * batch_n

        for step in range(max_horizon):
            if not any(active):
                break

            # Find envs that need new chunks
            needs_chunk = [
                i for i in range(batch_n)
                if active[i] and (
                    action_chunks[i] is None or chunk_idxs[i] >= chunk_stride
                )
            ]

            if needs_chunk:
                images = [extract_image(obs_list[i], image_size) for i in needs_chunk]
                states = [extract_state_12d(obs_list[i]) for i in needs_chunk]
                instructions = ["move"] * len(needs_chunk)

                chunks = client.get_action_batch(images, states, instructions)
                for j, idx in enumerate(needs_chunk):
                    action_chunks[idx] = np.array(chunks[j])
                    chunk_idxs[idx] = 0

            # Step each active env
            for i in range(batch_n):
                if not active[i] or action_chunks[i] is None:
                    continue
                if chunk_idxs[i] >= chunk_stride:
                    continue

                raw_action = action_chunks[i][chunk_idxs[i]].copy()
                chunk_idxs[i] += 1

                # EMA smoothing
                if ema_alpha > 0 and prev_actions[i] is not None:
                    raw_action = (
                        ema_alpha * prev_actions[i] + (1 - ema_alpha) * raw_action
                    )
                prev_actions[i] = raw_action.copy()

                action_12d = action_7d_to_12d(raw_action)
                obs_list[i], reward, done, info = envs[i].step(action_12d)
                steps[i] = step + 1

                try:
                    if envs[i]._check_success():
                        successes[i] = True
                        active[i] = False
                except Exception:
                    pass

        # Record results
        for i in range(batch_n):
            ep_num = episodes_done + i + 1
            tag = "SUCCESS" if successes[i] else "FAIL"
            print(f"  ep{ep_num}/{n_episodes}: {tag} steps={steps[i]}")
            all_results.append({"episode": ep_num, "success": successes[i], "steps": steps[i]})

        episodes_done += batch_n

    # Cleanup
    for env in envs:
        env.close()
    client.close()

    return all_results


# ---------------------------------------------------------------------------
# Pipeline-mode helpers
# ---------------------------------------------------------------------------

def build_state_12d_with_direction(obs, direction_vector=None, target_pos_rb=None):
    """Build 12D state vector with direction_vector injection.

    State layout: eef_base(3) + eef_quat_base(4) + gripper(2) + dir_vec(3)
    Matches pipeline's _build_state_vector() in vla_lora_controller.py.

    Args:
        obs: robosuite observation dict
        direction_vector: [X, Y, Z] floats; Y=None means compute from target_pos_rb
        target_pos_rb: (3,) perception target in robot-base frame (for Y fill)

    Returns:
        (12,) float32 state vector
    """
    from scipy.spatial.transform import Rotation as Rot

    eef_world = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)
    eef_quat_world = np.array(obs.get("robot0_eef_quat", [1, 0, 0, 0]), dtype=np.float64)
    gripper = np.array(obs.get("robot0_gripper_qpos", [0.04, 0.04]), dtype=np.float32)
    base_pos = obs.get("robot0_base_pos", None)
    base_quat = obs.get("robot0_base_quat", None)

    # eef in robot-base frame
    gt_eef_base = obs.get("robot0_base_to_eef_pos", None)
    if gt_eef_base is not None:
        eef_base = np.array(gt_eef_base, dtype=np.float32)
        rot_inv = Rot.from_quat(np.array(base_quat, dtype=np.float64)).inv() if base_quat is not None else None
    elif base_pos is not None and base_quat is not None:
        rot_inv = Rot.from_quat(np.array(base_quat, dtype=np.float64)).inv()
        eef_base = rot_inv.apply(eef_world - np.array(base_pos, dtype=np.float64)).astype(np.float32)
    else:
        eef_base = eef_world.astype(np.float32)
        rot_inv = None

    # quaternion in base frame
    if rot_inv is not None:
        quat_base = (rot_inv * Rot.from_quat(eef_quat_world)).as_quat().astype(np.float32)
    else:
        quat_base = eef_quat_world.astype(np.float32)

    # direction vector (state[9:12] = delta_base slot)
    if direction_vector is not None:
        dir_vec = np.array(
            [float("nan") if v is None else float(v) for v in direction_vector],
            dtype=np.float32,
        )
        # Fill null Y from perception target
        if np.isnan(dir_vec[1]):
            if target_pos_rb is not None:
                dir_vec[1] = float(target_pos_rb[1]) - float(eef_base[1])
            else:
                dir_vec[1] = 0.0
        # Unit-normalize (training data uses unit directions)
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-6:
            dir_vec = dir_vec / norm
    else:
        dir_vec = np.zeros(3, dtype=np.float32)

    return np.concatenate([eef_base, quat_base, gripper[:2], dir_vec]).astype(np.float32)


def get_perception_target_rb(env, obs):
    """Get target fixture position in robot-base frame via RoboCasaPerception.

    Returns None if perception fails.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from robobridge.wrappers.robocasa_perception import RoboCasaPerception
        perc = RoboCasaPerception(env)
        target_world = perc.get_target_pos()
        if target_world is None:
            return None
        # Transform to robot-base frame
        base_pos = obs.get("robot0_base_pos", None)
        base_quat = obs.get("robot0_base_quat", None)
        if base_pos is not None and base_quat is not None:
            from scipy.spatial.transform import Rotation as Rot
            rot_inv = Rot.from_quat(np.array(base_quat, dtype=np.float64)).inv()
            return rot_inv.apply(
                np.array(target_world, dtype=np.float64) - np.array(base_pos, dtype=np.float64)
            ).astype(np.float32)
        return np.array(target_world, dtype=np.float32)
    except Exception:
        return None


def run_pipeline_episode(
    env,
    client,
    task_name,
    plan_config,
    max_horizon,
    chunk_stride,
    ema_alpha,
    image_size,
    stuck_window=80,
    stuck_thresh=0.0005,
    replan_cooldown=100,
):
    """Run one pipeline episode: template plan FSM + direction_vector + stuck monitor.

    Returns: (success: bool, steps: int)
    """
    obs = env.reset()

    # Get primitives for this task
    task_cfg = plan_config.get(task_name)
    if task_cfg is None:
        # Fallback: single move primitive with no direction
        primitives = [{"type": "move", "steps_budget": max_horizon}]
    else:
        primitives = task_cfg.get("primitives", [])

    # Perception target (for null-Y direction_vector fill)
    target_pos_rb = get_perception_target_rb(env, obs)

    step = 0
    success = False

    for prim_idx, prim in enumerate(primitives):
        if step >= max_horizon or success:
            break

        prim_type = prim.get("type", "move")
        budget = prim.get("steps_budget", max_horizon)
        budget = min(budget, max_horizon - step)

        direction_vector = prim.get("direction_vector", prim.get("direction_target"))

        instruction = prim_type  # "move" or "grip"

        # Per-primitive state
        prev_action = None
        chunk_buf = None
        chunk_idx = 0
        prim_step = 0
        stuck_count = 0
        steps_since_start = 0
        prev_pos = None

        while prim_step < budget and step < max_horizon and not success:
            # Build state with direction injection
            state_12d = build_state_12d_with_direction(obs, direction_vector, target_pos_rb)

            # Need new chunk?
            if chunk_buf is None or chunk_idx >= chunk_stride:
                img = extract_image(obs, size=image_size)
                chunks = client.get_action_batch([img], [state_12d], [instruction])
                chunk_buf = np.array(chunks[0])  # (chunk_size, 7)
                chunk_idx = 0

            # Get action from chunk
            raw_action = chunk_buf[chunk_idx].copy()
            chunk_idx += 1

            # EMA smoothing
            if ema_alpha > 0 and prev_action is not None:
                raw_action = ema_alpha * prev_action + (1 - ema_alpha) * raw_action
            prev_action = raw_action.copy()

            # Execute
            action_12d = action_7d_to_12d(raw_action)
            obs, _reward, _done, info = env.step(action_12d)
            step += 1
            prim_step += 1
            steps_since_start += 1

            # Success check
            try:
                if env._check_success():
                    success = True
                    break
            except Exception:
                pass
            if info.get("success", False):
                success = True
                break

            # Stuck monitor (only after warmup)
            cur_pos = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float32)
            if prev_pos is not None and steps_since_start >= replan_cooldown:
                movement = float(np.linalg.norm(cur_pos - prev_pos))
                if movement < stuck_thresh:
                    stuck_count += 1
                else:
                    stuck_count = 0
                if stuck_count >= stuck_window:
                    # Stuck — move to next primitive
                    break
            prev_pos = cur_pos.copy()

    return success, step


# ---------------------------------------------------------------------------
# Pipeline-mode evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    task_name,
    n_envs,
    n_episodes,
    max_horizon,
    chunk_stride,
    ema_alpha,
    image_size,
    server_host,
    server_port,
    seed,
    template_plan_config,
    planner_provider,
    planner_model,
):
    """Run pipeline mode: template plan FSM + direction_vector injection + stuck monitor.

    For each episode:
      1. Load template plan for this task (direction_vector per primitive)
      2. For each primitive in plan:
         a. Inject direction_vector into state_12d[9:12]
         b. Run VLA chunks with EMA smoothing
         c. Stuck detection → advance to next primitive early
      3. Record success/failure
    """
    # Load template plan config
    if isinstance(template_plan_config, str):
        with open(template_plan_config) as f:
            plan_config = json.load(f)
    else:
        plan_config = template_plan_config

    client = VLAClient(server_host, server_port)
    assert client.ping(), "Server not responding"

    envs = [create_env(task_name, image_size, seed=seed) for _ in range(n_envs)]
    print(f"[{task_name}] pipeline mode | {n_envs} envs | {n_episodes} episodes | "
          f"horizon={max_horizon} | stride={chunk_stride} | ema={ema_alpha}")

    # Print plan
    task_plan = plan_config.get(task_name)
    if task_plan:
        prims = task_plan.get("primitives", [])
        print(f"  Plan: {len(prims)} primitives")
        for i, p in enumerate(prims):
            dv = p.get("direction_vector", p.get("direction_target", "none"))
            print(f"    [{i}] type={p['type']} budget={p.get('steps_budget','?')} dir={dv}")
    else:
        print(f"  WARNING: No template plan for {task_name}, using single move primitive")

    all_results = []
    episodes_done = 0

    while episodes_done < n_episodes:
        batch_n = min(n_envs, n_episodes - episodes_done)

        for i in range(batch_n):
            ep_num = episodes_done + i + 1
            t0 = time.time()
            success, steps = run_pipeline_episode(
                envs[i], client, task_name, plan_config,
                max_horizon=max_horizon,
                chunk_stride=chunk_stride,
                ema_alpha=ema_alpha,
                image_size=image_size,
            )
            dt = time.time() - t0
            tag = "SUCCESS" if success else "FAIL"
            print(f"  ep{ep_num}/{n_episodes}: {tag} steps={steps} t={dt:.0f}s")
            all_results.append({"episode": ep_num, "success": success, "steps": steps})

        episodes_done += batch_n

    for env in envs:
        env.close()
    client.close()

    return all_results


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(task_name, mode, episodes, output_dir):
    """Save per-task results as JSON."""
    n_episodes = len(episodes)
    n_successes = sum(1 for e in episodes if e["success"])
    success_rate = n_successes / n_episodes if n_episodes > 0 else 0.0

    result = {
        "task": task_name,
        "mode": mode,
        "n_episodes": n_episodes,
        "successes": n_successes,
        "total_episodes": n_episodes,
        "success_rate": round(success_rate, 4),
        "episodes": episodes,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"eval_{task_name}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{task_name}] {n_successes}/{n_episodes} = {success_rate:.1%}  -> {fname}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RoboCasa eval client for PAVE GROOT ZMQ server"
    )
    p.add_argument("--task", type=str, default=None,
                   help="Single task name (e.g. CloseDrawer)")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task list or 'all' for all 24 tasks")
    p.add_argument("--n-envs", type=int, default=5,
                   help="Number of parallel environments (default 5)")
    p.add_argument("--n-episodes", type=int, default=17,
                   help="Episodes per task (default 17)")
    p.add_argument("--max-horizon", type=int, default=1000,
                   help="Max steps per episode (default 1000)")
    p.add_argument("--chunk-stride", type=int, default=4,
                   help="Chunk stride (default 4)")
    p.add_argument("--ema-alpha", type=float, default=0.6,
                   help="EMA smoothing alpha (default 0.6)")
    p.add_argument("--image-size", type=int, default=224,
                   help="Image size (default 224)")
    p.add_argument("--server-host", type=str, default="127.0.0.1",
                   help="ZMQ server host (default 127.0.0.1)")
    p.add_argument("--server-port", type=int, default=5556,
                   help="ZMQ server port (default 5556)")
    p.add_argument("--output-dir", type=str, default="eval_results/pave_robocasa",
                   help="Output directory for results")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42)")
    p.add_argument("--mode", type=str, default="direct",
                   choices=["direct", "pipeline"],
                   help="Evaluation mode (default direct)")
    # Pipeline-only args
    p.add_argument("--template-plan", type=str, default=None,
                   help="Path to template plan config JSON (pipeline mode)")
    p.add_argument("--planner-provider", type=str, default="bedrock",
                   help="VLM planner provider (pipeline mode)")
    p.add_argument("--planner-model", type=str, default=None,
                   help="VLM planner model ID (pipeline mode)")
    return p.parse_args()


def resolve_tasks(args):
    """Resolve --task / --tasks into a list of task names."""
    if args.task:
        return [args.task]
    if args.tasks:
        if args.tasks.lower() == "all":
            return list(ALL_TASKS)
        return [t.strip() for t in args.tasks.split(",")]
    raise ValueError("Specify --task or --tasks")


def main():
    args = parse_args()
    tasks = resolve_tasks(args)

    print(f"=== PAVE RoboCasa Client ===")
    print(f"Mode: {args.mode} | Tasks: {len(tasks)} | "
          f"n_envs={args.n_envs} | n_episodes={args.n_episodes} | "
          f"horizon={args.max_horizon} | stride={args.chunk_stride} | "
          f"ema={args.ema_alpha} | seed={args.seed}")
    print(f"Server: {args.server_host}:{args.server_port}")
    print(f"Output: {args.output_dir}")
    print()

    summary = {}
    t_start = time.time()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        t_task = time.time()

        try:
            if args.mode == "direct":
                episodes = evaluate_direct(
                    task_name=task,
                    n_envs=args.n_envs,
                    n_episodes=args.n_episodes,
                    max_horizon=args.max_horizon,
                    chunk_stride=args.chunk_stride,
                    ema_alpha=args.ema_alpha,
                    image_size=args.image_size,
                    server_host=args.server_host,
                    server_port=args.server_port,
                    seed=args.seed,
                )
            elif args.mode == "pipeline":
                if not args.template_plan:
                    raise ValueError("--template-plan required for pipeline mode")
                episodes = evaluate_pipeline(
                    task_name=task,
                    n_envs=args.n_envs,
                    n_episodes=args.n_episodes,
                    max_horizon=args.max_horizon,
                    chunk_stride=args.chunk_stride,
                    ema_alpha=args.ema_alpha,
                    image_size=args.image_size,
                    server_host=args.server_host,
                    server_port=args.server_port,
                    seed=args.seed,
                    template_plan_config=args.template_plan,
                    planner_provider=args.planner_provider,
                    planner_model=args.planner_model,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            result = save_results(task, args.mode, episodes, args.output_dir)
            summary[task] = result["success_rate"]
            elapsed = time.time() - t_task
            print(f"[{task}] Done in {elapsed:.0f}s")

        except NotImplementedError as e:
            print(f"[{task}] SKIPPED: {e}")
            summary[task] = None
        except Exception as e:
            print(f"[{task}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            summary[task] = None

    # Print summary
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({elapsed_total:.0f}s total)")
    print(f"{'='*60}")

    completed = {k: v for k, v in summary.items() if v is not None}
    for task, rate in sorted(completed.items()):
        bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
        print(f"  {task:30s} [{bar}] {rate:.1%}")

    if completed:
        avg = sum(completed.values()) / len(completed)
        print(f"\n  Average: {avg:.1%} over {len(completed)} tasks")

    # Save summary JSON
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "n_episodes": args.n_episodes,
        "tasks": summary,
        "average": avg if completed else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
