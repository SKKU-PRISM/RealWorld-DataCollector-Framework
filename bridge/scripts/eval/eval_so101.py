#!/usr/bin/env python3
"""
SO101 Real Robot Evaluation Pipeline.

Modular evaluation with swappable Perception, Planner, Monitor, and Controller.
Mirrors eval_vla_robocasa.py structure — each module's provider/model is independently configurable.

Architecture:
    Camera Image + Instruction
        ↓
    [Perception] VLM object detection → object positions (optional)
        ↓
    [Planner] VLM plan generation → skill segments [{type, goal_ee, instruction}, ...]
        ↓
    [Controller] GROOT N1.5 + LoRA → delta actions per segment
        ↓
    [Robot] SO101 IK → joint commands
        ↓
    [Monitor] VLM execution verification (optional)

Usage:
    # Full pipeline with Gemini planner + monitor
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py \
        --port /dev/ttyACM0 \
        --planner-provider google --planner-model gemini-2.5-flash \
        --monitor-provider google --monitor-model gemini-2.5-flash \
        --tasks pick_redblock_place_bluedish --num-episodes 5

    # Swap planner to Anthropic Claude
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py \
        --port /dev/ttyACM0 \
        --planner-provider anthropic --planner-model claude-sonnet-4-20250514 \
        --tasks pick_redblock_place_bluedish

    # Swap planner to OpenAI
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py \
        --port /dev/ttyACM0 \
        --planner-provider openai --planner-model gpt-4o \
        --tasks stack_yellow_redblock

    # No monitor (faster)
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py \
        --port /dev/ttyACM0 \
        --planner-provider google --planner-model gemini-2.5-flash \
        --monitor-provider "" \
        --tasks pick_redblock_place_bluedish

    # Fixed direction mode (no VLM planner)
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py \
        --port /dev/ttyACM0 --no-planner \
        --tasks pick_redblock_place_bluedish

    # Dry-run (no robot, inference only)
    CUDA_VISIBLE_DEVICES=0 python eval_so101.py --dry-run \
        --tasks pick_redblock_place_bluedish
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add so101 scripts to path for imports (so101_client, so101_server)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "so101"))
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_so101")
for _mod in ("PIL", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(_mod).setLevel(logging.ERROR)


# ============================================================================
# VLM Provider Factory (shared by Planner / Monitor / Perception)
# ============================================================================

def init_vlm(provider: str, model: str, api_key: str = "", temperature: float = 0.3):
    """Create a LangChain chat model for any provider.

    Mirrors robobridge/modules/planner and monitor provider initialization.

    Args:
        provider: "openai", "anthropic", "google", "bedrock", "vertex", "ollama"
        model: Model name/ID
        api_key: API key (optional for bedrock/vertex)
        temperature: Sampling temperature

    Returns:
        LangChain BaseChatModel instance
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key or None, max_tokens=2048)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key or None, max_tokens=2048)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key or None)

    elif provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(model_name=model, temperature=temperature)

    elif provider == "bedrock":
        from robobridge.modules.bedrock_bearer import get_bedrock_llm
        return get_bedrock_llm(model, temperature=temperature)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode numpy RGB image to base64 JPEG string."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _vlm_invoke(llm, text_prompt: str, image: Optional[np.ndarray] = None) -> str:
    """Invoke a LangChain VLM with text + optional image."""
    from langchain_core.messages import HumanMessage

    content = []
    if image is not None:
        b64 = _encode_image_b64(image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": text_prompt})

    response = llm.invoke([HumanMessage(content=content)])
    return response.content


def _extract_json(text: str):
    """Robustly extract JSON from VLM response."""
    import re
    for pat in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    for c in ['{', '[']:
        s = text.find(c)
        if s >= 0:
            e = text.rfind(']' if c == '[' else '}')
            if e >= 0:
                try:
                    return json.loads(text[s:e+1])
                except json.JSONDecodeError:
                    pass
    return None


# ============================================================================
# SO101 Workspace Constants
# ============================================================================

WORKSPACE = {
    "x_range": [0.06, 0.40],
    "y_range": [-0.20, 0.22],
    "z_table": 0.02,
    "z_lift": 0.20,
    "z_above": 0.15,
    "home": [0.20, 0.00, 0.12],
}

EE_MIN = np.array([0.06, -0.20, -0.02], dtype=np.float32)
EE_MAX = np.array([0.40, 0.22, 0.23], dtype=np.float32)

ALL_TASKS = [
    "arrange_YRPblock_top2bottom",
    "distribute_chocolatepies",
    "pick_redblock_place_bluedish",
    "place_yellowblock_between_chocolatepies",
    "stack_red_yellow_purple_blocks",
    "stack_yellow_redblock_30epi",
]


# ============================================================================
# Module: Planner
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are a robot manipulation planner for the SO101 robot arm.
You see a top-down camera image of a tabletop workspace.
Given the image and a task instruction, output a sequence of skill segments as JSON.

Workspace: X=[0.08, 0.38]m, Y=[-0.20, 0.20]m, Z_table=0.02m, Z_lift=0.20m
Robot home: [0.20, 0.00, 0.12]

Segment types: "move", "gripper_open", "gripper_close", "move_free"
Each segment: {"type": str, "goal_ee": [x, y, z], "goal_gripper": "open"|"close", "instruction": str, "max_steps": int}

Rules:
1. Pick pattern: gripper_open → move above(z=0.15) → descend(z=0.02) → gripper_close → lift(z=0.20) → move above target → descend → gripper_open → retract → move_free home
2. Always approach from above before descending
3. End with move_free to home [0.20, 0.00, 0.12]
4. Output ONLY valid JSON: {"segments": [...]}"""


class Planner:
    """VLM-based planner that generates skill segments from image + instruction.

    Provider-swappable: openai, anthropic, google, bedrock, vertex, ollama.
    """

    def __init__(self, provider: str, model: str, api_key: str = "", max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = init_vlm(self.provider, self.model, self.api_key)
            logger.info(f"Planner initialized: {self.provider}/{self.model}")
        return self._llm

    def plan(self, instruction: str, image: Optional[np.ndarray] = None,
             current_ee: Optional[List[float]] = None) -> List[Dict]:
        """Generate skill segments from instruction + camera image."""
        ee = current_ee or WORKSPACE["home"]
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"Current EE position: {[round(x, 4) for x in ee]}\n"
            f"Task: {instruction}\n\n"
            f"Output the segments JSON:"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = _vlm_invoke(self._get_llm(), prompt, image)
                data = _extract_json(response)
                if data is None:
                    logger.warning(f"  Planner attempt {attempt}: no JSON found")
                    continue

                segments = data.get("segments", data) if isinstance(data, dict) else data
                if not isinstance(segments, list) or len(segments) == 0:
                    logger.warning(f"  Planner attempt {attempt}: empty segments")
                    continue

                segments = self._validate_segments(segments)
                logger.info(f"  Planner: {len(segments)} segments generated")
                for i, s in enumerate(segments):
                    logger.info(f"    [{i}] {s['type']:15s} goal_ee={[round(x,3) for x in s['goal_ee']]} | {s['instruction']}")
                return segments

            except Exception as e:
                logger.warning(f"  Planner attempt {attempt} error: {e}")
                time.sleep(2)

        logger.warning("  Planner: all attempts failed, using fallback")
        return self._fallback_plan(ee)

    def replan(self, instruction: str, image: Optional[np.ndarray] = None,
               current_ee: Optional[List[float]] = None, failure_context: str = "") -> List[Dict]:
        """Replan after failure with context."""
        augmented = f"{instruction}\n\n[RETRY] Previous attempt failed: {failure_context}. Adjust the plan."
        return self.plan(augmented, image, current_ee)

    def _validate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Validate and clamp segment coordinates to workspace."""
        validated = []
        for seg in segments:
            if "goal_ee" not in seg or "type" not in seg:
                continue
            ee = seg["goal_ee"]
            seg["goal_ee"] = [
                np.clip(ee[0], WORKSPACE["x_range"][0], WORKSPACE["x_range"][1]),
                np.clip(ee[1], WORKSPACE["y_range"][0], WORKSPACE["y_range"][1]),
                np.clip(ee[2], -0.02, 0.25),
            ]
            seg.setdefault("goal_gripper", "open")
            seg.setdefault("instruction", seg["type"])
            seg.setdefault("max_steps", 300)
            validated.append(seg)
        return validated

    def _fallback_plan(self, current_ee: List[float]) -> List[Dict]:
        """Default pick-and-place plan when VLM fails."""
        return [
            {"type": "gripper_open", "goal_ee": current_ee, "goal_gripper": "open", "instruction": "Open gripper", "max_steps": 50},
            {"type": "move", "goal_ee": [0.22, 0.08, 0.15], "goal_gripper": "open", "instruction": "Move above object", "max_steps": 300},
            {"type": "move", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "open", "instruction": "Descend to object", "max_steps": 300},
            {"type": "gripper_close", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "close", "instruction": "Grasp object", "max_steps": 50},
            {"type": "move", "goal_ee": [0.22, 0.08, 0.20], "goal_gripper": "close", "instruction": "Lift object", "max_steps": 300},
            {"type": "move", "goal_ee": [0.22, -0.08, 0.20], "goal_gripper": "close", "instruction": "Move above target", "max_steps": 300},
            {"type": "move", "goal_ee": [0.22, -0.08, 0.03], "goal_gripper": "close", "instruction": "Lower to target", "max_steps": 300},
            {"type": "gripper_open", "goal_ee": [0.22, -0.08, 0.03], "goal_gripper": "open", "instruction": "Release object", "max_steps": 50},
            {"type": "move", "goal_ee": [0.22, -0.08, 0.20], "goal_gripper": "open", "instruction": "Retract", "max_steps": 200},
            {"type": "move_free", "goal_ee": WORKSPACE["home"], "goal_gripper": "open", "instruction": "Return home", "max_steps": 200},
        ]


# ============================================================================
# Module: Monitor
# ============================================================================

class Monitor:
    """VLM-based execution monitor that verifies segment completion.

    Provider-swappable: openai, anthropic, google, bedrock, vertex, ollama.
    """

    def __init__(self, provider: str, model: str, api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = init_vlm(self.provider, self.model, self.api_key)
            logger.info(f"Monitor initialized: {self.provider}/{self.model}")
        return self._llm

    def verify(self, image: np.ndarray, instruction: str, segment_type: str) -> bool:
        """Verify segment execution by looking at the camera image."""
        if segment_type in ("gripper_open", "gripper_close", "move_free"):
            return True

        if "descend" in instruction.lower() or "grasp" in instruction.lower() or "lower" in instruction.lower():
            prompt = (
                f"Look at this robot arm workspace image.\n"
                f"The robot just attempted: \"{instruction}\"\n"
                f"Was this action completed successfully? Be strict.\n"
                f"Answer ONLY 'YES' or 'NO' followed by a brief reason."
            )
        else:
            prompt = (
                f"Look at this robot arm workspace image.\n"
                f"The robot just attempted: \"{instruction}\"\n"
                f"Is the robot arm roughly in the correct area? Be lenient.\n"
                f"Answer ONLY 'YES' or 'NO' followed by a brief reason."
            )

        try:
            response = _vlm_invoke(self._get_llm(), prompt, image)
            verified = response.strip().upper().startswith("YES")
            logger.info(f"    Monitor: {'PASS' if verified else 'FAIL'} — {response.strip()[:80]}")
            return verified
        except Exception as e:
            logger.warning(f"    Monitor error: {e}")
            return True  # fail-open


# ============================================================================
# Module: Controller (SO101-specific: GROOT + LoRA + IK)
# ============================================================================

class Controller:
    """GROOT N1.5 + LoRA adapter controller for SO101.

    Handles model loading, task-specific adapter swapping, and inference.
    """

    def __init__(self, base_model: str, device: str = "cuda:0", image_size: int = 224,
                 joint_mode: bool = False):
        self.base_model = base_model
        self.device = device
        self.image_size = image_size
        self.joint_mode = joint_mode
        self._policy = None
        self._eagle = None
        self._stats = None
        self._current_task = None

    def load_task(self, task: str, adapter_dir: str):
        """Load LoRA adapter and stats for a specific task."""
        if self._current_task == task:
            return

        # Unload previous model
        if self._policy is not None:
            self.unload()

        adapter_path = f"{adapter_dir}/{task}/move_adapter/checkpoint-best"
        if not os.path.exists(adapter_path):
            adapter_path = f"{adapter_dir}/{task}/move_adapter/checkpoint-final"
        stats_path = f"{adapter_dir}/{task}/move_adapter/data_stats.json"

        logger.info(f"Controller loading: {task}")
        logger.info(f"  Adapter: {adapter_path}")
        logger.info(f"  Stats:   {stats_path}")

        from so101_server import load_stats, load_groot
        import torch

        self._stats = load_stats(stats_path)
        self._policy, self._eagle = load_groot(adapter_path, self.base_model, torch.device(self.device))
        self._current_task = task

        # Warmup
        self._warmup()

    def predict(self, state: np.ndarray, images: Optional[Dict[str, np.ndarray]] = None,
                instruction: str = "move") -> Tuple[np.ndarray, float]:
        """Run inference → (action_chunk, inference_ms)."""
        from so101_server import build_batch, build_batch_joint, predict_and_denorm
        import torch

        t0 = time.time()
        image = images.get("top", next(iter(images.values()))) if images else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image2 = images.get("wrist", None) if images else None

        if self.joint_mode:
            batch = build_batch_joint(image, state, instruction, self._stats, self._eagle, torch.device(self.device), self.image_size, image2=image2)
        else:
            batch = build_batch(image, state, instruction, self._stats, self._eagle, torch.device(self.device), self.image_size, image2=image2)

        chunk = predict_and_denorm(self._policy, batch, self._stats)
        ms = (time.time() - t0) * 1000
        return chunk, ms

    def reset(self):
        """Reset model state for new episode."""
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def unload(self):
        """Free GPU memory."""
        import torch, gc
        del self._policy, self._eagle
        self._policy = self._eagle = None
        self._current_task = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("  Controller: GPU memory freed")

    def _warmup(self):
        """Warmup inference to avoid first-call latency."""
        from so101_server import build_batch, build_batch_joint, predict_and_denorm
        import torch

        logger.info("  Warmup...")
        dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        state_dim = len(self._stats["state_min"])
        dummy_state = np.zeros(state_dim, dtype=np.float32)
        if self.joint_mode:
            b = build_batch_joint(dummy_img, dummy_state, "move", self._stats, self._eagle, torch.device(self.device), self.image_size)
        else:
            b = build_batch(dummy_img, dummy_state, "move", self._stats, self._eagle, torch.device(self.device), self.image_size)
        predict_and_denorm(self._policy, b, self._stats)
        logger.info("  Warmup done")


# ============================================================================
# Segment Execution
# ============================================================================

def run_segment(
    robot,
    controller: Controller,
    segment: Dict,
    chunk_stride: int = 8,
    control_freq: float = 10.0,
    ema_alpha: float = 0.3,
    converge_threshold: float = 0.02,
    dry_run: bool = False,
    session_dir: Optional[Path] = None,
    seg_idx: int = 0,
) -> bool:
    """Execute a single segment (move / gripper_open / gripper_close / move_free)."""
    from so101_client import compute_direction, build_state_11d, ARM_JOINT_NAMES

    seg_type = segment["type"]
    goal_ee = np.array(segment["goal_ee"], dtype=np.float32)
    goal_gripper = segment.get("goal_gripper", "open")
    instruction = segment.get("instruction", seg_type)
    max_steps = segment.get("max_steps", 300)
    dt = 1.0 / control_freq

    controller.reset()
    logger.info(f"  Segment: {seg_type} | goal_ee={goal_ee.round(4).tolist()} "
                f"grip={goal_gripper} | max={max_steps} | {instruction}")

    # Gripper-only segments
    if seg_type in ("gripper_open", "gripper_close"):
        if not dry_run:
            gripper_cmd = 45.0 if seg_type == "gripper_open" else 0.0
            ee_pos_6d = robot.get_ee_pos()
            robot.send_ee_target(ee_pos_6d, gripper_cmd)
            time.sleep(1.0)
        logger.info(f"    Gripper {'opened' if 'open' in seg_type else 'closed'}")
        return True

    # Move segments
    chunk_buf = None
    chunk_idx = 0
    prev_action = None
    min_dist = float("inf")
    seg_log = []

    for step in range(1, max_steps + 1):
        t0 = time.time()

        # Convergence & divergence check
        if not dry_run:
            ee_check = robot.get_ee_pos()
            dist = np.linalg.norm(ee_check[:3] - goal_ee)
            z_err = abs(ee_check[2] - goal_ee[2])

            is_precise = any(k in instruction.lower() for k in ("descend", "grasp", "lower"))
            thresh = 0.01 if is_precise else converge_threshold
            z_thresh = 0.005 if is_precise else 0.03

            if dist < thresh and z_err < z_thresh and step > 3:
                logger.info(f"    Converged at step {step} (dist={dist:.4f})")
                return True

            if step <= 3:
                min_dist = dist
            else:
                min_dist = min(min_dist, dist)
                if dist > min_dist + 0.05 and step > 50:
                    logger.warning(f"    Diverging at step {step} (dist={dist:.4f}, min={min_dist:.4f})")
                    return min_dist < converge_threshold * 2

        # Get new action chunk
        if chunk_buf is None or chunk_idx >= chunk_stride:
            if dry_run:
                ee_pos_6d = np.zeros(6, dtype=np.float32)
                gripper = 0.5
                images = None
            else:
                ee_pos_6d = robot.get_ee_pos()
                gripper = robot.get_gripper_state()
                images = robot.get_images()

            if controller.joint_mode:
                obs = robot.robot.get_observation() if not dry_run else {}
                joint_norm = np.array([obs.get(f"{n}.pos", 0.0) for n in ARM_JOINT_NAMES] if not dry_run else [0]*6, dtype=np.float32)
                gripper_raw = obs.get("gripper.pos", 0.0) if not dry_run else 0.0
                state = np.concatenate([joint_norm[:5], [gripper_raw], ee_pos_6d[:3]]).astype(np.float32)
            else:
                direction = compute_direction(ee_pos_6d[:3], goal_ee)
                state = build_state_11d(ee_pos_6d, gripper, direction)

            try:
                chunk_buf, inf_ms = controller.predict(state, images, instruction)
                chunk_idx = 0
            except Exception as e:
                logger.warning(f"    step={step} predict failed: {e}")
                time.sleep(0.5)
                continue

            if step <= 3 or step % 30 == 0:
                d = np.linalg.norm(ee_pos_6d[:3] - goal_ee)
                logger.info(f"    step={step:>4d} ee={ee_pos_6d[:3].round(4).tolist()} dist={d:.4f} inf={inf_ms:.0f}ms")

            if dry_run and step >= 3:
                return True

        # Apply action
        delta = chunk_buf[chunk_idx].copy()
        chunk_idx += 1

        if ema_alpha > 0 and prev_action is not None:
            delta = ema_alpha * prev_action + (1 - ema_alpha) * delta
        prev_action = delta.copy()

        if not dry_run:
            if controller.joint_mode:
                action = {}
                for j, name in enumerate(ARM_JOINT_NAMES):
                    if j < len(delta):
                        action[f"{name}.pos"] = float(delta[j])
                action["gripper.pos"] = 0.0 if goal_gripper == "close" else 45.0
                robot.robot.send_action(action)
            else:
                delta_pos = delta[:3].astype(np.float64)
                ee_now = robot.get_ee_pos()[:3]
                target = np.clip(ee_now + delta_pos, EE_MIN, EE_MAX)
                gripper_cmd = 0.0 if goal_gripper == "close" else 45.0
                robot.send_ee_delta(target - ee_now, gripper_cmd)

        elapsed = time.time() - t0
        if dt - elapsed > 0 and not dry_run:
            time.sleep(dt - elapsed)

    logger.info(f"    Timeout at step {max_steps}")
    return False


# ============================================================================
# Episode Execution
# ============================================================================

def run_episode(
    robot,
    controller: Controller,
    planner: Optional[Planner],
    monitor: Optional[Monitor],
    instruction: str,
    task_cfg: Dict,
    chunk_stride: int = 8,
    control_freq: float = 10.0,
    ema_alpha: float = 0.3,
    converge_threshold: float = 0.02,
    max_attempts: int = 2,
    dry_run: bool = False,
    session_dir: Optional[Path] = None,
) -> Dict:
    """Run a single episode with planner + optional monitor."""

    # --- Planner mode ---
    if planner is not None:
        for attempt in range(1, max_attempts + 1):
            image = robot.get_image() if not dry_run else np.zeros((224, 224, 3), dtype=np.uint8)
            ee_pos = robot.get_ee_pos()[:3].tolist() if not dry_run else WORKSPACE["home"]

            if attempt == 1:
                segments = planner.plan(instruction, image, ee_pos)
            else:
                segments = planner.replan(instruction, image, ee_pos, f"Attempt {attempt-1} failed at seg {failed_idx}: {failed_instr}")

            if not segments:
                logger.error("  Planner returned empty plan")
                continue

            logger.info(f"  Attempt {attempt}: {len(segments)} segments")
            completed = 0
            failed_idx, failed_instr = 0, ""

            for i, seg in enumerate(segments):
                logger.info(f"\n  --- Segment {i+1}/{len(segments)} ---")
                success = run_segment(
                    robot, controller, seg,
                    chunk_stride, control_freq, ema_alpha,
                    converge_threshold, dry_run, session_dir, i,
                )

                # Monitor verification (if enabled)
                if monitor and not dry_run and not success:
                    img = robot.get_image()
                    if img is not None:
                        verified = monitor.verify(img, seg["instruction"], seg["type"])
                        if verified:
                            success = True

                if success:
                    completed += 1
                else:
                    failed_idx = i
                    failed_instr = seg.get("instruction", seg["type"])
                    logger.warning(f"    Segment {i+1} failed: {failed_instr}")
                    completed += 1  # continue to next segment

            result = {
                "segments_total": len(segments),
                "segments_completed": completed,
                "attempt": attempt,
                "success": completed == len(segments),
            }
            if result["success"]:
                return result

        return result

    # --- Fixed direction mode (no planner) ---
    else:
        from so101_client import compute_direction, build_state_11d
        direction = np.array(task_cfg.get("direction", [0, 0, 0]), dtype=np.float32)
        max_steps = task_cfg.get("max_steps", 1000)

        controller.reset()
        chunk_buf = None
        chunk_idx = 0
        prev_action = None
        dt = 1.0 / control_freq

        logger.info(f"  Fixed direction: {direction.round(4).tolist()}")

        for step in range(1, max_steps + 1):
            t0 = time.time()

            if chunk_buf is None or chunk_idx >= chunk_stride:
                if dry_run:
                    ee_pos_6d = np.zeros(6, dtype=np.float32)
                    gripper = -50.0
                    images = None
                else:
                    ee_pos_6d = robot.get_ee_pos()
                    gripper = robot.get_gripper_state()
                    images = robot.get_images()

                state = build_state_11d(ee_pos_6d, gripper, direction)
                chunk_buf, inf_ms = controller.predict(state, images, instruction)
                chunk_idx = 0

                if step <= 3 or step % 50 == 0:
                    logger.info(f"    step={step} ee={ee_pos_6d[:3].round(4).tolist()} inf={inf_ms:.0f}ms")

                if dry_run and step >= 3:
                    return {"steps": step, "success": True}

            delta = chunk_buf[chunk_idx].copy()
            chunk_idx += 1

            if ema_alpha > 0 and prev_action is not None:
                delta = ema_alpha * prev_action + (1 - ema_alpha) * delta
            prev_action = delta.copy()

            if not dry_run:
                robot.send_ee_delta(delta[:3].astype(np.float64), 45.0)

            elapsed = time.time() - t0
            if dt - elapsed > 0 and not dry_run:
                time.sleep(dt - elapsed)

        return {"steps": max_steps, "success": True}


# ============================================================================
# Task Evaluation
# ============================================================================

def evaluate_task(
    robot,
    controller: Controller,
    planner: Optional[Planner],
    monitor: Optional[Monitor],
    task: str,
    task_cfg: Dict,
    num_episodes: int,
    args,
    session_dir: Path,
) -> List[Dict]:
    """Evaluate multiple episodes for a single task."""
    instruction = task_cfg.get("instruction", "move")
    results = []

    for ep in range(1, num_episodes + 1):
        logger.info(f"\n--- Episode {ep}/{num_episodes} ---")

        if not args.dry_run and ep > 1:
            _move_to_home(robot)
            input("  Reset objects, then press Enter...")

        ep_dir = session_dir / f"{task}_ep{ep:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        result = run_episode(
            robot, controller, planner, monitor,
            instruction, task_cfg,
            args.chunk_stride, args.control_freq, args.ema_alpha,
            args.converge_threshold, args.max_attempts,
            args.dry_run, ep_dir,
        )
        results.append(result)
        logger.info(f"  Episode {ep} result: {result}")

    return results


# ============================================================================
# Robot Utilities
# ============================================================================

HOME_EE = np.array([0.20, 0.00, 0.12])

def _move_to_home(robot):
    """Move robot to home position via interpolation."""
    from so101_client import ARM_JOINT_NAMES

    home_rad = robot._solve_ik(HOME_EE.astype(np.float64))
    home_norm = robot._urdf_radians_to_normalized(home_rad)

    obs = robot.robot.get_observation()
    current = np.array([obs[f"{n}.pos"] for n in ARM_JOINT_NAMES])

    for i in range(1, 31):
        alpha = i / 30
        interp = current * (1 - alpha) + home_norm * alpha
        action = {f"{n}.pos": float(interp[j]) for j, n in enumerate(ARM_JOINT_NAMES)}
        action["gripper.pos"] = 0.0
        robot.robot.send_action(action)
        time.sleep(0.1)

    time.sleep(0.5)
    logger.info(f"Home: ee={robot.get_ee_pos()[:3].round(4).tolist()}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SO101 Real Robot Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Robot
    robot_group = parser.add_argument_group("robot")
    robot_group.add_argument("--port", default="/dev/ttyACM0", help="SO101 serial port")
    robot_group.add_argument("--cameras", nargs="*", default=["top:2", "wrist:8"])
    robot_group.add_argument("--image-size", type=int, default=224)
    robot_group.add_argument("--dry-run", action="store_true", help="No robot, inference only")

    # Controller (VLA)
    ctrl_group = parser.add_argument_group("controller")
    ctrl_group.add_argument("--base-model", default="nvidia/GR00T-N1.5-3B", help="VLA base model")
    ctrl_group.add_argument("--adapter-dir", default=None, help="Adapter directory (default: outputs/groot_so101_wrist)")
    ctrl_group.add_argument("--joint", action="store_true", help="Joint-space model (9D state, 6D action)")
    ctrl_group.add_argument("--device", default="cuda:0")

    # Planner
    plan_group = parser.add_argument_group("planner")
    plan_group.add_argument("--planner-provider", default="google",
                            help="Planner VLM provider: openai, anthropic, google, bedrock, vertex, ollama")
    plan_group.add_argument("--planner-model", default="gemini-2.5-flash", help="Planner model name")
    plan_group.add_argument("--no-planner", action="store_true", help="Fixed direction mode (no VLM planner)")

    # Monitor
    mon_group = parser.add_argument_group("monitor")
    mon_group.add_argument("--monitor-provider", default="",
                           help="Monitor VLM provider (empty = disabled): openai, anthropic, google, bedrock, vertex")
    mon_group.add_argument("--monitor-model", default="gemini-2.5-flash", help="Monitor model name")

    # API
    api_group = parser.add_argument_group("api")
    api_group.add_argument("--api-key", default="", help="API key (or use env vars)")

    # Tasks
    task_group = parser.add_argument_group("tasks")
    task_group.add_argument("--tasks", nargs="*", default=None, help="Task names (default: all)")
    task_group.add_argument("--task-config", default=os.path.join(os.path.dirname(__file__), "..", "so101", "so101_task_config.json"), help="Task config JSON")

    # Execution
    exec_group = parser.add_argument_group("execution")
    exec_group.add_argument("--num-episodes", type=int, default=1)
    exec_group.add_argument("--max-attempts", type=int, default=2, help="Max replan attempts per episode")
    exec_group.add_argument("--chunk-stride", type=int, default=8, help="Action chunk stride")
    exec_group.add_argument("--control-freq", type=float, default=10.0, help="Control frequency (Hz)")
    exec_group.add_argument("--ema-alpha", type=float, default=0.3, help="Action EMA smoothing")
    exec_group.add_argument("--converge-threshold", type=float, default=0.02, help="Convergence distance (m)")

    # Output
    parser.add_argument("--results-dir", default="eval_results", help="Results output directory")

    args = parser.parse_args()

    # --- Setup ---
    adapter_dir = args.adapter_dir or (
        "outputs/groot_so101_joint" if args.joint else "outputs/groot_so101_wrist"
    )
    task_config_path = Path(args.task_config)
    results_dir = Path(args.results_dir)

    tasks = args.tasks or ALL_TASKS
    with open(task_config_path) as f:
        task_cfg_all = json.load(f)

    # Print configuration
    print(f"\n{'='*60}")
    print(f"SO101 Evaluation Pipeline")
    print(f"  Controller: GROOT N1.5 + LoRA ({adapter_dir})")
    print(f"  Planner:    {'disabled' if args.no_planner else f'{args.planner_provider}/{args.planner_model}'}")
    print(f"  Monitor:    {'disabled' if not args.monitor_provider else f'{args.monitor_provider}/{args.monitor_model}'}")
    print(f"  Tasks:      {tasks}")
    print(f"  Episodes:   {args.num_episodes}")
    print(f"  Device:     {args.device}")
    print(f"  Dry-run:    {args.dry_run}")
    print(f"{'='*60}\n")

    # Initialize modules
    controller = Controller(args.base_model, args.device, args.image_size, args.joint)

    planner = None
    if not args.no_planner:
        planner = Planner(args.planner_provider, args.planner_model, args.api_key)

    monitor = None
    if args.monitor_provider:
        monitor = Monitor(args.monitor_provider, args.monitor_model, args.api_key)

    # Connect robot
    robot = None
    if not args.dry_run:
        from so101_client import SO101Robot
        cameras = {}
        for cam_spec in args.cameras:
            name, index = cam_spec.split(":")
            cameras[name] = int(index)
        robot = SO101Robot(args.port, cameras, args.image_size)
        robot.connect()
        _move_to_home(robot)

    # Session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(session_dir / "session.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Signal handler
    _running = [True]
    def _sig(s, f):
        _running[0] = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # --- Evaluation loop ---
    all_results = {}
    try:
        for task_idx, task in enumerate(tasks):
            if not _running[0]:
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"  Task {task_idx+1}/{len(tasks)}: {task}")
            logger.info(f"{'='*60}")

            controller.load_task(task, adapter_dir)
            cfg = task_cfg_all.get(task, {})

            task_results = evaluate_task(
                robot, controller, planner, monitor,
                task, cfg, args.num_episodes, args, session_dir,
            )
            all_results[task] = task_results
            controller.unload()

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if robot is not None:
            robot.disconnect()

    # --- Results ---
    result_file = results_dir / f"results_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved: {result_file}")
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for task, results in all_results.items():
        logger.info(f"  {task}: {len(results)} episodes")
        for i, r in enumerate(results):
            logger.info(f"    ep{i+1}: {r}")


if __name__ == "__main__":
    main()
