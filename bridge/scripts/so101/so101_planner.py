#!/usr/bin/env python3
"""
SO101 VLM Planner — 카메라 이미지 + 자연어 명령 → 세그먼트별 goal_ee 계획 생성.

VLM이 이미지를 보고 물체 위치를 추정하여 skill segment 시퀀스를 생성합니다.
각 세그먼트는 goal_ee(3D) + goal_gripper + type + instruction을 포함합니다.

Usage:
    from so101_planner import SO101Planner

    planner = SO101Planner(provider="bedrock", model="anthropic.claude-sonnet-4-20250514-v1:0")
    segments = planner.plan(
        instruction="pick red block and place on blue dish",
        image=rgb_array,       # (H, W, 3) uint8
        current_ee=[0.2, 0.0, 0.12],
    )
    # segments = [
    #   {"type": "move", "goal_ee": [0.21, 0.08, 0.15], "goal_gripper": "open", "instruction": "Move above red block"},
    #   {"type": "move", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "open", "instruction": "Descend to red block"},
    #   {"type": "gripper_close", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "close", "instruction": "Grasp red block"},
    #   ...
    # ]
"""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("so101_planner")

# SO101 workspace bounds (from training data analysis)
WORKSPACE = {
    "x_range": [0.08, 0.38],   # forward/backward
    "y_range": [-0.20, 0.20],  # left/right
    "z_table": 0.02,           # table surface height
    "z_lift": 0.20,            # lift/clearance height
    "z_above": 0.15,           # approach height above object
    "home": [0.20, 0.00, 0.12],  # home/initial position
}

SYSTEM_PROMPT = """You are a robot manipulation planner for the SO101 robot arm.
You see a top-down or angled camera image of a tabletop workspace.
Given the image and a task instruction, you must output a sequence of skill segments
that the robot should execute to complete the task.

IMPORTANT: You must output ONLY valid JSON. No explanations before or after."""

def _build_plan_prompt(
    instruction: str,
    current_ee: List[float],
    workspace: Dict = WORKSPACE,
) -> str:
    """Build the VLM prompt for segment planning."""
    return f"""## Task
"{instruction}"

## Current Robot State
- End-effector position: ({current_ee[0]:.4f}, {current_ee[1]:.4f}, {current_ee[2]:.4f})

## Workspace Coordinate System (meters, robot base frame)
- X axis: forward (away from robot base). Range: [{workspace['x_range'][0]:.2f}, {workspace['x_range'][1]:.2f}]
- Y axis: left (+) / right (-). Range: [{workspace['y_range'][0]:.2f}, {workspace['y_range'][1]:.2f}]
- Z axis: up (+) / down (-). Table surface ≈ {workspace['z_table']:.2f}, lift height ≈ {workspace['z_lift']:.2f}

## Segment Types
- "move": Move end-effector to goal_ee position
- "gripper_open": Open gripper (set goal_ee to current expected position)
- "gripper_close": Close gripper to grasp object (set goal_ee to current expected position)
- "move_free": Return to safe/home position

## Rules
1. For pick-and-place tasks, use this pattern:
   gripper_open → move(above object) → move(descend to object) → gripper_close → move(lift) → move(above target) → move(descend to target) → gripper_open → move(retract up) → move_free
2. For pushing/sliding tasks: move(behind object) → move(push through object toward target)
3. Always approach from above (z≈0.15-0.20) before descending to grasp (z≈0.01-0.03)
4. After grasping, lift to z≈0.20 before moving horizontally
5. Estimate object positions from the image. Objects on the table have z≈{workspace['z_table']:.2f}
6. Output 4-14 segments depending on task complexity
7. For multi-object tasks (e.g., "stack A on B", "distribute"), handle objects one at a time sequentially
8. End with move_free to home position ({workspace['home'][0]:.2f}, {workspace['home'][1]:.2f}, {workspace['home'][2]:.2f})

## Output Format
Output a JSON object with a "segments" array. Each segment has:
- "type": one of "move", "gripper_open", "gripper_close", "move_free"
- "goal_ee": [x, y, z] target position in meters
- "goal_gripper": "open" or "close"
- "instruction": brief description of what this segment does
- "max_steps": estimated steps needed (30-200)

## Example Output
```json
{{
  "segments": [
    {{"type": "gripper_open", "goal_ee": [0.20, 0.00, 0.12], "goal_gripper": "open", "instruction": "Open gripper", "max_steps": 30}},
    {{"type": "move", "goal_ee": [0.22, 0.08, 0.18], "goal_gripper": "open", "instruction": "Move above red block", "max_steps": 120}},
    {{"type": "move", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "open", "instruction": "Descend to red block", "max_steps": 100}},
    {{"type": "gripper_close", "goal_ee": [0.22, 0.08, 0.02], "goal_gripper": "close", "instruction": "Grasp red block", "max_steps": 40}},
    {{"type": "move", "goal_ee": [0.22, 0.08, 0.20], "goal_gripper": "close", "instruction": "Lift red block", "max_steps": 120}},
    {{"type": "move", "goal_ee": [0.25, -0.15, 0.20], "goal_gripper": "close", "instruction": "Move above blue dish", "max_steps": 150}},
    {{"type": "move", "goal_ee": [0.25, -0.15, 0.03], "goal_gripper": "close", "instruction": "Lower onto blue dish", "max_steps": 100}},
    {{"type": "gripper_open", "goal_ee": [0.25, -0.15, 0.03], "goal_gripper": "open", "instruction": "Release red block", "max_steps": 40}},
    {{"type": "move", "goal_ee": [0.25, -0.15, 0.20], "goal_gripper": "open", "instruction": "Retract upward", "max_steps": 80}},
    {{"type": "move_free", "goal_ee": [0.20, 0.00, 0.12], "goal_gripper": "open", "instruction": "Return to home", "max_steps": 80}}
  ]
}}
```

Now plan the segments for the given task. Look at the image carefully to estimate object positions.
Output ONLY the JSON."""


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode numpy image to base64 JPEG."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from VLM response text."""
    # Try ```json ... ``` blocks
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try ``` ... ``` blocks
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try outermost { ... }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


class SO101Planner:
    """VLM-based planner for SO101 robot tasks."""

    def __init__(
        self,
        provider: str = "bedrock",
        model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._llm = None

    def _init_llm(self):
        """Initialize LangChain LLM client."""
        if self._llm is not None:
            return

        if self.provider in ("bedrock", "aws"):
            from langchain_aws import ChatBedrockConverse
            self._llm = ChatBedrockConverse(
                model=self.model,
                region_name="eu-west-1",
                temperature=self.temperature,
                max_tokens=4096,
            )
        elif self.provider in ("anthropic",):
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=4096,
            )
        elif self.provider in ("openai",):
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=4096,
            )
        elif self.provider in ("google", "vertex", "gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=4096,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        logger.info(f"Planner initialized: {self.provider}/{self.model}")

    def plan(
        self,
        instruction: str,
        image: Optional[np.ndarray] = None,
        current_ee: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Generate skill segments from instruction and image.

        Args:
            instruction: Natural language task (e.g., "pick red block and place on blue dish")
            image: Camera image (H, W, 3) uint8 RGB. If None, plans without vision.
            current_ee: Current end-effector [x, y, z]. Defaults to home position.

        Returns:
            List of segment dicts with keys: type, goal_ee, goal_gripper, instruction, max_steps
        """
        self._init_llm()

        if current_ee is None:
            current_ee = WORKSPACE["home"]

        prompt_text = _build_plan_prompt(instruction, current_ee)

        # Build multimodal message
        from langchain_core.messages import HumanMessage, SystemMessage

        content = []
        if image is not None:
            img_b64 = _encode_image_b64(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": prompt_text})

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._llm.invoke(messages)
                resp_text = response.content
                if isinstance(resp_text, list):
                    resp_text = "\n".join(
                        p.get("text", str(p)) if isinstance(p, dict) else str(p)
                        for p in resp_text
                    )

                result = _extract_json(resp_text)
                if result is None:
                    logger.warning(f"JSON parse failed (attempt {attempt})")
                    continue

                segments = result.get("segments", [])
                if not segments:
                    logger.warning(f"Empty segments (attempt {attempt})")
                    continue

                # Validate segments
                validated = self._validate_segments(segments)
                if validated:
                    logger.info(f"Plan generated: {len(validated)} segments")
                    for i, seg in enumerate(validated):
                        logger.info(
                            f"  [{i}] {seg['type']:20s} goal_ee={[round(x,4) for x in seg['goal_ee']]} "
                            f"grip={seg['goal_gripper']:5s} | {seg['instruction']}"
                        )
                    return validated

            except Exception as e:
                logger.error(f"Planning failed (attempt {attempt}): {e}")

        logger.error("All planning attempts failed, using fallback")
        return self._fallback_plan(instruction, current_ee)

    def replan(
        self,
        instruction: str,
        image: Optional[np.ndarray] = None,
        current_ee: Optional[List[float]] = None,
        failure_context: str = "",
    ) -> List[Dict]:
        """Replan after failure with additional context.

        Args:
            instruction: Original task instruction
            image: Current camera image
            current_ee: Current end-effector position
            failure_context: Description of what failed (e.g., "grasp failed, object slipped")
        """
        augmented_instruction = instruction
        if failure_context:
            augmented_instruction = (
                f"{instruction}\n\n"
                f"[PREVIOUS ATTEMPT FAILED: {failure_context}. "
                f"Try a different approach — adjust positions or angles.]"
            )

        return self.plan(augmented_instruction, image, current_ee)

    def _validate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Validate and fix segment format."""
        validated = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue

            seg_type = seg.get("type", "move")
            goal_ee = seg.get("goal_ee")
            if goal_ee is None or len(goal_ee) != 3:
                logger.warning(f"Invalid goal_ee in segment: {seg}")
                continue

            # Clamp to workspace
            goal_ee = [
                np.clip(goal_ee[0], WORKSPACE["x_range"][0], WORKSPACE["x_range"][1]),
                np.clip(goal_ee[1], WORKSPACE["y_range"][0], WORKSPACE["y_range"][1]),
                np.clip(goal_ee[2], 0.0, 0.25),
            ]

            validated.append({
                "type": seg_type,
                "goal_ee": goal_ee,
                "goal_gripper": seg.get("goal_gripper", "open"),
                "instruction": seg.get("instruction", seg_type),
                "max_steps": seg.get("max_steps", 100),
            })

        return validated

    def _fallback_plan(self, instruction: str, current_ee: List[float]) -> List[Dict]:
        """Generate a simple fallback plan when VLM fails."""
        instr_lower = instruction.lower()

        # Default pick-and-place fallback
        pick_pos = [0.22, 0.08, 0.02]
        place_pos = [0.25, -0.15, 0.03]

        segments = [
            {"type": "gripper_open", "goal_ee": current_ee, "goal_gripper": "open",
             "instruction": "Open gripper", "max_steps": 30},
            {"type": "move", "goal_ee": [pick_pos[0], pick_pos[1], WORKSPACE["z_lift"]],
             "goal_gripper": "open", "instruction": "Move above object", "max_steps": 120},
            {"type": "move", "goal_ee": pick_pos, "goal_gripper": "open",
             "instruction": "Descend to object", "max_steps": 100},
            {"type": "gripper_close", "goal_ee": pick_pos, "goal_gripper": "close",
             "instruction": "Grasp object", "max_steps": 40},
            {"type": "move", "goal_ee": [pick_pos[0], pick_pos[1], WORKSPACE["z_lift"]],
             "goal_gripper": "close", "instruction": "Lift object", "max_steps": 120},
            {"type": "move", "goal_ee": [place_pos[0], place_pos[1], WORKSPACE["z_lift"]],
             "goal_gripper": "close", "instruction": "Move above target", "max_steps": 150},
            {"type": "move", "goal_ee": place_pos, "goal_gripper": "close",
             "instruction": "Lower to target", "max_steps": 100},
            {"type": "gripper_open", "goal_ee": place_pos, "goal_gripper": "open",
             "instruction": "Release object", "max_steps": 40},
            {"type": "move", "goal_ee": [place_pos[0], place_pos[1], WORKSPACE["z_lift"]],
             "goal_gripper": "open", "instruction": "Retract", "max_steps": 80},
            {"type": "move_free", "goal_ee": WORKSPACE["home"], "goal_gripper": "open",
             "instruction": "Return home", "max_steps": 80},
        ]

        logger.warning(f"Using fallback plan: {len(segments)} segments")
        return segments
