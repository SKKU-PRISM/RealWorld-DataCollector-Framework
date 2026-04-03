"""
VLM Planner - Single-stage vision-language planning.

Directly converts instruction + images + object coordinates into primitive actions.
Supports multiple VLM providers: OpenAI, Anthropic, Google, HuggingFace.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from robobridge.modules.base import BaseModule
from .types import (
    HighLevelAction,
    ObjectInfo,
    Position3D,
    PrimitiveAction,
    PrimitivePlan,
)

logger = logging.getLogger(__name__)


# ── Task-specific plan profiles ──────────────────────────────────────────
# Derived from training data analysis (data_direction_v4).
# Each profile defines the primitive sequence that matches what the VLA
# learned during training, ensuring direction vectors align exactly.
#
# Keys:
#   keywords   - instruction substrings to match (lowercase)
#   primitives - list of (type, params) tuples
#                move: {"offset": [dx,dy,dz]} relative to target, or "target" for exact pos
#                grip: {"width": float}
#   notes      - human-readable explanation
TASK_PLAN_PROFILES = {
    # ── Training: pure move (NO grip) ── direction: -x ──
    "TurnOnMicrowave": {
        "keywords": ["turn on the microwave", "turn on microwave"],
        "primitives": [
            ("move", {"offset": [0.0, 0.03, 0.05]}),   # approach from above/side
            ("move", {"offset": [0.0, 0.0, 0.0]}),     # contact target
            ("move", {"offset": [-0.08, -0.05, 0.0]}),  # sweep -x,-y (turn direction)
        ],
        "notes": "NO grip primitives. Training data shows single continuous move phase. "
                 "Gripper stays in whatever state it's in. Direction dominant: -x.",
    },
    # ── Training: pure move (NO grip) ── direction: -x ──
    "TurnOffMicrowave": {
        "keywords": ["turn off the microwave", "turn off microwave"],
        "primitives": [
            ("move", {"offset": [0.0, 0.03, 0.05]}),
            ("move", {"offset": [0.0, 0.0, 0.0]}),
            ("move", {"offset": [-0.08, 0.05, 0.0]}),   # sweep -x,+y (reverse turn)
        ],
        "notes": "NO grip primitives. Same as TurnOn but reverse direction.",
    },
    # ── Training: grip → move ── direction: +x ──
    "TurnSinkSpout": {
        "keywords": ["turn the sink spout", "turn sink spout"],
        "primitives": [
            ("grip", {"width": 0.0}),                    # grip first (direction=[0,0,0])
            ("move", {"offset": [0.0, 0.0, 0.0]}),      # move to target (+x dominant)
        ],
        "notes": "Grip FIRST then move. Training shows grip → move (2 phases). "
                 "VLA expects direction=[0,0,0] initially, then unit vector to target.",
    },
    # ── Training: move → grip → move ── direction: +x, first approach from -x/-z ──
    "CloseDrawer": {
        "keywords": ["close the drawer", "close drawer"],
        "primitives": [
            ("move", {"offset": [-0.08, 0.0, 0.0]}),    # approach from behind (-x)
            ("grip", {"width": 0.0}),                     # grip pause (direction=[0,0,0])
            ("move", {"offset": [0.12, 0.0, 0.0]}),     # push through (+x)
        ],
        "notes": "move → grip → move. Push +x to close. No actual gripping needed, "
                 "but grip pause matches training phase structure.",
    },
    # ── Training: move → grip → move ── direction: +x, approach from left/above ──
    "TurnOffSinkFaucet": {
        "keywords": ["turn off the sink faucet", "turn off sink faucet",
                      "turn off the faucet", "turn off faucet"],
        "primitives": [
            ("move", {"offset": [-0.03, 0.0, 0.05]}),   # approach from above/behind
            ("grip", {"width": 0.0}),                     # grip pause
            ("move", {"offset": [0.05, 0.0, -0.03]}),   # push forward and down
        ],
        "notes": "move → grip → move. Direction +x dominant. Faucet handle pushed down/forward.",
    },
    # ── Training: move → grip → move → grip → move ── direction: +x ──
    "TurnOnSinkFaucet": {
        "keywords": ["turn on the sink faucet", "turn on sink faucet",
                      "turn on the faucet", "turn on faucet"],
        "primitives": [
            ("move", {"offset": [-0.03, 0.0, 0.05]}),   # approach from above
            ("grip", {"width": 0.0}),                     # pause
            ("move", {"offset": [0.0, 0.0, 0.0]}),      # contact
            ("grip", {"width": 0.0}),                     # pause
            ("move", {"offset": [0.05, 0.0, -0.05]}),   # push forward/down
        ],
        "notes": "move → grip → move → grip → move. Multiple contact phases.",
    },
    # ── Training: grip → move → grip → move ── direction: +x ──
    "CoffeePressButton": {
        "keywords": ["press the coffee", "coffee press button", "press coffee button",
                      "coffee machine button", "button on the coffee machine",
                      "press the button on the coffee"],
        "primitives": [
            ("grip", {"width": 0.0}),                     # close gripper first
            ("move", {"offset": [0.0, 0.0, 0.0]}),      # approach + press target
            ("grip", {"width": 0.04}),                    # release
            ("move", {"offset": [0.0, 0.0, 0.08]}),     # retreat up
        ],
        "notes": "grip → move → grip → move. Close grip, press, release, retreat.",
    },
    # ── Training: move → grip → move → grip → move ── direction: -z ──
    "TurnOffStove": {
        "keywords": ["turn off the stove", "turn off stove"],
        "primitives": [
            ("move", {"offset": [0.0, 0.0, 0.05]}),     # approach from above
            ("grip", {"width": 0.0}),                     # pause
            ("move", {"offset": [0.0, 0.0, 0.0]}),      # contact knob
            ("grip", {"width": 0.0}),                     # pause
            ("move", {"offset": [0.0, 0.05, -0.03]}),   # turn knob
        ],
        "notes": "move → grip → move → grip → move. Direction -z dominant (push down to turn off).",
    },
    # ── Training: move → grip → move → grip → move ── direction: -z ──
    "TurnOnStove": {
        "keywords": ["turn on the stove", "turn on stove"],
        "primitives": [
            ("move", {"offset": [0.0, 0.0, 0.05]}),
            ("grip", {"width": 0.0}),
            ("move", {"offset": [0.0, 0.0, 0.0]}),
            ("grip", {"width": 0.0}),
            ("move", {"offset": [0.0, -0.05, -0.03]}),
        ],
        "notes": "move → grip → move → grip → move. Direction -z dominant.",
    },
    # ── Training: move → grip → move ── direction: +x, first from -x/+z ──
    "CloseSingleDoor": {
        "keywords": ["close the single door", "close single door", "close the door"],
        "primitives": [
            ("move", {"offset": [-0.06, 0.0, 0.05]}),   # approach from behind/above
            ("grip", {"width": 0.0}),                     # pause
            ("move", {"offset": [0.10, 0.0, 0.0]}),     # push forward
        ],
        "notes": "Simplified from 5+ segments. Push +x to close.",
    },
}


class Planner(BaseModule):
    """
    Vision-Language Model Planner.
    
    Takes instruction, images, and object coordinates to directly output
    primitive actions (move, grip) without intermediate planning stages.
    
    Supported providers:
    - openai: GPT-4V, GPT-4o, etc.
    - anthropic: Claude 3.5 Sonnet, etc.
    - google: Gemini Pro Vision, etc.
    - hf/huggingface: Qwen-VL, etc. (local)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5.2",
        device: str = "cpu",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        approach_height: float = 0.15,
        grip_open_width: float = 0.08,
        grip_close_width: float = 0.0,
        has_mobility: bool = False,
        link_mode: str = "direct",
        **kwargs,
    ):
        super().__init__(
            provider=provider,
            model=model,
            device=device,
            api_key=api_key,
            link_mode=link_mode,
            **kwargs,
        )
        
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.approach_height = approach_height
        self.grip_open_width = grip_open_width
        self.grip_close_width = grip_close_width
        self.has_mobility = has_mobility

        self._llm = None
        self._hf_model = None
        self._hf_processor = None

    def initialize_client(self) -> None:
        """Initialize VLM client based on provider."""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            self._init_openai()
        elif provider == "anthropic":
            self._init_anthropic()
        elif provider == "bedrock":
            self._init_bedrock()
        elif provider == "google":
            self._init_google()
        elif provider == "vertex":
            self._init_vertex()
        elif provider in ("hf", "huggingface"):
            self._init_huggingface()
        else:
            logger.warning(f"Unknown provider: {provider}, using OpenAI as default")
            self._init_openai()

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from langchain_openai import ChatOpenAI
            # gpt-5.x requires max_completion_tokens instead of max_tokens
            token_kwarg = {}
            if self.config.model.startswith("gpt-5") or self.config.model.startswith("o"):
                token_kwarg["max_completion_tokens"] = self.max_tokens
            else:
                token_kwarg["max_tokens"] = self.max_tokens
            self._llm = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.api_base,
                temperature=self.temperature,
                **token_kwarg,
            )
            logger.info(f"Initialized OpenAI VLM: {self.config.model}")
        except ImportError:
            logger.error("langchain-openai not installed. Run: pip install langchain-openai")

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info(f"Initialized Anthropic VLM: {self.config.model}")
        except ImportError:
            logger.error("langchain-anthropic not installed. Run: pip install langchain-anthropic")

    def _init_bedrock(self) -> None:
        """Initialize AWS Bedrock client using Bearer token auth."""
        from robobridge.modules.bedrock_bearer import create_bedrock_bearer_chat
        self._llm = create_bedrock_bearer_chat(
            model=self.config.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self._llm is None:
            logger.error("Failed to initialize Bedrock Bearer client. Set AWS_BEARER_TOKEN_BEDROCK.")

    def _init_google(self) -> None:
        """Initialize Google client."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.model,
                google_api_key=self.config.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            logger.info(f"Initialized Google VLM: {self.config.model}")
        except ImportError:
            logger.error("langchain-google-genai not installed. Run: pip install langchain-google-genai")

    def _init_vertex(self) -> None:
        """Initialize Google Vertex AI client (uses ADC, no API key needed)."""
        try:
            from langchain_google_vertexai import ChatVertexAI
            self._llm = ChatVertexAI(
                model_name=self.config.model,
                project="prism-485101",
                location="global",
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            logger.info(f"Initialized Vertex AI VLM: {self.config.model}")
        except ImportError:
            logger.error("langchain-google-vertexai not installed. Run: pip install langchain-google-vertexai")

    def _init_huggingface(self) -> None:
        """Initialize HuggingFace local VLM."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            model_name = self.config.model
            logger.info(f"Loading HuggingFace VLM: {model_name}")
            
            self._hf_processor = AutoProcessor.from_pretrained(model_name)
            self._hf_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.config.device else torch.float32,
                device_map="auto" if "cuda" in self.config.device else None,
                trust_remote_code=True,
            )
            logger.info(f"Loaded HuggingFace VLM: {model_name}")
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")

    def plan(
        self,
        instruction: str,
        images: Dict[str, np.ndarray],
        objects: Dict[str, ObjectInfo],
        bboxes: Optional[List[Dict]] = None,
        eef_pos: Optional[np.ndarray] = None,
    ) -> Optional[List[PrimitivePlan]]:
        """
        Generate primitive plans from instruction and images.

        Args:
            instruction: Natural language task instruction
            images: Dict of camera_name -> image array (e.g., {"robotview": array})
            objects: Dict of object_name -> ObjectInfo with coordinates
            bboxes: List of bounding boxes for collision avoidance
            eef_pos: Current end-effector position [x, y, z] in robot-base frame

        Returns:
            List of PrimitivePlan, one per logical action in the task
        """
        # Try task-specific template plan first (matches VLA training data exactly)
        template_result = self._try_template_plan(instruction, objects)
        if template_result is not None:
            return template_result

        if not self._llm and not self._hf_model:
            self.initialize_client()

        provider = self.config.provider.lower()

        if provider in ("hf", "huggingface"):
            return self._plan_with_hf(instruction, images, objects, bboxes, eef_pos=eef_pos)
        else:
            return self._plan_with_langchain(instruction, images, objects, bboxes, eef_pos=eef_pos)

    def _plan_with_langchain(
        self,
        instruction: str,
        images: Dict[str, np.ndarray],
        objects: Dict[str, ObjectInfo],
        bboxes: Optional[List[Dict]] = None,
        eef_pos: Optional[np.ndarray] = None,
    ) -> Optional[List[PrimitivePlan]]:
        """Plan using LangChain-based providers (OpenAI, Anthropic, Google)."""
        from langchain_core.messages import HumanMessage

        prompt = self._build_prompt(instruction, objects, bboxes, eef_pos=eef_pos)
        content = self._build_message_content(prompt, images)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._llm.invoke([HumanMessage(content=content)])
                resp_content = response.content
                # Vertex/Gemini may return content as list of parts
                if isinstance(resp_content, list):
                    resp_content = "\n".join(
                        p.get("text", str(p)) if isinstance(p, dict) else str(p)
                        for p in resp_content
                    )
                if not resp_content or not resp_content.strip():
                    logger.warning(f"VLM returned empty response (attempt {attempt + 1}/{max_retries})")
                    continue

                # Parse VLM response — try new primitive format first, fall back to action format
                result = self._parse_primitive_response(resp_content, objects)
                if result is not None:
                    return result

                # Fallback: try old action format
                action_plans = self._parse_action_response(resp_content)
                if action_plans:
                    result = self._expand_actions_to_primitives(action_plans, objects)
                    if result is not None:
                        return result

                logger.warning(f"VLM parse returned None (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"VLM planning failed (attempt {attempt + 1}/{max_retries}): {e}")

        logger.error(f"VLM planning failed after {max_retries} attempts, using fallback plan")
        return self._generate_fallback_plan(instruction)

    def _try_template_plan(
        self,
        instruction: str,
        objects: Dict[str, ObjectInfo],
    ) -> Optional[List[PrimitivePlan]]:
        """Try to generate a plan from task-specific templates.

        Templates are derived from VLA training data analysis to ensure
        the primitive sequence (move/grip phases) and direction vectors
        match exactly what the VLA learned during training.

        Returns None if no matching template is found (falls back to VLM).
        """
        from .types import (
            HighLevelAction, PrimitiveAction, PrimitivePlan, Position3D,
        )

        instr_lower = instruction.lower()

        # v5: Detect replan from instruction (contains failure context)
        is_replan = "robot timed out" in instr_lower or "try a different" in instr_lower
        replan_attempt = 0
        if is_replan:
            # Extract attempt number from failure context for deterministic perturbation
            import hashlib
            replan_attempt = int(hashlib.md5(instruction.encode()).hexdigest()[:4], 16) % 10 + 1

        # Find matching profile
        matched_profile = None
        matched_name = None
        for name, profile in TASK_PLAN_PROFILES.items():
            if any(kw in instr_lower for kw in profile["keywords"]):
                matched_profile = profile
                matched_name = name
                break

        if matched_profile is None:
            return None

        # Find target object position from perception
        target_pos = None
        if objects:
            # Use the target-role object, or the first object
            for obj_name, obj in objects.items():
                if obj.role == "target":
                    target_pos = Position3D(x=obj.position.x, y=obj.position.y, z=obj.position.z)
                    break
            if target_pos is None:
                obj = next(iter(objects.values()))
                target_pos = Position3D(x=obj.position.x, y=obj.position.y, z=obj.position.z)

        if target_pos is None:
            logger.warning(f"[PLANNER] Template '{matched_name}' matched but no target object found")
            return None

        # v5: Apply perturbation on replan to try different approach angles
        perturbation = [0.0, 0.0, 0.0]
        if is_replan and replan_attempt > 0:
            import numpy as _np
            _rng = _np.random.RandomState(replan_attempt)
            perturbation = [
                _rng.uniform(-0.03, 0.03),  # X: ±3cm
                _rng.uniform(-0.05, 0.05),  # Y: ±5cm (more variance)
                _rng.uniform(-0.02, 0.02),  # Z: ±2cm
            ]
            logger.info(
                f"[PLANNER] Replan perturbation (attempt {replan_attempt}): "
                f"dx={perturbation[0]:.3f}, dy={perturbation[1]:.3f}, dz={perturbation[2]:.3f}"
            )

        # Build primitives from template
        primitives = []
        for i, (ptype, params) in enumerate(matched_profile["primitives"]):
            if ptype == "grip":
                prim = PrimitiveAction(
                    primitive_id=i, primitive_type="grip",
                    parent_action_id=0,
                    grip_width=params["width"],
                    instruction="grip",
                )
            else:  # move
                offset = params.get("offset", [0, 0, 0])
                pos = Position3D(
                    x=target_pos.x + offset[0] + perturbation[0],
                    y=target_pos.y + offset[1] + perturbation[1],
                    z=target_pos.z + offset[2] + perturbation[2],
                )
                prim = PrimitiveAction(
                    primitive_id=i, primitive_type="move",
                    parent_action_id=0,
                    target_position=pos,
                    instruction="move",
                )
            primitives.append(prim)

        # Log template plan
        prim_summary = []
        for p in primitives:
            if p.primitive_type == "grip":
                prim_summary.append(f"grip({p.grip_width})")
            elif p.target_position:
                prim_summary.append(
                    f"move({p.target_position.x:.3f},{p.target_position.y:.3f},{p.target_position.z:.3f})"
                )
        logger.info(
            f"[PLANNER] Template plan '{matched_name}' ({len(primitives)} prims): {prim_summary}"
        )
        logger.info(f"[PLANNER] Template notes: {matched_profile.get('notes', '')}")

        parent_action = HighLevelAction(
            action_id=0, action_type="template",
            target_object=matched_name,
        )

        plan = PrimitivePlan(
            plan_id=str(uuid.uuid4())[:8],
            parent_action=parent_action,
            primitives=primitives,
        )
        return [plan]

    def _generate_fallback_plan(self, instruction: str) -> Optional[List[PrimitivePlan]]:
        """Generate a reasonable default plan when VLM fails.

        Infers a basic action sequence from the task instruction keywords.
        The VLA model handles actual motion — we just need the right
        primitive sequence (move/grip ordering).
        """
        instr_lower = instruction.lower()

        # Detect task type from instruction keywords
        if any(kw in instr_lower for kw in ["pick", "grab", "grasp", "take"]):
            plans_json = [
                {"action": "pick", "target": "object", "primitives": [
                    {"type": "move"}, {"type": "move"},
                    {"type": "grip", "grip_width": 0.0},
                    {"type": "move"},
                ]},
            ]
            if any(kw in instr_lower for kw in ["place", "put", "set"]):
                plans_json.append(
                    {"action": "place", "target": "target", "primitives": [
                        {"type": "move"}, {"type": "move"},
                        {"type": "grip", "grip_width": 0.08},
                        {"type": "move"},
                    ]}
                )
        elif any(kw in instr_lower for kw in ["open"]):
            plans_json = [
                {"action": "open", "target": "object", "primitives": [
                    {"type": "move"},
                    {"type": "grip", "grip_width": 0.0},
                    {"type": "move"}, {"type": "move"},
                    {"type": "grip", "grip_width": 0.08},
                ]},
            ]
        elif any(kw in instr_lower for kw in ["close"]):
            plans_json = [
                {"action": "close", "target": "object", "primitives": [
                    {"type": "move", "description": "approach handle"},
                    {"type": "move", "description": "push forward"},
                    {"type": "move", "description": "push further"},
                    {"type": "move", "description": "retreat"},
                ]},
            ]
        elif any(kw in instr_lower for kw in ["turn", "rotate", "twist"]):
            plans_json = [
                {"action": "turn", "target": "object", "primitives": [
                    {"type": "move"}, {"type": "move"},
                    {"type": "move"}, {"type": "move"},
                ]},
            ]
        elif any(kw in instr_lower for kw in ["push", "press", "toggle"]):
            plans_json = [
                {"action": "push", "target": "object", "primitives": [
                    {"type": "move"}, {"type": "move"},
                    {"type": "move"}, {"type": "move"},
                ]},
            ]
        else:
            # Generic manipulation fallback
            plans_json = [
                {"action": "manipulate", "target": "object", "primitives": [
                    {"type": "move"}, {"type": "move"},
                    {"type": "grip", "grip_width": 0.0},
                    {"type": "move"}, {"type": "move"},
                    {"type": "grip", "grip_width": 0.08},
                ]},
            ]

        logger.info(f"Fallback plan: {len(plans_json)} action(s) for '{instruction}'")
        return self._parse_response(json.dumps({"plans": plans_json}))

    def _plan_with_hf(
        self,
        instruction: str,
        images: Dict[str, np.ndarray],
        objects: Dict[str, ObjectInfo],
        bboxes: Optional[List[Dict]] = None,
        eef_pos: Optional[np.ndarray] = None,
    ) -> Optional[List[PrimitivePlan]]:
        """Plan using local HuggingFace VLM."""
        import torch

        if not self._hf_model or not self._hf_processor:
            logger.error("HuggingFace VLM not initialized")
            return None

        prompt = self._build_prompt(instruction, objects, bboxes, eef_pos=eef_pos)
        pil_images = [self._to_pil_image(img) for img in images.values() if img is not None]
        
        # Build conversation for Qwen-VL style
        content = []
        for img in pil_images:
            if img is not None:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        try:
            inputs = self._hf_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            device = self._hf_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self._hf_model.generate(**inputs, max_new_tokens=self.max_tokens)

            generated_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
            response_text = self._hf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            action_plans = self._parse_action_response(response_text)
            if action_plans:
                result = self._expand_actions_to_primitives(action_plans, objects)
                if result is not None:
                    return result
        except Exception as e:
            logger.error(f"HuggingFace VLM planning failed: {e}")

        logger.error("HuggingFace VLM planning failed, using fallback plan")
        return self._generate_fallback_plan(instruction)

    def _build_prompt(
        self,
        instruction: str,
        objects: Dict[str, ObjectInfo],
        bboxes: Optional[List[Dict]] = None,
        eef_pos: Optional[np.ndarray] = None,
    ) -> str:
        """Build planning prompt — VLM outputs primitive sequences directly."""
        # Build object list with positions
        obj_lines = []
        if objects:
            for name, obj in objects.items():
                pos = obj.position
                tag = " [TARGET]" if obj.role == "target" else ""
                obj_lines.append(f"- {name}{tag}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        obj_str = "\n".join(obj_lines) if obj_lines else "none detected"
        obj_names = list(objects.keys()) if objects else []
        target_objs = [n for n, o in objects.items() if o.role == "target"] if objects else []
        logger.info(f"Planner prompt objects: {obj_names}, targets: {target_objs}")

        eef_str = ""
        if eef_pos is not None:
            eef_str = f"\n## Current End-Effector Position\n({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f})\n"

        return f"""You are a robot task planner. Given the task, object positions, and current robot state, plan a sequence of waypoints.

## Task
{instruction}

## Available Objects (name: position as x,y,z in robot frame)
{obj_str}
{eef_str}
## Robot Frame
- x: forward/backward (+ = forward toward objects)
- y: left/right (+ = left, - = right)
- z: up/down (+ = up)

## Primitive Types
- move(x, y, z): Move end-effector to absolute coordinates.
- grip(width): Set gripper width. 0.0 = fully closed, 0.04 = fully open.

## Rules
1. Use 2-8 primitives. Each move should go to a DIFFERENT position.
2. Think carefully about the approach direction. To push/turn an object in a direction, approach from the OPPOSITE side first.
3. For picking: grip(open) → move(above) → move(grasp) → grip(close) → move(lift)
4. For sliding/pushing (no grasp needed): move(approach) → move(contact) → move(push through) → move(retreat)
5. For pressing buttons: grip(close) → move(above button) → move(press down) → grip(open) → move(retreat)
6. For turning knobs: grip(close) → move(knob) → move(turn direction) → grip(open) → move(retreat)
7. For opening doors/drawers: grip(open) → move(handle) → grip(close) → move(pull direction)
8. OBSTACLE AVOIDANCE: Objects listed above that are NOT the target are potential obstacles. Keep waypoints at least 0.05m away from non-target objects. If the task mentions a previous failure (e.g., "arm wedged against cabinet"), plan waypoints that approach from a DIFFERENT angle or height to avoid that obstacle.
9. If a previous attempt failed, DO NOT repeat similar waypoints. Change approach direction (e.g., from above instead of from the side) or add intermediate waypoints to clear obstacles.

## Examples
Objects: sink_spout at (0.50, -0.10, 0.21)
Current EEF: (0.42, 0.05, 0.35)
Task: "turn the sink spout to the right"
Reasoning: To turn right (y-), approach from the left side (y+) and sweep right.
{{"primitives": [{{"type": "move", "x": 0.50, "y": -0.05, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.10, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.15, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.15, "z": 0.28}}]}}

Objects: sink_spout at (0.50, -0.10, 0.21)
Current EEF: (0.42, 0.05, 0.35)
Task: "turn the sink spout to the left"
Reasoning: To turn left (y+), approach from the right side (y-) and sweep left.
{{"primitives": [{{"type": "move", "x": 0.50, "y": -0.15, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.10, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.05, "z": 0.21}}, {{"type": "move", "x": 0.50, "y": -0.05, "z": 0.28}}]}}

Objects: cup at (0.40, 0.10, 0.02), plate at (0.50, -0.20, 0.02)
Current EEF: (0.42, 0.05, 0.35)
Task: "pick up the cup and place it on the plate"
Reasoning: Open gripper, descend to cup, grasp, lift, move to plate, place, release.
{{"primitives": [{{"type": "grip", "width": 0.04}}, {{"type": "move", "x": 0.40, "y": 0.10, "z": 0.10}}, {{"type": "move", "x": 0.40, "y": 0.10, "z": 0.02}}, {{"type": "grip", "width": 0.0}}, {{"type": "move", "x": 0.40, "y": 0.10, "z": 0.10}}, {{"type": "move", "x": 0.50, "y": -0.20, "z": 0.10}}, {{"type": "move", "x": 0.50, "y": -0.20, "z": 0.02}}, {{"type": "grip", "width": 0.04}}]}}

Objects: drawer_handle at (0.45, 0.00, 0.30)
Current EEF: (0.42, 0.05, 0.35)
Task: "open the drawer"
Reasoning: Open gripper, move to handle, grasp, pull toward robot (-x).
{{"primitives": [{{"type": "grip", "width": 0.04}}, {{"type": "move", "x": 0.45, "y": 0.00, "z": 0.30}}, {{"type": "grip", "width": 0.0}}, {{"type": "move", "x": 0.35, "y": 0.00, "z": 0.30}}]}}

Objects: drawer_handle at (0.45, 0.00, 0.30)
Current EEF: (0.42, 0.05, 0.35)
Task: "close the drawer"
Reasoning: To close (push away, +x), move to handle first, then push forward. No gripping needed.
{{"primitives": [{{"type": "move", "x": 0.43, "y": 0.00, "z": 0.30}}, {{"type": "move", "x": 0.55, "y": 0.00, "z": 0.30}}, {{"type": "move", "x": 0.55, "y": 0.00, "z": 0.38}}]}}

Objects: coffee_machine_button at (0.50, 0.10, 0.25)
Current EEF: (0.42, 0.05, 0.35)
Task: "press the coffee machine button"
Reasoning: Close gripper to form a firm press, approach above button, push down, release, retreat.
{{"primitives": [{{"type": "grip", "width": 0.0}}, {{"type": "move", "x": 0.50, "y": 0.10, "z": 0.30}}, {{"type": "move", "x": 0.50, "y": 0.10, "z": 0.25}}, {{"type": "grip", "width": 0.04}}, {{"type": "move", "x": 0.50, "y": 0.10, "z": 0.35}}]}}

Objects: stovetop_knob at (0.55, -0.05, 0.22)
Current EEF: (0.42, 0.05, 0.35)
Task: "turn on the stove"
Reasoning: Close gripper to grip the knob, move to knob, turn by rotating, release, retreat.
{{"primitives": [{{"type": "grip", "width": 0.0}}, {{"type": "move", "x": 0.55, "y": -0.05, "z": 0.22}}, {{"type": "move", "x": 0.55, "y": -0.10, "z": 0.22}}, {{"type": "grip", "width": 0.04}}, {{"type": "move", "x": 0.55, "y": -0.10, "z": 0.30}}]}}

## Output
First write one line of reasoning about approach direction, then output JSON.
"""

    def _expand_actions_to_primitives(
        self,
        action_plans: List[Dict],
        objects: Dict[str, ObjectInfo],
    ) -> Optional[List[PrimitivePlan]]:
        """Expand high-level actions into primitive sequences using perception coordinates.

        VLM provides action structure (action type + target object).
        This method fills in move/grip primitives with actual coordinates from perception.
        """
        from .types import (
            HighLevelAction, PrimitiveAction, PrimitivePlan,
            Position3D,
        )

        result = []
        for i, plan_data in enumerate(action_plans):
            action_type = plan_data.get("action", "unknown")
            target_name = plan_data.get("target", "unknown")

            parent_action = HighLevelAction(
                action_id=i,
                action_type=action_type,
                target_object=target_name,
                target_location=target_name if action_type == "place" else None,
            )

            # Find target object position from perception
            target_pos = self._find_object_position(target_name, objects)

            if target_pos:
                logger.info(
                    f"[PLANNER] Action {i}: {action_type}({target_name}) → "
                    f"target=({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})"
                )
            else:
                logger.warning(
                    f"[PLANNER] Action {i}: {action_type}({target_name}) → "
                    f"target=None (object not found in {list(objects.keys())})"
                )

            # Generate primitives based on action type
            primitives = self._generate_primitives_for_action(
                action_type, target_pos, i
            )

            # Log primitive sequence
            prim_summary = []
            for p in primitives:
                if p.primitive_type == "grip":
                    prim_summary.append(f"grip({p.grip_width})")
                elif p.target_position:
                    prim_summary.append(
                        f"move({p.target_position.x:.3f},{p.target_position.y:.3f},{p.target_position.z:.3f})"
                    )
                else:
                    prim_summary.append("move(None)")
            logger.info(f"[PLANNER] Primitives: {prim_summary}")

            plan = PrimitivePlan(
                plan_id=str(uuid.uuid4())[:8],
                parent_action=parent_action,
                primitives=primitives,
            )
            result.append(plan)

        return result if result else None

    # Semantic aliases: map common task names to perception object names
    _SEMANTIC_ALIASES = {
        "faucet": ["sink_handle", "sink_faucet", "faucet_handle"],
        "sink_faucet": ["sink_handle", "sink_faucet", "faucet_handle"],
        "sink faucet": ["sink_handle", "sink_faucet", "faucet_handle"],
        "microwave_button": ["microwave_button", "microwave_knob"],
        "microwave_door": ["cab_micro_handle", "microwave_handle", "microwave_door_handle"],
        "microwave door": ["cab_micro_handle", "microwave_handle", "microwave_door_handle"],
        "microwave": ["microwave_button", "microwave_knob", "cab_micro_handle"],
        "stove": ["stovetop_knob", "stove_knob"],
        "stove_button": ["stovetop_knob", "stove_knob", "hood_button"],
        "coffee": ["coffee_machine_button"],
        "coffee_button": ["coffee_machine_button"],
        "coffee_machine": ["coffee_machine_button", "coffee_machine_handle"],
        "dishwasher": ["dishwasher_button"],
        "toaster": ["toaster_button", "toaster_knob"],
        "oven": ["oven_knob"],
        "fridge": ["fridge_top_handle", "fridge_cab_handle", "fridge_handle"],
        "drawer": ["stack_handle", "bottom_handle"],
    }

    def _find_object_position(
        self,
        target_name: str,
        objects: Dict[str, ObjectInfo],
    ) -> Optional[Position3D]:
        """Find object position by name, with fuzzy matching + semantic aliases."""
        from .types import Position3D

        if not objects:
            return None

        # Exact match
        if target_name in objects:
            pos = objects[target_name].position
            return Position3D(x=pos.x, y=pos.y, z=pos.z)

        # Semantic alias match
        target_lower = target_name.lower().replace("_", " ")
        for alias_key, alias_targets in self._SEMANTIC_ALIASES.items():
            if alias_key in target_lower or target_lower in alias_key:
                for alias_target in alias_targets:
                    if alias_target in objects:
                        pos = objects[alias_target].position
                        logger.info(f"[PLANNER] Semantic alias: '{target_name}' -> '{alias_target}'")
                        return Position3D(x=pos.x, y=pos.y, z=pos.z)

        # Fuzzy match: check if target_name is substring of any object name
        for name, obj in objects.items():
            name_lower = name.lower().replace("_", " ")
            if target_lower in name_lower or name_lower in target_lower:
                pos = obj.position
                return Position3D(x=pos.x, y=pos.y, z=pos.z)

        # If only one object, use it
        if len(objects) == 1:
            obj = next(iter(objects.values()))
            return Position3D(x=obj.position.x, y=obj.position.y, z=obj.position.z)

        logger.warning(f"Object '{target_name}' not found in perception: {list(objects.keys())}")
        return None

    def _generate_primitives_for_action(
        self,
        action_type: str,
        target_pos: Optional["Position3D"],
        action_id: int,
    ) -> List["PrimitiveAction"]:
        """Generate primitive sequence for a given action type + target position."""
        from .types import PrimitiveAction, Position3D

        primitives = []
        pid = 0

        def _move(pos: Optional[Position3D] = None):
            nonlocal pid
            p = PrimitiveAction(
                primitive_id=pid, primitive_type="move",
                parent_action_id=action_id,
                target_position=pos,
                instruction="move",
            )
            pid += 1
            return p

        def _grip(width: float):
            nonlocal pid
            p = PrimitiveAction(
                primitive_id=pid, primitive_type="grip",
                parent_action_id=action_id,
                grip_width=width, instruction="grip",
            )
            pid += 1
            return p

        # Simplified: use target_pos directly, no above/at/lift offsets.
        # Let VLA handle approach direction and fine positioning naturally.
        tp = target_pos

        if action_type in ("pick", "grab", "grasp", "take", "lift"):
            primitives = [_grip(self.grip_open_width), _move(tp), _grip(0.0), _move(tp)]
        elif action_type in ("place", "put", "set"):
            primitives = [_move(tp), _grip(self.grip_open_width)]
        elif action_type in ("turn", "rotate", "twist"):
            primitives = [_move(tp), _move(tp), _move(tp), _move(tp)]
        elif action_type in ("open",):
            primitives = [_grip(self.grip_open_width), _move(tp), _grip(0.0), _move(tp)]
        elif action_type in ("close",):
            # Push pattern: approach → push forward (no gripping needed)
            # Create intermediate push target: offset in push direction
            if tp is not None:
                push_offset = Position3D(
                    x=tp.x + 0.12, y=tp.y, z=tp.z
                )
            else:
                push_offset = None
            primitives = [
                _move(tp),           # approach handle
                _move(push_offset),  # push forward to close
            ]
        elif action_type in ("push", "press", "toggle", "slide"):
            primitives = [_move(tp), _move(tp), _move(tp), _move(tp)]
        else:
            primitives = [_move(tp), _move(tp), _move(tp), _move(tp)]

        return primitives

    def _build_message_content(self, prompt: str, images: Dict[str, np.ndarray]) -> List[Dict]:
        """Build message content with images for the specific provider."""
        content = []
        provider = self.config.provider.lower()
        
        for cam_name, image in images.items():
            if image is None:
                continue
            
            img_b64 = self._encode_image(image)
            if not img_b64:
                continue
            
            if provider in ("openai", "google", "vertex"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            elif provider in ("anthropic", "bedrock"):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                })
        
        content.append({"type": "text", "text": prompt})
        return content

    def _format_objects_info(self, objects: Dict[str, ObjectInfo]) -> str:
        """Format objects info for prompt with position and bbox."""
        lines = []
        for name, obj in objects.items():
            pos = obj.position
            role_str = f" [{obj.role}]" if obj.role else ""

            if obj.bbox_3d:
                bbox_min = obj.bbox_3d["min"]
                bbox_max = obj.bbox_3d["max"]
                lines.append(
                    f"- {name}{role_str}:\n"
                    f"    position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n"
                    f"    bbox: min=({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}), "
                    f"max=({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})"
                )
            else:
                lines.append(
                    f"- {name}{role_str}: position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
                )
        return "\n".join(lines) if lines else "No objects detected"

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Encode numpy image to base64 string."""
        try:
            pil_image = self._to_pil_image(image)
            if pil_image is None:
                return None
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _to_pil_image(self, image: np.ndarray) -> Optional[Image.Image]:
        """Convert numpy array to PIL Image."""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to convert to PIL Image: {e}")
            return None

    def _parse_primitive_response(
        self,
        content: str,
        objects: Dict[str, "ObjectInfo"],
    ) -> Optional[List["PrimitivePlan"]]:
        """Parse VLM response with direct primitive format.

        Expected: {"primitives": [{"type": "move", "target": "obj"}, {"type": "grip", "width": 0.0}, ...]}
        """
        from .types import (
            HighLevelAction, PrimitiveAction, PrimitivePlan, Position3D,
        )

        json_str = self._extract_json(content)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            fixed = self._fix_json(json_str)
            if fixed:
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    return None
            else:
                return None

        primitives_data = data.get("primitives", [])
        if not primitives_data:
            return None

        # Build primitive list
        primitives = []
        targets_used = set()
        for i, p in enumerate(primitives_data):
            ptype = p.get("type", "move")
            if ptype == "grip":
                width = float(p.get("width", 0.04))
                prim = PrimitiveAction(
                    primitive_id=i, primitive_type="grip",
                    parent_action_id=0, grip_width=width, instruction="grip",
                )
            else:
                # New format: explicit x,y,z coordinates
                if "x" in p and "y" in p and "z" in p:
                    target_pos = Position3D(
                        x=float(p["x"]), y=float(p["y"]), z=float(p["z"])
                    )
                    targets_used.add(f"({p['x']:.2f},{p['y']:.2f},{p['z']:.2f})")
                else:
                    # Fallback: object name lookup
                    target_name = p.get("target", "")
                    target_pos = self._find_object_position(target_name, objects)
                    targets_used.add(target_name)
                prim = PrimitiveAction(
                    primitive_id=i, primitive_type="move",
                    parent_action_id=0, target_position=target_pos,
                    instruction="move",
                )
            primitives.append(prim)

        if not primitives:
            return None

        # Log what VLM decided
        prim_summary = []
        for p in primitives:
            if p.primitive_type == "grip":
                prim_summary.append(f"grip({p.grip_width})")
            elif p.target_position:
                prim_summary.append(
                    f"move({p.target_position.x:.3f},{p.target_position.y:.3f},{p.target_position.z:.3f})"
                )
            else:
                prim_summary.append("move(None)")
        logger.info(f"[PLANNER] VLM primitives ({len(primitives)}): {prim_summary}")

        # Infer action type from primitive sequence
        has_grip = any(p.primitive_type == "grip" for p in primitives)
        main_target = next(iter(targets_used), "unknown")
        action_type = "manipulate"
        if has_grip:
            action_type = "pick" if len([p for p in primitives if p.primitive_type == "grip"]) >= 2 else "grip"

        parent_action = HighLevelAction(
            action_id=0, action_type=action_type,
            target_object=main_target,
        )

        plan = PrimitivePlan(
            plan_id=str(uuid.uuid4())[:8],
            parent_action=parent_action,
            primitives=primitives,
        )
        return [plan]

    def _parse_action_response(self, content: str) -> Optional[List[Dict]]:
        """Parse simplified VLM response into action list (action + target only).

        Returns list of dicts like [{"action": "pick", "target": "cup"}, ...]
        """
        if not content or not content.strip():
            logger.warning("Empty VLM response")
            return None

        logger.debug(f"VLM raw response: {content[:500]}")

        json_str = self._extract_json(content)
        if json_str is None:
            logger.error(f"Could not extract JSON from response: {content[:300]}...")
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            fixed = self._fix_json(json_str)
            if fixed:
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    logger.error(f"JSON parse failed even after fix: {e}")
                    return None
            else:
                logger.error(f"JSON parse failed: {e}")
                return None

        plans_data = data.get("plans", [])
        if not plans_data and isinstance(data, list):
            plans_data = data

        if not plans_data:
            logger.warning("No plans in VLM response")
            return None

        # Validate each plan has at least action and target
        valid_plans = []
        for p in plans_data:
            if isinstance(p, dict) and "action" in p:
                valid_plans.append(p)

        if not valid_plans:
            logger.warning("No valid action plans in VLM response")
            return None

        action_strs = [f"{p.get('action')}({p.get('target', '?')})" for p in valid_plans]
        logger.info(f"VLM actions: {action_strs}")
        return valid_plans

    def _parse_response(self, content: str) -> Optional[List[PrimitivePlan]]:
        """Parse VLM response into PrimitivePlan list.

        Tries multiple JSON extraction strategies for robustness.
        """
        if not content or not content.strip():
            logger.warning("Empty VLM response")
            return None

        logger.debug(f"VLM raw response: {content[:500]}")
        json_str = self._extract_json(content)
        if json_str is None:
            logger.error(f"Could not extract JSON from response: {content[:300]}...")
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try fixing common JSON issues
            fixed = self._fix_json(json_str)
            if fixed:
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    logger.error(f"JSON parse failed even after fix: {e}")
                    return None
            else:
                logger.error(f"JSON parse failed: {e}")
                return None

        plans_data = data.get("plans", [])
        if not plans_data:
            # Try treating the data itself as a list of plans
            if isinstance(data, list):
                plans_data = data

        result = []
        for i, plan_data in enumerate(plans_data):
            action_type = plan_data.get("action", "unknown")
            target = plan_data.get("target", "unknown")
            primitives_data = plan_data.get("primitives", [])

            if not primitives_data:
                logger.warning(f"Plan {i} has no primitives, skipping")
                continue

            parent_action = HighLevelAction(
                action_id=i,
                action_type=action_type,
                target_object=target,
                target_location=target if action_type == "place" else None,
            )

            primitives = []
            for j, prim_data in enumerate(primitives_data):
                prim_type = prim_data.get("type", "")

                target_pos = None
                if "target_position" in prim_data:
                    tp = prim_data["target_position"]
                    target_pos = Position3D(x=tp["x"], y=tp["y"], z=tp["z"])

                target_rot = None
                if "target_rotation" in prim_data:
                    tr = prim_data["target_rotation"]
                    target_rot = Rotation3D.from_dict(tr)

                instruction = prim_type if prim_type else "move"

                prim = PrimitiveAction(
                    primitive_id=j,
                    primitive_type=prim_type,
                    parent_action_id=i,
                    target_position=target_pos,
                    target_rotation=target_rot,
                    grip_width=prim_data.get("grip_width"),
                    instruction=instruction,
                )
                primitives.append(prim)

            plan = PrimitivePlan(
                plan_id=str(uuid.uuid4())[:8],
                parent_action=parent_action,
                primitives=primitives,
            )
            result.append(plan)

        logger.info(f"Parsed {len(result)} plans from VLM response")
        return result if result else None

    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON string from VLM response using multiple strategies."""
        import re

        # Strategy 1: ```json ... ``` block
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) >= 2:
                json_part = parts[1].split("```")[0].strip()
                if json_part:
                    return json_part

        # Strategy 2: ``` ... ``` block
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                json_part = parts[1].strip()
                if json_part:
                    return json_part

        # Strategy 3: Find outermost { ... } pair
        first_brace = content.find("{")
        if first_brace >= 0:
            depth = 0
            for i in range(first_brace, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return content[first_brace:i + 1]

        # Strategy 4: Find outermost [ ... ] pair
        first_bracket = content.find("[")
        if first_bracket >= 0:
            depth = 0
            for i in range(first_bracket, len(content)):
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return content[first_bracket:i + 1]

        return None

    def _fix_json(self, json_str: str) -> Optional[str]:
        """Try to fix common JSON formatting issues from various LLMs."""
        import re
        fixed = json_str.strip()
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        # Remove single-line comments
        fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
        # Remove multi-line comments
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        # Replace single quotes with double quotes
        if "'" in fixed:
            # Only replace single quotes used as JSON delimiters (not inside strings)
            fixed = re.sub(r"(?<![\\])'", '"', fixed)
        # Fix unquoted property names: {key: value} -> {"key": value}
        fixed = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', fixed)
        # Fix Python-style True/False/None
        fixed = fixed.replace(': True', ': true').replace(': False', ': false').replace(': None', ': null')
        # Remove control characters
        fixed = re.sub(r'[\x00-\x1f]+', ' ', fixed)
        # Try to fix unterminated strings by closing them
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
        # Last resort: extract any valid JSON object/array
        for match in re.finditer(r'\{[^{}]*\}|\[[^\[\]]*\]', fixed):
            try:
                json.loads(match.group())
                return match.group()
            except json.JSONDecodeError:
                continue
        return fixed if fixed != json_str else None

    def replan_primitives(
        self,
        actions: List[HighLevelAction],
        instruction: str,
        images: Dict[str, np.ndarray],
        objects: Dict[str, ObjectInfo],
    ) -> Optional[List[PrimitivePlan]]:
        """
        Selective replanning: regenerate only primitive sequences while
        preserving existing high-level actions.

        Called when scene divergence exceeds threshold. Keeps the action
        plan intact but regenerates low-level waypoints using updated
        object positions.

        Args:
            actions: Existing high-level actions to preserve
            instruction: Original task instruction
            images: Current camera images
            objects: Updated object info with new positions

        Returns:
            List of PrimitivePlan with regenerated primitives, or None on failure
        """
        if not self._llm and not self._hf_model:
            self.initialize_client()

        prompt = self._build_replan_prompt(actions, instruction, objects)
        provider = self.config.provider.lower()

        try:
            if provider in ("hf", "huggingface"):
                plans = self._replan_with_hf(prompt, images)
            else:
                plans = self._replan_with_langchain(prompt, images)

            if plans:
                # Restore original action metadata onto regenerated plans
                for i, plan in enumerate(plans):
                    if i < len(actions):
                        plan.parent_action = actions[i]
                return plans
        except Exception as e:
            logger.warning(f"Selective replanning failed: {e}")

        # Fallback: full replan
        logger.info("Selective replan failed, falling back to full plan()")
        return self.plan(instruction, images, objects)

    def _replan_with_langchain(
        self,
        prompt: str,
        images: Dict[str, np.ndarray],
    ) -> Optional[List[PrimitivePlan]]:
        """Replan primitives using LangChain-based providers."""
        from langchain_core.messages import HumanMessage

        content = self._build_message_content(prompt, images)
        try:
            response = self._llm.invoke([HumanMessage(content=content)])
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"LangChain replan failed: {e}")
            return None

    def _replan_with_hf(
        self,
        prompt: str,
        images: Dict[str, np.ndarray],
    ) -> Optional[List[PrimitivePlan]]:
        """Replan primitives using local HuggingFace VLM."""
        import torch

        if not self._hf_model or not self._hf_processor:
            logger.error("HuggingFace VLM not initialized")
            return None

        pil_images = [self._to_pil_image(img) for img in images.values() if img is not None]
        content = []
        for img in pil_images:
            if img is not None:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]

        try:
            inputs = self._hf_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = self._hf_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self._hf_model.generate(**inputs, max_new_tokens=self.max_tokens)

            generated_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
            response_text = self._hf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return self._parse_response(response_text)
        except Exception as e:
            logger.error(f"HuggingFace replan failed: {e}")
            return None

    def _build_replan_prompt(
        self,
        actions: List[HighLevelAction],
        instruction: str,
        objects: Dict[str, ObjectInfo],
    ) -> str:
        """Build prompt for selective primitive replanning."""
        objects_info = self._format_objects_info(objects)

        action_lines = []
        for i, action in enumerate(actions):
            action_lines.append(
                f"  {i+1}. {action.action_type}({action.target_object})"
            )
        actions_str = "\n".join(action_lines)

        return f"""You are a robot motion planner. The high-level action plan is already decided.
Regenerate ONLY the primitive motion sequences using the UPDATED object positions below.

## Original Task
{instruction}

## High-Level Actions (DO NOT CHANGE)
{actions_str}

## Updated Scene Objects (robot-base frame, meters)
{objects_info}

## Primitives
- move(x, y, z): Move gripper to position
  - x, y, z: position in meters (robot-base frame)
- grip(width): Control gripper ({self.grip_open_width}=open, 0=closed)

## Planning Rules

### 1. Preserve Actions
Keep the exact same action sequence above. Only regenerate primitives for each action.

### 2. Collision Avoidance
- All waypoints must stay OUTSIDE obstacle bboxes
- Add 0.03m safety margin from bbox boundaries

### 3. Short Waypoint Distance (IMPORTANT)
- Each move() must travel at most 0.05m (5cm) from the previous position
- Break long movements into multiple small waypoints

### 4. Height Guidelines
- Approach from above: target z + 0.10m
- Contact: target z + 0.02m
- Lift/Retreat: +0.15m above current position

## Output (JSON only)
{{"plans": [
  {{"action": "<action_type>", "target": "<object_name>", "primitives": [
    {{"type": "grip", "grip_width": ...}},
    {{"type": "move", "target_position": {{"x": ..., "y": ..., "z": ...}}}},
    ...
  ]}},
  ...
]}}"""

    def process(
        self,
        instruction: str = "",
        images: Optional[Dict[str, np.ndarray]] = None,
        objects: Optional[Dict[str, ObjectInfo]] = None,
        eef_pos: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Optional[List[PrimitivePlan]]:
        """
        BaseModule interface method.

        Delegates to plan() method for actual planning.

        Args:
            instruction: Natural language task instruction
            images: Dict of camera_name -> image array
            objects: Dict of object_name -> ObjectInfo
            eef_pos: Current end-effector position [x, y, z] in robot-base frame
            **kwargs: Additional arguments (ignored)

        Returns:
            List of PrimitivePlan
        """
        # HTTP remote mode
        if self.config.link_mode == "http":
            payload = {"instruction": instruction}
            if images:
                payload["images"] = {
                    k: self._encode_image(v) for k, v in images.items() if v is not None
                }
            if objects:
                payload["objects"] = {k: v.to_dict() for k, v in objects.items()}
            result = self._http_post("process", payload, timeout=30.0)
            plans = result.get("plans")
            if plans is None:
                return None
            from .types import PrimitivePlan
            return [PrimitivePlan.from_dict(p) for p in plans]

        return self.plan(
            instruction=instruction,
            images=images or {},
            objects=objects or {},
            eef_pos=eef_pos,
        )
