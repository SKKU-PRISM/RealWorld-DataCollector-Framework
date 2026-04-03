"""
VLA module types and data structures.

Defines input/output formats for VLA inference and LoRA adapter configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VLAInput:
    """Input to VLA model inference."""

    images: Dict[str, np.ndarray]  # camera_name -> (H, W, 3) uint8
    robot_state: np.ndarray  # (state_dim,) proprioception vector
    instruction: str  # natural language instruction
    primitive_type: str  # "move" or "grip" - determines adapter
    state_mask: Optional[np.ndarray] = None  # (64,) bool — GROOT normalized+padded state


@dataclass
class VLAOutput:
    """Output from VLA model inference."""

    action: np.ndarray  # (action_dim,) predicted action
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAAdapterConfig:
    """Configuration for a single LoRA adapter."""

    name: str  # e.g. "move", "grip"
    path: str  # path to adapter weights
    primitive_types: List[str]  # ["move"] or ["grip"]
    task_name: Optional[str] = None  # None = general, else task-specific


@dataclass
class VLAModelConfig:
    """Configuration for a VLA model backend."""

    backend: str  # "openvla", "lerobot", "hf_vlm"
    model_name: str  # HuggingFace model ID or path
    device: str = "cuda:0"
    quantize_4bit: bool = False
    action_dim: int = 7  # rel_pos(3) + rel_rot(3) + gripper(1)
    image_size: int = 224
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    chunk_stride: int = 8  # action chunk stride (re-infer every N steps)
    extra: Dict[str, Any] = field(default_factory=dict)
