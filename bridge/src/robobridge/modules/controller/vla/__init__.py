"""
VLA (Vision-Language-Action) module.

Provides model abstraction, LoRA adapter management, and inference
for multiple VLA backends (OpenVLA, LeRobot, HF VLM).
"""

from .base_vla import BaseVLA
from .lora_manager import LoRAManager
from .registry import get_vla_backend, list_vla_backends, register_vla
from .types import LoRAAdapterConfig, VLAInput, VLAModelConfig, VLAOutput

__all__ = [
    "BaseVLA",
    "LoRAManager",
    "VLAInput",
    "VLAOutput",
    "VLAModelConfig",
    "LoRAAdapterConfig",
    "register_vla",
    "get_vla_backend",
    "list_vla_backends",
]
