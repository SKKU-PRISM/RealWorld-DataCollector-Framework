"""
Multi-Arm Code Generation Module

Provides bi-arm (left_arm + right_arm) code generation for the unified multi-arm pipeline.
Turn 1 (bbox detection) and Turn 2 (crop-then-point) are reused from single-arm forward_execution.
Turn 0 (scene understanding) and Turn 3 (code generation) are multi-arm specific.
"""

from .forward_execution.system_prompt import MULTI_ARM_PERCEPTION_SYSTEM_PROMPT, MULTI_ARM_CODEGEN_SYSTEM_PROMPT
from .forward_execution.turn3_prompt import multi_arm_turn3_codegen_prompt
from .forward_execution.turn0_prompt import multi_arm_turn0_scene_understanding_prompt
from .multi_skill_api_doc import MULTI_ARM_API_DOC
from .reset_execution.turn3_prompt import multi_arm_turn3_reset_codegen_prompt

__all__ = [
    "MULTI_ARM_PERCEPTION_SYSTEM_PROMPT",
    "MULTI_ARM_CODEGEN_SYSTEM_PROMPT",
    "multi_arm_turn3_codegen_prompt",
    "multi_arm_turn0_scene_understanding_prompt",
    "MULTI_ARM_API_DOC",
    "multi_arm_turn3_reset_codegen_prompt",
]
