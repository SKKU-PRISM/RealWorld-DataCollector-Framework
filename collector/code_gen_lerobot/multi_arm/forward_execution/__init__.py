"""
Multi-Arm Forward Execution Code Generation

Turn structure:
  Turn 0: Scene understanding (multi-arm specific — describes two arms + object reach)
  Turn 1: BBox detection (reused from single-arm — shared camera)
  Turn 2: Crop-then-point (reused from single-arm — object property, not robot-specific)
  Turn 3: Code generation (multi-arm specific — move_to_position/execute_pick_object API)
"""

from .system_prompt import MULTI_ARM_PERCEPTION_SYSTEM_PROMPT, MULTI_ARM_CODEGEN_SYSTEM_PROMPT
from .turn3_prompt import multi_arm_turn3_codegen_prompt
from .turn0_prompt import multi_arm_turn0_scene_understanding_prompt
from ..multi_skill_api_doc import MULTI_ARM_API_DOC
