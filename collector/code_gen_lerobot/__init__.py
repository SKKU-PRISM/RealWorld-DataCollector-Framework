"""
Code Generation Module for LeRobot CaP Pipeline

This module provides Code-as-Policies (CaP) based code generation for LeRobot SO-101 robot control.

Submodules:
    - forward_execution: Forward task execution (pick & place, etc.)
    - reset_execution: Reset/reverse task execution (undo forward tasks)
    - judge: Task evaluation and success/failure judgment

Usage:
    # Forward execution
    from code_gen_lerobot.forward_execution import lerobot_code_gen
    code, positions = lerobot_code_gen(
        instruction="pick red cup and place on blue box",
        object_queries=["red cup", "blue box"],
    )

    # Save execution context
    from code_gen_lerobot.execution_context import save_forward_context
    context = save_forward_context(
        instruction="pick red cup and place on blue box",
        object_positions=positions,
        generated_code=code,
    )

    # Reset execution
    from code_gen_lerobot.reset_execution import lerobot_reset_code_gen
    reset_code, _ = lerobot_reset_code_gen(
        original_instruction=context.instruction,
        original_positions=context.object_positions,
        executed_code=context.generated_code,
    )
"""

# Backward compatibility imports
# These allow existing code to work without changes
from .forward_execution import (
    lerobot_code_gen,
    extract_code_from_response,
)

from .execution_context import (
    ExecutionContext,
    ExecutionContextManager,
    save_forward_context,
    load_latest_context,
)

# Version info
__version__ = "2.0.0"

__all__ = [
    # Forward execution
    "lerobot_code_gen",
    "extract_code_from_response",
    # Execution context
    "ExecutionContext",
    "ExecutionContextManager",
    "save_forward_context",
    "load_latest_context",
]
