"""
Execution Context Manager

Manages the execution context between forward and reset task executions.
Saves forward execution details so that reset execution can properly reverse the task.

Usage:
    # After forward execution
    context = ExecutionContext()
    context.save_forward_context(
        instruction="pick red cup and place on blue box",
        object_positions={"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]},
        generated_code=forward_code,
        execution_success=True,
    )

    # Before reset execution
    context = ExecutionContext.load("outputs/execution_context.json")
    reset_code = lerobot_reset_code_gen(
        original_instruction=context.instruction,
        original_positions=context.object_positions,
        executed_code=context.generated_code,
    )
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExecutionContext:
    """
    Execution context for CaP-based robot control.

    Stores all necessary information to reverse a forward execution.

    Attributes:
        instruction: Natural language goal (e.g., "pick red cup and place on blue box")
        object_positions: Original object positions before forward execution {name: [x, y, z]}
        generated_spec: Structured spec generated for forward execution (step-by-step plan)
        generated_code: Python code generated for forward execution
        execution_success: Whether forward execution completed successfully
        timestamp: When the forward execution occurred
        robot_id: Robot ID used for execution
        metadata: Additional metadata (optional)
    """

    instruction: str = ""
    object_positions: Dict[str, Any] = field(default_factory=dict)  # Extended format: {name: {"position": [...]}}
    generated_spec: Dict[str, Any] = field(default_factory=dict)
    generated_code: str = ""
    execution_success: bool = False
    timestamp: str = ""
    robot_id: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def save(self, path: str) -> str:
        """
        Save execution context to JSON file.

        Args:
            path: Output file path (e.g., "outputs/execution_context.json")

        Returns:
            Saved file path
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Convert to dict and handle numpy types
        data = asdict(self)

        def convert_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = convert_types(data)

        # JSON 저장
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[ExecutionContext] Saved to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ExecutionContext":
        """
        Load execution context from JSON file.

        Args:
            path: Input file path

        Returns:
            ExecutionContext instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        context = cls(**data)
        print(f"[ExecutionContext] Loaded from: {path}")
        return context

    def is_valid_for_reset(self) -> bool:
        """
        Check if context has all required fields for reset execution.

        Returns:
            True if context is valid for reset
        """
        if not self.instruction:
            print("[ExecutionContext] Missing: instruction")
            return False

        if not self.object_positions:
            print("[ExecutionContext] Missing: object_positions")
            return False

        if not self.generated_code:
            print("[ExecutionContext] Missing: generated_code")
            return False

        return True

    def summary(self) -> str:
        """
        Generate human-readable summary of execution context.

        Returns:
            Summary string
        """
        lines = [
            "=" * 60,
            "Execution Context Summary",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Robot ID: {self.robot_id}",
            f"Instruction: {self.instruction}",
            f"Execution Success: {self.execution_success}",
            "",
            "Object Positions:",
        ]

        for name, info in self.object_positions.items():
            if info is None:
                lines.append(f"  - {name}: None")
            elif isinstance(info, dict) and "position" in info:
                # Extended format
                pos = info["position"]
                lines.append(f"  - {name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            elif isinstance(info, (list, tuple)) and len(info) >= 3:
                # Legacy format
                lines.append(f"  - {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")
            else:
                lines.append(f"  - {name}: {info}")

        lines.append("")
        lines.append(f"Generated Code: {len(self.generated_code)} characters")

        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  - {key}: {value}")

        lines.append("=" * 60)

        return "\n".join(lines)


def save_forward_context(
    instruction: str,
    object_positions: Dict[str, Any],
    generated_spec: Dict[str, Any] = None,
    generated_code: str = "",
    execution_success: bool = True,
    robot_id: int = 3,
    output_dir: str = "outputs",
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    """
    Convenience function to save forward execution context.

    Args:
        instruction: Natural language goal
        object_positions: Object positions in extended format {name: {"position": [x,y,z]}}
        generated_spec: Generated spec (step-by-step plan)
        generated_code: Generated Python code
        execution_success: Whether execution succeeded
        robot_id: Robot ID
        output_dir: Output directory
        metadata: Additional metadata

    Returns:
        Created ExecutionContext instance
    """
    context = ExecutionContext(
        instruction=instruction,
        object_positions=object_positions,
        generated_spec=generated_spec or {},
        generated_code=generated_code,
        execution_success=execution_success,
        robot_id=robot_id,
        metadata=metadata or {},
    )

    # 최신 컨텍스트 파일만 저장 (latest)
    latest_path = os.path.join(output_dir, "execution_context.json")
    context.save(latest_path)

    return context


def load_latest_context(output_dir: str = "outputs") -> Optional[ExecutionContext]:
    """
    Load the most recent execution context.

    Args:
        output_dir: Output directory

    Returns:
        ExecutionContext or None if not found
    """
    latest_path = os.path.join(output_dir, "execution_context.json")

    if os.path.exists(latest_path):
        return ExecutionContext.load(latest_path)

    print(f"[ExecutionContext] No context found at: {latest_path}")
    return None


# For backward compatibility with original module location
class ExecutionContextManager:
    """
    Manager class for handling multiple execution contexts.
    Provides utilities for context history and cleanup.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        instruction: str,
        object_positions: Dict[str, List[float]],
        generated_spec: Dict[str, Any] = None,
        generated_code: str = "",
        execution_success: bool = True,
        robot_id: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """Save forward execution context."""
        return save_forward_context(
            instruction=instruction,
            object_positions=object_positions,
            generated_spec=generated_spec,
            generated_code=generated_code,
            execution_success=execution_success,
            robot_id=robot_id,
            output_dir=self.output_dir,
            metadata=metadata,
        )

    def load_latest(self) -> Optional[ExecutionContext]:
        """Load the most recent execution context."""
        return load_latest_context(self.output_dir)

    def list_contexts(self) -> List[str]:
        """List all saved execution contexts."""
        if not os.path.exists(self.output_dir):
            return []

        contexts = []
        for f in os.listdir(self.output_dir):
            if f.startswith("execution_context_") and f.endswith(".json"):
                if f != "execution_context_latest.json":
                    contexts.append(os.path.join(self.output_dir, f))

        return sorted(contexts, reverse=True)  # Most recent first

    def cleanup_old_contexts(self, keep_count: int = 10) -> int:
        """
        Remove old execution contexts, keeping only the most recent ones.

        Args:
            keep_count: Number of contexts to keep

        Returns:
            Number of deleted contexts
        """
        contexts = self.list_contexts()
        to_delete = contexts[keep_count:]

        for path in to_delete:
            os.remove(path)
            print(f"[ExecutionContext] Deleted: {path}")

        return len(to_delete)
