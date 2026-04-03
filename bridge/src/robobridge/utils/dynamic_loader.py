"""
Dynamic Class Loading Utilities

Helper functions for loading custom classes from file paths.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Type

logger = logging.getLogger(__name__)


def load_custom_class(model_path: str, base_class: Optional[Type] = None) -> Type:
    """
    Dynamically load a class from a file path.

    Loads a Python class from a file using the format:
    "/path/to/file.py:ClassName"

    Args:
        model_path: Path in format "/path/to/wrapper.py:ClassName"
        base_class: Optional base class to verify inheritance

    Returns:
        The loaded class (not an instance)

    Raises:
        ValueError: If model_path format is invalid
        FileNotFoundError: If the specified file doesn't exist
        AttributeError: If the class doesn't exist in the module
        TypeError: If the class doesn't inherit from base_class (when specified)

    Example:
        >>> cls = load_custom_class("/home/user/my_planner.py:MyLLMPlanner")
        >>> instance = cls(model_path="...", device="cuda:0")

        >>> from robobridge.wrappers import CustomPlanner
        >>> cls = load_custom_class(
        ...     "/home/user/my_planner.py:MyLLMPlanner",
        ...     base_class=CustomPlanner
        ... )
    """
    # Validate format
    if ":" not in model_path:
        raise ValueError(
            f"Invalid model_path format: '{model_path}'. "
            f"Expected format: '/path/to/file.py:ClassName'"
        )

    # Split path and class name
    file_path, class_name = model_path.rsplit(":", 1)

    # Validate file path
    file_path = os.path.expanduser(file_path)  # Handle ~ in path
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom wrapper file not found: {file_path}")

    if not file_path.endswith(".py"):
        raise ValueError(f"Custom wrapper file must be a .py file: {file_path}")

    # Generate unique module name to avoid conflicts
    module_name = f"robobridge_custom_{Path(file_path).stem}_{id(model_path)}"

    logger.debug(f"Loading custom class '{class_name}' from '{file_path}'")

    # Load module from file
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from module
    if not hasattr(module, class_name):
        available_classes = [
            name
            for name in dir(module)
            if isinstance(getattr(module, name), type) and not name.startswith("_")
        ]
        raise AttributeError(
            f"Class '{class_name}' not found in '{file_path}'. "
            f"Available classes: {available_classes}"
        )

    loaded_class = getattr(module, class_name)

    # Verify it's a class
    if not isinstance(loaded_class, type):
        raise TypeError(f"'{class_name}' is not a class, got: {type(loaded_class)}")

    # Verify inheritance if base_class is specified
    if base_class is not None:
        if not issubclass(loaded_class, base_class):
            raise TypeError(
                f"Class '{class_name}' must inherit from '{base_class.__name__}', "
                f"but inherits from: {[c.__name__ for c in loaded_class.__bases__]}"
            )

    logger.info(f"Successfully loaded custom class: {class_name} from {file_path}")
    return loaded_class


def parse_custom_model_path(model_path: str) -> Tuple[str, str]:
    """
    Parse a custom model path into file path and class name.

    Args:
        model_path: Path in format "/path/to/wrapper.py:ClassName"

    Returns:
        Tuple of (file_path, class_name)

    Raises:
        ValueError: If format is invalid
    """
    if ":" not in model_path:
        raise ValueError(
            f"Invalid model_path format: '{model_path}'. "
            f"Expected format: '/path/to/file.py:ClassName'"
        )

    file_path, class_name = model_path.rsplit(":", 1)
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    return file_path, class_name
