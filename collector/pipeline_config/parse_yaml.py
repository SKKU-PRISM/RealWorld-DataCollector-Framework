#!/usr/bin/env python3
"""
YAML Config Parser for Shell Scripts

Usage:
    # Parse entire config to shell variables
    eval "$(python3 pipeline_config/parse_yaml.py pipeline_config/paid_api_config.yaml)"

    # Get specific value
    python3 pipeline_config/parse_yaml.py pipeline_config/paid_api_config.yaml codegen_llm_model
"""

import sys
import yaml
from pathlib import Path


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dict to single level with underscore-separated keys.

    Returns: dict with values as tuples (value, is_list)
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to space-separated string for bash array
            items.append((new_key, (' '.join(f'"{item}"' for item in v), True)))
        else:
            items.append((new_key, (v, False)))
    return dict(items)


def to_shell_var(key, value, is_list=False):
    """Convert key-value pair to shell variable assignment."""
    key = key.upper()
    if isinstance(value, bool):
        return f'{key}={"true" if value else "false"}'
    elif isinstance(value, (int, float)):
        return f'{key}={value}'
    elif isinstance(value, str):
        # If this was originally a list, output as bash array
        if is_list:
            return f'{key}=({value})'
        return f'{key}="{value}"'
    else:
        return f'{key}="{value}"'


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_yaml.py <config.yaml> [key]", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Flatten nested config
    flat_config = flatten_dict(config)

    # If specific key requested, return just that value
    if len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key in flat_config:
            value, is_list = flat_config[key]
            if isinstance(value, bool):
                print("true" if value else "false")
            else:
                print(value)
        else:
            print(f"Error: Key '{key}' not found in config", file=sys.stderr)
            sys.exit(1)
    else:
        # Output all variables for eval
        for key, (value, is_list) in flat_config.items():
            print(to_shell_var(key, value, is_list))


if __name__ == "__main__":
    main()
