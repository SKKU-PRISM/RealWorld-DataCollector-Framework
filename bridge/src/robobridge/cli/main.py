#!/usr/bin/env python3
"""
RoboBridge CLI - Main Entry Point

Command line interface for RoboBridge framework.

Usage:
    robobridge server                    # Run RoboBridge server
    robobridge module --module planner   # Run specific module
    robobridge demo                      # Run demo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional

from modules import (
    Perception,
    Planner,
    Controller,
    Robot,
    Monitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Base prompt for planning and feedback
def build_base_prompt(robot_type: str, skill_list: list) -> str:
    skills_text = ", ".join(skill_list)
    
    return f"""You are an intelligent robot task planner for manipulation tasks.

Your role is to decompose natural language instructions into executable robot actions.
You have access to a {robot_type} robot arm.

Available skills: {skills_text}

When planning:
1. Consider object positions and reachability
2. Plan efficient motion sequences
3. Account for physical constraints
4. Include error handling for common failures

Always output valid JSON in the specified format."""


def create_robobridge_instance(config_path: str = "./config.py"):
    """Create and configure RoboBridge instance."""
    from ..core import RoboBridge

    robobridge = RoboBridge(
        ros_domain_id=0,
        ros_namespace="/robobridge",
        adapters_config_path=config_path,
        trace=True,
        log_dir="./logs/robobridge",
    )
    return robobridge


def _get_adapter_params(adapters: Dict[str, Any], module_name: str) -> dict:
    """Extract adapter connection parameters from ADAPTERS config."""
    adapter = adapters.get(module_name, {})
    if hasattr(adapter, "link_mode"):
        # It's an AdapterConfig object
        return {
            "link_mode": adapter.link_mode,
            "adapter_endpoint": (adapter.bind_host, adapter.bind_port),
            "adapter_protocol": "len_json",
            "auth_token": adapter.auth_token,
        }
    # It's a dict (legacy format)
    return {
        "link_mode": adapter.get("link_mode", "socket"),
        "adapter_endpoint": (
            adapter.get("bind_host", "127.0.0.1"),
            adapter.get("bind_port", 51000),
        ),
        "adapter_protocol": "len_json",
        "auth_token": adapter.get("auth_token"),
    }


def create_modules(config_path: str = "./config.py") -> Dict[str, Any]:
    """
    Create module instances using centralized configuration.

    All module settings are defined in config.py:
    - MODULE_CONFIGS: Model providers, parameters, and behavior settings
    - ADAPTERS: Socket connections, topics, and QoS profiles

    To change settings, edit config.py instead of this file.
    """
    from ..modules import Perception, Planner, Controller, Robot, Monitor
    from ..config import load_config

    config = load_config(config_path)
    module_configs = config.module_configs
    adapters = config.adapters

    modules = {}

    # Perception (object detector)
    cfg = module_configs.get("perception", module_configs.get("object_detector", {}))
    adapter = _get_adapter_params(adapters, "perception")
    modules["perception"] = Perception(
        provider=cfg.get("provider", "hf"),
        model=cfg.get("model", "florence-2"),
        device=cfg.get("device", "cuda:0"),
        api_key=cfg.get("api_key"),
        image_size=cfg.get("image_size", 640),
        conf_threshold=cfg.get("conf_threshold", 0.25),
        nms_threshold=cfg.get("nms_threshold", 0.5),
        max_dets=cfg.get("max_dets", 50),
        pose_format=cfg.get("pose_format", "pose_quat"),
        frame_id=cfg.get("frame_id", "camera_color_optical_frame"),
        timeout_s=cfg.get("timeout_s", 3.0),
        max_retries=cfg.get("max_retries", 2),
        rgb_topic="/camera/rgb",
        depth_topic="/camera/depth",
        object_list_topic="/robobridge/object_list",
        poses_topic="/perception/object_poses",
        **adapter,
    )

    # Planner (high-level planner)
    cfg = module_configs.get("planner", module_configs.get("high_level_planner", {}))
    adapter = _get_adapter_params(adapters, "planner")
    modules["planner"] = Planner(
        provider=cfg.get("provider", "openai"),
        model=cfg.get("model", "gpt-4o"),
        api_key=cfg.get("api_key"),
        api_base=cfg.get("api_base"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 1500),
        max_plan_steps=cfg.get("max_plan_steps", 12),
        retry_on_parse_error=cfg.get("retry_on_parse_error", True),
        skill_list=cfg.get("skill_list", ["pick", "place", "push", "pull"]),
        default_recovery_target=cfg.get("default_recovery_target", "low_level_planning"),
        timeout_s=cfg.get("timeout_s", 12.0),
        max_retries=cfg.get("max_retries", 2),
        base_prompt=BASE_PROMPT,
        plan_schema="robobridge/schemas/high_level_plan.json",
        instruction_topic="/robobridge/instruction",
        poses_topic="/perception/object_poses",
        feedback_topic="/feedback/failure_signal",
        plan_topic="/planning/high_level_plan",
        **adapter,
    )

    # Controller (low-level planner)
    cfg = module_configs.get("controller", module_configs.get("low_level_planner", {}))
    adapter = _get_adapter_params(adapters, "controller")
    modules["controller"] = Controller(
        provider=cfg.get("provider", "vla"),
        backend=cfg.get("backend", "primitives"),
        model=cfg.get("model", "pi0"),
        device=cfg.get("device", "cuda:0"),
        api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.5),
        action_space=cfg.get("action_space", "trajectory"),
        control_rate_hz=cfg.get("control_rate_hz", 20),
        horizon_steps=cfg.get("horizon_steps", 10),
        frame_convention=cfg.get("frame_convention", "base"),
        safety_limits=cfg.get("safety_limits", {}),
        timeout_s=cfg.get("timeout_s", 6.0),
        max_retries=cfg.get("max_retries", 1),
        plan_topic="/planning/high_level_plan",
        rgb_topic="/camera/rgb",
        depth_topic="/camera/depth",
        robot_state_topic="/robot/state",
        lowlevel_cmd_topic="/planning/low_level_cmd",
        **adapter,
    )

    # Robot (robot interface)
    cfg = module_configs.get("robot", module_configs.get("robot_interface", {}))
    adapter = _get_adapter_params(adapters, "robot")

    # Use simulation mode by default
    custom_interface = "simulation"
    if cfg.get("backend") != "sim":
        logger.warning(
            f"Backend '{cfg.get('backend')}' configured but no CustomRobot provided. "
            "Using simulation mode. To use real robot, provide a CustomRobot instance."
        )

    modules["robot"] = Robot(
        custom_interface=custom_interface,
        robot_type=cfg.get("robot_type", "franka"),
        rate_hz=cfg.get("rate_hz", 100),
        timeout_s=cfg.get("timeout_s", 15.0),
        units=cfg.get("units", "SI"),
        frame_convention=cfg.get("frame_convention", "base"),
        estop_policy=cfg.get("estop_policy", "stop_and_report"),
        max_retries=cfg.get("max_retries", 1),
        lowlevel_cmd_topic="/planning/low_level_cmd",
        exec_result_topic="/robot/execution_result",
        robot_state_topic="/robot/state",
        **adapter,
    )

    # Monitor (feedback)
    cfg = module_configs.get("monitor", module_configs.get("feedback", {}))
    adapter = _get_adapter_params(adapters, "monitor")
    modules["monitor"] = Monitor(
        provider=cfg.get("provider", "openai"),
        model=cfg.get("model", "gpt-4o"),
        api_key=cfg.get("api_key"),
        api_base=cfg.get("api_base"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 200),
        image_size=cfg.get("image_size", 224),
        observation_rate_hz=cfg.get("observation_rate_hz", 10.0),
        failure_confidence_threshold=cfg.get("failure_confidence_threshold", 0.7),
        stop_on_consecutive_failures=cfg.get("stop_on_consecutive_failures", 2),
        only_publish_on_failure=cfg.get("only_publish_on_failure", True),
        enable_continuous_mode=cfg.get("enable_continuous_mode", True),
        timeout_s=cfg.get("timeout_s", 4.0),
        max_retries=cfg.get("max_retries", 1),
        base_prompt=BASE_PROMPT,
        decision_schema="robobridge/schemas/feedback_decision.json",
        rgb_topic="/camera/rgb",
        plan_topic="/planning/high_level_plan",
        exec_result_topic="/robot/execution_result",
        feedback_topic="/feedback/failure_signal",
        **adapter,
    )

    return modules


def run_server(config_path: str = "./config.py"):
    """Run RoboBridge as the central orchestrator."""
    logger.info("Starting RoboBridge server...")

    robobridge = create_robobridge_instance(config_path)

    try:
        robobridge.start()
        logger.info("RoboBridge server running. Waiting for module connections...")
        robobridge.spin()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        robobridge.shutdown()
        logger.info("RoboBridge server stopped")


def run_module(module_name: str, config_path: str = "./config.py"):
    """Run a specific module."""
    logger.info(f"Starting module: {module_name}")

    modules = create_modules(config_path)

    if module_name not in modules:
        logger.error(f"Unknown module: {module_name}")
        logger.info(f"Available modules: {list(modules.keys())}")
        return

    module = modules[module_name]

    try:
        module.start()
        logger.info(f"Module {module_name} running...")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        module.stop()
        logger.info(f"Module {module_name} stopped")


def run_demo(config_path: str = "./config.py"):
    """Run a demo with simulated components."""
    import json

    logger.info("Starting RoboBridge demo...")

    robobridge = create_robobridge_instance(config_path)

    try:
        robobridge.start()
        logger.info("RoboBridge started")

        time.sleep(2.0)

        logger.info("Publishing test instruction...")
        robobridge.publish(
            "/robobridge/instruction",
            {"data": json.dumps({"instruction": "Pick up the red cup and place it on the table"})},
        )

        time.sleep(10.0)

    except KeyboardInterrupt:
        logger.info("Demo interrupted")
    finally:
        robobridge.shutdown()
        logger.info("Demo complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RoboBridge - General Robot Manipulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  robobridge server                           # Run RoboBridge server
  robobridge module --module planner          # Run specific module
  robobridge demo                             # Run demo

Available Modules:
  perception   - Object detection (Florence-2, YOLO, etc.)
  planner      - High-level planning (LangChain LLM)
  controller   - Low-level planning (VLA, MoveIt)
  robot        - Robot interface (Franka, UR, simulation)
  monitor      - Execution feedback (VLM)

Environment Variables:
  OPENAI_API_KEY      OpenAI API key (for openai provider)
  ANTHROPIC_API_KEY   Anthropic API key (for anthropic provider)
  GOOGLE_API_KEY      Google API key (for google provider)
  HF_API_KEY          HuggingFace API key (for huggingface provider)
        """,
    )
    parser.add_argument(
        "command",
        choices=["server", "module", "demo"],
        help="Command to run",
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Module name (for 'module' command)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.py",
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.command == "server":
        run_server(args.config)
    elif args.command == "module":
        if not args.module:
            logger.error("Module name required. Use --module <name>")
            logger.info("Available modules: perception, planner, controller, robot, monitor")
            sys.exit(1)
        run_module(args.module, args.config)
    elif args.command == "demo":
        run_demo(args.config)


if __name__ == "__main__":
    main()
