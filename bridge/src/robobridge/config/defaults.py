"""
Default configurations for RoboBridge.
"""

import os
from typing import Any, Dict

# Default QoS profiles
DEFAULT_QOS_PROFILES: Dict[str, Dict[str, Any]] = {
    "sensor_data": {
        "reliability": "best_effort",
        "durability": "volatile",
        "depth": 1,
    },
    "reliable_cmd": {
        "reliability": "reliable",
        "durability": "volatile",
        "depth": 10,
    },
}

# Default socket configuration
DEFAULT_SOCKET_DEFAULTS: Dict[str, Any] = {
    "protocol": "len_json",
    "recv_timeout_s": 3.0,
    "send_timeout_s": 3.0,
    "max_connections": 8,
    "max_payload_bytes": 10485760,  # 10MB
}

# Default module configurations
DEFAULT_MODULE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "perception": {
        "provider": "hf",
        "model": "microsoft/Florence-2-base",
        "device": "cuda:0",
        "api_key": os.getenv("HF_API_KEY"),
        "image_size": 640,
        "conf_threshold": 0.25,
        "nms_threshold": 0.50,
        "max_dets": 50,
        "pose_format": "pose_quat",
        "frame_id": "camera_color_optical_frame",
        "timeout_s": 3.0,
        "max_retries": 2,
    },
    "planner": {
        "provider": "hf",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "api_key": os.getenv("HF_API_KEY"),
        "api_base": None,
        "temperature": 0.7,
        "max_tokens": 1500,
        "max_plan_steps": 12,
        "retry_on_parse_error": True,
        "skill_list": ["pick", "place", "open", "close", "push", "pull"],
        "default_recovery_target": "controller",
        "timeout_s": 12.0,
        "max_retries": 2,
    },
    "controller": {
        "provider": "vla",
        "backend": "primitives",
        "model": "pi0",
        "device": "cuda:0",
        "api_key": os.getenv("VLA_API_KEY"),
        "temperature": 0.5,
        "action_space": "trajectory",
        "control_rate_hz": 20,
        "horizon_steps": 10,
        "frame_convention": "base",
        "safety_limits": {
            "max_joint_vel": 1.5,
            "max_ee_vel": 0.3,
            "workspace": [-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
        },
        "timeout_s": 6.0,
        "max_retries": 1,
    },
    "robot": {
        "backend": "sim",
        "robot_type": "franka",
        "controller_mode": "impedance",
        "rate_hz": 100,
        "timeout_s": 15.0,
        "units": "SI",
        "frame_convention": "base",
        "estop_policy": "stop_and_report",
        "max_retries": 1,
    },
    "monitor": {
        "provider": "hf",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "api_key": os.getenv("HF_API_KEY"),
        "api_base": None,
        "temperature": 0.0,
        "max_tokens": 200,
        "image_size": 224,
        "observation_rate_hz": 10.0,
        "failure_confidence_threshold": 0.7,
        "stop_on_consecutive_failures": 2,
        "only_publish_on_failure": True,
        "enable_continuous_mode": True,
        "timeout_s": 4.0,
        "max_retries": 1,
    },
}

# Default adapter configurations
DEFAULT_ADAPTERS: Dict[str, Dict[str, Any]] = {
    "perception": {
        "link_mode": "socket",
        "bind_host": "0.0.0.0",
        "bind_port": 51001,
        "auth_token": "auth_perception",
        "pub_topics": ["/perception/objects"],
        "sub_topics": ["/camera/rgb", "/camera/depth", "/perception/object_list"],
        "topic_types": {
            "/camera/rgb": "sensor_msgs/Image",
            "/camera/depth": "sensor_msgs/Image",
            "/perception/object_list": "std_msgs/String",
            "/perception/objects": "std_msgs/String",
        },
        "qos": {
            "/camera/rgb": "sensor_data",
            "/camera/depth": "sensor_data",
            "/perception/object_list": "reliable_cmd",
            "/perception/objects": "reliable_cmd",
        },
    },
    "planner": {
        "link_mode": "socket",
        "bind_host": "0.0.0.0",
        "bind_port": 51002,
        "auth_token": "auth_planner",
        "pub_topics": ["/planning/plan"],
        "sub_topics": ["/robobridge/instruction", "/perception/objects", "/feedback/signal"],
        "topic_types": {
            "/robobridge/instruction": "std_msgs/String",
            "/perception/objects": "std_msgs/String",
            "/feedback/signal": "std_msgs/String",
            "/planning/plan": "std_msgs/String",
        },
        "qos": {
            "/robobridge/instruction": "reliable_cmd",
            "/perception/objects": "reliable_cmd",
            "/feedback/signal": "reliable_cmd",
            "/planning/plan": "reliable_cmd",
        },
    },
    "controller": {
        "link_mode": "socket",
        "bind_host": "0.0.0.0",
        "bind_port": 51003,
        "auth_token": "auth_controller",
        "pub_topics": ["/control/command"],
        "sub_topics": ["/planning/plan", "/camera/rgb", "/camera/depth", "/robot/state"],
        "topic_types": {
            "/planning/plan": "std_msgs/String",
            "/camera/rgb": "sensor_msgs/Image",
            "/camera/depth": "sensor_msgs/Image",
            "/robot/state": "std_msgs/String",
            "/control/command": "std_msgs/String",
        },
        "qos": {
            "/planning/plan": "reliable_cmd",
            "/camera/rgb": "sensor_data",
            "/camera/depth": "sensor_data",
            "/robot/state": "reliable_cmd",
            "/control/command": "reliable_cmd",
        },
    },
    "robot": {
        "link_mode": "socket",
        "bind_host": "0.0.0.0",
        "bind_port": 51004,
        "auth_token": "auth_robot",
        "pub_topics": ["/robot/result", "/robot/state"],
        "sub_topics": ["/control/command"],
        "topic_types": {
            "/control/command": "std_msgs/String",
            "/robot/result": "std_msgs/String",
            "/robot/state": "std_msgs/String",
        },
        "qos": {
            "/control/command": "reliable_cmd",
            "/robot/result": "reliable_cmd",
            "/robot/state": "reliable_cmd",
        },
    },
    "monitor": {
        "link_mode": "socket",
        "bind_host": "0.0.0.0",
        "bind_port": 51005,
        "auth_token": "auth_monitor",
        "pub_topics": ["/feedback/signal"],
        "sub_topics": ["/camera/rgb", "/planning/plan", "/robot/result"],
        "topic_types": {
            "/camera/rgb": "sensor_msgs/Image",
            "/planning/plan": "std_msgs/String",
            "/robot/result": "std_msgs/String",
            "/feedback/signal": "std_msgs/String",
        },
        "qos": {
            "/camera/rgb": "sensor_data",
            "/planning/plan": "reliable_cmd",
            "/robot/result": "reliable_cmd",
            "/feedback/signal": "reliable_cmd",
        },
    },
}
