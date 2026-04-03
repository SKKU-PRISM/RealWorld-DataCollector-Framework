"""
Configuration constants for LeRobot Dataset Recording.

SO-101 로봇의 데이터셋 스키마 및 기본 설정값을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# =============================================================================
# Robot Configuration (공식 LeRobot 형식과 동일)
# =============================================================================

# 공식 LeRobot SO101 로봇 타입 (so101_follower.py 참조)
ROBOT_TYPE = "so101_follower"

# Joint names for SO-101 (5 arm joints + 1 gripper)
# 공식 형식: "{motor}.pos" (예: "shoulder_pan.pos")
# so101_follower.py의 observation_features/action_features와 동일
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# 공식 LeRobot 형식의 joint names (motor.pos 형태)
JOINT_NAMES = [f"{motor}.pos" for motor in MOTOR_NAMES]

NUM_JOINTS = 6  # 5 arm + 1 gripper

# =============================================================================
# Recording Configuration
# =============================================================================

# Default recording FPS (LeRobot standard)
DEFAULT_FPS = 30

# =============================================================================
# Camera Configuration
# =============================================================================

@dataclass
class CameraConfigRecord:
    """카메라 레코딩 설정

    LeRobot 공식 bi_so_follower 패턴과 호환:
      - shared: 공용 카메라 → observation.images.{name}
      - left_arm: 왼팔 카메라 → observation.images.left_{name}
      - right_arm: 오른팔 카메라 → observation.images.right_{name}
    """
    name: str              # 카메라 식별자 (예: "top", "wrist")
    type: str              # "realsense" 또는 "opencv"
    group: str = "shared"  # "shared" | "left_arm" | "right_arm"
    enabled: bool = True   # 레코딩 활성화 여부
    width: int = 640
    height: int = 480
    fps: int = 30

    # 공통 설정 (LeRobot 호환)
    index_or_path: Optional[str] = None  # 장치 경로 (예: "/dev/video7") 또는 인덱스

    # OpenCV 전용 설정 (레거시 호환)
    device_path: Optional[str] = None  # 예: "/dev/video7" (index_or_path 대체 가능)
    device_index: Optional[int] = None
    fourcc: Optional[str] = "MJPG"

    # RealSense 전용 설정
    serial_number: Optional[str] = None
    enable_depth: bool = False

    @property
    def feature_name(self) -> str:
        """최종 feature name (group prefix 포함).

        shared  → "top"
        left_arm  → "left_wrist"
        right_arm → "right_wrist"
        """
        if self.group == "left_arm":
            return f"left_{self.name}"
        elif self.group == "right_arm":
            return f"right_{self.name}"
        return self.name

    def get_device_path(self) -> Optional[str]:
        """장치 경로 반환 (index_or_path 또는 device_path)"""
        return self.index_or_path or self.device_path

    def to_feature_key(self) -> str:
        """LeRobot 데이터셋 feature 키"""
        return f"observation.images.{self.feature_name}"

    def to_feature_schema(self) -> Dict[str, Any]:
        """LeRobot 데이터셋 feature 스키마"""
        return {
            "dtype": "video",
            "shape": (self.height, self.width, 3),
            "names": ["height", "width", "channels"],
        }


# 기본 카메라 설정
DEFAULT_CAMERAS = [
    CameraConfigRecord(
        name="realsense",
        type="realsense",
        enabled=True,
        width=640,
        height=480,
        fps=30,
    ),
    CameraConfigRecord(
        name="innomaker",
        type="opencv",
        enabled=True,
        device_path="/dev/video7",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
    ),
]


def get_enabled_cameras() -> List[CameraConfigRecord]:
    """활성화된 카메라만 반환"""
    return [cam for cam in DEFAULT_CAMERAS if cam.enabled]


# =============================================================================
# Skill Features Configuration
# =============================================================================

# Observation feature keys (FK 기반 EE 자세 등)
OBSERVATION_FEATURE_KEYS = [
    "observation.ee_pos.robot_xyzrpy",
    "observation.gripper_binary",
    "observation.radian.state",
    "observation.radian.action",
    "observation.radian.state_urdf0",
    "observation.radian.action_urdf0",
]


# Skill feature keys
SKILL_FEATURE_KEYS = [
    "skill.natural_language",
    "skill.verification_question",
    "skill.type",
    "skill.progress",
    "skill.goal_position.joint",
    "skill.goal_position.robot_xyzrpy",
    "skill.goal_position.gripper",
]


def load_skill_features_from_yaml(yaml_path: str = None) -> Dict[str, bool]:
    """
    YAML에서 skill feature enabled 설정 로드.

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)

    Returns:
        Dict[str, bool]: 각 skill feature의 enabled 여부
                         없으면 전부 True (기본값)
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    # 기본값: 전부 True
    defaults = {key: True for key in SKILL_FEATURE_KEYS}

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        sf = data.get("skill_features")
        if not sf:
            return defaults

        gp = sf.get("goal_position", {})
        # goal_position이 bool이면 하위 전체에 적용
        if isinstance(gp, bool):
            gp = {"joint": gp, "robot_xyzrpy": gp, "gripper": gp}

        return {
            "skill.natural_language": sf.get("natural_language", True),
            "skill.verification_question": sf.get("verification_question", True),
            "skill.type": sf.get("type", True),
            "skill.progress": sf.get("progress", True),
            "skill.goal_position.joint": gp.get("joint", True),
            "skill.goal_position.robot_xyzrpy": gp.get("robot_xyzrpy", True),
            "skill.goal_position.gripper": gp.get("gripper", True),
        }
    except Exception as e:
        print(f"[Config] Warning: Failed to load skill_features from {yaml_path}: {e}")
        return defaults


def load_observation_features_from_yaml(yaml_path: str = None) -> Dict[str, bool]:
    """
    YAML에서 observation feature enabled 설정 로드.

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)

    Returns:
        Dict[str, bool]: 각 observation feature의 enabled 여부
                         없으면 전부 False (기본값 — 명시적 활성화 필요)
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    # 기본값: 전부 False (기존 동작과 호환 — 명시적으로 켜야 함)
    defaults = {key: False for key in OBSERVATION_FEATURE_KEYS}

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        of = data.get("observation_features")
        if not of:
            return defaults

        ep = of.get("ee_pos", {})
        # ee_pos가 bool이면 하위 전체에 적용
        if isinstance(ep, bool):
            ep = {"robot_xyzrpy": ep}

        rd = of.get("radian", {})
        if isinstance(rd, bool):
            rd = {"state": rd, "action": rd, "state_urdf0": rd, "action_urdf0": rd}

        return {
            "observation.ee_pos.robot_xyzrpy": ep.get("robot_xyzrpy", False),
            "observation.gripper_binary": of.get("gripper_binary", False),
            "observation.radian.state": rd.get("state", False),
            "observation.radian.action": rd.get("action", False),
            "observation.radian.state_urdf0": rd.get("state_urdf0", False),
            "observation.radian.action_urdf0": rd.get("action_urdf0", False),
        }
    except Exception as e:
        print(f"[Config] Warning: Failed to load observation_features from {yaml_path}: {e}")
        return defaults



def load_subtask_features_from_yaml(yaml_path: str = None) -> Dict[str, bool]:
    """
    YAML에서 subtask feature enabled 설정 로드.

    Returns:
        Dict[str, bool]: 각 subtask feature의 enabled 여부
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    defaults = {
        "subtask.natural_language": False,
        "subtask.object_name": False,
        "subtask.target_position": False,
    }

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        sf = data.get("subtask_features")
        if not sf:
            return defaults

        return {
            "subtask.natural_language": sf.get("natural_language", False),
            "subtask.object_name": sf.get("object_name", False),
            "subtask.target_position": sf.get("target_position", False),
        }
    except Exception as e:
        print(f"[Config] Warning: Failed to load subtask_features from {yaml_path}: {e}")
        return defaults


def get_camera_feature_keys() -> List[str]:
    """활성화된 카메라 feature 키 목록"""
    return [cam.to_feature_key() for cam in get_enabled_cameras()]


# =============================================================================
# Dataset Features Schema (LeRobot v3.0 format)
# =============================================================================

def build_dataset_features(
    cameras: List[CameraConfigRecord] = None,
    skill_enabled: Dict[str, bool] = None,
    obs_enabled: Dict[str, bool] = None,
    subtask_enabled: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """
    데이터셋 features 스키마 빌드

    Args:
        cameras: 카메라 설정 리스트 (None이면 DEFAULT_CAMERAS 사용)
        skill_enabled: skill feature별 enabled 여부 (None이면 전부 True)
        obs_enabled: observation feature별 enabled 여부 (None이면 전부 False)

    Returns:
        LeRobot 데이터셋 features 딕셔너리
    """
    if cameras is None:
        cameras = get_enabled_cameras()

    if skill_enabled is None:
        skill_enabled = {key: True for key in SKILL_FEATURE_KEYS}

    if obs_enabled is None:
        obs_enabled = {key: False for key in OBSERVATION_FEATURE_KEYS}


    features = {
        # Robot state: current joint positions (normalized -100 to +100)
        "observation.state": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },

        # Action: target joint positions (normalized -100 to +100)
        "action": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
    }

    # 카메라별 이미지 feature 추가
    for cam in cameras:
        features[cam.to_feature_key()] = cam.to_feature_schema()

    # Observation features (FK 기반 EE 자세 등, enabled인 것만 추가)
    obs_schemas = {
        "observation.ee_pos.robot_xyzrpy": {
            "dtype": "float32", "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.gripper_binary": {
            "dtype": "float32", "shape": (1,),
            "names": None,
        },
        "observation.radian.state": {
            "dtype": "float32", "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
        "observation.radian.action": {
            "dtype": "float32", "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
        "observation.radian.state_urdf0": {
            "dtype": "float32", "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
        "observation.radian.action_urdf0": {
            "dtype": "float32", "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
    }

    for key, schema in obs_schemas.items():
        if obs_enabled.get(key, False):
            features[key] = schema


    # Skill-level subgoal labels (enabled인 것만 추가)
    skill_schemas = {
        "skill.natural_language": {"dtype": "string", "shape": (1,), "names": None},
        "skill.verification_question": {"dtype": "string", "shape": (1,), "names": None},
        "skill.type": {"dtype": "string", "shape": (1,), "names": None},
        "skill.progress": {"dtype": "float32", "shape": (1,), "names": None},
        "skill.goal_position.joint": {"dtype": "float32", "shape": (NUM_JOINTS,), "names": JOINT_NAMES},
        "skill.goal_position.robot_xyzrpy": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
        "skill.goal_position.gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper.pos"]},
    }

    for key, schema in skill_schemas.items():
        if skill_enabled.get(key, True):
            features[key] = schema

    # Sub-task labels (enabled인 것만 추가)
    if subtask_enabled is None:
        subtask_enabled = {
            "subtask.natural_language": False,
            "subtask.object_name": False,
            "subtask.target_position": False,
        }

    subtask_schemas = {
        "subtask.natural_language": {"dtype": "string", "shape": (1,), "names": None},
        "subtask.object_name": {"dtype": "string", "shape": (1,), "names": None},
        "subtask.target_position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }

    for key, schema in subtask_schemas.items():
        if subtask_enabled.get(key, False):
            features[key] = schema

    return features


# =============================================================================
# Multi-Arm Features (ALOHA-style concat 12-axis)
# =============================================================================

# Multi-arm joint names: left_ and right_ prefix
MULTI_ARM_MOTOR_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]
MULTI_ARM_JOINT_NAMES = [f"{motor}.pos" for motor in MULTI_ARM_MOTOR_NAMES]
MULTI_ARM_NUM_JOINTS = 12  # 6 left + 6 right

# Multi-arm default cameras (shared RealSense + per-arm Innomaker)
DEFAULT_MULTI_ARM_CAMERAS = [
    CameraConfigRecord(
        name="realsense", type="realsense", enabled=True,
        width=640, height=480, fps=30,
    ),
    CameraConfigRecord(
        name="left_innomaker", type="opencv", enabled=True,
        device_path="/dev/video7", width=640, height=480, fps=30, fourcc="MJPG",
    ),
    CameraConfigRecord(
        name="right_innomaker", type="opencv", enabled=True,
        device_path="/dev/video8", width=640, height=480, fps=30, fourcc="MJPG",
    ),
]


def build_multi_arm_features(
    cameras: List[CameraConfigRecord] = None,
    skill_enabled: Dict[str, bool] = None,
    obs_enabled: Dict[str, bool] = None,
    subtask_enabled: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """
    Build dataset features for multi-arm (ALOHA-style) recording.

    State/action are 12-axis (concat left 6 + right 6).
    Skill and observation features are split into left_*/right_*.
    """
    if cameras is None:
        cameras = [cam for cam in DEFAULT_MULTI_ARM_CAMERAS if cam.enabled]

    features = {
        # Concat state: [left_6, right_6] = 12-axis (normalized -100 to +100)
        "observation.state": {
            "dtype": "float32",
            "shape": (MULTI_ARM_NUM_JOINTS,),
            "names": MULTI_ARM_JOINT_NAMES,
        },
        # Concat action: [left_6, right_6] = 12-axis
        "action": {
            "dtype": "float32",
            "shape": (MULTI_ARM_NUM_JOINTS,),
            "names": MULTI_ARM_JOINT_NAMES,
        },
    }

    # Camera images
    for cam in cameras:
        features[cam.to_feature_key()] = cam.to_feature_schema()

    # Per-arm observation features
    per_arm_obs_schemas = {}
    for prefix in ("left", "right"):
        arm_joint_names = [f"{prefix}_{m}.pos" for m in MOTOR_NAMES]
        per_arm_obs_schemas.update({
            f"observation.ee_pos.{prefix}_robot_xyzrpy": {
                "dtype": "float32", "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
            f"{prefix}_observation.gripper_binary": {
                "dtype": "float32", "shape": (1,),
                "names": None,
            },
            f"observation.radian.{prefix}_state": {
                "dtype": "float32", "shape": (NUM_JOINTS,),
                "names": arm_joint_names,
            },
            f"observation.radian.{prefix}_action": {
                "dtype": "float32", "shape": (NUM_JOINTS,),
                "names": arm_joint_names,
            },
        })

    if obs_enabled is None:
        obs_enabled = {}
    for key, schema in per_arm_obs_schemas.items():
        if obs_enabled.get(key, True):
            features[key] = schema

    # Per-arm skill features
    for prefix in ("left", "right"):
        skill_schemas = {
            f"{prefix}_skill.natural_language": {"dtype": "string", "shape": (1,), "names": None},
            f"{prefix}_skill.verification_question": {"dtype": "string", "shape": (1,), "names": None},
            f"{prefix}_skill.type": {"dtype": "string", "shape": (1,), "names": None},
            f"{prefix}_skill.progress": {"dtype": "float32", "shape": (1,), "names": None},
            f"{prefix}_skill.goal_position.joint": {
                "dtype": "float32", "shape": (NUM_JOINTS,),
                "names": [f"{prefix}_{m}.pos" for m in MOTOR_NAMES],
            },
            f"{prefix}_skill.goal_position.robot_xyzrpy": {
                "dtype": "float32", "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
            f"{prefix}_skill.goal_position.gripper": {
                "dtype": "float32", "shape": (1,),
                "names": ["gripper.pos"],
            },
        }
        if skill_enabled is None:
            se = {k: True for k in skill_schemas}
        else:
            se = skill_enabled
        for key, schema in skill_schemas.items():
            if se.get(key, True):
                features[key] = schema

    # Subtask features (shared — task-level, not per-arm)
    if subtask_enabled is None:
        subtask_enabled = {}
    subtask_schemas = {
        "subtask.natural_language": {"dtype": "string", "shape": (1,), "names": None},
        "subtask.object_name": {"dtype": "string", "shape": (1,), "names": None},
        "subtask.target_position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }
    for key, schema in subtask_schemas.items():
        if subtask_enabled.get(key, False):
            features[key] = schema

    return features


# 기본 features (공식 LeRobot 형식)
# names 필드는 공식 형식: ["shoulder_pan.pos", "shoulder_lift.pos", ...]
DATASET_FEATURES = {
    # Robot state: current joint positions (normalized -100 to +100)
    # 공식 형식: names에 "{motor}.pos" 형태 사용
    "observation.state": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,  # ["shoulder_pan.pos", "shoulder_lift.pos", ...]
    },

    # Action: target joint positions (normalized -100 to +100)
    # 공식 형식: names에 "{motor}.pos" 형태 사용
    "action": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,  # ["shoulder_pan.pos", "shoulder_lift.pos", ...]
    },

    # Camera images (RGB) - 멀티 카메라
    "observation.images.realsense": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.innomaker": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },

    # Skill-level subgoal labels
    "skill.natural_language": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "skill.verification_question": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "skill.type": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "skill.progress": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "skill.goal_position.joint": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "skill.goal_position.robot_xyzrpy": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "skill.goal_position.gripper": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper.pos"],
    },
}

# 레거시 호환 (기존 코드에서 사용)
DATASET_FEATURES_LEGACY = {
    "observation.state": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}

# =============================================================================
# Dynamic Camera Loading from YAML
# =============================================================================

def _parse_camera_entry(cam_data: dict, group: str = "shared") -> 'CameraConfigRecord':
    """YAML 카메라 항목 하나를 CameraConfigRecord로 변환."""
    return CameraConfigRecord(
        name=cam_data.get("name", "camera"),
        type=cam_data.get("type", "opencv"),
        group=group,
        enabled=cam_data.get("enabled", True),
        width=cam_data.get("width", 640),
        height=cam_data.get("height", 480),
        fps=cam_data.get("fps", 30),
        index_or_path=cam_data.get("index_or_path"),
        device_path=cam_data.get("device_path"),
        device_index=cam_data.get("device_index"),
        fourcc=cam_data.get("fourcc", "MJPG"),
        serial_number=cam_data.get("serial_number"),
        enable_depth=cam_data.get("enable_depth", False),
    )


def load_cameras_from_yaml(yaml_path: str = None, num_robots: int = None) -> List[CameraConfigRecord]:
    """
    YAML 파일에서 카메라 설정을 동적으로 로드.

    두 가지 YAML 구조를 지원:
      1) 기존 flat 리스트 (싱글암 호환):
           cameras:
             - name: "realsense" ...
      2) 새 그룹 구조 (멀티암):
           cameras:
             shared:
               - name: "top" ...
             left_arm:
               - name: "wrist" ...
             right_arm:
               - name: "wrist" ...

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)
        num_robots: 로봇 수. 순서대로 left/right/top/bottom arm 그룹 활성화.
                    None이면 모든 arm 그룹 로드.

    Returns:
        List[CameraConfigRecord]: 카메라 설정 리스트
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"[Config] Warning: {yaml_path} not found, using DEFAULT_CAMERAS")
        return DEFAULT_CAMERAS.copy()

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        cameras_data = data.get("cameras", [])
        if not cameras_data:
            print(f"[Config] Warning: No cameras defined in {yaml_path}, using DEFAULT_CAMERAS")
            return DEFAULT_CAMERAS.copy()

        cameras = []

        if isinstance(cameras_data, list):
            # 기존 flat 리스트 (싱글암 호환)
            for cam_data in cameras_data:
                cameras.append(_parse_camera_entry(cam_data, group="shared"))

        elif isinstance(cameras_data, dict):
            # ROBOT_IDS 순서 → arm 그룹 매핑:
            #   [0]=left_arm, [1]=right_arm, [2]=top_arm, [3]=bottom_arm
            # shared는 항상 포함, 제공된 num_robots 수만큼만 arm 그룹 활성화
            _ARM_GROUPS = ["left_arm", "right_arm", "top_arm", "bottom_arm"]
            active_arms = _ARM_GROUPS[:num_robots] if num_robots is not None else _ARM_GROUPS
            allowed_groups = ["shared"] + active_arms

            for group_key in allowed_groups:
                group_cams = cameras_data.get(group_key, [])
                if group_cams is None:
                    continue
                for cam_data in group_cams:
                    cameras.append(_parse_camera_entry(cam_data, group=group_key))

        print(f"[Config] Loaded {len(cameras)} camera(s) from {yaml_path}")
        for cam in cameras:
            status = "enabled" if cam.enabled else "disabled"
            group_str = f" [{cam.group}]" if cam.group != "shared" else ""
            print(f"  - {cam.feature_name} ({cam.type}): {cam.width}x{cam.height}@{cam.fps}fps [{status}]{group_str}")

        return cameras

    except Exception as e:
        print(f"[Config] Error loading {yaml_path}: {e}, using DEFAULT_CAMERAS")
        return DEFAULT_CAMERAS.copy()


def create_camera_manager_from_config(yaml_path: str = None, num_robots: int = None):
    """
    YAML 설정에서 MultiCameraManager 생성

    Args:
        yaml_path: recording_config.yaml 경로
        num_robots: 로봇 수. 순서대로 left/right/top/bottom arm 그룹 활성화.

    Returns:
        MultiCameraManager 인스턴스
    """
    import sys
    from pathlib import Path

    # cameras 모듈 경로 추가
    cameras_path = Path(__file__).parent.parent / "cameras"
    if str(cameras_path.parent) not in sys.path:
        sys.path.insert(0, str(cameras_path.parent))

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

    # YAML에서 카메라 설정 로드 (num_robots로 arm 그룹 필터링)
    camera_configs = load_cameras_from_yaml(yaml_path, num_robots=num_robots)

    # cameras 모듈용 config 객체로 변환
    configs = []
    for cam in camera_configs:
        if not cam.enabled:
            continue

        # feature_name을 카메라 등록 이름으로 사용 (shared: "top", left_arm: "left_wrist", ...)
        cam_name = cam.feature_name

        if cam.type == "realsense":
            configs.append(RealSenseCameraConfig(
                name=cam_name,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                serial_number=cam.serial_number,
                enable_depth=cam.enable_depth,
            ))
        else:  # opencv
            device = cam.get_device_path() or "/dev/video0"
            configs.append(OpenCVCameraConfig(
                name=cam_name,
                device_path=device,
                device_index=cam.device_index,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                fourcc=cam.fourcc,
            ))

    return MultiCameraManager(configs)


def build_features_from_yaml(yaml_path: str = None, num_robots: int = None) -> Dict[str, Any]:
    """
    YAML 설정 기반으로 LeRobot dataset features 생성

    Args:
        yaml_path: recording_config.yaml 경로
        num_robots: 로봇 수. 순서대로 left/right/top/bottom arm 그룹 활성화.

    Returns:
        LeRobot dataset features dict
    """
    cameras = load_cameras_from_yaml(yaml_path, num_robots=num_robots)
    enabled_cameras = [cam for cam in cameras if cam.enabled]
    skill_enabled = load_skill_features_from_yaml(yaml_path)
    obs_enabled = load_observation_features_from_yaml(yaml_path)
    subtask_enabled = load_subtask_features_from_yaml(yaml_path)
    return build_dataset_features(enabled_cameras, skill_enabled, obs_enabled, subtask_enabled)


# =============================================================================
# Image Configuration
# =============================================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANNELS = 3

# =============================================================================
# Storage Configuration
# =============================================================================

# Default local cache path (follows HuggingFace convention)
# Actual path: ~/.cache/huggingface/lerobot/{repo_id}
DEFAULT_ROOT = None  # Will use HF_LEROBOT_HOME

# Chunk settings for large datasets
DEFAULT_CHUNKS_SIZE = 1000  # Files per chunk directory
DEFAULT_DATA_FILES_SIZE_MB = 500  # Max parquet file size
DEFAULT_VIDEO_FILES_SIZE_MB = 500  # Max video file size

# =============================================================================
# Normalization Range
# =============================================================================

# Current system uses -100 to +100 normalized range
NORM_MIN = -100.0
NORM_MAX = 100.0
