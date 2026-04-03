"""
RoboCasa Perception Adapter.

Uses ground truth object positions from RoboCasa environment observation.
Converts world frame coordinates to robot-base frame for controller compatibility.
Also extracts fixture positions (cabinets, counters, etc.) for place actions.
Provides bounding boxes for collision avoidance.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from robobridge.modules.perception.types import Detection

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min: List[float]  # [x_min, y_min, z_min]
    max: List[float]  # [x_max, y_max, z_max]
    name: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
        }


class RoboCasaPerception:
    """
    Perception adapter for RoboCasa that extracts object information
    from environment observation and converts to robot-base frame.

    RoboCasa observation provides (in world frame):
    - obj_pos: target object position [x, y, z]
    - obj_quat: target object quaternion [x, y, z, w]
    - distr_*_pos/quat: distractor object positions
    - robot0_base_pos: robot base position in world frame

    Episode metadata (ep_meta) contains:
    - object_cfgs: list of object configurations with names, categories
    - fixture_refs: dict mapping fixture roles to fixture names
    - lang: natural language instruction
    """

    def __init__(self):
        """Initialize RoboCasa perception adapter."""
        self._obs: Optional[Dict[str, Any]] = None
        self._ep_meta: Optional[Dict[str, Any]] = None
        self._robot_base_pos: Optional[np.ndarray] = None
        self._robot_base_rot: Optional[R] = None
        self._env: Optional[Any] = None  # RoboCasa environment reference
        self._fixture_nat_lang: Dict[str, str] = {}  # naming_prefix → nat_lang
        self._target_prefix: Optional[str] = None    # 타겟 fixture의 naming_prefix

    def set_environment_state(
        self,
        obs: Dict[str, Any],
        ep_meta: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
    ) -> None:
        """
        Update the current observation and metadata.

        Args:
            obs: RoboCasa observation dictionary
            ep_meta: Episode metadata from env.get_ep_meta()
            env: RoboCasa environment reference (for fixture access)
        """
        self._obs = obs
        if ep_meta is not None:
            self._ep_meta = ep_meta
        if env is not None:
            self._env = env
            self._fixture_nat_lang = {}  # naming_prefix → nat_lang
            self._target_prefix = None   # 타겟 fixture의 naming_prefix
            self._target_category = None  # target interaction category filter (e.g. "spout", "handle")

            # env.fixtures에서 naming_prefix → nat_lang 매핑 생성
            if hasattr(env, 'fixtures'):
                for name, fxtr in env.fixtures.items():
                    prefix = getattr(fxtr, 'naming_prefix', '')
                    nat = getattr(fxtr, 'nat_lang', None)
                    if prefix and nat:
                        self._fixture_nat_lang[prefix] = nat

            # 타겟 fixture 식별 — task별 attr이 다름:
            #   CloseDrawer/OpenDrawer → env.drawer
            #   CloseSingleDoor/OpenSingleDoor → env.fxtr
            #   PnPCabToCounter → env.cab
            #   TurnOnStove → env.stove
            #   init_robot_base_ref는 모든 태스크에 존재 (최종 fallback)
            for attr in ('fxtr', 'drawer', 'sink', 'stove', 'microwave',
                         'coffee_machine', 'cab', 'init_robot_base_ref'):
                target = getattr(env, attr, None)
                if target is not None and hasattr(target, 'naming_prefix'):
                    self._target_prefix = target.naming_prefix
                    # Sink has multiple interaction sites (handle, spout) — disambiguate by task class
                    if attr == 'sink':
                        task_class = type(env).__name__
                        if 'Spout' in task_class:
                            self._target_category = 'spout'
                        elif 'Faucet' in task_class:
                            self._target_category = 'handle'
                        # else: None → all sink sites are target
                    break

            if self._fixture_nat_lang:
                logger.info(f"[PERCEPTION] Fixture nat_lang map: {self._fixture_nat_lang}")
            if self._target_prefix:
                cat_str = f" (category={self._target_category})" if self._target_category else ""
                logger.info(f"[PERCEPTION] Target fixture prefix: {self._target_prefix}{cat_str}")

        # Extract robot base pose for coordinate transformation
        if obs is not None and "robot0_base_pos" in obs:
            self._robot_base_pos = obs["robot0_base_pos"].copy()
            # Robot base quaternion (if available) - default to 180 deg rotation around Z
            if "robot0_base_quat" in obs:
                quat = obs["robot0_base_quat"]  # xyzw format
                self._robot_base_rot = R.from_quat(quat)
            else:
                # Default: 180 degree rotation around Z axis (common in RoboCasa kitchens)
                self._robot_base_rot = R.from_euler('z', 180, degrees=True)
        elif obs is not None:
            pass  # robot0_base_pos not in obs — fallback to world frame

    def _world_to_robot_base(self, world_pos: np.ndarray) -> np.ndarray:
        """
        Convert position from world frame to robot-base frame.

        Args:
            world_pos: Position in world frame [x, y, z]

        Returns:
            Position in robot-base frame [x, y, z]
        """
        if self._robot_base_pos is None or self._robot_base_rot is None:
            return world_pos  # Fallback: return as-is

        # Translate to robot base origin, then rotate
        relative_pos = world_pos - self._robot_base_pos
        return self._robot_base_rot.inv().apply(relative_pos)

    def _get_object_bbox(self, obj_name: str) -> Optional[BoundingBox]:
        """
        Get bounding box for an object in robot-base frame.

        Args:
            obj_name: Object key in env.objects (e.g., 'obj', 'distr_counter')

        Returns:
            BoundingBox in robot-base frame, or None if not found.
        """
        if self._env is None or self._obs is None:
            return None

        # Get object from environment
        obj = self._env.objects.get(obj_name)
        if obj is None:
            return None

        # Get position key
        pos_key = "obj_pos" if obj_name == "obj" else f"{obj_name}_pos"
        if pos_key not in self._obs:
            return None

        world_pos = self._obs[pos_key]
        robot_pos = self._world_to_robot_base(world_pos)

        # Get size from object attributes
        radius = getattr(obj, 'horizontal_radius', 0.05)
        top_offset = getattr(obj, 'top_offset', np.array([0, 0, 0.05]))
        bottom_offset = getattr(obj, 'bottom_offset', np.array([0, 0, -0.05]))

        # Calculate bbox in robot frame
        bbox_min = [
            robot_pos[0] - radius,
            robot_pos[1] - radius,
            robot_pos[2] + bottom_offset[2],
        ]
        bbox_max = [
            robot_pos[0] + radius,
            robot_pos[1] + radius,
            robot_pos[2] + top_offset[2],
        ]

        # Get category name
        obj_cfgs = self._ep_meta.get("object_cfgs", []) if self._ep_meta else []
        category = obj_name
        for cfg in obj_cfgs:
            if cfg.get("name") == obj_name:
                category = cfg.get("info", {}).get("cat", obj_name)
                break

        return BoundingBox(name=category, min=bbox_min, max=bbox_max)

    def _get_fixture_geom_bboxes(self, fixture_name: str) -> List[BoundingBox]:
        """
        Get bounding boxes for all geoms of a fixture (cabinet parts, shelves, etc.).

        Args:
            fixture_name: Name of the fixture (e.g., 'cab_1_main_group')

        Returns:
            List of BoundingBox for each geom in robot-base frame.
        """
        if self._env is None:
            return []

        sim = self._env.sim
        bboxes = []

        # Find all geoms related to this fixture
        for i, geom_name in enumerate(sim.model.geom_names):
            if geom_name and fixture_name in geom_name:
                # Skip visual geoms (only use collision geoms)
                if '_visual' in geom_name:
                    continue

                geom_type = sim.model.geom_type[i]
                # Only handle box type (type=6) for now
                if geom_type != 6:
                    continue

                pos = sim.data.geom_xpos[i]
                size = sim.model.geom_size[i]  # half-extents

                # Convert to robot-base frame
                robot_pos = self._world_to_robot_base(pos)

                # Calculate bbox
                bbox_min = [
                    robot_pos[0] - size[0],
                    robot_pos[1] - size[1],
                    robot_pos[2] - size[2],
                ]
                bbox_max = [
                    robot_pos[0] + size[0],
                    robot_pos[1] + size[1],
                    robot_pos[2] + size[2],
                ]

                # Extract part name (e.g., 'shelf' from 'cab_1_main_group_shelf')
                part_name = geom_name.replace(fixture_name + "_", "")
                bboxes.append(BoundingBox(name=part_name, min=bbox_min, max=bbox_max))

        return bboxes

    def get_all_bboxes(self) -> List[BoundingBox]:
        """
        Get all bounding boxes in the scene (objects + fixtures).

        Returns:
            List of BoundingBox in robot-base frame.
        """
        bboxes = []

        # Get object bounding boxes
        for obj_name in ["obj", "distr_counter", "distr_cab"]:
            bbox = self._get_object_bbox(obj_name)
            if bbox:
                bboxes.append(bbox)

        # Get fixture bounding boxes
        if self._ep_meta and self._env:
            fixture_refs = self._ep_meta.get("fixture_refs", {})
            for role, fixture_name in fixture_refs.items():
                fixture_bboxes = self._get_fixture_geom_bboxes(fixture_name)
                bboxes.extend(fixture_bboxes)

        return bboxes

    def _get_fixture_interior_position(self, fixture_name: str) -> Optional[np.ndarray]:
        """
        Get the interior center position of a fixture (cabinet, drawer, etc.).

        Uses interior site positions (int_p0, int_px, int_py) to calculate center.

        Args:
            fixture_name: Name of the fixture (e.g., 'cab_1_main_group')

        Returns:
            Interior center position in world frame, or None if not found.
        """
        if self._env is None:
            return None

        sim = self._env.sim

        # Try to find interior sites for this fixture
        # RoboCasa uses int_p0 (origin), int_px (x-extent), int_py (y-extent)
        site_prefix = f"{fixture_name}_int_"
        try:
            p0_id = sim.model.site_name2id(f"{site_prefix}p0")
            px_id = sim.model.site_name2id(f"{site_prefix}px")
            py_id = sim.model.site_name2id(f"{site_prefix}py")

            p0 = sim.data.site_xpos[p0_id]
            px = sim.data.site_xpos[px_id]
            py = sim.data.site_xpos[py_id]

            # Calculate center of the interior area
            # p0 is one corner, px and py define the extents
            center_x = (p0[0] + px[0]) / 2
            center_y = (p0[1] + py[1]) / 2
            center_z = p0[2]  # Use bottom z for placement surface

            return np.array([center_x, center_y, center_z])
        except (KeyError, ValueError):
            return None

    def _get_fixture_interior_bbox(self, fixture_name: str) -> Optional[Dict]:
        """
        Get interior bounding box of a fixture from interior sites.

        Returns:
            Dict with 'min', 'max', 'center' in world frame, or None.
        """
        if self._env is None:
            return None

        sim = self._env.sim
        site_prefix = f"{fixture_name}_int_"

        try:
            p0_id = sim.model.site_name2id(f"{site_prefix}p0")
            px_id = sim.model.site_name2id(f"{site_prefix}px")
            py_id = sim.model.site_name2id(f"{site_prefix}py")

            p0 = sim.data.site_xpos[p0_id]
            px = sim.data.site_xpos[px_id]
            py = sim.data.site_xpos[py_id]

            # Calculate interior bbox
            min_x = min(p0[0], px[0])
            max_x = max(p0[0], px[0])
            min_y = min(p0[1], py[1])
            max_y = max(p0[1], py[1])
            z = p0[2]  # Surface height

            # Estimate interior height (typical cabinet ~0.3m)
            height = 0.25

            return {
                "min": np.array([min_x, min_y, z]),
                "max": np.array([max_x, max_y, z + height]),
                "center": np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, z]),
            }
        except (KeyError, ValueError):
            return None

    # All known RoboCasa fixture type prefixes (appears before _main_group or _front_group)
    _KNOWN_FIXTURES = [
        # Longest first to avoid partial matches (e.g. "toaster_oven" before "toaster")
        "coffee_machine", "electric_kettle", "toaster_oven", "stand_mixer",
        "stove_hood", "oven_cabinet",
        "hingecabinet", "singlecabinet",
        "dishwasher", "microwave", "refrigerator",
        "blender", "toaster", "cabinet", "fridge", "stove", "oven",
        "kettle", "drawer", "faucet", "stack", "sink",
        "bottom", "top",
    ]

    @staticmethod
    def _extract_fixture_name(body_name_lower: str) -> str:
        """Extract fixture name from MuJoCo body name.

        RoboCasa body names follow: {fixture_type}_{group}_{id}_{component}
        Examples:
            "sink_front_group_1_handle" → "sink"
            "stove_main_group_1_knob_front_left" → "stove"
            "hingecabinet_02_main_group_1_left_door_handle_main" → "hingecabinet"
            "coffee_machine_main_group_1_button001" → "coffee_machine"
            "toaster_oven_main_group_1_knob_temp" → "toaster_oven"
            "dishwasher_1_front_group_1_button_power" → "dishwasher"
        """
        import re

        # Strategy 1: Split on group separators (covers 99% of RoboCasa fixtures)
        for sep in ["_main_group", "_front_group"]:
            if sep in body_name_lower:
                prefix = body_name_lower.split(sep)[0]
                prefix = re.sub(r'[_\d]+$', '', prefix)
                return prefix if prefix else "unknown"

        # Strategy 2: Match known RoboCasa fixture types (longest match first)
        for fixture in RoboCasaPerception._KNOWN_FIXTURES:
            if body_name_lower.startswith(fixture) or f"_{fixture}_" in body_name_lower:
                return fixture

        # Strategy 3: Take everything before interaction keywords
        for keyword in ["handle", "knob", "spout", "button", "faucet", "lever", "switch", "door"]:
            idx = body_name_lower.find(keyword)
            if idx > 0:
                prefix = body_name_lower[:idx].rstrip("_")
                prefix = re.sub(r'[_\d]+$', '', prefix)
                if prefix:
                    return prefix

        logger.warning(f"[PERCEPTION] Could not extract fixture name from body: '{body_name_lower}'")
        return "unknown"

    def _detect_interaction_sites(self) -> List[Detection]:
        """Detect fixture interaction sites (handles, knobs, spouts) from MuJoCo sim.

        Searches sim.model.body_names for interaction-related keywords and
        extracts their world positions, converting to robot-base frame.
        """
        if self._env is None or self._obs is None:
            return []

        sim = self._env.sim
        detections = []

        # Keyword -> detection category mapping
        keyword_categories = {
            "handle": "handle",
            "knob": "knob",
            "spout": "spout",
            "faucet": "faucet",
            "button": "button",
            "disc": "button",      # RoboCasa microwave uses "Disc001" for buttons
            "lever": "handle",
            "switch": "button",
        }

        # Search body names for interaction keywords
        # Track seen names to deduplicate (keep closest to robot)
        seen_sites: Dict[str, tuple] = {}  # name -> (distance, detection)

        for body_id, body_name in enumerate(sim.model.body_names):
            if not body_name:
                continue

            body_name_lower = body_name.lower()
            matched_category = None
            for keyword, category in keyword_categories.items():
                if keyword in body_name_lower:
                    matched_category = category
                    break

            if matched_category is None:
                continue

            # Skip robot body parts
            if any(skip in body_name_lower for skip in ["robot", "panda", "gripper", "finger"]):
                continue

            # Get world position of this body
            try:
                body_pos = sim.data.body_xpos[body_id].copy()
            except (IndexError, KeyError):
                continue

            # Use centroid of child geoms instead of body origin.
            # Body origin is often the joint/pivot (e.g. spout base),
            # while geom centroid better represents the interactable surface.
            geom_positions = []
            for gi in range(sim.model.ngeom):
                if sim.model.geom_bodyid[gi] == body_id:
                    geom_positions.append(sim.data.geom_xpos[gi].copy())

            if geom_positions:
                world_pos = np.mean(geom_positions, axis=0)
                offset = float(np.linalg.norm(world_pos - body_pos))
                if offset > 0.01:
                    logger.debug(
                        f"[PERCEPTION] {body_name}: geom centroid offset "
                        f"{offset:.3f}m from body origin ({len(geom_positions)} geoms)"
                    )
            else:
                world_pos = body_pos

            # Convert to robot-base frame
            robot_base_pos = self._world_to_robot_base(world_pos)

            # Extract fixture name using nat_lang mapping (preferred) or fallback
            # nat_lang gives human-readable names: "hingecabinet" → "cabinet", "coffee_machine" → "coffee machine"
            fixture_name = None
            for prefix, nat_lang in self._fixture_nat_lang.items():
                if body_name_lower.startswith(prefix.lower()):
                    fixture_name = nat_lang.replace(" ", "_")  # "coffee machine" → "coffee_machine"
                    break
            if fixture_name is None:
                fixture_name = self._extract_fixture_name(body_name_lower)  # fallback

            detection_name = f"{fixture_name}_{matched_category}"  # e.g. "cabinet_handle"

            # Mark target fixture detections (with optional category filter)
            prefix_match = (self._target_prefix is not None and
                            body_name_lower.startswith(self._target_prefix.lower()))
            if prefix_match and self._target_category is not None:
                is_target = (matched_category == self._target_category)
            else:
                is_target = prefix_match

            dist = float(np.linalg.norm(robot_base_pos))

            detection = Detection(
                name=detection_name,
                confidence=1.0,
                bbox=[0.3, 0.3, 0.7, 0.7],  # 2D placeholder
                pose={
                    "position": {
                        "x": float(robot_base_pos[0]),
                        "y": float(robot_base_pos[1]),
                        "z": float(robot_base_pos[2]),
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                    "frame": "robot_base",
                },
                metadata={
                    "role": "target" if is_target else "interaction_site",
                    "body_name": body_name,
                    "category": matched_category,
                    "fixture": fixture_name,
                    "is_fixture": True,
                },
            )

            # Deduplicate: keep only the closest instance per detection_name
            # BUT target fixtures always win over non-target (even if farther)
            if detection_name not in seen_sites:
                seen_sites[detection_name] = (dist, detection)
            else:
                existing_is_target = seen_sites[detection_name][1].metadata.get("role") == "target"
                if is_target and not existing_is_target:
                    # Target always replaces non-target
                    seen_sites[detection_name] = (dist, detection)
                elif not is_target and existing_is_target:
                    pass  # Never replace target with non-target
                elif dist < seen_sites[detection_name][0]:
                    seen_sites[detection_name] = (dist, detection)

        detections = [entry[1] for entry in seen_sites.values()]

        if detections:
            site_list = []
            for d in detections:
                p = d.pose["position"]
                site_list.append(f"{d.name}({p['x']:.2f},{p['y']:.2f},{p['z']:.2f})")
            logger.info(f"Detected {len(detections)} interaction sites: {site_list}")

        return detections

    def _detect_fixtures(self) -> List[Detection]:
        """
        Detect fixtures (cabinets, counters, etc.) from environment.

        Uses interior sites for accurate position and bbox.

        Returns:
            List of Detection objects for fixtures.
        """
        if self._ep_meta is None:
            return []

        detections = []
        fixture_refs = self._ep_meta.get("fixture_refs", {})

        for role, fixture_name in fixture_refs.items():
            # Determine category name from role
            if role == "cab":
                category = "cabinet"
            elif role == "counter":
                category = "counter"
            else:
                category = role

            # Try to get interior bbox (provides both position and bounds)
            interior = self._get_fixture_interior_bbox(fixture_name)

            if interior is not None:
                world_pos = interior["center"]
                world_bbox_min = interior["min"]
                world_bbox_max = interior["max"]
            else:
                # Fallback to distr object positions
                if role == "cab" and self._obs is not None and "distr_cab_pos" in self._obs:
                    world_pos = self._obs["distr_cab_pos"].copy()
                elif role == "counter" and self._obs is not None and "distr_counter_pos" in self._obs:
                    world_pos = self._obs["distr_counter_pos"].copy()
                    world_pos[2] -= 0.05
                else:
                    continue
                # No bbox available for fallback
                world_bbox_min = None
                world_bbox_max = None

            # Convert to robot-base frame
            robot_base_pos = self._world_to_robot_base(world_pos)

            bbox_3d = None
            if world_bbox_min is not None and world_bbox_max is not None:
                robot_bbox_min = self._world_to_robot_base(world_bbox_min)
                robot_bbox_max = self._world_to_robot_base(world_bbox_max)
                # Ensure min < max after rotation
                bbox_3d = {
                    "min": [
                        float(min(robot_bbox_min[0], robot_bbox_max[0])),
                        float(min(robot_bbox_min[1], robot_bbox_max[1])),
                        float(min(robot_bbox_min[2], robot_bbox_max[2])),
                    ],
                    "max": [
                        float(max(robot_bbox_min[0], robot_bbox_max[0])),
                        float(max(robot_bbox_min[1], robot_bbox_max[1])),
                        float(max(robot_bbox_min[2], robot_bbox_max[2])),
                    ],
                }

            detection = Detection(
                name=category,
                confidence=1.0,
                bbox=[0.3, 0.3, 0.7, 0.7],  # 2D placeholder
                pose={
                    "position": {
                        "x": float(robot_base_pos[0]),
                        "y": float(robot_base_pos[1]),
                        "z": float(robot_base_pos[2]),
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                    "frame": "robot_base",
                },
                metadata={
                    "role": "place_target" if role == "cab" else "pick_location",
                    "fixture_name": fixture_name,
                    "fixture_role": role,
                    "is_fixture": True,
                    "bbox_3d": bbox_3d,
                },
            )
            detections.append(detection)

        return detections

    def process(
        self,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        object_list: Optional[List[str]] = None,
        include_fixtures: bool = True,
    ) -> List[Detection]:
        """
        Extract object detections from observation (converted to robot-base frame).

        Args:
            rgb: RGB image (not used, kept for API compatibility)
            depth: Depth image (not used, kept for API compatibility)
            object_list: List of object names to detect. If None, returns all objects.
            include_fixtures: Whether to include fixture detections (cabinets, counters).

        Returns:
            List of Detection objects with positions in robot-base frame.
        """
        if self._obs is None:
            logger.warning("[PERCEPTION] No observation available, returning empty detections")
            return []

        detections = []

        # Get object configurations from episode metadata
        object_cfgs = self._ep_meta.get("object_cfgs", []) if self._ep_meta else []

        # Build category lookup
        obj_info_map = {}
        for cfg in object_cfgs:
            name = cfg.get("name", "")
            info = cfg.get("info", {})
            obj_info_map[name] = info.get("cat", "unknown")

        # Define available objects in observation
        available_objects = {
            "obj": ("obj_pos", "obj_quat"),
            "distr_counter": ("distr_counter_pos", "distr_counter_quat"),
            "distr_cab": ("distr_cab_pos", "distr_cab_quat"),
        }

        for obj_name, (pos_key, quat_key) in available_objects.items():
            if pos_key not in self._obs:
                continue

            # Get category name
            category = obj_info_map.get(obj_name, obj_name)

            # Filter by object_list if provided
            if object_list is not None:
                if category not in object_list and obj_name not in object_list:
                    continue

            # Get world frame position and convert to robot-base frame
            world_pos = self._obs[pos_key]
            robot_base_pos = self._world_to_robot_base(world_pos)

            quat = self._obs.get(quat_key, np.array([0, 0, 0, 1]))

            # Compute 3D bbox from object geometry
            bbox_3d = None
            if self._env is not None:
                obj = self._env.objects.get(obj_name)
                if obj is not None:
                    radius = getattr(obj, 'horizontal_radius', 0.05)
                    top_offset = getattr(obj, 'top_offset', np.array([0, 0, 0.05]))
                    bottom_offset = getattr(obj, 'bottom_offset', np.array([0, 0, -0.05]))
                    bbox_3d = {
                        "min": [
                            float(robot_base_pos[0] - radius),
                            float(robot_base_pos[1] - radius),
                            float(robot_base_pos[2] + bottom_offset[2]),
                        ],
                        "max": [
                            float(robot_base_pos[0] + radius),
                            float(robot_base_pos[1] + radius),
                            float(robot_base_pos[2] + top_offset[2]),
                        ],
                    }

            # Determine role for task relevance
            role = "pick_object" if obj_name == "obj" else "obstacle"

            detection = Detection(
                name=category,
                confidence=1.0,
                bbox=[0.4, 0.4, 0.6, 0.6],  # 2D placeholder
                pose={
                    "position": {
                        "x": float(robot_base_pos[0]),
                        "y": float(robot_base_pos[1]),
                        "z": float(robot_base_pos[2]),
                    },
                    "orientation": {
                        "x": float(quat[0]),
                        "y": float(quat[1]),
                        "z": float(quat[2]),
                        "w": float(quat[3]),
                    },
                    "frame": "robot_base",
                },
                metadata={
                    "role": role,
                    "obj_name": obj_name,
                    "bbox_3d": bbox_3d,
                },
            )
            detections.append(detection)

        # Add fixture detections (cabinets, counters, etc.)
        if include_fixtures:
            fixture_detections = self._detect_fixtures()
            # Filter fixtures by object_list if provided
            if object_list is not None:
                fixture_detections = [
                    d for d in fixture_detections
                    if d.name in object_list or d.metadata.get("fixture_role") in object_list
                ]
            detections.extend(fixture_detections)

        # Add interaction site detections (handles, knobs, spouts, faucets)
        interaction_sites = self._detect_interaction_sites()
        if object_list is not None:
            interaction_sites = [
                d for d in interaction_sites
                if d.name in object_list or d.metadata.get("category") in object_list
            ]
        detections.extend(interaction_sites)

        # Log detection summary for pipeline debugging
        if detections:
            det_summary = []
            for d in detections:
                pos = d.pose.get("position", {}) if isinstance(d.pose, dict) else {}
                det_summary.append(
                    f"{d.name}({pos.get('x', 0):.3f},{pos.get('y', 0):.3f},{pos.get('z', 0):.3f})"
                )
            logger.info(f"[PERCEPTION] {len(detections)} detections: {det_summary}")
        else:
            logger.warning("[PERCEPTION] 0 detections — planner will have no target positions")

        return detections

    def get_target_object_position(self) -> Optional[np.ndarray]:
        """Get target object position in robot-base frame."""
        if self._obs is None or "obj_pos" not in self._obs:
            return None
        world_pos = self._obs["obj_pos"]
        return self._world_to_robot_base(world_pos)

    def get_instruction(self) -> str:
        """Get the natural language instruction."""
        if self._ep_meta is None:
            return ""
        return self._ep_meta.get("lang", "")
