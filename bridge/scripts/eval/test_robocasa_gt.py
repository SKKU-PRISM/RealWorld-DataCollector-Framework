#!/usr/bin/env python3
"""
RoboBridge Pipeline Test with Ground Truth Perception

Tests the complete module pipeline in RoboCasa using ground truth
object positions from the environment instead of vision models.

Pipeline:
1. Perception (Ground Truth): Extract object positions from env observation
2. Planner (GPT-4): Generate primitive action plan  
3. Controller: Execute primitives in simulation

Usage:
    python scripts/test_robocasa_gt.py
    python scripts/test_robocasa_gt.py --output-dir /path/to/output
"""

import argparse
import logging
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline_test")

# Reduce noise from other loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class PipelineConfig:
    """Pipeline test configuration."""
    output_dir: str = "/tmp/robobridge_gt_pipeline"
    
    # Dataset - use recorded demos instead of random generation
    dataset_path: str = ""  # If empty, use random generation (legacy)
    demo_index: int = 0  # Which demo to load from dataset
    
    # Planner
    planner_provider: str = "openai"
    planner_model: str = "gpt-4.1"
    
    # Execution
    max_steps: int = 2000
    skip_cabinet_actions: bool = True  # Skip open/close cabinet (out of workspace)
    
    # Camera
    camera_height: int = 480
    camera_width: int = 640


class ModuleLogger:
    """Helper to log module inputs/outputs clearly."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = open(os.path.join(output_dir, "pipeline_log.txt"), "w")
    
    def section(self, title: str):
        sep = "=" * 70
        msg = f"\n{sep}\n{title}\n{sep}\n"
        print(msg)
        self.log_file.write(msg)
        self.log_file.flush()
    
    def log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
    
    def log_json(self, name: str, data: Any):
        self.log(f"\n{name}:")
        formatted = json.dumps(data, indent=2, default=str)
        self.log(formatted)
        
        # Also save to file
        path = os.path.join(self.output_dir, f"{name.lower().replace(' ', '_')}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_image(self, name: str, img: np.ndarray):
        path = os.path.join(self.output_dir, f"{name}.png")
        Image.fromarray(img).save(path)
        self.log(f"Saved image: {path}")
    
    def close(self):
        self.log_file.close()


class RoboCasaGTTest:
    """Pipeline test using ground truth perception from RoboCasa."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = ModuleLogger(config.output_dir)
        
        self.env = None
        self.perception = None
        self.planner = None
        self.obs = None
        self.ep_meta = None
        self.instruction = ""
        self.step_count = 0
        
        # Robot base frame transformation
        self.robot_base_pos = None
        self.robot_base_rot = None  # scipy Rotation object
        
        # Object geometry for proper grasp height calculation
        self.obj_top_offset = None
        self.obj_bottom_offset = None
        self.obj_horizontal_radius = None
        
        # Fixture positions (extracted from env after reset)
        self.fixture_positions = {}
        
        # Action dimension (varies by robot: Panda=7, PandaMobile=12)
        self.action_dim = None
        self.gripper_idx = None  # Index of gripper action in action vector
    
    def setup_environment(self):
        """Initialize RoboCasa environment.
        
        If config.dataset_path is set, load episode from HDF5 dataset.
        Otherwise, generate random episode (legacy behavior).
        """
        self.logger.section("1. ENVIRONMENT SETUP")
        
        import robosuite as suite
        
        # Load from dataset or generate random episode
        if self.config.dataset_path:
            self._load_from_dataset()
        else:
            # Legacy: create environment directly with random generation
            from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
            
            self.logger.log("Creating RoboCasa PnPCounterToCab environment with IK controller...")
            
            controller_config = suite.load_part_controller_config(default_controller="IK_POSE")
            
            self.env = PnPCounterToCab(
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,
                ignore_done=True,
                render_camera="robot0_robotview",
                camera_names=["robot0_robotview", "robot0_eye_in_hand"],
                camera_heights=self.config.camera_height,
                camera_widths=self.config.camera_width,
                obj_groups="condiment",
                controller_configs=controller_config,
            )
            
            self.obs = self.env.reset()
            self.ep_meta = self.env.get_ep_meta()
        
        self.instruction = self.ep_meta.get("lang", "pick up object")
        
        # Get action dimension - varies by robot (Panda=7, PandaMobile=12, etc.)
        self.action_dim = self.env.action_spec[0].shape[0]
        # Gripper is typically at index 6 for arm-only robots, or at end for mobile robots
        # For OSC_POSE: [dx, dy, dz, drx, dry, drz, gripper] = 7
        # For OSC_POSE + mobile: [dx, dy, dz, drx, dry, drz, gripper, base_x, base_y, base_theta, ...] = 12
        # Actually for PandaMobile: [arm_x, arm_y, arm_z, arm_rx, arm_ry, arm_rz, gripper, base_x, base_y, base_theta, base_vel_x, base_vel_y] = 12
        # The gripper index depends on robot config - let's detect it
        self.gripper_idx = 6  # Default for Panda arm
        
        self.logger.log(f"Environment created successfully")
        self.logger.log(f"Instruction: {self.instruction}")
        self.logger.log(f"Action dimension: {self.action_dim}")
        
        # Get robot base frame for coordinate transformation
        self._setup_robot_base_frame()
        
        # Get target object geometry for proper grasp calculations
        self._setup_object_geometry()
        
        # Extract fixture positions from environment
        self._setup_fixture_positions()
        
        # Log episode info
        object_cfgs = self.ep_meta.get("object_cfgs", [])
        self.logger.log(f"\nObjects in scene: {len(object_cfgs)}")
        for cfg in object_cfgs:
            info = cfg.get("info", {})
            self.logger.log(f"  - {cfg.get('name')}: {info.get('cat', 'unknown')}")
        
        # Save initial image
        self.logger.save_image("01_initial_scene", self.obs["robot0_robotview_image"])
    
    def _load_from_dataset(self):
        """Load episode from HDF5 dataset instead of random generation.
        
        This ensures reproducible episodes that are known to be achievable
        (since they were recorded from successful human demonstrations).
        
        Key insight: We must create the environment using the metadata from
        the dataset (env_args), not our own parameters. This ensures the
        robot configuration matches the recording.
        """
        from robocasa.scripts.playback_dataset import reset_to, get_env_metadata_from_dataset
        import robosuite
        
        self.logger.log(f"\nLoading from dataset: {self.config.dataset_path}")
        self.logger.log(f"Demo index: {self.config.demo_index}")
        
        # Step 1: Get environment metadata from dataset and create matching environment
        env_meta = get_env_metadata_from_dataset(self.config.dataset_path)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True  # We need camera observations
        env_kwargs["ignore_done"] = True  # Don't terminate early
        
        # Override camera settings - dataset may reference cameras that don't exist
        # in current RoboCasa version. Use cameras we know exist.
        env_kwargs["camera_names"] = ["robot0_robotview", "robot0_eye_in_hand"]
        env_kwargs["camera_heights"] = self.config.camera_height
        env_kwargs["camera_widths"] = self.config.camera_width
        
        self.logger.log(f"Creating environment from dataset metadata: {env_meta['env_name']}")
        self.logger.log(f"Robot: {env_kwargs.get('robots', 'unknown')}")
        self.env = robosuite.make(**env_kwargs)
        
        # Step 2: Load specific demo
        f = h5py.File(self.config.dataset_path, 'r')
        
        # Get sorted demo names (demo_1, demo_2, ..., not demo_0)
        demo_names = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))
        
        if self.config.demo_index >= len(demo_names):
            raise ValueError(f"Demo index {self.config.demo_index} out of range (max: {len(demo_names)-1})")
        
        demo_name = demo_names[self.config.demo_index]
        self.logger.log(f"Loading demo: {demo_name}")
        
        demo = f[f'data/{demo_name}']
        
        # Get initial state
        states = demo['states'][()]
        initial_state = {
            'states': states[0],
            'model': demo.attrs['model_file'],
            'ep_meta': demo.attrs.get('ep_meta', None),
        }
        
        # Parse ep_meta
        if initial_state['ep_meta']:
            self.ep_meta = json.loads(initial_state['ep_meta'])
        else:
            self.ep_meta = {}
        
        self.logger.log(f"Episode language: {self.ep_meta.get('lang', 'N/A')}")
        
        # Step 3: Reset environment to initial state from dataset
        reset_to(self.env, initial_state)
        
        # Get observation after reset
        self.obs = self.env._get_observations()
        
        f.close()
        
        self.logger.log(f"Successfully loaded episode from dataset")
    
    def _setup_robot_base_frame(self):
        """Extract robot base position and orientation for coordinate transformation."""
        robot = self.env.robots[0]
        sim = self.env.sim
        
        # Get robot base body position and orientation
        base_body_id = sim.model.body_name2id(robot.robot_model.root_body)
        self.robot_base_pos = sim.data.body_xpos[base_body_id].copy()
        base_quat_wxyz = sim.data.body_xquat[base_body_id]  # MuJoCo uses w,x,y,z
        
        # Convert to scipy Rotation (uses x,y,z,w)
        quat_xyzw = [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        self.robot_base_rot = R.from_quat(quat_xyzw)
        
        euler = self.robot_base_rot.as_euler('xyz', degrees=True)
        self.logger.log(f"Robot base position: {self.robot_base_pos}")
        self.logger.log(f"Robot base rotation (euler xyz deg): {euler}")
    
    def _setup_object_geometry(self):
        """Extract target object geometry for proper grasp height calculation.
        
        Objects in RoboCasa have:
        - top_offset: distance from center to top surface
        - bottom_offset: distance from center to bottom surface  
        - horizontal_radius: radius in XY plane
        
        For small objects like fruit (~5cm diameter), obj_pos gives CENTER.
        To grasp properly, gripper must approach at correct height.
        """
        try:
            obj = self.env.objects['obj']
            
            if hasattr(obj, 'top_offset'):
                self.obj_top_offset = obj.top_offset.copy()
            if hasattr(obj, 'bottom_offset'):
                self.obj_bottom_offset = obj.bottom_offset.copy()
            if hasattr(obj, 'horizontal_radius'):
                self.obj_horizontal_radius = obj.horizontal_radius
            
            self.logger.log(f"\nTarget object geometry:")
            self.logger.log(f"  Top offset: {self.obj_top_offset}")
            self.logger.log(f"  Bottom offset: {self.obj_bottom_offset}")
            self.logger.log(f"  Horizontal radius: {self.obj_horizontal_radius}")
            
            # Calculate object height
            if self.obj_top_offset is not None and self.obj_bottom_offset is not None:
                obj_height = self.obj_top_offset[2] - self.obj_bottom_offset[2]
                self.logger.log(f"  Object height: {obj_height:.4f}m")
        except Exception as e:
            self.logger.log(f"Warning: Could not get object geometry: {e}")
    
    def _get_world_object_position(self, obj_name: str = "obj") -> Optional[np.ndarray]:
        """Get object position in world frame from MuJoCo simulation.
        
        The observation obj_pos is relative to robot base frame.
        This method gets the absolute world position directly from simulation.
        
        Args:
            obj_name: Name of the object (default "obj" for target object)
            
        Returns:
            World-frame position [x, y, z] or None if object not found
        """
        try:
            obj = self.env.objects.get(obj_name)
            if obj is None:
                return None
            body_id = self.env.sim.model.body_name2id(obj.root_body)
            return self.env.sim.data.body_xpos[body_id].copy()
        except Exception as e:
            self.logger.log(f"Warning: Could not get world position for {obj_name}: {e}")
            return None
    
    def _setup_fixture_positions(self):
        """Extract fixture positions from environment after reset.
        
        Fixtures (cabinet, counter, sink, etc.) positions are only available
        via env.fixture_refs after the environment is loaded. The ep_meta
        only stores fixture class names, not positions.
        
        This is critical for:
        1. Planning cabinet/drawer manipulation
        2. Verifying object placement targets are reachable
        3. Providing context to the LLM planner
        """
        self.fixture_positions = {}
        
        try:
            # fixture_refs maps task-relevant names (e.g., 'cab', 'counter') 
            # to actual fixture names in the scene
            fixture_refs = getattr(self.env, 'fixture_refs', {})
            
            self.logger.log(f"\nFixture positions:")
            
            for ref_name, fixture in fixture_refs.items():
                # Fixtures have a 'pos' attribute for their position
                if hasattr(fixture, 'pos'):
                    pos = fixture.pos
                    self.fixture_positions[ref_name] = {
                        'position': pos.tolist() if hasattr(pos, 'tolist') else list(pos),
                        'name': fixture.name if hasattr(fixture, 'name') else ref_name,
                    }
                    self.logger.log(f"  {ref_name}: pos={pos}")
                    
                    # Get fixture bounding box if available (for planning)
                    if hasattr(fixture, 'size'):
                        self.fixture_positions[ref_name]['size'] = fixture.size.tolist() if hasattr(fixture.size, 'tolist') else list(fixture.size)
                        self.logger.log(f"    size={fixture.size}")
            
            # Get cabinet interior position (CRITICAL for place tasks!)
            # fixture.pos is the cabinet's reference point, NOT where objects go
            # We need the interior shelf position from MuJoCo sites
            for ref_name, fixture in fixture_refs.items():
                if 'cab' in ref_name.lower():
                    try:
                        fixture_name = fixture.name if hasattr(fixture, 'name') else ref_name
                        int_site_candidates = [
                            f"{fixture_name}_int_p0",
                            f"{fixture_name.replace('_', '_main_group_')}_int_p0",
                            "cab_main_main_group_int_p0",
                        ]
                        
                        for site_name in int_site_candidates:
                            try:
                                int_pos = self.env.sim.data.get_site_xpos(site_name)
                                self.fixture_positions[ref_name]['interior_position'] = int_pos.tolist()
                                self.fixture_positions[ref_name]['position'] = int_pos.tolist()
                                self.logger.log(f"  {ref_name} interior: {int_pos}")
                                break
                            except:
                                continue
                    except Exception as e:
                        self.logger.log(f"  Warning: Could not get interior for {ref_name}: {e}")
            
            # Add common aliases for fixture names (cab -> cabinet, etc.)
            if 'cab' in fixture_refs and 'cabinet' not in fixture_refs:
                self.fixture_positions['cabinet'] = self.fixture_positions['cab']
                self.logger.log(f"  Added alias: cabinet -> cab")
                    
        except Exception as e:
            self.logger.log(f"Warning: Could not extract fixture positions: {e}")
    
    def world_to_robot_base(self, world_vec: np.ndarray) -> np.ndarray:
        """Transform a direction vector from world frame to robot base frame.
        
        The OSC controller expects delta actions in the robot base frame.
        Since robot base is rotated 90 degrees (Z-axis) in kitchen scenes:
        - World +X → Robot base -Y  
        - World +Y → Robot base +X
        - World +Z → Robot base +Z
        """
        return self.robot_base_rot.inv().apply(world_vec)
    
    def _adjust_grasp_height(self, target: np.ndarray, gt_obj_pos: np.ndarray) -> np.ndarray:
        """Adjust target Z for proper grasping based on object geometry.
        
        Problem: Object position (obj_pos) is the geometric CENTER of the object.
        - Small objects (~5cm): Going to center Z pushes the object
        - Tall objects (bottles): Center might not be graspable (too wide)
        
        Solution: Calculate optimal grasp height based on object shape
        - Small/flat objects: Grasp near center with slight offset for fingers
        - Tall objects (bottles): Grasp higher (toward neck) where narrower
        
        Panda gripper specs:
        - Max opening: ~8cm
        - Fingertip to EEF: ~4cm
        
        For "approach from above" moves, we keep the higher Z.
        """
        # Check if this appears to be a grasp move (target Z close to object center Z)
        z_diff = abs(target[2] - gt_obj_pos[2])
        is_grasp_move = z_diff < 0.05  # Within 5cm of object center
        
        if not is_grasp_move:
            # This is probably an "above" or "lift" move, keep as is
            return target
        
        # Calculate proper grasp height
        if self.obj_top_offset is not None and self.obj_bottom_offset is not None:
            obj_half_height = self.obj_top_offset[2]  # Z component (distance from center to top)
            obj_height = 2 * obj_half_height
            obj_diameter = 2 * self.obj_horizontal_radius if self.obj_horizontal_radius else 0.05
            
            # Panda gripper constraints
            gripper_max_opening = 0.08  # 8cm
            
            # Determine grasp strategy based on object shape
            # KEY INSIGHT: obj_pos is object CENTER (halfway up the object).
            # For successful grasping, we need to go LOW enough that fingers
            # can wrap around the object, but not so low that we hit the table.
            # 
            # Panda gripper geometry:
            # - EEF frame is at the "palm" of the gripper  
            # - Finger contact point is ~3-4cm below EEF frame
            # - Fingers are ~4cm long and close toward center
            #
            # Strategy: Go to a height where gripper fingers are at object center
            # This means EEF should be slightly ABOVE the object center.
            # But we also need clearance from table (~1cm safety margin).
            
            # Table surface is approximately at object_bottom - small margin
            table_z_estimate = gt_obj_pos[2] - obj_half_height + 0.005  # 5mm margin from table
            
            # GRASP STRATEGY:
            # The Panda gripper EEF frame is at the "palm". Finger contact points
            # are approximately 4cm below EEF. For successful grasping:
            # - Go LOW enough that fingers wrap around widest part of object
            # - Don't go so low that we hit the table
            #
            # For bottles/cylinders: widest part is often the body (center to lower)
            # For most objects: center height works well
            
            # Minimum clearance from table
            min_z_clearance = table_z_estimate + 0.02  # 2cm above table
            
            # After testing: going below center causes collision with table.
            # The Panda gripper fingers are ~4cm long. If EEF is at object center,
            # the fingertips start at center-4cm. For successful grasping of a 
            # 6-7cm diameter object, we need the fingers to contact at ~diameter/2
            # from center on each side.
            #
            # Best strategy: EEF slightly above center, so fingers are at center level
            # This avoids table collision while still grasping the object body.
            
            # Use a small positive offset (1cm) so fingers reach object center
            if obj_height > 0.15:  # Tall object (>15cm) - bottles
                grasp_z = gt_obj_pos[2] + 0.01  # 1cm above center
                grasp_type = "tall_bottle"
            elif obj_height > 0.08:  # Medium object (8-15cm)
                grasp_z = gt_obj_pos[2] + 0.01  # 1cm above center
                grasp_type = "medium"
            else:  # Small object (<8cm)
                grasp_z = max(gt_obj_pos[2] + 0.01, min_z_clearance)
                grasp_type = "small"
            
            grasp_offset = grasp_z - gt_obj_pos[2]
            
            adjusted_z = gt_obj_pos[2] + grasp_offset
            
            self.logger.log(f"  Grasp height adjustment ({grasp_type}):")
            self.logger.log(f"    Object center Z: {gt_obj_pos[2]:.4f}")
            self.logger.log(f"    Object height: {obj_height:.4f}m, diameter: {obj_diameter:.4f}m")
            self.logger.log(f"    Grasp offset: {grasp_offset:.4f}")
            self.logger.log(f"    Original target Z: {target[2]:.4f}")
            self.logger.log(f"    Adjusted grasp Z: {adjusted_z:.4f}")
            
            target = np.array([target[0], target[1], adjusted_z])
        else:
            self.logger.log("  Warning: No object geometry, using original target")
        
        return target
    
    def setup_perception(self):
        """Initialize ground truth perception adapter."""
        self.logger.section("2. PERCEPTION MODULE SETUP (Ground Truth)")
        
        from robobridge.wrappers.robocasa_perception import RoboCasaPerception
        
        self.logger.log("Using RoboCasaPerception (ground truth from environment)")
        self.logger.log("No vision model loading required")
        
        self.perception = RoboCasaPerception()
        self.perception.set_environment_state(self.obs, self.ep_meta)
        
        self.logger.log("Perception module initialized successfully")
    
    def setup_planner(self):
        """Initialize Planner module."""
        self.logger.section("3. PLANNER MODULE SETUP")
        
        from robobridge.modules.planner import Planner
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.logger.log(f"Provider: {self.config.planner_provider}")
        self.logger.log(f"Model: {self.config.planner_model}")
        
        self.planner = Planner(
            provider=self.config.planner_provider,
            model=self.config.planner_model,
            api_key=api_key,
        )
        self.planner.initialize_client()
        
        self.logger.log("Planner module initialized successfully")
    
    def run_perception(self) -> List[Dict]:
        """Extract object detections from ground truth.
        
        IMPORTANT: The observation obj_pos is in robot-base frame, not world frame.
        For planning and execution, we need world-frame coordinates.
        This method gets positions directly from MuJoCo simulation for accuracy.
        """
        self.logger.section("4. PERCEPTION EXECUTION (Ground Truth)")
        
        self.logger.log("Extracting object positions from environment (world frame)...")
        start_time = time.time()
        
        detection_list = []
        
        # Get object configurations from episode metadata
        object_cfgs = self.ep_meta.get("object_cfgs", []) if self.ep_meta else []
        
        # Map object names to their configurations
        obj_info_map = {}
        for cfg in object_cfgs:
            name = cfg.get("name", "")
            info = cfg.get("info", {})
            obj_info_map[name] = {
                "name": name,
                "category": info.get("cat", "unknown"),
                "graspable": cfg.get("graspable", False),
            }
        
        # Get world-frame positions from MuJoCo simulation
        for obj_name, obj in self.env.objects.items():
            try:
                body_id = self.env.sim.model.body_name2id(obj.root_body)
                pos = self.env.sim.data.body_xpos[body_id].copy()
                quat_wxyz = self.env.sim.data.body_xquat[body_id].copy()
                # Convert from MuJoCo (w,x,y,z) to (x,y,z,w)
                quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                
                info = obj_info_map.get(obj_name, {"category": obj_name, "graspable": False})
                category = info.get("category", obj_name)
                is_target = (obj_name == "obj")
                
                det_dict = {
                    "name": category,
                    "confidence": 1.0,
                    "bbox": [0.4, 0.4, 0.6, 0.6],  # Placeholder
                    "pose": {
                        "position": {
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2]),
                        },
                        "orientation": {
                            "x": float(quat[0]),
                            "y": float(quat[1]),
                            "z": float(quat[2]),
                            "w": float(quat[3]),
                        },
                        "source": "ground_truth_world",
                        "graspable": info.get("graspable", False) or is_target,
                        "is_target": is_target,
                        "obj_name": obj_name,
                    },
                }
                detection_list.append(det_dict)
            except Exception as e:
                self.logger.log(f"Warning: Could not get position for {obj_name}: {e}")
        
        elapsed = time.time() - start_time
        self.logger.log(f"Extraction completed in {elapsed:.4f}s")
        self.logger.log(f"Number of objects detected: {len(detection_list)}")
        
        for i, det in enumerate(detection_list):
            pos = det["pose"]["position"]
            is_target = det["pose"].get("is_target", False)
            target_marker = " [TARGET]" if is_target else ""
            
            self.logger.log(f"\nObject {i+1}: {det['name']}{target_marker}")
            self.logger.log(f"  Position (world): ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            graspable = det["pose"].get("graspable", False)
            self.logger.log(f"  Graspable: {graspable}")
        
        self.logger.log_json("Perception Output", detection_list)
        
        return detection_list
    
    def run_planner(self, detections: List[Dict]) -> List[Any]:
        """Run planner to generate primitive plans."""
        self.logger.section("5. PLANNER EXECUTION")
        
        self.logger.log(f"Instruction: {self.instruction}")
        
        # Build world state with detections, robot state, and fixture positions
        world_state = {
            "detections": detections,
            "robot_ee_pos": self.obs["robot0_eef_pos"].tolist(),
            "fixtures": self.fixture_positions,  # Include fixture positions for LLM context
        }
        
        self.logger.log_json("Planner Input (World State)", world_state)
        
        self.logger.log("\nGenerating action plan with LLM...")
        start_time = time.time()
        
        primitive_plans = self.planner.process_full(self.instruction, world_state)
        
        elapsed = time.time() - start_time
        self.logger.log(f"Planning completed in {elapsed:.2f}s")
        
        if not primitive_plans:
            self.logger.log("ERROR: No plan generated!")
            return []
        
        self.logger.log(f"Number of action plans: {len(primitive_plans)}")
        
        # Log each plan in detail
        plans_data = []
        for i, plan in enumerate(primitive_plans):
            plan_dict = plan.to_dict()
            plans_data.append(plan_dict)
            
            action = plan.parent_action
            self.logger.log(f"\nPlan {i+1}: {action.action_type} -> {action.target_object}")
            self.logger.log(f"  Number of primitives: {len(plan.primitives)}")
            
            for j, prim in enumerate(plan.primitives):
                prim_dict = prim.to_dict()
                ptype = prim_dict["primitive_type"]
                self.logger.log(f"    {j+1}. {ptype}")
                
                if ptype == "move" and prim.target_position:
                    pos = prim.target_position
                    self.logger.log(f"       Target: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
                elif ptype == "grip":
                    self.logger.log(f"       Width: {prim.grip_width:.2f}")
        
        self.logger.log_json("Planner Output (Primitive Plans)", plans_data)
        
        return primitive_plans
    
    def execute_primitive(self, primitive: Dict, gt_obj_pos: np.ndarray) -> bool:
        """Execute a single primitive action."""
        ptype = primitive.get("primitive_type", "")
        
        if ptype == "grip":
            return self._execute_grip(primitive)
        elif ptype == "move":
            return self._execute_move(primitive, gt_obj_pos)
        elif ptype == "go":
            self.logger.log("  GO primitive: Skipping (no mobile base)")
            return True
        else:
            self.logger.log(f"  Unknown primitive type: {ptype}")
            return False
    
    def _execute_grip(self, primitive: Dict) -> bool:
        """Execute gripper action.
        
        Robosuite/Panda gripper action mapping:
        - action = -1 => OPEN (fingers apart)
        - action = +1 => CLOSED (fingers together)
        
        Planner grip_width convention:
        - grip_width = 0.08 (or high) => OPEN
        - grip_width = 0.00 => CLOSED
        
        So we need: gripper_action = 1 - (grip_width / 0.08) * 2
        Or simpler: -1 for open (width > 0), +1 for close (width ~= 0)
        """
        grip_width = primitive.get("grip_width", 0.5)
        is_closing = grip_width < 0.01
        
        # CORRECT MAPPING: grip_width=0.08 -> -1 (open), grip_width=0.0 -> +1 (close)
        # Invert the mapping: high width = open = -1, low width = close = +1
        gripper_action = 1.0 - (grip_width / 0.04)  # 0.08->-1, 0.0->+1
        gripper_action = np.clip(gripper_action, -1.0, 1.0)
        
        # Log gripper and object positions before gripping
        eef_pos = self.obs["robot0_eef_pos"]
        obj_pos = self.obs["obj_pos"]
        xy_error = np.linalg.norm(eef_pos[:2] - obj_pos[:2])
        self.logger.log(f"  Gripper: width={grip_width:.2f} (action={gripper_action:.2f})")
        self.logger.log(f"    EEF pos: ({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f})")
        self.logger.log(f"    Obj pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        self.logger.log(f"    XY error: {xy_error:.4f}m")
        
        # Execute grip action while actively holding position
        # Store target position to maintain during grip
        target_pos = self.obs["robot0_eef_pos"].copy()
        
        # Execute grip action (more steps for closing to ensure contact)
        num_steps = 50 if is_closing else 30
        gain = 5.0  # Position hold gain
        
        for _ in range(num_steps):
            # Calculate position error and correct
            current_pos = self.obs["robot0_eef_pos"]
            pos_error = target_pos - current_pos
            pos_correction = self.world_to_robot_base(pos_error) * gain
            pos_correction = np.clip(pos_correction, -0.5, 0.5)
            
            # Use correct action dimension (varies by robot)
            action = np.zeros(self.action_dim)
            action[0:3] = pos_correction  # Actively hold position
            action[self.gripper_idx] = gripper_action
            
            self.obs, reward, done, info = self.env.step(action)
            self.step_count += 1
        
        # Log final gripper state
        gripper_state = self.obs["robot0_gripper_qpos"]
        self.logger.log(f"    Gripper state after: {gripper_state}")
        
        # Check if grasp succeeded (gripper stopped at non-zero width = object in gripper)
        if is_closing:
            gripper_opening = abs(gripper_state[0]) + abs(gripper_state[1])
            if gripper_opening > 0.01:  # Fingers stopped with some opening
                self.logger.log(f"    >>> GRASP DETECTED (opening={gripper_opening:.4f})")
            else:
                self.logger.log(f"    >>> NO GRASP (fingers fully closed)")
        
        return True
    
    def _execute_move(self, primitive: Dict, gt_obj_pos: np.ndarray) -> bool:
        """Execute arm movement using IK controller with DELTA position control.
        
        IK controller in robosuite only supports delta mode.
        We compute the error between current and target position,
        then send scaled delta commands.
        """
        target_pos_dict = primitive.get("target_position", {})
        if not target_pos_dict:
            self.logger.log("  No target position!")
            return False
        
        target = np.array([
            target_pos_dict.get("x", 0),
            target_pos_dict.get("y", 0),
            target_pos_dict.get("z", 0),
        ])
        
        dist_to_robot_base = np.linalg.norm(target[:2] - self.robot_base_pos[:2])
        is_plausible_target = dist_to_robot_base < 1.5 and target[2] > 0.5 and target[2] < 2.5
        
        if not is_plausible_target:
            self.logger.log(f"  Planner target implausible: {target}")
            z_offset = 0.15
            target = np.array([gt_obj_pos[0], gt_obj_pos[1], gt_obj_pos[2] + z_offset])
            self.logger.log(f"  Using GT-adjusted target: {target}")
        
        target = self._adjust_grasp_height(target, gt_obj_pos)
        self.logger.log(f"  Target (absolute): ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        
        max_move_steps = 200
        position_threshold = 0.02
        gain = 5.0  # Delta gain for position control
        
        for i in range(max_move_steps):
            current_pos = self.obs["robot0_eef_pos"]
            error = target - current_pos
            dist = np.linalg.norm(error)
            
            if dist < position_threshold:
                self.logger.log(f"  Reached target in {i} steps (dist={dist:.4f})")
                return True
            
            # OSC_POSE controller expects deltas in ROBOT BASE FRAME
            # The robot base is rotated (e.g., 180° in kitchen scenes)
            # We must transform world-frame error to robot-base frame
            delta = self.world_to_robot_base(error) * gain
            delta = np.clip(delta, -1.0, 1.0)  # Clip to action limits
            
            # Use correct action dimension (varies by robot)
            action = np.zeros(self.action_dim)
            action[0:3] = delta  # Delta position in robot base frame
            # action[3:6] = 0  # No orientation change (default)
            # Gripper and base actions remain 0 (no change)
            
            self.obs, reward, done, info = self.env.step(action)
            self.step_count += 1
            
            if i > 0 and i % 50 == 0:
                self.logger.log(f"    Step {i}: dist={dist:.4f}")
            
            if self.step_count >= self.config.max_steps:
                self.logger.log(f"  Max total steps reached")
                break
        
        final_dist = np.linalg.norm(target - self.obs["robot0_eef_pos"])
        self.logger.log(f"  Move completed (final dist={final_dist:.4f})")
        return final_dist < 0.05
    
    def run_execution(self, primitive_plans: List[Any]) -> bool:
        """Execute all primitive plans."""
        self.logger.section("6. EXECUTION")
        
        # Get object position in WORLD FRAME (not robot-relative from obs)
        # obs["obj_pos"] is relative to robot base, which confuses the planner
        gt_obj_pos = self._get_world_object_position("obj")
        if gt_obj_pos is None:
            gt_obj_pos = self.obs["obj_pos"]  # Fallback to robot-relative
            self.logger.log(f"WARNING: Using robot-relative obj_pos (may cause issues)")
        
        self.logger.log(f"Target object position (world frame): {gt_obj_pos.tolist()}")
        
        success = True
        
        for i, plan in enumerate(primitive_plans):
            action = plan.parent_action
            
            # Skip cabinet actions if configured (they're typically out of workspace)
            if self.config.skip_cabinet_actions:
                if action.action_type in ["open", "close"] and "cabinet" in action.target_object.lower():
                    self.logger.log(f"\n--- SKIPPING: {action.action_type} -> {action.target_object} (out of workspace) ---")
                    continue
            
            self.logger.log(f"\n--- Executing: {action.action_type} -> {action.target_object} ---")
            
            for j, prim in enumerate(plan.primitives):
                prim_dict = prim.to_dict()
                ptype = prim_dict["primitive_type"]
                
                self.logger.log(f"\nPrimitive {j+1}: {ptype}")
                
                prim_success = self.execute_primitive(prim_dict, gt_obj_pos)
                
                if not prim_success:
                    self.logger.log(f"FAILED: {ptype}")
                    success = False
                    break
                
                # Update perception with new observation
                self.perception.set_environment_state(self.obs, self.ep_meta)
                
                # Save image after key primitives
                if ptype in ["move", "grip"]:
                    img_name = f"exec_{i+1}_{j+1}_{ptype}"
                    self.logger.save_image(img_name, self.obs["robot0_robotview_image"])
            
            if not success:
                break
        
        # Save final state
        self.logger.save_image("99_final_state", self.obs["robot0_robotview_image"])
        
        return success
    
    def run(self) -> bool:
        """Run the full pipeline test."""
        try:
            self.logger.section("ROBOBRIDGE PIPELINE TEST (Ground Truth Perception)")
            self.logger.log(f"Output directory: {self.config.output_dir}")
            self.logger.log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Setup all modules
            self.setup_environment()
            self.setup_perception()
            self.setup_planner()
            
            # Run pipeline
            detections = self.run_perception()
            primitive_plans = self.run_planner(detections)
            
            if not primitive_plans:
                self.logger.log("\nPipeline failed: No plan generated")
                return False
            
            success = self.run_execution(primitive_plans)
            
            # Summary
            self.logger.section("7. TEST SUMMARY")
            self.logger.log(f"Result: {'PASS' if success else 'FAIL'}")
            self.logger.log(f"Total simulation steps: {self.step_count}")
            self.logger.log(f"Output saved to: {self.config.output_dir}")
            
            # Final state
            final_state = {
                "object_position": self.obs["obj_pos"].tolist(),
                "robot_ee_position": self.obs["robot0_eef_pos"].tolist(),
                "gripper_state": self.obs["robot0_gripper_qpos"].tolist(),
            }
            self.logger.log_json("Final State", final_state)
            
            return success
            
        except Exception as e:
            self.logger.log(f"\nERROR: {e}")
            import traceback
            self.logger.log(traceback.format_exc())
            return False
            
        finally:
            if self.env:
                self.env.close()
            self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="RoboBridge Pipeline Test with GT Perception")
    parser.add_argument("--output-dir", "-o", default="/tmp/robobridge_gt_pipeline",
                        help="Output directory for logs and images")
    parser.add_argument("--model", "-m", default="gpt-5.2",
                        help="LLM model for planner")
    parser.add_argument("--dataset", "-d", 
                        default="./data/demo.hdf5",
                        help="Path to HDF5 dataset (empty for random generation)")
    parser.add_argument("--demo", type=int, default=0,
                        help="Demo index to load from dataset (0-indexed)")
    args = parser.parse_args()
    
    config = PipelineConfig(
        output_dir=args.output_dir,
        planner_model=args.model,
        dataset_path=args.dataset,
        demo_index=args.demo,
    )
    
    test = RoboCasaGTTest(config)
    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
