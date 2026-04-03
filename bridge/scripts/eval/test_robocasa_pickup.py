#!/usr/bin/env python3
"""
RoboCasa Pick Up Object Test Script.

Tests the full RoboBridge pipeline in RoboCasa simulation:
1. Perception: Detect objects using Florence-2
2. Planner: Generate pick/place plan using GPT-4
3. Controller: Execute primitives in simulation

Usage:
    python scripts/test_robocasa_pickup.py
    python scripts/test_robocasa_pickup.py --instruction "pick up the cup"
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration."""
    env_name: str = "PnPCounterToCab"
    robots: str = "Panda"
    instruction: Optional[str] = None  # If None, use environment's task
    save_images: bool = True
    output_dir: str = "/tmp/robocasa_test"
    max_steps: int = 500
    planner_provider: str = "openai"
    planner_model: str = "gpt-4.1"
    perception_model: str = "microsoft/Florence-2-base"
    use_perception: bool = True


class RoboCasaPickupTest:
    """Test harness for RoboCasa pick up task."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.env = None
        self.perception = None
        self.planner = None
        self.step_count = 0
        
        os.makedirs(config.output_dir, exist_ok=True)

    def setup_environment(self) -> None:
        """Initialize RoboCasa environment."""
        from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
        
        logger.info(f"Creating environment: {self.config.env_name}")
        
        self.env = PnPCounterToCab(
            robots=self.config.robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            render_camera="robot0_robotview",
            camera_names=["robot0_robotview", "robot0_eye_in_hand"],
            camera_heights=480,
            camera_widths=640,
        )
        
        self.obs = self.env.reset()
        self.ep_meta = self.env.get_ep_meta()
        self.task = self.ep_meta.get("lang", "pick up object")
        
        logger.info(f"Environment task: {self.task}")
        logger.info(f"Action dim: {self.env.action_spec[0].shape[0]}")

    def setup_perception(self) -> None:
        """Initialize perception module."""
        if not self.config.use_perception:
            logger.info("Perception disabled, using ground truth")
            return
            
        try:
            from robobridge.modules.perception import Perception
            
            logger.info(f"Loading perception: {self.config.perception_model}")
            self.perception = Perception(
                provider="hf",
                model=self.config.perception_model,
                device="cpu",
            )
            logger.info("Perception loaded")
        except Exception as e:
            logger.warning(f"Perception init failed: {e}, using ground truth")
            self.perception = None

    def setup_planner(self) -> None:
        """Initialize planner module."""
        try:
            from robobridge.modules.planner import Planner
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            logger.info(f"Loading planner: {self.config.planner_model}")
            self.planner = Planner(
                provider=self.config.planner_provider,
                model=self.config.planner_model,
                api_key=api_key,
            )
            # Initialize the LLM clients
            self.planner.initialize_client()
            logger.info("Planner loaded and initialized")
        except Exception as e:
            logger.error(f"Planner init failed: {e}")
            raise

    def get_detections(self) -> List[Dict]:
        """Get object detections from perception or ground truth."""
        if self.perception is not None:
            # Use perception model
            rgb = self.obs["robot0_robotview_image"]
            try:
                detections = self.perception.process(rgb=rgb)
                return [
                    {
                        "name": d.name,
                        "position": list(self.obs["obj_pos"]),  # Use GT position for now
                        "confidence": d.confidence,
                    }
                    for d in detections
                ]
            except Exception as e:
                logger.warning(f"Perception failed: {e}, using ground truth")
        
        # Use ground truth from simulation
        obj_pos = self.obs["obj_pos"]
        return [
            {
                "name": "object",  # Generic name since we don't know the specific object
                "position": list(obj_pos),
                "confidence": 1.0,
            }
        ]

    def generate_plan(self, instruction: str, detections: List[Dict]) -> List[Any]:
        """Generate action plan using planner."""
        world_state = {
            "detections": detections,
            "robot_ee_pos": list(self.obs["robot0_eef_pos"]),
        }
        
        logger.info(f"Planning for: {instruction}")
        logger.info(f"World state: {world_state}")
        
        try:
            primitive_plans = self.planner.process_full(instruction, world_state)
            return primitive_plans
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

    def execute_primitive(self, primitive: Dict) -> bool:
        """Execute a single primitive action in simulation."""
        ptype = primitive.get("primitive_type", "")
        
        if ptype == "grip":
            return self._execute_grip(primitive)
        elif ptype == "move":
            return self._execute_move(primitive)
        elif ptype == "go":
            return self._execute_go(primitive)
        else:
            logger.warning(f"Unknown primitive type: {ptype}")
            return False

    def _execute_grip(self, primitive: Dict) -> bool:
        """Execute gripper action."""
        grip_width = primitive.get("grip_width", 0.5)
        
        # In RoboCasa, gripper is controlled by last action dimension
        # grip_width: 1.0 = open, 0.0 = closed
        gripper_action = (grip_width * 2) - 1  # Map [0,1] to [-1,1]
        
        action = np.zeros(7)
        action[6] = gripper_action  # Gripper dimension
        
        logger.info(f"Gripper: width={grip_width:.2f}, action={gripper_action:.2f}")
        
        # Execute for multiple steps to complete the gripper motion
        for _ in range(20):
            self.obs, reward, done, info = self.env.step(action)
            self.step_count += 1
            
            if done or self.step_count >= self.config.max_steps:
                break
        
        return True

    def _execute_move(self, primitive: Dict) -> bool:
        """Execute arm movement."""
        target_pos = primitive.get("target_position", {})
        if not target_pos:
            logger.warning("No target position for move")
            return False
        
        target = np.array([
            target_pos.get("x", 0),
            target_pos.get("y", 0),
            target_pos.get("z", 0),
        ])
        
        # If target seems to be relative/placeholder (near origin), use object position
        if np.linalg.norm(target) < 1.0:
            obj_pos = self.obs["obj_pos"]
            z_offset = target[2] if target[2] != 0 else 0.15  # Use z as offset if provided
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + z_offset])
            logger.info(f"Using object-relative target: {target}")
        
        logger.info(f"Move to: {target}")
        
        # Simple proportional control to reach target
        for i in range(100):
            current_pos = self.obs["robot0_eef_pos"]
            error = target - current_pos
            
            # Check if reached
            if np.linalg.norm(error) < 0.02:
                logger.info(f"Reached target in {i} steps")
                break
            
            # Simple P control
            action = np.zeros(7)
            action[0:3] = np.clip(error * 5.0, -1.0, 1.0)  # Position control
            
            # Maintain gripper state
            if "grip_state" in primitive:
                action[6] = primitive["grip_state"]
            
            self.obs, reward, done, info = self.env.step(action)
            self.step_count += 1
            
            if done or self.step_count >= self.config.max_steps:
                break
        
        return True

    def _execute_go(self, primitive: Dict) -> bool:
        """Execute base movement (not supported for fixed arm)."""
        logger.info("Base movement not supported for Panda arm, skipping")
        return True

    def execute_plan(self, primitive_plans: List[Any]) -> bool:
        """Execute full plan."""
        for plan in primitive_plans:
            action_type = plan.parent_action.action_type
            target = plan.parent_action.target_object
            
            logger.info(f"\n{'='*40}")
            logger.info(f"Executing action: {action_type} -> {target}")
            logger.info(f"{'='*40}")
            
            for primitive in plan.primitives:
                prim_dict = primitive.to_dict()
                ptype = prim_dict.get("primitive_type", "unknown")
                
                logger.info(f"  Primitive: {ptype}")
                
                success = self.execute_primitive(prim_dict)
                
                if not success:
                    logger.error(f"Primitive {ptype} failed")
                    return False
                
                # Save image after each primitive
                if self.config.save_images:
                    self._save_image(f"step_{self.step_count:04d}_{ptype}")
        
        return True

    def _save_image(self, name: str) -> None:
        """Save current camera image."""
        img = self.obs["robot0_robotview_image"]
        path = os.path.join(self.config.output_dir, f"{name}.png")
        Image.fromarray(img).save(path)

    def run(self) -> bool:
        """Run the full test."""
        try:
            # Setup
            self.setup_environment()
            self.setup_perception()
            self.setup_planner()
            
            # Save initial state
            if self.config.save_images:
                self._save_image("00_initial")
            
            # Get instruction
            instruction = self.config.instruction or self.task
            logger.info(f"\nInstruction: {instruction}")
            
            # Perception
            logger.info("\n--- PERCEPTION ---")
            detections = self.get_detections()
            logger.info(f"Detections: {detections}")
            
            # Planning
            logger.info("\n--- PLANNING ---")
            plans = self.generate_plan(instruction, detections)
            
            if not plans:
                logger.error("No plan generated")
                return False
            
            logger.info(f"Generated {len(plans)} action plans")
            for i, plan in enumerate(plans):
                logger.info(f"  Plan {i+1}: {plan.parent_action.action_type} -> {plan.parent_action.target_object}")
                for j, prim in enumerate(plan.primitives):
                    logger.info(f"    {j+1}. {prim.primitive_type}")
            
            # Execution
            logger.info("\n--- EXECUTION ---")
            success = self.execute_plan(plans)
            
            # Save final state
            if self.config.save_images:
                self._save_image("99_final")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {'PASSED' if success else 'FAILED'}")
            logger.info(f"Total steps: {self.step_count}")
            logger.info(f"Images saved to: {self.config.output_dir}")
            logger.info(f"{'='*60}")
            
            return success
            
        except Exception as e:
            logger.exception(f"Test failed with error: {e}")
            return False
            
        finally:
            if self.env is not None:
                self.env.close()


def main():
    parser = argparse.ArgumentParser(description="RoboCasa Pick Up Test")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Custom instruction (default: use environment task)")
    parser.add_argument("--no-perception", action="store_true",
                        help="Disable perception, use ground truth")
    parser.add_argument("--output-dir", type=str, default="/tmp/robocasa_test",
                        help="Output directory for images")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum simulation steps")
    args = parser.parse_args()
    
    config = TestConfig(
        instruction=args.instruction,
        use_perception=not args.no_perception,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    
    test = RoboCasaPickupTest(config)
    success = test.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
