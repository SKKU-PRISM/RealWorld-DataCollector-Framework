#!/usr/bin/env python3
"""
RoboBridge Visual Pipeline Test

실시간 시뮬레이션 렌더링 + 각 모듈 출력 확인
- MuJoCo viewer로 시뮬레이션 시각화
- 각 모듈(Perception, Planner, Controller) 출력을 터미널에 표시
- 각 단계마다 사용자 입력을 기다림 (step-by-step 모드)

Usage:
    python scripts/test_pipeline_visual.py
    python scripts/test_pipeline_visual.py --auto  # 자동 진행
"""

import argparse
import logging
import os
import sys
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- {text} ---{Colors.ENDC}")

def print_module(module: str, text: str):
    colors = {
        "ENV": Colors.BLUE,
        "PERCEPTION": Colors.GREEN,
        "PLANNER": Colors.YELLOW,
        "CONTROLLER": Colors.RED,
    }
    color = colors.get(module, Colors.ENDC)
    print(f"{color}[{module}]{Colors.ENDC} {text}")

def print_json(data: Any, indent: int = 2):
    print(json.dumps(data, indent=indent, default=str))

def wait_for_user(auto_mode: bool, message: str = "Press Enter to continue..."):
    if not auto_mode:
        input(f"\n{Colors.BOLD}{message}{Colors.ENDC}")


class VisualPipelineTest:
    """Visual pipeline test with MuJoCo viewer."""
    
    def __init__(self, auto_mode: bool = False):
        self.auto_mode = auto_mode
        self.env = None
        self.perception = None
        self.planner = None
        self.viewer = None
        self.step_count = 0
        
        logging.basicConfig(level=logging.WARNING)
    
    def setup_environment(self):
        """Initialize RoboCasa environment with viewer."""
        print_header("1. ENVIRONMENT SETUP")
        
        from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
        
        print_module("ENV", "Creating RoboCasa PnPCounterToCab environment...")
        
        # has_renderer=True for on-screen rendering
        self.env = PnPCounterToCab(
            robots="Panda",
            has_renderer=True,  # Enable on-screen viewer
            has_offscreen_renderer=True,
            ignore_done=True,
            render_camera="robot0_robotview",
            camera_names=["robot0_robotview"],
            camera_heights=480,
            camera_widths=640,
        )
        
        self.obs = self.env.reset()
        self.ep_meta = self.env.get_ep_meta()
        self.instruction = self.ep_meta.get("lang", "pick up object")
        
        print_module("ENV", f"Environment created!")
        print_module("ENV", f"Instruction: {Colors.BOLD}{self.instruction}{Colors.ENDC}")
        print()
        
        # Ground truth info
        print_section("Ground Truth State")
        gt = {
            "object_position": self.obs["obj_pos"].tolist(),
            "robot_ee_position": self.obs["robot0_eef_pos"].tolist(),
        }
        print_json(gt)
        
        # Render initial frame
        self.env.render()
        
        wait_for_user(self.auto_mode, "Environment ready. Press Enter to setup Perception...")
    
    def setup_perception(self):
        """Initialize Perception module."""
        print_header("2. PERCEPTION MODULE")
        
        from robobridge.modules.perception import Perception
        
        print_module("PERCEPTION", "Loading Florence-2 model...")
        print_module("PERCEPTION", "  Provider: hf")
        print_module("PERCEPTION", "  Model: microsoft/Florence-2-base")
        print_module("PERCEPTION", "  Device: cpu")
        
        self.perception = Perception(
            provider="hf",
            model="microsoft/Florence-2-base",
            device="cpu",
        )
        self.perception.initialize_model()
        
        print_module("PERCEPTION", f"{Colors.GREEN}Model loaded!{Colors.ENDC}")
        
        wait_for_user(self.auto_mode, "Perception ready. Press Enter to setup Planner...")
    
    def setup_planner(self):
        """Initialize Planner module."""
        print_header("3. PLANNER MODULE")
        
        from robobridge.modules.planner import Planner
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print_module("PLANNER", f"{Colors.RED}ERROR: OPENAI_API_KEY not set{Colors.ENDC}")
            sys.exit(1)
        
        print_module("PLANNER", "Initializing GPT-4 planner...")
        print_module("PLANNER", "  Provider: openai")
        print_module("PLANNER", "  Model: gpt-4.1")
        
        self.planner = Planner(
            provider="openai",
            model="gpt-4.1",
            api_key=api_key,
        )
        self.planner.initialize_client()
        
        print_module("PLANNER", f"{Colors.GREEN}Planner initialized!{Colors.ENDC}")
        
        wait_for_user(self.auto_mode, "Planner ready. Press Enter to run Perception...")
    
    def run_perception(self) -> List[Dict]:
        """Run perception and display results."""
        print_header("4. RUNNING PERCEPTION")
        
        rgb = self.obs["robot0_robotview_image"]
        gt_pos = self.obs["obj_pos"]
        
        print_module("PERCEPTION", f"Input: RGB image {rgb.shape}")
        print_module("PERCEPTION", f"Ground truth object pos: {gt_pos.tolist()}")
        print_module("PERCEPTION", "Running Florence-2 detection...")
        
        start = time.time()
        detections = self.perception.process(rgb=rgb)
        elapsed = time.time() - start
        
        print_module("PERCEPTION", f"Detection completed in {elapsed:.2f}s")
        print()
        
        print_section("Perception Output")
        if detections:
            for i, det in enumerate(detections):
                print(f"  Detection {i+1}:")
                print(f"    name: {det.name}")
                print(f"    confidence: {det.confidence:.3f}")
                print(f"    bbox: {det.bbox}")
        else:
            print(f"  {Colors.YELLOW}No detections! Using ground truth as fallback.{Colors.ENDC}")
        
        # Build detection list for planner
        if detections:
            detection_list = [
                {
                    "name": d.name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "pose": d.pose,
                }
                for d in detections
            ]
        else:
            detection_list = [{
                "name": "target_object",
                "confidence": 1.0,
                "bbox": [0.4, 0.4, 0.6, 0.6],
                "pose": {
                    "position": {
                        "x": float(gt_pos[0]),
                        "y": float(gt_pos[1]),
                        "z": float(gt_pos[2])
                    }
                }
            }]
            print()
            print_section("Fallback Detection (Ground Truth)")
            print_json(detection_list[0])
        
        wait_for_user(self.auto_mode, "Perception complete. Press Enter to run Planner...")
        return detection_list
    
    def run_planner(self, detections: List[Dict]) -> List[Any]:
        """Run planner and display results."""
        print_header("5. RUNNING PLANNER")
        
        world_state = {
            "detections": detections,
            "robot_ee_pos": self.obs["robot0_eef_pos"].tolist(),
        }
        
        print_module("PLANNER", f"Instruction: {self.instruction}")
        print()
        print_section("Planner Input (World State)")
        print_json(world_state)
        
        print()
        print_module("PLANNER", "Generating action plan with GPT-4...")
        print_module("PLANNER", "  Stage 1: ActionPlanner (instruction → high-level actions)")
        print_module("PLANNER", "  Stage 2: PrimitivePlanner (actions → primitives)")
        
        start = time.time()
        primitive_plans = self.planner.process_full(self.instruction, world_state)
        elapsed = time.time() - start
        
        print_module("PLANNER", f"Planning completed in {elapsed:.2f}s")
        print()
        
        if not primitive_plans:
            print_module("PLANNER", f"{Colors.RED}ERROR: No plan generated!{Colors.ENDC}")
            return []
        
        # Display action plan
        print_section("Stage 1: High-Level Actions (ActionPlanner Output)")
        for i, plan in enumerate(primitive_plans):
            action = plan.parent_action
            print(f"  Action {i+1}: {Colors.BOLD}{action.action_type}{Colors.ENDC} → {action.target_object}")
            if action.target_location:
                print(f"           target_location: {action.target_location}")
        
        print()
        print_section("Stage 2: Primitive Actions (PrimitivePlanner Output)")
        for i, plan in enumerate(primitive_plans):
            action = plan.parent_action
            print(f"\n  {Colors.BOLD}Action {i+1}: {action.action_type} → {action.target_object}{Colors.ENDC}")
            print(f"  Primitives ({len(plan.primitives)}):")
            for j, prim in enumerate(plan.primitives):
                ptype = prim.primitive_type
                if ptype == "grip":
                    width = prim.grip_width
                    state = "OPEN" if width > 0.01 else "CLOSE"
                    print(f"    {j+1}. {ptype}: {state} (width={width:.3f}m)")
                elif ptype == "move":
                    pos = prim.target_position
                    print(f"    {j+1}. {ptype}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
                else:
                    print(f"    {j+1}. {ptype}")
        
        wait_for_user(self.auto_mode, "Plan generated. Press Enter to start execution...")
        return primitive_plans
    
    def execute_primitive(self, primitive: Any, gt_obj_pos: np.ndarray) -> bool:
        """Execute a single primitive with visualization."""
        ptype = primitive.primitive_type
        
        if ptype == "grip":
            return self._execute_grip(primitive)
        elif ptype == "move":
            return self._execute_move(primitive, gt_obj_pos)
        else:
            print_module("CONTROLLER", f"Unknown primitive: {ptype}")
            return True
    
    def _execute_grip(self, primitive) -> bool:
        """Execute gripper action."""
        width = primitive.grip_width if primitive.grip_width is not None else 0.0
        state = "OPEN" if width > 0.01 else "CLOSE"
        gripper_action = (width / 0.08 * 2) - 1  # Map to [-1, 1]
        gripper_action = np.clip(gripper_action, -1.0, 1.0)
        
        print_module("CONTROLLER", f"Gripper {state} (width={width:.3f}m)")
        
        action = np.zeros(7)
        action[6] = gripper_action
        
        for _ in range(30):
            self.obs, _, _, _ = self.env.step(action)
            self.env.render()
            self.step_count += 1
        
        return True
    
    def _execute_move(self, primitive, gt_obj_pos: np.ndarray) -> bool:
        """Execute arm movement."""
        if primitive.target_position is None:
            return True
        
        target = np.array([
            primitive.target_position.x,
            primitive.target_position.y,
            primitive.target_position.z,
        ])
        
        print_module("CONTROLLER", f"Moving to ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        
        for i in range(200):
            current_pos = self.obs["robot0_eef_pos"]
            error = target - current_pos
            dist = np.linalg.norm(error)
            
            if dist < 0.015:
                print_module("CONTROLLER", f"  Reached in {i} steps (dist={dist:.4f})")
                return True
            
            delta = np.clip(error / 0.05, -1.0, 1.0)
            action = np.zeros(7)
            action[0:3] = delta
            
            self.obs, _, _, _ = self.env.step(action)
            self.env.render()
            self.step_count += 1
        
        final_dist = np.linalg.norm(target - self.obs["robot0_eef_pos"])
        print_module("CONTROLLER", f"  Move done (final dist={final_dist:.4f})")
        return True
    
    def run_execution(self, primitive_plans: List[Any]) -> bool:
        """Execute all plans with visualization."""
        print_header("6. EXECUTION")
        
        gt_obj_pos = self.obs["obj_pos"]
        
        for i, plan in enumerate(primitive_plans):
            action = plan.parent_action
            
            print_section(f"Executing Action {i+1}: {action.action_type} → {action.target_object}")
            
            wait_for_user(self.auto_mode, f"Press Enter to execute {action.action_type}...")
            
            for j, prim in enumerate(plan.primitives):
                ptype = prim.primitive_type
                print(f"\n  {Colors.BOLD}Primitive {j+1}/{len(plan.primitives)}: {ptype}{Colors.ENDC}")
                
                self.execute_primitive(prim, gt_obj_pos)
            
            print_module("CONTROLLER", f"{Colors.GREEN}Action '{action.action_type}' complete{Colors.ENDC}")
        
        return True
    
    def run(self):
        """Run the full visual test."""
        try:
            print_header("ROBOBRIDGE VISUAL PIPELINE TEST")
            print(f"Mode: {'Automatic' if self.auto_mode else 'Step-by-step (interactive)'}")
            
            self.setup_environment()
            self.setup_perception()
            self.setup_planner()
            
            detections = self.run_perception()
            plans = self.run_planner(detections)
            
            if not plans:
                return False
            
            self.run_execution(plans)
            
            print_header("TEST COMPLETE")
            print(f"Total simulation steps: {self.step_count}")
            print()
            print("Press Enter to close...")
            input()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            return False
        finally:
            if self.env:
                self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Visual Pipeline Test")
    parser.add_argument("--auto", action="store_true", help="Auto mode (no user input)")
    args = parser.parse_args()
    
    test = VisualPipelineTest(auto_mode=args.auto)
    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
