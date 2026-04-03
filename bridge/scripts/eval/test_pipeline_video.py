#!/usr/bin/env python3
"""
RoboBridge Pipeline Test with Video Recording

시뮬레이션을 비디오로 녹화 + 각 모듈 출력 표시
- 실행 과정을 MP4 비디오로 저장
- 각 모듈(Perception, Planner, Controller) 출력을 터미널에 표시

Usage:
    python scripts/test_pipeline_video.py
    python scripts/test_pipeline_video.py --output /path/to/video.mp4
"""

import argparse
import logging
import os
import sys
import time
import json
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Colors for terminal output
class C:
    H = '\033[95m'  # Header
    B = '\033[94m'  # Blue
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    E = '\033[0m'   # End
    BOLD = '\033[1m'

def header(t): print(f"\n{C.H}{C.BOLD}{'='*60}\n  {t}\n{'='*60}{C.E}\n")
def section(t): print(f"\n{C.B}{C.BOLD}--- {t} ---{C.E}")
def env(t): print(f"{C.B}[ENV]{C.E} {t}")
def perc(t): print(f"{C.G}[PERCEPTION]{C.E} {t}")
def plan(t): print(f"{C.Y}[PLANNER]{C.E} {t}")
def ctrl(t): print(f"{C.R}[CONTROLLER]{C.E} {t}")


class VideoRecorder:
    """Simple video recorder using imageio."""
    
    def __init__(self, output_path: str, fps: int = 30):
        self.output_path = output_path
        self.fps = fps
        self.frames = []
    
    def add_frame(self, rgb: np.ndarray):
        self.frames.append(rgb.copy())
    
    def save(self):
        if not self.frames:
            return
        
        try:
            import imageio
            imageio.mimsave(self.output_path, self.frames, fps=self.fps)
            print(f"\n{C.G}Video saved: {self.output_path}{C.E}")
            print(f"  Frames: {len(self.frames)}, FPS: {self.fps}")
        except ImportError:
            # Fallback: save frames as images
            output_dir = self.output_path.replace('.mp4', '_frames')
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(self.frames):
                Image.fromarray(frame).save(f"{output_dir}/frame_{i:04d}.png")
            print(f"\n{C.Y}imageio not available. Frames saved to: {output_dir}{C.E}")


class PipelineTest:
    """Pipeline test with video recording."""
    
    def __init__(self, output_path: str):
        self.recorder = VideoRecorder(output_path)
        self.env = None
        self.perception = None
        self.planner = None
        self.step_count = 0
        logging.basicConfig(level=logging.WARNING)
    
    def setup(self):
        """Setup all modules."""
        header("SETUP")
        
        # Environment
        from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
        
        env("Creating RoboCasa environment...")
        self.env = PnPCounterToCab(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            render_camera="robot0_robotview",
            camera_names=["robot0_robotview"],
            camera_heights=480,
            camera_widths=640,
        )
        self.obs = self.env.reset()
        self.instruction = self.env.get_ep_meta().get("lang", "pick up object")
        env(f"Task: {C.BOLD}{self.instruction}{C.E}")
        env(f"Object pos: {self.obs['obj_pos'].tolist()}")
        
        # Perception
        from robobridge.modules.perception import Perception
        perc("Loading Florence-2 (cpu)...")
        self.perception = Perception(provider="hf", model="microsoft/Florence-2-base", device="cpu")
        self.perception.initialize_model()
        perc("Ready")
        
        # Planner
        from robobridge.modules.planner import Planner
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        plan("Initializing GPT-4.1...")
        self.planner = Planner(provider="openai", model="gpt-4.1", api_key=api_key)
        self.planner.initialize_client()
        plan("Ready")
        
        self.record_frame()
    
    def record_frame(self):
        """Record current frame."""
        self.recorder.add_frame(self.obs["robot0_robotview_image"])
    
    def run_perception(self) -> List[Dict]:
        """Run perception."""
        header("PERCEPTION")
        
        rgb = self.obs["robot0_robotview_image"]
        gt_pos = self.obs["obj_pos"]
        
        perc(f"Input: RGB {rgb.shape}")
        perc("Running Florence-2...")
        
        start = time.time()
        detections = self.perception.process(rgb=rgb)
        elapsed = time.time() - start
        
        perc(f"Done in {elapsed:.2f}s, detections: {len(detections)}")
        
        if detections:
            for i, d in enumerate(detections):
                perc(f"  {i+1}. {d.name} (conf={d.confidence:.2f})")
            return [{"name": d.name, "confidence": d.confidence, "bbox": d.bbox, "pose": d.pose} for d in detections]
        else:
            perc(f"{C.Y}No detections, using GT{C.E}")
            return [{"name": "target_object", "confidence": 1.0, "bbox": [0.4,0.4,0.6,0.6],
                    "pose": {"position": {"x": float(gt_pos[0]), "y": float(gt_pos[1]), "z": float(gt_pos[2])}}}]
    
    def run_planner(self, detections: List[Dict]) -> List[Any]:
        """Run planner."""
        header("PLANNER")
        
        world_state = {"detections": detections, "robot_ee_pos": self.obs["robot0_eef_pos"].tolist()}
        
        plan(f"Instruction: {self.instruction}")
        plan("Stage 1: ActionPlanner (instruction → actions)")
        plan("Stage 2: PrimitivePlanner (actions → primitives)")
        plan("Calling GPT-4...")
        
        start = time.time()
        plans = self.planner.process_full(self.instruction, world_state)
        elapsed = time.time() - start
        
        plan(f"Done in {elapsed:.2f}s")
        
        if not plans:
            plan(f"{C.R}ERROR: No plan generated{C.E}")
            return []
        
        section("High-Level Actions")
        for i, p in enumerate(plans):
            a = p.parent_action
            print(f"  {i+1}. {C.BOLD}{a.action_type}{C.E} → {a.target_object}")
        
        section("Primitive Plans")
        for i, p in enumerate(plans):
            a = p.parent_action
            print(f"\n  {C.BOLD}[{a.action_type} → {a.target_object}]{C.E}")
            for j, prim in enumerate(p.primitives):
                if prim.primitive_type == "grip":
                    w = prim.grip_width if prim.grip_width else 0
                    state = "OPEN" if w > 0.01 else "CLOSE"
                    print(f"    {j+1}. grip: {state} (w={w:.3f})")
                elif prim.primitive_type == "move":
                    pos = prim.target_position
                    print(f"    {j+1}. move: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        return plans
    
    def execute(self, plans: List[Any]):
        """Execute plans."""
        header("EXECUTION")
        
        gt_pos = self.obs["obj_pos"]
        
        for i, plan in enumerate(plans):
            action = plan.parent_action
            section(f"Action {i+1}: {action.action_type} → {action.target_object}")
            
            for j, prim in enumerate(plan.primitives):
                ptype = prim.primitive_type
                
                if ptype == "grip":
                    self._exec_grip(prim)
                elif ptype == "move":
                    self._exec_move(prim, gt_pos)
                
                self.record_frame()
            
            ctrl(f"{C.G}'{action.action_type}' done{C.E}")
    
    def _exec_grip(self, prim):
        w = prim.grip_width if prim.grip_width is not None else 0.0
        state = "OPEN" if w > 0.01 else "CLOSE"
        ctrl(f"Gripper {state}")
        
        action = np.zeros(7)
        action[6] = (w / 0.08 * 2) - 1
        
        for _ in range(30):
            self.obs, _, _, _ = self.env.step(action)
            self.step_count += 1
            if self.step_count % 10 == 0:
                self.record_frame()
    
    def _exec_move(self, prim, gt_pos):
        if not prim.target_position:
            return
        
        target = np.array([prim.target_position.x, prim.target_position.y, prim.target_position.z])
        ctrl(f"Move to ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        
        for i in range(200):
            cur = self.obs["robot0_eef_pos"]
            err = target - cur
            dist = np.linalg.norm(err)
            
            if dist < 0.015:
                ctrl(f"  Reached in {i} steps")
                return
            
            action = np.zeros(7)
            action[0:3] = np.clip(err / 0.05, -1, 1)
            
            self.obs, _, _, _ = self.env.step(action)
            self.step_count += 1
            
            if self.step_count % 5 == 0:
                self.record_frame()
        
        ctrl(f"  Move done (dist={np.linalg.norm(target - self.obs['robot0_eef_pos']):.4f})")
    
    def run(self):
        """Run full test."""
        try:
            header("ROBOBRIDGE PIPELINE TEST (VIDEO)")
            
            self.setup()
            detections = self.run_perception()
            plans = self.run_planner(detections)
            
            if plans:
                self.execute(plans)
            
            header("COMPLETE")
            print(f"Total steps: {self.step_count}")
            
            self.recorder.save()
            return True
            
        except KeyboardInterrupt:
            print("\nInterrupted")
            self.recorder.save()
            return False
        finally:
            if self.env:
                self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="/tmp/robobridge_pipeline.mp4", help="Output video path")
    args = parser.parse_args()
    
    test = PipelineTest(args.output)
    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
