#!/usr/bin/env python3
"""
RoboBridge Pipeline Debug CLI

Shows input/output of each module step-by-step with success/failure status.

Usage:
    python scripts/debug_pipeline.py --instruction "pick up the red cup"
    python scripts/debug_pipeline.py --dataset /path/to/demo.hdf5 --demo 0
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


@dataclass
class StepLog:
    step_id: int
    module: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    duration_ms: float
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class PipelineLog:
    instruction: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    steps: List[StepLog] = field(default_factory=list)
    final_result: Optional[str] = None


class PipelineDebugger:
    def __init__(self, output_dir: str = "/tmp/robobridge_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log = None
        self.step_counter = 0
        
    def start_session(self, instruction: str):
        self.log = PipelineLog(
            instruction=instruction,
            start_time=datetime.now().isoformat()
        )
        self.step_counter = 0
        self._print_header(f"RoboBridge Pipeline Debug")
        self._print_info(f"Instruction: {instruction}")
        self._print_info(f"Output dir: {self.output_dir}")
        print()
        
    def log_step(self, module: str, action: str, input_data: Dict, 
                 output_data: Dict, success: bool, duration_ms: float,
                 error: Optional[str] = None):
        self.step_counter += 1
        step = StepLog(
            step_id=self.step_counter,
            module=module,
            action=action,
            input_data=self._serialize(input_data),
            output_data=self._serialize(output_data),
            success=success,
            duration_ms=duration_ms,
            error=error
        )
        self.log.steps.append(step)
        self.log.total_steps += 1
        if success:
            self.log.successful_steps += 1
        else:
            self.log.failed_steps += 1
            
        self._print_step(step)
        
    def end_session(self, final_result: str):
        self.log.end_time = datetime.now().isoformat()
        self.log.final_result = final_result
        
        log_path = os.path.join(
            self.output_dir, 
            f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_path, 'w') as f:
            json.dump(asdict(self.log), f, indent=2, default=str)
            
        print()
        self._print_header("Pipeline Summary")
        self._print_info(f"Total steps: {self.log.total_steps}")
        self._print_success(f"Successful: {self.log.successful_steps}")
        if self.log.failed_steps > 0:
            self._print_error(f"Failed: {self.log.failed_steps}")
        self._print_info(f"Result: {final_result}")
        self._print_info(f"Log saved: {log_path}")
        
    def _serialize(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._serialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize(v) for v in data]
        elif hasattr(data, 'to_dict'):
            return data.to_dict()
        elif hasattr(data, '__dict__'):
            return {k: self._serialize(v) for k, v in data.__dict__.items() 
                    if not k.startswith('_')}
        return data
        
    def _print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
    def _print_module(self, module: str, action: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[{module}] {action}{Colors.ENDC}")
        print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")
        
    def _print_input(self, key: str, value: Any):
        val_str = self._format_value(value)
        print(f"  {Colors.BLUE}→ {key}:{Colors.ENDC} {val_str}")
        
    def _print_output(self, key: str, value: Any):
        val_str = self._format_value(value)
        print(f"  {Colors.YELLOW}← {key}:{Colors.ENDC} {val_str}")
        
    def _print_success(self, text: str):
        print(f"  {Colors.GREEN}✓ {text}{Colors.ENDC}")
        
    def _print_error(self, text: str):
        print(f"  {Colors.RED}✗ {text}{Colors.ENDC}")
        
    def _print_info(self, text: str):
        print(f"  {Colors.DIM}{text}{Colors.ENDC}")
        
    def _format_value(self, value: Any, max_len: int = 80) -> str:
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 5:
                return f"[{len(value)} items]"
            return str(value)
        elif isinstance(value, dict):
            if len(value) > 3:
                return f"{{{len(value)} keys}}"
            return str(value)
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s
        
    def _print_step(self, step: StepLog):
        self._print_module(step.module, step.action)
        
        print(f"  {Colors.DIM}Input:{Colors.ENDC}")
        for k, v in step.input_data.items():
            self._print_input(k, v)
            
        print(f"  {Colors.DIM}Output:{Colors.ENDC}")
        for k, v in step.output_data.items():
            self._print_output(k, v)
            
        if step.success:
            self._print_success(f"SUCCESS ({step.duration_ms:.1f}ms)")
        else:
            self._print_error(f"FAILED: {step.error}")


class RoboCasaPipelineRunner:
    def __init__(self, debugger: PipelineDebugger, dataset_path: str = None, 
                 demo_index: int = 0, fast_mode: bool = False):
        self.debugger = debugger
        self.dataset_path = dataset_path
        self.demo_index = demo_index
        self.fast_mode = fast_mode
        self.env = None
        self.obs = None
        self.ep_meta = None
        
    def setup_environment(self):
        import h5py
        import robosuite
        from robocasa.scripts.playback_dataset import reset_to, get_env_metadata_from_dataset
        
        start = time.time()
        
        env_meta = get_env_metadata_from_dataset(self.dataset_path)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["has_offscreen_renderer"] = not self.fast_mode
        env_kwargs["use_camera_obs"] = not self.fast_mode
        env_kwargs["ignore_done"] = True
        
        self.env = robosuite.make(**env_kwargs)
        
        f = h5py.File(self.dataset_path, 'r')
        demo_names = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))
        demo_name = demo_names[self.demo_index]
        demo = f[f'data/{demo_name}']
        
        initial_state = {
            'states': demo['states'][0],
            'model': demo.attrs['model_file'],
            'ep_meta': demo.attrs.get('ep_meta', None),
        }
        
        self.ep_meta = json.loads(initial_state['ep_meta']) if initial_state['ep_meta'] else {}
        reset_to(self.env, initial_state)
        self.obs = self.env._get_observations()
        f.close()
        
        duration = (time.time() - start) * 1000
        
        self.debugger.log_step(
            module="Environment",
            action="Setup",
            input_data={
                "dataset": self.dataset_path,
                "demo_index": self.demo_index,
                "robot": env_kwargs.get("robots", "unknown"),
            },
            output_data={
                "task": self.ep_meta.get("lang", "N/A"),
                "action_dim": self.env.action_spec[0].shape[0],
            },
            success=True,
            duration_ms=duration
        )
        
        return self.ep_meta.get("lang", "pick up the object")
        
    def run_perception(self) -> List[Dict]:
        start = time.time()
        
        detections = []
        object_cfgs = self.ep_meta.get("object_cfgs", [])
        
        for obj_name, obj in self.env.objects.items():
            try:
                body_id = self.env.sim.model.body_name2id(obj.root_body)
                pos = self.env.sim.data.body_xpos[body_id].copy()
                
                cfg = next((c for c in object_cfgs if c.get("name") == obj_name), {})
                category = cfg.get("info", {}).get("cat", obj_name)
                is_target = (obj_name == "obj")
                
                detections.append({
                    "name": category,
                    "obj_id": obj_name,
                    "position": pos.tolist(),
                    "is_target": is_target,
                    "confidence": 1.0,
                })
            except:
                pass
                
        duration = (time.time() - start) * 1000
        
        target = next((d for d in detections if d["is_target"]), None)
        
        self.debugger.log_step(
            module="Perception",
            action="Detect Objects (Ground Truth)",
            input_data={
                "source": "MuJoCo simulation",
                "num_objects_in_scene": len(self.env.objects),
            },
            output_data={
                "num_detections": len(detections),
                "target_object": target["name"] if target else "None",
                "target_position": target["position"] if target else None,
            },
            success=len(detections) > 0,
            duration_ms=duration
        )
        
        return detections
        
    def run_planner(self, instruction: str, detections: List[Dict]) -> List[Any]:
        from robobridge.modules.planner import Planner
        
        start = time.time()
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.debugger.log_step(
                module="Planner",
                action="Generate Plan",
                input_data={"instruction": instruction},
                output_data={},
                success=False,
                duration_ms=0,
                error="OPENAI_API_KEY not set"
            )
            return []
            
        planner = Planner(provider="openai", model="gpt-4.1", api_key=api_key)
        planner.initialize_client()
        
        world_state = {
            "detections": detections,
            "robot_ee_pos": self.obs["robot0_eef_pos"].tolist(),
        }
        
        primitive_plans = planner.process_full(instruction, world_state)
        
        duration = (time.time() - start) * 1000
        
        plan_summary = []
        for plan in primitive_plans:
            action = plan.parent_action
            primitives = [p.to_dict()["primitive_type"] for p in plan.primitives]
            plan_summary.append({
                "action": f"{action.action_type} -> {action.target_object}",
                "primitives": primitives,
            })
            
        self.debugger.log_step(
            module="Planner",
            action="Generate Primitive Plan",
            input_data={
                "instruction": instruction,
                "num_objects": len(detections),
                "robot_position": world_state["robot_ee_pos"],
            },
            output_data={
                "num_plans": len(primitive_plans),
                "plans": plan_summary,
            },
            success=len(primitive_plans) > 0,
            duration_ms=duration
        )
        
        return primitive_plans
        
    def run_execution(self, primitive_plans: List[Any]) -> bool:
        from scipy.spatial.transform import Rotation as R
        
        robot = self.env.robots[0]
        base_body_id = self.env.sim.model.body_name2id(robot.robot_model.root_body)
        robot_base_pos = self.env.sim.data.body_xpos[base_body_id].copy()
        base_quat = self.env.sim.data.body_xquat[base_body_id]
        quat_xyzw = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
        robot_base_rot = R.from_quat(quat_xyzw)
        
        def world_to_robot(vec):
            return robot_base_rot.inv().apply(vec)
            
        action_dim = self.env.action_spec[0].shape[0]
        overall_success = True
        
        for plan_idx, plan in enumerate(primitive_plans):
            parent_action = plan.parent_action
            
            for prim_idx, prim in enumerate(plan.primitives):
                prim_dict = prim.to_dict()
                ptype = prim_dict["primitive_type"]
                
                start = time.time()
                success = True
                error = None
                output_data = {}
                
                if ptype == "move":
                    target_pos = prim_dict.get("target_position", {})
                    target = np.array([
                        target_pos.get("x", 0),
                        target_pos.get("y", 0),
                        target_pos.get("z", 0),
                    ])
                    
                    initial_pos = self.obs["robot0_eef_pos"].copy()
                    
                    for step in range(200):
                        current = self.obs["robot0_eef_pos"]
                        error_vec = target - current
                        dist = np.linalg.norm(error_vec)
                        
                        if dist < 0.02:
                            break
                            
                        direction = error_vec / dist
                        delta = world_to_robot(direction * 0.3)
                        
                        action = np.zeros(action_dim)
                        action[0:3] = delta
                        action[6] = -1.0
                        
                        self.obs, _, _, _ = self.env.step(action)
                        
                    final_pos = self.obs["robot0_eef_pos"]
                    final_dist = np.linalg.norm(target - final_pos)
                    success = final_dist < 0.03
                    
                    output_data = {
                        "initial_pos": initial_pos.tolist(),
                        "final_pos": final_pos.tolist(),
                        "target_pos": target.tolist(),
                        "final_distance": f"{final_dist:.4f}m",
                        "steps": step + 1,
                    }
                    if not success:
                        error = f"Did not reach target (dist={final_dist:.3f}m)"
                        
                elif ptype == "grip":
                    grip_width = prim_dict.get("grip_width", 0.5)
                    is_closing = grip_width < 0.01
                    gripper_action = 1.0 if is_closing else -1.0
                    
                    initial_gripper = self.obs["robot0_gripper_qpos"].copy()
                    
                    for _ in range(50):
                        action = np.zeros(action_dim)
                        action[6] = gripper_action
                        self.obs, _, _, _ = self.env.step(action)
                        
                    final_gripper = self.obs["robot0_gripper_qpos"]
                    gripper_opening = abs(final_gripper[0]) + abs(final_gripper[1])
                    
                    if is_closing:
                        grasped = gripper_opening > 0.005
                        success = True
                        output_data = {
                            "action": "close",
                            "gripper_opening": f"{gripper_opening:.4f}",
                            "object_grasped": grasped,
                        }
                    else:
                        success = True
                        output_data = {
                            "action": "open",
                            "gripper_opening": f"{gripper_opening:.4f}",
                        }
                else:
                    output_data = {"skipped": True}
                    
                duration = (time.time() - start) * 1000
                
                self.debugger.log_step(
                    module="Controller/Robot",
                    action=f"{parent_action.action_type}.{ptype}",
                    input_data={
                        "plan": f"{plan_idx+1}/{len(primitive_plans)}",
                        "primitive": f"{prim_idx+1}/{len(plan.primitives)}",
                        "type": ptype,
                        "params": {k: v for k, v in prim_dict.items() 
                                   if k not in ["primitive_type"]},
                    },
                    output_data=output_data,
                    success=success,
                    duration_ms=duration,
                    error=error if not success else None
                )
                
                if not success:
                    overall_success = False
                    
        return overall_success
        
    def cleanup(self):
        if self.env:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="RoboBridge Pipeline Debug CLI")
    parser.add_argument("--instruction", "-i", type=str, 
                        help="Natural language instruction")
    parser.add_argument("--dataset", "-d", type=str,
                        default="./data/demo_gentex_im128_randcams.hdf5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0,
                        help="Demo index in dataset")
    parser.add_argument("--output-dir", "-o", type=str, 
                        default="/tmp/robobridge_debug",
                        help="Output directory for logs")
    parser.add_argument("--fast", "-f", action="store_true",
                        help="Fast mode: disable camera rendering")
    args = parser.parse_args()
    
    debugger = PipelineDebugger(output_dir=args.output_dir)
    runner = RoboCasaPipelineRunner(
        debugger=debugger,
        dataset_path=args.dataset,
        demo_index=args.demo,
        fast_mode=args.fast
    )
    
    try:
        debugger.start_session(args.instruction or "Loading from dataset...")
        
        instruction = runner.setup_environment()
        if args.instruction:
            instruction = args.instruction
        else:
            debugger.log.instruction = instruction
        
        detections = runner.run_perception()
        
        if not detections:
            debugger.end_session("FAILED - No objects detected")
            return 1
            
        plans = runner.run_planner(instruction, detections)
        
        if not plans:
            debugger.end_session("FAILED - No plan generated")
            return 1
            
        success = runner.run_execution(plans)
        
        debugger.end_session("SUCCESS" if success else "PARTIAL - Some actions failed")
        return 0 if success else 1
        
    except Exception as e:
        import traceback
        debugger.end_session(f"ERROR - {str(e)}")
        traceback.print_exc()
        return 1
        
    finally:
        runner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
