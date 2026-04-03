#!/usr/bin/env python3
"""
Simple RoboCasa Pick Up Test - Direct Control.

Tests pick up action using direct robot control without LLM planning.
This validates the low-level control before integrating with the full pipeline.

Usage:
    python scripts/test_robocasa_simple.py
"""

import logging
import os
import sys
import time

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_environment():
    """Create RoboCasa environment."""
    from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
    
    logger.info("Creating environment...")
    env = PnPCounterToCab(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        render_camera="robot0_robotview",
        camera_names=["robot0_robotview", "robot0_eye_in_hand"],
        camera_heights=480,
        camera_widths=640,
    )
    return env


def save_image(obs, name, output_dir="/tmp/robocasa_simple"):
    """Save camera image."""
    os.makedirs(output_dir, exist_ok=True)
    img = obs["robot0_robotview_image"]
    path = os.path.join(output_dir, f"{name}.png")
    Image.fromarray(img).save(path)
    logger.info(f"Saved: {path}")


def move_to_position(env, obs, target_pos, steps=200, grip_action=0.0):
    """Move end-effector to target position using delta control.
    
    RoboCasa uses OSC_POSE with delta input type.
    Action space: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
    Output is scaled by output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
    So action=1.0 means delta of 0.05m per step.
    """
    for i in range(steps):
        current_pos = obs["robot0_eef_pos"]
        error = target_pos - current_pos
        dist = np.linalg.norm(error)
        
        if dist < 0.015:
            logger.info(f"  Reached target in {i} steps (dist={dist:.4f})")
            return obs
        
        # Delta control: normalize error and scale appropriately
        # Max delta is 0.05m per step, so we scale error to [-1, 1] range
        # Using a gain that gives reasonable speed while maintaining stability
        delta = error / 0.05  # Scale to action range
        delta = np.clip(delta, -1.0, 1.0)
        
        action = np.zeros(7)
        action[0:3] = delta  # Position delta
        action[3:6] = 0.0    # Keep orientation fixed
        action[6] = grip_action
        
        obs, reward, done, info = env.step(action)
        
        if i % 30 == 0:
            logger.info(f"  Step {i}: dist={dist:.4f}, pos={current_pos}")
    
    final_dist = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
    logger.info(f"  Move completed (final dist={final_dist:.4f})")
    return obs


def set_gripper(env, obs, open_gripper=True, steps=30):
    """Open or close gripper."""
    grip_action = 1.0 if open_gripper else -1.0
    action = np.zeros(7)
    action[6] = grip_action
    
    for _ in range(steps):
        obs, reward, done, info = env.step(action)
    
    state = "open" if open_gripper else "closed"
    logger.info(f"  Gripper {state}")
    return obs


def run_pick_and_place(env, obs):
    """Execute pick and place sequence."""
    # Get positions
    obj_pos = obs["obj_pos"]
    ee_pos = obs["robot0_eef_pos"]
    
    logger.info(f"Object position: {obj_pos}")
    logger.info(f"EE position: {ee_pos}")
    
    # Define waypoints relative to object
    approach_height = 0.15
    grasp_height = 0.03
    lift_height = 0.20
    place_offset = np.array([0.2, 0.0, 0.0])  # Place 20cm to the side
    
    above_obj = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])
    grasp_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + grasp_height])
    lift_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + lift_height])
    place_pos = np.array([obj_pos[0] + place_offset[0], obj_pos[1], obj_pos[2] + approach_height])
    
    # Execute pick sequence
    logger.info("\n=== PICK SEQUENCE ===")
    
    logger.info("1. Open gripper")
    obs = set_gripper(env, obs, open_gripper=True)
    save_image(obs, "01_gripper_open")
    
    logger.info(f"2. Move above object: {above_obj}")
    obs = move_to_position(env, obs, above_obj, steps=150, grip_action=1.0)
    save_image(obs, "02_above_object")
    
    logger.info(f"3. Move to grasp position: {grasp_pos}")
    obs = move_to_position(env, obs, grasp_pos, steps=100, grip_action=1.0)
    save_image(obs, "03_grasp_position")
    
    logger.info("4. Close gripper")
    obs = set_gripper(env, obs, open_gripper=False)
    save_image(obs, "04_gripper_closed")
    
    logger.info(f"5. Lift object: {lift_pos}")
    obs = move_to_position(env, obs, lift_pos, steps=100, grip_action=-1.0)
    save_image(obs, "05_lift")
    
    # Execute place sequence
    logger.info("\n=== PLACE SEQUENCE ===")
    
    logger.info(f"6. Move to place position: {place_pos}")
    obs = move_to_position(env, obs, place_pos, steps=150, grip_action=-1.0)
    save_image(obs, "06_place_position")
    
    logger.info("7. Open gripper (release)")
    obs = set_gripper(env, obs, open_gripper=True)
    save_image(obs, "07_release")
    
    # Retreat
    retreat_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.1])
    logger.info(f"8. Retreat: {retreat_pos}")
    obs = move_to_position(env, obs, retreat_pos, steps=50, grip_action=1.0)
    save_image(obs, "08_retreat")
    
    return obs


def main():
    env = None
    try:
        env = create_environment()
        obs = env.reset()
        
        ep_meta = env.get_ep_meta()
        task = ep_meta.get("lang", "unknown task")
        logger.info(f"\nTask: {task}")
        logger.info(f"Action dim: {env.action_spec[0].shape[0]}")
        
        save_image(obs, "00_initial")
        
        # Run pick and place
        final_obs = run_pick_and_place(env, obs)
        
        save_image(final_obs, "99_final")
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed!")
        logger.info("Images saved to: /tmp/robocasa_simple")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1
    finally:
        if env is not None:
            env.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
