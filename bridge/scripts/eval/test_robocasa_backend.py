#!/usr/bin/env python3
"""
Test RoboCasaBackend's new move_to_position() with constant velocity control.

Usage:
    python scripts/test_robocasa_backend.py
    python scripts/test_robocasa_backend.py --demo 0 --verbose
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def get_world_object_position(env, obj_name: str = "obj") -> np.ndarray:
    obj = env.objects.get(obj_name)
    if obj is None:
        raise ValueError(f"Object {obj_name} not found")
    body_id = env.sim.model.body_name2id(obj.root_body)
    return env.sim.data.body_xpos[body_id].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d",
                        default="/home/franka/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5")
    parser.add_argument("--demo", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RoboCasaBackend Constant Velocity Control Test")
    print("=" * 60)

    from robobridge.modules.robot.backends.robocasa import RoboCasaBackend, RoboCasaConfig
    from robocasa.scripts.playback_dataset import reset_to, get_env_metadata_from_dataset
    import robosuite

    print(f"\nLoading dataset: {args.dataset}")
    print(f"Demo index: {args.demo}")

    env_meta = get_env_metadata_from_dataset(args.dataset)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["ignore_done"] = True
    env_kwargs["camera_names"] = ["robot0_robotview", "robot0_eye_in_hand"]
    env_kwargs["camera_heights"] = 256
    env_kwargs["camera_widths"] = 256

    print(f"Creating environment: {env_meta['env_name']}")
    print(f"Robot: {env_kwargs.get('robots', 'unknown')}")
    env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, 'r')
    demo_names = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))
    demo_name = demo_names[args.demo]
    print(f"Loading demo: {demo_name}")

    demo = f[f'data/{demo_name}']
    states = demo['states'][()]
    initial_state = {
        'states': states[0],
        'model': demo.attrs['model_file'],
        'ep_meta': demo.attrs.get('ep_meta', None),
    }

    ep_meta = json.loads(initial_state['ep_meta']) if initial_state['ep_meta'] else {}
    print(f"Task: {ep_meta.get('lang', 'N/A')}")

    reset_to(env, initial_state)
    obs = env._get_observations()
    f.close()

    config = RoboCasaConfig(
        move_speed=0.3,
        position_threshold_m=0.015,
        max_steps_per_move=400,
        gripper_action_steps=50,
        gripper_action_idx=6,
    )

    backend = RoboCasaBackend(config=config)
    backend._env = env
    backend._last_obs = obs
    backend._setup_robot_frame()

    obj_pos = get_world_object_position(env, "obj")
    eef_pos = obs["robot0_eef_pos"]

    print(f"\n--- Initial State ---")
    print(f"Object position (world): {obj_pos}")
    print(f"EEF position (world): {eef_pos}")
    print(f"Distance to object: {np.linalg.norm(eef_pos - obj_pos):.4f}m")

    directly_above = obj_pos.copy()
    directly_above[2] += 0.05
    print(f"\n--- Test 1: Move DIRECTLY ABOVE object (5cm up) ---")
    print(f"Target: {directly_above}")

    success = backend.move_to_position(directly_above, verbose=args.verbose)
    eef_pos = backend._last_obs["robot0_eef_pos"]
    dist = np.linalg.norm(eef_pos - directly_above)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Final EEF: {eef_pos}")
    print(f"Final distance: {dist:.4f}m")

    print(f"\n--- Test 2: Descend straight down to grasp ---")
    grasp_pos = obj_pos.copy()
    print(f"Target: {grasp_pos}")
    print(f"From directly above, descending vertically to object center")

    success = backend.move_to_position(grasp_pos, verbose=args.verbose)
    eef_pos = backend._last_obs["robot0_eef_pos"]
    dist = np.linalg.norm(eef_pos - grasp_pos)
    z_error = eef_pos[2] - grasp_pos[2]
    xy_error = np.linalg.norm(eef_pos[:2] - grasp_pos[:2])
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Final EEF: {eef_pos}")
    print(f"Final distance: {dist:.4f}m, XY error: {xy_error:.4f}m, Z error: {z_error:.4f}m")

    if abs(z_error) < 0.02:
        print("\n>>> SUCCESS! Z target reached!")
    else:
        print(f"\n>>> Still failing: Z error = {z_error:.4f}m")

    print(f"\n--- Test 3: Close gripper ---")
    success = backend.set_gripper(open_gripper=False, verbose=args.verbose)
    gripper_state = backend._last_obs["robot0_gripper_qpos"]
    gripper_opening = abs(gripper_state[0]) + abs(gripper_state[1])
    print(f"Gripper state: {gripper_state}")
    print(f"Gripper opening: {gripper_opening:.4f}")

    if gripper_opening > 0.01:
        print(">>> GRASP DETECTED (fingers stopped with object)")
    else:
        print(">>> NO GRASP (fingers fully closed)")

    lift_pos = eef_pos.copy()
    lift_pos[2] += 0.10
    print(f"\n--- Test 4: Lift object ---")
    print(f"Target: {lift_pos}")

    success = backend.move_to_position(lift_pos, verbose=args.verbose)
    eef_pos = backend._last_obs["robot0_eef_pos"]
    obj_pos_after = get_world_object_position(env, "obj")
    print(f"Final EEF: {eef_pos}")
    print(f"Object position after lift: {obj_pos_after}")
    print(f"Object Z change: {obj_pos_after[2] - obj_pos[2]:.4f}m")

    if obj_pos_after[2] - obj_pos[2] > 0.05:
        print("\n>>> SUCCESS! Object was lifted!")
    else:
        print("\n>>> FAILED: Object not lifted")

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
