#!/usr/bin/env python3
"""Verify that env.fxtr-based fixture naming produces correct target detections.

For each RoboCasa task:
1. Create env → reset
2. Inspect env.fxtr (or task-specific attr like sink, stove, etc.)
3. Run perception.process()
4. Check target detection has correct nat_lang-based name
5. Check target position is within arm reach (1.5m)
6. Print object list that planner would see

Usage:
    MUJOCO_GL=egl python3 robobridge/scripts/verify_fixture_targets.py --all-tasks
    MUJOCO_GL=egl python3 robobridge/scripts/verify_fixture_targets.py --tasks CloseDrawer CloseSingleDoor
"""

import argparse
import logging
import sys
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All 24 RoboCasa single-stage tasks
ALL_TASKS = [
    "CloseDrawer", "OpenDrawer",
    "CloseSingleDoor", "CloseDoubleDoor",
    "OpenSingleDoor", "OpenDoubleDoor",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
    "TurnOnStove", "TurnOffStove",
    "TurnOnMicrowave", "TurnOffMicrowave",
    "CoffeePressButton", "CoffeeServeMug", "CoffeeSetupMug",
    "PnPCabToCounter", "PnPCounterToCab",
    "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToStove", "PnPStoveToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter",
]

# Task-specific target fixture attributes (order matters: first match wins)
TARGET_ATTRS = {
    "CloseDrawer": ["fxtr"],
    "OpenDrawer": ["fxtr"],
    "CloseSingleDoor": ["fxtr"],
    "CloseDoubleDoor": ["fxtr"],
    "OpenSingleDoor": ["fxtr"],
    "OpenDoubleDoor": ["fxtr"],
    "TurnOnSinkFaucet": ["sink"],
    "TurnOffSinkFaucet": ["sink"],
    "TurnSinkSpout": ["sink"],
    "TurnOnStove": ["stove"],
    "TurnOffStove": ["stove"],
    "TurnOnMicrowave": ["microwave"],
    "TurnOffMicrowave": ["microwave"],
    "CoffeePressButton": ["coffee_machine"],
    "CoffeeServeMug": ["coffee_machine"],
    "CoffeeSetupMug": ["coffee_machine"],
    "PnPCabToCounter": ["fxtr"],
    "PnPCounterToCab": ["fxtr"],
    "PnPCounterToSink": ["sink"],
    "PnPSinkToCounter": ["sink"],
    "PnPCounterToStove": ["stove"],
    "PnPStoveToCounter": ["stove"],
    "PnPCounterToMicrowave": ["microwave"],
    "PnPMicrowaveToCounter": ["microwave"],
}


def verify_task(task_name: str, image_size: int = 224) -> dict:
    """Verify fixture target detection for a single task."""
    # Import here to avoid loading robocasa at module level
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from eval_vla_robocasa import create_env
    from robobridge.wrappers.robocasa_perception import RoboCasaPerception

    result = {
        "task": task_name,
        "success": False,
        "target_attr": None,
        "target_prefix": None,
        "target_nat_lang": None,
        "target_detection": None,
        "target_pos": None,
        "target_dist": None,
        "all_detections": [],
        "fixture_map": {},
        "error": None,
    }

    try:
        env = create_env(task_name, image_size=image_size)
        obs = env.reset()
        ep_meta = env.get_ep_meta()

        # Find target fixture
        target_fxtr = None
        target_attr_name = None
        for attr in TARGET_ATTRS.get(task_name, ["fxtr"]):
            target_fxtr = getattr(env, attr, None)
            if target_fxtr is not None:
                target_attr_name = attr
                break

        result["target_attr"] = target_attr_name

        if target_fxtr is not None:
            result["target_prefix"] = getattr(target_fxtr, "naming_prefix", "N/A")
            result["target_nat_lang"] = getattr(target_fxtr, "nat_lang", "N/A")

        # Build fixture map from env.fixtures
        fixture_map = {}
        if hasattr(env, "fixtures"):
            for name, fxtr in env.fixtures.items():
                prefix = getattr(fxtr, "naming_prefix", "")
                nat = getattr(fxtr, "nat_lang", "")
                if prefix:
                    fixture_map[prefix] = nat
        result["fixture_map"] = fixture_map

        # Run perception
        perception = RoboCasaPerception()
        perception.set_environment_state(obs, ep_meta=ep_meta, env=env)
        detections = perception.process()

        # Collect all detection info
        for d in detections:
            pos = d.pose.get("position", {}) if isinstance(d.pose, dict) else {}
            result["all_detections"].append({
                "name": d.name,
                "role": d.metadata.get("role", "unknown"),
                "pos": f"({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f})",
                "fixture": d.metadata.get("fixture", ""),
            })

        # Find target detections (role == "target")
        target_dets = [d for d in detections if d.metadata.get("role") == "target"]

        if target_dets:
            td = target_dets[0]
            pos = td.pose["position"]
            dist = np.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
            result["target_detection"] = td.name
            result["target_pos"] = f"({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})"
            result["target_dist"] = f"{dist:.3f}m"
            result["success"] = dist < 1.5  # within arm reach

        env.close()

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{task_name}] Error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify fixture target detection")
    parser.add_argument("--all-tasks", action="store_true", help="Test all 24 tasks")
    parser.add_argument("--tasks", nargs="+", default=[], help="Specific tasks to test")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    tasks = ALL_TASKS if args.all_tasks else (args.tasks or ["CloseDrawer"])

    print(f"\n{'='*80}")
    print(f"Fixture Target Verification — {len(tasks)} tasks")
    print(f"{'='*80}\n")

    results = []
    pass_count = 0

    for task in tasks:
        print(f"--- {task} ---")
        r = verify_task(task, image_size=args.image_size)
        results.append(r)

        if r["error"]:
            print(f"  ERROR: {r['error']}")
            continue

        status = "MATCH ✓" if r["success"] else "FAIL ✗"
        print(f"  target attr: env.{r['target_attr']}")
        print(f"  naming_prefix: {r['target_prefix']}")
        print(f"  nat_lang: {r['target_nat_lang']}")
        print(f"  detection name: {r['target_detection']}")
        print(f"  position: {r['target_pos']}, dist: {r['target_dist']}")
        print(f"  result: {status}")

        # Print all detections as planner would see them
        print(f"  all detections ({len(r['all_detections'])}):")
        for det in r["all_detections"]:
            marker = " ← TARGET" if det["role"] == "target" else ""
            print(f"    {det['name']} [{det['role']}] {det['pos']}{marker}")

        if r["success"]:
            pass_count += 1
        print()

    # Summary
    print(f"{'='*80}")
    print(f"Summary: {pass_count}/{len(results)} tasks passed")
    failed = [r["task"] for r in results if not r["success"]]
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
