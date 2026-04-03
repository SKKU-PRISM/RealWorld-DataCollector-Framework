#!/usr/bin/env python3
"""
Joint Compensation Calibration

Measures per-joint backlash (deadband), direction-dependent offset,
and base compensation for SO-101 robot arms.

Calibrates:
  1. Deadband — gear backlash dead-zone width per joint
  2. Direction offset — asymmetric error when approaching from +/- direction
  3. Base compensation — static gravity offset per joint

Usage:
    python joint_compensation_calibration/calibrate.py --robot-id 3
    python joint_compensation_calibration/calibrate.py --robot-id 3 --verify
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Joints to calibrate (only gravity-loaded joints have meaningful backlash)
CALIBRATE_JOINTS = [
    {"idx": 1, "name": "shoulder_lift"},
    {"idx": 2, "name": "elbow_flex"},
]

# Test angles per joint (normalized, -100 to +100)
# Spread across the working range
TEST_ANGLES = [-60, -30, 0, 30, 60]

# Movement amplitude for approach (normalized units)
APPROACH_OFFSET = 20.0

# Settle time after movement (seconds)
SETTLE_TIME = 1.5

# Movement duration (seconds)
MOVE_DURATION = 2.0


def read_joint(robot, joint_idx, num_samples=5, delay=0.05):
    """Read joint position multiple times and average (reduce noise)."""
    readings = []
    for _ in range(num_samples):
        pos = robot.read_positions(normalize=True)
        readings.append(float(pos[joint_idx]))
        time.sleep(delay)
    return float(np.median(readings))


def move_to_joint(robot, joint_idx, target, current_full, duration=MOVE_DURATION):
    """Move a single joint to target while keeping others fixed."""
    command = current_full.copy()
    num_steps = int(duration * 50)
    start_val = command[joint_idx]

    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        command[joint_idx] = start_val + alpha * (target - start_val)
        robot.write_positions(command, normalize=True)
        time.sleep(1.0 / 50)

    # Final command
    command[joint_idx] = target
    robot.write_positions(command, normalize=True)
    time.sleep(SETTLE_TIME)


def measure_deadband(robot, joint_idx, joint_name, test_angles):
    """Measure backlash deadband for one joint.

    Method: For each test angle, approach from +offset and -offset.
    Deadband = |arrival_from_positive - arrival_from_negative|

    Returns:
        List of {angle, from_pos, from_neg, deadband}
    """
    print(f"\n  [{joint_name}] Measuring deadband...")
    results = []

    current = robot.read_positions(normalize=True)

    for angle in test_angles:
        # Approach from positive side: go to angle+offset, then to angle
        above = angle + APPROACH_OFFSET
        move_to_joint(robot, joint_idx, above, current)
        current = robot.read_positions(normalize=True)
        move_to_joint(robot, joint_idx, angle, current)
        from_pos = read_joint(robot, joint_idx)

        # Approach from negative side: go to angle-offset, then to angle
        current = robot.read_positions(normalize=True)
        below = angle - APPROACH_OFFSET
        move_to_joint(robot, joint_idx, below, current)
        current = robot.read_positions(normalize=True)
        move_to_joint(robot, joint_idx, angle, current)
        from_neg = read_joint(robot, joint_idx)

        deadband = abs(from_pos - from_neg)

        results.append({
            "target_angle": angle,
            "from_positive": round(from_pos, 2),
            "from_negative": round(from_neg, 2),
            "deadband": round(deadband, 2),
        })

        print(f"    angle={angle:+4.0f}: from+={from_pos:+7.2f}, from-={from_neg:+7.2f}, "
              f"deadband={deadband:.2f}")

        # Return to test angle for next iteration
        current = robot.read_positions(normalize=True)

    return results


def measure_direction_offset(robot, joint_idx, joint_name, test_angles):
    """Measure direction-dependent offset for one joint.

    Method: For each test angle, command the same position and measure
    the signed error from both approach directions.
    offset_pos = mean(target - actual) when approaching from above
    offset_neg = mean(target - actual) when approaching from below

    Returns:
        List of {angle, error_from_pos, error_from_neg}
    """
    print(f"\n  [{joint_name}] Measuring direction offset...")
    results = []

    current = robot.read_positions(normalize=True)

    for angle in test_angles:
        # From positive
        above = angle + APPROACH_OFFSET
        move_to_joint(robot, joint_idx, above, current)
        current = robot.read_positions(normalize=True)
        move_to_joint(robot, joint_idx, angle, current)
        actual_from_pos = read_joint(robot, joint_idx)
        error_from_pos = angle - actual_from_pos

        # From negative
        current = robot.read_positions(normalize=True)
        below = angle - APPROACH_OFFSET
        move_to_joint(robot, joint_idx, below, current)
        current = robot.read_positions(normalize=True)
        move_to_joint(robot, joint_idx, angle, current)
        actual_from_neg = read_joint(robot, joint_idx)
        error_from_neg = angle - actual_from_neg

        results.append({
            "target_angle": angle,
            "error_from_positive": round(error_from_pos, 3),
            "error_from_negative": round(error_from_neg, 3),
        })

        print(f"    angle={angle:+4.0f}: err_from+={error_from_pos:+.3f}, "
              f"err_from-={error_from_neg:+.3f}")

        current = robot.read_positions(normalize=True)

    return results


def measure_base_compensation(robot, joint_idx, joint_name, test_angles):
    """Measure static gravity offset for one joint.

    Method: Command each angle, wait for settle, read actual.
    base_comp = mean(target - actual) across angles.

    Returns:
        List of {angle, target, actual, error}
    """
    print(f"\n  [{joint_name}] Measuring base compensation...")
    results = []

    current = robot.read_positions(normalize=True)

    for angle in test_angles:
        move_to_joint(robot, joint_idx, angle, current)
        actual = read_joint(robot, joint_idx)
        error = angle - actual

        results.append({
            "target_angle": angle,
            "actual": round(actual, 2),
            "error": round(error, 3),
        })

        print(f"    angle={angle:+4.0f}: actual={actual:+7.2f}, error={error:+.3f}")

        current = robot.read_positions(normalize=True)

    return results


def compute_parameters(deadband_results, offset_results, base_results, joint_name):
    """Compute calibration parameters from measurements."""
    # Deadband: median across test angles
    deadband = float(np.median([r["deadband"] for r in deadband_results]))

    # Direction offset: mean error from each direction
    offset_pos = float(np.mean([r["error_from_positive"] for r in offset_results]))
    offset_neg = float(np.mean([r["error_from_negative"] for r in offset_results]))

    # Base compensation: mean absolute error
    base_comp = float(np.mean([abs(r["error"]) for r in base_results]))
    # Sign: if mean error is positive (actual < target), we need to add
    mean_error = float(np.mean([r["error"] for r in base_results]))

    params = {
        "joint_name": joint_name,
        "deadband": round(deadband, 2),
        "offset_positive": round(offset_pos, 3),
        "offset_negative": round(offset_neg, 3),
        "base_compensation": round(base_comp, 2),
        "mean_signed_error": round(mean_error, 3),
    }

    print(f"\n  [{joint_name}] Parameters:")
    print(f"    deadband:          {params['deadband']:.2f}")
    print(f"    offset_positive:   {params['offset_positive']:+.3f}")
    print(f"    offset_negative:   {params['offset_negative']:+.3f}")
    print(f"    base_compensation: {params['base_compensation']:.2f}")

    return params


def save_to_compensation(robot_id, joint_params):
    """Update compensation JSON with calibrated values."""
    comp_path = (PROJECT_ROOT / "robot_configs" / "motor_calibration"
                 / "so101" / f"robot{robot_id}_compensation.json")
    if not comp_path.exists():
        print(f"[SAVE] File not found: {comp_path}")
        return False

    with open(comp_path) as f:
        data = json.load(f)

    for jp in joint_params:
        name = jp["joint_name"]
        data["deadband"][name] = jp["deadband"]
        data["offset_positive"][name] = jp["offset_positive"]
        data["offset_negative"][name] = jp["offset_negative"]
        data["base_compensation"][name] = jp["base_compensation"]

    data["calibration_date"] = time.strftime("%Y-%m-%d")
    data["description"] = f"Adaptive compensation parameters for SO-101 Robot {robot_id} (auto-calibrated)"

    with open(comp_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] Updated: {comp_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Joint Compensation Calibration")
    parser.add_argument("--robot-id", type=int, required=True)
    parser.add_argument("--test-angles", type=float, nargs="+", default=TEST_ANGLES)
    parser.add_argument("--verify", action="store_true",
                        help="After calibration, re-measure to verify improvement")
    parser.add_argument("--measure-only", action="store_true",
                        help="Only measure, do not save")
    args = parser.parse_args()

    robot_config = f"robot_configs/robot/so101_robot{args.robot_id}.yaml"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Joint Compensation Calibration — Robot {args.robot_id}")
    print("=" * 60)
    print(f"  Joints: {[j['name'] for j in CALIBRATE_JOINTS]}")
    print(f"  Test angles: {args.test_angles}")
    print(f"  Points per joint: {len(args.test_angles)}")
    print("=" * 60)

    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config=robot_config,
        frame="base_link",
        use_compensation=False,  # Disable compensation during measurement
        verbose=False,
    )
    skills.connect()
    robot = skills.robot

    # Move to initial state
    skills.move_to_initial_state()
    time.sleep(1.0)

    all_results = {}
    all_params = []

    for joint_info in CALIBRATE_JOINTS:
        idx = joint_info["idx"]
        name = joint_info["name"]
        print(f"\n{'='*50}")
        print(f"  Calibrating: {name} (joint {idx})")
        print(f"{'='*50}")

        # Measure
        db_results = measure_deadband(robot, idx, name, args.test_angles)
        offset_results = measure_direction_offset(robot, idx, name, args.test_angles)
        base_results = measure_base_compensation(robot, idx, name, args.test_angles)

        # Compute parameters
        params = compute_parameters(db_results, offset_results, base_results, name)
        all_params.append(params)

        all_results[name] = {
            "deadband_measurements": db_results,
            "offset_measurements": offset_results,
            "base_measurements": base_results,
            "parameters": params,
        }

        # Return to safe position
        skills.move_to_initial_state()
        time.sleep(0.5)

    skills.disconnect()

    # Save measurements
    meas_path = results_dir / f"robot{args.robot_id}_joint_measurements.json"
    with open(meas_path, "w") as f:
        json.dump({"robot_id": args.robot_id, "results": all_results}, f, indent=2)
    print(f"\n[SAVE] Measurements: {meas_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  CALIBRATION SUMMARY — Robot {args.robot_id}")
    print(f"{'='*60}")
    print(f"{'Joint':>15s} {'Deadband':>10s} {'Offset+':>10s} {'Offset-':>10s} {'BasComp':>10s}")
    print("-" * 58)

    # Load old values for comparison
    comp_path = (PROJECT_ROOT / "robot_configs" / "motor_calibration"
                 / "so101" / f"robot{args.robot_id}_compensation.json")
    with open(comp_path) as f:
        old_data = json.load(f)

    for p in all_params:
        name = p["joint_name"]
        old_db = old_data["deadband"].get(name, 0)
        old_op = old_data["offset_positive"].get(name, 0)
        old_on = old_data["offset_negative"].get(name, 0)
        old_bc = old_data["base_compensation"].get(name, 0)

        print(f"{'[OLD] '+name:>15s} {old_db:10.2f} {old_op:+10.3f} {old_on:+10.3f} {old_bc:10.2f}")
        print(f"{'[NEW] '+name:>15s} {p['deadband']:10.2f} {p['offset_positive']:+10.3f} "
              f"{p['offset_negative']:+10.3f} {p['base_compensation']:10.2f}")
        print()

    if args.measure_only:
        print("[Done] Measure-only mode.")
        return

    # Save to compensation config
    print("[SAVE] Updating compensation config...")
    save_to_compensation(args.robot_id, all_params)

    # Verify
    if args.verify:
        print(f"\n{'='*60}")
        print(f"  VERIFICATION — Robot {args.robot_id}")
        print(f"{'='*60}")

        skills2 = LeRobotSkills(
            robot_config=robot_config,
            frame="base_link",
            use_compensation=True,  # Enable compensation
            verbose=False,
        )
        skills2.connect()
        robot2 = skills2.robot

        skills2.move_to_initial_state()
        time.sleep(1.0)

        for joint_info in CALIBRATE_JOINTS:
            idx = joint_info["idx"]
            name = joint_info["name"]
            print(f"\n  Verifying {name}...")

            base_results_after = measure_base_compensation(
                robot2, idx, name, args.test_angles)

            before = [r for r in all_results[name]["base_measurements"]]
            after = base_results_after

            print(f"\n  {'Angle':>7s} {'Before':>10s} {'After':>10s} {'Improve':>10s}")
            print("  " + "-" * 40)
            for b, a in zip(before, after):
                imp = abs(b["error"]) - abs(a["error"])
                print(f"  {b['target_angle']:+7.0f} {b['error']:+10.3f} {a['error']:+10.3f} {imp:+10.3f}")

            mean_before = np.mean([abs(r["error"]) for r in before])
            mean_after = np.mean([abs(r["error"]) for r in after])
            pct = 100 * (1 - mean_after / mean_before) if mean_before > 0 else 0
            print(f"  {'Mean':>7s} {mean_before:10.3f} {mean_after:10.3f} {pct:9.0f}%")

        skills2.move_to_initial_state()
        skills2.disconnect()

    print("\n[Done] Calibration complete.")


if __name__ == "__main__":
    main()
