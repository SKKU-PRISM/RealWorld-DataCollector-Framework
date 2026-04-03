#!/usr/bin/env python3
"""
Gravity Sag Calibration

Measures z-axis error at multiple (reach, z) positions, then fits the
gravity sag compensation model:

    sag = gain * reach^power * max(0, z - deadzone)

Usage:
    # Calibrate robot3
    python gravity_sag_calibration/calibrate.py --robot-id 3

    # Calibrate and then verify with compensation applied
    python gravity_sag_calibration/calibrate.py --robot-id 3 --verify

    # Custom grid
    python gravity_sag_calibration/calibrate.py --robot-id 3 --reaches 0.20 0.25 0.30 0.35 --heights 0.05 0.10 0.15 0.20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def measure_z_errors(skills, reaches, heights, verbose=True):
    """Move to each (reach, z) grid point and measure z-error.

    Args:
        skills: Connected LeRobotSkills instance (gravity_sag disabled).
        reaches: List of reach distances (meters).
        heights: List of z heights (meters).
        verbose: Print progress.

    Returns:
        List of dicts: [{reach, z_target, z_actual, z_error}, ...]
    """
    measurements = []

    total = len(reaches) * len(heights)
    idx = 0

    for reach in reaches:
        for z_target in heights:
            idx += 1
            # Position: x=reach, y=0, z=z_target (directly in front of robot)
            position = [reach, 0.0, z_target]

            if verbose:
                print(f"\n[{idx}/{total}] reach={reach:.2f}m, z={z_target:.3f}m")

            # Check reachability
            if not skills.kinematics.is_position_reachable(np.array(position)):
                if verbose:
                    print(f"  SKIP: unreachable")
                continue

            # Move (no gravity_sag compensation)
            success = skills.move_to_position(position)

            # Measure actual EE position via FK
            _, actual_joints, actual_ee = skills._get_current_state()
            z_actual = actual_ee[2]
            z_error = z_target - z_actual  # positive = arm drooped below target

            measurements.append({
                "reach": float(reach),
                "z_target": float(z_target),
                "z_actual": float(z_actual),
                "z_error": float(z_error),
                "z_error_mm": float(z_error * 1000),
                "success": bool(success),
            })

            if verbose:
                status = "OK" if success else "FAIL"
                print(f"  z_actual={z_actual*1000:.1f}mm, z_error={z_error*1000:+.1f}mm [{status}]")

            # Return to initial to avoid path-dependent effects
            skills.move_to_initial_state()
            time.sleep(0.3)

    return measurements


def fit_sag_model(measurements, deadzone=0.05):
    """Fit sag = gain * reach^power * max(0, z - deadzone) to measurements.

    Args:
        measurements: List from measure_z_errors().
        deadzone: Fixed z deadzone (meters).

    Returns:
        (gain, power, deadzone, fit_stats)
    """
    from scipy.optimize import curve_fit

    # Filter: only successful measurements with positive z_error and z > deadzone
    valid = [m for m in measurements if m["success"] and m["z_target"] > deadzone]
    if len(valid) < 3:
        print(f"[FIT] Not enough valid measurements ({len(valid)}). Need at least 3.")
        return None, None, deadzone, {}

    reaches = np.array([m["reach"] for m in valid])
    z_targets = np.array([m["z_target"] for m in valid])
    z_errors = np.array([m["z_error"] for m in valid])

    # z_factor = max(0, z - deadzone)
    z_factors = np.maximum(0, z_targets - deadzone)

    def sag_model(X, gain, power):
        reach, z_factor = X
        return gain * (reach ** power) * z_factor

    try:
        popt, pcov = curve_fit(
            sag_model,
            (reaches, z_factors),
            z_errors,
            p0=[1.0, 2.0],        # initial guess
            bounds=([0.0, 0.5], [10.0, 5.0]),  # reasonable bounds
            maxfev=5000,
        )
        gain, power = popt

        # Compute fit quality
        predicted = sag_model((reaches, z_factors), gain, power)
        residuals = z_errors - predicted
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        r2 = float(1 - np.sum(residuals ** 2) / np.sum((z_errors - np.mean(z_errors)) ** 2))

        fit_stats = {
            "num_points": len(valid),
            "rmse_mm": round(rmse * 1000, 2),
            "r_squared": round(r2, 4),
            "mean_error_before_mm": round(float(np.mean(np.abs(z_errors))) * 1000, 2),
            "mean_error_after_mm": round(float(np.mean(np.abs(residuals))) * 1000, 2),
        }

        return float(gain), float(power), deadzone, fit_stats

    except Exception as e:
        print(f"[FIT] curve_fit failed: {e}")
        return None, None, deadzone, {}


def save_to_compensation(robot_id, gain, power, deadzone, max_offset=0.05):
    """Write gravity_sag parameters to the robot's compensation JSON."""
    comp_path = PROJECT_ROOT / "robot_configs" / "motor_calibration" / "so101" / f"robot{robot_id}_compensation.json"
    if not comp_path.exists():
        print(f"[SAVE] Compensation file not found: {comp_path}")
        return False

    with open(comp_path) as f:
        data = json.load(f)

    data["gravity_sag"] = {
        "enabled": True,
        "gain": round(gain, 4),
        "reach_power": round(power, 4),
        "z_deadzone": deadzone,
        "max_offset": max_offset,
        "_comment": f"Auto-calibrated. sag = {gain:.4f} * reach^{power:.2f} * max(0, z - {deadzone})"
    }

    with open(comp_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] gravity_sag written to {comp_path}")
    return True


def plot_results(measurements, gain, power, deadzone, robot_id, save_path):
    """Visualize measurements and fitted model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not available, skipping visualization")
        return

    valid = [m for m in measurements if m["success"]]
    reaches = np.array([m["reach"] for m in valid])
    z_targets = np.array([m["z_target"] for m in valid])
    z_errors = np.array([m["z_error_mm"] for m in valid])

    # Predicted
    z_factors = np.maximum(0, z_targets - deadzone)
    predicted_mm = gain * (reaches ** power) * z_factors * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: measured vs predicted
    ax = axes[0]
    scatter = ax.scatter(z_errors, predicted_mm, c=reaches, cmap='viridis', s=60, edgecolors='k', linewidths=0.5)
    lims = [min(z_errors.min(), predicted_mm.min()) - 1, max(z_errors.max(), predicted_mm.max()) + 1]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='perfect fit')
    ax.set_xlabel('Measured z-error (mm)')
    ax.set_ylabel('Predicted z-error (mm)')
    ax.set_title(f'Robot {robot_id}: Measured vs Predicted')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='reach (m)')

    # Right: z-error by reach, colored by z
    ax = axes[1]
    for z_val in sorted(set(z_targets)):
        mask = z_targets == z_val
        ax.plot(reaches[mask], z_errors[mask], 'o-', label=f'z={z_val:.2f}m', markersize=6)
        ax.plot(reaches[mask], predicted_mm[mask], 'x--', alpha=0.5, markersize=4)
    ax.set_xlabel('Reach (m)')
    ax.set_ylabel('z-error (mm)')
    ax.set_title(f'Robot {robot_id}: z-error by Reach (o=measured, x=predicted)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[PLOT] Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gravity Sag Calibration")
    parser.add_argument("--robot-id", type=int, required=True, help="Robot ID (e.g., 2, 3)")
    parser.add_argument("--reaches", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30, 0.35, 0.40],
                        help="Reach distances to measure (meters)")
    parser.add_argument("--heights", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20],
                        help="Z heights to measure (meters)")
    parser.add_argument("--deadzone", type=float, default=0.05,
                        help="Z deadzone for model (meters, default 0.05)")
    parser.add_argument("--verify", action="store_true",
                        help="After calibration, verify with compensation enabled")
    parser.add_argument("--measure-only", action="store_true",
                        help="Only measure, do not fit or save")
    args = parser.parse_args()

    robot_config = f"robot_configs/robot/so101_robot{args.robot_id}.yaml"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Gravity Sag Calibration — Robot {args.robot_id}")
    print("=" * 60)
    print(f"  Config: {robot_config}")
    print(f"  Reaches: {args.reaches}")
    print(f"  Heights: {args.heights}")
    print(f"  Grid: {len(args.reaches)} x {len(args.heights)} = {len(args.reaches) * len(args.heights)} points")
    print(f"  Deadzone: {args.deadzone}m")
    print("=" * 60)

    # ── Phase 1: Measure ──
    print("\n[Phase 1] Measuring z-errors (gravity_sag DISABLED)...")

    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config=robot_config,
        frame="base_link",
        verbose=True,
    )
    skills.connect()

    # Disable gravity_sag for measurement
    skills.gravity_sag = None
    print(f"  gravity_sag: DISABLED for measurement")

    skills.move_to_initial_state()
    skills.gripper_close()

    measurements = measure_z_errors(skills, args.reaches, args.heights)

    skills.move_to_initial_state()
    skills.disconnect()

    # Save measurements
    meas_path = results_dir / f"robot{args.robot_id}_sag_measurements.json"
    with open(meas_path, "w") as f:
        json.dump({"robot_id": args.robot_id, "measurements": measurements}, f, indent=2)
    print(f"\n[Phase 1] Measurements saved: {meas_path}")

    # Summary
    valid = [m for m in measurements if m["success"]]
    if valid:
        errors = [abs(m["z_error_mm"]) for m in valid]
        print(f"\n  Summary: {len(valid)} valid points")
        print(f"  z-error: mean={np.mean(errors):.1f}mm, max={np.max(errors):.1f}mm, std={np.std(errors):.1f}mm")

    if args.measure_only:
        print("\n[Done] Measure-only mode. Exiting.")
        return

    # ── Phase 2: Fit ──
    print("\n[Phase 2] Fitting sag model...")

    gain, power, deadzone, fit_stats = fit_sag_model(measurements, args.deadzone)

    if gain is None:
        print("[Phase 2] Fitting failed. Exiting.")
        return

    print(f"\n  Model: sag = {gain:.4f} * reach^{power:.2f} * max(0, z - {deadzone})")
    print(f"  RMSE: {fit_stats['rmse_mm']:.2f}mm")
    print(f"  R-squared: {fit_stats['r_squared']:.4f}")
    print(f"  Mean |error| before: {fit_stats['mean_error_before_mm']:.2f}mm")
    print(f"  Mean |error| after:  {fit_stats['mean_error_after_mm']:.2f}mm")
    improvement = 100 * (1 - fit_stats['mean_error_after_mm'] / fit_stats['mean_error_before_mm'])
    print(f"  Improvement: {improvement:.0f}%")

    # Save fit result
    fit_path = results_dir / f"robot{args.robot_id}_sag_fit.json"
    with open(fit_path, "w") as f:
        json.dump({
            "robot_id": args.robot_id,
            "gain": round(gain, 4),
            "reach_power": round(power, 4),
            "z_deadzone": deadzone,
            "fit_stats": fit_stats,
        }, f, indent=2)
    print(f"  Fit result saved: {fit_path}")

    # Plot
    plot_path = results_dir / f"robot{args.robot_id}_sag_fit.png"
    plot_results(measurements, gain, power, deadzone, args.robot_id, str(plot_path))

    # ── Phase 3: Save to compensation config ──
    print("\n[Phase 3] Saving to compensation config...")
    save_to_compensation(args.robot_id, gain, power, deadzone)

    # ── Phase 4: Verify (optional) ──
    if args.verify:
        print("\n[Phase 4] Verifying with compensation ENABLED...")

        skills2 = LeRobotSkills(
            robot_config=robot_config,
            frame="base_link",
            verbose=True,
        )
        skills2.connect()

        if skills2.gravity_sag is not None:
            print(f"  gravity_sag: ENABLED (gain={skills2.gravity_sag.gain}, power={skills2.gravity_sag.reach_power})")
        else:
            print("  WARNING: gravity_sag still not loaded!")

        skills2.move_to_initial_state()
        skills2.gripper_close()

        verify_measurements = measure_z_errors(skills2, args.reaches, args.heights)

        skills2.move_to_initial_state()
        skills2.disconnect()

        # Save verify results
        verify_path = results_dir / f"robot{args.robot_id}_sag_verify.json"
        with open(verify_path, "w") as f:
            json.dump({"robot_id": args.robot_id, "measurements": verify_measurements}, f, indent=2)

        # Compare before/after
        print(f"\n{'='*60}")
        print(f"  BEFORE vs AFTER Compensation")
        print(f"{'='*60}")
        print(f"{'Reach':>7s} {'Z':>6s} {'Before':>10s} {'After':>10s} {'Improve':>10s}")
        print("-" * 50)

        before_errs = []
        after_errs = []
        for bm, am in zip(measurements, verify_measurements):
            if bm["success"] and am["success"]:
                be = abs(bm["z_error_mm"])
                ae = abs(am["z_error_mm"])
                before_errs.append(be)
                after_errs.append(ae)
                imp = be - ae
                print(f"{bm['reach']:7.2f} {bm['z_target']:6.3f} {be:+10.1f}mm {ae:+10.1f}mm {imp:+10.1f}mm")

        if before_errs:
            mean_before = np.mean(before_errs)
            mean_after = np.mean(after_errs)
            total_imp = 100 * (1 - mean_after / mean_before) if mean_before > 0 else 0
            print("-" * 50)
            print(f"{'Mean':>7s} {'':>6s} {mean_before:10.1f}mm {mean_after:10.1f}mm {total_imp:9.0f}%")

    print("\n[Done] Calibration complete.")


if __name__ == "__main__":
    main()
