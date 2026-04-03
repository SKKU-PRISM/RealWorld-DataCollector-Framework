#!/usr/bin/env python3
"""
Record Initial/Free State Script

Records a "safe" initial state or free state position for robot.
- initial_state: Starting point for IK calculations
- free_state: Safe parking position when robot is idle

Usage:
    python scripts/record_initial_state.py --robot 2
    python scripts/record_initial_state.py --robot 3 --type free

Flow:
    1. Disable torque (robot becomes manually movable)
    2. User physically moves robot to a "safe" position
    3. User presses Enter
    4. Current position is saved to robot_configs/{type}/{robot_id}.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Project root directory (works regardless of where script is run from)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot_cap.hardware import FeetechController
from lerobot_cap.hardware.calibration import MotorCalibration


def load_config(config_path: str) -> dict:
    """Load robot configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_robot_id(config_path: str) -> str:
    """Extract robot ID from config path (e.g., 'so101_robot2.yaml' -> 'robot2')."""
    match = re.search(r'robot(\d+)', str(config_path))
    return f"robot{match.group(1)}" if match else "robot3"


def get_available_robot_ids():
    """robot_configs/robot/ 디렉토리에서 사용 가능한 robot ID 목록을 반환."""
    config_dir = PROJECT_ROOT / "robot_configs" / "robot"
    ids = []
    for f in sorted(config_dir.glob("so101_robot*.yaml")):
        match = re.search(r'robot(\d+)', f.stem)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def interactive_input():
    """대화형 모드: 사용자로부터 robot ID와 state type을 입력받음."""
    print("\n" + "=" * 60)
    print("         ROBOT STATE RECORDING - Interactive Mode")
    print("=" * 60)

    # 사용 가능한 robot ID 탐색
    available_ids = get_available_robot_ids()
    if not available_ids:
        print("  Error: robot_configs/robot/ 에 설정 파일이 없습니다.")
        sys.exit(1)

    ids_str = ", ".join(str(i) for i in available_ids)

    # Robot ID 입력
    while True:
        try:
            robot_input = input(f"\n  Robot ID를 입력하세요 ({ids_str}): ").strip()
            robot_id = int(robot_input)
            if robot_id in available_ids:
                break
            print(f"  Error: {ids_str} 중 하나만 입력 가능합니다.")
        except ValueError:
            print("  Error: 숫자를 입력해주세요.")

    # State type 입력
    print("\n  State type을 선택하세요:")
    print("    1. initial - IK 계산을 위한 시작 위치")
    print("    2. free    - 대기 시 안전한 parking 위치")

    while True:
        type_input = input("\n  선택 (1/initial 또는 2/free): ").strip().lower()
        if type_input in ['1', 'initial']:
            state_type = 'initial'
            break
        elif type_input in ['2', 'free']:
            state_type = 'free'
            break
        print("  Error: 1, 2, 'initial', 'free' 중 하나를 입력해주세요.")

    print(f"\n  선택됨: robot{robot_id}, {state_type}_state")
    return robot_id, state_type


def main():
    parser = argparse.ArgumentParser(description="Record initial/free state for robot")
    parser.add_argument("robot_id", type=int, nargs='?', default=None,
                        help="Robot ID (2 or 3)")
    parser.add_argument("state_type", type=str, nargs='?', default=None,
                        choices=["initial", "free"],
                        help="State type: 'initial' or 'free'")
    parser.add_argument("--robot", type=int, default=None,
                        help="Robot ID (2 or 3) - alternative to positional arg")
    parser.add_argument("--type", type=str, default=None,
                        choices=["initial", "free"],
                        help="State type - alternative to positional arg")
    parser.add_argument("--config", type=str, default=None,
                        help="Robot configuration file (optional, auto-detected from --robot)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (optional, auto-generated)")

    args = parser.parse_args()

    # positional args 또는 --robot/--type 옵션 사용
    robot = args.robot_id or args.robot
    state_type_arg = args.state_type or args.type

    # 인자가 없으면 대화형 모드
    if robot is None or state_type_arg is None:
        robot, state_type_arg = interactive_input()

    # args에 값 설정 (이후 코드 호환성 위해)
    args.robot = robot
    args.type = state_type_arg

    # Resolve config path relative to project root
    if args.config is None:
        config_path = PROJECT_ROOT / "robot_configs" / "robot" / f"so101_robot{args.robot}.yaml"
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    # YAML 내 상대경로를 YAML 파일 위치 기준으로 resolve
    yaml_dir = config_path.parent
    for key in ("calibration_file", "compensation_file"):
        val = config.get(key)
        if val and not Path(val).is_absolute():
            config[key] = str((yaml_dir / val).resolve())

    # Extract robot_id from config path
    robot_id = extract_robot_id(str(config_path))
    state_type = f"{args.type}_state"

    # Output path (relative to project root)
    # Format: configs/{initial_state|free_state}/{robot_id}_{state_type}.json
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = PROJECT_ROOT / "robot_configs" / state_type / f"{robot_id}_{state_type}.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Robot: {robot_id}")
    print(f"State type: {state_type}")
    print(f"Output: {output_path}")

    # Load calibration
    calibration_file = config.get("calibration_file")
    calibration_by_id = {}

    if calibration_file and Path(calibration_file).exists():
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            for name, data in calib_data.items():
                motor_id = data.get('motor_id', data.get('id'))
                if motor_id is None:
                    continue
                calibration_by_id[motor_id] = MotorCalibration(
                    motor_id=motor_id,
                    model=data.get('model', 'sts3215'),
                    drive_mode=data.get('drive_mode', 0),
                    homing_offset=data.get('homing_offset', 0),
                    range_min=data.get('range_min', 0),
                    range_max=data.get('range_max', 4095),
                )
            print(f"Calibration loaded: {len(calibration_by_id)} motors")
        except Exception as e:
            print(f"Error loading calibration: {e}")
            sys.exit(1)
    else:
        print(f"Error: Calibration file not found: {calibration_file}")
        sys.exit(1)

    # Motor IDs
    motor_ids = [config["motors"][f"motor_{i}"]["id"] for i in range(1, 7)]

    # Create controller
    robot = FeetechController(
        port=config.get("port"),
        baudrate=config.get("baudrate", 1000000),
        motor_ids=motor_ids,
        calibration=calibration_by_id,
    )

    try:
        # Connect
        if not robot.connect():
            print("Error: Failed to connect to robot")
            sys.exit(1)

        state_type_upper = state_type.upper().replace("_", " ")
        print("\n" + "=" * 60)
        print(f"         {state_type_upper} RECORDING ({robot_id})")
        print("=" * 60)

        # Disable torque so user can move robot
        print("\n[Step 1] Disabling torque...")
        robot.disable_torque()
        print("         Torque disabled. Robot is now manually movable.")

        # Instructions
        print(f"\n[Step 2] Move the robot to a SAFE {args.type} position")
        print("         ")
        if args.type == "initial":
            print("         Guidelines for a good INITIAL state:")
            print("         - Arm should be in a neutral, middle-range position")
            print("         - Good position: arm slightly extended forward")
            print("         - All joints near their center positions")
            print("         - Example: normalized values around [-20, 0, 20, 0, 0]")
        else:  # free state
            print("         Guidelines for a good FREE state:")
            print("         - Safe parking position when robot is idle")
            print("         - Arm folded or retracted to avoid obstacles")
            print("         - Clear of workspace for camera visibility")
        print("         ")
        print("         Avoid extreme joint angles (close to limits)")

        # Show current position in real-time
        print("\n[Step 3] Current position (updates every 0.5s):")
        print("         Press Enter when ready to save.\n")

        import select
        import time

        # For non-blocking input check
        def input_available():
            if sys.platform == 'win32':
                import msvcrt
                return msvcrt.kbhit()
            else:
                return select.select([sys.stdin], [], [], 0)[0]

        while True:
            # Read current position
            current_normalized = robot.read_positions(normalize=True)
            arm_normalized = current_normalized[:5]
            gripper_normalized = current_normalized[5]

            # Display
            pos_str = ", ".join([f"{p:6.1f}" for p in arm_normalized])
            print(f"\r  Arm: [{pos_str}]  Gripper: {gripper_normalized:5.1f}  | Press Enter to save", end="", flush=True)

            # Check for Enter key
            if input_available():
                line = sys.stdin.readline()
                break

            time.sleep(0.5)

        # Final reading
        final_normalized = robot.read_positions(normalize=True)
        arm_normalized = final_normalized[:5].tolist()
        gripper_normalized = float(final_normalized[5])

        print("\n")
        print("=" * 60)
        print("         RECORDED POSITION")
        print("=" * 60)
        print(f"  Arm joints (normalized):  {arm_normalized}")
        print(f"  Gripper (normalized):     {gripper_normalized:.1f}")

        # Check if position looks reasonable
        warnings = []
        for i, val in enumerate(arm_normalized):
            if abs(val) > 80:
                warnings.append(f"  - Joint {i+1}: {val:.1f} is close to limit (>80)")

        if warnings:
            print("\n  WARNINGS:")
            for w in warnings:
                print(w)
            print("\n  Consider choosing a more centered position.")

            response = input("\n  Save anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("  Cancelled. Position not saved.")
                return

        # Save to file
        if args.type == "initial":
            description = "Safe initial state for IK solving. Robot moves here before computing IK to target."
        else:
            description = "Safe free/parking state. Robot moves here when idle to clear workspace."

        state_data = {
            "initial_state_normalized": arm_normalized,
            "gripper_normalized": gripper_normalized,
            "recorded_at": datetime.now().isoformat(),
            "robot_id": robot_id,
            "config_file": str(config_path),
            "state_type": state_type,
            "description": description
        }

        with open(output_path, 'w') as f:
            json.dump(state_data, f, indent=2)

        print(f"\n  Saved to: {output_path}")
        print(f"  Robot: {robot_id}")
        print(f"  Type: {state_type}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        try:
            robot.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()
