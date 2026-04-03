#!/usr/bin/env python3
"""
Pix2Robot 캘리브레이션 CLI 진입점

사용법:
    python pix2robot_calibrator/run_calibration.py --robot 2
    python pix2robot_calibrator/run_calibration.py --robot 2 --resume  # 기존 데이터에 이어서 수집
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pix2robot_calibrator import Pix2RobotCalibrator


def main():
    parser = argparse.ArgumentParser(
        description="Pix2Robot 캘리브레이션: 픽셀(u,v) → 로봇(x,y,z) 직접 변환"
    )
    parser.add_argument(
        "--robot", type=int, required=True,
        help="로봇 번호 (예: 0, 2, 3)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="기존 캘리브레이션 데이터를 로드하여 이어서 포인트 수집",
    )
    args = parser.parse_args()

    calibrator = Pix2RobotCalibrator(robot_id=args.robot)

    try:
        success = calibrator.calibrate_interactive(resume=args.resume)
        if success:
            print("\n캘리브레이션 완료!")
        else:
            print("\n캘리브레이션 미완료 (종료됨)")
            return 1
    except KeyboardInterrupt:
        print("\n중단됨")
        calibrator.cleanup()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
