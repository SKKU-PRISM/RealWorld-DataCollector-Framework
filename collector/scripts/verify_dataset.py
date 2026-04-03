#!/usr/bin/env python3
"""Verify LeRobot dataset integrity and compatibility.

Usage:
    python scripts/verify_dataset.py --dataset cap_dataset_20260106_212506
"""

import sys
sys.path.insert(0, "/home/lerobot/AutoDataCollector/lerobot/src")

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Verify LeRobot dataset")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (e.g., cap_dataset_20260106_212506)"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    return parser.parse_args()


def check_file_exists(path: Path, name: str) -> bool:
    """Check if a file exists and print status."""
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {name}: {path}")
    return exists


def main():
    args = parse_args()

    print("=" * 60)
    print("LeRobot Dataset Verification")
    print("=" * 60)

    # Setup paths
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path.home() / ".cache/huggingface/lerobot/local" / args.dataset

    print(f"\nDataset: {args.dataset}")
    print(f"Path: {dataset_root}")

    # Check directory exists
    if not dataset_root.exists():
        print(f"\n[ERROR] Dataset directory not found: {dataset_root}")
        return 1

    # Check required files
    print(f"\n[1/5] Checking required files...")
    all_files_ok = True

    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    episodes_dir = dataset_root / "meta" / "episodes"
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"

    all_files_ok &= check_file_exists(info_path, "info.json")
    all_files_ok &= check_file_exists(stats_path, "stats.json")
    all_files_ok &= check_file_exists(tasks_path, "tasks.parquet")
    all_files_ok &= check_file_exists(episodes_dir, "episodes/")
    all_files_ok &= check_file_exists(data_dir, "data/")
    all_files_ok &= check_file_exists(videos_dir, "videos/")

    if not all_files_ok:
        print(f"\n[ERROR] Missing required files!")
        return 1

    # Load and verify info.json
    print(f"\n[2/5] Verifying metadata (info.json)...")
    with open(info_path) as f:
        info = json.load(f)

    print(f"  Codebase version: {info.get('codebase_version', 'unknown')}")
    print(f"  Robot type: {info.get('robot_type', 'unknown')}")
    print(f"  Total episodes: {info.get('total_episodes', 0)}")
    print(f"  Total frames: {info.get('total_frames', 0)}")
    print(f"  FPS: {info.get('fps', 0)}")

    # Verify version
    version = info.get('codebase_version', '')
    if not version.startswith('v3'):
        print(f"  [WARNING] Expected v3.x, got {version}")

    # Check features
    print(f"\n[3/5] Verifying features...")
    features = info.get('features', {})

    required_features = ['observation.state', 'action']
    for feat in required_features:
        if feat in features:
            f = features[feat]
            print(f"  [OK] {feat}: dtype={f.get('dtype')}, shape={f.get('shape')}")
        else:
            print(f"  [MISSING] {feat}")
            all_files_ok = False

    # Check for image feature
    image_features = [k for k in features.keys() if 'image' in k.lower()]
    if image_features:
        for feat in image_features:
            f = features[feat]
            print(f"  [OK] {feat}: dtype={f.get('dtype')}, shape={f.get('shape')}")
    else:
        print(f"  [WARNING] No image features found")

    # Verify stats.json
    print(f"\n[4/5] Verifying statistics (stats.json)...")
    with open(stats_path) as f:
        stats = json.load(f)

    for key in ['observation.state', 'action']:
        if key in stats:
            s = stats[key]
            has_required = all(k in s for k in ['min', 'max', 'mean', 'std'])
            status = "[OK]" if has_required else "[INCOMPLETE]"
            print(f"  {status} {key}: min/max/mean/std")
        else:
            print(f"  [MISSING] {key}")

    # Try loading with LeRobot
    print(f"\n[5/5] Testing LeRobot compatibility...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.configs.types import FeatureType

        # Load metadata
        repo_id = f"local/{args.dataset}"
        metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
        print(f"  [OK] LeRobotDatasetMetadata loaded")

        # Convert features
        policy_features = dataset_to_policy_features(metadata.features)
        print(f"  [OK] Policy features converted: {len(policy_features)} features")

        # Check feature types
        for key, feat in policy_features.items():
            print(f"       - {key}: {feat.type.name}, shape={feat.shape}")

        # Load dataset (small test)
        delta_timestamps = {
            "observation.state": [0.0],
            "action": [0.0],
        }
        # Add image feature if exists
        for feat_key in policy_features:
            if policy_features[feat_key].type == FeatureType.VISUAL:
                delta_timestamps[feat_key] = [0.0]
                break

        dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
        print(f"  [OK] LeRobotDataset loaded: {len(dataset)} samples")

        # Test sample access
        sample = dataset[0]
        print(f"  [OK] Sample access successful")

        if args.verbose:
            print(f"\n  Sample keys:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")

    except Exception as e:
        print(f"  [ERROR] LeRobot compatibility test failed: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Episodes: {info.get('total_episodes', 0)}")
    print(f"  Frames: {info.get('total_frames', 0)}")
    print(f"  Status: VALID")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
