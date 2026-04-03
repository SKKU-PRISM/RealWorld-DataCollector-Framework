#!/usr/bin/env python3
"""
Script to add LeRobot v3.0 tag and README to a HuggingFace dataset.
This makes the dataset compatible with smolVLA training.
"""

import json
import argparse
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


def get_dataset_info(repo_id: str) -> dict:
    """Download and parse the info.json from the dataset."""
    info_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/info.json",
        repo_type="dataset",
    )
    with open(info_path, "r") as f:
        return json.load(f)


def create_readme_content(info: dict, repo_id: str) -> str:
    """Create README.md content with proper LeRobot tags."""

    # Extract relevant info
    robot_type = info.get("robot_type", "unknown")
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    fps = info.get("fps", 30)
    codebase_version = info.get("codebase_version", "v3.0")
    features = info.get("features", {})

    # Get camera info
    cameras = [k.replace("observation.images.", "") for k in features.keys()
               if k.startswith("observation.images.")]

    # Get state/action dimensions
    state_shape = features.get("observation.state", {}).get("shape", [])
    action_shape = features.get("action", {}).get("shape", [])

    readme = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- lerobot
- {robot_type}
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

A robot manipulation dataset for imitation learning, compatible with **LeRobot v3.0** format.

- **Robot Type:** {robot_type}
- **Total Episodes:** {total_episodes}
- **Total Frames:** {total_frames}
- **FPS:** {fps}
- **Codebase Version:** {codebase_version}

### Features

| Feature | Shape | Type |
|---------|-------|------|
| observation.state | {state_shape} | float32 |
| action | {action_shape} | float32 |
"""

    for camera in cameras:
        cam_info = features.get(f"observation.images.{camera}", {})
        shape = cam_info.get("shape", [])
        readme += f"| observation.images.{camera} | {shape} | video |\n"

    readme += f"""
## Dataset Structure

[meta/info.json](meta/info.json):
```json
{json.dumps(info, indent=2)}
```

## Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load the dataset
dataset = LeRobotDataset("{repo_id}")

# Access a sample
sample = dataset[0]
print(sample.keys())
```

## Citation

```bibtex
@misc{{DATASET_NAME,
  title = {{DATASET_NAME}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{DATASET_URL}}}}
}}
```
"""

    # Replace placeholders
    dataset_name = repo_id.split("/")[-1]
    readme = readme.replace("DATASET_NAME", dataset_name)
    readme = readme.replace("DATASET_URL", f"https://huggingface.co/datasets/{repo_id}")
    return readme


def add_v3_tag(repo_id: str, dry_run: bool = False):
    """Add v3.0 tag and README to HuggingFace dataset."""

    api = HfApi()

    print(f"Processing dataset: {repo_id}")

    # 1. Get dataset info
    print("Downloading info.json...")
    info = get_dataset_info(repo_id)
    print(f"  - Codebase version: {info.get('codebase_version', 'unknown')}")
    print(f"  - Total episodes: {info.get('total_episodes', 0)}")
    print(f"  - Total frames: {info.get('total_frames', 0)}")

    # Verify it's v3.0 format
    if info.get("codebase_version") != "v3.0":
        print(f"WARNING: Dataset codebase_version is {info.get('codebase_version')}, not v3.0")
        print("The dataset may need to be converted first.")

    # 2. Create README content
    print("Creating README.md...")
    readme_content = create_readme_content(info, repo_id)

    if dry_run:
        print("\n--- README.md content (dry run) ---")
        print(readme_content)
        print("--- End of README.md ---\n")
        return

    # 3. Upload README.md
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add LeRobot dataset card with v3.0 tags",
    )
    print("  README.md uploaded successfully!")

    # 4. Create v3.0 git tag
    print("Creating v3.0 git tag...")
    try:
        # Get the latest commit SHA
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
        main_sha = None
        for branch in refs.branches:
            if branch.name == "main":
                main_sha = branch.target_commit
                break

        if main_sha:
            # Create the tag
            api.create_tag(
                repo_id=repo_id,
                repo_type="dataset",
                tag="v3.0",
                revision=main_sha,
            )
            print(f"  v3.0 tag created at commit {main_sha[:8]}!")
        else:
            print("  Could not find main branch SHA")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  v3.0 tag already exists!")
        else:
            print(f"  Error creating tag: {e}")

    print("\nDone! Dataset is now tagged for LeRobot v3.0 / smolVLA training.")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add LeRobot v3.0 tag to HuggingFace dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Show README content without uploading")

    args = parser.parse_args()
    add_v3_tag(args.repo_id, args.dry_run)
