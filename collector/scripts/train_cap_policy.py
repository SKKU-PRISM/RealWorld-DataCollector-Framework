#!/usr/bin/env python3
"""Train Diffusion Policy on CaP-collected dataset.

Usage:
    python scripts/train_cap_policy.py \
        --dataset cap_dataset_20260106_212506 \
        --steps 50000 \
        --batch-size 32
"""

import sys
sys.path.insert(0, "/home/lerobot/AutoDataCollector/lerobot/src")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on CaP dataset")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (e.g., cap_dataset_20260106_212506)"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Dataset root directory (default: ~/.cache/huggingface/lerobot/local)"
    )
    parser.add_argument(
        "--steps", type=int, default=50000,
        help="Total training steps (default: 50000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/cap_policy",
        help="Output directory (default: outputs/cap_policy)"
    )
    parser.add_argument(
        "--save-freq", type=int, default=5000,
        help="Checkpoint save frequency (default: 5000)"
    )
    parser.add_argument(
        "--log-freq", type=int, default=100,
        help="Log frequency (default: 100)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CaP Dataset → Diffusion Policy Training")
    print("=" * 60)

    # Setup paths
    if args.dataset_root:
        dataset_root = Path(args.dataset_root) / args.dataset
    else:
        dataset_root = Path.home() / ".cache/huggingface/lerobot/local" / args.dataset

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config]")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Dataset path: {dataset_root}")
    print(f"  Output: {output_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    # Load metadata
    print(f"\n[1/5] Loading dataset metadata...")
    repo_id = f"local/{args.dataset}"
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)

    print(f"  Episodes: {metadata.total_episodes}")
    print(f"  Total frames: {metadata.total_frames}")
    print(f"  FPS: {metadata.fps}")
    print(f"  Features: {list(metadata.features.keys())}")

    # Setup policy features
    print(f"\n[2/5] Configuring policy...")
    features = dataset_to_policy_features(metadata.features)

    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    print(f"  Input features: {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")

    # Create policy config
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
    )

    # Setup delta_timestamps
    delta_timestamps = {
        "observation.images.front": [i / metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / metadata.fps for i in cfg.action_delta_indices],
    }

    # Load dataset
    print(f"\n[3/5] Loading dataset...")
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"  Loaded {len(dataset)} samples")

    # Create policy
    print(f"\n[4/5] Initializing policy...")
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create preprocessors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    print(f"  Batches per epoch: {len(dataloader)}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
        print(f"  Resumed at step {start_step}")

    # Training loop
    print(f"\n[5/5] Training...")
    print("-" * 60)

    step = start_step
    epoch = 0
    losses = []
    start_time = time.time()

    training_log = {
        "dataset": args.dataset,
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "losses": [],
    }

    while step < args.steps:
        epoch += 1
        epoch_losses = []

        for batch in dataloader:
            # Preprocess
            batch = preprocessor(batch)

            # Forward
            loss, _ = policy.forward(batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            loss_val = loss.item()
            losses.append(loss_val)
            epoch_losses.append(loss_val)

            # Logging
            if step % args.log_freq == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (args.steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                current_lr = scheduler.get_last_lr()[0]

                print(f"Step {step:6d}/{args.steps} | "
                      f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | "
                      f"LR: {current_lr:.2e} | "
                      f"ETA: {eta/60:.1f}min")

                training_log["losses"].append({
                    "step": step,
                    "loss": loss_val,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                })

            # Save checkpoint
            if step % args.save_freq == 0:
                ckpt_path = output_dir / f"checkpoint_{step:06d}"
                ckpt_path.mkdir(parents=True, exist_ok=True)

                # Save policy
                policy.save_pretrained(ckpt_path)
                preprocessor.save_pretrained(ckpt_path)
                postprocessor.save_pretrained(ckpt_path)

                # Save training state
                torch.save({
                    'step': step,
                    'model': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, ckpt_path / "training_state.pt")

                print(f"  [Checkpoint saved: {ckpt_path}]")

            if step >= args.steps:
                break

        if epoch_losses:
            print(f"  Epoch {epoch} complete | Avg loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

    # Save final model
    print("-" * 60)
    print("\n[Complete] Saving final model...")

    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(final_path)
    preprocessor.save_pretrained(final_path)
    postprocessor.save_pretrained(final_path)

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total steps: {step}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Final model: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
