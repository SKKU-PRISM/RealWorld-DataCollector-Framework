#!/usr/bin/env python3
"""
VLA LoRA adapter training script.

Trains Move or Grip LoRA adapters on RoboCasa data.
Supports multiple VLA backends (OpenVLA, SmolVLA, pi0.5, GROOT N1, HF VLM)
and GPU profiles (A6000 QLoRA, A100 full precision).

Usage:
    # Train with raw HDF5 data (no preprocessing needed)
    python train_lora.py \
        --config configs/lora_adapter.yaml \
        --model-config smolvla \
        --task PnPCounterToCab \
        --hdf5-dir ~/.robocasa/datasets/v0.1

    # Train with custom overrides
    python train_lora.py \
        --config configs/lora_adapter.yaml \
        --model-config smolvla \
        --task PnPCounterToCab \
        --hdf5-dir ~/.robocasa/datasets/v0.1 \
        --lr 1e-3 --max-steps 2500
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.multiprocessing
import yaml
from PIL import Image as PILImage

# Use file-descriptor sharing instead of /dev/shm (Docker default is 64MB, too small for workers)
torch.multiprocessing.set_sharing_strategy("file_system")

# Speed: TF32 for matmul (A6000/A100 Ampere+), cuDNN autotuner for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Disable cuDNN SDPA and torch.compile/inductor to prevent compile worker spawning
torch.backends.cuda.enable_cudnn_sdp(False)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
import torch._inductor.config
torch._inductor.config.compile_threads = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

# Embodiment-agnostic state: eef_pos(3) + eef_quat(4) + target_pos(3) = 10D
# Indices into the 19D preprocessed robot_state (drops joint_pos[7:14] and gripper_qpos[14:16])
_STATE_KEEP_IDX = list(range(0, 7)) + list(range(16, 19))  # 19D → 10D
# 12D state (eef_pos(3)+eef_quat(4)+gripper(2)+target_pos(3)) → 10D (drop gripper at 7-8)
_STATE_KEEP_IDX_12D = list(range(0, 7)) + list(range(9, 12))
# 11D state (ee_pos(3)+ee_quat(4)+gripper(1)+goal_direction(3)) → 10D (drop gripper at 7)
_STATE_KEEP_IDX_11D = list(range(0, 7)) + list(range(8, 11))
# 9D state (joint_deg(6)+goal_xyz(3)) → 9D (keep all, no filtering needed)
_STATE_KEEP_IDX_9D = list(range(0, 9))


# =============================================================================
# Config Loading
# =============================================================================


def load_config(
    config_path: str,
    gpu_profile: Optional[str] = None,
    model_config_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
) -> Dict:
    """Load and merge configs: base.yaml + adapter config + model config + gpu profile + CLI overrides."""
    # Load base config
    base_path = CONFIGS_DIR / "base.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)

    # Merge adapter config
    with open(config_path) as f:
        adapter_config = yaml.safe_load(f)
    _deep_merge(config, adapter_config)

    # Merge model config (e.g., configs/models/smolvla.yaml)
    if model_config_path:
        # Support both bare name ("smolvla") and full path
        model_cfg_file = CONFIGS_DIR / "models" / f"{model_config_path}.yaml"
        if not model_cfg_file.exists():
            model_cfg_file = Path(model_config_path)
        if model_cfg_file.exists():
            with open(model_cfg_file) as f:
                model_config = yaml.safe_load(f)
            _deep_merge(config, model_config)
            logger.info(f"Applied model config: {model_cfg_file}")
        else:
            logger.warning(f"Model config not found: {model_config_path}")

    # Merge GPU profile
    if gpu_profile:
        profile_path = CONFIGS_DIR / "gpu_profiles" / f"{gpu_profile}.yaml"
        if profile_path.exists():
            with open(profile_path) as f:
                gpu_config = yaml.safe_load(f)
            _deep_merge(config, gpu_config)
            logger.info(f"Applied GPU profile: {gpu_profile}")
        else:
            logger.warning(f"GPU profile not found: {profile_path}")

    # Apply CLI overrides
    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# =============================================================================
# Dataset
# =============================================================================


def _compute_stats_from_processed(data_dir: str, task_filter: str, split: str = "train", state_dims: int = 0):
    """Recompute action/state stats from filtered preprocessed data."""
    from pathlib import Path

    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob(f"**/{split}/*.npz"))
    if not npz_files:
        npz_files = sorted(data_path.glob("*.npz"))

    all_actions = []
    all_states = []
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("task_name") != task_filter:
            continue

        data = np.load(npz_path)
        if "actions" in data:
            all_actions.append(data["actions"].astype(np.float32))
        if "robot_state" in data:
            s = data["robot_state"].astype(np.float32)
            if state_dims > 0:
                s = s[:, :state_dims]
            all_states.append(s)

    if not all_actions:
        raise ValueError(f"No data found for task '{task_filter}' in {data_dir}/{split}")

    all_actions = np.concatenate(all_actions, axis=0)
    action_stats = {
        "mean": np.mean(all_actions, axis=0).tolist(),
        "std": np.clip(np.std(all_actions, axis=0), 1e-6, None).tolist(),
        "min": np.min(all_actions, axis=0).tolist(),
        "max": np.max(all_actions, axis=0).tolist(),
        "mode": "min_max",
    }
    quantile_stats = {
        "q01": np.percentile(all_actions, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions, 99, axis=0).tolist(),
    }

    state_stats = None
    if all_states:
        all_states = np.concatenate(all_states, axis=0)
        state_stats = {
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.clip(np.std(all_states, axis=0), 1e-6, None).tolist(),
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
            "q01": np.percentile(all_states, 1, axis=0).tolist(),
            "q99": np.percentile(all_states, 99, axis=0).tolist(),
        }

    return action_stats, state_stats, quantile_stats


class VLADataset:
    """PyTorch Dataset for VLA LoRA training.

    Loads preprocessed .npz files with images, states, actions.
    Supports both flat structure (data_dir/*.npz) and hierarchical structure
    (data_dir/**/split/*.npz for v0.1_processed layout).
    """

    def __init__(
        self,
        data_dir: str,
        action_stats: Dict,
        task_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
        image_size: int = 224,
        split: Optional[str] = None,
        use_primitive_instructions: bool = True,
        state_dims: int = 0,
    ):
        import torch
        from torch.utils.data import Dataset

        self.data_dir = data_dir
        self.image_size = image_size
        self.torch = torch
        self.split = split
        self.use_primitive_instructions = use_primitive_instructions
        self._state_dims = state_dims

        # Load action stats for normalization
        self.action_mean = np.array(action_stats["mean"], dtype=np.float32)
        self.action_std = np.clip(
            np.array(action_stats["std"], dtype=np.float32), 1e-6, None
        )

        # Find all .npz files
        self.samples = []
        self._load_file_list(data_dir, task_filter, max_samples, split)
        logger.info(f"Dataset: {len(self.samples)} demos from {data_dir} (split={split})")

    def _load_file_list(
        self,
        data_dir: str,
        task_filter: Optional[str],
        max_samples: Optional[int],
        split: Optional[str] = None,
    ) -> None:
        """Build list of (npz_path, meta_path) tuples."""
        data_path = Path(data_dir)

        # Try flat structure first (legacy)
        npz_files = sorted(data_path.glob("*.npz"))

        # If no flat files, try hierarchical structure
        if not npz_files and split:
            npz_files = sorted(data_path.glob(f"**/{split}/*.npz"))

        for npz_path in npz_files:
            meta_path = npz_path.with_suffix(".json")
            if not meta_path.exists():
                continue

            # Filter by task
            if task_filter:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("task_name") != task_filter:
                    continue

            self.samples.append((str(npz_path), str(meta_path)))

            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def get_meta(self, idx: int) -> Dict:
        """Get metadata for a demo (reads from JSON sidecar)."""
        _, meta_path = self.samples[idx]
        with open(meta_path) as f:
            return json.load(f)

    def load_demo(self, idx: int) -> Dict:
        """Load a full demo's data."""
        npz_path, meta_path = self.samples[idx]

        data = np.load(npz_path)
        with open(meta_path) as f:
            meta = json.load(f)

        primitive_instructions = meta.get("primitive_instructions") if self.use_primitive_instructions else None

        state = data["robot_state"]
        if self._state_dims > 0:
            state = state[:, :self._state_dims]
        result = {
            "images": data["images"],
            "robot_state": state,
            "actions": data["actions"],
            "instruction": meta["instruction"],
            "primitive_instructions": primitive_instructions,
            "task_name": meta["task_name"],
        }
        if "wrist_images" in data:
            result["wrist_images"] = data["wrist_images"]
        return result


class RawHDF5Dataset:
    """Dataset for reading RoboCasa HDF5 files directly without preprocessing.

    Reads raw HDF5 structure and computes stats on-the-fly.
    Compatible with VLADataset interface.
    """

    def __init__(
        self,
        hdf5_root: str,
        task_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
        image_size: int = 224,
        split: str = "train",
    ):
        import torch
        from PIL import Image as PILImage

        self.hdf5_root = hdf5_root
        self.image_size = image_size
        self.torch = torch
        self.PILImage = PILImage
        self.split = split

        # Find all HDF5 files
        self._load_demos(hdf5_root, task_filter, max_samples, split)

        # Compute stats (quick pass without loading images)
        self._compute_stats()

        logger.info(f"RawHDF5Dataset: {len(self._demos)} demos from {hdf5_root} (split={split})")

    def _load_demos(self, hdf5_root: str, task_filter: Optional[str], max_samples: Optional[int], split: str):
        """Load list of HDF5 files and split train/val per file."""
        import random

        hdf5_path = Path(hdf5_root)
        hdf5_files = sorted(hdf5_path.glob("**/*.hdf5"))

        self._demos = []

        for ds_idx, hdf5_file in enumerate(hdf5_files):
            # Extract task name from path
            parts = hdf5_file.parts
            task_name = None
            for i, part in enumerate(parts):
                if part in ("single_stage", "multi_stage") and i + 2 < len(parts):
                    task_name = parts[i + 2]
                    break

            if not task_name:
                task_name = hdf5_file.stem

            # Filter by task
            if task_filter and task_name != task_filter:
                continue

            # Open HDF5 and split demos
            with h5py.File(hdf5_file, "r") as f:
                demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]

                # Split train/val at demo level (90/10 split, seed=42+ds_idx)
                rng = random.Random(42 + ds_idx)
                rng.shuffle(demo_keys)
                val_count = max(1, int(len(demo_keys) * 0.1))

                if split == "train":
                    selected_keys = demo_keys[val_count:]
                else:
                    selected_keys = demo_keys[:val_count]

                for demo_key in selected_keys:
                    ep_meta_str = f["data"][demo_key].attrs.get("ep_meta", "{}")
                    ep_meta = json.loads(ep_meta_str)
                    instruction = ep_meta.get("lang", "")

                    self._demos.append((str(hdf5_file), demo_key, task_name, instruction))

                    if max_samples and len(self._demos) >= max_samples:
                        return

    def _compute_stats(self):
        """Compute action_stats, state_stats, action_quantile_stats."""
        all_actions = []
        all_states = []

        for hdf5_path, demo_key, _, _ in self._demos:
            with h5py.File(hdf5_path, "r") as f:
                demo = f["data"][demo_key]

                actions = demo["actions"][:]
                all_actions.append(actions)

                eef_pos = demo["obs"]["robot0_eef_pos"][:]
                eef_quat = demo["obs"]["robot0_eef_quat"][:]
                joint_pos = demo["obs"]["robot0_joint_pos"][:]
                gripper = demo["obs"]["robot0_gripper_qpos"][:]

                state_16d = np.concatenate([eef_pos, eef_quat, joint_pos, gripper], axis=-1)
                all_states.append(state_16d)

        all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        all_states = np.concatenate(all_states, axis=0).astype(np.float32)

        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.clip(np.std(all_actions, axis=0), 1e-6, None)

        self.action_stats = {
            "mean": self.action_mean.tolist(),
            "std": self.action_std.tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
        }

        self.state_stats = {
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.std(all_states, axis=0).tolist(),
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
            "q01": np.percentile(all_states, 0.5, axis=0).tolist(),
            "q99": np.percentile(all_states, 99.5, axis=0).tolist(),
        }

        self.action_quantile_stats = {
            "q01": np.percentile(all_actions, 0.5, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99.5, axis=0).tolist(),
        }

    def __len__(self) -> int:
        return len(self._demos)

    @property
    def samples(self):
        return [(i, i) for i in range(len(self._demos))]

    def get_meta(self, idx: int) -> Dict:
        hdf5_path, demo_key, task_name, instruction = self._demos[idx]
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            n_steps = demo["actions"].shape[0]
        return {
            "n_steps": n_steps,
            "instruction": instruction,
            "task_name": task_name,
            "primitive_instructions": None,
        }

    def load_demo(self, idx: int) -> Dict:
        hdf5_path, demo_key, task_name, instruction = self._demos[idx]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]

            if "robot0_agentview_left_image" in obs:
                images = obs["robot0_agentview_left_image"][:]
            elif "robot0_agentview_right_image" in obs:
                images = obs["robot0_agentview_right_image"][:]
            elif "robot0_eye_in_hand_image" in obs:
                images = obs["robot0_eye_in_hand_image"][:]
            else:
                raise ValueError(f"No supported camera found in {hdf5_path}/{demo_key}")

            if images.shape[1:3] != (self.image_size, self.image_size):
                resized = []
                for img in images:
                    pil_img = self.PILImage.fromarray(img)
                    pil_img = pil_img.resize((self.image_size, self.image_size), self.PILImage.BILINEAR)
                    resized.append(np.array(pil_img))
                images = np.stack(resized, axis=0)

            eef_pos = obs["robot0_eef_pos"][:]
            eef_quat = obs["robot0_eef_quat"][:]
            joint_pos = obs["robot0_joint_pos"][:]
            gripper = obs["robot0_gripper_qpos"][:]

            robot_state = np.concatenate([eef_pos, eef_quat, joint_pos, gripper], axis=-1).astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)

        return {
            "images": images,
            "robot_state": robot_state,
            "actions": actions,
            "instruction": instruction,
            "primitive_instructions": None,
            "task_name": task_name,
        }


class VLAStepDataset:
    """Per-timestep dataset. Lazily loads demos and yields individual timesteps."""

    def __init__(self, demo_dataset, cache_size: int = 0):
        import torch
        from functools import lru_cache

        self.demo_dataset = demo_dataset
        self.torch = torch

        self.index = []
        self.demo_ranges = []
        offset = 0
        for demo_idx in range(len(demo_dataset)):
            meta = demo_dataset.get_meta(demo_idx)
            n_steps = meta["n_steps"]
            for step_idx in range(n_steps):
                self.index.append((demo_idx, step_idx))
            self.demo_ranges.append((offset, offset + n_steps))
            offset += n_steps

        logger.info(f"Step dataset: {len(self.index)} timesteps from {len(demo_dataset)} demos")

        # cache_size=0 means cache all demos in memory
        effective_cache = cache_size if cache_size > 0 else len(demo_dataset)

        @lru_cache(maxsize=effective_cache)
        def _cached_load(demo_idx):
            return demo_dataset.load_demo(demo_idx)

        self._cached_load = _cached_load

        # Pre-load all demos into cache
        if cache_size == 0:
            logger.info(f"Pre-loading all {len(demo_dataset)} demos into memory...")
            for i in range(len(demo_dataset)):
                _cached_load(i)
            logger.info("Pre-loading done")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        demo_idx, step_idx = self.index[idx]
        data = self._cached_load(demo_idx)

        image = data["images"][step_idx].astype(np.float32) / 255.0
        state = data["robot_state"][step_idx]
        action = data["actions"][step_idx]

        action_norm = (action - self.demo_dataset.action_mean) / self.demo_dataset.action_std

        prim_instructions = data.get("primitive_instructions")
        if prim_instructions and step_idx < len(prim_instructions):
            instruction = prim_instructions[step_idx]
        else:
            instruction = data["instruction"]

        return {
            "image": self.torch.from_numpy(image).permute(2, 0, 1),
            "robot_state": self.torch.from_numpy(state),
            "action": self.torch.from_numpy(action_norm),
            "instruction": instruction,
        }


class ActionChunkDataset:
    """Chunk-based dataset for action-chunk models (SmolVLA, PI0.5, GROOT)."""

    def __init__(self, demo_dataset, chunk_size: int, stride: int = 1, cache_size: int = 0):
        import torch
        from functools import lru_cache

        self.demo_dataset = demo_dataset
        self.chunk_size = chunk_size
        self.torch = torch

        self.index = []
        self.demo_ranges = []
        offset = 0
        for demo_idx in range(len(demo_dataset)):
            meta = demo_dataset.get_meta(demo_idx)
            n_steps = meta["n_steps"]
            demo_start = offset
            for start in range(0, n_steps, stride):
                self.index.append((demo_idx, start))
                offset += 1
            self.demo_ranges.append((demo_start, offset))

        logger.info(
            f"ActionChunkDataset: {len(self.index)} chunks (chunk_size={chunk_size}, stride={stride}) "
            f"from {len(demo_dataset)} demos"
        )

        # cache_size=0 means cache all demos in memory
        effective_cache = cache_size if cache_size > 0 else len(demo_dataset)

        @lru_cache(maxsize=effective_cache)
        def _cached_load(demo_idx):
            return demo_dataset.load_demo(demo_idx)

        self._cached_load = _cached_load

        # Pre-load all demos into cache
        if cache_size == 0:
            logger.info(f"Pre-loading all {len(demo_dataset)} demos into memory...")
            for i in range(len(demo_dataset)):
                _cached_load(i)
            logger.info("Pre-loading done")

    def __len__(self) -> int:
        return len(self.index)

    def _get_action_chunk(self, actions: np.ndarray, start_step: int):
        total_steps = actions.shape[0]
        end_step = min(start_step + self.chunk_size, total_steps)
        chunk = actions[start_step:end_step]
        actual_len = chunk.shape[0]

        if actual_len < self.chunk_size:
            pad_len = self.chunk_size - actual_len
            last_action = chunk[-1:]
            padding = np.repeat(last_action, pad_len, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[actual_len:] = True

        return chunk.astype(np.float32), is_pad

    def __getitem__(self, idx: int) -> Dict:
        demo_idx, start_step = self.index[idx]
        data = self._cached_load(demo_idx)

        action_chunk, action_pad_mask = self._get_action_chunk(data["actions"], start_step)
        image = data["images"][start_step].astype(np.float32) / 255.0

        wrist_image = None
        if "wrist_images" in data and data["wrist_images"] is not None:
            wrist_image = data["wrist_images"][start_step].astype(np.float32) / 255.0

        robot_state = data["robot_state"][start_step].astype(np.float32)

        prim_instructions = data.get("primitive_instructions")
        if prim_instructions and start_step < len(prim_instructions):
            instruction = prim_instructions[start_step]
        else:
            instruction = data["instruction"]

        result = {
            "images": self.torch.from_numpy(image).permute(2, 0, 1),
            "robot_state": self.torch.from_numpy(robot_state),
            "action_chunk": self.torch.from_numpy(action_chunk),
            "action_pad_mask": self.torch.from_numpy(action_pad_mask),
            "instruction": instruction,
            "step_idx": start_step,
        }

        if wrist_image is not None:
            result["wrist_images"] = self.torch.from_numpy(wrist_image).permute(2, 0, 1)

        return result


class SmolVLAChunkDataset:
    """SmolVLA-compatible dataset adapter."""

    def __init__(self, chunk_dataset, state_stats: Dict, action_stats: Dict, image_size: int = 512):
        from transformers import AutoTokenizer
        import torch
        import torch.nn.functional as F

        self.chunk_ds = chunk_dataset
        self.torch = torch
        self.F = F
        self.image_size = image_size

        _mean = np.array(state_stats["mean"], dtype=np.float32)
        _std = np.array(state_stats["std"], dtype=np.float32)
        if len(_mean) == 19:
            self.state_mean = _mean[_STATE_KEEP_IDX]
            self.state_std = _std[_STATE_KEEP_IDX]
        elif len(_mean) == 12:
            self.state_mean = _mean[_STATE_KEEP_IDX_12D]
            self.state_std = _std[_STATE_KEEP_IDX_12D]
        elif len(_mean) == 11:
            self.state_mean = _mean[_STATE_KEEP_IDX_11D]
            self.state_std = _std[_STATE_KEEP_IDX_11D]
        else:
            self.state_mean = _mean
            self.state_std = _std
        self._state_dim_in = len(_mean)
        self.action_mean = np.array(action_stats["mean"], dtype=np.float32)
        self.action_std = np.array(action_stats["std"], dtype=np.float32)

        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)

        self._instruction_cache = {}
        self._build_instruction_cache()

        logger.info(f"SmolVLAChunkDataset: {len(chunk_dataset)} samples, {len(self._instruction_cache)} unique instructions")

    def _build_instruction_cache(self):
        instructions = set()
        for demo_idx in range(len(self.chunk_ds.demo_dataset)):
            meta = self.chunk_ds.demo_dataset.get_meta(demo_idx)
            instructions.add(meta["instruction"])

        for inst in instructions:
            text = inst + "\n"
            tokens = self.tokenizer(
                text, max_length=48, padding="max_length", truncation=True, return_tensors="pt"
            )
            self._instruction_cache[inst] = {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

    def __len__(self):
        return len(self.chunk_ds)

    @property
    def demo_ranges(self):
        return self.chunk_ds.demo_ranges

    def __getitem__(self, idx):
        sample = self.chunk_ds[idx]

        img = sample["images"]
        img = self.F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)

        state = sample["robot_state"].numpy()
        if len(state) == 12:
            state = state[_STATE_KEEP_IDX_12D]  # 12D → 10D
        elif len(state) == 11:
            state = state[_STATE_KEEP_IDX_11D]  # 11D → 10D
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        action = sample["action_chunk"].numpy()
        action_norm = (action - self.action_mean) / (self.action_std + 1e-8)

        inst = sample["instruction"]
        if inst in self._instruction_cache:
            tokens = self._instruction_cache[inst]
        else:
            text = inst + "\n"
            tok = self.tokenizer(text, max_length=48, padding="max_length", truncation=True, return_tensors="pt")
            tokens = {"input_ids": tok["input_ids"].squeeze(0), "attention_mask": tok["attention_mask"].squeeze(0)}
            self._instruction_cache[inst] = tokens

        return {
            "observation.images.camera1": img,
            "observation.state": self.torch.from_numpy(state_norm.astype(np.float32)),
            "observation.language.tokens": tokens["input_ids"],
            "observation.language.attention_mask": tokens["attention_mask"].bool(),
            "action": self.torch.from_numpy(action_norm.astype(np.float32)),
            "actions_is_pad": sample["action_pad_mask"],
            "_sample_idx": self.torch.tensor(idx, dtype=self.torch.long),
        }


class PI05ChunkDataset:
    """PI0.5-compatible dataset adapter."""

    def __init__(self, chunk_dataset, state_stats: Dict, action_stats: Dict, image_size: int = 224):
        from transformers import AutoTokenizer
        import torch
        import torch.nn.functional as F

        self.chunk_ds = chunk_dataset
        self.torch = torch
        self.F = F
        self.image_size = image_size

        _q01 = np.array(state_stats["q01"], dtype=np.float32)
        _q99 = np.array(state_stats["q99"], dtype=np.float32)
        if len(_q01) == 19:
            self.state_q01 = _q01[_STATE_KEEP_IDX]
            self.state_q99 = _q99[_STATE_KEEP_IDX]
        elif len(_q01) == 12:
            self.state_q01 = _q01[_STATE_KEEP_IDX_12D]
            self.state_q99 = _q99[_STATE_KEEP_IDX_12D]
        elif len(_q01) == 11:
            self.state_q01 = _q01[_STATE_KEEP_IDX_11D]
            self.state_q99 = _q99[_STATE_KEEP_IDX_11D]
        else:
            self.state_q01 = _q01
            self.state_q99 = _q99

        self.action_q01 = np.array(action_stats["q01"], dtype=np.float32)
        self.action_q99 = np.array(action_stats["q99"], dtype=np.float32)

        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)

        self._instruction_set = set()
        for demo_idx in range(len(self.chunk_ds.demo_dataset)):
            meta = self.chunk_ds.demo_dataset.get_meta(demo_idx)
            self._instruction_set.add(meta["instruction"])

        logger.info(f"PI05ChunkDataset: {len(chunk_dataset)} samples, {len(self._instruction_set)} unique instructions")

    def _discretize_state(self, state: np.ndarray) -> str:
        state_norm = (state - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-8) * 2 - 1
        state_norm = np.clip(state_norm, -1, 1)

        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm

        bins = np.linspace(-1, 1, 257)[:-1]
        discretized = np.digitize(state_padded, bins) - 1
        discretized = np.clip(discretized, 0, 255)

        return " ".join(map(str, discretized))

    def __len__(self):
        return len(self.chunk_ds)

    @property
    def demo_ranges(self):
        return self.chunk_ds.demo_ranges

    def __getitem__(self, idx):
        sample = self.chunk_ds[idx]

        img = sample["images"]
        # Resize to model's expected resolution
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            img = self.F.interpolate(
                img.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        state = sample["robot_state"].numpy()
        if len(state) == 12:
            state = state[_STATE_KEEP_IDX_12D]  # 12D → 10D
        elif len(state) == 11:
            state = state[_STATE_KEEP_IDX_11D]  # 11D → 10D
        state_str = self._discretize_state(state)
        prompt = f"Task: {sample['instruction']}, State: {state_str};\nAction: "

        tokens = self.tokenizer(
            prompt, max_length=200, padding="max_length", truncation=True, return_tensors="pt"
        )

        action = sample["action_chunk"].numpy()
        action_norm = (action - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-8) * 2 - 1
        action_norm = np.clip(action_norm, -1, 1)

        return {
            "observation.images.base_0_rgb": img,
            "observation.language.tokens": tokens["input_ids"].squeeze(0),
            "observation.language.attention_mask": tokens["attention_mask"].squeeze(0).bool(),
            "action": self.torch.from_numpy(action_norm.astype(np.float32)),
        }


class PI05TensorStateChunkDataset:
    """PI0.5 tensor-state dataset: state as normalized tensor, no text discretization.

    Instead of encoding state as "State: 128 64 ..." in the prompt,
    passes state as a continuous 32D tensor via observation.state.
    Prompt is simplified to "Task: {instruction};\nAction: ".
    """

    def __init__(self, chunk_dataset, state_stats: Dict, action_stats: Dict, image_size: int = 224):
        from transformers import AutoTokenizer
        import torch
        import torch.nn.functional as F

        self.chunk_ds = chunk_dataset
        self.torch = torch
        self.F = F
        self.image_size = image_size

        # Quantile stats for state normalization (q01/q99 -> [-1,1])
        _q01 = np.array(state_stats["q01"], dtype=np.float32)
        _q99 = np.array(state_stats["q99"], dtype=np.float32)
        if len(_q01) == 19:
            self.state_q01 = _q01[_STATE_KEEP_IDX]
            self.state_q99 = _q99[_STATE_KEEP_IDX]
        elif len(_q01) == 12:
            self.state_q01 = _q01[_STATE_KEEP_IDX_12D]
            self.state_q99 = _q99[_STATE_KEEP_IDX_12D]
        elif len(_q01) == 11:
            self.state_q01 = _q01[_STATE_KEEP_IDX_11D]
            self.state_q99 = _q99[_STATE_KEEP_IDX_11D]
        else:
            self.state_q01 = _q01
            self.state_q99 = _q99

        # Quantile stats for action normalization
        self.action_q01 = np.array(action_stats["q01"], dtype=np.float32)
        self.action_q99 = np.array(action_stats["q99"], dtype=np.float32)

        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)

        self._instruction_set = set()
        for demo_idx in range(len(self.chunk_ds.demo_dataset)):
            meta = self.chunk_ds.demo_dataset.get_meta(demo_idx)
            self._instruction_set.add(meta["instruction"])

        logger.info(
            f"PI05TensorStateChunkDataset: {len(chunk_dataset)} samples, "
            f"{len(self._instruction_set)} unique instructions (tensor state mode)"
        )

    def __len__(self):
        return len(self.chunk_ds)

    @property
    def demo_ranges(self):
        return self.chunk_ds.demo_ranges

    def __getitem__(self, idx):
        sample = self.chunk_ds[idx]

        img = sample["images"]
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            img = self.F.interpolate(
                img.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # State: quantile normalize to [-1,1], pad to 32D tensor
        state = sample["robot_state"].numpy()
        if len(state) == 12:
            state = state[_STATE_KEEP_IDX_12D]  # 12D -> 10D
        elif len(state) == 11:
            state = state[_STATE_KEEP_IDX_11D]  # 11D -> 10D
        state_norm = (state - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-8) * 2 - 1
        state_norm = np.clip(state_norm, -1, 1)
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm

        # Prompt: no state text (state is in tensor)
        prompt = f"Task: {sample['instruction']};\nAction: "

        tokens = self.tokenizer(
            prompt, max_length=200, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Action: quantile normalize to [-1,1]
        action = sample["action_chunk"].numpy()
        action_norm = (action - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-8) * 2 - 1
        action_norm = np.clip(action_norm, -1, 1)

        return {
            "observation.images.base_0_rgb": img,
            "observation.language.tokens": tokens["input_ids"].squeeze(0),
            "observation.language.attention_mask": tokens["attention_mask"].squeeze(0).bool(),
            "observation.state": self.torch.from_numpy(state_padded),
            "action": self.torch.from_numpy(action_norm.astype(np.float32)),
            "_sample_idx": self.torch.tensor(idx, dtype=self.torch.long),
        }


class GrootChunkDataset:
    """GROOT N1.5-compatible dataset adapter."""

    def __init__(self, chunk_dataset, state_stats: Dict, action_stats: Dict,
                 use_wrist: bool = False, eagle_processor=None, precomputed_eagle: bool = False):
        import torch

        self.chunk_ds = chunk_dataset
        self.torch = torch
        self.use_wrist = use_wrist
        self.eagle_processor = eagle_processor
        self.precomputed_eagle = precomputed_eagle
        self._eagle_cache = {}

        if precomputed_eagle:
            self._load_precomputed_eagle()

        _smin = np.array(state_stats["min"], dtype=np.float32)
        _smax = np.array(state_stats["max"], dtype=np.float32)
        if len(_smin) == 19:
            # 19D → 12D: keep eef_pos(0-6), gripper(14-15), target_pos(16-18)
            _idx = list(range(0, 7)) + list(range(14, 19))
            self.state_min = _smin[_idx]
            self.state_max = _smax[_idx]
        else:
            # 12D/11D stats: use as-is (matches preprocessed data)
            self.state_min = _smin
            self.state_max = _smax
        self.action_min = np.array(action_stats["min"], dtype=np.float32)
        self.action_max = np.array(action_stats["max"], dtype=np.float32)

        self.embodiment_id = 31

        logger.info(f"GrootChunkDataset: {len(chunk_dataset)} samples (eagle_in_worker={eagle_processor is not None}, precomputed={precomputed_eagle})")

    def _load_precomputed_eagle(self):
        """Load precomputed eagle features for all demos."""
        demo_ds = self.chunk_ds.demo_dataset
        for i in range(len(demo_ds)):
            npz_path = demo_ds.samples[i][0]
            eagle_path = npz_path.replace(".npz", ".eagle.npz")
            if os.path.exists(eagle_path):
                data = np.load(eagle_path, mmap_mode='r')
                self._eagle_cache[i] = {
                    "input_ids": data["input_ids"],
                    "attention_mask": data["attention_mask"],
                    "pixel_values": data["pixel_values"],
                    "image_sizes": data["image_sizes"] if "image_sizes" in data else None,
                }
            else:
                logger.warning(f"Precomputed eagle not found: {eagle_path}")
        logger.info(f"Loaded precomputed eagle for {len(self._eagle_cache)} demos")

    def __len__(self):
        return len(self.chunk_ds)

    @property
    def demo_ranges(self):
        return self.chunk_ds.demo_ranges

    def __getitem__(self, idx):
        sample = self.chunk_ds[idx]

        state = sample["robot_state"].numpy()
        state_range = self.state_max - self.state_min + 1e-8
        state_norm = 2 * (state - self.state_min) / state_range - 1
        state_norm = np.clip(state_norm, -1, 1)

        state_padded = np.zeros(64, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm
        state_mask = np.zeros(64, dtype=bool)
        state_mask[:len(state_norm)] = True

        action = sample["action_chunk"].numpy()  # (chunk_size, action_dim)
        action_dim = action.shape[1]
        action_range = self.action_max - self.action_min + 1e-8
        action_norm = 2 * (action - self.action_min) / action_range - 1
        action_norm = np.clip(action_norm, -1, 1)

        chunk_len = action.shape[0]
        action_padded = np.zeros((chunk_len, 32), dtype=np.float32)
        action_padded[:, :action_dim] = action_norm
        action_mask = np.zeros((chunk_len, 32), dtype=bool)
        action_mask[:, :action_dim] = True
        if sample["action_pad_mask"].any():
            pad_start = (~sample["action_pad_mask"].numpy()).sum()
            action_mask[pad_start:, :] = False

        img = sample["images"].numpy()  # float [0, 1], (C, H, W)
        if self.use_wrist and "wrist_images" in sample and sample["wrist_images"] is not None:
            wrist_img = sample["wrist_images"].numpy()  # (C, H, W)
            video = np.stack([img, wrist_img], axis=0)[np.newaxis, ...]  # (1, 2, C, H, W)
        else:
            video = img[np.newaxis, np.newaxis, ...]  # (1, 1, C, H, W) float

        result = {
            "state": self.torch.from_numpy(state_padded).unsqueeze(0),
            "state_mask": self.torch.from_numpy(state_mask).unsqueeze(0),
            "action": self.torch.from_numpy(action_padded),
            "action_mask": self.torch.from_numpy(action_mask),
            "embodiment_id": self.torch.tensor(self.embodiment_id, dtype=self.torch.long),
        }

        # Prepare eagle features
        if self.precomputed_eagle:
            # Use precomputed eagle features — no CPU processing needed
            demo_idx, start_step = self.chunk_ds.index[idx]
            eagle_data = self._eagle_cache[demo_idx]
            result["eagle_input_ids"] = self.torch.from_numpy(eagle_data["input_ids"][start_step].copy())
            result["eagle_attention_mask"] = self.torch.from_numpy(eagle_data["attention_mask"][start_step].copy())
            result["eagle_pixel_values"] = self.torch.from_numpy(eagle_data["pixel_values"][start_step].copy())
            if eagle_data["image_sizes"] is not None:
                result["eagle_image_sizes"] = self.torch.from_numpy(eagle_data["image_sizes"][start_step].copy())
        elif self.eagle_processor is not None:
            from PIL import Image as PILImage

            # Build conversation format: each camera as a separate image
            content = []
            for t in range(video.shape[0]):
                for c in range(video.shape[1]):
                    img_np = (video[t, c].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    content.append({"type": "image", "image": PILImage.fromarray(img_np)})
            content.append({"type": "text", "text": sample["instruction"]})

            conversation = [{"role": "user", "content": content}]
            text_list = self.eagle_processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = self.eagle_processor.process_vision_info(conversation)

            result["eagle_content"] = {
                "text_list": [text_list],
                "image_inputs": image_inputs,
            }
        else:
            result["video"] = video
            result["language"] = sample["instruction"]

        return result


def groot_collate_fn(batch, eagle_processor=None):
    """Collate function for GROOT that handles Eagle encoding.

    If eagle keys are already in batch items (pre-encoded in worker),
    just stack them. Otherwise fall back to encoding here.
    """
    import torch

    result = {
        "state": torch.stack([b["state"] for b in batch]),
        "state_mask": torch.stack([b["state_mask"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "action_mask": torch.stack([b["action_mask"] for b in batch]),
        "embodiment_id": torch.stack([b["embodiment_id"] for b in batch]),
    }

    # Check if precomputed eagle features are available (fastest path)
    if "eagle_input_ids" in batch[0]:
        result["eagle_input_ids"] = torch.stack([b["eagle_input_ids"] for b in batch])
        result["eagle_attention_mask"] = torch.stack([b["eagle_attention_mask"] for b in batch])
        # pixel_values: (n_images, 3, H, W) per sample → concat along dim 0
        result["eagle_pixel_values"] = torch.cat([b["eagle_pixel_values"] for b in batch], dim=0)
        if "eagle_image_sizes" in batch[0]:
            result["eagle_image_sizes"] = torch.cat([b["eagle_image_sizes"] for b in batch], dim=0)
        return result

    # Check if eagle_content was prepared in worker (official GROOT multi-camera method)
    if "eagle_content" in batch[0]:
        text_list = []
        image_inputs = []
        for b in batch:
            ec = b["eagle_content"]
            text_list += ec["text_list"]
            image_inputs += ec["image_inputs"]
        eagle_inputs = eagle_processor(
            text=text_list, images=image_inputs, return_tensors="pt", padding=True,
        )
        for k, v in eagle_inputs.items():
            result["eagle_" + k] = v
    elif eagle_processor is not None:
        # Fallback: single-image legacy path
        from PIL import Image
        import numpy as np

        images = []
        texts = []
        for b in batch:
            video = b["video"]
            for t in range(video.shape[0]):
                for c in range(video.shape[1]):
                    img = video[t, c].transpose(1, 2, 0)
                    img_np = (img * 255).clip(0, 255).astype(np.uint8)
                    images.append(Image.fromarray(img_np))
            n_cams = video.shape[1]
            img_tags = " ".join(f"<image-{i+1}>" for i in range(n_cams))
            texts.append(f"{img_tags} {b['language']}")

        eagle_inputs = eagle_processor(
            text=texts,
            images=images,
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )
        for k, v in eagle_inputs.items():
            result["eagle_" + k] = v
    else:
        result["video"] = [b["video"] for b in batch]
        result["language"] = [b["language"] for b in batch]

    return result


class DemoGroupedBatchSampler:
    """Batch sampler that groups timesteps from the same demo for cache efficiency."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        import random

        demo_order = list(range(len(self.dataset.demo_ranges)))
        if self.shuffle:
            rng = random.Random(self._epoch)
            rng.shuffle(demo_order)

        # Collect all indices in demo-grouped order (same on all ranks)
        all_indices = []
        step_rng = random.Random(self._epoch)
        for demo_idx in demo_order:
            start, end = self.dataset.demo_ranges[demo_idx]
            step_indices = list(range(start, end))
            if self.shuffle:
                step_rng.shuffle(step_indices)
            all_indices.extend(step_indices)

        # Split samples evenly across ranks
        if self.world_size > 1:
            per_rank = len(all_indices) // self.world_size
            offset = self.rank * per_rank
            all_indices = all_indices[offset:offset + per_rank]

        # Yield full batches only
        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        n_samples = len(self.dataset)
        if self.world_size > 1:
            n_samples = n_samples // self.world_size
        return n_samples // self.batch_size


class OpenVLAStepDataset:
    """OpenVLA-specific step dataset with preprocessing in DataLoader workers."""

    IGNORE_INDEX = -100

    def __init__(self, step_dataset, processor, bin_centers, vocab_size):
        self.step_dataset = step_dataset
        self.demo_ranges = step_dataset.demo_ranges
        self.processor = processor
        self.bin_centers = (
            bin_centers.astype(np.float32)
            if hasattr(bin_centers, "astype")
            else np.array(bin_centers, dtype=np.float32)
        )
        self.vocab_size = int(vocab_size)

        self._token_cache = {}
        self._build_instruction_cache()

    def _build_instruction_cache(self):
        demo_ds = self.step_dataset.demo_dataset
        unique_instructions = set()

        for demo_idx in range(len(demo_ds)):
            meta = demo_ds.get_meta(demo_idx)
            unique_instructions.add(meta["instruction"])
            prim = meta.get("primitive_instructions")
            if prim:
                unique_instructions.update(prim)

        for instr in unique_instructions:
            prompt = f"In: What action should the robot take to {instr}?\nOut:"
            input_ids = self.processor.tokenizer(
                prompt, return_tensors="pt"
            )["input_ids"].squeeze(0)
            self._token_cache[instr] = input_ids

        logger.info(
            f"Pre-tokenized {len(self._token_cache)} unique instructions for OpenVLA"
        )

    def __len__(self):
        return len(self.step_dataset)

    def __getitem__(self, idx):
        demo_idx, step_idx = self.step_dataset.index[idx]
        data = self.step_dataset._cached_load(demo_idx)

        image = data["images"][step_idx]
        action = data["actions"][step_idx]

        demo_ds = self.step_dataset.demo_dataset
        action_norm = (
            action.astype(np.float32) - demo_ds.action_mean
        ) / demo_ds.action_std

        prim = data.get("primitive_instructions")
        if prim and step_idx < len(prim):
            instruction = prim[step_idx]
        else:
            instruction = data["instruction"]

        pil_img = PILImage.fromarray(image)
        pixel_values = self.processor.image_processor(
            pil_img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        input_ids = self._token_cache[instruction].clone()

        actions_clipped = np.clip(action_norm, -1.0, 1.0)
        bin_indices = np.argmin(
            np.abs(actions_clipped[:, None] - self.bin_centers[None, :]), axis=1
        )
        action_tokens = torch.tensor(
            self.vocab_size - 1 - bin_indices, dtype=torch.long
        )

        full_input_ids = torch.cat([input_ids, action_tokens])
        labels = torch.cat(
            [torch.full_like(input_ids, self.IGNORE_INDEX), action_tokens]
        )

        return {
            "input_ids": full_input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def openvla_collate_fn(batch):
    """Collate for OpenVLA: pad input_ids/labels to uniform length."""
    IGNORE_INDEX = -100
    batch_size = len(batch)
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_token_id = 0

    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    pixel_values_list = []
    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        padded_input_ids[i, :seq_len] = b["input_ids"]
        padded_labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1
        pixel_values_list.append(b["pixel_values"])

    return {
        "input_ids": padded_input_ids,
        "pixel_values": torch.stack(pixel_values_list),
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


# =============================================================================
# Training
# =============================================================================


class WeightedActionLoss:
    """Per-dimension weighted MSE loss."""

    def __init__(self, dim_weights: List[float], device: str = "cuda"):
        import torch
        self.weights = torch.tensor(dim_weights, dtype=torch.float32, device=device)

    def __call__(self, predicted, target):
        per_dim = (predicted - target) ** 2
        weighted = per_dim * self.weights.unsqueeze(0)
        return weighted.mean()


def setup_model_and_lora(config: Dict, local_rank: int = 0):
    """Load base model and apply LoRA configuration."""
    import torch
    from peft import LoraConfig, get_peft_model

    backend = config["model"]["backend"]
    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    quant_cfg = config["quantization"]

    logger.info(f"Loading model: {model_name} (backend={backend})")

    if backend == "openvla":
        from transformers import AutoModelForVision2Seq, AutoProcessor

        attn_impl = "sdpa"
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }

        if quant_cfg.get("enabled", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
            )
        # Pin to local_rank GPU for DDP compatibility (device_map="auto" breaks DDP)
        load_kwargs["device_map"] = {"": local_rank}

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        except (AttributeError, ValueError, TypeError, NotImplementedError):
            logger.warning(f"Attention '{attn_impl}' failed, falling back to eager")
            load_kwargs["attn_implementation"] = "eager"
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)

    elif backend == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(model_name)
        policy = policy.to(dtype=torch.bfloat16)

        # Monkey-patch: add dict-like methods to SmolVLAConfig for PEFT compatibility
        cfg_cls = type(policy.config)
        if not hasattr(cfg_cls, 'get'):
            cfg_cls.get = lambda self, key, default=None: getattr(self, key, default)
        if not hasattr(cfg_cls, '__contains__'):
            cfg_cls.__contains__ = lambda self, key: hasattr(self, key)

        peft_targets = policy._get_default_peft_targets()
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=peft_targets["target_modules"],
            modules_to_save=peft_targets.get("modules_to_save", []),
            bias="none",
        )
        # Patch: PEFT >=0.15 treats model.config as dict-like (calls .get(),
        # uses `in` operator). GrootConfig is not dict-like, so we patch
        # the CLASS (not instance) — Python looks up __contains__ on type.
        if hasattr(policy, 'config'):
            cfg_cls = type(policy.config)
            if not hasattr(cfg_cls, 'get'):
                cfg_cls.get = lambda self, key, default=None: getattr(self, key, default)
            if not hasattr(cfg_cls, '__contains__'):
                cfg_cls.__contains__ = lambda self, key: hasattr(self, key)

        model = get_peft_model(policy, lora_config)
        processor = None

    elif backend in ("pi05", "pi0.5"):
        from lerobot.configs.types import PolicyFeature, FeatureType

        # Monkey-patch: create fake siglip check module for PI0.5 compatibility
        import types, transformers.models.siglip as _siglip_pkg
        _check_mod = types.ModuleType("transformers.models.siglip.check")
        _check_mod.check_whether_transformers_replace_is_installed_correctly = lambda: True
        _siglip_pkg.check = _check_mod
        import sys
        sys.modules["transformers.models.siglip.check"] = _check_mod

        use_tensor_state = config["model"].get("tensor_state", False)
        if use_tensor_state:
            from robobridge.modules.controller.vla.pi05_tensor_state import PI05TensorStatePolicy
            policy = PI05TensorStatePolicy.from_pretrained(model_name)
            logger.info("PI0.5 tensor-state mode: state_proj added, text discretization disabled")
        else:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            policy = PI05Policy.from_pretrained(model_name)

        policy = policy.to(dtype=torch.bfloat16)

        # Monkey-patch: add dict-like methods to PI05Config for PEFT compatibility
        cfg_cls = type(policy.config)
        if not hasattr(cfg_cls, 'get'):
            cfg_cls.get = lambda self, key, default=None: getattr(self, key, default)
        if not hasattr(cfg_cls, '__contains__'):
            cfg_cls.__contains__ = lambda self, key: hasattr(self, key)

        keys_to_remove = [k for k, v in policy.config.input_features.items()
                         if v.type == FeatureType.VISUAL]
        for key in keys_to_remove:
            del policy.config.input_features[key]

        policy.config.input_features["observation.images.base_0_rgb"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *policy.config.image_resolution),
        )

        peft_targets = policy._get_default_peft_targets()
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=peft_targets["target_modules"],
            modules_to_save=peft_targets.get("modules_to_save", []),
            bias="none",
        )
        model = get_peft_model(policy, lora_config)
        processor = None

    elif backend == "groot_n1.5":
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        full_ft = config["model"].get("full_ft", False)
        if full_ft:
            # Full fine-tuning: no LoRA, train all components
            policy = GrootPolicy.from_pretrained(
                model_name,
                tune_llm=True,
                tune_visual=True,
                tune_projector=True,
                tune_diffusion_model=True,
                lora_rank=0,
            )
            logger.info("GROOT full fine-tuning: all components trainable (llm+visual+projector+diffusion)")
        else:
            # Use GROOT's internal LoRA (faster than PEFT, native support)
            policy = GrootPolicy.from_pretrained(
                model_name,
                lora_rank=lora_cfg["rank"],
                lora_alpha=lora_cfg["alpha"],
            )
        policy = policy.to(dtype=torch.bfloat16)
        model = policy

        # Patch transformers fast image processor to add missing method alias
        from transformers.image_processing_utils_fast import BaseImageProcessorFast
        if not hasattr(BaseImageProcessorFast, '_prepare_image_like_inputs'):
            BaseImageProcessorFast._prepare_image_like_inputs = BaseImageProcessorFast._prepare_input_images

        from lerobot.policies.groot.processor_groot import _build_eagle_processor
        processor = _build_eagle_processor()

    elif backend == "hf_vlm":
        from transformers import AutoModel, AutoProcessor

        load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
        if quant_cfg.get("enabled", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
            )

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        except (ValueError, OSError):
            model = AutoModel.from_pretrained(model_name, **load_kwargs)

    else:
        raise ValueError(f"Unknown model backend: {backend}")

    # Apply LoRA for openvla/hf_vlm
    if backend in ("openvla", "hf_vlm"):
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    return model, processor


# ---------------------------------------------------------------------------
# Vision feature caching for PI0.5 (SigLIP tower is frozen → deterministic)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_pi05_vision_cache(model, dataset, device, batch_size=64):
    """Pre-extract frozen SigLIP features for all training samples.

    Since the vision tower is frozen during LoRA training, its output is
    deterministic for the same input.  Caching avoids redundant forward passes
    through SigLIP every epoch, giving a 2-3x wall-clock speedup.

    Returns a fp16 tensor of shape [N, 256, 2048] on *device*.
    """
    raw = model.module if hasattr(model, "module") else model
    raw = raw.base_model.model if hasattr(raw, "base_model") else raw
    pge = raw.model.paligemma_with_expert          # PaliGemmaWithExpert

    n = len(dataset)
    # Probe feature dim with a single sample
    sample_img = dataset[0]["observation.images.base_0_rgb"].unsqueeze(0).to(device).float() * 2.0 - 1.0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        sample_feat = pge.embed_image(sample_img)
    feat_shape = sample_feat.shape[1:]              # (256, 2048)

    cache = torch.empty(n, *feat_shape, dtype=torch.float16, device="cpu", pin_memory=True)
    mem_gb = cache.element_size() * cache.nelement() / 1e9
    logger.info(f"Pre-computing PI0.5 vision cache: {n} samples → {list(cache.shape)} fp16 ({mem_gb:.2f} GB on CPU)")

    from tqdm import tqdm
    for i in tqdm(range(0, n, batch_size), desc="Vision cache"):
        end = min(i + batch_size, n)
        imgs = torch.stack([dataset[j]["observation.images.base_0_rgb"] for j in range(i, end)])
        imgs = imgs.to(device).float() * 2.0 - 1.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feats = pge.embed_image(imgs)
        cache[i:end] = feats.cpu().half()

    logger.info(f"Vision cache ready: {mem_gb:.2f} GB on CPU (pinned memory)")
    return cache


def setup_pi05_vision_cache(model, cache):
    """Monkey-patch embed_image to return cached features when indices are set.

    Usage in training loop:
        pge_ref._vision_cache_indices = batch["_sample_idx"]
        loss = model.forward(batch)
        pge_ref._vision_cache_indices = None
    """
    raw = model.module if hasattr(model, "module") else model
    raw = raw.base_model.model if hasattr(raw, "base_model") else raw
    pge = raw.model.paligemma_with_expert

    pge._vision_cache = cache
    pge._vision_cache_indices = None
    _original_embed = pge.embed_image

    def cached_embed_image(image):
        if pge._vision_cache_indices is not None:
            idx = pge._vision_cache_indices.cpu()  # cache is on CPU, indices must match
            return pge._vision_cache[idx].to(device=image.device, dtype=image.dtype)
        return _original_embed(image)

    pge.embed_image = cached_embed_image
    logger.info("PI0.5 vision cache hook installed (CPU→GPU per-batch transfer)")
    return pge


# ---------------------------------------------------------------------------
# Vision feature caching for SmolVLA (vision encoder + connector frozen when train_expert_only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_smolvla_vision_cache(model, dataset, device, batch_size=32):
    """Pre-extract frozen SigLIP + connector features for all SmolVLA training samples."""
    raw = model.module if hasattr(model, "module") else model
    raw = raw.base_model.model if hasattr(raw, "base_model") else raw
    vwe = raw.model.vlm_with_expert  # SmolVLMWithExpertModel

    n = len(dataset)
    # Probe feature dim
    sample_img = dataset[0]["observation.images.camera1"].unsqueeze(0).to(device).float() * 2.0 - 1.0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        sample_feat = vwe.embed_image(sample_img)
    feat_shape = sample_feat.shape[1:]

    cache = torch.empty(n, *feat_shape, dtype=torch.float16, device="cpu", pin_memory=True)
    mem_gb = cache.element_size() * cache.nelement() / 1e9
    logger.info(f"Pre-computing SmolVLA vision cache: {n} samples → {list(cache.shape)} fp16 ({mem_gb:.2f} GB on CPU)")

    from tqdm import tqdm
    for i in tqdm(range(0, n, batch_size), desc="SmolVLA vision cache"):
        end = min(i + batch_size, n)
        imgs = torch.stack([dataset[j]["observation.images.camera1"] for j in range(i, end)])
        imgs = imgs.to(device).float() * 2.0 - 1.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feats = vwe.embed_image(imgs)
        cache[i:end] = feats.cpu().half()

    logger.info(f"SmolVLA vision cache ready: {mem_gb:.2f} GB on CPU (pinned memory)")
    return cache


def setup_smolvla_vision_cache(model, cache):
    """Monkey-patch SmolVLA embed_image to return cached features."""
    raw = model.module if hasattr(model, "module") else model
    raw = raw.base_model.model if hasattr(raw, "base_model") else raw
    vwe = raw.model.vlm_with_expert

    vwe._vision_cache = cache
    vwe._vision_cache_indices = None
    _original_embed = vwe.embed_image

    def cached_embed_image(image):
        if vwe._vision_cache_indices is not None:
            idx = vwe._vision_cache_indices.cpu()  # cache is on CPU, indices must match
            return vwe._vision_cache[idx].to(device=image.device, dtype=image.dtype)
        return _original_embed(image)

    vwe.embed_image = cached_embed_image
    logger.info("SmolVLA vision cache hook installed (CPU→GPU per-batch transfer)")
    return vwe


def _pad_token_sequences(sequences: List[np.ndarray], pad_value: int) -> np.ndarray:
    """Pad variable-length 1D token sequences to a dense 2D array."""
    if not sequences:
        return np.empty((0, 0), dtype=np.int64)

    max_len = max(seq.shape[0] for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq
    return padded


def precompute_groot_eagle_features(demo_dataset, eagle_processor, use_wrist: bool = False) -> bool:
    """Precompute Eagle processor outputs beside each source NPZ file."""
    if not hasattr(demo_dataset, "samples"):
        logger.warning("GROOT Eagle precompute requires NPZ-backed datasets; skipping")
        return False

    sample_paths = [sample[0] for sample in demo_dataset.samples]
    if sample_paths and all(os.path.exists(path.replace(".npz", ".eagle.npz")) for path in sample_paths):
        logger.info("All precomputed Eagle feature files already exist")
        return True

    pad_token_id = 0
    if hasattr(eagle_processor, "tokenizer") and eagle_processor.tokenizer.pad_token_id is not None:
        pad_token_id = int(eagle_processor.tokenizer.pad_token_id)

    logger.info(f"Precomputing Eagle features for {len(sample_paths)} demos (use_wrist={use_wrist})")
    for demo_idx, npz_path in enumerate(sample_paths):
        eagle_path = npz_path.replace(".npz", ".eagle.npz")
        if os.path.exists(eagle_path):
            continue

        data = demo_dataset.load_demo(demo_idx)
        images = data["images"]
        wrist_images = data.get("wrist_images")
        prim_instructions = data.get("primitive_instructions")

        input_ids_steps = []
        attention_mask_steps = []
        pixel_values_steps = []
        image_sizes_steps = []

        for step_idx in range(images.shape[0]):
            instruction = data["instruction"]
            if prim_instructions and step_idx < len(prim_instructions):
                instruction = prim_instructions[step_idx]

            step_images = [images[step_idx]]
            if use_wrist and wrist_images is not None:
                step_images.append(wrist_images[step_idx])

            content = [{"type": "image", "image": PILImage.fromarray(img)} for img in step_images]
            content.append({"type": "text", "text": instruction})

            conversation = [{"role": "user", "content": content}]
            text = eagle_processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = eagle_processor.process_vision_info(conversation)
            eagle_inputs = eagle_processor(
                text=[text], images=image_inputs, return_tensors="pt", padding=True,
            )

            input_ids_steps.append(eagle_inputs["input_ids"].squeeze(0).cpu().numpy().copy())
            attention_mask_steps.append(
                eagle_inputs["attention_mask"].squeeze(0).cpu().numpy().copy()
            )
            pixel_values_steps.append(eagle_inputs["pixel_values"].cpu().numpy().copy())
            if "image_sizes" in eagle_inputs:
                image_sizes_steps.append(eagle_inputs["image_sizes"].cpu().numpy().copy())

        save_kwargs = {
            "input_ids": _pad_token_sequences(input_ids_steps, pad_token_id),
            "attention_mask": _pad_token_sequences(attention_mask_steps, 0),
            "pixel_values": np.stack(pixel_values_steps, axis=0),
        }
        if image_sizes_steps:
            save_kwargs["image_sizes"] = np.stack(image_sizes_steps, axis=0)

        np.savez(eagle_path, **save_kwargs)
        logger.info(
            f"Precomputed Eagle features: demo {demo_idx + 1}/{len(sample_paths)} -> {eagle_path}"
        )

    logger.info("Eagle precompute complete")
    return True


def train(config: Dict) -> None:
    """Main training loop."""
    import torch
    from torch.utils.data import DataLoader

    adapter_cfg = config["adapter"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    disable_validation = train_cfg.get("disable_validation", False)

    rank = int(os.environ.get("RANK", 0))
    adapter_name = adapter_cfg["name"]
    task_name = data_cfg.get("task") or "general"
    lr = train_cfg["learning_rate"]
    lr_str = f"lr{lr:.0e}".replace("+", "").replace("-0", "-")
    output_dir = os.path.join(output_cfg["base_dir"], task_name, f"{adapter_name}_adapter")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    # Barrier so rank>0 waits for dir creation
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create datasets
    # These will hold the effective stats (task-specific when filtered)
    task_action_stats = None
    task_state_stats = None
    task_quantile_stats = None

    processed_dir = data_cfg.get("processed_dir")
    hdf5_dir = data_cfg.get("hdf5_dir")
    if hdf5_dir:
        train_demo_ds = RawHDF5Dataset(
            hdf5_root=hdf5_dir,
            task_filter=data_cfg.get("task"),
            max_samples=data_cfg.get("max_samples"),
            image_size=config["model"]["image_size"],
            split="train",
        )
        val_demo_ds = None
        if not disable_validation:
            val_demo_ds = RawHDF5Dataset(
                hdf5_root=hdf5_dir,
                task_filter=data_cfg.get("task"),
                image_size=config["model"]["image_size"],
                split="val",
            )
        metadata = {"action_stats": train_demo_ds.action_stats}
        ext_meta = {
            "state_stats": train_demo_ds.state_stats,
            "action_quantile_stats": train_demo_ds.action_quantile_stats,
        }
        # HDF5 path already computes task-specific stats
        task_action_stats = train_demo_ds.action_stats
        task_state_stats = train_demo_ds.state_stats
        task_quantile_stats = train_demo_ds.action_quantile_stats

        # Save data_stats.json for evaluation
        if rank == 0:
            data_stats_path = os.path.join(output_dir, "data_stats.json")
            _hdf5_data_stats = {
                "action_stats": train_demo_ds.action_stats,
                "state_stats": train_demo_ds.state_stats,
                "action_quantile_stats": train_demo_ds.action_quantile_stats,
            }
            if config["model"].get("tensor_state", False):
                _hdf5_data_stats["pi05_tensor_state"] = True
            if config["model"].get("full_ft", False):
                _hdf5_data_stats["groot_full_ft"] = True
            with open(data_stats_path, "w") as f:
                json.dump(_hdf5_data_stats, f, indent=2)
    else:
        metadata_path = os.path.join(processed_dir, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        use_prim = data_cfg.get("use_primitive_instructions", False)
        task_filter = data_cfg.get("task")

        # Recompute action/state stats from filtered data if task-specific
        state_dims = data_cfg.get("state_dims", 0)
        if task_filter:
            logger.info(f"Recomputing action/state stats for task: {task_filter}")
            task_action_stats, task_state_stats, task_quantile_stats = _compute_stats_from_processed(
                processed_dir, task_filter, split="train", state_dims=state_dims
            )
            logger.info(f"  Task-specific action min: {task_action_stats['min']}")
            logger.info(f"  Task-specific action max: {task_action_stats['max']}")
        else:
            task_action_stats = metadata["action_stats"]
            task_state_stats = None
            task_quantile_stats = None

        train_demo_ds = VLADataset(
            data_dir=processed_dir,
            action_stats=task_action_stats,
            task_filter=task_filter,
            max_samples=data_cfg.get("max_samples"),
            image_size=config["model"]["image_size"],
            split="train",
            use_primitive_instructions=use_prim,
            state_dims=state_dims,
        )
        val_demo_ds = None
        if not disable_validation:
            val_demo_ds = VLADataset(
                data_dir=processed_dir,
                action_stats=task_action_stats,
                task_filter=task_filter,
                image_size=config["model"]["image_size"],
                split="val",
                use_primitive_instructions=use_prim,
                state_dims=state_dims,
            )
        ext_meta = None

        # Save data_stats.json for pipeline inference
        ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
        if os.path.exists(ext_meta_path):
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)

        if rank == 0:
            data_stats_path = os.path.join(output_dir, "data_stats.json")
            data_stats = {"action_stats": task_action_stats}
            if task_state_stats:
                data_stats["state_stats"] = task_state_stats
                data_stats["action_quantile_stats"] = task_quantile_stats
            elif ext_meta:
                data_stats["state_stats"] = ext_meta["state_stats"]
                data_stats["action_quantile_stats"] = ext_meta["action_quantile_stats"]
            # Propagate state_format from preprocessing metadata
            if ext_meta and "state_format" in ext_meta:
                data_stats["state_format"] = ext_meta["state_format"]
            # Mark tensor-state mode for eval auto-detection
            if config["model"].get("tensor_state", False):
                data_stats["pi05_tensor_state"] = True
            # Mark full fine-tuning mode for eval auto-detection
            if config["model"].get("full_ft", False):
                data_stats["groot_full_ft"] = True
            # Mark action_dim for CycleVLA (9D) vs standard (7D)
            _action_dim_cfg = config.get("_action_dim", 7)
            if _action_dim_cfg != 7:
                data_stats["action_dim"] = _action_dim_cfg
            with open(data_stats_path, "w") as f:
                json.dump(data_stats, f, indent=2)
            logger.info(f"Saved data_stats.json to {data_stats_path}")

    train_ds = VLAStepDataset(train_demo_ds)
    if val_demo_ds is not None and len(val_demo_ds) > 0:
        val_ds = VLAStepDataset(val_demo_ds)
    else:
        val_ds = None
        if disable_validation:
            logger.info("Validation disabled by config")
        else:
            logger.info("No validation data found — skipping validation")

    logger.info(f"Train: {len(train_ds)} steps, Val: {len(val_ds) if val_ds is not None else 0} steps")

    distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, "is_initialized") else False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size() if distributed else 1

    model, processor = setup_model_and_lora(config, local_rank=local_rank)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resume from checkpoint: load LoRA weights
    resume_from = config.get("_resume_from")
    if resume_from:
        ckpt_file = os.path.join(resume_from, "model.safetensors")
        if os.path.exists(ckpt_file):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_file)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Resumed from {resume_from}: loaded {len(state_dict)} keys, "
                       f"missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    if not hasattr(model, "hf_device_map"):
        model = model.to(device)

    if distributed:
        # Ensure all ranks finish model loading before DDP
        torch.distributed.barrier()
        n_params = sum(1 for p in model.parameters())
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(f"Rank {local_rank}: {n_params} param tensors, {n_trainable} trainable")

        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters needed for LoRA (some params inactive per forward)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        logger.info(f"Distributed DDP: rank {local_rank}/{world_size}")

    backend = config["model"]["backend"]
    if train_cfg.get("gradient_checkpointing", False):
        gc_model = model.module if hasattr(model, "module") else model
        if backend in ("pi05", "pi0.5"):
            inner = gc_model
            if hasattr(inner, 'base_model'):
                inner = inner.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
            if hasattr(inner, 'model'):
                inner = inner.model
            if hasattr(inner, 'gradient_checkpointing_enabled'):
                inner.gradient_checkpointing_enabled = True
                if hasattr(inner, 'paligemma_with_expert'):
                    for submodel in [inner.paligemma_with_expert]:
                        if hasattr(submodel, 'gradient_checkpointing_enable'):
                            submodel.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs={"use_reentrant": False}
                            )
                logger.info("Gradient checkpointing enabled (PI05)")
        elif backend == "groot_n1.5":
            # GROOT: enable gradient checkpointing on inner backbone components
            inner = gc_model._groot_model if hasattr(gc_model, '_groot_model') else gc_model
            gc_enabled = False
            for name, module in inner.named_modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    gc_enabled = True
                    logger.info(f"Gradient checkpointing enabled on: {name}")
            if not gc_enabled:
                # Fallback: use torch.utils.checkpoint manually via hooks
                logger.warning("GROOT: no gradient_checkpointing_enable found, using torch checkpoint wrapper")
                import torch.utils.checkpoint as ckpt
                for name, module in inner.named_children():
                    if hasattr(module, 'forward') and sum(p.numel() for p in module.parameters()) > 1e6:
                        orig_forward = module.forward
                        def make_ckpt_forward(mod, orig_fn):
                            def ckpt_forward(*args, **kwargs):
                                return ckpt.checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
                            return ckpt_forward
                        module.forward = make_ckpt_forward(module, orig_forward)
                        logger.info(f"  Wrapped {name} with checkpoint ({sum(p.numel() for p in module.parameters())/1e6:.1f}M params)")
        else:
            gc_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled")

    if train_cfg.get("compile", False):
        model = torch.compile(model)

    loss_fn = WeightedActionLoss(adapter_cfg["loss_weights"], device=str(device))

    # Wrap datasets for backend-specific preprocessing
    if backend == "openvla":
        raw_model = model.module if hasattr(model, "module") else model
        raw_model = raw_model.base_model.model if hasattr(raw_model, "base_model") else raw_model
        train_ds = OpenVLAStepDataset(train_ds, processor, raw_model.bin_centers, raw_model.vocab_size)
        if val_ds is not None:
            val_ds = OpenVLAStepDataset(val_ds, processor, raw_model.bin_centers, raw_model.vocab_size)
    elif backend == "smolvla":
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 50)
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        img_size = config["model"].get("image_size", 512)
        _action_stats_smol = task_action_stats if task_action_stats else metadata.get("action_stats", {})
        _state_stats_smol = task_state_stats if task_state_stats else ext_meta["state_stats"]
        train_ds = SmolVLAChunkDataset(train_chunk, _state_stats_smol, _action_stats_smol, image_size=img_size)
        if val_ds is not None:
            val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
            val_ds = SmolVLAChunkDataset(val_chunk, _state_stats_smol, _action_stats_smol, image_size=img_size)
    elif backend in ("pi05", "pi0.5"):
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 50)
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        img_size = config["model"].get("image_size", 224)
        use_tensor_state = config["model"].get("tensor_state", False)
        if use_tensor_state:
            train_ds = PI05TensorStateChunkDataset(train_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"], image_size=img_size)
            if val_ds is not None:
                val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
                val_ds = PI05TensorStateChunkDataset(val_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"], image_size=img_size)
        else:
            train_ds = PI05ChunkDataset(train_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"], image_size=img_size)
            if val_ds is not None:
                val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
                val_ds = PI05ChunkDataset(val_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"], image_size=img_size)
    elif backend == "groot_n1.5":
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 16)
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        # Use task-specific stats if available (from task_filter recomputation)
        _action_stats = task_action_stats if task_action_stats else metadata["action_stats"]
        _state_stats = task_state_stats if task_state_stats else ext_meta["state_stats"]

        # Quantile normalization: use q01/q99 as min/max for tighter action range
        action_norm_mode = config.get("_action_norm", "min_max")
        if action_norm_mode == "quantile":
            _qstats = task_quantile_stats if task_quantile_stats else ext_meta.get("action_quantile_stats")
            if _qstats and "q01" in _qstats and "q99" in _qstats:
                logger.info(f"Using QUANTILE normalization (q01/q99) instead of min/max")
                logger.info(f"  min_max min: {_action_stats['min']}")
                logger.info(f"  min_max max: {_action_stats['max']}")
                _action_stats = dict(_action_stats)  # copy
                _action_stats["min"] = _qstats["q01"]
                _action_stats["max"] = _qstats["q99"]
                _action_stats["mode"] = "min_max"  # eval still uses min_max denorm
                logger.info(f"  quantile min (q01): {_action_stats['min']}")
                logger.info(f"  quantile max (q99): {_action_stats['max']}")
            else:
                logger.warning("Quantile stats not found, falling back to min_max")

        _use_wrist = config.get("data", {}).get("use_wrist", False)
        # Check if precomputed eagle features exist
        _sample_npz = train_demo_ds.samples[0][0] if train_demo_ds.samples else ""
        _has_precomputed = os.path.exists(_sample_npz.replace(".npz", ".eagle.npz")) if _sample_npz else False
        if train_cfg.get("precompute_eagle", False) and not _has_precomputed:
            if rank == 0:
                precompute_groot_eagle_features(train_demo_ds, processor, use_wrist=_use_wrist)
            if distributed:
                torch.distributed.barrier()
            _has_precomputed = os.path.exists(_sample_npz.replace(".npz", ".eagle.npz")) if _sample_npz else False
        if _has_precomputed:
            logger.info("Using precomputed eagle features (no CPU eagle processing)")
        train_ds = GrootChunkDataset(train_chunk, _state_stats, _action_stats, use_wrist=_use_wrist,
                                     eagle_processor=None if _has_precomputed else processor,
                                     precomputed_eagle=_has_precomputed)
        if val_ds is not None:
            val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
            val_ds = GrootChunkDataset(val_chunk, _state_stats, _action_stats, use_wrist=_use_wrist,
                                       eagle_processor=None if _has_precomputed else processor,
                                       precomputed_eagle=_has_precomputed)

        # Re-save data_stats.json with effective action bounds (quantile-overridden if applicable)
        if rank == 0 and action_norm_mode == "quantile":
            data_stats_path = os.path.join(output_dir, "data_stats.json")
            data_stats = {"action_stats": _action_stats}
            if _state_stats:
                data_stats["state_stats"] = _state_stats
            _qstats_save = task_quantile_stats if task_quantile_stats else (ext_meta.get("action_quantile_stats") if ext_meta else None)
            if _qstats_save:
                data_stats["action_quantile_stats"] = _qstats_save
            # Propagate state_format from preprocessing metadata
            if ext_meta and "state_format" in ext_meta:
                data_stats["state_format"] = ext_meta["state_format"]
            if config["model"].get("full_ft", False):
                data_stats["groot_full_ft"] = True
            # Mark action_dim for CycleVLA (9D) vs standard (7D)
            _action_dim_cfg = config.get("_action_dim", 7)
            if _action_dim_cfg != 7:
                data_stats["action_dim"] = _action_dim_cfg
            with open(data_stats_path, "w") as f:
                json.dump(data_stats, f, indent=2)
            logger.info(f"Re-saved data_stats.json with quantile bounds to {data_stats_path}")

    # Optimizer
    opt_name = train_cfg.get("optimizer", "adamw")
    base_lr = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]

    # Separate param groups: state_proj gets 10x LR (random init needs higher LR)
    state_proj_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "state_proj" in name:
            state_proj_params.append(param)
            logger.info(f"  state_proj param: {name} (shape={param.shape}, 10x LR)")
        else:
            other_params.append(param)

    if state_proj_params:
        state_proj_lr = base_lr * 10
        logger.info(f"  state_proj LR: {state_proj_lr} (10x base {base_lr})")
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": state_proj_params, "lr": state_proj_lr},
        ]
    else:
        param_groups = [{"params": other_params, "lr": base_lr}]

    adamw_betas = tuple(train_cfg.get("adamw_betas", (0.9, 0.999)))
    adamw_eps = train_cfg.get("adamw_eps", 1e-8)

    if opt_name == "adamw_8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=base_lr, weight_decay=weight_decay,
                                         betas=adamw_betas, eps=adamw_eps)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay,
                                       betas=adamw_betas, eps=adamw_eps)
    logger.info(f"Optimizer: {opt_name}, lr={base_lr}, wd={weight_decay}, betas={adamw_betas}, eps={adamw_eps}")

    # DataLoader
    if backend == "openvla":
        active_collate_fn = openvla_collate_fn
    elif backend == "groot_n1.5":
        from functools import partial
        active_collate_fn = partial(groot_collate_fn, eagle_processor=processor)
    elif backend in ("smolvla", "pi05", "pi0.5"):
        def active_collate_fn(batch):
            result = {}
            for key in batch[0].keys():
                vals = [b[key] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    result[key] = torch.stack(vals)
                else:
                    result[key] = vals
            return result
    else:
        def active_collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "robot_state": torch.stack([b["robot_state"] for b in batch]),
                "action": torch.stack([b["action"] for b in batch]),
                "instruction": [b["instruction"] for b in batch],
            }

    num_workers = train_cfg.get("dataloader_num_workers", 4)

    # Auto-scale batch size for DDP: config batch_size is the effective (total) batch
    per_gpu_batch = train_cfg["batch_size"]
    if distributed:
        assert per_gpu_batch % world_size == 0, (
            f"batch_size {per_gpu_batch} must be divisible by world_size {world_size}"
        )
        per_gpu_batch = per_gpu_batch // world_size
    logger.info(f"DataLoader: per_gpu_batch={per_gpu_batch}, num_workers={num_workers}, world_size={world_size}")

    train_sampler = DemoGroupedBatchSampler(train_ds, batch_size=per_gpu_batch, shuffle=True, rank=rank, world_size=world_size)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=active_collate_fn, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None)

    if val_ds is not None:
        val_sampler = DemoGroupedBatchSampler(val_ds, batch_size=per_gpu_batch, shuffle=False, rank=rank, world_size=world_size)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0, collate_fn=active_collate_fn, pin_memory=True)
    else:
        val_loader = None

    # Scheduler
    grad_accum = train_cfg["gradient_accumulation_steps"]
    max_steps = train_cfg.get("max_steps", -1)

    micro_batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = max(micro_batches_per_epoch // grad_accum, 1)

    if max_steps > 0:
        num_epochs = (max_steps + optimizer_steps_per_epoch - 1) // optimizer_steps_per_epoch
        total_optimizer_steps = max_steps
    else:
        num_epochs = train_cfg["num_epochs"]
        total_optimizer_steps = optimizer_steps_per_epoch * num_epochs

    scheduler_type = train_cfg.get("scheduler_type", "onecycle")
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif scheduler_type == "cosine":
        warmup_steps = int(total_optimizer_steps * train_cfg.get("warmup_ratio", 0.0))
        if warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps - warmup_steps)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_cfg["learning_rate"], total_steps=total_optimizer_steps, pct_start=train_cfg["warmup_ratio"])

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    early_stopping_patience = train_cfg.get("early_stopping_patience", 0)
    no_improve_count = 0
    best_model_checkpoint = None
    log_history = []

    import time
    train_start_time = time.time()

    effective_batch = per_gpu_batch * grad_accum * world_size
    logger.info("Starting training...")
    logger.info(f"  Adapter: {adapter_name}")
    logger.info(f"  Task: {task_name}")
    logger.info(f"  Batch size: {per_gpu_batch} x {grad_accum} x {world_size}gpu = {effective_batch}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Optimizer steps: {total_optimizer_steps}")

    from tqdm import tqdm

    # --- Vision feature caching (PI0.5 / SmolVLA — frozen vision encoder) ---
    pge_ref = None  # holds the object with _vision_cache_indices attribute
    if train_cfg.get("vision_cache", False):
        try:
            if backend in ("pi05", "pi0.5"):
                vision_cache = precompute_pi05_vision_cache(model, train_ds, device, batch_size=64)
                pge_ref = setup_pi05_vision_cache(model, vision_cache)
            elif backend == "smolvla":
                vision_cache = precompute_smolvla_vision_cache(model, train_ds, device, batch_size=32)
                pge_ref = setup_smolvla_vision_cache(model, vision_cache)
        except Exception as e:
            logger.warning(f"Vision cache setup failed, continuing without cache: {e}")
            pge_ref = None

    model.train()

    start_epoch = config.get("_start_epoch", 0)
    if start_epoch > 0:
        logger.info(f"Skipping epochs 1-{start_epoch}, starting from epoch {start_epoch + 1}")

    for epoch in range(num_epochs):
        if epoch < start_epoch:
            train_sampler.set_epoch(epoch)
            # Advance scheduler steps for skipped epochs
            for _ in range(optimizer_steps_per_epoch):
                scheduler.step()
            continue
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for batch_idx, batch in pbar:
            with torch.amp.autocast("cuda", enabled=train_cfg.get("bf16", True), dtype=torch.bfloat16):
                if backend == "openvla":
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    loss = outputs.loss / grad_accum
                elif backend in ("smolvla", "pi05", "pi0.5", "groot_n1.5"):
                    batch_on_device = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    # Inject vision cache indices for PI0.5
                    if pge_ref is not None and "_sample_idx" in batch_on_device:
                        pge_ref._vision_cache_indices = batch_on_device.pop("_sample_idx")
                    elif "_sample_idx" in batch_on_device:
                        batch_on_device.pop("_sample_idx")  # remove non-model key
                    loss_out = model.forward(batch_on_device)
                    if pge_ref is not None:
                        pge_ref._vision_cache_indices = None
                    if isinstance(loss_out, tuple):
                        loss = loss_out[0] / grad_accum
                    else:
                        loss = loss_out / grad_accum
                else:
                    raise ValueError(f"Unknown backend: {backend}")

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # DDP auto-syncs gradients on backward(); no manual all_reduce needed
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                lr = scheduler.get_last_lr()[0]
                step_loss = loss.item() * grad_accum

                pbar.set_postfix({"step": global_step, "loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"})

                if global_step % train_cfg["logging_steps"] == 0:
                    log_history.append({"epoch": epoch + (batch_idx + 1) / len(train_loader), "step": global_step, "loss": step_loss, "learning_rate": lr})

                if global_step % train_cfg["save_steps"] == 0:
                    _save_checkpoint(model, output_dir, global_step, lora_config=config.get("lora"))

                if max_steps > 0 and global_step >= max_steps:
                    break

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        pbar.close()
        avg_epoch_loss = epoch_loss / max(n_batches, 1)

        # Validate only at save_every_epochs intervals (default: every epoch)
        val_every = train_cfg.get("save_every_epochs", 0) or 1
        do_val = (epoch + 1) % val_every == 0 or (epoch + 1) == num_epochs
        if disable_validation:
            do_val = False

        if do_val and val_loader is not None and len(val_loader) > 0 and train_cfg.get("max_val_batches", -1) != 0:
            val_loss = _validate(model, processor, val_loader, loss_fn, device, train_cfg, backend)
            if distributed:
                val_loss_t = torch.tensor(val_loss, device=device)
                torch.distributed.all_reduce(val_loss_t, op=torch.distributed.ReduceOp.AVG)
                val_loss = val_loss_t.item()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            val_loss = avg_epoch_loss
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f}")

        log_history.append({"epoch": epoch + 1, "step": global_step, "train_loss": avg_epoch_loss, "eval_loss": val_loss})

        if do_val and val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_checkpoint = os.path.join(output_dir, "checkpoint-best")
            _save_checkpoint(model, output_dir, "best", lora_config=config.get("lora"))
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")
        elif do_val:
            no_improve_count += 1
            logger.info(f"  No improvement for {no_improve_count}/{early_stopping_patience} epochs")
            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered.")
                break

        # Epoch-based periodic checkpoint saving
        save_every = train_cfg.get("save_every_epochs", 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            _save_checkpoint(model, output_dir, f"epoch-{epoch+1}", lora_config=config.get("lora"))
            logger.info(f"  Periodic checkpoint saved: epoch-{epoch+1}")

        if max_steps > 0 and global_step >= max_steps:
            break

    _save_checkpoint(model, output_dir, "final", lora_config=config.get("lora"))

    train_end_time = time.time()
    if rank == 0:
        trainer_state = {
            "best_model_checkpoint": best_model_checkpoint,
            "best_metric": None if disable_validation else best_val_loss,
            "epoch": epoch + 1,
            "global_step": global_step,
            "max_steps": max_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "train_runtime": train_end_time - train_start_time,
            "log_history": log_history,
        }
        trainer_state_path = os.path.join(output_dir, "trainer_state.json")
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)

    logger.info(f"Training complete! Adapters saved to {output_dir}")


@torch.no_grad()
def _validate(model, processor, val_loader, loss_fn, device, train_cfg, backend):
    import torch

    model.eval()
    total_loss = 0.0
    n_batches = 0
    max_val_batches = train_cfg.get("max_val_batches", -1)

    for batch in val_loader:
        with torch.amp.autocast("cuda", enabled=train_cfg.get("bf16", True), dtype=torch.bfloat16):
            if backend == "openvla":
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss
            elif backend in ("smolvla", "pi05", "pi0.5", "groot_n1.5"):
                batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_on_device.pop("_sample_idx", None)  # remove non-model key
                loss_out = model.forward(batch_on_device)
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out
            else:
                raise ValueError(f"Unknown backend: {backend}")

        total_loss += loss.item()
        n_batches += 1

        if max_val_batches > 0 and n_batches >= max_val_batches:
            break

    model.train()
    return total_loss / max(n_batches, 1)


def _save_checkpoint(model, output_dir: str, step, lora_config: Optional[Dict] = None) -> None:
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return

    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    save_model = model.module if hasattr(model, "module") else model
    import torch
    from safetensors.torch import save_file
    # Save only trainable (LoRA/adapter) parameters, not the full model
    trainable_keys = {n for n, p in save_model.named_parameters() if p.requires_grad}
    state = {k: v for k, v in save_model.state_dict().items() if k in trainable_keys}
    # Save in both formats for compatibility
    save_file(state, os.path.join(save_dir, "model_diff.safetensors"))
    torch.save(state, os.path.join(save_dir, "adapter_model.pt"))
    logger.info(f"  Saved {len(state)} trainable params ({sum(v.numel() for v in state.values()):,} elements)")

    # Save LoRA config for evaluation script compatibility
    if lora_config:
        import json
        config_data = {
            "lora_rank": lora_config.get("rank", 64),
            "lora_alpha": lora_config.get("alpha", 128),
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

    logger.info(f"Checkpoint saved: {save_dir}")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="Train VLA LoRA adapter")
    parser.add_argument("--config", type=str, required=True, help="Adapter config YAML")
    parser.add_argument("--gpu-profile", type=str, default=None, choices=["a6000", "a100"])
    parser.add_argument("--model-config", type=str, default=None, help="Model config name from configs/models/")
    parser.add_argument("--scheduler-type", type=str, default=None, choices=["onecycle", "constant", "cosine"])
    parser.add_argument("--chunk-stride", type=int, default=None)
    parser.add_argument("--action-norm", type=str, default="min_max", choices=["min_max", "quantile"],
                        help="Action normalization mode: min_max (default) or quantile (q01/q99)")
    parser.add_argument("--task", type=str, default=None, help="Train on specific task")
    parser.add_argument("--model-backend", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--hdf5-dir", type=str, default=None, help="RoboCasa HDF5 data root (raw data, no preprocessing)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--save-every-epochs", type=int, default=0,
                        help="Save checkpoint every N epochs (0=disabled)")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-base-dir", type=str, default=None)
    parser.add_argument("--tensor-state", action="store_true",
                        help="PI0.5: use tensor-based state conditioning instead of text discretization")
    parser.add_argument("--full-ft", action="store_true",
                        help="GROOT: full fine-tuning instead of LoRA (trains all parameters)")
    parser.add_argument("--action-dim", type=int, default=7,
                        help="Action dimension: 7 for standard (default), 9 for CycleVLA (7D + stop_signal + progress)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Preprocessed data directory (alias for --processed-dir). Default: data_delta. Use data_cyclevla for CycleVLA.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint dir to resume LoRA weights from")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch to start from (skip earlier epochs)")
    args = parser.parse_args()

    overrides = {}
    if args.task:
        overrides.setdefault("data", {})["task"] = args.task
    if args.model_backend:
        overrides.setdefault("model", {})["backend"] = args.model_backend
    if args.model_name:
        overrides.setdefault("model", {})["name"] = args.model_name
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.epochs:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.lora_rank:
        overrides.setdefault("lora", {})["rank"] = args.lora_rank
    if args.processed_dir:
        overrides.setdefault("data", {})["processed_dir"] = args.processed_dir
    if args.hdf5_dir:
        overrides.setdefault("data", {})["hdf5_dir"] = args.hdf5_dir
    if args.max_steps:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.grad_accum:
        overrides.setdefault("training", {})["gradient_accumulation_steps"] = args.grad_accum
    if args.gradient_checkpointing:
        overrides.setdefault("training", {})["gradient_checkpointing"] = True
    if args.max_val_batches:
        overrides.setdefault("training", {})["max_val_batches"] = args.max_val_batches
    if args.scheduler_type:
        overrides.setdefault("training", {})["scheduler_type"] = args.scheduler_type
    if args.chunk_stride:
        overrides.setdefault("model", {})["chunk_stride"] = args.chunk_stride
    if args.action_norm:
        overrides["_action_norm"] = args.action_norm
    if args.output_base_dir:
        overrides.setdefault("output", {})["base_dir"] = args.output_base_dir
    if args.skip_checkpoints:
        overrides.setdefault("training", {})["save_steps"] = 999999
    if args.save_every_epochs:
        overrides.setdefault("training", {})["save_every_epochs"] = args.save_every_epochs
    if args.tensor_state:
        overrides.setdefault("model", {})["tensor_state"] = True
    if args.full_ft:
        overrides.setdefault("model", {})["full_ft"] = True
    if args.data_dir:
        overrides.setdefault("data", {})["processed_dir"] = args.data_dir
    if args.action_dim != 7:
        overrides["_action_dim"] = args.action_dim

    config = load_config(args.config, args.gpu_profile, args.model_config, overrides or None)
    if args.resume_from:
        config["_resume_from"] = args.resume_from
    if args.start_epoch > 0:
        config["_start_epoch"] = args.start_epoch

    logger.info("=" * 60)
    logger.info("VLA LoRA Training")
    logger.info(f"  Backend: {config['model']['backend']}")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Adapter: {config['adapter']['name']}")
    logger.info(f"  Task: {config['data'].get('task', 'all')}")
    logger.info("=" * 60)

    try:
        train(config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
