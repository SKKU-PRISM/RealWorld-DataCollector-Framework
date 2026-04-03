#!/usr/bin/env python3
"""
VLA LoRA adapter training script.

Trains Move or Grip LoRA adapters on preprocessed RoboCasa data.
Supports multiple VLA backends (OpenVLA, SmolVLA, pi0.5, GROOT N1, HF VLM)
and GPU profiles (A6000 QLoRA, A100 full precision).

Usage:
    # Train move adapter on A6000
    python scripts/vla/train_lora.py \
        --config scripts/vla/configs/move_adapter.yaml \
        --gpu-profile a6000

    # Train grip adapter on A100 for specific task
    python scripts/vla/train_lora.py \
        --config scripts/vla/configs/grip_adapter.yaml \
        --gpu-profile a100 \
        --task PnPCounterToCab

    # Train with SmolVLA backend
    python scripts/vla/train_lora.py \
        --config scripts/vla/configs/move_adapter.yaml \
        --gpu-profile a6000 \
        --model-backend smolvla \
        --model-name lerobot/smolvla_base

    # Train with custom overrides
    python scripts/vla/train_lora.py \
        --config scripts/vla/configs/move_adapter.yaml \
        --gpu-profile a6000 \
        --lr 1e-4 --epochs 10 --lora-rank 32
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

# Embodiment-agnostic state: eef_pos(3) + eef_quat(4) + target_pos(3) = 10D
# Indices into the 19D preprocessed robot_state (drops joint_pos[7:14] and gripper_qpos[14:16])
_STATE_KEEP_IDX = list(range(0, 7)) + list(range(16, 19))


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


class VLADataset:
    """PyTorch Dataset for VLA LoRA training.

    Loads preprocessed .npz files with images, states, actions, and phase labels.
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
    ):
        import torch
        from torch.utils.data import Dataset

        self.data_dir = data_dir
        self.image_size = image_size
        self.torch = torch
        self.split = split

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
        """Build list of (npz_path, meta_path) tuples.

        Supports:
        - Flat structure: data_dir/*.npz
        - Hierarchical structure: data_dir/**/{split}/*.npz (v0.1_processed layout)
        """
        data_path = Path(data_dir)

        # Try flat structure first (legacy)
        npz_files = sorted(data_path.glob("*.npz"))

        # If no flat files, try hierarchical structure
        if not npz_files and split:
            # v0.1_processed layout: multi_stage/*/TaskName/{split}/*.npz
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

        # Per-timestep primitive instructions (if available)
        primitive_instructions = meta.get("primitive_instructions")

        return {
            "images": data["images"],         # (N, H, W, 3)
            "robot_state": data["robot_state"][:, _STATE_KEEP_IDX],  # (N, 10) = eef_pos(3)+eef_quat(4)+target_pos(3)
            "actions": data["actions"],       # (N, 7)
            "phases": data["phases"],         # (N,)
            "instruction": meta["instruction"],
            "primitive_instructions": primitive_instructions,
            "task_name": meta["task_name"],
        }


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

        self._demos = []  # List of (hdf5_path, demo_key, task_name, instruction)

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
                else:  # val
                    selected_keys = demo_keys[:val_count]

                # Read instruction from first demo's ep_meta
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

                # Actions: (N, 7)
                actions = demo["actions"][:]
                all_actions.append(actions)

                # State: eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper(2) = 16D -> pad to 19D
                eef_pos = demo["obs"]["robot0_eef_pos"][:]
                eef_quat = demo["obs"]["robot0_eef_quat"][:]
                joint_pos = demo["obs"]["robot0_joint_pos"][:]
                gripper = demo["obs"]["robot0_gripper_qpos"][:]

                # Concatenate to 16D (raw, no slicing)
                state_16d = np.concatenate([eef_pos, eef_quat, joint_pos, gripper], axis=-1)
                all_states.append(state_16d)

        # Stack and compute stats
        all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        all_states = np.concatenate(all_states, axis=0).astype(np.float32)

        # Action stats (for action_mean, action_std)
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.clip(np.std(all_actions, axis=0), 1e-6, None)

        self.action_stats = {
            "mean": self.action_mean.tolist(),
            "std": self.action_std.tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
        }

        # State stats
        self.state_stats = {
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.std(all_states, axis=0).tolist(),
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
            "q01": np.percentile(all_states, 0.5, axis=0).tolist(),
            "q99": np.percentile(all_states, 99.5, axis=0).tolist(),
        }

        # Action quantile stats
        self.action_quantile_stats = {
            "q01": np.percentile(all_actions, 0.5, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99.5, axis=0).tolist(),
        }

    def __len__(self) -> int:
        return len(self._demos)

    @property
    def samples(self):
        """Compatibility with VLADataset interface."""
        return [(i, i) for i in range(len(self._demos))]

    def get_meta(self, idx: int) -> Dict:
        """Get metadata for a demo."""
        hdf5_path, demo_key, task_name, instruction = self._demos[idx]

        # Count steps
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
        """Load a full demo's data."""
        hdf5_path, demo_key, task_name, instruction = self._demos[idx]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]

            # Camera selection (priority: agentview_left > agentview_right > eye_in_hand)
            if "robot0_agentview_left_image" in obs:
                images = obs["robot0_agentview_left_image"][:]
            elif "robot0_agentview_right_image" in obs:
                images = obs["robot0_agentview_right_image"][:]
            elif "robot0_eye_in_hand_image" in obs:
                images = obs["robot0_eye_in_hand_image"][:]
            else:
                raise ValueError(f"No supported camera found in {hdf5_path}/{demo_key}")

            # Resize images if needed
            if images.shape[1:3] != (self.image_size, self.image_size):
                resized = []
                for img in images:
                    pil_img = self.PILImage.fromarray(img)
                    pil_img = pil_img.resize((self.image_size, self.image_size), self.PILImage.BILINEAR)
                    resized.append(np.array(pil_img))
                images = np.stack(resized, axis=0)

            # Robot state: eef_pos(3) + eef_quat(4) + joint_pos(7) + gripper(2) = 16D -> pad to 19D
            eef_pos = obs["robot0_eef_pos"][:]
            eef_quat = obs["robot0_eef_quat"][:]
            joint_pos = obs["robot0_joint_pos"][:]
            gripper = obs["robot0_gripper_qpos"][:]

            robot_state = np.concatenate([eef_pos, eef_quat, joint_pos, gripper], axis=-1).astype(np.float32)

            # Actions
            actions = demo["actions"][:].astype(np.float32)

            # Phases (all zeros for RoboCasa)
            phases = np.zeros(actions.shape[0], dtype=np.int64)

        return {
            "images": images,  # (N, H, W, 3) uint8
            "robot_state": robot_state,  # (N, 16) raw
            "actions": actions,  # (N, 7)
            "phases": phases,  # (N,) all zeros
            "instruction": instruction,
            "primitive_instructions": None,
            "task_name": task_name,
        }


class VLAStepDataset:
    """Per-timestep dataset that wraps VLADataset for training.

    Lazily loads demos and yields individual timesteps.
    Uses LRU cache to reduce redundant disk I/O across workers.
    """

    def __init__(self, demo_dataset: VLADataset, cache_size: int = 8):
        import torch
        from functools import lru_cache

        self.demo_dataset = demo_dataset
        self.torch = torch

        # Build index: (demo_idx, step_idx) for all timesteps
        # Also build per-demo ranges for DemoGroupedBatchSampler
        self.index = []
        self.demo_ranges = []  # [(start_idx, end_idx), ...] per demo
        offset = 0
        for demo_idx in range(len(demo_dataset)):
            meta = demo_dataset.get_meta(demo_idx)
            n_steps = meta["n_steps"]
            for step_idx in range(n_steps):
                self.index.append((demo_idx, step_idx))
            self.demo_ranges.append((offset, offset + n_steps))
            offset += n_steps

        logger.info(f"Step dataset: {len(self.index)} timesteps from {len(demo_dataset)} demos")

        # LRU cache for demo loading (shared within each worker)
        @lru_cache(maxsize=cache_size)
        def _cached_load(demo_idx):
            return demo_dataset.load_demo(demo_idx)

        self._cached_load = _cached_load

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        demo_idx, step_idx = self.index[idx]
        data = self._cached_load(demo_idx)

        # Extract single timestep
        image = data["images"][step_idx]  # (H, W, 3)
        state = data["robot_state"][step_idx]  # (10,)
        action = data["actions"][step_idx]  # (7,)
        phase = data["phases"][step_idx]  # 0 or 1

        # Normalize image to [0, 1] (images are pre-resized to target size during preprocessing)
        image = image.astype(np.float32) / 255.0

        # Normalize action (z-score)
        action_norm = (action - self.demo_dataset.action_mean) / self.demo_dataset.action_std

        # Use per-timestep primitive instruction if available, else full-task instruction
        prim_instructions = data.get("primitive_instructions")
        if prim_instructions and step_idx < len(prim_instructions):
            instruction = prim_instructions[step_idx]
        else:
            instruction = data["instruction"]

        return {
            "image": self.torch.from_numpy(image).permute(2, 0, 1),  # (3, H, W)
            "robot_state": self.torch.from_numpy(state),
            "action": self.torch.from_numpy(action_norm),
            "phase": self.torch.tensor(phase, dtype=self.torch.long),
            "instruction": instruction,
        }


class ActionChunkDataset:
    """Chunk-based dataset that wraps VLADataset for action-chunk models (SmolVLA, PI0.5, GROOT).

    Returns action chunks of fixed size with padding for short demos.
    Uses LRU cache for demo loading (same pattern as VLAStepDataset).
    """

    def __init__(self, demo_dataset: VLADataset, chunk_size: int, stride: int = 1, cache_size: int = 8):
        import torch
        from functools import lru_cache

        self.demo_dataset = demo_dataset
        self.chunk_size = chunk_size
        self.torch = torch

        # Build index: (demo_idx, start_step) for each valid chunk start
        # Also build demo_ranges for DemoGroupedBatchSampler compatibility
        self.index = []
        self.demo_ranges = []
        offset = 0
        for demo_idx in range(len(demo_dataset)):
            meta = demo_dataset.get_meta(demo_idx)
            n_steps = meta["n_steps"]
            demo_start = offset
            # Generate chunk starts at every `stride` steps
            for start in range(0, n_steps, stride):
                self.index.append((demo_idx, start))
                offset += 1
            self.demo_ranges.append((demo_start, offset))

        logger.info(
            f"ActionChunkDataset: {len(self.index)} chunks (chunk_size={chunk_size}, stride={stride}) "
            f"from {len(demo_dataset)} demos"
        )

        # LRU cache for demo loading
        @lru_cache(maxsize=cache_size)
        def _cached_load(demo_idx):
            return demo_dataset.load_demo(demo_idx)

        self._cached_load = _cached_load

    def __len__(self) -> int:
        return len(self.index)

    def _get_action_chunk(self, actions: np.ndarray, start_step: int):
        """Extract action chunk with padding if needed.

        Args:
            actions: Full demo actions array (T, action_dim)
            start_step: Starting timestep

        Returns:
            action_chunk: (chunk_size, action_dim) float32 array
            is_pad: (chunk_size,) bool array -- True where padded
        """
        total_steps = actions.shape[0]
        end_step = min(start_step + self.chunk_size, total_steps)
        chunk = actions[start_step:end_step]
        actual_len = chunk.shape[0]

        # Pad by repeating last action if needed
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

        # Get action chunk
        action_chunk, action_pad_mask = self._get_action_chunk(data["actions"], start_step)

        # Get image at start_step (the observation that triggers the action chunk)
        image = data["images"][start_step].astype(np.float32) / 255.0

        # Get wrist image if available
        wrist_image = None
        if "wrist_images" in data and data["wrist_images"] is not None:
            wrist_image = data["wrist_images"][start_step].astype(np.float32) / 255.0

        # Get robot state at start_step
        robot_state = data["robot_state"][start_step].astype(np.float32)

        # Get instruction
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
    """SmolVLA-compatible dataset adapter wrapping ActionChunkDataset.
    
    Produces batches with exact keys expected by SmolVLAPolicy.forward():
    - observation.images.camera: (3, 512, 512) float32 [0, 1]
    - observation.state: (19,) float32 MEAN_STD normalized
    - observation.language.tokens: (48,) long
    - observation.language.attention_mask: (48,) long
    - action: (50, 7) float32 MEAN_STD normalized
    - actions_is_pad: (50,) bool
    """

    def __init__(self, chunk_dataset: ActionChunkDataset, state_stats: Dict, action_stats: Dict):
        from transformers import AutoTokenizer
        import torch
        import torch.nn.functional as F
        
        self.chunk_ds = chunk_dataset
        self.torch = torch
        self.F = F
        
        # Normalization stats
        _mean = np.array(state_stats["mean"], dtype=np.float32)
        _std = np.array(state_stats["std"], dtype=np.float32)
        if len(_mean) == 19:  # preprocessed 19D format
            self.state_mean = _mean[_STATE_KEEP_IDX]
            self.state_std = _std[_STATE_KEEP_IDX]
        else:  # raw format (e.g. 16D)
            self.state_mean = _mean
            self.state_std = _std
        self.action_mean = np.array(action_stats["mean"], dtype=np.float32)
        self.action_std = np.array(action_stats["std"], dtype=np.float32)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        
        # Build instruction cache
        self._instruction_cache = {}
        self._build_instruction_cache()
        
        logger.info(f"SmolVLAChunkDataset: {len(chunk_dataset)} samples, {len(self._instruction_cache)} unique instructions")
    
    def _build_instruction_cache(self):
        """Pre-tokenize all unique instructions."""
        instructions = set()
        for demo_idx in range(len(self.chunk_ds.demo_dataset)):
            meta = self.chunk_ds.demo_dataset.get_meta(demo_idx)
            instructions.add(meta["instruction"])
        
        for inst in instructions:
            text = inst + "\n"  # SmolVLA requires trailing newline
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
        
        # Resize image to 512x512
        img = sample["images"]  # (3, 224, 224)
        img = self.F.interpolate(img.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False).squeeze(0)
        
        # Normalize state with MEAN_STD
        state = sample["robot_state"].numpy()
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        
        # Normalize action with MEAN_STD
        action = sample["action_chunk"].numpy()
        action_norm = (action - self.action_mean) / (self.action_std + 1e-8)
        
        # Get cached tokens or tokenize on-the-fly
        inst = sample["instruction"]
        if inst in self._instruction_cache:
            tokens = self._instruction_cache[inst]
        else:
            # Tokenize and cache
            text = inst + "\n"
            tok = self.tokenizer(text, max_length=48, padding="max_length", truncation=True, return_tensors="pt")
            tokens = {"input_ids": tok["input_ids"].squeeze(0), "attention_mask": tok["attention_mask"].squeeze(0)}
            self._instruction_cache[inst] = tokens
        
        return {
            "observation.images.camera1": img,  # SmolVLA expects camera1/2/3
            "observation.state": self.torch.from_numpy(state_norm.astype(np.float32)),
            "observation.language.tokens": tokens["input_ids"],
            "observation.language.attention_mask": tokens["attention_mask"].bool(),
            "action": self.torch.from_numpy(action_norm.astype(np.float32)),
            "actions_is_pad": sample["action_pad_mask"],
        }


class PI05ChunkDataset:
    """PI0.5-compatible dataset adapter wrapping ActionChunkDataset.
    
    PI0.5 embeds state INTO the language prompt via discretization.
    Produces batches with exact keys expected by PI05Policy.forward():
    - observation.images.camera: (3, 224, 224) float32 [0, 1]
    - observation.language.tokens: (200,) long (includes discretized state)
    - observation.language.attention_mask: (200,) long
    - action: (50, 7) float32 QUANTILES normalized [-1, 1]
    """

    def __init__(self, chunk_dataset: ActionChunkDataset, state_stats: Dict, action_stats: Dict):
        from transformers import AutoTokenizer
        import torch
        
        self.chunk_ds = chunk_dataset
        self.torch = torch
        
        # State quantile stats for discretization
        _q01 = np.array(state_stats["q01"], dtype=np.float32)
        _q99 = np.array(state_stats["q99"], dtype=np.float32)
        if len(_q01) == 19:  # preprocessed 19D format
            self.state_q01 = _q01[_STATE_KEEP_IDX]
            self.state_q99 = _q99[_STATE_KEEP_IDX]
        else:  # raw format
            self.state_q01 = _q01
            self.state_q99 = _q99
        
        # Action quantile stats
        self.action_q01 = np.array(action_stats["q01"], dtype=np.float32)
        self.action_q99 = np.array(action_stats["q99"], dtype=np.float32)
        
        # Tokenizer - must match PI0.5's Gemma backbone (NOT SmolVLA tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b", padding_side="right")
        
        # Cache instructions only (state varies per sample, so full prompt can't be cached)
        self._instruction_set = set()
        for demo_idx in range(len(self.chunk_ds.demo_dataset)):
            meta = self.chunk_ds.demo_dataset.get_meta(demo_idx)
            self._instruction_set.add(meta["instruction"])
        
        logger.info(f"PI05ChunkDataset: {len(chunk_dataset)} samples, {len(self._instruction_set)} unique instructions")
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize state to 256 bins following PI05 reference implementation."""
        # Normalize to [-1, 1] using quantiles
        state_norm = (state - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-8) * 2 - 1
        state_norm = np.clip(state_norm, -1, 1)
        
        # Pad to 32 dims
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm
        
        # Discretize to 256 bins
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
        
        # Image stays at 224x224
        img = sample["images"]  # (3, 224, 224)
        
        # Build prompt with discretized state
        state = sample["robot_state"].numpy()
        state_str = self._discretize_state(state)
        prompt = f"Task: {sample['instruction']}, State: {state_str};\nAction: "
        
        # Tokenize
        tokens = self.tokenizer(
            prompt, max_length=200, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # Normalize action with QUANTILES
        action = sample["action_chunk"].numpy()
        action_norm = (action - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-8) * 2 - 1
        action_norm = np.clip(action_norm, -1, 1)
        
        return {
            "observation.images.base_0_rgb": img,
            "observation.language.tokens": tokens["input_ids"].squeeze(0),
            "observation.language.attention_mask": tokens["attention_mask"].squeeze(0).bool(),
            "action": self.torch.from_numpy(action_norm.astype(np.float32)),
        }


class GrootChunkDataset:
    """GROOT N1.5-compatible dataset adapter wrapping ActionChunkDataset.
    
    GROOT uses Eagle VLM encoding for images/text and MIN_MAX normalization.
    Produces batches with exact keys expected by GrootPolicy.forward():
    - state: (1, 64) float32 MIN_MAX normalized [-1, 1]
    - state_mask: (1, 64) bool
    - action: (16, 32) float32 MIN_MAX normalized [-1, 1]  
    - action_mask: (16, 32) bool
    - embodiment_id: long scalar
    - video: (1, num_cams, 3, H, W) uint8 for Eagle encoding
    - language: str instruction
    
    Note: Eagle encoding is done in collate_fn, not per-sample.
    """

    def __init__(self, chunk_dataset: ActionChunkDataset, state_stats: Dict, action_stats: Dict):
        import torch
        
        self.chunk_ds = chunk_dataset
        self.torch = torch
        
        # MIN_MAX stats (using 0.5/99.5 percentiles as min/max)
        _smin = np.array(state_stats["min"], dtype=np.float32)
        _smax = np.array(state_stats["max"], dtype=np.float32)
        if len(_smin) == 19:  # preprocessed 19D format
            self.state_min = _smin[_STATE_KEEP_IDX]
            self.state_max = _smax[_STATE_KEEP_IDX]
        else:  # raw format
            self.state_min = _smin
            self.state_max = _smax
        self.action_min = np.array(action_stats["min"], dtype=np.float32)
        self.action_max = np.array(action_stats["max"], dtype=np.float32)
        
        # Embodiment ID for new embodiments
        self.embodiment_id = 31  # "new_embodiment" in GROOT
        
        logger.info(f"GrootChunkDataset: {len(chunk_dataset)} samples (chunk_size=16)")
    
    def __len__(self):
        return len(self.chunk_ds)
    
    @property
    def demo_ranges(self):
        return self.chunk_ds.demo_ranges
    
    def __getitem__(self, idx):
        sample = self.chunk_ds[idx]
        
        # State: normalize with MIN_MAX, pad to 64
        state = sample["robot_state"].numpy()  # (10,)
        state_range = self.state_max - self.state_min + 1e-8
        state_norm = 2 * (state - self.state_min) / state_range - 1
        state_norm = np.clip(state_norm, -1, 1)
        
        state_padded = np.zeros(64, dtype=np.float32)
        state_padded[:len(state_norm)] = state_norm
        state_mask = np.zeros(64, dtype=bool)
        state_mask[:len(state_norm)] = True
        
        # Action: normalize with MIN_MAX, pad to 32
        action = sample["action_chunk"].numpy()  # (chunk_size, action_dim)
        action_dim = action.shape[1]
        action_range = self.action_max - self.action_min + 1e-8
        action_norm = 2 * (action - self.action_min) / action_range - 1
        action_norm = np.clip(action_norm, -1, 1)

        action_padded = np.zeros((16, 32), dtype=np.float32)
        action_padded[:, :action_dim] = action_norm
        action_mask = np.zeros((16, 32), dtype=bool)
        action_mask[:, :action_dim] = True
        # Mark padded timesteps
        if sample["action_pad_mask"].any():
            pad_start = (~sample["action_pad_mask"].numpy()).sum()
            action_mask[pad_start:, :] = False
        
        # Video: uint8 for Eagle encoding (1, num_cams, 3, H, W)
        img = (sample["images"].numpy() * 255).astype(np.uint8)  # (3, 224, 224)
        video = img[np.newaxis, np.newaxis, ...]  # (1, 1, 3, 224, 224)
        
        return {
            "state": self.torch.from_numpy(state_padded).unsqueeze(0),  # (1, 64)
            "state_mask": self.torch.from_numpy(state_mask).unsqueeze(0),  # (1, 64)
            "action": self.torch.from_numpy(action_padded),  # (16, 32)
            "action_mask": self.torch.from_numpy(action_mask),  # (16, 32)
            "embodiment_id": self.torch.tensor(self.embodiment_id, dtype=self.torch.long),
            "video": video,  # numpy for Eagle collate
            "language": sample["instruction"],
        }


def groot_collate_fn(batch, eagle_processor=None):
    """Collate function for GROOT that handles Eagle encoding.
    
    If eagle_processor is provided, encodes video+language into eagle_* tensors.
    Otherwise, returns raw batch for debugging.
    """
    import torch
    
    result = {
        "state": torch.stack([b["state"] for b in batch]),
        "state_mask": torch.stack([b["state_mask"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "action_mask": torch.stack([b["action_mask"] for b in batch]),
        "embodiment_id": torch.stack([b["embodiment_id"] for b in batch]),
    }
    
    if eagle_processor is not None:
        # Eagle encoding
        from PIL import Image
        import numpy as np

        images = []
        texts = []
        for b in batch:
            # video: (1, num_cams, 3, H, W) -> list of PIL Images
            video = b["video"]  # (1, 1, 3, 224, 224)
            for t in range(video.shape[0]):
                for c in range(video.shape[1]):
                    img = video[t, c].transpose(1, 2, 0)  # (H, W, 3)
                    # Convert from float tensor to uint8 numpy array
                    img_np = (img * 255).clip(0, 255).astype(np.uint8)
                    images.append(Image.fromarray(img_np))
            # Format text with image placeholder: "<image-1> task description"
            texts.append(f"<image-1> {b['language']}")
        
        # Batch encode
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
        # Pass through for debugging
        result["video"] = [b["video"] for b in batch]
        result["language"] = [b["language"] for b in batch]
    
    return result


class DemoGroupedBatchSampler:
    """Batch sampler that groups timesteps from the same demo.

    Minimizes cache misses by keeping consecutive batches from the same demo.
    Demo order is shuffled each epoch for training diversity.
    Supports distributed training by sharding demos across ranks.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across ranks."""
        self._epoch = epoch

    def __iter__(self):
        import random

        demo_order = list(range(len(self.dataset.demo_ranges)))
        if self.shuffle:
            # Epoch-based seed ensures all ranks get the same order before sharding
            rng = random.Random(self._epoch)
            rng.shuffle(demo_order)

        # Shard demos across ranks for distributed training
        if self.world_size > 1:
            demo_order = demo_order[self.rank::self.world_size]

        # Collect all step indices for this rank
        all_indices = []
        step_rng = random.Random(self._epoch + self.rank)
        for demo_idx in demo_order:
            start, end = self.dataset.demo_ranges[demo_idx]
            step_indices = list(range(start, end))
            if self.shuffle:
                step_rng.shuffle(step_indices)
            all_indices.extend(step_indices)

        # All ranks must yield the EXACT same number of batches to avoid NCCL
        # deadlocks from mismatched allreduce calls at epoch boundaries.
        # Pad shorter ranks by cycling data; truncate longer ranks.
        target_batches = len(self)
        target_samples = target_batches * self.batch_size

        if len(all_indices) < target_samples:
            # Pad by cycling from the beginning
            shortage = target_samples - len(all_indices)
            all_indices.extend(all_indices[:shortage])

        # Yield exactly target_batches full batches
        for i in range(target_batches):
            start = i * self.batch_size
            yield all_indices[start : start + self.batch_size]

    def __len__(self):
        # Deterministic: same value for all ranks
        return len(self.dataset) // (self.batch_size * self.world_size)


class OpenVLAStepDataset:
    """OpenVLA-specific step dataset with preprocessing in DataLoader workers.

    Moves tokenization, image processing, and action discretization from
    the training thread to DataLoader workers for better GPU utilization.

    To add similar optimization for other backends, follow this pattern:
    create a backend-specific dataset class that wraps VLAStepDataset and
    does model-specific preprocessing in __getitem__().
    """

    IGNORE_INDEX = -100

    def __init__(self, step_dataset, processor, bin_centers, vocab_size,
                 cache_dir=None, split="train"):
        self.step_dataset = step_dataset
        self.demo_ranges = step_dataset.demo_ranges
        self.processor = processor
        self.bin_centers = (
            bin_centers.astype(np.float32)
            if hasattr(bin_centers, "astype")
            else np.array(bin_centers, dtype=np.float32)
        )
        self.vocab_size = int(vocab_size)

        # Pre-tokenize all unique instructions
        self._token_cache = {}
        self._build_instruction_cache()

        # Pixel cache: stored as a single bf16 tensor on disk, loaded via mmap.
        # Multiple processes share the same physical RAM pages through mmap.
        distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, "is_initialized") else False
        rank = torch.distributed.get_rank() if distributed else 0

        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"openvla_pv_{split}.pt")

        need_compute = cache_path is not None and not os.path.exists(cache_path)

        if need_compute and rank == 0:
            # Rank 0: compute all pixel_values and save to disk
            logger.info("OpenVLAStepDataset: pre-processing all images (rank 0, one-time cost)...")
            total_steps = len(self.step_dataset)

            # Determine pixel_values shape from first sample
            d0, s0 = self.step_dataset.index[0]
            first_data = self.step_dataset._cached_load(d0)
            first_img = first_data["images"][s0]
            if first_img.ndim == 3 and first_img.shape[0] == 3:
                first_img = (first_img.transpose(1, 2, 0) * 255).astype(np.uint8)
            pv0 = self.processor.image_processor(
                PILImage.fromarray(first_img), return_tensors="pt"
            )["pixel_values"].squeeze(0)

            all_pv = torch.empty(total_steps, *pv0.shape, dtype=torch.bfloat16)
            all_pv[0] = pv0.to(torch.bfloat16)

            prev_demo = d0
            cur_data = first_data
            for i in range(1, total_steps):
                demo_idx, step_idx = self.step_dataset.index[i]
                if demo_idx != prev_demo:
                    cur_data = self.step_dataset._cached_load(demo_idx)
                    prev_demo = demo_idx
                img = cur_data["images"][step_idx]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                pv = self.processor.image_processor(
                    PILImage.fromarray(img), return_tensors="pt"
                )["pixel_values"].squeeze(0)
                all_pv[i] = pv.to(torch.bfloat16)
                if (i + 1) % 10000 == 0 or i == total_steps - 1:
                    logger.info(f"  Preprocessed {i + 1}/{total_steps} steps")

            torch.save(all_pv, cache_path)
            size_gb = os.path.getsize(cache_path) / 1e9
            logger.info(f"OpenVLAStepDataset: saved cache to {cache_path} ({size_gb:.1f} GB)")
            del all_pv

        # Wait for rank 0 to finish saving
        if distributed:
            torch.distributed.barrier()

        # All ranks: load via mmap (shared physical RAM)
        self._pv_tensor = None
        if cache_path and os.path.exists(cache_path):
            logger.info(f"OpenVLAStepDataset: loading cache via mmap ({cache_path})")
            self._pv_tensor = torch.load(cache_path, mmap=True, weights_only=True)
            logger.info(f"OpenVLAStepDataset: mmap loaded {self._pv_tensor.shape[0]} pixel_values")

    def _build_instruction_cache(self):
        """Pre-tokenize all unique instructions for cache-friendly lookup."""
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

        action = data["actions"][step_idx]  # (7,) raw float

        # Normalize action (z-score)
        demo_ds = self.step_dataset.demo_dataset
        action_norm = (
            action.astype(np.float32) - demo_ds.action_mean
        ) / demo_ds.action_std

        # Get instruction
        prim = data.get("primitive_instructions")
        if prim and step_idx < len(prim):
            instruction = prim[step_idx]
        else:
            instruction = data["instruction"]

        # === OpenVLA-specific preprocessing ===

        # 1. Get pre-processed pixel_values (bf16 mmap or compute on the fly)
        if self._pv_tensor is not None:
            pixel_values = self._pv_tensor[idx]
        else:
            img = data["images"][step_idx]
            if img.ndim == 3 and img.shape[0] == 3:
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img)
            pixel_values = self.processor.image_processor(
                pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(0)

        # 2. Look up cached tokenized instruction (no tokenizer call)
        input_ids = self._token_cache[instruction].clone()

        # 3. Discretize actions → token IDs (vectorized, no loops)
        actions_clipped = np.clip(action_norm, -1.0, 1.0)
        bin_indices = np.argmin(
            np.abs(actions_clipped[:, None] - self.bin_centers[None, :]), axis=1
        )
        action_tokens = torch.tensor(
            self.vocab_size - 1 - bin_indices, dtype=torch.long
        )

        # 4. Build full input_ids and labels
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
    """Collate for OpenVLA: pad input_ids/labels to uniform length, stack pixel_values."""
    IGNORE_INDEX = -100
    batch_size = len(batch)
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_token_id = 0

    padded_input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long
    )
    padded_labels = torch.full(
        (batch_size, max_len), IGNORE_INDEX, dtype=torch.long
    )
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
    """Per-dimension weighted MSE loss for action prediction."""

    def __init__(self, dim_weights: List[float], device: str = "cuda"):
        import torch

        self.weights = torch.tensor(dim_weights, dtype=torch.float32, device=device)

    def __call__(self, predicted, target):
        import torch.nn.functional as F

        per_dim = (predicted - target) ** 2  # (batch, 7)
        weighted = per_dim * self.weights.unsqueeze(0)
        return weighted.mean()


def setup_model_and_lora(config: Dict, local_rank: int = 0):
    """Load base model and apply LoRA configuration.

    Args:
        config: Full training configuration dict.
        local_rank: GPU index for this process (for DDP).

    Returns:
        (model, processor/tokenizer)
    """
    import torch
    from peft import LoraConfig, get_peft_model

    backend = config["model"]["backend"]
    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    quant_cfg = config["quantization"]

    logger.info(f"Loading model: {model_name} (backend={backend})")

    if backend == "openvla":
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # flash_attn broken (ABI mismatch with torch 2.7.1)
        # SDPA has built-in flash backend via torch.backends.cuda.flash_sdp_enabled()
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
            # Pin model to specific GPU for DDP compatibility
            load_kwargs["device_map"] = {"": local_rank}
        else:
            # DDP: pin model to specific GPU; single-GPU: auto placement
            distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, "is_initialized") else False
            if distributed:
                load_kwargs["device_map"] = {"": local_rank}
            else:
                load_kwargs["device_map"] = "auto"

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
            logger.info(f"Attention: {load_kwargs['attn_implementation']}")
        except (AttributeError, ValueError, TypeError, NotImplementedError) as e:
            logger.warning(f"Attention '{attn_impl}' failed ({type(e).__name__}: {e}), falling back to eager")
            load_kwargs["attn_implementation"] = "eager"
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)

    elif backend == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(model_name)
        policy = policy.to(dtype=torch.bfloat16)
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

    elif backend in ("pi05", "pi0.5"):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.configs.types import PolicyFeature, FeatureType

        policy = PI05Policy.from_pretrained(model_name)
        policy = policy.to(dtype=torch.bfloat16)

        # Override input_features to match our dataset (single camera)
        # Clear existing image features and add only our camera key
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

        # Load base model WITHOUT internal LoRA — use PEFT instead
        policy = GrootPolicy.from_pretrained(model_name)
        policy = policy.to(dtype=torch.bfloat16)

        # Auto-detect PEFT targets: Linear modules in action_head get LoRA,
        # non-Linear trainable leaf modules go into modules_to_save.
        target_modules = []
        modules_to_save = []
        for name, mod in policy.named_modules():
            if "action_head" not in name:
                continue
            if list(mod.children()):
                continue  # skip non-leaf
            has_params = any(True for _ in mod.parameters(recurse=False))
            if not has_params:
                continue
            if isinstance(mod, torch.nn.Linear):
                target_modules.append(name)
            else:
                modules_to_save.append(name)

        logger.info(f"GROOT PEFT targets: {len(target_modules)} Linear (LoRA), "
                     f"{len(modules_to_save)} non-Linear (modules_to_save)")

        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=target_modules,
            modules_to_save=modules_to_save,
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

        # Eagle processor needed for collate_fn
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

    # Apply LoRA
    if backend in ("openvla", "hf_vlm"):
        # Standard PEFT LoRA
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    return model, processor


def _load_lerobot_model(backend: str, model_name: str, config: Dict):
    """Load a LeRobot-based model."""
    import torch

    policy_map = {
        "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        "pi05": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        "pi0.5": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        "groot_n1.5": ("lerobot.policies.groot.modeling_groot", "GrootPolicy"),
    }

    if backend not in policy_map:
        raise ValueError(f"Unknown LeRobot backend: {backend}")

    module_path, class_name = policy_map[backend]
    import importlib
    module = importlib.import_module(module_path)
    policy_cls = getattr(module, class_name)

    policy = policy_cls.from_pretrained(model_name)
    return policy


def train(config: Dict) -> None:
    """Main training loop."""
    import torch
    from torch.utils.data import DataLoader

    adapter_cfg = config["adapter"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    # Determine output directory
    adapter_name = adapter_cfg["name"]
    task_name = data_cfg.get("task") or "general"
    output_dir = os.path.join(output_cfg["base_dir"], task_name, f"{adapter_name}_adapter")
    os.makedirs(output_dir, exist_ok=True)

    # Save full config
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Output directory: {output_dir}")

    # Create datasets
    hdf5_dir = data_cfg.get("hdf5_dir")
    if hdf5_dir:
        # Direct HDF5 mode: read raw RoboCasa data without preprocessing
        train_demo_ds = RawHDF5Dataset(
            hdf5_root=hdf5_dir,
            task_filter=data_cfg.get("task"),
            max_samples=data_cfg.get("max_samples"),
            image_size=config["model"]["image_size"],
            split="train",
        )
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
    else:
        # NPZ mode: read preprocessed data
        processed_dir = data_cfg["processed_dir"]
        metadata_path = os.path.join(processed_dir, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        train_demo_ds = VLADataset(
            data_dir=processed_dir,
            action_stats=metadata["action_stats"],
            task_filter=data_cfg.get("task"),
            max_samples=data_cfg.get("max_samples"),
            image_size=config["model"]["image_size"],
            split="train",
        )
        val_demo_ds = VLADataset(
            data_dir=processed_dir,
            action_stats=metadata["action_stats"],
            task_filter=data_cfg.get("task"),
            image_size=config["model"]["image_size"],
            split="val",
        )
        ext_meta = None

    train_ds = VLAStepDataset(train_demo_ds)
    val_ds = VLAStepDataset(val_demo_ds)

    logger.info(f"Train: {len(train_ds)} steps, Val: {len(val_ds)} steps")

    # Distributed training setup (detect before model loading for device placement)
    distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, "is_initialized") else False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size() if distributed else 1

    # Setup model (pass local_rank for correct GPU placement in DDP)
    model, processor = setup_model_and_lora(config, local_rank=local_rank)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not hasattr(model, "hf_device_map"):
        model = model.to(device)

    # For distributed training with QLoRA, skip DDP wrapping (quantized params break DDP).
    # Instead, manually sync LoRA gradients after each backward pass.
    if distributed:
        logger.info(f"Distributed: rank {local_rank}/{world_size} (manual gradient sync for QLoRA)")

    # Speed optimizations
    backend = config["model"]["backend"]
    if train_cfg.get("gradient_checkpointing", False):
        gc_model = model.module if hasattr(model, "module") else model
        if backend in ("pi05", "pi0.5"):
            # PI05Policy doesn't inherit PreTrainedModel; set flag on inner PI05Pytorch
            inner = gc_model
            # Unwrap PEFT layers to find PI05Pytorch
            if hasattr(inner, 'base_model'):
                inner = inner.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
            if hasattr(inner, 'model'):  # PI05Policy.model -> PI05Pytorch
                inner = inner.model
            if hasattr(inner, 'gradient_checkpointing_enabled'):
                inner.gradient_checkpointing_enabled = True
                # Also enable on paligemma submodels
                if hasattr(inner, 'paligemma_with_expert'):
                    for submodel in [inner.paligemma_with_expert]:
                        if hasattr(submodel, 'gradient_checkpointing_enable'):
                            submodel.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs={"use_reentrant": False}
                            )
                logger.info("Gradient checkpointing enabled (PI05 manual flag)")
            else:
                logger.warning("Could not enable gradient checkpointing for PI05")
        else:
            gc_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled")

    if train_cfg.get("compile", False):
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    # Loss function with dimension weights (used for non-OpenVLA backends)
    loss_fn = WeightedActionLoss(adapter_cfg["loss_weights"], device=str(device))

    # Wrap datasets for backend-specific preprocessing (moves CPU work to DataLoader workers)
    if backend == "openvla":
        # Unwrap DDP/LoRA to get base model attributes
        raw_model = model.module if hasattr(model, "module") else model
        raw_model = raw_model.base_model.model if hasattr(raw_model, "base_model") else raw_model
        pixel_cache_dir = os.path.join(output_dir, "pixel_cache")
        train_ds = OpenVLAStepDataset(
            train_ds, processor, raw_model.bin_centers, raw_model.vocab_size,
            cache_dir=pixel_cache_dir, split="train"
        )
        val_ds = OpenVLAStepDataset(
            val_ds, processor, raw_model.bin_centers, raw_model.vocab_size,
            cache_dir=pixel_cache_dir, split="val"
        )
    elif backend == "smolvla":
        # Load extended stats for state normalization
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 50)
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        train_ds = SmolVLAChunkDataset(train_chunk, ext_meta["state_stats"], metadata["action_stats"])
        val_ds = SmolVLAChunkDataset(val_chunk, ext_meta["state_stats"], metadata["action_stats"])
    elif backend in ("pi05", "pi0.5"):
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 50)
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        train_ds = PI05ChunkDataset(train_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"])
        val_ds = PI05ChunkDataset(val_chunk, ext_meta["state_stats"], ext_meta["action_quantile_stats"])
    elif backend == "groot_n1.5":
        if ext_meta is None:
            ext_meta_path = os.path.join(processed_dir, "metadata_extended.json")
            with open(ext_meta_path) as f:
                ext_meta = json.load(f)
        chunk_size = config["model"].get("chunk_size", 16)  # GROOT uses 16
        chunk_stride = config["model"].get("chunk_stride", 1)
        train_chunk = ActionChunkDataset(train_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        val_chunk = ActionChunkDataset(val_demo_ds, chunk_size=chunk_size, stride=chunk_stride)
        train_ds = GrootChunkDataset(train_chunk, ext_meta["state_stats"], metadata["action_stats"])
        val_ds = GrootChunkDataset(val_chunk, ext_meta["state_stats"], metadata["action_stats"])

    # Optimizer
    opt_name = train_cfg.get("optimizer", "adamw")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    betas = (train_cfg.get("adam_beta1", 0.95), train_cfg.get("adam_beta2", 0.999))
    if opt_name == "adamw_8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            betas=betas,
        )
        logger.info("Using 8-bit AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            betas=betas,
        )
    logger.info(f"Optimizer: {opt_name}, betas={betas}")

    # DataLoader with backend-specific collate function
    if backend == "openvla":
        active_collate_fn = openvla_collate_fn
    elif backend == "groot_n1.5":
        # GROOT needs Eagle encoding in collate
        from functools import partial
        active_collate_fn = partial(groot_collate_fn, eagle_processor=processor)
    elif backend in ("smolvla", "pi05", "pi0.5"):
        # SmolVLA and PI05 use standard stack collate
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
        # Legacy default collate for VLAStepDataset
        def active_collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "robot_state": torch.stack([b["robot_state"] for b in batch]),
                "action": torch.stack([b["action"] for b in batch]),
                "phase": torch.stack([b["phase"] for b in batch]),
                "instruction": [b["instruction"] for b in batch],
            }

    num_workers = train_cfg.get("dataloader_num_workers", 4)
    # /dev/shm check removed: torch.multiprocessing sharing_strategy is
    # already set to "file_system" (line 51), so /dev/shm size is irrelevant.
    logger.info(f"DataLoader num_workers={num_workers} (file_system sharing strategy)")
    if distributed:
        num_workers = 0  # Distributed mode also needs 0 workers
        logger.info(f"Distributed: using {num_workers} workers (shared memory constraint)")

    train_sampler = DemoGroupedBatchSampler(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        rank=local_rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=active_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_sampler = DemoGroupedBatchSampler(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        rank=local_rank,
        world_size=world_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=0,
        collate_fn=active_collate_fn,
        pin_memory=True,
    )

    # Gradient accumulation and max_steps
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

    # Learning rate scheduler (total_steps = actual optimizer steps, not micro-batches)
    scheduler_type = train_cfg.get("scheduler_type", "onecycle")
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif scheduler_type == "cosine":
        warmup_steps = int(total_optimizer_steps * train_cfg.get("warmup_ratio", 0.0))
        if warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_optimizer_steps - warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_optimizer_steps
            )
    else:  # "onecycle" (default, backward compatible)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg["learning_rate"],
            total_steps=total_optimizer_steps,
            pct_start=train_cfg["warmup_ratio"],
        )
    logger.info(f"Scheduler: {scheduler_type}")

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    early_stopping_patience = train_cfg.get("early_stopping_patience", 0)  # 0 = disabled
    no_improve_count = 0
    best_model_checkpoint = None
    log_history = []

    import time
    train_start_time = time.time()

    effective_batch = train_cfg["batch_size"] * grad_accum * world_size
    logger.info("Starting training...")
    logger.info(f"  Adapter: {adapter_name}")
    logger.info(f"  Task: {task_name}")
    logger.info(f"  Loss weights: {adapter_cfg['loss_weights']}")
    if world_size > 1:
        logger.info(f"  Batch size: {train_cfg['batch_size']} x {grad_accum} x {world_size}gpu = {effective_batch}")
    else:
        logger.info(f"  Batch size: {train_cfg['batch_size']} x {grad_accum} = {effective_batch}")
    logger.info(f"  Epochs: {num_epochs}" + (f" (auto from max_steps={max_steps})" if max_steps > 0 else ""))
    logger.info(f"  Optimizer steps: {total_optimizer_steps}" + (f" (max_steps={max_steps})" if max_steps > 0 else ""))
    logger.info(f"  Steps/epoch: {optimizer_steps_per_epoch}")

    from tqdm import tqdm

    model.train()

    for epoch in range(num_epochs):
        # Sync all ranks before starting new epoch
        if distributed:
            torch.distributed.barrier()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0

        # Progress bar for each epoch
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for batch_idx, batch in pbar:
            with torch.amp.autocast("cuda", enabled=train_cfg.get("bf16", True), dtype=torch.bfloat16):
                if backend == "openvla":
                    # Batch already preprocessed by OpenVLAStepDataset
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    loss = outputs.loss / grad_accum
                elif backend in ("smolvla", "pi05", "pi0.5", "groot_n1.5"):
                    # Native LeRobot policy forward -- batch already preprocessed by model-specific dataset
                    batch_on_device = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    loss_out = model.forward(batch_on_device)
                    # All LeRobot models return (loss, loss_dict) with default reduction="mean"
                    if isinstance(loss_out, tuple):
                        loss = loss_out[0] / grad_accum
                    else:
                        loss = loss_out / grad_accum
                else:
                    raise ValueError(f"Unknown backend: {backend}")

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # Sync LoRA gradients across ranks (manual AllReduce for QLoRA compat)
                if distributed:
                    for p in trainable_params:
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                lr = scheduler.get_last_lr()[0]
                step_loss = loss.item() * grad_accum

                # Update progress bar
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "lr": f"{lr:.2e}",
                })

                if global_step % train_cfg["logging_steps"] == 0:
                    log_history.append({
                        "epoch": epoch + (batch_idx + 1) / len(train_loader),
                        "step": global_step,
                        "loss": step_loss,
                        "learning_rate": lr,
                    })

                if global_step % train_cfg["save_steps"] == 0:
                    _save_checkpoint(model, output_dir, global_step)

                if max_steps > 0 and global_step >= max_steps:
                    break

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        pbar.close()
        avg_epoch_loss = epoch_loss / max(n_batches, 1)

        # Validation (average loss across ranks for distributed)
        val_loss = _validate(model, processor, val_loader, loss_fn, device, train_cfg, backend)
        if distributed:
            val_loss_t = torch.tensor(val_loss, device=device)
            torch.distributed.all_reduce(val_loss_t, op=torch.distributed.ReduceOp.AVG)
            val_loss = val_loss_t.item()
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Log epoch metrics
        log_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": avg_epoch_loss,
            "eval_loss": val_loss,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_checkpoint = os.path.join(output_dir, "checkpoint-best")
            _save_checkpoint(model, output_dir, "best")
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1
            logger.info(f"  No improvement for {no_improve_count}/{early_stopping_patience} epochs")
            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                break

        if max_steps > 0 and global_step >= max_steps:
            logger.info(f"Reached max_steps={max_steps}, stopping.")
            break

        # Sync all ranks at end of epoch before next iteration
        if distributed:
            torch.distributed.barrier()

    # Save final model
    _save_checkpoint(model, output_dir, "final")

    # Save trainer_state.json (rank 0 only)
    rank = int(os.environ.get("RANK", 0))
    train_end_time = time.time()
    if rank == 0:
        trainer_state = {
            "best_model_checkpoint": best_model_checkpoint,
            "best_metric": best_val_loss,
            "epoch": epoch + 1,
            "global_step": global_step,
            "max_steps": max_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "num_train_epochs": num_epochs,
            "total_flos": 0,
            "train_batch_size": train_cfg["batch_size"],
            "train_samples": len(train_ds),
            "eval_samples": len(val_ds),
            "learning_rate": train_cfg["learning_rate"],
            "train_runtime": train_end_time - train_start_time,
            "log_history": log_history,
        }
        trainer_state_path = os.path.join(output_dir, "trainer_state.json")
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)
        logger.info(f"Trainer state saved to: {trainer_state_path}")

    logger.info(f"Training complete! Adapters saved to {output_dir}")


@torch.no_grad()
def _validate(model, processor, val_loader, loss_fn, device, train_cfg, backend):
    """Run validation and return average loss."""
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
                # Native LeRobot policy forward
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
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


def _save_checkpoint(model, output_dir: str, step) -> None:
    """Save LoRA adapter weights. Only rank 0 saves in distributed training."""
    # Only save on rank 0 for distributed training
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return

    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    # Unwrap DDP if needed
    save_model = model.module if hasattr(model, "module") else model
    if hasattr(save_model, "save_pretrained"):
        save_model.save_pretrained(save_dir)
    else:
        import torch
        state = {k: v for k, v in save_model.state_dict().items() if v.requires_grad}
        torch.save(state, os.path.join(save_dir, "adapter_model.pt"))

    logger.info(f"Checkpoint saved: {save_dir}")


def main():
    # Initialize distributed training if launched with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="Train VLA LoRA adapter")
    parser.add_argument("--config", type=str, required=True,
                        help="Adapter config YAML (move_adapter.yaml or grip_adapter.yaml)")
    parser.add_argument("--gpu-profile", type=str, default=None,
                        choices=["a6000", "a100"],
                        help="GPU profile for batch size and quantization settings")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Model config name (e.g., smolvla, pi05, groot) from configs/models/")
    parser.add_argument("--scheduler-type", type=str, default=None,
                        choices=["onecycle", "constant", "cosine"],
                        help="LR scheduler type")
    parser.add_argument("--chunk-stride", type=int, default=None,
                        help="Stride for ActionChunkDataset (>1 reduces overlapping chunks)")
    parser.add_argument("--task", type=str, default=None,
                        help="Train on specific task only (e.g., PnPCounterToCab)")
    parser.add_argument("--model-backend", type=str, default=None,
                        help="Override model backend (openvla, smolvla, pi05, groot_n1.5, hf_vlm)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name/path")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="Override LoRA rank")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override processed data directory")
    parser.add_argument("--hdf5-dir", type=str, default=None,
                        help="RoboCasa HDF5 data root (reads raw data directly, no preprocessing)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N optimizer steps (overrides epochs)")
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="Skip saving intermediate checkpoints (faster for sweeps)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Limit validation to N batches (faster eval)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override micro-batch size per GPU")
    parser.add_argument("--output-base-dir", type=str, default=None,
                        help="Override output base directory")
    args = parser.parse_args()

    # Build overrides from CLI
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
    if args.output_base_dir:
        overrides.setdefault("output", {})["base_dir"] = args.output_base_dir
    if args.skip_checkpoints:
        overrides.setdefault("training", {})["save_steps"] = 999999

    config = load_config(args.config, args.gpu_profile, args.model_config, overrides or None)

    logger.info("=" * 60)
    logger.info("VLA LoRA Training")
    logger.info(f"  Backend: {config['model']['backend']}")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Adapter: {config['adapter']['name']}")
    logger.info(f"  Task: {config['data'].get('task', 'all')}")
    logger.info(f"  GPU profile: {args.gpu_profile or 'default'}")
    logger.info(f"  LoRA rank: {config['lora']['rank']}")
    logger.info(f"  Quantization: {'4-bit' if config['quantization'].get('enabled') else 'none'}")
    logger.info("=" * 60)

    try:
        train(config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
