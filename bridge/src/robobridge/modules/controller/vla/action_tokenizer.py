"""
Action tokenization and normalization for VLA models.

Supports two modes:
- Discrete: 256-bin tokenization for OpenVLA-style models (cross-entropy loss)
- Continuous: normalization for flow-matching models like SmolVLA/pi0.5/GROOT N1.5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ActionTokenizer:
    """
    Converts between continuous robot actions and normalized/tokenized space.

    For discrete models (OpenVLA):
        continuous action -> 256-bin indices -> model tokens
    For continuous models (SmolVLA, pi0.5, GROOT N1.5):
        continuous action -> normalized [-1, 1] -> flow matching target
    """

    def __init__(
        self,
        action_stats: Dict[str, list],
        n_bins: int = 256,
        mode: str = "discrete",
    ):
        """
        Args:
            action_stats: {"min": [7 floats], "max": [7 floats],
                           "mean": [7 floats], "std": [7 floats]}
            n_bins: Number of bins for discrete tokenization.
            mode: "discrete" (OpenVLA) or "continuous" (flow matching).
        """
        self.n_bins = n_bins
        self.mode = mode

        self.mins = np.array(action_stats["min"], dtype=np.float32)
        self.maxs = np.array(action_stats["max"], dtype=np.float32)
        self.mean = np.array(action_stats.get("mean", np.zeros_like(self.mins)), dtype=np.float32)
        self.std = np.array(action_stats.get("std", np.ones_like(self.mins)), dtype=np.float32)
        self.std = np.clip(self.std, 1e-6, None)
        # Quantile stats for quantile normalization
        self.q01 = np.array(action_stats["q01"], dtype=np.float32) if "q01" in action_stats else self.mins
        self.q99 = np.array(action_stats["q99"], dtype=np.float32) if "q99" in action_stats else self.maxs

        self.action_dim = len(self.mins)

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using min-max scaling."""
        range_ = self.maxs - self.mins
        range_ = np.clip(range_, 1e-6, None)
        return 2.0 * (action - self.mins) / range_ - 1.0

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to original scale."""
        range_ = self.maxs - self.mins
        return (normalized + 1.0) / 2.0 * range_ + self.mins

    def quantile_normalize(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using q01/q99 quantile range."""
        range_ = self.q99 - self.q01
        range_ = np.clip(range_, 1e-6, None)
        return 2.0 * (action - self.q01) / range_ - 1.0

    def quantile_denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] using q01/q99 quantile range."""
        range_ = self.q99 - self.q01
        return (normalized + 1.0) / 2.0 * range_ + self.q01

    def standardize(self, action: np.ndarray) -> np.ndarray:
        """Standardize action using mean/std (z-score)."""
        return (action - self.mean) / self.std

    def destandardize(self, standardized: np.ndarray) -> np.ndarray:
        """Reverse z-score standardization."""
        return standardized * self.std + self.mean

    def encode(self, action: np.ndarray) -> np.ndarray:
        """Encode continuous action to bin indices (discrete mode)."""
        normalized = (action - self.mins) / np.clip(self.maxs - self.mins, 1e-6, None)
        bins = np.clip((normalized * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        return bins

    def decode(self, bin_indices: np.ndarray) -> np.ndarray:
        """Decode bin indices to continuous action (discrete mode)."""
        normalized = (bin_indices.astype(np.float32) + 0.5) / self.n_bins
        return normalized * (self.maxs - self.mins) + self.mins

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Process action based on mode.

        Discrete mode: returns bin indices (int).
        Continuous mode: returns normalized [-1, 1] (float).
        Zscore mode: returns z-score standardized (float).
        Quantile mode: returns quantile-normalized [-1, 1] (float).
        """
        if self.mode == "discrete":
            return self.encode(action)
        elif self.mode == "zscore":
            return self.standardize(action)
        elif self.mode == "quantile":
            return self.quantile_normalize(action)
        else:
            return self.normalize(action)

    def recover_action(self, processed: np.ndarray) -> np.ndarray:
        """Recover original-scale action from processed form."""
        if self.mode == "discrete":
            return self.decode(processed)
        elif self.mode == "zscore":
            return self.destandardize(processed)
        elif self.mode == "quantile":
            return self.quantile_denormalize(processed)
        else:
            return self.denormalize(processed)

    def to_dict(self) -> Dict:
        """Serialize stats for saving."""
        return {
            "min": self.mins.tolist(),
            "max": self.maxs.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_bins": self.n_bins,
            "mode": self.mode,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ActionTokenizer":
        """Load from metadata JSON file."""
        with open(path) as f:
            data = json.load(f)

        action_stats = data.get("action_stats", data)
        # Infer mode: if mean/std present and mode not specified, use zscore
        default_mode = "discrete"
        if "mode" not in action_stats and "mean" in action_stats and "std" in action_stats:
            default_mode = "zscore"
        return cls(
            action_stats=action_stats,
            n_bins=action_stats.get("n_bins", 256),
            mode=action_stats.get("mode", default_mode),
        )

    @classmethod
    def from_dataset(
        cls,
        actions: np.ndarray,
        n_bins: int = 256,
        mode: str = "discrete",
        percentile: float = 99.0,
    ) -> "ActionTokenizer":
        """Compute stats from training data.

        Args:
            actions: (N, action_dim) array of all training actions.
            n_bins: Number of discretization bins.
            mode: "discrete" or "continuous".
            percentile: Use percentile for min/max to handle outliers.
        """
        low = (100.0 - percentile) / 2.0
        high = 100.0 - low

        stats = {
            "min": np.percentile(actions, low, axis=0).tolist(),
            "max": np.percentile(actions, high, axis=0).tolist(),
            "mean": np.mean(actions, axis=0).tolist(),
            "std": np.std(actions, axis=0).tolist(),
        }

        return cls(action_stats=stats, n_bins=n_bins, mode=mode)
