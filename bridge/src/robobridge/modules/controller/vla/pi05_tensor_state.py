"""
PI0.5 with tensor-based state conditioning.

Instead of encoding state as discretized text in the prompt (which loses precision
through quantization and KV cache dilution), this module injects state as a continuous
tensor in the suffix alongside noisy actions.

The state embedding is prepended to action embeddings in embed_suffix(), so the model
attends to fresh state information at every denoising step.

Architecture:
    [Original] embed_suffix(noisy_actions, timestep) -> [action_emb_1...action_emb_50]
    [Modified] embed_suffix(noisy_actions, timestep) -> [state_emb, action_emb_1...action_emb_50]

    suffix_out[:, -chunk_size:] extracts only action tokens, auto-skipping state_emb.

LoRA compatibility:
    _get_default_peft_targets() regex already matches "state_proj" (line 1287 of modeling_pi05.py).
    We override to put state_proj in modules_to_save for full training (not LoRA'd on random init).
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Config,
    PI05Policy,
    PI05Pytorch,
    ActionSelectKwargs,
    create_sinusoidal_pos_embedding,
)

try:
    from lerobot.policies.pi05.modeling_pi05 import (
        ACTION,
        PreTrainedConfig,
        PreTrainedPolicy,
    )
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
except ImportError:
    from lerobot.common.policies.pretrained import PreTrainedConfig, PreTrainedPolicy
    ACTION = "action"
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

logger = logging.getLogger(__name__)

# Max padded state dimension (matches training preprocessing)
MAX_STATE_DIM = 32

T = builtins.type["PI05TensorStatePolicy"]


class PI05TensorStatePytorch(PI05Pytorch):
    """PI05 core model with tensor-based state conditioning in the denoising suffix.

    Adds a state_proj linear layer that projects the padded 32D state vector
    into the expert embedding space. The resulting state token is prepended
    to action tokens in embed_suffix(), ensuring the model re-reads fresh
    state information at every denoising step (not cached once in KV prefix).
    """

    def __init__(self, config: PI05Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        # State projection: max_state_dim -> expert_width
        expert_width = self.action_in_proj.out_features
        self.state_proj = nn.Linear(MAX_STATE_DIM, expert_width)

        # Current state tensor (set before each forward/sample_actions call)
        self._current_state: Tensor | None = None

    def set_state(self, state: Tensor):
        """Store state tensor for use in embed_suffix.

        Args:
            state: (B, 32) float tensor, quantile-normalized to [-1, 1], zero-padded.
        """
        self._current_state = state

    def embed_suffix(self, noisy_actions, timestep):
        """Override to inject state via adaRMS conditioning pathway.

        State embedding is added to the time_emb BEFORE time_mlp, so it flows
        through the adaRMS normalization that modulates every transformer layer.
        This is much harder for LoRA to cancel than a constant additive bias on
        action embeddings (v3), because adaRMS controls scale/shift of layernorms.

        Architecture:
            adarms_cond = time_mlp(time_emb + state_emb)  -- state modulates normalization
            action_emb = action_in_proj(noisy_actions)      -- unchanged

        Output layout: [action_emb_1, ..., action_emb_N]  (same as original, no extra token)
        """
        # --- Time embedding ---
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # --- Inject state into time embedding BEFORE time_mlp ---
        if self._current_state is not None:
            state = self._current_state.to(
                dtype=time_emb.dtype, device=time_emb.device
            )
            if state.dim() == 3:
                state = state.squeeze(1)  # (B, 1, 32) -> (B, 32)

            def state_proj_func(s):
                return self.state_proj(s)

            state_emb = self._apply_checkpoint(state_proj_func, state)  # (B, expert_width)
            # Add to time_emb — flows through time_mlp → adaRMS conditioning
            time_emb = time_emb + state_emb

        # --- Action projection ---
        def action_proj_func(x):
            return self.action_in_proj(x)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # --- Time MLP (adaRMS conditioning) — now includes state info ---
        def time_mlp_func(t):
            x = self.time_mlp_in(t)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb  # carries both time + state information

        bsize = action_time_emb.shape[0]

        # --- Standard suffix layout (same as original PI0.5) ---
        embs = action_time_emb
        action_time_dim = action_time_emb.shape[1]
        pad_masks = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        att_masks = torch.tensor(
            [1] + ([0] * (self.config.chunk_size - 1)),
            dtype=embs.dtype, device=embs.device,
        )
        att_masks = att_masks[None, :].expand(bsize, self.config.chunk_size)

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self, images, img_masks, tokens, masks, actions,
        noise=None, time=None, state=None,
    ) -> Tensor:
        """Training forward pass with optional state tensor."""
        if state is not None:
            self.set_state(state)
        return super().forward(
            images, img_masks, tokens, masks, actions, noise=noise, time=time,
        )

    def sample_actions(
        self, images, img_masks, tokens, masks,
        noise=None, num_steps=None, state=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Inference with optional state tensor."""
        if state is not None:
            self.set_state(state)
        return super().sample_actions(
            images, img_masks, tokens, masks,
            noise=noise, num_steps=num_steps, **kwargs,
        )


class PI05TensorStatePolicy(PI05Policy):
    """PI05Policy with tensor-based state conditioning.

    Extracts observation.state from the batch and passes it as a continuous
    tensor to the model, bypassing the text-based discretization.
    """

    def __init__(self, config: PI05Config, **kwargs):
        # Skip PI05Policy.__init__ to swap model class, call grandparent instead
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05TensorStatePytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Load pretrained weights with strict=False to allow missing state_proj keys."""
        return super().from_pretrained(
            pretrained_name_or_path, strict=False, **kwargs,
        )

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Override to put state_proj in modules_to_save (fully trained, not LoRA'd).

        state_proj has random init (not in pretrained checkpoint), so it should be
        fully trained rather than LoRA-adapted on random weights.
        """
        targets = super()._get_default_peft_targets()
        # Remove state_proj from LoRA target_modules regex
        common_projections = "action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        targets["target_modules"] = (
            rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        )
        # Add state_proj to modules_to_save (fully trained)
        targets["modules_to_save"] = ["model.state_proj"]
        return targets

    def forward(
        self, batch: dict[str, Tensor], reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Training forward with state tensor from batch."""
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        # Extract state tensor (B, 32) if present
        state = batch.get("observation.state")

        losses = self.model.forward(
            images, img_masks, tokens, masks, actions, state=state,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Predict with state tensor from batch."""
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Extract state tensor (B, 32) if present
        state = batch.get("observation.state")

        actions = self.model.sample_actions(
            images, img_masks, tokens, masks, state=state, **kwargs,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions
