"""agents/fb/model.py — Literal port of td_jepa/metamotivo/agents/fb/model.py.

Substitutions vs upstream:
    self.cfg.archi.X    -> self.X        (architecture hyperparams as instance attrs)
    self.cfg.X          -> self.X        (top-level model hyperparams)
    self._normalize     -> kept name (used by `actor`, `backward_map`, etc.)
    safetensors load/save -> dropped (use plain nn.Module state_dict)
    pydantic config classes -> replaced by ctor kwargs

The network construction (encoders, normalizer, augmentator, left_encoder,
backward_map, forward_map, actor, target nets) is line-for-line equivalent
to td_jepa's, in the same order, with the same gradient/eval-mode setup.
"""

from __future__ import annotations

import copy
import math
import typing as tp
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from normalizers import IdentityNormalizerConfig


class FBModel(torch.nn.Module):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        z_dim: int = 50,
        L_dim: int = 50,
        norm_z: bool = True,
        actor_encode_obs: bool = False,
        actor_std: float = 0.2,
        amp: bool = False,
        inference_batch_size: int = 500_000,
        forward_cfg: Optional[ForwardArchiConfig] = None,
        backward_cfg: Optional[BackwardArchiConfig] = None,
        left_encoder_cfg=None,
        actor_cfg=None,
        rgb_encoder_cfg=None,
        augmentator_cfg=None,
        obs_normalizer_cfg=None,
        fixed_b: str = "none",
        goal_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.fixed_b = str(fixed_b)
        self.goal_dim = int(goal_dim)
        # Store every config attr as a plain instance attribute (stand-in for cfg.archi.X / cfg.X).
        self.z_dim = z_dim
        self.L_dim = L_dim
        self.norm_z = norm_z
        self.actor_encode_obs = actor_encode_obs
        self.actor_std = actor_std
        self.amp = amp
        self.inference_batch_size = inference_batch_size
        self.device = device
        self.amp_dtype = torch.bfloat16

        # Defaults matching td_jepa's FBModelArchiConfig
        if forward_cfg is None:
            forward_cfg = ForwardArchiConfig()
        if backward_cfg is None:
            backward_cfg = BackwardArchiConfig()
        if actor_cfg is None:
            actor_cfg = SimpleActorArchiConfig()
        if left_encoder_cfg is None:
            left_encoder_cfg = IdentityNNConfig()
        if rgb_encoder_cfg is None:
            rgb_encoder_cfg = IdentityNNConfig()
        if augmentator_cfg is None:
            augmentator_cfg = IdentityNNConfig()
        if obs_normalizer_cfg is None:
            obs_normalizer_cfg = IdentityNormalizerConfig()

        # create networks (order and naming mirror td_jepa exactly)
        self._obs_normalizer = obs_normalizer_cfg.build(obs_space)
        self._bw_encoder = rgb_encoder_cfg.build(obs_space)
        self._augmentator = augmentator_cfg.build(obs_space)
        self._fw_encoder = rgb_encoder_cfg.build(obs_space)
        self._left_encoder = left_encoder_cfg.build(self._fw_encoder.output_space, L_dim)

        if self.fixed_b == "cube_xyz":
            from agents.fb.fixed_backward import FixedCubeBackward
            self._bw_encoder = torch.nn.Identity()           # raw obs -> backward
            self._backward_map = FixedCubeBackward()          # B(obs) = scaled cube xyz, z_dim=3
            self.z_dim = 3
            z_dim = 3                                          # forward map output dim follows
        else:
            self._backward_map = backward_cfg.build(self._bw_encoder.output_space, z_dim)
        self._forward_map = forward_cfg.build(self._left_encoder.output_space, z_dim, action_dim, goal_dim=self.goal_dim)
        self._actor = actor_cfg.build(
            self._left_encoder.output_space if self.actor_encode_obs else self._fw_encoder.output_space,
            z_dim,
            action_dim,
            goal_dim=self.goal_dim,
        )

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            return self._backward_map(self._bw_encoder(self._normalize(obs)))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            return self._forward_map(self._left_encoder(self._fw_encoder(self._normalize(obs))), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float, goal=None):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.actor_encode_obs else obs
            return self._actor(obs, z, std, **({} if goal is None else {"goal": goal}))

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True, goal=None) -> torch.Tensor:
        dist = self.actor(obs, z, self.actor_std, goal=goal)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    def reward_inference(
        self, next_obs: torch.Tensor, reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            batch_size = next_obs.shape[0]
            num_batches = int(np.ceil(batch_size / self.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = i * self.inference_batch_size, (i + 1) * self.inference_batch_size
                next_obs_slice = next_obs[start_idx:end_idx].to(self.device)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        # td_jepa's reward_inference returns shape [1, z_dim]; our callers expect [z_dim].
        z = self.project_z(z)
        return z.squeeze(0) if z.dim() == 2 and z.shape[0] == 1 else z

    # Backwards-compatible alias used by FBAgent.infer_z
    infer_z = reward_inference
