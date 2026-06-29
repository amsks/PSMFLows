"""agents/rldp/flow_bc/model.py — Adapted from td_jepa/.../rldp/flow_bc/model.py.

Mirrors the FBFlowBCModel pattern in agents/fb/flow_bc/model.py: subclasses
RLDPModel and adds `_actor_vf` (the action-flow vector field) for flow-matching.
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
import torch
from torch.amp import autocast

from agents.rldp.model import RLDPModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class RLDPFlowBCModel(RLDPModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        actor_cfg=None,
        **rldp_kwargs,
    ):
        # td_jepa's RLDPFlowBCModelArchiConfig defaults actor to NoiseConditionedActor.
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, actor_cfg=actor_cfg, **rldp_kwargs)

        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig()

        actor_vf_obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.L_dim,), dtype=np.float32)
            if self.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = actor_vf_cfg.build(actor_vf_obs_space, action_dim)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.actor_encode_obs else obs
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean
        return self.actor(obs, z)
