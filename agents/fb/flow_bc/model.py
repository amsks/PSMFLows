"""agents/fb/flow_bc/model.py — Literal port of td_jepa/.../fb/flow_bc/model.py.

Subclasses FBModel and adds `_actor_vf` (a NoiseConditionedActor-free
unconditional vector field over actions, used by flow matching). Substitutions
match agents/fb/model.py.
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
import torch
from torch.amp import autocast

from agents.fb.model import FBModel
from nn_models import SimpleVectorFieldArchiConfig


class FBFlowBCModel(FBModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        **kwargs,
    ):
        super().__init__(obs_space=obs_space, action_dim=action_dim, **kwargs)

        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig()

        actor_vf_obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.L_dim,), dtype=np.float32)
            if self.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = actor_vf_cfg.build(actor_vf_obs_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, goal: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.actor_encode_obs else obs
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises, **({} if goal is None else {"goal": goal}))
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True, goal=None) -> torch.Tensor:
        del mean  # not used
        return self.actor(obs, z, goal=goal)
