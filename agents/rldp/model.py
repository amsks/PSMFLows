"""agents/rldp/model.py — Adapted from td_jepa/.../rldp/model.py.

Same substitution scheme as agents/fb/flow_bc/model.py:
    self.cfg.archi.X     -> self.X / kwarg
    cfg.archi.predictor  -> predictor_cfg kwarg
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
import torch

from agents.fb.model import FBModel
from nn_models import VForwardArchiConfig


class RLDPModel(FBModel):
    """FB model with an added self-predictive (SP) `_predictor` head.

    `_predictor` maps a current z-embedding + an action to a *predicted next*
    z-embedding. Used by RLDPAgent.update_fb to add a self-predictive loss
    that regularizes the encoder via dynamics prediction (the 'DP' in RLDP).
    """

    def __init__(
        self,
        obs_space,
        action_dim: int,
        predictor_cfg: Optional[VForwardArchiConfig] = None,
        **fb_kwargs,
    ):
        self._predictor_cfg = predictor_cfg if predictor_cfg is not None else VForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, **fb_kwargs)

        z_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.z_dim,), dtype=np.float32)
        # Predictor: VForwardMap(obs=z_embedding[z_dim], z=action[action_dim]) -> z_dim.
        # Mirrors td_jepa rldp/model.py: build(z_space, action_dim, output_dim=z_dim).
        self._predictor = self._predictor_cfg.build(z_space, action_dim, output_dim=self.z_dim)

        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)
