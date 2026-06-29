"""agents/psm/flow_bc/model.py — PSMFlowBCModel.

Mirror of agents/fb/flow_bc/model.py.FBFlowBCModel, but the base is PSMModel
(not FBModel). PSMModel builds NO TD3 actor when actor_kind != "td3"
(self.actor is None); this subclass attaches the noise-conditioned flow actor
(_actor) + the unconditional flow-matching vector field (_actor_vf), and
overrides actor()/act() exactly like FBFlowBCModel.

Substitutions vs FBFlowBCModel:
  - base class FBModel -> PSMModel
  - FB has _left_encoder / actor_encode_obs / L_dim; PSM does not. The actor /
    vector-field obs_space is the fw-encoder output_space (Identity for state).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.amp import autocast

from agents.psm.model import PSMModel
from agents.psm.psm_nets import weight_init
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class PSMFlowBCModel(PSMModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: Optional[NoiseConditionedActorArchiConfig] = None,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        **kwargs,
    ):
        # actor_kind != "td3" => PSMModel.__init__ sets the instance attribute
        # self.actor = None, which would shadow the actor() method defined below.
        super().__init__(obs_space=obs_space, action_dim=action_dim, **kwargs)
        # Drop the None instance attribute so the actor() method (class-level)
        # is the one resolved on lookup.
        del self.actor

        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig()

        # State: _fw_encoder is Identity, so its output_space is obs_space.
        actor_obs_space = self._fw_encoder.output_space
        self._actor = actor_cfg.build(actor_obs_space, self.z_dim, action_dim)
        self._actor_vf = actor_vf_cfg.build(actor_obs_space, action_dim)
        self._actor.apply(weight_init)
        self._actor_vf.apply(weight_init)

        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True, goal=None) -> torch.Tensor:
        del mean, goal  # not used
        return self.actor(obs, z)
