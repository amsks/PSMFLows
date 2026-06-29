"""agents/psm/flow_psm/model.py — FlowPSMModel.

Extends PSMFlowBCModel (which provides the behavior flow _actor_vf + the PSM
encoders/normalizer) with the PSMFlows representation nets:
  - phi_uu: phi(s, u0, u0') -> R^z_dim, a PhiMap on [s, u0, u0'] (final L2 Norm).
  - psi_goal: psi(s+) -> R^z_dim, a PhiMap on the goal/next-state encoding.
Target copies (target_phi_uu / target_psi_goal) are created for the future
SM-TD bootstrap so soft updates have something to track. The successor measure
is M = psi(s+) phi(s,u0,u0')^T.

The inherited proto nets (psm_psi, proto_sampler) and sf_psi are NOT used by
FlowPSM; they are built by the PSMModel base and left untouched.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from agents.psm.flow_bc.model import PSMFlowBCModel
from agents.psm.psm_nets import PhiMap, weight_init


class FlowPSMModel(PSMFlowBCModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        u0_dim: Optional[int] = None,
        actor_cfg=None,
        actor_vf_cfg=None,
        phi_uu_hidden_dim: int = 256,
        phi_uu_hidden_layers: int = 2,
        **kwargs,
    ):
        # FlowPSM always uses the behavior flow actor, never a TD3 actor. Default
        # actor_kind to "flowbc" so PSMModel sets self.actor=None (and PSMFlowBCModel
        # del's it), letting PSMFlowBCModel.actor() resolve. Without this, direct
        # construction would default to "td3" and build an Actor module that collides
        # with the inherited actor() method. The agent path passes actor_kind explicitly.
        kwargs.setdefault("actor_kind", "flowbc")
        super().__init__(
            obs_space=obs_space,
            action_dim=action_dim,
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            **kwargs,
        )
        # State: _fw_encoder is Identity, so obs_dim is the encoder output dim.
        obs_dim = self._fw_encoder.output_space.shape[0]
        self.u0_dim = u0_dim if u0_dim is not None else action_dim

        # phi(s, u0, u0'): input is [s, u0, u0'] -> z_dim, L2-normalised (norm=True).
        phi_in = obs_dim + 2 * self.u0_dim
        self.phi_uu = PhiMap(phi_in, self.z_dim, phi_uu_hidden_dim, phi_uu_hidden_layers, norm=True)
        # psi(s+): goal/next-state encoding -> z_dim.
        self.psi_goal = PhiMap(obs_dim, self.z_dim, phi_uu_hidden_dim, phi_uu_hidden_layers, norm=True)

        self.phi_uu.apply(weight_init)
        self.psi_goal.apply(weight_init)
        self.target_phi_uu = copy.deepcopy(self.phi_uu)
        self.target_psi_goal = copy.deepcopy(self.psi_goal)

        self.to(self.device)

    # NOTE: named flow_phi/flow_psi (not phi/psi) to avoid shadowing the base
    # PSMModel's `self.phi` submodule, which nn.Module resolves via attribute
    # lookup that a same-named method would otherwise intercept.
    def flow_phi(self, s: torch.Tensor, u0: torch.Tensor, u0p: torch.Tensor) -> torch.Tensor:
        return self.phi_uu(torch.cat([s, u0, u0p], dim=-1))

    def flow_psi(self, sp: torch.Tensor) -> torch.Tensor:
        return self.psi_goal(sp)
