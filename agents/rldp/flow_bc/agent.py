"""agents/rldp/flow_bc/agent.py — Adapted from td_jepa/.../rldp/flow_bc/agent.py.

Mirrors the FBFlowBCAgent pattern in agents/fb/flow_bc/agent.py: subclasses
RLDPAgent and adds (a) actor_vf_optimizer for the action-flow vector field,
(b) flow-matching loss in update_actor, (c) compute_flow_actions Euler
integration helper used by the BC loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.amp import autocast

from agents.rldp.agent import RLDPAgent
from agents.rldp.flow_bc.model import RLDPFlowBCModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class RLDPFlowBCAgent(RLDPAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: Optional[NoiseConditionedActorArchiConfig] = None,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        **rldp_kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        self.flow_steps = int(flow_steps)
        self.lr_actor_vf = float(lr_actor_vf)
        self._actor_vf_cfg = actor_vf_cfg
        super().__init__(
            obs_space=obs_space, action_dim=action_dim,
            actor_cfg=actor_cfg, **rldp_kwargs,
        )

    def _make_model(self, **kwargs):
        return RLDPFlowBCModel(
            actor_vf_cfg=self._actor_vf_cfg,
            predictor_cfg=self._predictor_cfg,
            **kwargs,
        )

    def setup_training(self) -> None:
        super().setup_training()
        self.actor_vf_optimizer = torch.optim.Adam(
            self.model._actor_vf.parameters(),
            lr=self.lr_actor_vf,
            weight_decay=self.weight_decay,
        )

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        return self.model._actor(obs, z, noises)

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

            with torch.no_grad():
                left_enc = self.model._left_encoder(obs)
            actor_in = left_enc if self.actor_encode_obs else obs
            noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            actor_actions = self.model._actor(actor_in, z, noises)
            Fs = self.model._forward_map(left_enc, z, actor_actions)
            Qs = (Fs * z).sum(-1)
            _, _, Q = self.get_targets_uncertainty(Qs, self.actor_pessimism_penalty)
            actor_loss = -Q.mean()

            bc_loss = torch.tensor([0.0], device=action.device)
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                with torch.no_grad():
                    target_flow_actions = self.compute_flow_actions(obs, noises)
                bc_error = torch.pow(actor_actions - target_flow_actions, 2).mean()
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

            actor_loss = actor_loss + bc_flow_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()
        self.actor_vf_optimizer.step()

        return {
            "actor_loss": actor_loss.mean().detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "bc_error": bc_error.detach(),
            "q": Q.mean().detach(),
        }

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.flow_steps
            vels = self.model._actor_vf(obs, actions, t)
            actions = actions + vels / self.flow_steps
        return torch.clamp(actions, min=-1, max=1)

    def act(self, obs: torch.Tensor, z: torch.Tensor, *, eval_mode: bool = True) -> torch.Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)
