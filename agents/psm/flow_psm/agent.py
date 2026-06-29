"""agents/psm/flow_psm/agent.py — FlowPSMAgent.

Subclasses PSMFlowBCAgent (a PSMAgent) to inherit the behavior flow (_actor_vf,
compute_flow_actions, flow optimizers) and the encoder/normalizer plumbing.

This SCAFFOLD trains only the behavior-flow vector field (flow-matching BC of
dataset actions). The proto/SF/Q machinery inherited from the base is NOT
invoked. The PSMFlows-specific updates are stubbed seams:
  - _update_flow_psm:  flow_phi(s,u0,u0')^T flow_psi(s+) contrastive SM-TD + ortho.
  - _update_u0_critic: u0-space constrained Q-learning.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.psm.flow_bc.agent import PSMFlowBCAgent
from agents.psm.flow_psm.model import FlowPSMModel
from nn_models import eval_mode


class FlowPSMAgent(PSMFlowBCAgent):
    def __init__(self, *args, u0_dim=None, **kwargs):
        # Stash u0_dim BEFORE super().__init__ so _build_model (called inside it)
        # can pass it to FlowPSMModel.
        self._u0_dim = u0_dim
        super().__init__(*args, **kwargs)

    def _build_model(self, obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
                     phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
                     obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
                     num_parallel, device, amp):
        """Build FlowPSMModel (adds flow_phi/flow_psi on top of the behavior flow)."""
        return FlowPSMModel(
            obs_space, action_dim, u0_dim=self._u0_dim,
            actor_cfg=actor_cfg, actor_vf_cfg=self._actor_vf_cfg,
            z_dim=z_dim, max_log_seed=max_log_seed, batch_size=batch_size,
            norm_z=norm_z, phi_input=phi_input, phi_cfg=phi_cfg, sf_cfg=sf_cfg,
            actor_kind=actor_kind, obs_normalizer_cfg=obs_normalizer_cfg,
            rgb_encoder_cfg=rgb_encoder_cfg, augmentator_cfg=augmentator_cfg,
            num_parallel=num_parallel, device=device, amp=amp,
        )

    def setup_training(self) -> None:
        # Base builds phi/sf/psm optimizers + the flow actor/vf optimizers.
        super().setup_training()
        m = self.model
        # Optimizers + target paramlists for the FlowPSM nets (unused by the
        # scaffold's stubbed updates; created so the seam is ready).
        self.optim_phi_uu = torch.optim.Adam(m.phi_uu.parameters(), lr=self.lr_phi, weight_decay=self.weight_decay)
        self.optim_psi_goal = torch.optim.Adam(m.psi_goal.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        self._phi_uu_paramlist = tuple(m.phi_uu.parameters())
        self._target_phi_uu_paramlist = tuple(m.target_phi_uu.parameters())
        self._psi_goal_paramlist = tuple(m.psi_goal.parameters())
        self._target_psi_goal_paramlist = tuple(m.target_psi_goal.parameters())

    # ── behavior-flow BC step (flow matching of dataset actions; _actor_vf only) ── #
    def _update_behavior_flow(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0
            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        bc_flow_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model._actor_vf.parameters(), self.clip_grad_norm)
        self.actor_vf_optimizer.step()
        return {"bc_flow_loss": bc_flow_loss.detach()}

    # ── stubbed seams (next implementation plans) ── #
    def _update_flow_psm(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """STUB: flow_phi(s,u0,u0')^T flow_psi(s+) contrastive SM-TD + ortho over flow-noise policies."""
        return {}

    def _update_u0_critic(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """STUB: u0-space constrained Q-learning critic."""
        return {}

    # ── update orchestration (behavior flow only; stubs are no-ops) ── #
    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs_n = self.model._obs_normalizer(obs)
            next_obs_n = self.model._obs_normalizer(next_obs)
        obs_n, next_obs_n = self.aug(obs_n, next_obs_n)

        metrics: Dict = {}
        obs_enc = self._enc_obs(obs_n).detach()
        metrics.update(self._update_behavior_flow(obs_enc, action))
        metrics.update(self._update_flow_psm())
        metrics.update(self._update_u0_critic())

        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}
