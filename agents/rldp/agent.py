"""agents/rldp/agent.py — Adapted from td_jepa/.../rldp/agent.py.

Same substitution scheme as agents/fb/agent.py:
    self.cfg.train.X     -> self.X
    self._model.X        -> self.model.X
    cudagraphs/compile   -> dropped (eager mode)

RLDPAgent inherits from FBAgent. The novel pieces vs FB:
  - _predictor in self.model (added by RLDPModel)
  - backward_optimizer also updates _predictor params
  - update_fb adds a self-predictive (SP) loss:
      starting from B(obs), roll out _predictor(curr, action_t) for h steps,
      and L2-match each step to _target_backward_map(future_obs_t).
  - update() consumes windowed [B, horizon, ...] batches from
    DictBuffer.sample(batch_size, horizon=h) and slices inline.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from agents.fb.agent import FBAgent
from agents.rldp.model import RLDPModel
from nn_models import VForwardArchiConfig, _soft_update_params, eval_mode


class RLDPAgent(FBAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        horizon: int = 5,
        predictor_cfg: Optional[VForwardArchiConfig] = None,
        **fb_kwargs,
    ):
        self.horizon = int(horizon)
        self._predictor_cfg = predictor_cfg if predictor_cfg is not None else VForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, **fb_kwargs)

    def _make_model(self, **kwargs):
        """Override FBAgent._make_model to build RLDPModel."""
        return RLDPModel(predictor_cfg=self._predictor_cfg, **kwargs)

    def setup_training(self) -> None:
        super().setup_training()
        # Extend backward_optimizer to include _predictor params (matches td_jepa rldp/agent.py).
        self.backward_optimizer = torch.optim.Adam(
            list(self.model._backward_map.parameters())
            + list(self.model._bw_encoder.parameters())
            + list(self.model._predictor.parameters()),
            lr=self.lr_b,
            weight_decay=self.weight_decay,
        )

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        """Consume a windowed batch (shape [B, horizon, ...] per key) and run one
        RLDP update. Inline-slices windowed batch into td_jepa-shaped intermediates.
        """
        obs        = batch["observation"][:, 0].to(self.device)
        action     = batch["action"][:, 0].to(self.device)
        next_obs   = batch["next"]["observation"][:, 0].to(self.device)
        terminated = batch["next"]["terminated"][:, 0].to(self.device)

        if self.horizon > 1:
            future_obs = batch["observation"][:, 1:].to(self.device)
            future_act = batch["action"][:, 1:].to(self.device)
            future_obs_flat = future_obs.reshape(-1, *future_obs.shape[2:])
        else:
            future_obs_flat = None
            future_act = None

        discount = self.discount * (~terminated.bool())

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs = self.model._obs_normalizer(obs)
            next_obs = self.model._obs_normalizer(next_obs)
            if future_obs_flat is not None:
                future_obs_flat = self.model._obs_normalizer(future_obs_flat)

        obs       = self.model._augmentator(obs)
        next_obs  = self.model._augmentator(next_obs)
        if future_obs_flat is not None:
            future_obs_flat = self.model._augmentator(future_obs_flat)

        obs_fw  = self.model._fw_encoder(obs)
        obs_bw  = self.model._bw_encoder(obs)
        with torch.no_grad():
            goal = self.model._bw_encoder(next_obs)
            next_obs_fw = self.model._fw_encoder(next_obs)
            future_obs_bw = self.model._bw_encoder(future_obs_flat) if future_obs_flat is not None else None

        z, _ = self.sample_mixed_z(train_goal=goal)
        z = z.clone()

        clip_grad_norm = self.clip_grad_norm if self.clip_grad_norm > 0 else None
        q_loss_coef = self.q_loss_coef if self.q_loss_coef > 0 else None

        metrics = self.update_fb(
            obs=obs_fw,
            obs_bw=obs_bw,
            action=action,
            future_act=future_act,
            discount=discount,
            next_obs=next_obs_fw,
            future_obs=future_obs_bw,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(self.update_actor(
            obs=obs_fw.detach(),
            action=action,
            z=z,
            clip_grad_norm=clip_grad_norm,
        ))

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.f_target_tau)

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        obs_bw: torch.Tensor,
        action: torch.Tensor,
        future_act: Optional[torch.Tensor],
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: Optional[torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: Optional[float],
        clip_grad_norm: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                next_action = self.sample_action_from_norm_obs(actor_in, z)
                target_Fs = self.model._target_forward_map(next_left_enc, z, next_action)
                target_B = self.model._target_backward_map(goal)
                target_Ms = torch.matmul(target_Fs, target_B.T)
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.fb_pessimism_penalty)

            left_enc = self.model._left_encoder(obs)
            Fs = self.model._forward_map(left_enc, z, action)
            B = self.model._backward_map(goal).detach()
            Ms = torch.matmul(Fs, B.T)

            diff = Ms - discount * target_M
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # ── Self-predictive (SP) loss — the RLDP-specific addition ──
            with torch.no_grad():
                targets = [self.model._target_backward_map(next_obs)]
                actions = [action]
                if future_obs is not None and future_act is not None:
                    future_phi = self.model._target_backward_map(future_obs)
                    future_phi = future_phi.reshape(*future_act.shape[:2], -1)  # [B, h-1, z_dim]
                    targets += [future_phi[:, i] for i in range(future_phi.shape[1])]
                    actions += [future_act[:, i] for i in range(future_act.shape[1])]
            B_pred = self.model._backward_map(obs_bw)
            curr = B_pred
            sp_loss = 0
            for act_t, target_t in zip(actions, targets):
                curr = self.model._predictor(curr, act_t)  # [num_parallel, B, z_dim]
                sp_loss = sp_loss + (curr - target_t.unsqueeze(0)).pow(2).sum(-1).mean()
            sp_loss = sp_loss / max(len(actions), 1)
            fb_loss = fb_loss + sp_loss

            # ── Orthonormality loss on B ──
            Cov = torch.matmul(B_pred, B_pred.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss = fb_loss + self.ortho_coef * orth_loss

            # ── Optional Q-MSE auxiliary loss ──
            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.fb_pessimism_penalty)
                    with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=False):
                        cov = torch.matmul(B_pred.T, B_pred) / B_pred.shape[0]
                    B_inv_conv = torch.linalg.solve(cov, B_pred, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss = fb_loss + q_loss_coef * q_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B_pred.mean(),
                "B_norm": torch.norm(B_pred, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "sp_loss": sp_loss,
                "q_loss": q_loss,
            }
        return output_metrics
