"""agents/psm/flow_bc/agent.py — PSMFlowBCAgent.

PSM representation (proto + SF psi, phi basis) with the TD3 actor REPLACED by
our FlowBC actor: a noise-conditioned flow vector field (behavior cloning of
dataset actions) plus a Q-driven actor head, where Q = sf_psi(obs, z, a) · z.

This is the FBFlowBCAgent pattern (agents/fb/flow_bc/agent.py) ported onto the
PSM base. The deltas vs FBFlowBCAgent:
  - base class FBAgent -> PSMAgent
  - the Q term uses PSM's sf_psi (PsiMap, signature (obs, z, action)) in place
    of FB's _forward_map
  - the real z is used (PSM has no one-step zeroing; there is NO _fb_z)
  - PSM's get_targets_uncertainty returns (mean, unc) (2-tuple), so the
    pessimistic Q is Q_mean - actor_pessimism_penalty * Q_unc (matches PSMAgent
    ._update_actor exactly)
  - the SF target next-action comes from the flow actor (sample_action_from_norm_obs)

The base PSMAgent TD3 update math (_update_psm / _update_sf / _update_td3_actor)
is NOT modified; PSMFlowBCAgent only overrides the actor path. actor_kind is
forced to "flowbc" so PSMModel builds no TD3 actor; PSMFlowBCModel attaches the
flow actor (_actor) + vector field (_actor_vf).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.psm.agent import PSMAgent
from agents.psm.flow_bc.model import PSMFlowBCModel
from nn_models import (
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
    _soft_update_params,
    eval_mode,
)


class PSMFlowBCAgent(PSMAgent):
    def __init__(
        self,
        obs_space,
        action_dim,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        bc_coeff: float = 0.0,
        batch_size=1024,
        z_dim=128,
        max_log_seed=16,
        phi_cfg=None,
        sf_cfg=None,
        norm_z=True,
        phi_input="s",
        obs_normalizer_cfg=None,
        rgb_encoder_cfg=None,
        augmentator_cfg=None,
        num_parallel=2,
        discount=0.98,
        lr_sf=1e-4,
        lr_phi=1e-4,
        lr_actor=1e-4,
        weight_decay=0.0,
        clip_grad_norm=0.0,
        target_tau=0.01,
        ortho_coef=1.0,
        mix_ratio=0.5,
        pessimism_penalty=0.0,
        actor_pessimism_penalty=0.5,
        actor_std=0.2,
        stddev_clip=0.3,
        amp=False,
        device="cpu",
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()

        # Stash FlowBC-specific config BEFORE super().__init__ so _build_model
        # (called inside super().__init__) sees actor_vf_cfg, and setup_training
        # (also called inside super().__init__) sees flow_steps/lr_actor_vf/bc_coeff.
        self.flow_steps = flow_steps
        self.lr_actor_vf = lr_actor_vf
        self.bc_coeff = bc_coeff
        self._actor_vf_cfg = actor_vf_cfg

        # actor_kind="flowbc" => PSMModel builds NO TD3 actor; _build_model below
        # returns a PSMFlowBCModel that attaches _actor (NoiseConditionedActor) +
        # _actor_vf (SimpleVectorField). Everything else (attr assignment, std/clip
        # sync, setup_training, model.to) is inherited from PSMAgent.__init__.
        super().__init__(
            obs_space, action_dim, batch_size=batch_size, z_dim=z_dim,
            max_log_seed=max_log_seed, phi_cfg=phi_cfg, sf_cfg=sf_cfg,
            actor_cfg=actor_cfg, norm_z=norm_z, phi_input=phi_input,
            obs_normalizer_cfg=obs_normalizer_cfg, rgb_encoder_cfg=rgb_encoder_cfg,
            augmentator_cfg=augmentator_cfg, num_parallel=num_parallel, discount=discount,
            lr_sf=lr_sf, lr_phi=lr_phi, lr_actor=lr_actor, weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm, target_tau=target_tau, ortho_coef=ortho_coef,
            mix_ratio=mix_ratio, pessimism_penalty=pessimism_penalty,
            actor_pessimism_penalty=actor_pessimism_penalty, actor_std=actor_std,
            stddev_clip=stddev_clip, amp=amp, device=device, actor_kind="flowbc",
        )

    def _build_model(self, obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
                     phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
                     obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
                     num_parallel, device, amp):
        """Build PSMFlowBCModel (instead of PSMModel) so the flow actor + vector
        field are attached. actor_kind is forced to "flowbc" by PSMFlowBCAgent.__init__."""
        return PSMFlowBCModel(
            obs_space, action_dim, actor_cfg=actor_cfg, actor_vf_cfg=self._actor_vf_cfg,
            z_dim=z_dim, max_log_seed=max_log_seed, batch_size=batch_size,
            norm_z=norm_z, phi_input=phi_input, phi_cfg=phi_cfg, sf_cfg=sf_cfg,
            actor_kind=actor_kind, obs_normalizer_cfg=obs_normalizer_cfg, rgb_encoder_cfg=rgb_encoder_cfg,
            augmentator_cfg=augmentator_cfg, num_parallel=num_parallel, device=device, amp=amp,
        )

    def setup_training(self) -> None:
        # Run base setup (train/requires_grad, phi/sf/psm optimizers, target
        # paramlists, off_diag). Base PSMAgent.setup_training builds optim_actor
        # only when m.actor is not None; for flowbc m.actor is None, so the flow
        # actor + vector-field optimizers are added here.
        super().setup_training()
        m = self.model
        # actor = the noise-conditioned flow actor head (NOT a TD3 actor)
        self.optim_actor = torch.optim.Adam(m._actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)
        self.actor_vf_optimizer = torch.optim.Adam(
            m._actor_vf.parameters(), lr=self.lr_actor_vf, weight_decay=self.weight_decay
        )

    # ── flow actor sampler (replaces the TD3 actor's stochastic sample) ── #
    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        action = self.model._actor(obs, z, noises)
        return action

    # ── FlowBC actor update: flow-matching BC + Q-driven head (Q from sf_psi·z) ── #
    def update_actor(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            # flow matching l2 loss (goal/z-UNconditioned — pure BC of dataset actions)
            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

            # Q loss. actor_in = obs (already fw-encoded; Identity for state).
            actor_in = obs
            noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            actor_actions = self.model._actor(actor_in, z, noises)
            # Q via PSM's sf_psi (PsiMap signature (obs, z, action)); dots with the real z.
            Fs = self.model.sf_psi(actor_in, z, actor_actions)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            Q_mean, Q_unc = self.model.get_targets_uncertainty(Qs)  # batch
            Q = Q_mean - self.actor_pessimism_penalty * Q_unc  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_loss = torch.tensor([0.0], device=action.device)
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                with torch.no_grad():
                    target_flow_actions = self.compute_flow_actions(actor_in, noises)
                bc_error = torch.pow(actor_actions - target_flow_actions, 2).mean()
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

            actor_loss = actor_loss + bc_flow_loss

        self.optim_actor.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), self.clip_grad_norm)
        self.optim_actor.step()
        self.actor_vf_optimizer.step()

        return {
            "actor_loss": actor_loss.mean().detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "bc_error": bc_error.detach() if isinstance(bc_error, torch.Tensor) else torch.tensor(bc_error),
            "q": Q.mean().detach(),
            "z_norm": z.norm(dim=-1).mean().detach(),
            "actor_action_abs": actor_actions.abs().mean().detach(),  # is the flow actor saturating?
        }

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.flow_steps
            vels = self.model._actor_vf(obs, actions, t)
            actions = actions + vels / self.flow_steps
        actions = torch.clamp(actions, min=-1, max=1)
        return actions

    # ── update orchestration (mirror PSMAgent.update; actor path swapped) ── #
    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        obs = batch["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next"]["observation"].to(self.device).float()
        terminated = batch["next"]["terminated"].to(self.device)
        idx = batch.get("index")
        next_obs_hash = (idx.to(self.device) if idx is not None
                         else torch.arange(self.batch_size, device=self.device))

        if terminated.dtype == torch.bool:
            discount = self.discount * ~terminated
        else:
            discount = self.discount * (1.0 - terminated)
        discount = discount.reshape(-1, 1)

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs_n = self.model._obs_normalizer(obs)
            next_obs_n = self.model._obs_normalizer(next_obs)

        # augment (Identity for state; random-shift for pixel), then re-encode per branch
        # so each of the THREE backward calls (psm / sf / actor) owns its grad graph (the
        # DrQ encoders carry a graph for pixels; a single shared encode would be backward-ed
        # multiple times). Encoder ownership: _fw_encoder -> optim_sf_psi, _bw_encoder ->
        # optim_phi. psm/actor branches use detached obs; the SF branch owns fw-encoder grad.
        # For STATE the encoders are Identity (no params, passthrough) so this is byte-identical.
        obs_n, next_obs_n = self.aug(obs_n, next_obs_n)

        metrics: Dict = {}

        # 1) proto-successor branch (unchanged from PSMAgent): obs detached, goal with bw-grad.
        z_psm = self.model.sample_z_psm(self.batch_size, device=self.device)
        obs_enc_psm = self._enc_obs(obs_n).detach()
        next_obs_enc_psm = self._enc_obs(next_obs_n).detach()
        goal_psm = self._enc_goal(next_obs_n)
        metrics.update(self._update_psm(obs_enc_psm, action, discount, next_obs_enc_psm,
                                        next_obs_hash, goal_psm, z_psm))
        with torch.no_grad():
            _soft_update_params(self._psm_psi_paramlist, self._target_psm_psi_paramlist, self.target_tau)
            _soft_update_params(self._phi_paramlist, self._target_phi_paramlist, self.target_tau)

        # 2) SF + FlowBC-actor branch: mixed Gaussian z. SF owns the fw-encoder grad.
        goal_sf = self._enc_goal(next_obs_n).detach()
        z = self.sample_mixed_z(goal_sf)
        obs_enc_sf = self._enc_obs(obs_n)
        next_obs_enc_sf = self._enc_obs(next_obs_n).detach()
        # SF target next-action comes from the FLOW actor (base _update_sf would
        # otherwise call self.model.actor(...), which is None for flowbc).
        with torch.no_grad():
            sf_next_action = self.sample_action_from_norm_obs(next_obs_enc_sf, z)
        metrics.update(self._update_sf(obs_enc_sf, action, discount, next_obs_enc_sf, goal_sf, z,
                                       next_action=sf_next_action))
        metrics.update(self.update_actor(self._enc_obs(obs_n).detach(), action, z))
        with torch.no_grad():
            _soft_update_params(self._sf_psi_paramlist, self._target_sf_psi_paramlist, self.target_tau)

        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def act(self, obs, z, *, eval_mode: bool = True):
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    def state_dict(self) -> dict:
        state = {
            "model": self.model.state_dict(),
            "optim_phi": self.optim_phi.state_dict(),
            "optim_sf_psi": self.optim_sf_psi.state_dict(),
            "optim_psm_psi": self.optim_psm_psi.state_dict(),
            "optim_actor": self.optim_actor.state_dict(),
            "actor_vf_optimizer": self.actor_vf_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optim_phi.load_state_dict(state["optim_phi"])
        self.optim_sf_psi.load_state_dict(state["optim_sf_psi"])
        self.optim_psm_psi.load_state_dict(state["optim_psm_psi"])
        self.optim_actor.load_state_dict(state["optim_actor"])
        self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
