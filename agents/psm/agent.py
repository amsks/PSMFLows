"""agents/psm/agent.py — PSMAgent: dual-psi (proto + SF) + TD3 actor update.

The update orchestration mirrors agents/fb/agent.py.FBAgent (device/device_type/
amp_dtype, obs-normalizer update + eval-mode normalize, autocast wrapping,
_soft_update_params with precomputed param tuples, the final {k: v.item()}
metric conversion, state_dict/load_state_dict shape).

The loss math in _update_psm / _update_sf / _update_td3_actor is transcribed
VERBATIM from the reference (/home/adityamohan/git/baselines/PSM/agent/psm.py),
substituting our attribute names (self.model.X, self.X). A later test asserts
bit-exact equivalence, so the diag/offdiag/ortho/Q signs, the `* Ms.shape[0]`
diagonal scaling, the `phi.detach()` in SF, the SF target using `self.phi` (not
target_phi), and the `0*ortho_coef` SF-ortho term are preserved exactly.

NOTE: _update_psm's phi-orthonormality term is now `self.ortho_coef * orth_loss`
(default 1.0 == the reference's hardcoded `1 *`, so the bit-exact equivalence
test still passes). ortho_coef is an intentional ablation knob BEYOND the paper
(which fixes the orthonormality weight at 1).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.amp import autocast

from agents.base import BaseAgent
from agents.psm.model import PSMModel
from nn_models import _soft_update_params, eval_mode


class PSMAgent(BaseAgent):
    def __init__(self, obs_space, action_dim, batch_size=1024, z_dim=128, max_log_seed=16,
                 phi_cfg=None, sf_cfg=None, actor_cfg=None, norm_z=True, phi_input="s",
                 obs_normalizer_cfg=None, rgb_encoder_cfg=None, augmentator_cfg=None,
                 num_parallel=2, discount=0.98, lr_sf=1e-4, lr_phi=1e-4, lr_actor=1e-4,
                 weight_decay=0.0, clip_grad_norm=0.0, target_tau=0.01, ortho_coef=1.0,
                 mix_ratio=0.5, pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
                 actor_std=0.2, stddev_clip=0.3, amp=False, device="cpu", actor_kind="td3"):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.max_log_seed = max_log_seed
        self.discount = discount
        self.lr_sf = lr_sf
        self.lr_phi = lr_phi
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.target_tau = target_tau
        self.ortho_coef = ortho_coef
        self.mix_ratio = mix_ratio
        self.pessimism_penalty = pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.num_parallel = num_parallel
        self.amp = amp
        self.device = device
        self.device_type = str(device).split(":")[0]  # "cuda:0" -> "cuda" for autocast

        # Build the model via a hook so subclasses (PSMFlowBCAgent) can return
        # their own variant (PSMFlowBCModel) without re-implementing __init__.
        self.model = self._build_model(
            obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
            phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
            obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
            num_parallel, device, amp)
        # keep the model's fixed actor std/clip in sync with the agent kwargs
        self.model.actor_std = actor_std
        self.model.stddev_clip = stddev_clip
        self.setup_training()
        self.model.to(self.device)

    def _build_model(self, obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
                     phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
                     obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
                     num_parallel, device, amp):
        """Construct the model. Subclasses override to return their variant."""
        return PSMModel(obs_space, action_dim, z_dim, max_log_seed, batch_size, norm_z,
                        phi_input, phi_cfg, sf_cfg, actor_cfg, actor_kind,
                        obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg,
                        num_parallel, device, amp)

    @property
    def amp_dtype(self):
        return self.model.amp_dtype

    def setup_training(self) -> None:
        m = self.model
        m.train(True)
        m.requires_grad_(True)
        # Encoder ownership (pixels only; empty param lists for state Identity encoders):
        #   _bw_encoder (goal) trained with phi; _fw_encoder (obs) trained with sf_psi.
        self.optim_phi = torch.optim.Adam(
            list(m.phi.parameters()) + list(m._bw_encoder.parameters()),
            lr=self.lr_phi, weight_decay=self.weight_decay)
        self.optim_sf_psi = torch.optim.Adam(
            list(m.sf_psi.parameters()) + list(m._fw_encoder.parameters()),
            lr=self.lr_sf, weight_decay=self.weight_decay)
        self.optim_psm_psi = torch.optim.Adam(m.psm_psi.parameters(), lr=self.lr_sf, weight_decay=self.weight_decay)
        # TD3: m.actor is an nn.Module (or None). For the flowbc variant the
        # PSMFlowBCModel exposes actor() as a method (not a Module); the subclass
        # builds its own actor/vector-field optimizers in its setup_training, so
        # only create optim_actor here when there is a real TD3 actor module.
        if isinstance(m.actor, torch.nn.Module):
            self.optim_actor = torch.optim.Adam(m.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)

        # precompute parameter tuples for the in-place _soft_update_params
        # (torch._foreach_* needs concrete lists, not generators) — mirrors FBAgent.
        self._phi_paramlist = tuple(m.phi.parameters())
        self._target_phi_paramlist = tuple(m.target_phi.parameters())
        self._sf_psi_paramlist = tuple(m.sf_psi.parameters())
        self._target_sf_psi_paramlist = tuple(m.target_sf_psi.parameters())
        self._psm_psi_paramlist = tuple(m.psm_psi.parameters())
        self._target_psm_psi_paramlist = tuple(m.target_psm_psi.parameters())

        # precompute some useful variables (mirror FBAgent / reference psm.py:232-233)
        self.off_diag = 1 - torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    # ── aug/encode helpers (Identity for state; DrQ CNN for pixel) ── #
    def aug(self, obs: Tensor, next_obs: Tensor):
        """Augments observations when training from pixels; no-op otherwise (mirror FBAgent.aug)."""
        return self.model._augmentator(obs), self.model._augmentator(next_obs)

    def _enc_obs(self, obs: Tensor) -> Tensor:
        return self.model._fw_encoder(obs)

    def _enc_goal(self, next_obs: Tensor) -> Tensor:
        return self.model._bw_encoder(next_obs)

    # ── z sampling for the SF/actor branch (reference psm.py:744-756) ── #
    def sample_mixed_z(self, goal: Tensor) -> Tensor:
        z = self.model.sample_z(self.batch_size, device=self.device)
        perm = torch.randperm(self.batch_size, device=self.device)
        phi_in = goal[perm]
        if self.mix_ratio > 0:
            mix_idxs = torch.where(torch.rand(self.batch_size, device=self.device) < self.mix_ratio)[0]
            with torch.no_grad(), eval_mode(self.model):
                mix_z = self.model.phi(phi_in[mix_idxs]).detach()
            mix_z = self.model.project_z(mix_z)
            z[mix_idxs] = mix_z
        return z

    # ── proto successor net update (reference _update_psm, psm.py:438-477) ── #
    def _update_psm(self, obs, action, discount, next_obs, next_obs_hash, goal, z, next_action=None):
        # `next_action` is a test-only injection hook (default None = current behavior:
        # sample from the proto behavior sampler). Used by the bit-exact reference-
        # equivalence test to bypass stochastic sampling; never set in training.
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            # compute target successor measure
            with torch.no_grad():
                if next_action is None:
                    next_action = self.model.proto_sampler(next_obs_hash, z)
                target_psm_psis = self.model.target_psm_psi(next_obs, z, next_action)  # P x B x z_dim
                target_phi = self.model.target_phi(goal)  # B x z_dim
                target_Ms = torch.matmul(target_psm_psis, target_phi.T)  # P x B x B
                target_M_mean, target_M_unc = self.model.get_targets_uncertainty(target_Ms)  # B x B
                target_M = target_M_mean - self.pessimism_penalty * target_M_unc  # B x B

            # compute PSM loss
            psis = self.model.psm_psi(obs, z, action)  # P x B x z_dim
            phi = self.model.phi(goal)  # B x z_dim
            Ms = torch.matmul(psis, phi.T)  # P x B x B

            diff = Ms - discount * target_M  # P x B x B
            psm_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            psm_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            psm_loss = psm_offdiag + psm_diag

            # orthonormality loss for phi embedding
            Cov = torch.matmul(phi, phi.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            psm_loss = psm_loss + self.ortho_coef * orth_loss

        # optimize PSM (psm_psi + phi)
        self.optim_psm_psi.zero_grad(set_to_none=True)
        self.optim_phi.zero_grad(set_to_none=True)
        psm_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.psm_psi.parameters(), self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.phi.parameters(), self.clip_grad_norm)
        self.optim_psm_psi.step()
        self.optim_phi.step()

        # ── diagnostics (pure logging; no grad/RNG effect on the update above) ──
        with torch.no_grad():
            diag = {
                "phi_norm": phi.norm(dim=-1).mean(),                       # basis row norm (orth target ~1)
                "psm_psi_norm": psis.norm(dim=-1).mean(),                  # proto successor-feature norm
                "psm_M_diag": torch.diagonal(Ms, dim1=1, dim2=2).mean(),   # measure self-term scale
                "psm_M_offdiag_abs": (Ms.abs() * self.off_diag).sum() / (Ms.shape[0] * self.off_diag_sum),
                "psm_targetM_diag": torch.diagonal(target_M).mean(),       # bootstrap target scale
            }

        return {
            "psm_loss": psm_loss.detach(),
            "psm_diag": psm_diag.detach(),
            "psm_offdiag": psm_offdiag.detach(),
            "orth_loss": orth_loss.detach(),
            "orth_loss_diag": orth_loss_diag.detach(),
            "orth_loss_offdiag": orth_loss_offdiag.detach(),
            **diag,
        }

    # ── SF net update (reference _update_sf, psm.py:495-532) ── #
    def _update_sf(self, obs, action, discount, next_obs, goal, z, next_action=None):
        # `next_action` is a test-only injection hook (default None = current behavior:
        # sample the next action from the learned actor). Used by the bit-exact
        # reference-equivalence test to bypass stochastic sampling; never set in training.
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            # compute target successor measure (target next-action from the LEARNED policy)
            with torch.no_grad():
                if next_action is None:
                    dist = self.model.actor(next_obs, z, self.actor_std)
                    next_action = dist.sample(clip=self.stddev_clip)
                target_psis = self.model.target_sf_psi(next_obs, z, next_action)  # P x B x z_dim
                target_phi = self.model.phi(goal)  # B x z_dim  (NOT target_phi — ref line 499)
                target_Ms = torch.matmul(target_psis, target_phi.T)  # P x B x B
                target_M_mean, target_M_unc = self.model.get_targets_uncertainty(target_Ms)  # B x B
                target_M = target_M_mean - self.pessimism_penalty * target_M_unc  # B x B

            # compute SF loss
            psis = self.model.sf_psi(obs, z, action)  # P x B x z_dim
            phi = self.model.phi(goal).detach()  # B x z_dim (DETACHED — ref line 506)
            Ms = torch.matmul(psis, phi.T)  # P x B x B

            diff = Ms - discount * target_M  # P x B x B
            sf_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            sf_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            sf_loss = sf_offdiag + sf_diag

            # orthonormality loss for phi embedding (ortho effectively OFF for SF: 0*)
            Cov = torch.matmul(phi, phi.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            sf_loss = sf_loss + 0 * self.ortho_coef * orth_loss

        # optimize SF (sf_psi only)
        self.optim_sf_psi.zero_grad(set_to_none=True)
        sf_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.sf_psi.parameters(), self.clip_grad_norm)
        self.optim_sf_psi.step()

        # ── diagnostics (pure logging; no grad/RNG effect on the update above) ──
        with torch.no_grad():
            diag = {
                "sf_phi_norm": phi.norm(dim=-1).mean(),                    # (detached proto) basis norm seen by SF
                "sf_psi_norm": psis.norm(dim=-1).mean(),                   # SF successor-feature norm (drives q)
                "sf_M_diag": torch.diagonal(Ms, dim1=1, dim2=2).mean(),
                "sf_M_offdiag_abs": (Ms.abs() * self.off_diag).sum() / (Ms.shape[0] * self.off_diag_sum),
                "sf_targetM_diag": torch.diagonal(target_M).mean(),
            }

        return {
            "sf_loss": sf_loss.detach(),
            "sf_diag": sf_diag.detach(),
            "sf_offdiag": sf_offdiag.detach(),
            **diag,
        }

    # ── TD3 actor update (reference _update_td3_actor, psm.py:651-667) ── #
    def _update_actor(self, obs, z, action=None):
        # `action` is a test-only injection hook (default None = current behavior:
        # sample from the learned actor). Used by the bit-exact reference-equivalence
        # test to bypass the stochastic actor sample; never set in training.
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            dist = self.model.actor(obs, z, self.actor_std)
            if action is None:
                action = dist.sample(clip=self.stddev_clip)
            else:
                # inject a fixed clipped-noise sample while preserving the
                # reference straight-through estimator: forward = injected value,
                # grad-to-actor = d(dist.loc) (TruncatedNormal._clamp pass-through).
                action = dist._clamp(dist.loc + (action - dist.loc).detach())
            psis = self.model.sf_psi(obs, z, action)  # P x B x z_dim
            Qs = (psis * z).sum(-1)  # P x B
            Q_mean, Q_unc = self.model.get_targets_uncertainty(Qs)  # B
            Q = Q_mean - self.actor_pessimism_penalty * Q_unc  # B
            actor_loss = -Q.mean()

        # optimize actor
        self.optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.clip_grad_norm)
        self.optim_actor.step()

        return {"actor_loss": actor_loss.detach(), "q": Q.mean().detach(),
                "z_norm": z.norm(dim=-1).mean().detach()}

    # ── update orchestration (reference update, psm.py:695-773) ── #
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
        discount = discount.reshape(-1, 1)  # [B,1] => row-wise discount over Ms-discount*target_M

        # update obs-normalizer running stats then normalize in eval_mode (mirror FBAgent)
        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs_n = self.model._obs_normalizer(obs)
            next_obs_n = self.model._obs_normalizer(next_obs)

        # augment (Identity for state; random-shift for pixel) then encode. goal =
        # bw-encoded next_obs (phi_input="s"). Mirrors FBAgent.aug/enc; no-op for state.
        #
        # PSM runs THREE separate .backward() calls per step (psm / sf / actor). For
        # pixels the DrQ encoders carry a grad graph, so a single shared encode would
        # be backward-ed multiple times (graph freed after the first). We therefore
        # re-encode per branch (each backward owns its graph) and assign encoder
        # ownership to a single optimizer to avoid double-stepping:
        #   _fw_encoder (obs) -> optim_sf_psi ;  _bw_encoder (goal) -> optim_phi.
        # The psm/actor branches encode obs WITHOUT grad to the fw-encoder; the sf
        # branch's goal use is already detached/no-grad. For STATE the encoders are
        # Identity (no params, passthrough) so .detach()/re-encode are no-ops and the
        # update stays byte-identical.
        obs_n, next_obs_n = self.aug(obs_n, next_obs_n)

        metrics: Dict = {}

        # 1) proto-successor branch: binary z -> _update_psm -> soft-update psm_psi + phi targets.
        #    obs encoded WITHOUT fw-encoder grad (psm_psi-owned); goal encoded WITH bw-encoder grad.
        z_psm = self.model.sample_z_psm(self.batch_size, device=self.device)
        obs_enc_psm = self._enc_obs(obs_n).detach()
        next_obs_enc_psm = self._enc_obs(next_obs_n).detach()
        goal_psm = self._enc_goal(next_obs_n)
        metrics.update(self._update_psm(obs_enc_psm, action, discount, next_obs_enc_psm,
                                        next_obs_hash, goal_psm, z_psm))
        with torch.no_grad():
            _soft_update_params(self._psm_psi_paramlist, self._target_psm_psi_paramlist, self.target_tau)
            _soft_update_params(self._phi_paramlist, self._target_phi_paramlist, self.target_tau)

        # 2) SF + actor branch: mixed Gaussian z -> _update_sf -> _update_actor -> soft-update sf_psi.
        #    SF owns the fw-encoder grad (encode WITH grad); actor uses detached obs.
        goal_sf = self._enc_goal(next_obs_n).detach()
        z = self.sample_mixed_z(goal_sf)
        obs_enc_sf = self._enc_obs(obs_n)
        next_obs_enc_sf = self._enc_obs(next_obs_n).detach()
        metrics.update(self._update_sf(obs_enc_sf, action, discount, next_obs_enc_sf, goal_sf, z))
        metrics.update(self._update_actor(self._enc_obs(obs_n).detach(), z))
        with torch.no_grad():
            _soft_update_params(self._sf_psi_paramlist, self._target_sf_psi_paramlist, self.target_tau)

        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.reward_inference(obs.to(self.device), reward.to(self.device))

    def state_dict(self) -> dict:
        state = {
            "model": self.model.state_dict(),
            "optim_phi": self.optim_phi.state_dict(),
            "optim_sf_psi": self.optim_sf_psi.state_dict(),
            "optim_psm_psi": self.optim_psm_psi.state_dict(),
        }
        if self.model.actor is not None:
            state["optim_actor"] = self.optim_actor.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optim_phi.load_state_dict(state["optim_phi"])
        self.optim_sf_psi.load_state_dict(state["optim_sf_psi"])
        self.optim_psm_psi.load_state_dict(state["optim_psm_psi"])
        if self.model.actor is not None and "optim_actor" in state:
            self.optim_actor.load_state_dict(state["optim_actor"])
