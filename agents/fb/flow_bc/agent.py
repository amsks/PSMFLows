"""agents/fb/flow_bc/agent.py — Literal port of td_jepa/.../fb/flow_bc/agent.py.

Same substitution scheme as agents/fb/agent.py:
    self.cfg.train.X     -> self.X
    self._model.X        -> self.model.X
    cudagraphs/compile   -> dropped
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.model import FBFlowBCModel
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class FBFlowBCAgent(FBAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_cfg: NoiseConditionedActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        flow_steps: int = 10,
        lr_actor_vf: float = 3e-4,
        **fb_kwargs,
    ):
        if actor_cfg is None:
            actor_cfg = NoiseConditionedActorArchiConfig()
        # Stash FlowBC-specific config BEFORE super().__init__ so _make_model
        # (called inside super().__init__) sees actor_vf_cfg.
        self.flow_steps = flow_steps
        self.lr_actor_vf = lr_actor_vf
        self._actor_vf_cfg = actor_vf_cfg
        super().__init__(obs_space=obs_space, action_dim=action_dim, actor_cfg=actor_cfg, **fb_kwargs)

    def _make_model(self, **kwargs):
        """Override FBAgent._make_model to build FBFlowBCModel directly.
        Matches td_jepa's `cfg.model.build(...)` one-shot construction so
        RNG state at weight_init time is identical."""
        return FBFlowBCModel(actor_vf_cfg=self._actor_vf_cfg, **kwargs)

    def setup_training(self) -> None:
        # Model is already an FBFlowBCModel (built via _make_model). Just run
        # parent setup (weight_init, target nets, optimizers) and add actor_vf.
        super().setup_training()
        self.actor_vf_optimizer = torch.optim.Adam(
            self.model._actor_vf.parameters(),
            lr=self.lr_actor_vf,
            weight_decay=self.weight_decay,
        )

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        gk = {} if goal is None else {"goal": goal}
        action = self.model._actor(obs, z, noises, **gk)
        return action

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
        goal: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp):
            x_1 = action
            x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            t = torch.rand((x_1.shape[0], 1), device=action.device)
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            # flow matching l2 loss
            pred = self.model._actor_vf(obs, x_t, t)
            bc_flow_loss = torch.pow(pred - vel, 2).mean()

            # Q loss.
            with torch.no_grad():
                left_enc = self.model._left_encoder(obs)
            actor_in = left_enc if self.actor_encode_obs else obs
            noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
            # V1 goal-conditioning: the Q-driven actor head and F see the goal; the BC flow
            # vector field (_actor_vf above) stays goal-UNconditioned. goal=None => baseline.
            gk = {} if goal is None else {"goal": goal}
            actor_actions = self.model._actor(actor_in, z, noises, **gk)
            # one-step FB: F is z-independent (zero-z); Q still dots with the real z.
            Fs = self.model._forward_map(left_enc, self._fb_z(z), actor_actions, **gk)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

            # compute bc loss
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
            "bc_error": bc_error.detach() if isinstance(bc_error, torch.Tensor) else torch.tensor(bc_error),
            "q": Q.mean().detach(),
        }

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.flow_steps
            vels = self.model._actor_vf(obs, actions, t)
            actions = actions + vels / self.flow_steps
        actions = torch.clamp(actions, min=-1, max=1)
        return actions

    @torch.no_grad()
    def act(self, obs, z, *, eval_mode: bool = True):
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        goal = getattr(self, "_eval_goal", None) if self.goal_cond else None
        if goal is not None:
            goal = goal.to(self.device).float()
        return self.model.act(obs, z, mean=True, goal=goal)

    def state_dict(self) -> dict:
        s = super().state_dict()
        s["actor_vf_optimizer"] = self.actor_vf_optimizer.state_dict()
        return s

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.actor_vf_optimizer.load_state_dict(state["actor_vf_optimizer"])
