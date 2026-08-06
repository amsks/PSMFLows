"""PSMFlow agent — latent-space PSM over a frozen behavior flow.

Redesigned 2026-08-05 (PAPER/decisions.tex, first entry): the original Rung-1 semantics
indexed policies by a FIXED latent u' and bootstrapped psi(s', u', u') — faithful to a
family that is structurally non-goal-covering (fixed-u policies are orbiters or
tanh-saturated constant headings; expert routes are latent white noise). This version is
PSM transposed to the flow's latent space: policy identity lives in the task vector w,
actions live in latent space, and the deployed action is always a flow decode.

Code <-> write-up:
  phi (PhiMap)            -> varphi(x)        shared basis over future states
  psi (PsiMap)            -> psi(s, w, u)     measure head: z-slot = task vector w,
                                              action-slot = current latent u
  actor                   -> u(s, w, noise)   per-step latent actor (flowBC recipe)
  flow_vf / flow_onestep  -> G_theta          FROZEN behavior flow (FQL Stage-A ckpt)
  batch['noise_preimage'] -> u = E_theta(s,a) dataset latent (preimage pipeline)
  infer_z                 -> w = E[r varphi]  closed-form reward inference
  sample_actions          -> decode(actor latent) (acting=actor, default) or per-step
                             argmax_u psi(s,w,u)^T w over prior draws (acting=gpi)

TD target: M^{pi_w}(s, u_data, .) backs up onto psi(s', w, u(s', w)) — the ACTOR's
latent at s' (PSM's sf_next_action, in latent space). Policy improvement happens in
latent space; every action either branch implies is still a flow decode, so the
in-support constraint (C=1) is preserved. u_data (the Stage-B preimage) is the dataset
action in latent space — the in-sample anchor of the TD.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.psm import (
    _plain_config, contrastive_loss, off_diagonal_mask, ortho_loss, polyak_update,
    project_z, targets_uncertainty,
)
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField
from utils.psm_networks import FlowVectorField, NoiseConditionedActor, PhiMap, PsiMap


@flax.struct.dataclass
class StepInputs:
    """Per-step sampled quantities.

    u_data:  (B, d_a) dataset latent (preimage of the batch action)
    u_next:  (B, d_a) actor's latent at s' under task_w (the TD bootstrap action)
    task_w:  (B, z_dim) task vector, shared by the measure and actor branches
             (PSM's sample_mixed_z: Gaussian mixed with phi(next_obs[perm]))
    flow_*:  CFM draws for the actor's latent-space BC velocity field
    """
    u_data: Any
    u_next: Any
    task_w: Any
    flow_x0: Any
    flow_t: Any
    flow_noise: Any


class PSMFlowAgent(flax.struct.PyTreeNode):
    rng: Any
    phi: TrainState
    psi: TrainState
    target_phi: Any
    target_psi: Any
    actor: TrainState       # amortized LATENT actor (s, w, noise) -> u (flowBC recipe)
    actor_vf: TrainState    # CFM velocity field over preimage latents (the actor's BC anchor)
    flow_vf: Any        # FROZEN params: multi-step BC velocity field (ODE decode)
    flow_onestep: Any   # FROZEN params: one-step distilled decoder (fast decode)
    task_z: Any         # (z_dim,) eval task vector w, set via infer_eval_z
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ---- training ----
    def measure_loss(self, batch, sampled, phi_params, psi_params):
        c = self.config
        obs, next_obs = batch["observations"], batch["next_observations"]
        goal = next_obs  # phi_input='s' convention (as PSM)
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        u, w, u_next = sampled.u_data, sampled.task_w, sampled.u_next

        phi_g = self.phi(goal, params=phi_params)
        M = self.psi(obs, w, u, params=psi_params) @ phi_g.T
        tphi = self.phi(goal, params=self.target_phi)
        # Bootstrap action = the ACTOR's latent at s' (PSM sf_loss's sf_next_action,
        # transposed to latent space). This is where policy improvement enters the
        # measure: the continuation is pi_w, not a fixed-u orbit (decisions.tex 08-05).
        tM = self.psi(next_obs, w, u_next, params=self.target_psi) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = tmean - c["pessimism_penalty"] * tunc
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
        loss = cl + c["ortho_coef"] * ol
        return loss, {"psm_loss": cl, "psm_diag": cdiag, "psm_offdiag": coff,
                      "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B, adim = batch["observations"].shape[0], c["action_dim"]
        # Clip u_data: `u_clip` is the typical-set clamp on ALL latent draws, and the
        # bootstrap side only ever sees tanh-bounded actor latents. Leaving u_data raw
        # gives the online branch inputs the target branch never sees, and the stored EM
        # mixtures are broad enough for that to be catastrophic: sampled |u| reaches 377
        # per-dim on pointmaze (43% of rows exceed u_clip), which drove psm_offdiag from
        # ~300 to 3.5e7 and left psm_diag flat. Clipping restores train/target
        # consistency; it does NOT fix the underlying mixture calibration.
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        # Task vector w, shared by measure + actor branches: Gaussian, with a mix_ratio
        # fraction replaced by project_z(phi(next_obs[perm])) — PSM's sample_mixed_z.
        # stop_gradient: sampling must not shape the basis.
        r_w, r_wmix, r_wperm, r_next, r_x0, r_t, r_noise = jax.random.split(rng, 7)
        gauss_w = project_z(jax.random.normal(r_w, (B, c["z_dim"])), c["norm_z"])
        wperm = jax.random.permutation(r_wperm, B)
        goal_w = project_z(jax.lax.stop_gradient(self.phi(batch["next_observations"]))[wperm],
                           c["norm_z"])
        wmask = (jax.random.uniform(r_wmix, (B,)) < c["mix_ratio"])[:, None]
        task_w = jnp.where(wmask, goal_w, gauss_w)
        # TD bootstrap action: the actor's latent at s' under task_w (PSM's flow-mode
        # sf_next_action). NoiseConditionedActor is tanh-bounded; scale to the u box.
        u_next = c["u_clip"] * self.actor(batch["next_observations"], task_w,
                                          jax.random.normal(r_next, (B, adim)))
        return StepInputs(
            u_data=u_data, u_next=u_next, task_w=task_w,
            flow_x0=jax.random.normal(r_x0, (B, adim)),
            flow_t=jax.random.uniform(r_t, (B, 1)),
            flow_noise=jax.random.normal(r_noise, (B, adim)))

    def flow_actor_loss(self, batch, sampled, actor_params, vf_params):
        """flowBC actor (fb_flowbc recipe), transposed to LATENT space.

        Same three terms as agents/psm.py:flow_actor_loss, with the action space
        replaced by the flow's latent space:
          - CFM velocity field v(s, x_t, t) trained toward the dataset PREIMAGE latents
            u_data — the latent-space behavior distribution. (Under an exact flow this is
            N(0, I) everywhere; measured, it is not — D3 — which is exactly why the BC
            anchor is worth learning rather than assuming.)
          - one-step NoiseConditionedActor(s, w, noise) -> tanh * u_clip, distilled from
            the vf's Euler rollout.
          - Q term: psi(s, w, u_a)^T w — the task-w value of emitting latent u_a now —
            with ensemble pessimism (PSM's flow_actor_loss, action slot = latent).
        psi is read at fixed params (no gradient); the frozen flow is untouched. Deployed
        action = flow decode of the actor's latent, so it stays data-like by construction.
        """
        c = self.config
        obs = batch["observations"]
        w = sampled.task_w
        x0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
        steps = c["actor"]["flow_steps"]
        u_clip = c["u_clip"]

        def rollout(vf_p, o, n):
            u = n
            for i in range(steps):
                ti = jnp.full((o.shape[0], 1), i / steps)
                u = u + self.actor_vf(o, u, ti, params=vf_p) / steps
            return jnp.clip(u, -u_clip, u_clip)

        # CFM toward the dataset's latents (u_data plays the role of the data action).
        x1 = sampled.u_data
        xt = (1 - t) * x0 + t * x1
        pred = self.actor_vf(obs, xt, t, params=vf_params)
        bc_flow_loss = jnp.mean((pred - (x1 - x0)) ** 2)

        u_a = u_clip * self.actor(obs, w, noise, params=actor_params)
        # psi at its stored params: the DPG gradient reaches the actor THROUGH psi's
        # inputs; psi's own params are outside the argnums and receive no gradient.
        Qs = (self.psi(obs, w, u_a) * w).sum(-1)  # (P, B)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        Q = qmean - c["actor_pessimism_penalty"] * qunc
        q_loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(Qs).mean() + 1e-8)
        distill = jnp.mean((u_a - jax.lax.stop_gradient(rollout(vf_params, obs, noise))) ** 2)
        loss = q_loss + c["actor"]["bc_coeff"] * distill + bc_flow_loss
        return loss, {"actor_loss": loss, "actor_q": Q.mean(),
                      "actor_bc_flow_loss": bc_flow_loss, "actor_bc_error": distill}

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_phi, g_psi) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.psi.params)
        phi = self.phi.apply_gradients(grads=g_phi)
        psi = self.psi.apply_gradients(grads=g_psi)
        target_phi = polyak_update(phi.params, self.target_phi, tau)
        target_psi = polyak_update(psi.params, self.target_psi, tau)
        new = self.replace(phi=phi, psi=psi, target_phi=target_phi, target_psi=target_psi)
        # Actor branch at the PRE-update psi (PSM's no-interleave convention). The actor
        # is load-bearing: the measure backup bootstraps its latent at s', so it always
        # trains (no enable switch).
        (_, a_info), (g_a, g_vf) = jax.value_and_grad(
            self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.actor.params, self.actor_vf.params)
        new = new.replace(actor=new.actor.apply_gradients(grads=g_a),
                          actor_vf=new.actor_vf.apply_gradients(grads=g_vf))
        return new, {**info, **a_info}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss at current params (no step)."""
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.phi.params, self.psi.params)
        return loss, info

    # ---- inference ----
    def infer_z(self, next_observations, rewards):
        phi = self.phi(next_observations)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_eval_z(self, next_observations, rewards):
        """Copy of this agent with task_z set from rewards. Picked up generically by
        main.py's eval hook (hasattr(agent, 'infer_eval_z'))."""
        return self.replace(task_z=self.infer_z(next_observations, rewards))

    def decode(self, observations, u):
        """Decode latents through the FROZEN flow. observations (B, ob), u (B, d_a)."""
        if self.config["gpi_decode"] == "onestep":
            a = self.flow_onestep_def.apply({"params": self.flow_onestep}, observations, u)
        else:
            a = u
            steps = self.config["flow_decode_steps"]
            for i in range(steps):
                t = jnp.full((*observations.shape[:-1], 1), i / steps)
                a = a + self.flow_vf_def.apply({"params": self.flow_vf}, observations, a, t) / steps
        return jnp.clip(a, -1.0, 1.0)

    @jax.jit
    def gpi_select(self, observations, seed=None):
        """Per-step latent argmax: argmax_u [mean_P - pess * unc](psi(s, w, u)^T w) over
        K clipped prior draws. The actor-free acting path (ablation vs acting=actor);
        re-run every step, so this is per-step reselection, not a fixed index.

        Single observation (ob_dims,) — the eval/acting path."""
        c = self.config
        assert observations.ndim == 1, "gpi_select acts on a single observation"
        K, d_a = c["gpi_num_u"], c["action_dim"]
        seed = self.rng if seed is None else seed
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        qpsis = self.psi(obs, w, u_cand)                    # (P, K, z_dim)
        Qs = (qpsis * self.task_z).sum(-1)                  # (P, K)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        Q = qmean - c["actor_pessimism_penalty"] * qunc
        return u_cand[jnp.argmax(Q)]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        seed = self.rng if seed is None else seed
        if self.config["acting"] == "actor":
            # One shot through the amortized latent actor, then decode (default).
            noise = jax.random.normal(seed, (1, self.config["action_dim"]))
            u_star = self.config["u_clip"] * self.actor(
                observations[None], self.task_z[None], noise)[0]
        else:
            u_star = self.gpi_select(observations, seed=seed)
        return self.decode(observations[None], u_star[None])[0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rphi, rpsi, rvf, ronestep = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "psmflow does not support visual encoders yet."
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))
        ex_w = jnp.zeros((ex_observations.shape[0], z_dim))

        phi_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                         hidden_layers=config["phi"]["hidden_layers"], norm=True)
        psi_def = PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                         num_parallel=config["num_parallel"],
                         embedding_layers=config["sf"]["embedding_layers"],
                         hidden_layers=config["sf"]["hidden_layers"])
        phi = TrainState.create(phi_def, phi_def.init(rphi, ex_observations)["params"],
                                tx=optax.adam(config["lr_phi"]))
        # psi z-slot carries the task vector w (z_dim), action slot the latent u (d_a).
        psi = TrainState.create(psi_def, psi_def.init(rpsi, ex_observations, ex_w, ex_u)["params"],
                                tx=optax.adam(config["lr_sf"]))

        # Amortized latent actor (flowBC recipe over latents). Load-bearing: the measure
        # backup bootstraps its latent at s'.
        rng, ractor, ravf = jax.random.split(rng, 3)
        actor_def = NoiseConditionedActor(
            action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
            hidden_layers=config["actor"]["hidden_layers"],
            embedding_layers=config["actor"]["embedding_layers"])
        actor_vf_def = FlowVectorField(
            action_dim=action_dim, hidden_dim=config["actor"]["vf_hidden_dim"],
            hidden_layers=config["actor"]["vf_hidden_layers"])
        actor = TrainState.create(
            actor_def, actor_def.init(ractor, ex_observations, ex_w, ex_u)["params"],
            tx=optax.adam(config["lr_actor"]))
        actor_vf = TrainState.create(
            actor_vf_def, actor_vf_def.init(ravf, ex_observations, ex_u, ex_u[..., :1])["params"],
            tx=optax.adam(config["lr_actor_vf"]))

        # FROZEN behavior flow (FQL Stage-A checkpoint). Defs are rebuilt from config;
        # shapes must match the checkpointed run.
        flow_hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                       layer_norm=config["flow"]["layer_norm"])
        ckpt = config.get("flow_ckpt_path", None)
        if ckpt:
            flow_vf, flow_onestep = _load_flow_params(
                ckpt, config.get("flow_ckpt_epoch", None), config,
                ex_observations, ex_actions)
        else:
            assert config.get("allow_untrained_flow", False), (
                "psmflow requires agent.flow_ckpt_path (a Stage-A fql bc_only ckpt dir); "
                "set agent.allow_untrained_flow=true only for tests/smokes.")
            ex_times = ex_actions[..., :1]
            flow_vf = vf_def.init(rvf, ex_observations, ex_actions, ex_times)["params"]
            flow_onestep = onestep_def.init(ronestep, ex_observations, ex_actions)["params"]

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        return cls(rng=rng, phi=phi, psi=psi,
                   target_phi=copy.deepcopy(phi.params), target_psi=copy.deepcopy(psi.params),
                   actor=actor, actor_vf=actor_vf,
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   task_z=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)


def _load_flow_params(ckpt_path, ckpt_epoch, config, ex_observations, ex_actions):
    """Extract the frozen flow subtrees from a Stage-A FQL(bc_only) checkpoint.

    Builds a throwaway FQLAgent with matching shapes and restores the pickle, then
    pulls modules_actor_bc_flow / modules_actor_onestep_flow.
    """
    from agents.fql import FQLAgent, get_config as fql_get_config
    from utils.flax_utils import restore_agent

    fql_cfg = fql_get_config()
    fql_cfg["actor_hidden_dims"] = tuple(config["flow"]["hidden_dims"])
    fql_cfg["value_hidden_dims"] = tuple(config["flow"]["value_hidden_dims"])
    fql_cfg["actor_layer_norm"] = config["flow"]["layer_norm"]
    fql_cfg["layer_norm"] = config["flow"]["critic_layer_norm"]
    fql_agent = FQLAgent.create(0, ex_observations, ex_actions, fql_cfg)
    fql_agent = restore_agent(fql_agent, ckpt_path, ckpt_epoch)
    params = fql_agent.network.params
    vf, onestep = params["modules_actor_bc_flow"], params["modules_actor_onestep_flow"]

    # restore_agent replaces the param tree wholesale and does NOT check shapes, so a
    # checkpoint from a different environment loads without complaint and only misbehaves
    # much later, at decode time. The bc-flow trunk takes concat[obs, action, t], so its
    # first kernel pins the env this flow was trained on.
    ob_dim = int(ex_observations.shape[-1])
    action_dim = int(ex_actions.shape[-1])
    got = int(vf["mlp"]["Dense_0"]["kernel"].shape[0])
    expected = ob_dim + action_dim + 1
    assert got == expected, (
        f'flow checkpoint {ckpt_path} expects concat[obs, action, t] of width {got}, but '
        f'this env gives {expected} (obs {ob_dim} + action {action_dim} + t 1); '
        'the Stage-A checkpoint was trained on a different environment')
    return vf, onestep


def get_config():
    """Importable default config, mirrored by configs/agent/psmflow.yaml."""
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            agent_name="psmflow",
            batch_size=1024,
            z_dim=128,
            num_parallel=2,
            discount=0.98,
            tau=0.01,
            ortho_coef=1000.0,       # reference PSM sweep winner
            # 0.5 with num_parallel=2 = exact min-Q in the TD target; see the yaml note.
            pessimism_penalty=0.5,
            actor_pessimism_penalty=0.5,
            norm_z=True,
            lr_phi=1.0e-5,
            lr_sf=1.0e-4,
            lr_actor=1.0e-4,
            lr_actor_vf=3.0e-4,
            phi=dict(hidden_dim=256, hidden_layers=2),
            sf=dict(hidden_dim=1024, hidden_layers=1, embedding_layers=2),
            mix_ratio=0.5,           # P(w from phi(next_obs[perm])) vs random unit z
            # amortized latent actor (flowBC recipe over latents). Load-bearing: the
            # measure backup bootstraps its latent at s'; always trained.
            actor=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                       vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10,
                       bc_coeff=1.0),
            acting="actor",          # actor | gpi (per-step latent argmax, actor-free)
            # frozen behavior flow (must match the Stage-A fql bc_only run)
            flow=dict(hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
                      layer_norm=False, critic_layer_norm=True),
            flow_ckpt_path=ml_collections.config_dict.placeholder(str),
            flow_ckpt_epoch=ml_collections.config_dict.placeholder(int),
            allow_untrained_flow=False,
            preimage_path=ml_collections.config_dict.placeholder(str),
            use_point_preimage=False,
            u_clip=3.0,              # typical-set clamp on all latent draws
            # acting=gpi (per-step latent argmax) inference
            gpi_num_u=64,
            gpi_decode="onestep",    # onestep | ode
            flow_decode_steps=10,    # Euler steps for gpi_decode=ode
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
