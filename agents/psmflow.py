"""PSMFlow agent — measure learning in a frozen behavior flow's latent action space.

Actions live in the flow's latent space and the deployed action is always a flow decode,
so every policy stays inside the cloned behaviour. The measure branch is FB-shaped (one
branch, basis trained against the task head), not PSM's proto/sf split.

Two config flags select what psi is indexed by and what the TD bootstrap is; both default
to the shipped behaviour, under which Prop. insample's C=1 does NOT apply (the bootstrap
latent is the actor's, hence not independent of s'). See `_index`, `backup_explore_frac`,
and docs/COMPENDIUM.md §3 for the full code <-> write-up correspondence.
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
    # Policy-index latent u' ~ p0 (policy_index='latent' only; None otherwise). Under the
    # paper's psi(s, u, u') this occupies the index slot task_w occupies by default.
    u_index: Any
    flow_x0: Any
    flow_t: Any
    flow_noise: Any
    # Task vector for the ACTION branch. Identical to task_w unless the P2 FB graft is on,
    # in which case it is sampled against the branch's own basis B_a (see sample_step_inputs).
    task_w_a: Any


class PSMFlowAgent(flax.struct.PyTreeNode):
    rng: Any
    phi: TrainState
    psi: TrainState
    target_phi: Any
    target_psi: Any
    actor: TrainState       # amortized LATENT actor (s, w, noise) -> u (flowBC recipe)
    actor_vf: TrainState    # CFM velocity field over preimage latents (the actor's BC anchor)
    # Action branch (config action_critic.enabled, default false): a second
    # successor-feature head over EXECUTED actions, psi_a(s, w, a), sharing phi and the
    # w-machinery, plus a bounded residual delta(s, w, u). Q_a = psi_a^T w supplies the
    # value-directed gradient a latent critic cannot when the decode is narrow; the
    # executed action a = G(s,u) + eps*delta stays within eps of the decode.
    psi_a: TrainState
    target_psi_a: Any
    residual: TrainState
    # FB graft (config action_critic.fb_graft, default false): the action branch's own
    # backward map B_a, the basis psi_a's measure is written against and the source of w_a.
    # Always created so the pytree stays static; trained only when the graft is on.
    phi_a: TrainState
    target_phi_a: Any
    flow_vf: Any        # FROZEN params: multi-step BC velocity field (ODE decode)
    flow_onestep: Any   # FROZEN params: one-step distilled decoder (fast decode)
    task_z: Any         # (z_dim,) eval task vector w, set via infer_eval_z
    task_z_a: Any       # (z_dim,) eval task vector for the ACTION branch: inferred from
                        # B_a under the graft, else a copy of task_z
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

        idx = self._index(sampled)
        phi_g = self.phi(goal, params=phi_params)
        M = self.psi(obs, idx, u, params=psi_params) @ phi_g.T
        tphi = self.phi(goal, params=self.target_phi)
        # Bootstrap action = the ACTOR's latent at s' (PSM sf_loss's sf_next_action in
        # latent space): the continuation is pi_w, which is where policy improvement
        # enters the measure. Under backup_explore_frac>0 a fraction of those latents are
        # prior draws instead; under policy_index='latent' all of them are.
        # policy_index='latent': u_next IS u_index, so this reads psibar(s', u', u') --
        # both slots on the same continuation index, which is the paper's backup.
        tM = self.psi(next_obs, idx, u_next, params=self.target_psi) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = tmean - c["pessimism_penalty"] * tunc
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
        loss = cl + c["ortho_coef"] * ol
        return loss, {"psm_loss": cl, "psm_diag": cdiag, "psm_offdiag": coff,
                      "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}

    def _index(self, sampled):
        """Whatever occupies psi's index slot.

        'task_vector' (default, shipped): the task vector w, so psi is a task-conditioned
        head and policy identity lives in w. 'latent': a fresh prior draw u' ~ p0 per
        batch element, i.e. the paper's psi(s, u, u') where the index is a POLICY latent
        and w never enters psi at all -- it appears only in the readout Q = psi^T w.
        """
        return sampled.u_index if self.config["policy_index"] == "latent" else sampled.task_w

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B, adim = batch["observations"].shape[0], c["action_dim"]
        # u_clip is the typical-set clamp on EVERY latent here: the bootstrap side is a
        # tanh-bounded actor latent, so an unclipped u_data would hand the online branch
        # inputs the target branch can never produce (HANDOFF 2026-08-05).
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        # Task vector w, shared by the measure and actor branches: Gaussian, with a
        # mix_ratio fraction replaced by phi(next_obs[perm]) — PSM's sample_mixed_z.
        # Sampling must not shape the basis, hence stop_gradient.
        r_w, r_wmix, r_wperm, r_next, r_x0, r_t, r_noise = jax.random.split(rng, 7)
        gauss_w = project_z(jax.random.normal(r_w, (B, c["z_dim"])), c["norm_z"])
        wperm = jax.random.permutation(r_wperm, B)
        goal_w = project_z(jax.lax.stop_gradient(self.phi(batch["next_observations"]))[wperm],
                           c["norm_z"])
        wmask = (jax.random.uniform(r_wmix, (B,)) < c["mix_ratio"])[:, None]
        task_w = jnp.where(wmask, goal_w, gauss_w)
        # Policy index u' ~ p0, one per batch element, CLIPPED exactly as the deployed
        # draw is (gpi_select). Key folded out of rng rather than split from it, so with
        # policy_index='task_vector' every draw below is byte-identical to before.
        u_index = None
        if c["policy_index"] == "latent":
            u_index = jnp.clip(
                jax.random.normal(jax.random.fold_in(rng, 106), (B, adim)),
                -c["u_clip"], c["u_clip"])

        # TD bootstrap: the actor's latent at s' under task_w (PSM's sf_next_action),
        # tanh-bounded and scaled to the u box.
        u_next = c["u_clip"] * self.actor(batch["next_observations"], task_w,
                                          jax.random.normal(r_next, (B, adim)))
        # Backup exploration (default off): replace a fraction of bootstrap latents with
        # prior draws, so psi is fit over the whole latent family rather than the slice
        # the actor already occupies. Keys are drawn inside the branch, so at frac=0 the
        # random stream is unchanged.
        if c["backup_explore_frac"] > 0.0:
            r_expl, r_emask = jax.random.split(r_next)
            u_prior = jnp.clip(jax.random.normal(r_expl, (B, adim)),
                               -c["u_clip"], c["u_clip"])
            emask = (jax.random.uniform(r_emask, (B,)) < c["backup_explore_frac"])[:, None]
            u_next = jnp.where(emask, u_prior, u_next)
        if c["policy_index"] == "latent":
            # The continuation at s' is pi_{u'}, the SAME index the online side carries --
            # that is what makes u' a policy index rather than one more noise draw. It also
            # makes the bootstrap action G(s', u') a p0 decode, which is Prop. insample's
            # hypothesis, so backup_explore_frac is inert here by construction.
            u_next = u_index
        # FB graft (default off): the action branch gets its own task vector, sampled
        # against its own backward map B_a the way FB samples z against B. Keys are folded
        # out of `rng` inside the branch, so with the graft off no draw above changes.
        task_w_a = task_w
        if c["action_critic"]["fb_graft"]:
            r_wa, r_wamix, r_waperm = jax.random.split(jax.random.fold_in(rng, 104), 3)
            gauss_wa = project_z(jax.random.normal(r_wa, (B, c["z_dim"])), c["norm_z"])
            goal_wa = project_z(
                jax.lax.stop_gradient(self.phi_a(batch["next_observations"]))[
                    jax.random.permutation(r_waperm, B)], c["norm_z"])
            wa_mask = (jax.random.uniform(r_wamix, (B,)) < c["mix_ratio"])[:, None]
            task_w_a = jnp.where(wa_mask, goal_wa, gauss_wa)
        return StepInputs(
            u_data=u_data, u_next=u_next, task_w=task_w, u_index=u_index,
            flow_x0=jax.random.normal(r_x0, (B, adim)),
            flow_t=jax.random.uniform(r_t, (B, 1)),
            flow_noise=jax.random.normal(r_noise, (B, adim)),
            task_w_a=task_w_a)

    def flow_actor_loss(self, batch, sampled, actor_params, vf_params):
        """flowBC actor (fb_flowbc recipe) with the action space replaced by latents.

        Three terms, as in agents/psm.py:flow_actor_loss:
          - CFM velocity field v(s, x_t, t) toward the dataset latents u_data (the
            latent-space behavior distribution, which is measurably not N(0, I));
          - a one-step actor(s, w, noise) -> tanh * u_clip, distilled from its rollout;
          - Q = psi(s, w, u_a)^T w with ensemble pessimism.
        psi is read at fixed params and the frozen flow is untouched, so the deployed
        action stays a flow decode.
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

    # ---- Idea-1 action branch ----
    def execute(self, observations, w, u, residual_params=None):
        """Executed action: frozen decode plus the eps-bounded, w-conditioned residual.

        a = clip(G(s, u) + eps * delta(s, w, u)). NoiseConditionedActor tanh-bounds its
        output, so eps is a hard per-dimension budget on the distance from the decode.
        Only used when action_critic.enabled; at eps=0 it reduces to the pure decode.
        """
        a = self.decode(observations, u)
        eps = self.config["residual_eps"]
        if eps > 0.0:
            p = self.residual.params if residual_params is None else residual_params
            a = a + eps * self.residual(observations, w, u, params=p)
        return jnp.clip(a, -1.0, 1.0)

    def action_critic_loss(self, batch, sampled, psi_a_params):
        """Vector TD for psi_a(s, w, a): successor features of the SHARED phi under pi_w.

        target = phi(s') + gamma_ac * psi_a_bar(s', w, a'_exec), with a'_exec the actor's
        executed action at s' (decode + residual, stop-grad). Reward-free, so the branch
        stays zero-shot: Q_a for any reward is psi_a^T w with w = E[r phi]. Pessimism on
        this branch defaults to 0.0 — the collapse forensics (08-13 HANDOFF) showed
        exact-min pessimism DRIVES the value collapse under a residual, so the disease is
        not ported here.
        """
        c = self.config
        ac = c["action_critic"]
        obs, next_obs = batch["observations"], batch["next_observations"]
        w = sampled.task_w
        a_next = self.execute(next_obs, w, sampled.u_next)
        pred = self.psi_a(obs, w, batch["actions"], params=psi_a_params)      # (P, B, z)
        phi_next = jax.lax.stop_gradient(self.phi(next_obs, params=self.target_phi))
        t_next = self.psi_a(next_obs, w, jax.lax.stop_gradient(a_next),
                            params=self.target_psi_a)
        tmean, tunc = targets_uncertainty(t_next, c["num_parallel"])
        # Pessimism in Q-space, not per-feature: a per-feature spread shifts Q by
        # -pessimism * unc^T w, whose sign flips wherever w < 0, so it raised the target
        # on half the basis. Blend the ensemble mean toward the member this task values
        # least instead; at lambda = 0 (the default) this is the plain ensemble mean.
        q_next = (t_next * w[None]).sum(-1)                                  # (P, B)
        worst = jnp.argmin(q_next, axis=0)                                   # (B,)
        t_worst = jnp.take_along_axis(t_next, worst[None, :, None], axis=0)[0]
        lam = ac["pessimism"]
        target = phi_next + ac["discount"] * ((1.0 - lam) * tmean + lam * t_worst)
        loss = jnp.mean((pred - jax.lax.stop_gradient(target)[None]) ** 2)
        q_data = (pred * w[None]).sum(-1).mean()
        return loss, {"ac_loss": loss, "ac_q_data": q_data,
                      "ac_target_q_gap": (q_next.mean(0) - jnp.min(q_next, axis=0)).mean(),
                      "ac_feature_unc": tunc.mean()}

    def action_critic_fb_loss(self, batch, sampled, psi_a_params, phi_a_params):
        """FB graft: train (psi_a, B_a) as a self-contained FB pair over EXECUTED actions.

        B_a(s') is the branch's own backward map, trained by the FB measure loss verbatim
        (contrastive off-diag/diag + ortho on B_a, no pessimism, FB's target tau), and w_a
        is inferred from it — so delta and Q_a stop depending on the shared basis phi.
        phi / psi / w keep training as before for the latent actor. No gradient crosses
        between this branch and the shared measure in either direction.

        Result on record: this FAILED its gate (deployed 0.064 vs a 0.45 bar).
        """
        c = self.config
        ac = c["action_critic"]
        obs, next_obs = batch["observations"], batch["next_observations"]
        goal = next_obs                      # phi_input='s' convention, as the shared branch
        off, off_sum = off_diagonal_mask(obs.shape[0])
        w = sampled.task_w_a
        a_next = jax.lax.stop_gradient(self.execute(next_obs, w, sampled.u_next))

        tB = self.phi_a(goal, params=self.target_phi_a)
        tM = self.psi_a(next_obs, w, a_next, params=self.target_psi_a) @ tB.T
        tmean, _ = targets_uncertainty(tM, c["num_parallel"])
        B_a = self.phi_a(goal, params=phi_a_params)
        M = self.psi_a(obs, w, batch["actions"], params=psi_a_params) @ B_a.T
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(tmean),
                                           ac["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(B_a, off, off_sum)
        loss = cl + ac["ortho_coef"] * ol
        q_data = (self.psi_a(obs, w, batch["actions"], params=psi_a_params)
                  * w[None]).sum(-1).mean()
        return loss, {"ac_loss": loss, "ac_fb_diag": cdiag, "ac_fb_offdiag": coff,
                      "ac_orth_loss": ol, "ac_orth_diag": odiag, "ac_q_data": q_data}

    def action_critic_spread(self, batch, sampled, key):
        """Live "is the action critic awake?" signal, logged every eval.

        Q_a(s, a) = psi_a(s, w, a)^T w evaluated over `spread_candidates` decoded
        candidates at the SAME state. If that spread is ~0 the critic cannot rank the
        actions the decoder can produce, the -Q_a gradient carries no direction, and the
        residual head is climbing noise -- the D3 failure, which until now was only ever
        found post-hoc by re-loading a finished checkpoint (one wasted run per discovery).
        Reported relative to |Q| so it is comparable across tasks and training steps.
        """
        c = self.config
        n = min(64, batch["observations"].shape[0])
        obs, w = batch["observations"][:n], sampled.task_w_a[:n]
        cand = c["action_critic"]["spread_candidates"]
        u = jnp.clip(jax.random.normal(key, (cand, n, c["action_dim"])),
                     -c["u_clip"], c["u_clip"])

        def q_of(u_k):
            a = self.execute(obs, w, u_k)
            return (self.psi_a(obs, w, a) * w[None]).sum(-1).mean(0)         # (n,)

        Q = jax.vmap(q_of)(u)                                                # (cand, n)
        scale = jnp.abs(Q).mean() + 1e-8
        return {"ac_q_spread": Q.std(0).mean(),
                "ac_q_spread_rel": Q.std(0).mean() / scale,
                "ac_q_range_rel": (Q.max(0) - Q.min(0)).mean() / scale}

    def residual_loss(self, batch, sampled, residual_params):
        """-Q_a on the executed action; the value-directed step the decoder cannot take.

        The latent actor is read at stop-grad (its own training is untouched); only the
        residual head chases Q_a, and only within its eps budget.
        """
        c = self.config
        obs, w = batch["observations"], sampled.task_w_a
        # The LATENT actor stays on the shared w (its own training is untouched); only the
        # residual and Q_a move to the action branch's w_a, which differs under the graft.
        u_a = jax.lax.stop_gradient(
            c["u_clip"] * self.actor(obs, sampled.task_w, sampled.flow_noise))
        a_exec = self.execute(obs, w, u_a, residual_params=residual_params)
        Qs = (self.psi_a(obs, w, a_exec) * w[None]).sum(-1)                   # (P, B)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        Q = qmean - c["action_critic"]["pessimism"] * qunc
        loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(Qs).mean() + 1e-8)
        spend = jnp.mean(jnp.abs(a_exec - self.decode(obs, u_a)))
        return loss, {"residual_loss": loss, "residual_q": Q.mean(),
                      "residual_spend": spend}

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_phi, g_psi) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.psi.params)
        phi = self.phi.apply_gradients(grads=g_phi)
        psi = self.psi.apply_gradients(grads=g_psi)
        target_phi = polyak_update(phi.params, self.target_phi, tau)
        target_psi = polyak_update(psi.params, self.target_psi, tau)
        new = self.replace(phi=phi, psi=psi, target_phi=target_phi, target_psi=target_psi)
        # Actor branch at the PRE-update psi (PSM's no-interleave convention). Under the
        # shipped index the actor is load-bearing -- the measure backup bootstraps its
        # latent at s'. Under policy_index='latent' the backup bootstraps u' instead, so
        # nothing in the measure depends on the actor and train_actor=false drops it
        # entirely (the paper's Sec. PSMFlows has no actor). Static switch: the enabled
        # path traces to exactly the pre-existing computation.
        a_info = {}
        if self.config["train_actor"]:
            (_, a_info), (g_a, g_vf) = jax.value_and_grad(
                self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
                batch, sampled, self.actor.params, self.actor_vf.params)
            new = new.replace(actor=new.actor.apply_gradients(grads=g_a),
                              actor_vf=new.actor_vf.apply_gradients(grads=g_vf))
        # Idea-1 action branch — static config switch, so the disabled path traces to
        # exactly the pre-existing computation (no extra rng, no shared-branch gradients).
        if self.config["action_critic"]["enabled"]:
            ac_cfg = self.config["action_critic"]
            if ac_cfg["fb_graft"]:
                # P2: psi_a and its OWN backward map B_a step together on the FB measure
                # loss; B_a's target follows FB's separate tau.
                (_, ac_info), (g_pa, g_ba) = jax.value_and_grad(
                    self.action_critic_fb_loss, argnums=(2, 3), has_aux=True)(
                    batch, sampled, new.psi_a.params, new.phi_a.params)
                phi_a = new.phi_a.apply_gradients(grads=g_ba)
                new = new.replace(phi_a=phi_a,
                                  target_phi_a=polyak_update(phi_a.params, new.target_phi_a,
                                                             ac_cfg["b_tau"]))
            else:
                (_, ac_info), g_pa = jax.value_and_grad(
                    self.action_critic_loss, argnums=2, has_aux=True)(
                    batch, sampled, new.psi_a.params)
            psi_a = new.psi_a.apply_gradients(grads=g_pa)
            target_psi_a = polyak_update(psi_a.params, new.target_psi_a,
                                         self.config["action_critic"]["tau"])
            (_, r_info), g_r = jax.value_and_grad(
                self.residual_loss, argnums=2, has_aux=True)(
                batch, sampled, new.residual.params)
            new = new.replace(psi_a=psi_a, target_psi_a=target_psi_a,
                              residual=new.residual.apply_gradients(grads=g_r))
            # Diagnostic only. The key is folded out of the stored rng rather than split
            # from it, so the branch's own random stream is untouched.
            s_info = new.action_critic_spread(batch, sampled,
                                              jax.random.fold_in(self.rng, 103))
            a_info = {**a_info, **ac_info, **r_info, **s_info}
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

    def infer_z_a(self, next_observations, rewards):
        """w_a = E[r B_a(s')], the action branch's own reward inference (FB's z formula
        against B_a). Without the graft B_a is untrained, so w_a falls back to w."""
        b = self.phi_a(next_observations)
        z = (rewards.reshape(1, -1) @ b).reshape(-1) / b.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_eval_z(self, next_observations, rewards):
        """Copy of this agent with task_z set from rewards. Picked up generically by
        main.py's eval hook (hasattr(agent, 'infer_eval_z')). Under the FB graft the
        action branch gets its OWN task vector, inferred through B_a -- the whole point of
        the graft is that delta and Q_a stop depending on the shared phi."""
        z = self.infer_z(next_observations, rewards)
        z_a = (self.infer_z_a(next_observations, rewards)
               if self.config["action_critic"]["fb_graft"] else z)
        return self.replace(task_z=z, task_z_a=z_a)

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
        if c["policy_index"] == "latent":
            # Alg. "Rung 1" verbatim: draw K action latents AND K policy indices, score
            # every (u_i, u'_j) pair, return G(s, u_ihat). The max over Lambda_K is the
            # GPI the bound is stated for; the max over u_i is the within-policy argmax.
            r_u, r_up = jax.random.split(seed)
            u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -c["u_clip"], c["u_clip"])
            u_idx = jnp.clip(jax.random.normal(r_up, (K, d_a)), -c["u_clip"], c["u_clip"])
            uu = jnp.repeat(u_cand, K, axis=0)              # (K*K, d_a), i-major
            ii = jnp.tile(u_idx, (K, 1))                    # (K*K, d_a)
            obs = jnp.broadcast_to(observations, (K * K, *observations.shape))
            qpsis = self.psi(obs, ii, uu)                   # (P, K*K, z_dim)
            Qs = (qpsis * self.task_z).sum(-1)              # (P, K*K)
            qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
            Q = qmean - c["actor_pessimism_penalty"] * qunc
            return uu[jnp.argmax(Q)]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        qpsis = self.psi(obs, w, u_cand)                    # (P, K, z_dim)
        Qs = (qpsis * self.task_z).sum(-1)                  # (P, K)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        Q = qmean - c["actor_pessimism_penalty"] * qunc
        return u_cand[jnp.argmax(Q)]

    def _acting_w_a(self):
        """Task vector the ACTION branch acts on: its own w_a under the FB graft, the
        shared w otherwise. Reading task_z_a unconditionally would silently zero Q_a for
        any caller that sets task_z directly instead of going through infer_eval_z."""
        return self.task_z_a if self.config["action_critic"]["fb_graft"] else self.task_z

    def qa_rank_select(self, observations, seed):
        """Eval-time ablation: RANK decoded candidates by Q_a instead of pushing on them.

        K actor draws -> K decodes (NO residual) -> execute argmax_a psi_a(s, w, a)^T w.
        Every action taken is therefore something the frozen flow could have produced,
        and the only thing the action critic contributes is a preference order. Run
        against the same checkpoint's residual acting, this separates "Q_a ranks better"
        from "Q_a pushes better" -- the hybrid's gain is otherwise a single number with
        two candidate causes. Off by default (eval_rank_k = 0); training never calls it.
        """
        c = self.config
        K, d_a = c["action_critic"]["eval_rank_k"], c["action_dim"]
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        noise = jax.random.normal(seed, (K, d_a))
        w_a = jnp.broadcast_to(self._acting_w_a(), (K, *self.task_z.shape))
        u_cand = c["u_clip"] * self.actor(obs, w, noise)
        a_cand = self.decode(obs, u_cand)
        Q = (self.psi_a(obs, w_a, a_cand) * w_a).sum(-1).mean(0)          # (K,)
        return a_cand[jnp.argmax(Q)]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        seed = self.rng if seed is None else seed
        ac = self.config["action_critic"]
        if ac["enabled"] and ac["eval_rank_k"] > 0:
            return self.qa_rank_select(observations, seed)
        assert not (self.config["acting"] == "actor" and not self.config["train_actor"]), (
            "acting=actor with train_actor=false would deploy an untrained actor; "
            "the paper's Sec. PSMFlows has no actor -- use acting=gpi")
        if self.config["acting"] == "actor":
            # One shot through the amortized latent actor, then decode (default).
            noise = jax.random.normal(seed, (1, self.config["action_dim"]))
            u_star = self.config["u_clip"] * self.actor(
                observations[None], self.task_z[None], noise)[0]
        else:
            u_star = self.gpi_select(observations, seed=seed)
        if ac["enabled"]:
            return self.execute(observations[None], self._acting_w_a()[None], u_star[None])[0]
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
        # psi index slot: the task vector w (z_dim) by default, the policy latent u'
        # (d_a) under policy_index='latent'. Action slot is the latent u either way.
        ex_index = ex_u if config["policy_index"] == "latent" else ex_w
        psi = TrainState.create(psi_def, psi_def.init(rpsi, ex_observations, ex_index, ex_u)["params"],
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

        # Idea-1 action branch: psi_a is the same tower class as psi with the action slot
        # carrying the RAW action (same width as the latent, d_a); the residual head is a
        # NoiseConditionedActor read as delta(s, w, u). Keys come from fold_in so the
        # stored rng stream — and therefore every draw of the pre-existing branches — is
        # byte-identical to the pre-Idea-1 code whether or not the branch is enabled.
        ac_cfg = config["action_critic"]
        psi_a_def = PsiMap(output_dim=z_dim, hidden_dim=ac_cfg["hidden_dim"],
                           num_parallel=config["num_parallel"],
                           embedding_layers=ac_cfg["embedding_layers"],
                           hidden_layers=ac_cfg["hidden_layers"])
        psi_a = TrainState.create(
            psi_a_def,
            psi_a_def.init(jax.random.fold_in(rng, 101), ex_observations, ex_w, ex_actions)["params"],
            tx=optax.adam(ac_cfg["lr"]))
        residual_def = NoiseConditionedActor(
            action_dim=action_dim, hidden_dim=config["residual"]["hidden_dim"],
            hidden_layers=config["residual"]["hidden_layers"],
            embedding_layers=config["residual"]["embedding_layers"])
        residual = TrainState.create(
            residual_def,
            residual_def.init(jax.random.fold_in(rng, 102), ex_observations, ex_w, ex_u)["params"],
            tx=optax.adam(ac_cfg["lr"]))
        # B_a: same class and shape as the shared basis (BackwardMap IS PhiMap in this
        # codebase), but a separate parameter set with its own optimizer and target.
        phi_a_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                           hidden_layers=config["phi"]["hidden_layers"], norm=True)
        phi_a = TrainState.create(
            phi_a_def,
            phi_a_def.init(jax.random.fold_in(rng, 105), ex_observations)["params"],
            tx=optax.adam(ac_cfg["lr"]))

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        return cls(rng=rng, phi=phi, psi=psi,
                   target_phi=copy.deepcopy(phi.params), target_psi=copy.deepcopy(psi.params),
                   actor=actor, actor_vf=actor_vf,
                   psi_a=psi_a, target_psi_a=copy.deepcopy(psi_a.params), residual=residual,
                   phi_a=phi_a, target_phi_a=copy.deepcopy(phi_a.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   task_z=jnp.zeros((z_dim,), jnp.float32),
                   task_z_a=jnp.zeros((z_dim,), jnp.float32),
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
            # Fraction of TD bootstrap latents drawn from the prior instead of the actor.
            # 0.0 reproduces the as-published backup exactly.
            backup_explore_frac=0.0,
            # amortized latent actor (flowBC recipe over latents). Load-bearing: the
            # measure backup bootstraps its latent at s'; always trained.
            actor=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                       vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10,
                       bc_coeff=1.0),
            acting="actor",          # actor | gpi (per-step latent argmax, actor-free)
            # psi's index slot. "task_vector" is the shipped agent (psi(s, w, u)).
            # "latent" is the write-up's psi(s, u, u'): a fresh u' ~ p0 per batch element
            # indexes the policy, the backup continues that SAME u' at s', and w reaches
            # psi only through the readout Q = psi^T w.
            policy_index="task_vector",   # task_vector | latent
            # Train the amortized latent actor. False drops the actor/CFM branch entirely
            # (the write-up's Sec. PSMFlows has none); only valid with acting=gpi, and only
            # sensible with policy_index="latent", where the backup does not read it.
            train_actor=True,
            # Idea-1 action branch: w-conditioned successor features over EXECUTED
            # actions + eps-bounded residual. Off by default; enabling changes only the
            # new branch and the acting path (shared psi/phi/actor updates are untouched).
            # `pessimism` is a BLEND WEIGHT in [0, 1] between the target ensemble's mean
            # and its least-task-valued member (scalar-Q pessimism); it is not a
            # multiplier on a per-feature spread. See action_critic_loss.
            action_critic=dict(enabled=False, discount=0.99, tau=0.005, pessimism=0.0,
                               lr=3.0e-4, hidden_dim=512, hidden_layers=1,
                               embedding_layers=2, spread_candidates=16,
                               eval_rank_k=0,
                               # P2 graft: psi_a written against its OWN backward map B_a,
                               # trained by the FB measure loss; w_a inferred from B_a.
                               fb_graft=False, ortho_coef=1000.0, b_tau=0.005),
            residual_eps=0.05,       # hard per-dim budget on |a_exec - decode|
            residual=dict(hidden_dim=256, hidden_layers=2, embedding_layers=2),
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
