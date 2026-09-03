"""Affine (full) PSM agent — successor measure M(s,a,x) = Phi(s,a,x)·w + b(s,a,x).

The affine decomposition (RLU controllable_agent, and arXiv 2411.19418) makes the task
coordinate `w` enter LINEARLY, which is what makes the constrained-LP `full` inference
well-defined — the distinguishing feature vs the bilinear agents/psm.py.

Networks live in utils/psm_networks.py; this agent holds only losses and orchestration.
Phi and w are sqrt(d)-normalized and b is tanh-bounded so the TD bootstrap cannot diverge.
Inference is `full` (LP) or `zero_shot` (closed form).
See docs/design/2026-07-22-affine-psm-design.md.
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.psm_networks import (AffineMeasureNet, FactoredAffineMeasureNet, FlowVectorField,
                                LagrangeNet, NoiseConditionedActor, PSMActor, WNet)
from agents.psm import proto_sample, project_z, off_diagonal_mask, polyak_update, ortho_loss, _plain_config


@flax.struct.dataclass
class StepInputs:
    """Per-step quantities sampled once and shared across the loss branches."""
    proto_seed: Any             # (B, max_log_seed) binary codebook code z for the proto branch
    proto_next_action: Any      # (B, action_dim) codebook policy action pi_z(s') at s'
    actor_goal: Any = None      # (ob_dim,) a future state used as the goal for the amortized actor stage
    flow_x0: Any = None         # (B, action_dim) flow-matching source sample u0 ~ N(0,I)   [actor.type=flow]
    flow_t: Any = None          # (B, 1) flow-matching time t ~ U[0,1)                     [actor.type=flow]
    flow_noise: Any = None      # (B, action_dim) noise fed to the one-step noise actor     [actor.type=flow]


class AffinePSMAgent(flax.struct.PyTreeNode):
    """PSM agent holding one TrainState per network + soft-updated target param trees.

    All networks are defined in utils/psm_networks.py; this class owns only the losses,
    the jitted update orchestration, and the eval-time inference. Following the repo
    convention (agents/psm.py), each network is its own TrainState (module + optimizer).
    """

    rng: Any                   # PRNGKey, split each update step
    measure: TrainState        # AffineMeasureNet: (s,a,x) -> (Phi in R^d, b in R)
    w: TrainState              # WNet: codebook code z -> task coordinate w in R^d
    target_measure: Any        # soft-updated copy of measure.params (TD bootstrap target)
    target_w: Any              # soft-updated copy of w.params
    w_inf: Any                 # (d_dim,) task coordinate solved at eval (LP or closed-form)
    actor: TrainState          # pi(s, w): AMORTIZED in-loop (see actor_loss), not distilled
    config: Any = nonpytree_field()   # plain hashable config (jit static aux)
    actor_vf: Any = None       # TrainState for the flow actor's velocity field; None for ddpgbc
    proto: Any = None          # (seed_to_action table, bit powers) for the hash codebook policy
    task_goal: Any = None      # (ob_dim,) goal fixed at inference; retained for diagnostics

    def sample_step_inputs(self, batch, rng):
        """Sample the per-step quantities (codebook code, its next-action, actor goal)."""
        c = self.config
        B = batch["observations"].shape[0]
        r_seed, r_goal, r_flow = jax.random.split(rng, 3)      # keys: codebook code, goal, flow draws
        # --- codebook codes z: B distinct codes, one per row, so each step trains the
        # basis on B proto-policies. Each code is tied to the source/row index i, so row
        # i's target uses pi_{z_i}(s'_i). Tying z to the goal/column instead misaligns. ---
        n_bits = c["max_log_seed"]                             # binary code width
        codes = jax.random.randint(r_seed, (B,), 0, 2 ** n_bits)         # B integer seeds
        bits = ((codes[:, None] >> jnp.arange(n_bits)) & 1).astype(jnp.float32)  # LSB-first
        proto_seed = bits                                      # (B, n_bits), row-aligned
        seed_to_action, powers = self.proto                    # the frozen hash-codebook lookup table
        obs_hash = batch["index"] if "index" in batch else jnp.arange(B)   # key on global row index if present
        # pi_z(s') = table[(z·powers + hash(s')) mod max_seed]: the codebook action at each s'.
        proto_next_action = proto_sample(seed_to_action, powers, obs_hash, proto_seed, c["proto_max_seed"])
        # --- amortized-actor goal: a random future state. Over training this covers the
        # goal distribution, so the w-conditioned actor learns to reach any goal. ---
        gidx = jax.random.randint(r_goal, (), 0, B)
        actor_goal = batch["next_observations"][gidx]
        if c["actor_type"] != "flow":
            return StepInputs(proto_seed=proto_seed, proto_next_action=proto_next_action,
                              actor_goal=actor_goal)
        # --- flow-actor draws: the flow-matching pair (u0, t) for the velocity field, and
        # the noise for the one-step actor (mirrors agents/psm.py). ---
        adim = c["action_dim"]
        r_x0, r_t, r_noise = jax.random.split(r_flow, 3)
        return StepInputs(proto_seed=proto_seed, proto_next_action=proto_next_action,
                          actor_goal=actor_goal,
                          flow_x0=jax.random.normal(r_x0, (B, adim)),
                          flow_t=jax.random.uniform(r_t, (B, 1)),
                          flow_noise=jax.random.normal(r_noise, (B, adim)))

    # ---- action-slot seams ----
    # The measure, actor and codebook all read one array per transition: whatever fills
    # M(s, ., x)'s middle slot. Here that is the dataset action. The latent subclass
    # (agents/latent_affine_psm.py) serves a flow preimage latent and decodes it only at
    # act time, so nothing below needs to know which space it is in.

    def measure_action(self, batch):
        """The array filling the measure's action slot for this batch."""
        return batch["actions"]

    def action_scale(self):
        """Multiplier on the actor's tanh output; 1.0 keeps actions in [-1, 1]."""
        return 1.0

    def to_env_action(self, observations, a):
        """Environment action for a slot value. Identity (plus the clip the acting path
        already applied) in action space."""
        return jnp.clip(a, -1.0, 1.0)

    @staticmethod
    def codebook_table(rng, max_seed, dim, config):
        """Hash-codebook lookup table: RLU draws actions uniform in [-2, 0)."""
        return (jax.random.uniform(rng, (max_seed, dim)) - 1.0) * 2.0

    def measure_loss(self, batch, sampled, measure_params, w_params):
        """PSM contrastive TD loss on the affine measure M(s,a,x) = Phi(s,a,x)·w(z) + b.
        Cor. 4.2 measure, fit by Eq. 7 under E_z (Eq. 9).

        Reproduces RLU-continuous `update_psm`: a squared off-diagonal TD residual plus a
        diagonal source term, over the B^2 grid of (source (s_i,a_i), measure-arg x_j) pairs.
        `measure_params`/`w_params` flow gradients; the target nets are stop-gradient.
        """
        c = self.config
        obs, next_obs = batch["observations"], batch["next_observations"]
        action = self.measure_action(batch)
        x = next_obs                              # the measure argument (a future state s')
        B = obs.shape[0]
        off, off_sum = off_diagonal_mask(B)       # (1 - I) mask and its sum, for the off-diagonal term
        z, proto_na = sampled.proto_seed, sampled.proto_next_action   # codebook code + its next action

        wz = project_z(self.w(z, params=w_params), True)     # (B, d); rows identical (one code)
        twz = project_z(self.w(z, params=self.target_w), True)

        if c["measure_factored"]:
            # --- FACTORIZED mesh (Thm 6.3): Phi = A(s,a) phi_x(x), so
            #       M_ij = phi_x(x_j)·(A(s_i,a_i)^T w) + bound(beta_i·phi_x(x_j))
            # B evals per tower plus two (B,B) matmuls, instead of B^2 evals. That is what
            # makes batch_size=1024 affordable: the unfactored path is ~1.1 s/step vs a few ms.
            def mesh(o, a, xs, mp, wv):
                A, beta, px, pb = self.measure.model_def.apply(
                    {"params": mp}, o, a, xs, method="mesh_terms")   # (B,d,k),(B,k),(B,k),(B,k)
                psi = jnp.einsum("bdk,bd->bk", A, wv)                # A^T w, the psi side
                raw_b = beta @ pb.T                                  # (B,B) offset pre-bound
                bs = float(c["measure"]["b_scale"])
                return psi @ px.T + (bs * jnp.tanh(raw_b) if bs > 0 else raw_b), px

            M, phi_basis = mesh(obs, action, x, measure_params, wz)
            target_M, _ = mesh(next_obs, proto_na, x, self.target_measure, twz)
        else:
            # --- naive mesh: B^2 evaluations of the joint (s,a,x) trunk ---
            i_idx = jnp.repeat(jnp.arange(B), B)   # (B^2,) 0,0,..,0,1,1,.. (source index)
            j_idx = jnp.tile(jnp.arange(B), B)     # (B^2,) 0,1,..,B-1,0,1,.. (measure-arg index)
            m_obs, m_act = obs[i_idx], action[i_idx]                    # source (s_i, a_i)
            m_next_obs, m_next_act = next_obs[i_idx], proto_na[i_idx]   # bootstrap (s'_i, pi_z(s'_i))
            m_x = x[j_idx]                                             # measure argument x_j
            # w and Phi are both sqrt(d)-normalized, so M is bounded and the TD bootstrap
            # cannot diverge.
            phi, b = self.measure(m_obs, m_act, m_x, params=measure_params)   # (B^2,d), (B^2,1)
            M = ((phi * wz[i_idx]).sum(-1, keepdims=True) + b).reshape(B, B)
            tphi, tb = self.measure(m_next_obs, m_next_act, m_x, params=self.target_measure)
            target_M = ((tphi * twz[i_idx]).sum(-1, keepdims=True) + tb).reshape(B, B)
            phi_basis = None
        target_M = jax.lax.stop_gradient(target_M)           # no gradient through the target

        # --- contrastive loss = off-diagonal TD residual + diagonal source term (Eq. 7) ---
        diff = M - c["discount"] * target_M                  # Bellman residual (B,B)
        offdiag = 0.5 * jnp.sum((diff * off) ** 2) / off_sum  # mean squared residual on s' != s
        diag = -((1 - c["discount"]) * jnp.diagonal(M)).mean()   # RLU psm.py:352 source term; App. B.3.2
        loss = offdiag + diag
        info = {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag}

        # --- orthonormality regularizer on Phi ---
        # On sqrt(d)-normalized Phi the diagonal term is inert, so a large ortho_coef acts
        # as a pure decorrelator (Phi Phi^T -> I) and stops rank collapse.
        if c["ortho_coef"] > 0:
            if c["measure_factored"]:
                # Regularize the x-side basis phi_x: it is the sqrt(k)-normalized piece.
                # On the unnormalized product Phi the diagonal term goes live and
                # ortho_coef=1000 shrinks norms instead of decorrelating.
                phi_batch = phi_basis
            else:
                phi_batch, _ = self.measure(obs, action, x, params=measure_params)   # (B,d)
            ol, oldiag, oloff = ortho_loss(phi_batch, off, off_sum)
            loss = loss + c["ortho_coef"] * ol
            info.update({"psm_loss": loss, "orth_loss": ol, "orth_diag": oldiag, "orth_offdiag": oloff})
        return loss, info

    def _goal_task_coord(self, batch, sampled):
        """(g_rep, w_g) for this step's actor goal: the closed-form task coordinate w_g for
        goal g is the mean basis response over dataset (s,a) toward g, sqrt(d)-normalized —
        i.e. the zero_shot inference, batch-estimated. The measure is frozen here (no grad
        flows to it from the actor stage), so both actor branches share this."""
        obs, action = batch["observations"], self.measure_action(batch)
        B = obs.shape[0]
        g = sampled.actor_goal                               # this step's goal (a future state)
        g_rep = jnp.broadcast_to(g, (B,) + g.shape)          # (B, ob_dim) broadcast to the batch
        phi_toward_g, _ = self.measure(obs, action, g_rep)   # (B, d)
        return g_rep, project_z(phi_toward_g.mean(0), True)  # (B, ob_dim), (d,)

    def _goal_q(self, obs, a, g_rep, w_g):
        """Q(s,a) for reaching g under the frozen measure: Phi(s,a,g)·w_g + b(s,a,g)."""
        phi_a, b_a = self.measure(obs, a, g_rep)
        return (phi_a * w_g).sum(-1) + b_a.squeeze(-1)

    def _bc_coeff(self):
        c = self.config["actor"]
        return c.get("bc_coeff", 0.0) if hasattr(c, "get") else c["bc_coeff"]

    def actor_loss(self, batch, sampled, actor_params):
        """Amortized (in-loop) w-conditioned DDPGBC actor, FB-style. For a sampled goal g,
        form its closed-form task coord w_g and train pi(s, w_g) to maximize the measure's
        Q = Phi(s,a,g)·w_g + b(s,a,g). Over training this amortizes goal-reaching across the
        whole goal/w distribution (vs the paper's from-scratch eval-time distillation)."""
        c = self.config
        obs, action = batch["observations"], self.measure_action(batch)
        B = obs.shape[0]
        g_rep, w_g = self._goal_task_coord(batch, sampled)
        w_rep = jnp.broadcast_to(w_g, (B, c["d_dim"]))       # condition the actor on w_g
        # tanh mean in [-1,1], rescaled to the slot's box (a no-op at scale 1.0).
        a = self.action_scale() * self.actor(obs, w_rep, params=actor_params)
        Q = self._goal_q(obs, a, g_rep, w_g)
        loss = -Q.mean()                                     # maximize Q -> deterministic policy gradient
        info = {"actor_loss": loss, "actor_q": Q.mean()}
        bc_coeff = self._bc_coeff()
        if bc_coeff > 0:
            bc = jnp.mean((a - action) ** 2)                 # keep the actor near dataset actions (in-support)
            # normalize the Q term by |Q| so the BC weight has a stable relative scale (FB/psm).
            loss = loss / jax.lax.stop_gradient(jnp.abs(Q).mean() + 1e-6) + bc_coeff * bc
            info = {"actor_loss": loss, "actor_q": Q.mean(), "actor_bc": bc}
        return loss, info

    def flow_actor_loss(self, batch, sampled, actor_params, vf_params):
        """Flow (FQL-style) actor, the reference's cube/manipulation recipe (Factored-FB
        psm_flowbc): a BC flow-matching velocity field v(s, x_t, t) trained on dataset
        actions, plus a one-step w-conditioned noise actor distilled from its ODE rollout.

        The distillation target is a SAMPLE from the behavior flow rather than the dataset
        action's conditional mean, which is why this is the right actor on multimodal
        manipulation data: a tanh mean fit by MSE regresses to the average of the modes.
        Q is the same frozen-measure goal Q as `actor_loss`. Differentiated w.r.t.
        (actor_params, vf_params)."""
        c = self.config
        obs, action = batch["observations"], self.measure_action(batch)
        B = obs.shape[0]
        x0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
        bc_coeff, steps = self._bc_coeff(), c["flow_steps"]
        scale = self.action_scale()

        def rollout(vf_p, o, n):
            """Euler-integrate the velocity field from noise n to an action (t: 0 -> 1)."""
            a = n
            for i in range(steps):
                ti = jnp.full((o.shape[0], 1), i / steps)
                a = a + self.actor_vf(o, a, ti, params=vf_p) / steps
            return jnp.clip(a, -scale, scale)

        # --- conditional flow matching: regress v(s, x_t, t) onto the straight-line velocity ---
        xt = (1 - t) * x0 + t * action                       # interpolant between noise and data action
        pred = self.actor_vf(obs, xt, t, params=vf_params)
        bc_flow_loss = jnp.mean((pred - (action - x0)) ** 2)

        # --- one-step w-conditioned noise actor, scored by the frozen measure's goal Q ---
        g_rep, w_g = self._goal_task_coord(batch, sampled)
        w_rep = jnp.broadcast_to(w_g, (B, c["d_dim"]))
        a = scale * self.actor(obs, w_rep, noise, params=actor_params)  # tanh, rescaled to the slot box
        Q = self._goal_q(obs, a, g_rep, w_g)
        q_loss = -Q.mean()
        info = {"actor_q": Q.mean(), "bc_flow_loss": bc_flow_loss}
        if bc_coeff > 0:
            # Distill the stop-gradient ODE rollout: same noise in, so the one-step actor
            # learns the flow's noise->action map, then bends it toward high Q.
            target = jax.lax.stop_gradient(rollout(vf_params, obs, noise))
            distill = jnp.mean((a - target) ** 2)
            q_loss = q_loss / jax.lax.stop_gradient(jnp.abs(Q).mean() + 1e-6) + bc_coeff * distill
            info["actor_bc"] = distill
        loss = q_loss + bc_flow_loss
        info["actor_loss"] = loss
        return loss, info

    def apply_update(self, batch, sampled):
        """One gradient step for all three networks: measure + w, then the amortized actor."""
        tau = self.config["tau"]
        # --- stage 1: measure + w (the successor-measure representation) ---
        (_, info), (g_m, g_w) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.measure.params, self.w.params)
        measure = self.measure.apply_gradients(grads=g_m)
        w = self.w.apply_gradients(grads=g_w)
        target_measure = polyak_update(measure.params, self.target_measure, tau)   # soft-update targets
        target_w = polyak_update(w.params, self.target_w, tau)
        # --- stage 2: amortized actor (trained against the measure's Q; measure is frozen inside) ---
        actor_vf = self.actor_vf
        if self.config["actor_type"] == "flow":
            (_, a_info), (g_actor, g_vf) = jax.value_and_grad(
                self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
                batch, sampled, self.actor.params, self.actor_vf.params)
            actor = self.actor.apply_gradients(grads=g_actor)
            actor_vf = self.actor_vf.apply_gradients(grads=g_vf)
        else:
            (_, a_info), g_actor = jax.value_and_grad(self.actor_loss, argnums=2, has_aux=True)(
                batch, sampled, self.actor.params)
            actor = self.actor.apply_gradients(grads=g_actor)
        return self.replace(measure=measure, w=w, actor=actor, actor_vf=actor_vf,
                            target_measure=target_measure,
                            target_w=target_w), {**info, **a_info}

    @jax.jit
    def update(self, batch):
        """Jitted training step: split rng, sample step inputs, apply the gradient update."""
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.measure.params, self.w.params)
        if self.config["actor_type"] == "flow":
            _, a_info = self.flow_actor_loss(batch, sampled, self.actor.params, self.actor_vf.params)
        else:
            _, a_info = self.actor_loss(batch, sampled, self.actor.params)
        return loss, {**info, **a_info}

    # ---- inference ----
    def infer_w_goal(self, dataset, goal, seed=0):
        """Solve w_inf by constrained optimization (Eq. 10; RLU infer_w_goal/_infer_step_gc).
        Objective: maximize measure at `goal`; constraint: Phi·w+b >= 0 off-goal.
        Returns a new agent with w_inf set (Python loop; jitted inner step)."""
        c = self.config
        ic = c["inference"]
        d = c["d_dim"]
        goal = jnp.asarray(goal, jnp.float32)

        w_inf = jnp.ones((d,), jnp.float32)
        w_opt = optax.adam(c["lr_w"])
        w_state = w_opt.init(w_inf)

        use_dgd = bool(ic["use_dgd"])
        inf_coeff = float(ic["inf_coeff"])
        lam_def = LagrangeNet(hidden_dim=ic["lagrange_hidden_dim"],
                              hidden_layers=ic["lagrange_hidden_layers"])
        ex = dataset.sample(c["batch_size"])
        lam_params = lam_def.init(jax.random.PRNGKey(seed),
                                  jnp.asarray(ex["observations"]), jnp.asarray(self.measure_action(ex)),
                                  jnp.asarray(ex["next_observations"]))["params"]
        lam_opt = optax.adam(c["lr_w"])
        lam_state = lam_opt.init(lam_params)

        @jax.jit
        def step(w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep):
            def primal(w_raw):
                # Re-project onto the sqrt(d) sphere at every use. Without it the objective
                # -mean(phi_g·w) is linear in w and unbounded below; only the step budget
                # kept it finite.
                w = project_z(w_raw, True)
                phi_g, _ = self.measure(obs, act, goal_rep)
                phi_p, b_p = self.measure(obs, act, xperm)
                obj = -(phi_g * w).sum(-1).mean()
                Mp = (phi_p * w).sum(-1, keepdims=True) + b_p
                if use_dgd:
                    lam = jax.lax.stop_gradient(lam_def.apply({"params": lam_params}, obs, act, xperm))
                    con = -(Mp * lam).mean()
                else:
                    con = -(jnp.minimum(Mp, 0.0) * inf_coeff).mean()
                return obj + con, (obj, con)

            (_, (obj, con)), gw = jax.value_and_grad(primal, has_aux=True)(w_inf)
            upd, w_state = w_opt.update(gw, w_state, w_inf)
            w_inf = optax.apply_updates(w_inf, upd)
            if use_dgd:
                def dual(lp):
                    phi_p, b_p = self.measure(obs, act, xperm)
                    Mp = (phi_p * jax.lax.stop_gradient(project_z(w_inf, True))).sum(-1, keepdims=True) + b_p
                    lam = lam_def.apply({"params": lp}, obs, act, xperm)
                    return (Mp * lam).mean()

                # RLU minimizes (M*lam) w.r.t. lam; against the primal penalty -(M*lam) that
                # grows lam where M<0. optax minimizes, so feed the raw grad: negating it
                # inverts constraint enforcement.
                gl = jax.grad(dual)(lam_params)
                ul, lam_state = lam_opt.update(gl, lam_state, lam_params)
                lam_params = optax.apply_updates(lam_params, ul)
            return w_inf, w_state, lam_params, lam_state, obj, con

        # Seed the permutation: default_rng() with no argument draws OS entropy, so equal
        # seeds and weights gave different constraint sets xperm, and different w_inf.
        perm_rng = np.random.default_rng(seed)
        for _ in range(int(ic["num_inference_steps"])):
            b = dataset.sample(c["batch_size"])
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(self.measure_action(b))
            xb = jnp.asarray(b["next_observations"])
            perm = perm_rng.permutation(obs.shape[0])
            xperm = xb[perm]
            goal_rep = jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape)
            w_inf, w_state, lam_params, lam_state, _, _ = step(
                w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep)

        if bool(ic["norm_w"]):
            w_inf = project_z(w_inf, True)
        # The amortized actor was trained on the closed-form w_g only. The LP w_inf is a
        # different distribution, so fine-tune the actor against the w_inf just solved for.
        return self.replace(w_inf=w_inf, task_goal=goal).distill_actor(dataset, goal)

    def distill_actor(self, dataset, goal, steps=None):
        """Fine-tune the amortized actor against THIS agent's w_inf and goal.

        Starts from the in-loop amortized weights (so it converges in far fewer steps than
        RLU's from-scratch distill_actor_ddpg) and maximizes the same frozen-measure goal Q
        the training-time actor loss uses, so the objective is identical apart from w."""
        c = self.config
        ic = c["inference"]
        n = int(ic["num_actor_inference_steps"]) if steps is None else int(steps)
        if n <= 0:
            return self
        goal = jnp.asarray(goal, jnp.float32)
        flow = c["actor_type"] == "flow"

        @jax.jit
        def step(actor, rng, obs, action):
            B = obs.shape[0]
            g_rep = jnp.broadcast_to(goal, (B,) + goal.shape)
            w_rep = jnp.broadcast_to(self.w_inf, (B, c["d_dim"]))

            def loss_fn(params):
                if flow:
                    noise = jax.random.normal(rng, (B, c["action_dim"]))
                    a = self.action_scale() * self.actor(obs, w_rep, noise, params=params)
                else:
                    a = self.action_scale() * self.actor(obs, w_rep, params=params)
                Q = self._goal_q(obs, a, g_rep, self.w_inf)
                loss = -Q.mean()
                bc_coeff = self._bc_coeff()
                if bc_coeff > 0:   # keep it in-support, as the training-time loss does
                    loss = (loss / jax.lax.stop_gradient(jnp.abs(Q).mean() + 1e-6)
                            + bc_coeff * jnp.mean((a - action) ** 2))
                return loss, Q.mean()

            (_, q), g = jax.value_and_grad(loss_fn, has_aux=True)(actor.params)
            return actor.apply_gradients(grads=g), q

        actor, rng = self.actor, self.rng
        for _ in range(n):
            b = dataset.sample(c["batch_size"])
            rng, k = jax.random.split(rng)
            actor, _ = step(actor, k, jnp.asarray(b["observations"]), jnp.asarray(self.measure_action(b)))
        return self.replace(actor=actor, rng=rng)

    def infer_w_zeroshot(self, dataset, goal, num_samples=4096):
        """Closed-form goal code (not Eq. 10; FB get_goal_meta analogue): w = sqrt(d)*normalize(
        E_{(s,a)~D}[Phi(s,a,goal)]) — the affine-net analogue of FB get_goal_meta."""
        c = self.config
        goal = jnp.asarray(goal, jnp.float32)
        # Full-array reduction when the dataset exposes raw arrays (test fakes). The
        # production ReplayBuffer has no `.obs`, so it samples.
        if hasattr(dataset, "obs") and hasattr(dataset, "act") and num_samples >= getattr(dataset, "n", 0):
            obs = jnp.asarray(dataset.obs); act = jnp.asarray(dataset.act)
        else:
            b = dataset.sample(num_samples)
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(self.measure_action(b))
        goal_rep = jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape)
        phi, _ = self.measure(obs, act, goal_rep)
        w = phi.mean(0)
        if bool(c["inference"]["norm_w"]):
            w = project_z(w, True)
        return self.replace(w_inf=w, task_goal=goal)

    def infer_eval(self, dataset, goal):
        """Dispatch on config inference.mode: 'full' (constrained) or 'zero_shot'."""
        mode = self.config["inference"]["mode"]
        if mode == "zero_shot":
            return self.infer_w_zeroshot(dataset, goal)
        if mode == "full":
            return self.infer_w_goal(dataset, goal)
        raise ValueError(f"unknown inference.mode {mode!r}")

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        # The actor is amortized and conditioned on w. infer_eval sets w_inf (LP or closed
        # form); the actor acts greedily for it.
        w = jnp.broadcast_to(self.w_inf, (*observations.shape[:-1], self.config["d_dim"]))
        if self.config["actor_type"] == "flow":
            # One-step noise actor: draw the flow latent, decode, clip (mirrors agents/psm.py).
            seed = self.rng if seed is None else seed
            noise = jax.random.normal(seed, (*observations.shape[:-1], self.config["action_dim"]))
            return self.to_env_action(observations, self.action_scale() * self.actor(observations, w, noise))
        return self.to_env_action(observations, self.action_scale() * self.actor(observations, w))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Build the agent: instantiate the four networks + their TrainStates, the hash
        codebook table, and the target param trees, from example observations/actions."""
        rng = jax.random.PRNGKey(seed)
        rng, rm, rw, ract, rproto = jax.random.split(rng, 5)   # keys: measure, w, actor, proto
        assert config.get("encoder", None) is None, "affine_psm does not support visual encoders."
        action_dim = ex_actions.shape[-1]
        d_dim, z_dim = config["d_dim"], config["z_dim"]
        ex_obs, ex_act = ex_observations, ex_actions
        ex_x = ex_obs                                        # example measure argument (a state)
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))           # actor task coord w placeholder (z_dim == d_dim)
        ex_zbin = jnp.zeros((ex_obs.shape[0], config["max_log_seed"]))   # binary codebook code -> WNet

        # measure net: (s,a,x) -> (Phi in R^d, b in R), b tanh-bounded.
        mcfg = config["measure"]
        factored = bool(mcfg["factored"]) if "factored" in mcfg else False
        if factored:
            # Factorized basis Phi = A(s,a) phi_x(x): the B^2 mesh becomes B evals + two
            # matmuls, which makes batch_size=1024 affordable.
            measure_def = FactoredAffineMeasureNet(
                d_dim=d_dim, k_dim=int(mcfg["k_dim"]) if "k_dim" in mcfg else 0,
                hidden_dim=mcfg["hidden_dim"], hidden_layers=mcfg["hidden_layers"],
                b_scale=float(mcfg.get("b_scale", 10.0)))
        else:
            measure_def = AffineMeasureNet(d_dim=d_dim, hidden_dim=mcfg["hidden_dim"],
                                           hidden_layers=mcfg["hidden_layers"],
                                           b_scale=float(mcfg.get("b_scale", 10.0)))
        measure = TrainState.create(measure_def, measure_def.init(rm, ex_obs, ex_act, ex_x)["params"],
                                    tx=optax.adam(config["lr"]))

        # w net: binary codebook code (max_log_seed bits) -> d-dim task coordinate (RLU psm.py self.w).
        w_def = WNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"])
        w = TrainState.create(w_def, w_def.init(rw, ex_zbin)["params"], tx=optax.adam(config["lr_w"]))

        # actor net: pi(s, w) -> action; amortized in the main loop (see actor_loss).
        actor_cfg = config["actor"]
        actor_type = actor_cfg["type"] if "type" in actor_cfg else "ddpgbc"
        assert actor_type in ("ddpgbc", "flow"), f"unknown actor.type {actor_type!r}"
        # actor LR defaults to the measure LR; set it lower (lr_actor) for a two-timescale
        # scheme, with a slow actor tracking a quasi-static Q.
        lr_actor = float(config.get("lr_actor", config["lr"]))
        actor_vf = None
        if actor_type == "flow":
            # Flow actor (reference psm_flowbc): a one-step NoiseConditionedActor
            # a = tanh(net(s, w, noise)), distilled from a GELU velocity field v(s, x_t, t).
            def _acfg(key, default):
                return actor_cfg[key] if key in actor_cfg else default
            actor_def = NoiseConditionedActor(
                action_dim=action_dim, hidden_dim=int(_acfg("flow_actor_hidden_dim", 512)),
                hidden_layers=int(_acfg("flow_actor_hidden_layers", 2)),
                embedding_layers=int(_acfg("flow_actor_embedding_layers", 2)))
            vf_def = FlowVectorField(action_dim=action_dim,
                                     hidden_dim=int(_acfg("flow_vf_hidden_dim", 512)),
                                     hidden_layers=int(_acfg("flow_vf_hidden_layers", 4)))
            ract, rvf = jax.random.split(ract)
            ex_noise = ex_act
            ex_times = ex_act[..., :1]
            actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z, ex_noise)["params"],
                                      tx=optax.adam(lr_actor))
            actor_vf = TrainState.create(vf_def, vf_def.init(rvf, ex_obs, ex_act, ex_times)["params"],
                                         tx=optax.adam(float(_acfg("lr_actor_vf", 3e-4))))
        else:
            actor_def = PSMActor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                                 embedding_layers=actor_cfg["embedding_layers"],
                                 hidden_layers=actor_cfg["hidden_layers"])
            actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z)["params"],
                                      tx=optax.adam(lr_actor))

        # hash-codebook table: max_seed rows of actions in [-2, 0); pi_z looks one up per (z, s).
        max_seed = 2 ** config["max_log_seed"] + 20000
        table = cls.codebook_table(rproto, max_seed, action_dim, config)
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)   # bit -> integer weights
        proto = (table.astype(jnp.float32), powers)

        # freeze the config into a plain hashable dict so it can serve as the jit static aux.
        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["proto_max_seed"] = max_seed
        config["actor_type"] = actor_type
        config["measure_factored"] = factored
        config["flow_steps"] = int(actor_cfg["flow_steps"]) if "flow_steps" in actor_cfg else 10
        return cls(rng=rng, measure=measure, w=w,
                   target_measure=copy.deepcopy(measure.params),   # targets start equal to online params
                   target_w=copy.deepcopy(w.params),
                   w_inf=jnp.ones((d_dim,), jnp.float32), actor=actor, actor_vf=actor_vf,
                   config=flax.core.FrozenDict(config), proto=proto)
