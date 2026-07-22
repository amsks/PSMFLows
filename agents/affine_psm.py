"""Affine (full) PSM agent — M(s,a,x) = Phi(s,a,x)·w + b(s,a,x).

Faithful JAX port of RLU controllable_agent/url_benchmark/agent/psm.py (continuous)
+ discrete_psm.py `_infer_step`. Distinct from the bilinear agents/psm.py: the task
coordinate w enters LINEARLY, which is what makes the constrained-LP `full` inference
well-defined. See docs/design/2026-07-22-affine-psm-design.md.
"""
import copy
from typing import Any

import flax
import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.psm_networks import AffineMeasureNet, LagrangeNet, PSMActor
from agents.psm import proto_sample, project_z, off_diagonal_mask, polyak_update, ortho_loss, _plain_config


class _WNet(fnn.Module):
    """z (codebook code) -> w (task coordinates), an MLP (RLU psm.py self.w)."""
    d_dim: int
    hidden_dim: int

    @fnn.compact
    def __call__(self, z):
        h = z
        for _ in range(3):
            h = fnn.relu(fnn.Dense(self.hidden_dim)(h))
        return fnn.Dense(self.d_dim)(h)


@flax.struct.dataclass
class StepInputs:
    proto_seed: Any
    proto_next_action: Any
    actor_goal: Any = None      # a future state used as the goal for the amortized actor stage


class AffinePSMAgent(flax.struct.PyTreeNode):
    rng: Any
    measure: TrainState
    w: TrainState
    target_measure: Any
    target_w: Any
    w_inf: Any                 # (d_dim,) inference task coordinate, solved at eval
    actor: TrainState          # PSMActor distilled against Q = Phi(s,a,goal)·w_inf + b
    config: Any = nonpytree_field()
    proto: Any = None          # (seed_to_action, powers)
    task_goal: Any = None      # (ob_dim,) goal state fixed at inference; the measure arg x

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        r_seed, r_goal = jax.random.split(rng)
        # ONE binary codebook code per batch (a simplification; RLU-continuous samples B
        # distinct codes per batch — a known reduction in codebook diversity, not fidelity).
        w = c["max_log_seed"]
        code = jax.random.randint(r_seed, (), 0, 2 ** w)
        bits = ((code >> jnp.arange(w)) & 1).astype(jnp.float32)
        proto_seed = jnp.broadcast_to(bits, (B, w))
        seed_to_action, powers = self.proto
        obs_hash = batch["index"] if "index" in batch else jnp.arange(B)
        proto_next_action = proto_sample(seed_to_action, powers, obs_hash, proto_seed, c["proto_max_seed"])
        # Amortized-actor goal: a random future state from the batch. Over training this
        # covers the goal distribution, so the w-conditioned actor learns to reach any goal.
        gidx = jax.random.randint(r_goal, (), 0, B)
        actor_goal = batch["next_observations"][gidx]
        return StepInputs(proto_seed=proto_seed, proto_next_action=proto_next_action, actor_goal=actor_goal)

    def measure_loss(self, batch, sampled, measure_params, w_params):
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        x = next_obs                      # measure argument (future state)
        B = obs.shape[0]
        off, off_sum = off_diagonal_mask(B)
        z, proto_na = sampled.proto_seed, sampled.proto_next_action

        # B^2 mesh over (s_i, a_i, x_j): i indexes the source (s,a), j the measure arg x.
        i_idx = jnp.repeat(jnp.arange(B), B)     # (B^2,)
        j_idx = jnp.tile(jnp.arange(B), B)       # (B^2,)
        m_obs, m_act = obs[i_idx], action[i_idx]
        m_next_obs, m_next_act = next_obs[i_idx], proto_na[i_idx]
        m_x = x[j_idx]

        # Normalize w to the sqrt(d)-sphere (project_z), matching the bilinear task-vector
        # normalization. With ||phi||=sqrt(d) too, the measure Phi·w+b is bounded, so the
        # TD bootstrap target cannot diverge (the 108-spike-then-collapse failure).
        wz = project_z(self.w(z, params=w_params), True)   # (B, d) — identical rows
        wz_i = wz[i_idx]
        phi, b = self.measure(m_obs, m_act, m_x, params=measure_params)
        M = ((phi * wz_i).sum(-1, keepdims=True) + b).reshape(B, B)

        tphi, tb = self.measure(m_next_obs, m_next_act, m_x, params=self.target_measure)
        twz = project_z(self.w(z, params=self.target_w), True)[i_idx]
        target_M = ((tphi * twz).sum(-1, keepdims=True) + tb).reshape(B, B)
        target_M = jax.lax.stop_gradient(target_M)

        diff = M - c["discount"] * target_M
        offdiag = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        # RLU psm.py:352 diagonal source term: -(1-gamma)*mean(diag(M)).
        diag = -((1 - c["discount"]) * jnp.diagonal(M)).mean()
        loss = offdiag + diag
        info = {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag}
        # Optional orthonormality regularizer on the basis Phi(s,a,x) evaluated at the
        # batch's own transitions (B,d) — pushes Phi Phi^T -> I, preventing the basis from
        # collapsing/exploding (the main stability lever in PSM; RLU leaves it off).
        if c["ortho_coef"] > 0:
            phi_batch, _ = self.measure(obs, action, x, params=measure_params)
            ol, oldiag, oloff = ortho_loss(phi_batch, off, off_sum)
            loss = loss + c["ortho_coef"] * ol
            info.update({"psm_loss": loss, "orth_loss": ol, "orth_diag": oldiag, "orth_offdiag": oloff})
        return loss, info

    def actor_loss(self, batch, sampled, actor_params):
        """Amortized (in-loop) w-conditioned actor, FB-style. For a sampled goal g, form
        its closed-form task coord w_g from the (frozen) measure, and train pi(s, w_g) to
        maximize the measure's Q = Phi(s,a,g)·w_g + b(s,a,g). Over training this amortizes
        goal-reaching across the whole goal/w distribution (vs the paper's from-scratch
        eval-time distillation). The measure is a stop-gradient constant here."""
        c = self.config
        obs, action = batch["observations"], batch["actions"]
        B = obs.shape[0]
        g = sampled.actor_goal
        g_rep = jnp.broadcast_to(g, (B,) + g.shape)
        # closed-form w_g for this goal from the frozen measure (dataset actions toward g).
        phi_toward_g, _ = self.measure(obs, action, g_rep)
        w_g = project_z(phi_toward_g.mean(0), True)          # (d,)
        w_rep = jnp.broadcast_to(w_g, (B, c["d_dim"]))
        a = self.actor(obs, w_rep, params=actor_params)      # PSMActor: tanh mean in [-1,1]
        phi_a, b_a = self.measure(obs, a, g_rep)
        Q = (phi_a * w_g).sum(-1) + b_a.squeeze(-1)
        loss = -Q.mean()
        info = {"actor_loss": loss, "actor_q": Q.mean()}
        bc_coeff = c["actor"].get("bc_coeff", 0.0) if hasattr(c["actor"], "get") else c["actor"]["bc_coeff"]
        if bc_coeff > 0:
            bc = jnp.mean((a - action) ** 2)
            # normalize the Q term by |Q| so the BC weight has a stable relative scale (FB/psm).
            loss = loss / jax.lax.stop_gradient(jnp.abs(Q).mean() + 1e-6) + bc_coeff * bc
            info = {"actor_loss": loss, "actor_q": Q.mean(), "actor_bc": bc}
        return loss, info

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_m, g_w) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.measure.params, self.w.params)
        measure = self.measure.apply_gradients(grads=g_m)
        w = self.w.apply_gradients(grads=g_w)
        target_measure = polyak_update(measure.params, self.target_measure, tau)
        target_w = polyak_update(w.params, self.target_w, tau)
        # amortized actor stage: train the w-conditioned actor against the measure's Q.
        (_, a_info), g_actor = jax.value_and_grad(self.actor_loss, argnums=2, has_aux=True)(
            batch, sampled, self.actor.params)
        actor = self.actor.apply_gradients(grads=g_actor)
        return self.replace(measure=measure, w=w, actor=actor, target_measure=target_measure,
                            target_w=target_w), {**info, **a_info}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.measure.params, self.w.params)
        _, a_info = self.actor_loss(batch, sampled, self.actor.params)
        return loss, {**info, **a_info}

    # ---- inference ----
    def infer_w_goal(self, dataset, goal, seed=0):
        """Solve w_inf by constrained optimization (RLU infer_w_goal/_infer_step_gc).
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
                                  jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]),
                                  jnp.asarray(ex["next_observations"]))["params"]
        lam_opt = optax.adam(c["lr_w"])
        lam_state = lam_opt.init(lam_params)

        @jax.jit
        def step(w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep):
            def primal(w):
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
                    Mp = (phi_p * jax.lax.stop_gradient(w_inf)).sum(-1, keepdims=True) + b_p
                    lam = lam_def.apply({"params": lp}, obs, act, xperm)
                    return (Mp * lam).mean()

                # RLU minimizes (M*lam) w.r.t. lam (psm.py:559-565): with the primal penalty
                # -(M*lam), this GROWS lam where M<0 (violated). optax minimizes, so feed the
                # raw grad — the prior negation inverted constraint enforcement (audit bug 5b).
                gl = jax.grad(dual)(lam_params)
                ul, lam_state = lam_opt.update(gl, lam_state, lam_params)
                lam_params = optax.apply_updates(lam_params, ul)
            return w_inf, w_state, lam_params, lam_state, obj, con

        for _ in range(int(ic["num_inference_steps"])):
            b = dataset.sample(c["batch_size"])
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(b["actions"])
            xb = jnp.asarray(b["next_observations"])
            perm = np.random.default_rng().permutation(obs.shape[0])
            xperm = xb[perm]
            goal_rep = jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape)
            w_inf, w_state, lam_params, lam_state, _, _ = step(
                w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep)

        if bool(ic["norm_w"]):
            w_inf = project_z(w_inf, True)
        return self.replace(w_inf=w_inf, task_goal=goal)

    def infer_w_zeroshot(self, dataset, goal, num_samples=4096):
        """Closed-form goal code (no test-time optimization): w = sqrt(d)*normalize(
        E_{(s,a)~D}[Phi(s,a,goal)]) — the affine-net analogue of FB get_goal_meta."""
        c = self.config
        goal = jnp.asarray(goal, jnp.float32)
        # Deterministic full-array reduction when the dataset exposes its raw arrays
        # (unit-test _FakeDataset). Production ReplayBuffer has no `.obs`, so it samples.
        if hasattr(dataset, "obs") and num_samples >= getattr(dataset, "n", 0):
            obs = jnp.asarray(dataset.obs); act = jnp.asarray(dataset.act)
        else:
            b = dataset.sample(num_samples)
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(b["actions"])
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
        # The actor is amortized (trained in-loop) and conditioned on the task coord w.
        # infer_eval sets w_inf (LP or closed-form); the actor acts greedily for it.
        w = jnp.broadcast_to(self.w_inf, (*observations.shape[:-1], self.config["d_dim"]))
        return self.actor(observations, w)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rm, rw, ract, rproto = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "affine_psm does not support visual encoders."
        action_dim = ex_actions.shape[-1]
        d_dim, z_dim = config["d_dim"], config["z_dim"]
        ex_obs, ex_act = ex_observations, ex_actions
        ex_x = ex_obs
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))          # actor task-conditioning (zeroed placeholder)
        ex_zbin = jnp.zeros((ex_obs.shape[0], config["max_log_seed"]))  # binary codebook code -> w

        measure_def = AffineMeasureNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"],
                                       hidden_layers=config["measure"]["hidden_layers"],
                                       b_scale=float(config["measure"].get("b_scale", 10.0)))
        measure = TrainState.create(measure_def, measure_def.init(rm, ex_obs, ex_act, ex_x)["params"],
                                    tx=optax.adam(config["lr"]))

        # w: binary codebook code (max_log_seed bits) -> d task coords (RLU psm.py self.w).
        w_def = _WNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"])
        w = TrainState.create(w_def, w_def.init(rw, ex_zbin)["params"], tx=optax.adam(config["lr_w"]))

        actor_cfg = config["actor"]
        actor_def = PSMActor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                             embedding_layers=actor_cfg["embedding_layers"],
                             hidden_layers=actor_cfg["hidden_layers"])
        actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z)["params"],
                                  tx=optax.adam(config["lr"]))

        # proto codebook table (reuse the psm.py scheme).
        max_seed = 2 ** config["max_log_seed"] + 20000
        table = (jax.random.uniform(rproto, (max_seed, action_dim)) - 1.0) * 2.0
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)
        proto = (table.astype(jnp.float32), powers)

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["proto_max_seed"] = max_seed
        return cls(rng=rng, measure=measure, w=w,
                   target_measure=copy.deepcopy(measure.params),
                   target_w=copy.deepcopy(w.params),
                   w_inf=jnp.ones((d_dim,), jnp.float32), actor=actor,
                   config=flax.core.FrozenDict(config), proto=proto)
