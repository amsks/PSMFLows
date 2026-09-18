"""psmgoal: RLU Proto Successor Measure (arXiv 2411.19418) on frozen-flow latents."""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.f_psmflow import _load_flow_params
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField
from utils.psm_common import _plain_config, polyak_update
from utils.psm_networks import (
    GoalCoefficient,
    PolicyCoefficient,
    RLUMeasure,
    TanhGaussianLatentActor,
    TripleMultiplier,
    tanh_gaussian_sample,
)
from utils.psm_proto import proto_latents, proto_seed_ints, sample_z_bin

#: A relabel row is rewarding when its (shifted) reward exceeds this. main.py and
#: tools/eval_checkpoint.py pass r + eval_reward_shift, so cube's {-1, 0} arrives as {0, 1}.
REWARDING_THRESHOLD = 0.5


@flax.struct.dataclass
class InferenceState:
    """The mutable state of the Stage-2 coefficient optimization (RLU `init_inference`)."""
    w: Any                    # (z_dim,) the free coefficient, on the sqrt(z_dim) sphere
    w_opt: Any                # optax state for w
    l_params: Any          # l(s,u,g) params (softplus multiplier network)
    l_opt: Any             # optax state for the multiplier


class PSMGoalAgent(flax.struct.PyTreeNode):
    """RLU Proto Successor Measure on flow latents; see the module docstring."""

    rng: Any
    basis: TrainState           # (phi, b) on [s, u, g], single critic
    w: TrainState            # w(z): binary policy code -> sqrt(D) sphere
    l: TrainState            # l(s, u, g) >= 0, softplus (used at inference)
    actor: TrainState           # DSRL latent actor (used only for acting=distill)
    w_star: TrainState       # h(g): goal state -> sqrt(D) sphere (A/B goal_conditioned mode;
                                # always built for restore-safety, trained only if train_goal_head)
    target_basis: Any
    target_w: Any
    flow_vf: Any                # FROZEN behaviour-flow velocity field
    flow_onestep: Any           # FROZEN one-step distilled decoder
    eval_goals: Any             # (k_goals, ob) goal set G, set by infer_eval_goals
    eval_w_star: Any                 # (z_dim,) the inferred coefficient w
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert config.get("encoder", None) is None, "psmgoal does not support visual encoders."
        assert int(config["max_log_seed"]) >= 1, "max_log_seed (policy-code width) must be >= 1"
        assert int(config["k_goals"]) >= 1 and int(config["gpi_num_u"]) >= 1
        assert float(config["inf_coeff"]) >= 0.0
        rng = jax.random.PRNGKey(seed)
        rng, r_b, r_c, r_l, r_a, r_vf, r_os = jax.random.split(rng, 7)
        action_dim = ex_actions.shape[-1]
        z_dim = int(config["z_dim"])
        code_dim = int(config["max_log_seed"])
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))
        ex_z = jnp.zeros((ex_observations.shape[0], code_dim))

        basis_def = RLUMeasure(z_dim=z_dim, hidden_dim=config["measure"]["hidden_dim"],
                               hidden_layers=config["measure"]["hidden_layers"])
        basis = TrainState.create(
            basis_def, basis_def.init(r_b, ex_observations, ex_u, ex_observations)["params"],
            tx=optax.adam(config["lr_measure"]))
        w_def = PolicyCoefficient(z_dim=z_dim, hidden_dim=config["w"]["hidden_dim"],
                                     hidden_layers=config["w"]["hidden_layers"])
        w = TrainState.create(w_def, w_def.init(r_c, ex_z)["params"],
                                 tx=optax.adam(config["lr_w"]))
        l_def = TripleMultiplier(hidden_dim=config["l"]["hidden_dim"],
                                    hidden_layers=config["l"]["hidden_layers"])
        l = TrainState.create(
            l_def, l_def.init(r_l, ex_observations, ex_u, ex_observations)["params"],
            tx=optax.adam(config["lr_l"]))
        actor_def = TanhGaussianLatentActor(action_dim=action_dim,
                                            hidden_dim=config["actor"]["hidden_dim"],
                                            hidden_layers=config["actor"]["hidden_layers"])
        actor = TrainState.create(actor_def, actor_def.init(r_a, ex_observations, ex_u)["params"],
                                  tx=optax.adam(config["lr_actor"]))
        # h(g): goal -> sqrt(D) sphere. Always built so a restore has the slot; trained only
        # when train_goal_head.
        rng, r_g = jax.random.split(rng)
        gc = config.get("w_star", {"hidden_dim": 256, "hidden_layers": 2})
        w_star_def = GoalCoefficient(z_dim=z_dim, hidden_dim=gc["hidden_dim"],
                                        hidden_layers=gc["hidden_layers"])
        w_star = TrainState.create(w_star_def, w_star_def.init(r_g, ex_observations)["params"],
                                      tx=optax.adam(config.get("lr_goal", 1.0e-4)))

        flow_hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                       layer_norm=config["flow"]["layer_norm"])
        ckpt = config.get("flow_ckpt_path", None)
        if ckpt:
            flow_vf, flow_onestep = _load_flow_params(
                ckpt, config.get("flow_ckpt_epoch", None), config, ex_observations, ex_actions)
        else:
            assert config.get("allow_untrained_flow", False), (
                "psmgoal requires agent.flow_ckpt_path (a Stage-A fql bc_only ckpt dir); "
                "set agent.allow_untrained_flow=true only for tests/smokes.")
            flow_vf = vf_def.init(r_vf, ex_observations, ex_actions, ex_actions[..., :1])["params"]
            flow_onestep = onestep_def.init(r_os, ex_observations, ex_actions)["params"]

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        ob_dim = int(ex_observations.shape[-1])
        return cls(rng=rng, basis=basis, w=w, l=l, actor=actor, w_star=w_star,
                   target_basis=copy.deepcopy(basis.params),
                   target_w=copy.deepcopy(w.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   eval_goals=jnp.zeros((int(config["k_goals"]), ob_dim), jnp.float32),
                   eval_w_star=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)

    # ------------------------------------------------------------------ the measure
    def M(self, obs, u, g, w, params=None):
        """M(s,u,g) = phi(s,u,g)^T w + b -> (B,). Single-point measure; the mesh_M reference
        used in tests. w is (B, z_dim) or (z_dim,)."""
        phi, b = self.basis(obs, u, g, params=self.basis.params if params is None else params)
        return (phi * jnp.atleast_2d(w)).sum(-1) + b

    def mesh_M(self, obs, u, goals, w_rows, params=None):
        """The state x goal mesh: M_ij = phi(s_i,u_i,g_j)^T w_i + b(s_i,u_i,g_j).

        obs, u are (N, .) rows; goals (G, ob); w_rows (N, z_dim). Returns (N, G).
        """
        # all-pairs state x goal mesh (RLU discrete_psm.py:374-384): row i = (s_i,u_i), col j = g_j.
        N, G = obs.shape[0], goals.shape[0]
        obs_r = jnp.broadcast_to(obs[:, None], (N, G, obs.shape[-1])).reshape(N * G, -1)
        u_r = jnp.broadcast_to(u[:, None], (N, G, u.shape[-1])).reshape(N * G, -1)
        g_r = jnp.broadcast_to(goals[None], (N, G, goals.shape[-1])).reshape(N * G, -1)
        # (phi, b) from the single-critic basis; params=None -> online params.
        phi, b = self.basis(obs_r, u_r, g_r, params=self.basis.params if params is None else params)
        phi = phi.reshape(N, G, -1)
        b = b.reshape(N, G)
        return (phi * w_rows[:, None, :]).sum(-1) + b

    # ------------------------------------------------------------------ proto bootstrap
    def proto_bootstrap(self, z, index):
        """u^+: the fixed z-indexed policy's latent at each row. A clipped prior draw keyed on
        (z, row), so G(s', u^+) stays in-support. RLU SamplingSeedActor, discrete_psm.py:137-171.
        Returns (N, d_a)."""
        c = self.config
        seeds = proto_seed_ints(z, index, int(c["max_log_seed"]))
        base = jax.random.PRNGKey(int(c["proto_seed"]))
        return proto_latents(seeds, c["action_dim"], c["u_clip"], base)

    # ------------------------------------------------------------------ basis loss
    def measure_loss(self, basis_params, w_params, batch, z, u_next):
        """RLU `update_psm`: squared TD on the off-diagonal of the state x goal mesh plus a
        `-(1-gamma)` diagonal pull. Gradient reaches phi, b (basis) and w (w)."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        next_obs = jnp.asarray(batch["next_observations"])
        u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        goals = next_obs                                         # g_j = s'_j
        w = self.w(z, params=w_params)                 # (N, z), online
        w_t = self.w(z, params=self.target_w)          # (N, z), target
        M = self.mesh_M(obs, u, goals, w, params=basis_params)                    # (N, N)
        M_bar = self.mesh_M(next_obs, u_next, goals, w_t, params=self.target_basis)
        target_M = jax.lax.stop_gradient(M_bar)
        # Successor-measure Bellman fit (PSM Eq. 2 + Cor. 4.2, arXiv 2411.19418 v2; RLU
        # discrete_psm.py:427-434): off-diagonal = squared TD to gamma*target; diagonal =
        # -(1-gamma) pull toward reaching your own next state.
        gamma = c["discount"]
        N = M.shape[0]
        off = 1.0 - jnp.eye(N)
        diff = M - gamma * target_M
        off_diag = 0.5 * jnp.sum((diff ** 2) * off) / jnp.maximum(jnp.sum(off), 1.0)
        diag = -(1.0 - gamma) * jnp.mean(jnp.diagonal(M))
        loss = off_diag + diag
        info = {"psm_loss": loss, "psm_offdiag": off_diag, "psm_diag": diag,
                "m_diag_mean": jnp.mean(jnp.diagonal(M)),
                "m_offdiag_mean": jnp.sum(M * off) / jnp.maximum(jnp.sum(off), 1.0),
                "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                "w_norm": jnp.mean(jnp.linalg.norm(w, axis=-1))}
        return loss, info

    # --------------------------------------------------------- goal-conditioned head (A/B)
    def goal_head_loss(self, w_star_params, batch, perm):
        """Amortize the per-goal coefficient into h(g) (A/B goal_conditioned mode).

        J(theta|g) = mean_i M(s_i,u_i,g_i) at w*(g)=h(g), minus a hinge on the non-negativity
        constraint phi^T w*(g)+b >= 0 over off-goal (permuted) next states. Grad reaches h only.
        Keep the constraint: without it h(g) -> E[r_g f] = FB, which ranked at chance. Goals are
        the hindsight mixture batch['goals']. RLU solves this LP per goal; here it is amortized."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        goals = jnp.asarray(batch["goals"])                    # (N, ob) hindsight goal per row
        next_obs = jnp.asarray(batch["next_observations"])
        bp = jax.lax.stop_gradient(self.basis.params)
        w_g = self.w_star(goals, params=w_star_params)   # (N, z), grad to h only
        obj = jnp.mean(self.M(obs, u, goals, w_g, params=bp))  # value at own goal
        cons = self.mesh_M(obs, u, next_obs[perm], w_g, params=bp)   # (N, N) M at off-goals
        pen = jnp.mean(jax.nn.relu(-cons))                     # hinge on non-negativity
        loss = -obj + float(c.get("j_constraint_coef", 1.0)) * pen
        info = {"goal_obj": obj, "goal_pen": pen,
                "goal_viol_frac": jnp.mean((cons < 0.0).astype(jnp.float32)),
                "goal_w_norm": jnp.mean(jnp.linalg.norm(w_g, axis=-1))}
        return loss, info

    # ------------------------------------------------------------------ update
    def apply_update(self, batch, z, u_next):
        c = self.config
        grad_fn = jax.value_and_grad(self.measure_loss, argnums=(0, 1), has_aux=True)
        (_, info), (g_b, g_c) = grad_fn(self.basis.params, self.w.params, batch, z, u_next)
        basis = self.basis.apply_gradients(grads=g_b)
        w = self.w.apply_gradients(grads=g_c)
        w_star = self.w_star
        if bool(c.get("train_goal_head", False)) and "goals" in batch:
            perm = jax.random.permutation(jax.random.fold_in(self.rng, 55),
                                          jnp.asarray(batch["observations"]).shape[0])
            (_, gh_info), g_gc = jax.value_and_grad(self.goal_head_loss, has_aux=True)(
                self.w_star.params, batch, perm)
            w_star = self.w_star.apply_gradients(grads=g_gc)
            info = {**info, **gh_info}
        new = self.replace(
            basis=basis, w=w, w_star=w_star,
            target_basis=polyak_update(basis.params, self.target_basis, c["tau"]),
            target_w=polyak_update(w.params, self.target_w, c["tau"]))
        return new, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        N = jnp.asarray(batch["observations"]).shape[0]
        z = sample_z_bin(rng, N, int(self.config["max_log_seed"]))
        index = jnp.asarray(batch["index"]) if "index" in batch else jnp.arange(N)
        u_next = self.proto_bootstrap(z, index)
        new_agent, info = self.apply_update(batch, z, u_next)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss at current params (no step)."""
        rng = rng if rng is not None else self.rng
        N = jnp.asarray(batch["observations"]).shape[0]
        z = sample_z_bin(rng, N, int(self.config["max_log_seed"]))
        index = jnp.asarray(batch["index"]) if "index" in batch else jnp.arange(N)
        u_next = self.proto_bootstrap(z, index)
        return self.measure_loss(self.basis.params, self.w.params, batch, z, u_next)

    # ------------------------------------------------------------------ inference (RLU infer_w)
    def init_inference(self, key):
        """Fresh Stage-2 state: w on the sphere, a re-initialized multiplier net."""
        c = self.config
        z_dim = int(c["z_dim"])
        w = self._project(jax.random.normal(key, (z_dim,)))
        w_tx = optax.adam(c["lr_infer"])
        l_tx = optax.adam(c["lr_l"])
        return InferenceState(w=w, w_opt=w_tx.init(w),
                              l_params=self.l.params, l_opt=l_tx.init(self.l.params))

    def _project(self, w):
        """Renormalize w_inf onto the sqrt(z_dim) sphere each step; RLU discrete_psm.py:662,698."""
        r = jnp.sqrt(float(self.config["z_dim"]))
        return r * w / jnp.maximum(jnp.linalg.norm(w), 1e-12)

    def _obj_and_constraint(self, w, l_params, obs, u, goals, perm):
        """PSM Eq. 10: obj = mean phi(s,u,g)^T w, and the non-negativity constraint
        phi^T w + b >= 0 on permuted off-goal samples, priced by the multiplier l(s,u,g)."""
        bp = jax.lax.stop_gradient(self.basis.params)
        obj = jnp.mean(self.mesh_M(obs, u, goals, jnp.broadcast_to(w, (obs.shape[0], w.shape[0])), params=bp))
        obs_p, u_p = obs[perm], u[perm]
        # constraint value per (permuted sample, goal): phi.w + b, averaged over goals
        cons = self.mesh_M(obs_p, u_p, goals, jnp.broadcast_to(w, (obs_p.shape[0], w.shape[0])), params=bp)
        l = self.l(
            jnp.repeat(obs_p, goals.shape[0], axis=0),
            jnp.repeat(u_p, goals.shape[0], axis=0),
            jnp.tile(goals, (obs_p.shape[0], 1)),
            params=l_params).reshape(obs_p.shape[0], goals.shape[0])
        return obj, cons, l

    def inference_step(self, state, obs, u, goals, key):
        """One RLU coefficient step: descend w on -obj + penalty, ascend the multiplier,
        renormalize w onto the sphere. Returns (state', info)."""
        c = self.config
        use_dgd = bool(c["use_dgd"])
        perm = jax.random.permutation(key, obs.shape[0])
        w_tx = optax.adam(c["lr_infer"])
        l_tx = optax.adam(c["lr_l"])

        def w_loss(w):
            obj, cons, l = self._obj_and_constraint(w, state.l_params, obs, u, goals, perm)
            if use_dgd:
                pen = -jnp.mean(cons * jax.lax.stop_gradient(l))
            else:
                pen = -jnp.mean(jnp.minimum(cons, 0.0)) * c["inf_coeff"]
            return -obj + pen, (obj, cons)

        (_, (obj, cons)), g_w = jax.value_and_grad(w_loss, has_aux=True)(state.w)
        upd, w_opt = w_tx.update(g_w, state.w_opt)
        w = self._project(optax.apply_updates(state.w, upd))

        l_params, l_opt = state.l_params, state.l_opt
        if use_dgd:
            cons_sg = jax.lax.stop_gradient(cons)

            def l_loss(mp):
                # Dual ascent on l (RLU psm.py:560-565): minimizing mean(cons*l) raises l where
                # the constraint is violated (cons < 0).
                _, _, l = self._obj_and_constraint(state.w, mp, obs, u, goals, perm)
                return jnp.mean(cons_sg * l)

            _, g_l = jax.value_and_grad(l_loss)(l_params)
            upd_l, l_opt = l_tx.update(g_l, l_opt)
            l_params = optax.apply_updates(l_params, upd_l)

        info = {"obj": obj, "viol_frac": jnp.mean((cons < 0.0).astype(jnp.float32)),
                "viol_mean": jnp.mean(jax.nn.relu(-cons)), "w_norm": jnp.linalg.norm(w)}
        return state.replace(w=w, w_opt=w_opt, l_params=l_params, l_opt=l_opt), info

    def run_inference(self, obs, u, goals, key):
        """Full Stage-2 loop; returns the inferred coefficient w (z_dim,)."""
        state = self.init_inference(key)
        step = jax.jit(self.inference_step)
        for i in range(int(self.config["num_inference_steps"])):
            state, _ = step(state, obs, u, goals, jax.random.fold_in(key, i))
        return state.w

    # ------------------------------------------------------------------ acting
    def decode(self, observations, u):
        """G(s, u) through the FROZEN flow. observations (B, ob), u (B, d_a)."""
        if self.config["gpi_decode"] == "onestep":
            a = self.flow_onestep_def.apply({"params": self.flow_onestep}, observations, u)
        else:
            a = u
            steps = self.config["flow_decode_steps"]
            for i in range(steps):
                t = jnp.full((*observations.shape[:-1], 1), i / steps)
                a = a + self.flow_vf_def.apply({"params": self.flow_vf}, observations, a, t) / steps
        return jnp.clip(a, -1.0, 1.0)

    def select_latent(self, observations, seed):
        """acting=gpi: argmax over gpi_num_u clipped prior draws of the goal-averaged Q,
        Q(s, u_m) = mean_{g in G} phi(s,u_m,g)^T eval_w_star + b(s,u_m,g)."""
        c = self.config
        assert observations.ndim == 1, "select_latent acts on a single observation"
        K, d_a = int(c["gpi_num_u"]), c["action_dim"]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, observations.shape[-1]))
        w = jnp.broadcast_to(self.eval_w_star, (K, self.eval_w_star.shape[0]))
        q = jnp.mean(self.mesh_M(obs, u_cand, self.eval_goals, w), axis=1)   # (K,)
        return u_cand[jnp.argmax(q)]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """The deployed action. acting=gpi decodes the argmax latent; acting=distill samples
        the distilled DSRL actor's mode."""
        seed = self.rng if seed is None else seed
        if self.config["acting"] == "distill":
            mu, _ = self.actor(observations[None], self.eval_w_star[None])
            u_star = self.config["u_clip"] * jnp.tanh(mu[0])
        else:
            u_star = self.select_latent(observations, seed)
        return self.decode(observations[None], u_star[None])[0]

    def distill_actor(self, obs, key, steps=None):
        """Train the DSRL actor to maximize Q = phi^T eval_w_star + b over the goal set. Minimize
        temp*log_prob - Q on a reparameterized draw. RLU distill_actor_ddpg, psm.py:569-603."""
        c = self.config
        steps = int(c["num_actor_steps"]) if steps is None else int(steps)
        temp, scale = float(c["actor_temp"]), float(c["u_clip"])

        q_coeff, bc_coeff = float(c.get("q_coeff", 1.0)), float(c.get("bc_coeff", 0.0))

        def actor_loss(params, o, noise):
            z_in = jnp.broadcast_to(self.eval_w_star, (o.shape[0], self.eval_w_star.shape[0]))
            mu, log_std = self.actor(o, z_in, params=params)
            u, logp = tanh_gaussian_sample(mu, log_std, noise, scale)
            q = jax.vmap(lambda oo, uu: jnp.mean(
                self.mesh_M(oo[None], uu[None], self.eval_goals,
                            self.eval_w_star[None])))(o, u)
            # Q has no natural scale, so normalise by stopgrad|Q|; temp*logp is the entropy term,
            # bc anchors u to the prior centre. Full flow-matching BC is deferred.
            qscale = jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-8)
            return q_coeff * (-jnp.mean(q) / qscale) + temp * jnp.mean(logp) + bc_coeff * jnp.mean(u ** 2)

        actor = self.actor
        step = jax.jit(lambda a, o, n: a.apply_gradients(
            grads=jax.grad(actor_loss)(a.params, o, n)))
        for i in range(steps):
            noise = jax.random.normal(jax.random.fold_in(key, i), obs.shape[:-1] + (c["action_dim"],))
            actor = step(actor, obs, noise)
        return self.replace(actor=actor)

    # ------------------------------------------------------------------ eval inference
    def infer_eval_goals(self, batch, rewards):
        """Set the goal set G and the eval coefficient from a relabel batch. G = k_goals
        rewarding next states; the coefficient is the LP solution (or h(g) if amortized)."""
        c = self.config
        rewards = np.asarray(rewards).reshape(-1)
        rows = np.nonzero(rewards > REWARDING_THRESHOLD)[0]
        if rows.size == 0:
            raise ValueError("infer_eval_goals: relabel batch has no rewarding row "
                             f"(reward > {REWARDING_THRESHOLD}); cannot build a goal set")
        k = int(c["k_goals"])
        pick = np.random.choice(rows, size=k, replace=rows.size < k)
        goals = jnp.asarray(np.asarray(batch["next_observations"])[pick], jnp.float32)
        all_obs = np.asarray(batch["observations"])
        n = min(int(c["infer_batch"]), all_obs.shape[0])
        sub = np.random.choice(np.arange(all_obs.shape[0]), size=n, replace=False)
        obs = jnp.asarray(all_obs[sub], jnp.float32)
        key = jax.random.fold_in(self.rng, 7)
        # Inference uses E_{(s,u)~D}: the dataset preimages (recorded actions' latents) for the
        # sub-rows, not prior draws (prior draws made the LP's a_bar the wrong direction).
        u = jnp.clip(jnp.asarray(np.asarray(batch["noise_preimage"])[sub], jnp.float32),
                     -c["u_clip"], c["u_clip"])
        if str(c.get("coef_source", "lp")) == "amortized":
            # A/B goal_conditioned: amortized coefficient from the trained goal head.
            # w = normalize(mean_{g in G} h(g)) -- the design's sum-over-goals readout.
            w = self._project(jnp.mean(self.w_star(goals), axis=0))
        else:
            w = self.run_inference(obs, u, goals, key)   # RLU Lagrangian inference (default)
        agent = self.replace(eval_goals=goals, eval_w_star=w)
        if c["acting"] == "distill":
            agent = agent.distill_actor(obs, jax.random.fold_in(self.rng, 11))
        return agent


def get_config():
    """Importable default config, mirrored by configs/agent/psmgoal.yaml."""
    import ml_collections

    return ml_collections.ConfigDict({
        "agent_name": "psmgoal",
        "batch_size": 256,          # mesh is batch^2, so smaller than f_psmflow's 1024
        "z_dim": 128,               # D: width of phi and of the coefficient
        "discount": 0.98,           # gamma
        "tau": 0.01,                # Polyak rate of the phi,b,w targets
        "lr_measure": 1.0e-4,       # phi, b
        "lr_w": 1.0e-4,          # w(z)
        "lr_l": 3.0e-4,             # multiplier (RLU lr_infer scale)
        "max_log_seed": 8,          # policy-index code width (RLU proto family)
        "proto_seed": 0,            # base key for the fixed z-indexed policy (a code constant)
        # --- Stage-2 coefficient inference (RLU infer_w) ---
        "lr_infer": 3.0e-4,             # the free coefficient
        "num_inference_steps": 20000,
        "use_dgd": True,            # True: learned multiplier network; False: hinge
        "inf_coeff": 5.0,           # hinge weight (use_dgd=False)
        "infer_batch": 1024,        # (s, u) samples per inference step
        "k_goals": 32,              # size of the eval goal set G
        # --- acting ---
        "acting": "gpi",            # gpi (argmax over prior draws) | distill (DSRL actor)
        "gpi_num_u": 64,            # prior draws the gpi argmax scans
        "num_actor_steps": 10000,   # distillation steps (acting=distill)
        "actor_temp": 0.0,          # entropy weight (alpha) in the distillation objective
        "q_coeff": 1.0,             # distill: weight on the (scale-normalised) Q term
        "bc_coeff": 0.0,            # distill: prior-centre anchor weight (in-support pull)
        # --- A/B goal_conditioned mode (all OFF by default -> byte-identical to the RLU core) ---
        "train_goal_head": False,   # train h(g) by J(theta|g)+constraint during update()
        "coef_source": "lp",        # lp (RLU Lagrangian inference) | amortized (use h(g))
        "lr_goal": 1.0e-4,          # h(g) learning rate
        "j_constraint_coef": 1.0,   # hinge weight on non-negativity in J(theta|g)
        "w_star": {"hidden_dim": 256, "hidden_layers": 2},   # h(g) net
        "goal_discount": 0.98,      # dataset hindsight-goal geometric horizon (main.py)
        "goal_random_frac": 0.3,    # dataset off-trajectory random-goal fraction (OGBench-style)
        # --- nets ---
        "measure": {"hidden_dim": 512, "hidden_layers": 2},
        "w": {"hidden_dim": 256, "hidden_layers": 2},
        "l": {"hidden_dim": 256, "hidden_layers": 2},
        "actor": {"hidden_dim": 512, "hidden_layers": 2},
        "lr_actor": 3.0e-4,
        # --- frozen behaviour flow (must match the Stage-A fql bc_only run) ---
        "flow": {"hidden_dims": (512, 512, 512, 512), "value_hidden_dims": (512, 512, 512, 512),
                 "layer_norm": False, "critic_layer_norm": True},
        "flow_ckpt_path": ml_collections.config_dict.placeholder(str),
        "flow_ckpt_epoch": ml_collections.config_dict.placeholder(int),
        "allow_untrained_flow": False,
        "preimage_path": ml_collections.config_dict.placeholder(str),
        "use_point_preimage": True,
        "u_clip": 3.0,
        "gpi_decode": "onestep",    # onestep | ode
        "flow_decode_steps": 10,
        "ob_dims": ml_collections.config_dict.placeholder(list),
        "action_dim": ml_collections.config_dict.placeholder(int),
        "encoder": ml_collections.config_dict.placeholder(str),
    })
