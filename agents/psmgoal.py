"""psmgoal: goal-indexed affine successor measure over the frozen flow's latents.

Spec: docs/design/2026-09-17-psmgoal.md. Every symbol below is that document's.

    M^{pi_g}(s,u,s+) = phi(s,u,s+)^T w*(g) + b(s,u,s+)        phi, b: networks on the triple
    w*(g)            = h(g) / ||h(g)||                          h: goal encoder
    Q_g(s,u)         = M(s,u,g)                                 r_g(s+) = 1[s+ = g]
    l(s,u,s+) >= 0                                              per-triple multiplier (softplus)

Per update, on N rows (s_i, u_i, s'_i, g_i) with C negative columns (other rows' next states)
and the positive column s'_i:

    1. c_i = w*(g_i), stop-gradded into 2 and 3.
    2. phi, b:  contrastive TD (`contrastive_loss_rect`) of M_ij against gamma * Mbar_ij, the
       TARGET measure at (s'_i, u'_i, s+_j) with u'_i the pessimistic argmax of the target
       Q_{g_i}(s'_i, .) over n_backup clipped prior draws.  Descent on phi, b only.
    3. h:       loss_h = -obj + constraint_coef * pen, obj the mean goal value
       M(s_i,u_i,g_i) at w*(g_i), pen = mean_ij l_ij * max(-M_ij, 0) over the negatives with
       phi, b, l stop-gradded.  Descent on h only.
    4. l:       ascent on pen with everything else fixed (constraint_coef > 0 only).
    5. Polyak targets for phi, b.

Test time: `infer_eval_goals` collects k_goals rewarding next states G, w = normalised
sum of w*(g); `sample_actions` decodes argmax_m [min_P] mean_g M(s, u_m, g) over gpi_num_u
prior draws through the frozen flow.

Reused from `agents/psmflow.py`: the frozen-flow load and decode, the preimage dataset path
(`noise_preimage` in the batch), `restore_agent`, the single-observation jitted
`sample_actions`. File order: construction, batch sampling, losses, update, acting, inference.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.psmflow import _load_flow_params
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField
from utils.psm_common import _plain_config, contrastive_loss_rect, polyak_update, targets_uncertainty
from utils.psm_networks import GoalCoefficient, TripleMeasure, TripleMultiplier

#: A relabel row counts as rewarding when its (shifted) reward exceeds this. main.py and
#: tools/eval_checkpoint.py pass r + eval_reward_shift, so cube's {-1, 0} arrives as {0, 1}.
REWARDING_THRESHOLD = 0.5


@flax.struct.dataclass
class StepInputs:
    """Quantities drawn once per update and shared by every loss.

    u_data      (N, d_a)        dataset latent, the flow preimage of the recorded action
    u_next      (N, d_a)        backup latent u'_i at s'_i: pessimistic argmax of the target
                                Q_{g_i} over n_backup clipped prior draws
    backup_adv  (N,)            Q(u'_i) - mean_m Q(u_m) at s'_i, telemetry
    cols        (N, C+1, ob)    the s+ columns: column 0 is s'_i, columns 1..C are the next
                                states of C other rows
    """
    u_data: Any
    u_next: Any
    backup_adv: Any
    cols: Any


class PSMGoalAgent(flax.struct.PyTreeNode):
    """Goal-indexed affine measure; see the module docstring and the spec."""

    rng: Any
    phi_b: TrainState           # (phi, b) on the triple, P-fold ensemble
    coef: TrainState            # h: goal -> unit coefficient w*(g)
    mult: TrainState            # l: per-triple multiplier >= 0
    target_phi_b: Any
    flow_vf: Any                # FROZEN: multi-step behaviour-flow velocity field
    flow_onestep: Any           # FROZEN: one-step distilled decoder
    eval_goals: Any             # (k_goals, ob) goal set G, set by infer_eval_goals
    eval_w: Any                 # (z_dim,) sum of w*(g) over G, projected onto the sqrt(z_dim) sphere
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert config.get("encoder", None) is None, "psmgoal does not support visual encoders."
        assert int(config["num_parallel"]) >= 2, "num_parallel must be >= 2 (the pessimistic min)"
        assert int(config["n_backup"]) >= 1 and int(config["n_neg"]) >= 1 and int(config["k_goals"]) >= 1
        assert 0.0 <= float(config["goal_random_frac"]) <= 1.0, "goal_random_frac must be in [0, 1]"
        assert float(config["constraint_coef"]) >= 0.0, "constraint_coef must be >= 0"
        rng = jax.random.PRNGKey(seed)
        rng, r_pb, r_h, r_l, r_vf, r_onestep = jax.random.split(rng, 6)
        action_dim = ex_actions.shape[-1]
        z_dim = int(config["z_dim"])
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))

        phi_b_def = TripleMeasure(z_dim=z_dim, hidden_dim=config["measure"]["hidden_dim"],
                                  hidden_layers=config["measure"]["hidden_layers"],
                                  num_parallel=config["num_parallel"])
        phi_b = TrainState.create(
            phi_b_def, phi_b_def.init(r_pb, ex_observations, ex_u, ex_observations)["params"],
            tx=optax.adam(config["lr_phi"]))
        coef_def = GoalCoefficient(z_dim=z_dim, hidden_dim=config["coef"]["hidden_dim"],
                                   hidden_layers=config["coef"]["hidden_layers"])
        coef = TrainState.create(coef_def, coef_def.init(r_h, ex_observations)["params"],
                                 tx=optax.adam(config["lr_h"]))
        mult_def = TripleMultiplier(hidden_dim=config["mult"]["hidden_dim"],
                                    hidden_layers=config["mult"]["hidden_layers"])
        mult = TrainState.create(
            mult_def, mult_def.init(r_l, ex_observations, ex_u, ex_observations)["params"],
            tx=optax.adam(config["lr_l"]))

        # Frozen behaviour flow (a Stage-A FQL bc_only checkpoint), as in psmflow.
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
            flow_onestep = onestep_def.init(r_onestep, ex_observations, ex_actions)["params"]

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        ob_dim = int(ex_observations.shape[-1])
        return cls(rng=rng, phi_b=phi_b, coef=coef, mult=mult,
                   target_phi_b=copy.deepcopy(phi_b.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   eval_goals=jnp.zeros((int(config["k_goals"]), ob_dim), jnp.float32),
                   eval_w=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)

    # ------------------------------------------------------------------ the measure
    def measure(self, obs, u, s_plus, coef, params=None):
        """M(s,u,s+) = phi(s,u,s+)^T coef + b(s,u,s+) -> (P, B). `coef` (B, z_dim) is any
        coefficient; the losses pass w*(g), acting passes the goal-averaged w."""
        phi, b = self.phi_b(obs, u, s_plus, params=params)
        return (phi * coef[None]).sum(-1) + b

    def _measure_cols(self, obs, u, cols, coef, params=None):
        """M at every column: obs (N, ob), u (N, d_a), cols (N, J, ob), coef (N, z) -> (P, N, J)."""
        N, J = cols.shape[0], cols.shape[1]
        obs_r = jnp.broadcast_to(obs[:, None], (N, J, obs.shape[-1])).reshape(N * J, -1)
        u_r = jnp.broadcast_to(u[:, None], (N, J, u.shape[-1])).reshape(N * J, -1)
        phi, b = self.phi_b(obs_r, u_r, cols.reshape(N * J, -1), params=params)
        P = phi.shape[0]
        phi = phi.reshape(P, N, J, -1)
        return (phi * coef[None, :, None, :]).sum(-1) + b.reshape(P, N, J)

    def _pessimistic(self, q):
        """[mean_P - pessimism * spread] of an ensemble score; = min at P=2, pessimism=0.5."""
        q_mean, q_unc = targets_uncertainty(q, self.config["num_parallel"])
        return q_mean - self.config["pessimism"] * q_unc

    def backup_latent(self, next_obs, goals, coef, key):
        """u'_i = argmax over n_backup clipped prior draws of the pessimistic TARGET
        Q_{g_i}(s'_i, u_m) = phibar(s'_i,u_m,g_i)^T c_i + bbar(s'_i,u_m,g_i).

        Every candidate is a prior draw, so G(s', u') is in-support. Returns
        (u_star (N, d_a), adv (N,)) with adv = Q(u*) - mean_m Q(u_m), both stop-gradded.
        """
        c = self.config
        N, d_a = next_obs.shape[0], c["action_dim"]
        u_cand = jnp.clip(jax.random.normal(key, (int(c["n_backup"]), N, d_a)), -c["u_clip"], c["u_clip"])

        def score(u_m):
            return self._pessimistic(self.measure(next_obs, u_m, goals, coef, params=self.target_phi_b))

        Q = jax.vmap(score)(u_cand)                                           # (n, N)
        best = jnp.argmax(Q, axis=0)
        u_star = jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0]
        adv = Q.max(0) - Q.mean(0)
        return jax.lax.stop_gradient(u_star), jax.lax.stop_gradient(adv)

    # ------------------------------------------------------------------ batch sampling
    def sample_step_inputs(self, batch, rng):
        """Draw the per-update quantities of `StepInputs` from one batch."""
        c = self.config
        next_obs, goals = jnp.asarray(batch["next_observations"]), jnp.asarray(batch["goals"])
        N, C = next_obs.shape[0], int(c["n_neg"])
        assert C < N, f"n_neg={C} negatives per row need a batch larger than {C}, got {N}"
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        # Negatives: the next states of the C rows following i in the batch (cyclically).
        # The batch is a uniform draw, so this is C uniformly random other rows without
        # repeats and without i itself.
        neg = (jnp.arange(N)[:, None] + 1 + jnp.arange(C)[None, :]) % N       # (N, C)
        cols = jnp.concatenate([next_obs[:, None], next_obs[neg]], axis=1)   # (N, C+1, ob)
        coef = jax.lax.stop_gradient(self.coef(goals))
        u_next, adv = self.backup_latent(next_obs, goals, coef, jax.random.fold_in(rng, 108))
        return StepInputs(u_data=u_data, u_next=u_next, backup_adv=adv, cols=cols)

    # ------------------------------------------------------------------ losses
    def measure_loss(self, batch, sampled, phi_b_params, coef_params):
        """Step 2: contrastive TD of M_ij = phi(s_i,u_i,s+_j)^T c_i + b(s_i,u_i,s+_j) against
        gamma * Mbar_ij at the target (s'_i, u'_i, s+_j), pessimistic over the ensemble.
        c_i = w*(g_i) is stop-gradded, so the gradient reaches phi and b only."""
        c = self.config
        obs, goals = jnp.asarray(batch["observations"]), jnp.asarray(batch["goals"])
        next_obs = jnp.asarray(batch["next_observations"])
        coef = jax.lax.stop_gradient(self.coef(goals, params=coef_params))              # (N, z)
        M = self._measure_cols(obs, sampled.u_data, sampled.cols, coef, params=phi_b_params)
        M_bar = self._measure_cols(next_obs, sampled.u_next, sampled.cols, coef,
                                   params=self.target_phi_b)                              # (P, N, C+1)
        target_M = jax.lax.stop_gradient(self._pessimistic(M_bar))                       # (N, C+1)
        loss, diag, offdiag = contrastive_loss_rect(M, target_M, c["discount"])
        info = {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag,
                "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                "m_absmean": jnp.mean(jnp.abs(M)),
                "m_pos_mean": jnp.mean(M[..., 0]),
                "m_neg_mean": jnp.mean(M[..., 1:]),
                "w_norm": jnp.mean(jnp.linalg.norm(coef, axis=-1))}
        return loss, info

    def multipliers(self, batch, sampled, params=None):
        """l(s_i, u_i, s+_j) over the C negative columns -> (N, C)."""
        obs = jnp.asarray(batch["observations"])
        cols = sampled.cols[:, 1:]
        N, C = cols.shape[0], cols.shape[1]
        obs_r = jnp.broadcast_to(obs[:, None], (N, C, obs.shape[-1])).reshape(N * C, -1)
        u_r = jnp.broadcast_to(sampled.u_data[:, None], (N, C, sampled.u_data.shape[-1])).reshape(N * C, -1)
        return self.mult(obs_r, u_r, cols.reshape(N * C, -1), params=params).reshape(N, C)

    @staticmethod
    def penalty(l, viol):
        """pen = (1/NC) sum_ij l_ij * viol_ij."""
        return jnp.mean(l * viol)

    def coef_loss(self, batch, sampled, phi_b_params, coef_params, mult_params):
        """Step 3: loss_h = -obj + constraint_coef * pen, with phi, b and l stop-gradded.

            obj     = (1/N) sum_i  M(s_i, u_i, g_i)          at w*(g_i)
            viol_ij = max(-M(s_i, u_i, s+_j), 0)              over the C negatives
            pen     = (1/NC) sum_ij l_ij * viol_ij

        M is the ensemble MEAN here. Returns (loss, (info, viol)); `viol` (N, C) is
        stop-gradded and is what `multiplier_step` ascends on.
        """
        c = self.config
        obs, goals = jnp.asarray(batch["observations"]), jnp.asarray(batch["goals"])
        w = self.coef(goals, params=coef_params)                                          # (N, z), live
        frozen = jax.lax.stop_gradient(phi_b_params)
        cols = jnp.concatenate([goals[:, None], sampled.cols[:, 1:]], axis=1)             # (N, C+1, ob)
        M = self._measure_cols(obs, sampled.u_data, cols, w, params=frozen).mean(0)       # (N, C+1)
        obj = jnp.mean(M[:, 0])
        viol = jax.nn.relu(-M[:, 1:])                                                     # (N, C)
        l = jax.lax.stop_gradient(self.multipliers(batch, sampled, params=jax.lax.stop_gradient(mult_params)))
        pen = self.penalty(l, viol)
        loss = -obj
        if float(c["constraint_coef"]) > 0.0:
            loss = loss + c["constraint_coef"] * pen
        info = {"obj": obj, "pen": pen,
                "viol_frac": jnp.mean((viol > 0.0).astype(jnp.float32)),
                "viol_mean": jnp.mean(viol),
                "mult_mean": jnp.mean(l), "mult_max": jnp.max(l)}
        return loss, (info, jax.lax.stop_gradient(viol))

    def mult_loss(self, batch, sampled, mult_params, viol):
        """Step 4: descent on -pen with l live and viol fixed, i.e. ascent on pen."""
        l = self.multipliers(batch, sampled, params=mult_params)
        pen = self.penalty(l, viol)
        return -pen, {"pen": pen}

    def multiplier_step(self, batch, sampled, viol):
        """One ascent step of the multiplier on `viol` (N, C). Returns (agent, info)."""
        (_, info), g_l = jax.value_and_grad(self.mult_loss, argnums=2, has_aux=True)(
            batch, sampled, self.mult.params, viol)
        return self.replace(mult=self.mult.apply_gradients(grads=g_l)), info

    # ------------------------------------------------------------------ update
    def apply_update(self, batch, sampled):
        """Steps 2-5, each loss read at the PRE-update parameters of the others."""
        c = self.config
        (_, m_info), g_pb = jax.value_and_grad(self.measure_loss, argnums=2, has_aux=True)(
            batch, sampled, self.phi_b.params, self.coef.params)
        (_, (h_info, viol)), g_h = jax.value_and_grad(self.coef_loss, argnums=3, has_aux=True)(
            batch, sampled, self.phi_b.params, self.coef.params, self.mult.params)
        phi_b = self.phi_b.apply_gradients(grads=g_pb)
        new = self.replace(phi_b=phi_b, coef=self.coef.apply_gradients(grads=g_h),
                           target_phi_b=polyak_update(phi_b.params, self.target_phi_b, c["tau"]))
        if float(c["constraint_coef"]) > 0.0:
            stepped, _ = self.multiplier_step(batch, sampled, viol)
            new = new.replace(mult=stepped.mult)
        info = {**m_info, **h_info, "backup_adv": jnp.mean(sampled.backup_adv)}
        return new, info

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
        return self.measure_loss(batch, sampled, self.phi_b.params, self.coef.params)

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
        """u*(s) = argmax_{m <= gpi_num_u} [min_P] (1/K_g) sum_{g in G} M(s, u_m, g) at the
        goal-averaged w, for ONE observation. Every candidate is a clipped prior draw."""
        c = self.config
        assert observations.ndim == 1, "select_latent acts on a single observation"
        K, d_a = int(c["gpi_num_u"]), c["action_dim"]
        Kg = self.eval_goals.shape[0]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        cols = jnp.broadcast_to(self.eval_goals[None], (K, Kg, self.eval_goals.shape[-1]))
        w = jnp.broadcast_to(self.eval_w[None], (K, self.eval_w.shape[-1]))
        Q = self._measure_cols(obs, u_cand, cols, w).mean(-1)                # (P, K)
        return u_cand[jnp.argmax(self._pessimistic(Q))]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """The deployed action: select a latent over the goal set, decode it through the flow."""
        seed = self.rng if seed is None else seed
        u_star = self.select_latent(observations, seed)
        return self.decode(observations[None], u_star[None])[0]

    # ------------------------------------------------------------------ inference
    def infer_eval_goals(self, next_observations, rewards):
        """Copy of this agent with the goal set G and w set from a relabelling batch.

        G = k_goals rows whose (shifted) reward exceeds `REWARDING_THRESHOLD`, drawn without
        replacement when enough exist (numpy's global stream, like the relabel batch itself);
        w = sum_{g in G} w*(g) / ||.||. Picked up by main.py / tools/eval_checkpoint.py via
        hasattr(agent, 'infer_eval_goals').
        """
        rewards = np.asarray(rewards).reshape(-1)
        rows = np.nonzero(rewards > REWARDING_THRESHOLD)[0]
        if rows.size == 0:
            raise ValueError("infer_eval_goals: the relabel batch has no rewarding row "
                             f"(reward > {REWARDING_THRESHOLD}); cannot build a goal set")
        k = int(self.config["k_goals"])
        pick = np.random.choice(rows, size=k, replace=rows.size < k)
        goals = jnp.asarray(np.asarray(next_observations)[pick], jnp.float32)
        w = self.coef(goals).sum(0)
        w = jnp.sqrt(float(self.config["z_dim"])) * w / jnp.maximum(jnp.linalg.norm(w), 1e-12)
        return self.replace(eval_goals=goals, eval_w=w)


def get_config():
    """Importable default config, mirrored by configs/agent/psmgoal.yaml."""
    import ml_collections

    return ml_collections.ConfigDict({
        "agent_name": "psmgoal",
        "batch_size": 1024,
        "z_dim": 128,               # D, the width of phi and of w*(g)
        "num_parallel": 2,          # P, the (phi, b) ensemble size
        "discount": 0.98,           # gamma; also the hindsight-goal geometric horizon
        "tau": 0.01,                # Polyak rate of the (phi, b) target
        "pessimism": 0.5,           # mean - pessimism * spread over P; = min at P=2
        "lr_phi": 1.0e-4,           # phi, b
        "lr_h": 1.0e-4,             # h (the coefficient encoder)
        "lr_l": 3.0e-5,             # l (the multiplier); <= lr_h
        "constraint_coef": 0.0,     # 0 = no multiplier term in loss_h and l is not stepped
        "n_backup": 8,              # prior draws the backup latent is the argmax over
        "n_neg": 32,                # C, negative columns per row
        "k_goals": 32,              # K_g, size of the eval goal set G
        "goal_random_frac": 0.3,    # fraction of hindsight goals replaced by random states
        "measure": {"hidden_dim": 512, "hidden_layers": 2},   # phi, b trunk on [s, u, s+]
        "coef": {"hidden_dim": 256, "hidden_layers": 2},      # h
        "mult": {"hidden_dim": 256, "hidden_layers": 2},      # l
        # Frozen behaviour flow (must match the Stage-A fql bc_only run).
        "flow": {"hidden_dims": (512, 512, 512, 512), "value_hidden_dims": (512, 512, 512, 512),
                 "layer_norm": False, "critic_layer_norm": True},
        "flow_ckpt_path": ml_collections.config_dict.placeholder(str),
        "flow_ckpt_epoch": ml_collections.config_dict.placeholder(int),
        "allow_untrained_flow": False,
        "preimage_path": ml_collections.config_dict.placeholder(str),
        "use_point_preimage": True,
        "u_clip": 3.0,              # typical-set clamp on every latent draw
        "gpi_num_u": 64,            # prior draws the acting argmax scans
        "gpi_decode": "onestep",    # onestep | ode
        "flow_decode_steps": 10,    # Euler steps for gpi_decode=ode
        "ob_dims": ml_collections.config_dict.placeholder(list),
        "action_dim": ml_collections.config_dict.placeholder(int),
        "encoder": ml_collections.config_dict.placeholder(str),
    })
