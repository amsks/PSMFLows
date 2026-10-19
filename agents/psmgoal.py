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
from utils.psm_common import _plain_config, polyak_update, pool_anchor, pool_hash, pool_sigma
from utils.psm_networks import (
    FactorizedMeasure,
    GoalCoefficient,
    LatentResidualActor,
    LogAlpha,
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


def tanh_gaussian_logprob(mu, log_std, u, scale):
    """log pi(u | s, w) of a GIVEN latent u under the actor's tanh-Gaussian.

    The mirror of `tanh_gaussian_sample`: invert the box rescale and the tanh squash to the
    pre-squash value, score it under N(mu, exp(log_std)), then correct by the tanh Jacobian.
    Same unscaled [-1, 1] convention as the sampler (the d_a*log(scale) box term is omitted).
    Used by the behaviour-clone extractors, which need the density at u_data, not a fresh draw.
    """
    tanh_pre = jnp.clip(u / scale, -1.0 + 1e-6, 1.0 - 1e-6)
    pre = jnp.arctanh(tanh_pre)
    logp = (-0.5 * ((pre - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)).sum(-1)
    logp = logp - jnp.log(1.0 - tanh_pre ** 2 + 1e-6).sum(-1)
    return logp


#: measure_form options. joint = RLUMeasure on [s, u, g]; factorized = A(s,u)^T f(g), beta(s,u)^T f(g).
MEASURE_FORMS = ("joint", "factorized")


def f_ortho_loss(f):
    """||E[f f^T] - I||_F^2 over a pool of features f (P, K)."""
    gram = f.T @ f / f.shape[0]
    return jnp.sum((gram - jnp.eye(gram.shape[0])) ** 2)


#: actor_value options. none = the pre-existing in-loop actor objectives (actor_objective).
ACTOR_VALUES = ("none", "measure_reward_raw", "measure_reward_softmax")


# ------------------------------------------------ Factored-FB induced-reward helpers
# Ported from Factored-FB (branch density-fb) impls/critics/density_fb.py: `_delta_phi`,
# `_sphere_or_neutral`, `_z_centered_moment`, `_z_centered_coef`. Pure functions so the tests
# can check them against numpy.

def parse_hit_spec(spec):
    """'a:b:eps[,a:b:eps...]' -> ((a, b, eps), ...): observation column blocks and kernel radii."""
    blocks = []
    for part in str(spec).split(","):
        a, b, e = part.split(":")
        blocks.append((int(a), int(b), float(e)))
    return tuple(blocks)


def kernel_reward(goals, pool, blocks):
    """Hindsight-goal reward on the pool: softmax_j(-0.5 sum_blocks ||(g - s+_j)/eps||^2).
    goals (B, ob), pool (M, ob) -> (B, M). Factored-FB `_delta_phi`."""
    xg = jnp.concatenate([goals[:, a:b] / e for (a, b, e) in blocks], axis=-1)
    xp = jnp.concatenate([pool[:, a:b] / e for (a, b, e) in blocks], axis=-1)
    logits = -0.5 * jnp.sum((xg[:, None, :] - xp[None, :, :]) ** 2, axis=-1)
    return jax.nn.softmax(logits, axis=-1)


def sphere_or_neutral(x, d, eps):
    """Exact sqrt(d) normalization; rows with ||x|| < eps (constant reward) map to 1_d."""
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    degenerate = nrm < eps
    safe = jnp.where(degenerate, jnp.ones_like(nrm), nrm)
    return jnp.where(degenerate, jnp.ones_like(x), x * jnp.sqrt(d * 1.0) / safe)


def ridge_task(f, mu, cov, r, lam_mult, eig_floor, const_eps):
    """Centered ridge fit of reward samples onto the state feature, projected to the sphere.

    f (M, D) feature of the pool, mu (D,), cov (D, D) the EMA moments, r (B, M) rewards on the
    pool. m_c = (1/M) sum_j (f_j - mu)(r_j - rbar); c = (C + lam I)^-1 m_c with C's spectrum
    floored at eig_floor*tr(C)/D and lam = lam_mult*tr(C)/D; w = sqrt(D) c/||c|| (or 1_D when
    ||c|| < const_eps). Returns (B, D), stop-gradded."""
    d = float(f.shape[-1])
    r2 = jnp.atleast_2d(r)
    dr = r2 - jnp.mean(r2, axis=-1, keepdims=True)
    m_c = jnp.einsum('bm,md->bd', dr, f - mu[None]) / r2.shape[-1]
    cov = 0.5 * (cov + cov.T)
    tr = jnp.trace(cov)
    ev, vec = jnp.linalg.eigh(cov)
    ev = jnp.maximum(ev, jnp.maximum(eig_floor * tr / d, 1e-12))
    lam = lam_mult * tr / d
    c = ((m_c @ vec) / (ev + lam)[None, :]) @ vec.T
    return jax.lax.stop_gradient(sphere_or_neutral(c, d, const_eps))


def induced_value(M, r, mode, temp):
    """The actor's value from the measure mesh M (N, P) and the induced reward r (N, P).
    raw: Q_i = (1/P) sum_j M_ij r_ij. softmax: Q_i = sum_j softmax_j(M_ij/temp) r_ij."""
    if mode == "measure_reward_raw":
        return jnp.mean(M * r, axis=1)
    if mode == "measure_reward_softmax":
        return jnp.sum(jax.nn.softmax(M / temp, axis=1) * r, axis=1)
    raise ValueError(f"unknown actor_value {mode!r}")


# ------------------------------------------------ goal-conditioned training (2026-10-01)
# docs/design/2026-10-01-psmgoal-goal-conditioned.md; ported from Factored-FB (branch
# density-fb) impls/critics/density_fb.py.

#: policy_index options. code = w(z) on the binary policy code; goal = w = h(g) on the row's goal.
#: bank (2026-10-19, docs/design/2026-10-19-pool-measure.md): a bank of n_bank policies; only n_bank=1
#: is implemented, the behaviour policy -- the bootstrap latent is one clipped prior draw (a BC draw,
#: FLOWBC_BOOT_KEY) and the coefficient is ONE shared learned w, read as the existing w(z) net at the
#: constant all-ones code BANK_CODE (no new net; the all-zero code would give enc(0) = 0, w = 0 at init).
#: No actor is trained (train_actor must be false); a plain policy evaluation of BC.
POLICY_INDEXES = ("code", "goal", "bank", "reward")
#: reward (2026-10-19, docs/design/2026-10-19-reward-index.md): Density FB's z arm. w_i is the centred ridge
#: solve of a reward on the projected-phi feature f (fb_feature at the slow tower fb_zenc): at training the
#: kernel reward of the row's hindsight goal over the batch's next states (fb_rand_frac of the rows a random
#: sphere draw), at test the relabel rewards (coef_source=ridge). Critic A's loss, bootstrap and flowbc
#: actor otherwise; the actor takes w. ridge_score picks the best-of-64 score at test time.
RIDGE_SCORES = ("goal", "reward")
REWARD_ACTOR_REWARDS = ("latent_centered", "kernel")
REWARD_INDEX_KEY = 107
REWARD_ACTOR_KEY = 108
BANK_CODE = 1.0
#: measure_loss options. squared = TD on the mesh; softmax = cross-entropy over the columns.
#: pool (2026-10-19, docs/design/2026-10-19-pool-measure.md): the softmax cross-entropy with the
#: batch's next states replaced by a FIXED reference pool of pool_size dataset states carried on
#: the agent (`pool_states`), and the one-hot on the pool state nearest the row's next state.
MEASURE_LOSSES = ("squared", "softmax", "pool")
#: pool_u_tv: rows of the batch the u-dependence diagnostic is computed on each update.
POOL_TV_ROWS = 32
#: fold_in constant of the fresh prior draw the pool_u_tv diagnostic compares the row's u against.
POOL_TV_KEY = 106
#: actor_input options. w = the coefficient; goal = the raw goal state.
ACTOR_INPUTS = ("w", "goal")
#: eval_goal_source options: where the actor readout's goal comes from.
EVAL_GOAL_SOURCES = ("relabel", "env")
#: eval_goal_pool options: the rows the goal set is drawn from. relabel = the relabel batch's
#: rewarding rows; dataset = the task dataset's rewarding rows, without replacement.
EVAL_GOAL_POOLS = ("relabel", "dataset")
#: Goal columns per network call when a goal set is scored in chunks (acting).
GOAL_CHUNK = 1024
#: (s, u, g) rows per network call when the frozen measure is cached on a pairs x goals mesh.
MESH_ROWS = 65536
#: Cached Lagrangian: goal columns per (s, u) pair that the constraint and the multiplier l
#: read. A goal set of at most this many goals is read whole.
LP_CONS_COLS = 256
#: coef_source options: where the test-time coefficient w comes from. lp = the Lagrangian;
#: amortized = h over the goal set; regression = least squares on phi; ridge = the actor_value
#: feature fit; trained = the coefficient the measure was trained with (policy_index=code).
COEF_SOURCES = ("lp", "amortized", "regression", "ridge", "trained")
#: coef_source values that exist only under policy_index=goal (eval only). hgoal_each: each
#: goal g_j is scored with its own coefficient h(g_j), no single w (2026-10-03). reward_gpi
#: (2026-10-15): the task is a reward on states r(s+), not a goal; each candidate coefficient
#: h(g_c) over k_goals candidate goals is read as the expected reward over reference states,
#: and the score is the max over candidates (GPI over goal indices).
GOAL_COEF_SOURCES = ("hgoal_each", "reward_gpi")
#: coef_source=pool_reward (2026-10-19, eval only; policy_index=goal or bank;
#: docs/design/2026-10-19-pool-measure.md): value by integrating the reward over a bank of
#: pool_eval_size relabel next states with importance weights omega_j = (class count in the relabel
#: batch) / (class count in the bank), n_r = min(pool_eval_size // 4, rewarding rows) of them
#: rewarding: Q(s,u;w) = sum_j omega_j exp(M(s,u,y_j;w)/t) r_j / sum_j omega_j exp(M/t), and the
#: acting score is max over the candidate coefficients (h(g_c) over the k_goals goals under the goal
#: index, the single w under the bank). The same formula under every measure_loss.
POOL_COEF_SOURCES = ("pool_reward",)
#: reward_gpi_cands options (coef_source=reward_gpi): which relabel rows are the candidate
#: goals. top = the k_goals highest-reward next states; random = k_goals uniform rows;
#: both = k_goals//2 top + the rest uniform from the remaining rows.
REWARD_GPI_CANDS = ("top", "random", "both")
#: bootstrap_source options (policy_index=goal). actor = the goal-fed actor's mode at (s', g);
#: data = the data's next latent u_{i+1} (batch['next_noise_preimage']), no actor trained.
#: docs/design/2026-10-03-psmgoal-data-bootstrap.md. argmax = the best of bootstrap_n clipped
#: prior draws at s', scored by the target measure on the row's goal (2026-10-07); no actor.
#: bc_mean = the target measure averaged over bootstrap_n clipped prior draws at s' (2026-10-13;
#: softmax: the mean of their softmax rows); no actor.
BOOTSTRAP_SOURCES = ("actor", "data", "argmax", "bc_mean", "frozen_delta")
#: bootstrap_source=frozen_delta (2026-10-17): the TD target latent is clip(eps + step) with step built from a
#: FROZEN flowbc delta loaded from another run (frozen_delta_path/epoch); no co-adaptation. plus = +delta
#: (critic A's target with a fixed field), minus = -delta, random = a random direction of the same length
#: per row, zero = clip(eps) (critic B). The live actor still trains (policy_index=goal needs it) but is
#: not read by the target.
FROZEN_DELTA_MODES = ("plus", "minus", "random", "zero")
FROZEN_DELTA_BOOT_KEY = 105
#: goal_sampling options (policy_index=goal, 2026-10-04). geometric = the hindsight goal at
#: offset ~ Geometric(1 - goal_discount) (utils/datasets.py hindsight_goal_idxs); uniform =
#: a uniform draw over the row's own s' and every later state of the same trajectory
#: (future_goal_idxs). The agent reads batch['goals'] either way; the dataset draws them.
GOAL_SAMPLINGS = ("geometric", "uniform")
#: coef_source=trained: how many policy codes z the mean of w(z) is taken over, and the key
#: they are drawn from (a code constant, so every eval of a checkpoint reads the same w).
TRAINED_COEF_CODES = 256
TRAINED_COEF_SEED = 0
#: actor_kind options (policy_index=goal, train_actor=true; 2026-10-05). tanh = the
#: tanh-Gaussian actor with the fb_bc_coeff anchor at the data latent (`actor_loss_goal`);
#: flowbc = u = clip(eps + delta(s, g, eps)), eps a prior draw, BC anchor at eps
#: (`actor_loss_flowbc`); dsrl = the tanh-Gaussian actor with no BC term and a learned
#: entropy weight, the f_psmflow dsrl_sac head (`actor_loss_dsrl`).
ACTOR_KINDS = ("tanh", "flowbc", "dsrl")
#: actor_value_kind options: the value the goal actor climbs. shaped = the kernel-weighted
#: measure over the batch's next states; point = M(s,u,g) at the actor goal itself
#: (softmax: the share of g among [g + the batch's next states]).
ACTOR_VALUE_KINDS = ("shaped", "point")
#: actor_goal_source: the goal the goal actor is trained at. future = a uniformly drawn later state
#: of the row's trajectory (batch['fb_goals']); measure = the row's measure goal batch['goals'].
ACTOR_GOAL_SOURCES = ("future", "measure")
#: bootstrap_switch (actor_kind=flowbc): when the bootstrap latent is one clipped prior draw (a BC
#: draw) instead of the actor's latent. prior_then_actor = before bootstrap_switch_step training
#: steps; actor_then_prior = from that step on. The actor trains throughout.
BOOTSTRAP_SWITCHES = ("none", "prior_then_actor", "actor_then_prior")
#: fold_in constant of the fresh prior draw eps the flowbc bootstrap uses at (s', g).
FLOWBC_BOOT_KEY = 103
#: fold_in constant of the bootstrap_n prior draws the argmax bootstrap scores at (s', g).
ARGMAX_BOOT_KEY = 104
#: name of the flowbc actor's output layer in its params (zero-initialised).
DELTA_OUT_LAYER = "out"


def softmax_td_loss(M, M_bar, discount, temp):
    """Cross-entropy between softmax_j(M_ij / temp) and the stop-gradded target
    (1 - discount) * onehot(j = i) + discount * softmax_j(M_bar_ij / temp), mean over rows.
    M, M_bar (N, S), S >= N: row i is (s_i, u_i) resp. (s'_i, u'_i), column j < N the batch's
    next state s+_j (so column i is the row's own s'_i) and columns j >= N the rows' future
    states under norm_future_k (S = N when it is 0). Both softmaxes ignore a per-row shift of
    their logits."""
    target = (1.0 - discount) * jnp.eye(M.shape[0], M.shape[1]) + discount * jax.nn.softmax(M_bar / temp, axis=-1)
    logp = jax.nn.log_softmax(M / temp, axis=-1)
    return -jnp.mean(jnp.sum(jax.lax.stop_gradient(target) * logp, axis=-1))


@flax.struct.dataclass
class InferenceState:
    """The mutable state of the Stage-2 coefficient optimization (RLU `init_inference`)."""
    w: Any                    # (z_dim,) the free coefficient, on the sqrt(z_dim) sphere
    w_opt: Any                # optax state for w
    l_params: Any          # l(s,u,g) params (softplus multiplier network)
    l_opt: Any             # optax state for the multiplier


def _load_frozen_delta_params(run_dir, epoch, template):
    """bootstrap_source=frozen_delta: read `delta.params` out of another psmgoal run's params_<epoch>.pkl
    (the full agent state dict) and restore it onto `template` (this agent's delta params pytree), so
    the architectures must match (actor.hidden_dim / hidden_layers, action_dim, ob_dim)."""
    import glob
    import pickle
    cands = glob.glob(run_dir)
    assert len(cands) == 1, f"frozen_delta_path matched {len(cands)} dirs: {cands}"
    path = f"{cands[0]}/params_{int(epoch)}.pkl"
    with open(path, "rb") as f:
        saved = pickle.load(f)["agent"]
    assert "delta" in saved and "params" in saved["delta"], f"{path} has no delta params (not a flowbc run?)"
    return flax.serialization.from_state_dict(template, saved["delta"]["params"])


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
    # Factored-FB induced-reward actor state (actor_value != none; carried but unused otherwise).
    fb_zenc: Any              # slow-EMA copy of the basis params: f(s+) is read from it
    fb_mu: Any                # (z_dim,) EMA mean of f over the pool
    fb_cov: Any               # (z_dim, z_dim) EMA centered covariance of f
    fb_anchor: Any            # (fb_k_anchor, ob) fixed anchor goal states, set on step 1
    fb_filled: Any            # scalar, 1 once the anchors are written
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)
    # Goal-conditioned training state (policy_index=goal / measure_loss=softmax; carried but
    # unused otherwise). A checkpoint written before these existed restores with fresh values.
    target_w_star: Any = None   # Polyak copy of h's params: w = h(g) in the target measure
    eval_ref: Any = None        # (batch_size - k_goals, ob) non-rewarding next states: the
                                # softmax readout's reference columns, set by infer_eval_goals
    eval_actor_goal: Any = None  # (ob,) the goal the actor readout is fed (actor_input=goal)
    eval_goal_w: Any = None      # (k_goals, z_dim) h(g_j) per eval goal (coef_source=hgoal_each)
    # Joint actor heads (2026-10-05; carried but unused unless actor_kind selects them).
    delta: Any = None            # actor_kind=flowbc: delta(s, g, eps) TrainState
    frozen_delta_params: Any = None   # bootstrap_source=frozen_delta: the loaded, never-updated delta params
    log_alpha: Any = None        # actor_kind=dsrl: SAC's log alpha TrainState
    # coef_source=reward_gpi (2026-10-15, eval only; carried but unused otherwise).
    eval_cand_w: Any = None      # (k_goals, z_dim) h(g_c) per candidate goal
    eval_ref_reward: Any = None  # (R,) r(s_j) at the reference states eval_ref
    # measure_loss=pool (2026-10-19): the reference pool R, fixed for the run, never trained. None
    # under the other losses. main.py / tools/eval_checkpoint.py draw it from the dataset's
    # valid-preimage rows by pool_seed (utils.psm_common.draw_pool_indices) and hand it to `create`.
    pool_states: Any = None      # (P, ob) the pool states y_j (the rows' next states)
    pool_indices: Any = None     # (P,) int32 dataset rows of the pool (sha1 -> config['pool_hash'])
    pool_scale: Any = None       # (ob,) sigma of the anchor distance: per-dimension std of pool_states
    # coef_source=pool_reward (2026-10-19, eval only; carried but unused otherwise), set by infer_eval_goals.
    eval_bank: Any = None        # (pool_eval_size, ob) the bank's next states y_j
    eval_bank_r: Any = None      # (pool_eval_size,) r(y_j), the shifted relabel reward
    eval_bank_w: Any = None      # (pool_eval_size,) omega_j, the class importance weights

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, pool_obs=None, pool_indices=None):
        """measure_loss=pool needs the reference pool: `pool_obs` (pool_size, ob) the pool rows' next
        states and `pool_indices` (pool_size,) their dataset rows, drawn by the caller
        (utils.psm_common.draw_pool_indices from the valid-preimage rows at pool_seed). The hash of
        the rows lands in config['pool_hash']; a config that already carries one (an eval reading
        the run's flags.json) must match. Both are ignored under the other losses."""
        assert config.get("encoder", None) is None, "psmgoal does not support visual encoders."
        coef_source = str(config.get("coef_source", "lp"))
        assert coef_source in COEF_SOURCES + GOAL_COEF_SOURCES + POOL_COEF_SOURCES, \
            "coef_source must be one of lp | amortized | regression | ridge | trained | hgoal_each | reward_gpi | pool_reward"
        assert str(config.get("reward_gpi_cands", "both")) in REWARD_GPI_CANDS, \
            f"reward_gpi_cands must be one of {REWARD_GPI_CANDS}"
        actor_value = str(config.get("actor_value", "none"))
        assert actor_value in ACTOR_VALUES, f"actor_value must be one of {ACTOR_VALUES}"
        if actor_value != "none":
            assert bool(config.get("train_actor", False)), \
                "actor_value != none trains the in-loop actor: set train_actor=true"
        if str(config.get("coef_source", "lp")) == "ridge":
            assert actor_value != "none" or str(config.get("policy_index", "code")) == "reward", \
                "coef_source=ridge reads the actor_value feature state (or policy_index=reward)"
        assert int(config["max_log_seed"]) >= 1, "max_log_seed (policy-code width) must be >= 1"
        assert int(config["k_goals"]) >= 1 and int(config["gpi_num_u"]) >= 1
        assert float(config["inf_coeff"]) >= 0.0
        measure_form = str(config.get("measure_form", "joint"))
        assert measure_form in MEASURE_FORMS, f"measure_form must be one of {MEASURE_FORMS}"
        if measure_form == "factorized":
            # fb_feature evaluates the basis per (row, u, anchor) triple, which would build
            # A(s,u) (f_dim x z_dim) per triple; not wired.
            assert actor_value == "none", "measure_form=factorized is not wired with actor_value"
        # goal-conditioned training switches (docs/design/2026-10-01-psmgoal-goal-conditioned.md)
        policy_index = str(config.get("policy_index", "code"))
        measure_loss = str(config.get("measure_loss", "squared"))
        actor_input = str(config.get("actor_input", "w"))
        assert policy_index in POLICY_INDEXES, f"policy_index must be one of {POLICY_INDEXES}"
        assert measure_loss in MEASURE_LOSSES, f"measure_loss must be one of {MEASURE_LOSSES}"
        assert actor_input in ACTOR_INPUTS, f"actor_input must be one of {ACTOR_INPUTS}"
        assert str(config.get("eval_goal_source", "relabel")) in EVAL_GOAL_SOURCES, \
            f"eval_goal_source must be one of {EVAL_GOAL_SOURCES}"
        assert str(config.get("eval_goal_pool", "relabel")) in EVAL_GOAL_POOLS, \
            f"eval_goal_pool must be one of {EVAL_GOAL_POOLS}"
        bootstrap_source = str(config.get("bootstrap_source", "actor"))
        assert bootstrap_source in BOOTSTRAP_SOURCES, f"bootstrap_source must be one of {BOOTSTRAP_SOURCES}"
        if bootstrap_source in ("argmax", "bc_mean"):
            assert policy_index == "goal", f"bootstrap_source={bootstrap_source} reads h(g): needs policy_index=goal"
            assert not bool(config.get("train_actor", False)), \
                f"bootstrap_source={bootstrap_source} replaces the actor in the bootstrap: set train_actor=false"
            assert int(config.get("bootstrap_n", 8)) >= 1, "bootstrap_n must be >= 1"
        if bootstrap_source == "frozen_delta":
            assert policy_index == "goal" and str(config.get("actor_kind", "tanh")) == "flowbc", \
                "bootstrap_source=frozen_delta applies a flowbc delta(s', g, eps): needs policy_index=goal, actor_kind=flowbc"
            assert str(config.get("frozen_delta_mode", "plus")) in FROZEN_DELTA_MODES, \
                f"frozen_delta_mode must be one of {FROZEN_DELTA_MODES}"
            assert config.get("frozen_delta_path", None) or config.get("allow_untrained_flow", False), \
                "bootstrap_source=frozen_delta needs frozen_delta_path (a run dir with params_<frozen_delta_epoch>.pkl)"
        if bootstrap_source == "data":
            # docs/design/2026-10-03-psmgoal-data-bootstrap.md: u' = u_{i+1} is a sample of the
            # goal's policy at s' only for a hindsight goal from the row's own future.
            assert policy_index == "goal", "bootstrap_source=data needs policy_index=goal (w = h(g))"
            assert float(config.get("goal_random_frac", 0.3)) == 0.0, \
                "bootstrap_source=data needs goal_random_frac=0 (goals from the row's own future only)"
            assert float(config.get("goal_cur_frac", 0.0)) == 0.0, \
                "bootstrap_source=data needs goal_cur_frac=0 (geometric hindsight goals only)"
        assert str(config.get("goal_sampling", "geometric")) in GOAL_SAMPLINGS, \
            f"goal_sampling must be one of {GOAL_SAMPLINGS}"
        assert int(config.get("gpi_hold", 1)) >= 1, "gpi_hold (env steps one gpi latent is held) must be >= 1"
        if coef_source in GOAL_COEF_SOURCES:
            assert policy_index == "goal", f"coef_source={coef_source} reads h(g) per goal: needs policy_index=goal"
        if coef_source == "reward_gpi":
            assert str(config.get("eval_goal_pool", "relabel")) == "relabel", \
                "coef_source=reward_gpi draws its candidates and reference states from the relabel batch: needs eval_goal_pool=relabel"
        if coef_source == "pool_reward":
            assert policy_index in ("goal", "bank"), \
                "coef_source=pool_reward reads h(g) per goal or the bank's w: needs policy_index=goal|bank"
            assert str(config.get("eval_goal_pool", "relabel")) == "relabel", \
                "coef_source=pool_reward draws its bank from the relabel batch: needs eval_goal_pool=relabel"
            assert int(config.get("pool_eval_size", 512)) >= 1, "pool_eval_size must be >= 1"
        if policy_index == "bank":
            assert int(config.get("n_bank", 1)) == 1, "policy_index=bank: only n_bank=1 (the behaviour policy) is implemented"
            assert not bool(config.get("train_actor", False)), \
                "policy_index=bank evaluates the behaviour policy: no actor is trained, set train_actor=false"
            assert bootstrap_source == "actor", (
                "policy_index=bank bootstraps from one clipped prior draw and ignores bootstrap_source: "
                "leave it at its default (actor)")
            assert not bool(config.get("train_goal_head", False)), "policy_index=bank has no goal head to train"
            assert coef_source not in GOAL_COEF_SOURCES, f"coef_source={coef_source} reads h(g): needs policy_index=goal"
        if policy_index == "goal":
            assert bool(config.get("train_actor", False)) or bootstrap_source in ("data", "argmax", "bc_mean", "frozen_delta"), \
                ("policy_index=goal bootstraps from the goal-fed actor: set train_actor=true "
                 "(or bootstrap_source=data|argmax|bc_mean|frozen_delta)")
            assert not bool(config.get("train_goal_head", False)), \
                "policy_index=goal trains h by the measure loss; train_goal_head must be false"
            assert actor_value == "none", "policy_index=goal has its own actor loss; set actor_value=none"
            assert str(config.get("coef_source", "lp")) != "trained", (
                "coef_source=trained reads w(z) on the policy code; under policy_index=goal the "
                "trained coefficient is h(g): use coef_source=amortized")
        if policy_index == "reward":
            # docs/design/2026-10-19-reward-index.md: Density FB's z arm in critic A's recipe.
            assert measure_loss == "softmax", "policy_index=reward uses critic A's softmax loss: set measure_loss=softmax"
            assert bool(config.get("train_actor", False)), \
                "policy_index=reward bootstraps from the w-fed flowbc actor: set train_actor=true"
            assert str(config.get("actor_kind", "tanh")) == "flowbc", "policy_index=reward: set actor_kind=flowbc"
            assert actor_input == "w", "policy_index=reward feeds the actor the index w: actor_input=w"
            assert bootstrap_source == "actor", "policy_index=reward bootstraps from the actor at (s', w): bootstrap_source=actor"
            assert actor_value == "none", "policy_index=reward has its own actor loss; set actor_value=none"
            assert not bool(config.get("train_goal_head", False)), "policy_index=reward has no goal head to train"
            assert int(config.get("norm_future_k", 0)) == 0, "policy_index=reward: norm_future_k=0 (first arm)"
            assert coef_source in ("ridge", "lp"), \
                "policy_index=reward is read at test time by coef_source=ridge (lp only as a diagnostic)"
            assert str(config.get("ridge_score", "goal")) in RIDGE_SCORES, f"ridge_score must be one of {RIDGE_SCORES}"
            assert str(config.get("reward_actor_reward", "latent_centered")) in REWARD_ACTOR_REWARDS, \
                f"reward_actor_reward must be one of {REWARD_ACTOR_REWARDS}"
            assert str(config.get("bootstrap_switch", "none")) == "none" and float(config.get("bootstrap_delta_scale", 1.0)) == 1.0, \
                "policy_index=reward: bootstrap_switch=none, bootstrap_delta_scale=1"
            assert str(config["acting"]) in ("gpi", "distill"), "policy_index=reward acts by gpi (best-of-64 at w) or distill (the actor at w)"
            if str(config["acting"]) == "distill":
                assert not bool(config.get("eval_redistill", True)), "policy_index=reward: the flowbc actor is not re-distilled; set eval_redistill=false"
        norm_future_k = int(config.get("norm_future_k", 0))
        assert norm_future_k >= 0, "norm_future_k must be >= 0"
        if norm_future_k > 0:
            assert measure_loss == "softmax", "norm_future_k > 0 extends the softmax normalisation set: set measure_loss=softmax"
        if measure_loss == "pool":
            # the pool REPLACES the candidate set; mixing it with the rows' future states is not defined
            assert norm_future_k == 0, "measure_loss=pool replaces the candidate set by the pool: needs norm_future_k=0"
            assert config.get("bootstrap_source", "actor") != "argmax", \
                "measure_loss=pool: the argmax bootstrap scores draws over the batch's next states, not the pool"
            assert pool_obs is not None and pool_indices is not None, \
                "measure_loss=pool needs the reference pool (create(..., pool_obs=, pool_indices=))"
            pool_obs = np.asarray(pool_obs, np.float32)
            pool_indices = np.asarray(pool_indices, np.int64).reshape(-1)
            assert pool_obs.ndim == 2 and pool_obs.shape[0] == pool_indices.shape[0] == int(config["pool_size"]), \
                f"pool_obs {pool_obs.shape} / pool_indices {pool_indices.shape} do not match pool_size={config['pool_size']}"
        if measure_loss in ("softmax", "pool"):
            assert measure_form != "factorized", "measure_loss=softmax is not wired with measure_form=factorized"
            assert float(config.get("ortho_coef", 0.0)) == 0.0, "measure_loss=softmax needs ortho_coef=0"
            assert int(config["batch_size"]) > int(config["k_goals"]), \
                "measure_loss=softmax reads batch_size - k_goals reference columns at test time"
        if actor_input == "goal":
            assert policy_index == "goal", "actor_input=goal needs policy_index=goal (the actor is fed the row's goal)"
            # Both of these feed the actor the eval coefficient, which a goal-fed actor cannot take.
            assert not (str(config["acting"]) == "distill" and bool(config.get("eval_redistill", True))), \
                "actor_input=goal cannot be re-distilled at eval: set eval_redistill=false"
            assert str(config["acting"]) != "sfbc", "actor_input=goal is not wired with acting=sfbc"
        # joint actor heads (2026-10-05)
        actor_kind = str(config.get("actor_kind", "tanh"))
        actor_value_kind = str(config.get("actor_value_kind", "shaped"))
        assert actor_kind in ACTOR_KINDS, f"actor_kind must be one of {ACTOR_KINDS}"
        assert actor_value_kind in ACTOR_VALUE_KINDS, f"actor_value_kind must be one of {ACTOR_VALUE_KINDS}"
        if actor_kind != "tanh":
            assert policy_index in ("goal", "reward"), f"actor_kind={actor_kind} is the goal actor: needs policy_index=goal|reward"
            assert bool(config.get("train_actor", False)), f"actor_kind={actor_kind} trains an actor: set train_actor=true"
            assert policy_index == "reward" or actor_input == "goal", \
                f"actor_kind={actor_kind} takes the raw goal: set actor_input=goal"
        if actor_kind == "dsrl":
            assert float(config.get("fb_bc_coeff", 3.0)) == 0.0, \
                "actor_kind=dsrl has no BC term: set fb_bc_coeff=0"
        if actor_value_kind == "point":
            assert policy_index == "goal", "actor_value_kind=point reads M(s,u,g) at w = h(g): needs policy_index=goal"
        # design-decision ablations (2026-10-09)
        actor_goal_source = str(config.get("actor_goal_source", "future"))
        assert actor_goal_source in ACTOR_GOAL_SOURCES, f"actor_goal_source must be one of {ACTOR_GOAL_SOURCES}"
        bootstrap_switch = str(config.get("bootstrap_switch", "none"))
        assert bootstrap_switch in BOOTSTRAP_SWITCHES, f"bootstrap_switch must be one of {BOOTSTRAP_SWITCHES}"
        if bootstrap_switch != "none":
            assert actor_kind == "flowbc" and str(config.get("bootstrap_source", "actor")) == "actor", \
                "bootstrap_switch swaps the flowbc actor's bootstrap latent: needs actor_kind=flowbc, bootstrap_source=actor"
        if str(config["acting"]) == "flowbc_gpi":
            assert actor_kind == "flowbc", "acting=flowbc_gpi draws its candidates from the flowbc actor"
            assert str(config.get("coef_source", "lp")) == "hgoal_each", \
                "acting=flowbc_gpi scores candidates per goal: needs coef_source=hgoal_each"
        rng = jax.random.PRNGKey(seed)
        rng, r_b, r_c, r_l, r_a, r_vf, r_os = jax.random.split(rng, 7)
        action_dim = ex_actions.shape[-1]
        z_dim = int(config["z_dim"])
        code_dim = int(config["max_log_seed"])
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))
        ex_z = jnp.zeros((ex_observations.shape[0], code_dim))
        ex_w = jnp.zeros((ex_observations.shape[0], z_dim))   # actor's z-input is the coefficient w (z_dim)

        if measure_form == "factorized":
            basis_def = FactorizedMeasure(z_dim=z_dim, f_dim=int(config.get("f_dim", 128)),
                                          hidden_dim=config["measure"]["hidden_dim"],
                                          hidden_layers=config["measure"]["hidden_layers"])
        else:
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
        # actor_input=goal: the actor's second input is the raw goal state, not the coefficient.
        ex_actor_in = ex_observations if actor_input == "goal" else ex_w
        actor = TrainState.create(actor_def, actor_def.init(r_a, ex_observations, ex_actor_in)["params"],
                                  tx=optax.adam(config["lr_actor"]))
        # h(g): goal -> sqrt(D) sphere. Always built so a restore has the slot; trained only
        # when train_goal_head.
        rng, r_g = jax.random.split(rng)
        gc = config.get("w_star", {"hidden_dim": 256, "hidden_layers": 2})
        w_star_def = GoalCoefficient(z_dim=z_dim, hidden_dim=gc["hidden_dim"],
                                        hidden_layers=gc["hidden_layers"])
        w_star = TrainState.create(w_star_def, w_star_def.init(r_g, ex_observations)["params"],
                                      tx=optax.adam(config.get("lr_goal", 1.0e-4)))
        # Joint actor heads (2026-10-05). Keyed off r_a by fold_in so the rng chain above is
        # unchanged (the default agent stays bit-identical). Always built so a restore has
        # the slots; trained only under their actor_kind.
        delta_def = LatentResidualActor(action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
                                        hidden_layers=config["actor"]["hidden_layers"])
        # policy_index=reward: delta's goal slot takes the index w (z_dim) instead of a goal state.
        ex_delta_cond = ex_w if policy_index == "reward" else ex_observations
        delta = TrainState.create(
            delta_def, delta_def.init(jax.random.fold_in(r_a, 1), ex_observations, ex_delta_cond, ex_u)["params"],
            tx=optax.adam(config["lr_actor"]))
        # bootstrap_source=frozen_delta: load delta's params from another run and keep them fixed.
        frozen_delta_params = None
        if str(config.get("bootstrap_source", "actor")) == "frozen_delta":
            fpath = config.get("frozen_delta_path", None)
            if fpath:
                frozen_delta_params = _load_frozen_delta_params(
                    str(fpath), int(config.get("frozen_delta_epoch", 500000)), delta.params)
            else:   # tests/smokes only (allow_untrained_flow): start from delta's own init, replace() later
                frozen_delta_params = copy.deepcopy(delta.params)
        dsrl = config.get("dsrl_actor", {"target_entropy": 0.0, "init_alpha": 1.0, "lr_alpha": 3.0e-4})
        log_alpha_def = LogAlpha(init_value=float(np.log(float(dsrl["init_alpha"]))))
        log_alpha = TrainState.create(log_alpha_def, log_alpha_def.init(jax.random.fold_in(r_a, 2))["params"],
                                      tx=optax.adam(float(dsrl["lr_alpha"])))

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
        pool_states = pool_scale = pool_rows = None
        if measure_loss == "pool":
            h = pool_hash(pool_indices)
            recorded = config.get("pool_hash", None)
            assert recorded in (None, "") or str(recorded) == h, (
                f"pool_hash mismatch: the config records {recorded!r} but the pool handed to create hashes "
                f"to {h!r} (different dataset, pool_seed or pool_size than the run was trained with)")
            config["pool_hash"] = h
            pool_states = jnp.asarray(pool_obs, jnp.float32)
            pool_scale = pool_sigma(pool_states)
            pool_rows = jnp.asarray(pool_indices, jnp.int32)
        # Factored-FB state: the index tower starts as a copy of the basis, mu = 0, C = I
        # (density_fb _ZStatsState init). Built always so a restore has the slot.
        k_anchor = int(config.get("fb_k_anchor", 32))
        return cls(rng=rng, basis=basis, w=w, l=l, actor=actor, w_star=w_star,
                   target_basis=copy.deepcopy(basis.params),
                   target_w=copy.deepcopy(w.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   eval_goals=jnp.zeros((int(config["k_goals"]), ob_dim), jnp.float32),
                   eval_w_star=jnp.zeros((z_dim,), jnp.float32),
                   fb_zenc=copy.deepcopy(basis.params),
                   fb_mu=jnp.zeros((z_dim,), jnp.float32),
                   fb_cov=jnp.eye(z_dim, dtype=jnp.float32),
                   fb_anchor=jnp.zeros((k_anchor, ob_dim), jnp.float32),
                   fb_filled=jnp.zeros((), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def,
                   target_w_star=copy.deepcopy(w_star.params),
                   eval_ref=jnp.zeros((max(int(config["batch_size"]) - int(config["k_goals"]), 1),
                                       ob_dim), jnp.float32),
                   eval_actor_goal=jnp.zeros((ob_dim,), jnp.float32),
                   eval_goal_w=jnp.zeros((int(config["k_goals"]), z_dim), jnp.float32),
                   delta=delta, log_alpha=log_alpha, frozen_delta_params=frozen_delta_params,
                   eval_cand_w=jnp.zeros((int(config["k_goals"]), z_dim), jnp.float32),
                   eval_ref_reward=jnp.zeros((max(int(config["batch_size"]) - int(config["k_goals"]), 1),),
                                             jnp.float32),
                   pool_states=pool_states, pool_indices=pool_rows, pool_scale=pool_scale,
                   eval_bank=jnp.zeros((int(config.get("pool_eval_size", 512)), ob_dim), jnp.float32),
                   eval_bank_r=jnp.zeros((int(config.get("pool_eval_size", 512)),), jnp.float32),
                   eval_bank_w=jnp.ones((int(config.get("pool_eval_size", 512)),), jnp.float32))

    # ------------------------------------------------------------------ the measure
    def M(self, obs, u, g, w, params=None):
        """M(s,u,g) = phi(s,u,g)^T w + b -> (B,). Single-point measure; the mesh_M reference
        used in tests. w is (B, z_dim) or (z_dim,)."""
        phi, b = self.basis(obs, u, g, params=self.basis.params if params is None else params)
        return (phi * jnp.atleast_2d(w)).sum(-1) + b

    def mesh_phi_b(self, obs, u, goals, params=None):
        """The state x goal mesh of the basis: phi_ij = phi(s_i,u_i,g_j) (N, G, z_dim) and
        b_ij = b(s_i,u_i,g_j) (N, G). obs, u are (N, .) rows; goals (G, ob)."""
        # all-pairs state x goal mesh (RLU discrete_psm.py:374-384): row i = (s_i,u_i), col j = g_j.
        if self._factorized():
            # phi_ij = A(s_i,u_i)^T f(g_j), b_ij = beta(s_i,u_i)^T f(g_j): A once per row.
            p = self.basis.params if params is None else params
            A, beta = self.basis(obs, u, params=p, method="operators")       # (N, K, D), (N, K)
            f = self.basis(goals, params=p, method="features")               # (G, K)
            return jnp.einsum('nkd,gk->ngd', A, f), jnp.einsum('nk,gk->ng', beta, f)
        N, G = obs.shape[0], goals.shape[0]
        obs_r = jnp.broadcast_to(obs[:, None], (N, G, obs.shape[-1])).reshape(N * G, -1)
        u_r = jnp.broadcast_to(u[:, None], (N, G, u.shape[-1])).reshape(N * G, -1)
        g_r = jnp.broadcast_to(goals[None], (N, G, goals.shape[-1])).reshape(N * G, -1)
        # (phi, b) from the single-critic basis; params=None -> online params.
        phi, b = self.basis(obs_r, u_r, g_r, params=self.basis.params if params is None else params)
        return phi.reshape(N, G, -1), b.reshape(N, G)

    def mesh_M(self, obs, u, goals, w_rows, params=None):
        """The state x goal mesh: M_ij = phi(s_i,u_i,g_j)^T w_i + b(s_i,u_i,g_j).

        obs, u are (N, .) rows; goals (G, ob); w_rows (N, z_dim). Returns (N, G).
        """
        phi, b = self.mesh_phi_b(obs, u, goals, params=params)
        return (phi * w_rows[:, None, :]).sum(-1) + b

    def _factorized(self):
        return str(self.config.get("measure_form", "joint")) == "factorized"

    def _goal_index(self):
        return str(self.config.get("policy_index", "code")) == "goal"

    def _bank_index(self):
        return str(self.config.get("policy_index", "code")) == "bank"

    def _reward_index(self):
        return str(self.config.get("policy_index", "code")) == "reward"

    def _ridge_reward_score(self):
        return self._reward_index() and str(self.config.get("ridge_score", "goal")) == "reward"

    def bank_w(self, params=None, target=False):
        """policy_index=bank: the single shared coefficient, (1, z_dim): the w(z) net at the constant code
        BANK_CODE (on the sqrt(D) sphere like every w(z)). target=True reads the Polyak copy."""
        code = jnp.full((1, int(self.config["max_log_seed"])), BANK_CODE, jnp.float32)
        return self.w(code, params=self.target_w if target else params)

    def _softmax(self):
        """The measure is a distribution over candidate states and is read as shares: measure_loss=softmax
        and measure_loss=pool (the pool loss is the softmax cross-entropy over the pool). Every readout
        and actor value that branches on this treats both the same; the loss itself branches on `_pool`."""
        return str(self.config.get("measure_loss", "squared")) in ("softmax", "pool")

    def _pool(self):
        return str(self.config.get("measure_loss", "squared")) == "pool"

    def pool_anchors(self, next_obs):
        """measure_loss=pool: the anchor of each next state, the pool state nearest to it in the pool's
        per-dimension scale (utils.psm_common.pool_anchor; first index on ties). (N,) int32."""
        return pool_anchor(next_obs, self.pool_states, self.pool_scale)

    def _actor_takes_goal(self):
        return str(self.config.get("actor_input", "w")) == "goal"

    def _data_bootstrap(self):
        return str(self.config.get("bootstrap_source", "actor")) == "data"

    def _argmax_bootstrap(self):
        return str(self.config.get("bootstrap_source", "actor")) == "argmax"

    def _frozen_delta_bootstrap(self):
        return str(self.config.get("bootstrap_source", "actor")) == "frozen_delta"

    def _bc_mean_bootstrap(self):
        return str(self.config.get("bootstrap_source", "actor")) == "bc_mean"

    def _hgoal_each(self):
        return str(self.config.get("coef_source", "lp")) == "hgoal_each"

    def _reward_gpi(self):
        return str(self.config.get("coef_source", "lp")) == "reward_gpi"

    def _pool_reward(self):
        return str(self.config.get("coef_source", "lp")) == "pool_reward"

    def _actor_kind(self):
        return str(self.config.get("actor_kind", "tanh"))

    def _point_value(self):
        return str(self.config.get("actor_value_kind", "shaped")) == "point"

    def _coef(self, z, goals, params=None, target=False):
        """The per-row coefficient w of the measure, (N, z_dim). policy_index=code: w(z) on the
        binary policy code z. policy_index=goal: w = h(g) on the row's goal, z unused.
        policy_index=bank: the one shared w (`bank_w`) broadcast to every row; z, goals unused.
        target=True reads the Polyak copy; otherwise `params` (None -> the online params)."""
        if self._goal_index():
            return self.w_star(goals, params=self.target_w_star if target else params)
        if self._bank_index():
            n = (z if z is not None else goals).shape[0]
            return jnp.broadcast_to(self.bank_w(params=params, target=target), (n, int(self.config["z_dim"])))
        return self.w(z, params=self.target_w if target else params)

    def _goal_score(self, obs, u, goals, refs, w_rows, params=None):
        """The test-time score of each (s_i, u_i) row for the goal set, (N,).
        measure_loss=squared: mean_j M(s_i,u_i,g_j) over the goal columns.
        measure_loss=softmax: the softmax mass on the goal columns over [goal columns +
        reference next states], sum_{j in goals} softmax_j(M_ij / measure_temp); the logits
        are only defined up to a per-row shift, so their mean is not a score."""
        if self._softmax():
            cols = jnp.concatenate([goals, refs], axis=0)
            p = jax.nn.softmax(self.mesh_M(obs, u, cols, w_rows, params=params)
                               / float(self.config["measure_temp"]), axis=1)
            return jnp.sum(p[:, :goals.shape[0]], axis=1)
        return self._goal_mean_M(obs, u, goals, w_rows, params=params)

    def _goal_mean_M(self, obs, u, goals, w_rows, params=None, chunk=None):
        """mean_j M(s_i,u_i,g_j) over the goal columns, (N,). A goal set of more than `chunk`
        (GOAL_CHUNK) goals is scored `chunk` columns per network call, so memory does not
        grow with the goal set."""
        chunk = GOAL_CHUNK if chunk is None else int(chunk)
        G = goals.shape[0]
        if G <= chunk:
            return jnp.mean(self.mesh_M(obs, u, goals, w_rows, params=params), axis=1)

        def col_sum(g):
            return jnp.sum(self.mesh_M(obs, u, g, w_rows, params=params), axis=1)

        n_full = G // chunk
        total = jax.lax.map(col_sum, goals[:n_full * chunk].reshape(n_full, chunk, goals.shape[-1])).sum(axis=0)
        if G % chunk:
            total = total + col_sum(goals[n_full * chunk:])
        return total / G

    def _hgoal_each_score(self, obs, u, goals, goal_w, refs, params=None, chunk=None):
        """coef_source=hgoal_each: each goal scored with its own coefficient h(g_j), (N,).
        measure_loss=squared: mean_j M(s_i,u_i,g_j; h(g_j)).
        measure_loss=softmax: for each j the share of g_j among [g_j + refs] at w = h(g_j),
        exp(M_ij/t) / (exp(M_ij/t) + sum_r exp(M(s_i,u_i,ref_r; h(g_j))/t)), averaged over j.
        goal_w (G, z_dim) is h(g_j). Goals are scored `chunk` (GOAL_CHUNK) columns per network
        call; phi, b at the reference columns are computed once."""
        chunk = GOAL_CHUNK if chunk is None else int(chunk)
        G = goals.shape[0]
        temp = float(self.config["measure_temp"])
        softmax = self._softmax()
        if softmax:
            phi_r, b_r = self.mesh_phi_b(obs, u, refs, params=params)      # (N, R, z), (N, R)

        def col_sum(g, hw):
            phi_g, b_g = self.mesh_phi_b(obs, u, g, params=params)         # (N, C, z), (N, C)
            m_g = jnp.einsum('ncz,cz->nc', phi_g, hw) + b_g                # M(s_i,u_i,g_j; h_j)
            if not softmax:
                return jnp.sum(m_g, axis=1)
            m_r = jnp.einsum('nrz,cz->ncr', phi_r, hw) + b_r[:, None, :]  # (N, C, R) at h_j
            lse = jax.nn.logsumexp(jnp.concatenate([m_g[:, :, None], m_r], axis=2) / temp, axis=2)
            return jnp.sum(jnp.exp(m_g / temp - lse), axis=1)

        if G <= chunk:
            return col_sum(goals, goal_w) / G
        n_full = G // chunk
        head = (goals[:n_full * chunk].reshape(n_full, chunk, goals.shape[-1]),
                goal_w[:n_full * chunk].reshape(n_full, chunk, goal_w.shape[-1]))
        total = jax.lax.map(lambda gw: col_sum(*gw), head).sum(axis=0)
        if G % chunk:
            total = total + col_sum(goals[n_full * chunk:], goal_w[n_full * chunk:])
        return total / G

    def _reward_gpi_values(self, obs, u, cand_w, refs, ref_reward, params=None, chunk=None):
        """coef_source=reward_gpi: the value of each (s_i, u_i) row under each candidate
        coefficient h_c, read as expected reward over the reference states, (N, C).
        measure_loss=softmax: m_icr = phi(s_i,u_i,ref_r)^T h_c + b; value_ic = sum_r
        softmax_r(m_icr / measure_temp) r_r (the shares sum to one, so a per-(s,u) shift of M
        drops out). measure_loss=squared: value_ic = mean_r m_icr r_r.
        cand_w (C, z_dim) is h(g_c); ref_reward (R,) is r at refs (R, ob). Candidates are
        scored `chunk` (GOAL_CHUNK) at a time; phi, b at the reference columns are computed
        once."""
        chunk = GOAL_CHUNK if chunk is None else int(chunk)
        C = cand_w.shape[0]
        temp = float(self.config["measure_temp"])
        softmax = self._softmax()
        phi_r, b_r = self.mesh_phi_b(obs, u, refs, params=params)          # (N, R, z), (N, R)
        r = jnp.asarray(ref_reward, jnp.float32).reshape(-1)

        def values(hw):
            m_r = jnp.einsum('nrz,cz->ncr', phi_r, hw) + b_r[:, None, :]  # (N, C', R) at h_c
            if not softmax:
                return jnp.einsum('ncr,r->nc', m_r, r) / r.shape[0]
            share = jax.nn.softmax(m_r / temp, axis=2)
            return jnp.einsum('ncr,r->nc', share, r)

        if C <= chunk:
            return values(cand_w)
        n_full = C // chunk
        head = jax.lax.map(values, cand_w[:n_full * chunk].reshape(n_full, chunk, cand_w.shape[-1]))
        out = jnp.transpose(head, (1, 0, 2)).reshape(obs.shape[0], n_full * chunk)   # (N, n_full*chunk)
        if C % chunk:
            out = jnp.concatenate([out, values(cand_w[n_full * chunk:])], axis=1)
        return out

    def _reward_gpi_score(self, obs, u, cand_w, refs, ref_reward, params=None, chunk=None):
        """coef_source=reward_gpi: max over the candidate coefficients of `_reward_gpi_values`
        (GPI over goal indices), (N,)."""
        return jnp.max(self._reward_gpi_values(obs, u, cand_w, refs, ref_reward, params=params, chunk=chunk),
                       axis=1)

    def _pool_reward_score(self, obs, u, cand_w, bank, bank_r, bank_w, params=None):
        """coef_source=pool_reward: the acting score of each (s_i, u_i) row, (N,):
        max_c Q_ic with Q_ic = sum_j omega_j exp(m_icj / t) r_j / sum_j omega_j exp(m_icj / t),
        m_icj = phi(s_i,u_i,y_j)^T w_c + b(s_i,u_i,y_j) over the bank y_j (B, ob), r_j = bank_r, omega_j =
        bank_w, t = measure_temp. Computed as softmax_j(m/t + log omega_j) . r, so a per-(s,u) shift of
        M drops out and the weights never divide by zero (every bank row's class is present in the
        relabel batch, so omega_j > 0). cand_w (C, z_dim): h(g_c) under policy_index=goal, the single
        w under bank. With no rewarding row r = 0 everywhere and the score is exactly 0."""
        temp = float(self.config["measure_temp"])
        phi_r, b_r = self.mesh_phi_b(obs, u, bank, params=params)              # (N, B, z), (N, B)
        m = jnp.einsum('nbz,cz->ncb', phi_r, cand_w) + b_r[:, None, :]          # (N, C, B)
        log_w = jnp.log(jnp.maximum(jnp.asarray(bank_w, jnp.float32).reshape(-1), 1e-30))
        share = jax.nn.softmax(m / temp + log_w[None, None, :], axis=2)
        q = jnp.einsum('ncb,b->nc', share, jnp.asarray(bank_r, jnp.float32).reshape(-1))
        return jnp.max(q, axis=1)

    def _reward_gpi_argmax(self, obs, u, cand_w, refs, ref_reward, params=None, chunk=None):
        """The candidate index each row's `_reward_gpi_score` comes from, (N,) int. Diagnostics."""
        return jnp.argmax(self._reward_gpi_values(obs, u, cand_w, refs, ref_reward, params=params, chunk=chunk),
                          axis=1)

    # ------------------------------------------------------------------ proto bootstrap
    def proto_bootstrap(self, z, index):
        """u^+: the fixed z-indexed policy's latent at each row. A clipped prior draw keyed on
        (z, row), so G(s', u^+) stays in-support. RLU SamplingSeedActor, discrete_psm.py:137-171.
        Returns (N, d_a)."""
        c = self.config
        seeds = proto_seed_ints(z, index, int(c["max_log_seed"]))
        base = jax.random.PRNGKey(int(c["proto_seed"]))
        return proto_latents(seeds, c["action_dim"], c["u_clip"], base)

    def actor_backup(self, next_obs, w):
        """u^+ from the actor a_theta(s', w): the deterministic mode u_clip*tanh(mu), clipped to
        the latent box. Used as the TD backup when train_actor (the FB idea: the bootstrap
        action is the actor's, not a fixed proto draw). Returns (N, d_a)."""
        u_clip = float(self.config["u_clip"])
        mu, _ = self.actor(next_obs, w)
        return jnp.clip(u_clip * jnp.tanh(mu), -u_clip, u_clip)

    def _select_u_next(self, batch, z):
        """The bootstrap latent u^+. train_actor -> the actor's draw at the target coefficient
        self.w(z, target_w) (matches the target measure); else the fixed z-indexed proto draw.
        policy_index=goal: the actor's mode at (s'_i, g_i), g_i the row's own measure goal.
        bootstrap_source=data: the data's next latent u_{i+1} (batch['next_noise_preimage'],
        the move the data made at s'_i), clipped like the row's own u.
        bootstrap_source=bc_mean: all bootstrap_n clipped prior draws, (n, N, d); measure_loss
        averages the target over them."""
        c = self.config
        if self._reward_index():
            # the flowbc actor at (s', w_i), w_i the row's index (batch['index_w'], `reward_index_rows`)
            next_obs = jnp.asarray(batch["next_observations"])
            eps = jax.random.normal(jax.random.fold_in(self.rng, FLOWBC_BOOT_KEY), (next_obs.shape[0], int(c["action_dim"])))
            return jax.lax.stop_gradient(self.flowbc_latent(next_obs, jnp.asarray(batch["index_w"]), eps)[0])
        if self._bank_index():
            # the behaviour policy: one clipped prior draw at s', the same key stream as the flowbc bootstrap
            # (so it equals that bootstrap with delta = 0)
            next_obs = jnp.asarray(batch["next_observations"])
            eps = jax.random.normal(jax.random.fold_in(self.rng, FLOWBC_BOOT_KEY), (next_obs.shape[0], int(c["action_dim"])))
            return jax.lax.stop_gradient(jnp.clip(eps, -c["u_clip"], c["u_clip"]))
        if self._data_bootstrap():
            return jnp.clip(jnp.asarray(batch["next_noise_preimage"]), -c["u_clip"], c["u_clip"])
        if self._argmax_bootstrap():
            return self.argmax_bootstrap(batch)[0]
        if self._bc_mean_bootstrap():
            return self.prior_draws(jnp.asarray(batch["next_observations"]).shape[0])
        if self._frozen_delta_bootstrap():
            return self.frozen_delta_bootstrap(batch)
        if bool(c.get("train_actor", False)):
            next_obs = jnp.asarray(batch["next_observations"])
            goals = jnp.asarray(batch["goals"]) if self._goal_index() else None
            if self._actor_takes_goal():
                if self._actor_kind() == "flowbc":
                    # the flowbc actor at (s', g) with a fresh prior draw eps
                    eps = jax.random.normal(jax.random.fold_in(self.rng, FLOWBC_BOOT_KEY),
                                            (next_obs.shape[0], int(c["action_dim"])))
                    u_actor, d_actor = self.flowbc_latent(next_obs, goals, eps)
                    # bootstrap_delta_scale (2026-10-15): scale the actor's step in the TD target
                    # ONLY (acting and the actor loss keep the unscaled step). 1.0 = unchanged.
                    m = float(c.get("bootstrap_delta_scale", 1.0))
                    if m != 1.0:
                        u_actor = jnp.clip(eps + m * d_actor, -c["u_clip"], c["u_clip"])
                    switch = str(c.get("bootstrap_switch", "none"))
                    if switch == "none":
                        return u_actor
                    # the BC draw: the same eps, clipped, without the actor's delta
                    u_prior = jnp.clip(eps, -c["u_clip"], c["u_clip"])
                    # basis.step starts at 1 and counts updates, so it equals main.py's step
                    early = self.basis.step < int(c["bootstrap_switch_step"])
                    use_prior = early if switch == "prior_then_actor" else jnp.logical_not(early)
                    return jnp.where(use_prior, u_prior, u_actor)
                # tanh and dsrl: the tanh-Gaussian actor's mode
                return self.actor_backup(next_obs, goals)
            return self.actor_backup(next_obs, self._coef(z, goals, target=True))
        N = jnp.asarray(batch["observations"]).shape[0]
        index = jnp.asarray(batch["index"]) if "index" in batch else jnp.arange(N)
        return self.proto_bootstrap(z, index)

    def frozen_delta_bootstrap(self, batch):
        """bootstrap_source=frozen_delta (2026-10-17). eps ~ N(0, I) from the same key stream as the
        flowbc actor bootstrap (so mode=plus with the live params would be bit-identical to it);
        d = delta_frozen(s', g, eps) from the loaded params; the target latent is
          plus:   clip(eps + d)        minus: clip(eps - d)
          random: clip(eps + ||d|| * r/||r||), r ~ N(0, I) per row (the step's length, a random direction)
          zero:   clip(eps)
        Lengths are matched before the clip (the clip touches ~1% of rows at u_clip=3)."""
        c = self.config
        next_obs = jnp.asarray(batch["next_observations"])
        goals = jnp.asarray(batch["goals"])
        N, d_a = next_obs.shape[0], int(c["action_dim"])
        eps = jax.random.normal(jax.random.fold_in(self.rng, FLOWBC_BOOT_KEY), (N, d_a))
        d = self.delta(next_obs, goals, eps, params=self.frozen_delta_params)
        mode = str(c.get("frozen_delta_mode", "plus"))
        if mode == "plus":
            step = d
        elif mode == "minus":
            step = -d
        elif mode == "zero":
            step = jnp.zeros_like(d)
        else:   # random: a random direction with the frozen step's length, per row
            r = jax.random.normal(jax.random.fold_in(self.rng, FROZEN_DELTA_BOOT_KEY), (N, d_a))
            r = r / jnp.maximum(jnp.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
            step = r * jnp.linalg.norm(d, axis=-1, keepdims=True)
        u_clip = float(c["u_clip"])
        return jax.lax.stop_gradient(jnp.clip(eps + step, -u_clip, u_clip))

    def argmax_bootstrap(self, batch):
        """bootstrap_source=argmax: u'_i = the best of bootstrap_n clipped prior draws at s'_i,
        scored by the TARGET measure at w_i = h_target(g_i) on the row's goal g_i. Softmax loss:
        the share of g_i among [g_i + the batch's next states] (the point actor value on target
        params); squared loss: M_bar(s'_i, u, g_i). Every candidate is a behaviour-cloning draw,
        so u' stays on the flow's support; bootstrap_n=1 is a plain BC draw. The draws are
        scored one at a time (lax.map) so memory stays at one target mesh.
        Returns (u' (N, d), scores (n, N)), stop-gradded."""
        c = self.config
        next_obs = jnp.asarray(batch["next_observations"])
        goals = jnp.asarray(batch["goals"])
        N, temp = next_obs.shape[0], float(c["measure_temp"])
        eps = self.prior_draws(N)
        w_t = self._coef(None, goals, target=True)                               # (N, z)

        def score(u):
            m_g = self.M(next_obs, u, goals, w_t, params=self.target_basis)       # (N,)
            if not self._softmax():
                return m_g
            M = self.mesh_M(next_obs, u, next_obs, w_t, params=self.target_basis)  # (N, N)
            lse = jax.nn.logsumexp(jnp.concatenate([m_g[:, None], M], axis=1) / temp, axis=1)
            return jnp.exp(m_g / temp - lse)

        scores = jax.lax.map(score, eps)                                          # (n, N)
        best = jnp.argmax(scores, axis=0)                                         # (N,)
        u_next = eps[best, jnp.arange(N)]
        return jax.lax.stop_gradient(u_next), jax.lax.stop_gradient(scores)

    def prior_draws(self, N):
        """bootstrap_n clipped prior draws per row, (n, N, d), from the ARGMAX_BOOT_KEY stream that
        bootstrap_source=argmax and bc_mean share (so bootstrap_n=1 is the same BC draw in both)."""
        c = self.config
        n, d, u_clip = int(c.get("bootstrap_n", 8)), int(c["action_dim"]), float(c["u_clip"])
        eps = jax.random.normal(jax.random.fold_in(self.rng, ARGMAX_BOOT_KEY), (n, N, d))
        return jax.lax.stop_gradient(jnp.clip(eps, -u_clip, u_clip))

    def bc_mean_target(self, next_obs, eps, goals, w_t):
        """bootstrap_source=bc_mean: the target measure at s' averaged over the BC draws eps
        (n, N, d), as (N, N) target logits. Squared loss: mean_n M_bar(s', u_n, s+). Softmax loss:
        temp * log mean_n softmax_j(M_bar_n / temp), so softmax_j(target / temp) is the mean of the
        draws' softmax rows (the successor measure of the BC policy at s'), not the softmax of the
        mean logit. n = 1 returns the single draw's target up to a per-row shift."""
        Ms = jax.lax.map(lambda u: self.mesh_M(next_obs, u, goals, w_t, params=self.target_basis), eps)
        if not self._softmax():
            return Ms.mean(0)
        temp = float(self.config["measure_temp"])
        logp = jax.nn.log_softmax(Ms / temp, axis=-1)                          # (n, N, N)
        return temp * (jax.nn.logsumexp(logp, axis=0) - jnp.log(eps.shape[0]))

    # ------------------------------------------------------------------ basis loss
    def measure_loss(self, basis_params, w_params, batch, z, u_next):
        """RLU `update_psm`: squared TD on the off-diagonal of the state x goal mesh plus a
        `-(1-gamma)` diagonal pull. Gradient reaches phi, b (basis) and w (w).

        policy_index=goal: the row's coefficient is w_i = h(g_i) on its measure goal
        batch['goals'] (w_params are then h's params, and the target reads h's Polyak copy);
        the columns stay the batch's next states. measure_loss=softmax swaps the squared TD
        for `softmax_td_loss` on the same mesh.

        norm_future_k = K > 0 (softmax only, 2026-10-15): the columns are the batch's N next
        states followed by every row's K future states batch['future_states'] flattened
        (column N + i*K + k is row i's k-th future state), S = N + N*K columns shared by every
        row; the one-hot stays on column i and the bootstrap softmax runs over all S."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        next_obs = jnp.asarray(batch["next_observations"])
        u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        goals = next_obs                                         # g_j = s'_j
        anchor = None
        if self._pool():
            # the fixed reference pool is the candidate set of every row; the row's own next state
            # enters only through its anchor, the pool state nearest to it
            goals = self.pool_states                                                # (P, ob)
            anchor = self.pool_anchors(next_obs)                                    # (N,)
        if int(c.get("norm_future_k", 0)) > 0:
            assert "future_states" in batch, "norm_future_k > 0 needs batch['future_states'] (Dataset.norm_future_k)"
            fut = jnp.asarray(batch["future_states"])                           # (N, K, ob)
            assert fut.shape[:2] == (next_obs.shape[0], int(c["norm_future_k"])), fut.shape
            goals = jnp.concatenate([next_obs, fut.reshape(-1, fut.shape[-1])], axis=0)  # (N + N*K, ob)
        row_goals = jnp.asarray(batch["goals"]) if self._goal_index() else None
        if self._reward_index():
            # the index is a solve, not a net: online and target rows are the same stop-gradded w_i
            w = w_t = jax.lax.stop_gradient(jnp.asarray(batch["index_w"]))        # (N, z)
        else:
            w = self._coef(z, row_goals, params=w_params)            # (N, z), online
            w_t = self._coef(z, row_goals, target=True)              # (N, z), target
        phi, b = self.mesh_phi_b(obs, u, goals, params=basis_params)             # (N, S, z), (N, S)
        M = (phi * w[:, None, :]).sum(-1) + b                                     # (N, S); S = N unless norm_future_k
        if u_next.ndim == 3:                                     # bootstrap_source=bc_mean
            M_bar = self.bc_mean_target(next_obs, u_next, goals, w_t)
        else:
            M_bar = self.mesh_M(next_obs, u_next, goals, w_t, params=self.target_basis)
        target_M = jax.lax.stop_gradient(M_bar)
        if self._pool():
            return self._pool_measure_loss(M, target_M, phi, w, anchor, obs, basis_params)
        if self._softmax():
            loss, info = self._softmax_measure_loss(M, target_M, phi, w)
            if self._reward_index():
                info = {**info, **self.reward_index_info(batch, w)}
            return loss, info
        # Successor-measure Bellman fit (PSM Eq. 2 + Cor. 4.2, arXiv 2411.19418 v2; RLU
        # discrete_psm.py:427-434): off-diagonal = squared TD to gamma*target; diagonal =
        # -(1-gamma) pull toward reaching your own next state.
        gamma = c["discount"]
        N = M.shape[0]
        off = 1.0 - jnp.eye(N)
        diff = M - gamma * target_M
        off_diag = 0.5 * jnp.sum((diff ** 2) * off) / jnp.maximum(jnp.sum(off), 1.0)
        diag = -(1.0 - gamma) * jnp.mean(jnp.diagonal(M))
        # F1 (docs/design/2026-09-22-psmgoal-fixes.md): Gram of phi over the mesh. Without a
        # penalty phi collapsed to rank one (effective rank 1.01-1.02, cube, all seeds), which
        # makes the Lagrangian's linear objective point away from the least-squares w.
        # L_ortho = ||mean_ij phi_ij phi_ij^T - I||_F^2, weighted by ortho_coef (0 = off).
        phi_flat = phi.reshape(-1, phi.shape[-1])
        gram = phi_flat.T @ phi_flat / phi_flat.shape[0]
        ortho = jnp.sum((gram - jnp.eye(gram.shape[0])) ** 2)
        loss = off_diag + diag + float(c.get("ortho_coef", 0.0)) * ortho
        f_info = {}
        if self._factorized():
            # f orthogonality over the batch's s+ pool: ||E[f f^T] - I||_F^2 (f on the
            # sqrt(f_dim) sphere, so tr E[f f^T] = f_dim = tr I).
            f_ortho = f_ortho_loss(self.basis(goals, params=basis_params, method="features"))
            loss = loss + float(c.get("f_ortho_coef", 1.0)) * f_ortho
            f_info = {"f_ortho_loss": f_ortho, "psm_loss": loss}
        info = {"psm_loss": loss, "psm_offdiag": off_diag, "psm_diag": diag,
                "ortho_loss": ortho,
                # participation ratio (tr G)^2 / ||G||_F^2: z_dim if isotropic, 1 if rank one
                "phi_eff_rank": jnp.trace(gram) ** 2 / jnp.maximum(jnp.sum(gram ** 2), 1e-12),
                "m_diag_mean": jnp.mean(jnp.diagonal(M)),
                "m_offdiag_mean": jnp.sum(M * off) / jnp.maximum(jnp.sum(off), 1.0),
                "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                "w_norm": jnp.mean(jnp.linalg.norm(w, axis=-1)), **f_info}
        return loss, info

    def _softmax_measure_loss(self, M, target_M, phi, w):
        """measure_loss=softmax: cross-entropy between softmax_j(M_ij/measure_temp) and
        (1-gamma)*onehot(j=i) + gamma*softmax_j(target_M_ij/measure_temp) over the batch's next
        states (plus the rows' future states under norm_future_k; M is (N, S), S = N + N*K).
        Logs the row softmax p next to the raw M: its mass on the row's own next state
        (measure_p_own_next = m_softmax_diag), its total mass on the row's own K future states
        (measure_p_future, 0 when norm_future_k = 0), its largest entry, and the spread
        max_j - min_j of the logits M_ij/measure_temp. phi_eff_rank is the participation ratio
        of phi's Gram over the whole mesh (z_dim if isotropic, 1 if rank one)."""
        c = self.config
        temp = float(c["measure_temp"])
        loss = softmax_td_loss(M, target_M, c["discount"], temp)
        p = jax.nn.softmax(M / temp, axis=-1)
        N, S = M.shape
        K = int(c.get("norm_future_k", 0))
        assert S == N + N * K, (M.shape, K)
        off = 1.0 - jnp.eye(N, S)
        p_own = jnp.mean(jnp.diagonal(p))
        if K > 0:
            rows = jnp.arange(N)
            p_future = jnp.mean(jnp.sum(p[:, N:].reshape(N, N, K)[rows, rows, :], axis=-1))
        else:
            p_future = jnp.zeros(())
        phi_flat = phi.reshape(-1, phi.shape[-1])
        gram = phi_flat.T @ phi_flat / phi_flat.shape[0]
        info = {"psm_loss": loss,
                "m_softmax_diag": p_own,
                "measure_p_own_next": p_own,
                "measure_p_future": p_future,
                "m_softmax_max": jnp.mean(jnp.max(p, axis=-1)),
                "m_logit_spread": jnp.mean(jnp.max(M, axis=-1) - jnp.min(M, axis=-1)) / temp,
                "phi_eff_rank": jnp.trace(gram) ** 2 / jnp.maximum(jnp.sum(gram ** 2), 1e-12),
                "m_diag_mean": jnp.mean(jnp.diagonal(M)),
                "m_offdiag_mean": jnp.sum(M * off) / jnp.maximum(jnp.sum(off), 1.0),
                "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                "w_norm": jnp.mean(jnp.linalg.norm(w, axis=-1))}
        return loss, info

    @staticmethod
    def pool_td_loss(M, M_bar, anchor, discount, temp):
        """measure_loss=pool: cross-entropy between softmax_j(M_ij / temp) and the stop-gradded target
        (1 - discount) * onehot(j = anchor_i) + discount * softmax_j(M_bar_ij / temp), mean over rows.
        M, M_bar (N, P) over the P pool states: row i is (s_i, u_i) resp. (s'_i, u'_i); anchor (N,) is the
        pool state nearest s'_i. `softmax_td_loss` with the identity one-hot replaced by the anchor."""
        onehot = jax.nn.one_hot(anchor, M.shape[1], dtype=M.dtype)
        target = (1.0 - discount) * onehot + discount * jax.nn.softmax(M_bar / temp, axis=-1)
        logp = jax.nn.log_softmax(M / temp, axis=-1)
        return -jnp.mean(jnp.sum(jax.lax.stop_gradient(target) * logp, axis=-1))

    def _pool_measure_loss(self, M, target_M, phi, w, anchor, obs, basis_params):
        """measure_loss=pool: `pool_td_loss` over the pool, plus the diagnostics of the design doc.
        measure_p_anchor: mean mass of the row softmax p_i on its anchor state. pool_entropy: mean
        entropy of p_i over the pool, nats (ln P at most). pool_u_tv: on the first POOL_TV_ROWS rows,
        the mean total variation 0.5 * sum_j |p_i(j) - p'_i(j)| between p_i (the row's own u_i) and
        p'_i under a fresh clipped prior draw at the same (s_i, w_i), online params, no gradient -- the
        number that says whether the measure depends on the action. phi_eff_rank, m_softmax_max,
        m_logit_spread, td_target_absmean, w_norm as under measure_loss=softmax."""
        c = self.config
        temp = float(c["measure_temp"])
        loss = self.pool_td_loss(M, target_M, anchor, c["discount"], temp)
        logp = jax.nn.log_softmax(M / temp, axis=-1)
        p = jnp.exp(logp)
        N = M.shape[0]
        p_anchor = jnp.mean(p[jnp.arange(N), anchor])
        entropy = -jnp.mean(jnp.sum(p * logp, axis=-1))
        n = min(POOL_TV_ROWS, N)
        eps = jax.random.normal(jax.random.fold_in(self.rng, POOL_TV_KEY), (n, int(c["action_dim"])))
        eps = jnp.clip(eps, -c["u_clip"], c["u_clip"])
        M_prior = self.mesh_M(obs[:n], eps, self.pool_states, w[:n], params=basis_params)        # (n, P)
        p_prior = jax.nn.softmax(jax.lax.stop_gradient(M_prior) / temp, axis=-1)
        tv = jnp.mean(0.5 * jnp.sum(jnp.abs(jax.lax.stop_gradient(p[:n]) - p_prior), axis=-1))
        phi_flat = phi.reshape(-1, phi.shape[-1])
        gram = phi_flat.T @ phi_flat / phi_flat.shape[0]
        info = {"psm_loss": loss,
                "measure_p_anchor": p_anchor,
                "pool_entropy": entropy,
                "pool_u_tv": tv,
                "m_softmax_max": jnp.mean(jnp.max(p, axis=-1)),
                "m_logit_spread": jnp.mean(jnp.max(M, axis=-1) - jnp.min(M, axis=-1)) / temp,
                "phi_eff_rank": jnp.trace(gram) ** 2 / jnp.maximum(jnp.sum(gram ** 2), 1e-12),
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

    # ------------------------------------------------------------- in-loop actor loss
    def _awr_weights(self, obs, goals, w_rows, u_data, base_key):
        """AWR advantage weights per row: weight_i = clip(exp((Q_data_i - V_i)/beta), 0, wmax).

        Q_data_i = mean_g M(s_i, u_data_i, g, w_i) reads the FROZEN measure at the DATA latent.
        V_i is a Monte-Carlo baseline: mean over awr_baseline_k clipped prior draws of the same
        goal-averaged measure. The measure is queried only at u_data and at prior draws, never
        at an out-of-support actor latent. Weights are returned stop-gradded. Returns
        (weight, adv, Q_data, V), each (N,)."""
        c = self.config
        scale = float(c["u_clip"])
        bp = jax.lax.stop_gradient(self.basis.params)
        q_data = jnp.mean(self.mesh_M(obs, u_data, goals, w_rows, params=bp), axis=1)   # (N,)
        kv, da = int(c["awr_baseline_k"]), int(c["action_dim"])
        u_prior = jnp.clip(jax.random.normal(base_key, (kv, obs.shape[0], da)), -scale, scale)
        v = jnp.mean(jax.vmap(lambda uk: jnp.mean(
            self.mesh_M(obs, uk, goals, w_rows, params=bp), axis=1))(u_prior), axis=0)    # (N,)
        adv = q_data - v
        beta, wmax = float(c["awr_beta"]), float(c["awr_wmax"])
        weight = jax.lax.stop_gradient(jnp.clip(jnp.exp(adv / beta), 0.0, wmax))
        return weight, adv, q_data, v

    def actor_loss_inloop(self, actor_params, obs, goals, w_rows, noise, u_data, base_key):
        """In-loop actor step on the TRAINING-time coefficient w=self.w(z), with phi read at
        frozen (stop-grad) basis params so only the actor moves. `actor_objective` picks the loss:

        - distill (default): DDPG-style max of the goal-averaged measure at the actor's own draw;
          Q scale-normalised, entropy at actor_temp, prior anchor at bc_coeff. Queries the measure
          at the actor latent (can be out-of-support) -- the DSRL path, unchanged.
        - awr: -mean(weight_i * logpi(u_data_i)) with weight from `_awr_weights`. Behaviour-clones
          the DATA latent, weighted by its measured advantage. In-support: the loss target is u_data.
        - bc: -mean(logpi(u_data_i)). Plain behaviour clone of the data latent."""
        c = self.config
        scale = float(c["u_clip"])
        objective = str(c.get("actor_objective", "distill"))
        if objective == "distill":
            temp = float(c["actor_temp"])
            q_coeff, bc_coeff = float(c.get("q_coeff", 1.0)), float(c.get("bc_coeff", 0.0))
            bp = jax.lax.stop_gradient(self.basis.params)
            mu, log_std = self.actor(obs, w_rows, params=actor_params)
            u, logp = tanh_gaussian_sample(mu, log_std, noise, scale)
            q = jnp.mean(self.mesh_M(obs, u, goals, w_rows, params=bp), axis=1)     # (N,) mean over goals
            qscale = jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-8)
            loss = q_coeff * (-jnp.mean(q) / qscale) + temp * jnp.mean(logp) + bc_coeff * jnp.mean(u ** 2)
            return loss, {"actor_q": jnp.mean(q), "actor_logp": jnp.mean(logp),
                          "actor_u_absmean": jnp.mean(jnp.abs(u))}
        mu, log_std = self.actor(obs, w_rows, params=actor_params)
        logpi = tanh_gaussian_logprob(mu, log_std, u_data, scale)                   # (N,)
        if objective == "bc":
            loss = -jnp.mean(logpi)
            return loss, {"actor_bc_logp": jnp.mean(logpi), "actor_logp": jnp.mean(logpi)}
        if objective == "awr":
            weight, adv, q_data, v = self._awr_weights(obs, goals, w_rows, u_data, base_key)
            loss = -jnp.mean(weight * logpi)
            return loss, {"actor_logp": jnp.mean(logpi), "awr_adv_mean": jnp.mean(adv),
                          "awr_q_data_mean": jnp.mean(q_data), "awr_v_mean": jnp.mean(v),
                          "awr_weight_mean": jnp.mean(weight), "awr_weight_max": jnp.max(weight)}
        raise ValueError(f"unknown actor_objective: {objective!r} (distill | awr | bc)")

    # ------------------------------------------- Factored-FB induced-reward actor (actor_value)
    def fb_feature(self, states, params, anchors):
        """f(s+) = psm_norm(mean_{u in U} mean_{a in anchors} phi(s+, u, a)): the goal- and
        action-marginalized projected phi (PsmgoalProjectedPhi), read at `params` (the slow
        EMA basis). U = fb_k_u fixed clipped prior latents from fb_proj_seed. (n, ob) -> (n, D)."""
        from agents.f_psmflow import _sample_projection_u
        from utils.psm_networks import psm_norm

        c = self.config
        U = _sample_projection_u(int(c["fb_proj_seed"]), int(c["fb_k_u"]),
                                 int(c["action_dim"]), float(c["u_clip"]))
        n, d = states.shape
        ku, kg = U.shape[0], anchors.shape[0]
        o = jnp.broadcast_to(states[:, None, None, :], (n, ku, kg, d)).reshape(n * ku * kg, d)
        u = jnp.broadcast_to(U[None, :, None, :], (n, ku, kg, U.shape[-1])).reshape(n * ku * kg, -1)
        g = jnp.broadcast_to(anchors[None, None], (n, ku, kg, anchors.shape[-1])).reshape(n * ku * kg, -1)
        phi, _ = self.basis(o, u, g, params=params)
        return psm_norm(phi.reshape(n, ku, kg, -1).mean(axis=(1, 2)))

    def fb_task_fit(self, f_pool, rewards):
        """infer_cond: reward samples on the pool -> task vector(s) on the sqrt(D) sphere, at
        the STORED moments. f_pool (M, D), rewards (B, M) or (M,) -> (B, D)."""
        c = self.config
        return ridge_task(f_pool, self.fb_mu, self.fb_cov, rewards,
                          float(c["fb_ridge_lambda_mult"]), float(c["fb_cov_eig_floor"]),
                          float(c["fb_const_eps"]))

    def fb_sample_tasks(self, f_pool, fb_goals, pool, key):
        """Per-row task vector (Factored-FB sample_cond + _z_mix_masked): the centered ridge
        fit of the hindsight goal's kernel reward on the pool, replaced by a fresh sqrt(D)
        sphere draw with probability fb_rand_frac. Returns (w (N, D), hindsight (N, 1))."""
        c = self.config
        d = int(c["z_dim"])
        r = jax.lax.stop_gradient(kernel_reward(fb_goals, pool, parse_hit_spec(c["fb_hit_spec"])))
        w_goal = self.fb_task_fit(f_pool, r)
        k_sphere, k_mix = jax.random.split(key)
        n = w_goal.shape[0]
        x = jax.random.normal(k_sphere, (n, d))
        w_rand = x * jnp.sqrt(float(d)) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        use_rand = jax.random.bernoulli(k_mix, float(c["fb_rand_frac"]), (n, 1))
        w = jax.lax.stop_gradient(jnp.where(use_rand, w_rand, w_goal))
        return w, 1.0 - use_rand.astype(w.dtype)

    def actor_loss_fbtrick(self, actor_params, obs, pool, w_task, f_centered, noise, u_data):
        """Factored-FB FlowBC actor on psmgoal's tanh-Gaussian latent actor.

        Q(s,u,w) from the measure mesh M_ij = phi(s_i,u_i,s+_j)^T w_i + b_ij at the TASK vector
        w_i (the reference conditions its measure on the same z the actor ascends) and the
        induced reward r_ij = (f(s+_j) - mu)^T w_i / sqrt(D); `induced_value` combines them.
        Basis params are stop-gradded, so only the actor moves.
        loss = -mean Q / sg(mean|Q|) + actor_temp*mean logp + fb_bc_coeff*mean (u - u_data)^2."""
        c = self.config
        scale = float(c["u_clip"])
        bp = jax.lax.stop_gradient(self.basis.params)
        mu, log_std = self.actor(obs, w_task, params=actor_params)
        u, logp = tanh_gaussian_sample(mu, log_std, noise, scale)
        M = self.mesh_M(obs, u, pool, w_task, params=bp)                          # (N, P)
        r = jax.lax.stop_gradient(w_task @ f_centered.T / jnp.sqrt(float(c["z_dim"])))  # (N, P)
        q = induced_value(M, r, str(c["actor_value"]), float(c["fb_value_temp"]))  # (N,)
        qscale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        bc = jnp.mean((u - u_data) ** 2)
        loss = (-jnp.mean(q) / qscale + float(c["actor_temp"]) * jnp.mean(logp)
                + float(c["fb_bc_coeff"]) * bc)
        return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "actor_q_absmean": qscale,
                      "actor_bc_err": bc, "actor_logp": jnp.mean(logp),
                      "actor_u_absmean": jnp.mean(jnp.abs(u)),
                      "fb_M_mean": jnp.mean(M), "fb_M_std": jnp.std(M),
                      "fb_r_std": jnp.std(r)}

    def fb_actor_step(self, batch):
        """One actor step under actor_value, plus the index-state update (density_fb
        update_stats order: f read at the OLD index tower; the tower then EMAs toward the
        PRE-step online basis; moments EMA'd from the OLD-tower f). Returns (actor, fb_state, info)."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        pool = jnp.asarray(batch["next_observations"])
        k_anchor = self.fb_anchor.shape[0]
        assert pool.shape[0] >= k_anchor, "batch must hold at least fb_k_anchor next states"
        anchors = jnp.where(self.fb_filled > 0.5, self.fb_anchor, pool[:k_anchor])
        f_pool = jax.lax.stop_gradient(self.fb_feature(pool, self.fb_zenc, anchors))   # (P, D)
        w_task, hindsight = self.fb_sample_tasks(
            f_pool, jnp.asarray(batch["fb_goals"]), pool, jax.random.fold_in(self.rng, 101))
        f_centered = f_pool - self.fb_mu[None]
        noise = jax.random.normal(jax.random.fold_in(self.rng, 102),
                                  (obs.shape[0], int(c["action_dim"])))
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        (_, info), g_a = jax.value_and_grad(self.actor_loss_fbtrick, has_aux=True)(
            self.actor.params, obs, pool, w_task, f_centered, noise, u_data)
        actor = self.actor.apply_gradients(grads=g_a)
        # index state
        tz, ts = float(c["fb_enc_tau"]), float(c["fb_sigma_tau"])
        zenc = jax.tree_util.tree_map(lambda p, e: p * tz + e * (1.0 - tz),
                                      self.basis.params, self.fb_zenc)
        mu_b = jnp.mean(f_pool, axis=0)
        db = f_pool - mu_b[None]
        cov_b = db.T @ db / f_pool.shape[0]
        mu = self.fb_mu * (1.0 - ts) + mu_b * ts
        cov = self.fb_cov * (1.0 - ts) + cov_b * ts
        cov = 0.5 * (cov + cov.T)
        fb_state = {"fb_zenc": zenc, "fb_mu": mu, "fb_cov": cov, "fb_anchor": anchors,
                    "fb_filled": jnp.ones_like(self.fb_filled)}
        info = {**info, "fb_hindsight_frac": jnp.mean(hindsight),
                "fb_task_norm": jnp.mean(jnp.linalg.norm(w_task, axis=-1)),
                "fb_cov_trace": jnp.trace(cov), "fb_f_mu_norm": jnp.linalg.norm(mu)}
        return actor, fb_state, info

    # ----------------------------------------------- reward index (policy_index=reward, 2026-10-19)
    def reward_index_rows(self, batch):
        """policy_index=reward: the index rows of one update, computed once per step from the batch (the
        index is a solve on f at the slow tower, not a net). Returned as a dict merged into the batch:
        index_w (N, D) the measure rows' w = ridge of the kernel reward of batch['goals'] over the batch's
        next states (fb_rand_frac of the rows a random sqrt(D) sphere draw); index_w_actor (N, D) the same
        at the actor goal batch['fb_goals'] with its own draw; index_f (P, D) f of the next states at the
        index tower; index_anchor (k_anchor, ob) the anchors read; index_hindsight(_actor) (N, 1) 1 on
        hindsight rows."""
        pool = jnp.asarray(batch["next_observations"])
        k_anchor = self.fb_anchor.shape[0]
        assert pool.shape[0] >= k_anchor, "batch must hold at least fb_k_anchor next states"
        anchors = jnp.where(self.fb_filled > 0.5, self.fb_anchor, pool[:k_anchor])
        f_pool = jax.lax.stop_gradient(self.fb_feature(pool, self.fb_zenc, anchors))          # (P, D)
        w_m, h_m = self.fb_sample_tasks(f_pool, jnp.asarray(batch["goals"]), pool,
                                        jax.random.fold_in(self.rng, REWARD_INDEX_KEY))
        w_a, h_a = self.fb_sample_tasks(f_pool, jnp.asarray(batch["fb_goals"]), pool,
                                        jax.random.fold_in(self.rng, REWARD_ACTOR_KEY))
        return {"index_w": w_m, "index_w_actor": w_a, "index_f": f_pool, "index_anchor": anchors,
                "index_hindsight": h_m, "index_hindsight_actor": h_a}

    def reward_index_info(self, batch, w):
        """Training-log diagnostics of the reward index (docs/design/2026-10-19-reward-index.md):
        index_proxy_corr = mean over the hindsight rows of the per-row correlation between the index's own
        reward view (f(y_j) - mu)^T w_i / sqrt(D) and the kernel reward k(g_i, y_j) over the batch's next
        states; index_w_norm, index_hindsight_frac, index_cov_trace."""
        c = self.config
        f_c = jnp.asarray(batch["index_f"]) - self.fb_mu[None]
        proxy = (w @ f_c.T) / jnp.sqrt(float(c["z_dim"]))                                     # (N, P)
        kern = kernel_reward(jnp.asarray(batch["goals"]), jnp.asarray(batch["next_observations"]),
                             parse_hit_spec(c["fb_hit_spec"]))                                 # (N, P)
        hs = jnp.asarray(batch["index_hindsight"]).reshape(-1)                                  # (N,)
        a = proxy - proxy.mean(axis=1, keepdims=True)
        b = kern - kern.mean(axis=1, keepdims=True)
        corr = jnp.sum(a * b, axis=1) / jnp.sqrt(jnp.sum(a * a, axis=1) * jnp.sum(b * b, axis=1) + 1e-12)
        n_h = jnp.maximum(jnp.sum(hs), 1.0)
        return {"index_proxy_corr": jnp.sum(corr * hs) / n_h,
                "index_w_norm": jnp.mean(jnp.linalg.norm(w, axis=-1)),
                "index_hindsight_frac": jnp.mean(hs),
                "index_cov_trace": jnp.trace(self.fb_cov)}

    def reward_actor_rows(self, batch, w_rows, f_centered):
        """The reward r_i(y_j) (N, P) the reward-index actor climbs over the batch's next states y_j.
        reward_actor_reward=latent_centered: the index's own reward view (f(y_j) - mu)^T w_i / sqrt(D) on every
        row (Density FB Eq. pretraining-reward). kernel: the exact kernel reward k(g_i, y_j) of the row's own
        hindsight goal on the hindsight rows (index_hindsight_actor == 1), the centered view on the random
        rows, which carry no goal (Density FB `measure_density_z_actor_reward=kernel`). Stop-gradded."""
        c = self.config
        latent = w_rows @ f_centered.T / jnp.sqrt(float(c["z_dim"]))                            # (N, P)
        if str(c.get("reward_actor_reward", "latent_centered")) == "kernel":
            kern = kernel_reward(jnp.asarray(batch["fb_goals"]), jnp.asarray(batch["next_observations"]),
                                 parse_hit_spec(c["fb_hit_spec"]))                             # (N, P)
            hs = jnp.asarray(batch["index_hindsight_actor"]).reshape(-1, 1)
            latent = hs * kern + (1.0 - hs) * latent
        return jax.lax.stop_gradient(latent)

    def actor_loss_flowbc_reward(self, delta_params, obs, pool, w_rows, r, eps, u_data):
        """policy_index=reward, actor_kind=flowbc. u_i = clip(eps_i + delta(s_i, w_i, eps_i)); value
        Q_i = sum_j softmax_j(M(s_i,u_i,y_j; w_i)/measure_temp) r_i(y_j), r the rows of `reward_actor_rows`;
        loss -q_coeff mean Q / sg|Q| + fb_bc_coeff mean delta^2. phi, b stop-gradded; only delta moves."""
        c = self.config
        bp = jax.lax.stop_gradient(self.basis.params)
        temp = float(c["measure_temp"])
        u, d = self.flowbc_latent(obs, w_rows, eps, params=delta_params)
        M = self.mesh_M(obs, u, pool, w_rows, params=bp)                                       # (N, P)
        r = jax.lax.stop_gradient(r)                                                           # (N, P)
        p = jax.nn.softmax(M / temp, axis=1)
        q = jnp.sum(p * r, axis=1)
        qscale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        bc = jnp.mean(d ** 2)
        loss = -float(c.get("q_coeff", 1.0)) * jnp.mean(q) / qscale + float(c["fb_bc_coeff"]) * bc
        n = min(M.shape[0], M.shape[1])
        return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "actor_q_absmean": qscale,
                      "actor_bc_err": bc, "actor_delta_norm": jnp.mean(jnp.linalg.norm(d, axis=-1)),
                      "actor_u_absmean": jnp.mean(jnp.abs(u)),
                      "actor_u_data_gap": jnp.mean((u - u_data) ** 2),
                      "actor_p_own": jnp.mean(jnp.diagonal(p[:n, :n])),
                      "actor_M_mean": jnp.mean(M), "actor_M_std": jnp.std(M), "actor_r_std": jnp.std(r)}

    def reward_actor_step(self, batch):
        """One flowbc actor step at the actor index rows batch['index_w_actor'], then the index-state update
        in `fb_actor_step`'s order (f read at the OLD tower; the tower EMAs toward the PRE-step basis; the
        moments EMA from the OLD-tower f). Returns (updates, info)."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        pool = jnp.asarray(batch["next_observations"])
        w_a = jnp.asarray(batch["index_w_actor"])
        f_pool = jnp.asarray(batch["index_f"])
        anchors = jnp.asarray(batch["index_anchor"])
        f_centered = f_pool - self.fb_mu[None]
        eps = jax.random.normal(jax.random.fold_in(self.rng, 102), (obs.shape[0], int(c["action_dim"])))
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        r = self.reward_actor_rows(batch, w_a, f_centered)
        (_, info), g_d = jax.value_and_grad(self.actor_loss_flowbc_reward, has_aux=True)(
            self.delta.params, obs, pool, w_a, r, eps, u_data)
        delta = self.delta.apply_gradients(grads=g_d)
        tz, ts = float(c["fb_enc_tau"]), float(c["fb_sigma_tau"])
        zenc = jax.tree_util.tree_map(lambda p, e: p * tz + e * (1.0 - tz), self.basis.params, self.fb_zenc)
        mu_b = jnp.mean(f_pool, axis=0)
        db = f_pool - mu_b[None]
        cov_b = db.T @ db / f_pool.shape[0]
        mu = self.fb_mu * (1.0 - ts) + mu_b * ts
        cov = self.fb_cov * (1.0 - ts) + cov_b * ts
        cov = 0.5 * (cov + cov.T)
        upd = {"delta": delta, "fb_zenc": zenc, "fb_mu": mu, "fb_cov": cov, "fb_anchor": anchors,
               "fb_filled": jnp.ones_like(self.fb_filled)}
        return upd, {**info, "index_hindsight_frac_actor": jnp.mean(jnp.asarray(batch["index_hindsight_actor"]))}

    # ----------------------------------------------- goal-conditioned actor (policy_index=goal)
    def actor_loss_goal(self, actor_params, obs, pool, goals, noise, u_data):
        """Actor loss under policy_index=goal (Factored-FB FlowBC on the tanh-Gaussian latent actor).

        u_i is the actor's draw at (s_i, g_i), g_i the actor goal. Q_i is the measure over the
        batch's next states s+_j at w_i = h(g_i), weighted by the reward k(g_i, s+_j): the
        normalised Gaussian kernel around the goal on the success coordinates (fb_hit_spec).
        squared run: Q_i = sum_j M_ij k_ij. softmax run: Q_i = sum_j softmax_j(M_ij/measure_temp) k_ij.
        phi, b and h are stop-gradded, so only the actor moves.
        loss = -q_coeff*mean Q / sg(mean|Q|) + actor_temp*mean logp + fb_bc_coeff*mean (u - u_data)^2."""
        c = self.config
        scale = float(c["u_clip"])
        bp = jax.lax.stop_gradient(self.basis.params)
        w_rows = jax.lax.stop_gradient(self.w_star(goals))                        # (N, z) h(g_i)
        mu, log_std = self.actor(obs, goals if self._actor_takes_goal() else w_rows,
                                 params=actor_params)
        u, logp = tanh_gaussian_sample(mu, log_std, noise, scale)
        q, v_info = self.actor_value(obs, u, pool, goals, w_rows, bp, with_info=True)
        qscale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        bc = jnp.mean((u - u_data) ** 2)
        loss = (-float(c.get("q_coeff", 1.0)) * jnp.mean(q) / qscale
                + float(c["actor_temp"]) * jnp.mean(logp) + float(c["fb_bc_coeff"]) * bc)
        return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "actor_q_absmean": qscale,
                      "actor_bc_err": bc, "actor_logp": jnp.mean(logp),
                      "actor_u_absmean": jnp.mean(jnp.abs(u)), **v_info}

    # ------------------------------------------------- joint actor heads (2026-10-05)
    def actor_value(self, obs, u, pool, goals, w_rows, basis_params, with_info=False):
        """The value the goal actor climbs, (N,). Rows (s_i, u_i) at w_i = h(g_i), g_i the
        actor goal, pool the batch's next states s+_j.
        actor_value_kind=shaped: Q_i = sum_j p_ij k(g_i, s+_j), p = softmax_j(M_ij/measure_temp)
          (softmax loss) or p = M (squared loss); k the kernel reward (fb_hit_spec).
        actor_value_kind=point: softmax loss: the share of g_i among [g_i + pool],
          exp(M_ig/t) / (exp(M_ig/t) + sum_j exp(M_ij/t)); squared loss: M(s_i,u_i,g_i).
        with_info=True also returns the softmax weight on the row's own next state
        (column i of the pool) and the mesh statistics."""
        c = self.config
        temp = float(c["measure_temp"])
        M = self.mesh_M(obs, u, pool, w_rows, params=basis_params)                 # (N, P)
        if self._point_value():
            m_g = self.M(obs, u, goals, w_rows, params=basis_params)                # (N,)
            if self._softmax():
                lse = jax.nn.logsumexp(jnp.concatenate([m_g[:, None], M], axis=1) / temp, axis=1)
                q = jnp.exp(m_g / temp - lse)
            else:
                q = m_g
        else:
            k = jax.lax.stop_gradient(kernel_reward(goals, pool, parse_hit_spec(c["fb_hit_spec"])))
            if self._softmax():
                q = jnp.sum(jax.nn.softmax(M / temp, axis=1) * k, axis=1)
            else:
                q = jnp.sum(M * k, axis=1)
        if not with_info:
            return q
        p = jax.nn.softmax(M / temp, axis=1)
        n = min(M.shape[0], M.shape[1])
        info = {"actor_p_own": jnp.mean(jnp.diagonal(p[:n, :n])),
                "actor_M_mean": jnp.mean(M), "actor_M_std": jnp.std(M)}
        return q, info

    def flowbc_latent(self, obs, goals, eps, params=None):
        """actor_kind=flowbc: u = clip(eps + delta(s, g, eps), +-u_clip). Returns (u, delta)."""
        u_clip = float(self.config["u_clip"])
        d = self.delta(obs, goals, eps, params=self.delta.params if params is None else params)
        return jnp.clip(eps + d, -u_clip, u_clip), d

    def actor_loss_flowbc(self, delta_params, obs, pool, goals, eps, u_data):
        """actor_kind=flowbc. u_i = clip(eps_i + delta(s_i, g_i, eps_i)), eps_i a prior draw.
        loss = -q_coeff * mean Q / sg(mean|Q|) + fb_bc_coeff * mean delta^2 (mean over rows
        and dims, the same normalisation as the data-latent anchor it replaces). The anchor
        is eps itself: u = eps is a behaviour-cloning sample, so delta = 0 is exact BC.
        phi, b and h are stop-gradded; only delta moves. u_data is read for logging only."""
        c = self.config
        bp = jax.lax.stop_gradient(self.basis.params)
        w_rows = jax.lax.stop_gradient(self.w_star(goals))                        # (N, z) h(g_i)
        u, d = self.flowbc_latent(obs, goals, eps, params=delta_params)
        q, v_info = self.actor_value(obs, u, pool, goals, w_rows, bp, with_info=True)
        qscale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        bc = jnp.mean(d ** 2)
        loss = -float(c.get("q_coeff", 1.0)) * jnp.mean(q) / qscale + float(c["fb_bc_coeff"]) * bc
        return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "actor_q_absmean": qscale,
                      "actor_bc_err": bc, "actor_delta_norm": jnp.mean(jnp.linalg.norm(d, axis=-1)),
                      "actor_u_absmean": jnp.mean(jnp.abs(u)),
                      "actor_u_data_gap": jnp.mean((u - u_data) ** 2), **v_info}

    def actor_loss_dsrl(self, actor_params, obs, pool, goals, noise):
        """actor_kind=dsrl: the f_psmflow dsrl_sac head on the goal actor. A tanh-Gaussian
        draw u_i at (s_i, g_i); loss = -q_coeff * mean Q / sg(mean|Q|) + alpha * mean logp,
        alpha = exp(log_alpha) stop-gradded (trained by `alpha_loss` toward
        dsrl_actor.target_entropy). No BC term. phi, b and h are stop-gradded."""
        c = self.config
        bp = jax.lax.stop_gradient(self.basis.params)
        w_rows = jax.lax.stop_gradient(self.w_star(goals))
        mu, log_std = self.actor(obs, goals, params=actor_params)
        u, logp = tanh_gaussian_sample(mu, log_std, noise, float(c["u_clip"]))
        q, v_info = self.actor_value(obs, u, pool, goals, w_rows, bp, with_info=True)
        qscale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1e-6))
        alpha = jax.lax.stop_gradient(jnp.exp(self.log_alpha()))
        loss = -float(c.get("q_coeff", 1.0)) * jnp.mean(q) / qscale + alpha * jnp.mean(logp)
        return loss, {"actor_loss": loss, "actor_q": jnp.mean(q), "actor_q_absmean": qscale,
                      "actor_logp": jnp.mean(logp), "actor_alpha": alpha,
                      "actor_u_absmean": jnp.mean(jnp.abs(u)), **v_info}

    def alpha_loss(self, alpha_params, obs, goals, noise):
        """SAC's entropy-coefficient loss, -log_alpha * sg(logp + target_entropy), mean over
        rows; DSRL's target_ent = 0 in the squashed, unscaled space of `tanh_gaussian_sample`."""
        c = self.config
        mu, log_std = self.actor(obs, goals)
        _, logp = tanh_gaussian_sample(mu, log_std, noise, float(c["u_clip"]))
        la = self.log_alpha(params=alpha_params)
        target = float(c["dsrl_actor"]["target_entropy"])
        loss = -(la * jax.lax.stop_gradient(logp + target)).mean()
        return loss, {"alpha_loss": loss, "log_alpha": la}

    def goal_actor_step(self, batch):
        """One actor step under policy_index=goal, at the actor goal batch['fb_goals'] (a
        uniformly drawn future state of the row's trajectory), dispatched on actor_kind.
        Returns (updates, info); updates is a dict of the TrainStates that moved."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        pool = jnp.asarray(batch["next_observations"])
        goals = jnp.asarray(batch["goals"] if str(c.get("actor_goal_source", "future")) == "measure"
                            else batch["fb_goals"])
        noise = jax.random.normal(jax.random.fold_in(self.rng, 102),
                                  (obs.shape[0], int(c["action_dim"])))
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        kind = self._actor_kind()
        if kind == "flowbc":
            (_, info), g_d = jax.value_and_grad(self.actor_loss_flowbc, has_aux=True)(
                self.delta.params, obs, pool, goals, noise, u_data)
            return {"delta": self.delta.apply_gradients(grads=g_d)}, info
        if kind == "dsrl":
            (_, info), g_a = jax.value_and_grad(self.actor_loss_dsrl, has_aux=True)(
                self.actor.params, obs, pool, goals, noise)
            (_, al_info), g_al = jax.value_and_grad(self.alpha_loss, has_aux=True)(
                self.log_alpha.params, obs, goals, noise)
            return ({"actor": self.actor.apply_gradients(grads=g_a),
                     "log_alpha": self.log_alpha.apply_gradients(grads=g_al)}, {**info, **al_info})
        (_, info), g_a = jax.value_and_grad(self.actor_loss_goal, has_aux=True)(
            self.actor.params, obs, pool, goals, noise, u_data)
        return {"actor": self.actor.apply_gradients(grads=g_a)}, info

    # ------------------------------------------------------------------ update
    def apply_update(self, batch, z, u_next):
        c = self.config
        train_basis = bool(c.get("train_basis", True))
        goal_index = self._goal_index()
        basis, w, w_star = self.basis, self.w, self.w_star
        target_basis, target_w, target_w_star = self.target_basis, self.target_w, self.target_w_star
        # The coefficient net the measure loss trains: w(z), or h(g) under policy_index=goal.
        coef = self.w_star if goal_index else self.w
        if train_basis:
            grad_fn = jax.value_and_grad(self.measure_loss, argnums=(0, 1), has_aux=True)
            (_, info), (g_b, g_c) = grad_fn(self.basis.params, coef.params, batch, z, u_next)
            basis = self.basis.apply_gradients(grads=g_b)
            target_basis = polyak_update(basis.params, self.target_basis, c["tau"])
            if goal_index:
                w_star = self.w_star.apply_gradients(grads=g_c)
                target_w_star = polyak_update(w_star.params, self.target_w_star, c["tau"])
            else:
                w = self.w.apply_gradients(grads=g_c)
                target_w = polyak_update(w.params, self.target_w, c["tau"])
        else:
            # frozen measure (phi, b, w(z) held fixed, targets held fixed): log the loss only.
            _, info = self.measure_loss(self.basis.params, coef.params, batch, z, u_next)
        if bool(c.get("train_goal_head", False)) and "goals" in batch:
            perm = jax.random.permutation(jax.random.fold_in(self.rng, 55),
                                          jnp.asarray(batch["observations"]).shape[0])
            (_, gh_info), g_gc = jax.value_and_grad(self.goal_head_loss, has_aux=True)(
                self.w_star.params, batch, perm)
            w_star = self.w_star.apply_gradients(grads=g_gc)
            info = {**info, **gh_info}
        actor = self.actor
        fb_state = {}
        if goal_index and bool(c.get("train_actor", False)):
            # goal-conditioned actor at the actor goal; replaces the two actor losses below.
            # bootstrap_source=data with train_actor=false trains no actor at all.
            # actor_kind=flowbc moves `delta`, dsrl moves `actor` and `log_alpha`.
            upd, a_info = self.goal_actor_step(batch)
            actor = upd.pop("actor", self.actor)
            fb_state = upd
            info = {**info, **a_info}
        elif self._reward_index() and bool(c.get("train_actor", False)):
            # reward index: the flowbc actor at the actor index rows, plus the index-state update.
            upd, a_info = self.reward_actor_step(batch)
            fb_state = upd
            info = {**info, **a_info}
        elif str(c.get("actor_value", "none")) != "none":
            # Factored-FB induced-reward actor; replaces actor_objective's loss.
            actor, fb_state, a_info = self.fb_actor_step(batch)
            info = {**info, **a_info}
        elif bool(c.get("train_actor", False)):
            obs = jnp.asarray(batch["observations"])
            goals = jnp.asarray(batch["next_observations"])
            w_rows = jax.lax.stop_gradient(self.w(z))          # training coefficient
            noise = jax.random.normal(jax.random.fold_in(self.rng, 77),
                                      (obs.shape[0], int(c["action_dim"])))
            # awr/bc behaviour-clone the DATA latent; distill ignores u_data/base_key.
            u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
            base_key = jax.random.fold_in(self.rng, 88)
            (_, a_info), g_a = jax.value_and_grad(self.actor_loss_inloop, has_aux=True)(
                self.actor.params, obs, goals, w_rows, noise, u_data, base_key)
            actor = self.actor.apply_gradients(grads=g_a)
            info = {**info, **a_info}
        new = self.replace(basis=basis, w=w, w_star=w_star, actor=actor,
                           target_basis=target_basis, target_w=target_w,
                           target_w_star=target_w_star, **fb_state)
        return new, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        N = jnp.asarray(batch["observations"]).shape[0]
        z = sample_z_bin(rng, N, int(self.config["max_log_seed"]))
        if self._reward_index():
            batch = {**batch, **self.reward_index_rows(batch)}
        u_next = self._select_u_next(batch, z)
        new_agent, info = self.apply_update(batch, z, u_next)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss at current params (no step)."""
        rng = rng if rng is not None else self.rng
        N = jnp.asarray(batch["observations"]).shape[0]
        z = sample_z_bin(rng, N, int(self.config["max_log_seed"]))
        if self._reward_index():
            batch = {**batch, **self.reward_index_rows(batch)}
        u_next = self._select_u_next(batch, z)
        coef = self.w_star if self._goal_index() else self.w
        return self.measure_loss(self.basis.params, coef.params, batch, z, u_next)

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

    def _obj_and_constraint(self, w, l_params, obs, u, goals, perm, refs=None):
        """PSM Eq. 10: obj = mean phi(s,u,g)^T w, and the non-negativity constraint
        phi^T w + b >= 0 on permuted off-goal samples, priced by the multiplier l(s,u,g).
        measure_loss=softmax: obj is the mean `_goal_score`, the softmax mass on the goal
        columns over [goals + refs]."""
        bp = jax.lax.stop_gradient(self.basis.params)
        if self._softmax():
            assert refs is not None, "measure_loss=softmax inference needs the reference columns"
            obj = jnp.mean(self._goal_score(
                obs, u, goals, refs, jnp.broadcast_to(w, (obs.shape[0], w.shape[0])), params=bp))
        else:
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

    def inference_step(self, state, obs, u, goals, key, refs=None):
        """One RLU coefficient step: descend w on -obj + penalty, ascend the multiplier,
        renormalize w onto the sphere. Returns (state', info).
        measure_loss=softmax: the objective is the softmax mass on the goal columns, which is
        non-negative for every w, so the penalty is off and the multiplier is not stepped."""
        c = self.config
        # the multiplier only prices the squared measure's non-negativity constraint
        use_dgd = bool(c["use_dgd"]) and not self._softmax()
        perm = jax.random.permutation(key, obs.shape[0])
        w_tx = optax.adam(c["lr_infer"])
        l_tx = optax.adam(c["lr_l"])

        def w_loss(w):
            obj, cons, l = self._obj_and_constraint(w, state.l_params, obs, u, goals, perm, refs=refs)
            if self._softmax():
                pen = 0.0
            elif use_dgd:
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

    def run_inference(self, obs, u, goals, key, refs=None):
        """Full Stage-2 loop; returns the inferred coefficient w (z_dim,). `refs` are the
        reference columns of the softmax objective (measure_loss=softmax only)."""
        state = self.init_inference(key)
        step = jax.jit(self.inference_step)
        for i in range(int(self.config["num_inference_steps"])):
            state, _ = step(state, obs, u, goals, jax.random.fold_in(key, i), refs)
        return state.w

    # ------------------------------------------------------------------ inference on a cached mesh
    def _mesh_goal_means(self, obs, u, goals, cols=None):
        """The frozen measure on the (N pairs x G goals) mesh, reduced as it is computed:
        the per-pair means over ALL goals, phi_mean (N, z_dim) and b_mean (N,), and, when
        `cols` (N, C) is given, the mesh entries at those goal columns, phi_cols (N, C, z_dim)
        and b_cols (N, C) (else None). The network sees MESH_ROWS (s, u, g) rows per call, so
        the full (N, G, z_dim) mesh is never held."""
        N, G = obs.shape[0], goals.shape[0]
        rows = max(1, min(N, MESH_ROWS // G))
        n_pad = (-N) % rows                                   # pad to whole chunks, trimmed below
        pad = (lambda x: jnp.concatenate([x, jnp.repeat(x[:1], n_pad, axis=0)], axis=0)) if n_pad else (lambda x: x)

        @jax.jit
        def chunk_fn(bp, g, o, uu, cc):
            phi, b = self.mesh_phi_b(o, uu, g, params=bp)              # (rows, G, z_dim), (rows, G)
            if cc is None:
                return phi.mean(axis=1), b.mean(axis=1), None, None
            return (phi.mean(axis=1), b.mean(axis=1),
                    jnp.take_along_axis(phi, cc[:, :, None], axis=1), jnp.take_along_axis(b, cc, axis=1))

        obs_p, u_p = pad(obs), pad(u)
        cols_p = None if cols is None else pad(cols)
        parts = [chunk_fn(self.basis.params, goals, obs_p[lo:lo + rows], u_p[lo:lo + rows],
                          None if cols_p is None else cols_p[lo:lo + rows])
                 for lo in range(0, N + n_pad, rows)]

        def cat(i):
            return jnp.concatenate([p[i] for p in parts], axis=0)[:N]

        return cat(0), cat(1), (None if cols is None else cat(2)), (None if cols is None else cat(3))

    def _lp_cache(self, obs, u, goals, key):
        """Everything the Lagrangian reads on the (N pairs x G goals) mesh, computed once with
        the frozen measure. M is a straight line in w, so w never passes through the network.

        phi_mean (N, z_dim), b_mean (N,): per-pair means over ALL G goals -- the objective.
        cols (N, C): the goal columns the constraint and the multiplier read for each pair,
            C = min(G, LP_CONS_COLS). G <= LP_CONS_COLS: every column, in order (the whole
            mesh). Otherwise C columns drawn per pair without replacement, fixed for all steps.
        phi_cons (N, C, z_dim), b_cons (N, C): the mesh at those columns.
        l_obs, l_u, l_goals (N*C, .): the same entries as the multiplier's inputs, row-major."""
        N, G = obs.shape[0], goals.shape[0]
        C = min(G, int(LP_CONS_COLS))
        if C == G:
            cols = jnp.broadcast_to(jnp.arange(G), (N, G))
        else:
            cols = jnp.argsort(jax.random.uniform(key, (N, G)), axis=1)[:, :C]
        phi_mean, b_mean, phi_cons, b_cons = self._mesh_goal_means(obs, u, goals, cols=cols)
        return {"phi_mean": phi_mean, "b_mean": b_mean, "cols": cols,
                "phi_cons": phi_cons, "b_cons": b_cons,
                "l_obs": jnp.repeat(obs, C, axis=0), "l_u": jnp.repeat(u, C, axis=0),
                "l_goals": goals[cols].reshape(N * C, goals.shape[-1])}

    def inference_step_cached(self, state, cache):
        """`inference_step` (squared measure) on the `_lp_cache` arrays: same losses, same
        optimizers, same projection. obj = mean_ij (phi_ij^T w + b_ij) over the whole mesh,
        read from the per-pair means; the constraint phi^T w + b and the multiplier l are
        evaluated on the cached columns. `inference_step` permutes the pairs each step, which
        leaves every mean it takes unchanged, so no permutation is drawn here."""
        c = self.config
        use_dgd = bool(c["use_dgd"])
        w_tx = optax.adam(c["lr_infer"])
        l_tx = optax.adam(c["lr_l"])
        phi_cons, b_cons = cache["phi_cons"], cache["b_cons"]

        def multiplier(l_params):
            return self.l(cache["l_obs"], cache["l_u"], cache["l_goals"], params=l_params).reshape(b_cons.shape)

        def w_loss(w):
            obj = jnp.mean(cache["phi_mean"] @ w + cache["b_mean"])
            cons = phi_cons @ w + b_cons
            if use_dgd:
                pen = -jnp.mean(cons * jax.lax.stop_gradient(multiplier(state.l_params)))
            else:
                pen = -jnp.mean(jnp.minimum(cons, 0.0)) * c["inf_coeff"]
            return -obj + pen, (obj, cons)

        (_, (obj, cons)), g_w = jax.value_and_grad(w_loss, has_aux=True)(state.w)
        upd, w_opt = w_tx.update(g_w, state.w_opt)
        w = self._project(optax.apply_updates(state.w, upd))

        l_params, l_opt = state.l_params, state.l_opt
        if use_dgd:
            cons_sg = jax.lax.stop_gradient(cons)
            _, g_l = jax.value_and_grad(lambda mp: jnp.mean(cons_sg * multiplier(mp)))(l_params)
            upd_l, l_opt = l_tx.update(g_l, l_opt)
            l_params = optax.apply_updates(l_params, upd_l)

        info = {"obj": obj, "viol_frac": jnp.mean((cons < 0.0).astype(jnp.float32)),
                "viol_mean": jnp.mean(jax.nn.relu(-cons)), "w_norm": jnp.linalg.norm(w)}
        return state.replace(w=w, w_opt=w_opt, l_params=l_params, l_opt=l_opt), info

    def run_inference_cached(self, obs, u, goals, key):
        """`run_inference` for a large goal set (squared measure): the frozen phi and b are
        computed once (`_lp_cache`) and the w / l steps run on the cached arrays. Same
        initial w and multiplier as `run_inference`. Returns the coefficient w (z_dim,)."""
        assert not self._softmax(), "the cached Lagrangian is the squared measure's (measure_loss=squared)"
        state = self.init_inference(key)
        cache = self._lp_cache(obs, u, goals, jax.random.fold_in(key, 13))
        step = jax.jit(self.inference_step_cached)
        for _ in range(int(self.config["num_inference_steps"])):
            state, _ = step(state, cache)
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
        Q(s, u_m) = mean_{g in G} phi(s,u_m,g)^T eval_w_star + b(s,u_m,g).
        measure_loss=softmax: the score is the softmax mass on G over [G + eval_ref].
        coef_source=hgoal_each: `_hgoal_each_score`, each goal at its own h(g_j).
        coef_source=reward_gpi: `_reward_gpi_score`, the max over candidate coefficients of
        the expected state reward over eval_ref.
        coef_source=pool_reward: `_pool_reward_score`, the max over candidate coefficients of the
        importance-weighted reward integral over eval_bank."""
        c = self.config
        assert observations.ndim == 1, "select_latent acts on a single observation"
        K, d_a = int(c["gpi_num_u"]), c["action_dim"]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, observations.shape[-1]))
        if self._hgoal_each():
            q = self._hgoal_each_score(obs, u_cand, self.eval_goals, self.eval_goal_w, self.eval_ref)
        elif self._reward_gpi():
            q = self._reward_gpi_score(obs, u_cand, self.eval_cand_w, self.eval_ref, self.eval_ref_reward)
        elif self._pool_reward():
            q = self._pool_reward_score(obs, u_cand, self.eval_cand_w, self.eval_bank, self.eval_bank_r,
                                        self.eval_bank_w)
        elif self._ridge_reward_score():
            q = self._reward_gpi_score(obs, u_cand, self.eval_cand_w, self.eval_ref, self.eval_ref_reward)
        else:
            w = jnp.broadcast_to(self.eval_w_star, (K, self.eval_w_star.shape[0]))
            q = self._goal_score(obs, u_cand, self.eval_goals, self.eval_ref, w)   # (K,)
        return u_cand[jnp.argmax(q)]

    def select_latent_sfbc(self, observations, seed):
        """acting=sfbc: draw sfbc_num_u candidate latents from the trained actor a_theta(s,
        eval_w_star) (reparameterised, clipped to u_clip), rerank by the goal-averaged measure
        Q(s, u) = mean_g M(s,u,g,eval_w_star), return the argmax. Select-from-behaviour: the
        candidates come from the state-conditioned BC actor, a tighter support than the raw
        prior draws acting=gpi scans."""
        c = self.config
        assert observations.ndim == 1, "select_latent_sfbc acts on a single observation"
        K, d_a, scale = int(c["sfbc_num_u"]), c["action_dim"], float(c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, observations.shape[-1]))
        w = jnp.broadcast_to(self.eval_w_star, (K, self.eval_w_star.shape[0]))
        mu, log_std = self.actor(obs, w)
        u_cand, _ = tanh_gaussian_sample(mu, log_std, jax.random.normal(seed, (K, d_a)), scale)
        u_cand = jnp.clip(u_cand, -scale, scale)
        q = jnp.mean(self.mesh_M(obs, u_cand, self.eval_goals, w), axis=1)   # (K,)
        return u_cand[jnp.argmax(q)]

    def select_latent_flowbc_gpi(self, observations, seed):
        """acting=flowbc_gpi (2026-10-09, eval only): acting=gpi with the candidates drawn from the
        flowbc actor instead of the prior. Candidate k: eps_k a clipped prior draw, fed the goal
        g_k = eval_goals[k mod K_g]; u_k = clip(eps_k + delta(s, g_k, eps_k)). Scored as acting=gpi
        under coef_source=hgoal_each (`_hgoal_each_score`); the argmax is returned."""
        c = self.config
        assert observations.ndim == 1, "select_latent_flowbc_gpi acts on a single observation"
        K, d_a = int(c["gpi_num_u"]), c["action_dim"]
        eps = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, observations.shape[-1]))
        g = self.eval_goals[jnp.arange(K) % self.eval_goals.shape[0]]
        u_cand = self.flowbc_latent(obs, g, eps)[0]
        q = self._hgoal_each_score(obs, u_cand, self.eval_goals, self.eval_goal_w, self.eval_ref)
        return u_cand[jnp.argmax(q)]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """The deployed action. acting=gpi decodes the argmax prior draw; acting=distill samples
        the distilled/in-loop DSRL actor's mode; acting=sfbc reranks actor draws by the measure.
        actor_input=goal: the actor is fed the goal state eval_actor_goal, not the coefficient."""
        seed = self.rng if seed is None else seed
        if self.config["acting"] == "distill":
            actor_in = self.eval_actor_goal if self._actor_takes_goal() else self.eval_w_star
            if self._actor_kind() == "flowbc":
                # one prior draw eps per step, corrected by delta(s, g, eps) (policy_index=goal) or
                # delta(s, w, eps) at the ridge w (policy_index=reward)
                eps = jax.random.normal(seed, (1, self.config["action_dim"]))
                u_star = self.flowbc_latent(observations[None], actor_in[None], eps)[0][0]
            else:
                mu, _ = self.actor(observations[None], actor_in[None])
                u_star = self.config["u_clip"] * jnp.tanh(mu[0])
        elif self.config["acting"] == "sfbc":
            u_star = self.select_latent_sfbc(observations, seed)
        elif self.config["acting"] == "flowbc_gpi":
            u_star = self.select_latent_flowbc_gpi(observations, seed)
        else:
            u_star = self.select_latent(observations, seed)
        return self.decode(observations[None], u_star[None])[0]

    # gpi_hold (2026-10-04, eval only): utils/evaluation.py keeps the latent `choose_latent`
    # returns for gpi_hold env steps and decodes it at each step's state with `act_with_latent`;
    # the two together at one key equal `sample_actions` under acting=gpi. The counter lives
    # in the eval loop, which knows the episode boundaries; the agent stays stateless.
    @jax.jit
    def choose_latent(self, observations, seed):
        """acting=gpi: the argmax prior draw at one observation, (d_a,)."""
        return self.select_latent(observations, seed)

    @jax.jit
    def act_with_latent(self, observations, u):
        """The action G(s, u) of a given latent at one observation."""
        return self.decode(observations[None], u[None])[0]

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
    def fb_eval_task(self, next_obs, rewards, chunk=512):
        """The ridge task vector from reward samples on a relabel pool: f on the pool at the
        stored index tower and anchors (chunked), then `fb_task_fit`. Returns (D,)."""
        feat = jax.jit(lambda s: self.fb_feature(s, self.fb_zenc, self.fb_anchor))
        parts = []
        for lo in range(0, next_obs.shape[0], chunk):
            parts.append(np.asarray(feat(jnp.asarray(next_obs[lo:lo + chunk], jnp.float32))))
        f_pool = jnp.asarray(np.concatenate(parts, axis=0))
        r = jnp.asarray(np.asarray(rewards, np.float32).reshape(-1))
        return self.fb_task_fit(f_pool, r)[0]

    def trained_coef(self):
        """coef_source=trained: the coefficient the measure was trained with, (z_dim,).
        policy_index=code: the mean of w(z) over TRAINED_COEF_CODES policy codes z drawn from
        a fixed key, projected onto the sqrt(z_dim) sphere like the other sources. It reads no
        reward and runs no Lagrangian or regression; the task enters only through the goal set
        the score is taken over."""
        if self._goal_index():
            raise ValueError("coef_source=trained reads w(z) on the policy code; under "
                             "policy_index=goal the trained coefficient is h(g): use coef_source=amortized")
        z = sample_z_bin(jax.random.PRNGKey(TRAINED_COEF_SEED), TRAINED_COEF_CODES,
                         int(self.config["max_log_seed"]))
        return self._project(jnp.mean(self.w(z), axis=0))

    def infer_eval_goals(self, batch, rewards, env_goal=None, goal_pool=None):
        """Set the goal set G and the eval coefficient from a relabel batch. G = k_goals
        rewarding next states; the coefficient is the LP solution (or h(g) if amortized, or
        the mean of w(z) over sampled policy codes if trained).

        eval_goal_pool=dataset: G is k_goals rewarding next states of `goal_pool`, drawn
        without replacement. `goal_pool` is the task dataset as {"next_observations": (P, ob),
        "rewards": (P,) shifted like `rewards`}. The (s, u) pairs still come from `batch`; the
        Lagrangian runs on a cached mesh (`run_inference_cached`) and coef_source=regression
        averages phi over G in chunks. eval_goal_pool=relabel ignores `goal_pool`.

        measure_loss=softmax also stores batch_size - k_goals non-rewarding next states as the
        readout's reference columns. With acting=distill and actor_input=goal no coefficient
        is inferred: the actor is fed one rewarding state (eval_goal_source=relabel) or
        `env_goal`, the env's own task goal (eval_goal_source=env).

        coef_source=reward_gpi: `rewards` is a continuous state reward r(s'_i) over the batch
        rows (no threshold). G = k_goals candidate goals chosen by reward_gpi_cands, with
        eval_cand_w = h(G); eval_ref = R uniform rows of the batch's next states, with
        eval_ref_reward their rewards. No coefficient is inferred.

        coef_source=pool_reward: eval_bank = pool_eval_size next states of the batch, n_r =
        min(pool_eval_size // 4, rewarding rows) of them rewarding, with eval_bank_r their rewards and
        eval_bank_w the class importance weights (`pool_reward_bank_rows`); eval_cand_w = h(G) over
        the k_goals rewarding goals (policy_index=goal) or the single bank w (policy_index=bank, which
        needs no goal set, so a batch without a rewarding row is accepted). No coefficient is inferred."""
        c = self.config
        rewards = np.asarray(rewards).reshape(-1)
        k = int(c["k_goals"])
        from_dataset = str(c.get("eval_goal_pool", "relabel")) == "dataset"
        reward_gpi = self._reward_gpi()
        pool_reward = self._pool_reward()
        ref_reward = self.eval_ref_reward
        cand_w = self.eval_cand_w
        bank, bank_r, bank_w = self.eval_bank, self.eval_bank_r, self.eval_bank_w
        if pool_reward and self._bank_index():
            goals = self.eval_goals          # the bank's single w needs no goal set
        elif reward_gpi:
            next_obs_all = np.asarray(batch["next_observations"])
            pick = reward_gpi_candidate_rows(rewards, k, str(c.get("reward_gpi_cands", "both")))
            goals = jnp.asarray(next_obs_all[pick], jnp.float32)
            n_ref = int(self.eval_ref.shape[0])
            ref_rows = np.random.choice(rewards.shape[0], size=n_ref, replace=rewards.shape[0] < n_ref)
            ref_reward = jnp.asarray(rewards[ref_rows], jnp.float32)
        elif from_dataset:
            assert goal_pool is not None, \
                "eval_goal_pool=dataset needs goal_pool (the task dataset's next states and rewards)"
            pool_rows = np.nonzero(np.asarray(goal_pool["rewards"]).reshape(-1) > REWARDING_THRESHOLD)[0]
            if pool_rows.size < k:
                raise ValueError(f"infer_eval_goals: the goal pool has {pool_rows.size} rewarding rows "
                                 f"(reward > {REWARDING_THRESHOLD}), fewer than k_goals={k}")
            pick = np.random.choice(pool_rows, size=k, replace=False)
            goals = jnp.asarray(np.asarray(goal_pool["next_observations"])[pick], jnp.float32)
        else:
            rows = np.nonzero(rewards > REWARDING_THRESHOLD)[0]
            if rows.size == 0:
                raise ValueError("infer_eval_goals: relabel batch has no rewarding row "
                                 f"(reward > {REWARDING_THRESHOLD}); cannot build a goal set")
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
        coef_source = str(c.get("coef_source", "lp"))
        refs = self.eval_ref
        if pool_reward:
            bank_rows, omega = pool_reward_bank_rows(rewards, int(c["pool_eval_size"]))
            bank = jnp.asarray(np.asarray(batch["next_observations"])[bank_rows], jnp.float32)
            bank_r = jnp.asarray(rewards[bank_rows], jnp.float32)
            bank_w = jnp.asarray(omega, jnp.float32)
        if reward_gpi:
            # Reference states of the reward readout (both losses): uniform rows of the batch.
            refs = jnp.asarray(np.asarray(batch["next_observations"])[ref_rows], jnp.float32)
        elif self._softmax():
            # Reference columns of the softmax score: non-rewarding next states of the batch.
            other = np.nonzero(rewards <= REWARDING_THRESHOLD)[0]
            assert other.size > 0, "measure_loss=softmax needs non-rewarding relabel rows as reference columns"
            n_ref = int(self.eval_ref.shape[0])
            ref_rows = np.random.choice(other, size=n_ref, replace=other.size < n_ref)
            refs = jnp.asarray(np.asarray(batch["next_observations"])[ref_rows], jnp.float32)
        actor_goal = self.eval_actor_goal
        goal_w = self.eval_goal_w
        goal_actor = c["acting"] == "distill" and self._actor_takes_goal()
        if goal_actor:
            if str(c.get("eval_goal_source", "relabel")) == "env":
                assert env_goal is not None, \
                    "eval_goal_source=env needs the env's task goal (info['goal'] on reset)"
                actor_goal = jnp.asarray(np.asarray(env_goal), jnp.float32).reshape(-1)
                assert actor_goal.shape == self.eval_actor_goal.shape, \
                    f"env goal has shape {actor_goal.shape}, observations {self.eval_actor_goal.shape}"
            else:
                actor_goal = goals[0]            # one rewarding dataset state
            w = jnp.zeros_like(self.eval_w_star)  # the actor takes the goal; no coefficient is read
        elif coef_source == "amortized":
            # A/B goal_conditioned: amortized coefficient from the trained goal head.
            # w = normalize(mean_{g in G} h(g)) -- the design's sum-over-goals readout.
            w = self._project(jnp.mean(self.w_star(goals), axis=0))
        elif coef_source == "hgoal_each":
            # Each goal keeps its own h(g_j); the score is taken per goal (`_hgoal_each_score`),
            # so there is no single coefficient.
            goal_w = self.w_star(goals)
            w = jnp.zeros_like(self.eval_w_star)
        elif coef_source == "reward_gpi":
            # Each candidate goal keeps its own h(g_c); the score is the max over candidates of
            # the expected state reward (`_reward_gpi_score`), so there is no single coefficient.
            cand_w = self.w_star(goals)
            goal_w = cand_w
            w = jnp.zeros_like(self.eval_w_star)
        elif coef_source == "pool_reward":
            # The candidate coefficients the reward integral is maximised over: h(g_c) per goal, or
            # the bank's one w. No single coefficient.
            cand_w = self.w_star(goals) if self._goal_index() else self.bank_w()
            w = jnp.zeros_like(self.eval_w_star)
        elif coef_source == "regression":
            # Least-squares coefficient on the same features the LP path reads. Fit reward from
            # the goal-averaged phi: Phi[i] = mean_{g in G} phi(s'_i, u'_i, g), w = lstsq(Phi, r).
            # Then mean_g phi^T w tracks reward by construction (diagnostic 2026-09-21: the
            # amortized readout was reward-uncorrelated, the least-squares one was not). u is the
            # sub-rows' preimage read above; next states and reward use the SAME sub rows.
            r_sub = jnp.asarray(rewards[sub], jnp.float32)
            next_obs = jnp.asarray(np.asarray(batch["next_observations"])[sub], jnp.float32)
            ng, nrow = goals.shape[0], next_obs.shape[0]
            if from_dataset:
                # the same per-row mean over the goal set, MESH_ROWS mesh rows per network call
                phi_mean = self._mesh_goal_means(next_obs, u, goals)[0]          # (n, z_dim)
            elif self._factorized():
                # same (row, goal) mesh, built from A once per row
                phi_mean = self.mesh_phi_b(next_obs, u, goals)[0].mean(axis=1)   # (n, z_dim)
            else:
                o_r = jnp.broadcast_to(next_obs[:, None], (nrow, ng, next_obs.shape[-1])).reshape(nrow * ng, -1)
                u_r = jnp.broadcast_to(u[:, None], (nrow, ng, u.shape[-1])).reshape(nrow * ng, -1)
                g_r = jnp.broadcast_to(goals[None], (nrow, ng, goals.shape[-1])).reshape(nrow * ng, -1)
                phi_part, _ = self.basis(o_r, u_r, g_r, params=self.basis.params)
                phi_mean = phi_part.reshape(nrow, ng, -1).mean(axis=1)     # (n, z_dim)
            w_reg = jnp.linalg.lstsq(phi_mean, r_sub, rcond=None)[0]   # (z_dim,)
            w = self._project(w_reg)
        elif coef_source == "ridge":
            # Factored-FB infer_cond: centered ridge fit of the task's reward (shifted; the
            # centering removes the shift) on f(s+) over the WHOLE relabel batch, at the
            # stored EMA moments and index tower, normalized to sqrt(D). Zero-shot: no training.
            w = self.fb_eval_task(np.asarray(batch["next_observations"]), rewards)
            if self._ridge_reward_score():
                # ridge_score=reward: best-of-64 by the expected relabel reward over eval_ref uniform rows
                # of the batch (`_reward_gpi_score`) at the single ridge w (every candidate row = w).
                cand_w = jnp.broadcast_to(w[None], (k, w.shape[0]))
                n_ref = int(self.eval_ref.shape[0])
                rows_r = np.random.choice(rewards.shape[0], size=n_ref, replace=rewards.shape[0] < n_ref)
                refs = jnp.asarray(np.asarray(batch["next_observations"])[rows_r], jnp.float32)
                ref_reward = jnp.asarray(rewards[rows_r], jnp.float32)
        elif coef_source == "trained":
            # The coefficient M was trained with: project(mean_z w(z)). No inference is run.
            w = self.trained_coef()
        elif from_dataset and not self._softmax():
            # the same Lagrangian on phi and b computed once on the (pairs x goals) mesh
            w = self.run_inference_cached(obs, u, goals, key)
        else:
            # RLU Lagrangian inference (default)
            w = self.run_inference(obs, u, goals, key, refs=refs if self._softmax() else None)
        agent = self.replace(eval_goals=goals, eval_w_star=w, eval_ref=refs, eval_actor_goal=actor_goal,
                             eval_goal_w=goal_w, eval_cand_w=cand_w, eval_ref_reward=ref_reward,
                             eval_bank=bank, eval_bank_r=bank_r, eval_bank_w=bank_w)
        # acting=distill re-distills the actor at eval by default. With eval_redistill=false the
        # in-loop-trained actor (train_actor) is used directly, without eval re-distillation.
        if c["acting"] == "distill" and bool(c.get("eval_redistill", True)):
            agent = agent.distill_actor(obs, jax.random.fold_in(self.rng, 11))
        return agent


def reward_gpi_candidate_rows(rewards, k, mode):
    """coef_source=reward_gpi: the k candidate rows of a relabel batch, by `mode`
    (REWARD_GPI_CANDS). top: the k highest-reward rows. random: k uniform rows without
    replacement. both: k//2 top rows plus k - k//2 uniform rows from the remaining ones.
    Draws from the global numpy stream like the other eval row picks. With replacement only
    when the batch is smaller than the request."""
    rewards = np.asarray(rewards).reshape(-1)
    n = rewards.shape[0]
    assert mode in REWARD_GPI_CANDS, f"reward_gpi_cands must be one of {REWARD_GPI_CANDS}"
    order = np.argsort(-rewards, kind="stable")
    if mode == "top":
        return order[:k] if n >= k else np.random.choice(order, size=k, replace=True)
    if mode == "random":
        return np.random.choice(n, size=k, replace=n < k)
    k_top = k // 2
    top = order[:k_top]
    rest = order[k_top:]
    if rest.size == 0:
        rest = np.arange(n)
    rand = np.random.choice(rest, size=k - k_top, replace=rest.size < k - k_top)
    return np.concatenate([top, rand])


def pool_reward_bank_rows(rewards, size):
    """coef_source=pool_reward: the bank's rows of a relabel batch and their importance weights.
    n_r = min(size // 4, number of rewarding rows (reward > REWARDING_THRESHOLD)) rewarding rows and
    size - n_r non-rewarding rows, each class without replacement when it is large enough (with,
    otherwise); omega_j = (count of j's class in the batch) / (count of that class in the bank), so
    sum_j omega_j = the batch size and each class weighs what it weighs in the batch. A class absent
    from the batch is absent from the bank (the other class fills it). Draws from the global numpy
    stream like the other eval row picks. Returns (rows (size,) int, omega (size,) float32)."""
    rewards = np.asarray(rewards).reshape(-1)
    size = int(size)
    rew_rows = np.nonzero(rewards > REWARDING_THRESHOLD)[0]
    other_rows = np.nonzero(rewards <= REWARDING_THRESHOLD)[0]
    assert rew_rows.size + other_rows.size > 0, "pool_reward_bank_rows: empty relabel batch"
    n_r = min(size // 4, rew_rows.size)
    if other_rows.size == 0:
        n_r = size
    n_o = size - n_r
    picks, omega = [], []
    if n_r > 0:
        picks.append(np.random.choice(rew_rows, size=n_r, replace=rew_rows.size < n_r))
        omega.append(np.full((n_r,), rew_rows.size / n_r, np.float32))
    if n_o > 0:
        picks.append(np.random.choice(other_rows, size=n_o, replace=other_rows.size < n_o))
        omega.append(np.full((n_o,), other_rows.size / n_o, np.float32))
    return np.concatenate(picks), np.concatenate(omega)


def get_config():
    """Importable default config, mirrored by configs/agent/psmgoal.yaml."""
    import ml_collections

    return ml_collections.ConfigDict({
        "agent_name": "psmgoal",
        "batch_size": 256,          # mesh is batch^2, so smaller than f_psmflow's 1024
        "z_dim": 128,               # D: width of phi and of the coefficient
        "discount": 0.98,           # gamma
        "tau": 0.01,                # Polyak rate of the phi,b,w targets
        "measure_form": "joint",    # joint (RLUMeasure on [s,u,g]) | factorized (A(s,u)^T f(g), beta^T f(g))
        "f_dim": 128,               # factorized: width of f(g) (on the sqrt(f_dim) sphere)
        "f_ortho_coef": 1.0,        # factorized: weight on ||E[f f^T] - I||_F^2 over s+ (ignored when joint)
        "ortho_coef": 0.0,          # F1: weight on ||mean phi phi^T - I||_F^2 over the TD mesh (0 = off)
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
        "eval_goal_pool": "relabel",    # rows G is drawn from: relabel (the relabel batch) | dataset (the task dataset)
        # --- acting ---
        "acting": "gpi",            # gpi (argmax over prior draws) | distill (DSRL actor) | sfbc (actor draws reranked) | flowbc_gpi (flowbc actor draws reranked)
        "gpi_num_u": 64,            # prior draws the gpi argmax scans
        "sfbc_num_u": 64,           # acting=sfbc: actor draws reranked by the measure
        "num_actor_steps": 10000,   # distillation steps (acting=distill)
        "actor_temp": 0.0,          # entropy weight (alpha) in the distillation objective
        "q_coeff": 1.0,             # distill: weight on the (scale-normalised) Q term
        "bc_coeff": 0.0,            # distill: prior-centre anchor weight (in-support pull)
        # --- FB-style in-loop actor+measure interleaving (OFF-equivalent by default) ---
        "train_actor": False,       # co-train the actor in update(); when true it supplies the TD backup u'
        "train_basis": True,        # step phi, b, w(z); False freezes the whole measure (actor-only arm)
        "eval_redistill": True,     # acting=distill re-distills at eval; False acts with the in-loop actor
        # in-loop actor loss (train_actor=true only): distill (DSRL, out-of-support) | awr | bc (in-support)
        "actor_objective": "distill",
        "awr_beta": 3.0,            # awr temperature: weight_i = clip(exp(adv_i/awr_beta), 0, awr_wmax)
        "awr_wmax": 100.0,          # awr weight ceiling
        "awr_baseline_k": 32,       # awr baseline: prior draws averaged for V_i
        # --- Factored-FB induced-reward actor (OFF by default: actor_value=none) ---
        "actor_value": "none",      # none | measure_reward_raw | measure_reward_softmax
        "fb_value_temp": 1.0,       # softmax arm: p_j = softmax_j(M_j / temp)
        "fb_bc_coeff": 3.0,         # in-support anchor weight on mean (u - u_data)^2
        "bootstrap_delta_scale": 1.0,  # factor on the flowbc actor's step in the TD target only
        "frozen_delta_path": None,     # bootstrap_source=frozen_delta: run dir whose delta params are loaded
        "frozen_delta_epoch": 500000,  # ... at this checkpoint
        "frozen_delta_mode": "plus",   # plus | minus | random | zero
        "fb_rand_frac": 0.5,        # fraction of rows whose task is a random sqrt(D) direction
        "fb_hit_spec": "19:22:0.4",  # kernel reward blocks a:b:eps (cube: cube xyz, eps 0.4)
        "fb_ridge_lambda_mult": 0.01,   # lam = mult * tr(C)/D
        "fb_cov_eig_floor": 1.0e-6,     # eigenvalue floor on C, multiple of tr(C)/D
        "fb_const_eps": 1.0e-6,         # ||c|| below this -> neutral task 1_D
        "fb_enc_tau": 0.0005,       # EMA rate of the slow basis copy that f is read from
        "fb_sigma_tau": 0.005,      # EMA rate of the f mean/covariance
        "fb_k_u": 8,                # f: prior latents averaged
        "fb_k_anchor": 32,          # f: fixed anchor goal states averaged (first batch's s+)
        "fb_proj_seed": 0,          # f: code seed of the prior latents
        # --- A/B goal_conditioned mode (all OFF by default -> byte-identical to the RLU core) ---
        "train_goal_head": False,   # train h(g) by J(theta|g)+constraint during update()
        # lp (RLU Lagrangian) | amortized (h(g)) | regression (lstsq) | ridge (actor_value)
        # | trained (policy_index=code: mean of w(z) over 256 fixed policy codes, no inference)
        # | hgoal_each, reward_gpi (policy_index=goal, eval only) | pool_reward (policy_index=goal|bank,
        # eval only: the importance-weighted reward integral over a bank of pool_eval_size relabel next states)
        "coef_source": "lp",
        "reward_actor_reward": "latent_centered",  # policy_index=reward, actor reward rows: the index's centered view | the exact kernel reward on hindsight rows (kernel)
        "ridge_score": "goal",      # policy_index=reward, eval: best-of-64 by the goal mass at w (goal) | the expected relabel reward over eval_ref at w (reward)
        "pool_eval_size": 512,      # coef_source=pool_reward: bank size (a quarter rewarding at most)
        # coef_source=reward_gpi: candidate goals = top (k_goals highest-reward rows) | random
        # (k_goals uniform rows) | both (k_goals//2 top + the rest uniform)
        "reward_gpi_cands": "both",
        "lr_goal": 1.0e-4,          # h(g) learning rate
        "j_constraint_coef": 1.0,   # hinge weight on non-negativity in J(theta|g)
        "w_star": {"hidden_dim": 256, "hidden_layers": 2},   # h(g) net
        "goal_discount": 0.98,      # dataset hindsight-goal geometric horizon (main.py)
        "goal_random_frac": 0.3,    # dataset off-trajectory random-goal fraction (OGBench-style)
        # --- goal-conditioned training (2026-10-01; all OFF by default) ---
        # code (w(z) on the binary policy code) | goal (w = h(g), actor bootstrap at g) | bank (n_bank=1:
        # the behaviour policy, one shared w = w(z) at a constant code, bootstrap = a clipped prior draw)
        "policy_index": "code",
        "n_bank": 1,                # policy_index=bank: members of the bank; only 1 is implemented
        # squared (TD on the mesh) | softmax (cross-entropy over the batch's next states) | pool
        # (the softmax cross-entropy over a fixed pool of pool_size dataset states, one-hot on the
        # pool state nearest the row's next state; docs/design/2026-10-19-pool-measure.md)
        "measure_loss": "squared",
        "measure_temp": 1.0,        # softmax temperature, in the loss and the readouts
        "pool_size": 1024,          # measure_loss=pool: P, states in the reference pool
        "pool_seed": 0,             # measure_loss=pool: seed of the pool's draw from the valid-preimage rows
        "pool_hash": ml_collections.config_dict.placeholder(str),   # set by create: sha1 of the pool's rows
        # measure_loss=softmax: K later states of each row's own trajectory (offset ~ Geometric(1 -
        # goal_discount), clipped at the trajectory end) join the batch's next states as the
        # softmax normalisation set of the loss and its target (S = N + N*K columns). 0 = off.
        "norm_future_k": 0,
        "actor_input": "w",         # w (the coefficient) | goal (the raw goal state)
        "goal_cur_frac": 0.0,       # share of rows whose measure goal is their own s'
        "eval_goal_source": "relabel",  # actor readout's goal: relabel (a rewarding dataset state) | env (the task goal)
        # --- data bootstrap (2026-10-03; OFF by default) ---
        # actor (the goal-fed actor's mode at (s', g)) | data (the data's next latent u_{i+1};
        # needs policy_index=goal, goal_random_frac=0, goal_cur_frac=0; train_actor may be false)
        "bootstrap_source": "actor",
        # bootstrap_source=argmax: number of clipped prior draws scored at s' (1 = a BC draw);
        # bc_mean: number of draws the target is averaged over
        "bootstrap_n": 8,
        # --- 2026-10-04 ablations (OFF by default) ---
        # policy_index=goal: geometric (hindsight offset ~ Geometric(1 - goal_discount)) |
        # uniform (the row's own s' and every later state of its trajectory, equally likely)
        "goal_sampling": "geometric",
        # acting=gpi, eval only: env steps one chosen latent is held before the next argmax
        # (utils/evaluation.py); the action is decoded at the current state every step
        "gpi_hold": 1,
        # --- joint actor heads (2026-10-05; OFF by default) ---
        # policy_index=goal train_actor=true actor_input=goal: tanh (the tanh-Gaussian actor
        # with the fb_bc_coeff anchor at the data latent) | flowbc (u = clip(eps + delta(s,g,eps)),
        # BC anchor fb_bc_coeff*mean delta^2 at the prior draw eps) | dsrl (tanh-Gaussian, no BC
        # term, alpha*logp with alpha learned; needs fb_bc_coeff=0)
        "actor_kind": "tanh",
        # the value the goal actor climbs: shaped (kernel-weighted measure over the batch's
        # next states) | point (M(s,u,g) at the actor goal; softmax: g's share among g + pool)
        "actor_value_kind": "shaped",
        # --- design-decision ablations (2026-10-09; OFF by default) ---
        # future (uniform later state of the row's trajectory, batch['fb_goals']) | measure (batch['goals'])
        "actor_goal_source": "future",
        # actor_kind=flowbc: none | prior_then_actor | actor_then_prior (the BC draw before / from
        # bootstrap_switch_step, the actor's latent otherwise)
        "bootstrap_switch": "none",
        "bootstrap_switch_step": 100000,
        # actor_kind=dsrl: SAC's alpha, the f_psmflow dsrl_sac head's values
        "dsrl_actor": {"target_entropy": 0.0, "init_alpha": 1.0, "lr_alpha": 3.0e-4},
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
