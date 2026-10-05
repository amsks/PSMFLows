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
POLICY_INDEXES = ("code", "goal")
#: measure_loss options. squared = TD on the mesh; softmax = cross-entropy over the columns.
MEASURE_LOSSES = ("squared", "softmax")
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
#: goal g_j is scored with its own coefficient h(g_j), no single w (2026-10-03).
GOAL_COEF_SOURCES = ("hgoal_each",)
#: bootstrap_source options (policy_index=goal). actor = the goal-fed actor's mode at (s', g);
#: data = the data's next latent u_{i+1} (batch['next_noise_preimage']), no actor trained.
#: docs/design/2026-10-03-psmgoal-data-bootstrap.md.
BOOTSTRAP_SOURCES = ("actor", "data")
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
#: fold_in constant of the fresh prior draw eps the flowbc bootstrap uses at (s', g).
FLOWBC_BOOT_KEY = 103
#: name of the flowbc actor's output layer in its params (zero-initialised).
DELTA_OUT_LAYER = "out"


def softmax_td_loss(M, M_bar, discount, temp):
    """Cross-entropy between softmax_j(M_ij / temp) and the stop-gradded target
    (1 - discount) * onehot(j = i) + discount * softmax_j(M_bar_ij / temp), mean over rows.
    M, M_bar (N, N): row i is (s_i, u_i) resp. (s'_i, u'_i), column j the batch's next state
    s+_j. Both softmaxes ignore a per-row shift of their logits."""
    target = (1.0 - discount) * jnp.eye(M.shape[0]) + discount * jax.nn.softmax(M_bar / temp, axis=-1)
    logp = jax.nn.log_softmax(M / temp, axis=-1)
    return -jnp.mean(jnp.sum(jax.lax.stop_gradient(target) * logp, axis=-1))


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
    log_alpha: Any = None        # actor_kind=dsrl: SAC's log alpha TrainState

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert config.get("encoder", None) is None, "psmgoal does not support visual encoders."
        coef_source = str(config.get("coef_source", "lp"))
        assert coef_source in COEF_SOURCES + GOAL_COEF_SOURCES, \
            "coef_source must be one of lp | amortized | regression | ridge | trained | hgoal_each"
        actor_value = str(config.get("actor_value", "none"))
        assert actor_value in ACTOR_VALUES, f"actor_value must be one of {ACTOR_VALUES}"
        if actor_value != "none":
            assert bool(config.get("train_actor", False)), \
                "actor_value != none trains the in-loop actor: set train_actor=true"
        if str(config.get("coef_source", "lp")) == "ridge":
            assert actor_value != "none", "coef_source=ridge reads the actor_value feature state"
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
        if policy_index == "goal":
            assert bool(config.get("train_actor", False)) or bootstrap_source == "data", \
                "policy_index=goal bootstraps from the goal-fed actor: set train_actor=true (or bootstrap_source=data)"
            assert not bool(config.get("train_goal_head", False)), \
                "policy_index=goal trains h by the measure loss; train_goal_head must be false"
            assert actor_value == "none", "policy_index=goal has its own actor loss; set actor_value=none"
            assert str(config.get("coef_source", "lp")) != "trained", (
                "coef_source=trained reads w(z) on the policy code; under policy_index=goal the "
                "trained coefficient is h(g): use coef_source=amortized")
        if measure_loss == "softmax":
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
            assert policy_index == "goal", f"actor_kind={actor_kind} is the goal actor: needs policy_index=goal"
            assert bool(config.get("train_actor", False)), f"actor_kind={actor_kind} trains an actor: set train_actor=true"
            assert actor_input == "goal", f"actor_kind={actor_kind} takes the raw goal: set actor_input=goal"
        if actor_kind == "dsrl":
            assert float(config.get("fb_bc_coeff", 3.0)) == 0.0, \
                "actor_kind=dsrl has no BC term: set fb_bc_coeff=0"
        if actor_value_kind == "point":
            assert policy_index == "goal", "actor_value_kind=point reads M(s,u,g) at w = h(g): needs policy_index=goal"
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
        delta = TrainState.create(
            delta_def, delta_def.init(jax.random.fold_in(r_a, 1), ex_observations, ex_observations, ex_u)["params"],
            tx=optax.adam(config["lr_actor"]))
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
                   delta=delta, log_alpha=log_alpha)

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

    def _softmax(self):
        return str(self.config.get("measure_loss", "squared")) == "softmax"

    def _actor_takes_goal(self):
        return str(self.config.get("actor_input", "w")) == "goal"

    def _data_bootstrap(self):
        return str(self.config.get("bootstrap_source", "actor")) == "data"

    def _hgoal_each(self):
        return str(self.config.get("coef_source", "lp")) == "hgoal_each"

    def _actor_kind(self):
        return str(self.config.get("actor_kind", "tanh"))

    def _point_value(self):
        return str(self.config.get("actor_value_kind", "shaped")) == "point"

    def _coef(self, z, goals, params=None, target=False):
        """The per-row coefficient w of the measure, (N, z_dim). policy_index=code: w(z) on the
        binary policy code z. policy_index=goal: w = h(g) on the row's goal, z unused.
        target=True reads the Polyak copy; otherwise `params` (None -> the online params)."""
        if self._goal_index():
            return self.w_star(goals, params=self.target_w_star if target else params)
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
        the move the data made at s'_i), clipped like the row's own u."""
        c = self.config
        if self._data_bootstrap():
            return jnp.clip(jnp.asarray(batch["next_noise_preimage"]), -c["u_clip"], c["u_clip"])
        if bool(c.get("train_actor", False)):
            next_obs = jnp.asarray(batch["next_observations"])
            goals = jnp.asarray(batch["goals"]) if self._goal_index() else None
            if self._actor_takes_goal():
                if self._actor_kind() == "flowbc":
                    # the flowbc actor at (s', g) with a fresh prior draw eps
                    eps = jax.random.normal(jax.random.fold_in(self.rng, FLOWBC_BOOT_KEY),
                                            (next_obs.shape[0], int(c["action_dim"])))
                    return self.flowbc_latent(next_obs, goals, eps)[0]
                # tanh and dsrl: the tanh-Gaussian actor's mode
                return self.actor_backup(next_obs, goals)
            return self.actor_backup(next_obs, self._coef(z, goals, target=True))
        N = jnp.asarray(batch["observations"]).shape[0]
        index = jnp.asarray(batch["index"]) if "index" in batch else jnp.arange(N)
        return self.proto_bootstrap(z, index)

    # ------------------------------------------------------------------ basis loss
    def measure_loss(self, basis_params, w_params, batch, z, u_next):
        """RLU `update_psm`: squared TD on the off-diagonal of the state x goal mesh plus a
        `-(1-gamma)` diagonal pull. Gradient reaches phi, b (basis) and w (w).

        policy_index=goal: the row's coefficient is w_i = h(g_i) on its measure goal
        batch['goals'] (w_params are then h's params, and the target reads h's Polyak copy);
        the columns stay the batch's next states. measure_loss=softmax swaps the squared TD
        for `softmax_td_loss` on the same mesh."""
        c = self.config
        obs = jnp.asarray(batch["observations"])
        next_obs = jnp.asarray(batch["next_observations"])
        u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        goals = next_obs                                         # g_j = s'_j
        row_goals = jnp.asarray(batch["goals"]) if self._goal_index() else None
        w = self._coef(z, row_goals, params=w_params)            # (N, z), online
        w_t = self._coef(z, row_goals, target=True)              # (N, z), target
        phi, b = self.mesh_phi_b(obs, u, goals, params=basis_params)             # (N, N, z), (N, N)
        M = (phi * w[:, None, :]).sum(-1) + b                                     # (N, N)
        M_bar = self.mesh_M(next_obs, u_next, goals, w_t, params=self.target_basis)
        target_M = jax.lax.stop_gradient(M_bar)
        if self._softmax():
            return self._softmax_measure_loss(M, target_M, phi, w)
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
        states. Logs the row softmax p next to the raw M: its mass on the row's own next state,
        its largest entry, and the spread max_j - min_j of the logits M_ij/measure_temp."""
        c = self.config
        temp = float(c["measure_temp"])
        loss = softmax_td_loss(M, target_M, c["discount"], temp)
        p = jax.nn.softmax(M / temp, axis=-1)
        off = 1.0 - jnp.eye(M.shape[0])
        phi_flat = phi.reshape(-1, phi.shape[-1])
        gram = phi_flat.T @ phi_flat / phi_flat.shape[0]
        info = {"psm_loss": loss,
                "m_softmax_diag": jnp.mean(jnp.diagonal(p)),
                "m_softmax_max": jnp.mean(jnp.max(p, axis=-1)),
                "m_logit_spread": jnp.mean(jnp.max(M, axis=-1) - jnp.min(M, axis=-1)) / temp,
                "phi_eff_rank": jnp.trace(gram) ** 2 / jnp.maximum(jnp.sum(gram ** 2), 1e-12),
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
        goals = jnp.asarray(batch["fb_goals"])
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
        u_next = self._select_u_next(batch, z)
        new_agent, info = self.apply_update(batch, z, u_next)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss at current params (no step)."""
        rng = rng if rng is not None else self.rng
        N = jnp.asarray(batch["observations"]).shape[0]
        z = sample_z_bin(rng, N, int(self.config["max_log_seed"]))
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
        coef_source=hgoal_each: `_hgoal_each_score`, each goal at its own h(g_j)."""
        c = self.config
        assert observations.ndim == 1, "select_latent acts on a single observation"
        K, d_a = int(c["gpi_num_u"]), c["action_dim"]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, observations.shape[-1]))
        if self._hgoal_each():
            q = self._hgoal_each_score(obs, u_cand, self.eval_goals, self.eval_goal_w, self.eval_ref)
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

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """The deployed action. acting=gpi decodes the argmax prior draw; acting=distill samples
        the distilled/in-loop DSRL actor's mode; acting=sfbc reranks actor draws by the measure.
        actor_input=goal: the actor is fed the goal state eval_actor_goal, not the coefficient."""
        seed = self.rng if seed is None else seed
        if self.config["acting"] == "distill":
            actor_in = self.eval_actor_goal if self._actor_takes_goal() else self.eval_w_star
            if self._actor_takes_goal() and self._actor_kind() == "flowbc":
                # one prior draw eps per step, corrected by delta(s, g, eps)
                eps = jax.random.normal(seed, (1, self.config["action_dim"]))
                u_star = self.flowbc_latent(observations[None], actor_in[None], eps)[0][0]
            else:
                mu, _ = self.actor(observations[None], actor_in[None])
                u_star = self.config["u_clip"] * jnp.tanh(mu[0])
        elif self.config["acting"] == "sfbc":
            u_star = self.select_latent_sfbc(observations, seed)
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
        `env_goal`, the env's own task goal (eval_goal_source=env)."""
        c = self.config
        rewards = np.asarray(rewards).reshape(-1)
        k = int(c["k_goals"])
        from_dataset = str(c.get("eval_goal_pool", "relabel")) == "dataset"
        if from_dataset:
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
        if self._softmax():
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
                             eval_goal_w=goal_w)
        # acting=distill re-distills the actor at eval by default. With eval_redistill=false the
        # in-loop-trained actor (train_actor) is used directly, without eval re-distillation.
        if c["acting"] == "distill" and bool(c.get("eval_redistill", True)):
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
        "acting": "gpi",            # gpi (argmax over prior draws) | distill (DSRL actor) | sfbc (actor draws reranked)
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
        "coef_source": "lp",
        "lr_goal": 1.0e-4,          # h(g) learning rate
        "j_constraint_coef": 1.0,   # hinge weight on non-negativity in J(theta|g)
        "w_star": {"hidden_dim": 256, "hidden_layers": 2},   # h(g) net
        "goal_discount": 0.98,      # dataset hindsight-goal geometric horizon (main.py)
        "goal_random_frac": 0.3,    # dataset off-trajectory random-goal fraction (OGBench-style)
        # --- goal-conditioned training (2026-10-01; all OFF by default) ---
        "policy_index": "code",     # code (w(z) on the binary policy code) | goal (w = h(g), actor bootstrap at g)
        "measure_loss": "squared",  # squared (TD on the mesh) | softmax (cross-entropy over the batch's next states)
        "measure_temp": 1.0,        # softmax temperature, in the loss and the readouts
        "actor_input": "w",         # w (the coefficient) | goal (the raw goal state)
        "goal_cur_frac": 0.0,       # share of rows whose measure goal is their own s'
        "eval_goal_source": "relabel",  # actor readout's goal: relabel (a rewarding dataset state) | env (the task goal)
        # --- data bootstrap (2026-10-03; OFF by default) ---
        # actor (the goal-fed actor's mode at (s', g)) | data (the data's next latent u_{i+1};
        # needs policy_index=goal, goal_random_frac=0, goal_cur_frac=0; train_actor may be false)
        "bootstrap_source": "actor",
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
