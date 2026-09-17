"""LatentFlowPSM: successor measures over the latent action space of a frozen behaviour flow.

Actions are indexed by the latent ``u`` of a frozen conditional flow ``G(s, u)``, so every
action the measure evaluates, bootstraps or executes is a flow decode and the bootstrap
distribution equals the data distribution.

The defaults are the affine form of Prop. `bilinear`: ``psi_form=affine``,
``policy_index=latent``, ``train_actor=false``, ``acting=gpi``. Every other arm is an
ablation, and the two loss stabilisers (``ortho_mode``, ``psi_bound``) default OFF.

``docs/reference/psmflow-symbols.md`` is the full code<->paper symbol map, the defaults and
the stabilisers -- read it before touching the losses. File order: construction, losses,
update, acting, inference.

DESIGN FORK -- the affine head under the default index and a faithful DSRL arm do NOT
compose. Under ``policy_index=latent`` the TD bootstrap latent is the prior
draw u', not the actor's: ``sample_step_inputs`` computes the actor's u^+ and then
overwrites it. So psi is the successor measure of the constant-latent policy pi_{u'},
which is exactly what makes GPI well-posed -- and exactly why an actor trained against it
is not doing SAC, since the backup never contains the actor's own action. The one
configuration in which the latent MDP is genuinely on-policy is

    actor_mode=dsrl_sac policy_index=task_vector psi_form=free index_agg=max
    actor.index_panel=0 train_actor=true acting=actor

and, with ``psi_form=free``, gives up Prop. `bilinear`. Since 2026-09-14 ``psi_form=affine``
also accepts ``policy_index=task_vector`` (w_enc then encodes the task vector), which is the
Section 10 agent on the affine head. Pick one; the config space does not enforce the choice.

That fork applies to an actor climbing PSI. ``dsrl_na.enabled`` (2026-09-08) is the other
resolution: the actor climbs a separate reward-specific dual critic (``qa`` over actions,
``qw`` over latents), psi is not in its gradient at all, and the two therefore compose with
the affine defaults. It is NOT zero-shot -- it is the upper bound that says whether the
frozen flow can be steered on an env, and its number never goes beside a zero-shot row.

``proto.enabled`` (2026-09-14) swaps the critic for the REFERENCE PSM one, on latent inputs:
a proto stage trains phi together with a second tower ``proto_psi(s, z_bin, u)`` on the
measures of a fixed pseudo-random latent-policy family (``utils.psm_proto``), and the
existing psi becomes PSM's separate reward-conditioned SF head, fitted on that phi with phi
stop-gradded and its target read at the online phi. Off, the agent is bit-for-bit the one
above (``tests/test_psmflow_psm_ref.py``).
"""

import copy
import math
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.psm_common import (
    _plain_config, contrastive_loss, off_diagonal_mask, ortho_loss, polyak_update,
    project_z, targets_uncertainty,
)
from utils.psm_networks import (
    AffinePsiMap, FlowVectorField, LogAlpha, NoiseConditionedActor, PhiMap, PsiMap,
    TanhGaussianLatentActor, tanh_gaussian_sample,
)
from utils.psm_proto import proto_latents, proto_seed_ints, sample_z_bin

# gpi_select="small_ball" oversample factor: keeps the accepted pool >= K after
# the median-norm cut.
_SMALL_BALL_OVERSAMPLE = 32

#: Eval-time selection rules over the K prior draws; see `_gpi_select_ablation`.
GPI_SELECT_MODES = ("argmax", "max_norm", "top_quartile_random", "small_ball",
                    "soft_topm", "mean", "fixed_index", "prior_shrunk")

#: Latent-actor heads. "ddpg" is the NoiseConditionedActor + flow-BC recipe; the two DSRL
#: modes share the tanh-Gaussian head.
ACTOR_MODES = ("ddpg", "dsrl_sac", "gpi_distill")

#: 2026-09-08 rename. What shipped as `dsrl_na` is NOT DSRL-NA: DSRL-NA is a dual-critic
#: scheme that trains an ACTION-space critic Q_A on real transitions and distils it into
#: the latent critic, exploiting the fact that many latents decode to the same action.
#: Ours regresses the actor onto the argmax of `na_candidates` prior draws scored by the
#: GPI rule -- candidate-argmax behaviour distillation (the SfBC/IDQL family), with no
#: second critic and no action-space signal. Old flags.json files are mapped on read.
ACTOR_MODE_ALIASES = {"dsrl_na": "gpi_distill"}

#: Defaults for the `actor` sub-keys, applied in `create` so a partial dict -- a test
#: passing only the widths, or a flags.json written before a key existed -- still builds.
ACTOR_DEFAULTS = dict(index_panel=0, q_coeff=1.0, na_coeff=1.0, na_candidates=16,
                      na_states=256, na_advantage_weight=False, entropy="auto",
                      target_entropy=0.0, init_alpha=1.0, lr_alpha=3.0e-4,
                      log_std_min=-20.0, log_std_max=2.0,
                      prior_init=False, prior_init_std=0.3, layer_norm=False)


def _actor_opt(config, key):
    """`config.actor[key]`, falling back to ACTOR_DEFAULTS when the dict lacks it."""
    try:
        return config["actor"][key]
    except (KeyError, AttributeError):
        return ACTOR_DEFAULTS[key]


#: DSRL-NA keys (2026-09-08). `enabled=False` is the OFF value: the branch builds its two
#: heads either way (the pytree must be static across the switch, as everywhere else here)
#: but never steps them and never reaches acting, so a config predating the block restores
#: onto exactly the behaviour it had.
DSRL_NA_DEFAULTS = dict(enabled=False, discount=0.99, tau=0.005, lr=3.0e-4,
                        hidden_dim=2048, hidden_layers=3, layer_norm=True,
                        num_ensembles=2, inner_steps=10, n_latent=1,
                        reward_source="real", reward_shift=1.0,
                        task_conditioned=False, reward_refit_every=10000,
                        # 2026-09-15 (Fix 1): continuing-formulation TD target (mask=1
                        # everywhere) and a scale on the synthetic reward phi(s')^T w.
                        ignore_masks=False, reward_scale=1.0)

#: Where `dsrl_qa_loss` gets its reward.
#:   real         Item 2's arm: the task's own reward. NOT zero-shot.
#:   phi_readout  Arm D1: r_hat = phi(s')^T w with w = project(E_batch[(r+shift) phi]), the
#:                deployed zero-shot reward channel. Still reward-specific.
#:   synthetic_w  Arm D2: r_w = phi(s')^T w at the SAMPLED task vector, which is what makes
#:                the arm zero-shot -- no real reward enters training at any point.
#:   phi_readout_fixed
#:                Arm D1b: the same channel as `phi_readout`, fit the way the agent
#:                deploys it. `phi_readout` refits w on each 256-row BATCH, where a
#:                2.1%-sparse reward leaves ~5 rewarding rows, so w is the mean of about
#:                five phi vectors and is redrawn every step -- Q_A is then trained on a
#:                per-batch random reward, and its measured std was 13.6 against the real
#:                reward's 0.15. Here w is the closed form on a relabel batch of
#:                `eval_relabel_size` rows (the estimator `infer_z` uses at eval), refit
#:                every `reward_refit_every` steps and held FIXED in between, and r_hat is
#:                rescaled so its std over that batch equals the real reward's. Refit and
#:                rescale are driven from main.py via `refit_na_reward`, not from inside
#:                the jitted update.
DSRL_NA_REWARD_SOURCES = ("real", "phi_readout", "phi_readout_fixed", "synthetic_w")


def _na_opt(config, key):
    """`config.dsrl_na[key]`, falling back to DSRL_NA_DEFAULTS when the config predates it."""
    try:
        return config["dsrl_na"][key]
    except (KeyError, AttributeError, TypeError):
        return DSRL_NA_DEFAULTS[key]


def fill_dsrl_na_defaults(config):
    """Give `config` a complete `dsrl_na` block, in place where the container allows.

    Same contract as `fill_actor_defaults`: absent means OFF, so an older flags.json
    restores onto the old behaviour rather than dying on the first read.
    """
    try:
        if "dsrl_na" not in config:
            config["dsrl_na"] = dict(DSRL_NA_DEFAULTS)
        else:
            for k, v in DSRL_NA_DEFAULTS.items():
                if k not in config["dsrl_na"]:
                    config["dsrl_na"][k] = v
    except (AttributeError, KeyError, TypeError):
        pass
    return config


#: Reference PSM proto stage keys (2026-09-14). `enabled=False` is the OFF value: the proto
#: tower is built either way (the pytree must be static across the switch) but never
#: sampled for, stepped or read, so a config predating the block restores onto exactly the
#: behaviour it had. `ortho_coef=None` means "the agent's own ortho_coef".
PROTO_DEFAULTS = {"enabled": False, "max_log_seed": 16, "proto_seed": 0, "lr": 1.0e-4,
                  "ortho_coef": None}


def _proto_opt(config, key):
    """`config.proto[key]`, falling back to PROTO_DEFAULTS when the config predates it."""
    try:
        return config["proto"][key]
    except (KeyError, AttributeError, TypeError):
        return PROTO_DEFAULTS[key]


def fill_proto_defaults(config):
    """Give `config` a complete `proto` block, in place where the container allows.

    Same contract as `fill_dsrl_na_defaults`: absent means OFF, so an older flags.json
    restores onto the old behaviour rather than dying on the first read.
    """
    try:
        if "proto" not in config:
            config["proto"] = dict(PROTO_DEFAULTS)
        else:
            for k, v in PROTO_DEFAULTS.items():
                if k not in config["proto"]:
                    config["proto"][k] = v
    except (AttributeError, KeyError, TypeError):
        pass
    return config


#: Loss-stabiliser keys (2026-09-07). Every default is the OFF value -- the loss the
#: published numbers used. `fill_stability_defaults` backfills an older flags.json.
STABILITY_DEFAULTS = dict(ortho_mode="fixed", ortho_rel_coef=1.0,
                          psi_bound="none", psi_bound_scale=200.0)

#: Measure/acting keys `create` guards with `.get(<legacy default>)` but the runtime then
#: reads unguarded. Backfilled with the SAME legacy values, so a flags.json predating any
#: of them restores onto the old behaviour instead of dying at the first update or eval.
MEASURE_DEFAULTS = dict(psi_form="free", index_agg="max", gpi_select="argmax",
                        index_clip=None, index_panel=16, expectile_mu=0.9,
                        mask_invalid_preimages=False, measure_u_samples=1,
                        measure_action_input="latent",
                        measure_u_source="mixture", measure_u_jitter_std=0.3,
                        measure_u_mixture_shrink=None,
                        reward_inference="closed_form", reward_inference_eps=1e-3,
                        # 2026-09-15 (Fix 2): scalar grounding of the readout psi^T w and
                        # the dueling affine head. Both OFF values reproduce the loss and
                        # the head every number before this date was produced with.
                        psm_scalar_coef=0.0, psi_dueling=False, psi_dueling_samples=8,
                        # 2026-09-15 (PSM Eq. 10): `acting=fixed_coeff` reads the affine
                        # head at ONE policy coefficient c loaded from this npz instead of
                        # the per-step u' panel. None = the seam is inert.
                        fixed_index_coeff_path=None,
                        # 2026-09-16: EMaQ-style backup. `index` reproduces every earlier
                        # arm bit-for-bit (u_next = u' under latent, the actor's latent
                        # under task_vector).
                        bootstrap="index", bootstrap_candidates=8)

#: `bootstrap`: what fills the backup's action slot at s'. `gpi_argmax` (task_vector only)
#: is argmax over `bootstrap_candidates` clipped prior draws of the pessimistic
#: psi_target(s', w, u_m)^T w -- EMaQ's max-over-behaviour-samples backup, so psi(s, w, u)
#: is the successor measure of "take u, then act by GPI-under-w", i.e. of the deployed policy.
BOOTSTRAP_MODES = ("index", "gpi_argmax")

#: `acting`: how `sample_actions` picks the latent it decodes.
ACTING_MODES = ("gpi", "actor", "fixed_coeff")

#: How `infer_z` turns a relabelling batch into w. See `PSMFlowAgent.infer_z`.
REWARD_INFERENCE_MODES = ("closed_form", "whitened")

#: Where `measure_u_samples > 1` gets its extra action latents.
MEASURE_U_SOURCES = ("mixture", "jitter")

#: `ortho_mode`: how the orthonormality regulariser is weighted against the TD term.
ORTHO_MODES = ("fixed", "relative")
#: `psi_bound`: where the magnitude ceiling on the measure is imposed, if anywhere.
PSI_BOUND_MODES = ("none", "tanh", "clip_target")


def _stab_opt(config, key):
    """`config[key]`, falling back to STABILITY_DEFAULTS when the config predates it."""
    try:
        return config[key]
    except (KeyError, AttributeError):
        return STABILITY_DEFAULTS[key]


def fill_stability_defaults(config):
    """Give `config` every key in STABILITY_DEFAULTS, in place where the container allows.

    Same contract as `fill_actor_defaults`: every key added here must read as its OFF
    default when absent, so an older flags.json restores onto the old behaviour.
    """
    try:
        for k, v in STABILITY_DEFAULTS.items():
            if k not in config:
                config[k] = v
    except (AttributeError, KeyError, TypeError):
        pass
    return config


def fill_actor_defaults(config):
    """Give `config.actor` every key in ACTOR_DEFAULTS, in place where the container allows.

    Called as the FIRST statement of `create`, so no read below it can hit a key the
    caller's config predates -- a restored checkpoint's flags.json is by definition older
    than the code restoring it. Pinned by `tests/test_psmflow_config_compat.py`. Silently
    does nothing on a locked ConfigDict; the `_actor_opt` reads and the post-`_plain_config`
    backfill in `create` cover that case.
    """
    try:
        actor = config["actor"]
        for k, v in ACTOR_DEFAULTS.items():
            if k not in actor:
                actor[k] = v
        if "actor_mode" not in config:
            config["actor_mode"] = "ddpg"
        config["actor_mode"] = ACTOR_MODE_ALIASES.get(config["actor_mode"],
                                                      config["actor_mode"])
    except (AttributeError, KeyError, TypeError):
        pass
    return config


@flax.struct.dataclass
class StepInputs:
    """Quantities drawn once per update and shared by every branch.

    u_data   (B, d_a) dataset latent, the flow preimage of the recorded action
    u_next   (B, d_a) bootstrap latent at s' (the write-up's u^+)
    task_w   (B, z_dim) task vector w, shared by the measure and actor branches
    u_index  (B, d_a) policy index u' ~ p0, or None under policy_index='task_vector'
    flow_*   the CFM draws for the latent actor's behaviour-cloning anchor
    task_w_a (B, z_dim) task vector of the ACTION branch: task_w unless the FB graft is
             on, in which case it is sampled against that branch's own basis B_a
    u_valid  (B,) 1.0 where the Stage-B inversion converged, else 0.0; all ones unless
             `mask_invalid_preimages`
    u_extra  (n-1, B, d_a) further action latents for the SAME transition, drawn from the
             stored EM preimage mixture; None unless `measure_u_samples` > 1
    z_bin    (B, max_log_seed) binary proto code; None unless `proto.enabled`
    u_proto  (B, d_a) the fixed pseudo-random latent policy pi_{z_bin}'s draw at this row,
             the proto stage's continuation at s'; None unless `proto.enabled`
    u_duel   (K, d_a) the clipped prior panel the dueling head's advantage baseline is
             averaged over, shared by every psi call of this update; None unless
             `psi_dueling`
    """
    u_data: Any
    u_next: Any
    task_w: Any
    u_index: Any
    flow_x0: Any
    flow_t: Any
    flow_noise: Any
    task_w_a: Any
    u_valid: Any
    u_extra: Any
    z_bin: Any = None
    u_proto: Any = None
    u_duel: Any = None
    u_adv: Any = None       # bootstrap=gpi_argmax: Q(u*) - mean_m Q(u_m) at s', for telemetry


class PSMFlowAgent(flax.struct.PyTreeNode):
    """LatentFlowPSM. See `docs/reference/psmflow-symbols.md` for the symbol map."""

    rng: Any
    phi: TrainState             # basis phi(x) over future states
    psi: TrainState             # successor features psi(s, index, u), P-fold ensemble
    target_phi: Any
    target_psi: Any
    # Reference PSM proto stage (`proto.enabled`): the codebook tower proto_psi(s, z_bin, u),
    # trained WITH phi on the measures of a fixed pseudo-random latent-policy family; psi
    # above is then PSM's SF head on that basis. Built unconditionally so the pytree is
    # static across the switch; a checkpoint predating it keeps the fresh params.
    proto_psi: TrainState
    target_proto_psi: Any
    actor: TrainState           # amortized latent actor pi_eta(s, w, eps) -> u
    actor_vf: TrainState        # v_xi: CFM velocity field over preimage latents
    # DSRL latent actor (`actor_mode` != 'ddpg'). Always created so the pytree stays
    # static across the switch; a checkpoint predating it keeps the fresh params.
    sac_actor: TrainState       # tanh-Gaussian pi(u | s, w) over the latent box
    log_alpha: TrainState       # SAC entropy coefficient, held in log space
    # `index_agg='expectile'`: upper-expectile regression onto psi(s, u', u)^T w over
    # prior u'. On (s, w, u), not (s, u) -- w is redrawn per row, so a head without it
    # could only fit the task-marginal expectile.
    q_dist: TrainState
    # DSRL-NA (`dsrl_na.enabled`; NOT zero-shot) -- the dual critic the 2026-09-08 audit
    # found missing. `qa(s, a)` is a scalar TD critic on the task's REAL reward; `qw(s, u)`
    # is fitted by regression onto qa(s, G(s, u)) at prior latents and is the only thing
    # the latent actor climbs. `actor_mode=dsrl_sac` supplies DSRL's ACTOR; this pair
    # supplies DSRL's CRITIC. Both are needed for an arm that is actually DSRL-NA.
    qa: TrainState
    target_qa: Any
    qw: TrainState
    # Action branch (`action_critic.enabled`): successor features over EXECUTED actions,
    # psi_a(s, w, a), plus a bounded residual delta(s, w, u); a = G(s, u) + eps * delta.
    psi_a: TrainState
    target_psi_a: Any
    residual: TrainState
    # `action_critic.fb_graft`: the action branch's own backward map B_a, the basis its
    # measure is written against and the source of w_a. Trained only under the graft.
    phi_a: TrainState
    target_phi_a: Any
    flow_vf: Any                # FROZEN: multi-step behaviour-flow velocity field
    flow_onestep: Any           # FROZEN: one-step distilled decoder
    # Arm D1b: the reward-readout w held FIXED between refits, and the scalar that puts
    # r_hat on the real reward's scale. Set by `refit_na_reward`; unused by every other
    # reward_source, and zero-initialised so a checkpoint predating them restores cleanly.
    na_rw: Any                  # (z_dim,) reward-readout task vector
    na_rw_scale: Any            # () scalar, std(r_true) / std(phi @ na_rw)
    task_z: Any                 # (z_dim,) eval task vector w, set by infer_eval_z
    task_z_a: Any               # (z_dim,) eval task vector of the action branch
    # `acting=fixed_coeff` (2026-09-15, PSM Eq. 10): ONE policy coefficient c in R^{w_dim}
    # that replaces w(u') in psi = A(s,u)^T c + beta(s,u) at every step. Loaded from
    # `fixed_index_coeff_path` in `create`; zeros (and unread) under every other `acting`.
    fixed_coeff: Any
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        # FIRST: a config written before an `actor` key existed (any restored checkpoint's
        # flags.json) must read as that key's pre-existing default. See fill_actor_defaults.
        fill_actor_defaults(config)
        fill_stability_defaults(config)
        fill_dsrl_na_defaults(config)
        fill_proto_defaults(config)
        rng = jax.random.PRNGKey(seed)
        rng, rphi, rpsi, rvf, ronestep = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "psmflow does not support visual encoders yet."
        assert config.get("index_agg", "max") in ("max", "expectile"), "index_agg: max | expectile"
        assert not (config.get("index_agg", "max") == "expectile"
                    and config["policy_index"] != "latent"), (
            "index_agg=expectile requires policy_index=latent: under task_vector the index "
            "slot carries w, so there is no index distribution to take an expectile over.")
        gpi_select = config.get("gpi_select", "argmax")
        assert gpi_select in GPI_SELECT_MODES, f"gpi_select: {'|'.join(GPI_SELECT_MODES)}"
        assert config["acting"] in ACTING_MODES, f"acting: {'|'.join(ACTING_MODES)}"
        bootstrap = config.get("bootstrap", "index")
        assert bootstrap in BOOTSTRAP_MODES, f"bootstrap: {'|'.join(BOOTSTRAP_MODES)}"
        assert not (bootstrap == "gpi_argmax" and config["policy_index"] != "task_vector"), (
            "bootstrap=gpi_argmax defines the continuation policy as GPI under the task vector "
            "w, so psi's index slot must carry w: it needs policy_index=task_vector")
        coeff_path = config.get("fixed_index_coeff_path", None)
        assert not (config["acting"] == "fixed_coeff" and not coeff_path), (
            "acting=fixed_coeff needs fixed_index_coeff_path (an npz with `c` of shape "
            "(w_dim,), written by tools/infer_policy_lagrangian.py)")
        assert not (config["acting"] == "fixed_coeff"
                    and (config.get("psi_form", "free") != "affine"
                         or config["policy_index"] != "latent")), (
            "acting=fixed_coeff reads psi = A(s,u)^T c + beta(s,u) at a fixed c, which "
            "exists only under psi_form=affine with policy_index=latent")
        assert not (gpi_select != "argmax"
                    and (config["policy_index"] != "latent"
                         or config.get("index_agg", "max") != "max")), (
            "gpi_select != argmax requires policy_index=latent and index_agg=max: the "
            "selection ablations act on the (u, u') pair scan, which the other arms do "
            "not run.")
        actor_mode = ACTOR_MODE_ALIASES.get(config.get("actor_mode", "ddpg"),
                                            config.get("actor_mode", "ddpg"))
        config["actor_mode"] = actor_mode
        assert actor_mode in ACTOR_MODES, f"actor_mode: {'|'.join(ACTOR_MODES)}"
        assert not (actor_mode != "ddpg" and not config["train_actor"]), (
            f"actor_mode={actor_mode} trains the tanh-Gaussian latent head; it needs "
            "train_actor=true (and acting=actor to deploy it)")
        assert _actor_opt(config, "entropy") in ("auto", "fixed"), (
            "actor.entropy: auto | fixed")
        panel = int(_actor_opt(config, "index_panel"))
        assert panel >= 0, "actor.index_panel must be >= 0"
        assert not (panel > 0 and config["policy_index"] != "latent"), (
            "actor.index_panel > 0 aggregates psi over POLICY latents u'; under "
            "policy_index=task_vector the index slot carries w and there is no panel.")
        assert not (panel > 0 and config.get("index_agg", "max") != "max"), (
            "actor.index_panel > 0 requires index_agg=max: under expectile the index "
            "distribution is already marginalized into q_dist and the panel is redundant.")
        assert not (actor_mode == "gpi_distill" and panel == 0), (
            "actor_mode=gpi_distill regresses onto max over the index panel, so panel=0 would "
            "silently distil a ONE-index argmax while the deployed rule scans gpi_num_u; "
            "set actor.index_panel > 0.")
        # DSRL-NA is the ACTOR plus the CRITIC. Shipping one without the other is the
        # 2026-09-08 mislabelling; these asserts are what stop it recurring.
        if _na_opt(config, "enabled"):
            assert _na_opt(config, "reward_source") in DSRL_NA_REWARD_SOURCES, (
                f"dsrl_na.reward_source: {'|'.join(DSRL_NA_REWARD_SOURCES)}")
            assert actor_mode == "dsrl_sac", (
                "dsrl_na.enabled supplies DSRL's dual CRITIC; actor_mode=dsrl_sac supplies "
                "DSRL's tanh-Gaussian ACTOR. DSRL-NA is both -- the 2026-09-08 audit found "
                "we had shipped the actor alone and called it DSRL-NA.")
            assert config["acting"] == "actor", (
                "dsrl_na deploys the latent actor it trains; acting=gpi would evaluate the "
                "actor-free argmax over psi, which this arm never fits.")
            assert not (_na_opt(config, "reward_source") == "synthetic_w"
                        and not _na_opt(config, "task_conditioned")), (
                "dsrl_na.reward_source=synthetic_w draws a fresh task vector per row, so the "
                "critics and the actor must SEE it: set dsrl_na.task_conditioned=true. "
                "Without that the arm trains one head against a reward that changes every "
                "batch, which is noise, not zero-shot.")
            assert not (_na_opt(config, "reward_source") == "phi_readout_fixed"
                        and int(_na_opt(config, "reward_refit_every")) <= 0), (
                "dsrl_na.reward_source=phi_readout_fixed needs reward_refit_every > 0. "
                "With no refit the held w stays at its zero init and the reward is "
                "identically 0, which trains Q_A on nothing.")
            assert not (_na_opt(config, "task_conditioned")
                        and _na_opt(config, "reward_source") != "synthetic_w"), (
                "dsrl_na.task_conditioned=true with a reward_source other than synthetic_w "
                "conditions the critics on a w the reward does not depend on.")
            assert float(_actor_opt(config, "bc_coeff")) == 0.0, (
                "DSRL-NA has no behaviour-cloning term; set actor.bc_coeff=0. Leaving the "
                "1.0 default on would anchor the actor to the flow prior and the arm would "
                "measure the BC control.")
            assert float(_actor_opt(config, "q_coeff")) > 0.0, (
                "dsrl_na with actor.q_coeff=0 trains an actor that climbs nothing.")
        if _proto_opt(config, "enabled"):
            # The reference critic is proto stage + a w-indexed SF head whose bootstrap is
            # the learned policy's own latent at s'. Under policy_index=latent the SF slot
            # carries u' and the bootstrap is a prior draw, which is a different object.
            assert config["policy_index"] == "task_vector", (
                "proto.enabled fits psi as PSM's SF head psi(s, w, u); it needs "
                "policy_index=task_vector (the index slot carries the task vector).")
            assert config["train_actor"], (
                "proto.enabled bootstraps the SF head at the actor's latent u+ = pi(s', w); "
                "it needs train_actor=true (and acting=actor to deploy it).")
            assert config.get("train_phi", True), (
                "proto.enabled: the proto stage is what trains phi; train_phi=false would "
                "leave the basis at its init and the SF head fitting nothing.")
            assert 1 <= int(_proto_opt(config, "max_log_seed")) <= 30, (
                "proto.max_log_seed must be in [1, 30]: the code is drawn as an int32 in "
                "[0, 2**max_log_seed) and unpacked bitwise.")
        assert config.get("reward_inference", "closed_form") in REWARD_INFERENCE_MODES, (
            f"reward_inference: {'|'.join(REWARD_INFERENCE_MODES)}")
        n_u = int(config.get("measure_u_samples", 1))
        assert n_u >= 1, "measure_u_samples must be >= 1 (1 = the published loss)"
        u_src = config.get("measure_u_source", "mixture")
        assert u_src in MEASURE_U_SOURCES, f"measure_u_source: {'|'.join(MEASURE_U_SOURCES)}"
        assert not (n_u > 1 and u_src == "jitter"
                    and float(config.get("measure_u_jitter_std", 0.3)) <= 0.0), (
            "measure_u_jitter_std must be > 0: at 0 the extra latents are copies of "
            "u_data and the arm is the published loss with n times the compute.")
        # No default for the shrink: it is the ONE knob that decides whether the extra
        # latents decode to this transition's action at all, it differs per environment
        # (cube 0.5, antmaze 1.0, pointmaze 0.5 on the published npz files), and there is no
        # value that is safe everywhere. Requiring it to be stated is the gate that replaces
        # the prior_scale proxy main.py used to apply to this path.
        assert not (n_u > 1 and u_src == "mixture"
                    and config.get("measure_u_mixture_shrink", None) is None), (
            "measure_u_source=mixture needs an explicit measure_u_mixture_shrink. Pick it "
            "per environment with tools/diag_mixture_decode.py --shrink: the largest c "
            "whose mean decode error stays inside the point inverse's own p90.")
        psi_form = config.get("psi_form", "free")
        assert psi_form in ("free", "affine"), "psi_form: free | affine"
        measure_action_input = config.get("measure_action_input", "latent")
        assert measure_action_input in ("latent", "action"), (
            "measure_action_input: latent | action")
        assert not (measure_action_input == "action" and psi_form != "affine"), (
            "measure_action_input=action requires psi_form=affine")
        # measure_action_input=action composes with both indices (2026-09-15, Fix 3; the
        # 09-11 arm ran it under `latent` only). The online row is fitted at the recorded
        # action a_i either way; the bootstrap decodes the index-appropriate latent at s'
        # through the frozen flow -- the same u' under `latent`, the actor's
        # u+ = _deploy_latent(s', task_w) under `task_vector` -- and every other psi query
        # (`_actor_q`, `gpi_select`, the panels) goes through `psi_b` -> `_measure_input`,
        # so the actor's Q is psi(s, G(s, u_actor), w)^T w with gradient through the decoder.
        assert not (measure_action_input == "action" and n_u != 1), (
            "measure_action_input=action requires measure_u_samples=1: the recorded action "
            "defines one exact transition input")
        psi_dueling = bool(config.get("psi_dueling", False))
        assert not (psi_dueling and psi_form != "affine"), (
            "psi_dueling is a decomposition of the AFFINE head (V(s,z) + Adv(s,u,z) - "
            "baseline); it requires psi_form=affine")
        assert not (psi_dueling and measure_action_input == "action"), (
            "psi_dueling averages the advantage tower over PRIOR LATENTS in its action slot; "
            "under measure_action_input=action that slot carries decoded actions, so the "
            "baseline would mix coordinates. Not supported together.")
        assert not (psi_dueling and int(config.get("psi_dueling_samples", 8)) < 1), (
            "psi_dueling_samples must be >= 1: it is the number of prior latents the "
            "advantage baseline is averaged over")
        assert float(config.get("psm_scalar_coef", 0.0)) >= 0.0, "psm_scalar_coef must be >= 0"
        # psi_form=affine composes with both indices (2026-09-14). policy_index=latent: the
        # index slot carries u' and w_enc encodes it into w(u') (Prop. `bilinear`).
        # policy_index=task_vector: the slot carries the z_dim task vector and w_enc encodes
        # THAT (the Section 10 agent on the affine head); w_enc's input width is inferred at
        # init from `ex_index` below. A(s,u) and beta(s,u) never see the index either way.
        ortho_mode = _stab_opt(config, "ortho_mode")
        assert ortho_mode in ORTHO_MODES, f"ortho_mode: {'|'.join(ORTHO_MODES)}"
        assert float(_stab_opt(config, "ortho_rel_coef")) >= 0.0, "ortho_rel_coef must be >= 0"
        psi_bound = _stab_opt(config, "psi_bound")
        assert psi_bound in PSI_BOUND_MODES, f"psi_bound: {'|'.join(PSI_BOUND_MODES)}"
        assert not (psi_bound != "none" and float(_stab_opt(config, "psi_bound_scale")) <= 0.0), (
            "psi_bound_scale must be > 0: it is the per-component ceiling on psi, and the "
            "TD-target clip is its Cauchy-Schwarz image psi_bound_scale * z_dim.")
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))
        ex_w = jnp.zeros((ex_observations.shape[0], z_dim))

        phi_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                         hidden_layers=config["phi"]["hidden_layers"], norm=True)
        if psi_form == "affine":
            af = config["affine"]
            psi_def = AffinePsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                                   num_parallel=config["num_parallel"],
                                   embedding_layers=config["sf"]["embedding_layers"],
                                   hidden_layers=config["sf"]["hidden_layers"],
                                   w_dim=af["w_dim"], encoder_hidden=af["encoder_hidden"],
                                   encoder_layers=af["encoder_layers"], norm_w=af["norm_w"],
                                   dueling=psi_dueling)
        else:
            psi_def = PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                             num_parallel=config["num_parallel"],
                             embedding_layers=config["sf"]["embedding_layers"],
                             hidden_layers=config["sf"]["hidden_layers"])
        phi = TrainState.create(phi_def, phi_def.init(rphi, ex_observations)["params"],
                                tx=optax.adam(config["lr_phi"]))
        # `phi_restore_path` (2026-09-15, Fix 1): phi's parameters from ANOTHER run's
        # params_<phi_restore_epoch>.pkl, so a scalar critic can be trained on phi(s')^T w
        # with the basis of a finished measure run (held fixed by train_phi=false). The
        # checkpoint's ONLINE phi goes into phi and, through the deepcopy below, target_phi;
        # its optimiser state is not carried. Nothing else is read from that checkpoint.
        if config.get("phi_restore_path", None):
            phi = phi.replace(params=_load_phi_params(
                config["phi_restore_path"], config.get("phi_restore_epoch", None), phi.params))
        # psi's index slot: the policy latent u' (d_a wide) under policy_index='latent',
        # the task vector w (z_dim wide) otherwise. The action slot is the latent either way.
        ex_index = ex_u if config["policy_index"] == "latent" else ex_w
        # The dueling head reads a (K, d_a) prior panel on every call, init included.
        ex_duel = ({"u_prior": jnp.zeros((int(config.get("psi_dueling_samples", 8)), action_dim))}
                   if psi_dueling else {})
        psi = TrainState.create(
            psi_def, psi_def.init(rpsi, ex_observations, ex_index, ex_u, **ex_duel)["params"],
            tx=optax.adam(config["lr_sf"]))

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

        # Heads are created unconditionally so the pytree is static across the config
        # switches; keys come from fold_in, so a disabled branch leaves rng untouched.
        sac_actor_def = TanhGaussianLatentActor(
            action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
            hidden_layers=config["actor"]["hidden_layers"],
            embedding_layers=config["actor"]["embedding_layers"],
            log_std_min=_actor_opt(config, "log_std_min"),
            log_std_max=_actor_opt(config, "log_std_max"),
            prior_init=bool(_actor_opt(config, "prior_init")),
            prior_init_std=float(_actor_opt(config, "prior_init_std")),
            layer_norm=bool(_actor_opt(config, "layer_norm")))
        sac_actor = TrainState.create(
            sac_actor_def,
            sac_actor_def.init(jax.random.fold_in(rng, 112), ex_observations, ex_w)["params"],
            tx=optax.adam(config["lr_actor"]))
        log_alpha_def = LogAlpha(init_value=float(math.log(_actor_opt(config, "init_alpha"))))
        log_alpha = TrainState.create(
            log_alpha_def, log_alpha_def.init(jax.random.fold_in(rng, 113))["params"],
            tx=optax.adam(_actor_opt(config, "lr_alpha")))

        # num_ensembles=1: the pessimism this head would otherwise need is already inside
        # its regression target.
        qd_cfg = config["q_dist"]
        q_dist_def = Value(hidden_dims=(qd_cfg["hidden_dim"],) * (qd_cfg["hidden_layers"] + 1),
                           layer_norm=True, num_ensembles=1)
        q_dist = TrainState.create(
            q_dist_def,
            q_dist_def.init(jax.random.fold_in(rng, 108), ex_observations,
                            jnp.concatenate([ex_w, ex_u], -1))["params"],
            tx=optax.adam(qd_cfg["lr"]))

        # DSRL-NA's two scalar critics. Both are `Value`, the same head q_dist uses, at
        # DSRL's offline OGBench widths (3 x 2048, LayerNorm on every hidden layer).
        # qa reads a raw ACTION; qw reads a LATENT. Built unconditionally so the pytree is
        # static across `dsrl_na.enabled`.
        na_cfg = config["dsrl_na"]
        na_dims = (na_cfg["hidden_dim"],) * na_cfg["hidden_layers"]
        # Arm D2: both critics take the task vector alongside the action / latent, so the
        # arm generalises over w instead of solving one task. `Value` concatenates its two
        # arguments, so appending w to the second one is the whole change.
        na_tc = bool(na_cfg["task_conditioned"])
        ex_qa_in = jnp.concatenate([ex_actions, ex_w], -1) if na_tc else ex_actions
        ex_qw_in = jnp.concatenate([ex_u, ex_w], -1) if na_tc else ex_u
        qa_def = Value(hidden_dims=na_dims, layer_norm=na_cfg["layer_norm"],
                       num_ensembles=na_cfg["num_ensembles"])
        qa = TrainState.create(
            qa_def,
            qa_def.init(jax.random.fold_in(rng, 114), ex_observations, ex_qa_in)["params"],
            tx=optax.adam(na_cfg["lr"]))
        qw_def = Value(hidden_dims=na_dims, layer_norm=na_cfg["layer_norm"], num_ensembles=1)
        qw = TrainState.create(
            qw_def, qw_def.init(jax.random.fold_in(rng, 115), ex_observations, ex_qw_in)["params"],
            tx=optax.adam(na_cfg["lr"]))

        # Frozen behaviour flow (a Stage-A FQL bc_only checkpoint). The defs are rebuilt
        # from config; their shapes must match the checkpointed run.
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

        # Action branch: psi_a is a PsiMap whose action slot carries the raw action; the
        # residual is a NoiseConditionedActor read as delta(s, w, u); B_a a second PhiMap.
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
        phi_a_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                           hidden_layers=config["phi"]["hidden_layers"], norm=True)
        phi_a = TrainState.create(
            phi_a_def,
            phi_a_def.init(jax.random.fold_in(rng, 105), ex_observations)["params"],
            tx=optax.adam(ac_cfg["lr"]))

        # Reference PSM proto tower: the same free PsiMap shape as the SF head, index slot
        # max_log_seed wide (the binary code), action slot the latent. fold_in, not split,
        # so the keys above and the agent rng are exactly what they were without it.
        proto_def = PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                           num_parallel=config["num_parallel"],
                           embedding_layers=config["sf"]["embedding_layers"],
                           hidden_layers=config["sf"]["hidden_layers"])
        ex_zbin = jnp.zeros((ex_observations.shape[0], int(_proto_opt(config, "max_log_seed"))))
        proto_psi = TrainState.create(
            proto_def,
            proto_def.init(jax.random.fold_in(rng, 118), ex_observations, ex_zbin, ex_u)["params"],
            tx=optax.adam(float(_proto_opt(config, "lr"))))

        config = _plain_config(config)
        # Backfill the `actor` sub-keys so the runtime never sees a partial dict; every
        # default reproduces the behaviour of a config written before that key existed.
        config.setdefault("train_phi", True)
        config.setdefault("actor_mode", "ddpg")
        for _k, _v in ACTOR_DEFAULTS.items():
            config["actor"].setdefault(_k, _v)
        for _k, _v in STABILITY_DEFAULTS.items():
            config.setdefault(_k, _v)
        for _k, _v in MEASURE_DEFAULTS.items():
            config.setdefault(_k, _v)
        config.setdefault("dsrl_na", {})
        for _k, _v in DSRL_NA_DEFAULTS.items():
            config["dsrl_na"].setdefault(_k, _v)
        config.setdefault("proto", {})
        for _k, _v in PROTO_DEFAULTS.items():
            config["proto"].setdefault(_k, _v)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        # Width of the affine head's coefficient; a free psi has no such slot and never
        # reads the field, so it is zero-width there.
        w_dim_fc = int((config.get("affine") or {}).get("w_dim", 0)) if psi_form == "affine" else 0
        fixed_coeff = jnp.zeros((w_dim_fc,), jnp.float32)
        if config["acting"] == "fixed_coeff":
            fixed_coeff = jnp.asarray(load_fixed_index_coeff(coeff_path, w_dim_fc), jnp.float32)
        return cls(rng=rng, phi=phi, psi=psi,
                   target_phi=copy.deepcopy(phi.params), target_psi=copy.deepcopy(psi.params),
                   proto_psi=proto_psi, target_proto_psi=copy.deepcopy(proto_psi.params),
                   actor=actor, actor_vf=actor_vf,
                   sac_actor=sac_actor, log_alpha=log_alpha, q_dist=q_dist,
                   qa=qa, target_qa=copy.deepcopy(qa.params), qw=qw,
                   psi_a=psi_a, target_psi_a=copy.deepcopy(psi_a.params), residual=residual,
                   phi_a=phi_a, target_phi_a=copy.deepcopy(phi_a.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   na_rw=jnp.zeros((z_dim,), jnp.float32),
                   na_rw_scale=jnp.ones((), jnp.float32),
                   task_z=jnp.zeros((z_dim,), jnp.float32),
                   task_z_a=jnp.zeros((z_dim,), jnp.float32),
                   fixed_coeff=fixed_coeff,
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)

    # ------------------------------------------------------------------ batch sampling
    def _index_clip(self):
        """Box for the policy-index draws u': ``index_clip``, or ``u_clip`` when null."""
        v = self.config["index_clip"]
        return float(self.config["u_clip"]) if v is None else float(v)

    def sample_step_inputs(self, batch, rng):
        """Draw the per-update quantities of `StepInputs` from one batch."""
        c = self.config
        B, adim = batch["observations"].shape[0], c["action_dim"]
        # u_clip clamps every latent to the typical set. Under policy_index='task_vector'
        # the bootstrap is a tanh-bounded actor latent, so an unclipped u_data would feed
        # the online branch inputs the target branch cannot produce. Under 'latent' the
        # bootstrap is a prior draw and that argument does NOT apply -- the clamp is then
        # only the typical-set convention, and it costs G(s, u_data) != a on clipped rows.
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        # Rows whose inversion diverged: `repair_invalid_preimages` reset them to the prior
        # (mixture) or 0 (point) and recorded the mask, so for these u_data does NOT decode
        # to the recorded action and the (u, a) pair the measure is fit on is a fiction.
        # Off by default: every number before 2026-09-08 trained on them.
        valid = batch.get("preimage_valid")
        u_valid = (jnp.asarray(valid).astype(jnp.float32).reshape(-1)
                   if (valid is not None and c["mask_invalid_preimages"])
                   else jnp.ones((B,), jnp.float32))
        # Task vector w: Gaussian, mix_ratio of it replaced by phi(next_obs[perm]) (PSM's
        # sample_mixed_z). stop_gradient so sampling does not shape the basis.
        r_w, r_wmix, r_wperm, r_next, r_x0, r_t, r_noise = jax.random.split(rng, 7)
        w_gauss = project_z(jax.random.normal(r_w, (B, c["z_dim"])), c["norm_z"])
        w_perm = jax.random.permutation(r_wperm, B)
        w_goal = project_z(jax.lax.stop_gradient(self.phi(batch["next_observations"]))[w_perm],
                           c["norm_z"])
        w_mask = (jax.random.uniform(r_wmix, (B,)) < c["mix_ratio"])[:, None]
        task_w = jnp.where(w_mask, w_goal, w_gauss)
        # Policy index u' ~ p0 per row, clipped to index_clip. The key is folded out of rng
        # rather than split, so policy_index='task_vector' leaves every draw below alone.
        u_index = None
        if c["policy_index"] == "latent":
            ic = self._index_clip()
            u_index = jnp.clip(
                jax.random.normal(jax.random.fold_in(rng, 106), (B, adim)), -ic, ic)

        # TD bootstrap: the actor's latent at s' under task_w (the write-up's u^+),
        # tanh-bounded and scaled to the u box.
        u_next = self._deploy_latent(batch["next_observations"], task_w,
                                     jax.random.normal(r_next, (B, adim)))
        # Backup exploration: replace a fraction of bootstrap latents with prior draws, so
        # psi is fit over the whole latent family rather than the slice the actor occupies.
        if c["backup_explore_frac"] > 0.0:
            r_expl, r_emask = jax.random.split(jax.random.fold_in(rng, 107))
            u_prior = jnp.clip(jax.random.normal(r_expl, (B, adim)),
                               -c["u_clip"], c["u_clip"])
            explore_mask = (jax.random.uniform(r_emask, (B,)) < c["backup_explore_frac"])[:, None]
            u_next = jnp.where(explore_mask, u_prior, u_next)
        u_adv = None
        if c["policy_index"] == "task_vector" and c["bootstrap"] == "gpi_argmax":
            # EMaQ-style backup: the continuation at s' is GPI under this row's w, scored
            # with the target psi over bootstrap_candidates prior draws (key folded out of
            # rng, so every draw above is untouched under bootstrap=index).
            u_next, u_adv = self._gpi_argmax_latent(
                batch["next_observations"], task_w, jax.random.fold_in(rng, 108),
                int(c["bootstrap_candidates"]), params=self.target_psi)
        if c["policy_index"] == "latent":
            # The continuation at s' is pi_{u'}, the same index the online side carries --
            # that makes G(s', u') a p0 decode (Prop. `insample`), so explore_frac is inert.
            u_next = u_index
        # `measure_u_samples` > 1: further action latents for the SAME (s, a, s'), drawn
        # from the stored EM preimage posterior. The measure head is otherwise fitted at
        # ONE u per state (u_data) and extrapolates over the whole box, which is why the
        # signal over u is half the ensemble's own disagreement
        # (docs/design/2026-09-08-critic-signal-and-dsrl-na.md 1).
        u_extra = None
        n_extra = int(c["measure_u_samples"]) - 1
        if n_extra > 0:
            k_x = jax.random.fold_in(rng, 116)
            if c["measure_u_source"] == "jitter":
                u_extra = jnp.clip(
                    u_data[None] + c["measure_u_jitter_std"] * jax.random.normal(
                        k_x, (n_extra, B, adim)), -c["u_clip"], c["u_clip"])
            else:
                u_extra = self._sample_preimage_mixture(
                    batch, n_extra, k_x, float(c["measure_u_mixture_shrink"]))
        # FB graft: the action branch gets its own task vector, sampled against its
        # backward map B_a the way FB samples z against B.
        task_w_a = task_w
        if c["action_critic"]["fb_graft"]:
            r_wa, r_wamix, r_waperm = jax.random.split(jax.random.fold_in(rng, 104), 3)
            w_a_gauss = project_z(jax.random.normal(r_wa, (B, c["z_dim"])), c["norm_z"])
            w_a_goal = project_z(
                jax.lax.stop_gradient(self.phi_a(batch["next_observations"]))[
                    jax.random.permutation(r_waperm, B)], c["norm_z"])
            w_a_mask = (jax.random.uniform(r_wamix, (B,)) < c["mix_ratio"])[:, None]
            task_w_a = jnp.where(w_a_mask, w_a_goal, w_a_gauss)
        # Reference PSM proto stage: a binary code per row, and the fixed pseudo-random
        # latent policy's draw at that (row, code) -- the proto target's continuation at
        # s'. The code key is folded out of rng, so the draws above are untouched when
        # the stage is off; the latent draws no run randomness at all (`utils.psm_proto`).
        z_bin = u_proto = None
        if c["proto"]["enabled"]:
            if "index" not in batch:
                raise KeyError(
                    "proto.enabled needs batch['index'], the dataset ROW index: keyed on "
                    "batch position a transition meets a different proto policy on every "
                    "resample and proto_psi has no fixed policy to converge to. main.py sets "
                    "dataset.return_index for this arm.")
            n_bits = int(c["proto"]["max_log_seed"])
            z_bin = sample_z_bin(jax.random.fold_in(rng, 117), B, n_bits)
            u_proto = proto_latents(proto_seed_ints(z_bin, batch["index"], n_bits), adim,
                                    c["u_clip"], jax.random.PRNGKey(int(c["proto"]["proto_seed"])))
        return StepInputs(
            u_data=u_data, u_next=u_next, task_w=task_w, u_index=u_index,
            flow_x0=jax.random.normal(r_x0, (B, adim)),
            flow_t=jax.random.uniform(r_t, (B, 1)),
            flow_noise=jax.random.normal(r_noise, (B, adim)),
            task_w_a=task_w_a, u_valid=u_valid, u_extra=u_extra,
            z_bin=z_bin, u_proto=u_proto,
            # Folded out of rng, so the draws above are untouched when the head is off.
            u_duel=self._duel_panel(jax.random.fold_in(rng, 119)),
            u_adv=u_adv)

    def _duel_panel(self, key):
        """`psi_dueling`: the (K, d_a) clipped prior panel the advantage baseline averages over.

        One panel per call, shared by every state in that call (the baseline is a
        Monte-Carlo estimate of E_{u ~ p0} Adv(s, u, z); the same K draws at every s is a
        valid estimate at a K-th of the cost). None when the head is off, and every psi
        call site passes the result through, so the free head never sees the kwarg.
        """
        c = self.config
        if not c["psi_dueling"]:
            return None
        K = int(c["psi_dueling_samples"])
        return jnp.clip(jax.random.normal(key, (K, c["action_dim"])), -c["u_clip"], c["u_clip"])

    @staticmethod
    def _duel_kw(u_prior):
        """kwargs for a psi call: `u_prior` only when there is a panel to pass."""
        return {} if u_prior is None else {"u_prior": u_prior}

    def _sample_preimage_mixture(self, batch, n, key, scale=1.0):
        """n draws per row from the stored EM preimage mixture. Returns (n, B, d_a).

        The jax twin of `utils.flow_inversion.sample_preimage_noise`, which runs on the
        dataset path in numpy; this one runs inside the jitted update. Same construction --
        a categorical draw over the components, then mu + L z -- but the factor comes from
        an eigendecomposition rather than a Cholesky with a fallback, because a traced
        `try` cannot branch: `L = V sqrt(max(lambda, 0))` is defined for every symmetric
        input, including the float-error-indefinite covariances the EM fit stores.

        `scale` multiplies the Cholesky factor, i.e. scales each component's covariance by
        `scale ** 2` -- Arm C. It exists because the posterior as stored decodes too far
        from the recorded action to be usable raw, and shrinking it toward its own mean
        recovers fidelity while keeping the covariance's SHAPE. The shape is the point: the
        EM covariance is stretched along the directions in which the decode barely moves, so
        at a matched decode error a shrunk anisotropic draw sits further from u_data than
        the round `jitter` ball -- measured at 1.6x on cube, 1.9x on antmaze, 3.2x on
        pointmaze (`tools/diag_mixture_decode.py --shrink`).

        The floor to be aware of: `scale -> 0` collapses onto the posterior MEAN, not onto
        u_data, and that mean sits 1.14 (cube) / 1.74 (antmaze) / 2.79 (pointmaze) away from
        the point inverse. Shrinking cannot buy fidelity below the mean's own displacement,
        which is why the sweep flattens out at small c.

        CAVEAT ON WHAT THESE ARE (measured, `tools/diag_mixture_decode.py`, and sharper than
        COMPENDIUM 4.9's reading): the posterior is not a blurred point. An UNSHRUNK cube
        sample decodes 0.167 from the recorded action against the point inverse's 0.089 --
        40% of the way to an uninverted prior draw. So the extra rows are only preimages of
        the same action once `scale` has been chosen against that probe's gate; the width is
        not a free parameter.
        """
        c = self.config
        mean = jnp.asarray(batch["noise_preimage_mean"])          # (B, K, d_a)
        cov = jnp.asarray(batch["noise_preimage_cov"])            # (B, K, d_a, d_a)
        wts = jnp.asarray(batch["noise_preimage_weights"])        # (B, K)
        B, K, d_a = mean.shape
        k_c, k_z = jax.random.split(key)
        # A collapsed EM fit can leave a row's weights summing to ~0; fall back to uniform
        # so the row stays a valid draw instead of silently taking the last component.
        tot = wts.sum(-1, keepdims=True)
        wts = jnp.where(tot > 1e-12, wts, jnp.full_like(wts, 1.0 / K))
        comp = jax.random.categorical(k_c, jnp.log(wts + 1e-12)[None], axis=-1,
                                      shape=(n, B))               # (n, B)
        mu = jnp.take_along_axis(mean[None], comp[..., None, None], axis=2)[:, :, 0]
        cv = jnp.take_along_axis(cov[None], comp[..., None, None, None], axis=2)[:, :, 0]
        cv = 0.5 * (cv + jnp.swapaxes(cv, -1, -2))
        lam, V = jnp.linalg.eigh(cv)
        L = V * jnp.sqrt(jnp.clip(lam, 0.0, None))[..., None, :]
        z = jax.random.normal(k_z, (n, B, d_a))
        u = mu + scale * jnp.einsum("nbij,nbj->nbi", L, z)
        return jnp.clip(u, -c["u_clip"], c["u_clip"])

    def _index(self, sampled):
        """Whatever occupies psi's index slot.

        'latent' (default) puts a fresh prior draw u' there -- psi(s, u, u'), with w
        reaching psi only through the readout Q = psi^T w. 'task_vector' puts w there.
        """
        return sampled.u_index if self.config["policy_index"] == "latent" else sampled.task_w

    # ------------------------------------------------------------------ psi magnitude bound
    def bound_psi(self, psi_out):
        """`psi_bound='tanh'`: S * tanh(psi / S). Identity in every other mode.

        A bounded reparameterisation, not a penalty: elementwise and monotone, so psi keeps
        its sign structure and its ordering in the policy coordinate. S = 1/(1-gamma) is the
        per-component ceiling on a successor feature under E[phi phi^T] = I.
        """
        if self.config["psi_bound"] != "tanh":
            return psi_out
        s = self.config["psi_bound_scale"]
        return s * jnp.tanh(psi_out / s)

    def _measure_input(self, observations, u):
        """The measure head's current-action coordinate for a latent query ``u``."""
        if self.config["measure_action_input"] == "action":
            return self.decode(observations, u)
        return u

    def psi_b(self, observations, index, u, u_prior=None, **kwargs):
        """Bounded psi at a latent query, decoding its current-action input when requested.

        Only for calls that return psi itself; `method='sa_terms'` / `'encode_index'`
        return the factors A, beta, w(u'), which are not the bounded object. `u_prior` is
        the dueling head's prior panel (None unless `psi_dueling`).
        """
        return self.bound_psi(self.psi(
            observations, index, self._measure_input(observations, u),
            **self._duel_kw(u_prior), **kwargs))

    # ------------------------------------------------------------------ measure loss
    def measure_loss(self, batch, sampled, phi_params, psi_params):
        """Contrastive fit of m(s,u,u',x) = psi(s,u,u')^T phi(x), plus phi's orthonormality.

        The TD target is psibar(s', u_next, index)^T phibar(s'_j), ensemble-reduced to
        mean - kappa * spread. u_next is where policy improvement enters the measure.
        """
        c = self.config
        obs, next_obs = batch["observations"], batch["next_observations"]
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        u, u_next = sampled.u_data, sampled.u_next

        index = self._index(sampled)
        phi_next = self.phi(next_obs, params=phi_params)
        proto_on = c["proto"]["enabled"]
        if proto_on:
            # Reference `_update_sf`: phi is FROZEN in the SF stage. The proto stage owns
            # it; here it is the basis the SF head is fitted on, nothing more.
            phi_next = jax.lax.stop_gradient(phi_next)
        # In action mode the online transition is fitted at the recorded action exactly;
        # every latent query (including the bootstrap below) goes through ``psi_b`` and
        # therefore through the frozen decoder.
        online_input = batch["actions"] if c["measure_action_input"] == "action" else u
        psi_raw = self.psi(obs, index, online_input, params=psi_params,
                           **self._duel_kw(sampled.u_duel))
        psi_online = self.bound_psi(psi_raw)
        M = psi_online @ phi_next.T
        # A frozen basis has no lagging target coordinates: the online parameters are the
        # fixed target. This also makes a restored stale target inert on the first step.
        target_phi_params = self.target_phi if c["train_phi"] else phi_params
        if proto_on:
            # ... and the reference's target measure is read at the ONLINE phi -- the one
            # the proto stage stepped this update (`apply_update` passes it) -- not at
            # target_phi. Only psi has a target net in this stage.
            target_phi_params = phi_params
        target_phi_next = self.phi(next_obs, params=target_phi_params)
        psi_boot = self.psi_b(next_obs, index, u_next, params=self.target_psi,
                              u_prior=sampled.u_duel)                      # (P, B, z)
        M_boot = psi_boot @ target_phi_next.T
        M_mean, M_unc = targets_uncertainty(M_boot, P)
        target_M = M_mean - c["pessimism_penalty"] * M_unc
        # `psi_bound='clip_target'`: the tanh head's admissible set imposed on the BOOTSTRAP.
        # S * z_dim is the Cauchy-Schwarz image of the psi ball, so both modes cap the same
        # quantity and psi_bound_scale keeps one meaning.
        bound_frac = jnp.asarray(0.0)
        if c["psi_bound"] == "clip_target":
            m_cap = c["psi_bound_scale"] * c["z_dim"]
            bound_frac = jnp.mean((jnp.abs(target_M) > m_cap).astype(jnp.float32))
            target_M = jnp.clip(target_M, -m_cap, m_cap)
        elif c["psi_bound"] == "tanh":
            bound_frac = jnp.mean(
                (jnp.abs(psi_raw) > c["psi_bound_scale"]).astype(jnp.float32))
        # `mask_invalid_preimages`: drop rows whose Stage-B inversion diverged, where
        # u_data does not decode to the recorded action. None => every row weight 1, the
        # published loss.
        row_w = sampled.u_valid if c["mask_invalid_preimages"] else None
        sm, sm_diag, sm_offdiag = contrastive_loss(
            M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum,
            row_weight=row_w)
        # `measure_u_samples` > 1: the SAME loss at further latents for the same
        # transition, averaged in. The TD target does not depend on the online u -- it is
        # psibar(s', u_next, u') -- so it is computed once and shared, and every extra
        # sample costs one online psi forward/backward and nothing else.
        u_extra_spread = jnp.asarray(0.0)
        u_extra_clip = jnp.asarray(0.0)
        if sampled.u_extra is not None:
            terms = [(sm, sm_diag, sm_offdiag)]
            for u_k in sampled.u_extra:
                M_k = self.psi_b(obs, index, u_k, params=psi_params,
                                 u_prior=sampled.u_duel) @ phi_next.T
                terms.append(contrastive_loss(
                    M_k, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum,
                    row_weight=row_w))
            sm, sm_diag, sm_offdiag = (
                sum(t[i] for t in terms) / len(terms) for i in range(3))
            # How far the extra latents sit from the point preimage, in the u box. If this
            # is ~0 the augmentation is doing nothing; if it is ~u_clip the extra rows are
            # not preimages of this transition's action in any useful sense.
            u_extra_spread = jnp.mean(jnp.linalg.norm(sampled.u_extra - u[None], axis=-1))
            # Fraction of the extra latents' COMPONENTS sitting on the box wall. Part of the
            # result, not just telemetry: the stored posterior mean lies outside
            # [-u_clip, u_clip] on a real fraction of rows -- 2.79 from the point inverse on
            # pointmaze -- so where a shrunk mixture draw actually lands is decided by the
            # BOX there, not by the posterior. A large number means the arm is training on
            # the wall.
            u_extra_clip = jnp.mean(
                (jnp.abs(sampled.u_extra) >= c["u_clip"] - 1e-6).astype(jnp.float32))
        ortho, ortho_diag, ortho_offdiag = ortho_loss(phi_next, off, off_sum)
        # The VALIDITY CONDITION for reward inference, logged beside the loss that enforces
        # it. Cor. `reward-inference`: w = E_D[r phi] is the least-squares projection exactly
        # when E[phi phi^T] = I. Measured 2026-09-09 on pointmaze, `orth_loss` tracks this
        # Gram's deviation at rank correlation 1.000 while the deviation itself swings four
        # orders of magnitude across checkpoints -- so the loss was never blind, it was
        # losing, and nobody was reading it as the thing that decides whether w means
        # anything. One matrix product plus a 128x128 eigendecomposition per step, both
        # stop-gradded: this is telemetry, not an objective.
        pn = jax.lax.stop_gradient(phi_next)
        gram = (pn.T @ pn) / pn.shape[0]
        gram_ev = jnp.linalg.eigvalsh(gram)
        gram_dev = (jnp.linalg.norm(gram - jnp.eye(gram.shape[0]))
                    / jnp.sqrt(gram.shape[0]))
        # `ortho_mode='relative'`: weight the geometry regulariser BY the term it holds
        # down, ortho_coef + ortho_rel_coef * stopgrad(|psm_loss|), so the two gradient
        # directions scale together and ortho cannot be outgrown. The weight is
        # stop-gradded, so only phi's gradient direction changes.
        ortho_weight = c["ortho_coef"]
        if c["ortho_mode"] == "relative":
            ortho_weight = ortho_weight + c["ortho_rel_coef"] * jnp.abs(
                jax.lax.stop_gradient(sm))
        loss = sm + ortho_weight * ortho
        if proto_on:
            # The reference multiplies the SF stage's copy of the ortho term by a literal
            # 0: phi is not this stage's to move. Kept above as telemetry only.
            loss = sm
        # `psm_scalar_coef` (2026-09-15, Fix 2): the measure's Bellman equation PROJECTED
        # along the sampled task vector z = task_w, as a scalar TD loss on the readout,
        #
        #   ( psi(s, u, z)^T z - gamma * [mean_P - kappa unc](psibar(s', u+, z)^T z)
        #                      - stopgrad(phi(s'))^T z )^2  /  Var_batch[stopgrad(phi(s')^T z)]
        #
        # with u+ the bootstrap latent the measure loss already uses (the actor's at s'
        # under policy_index=task_vector). This is the ICLR draft's scalar Bellman equation
        # of the readout Q_z = psi^T z (PAPER/ICLR), and Meta Motivo's `q_loss` on F^T z
        # (arXiv 2410.20096, `metamotivo/fb/agent.py`, coefficient 0 there by default); the
        # ensemble reduction is the measure target's own mean - kappa * spread, which at
        # P = 2, kappa = 0.5 is the min Meta Motivo takes. Gradient reaches psi only: phi is
        # stop-gradded in the reward term and the target is stop-gradded whole. The batch
        # variance of the reward term (stop-gradded) normalises it, so coef = 1 weights it
        # like a unit-variance TD loss whatever scale phi^T z happens to sit at.
        scalar = jnp.asarray(0.0)
        scalar_target_std = jnp.asarray(0.0)
        if float(c["psm_scalar_coef"]) > 0.0:
            z = sampled.task_w
            r_z = jax.lax.stop_gradient((phi_next * z).sum(-1))                # (B,)
            q_online = (psi_online * z[None]).sum(-1)                          # (P, B)
            q_boot = (psi_boot * z[None]).sum(-1)                              # (P, B)
            q_mean, q_unc = targets_uncertainty(q_boot, P)
            q_target = jax.lax.stop_gradient(
                r_z + c["discount"] * (q_mean - c["pessimism_penalty"] * q_unc))  # (B,)
            r_var = jax.lax.stop_gradient(jnp.var(r_z)) + 1e-8
            scalar = jnp.mean((q_online - q_target[None]) ** 2) / r_var
            scalar_target_std = q_target.std()
            loss = loss + c["psm_scalar_coef"] * scalar
        info = {"psm_loss": sm, "psm_diag": sm_diag, "psm_offdiag": sm_offdiag,
                      "psm_scalar_loss": scalar, "psm_scalar_target_std": scalar_target_std,
                      "orth_loss": ortho, "orth_diag": ortho_diag, "orth_offdiag": ortho_offdiag,
                      # Stabiliser telemetry, logged in every arm so the fixed and relative
                      # runs share a CSV schema.
                      "preimage_valid_frac": jnp.mean(sampled.u_valid),
                      "ortho_weight": jnp.asarray(ortho_weight, jnp.float32),
                      "ortho_term_abs": jnp.abs(ortho_weight * ortho),
                      "psi_absmean": jnp.mean(jnp.abs(psi_raw)),
                      "psi_absmax": jnp.max(jnp.abs(psi_raw)),
                      "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                      "psi_bound_frac": bound_frac,
                      "u_extra_dist": u_extra_spread,
                      "u_extra_clipfrac": u_extra_clip,
                      "phi_gram_dev": gram_dev,
                      "phi_gram_eig_min": gram_ev.min(),
                      "phi_gram_eig_max": gram_ev.max(),
                      "phi_gram_cond": gram_ev.max() / jnp.maximum(gram_ev.min(), 1e-12)}
        if sampled.u_adv is not None:
            info["bootstrap/adv"] = jnp.mean(sampled.u_adv)
        return loss, info

    # ------------------------------------------------------------------ reference PSM proto stage
    def proto_measure_loss(self, batch, sampled, phi_params, proto_params):
        """`proto.enabled`: PSM's proto branch (Eq. 9) on latent inputs -- the stage that trains phi.

        m^z(s, u, x) = proto_psi(s, z_bin, u)^T phi(x), fitted by the same contrastive loss
        against psibar(s', z_bin, u_proto)^T phibar(s'_j), where u_proto is the FIXED
        pseudo-random latent policy pi_z's draw at (row, z_bin) (`utils.psm_proto`), so the
        bootstrap action G(s', u_proto) is a decode of a prior-typical latent. Reference
        `proto_loss` verbatim otherwise: both target nets, ensemble mean minus kappa times
        spread, plus the orthonormality term on phi(s'). Gradients reach phi and proto_psi;
        the SF head never sees this loss.
        """
        c = self.config
        obs, next_obs = batch["observations"], batch["next_observations"]
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        z_bin, u, u_proto = sampled.z_bin, sampled.u_data, sampled.u_proto

        phi_next = self.phi(next_obs, params=phi_params)
        psi_raw = self.proto_psi(obs, z_bin, u, params=proto_params)
        M = psi_raw @ phi_next.T
        target_phi_next = self.phi(next_obs, params=self.target_phi)
        M_boot = self.proto_psi(next_obs, z_bin, u_proto, params=self.target_proto_psi) @ target_phi_next.T
        M_mean, M_unc = targets_uncertainty(M_boot, P)
        target_M = M_mean - c["pessimism_penalty"] * M_unc
        row_w = sampled.u_valid if c["mask_invalid_preimages"] else None
        sm, sm_diag, sm_offdiag = contrastive_loss(
            M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum, row_weight=row_w)
        ortho, ortho_diag, ortho_offdiag = ortho_loss(phi_next, off, off_sum)
        # `proto.ortho_coef` null = the agent's own ortho_coef. `ortho_mode` applies to
        # whichever stage carries the term, which under proto is this one.
        ortho_weight = c["proto"]["ortho_coef"]
        ortho_weight = c["ortho_coef"] if ortho_weight is None else float(ortho_weight)
        if c["ortho_mode"] == "relative":
            ortho_weight = ortho_weight + c["ortho_rel_coef"] * jnp.abs(
                jax.lax.stop_gradient(sm))
        loss = sm + ortho_weight * ortho
        return loss, {"proto_loss": sm, "proto_diag": sm_diag, "proto_offdiag": sm_offdiag,
                      "proto_orth_loss": ortho, "proto_orth_diag": ortho_diag,
                      "proto_orth_offdiag": ortho_offdiag,
                      "proto_ortho_weight": jnp.asarray(ortho_weight, jnp.float32),
                      "proto_psi_absmean": jnp.mean(jnp.abs(psi_raw)),
                      "proto_td_target_absmean": jnp.mean(jnp.abs(target_M))}

    # ------------------------------------------------------------------ latent-actor losses
    def _psi_q_over_indices(self, obs, u, w, u_index, params=None, u_prior=None):
        """Q(s, u, u'_k) = psi(s, u, u'_k)^T w for K policy indices at once. Returns (P, K, B).

        Under psi_form=affine, A and beta do not depend on the index, so an index panel
        costs one einsum in w-space instead of K towers. psi_form=free and psi_bound='tanh'
        have no such factorisation and fall back to K full psi calls.
        """
        if self.config["psi_form"] == "affine" and self.config["psi_bound"] != "tanh":
            measure_input = self._measure_input(obs, u)
            A, beta = self.psi(
                obs, measure_input, params=params, method="sa_terms",
                **self._duel_kw(u_prior))                                  # (P,B,z,d_w),(P,B,z)
            w_index = self.psi(u_index, params=params, method="encode_index")   # (K, B, d_w)
            Aw = jnp.einsum("pbzw,bz->pbw", A, w)
            beta_w = jnp.einsum("pbz,bz->pb", beta, w)
            return jnp.einsum("pbw,kbw->pkb", Aw, w_index) + beta_w[:, None, :]   # (P, K, B)
        return jax.vmap(lambda idx: (self.psi_b(obs, idx, u, params=params,
                                                u_prior=u_prior) * w).sum(-1),
                        out_axes=1)(u_index)                               # (P, K, B)

    def _psi_q_fixed_coeff(self, obs, u, w, coeff, params=None, u_prior=None):
        """Q_c(s, u) = psi_c(s, u)^T w with psi_c = A(s,u)^T c + beta(s,u): (P, B).

        PSM's test-time policy (arXiv 2411.19418 Eq. 10) is one coefficient vector c on
        the affine basis, in place of the encoder output w(u') of one family member.
        With c = w(u'_0) this is `_psi_q_over_indices(..., u_index=u'_0[None])[:, 0]`
        exactly; c is otherwise free (not on the encoder's unit sphere).
        """
        assert self.config["psi_form"] == "affine", "a fixed coefficient needs the affine head"
        measure_input = self._measure_input(obs, u)
        A, beta = self.psi(obs, measure_input, params=params, method="sa_terms",
                           **self._duel_kw(u_prior))                       # (P,B,z,d_w),(P,B,z)
        coeff = jnp.broadcast_to(coeff, (*A.shape[:-2], A.shape[-1]))
        psi_c = self.bound_psi(jnp.einsum("pbzw,pbw->pbz", A, coeff) + beta)   # (P, B, z)
        return (psi_c * w).sum(-1)                                          # (P, B)

    @jax.jit
    def fixed_coeff_select(self, observations, seed=None):
        """`acting=fixed_coeff`: argmax over K clipped prior draws u of the pessimistic
        readout [mean_P - kappa * unc] Q_c(s, u) at the loaded coefficient `fixed_coeff`.

        The candidate draws come off the same split of `seed` as `gpi_select`'s u_cand, so
        at c = w(u'_0) the score ranks the very candidates the pinned-index ablation ranks.
        """
        c = self.config
        assert observations.ndim == 1, "fixed_coeff_select acts on a single observation"
        K, d_a = c["gpi_num_u"], c["action_dim"]
        seed = self.rng if seed is None else seed
        u_duel = self._duel_panel(jax.random.fold_in(seed, 5))
        r_u, _ = jax.random.split(seed)
        u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        q_ens = self._psi_q_fixed_coeff(obs, u_cand, w, self.fixed_coeff, u_prior=u_duel)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        Q = q_mean - c["actor_pessimism_penalty"] * q_unc
        return u_cand[jnp.argmax(Q)]

    def _actor_q(self, obs, u_a, sampled):
        """The Q the latent actor climbs, and the raw ensemble readout it is normalised by.

        `actor.index_panel = 0` reads psi at the ONE prior index this row drew; K > 0
        replaces it with the max over K clipped prior draws -- the object `gpi_select`
        maximises. Redundant under index_agg='expectile', which `create` rejects.
        """
        c = self.config
        w = sampled.task_w
        if c["dsrl_na"]["enabled"]:
            # DSRL-NA: the actor climbs the LATENT critic and nothing else. Q_W is already
            # a scalar on the reward's own scale, and no gradient reaches Q_A or the flow.
            Q = q_ens = self.qw(obs, self._na_in(u_a, self._actor_w(w)))    # (B,)
            return Q, q_ens
        if c["index_agg"] == "expectile":
            Q = q_ens = self.q_dist(obs, jnp.concatenate([w, u_a], -1))    # (B,)
            return Q, q_ens
        K = int(c["actor"]["index_panel"])
        if K <= 0:
            q_ens = (self.psi_b(obs, self._index(sampled), u_a,
                                u_prior=sampled.u_duel) * w).sum(-1)       # (P, B)
            q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
            return q_mean - c["actor_pessimism_penalty"] * q_unc, q_ens
        B, d_a = obs.shape[0], c["action_dim"]
        ic = self._index_clip()
        u_index = jnp.clip(jax.random.normal(jax.random.fold_in(self.rng, 112), (K, B, d_a)),
                           -ic, ic)
        q_panel = self._psi_q_over_indices(obs, u_a, w, u_index,
                                           u_prior=sampled.u_duel)         # (P, K, B)
        q_mean, q_unc = targets_uncertainty(q_panel, c["num_parallel"])    # (K, B)
        Q = (q_mean - c["actor_pessimism_penalty"] * q_unc).max(0)         # (B,)
        return Q, q_panel.reshape(q_panel.shape[0], -1)

    def _actor_w(self, w):
        """The task slot the latent actor reads. Zeros under DSRL-NA.

        DSRL-NA is SINGLE-TASK: Q_A is fitted on one real reward and Q_W(s, u) has no task
        input at all, so a task-conditioned policy would be conditioning on a random vector
        its own critic ignores -- and would then meet a specific `task_z` at eval that it
        never saw in training. Zeroing the slot makes the head pi(u | s) exactly, without
        changing the module's shape, so the pytree stays static across the switch.
        """
        na = self.config["dsrl_na"]
        if na["enabled"] and not na["task_conditioned"]:
            return jnp.zeros_like(w)
        return w

    def _na_in(self, x, w):
        """The second argument of the DSRL-NA critics: (a, w) or (u, w) when task-conditioned.

        `Value` concatenates its two inputs, so this is exactly "append the task vector".
        """
        return jnp.concatenate([x, w], -1) if self.config["dsrl_na"]["task_conditioned"] else x

    def _sac_latent(self, obs, w, noise, params=None):
        """Reparameterised draw from the tanh-Gaussian latent policy, and its log-density."""
        mu, log_std = self.sac_actor(obs, w, params=params)
        return tanh_gaussian_sample(mu, log_std, noise, self.config["u_clip"])

    def _deploy_latent(self, obs, w, noise):
        """The TRAINED latent actor's draw, scaled to the u box, dispatched on `actor_mode`.

        Every read of the actor outside its own loss must go through here: `apply_update`
        steps `sac_actor` under the DSRL modes and `actor` only under 'ddpg', so a
        hardcoded `self.actor` silently reads an untrained head on the other arms.
        """
        if self.config["actor_mode"] == "ddpg":
            return self.config["u_clip"] * self.actor(obs, w, noise)
        return self._sac_latent(obs, w, noise)[0]

    def _gpi_argmax_latent(self, obs, w, key, n, params=None):
        """EMaQ-style argmax under the task-vector index: for each row,

            u*(s) = argmax_{u_m ~ p0, m<=n} [mean_P - pessimism_penalty * unc] psi(s, w, u_m)^T w

        with psi read at `params` (the TARGET psi when this builds the TD bootstrap). Every
        candidate is a clipped prior draw, so the decode G(s, u*) is in-support. Returns
        (u_star (B, d_a), adv (B,)) with adv = Q(u*) - mean_m Q(u_m), both stop-gradded.
        Cost: n psi forwards per row.
        """
        c = self.config
        B, d_a = obs.shape[0], c["action_dim"]
        u_cand = jnp.clip(jax.random.normal(key, (n, B, d_a)), -c["u_clip"], c["u_clip"])

        def score(u_m):
            q = (self.psi_b(obs, w, u_m, params=params) * w).sum(-1)          # (P, B)
            q_mean, q_unc = targets_uncertainty(q, c["num_parallel"])
            return q_mean - c["pessimism_penalty"] * q_unc                    # (B,)

        Q = jax.vmap(score)(u_cand)                                           # (n, B)
        best = jnp.argmax(Q, axis=0)
        u_star = jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0]
        adv = Q.max(0) - Q.mean(0)
        return jax.lax.stop_gradient(u_star), jax.lax.stop_gradient(adv)

    def _gpi_argmax_target(self, obs, w, key):
        """DSRL-NA's regression target: the acting rule's own choice, per state.

            u*(s) = argmax_{u_m ~ p0} max_{u'_k ~ p0} [mean_P - kappa*unc] psi(s, u_m, u'_k)^T w

        Every candidate is a prior draw, so the actor is regressed only toward latents p0
        produces. Returns (u_star (B, d_a), advantage (B,)) with
        advantage = Q(u*) - mean_m Q(u_m).
        """
        c = self.config
        B, d_a = obs.shape[0], c["action_dim"]
        M, K = int(c["actor"]["na_candidates"]), max(int(c["actor"]["index_panel"]), 1)
        k_u, k_i = jax.random.split(key)
        ic = self._index_clip()
        u_cand = jnp.clip(jax.random.normal(k_u, (M, B, d_a)), -c["u_clip"], c["u_clip"])
        u_index = jnp.clip(jax.random.normal(k_i, (K, B, d_a)), -ic, ic)
        u_duel = self._duel_panel(jax.random.fold_in(key, 2))

        def score(u_m):
            q_panel = self._psi_q_over_indices(obs, u_m, w, u_index,
                                               u_prior=u_duel)              # (P, K, B)
            q_mean, q_unc = targets_uncertainty(q_panel, c["num_parallel"])
            return (q_mean - c["actor_pessimism_penalty"] * q_unc).max(0)   # (B,)

        Q = jax.vmap(score)(u_cand)                                         # (M, B)
        best = jnp.argmax(Q, axis=0)                                        # (B,)
        u_star = jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0]
        adv = jnp.max(Q, axis=0) - Q.mean(0)
        return jax.lax.stop_gradient(u_star), jax.lax.stop_gradient(adv)

    def flow_actor_loss(self, batch, sampled, actor_params, vf_params):
        """`actor_mode='ddpg'`: the flow-BC recipe with the action space replaced by latents.

        Three terms: the CFM field v_xi toward the dataset latents, a one-step actor
        distilled from its own rollout, and the normalised value term -Q/|Q|. psi is read at
        fixed params, so the DPG gradient reaches the actor through psi's INPUTS only.
        """
        c = self.config
        obs = batch["observations"]
        w = sampled.task_w
        u0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
        steps = c["actor"]["flow_steps"]
        u_clip = c["u_clip"]

        def rollout(vf_p, o, n):
            u = n
            for i in range(steps):
                ti = jnp.full((o.shape[0], 1), i / steps)
                u = u + self.actor_vf(o, u, ti, params=vf_p) / steps
            return jnp.clip(u, -u_clip, u_clip)

        u_t = (1 - t) * u0 + t * sampled.u_data
        bc_flow_loss = jnp.mean(
            (self.actor_vf(obs, u_t, t, params=vf_params) - (sampled.u_data - u0)) ** 2)

        u_a = u_clip * self.actor(obs, w, noise, params=actor_params)
        Q, q_ens = self._actor_q(obs, u_a, sampled)
        q_loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(q_ens).mean() + 1e-8)
        distill = jnp.mean((u_a - jax.lax.stop_gradient(rollout(vf_params, obs, noise))) ** 2)
        loss = q_loss + c["actor"]["bc_coeff"] * distill + bc_flow_loss
        return loss, {"actor_loss": loss, "actor_q": Q.mean(),
                      "actor_bc_flow_loss": bc_flow_loss, "actor_bc_error": distill,
                      # mean ||u_actor||: the prior's typical norm is ~sqrt(d_a) (2.1 at
                      # d_a = 5); the Section 10 affine checkpoint sits at 0.67 (09-15).
                      "actor_u_norm": jnp.linalg.norm(u_a, axis=-1).mean()}

    def dsrl_actor_loss(self, batch, sampled, sac_params, vf_params):
        """`actor_mode` in {'dsrl_sac', 'dsrl_na'}: the DSRL latent actor on this substrate.

        Both modes share the tanh-Gaussian head pi(u | s, w) over the latent box
        [-u_clip, u_clip] and read Q through `_actor_q`.

        dsrl_sac  SAC's (alpha * log pi - Q); at P=2, mean - 0.5 * spread IS the ensemble min.
        dsrl_na   a regression onto the argmax over prior draws of the GPI score, optionally
                  weighted by that state's Q advantage.

        Q is normalised by stopgrad(|Q|) and the entropy term added AFTER, so psi^T w having
        no natural return scale does not make `target_entropy` a per-environment constant.
        """
        c = self.config
        obs, w = batch["observations"], self._actor_w(sampled.task_w)
        u0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
        u_clip, steps = c["u_clip"], c["actor"]["flow_steps"]
        mode = c["actor_mode"]

        def rollout(vf_p, o, n):
            u = n
            for i in range(steps):
                ti = jnp.full((o.shape[0], 1), i / steps)
                u = u + self.actor_vf(o, u, ti, params=vf_p) / steps
            return jnp.clip(u, -u_clip, u_clip)

        u_t = (1 - t) * u0 + t * sampled.u_data
        bc_flow_loss = jnp.mean(
            (self.actor_vf(obs, u_t, t, params=vf_params) - (sampled.u_data - u0)) ** 2)

        u_a, logp = self._sac_latent(obs, w, noise, params=sac_params)
        Q, q_ens = self._actor_q(obs, u_a, sampled)
        # The normalisation exists because psi^T w has no natural return scale, which is
        # what makes `target_entropy` otherwise a per-environment constant. Under DSRL-NA
        # Q_W IS on the reward's scale, and DSRL/SB3 climbs it raw -- normalising there
        # would silently rescale alpha against the entropy target.
        scale = (jnp.asarray(1.0) if c["dsrl_na"]["enabled"]
                 else jax.lax.stop_gradient(jnp.abs(q_ens).mean() + 1e-8))
        alpha = jax.lax.stop_gradient(jnp.exp(self.log_alpha()))
        q_term = -Q.mean() / scale
        # The entropy term sits OUTSIDE q_coeff: at q_coeff=0 it is the only thing stopping
        # the tanh-Gaussian from shrinking its scale onto the regression target's mean.
        ent_term = alpha * logp.mean()

        info = {"actor_q": Q.mean(), "actor_logp": logp.mean(), "actor_alpha": alpha,
                "actor_entropy": -logp.mean(), "actor_u_norm": jnp.linalg.norm(u_a, axis=-1).mean()}
        loss = c["actor"]["q_coeff"] * q_term + ent_term
        if mode == "gpi_distill":
            # na_candidates psi passes per state, so the target is built on the first
            # na_states rows only.
            n = min(int(c["actor"]["na_states"]), obs.shape[0])
            u_star, adv = self._gpi_argmax_target(obs[:n], w[:n],
                                                  jax.random.fold_in(self.rng, 113))
            sq = ((u_a[:n] - u_star) ** 2).mean(-1)                           # (n,)
            if c["actor"]["na_advantage_weight"]:
                # Mean-1 normalised so na_coeff keeps its meaning as |Q| drifts.
                wgt = jax.nn.relu(adv)
                wgt = wgt / (wgt.mean() + 1e-8)
                na_loss = (wgt * sq).mean()
            else:
                na_loss = sq.mean()
            loss = loss + c["actor"]["na_coeff"] * na_loss
            info["actor_na_error"] = na_loss
            # Always UNWEIGHTED, unlike `actor_na_error`, so it stays comparable to the
            # independence floor (‖u_a‖² + ‖u*‖²)/d_a the 2026-09-07 plateau was measured
            # against. Read progress off this one.
            info["actor_na_mse"] = sq.mean()
            info["actor_na_advantage"] = adv.mean()
            info["actor_na_target_norm"] = jnp.linalg.norm(u_star, axis=-1).mean()
        distill = jnp.mean((u_a - jax.lax.stop_gradient(rollout(vf_params, obs, noise))) ** 2)
        loss = loss + c["actor"]["bc_coeff"] * distill + bc_flow_loss
        info.update({"actor_loss": loss, "actor_bc_flow_loss": bc_flow_loss,
                     "actor_bc_error": distill})
        return loss, info

    def alpha_loss(self, batch, sampled, alpha_params):
        """SAC's entropy-coefficient loss, L = -log_alpha * stopgrad(log pi + target_entropy).

        DSRL sets target_ent = 0 rather than the usual -dim(A), stated in the squashed,
        UNSCALED space `tanh_gaussian_sample` computes log pi in.
        """
        c = self.config
        obs, w = batch["observations"], self._actor_w(sampled.task_w)
        _, logp = self._sac_latent(obs, w, sampled.flow_noise)
        target = c["actor"]["target_entropy"]
        la = self.log_alpha(params=alpha_params)
        loss = -(la * jax.lax.stop_gradient(logp + target)).mean()
        return loss, {"alpha_loss": loss, "log_alpha": la}

    # ------------------------------------------------------------------ index-aggregation head
    def q_dist_loss(self, batch, sampled, q_params):
        """`index_agg='expectile'`: upper-expectile regression of q_dist onto psi^T w.

            L = E_{(s,u), u'~p0} [ |mu - 1{d < 0}| * d^2 ],
            d = stopgrad(Q(s, u', u)) - q_dist(s, w, u)

        The panel of `index_panel` draws u' is NOT aggregated before the regression: each
        target is its own residual, which makes the fit an expectile OF THE INDEX
        DISTRIBUTION rather than of its mean. psi is read at stored params under
        stop_gradient -- this head is a distillation target and must not push gradient back
        into the measure.
        """
        c = self.config
        obs, u = batch["observations"], sampled.u_data
        w = sampled.task_w
        B, d_a = u.shape[0], c["action_dim"]
        panel = c["index_panel"]
        key = jax.random.fold_in(self.rng, 110)
        ic = self._index_clip()
        u_index = jnp.clip(jax.random.normal(key, (panel, B, d_a)), -ic, ic)

        obs_rep = jnp.broadcast_to(obs[None], (panel, *obs.shape)).reshape(panel * B, -1)
        u_rep = jnp.broadcast_to(u[None], (panel, *u.shape)).reshape(panel * B, d_a)
        w_rep = jnp.broadcast_to(w[None], (panel, *w.shape)).reshape(panel * B, -1)
        q_ens = (self.psi_b(obs_rep, u_index.reshape(panel * B, d_a), u_rep,
                            u_prior=sampled.u_duel) * w_rep).sum(-1)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        target = jax.lax.stop_gradient(
            (q_mean - c["actor_pessimism_penalty"] * q_unc).reshape(panel, B))

        pred = self.q_dist(obs, jnp.concatenate([w, u], -1), params=q_params)   # (B,)
        diff = target - pred[None]                                        # (panel, B)
        mu = c["expectile_mu"]
        loss = (jnp.where(diff >= 0, mu, 1.0 - mu) * diff ** 2).mean()
        return loss, {"q_dist_loss": loss, "q_dist_pred": pred.mean(),
                      "q_dist_target": target.mean(),
                      "q_dist_target_spread": target.std(axis=0).mean()}

    # ------------------------------------------------------------------ DSRL-NA dual critic
    def dsrl_qa_loss(self, batch, sampled, qa_params):
        """`dsrl_na`: the ACTION-space critic. Scalar TD on the task's REAL reward.

            target = r + gamma_na * mask * min_e Q_A_bar(s', G(s', u')),   u' ~ pi(. | s')

        This is the ONLY place in the agent that reads `batch['rewards']`, and it is why
        the arm is not zero-shot. It is DSRL's own critic: fitted over raw actions, where
        many latents collapse onto the same action, which is what gives the distillation
        below something to be a function of.

        No entropy bonus in the bootstrap. Our log pi is a density over the LATENT u, not
        over the action a, so an entropy term here would be measured in the wrong
        coordinates; the entropy target lives with the actor and Q_W, where the policy is.
        The bootstrap action is stop-gradded -- Q_A is a critic, not a path to the actor.

        `reward_source='phi_readout'` (Arm D1) replaces r with its reconstruction through
        the frozen basis, `r_hat = phi(s')^T w` with `w = project(E_batch[(r + shift) phi])`
        -- the deployed zero-shot reward channel, recomputed per batch because phi is still
        training. It is the FIRST of the four differences between this arm and the zero-shot
        one, isolated. Two things it also changes, stated because they are unavoidable:
        the reward becomes the SHIFTED one (the 2026-09-09 gate measured topline R^2 = -0.20
        on the raw -1/0 reward against 0.116 on the shifted one, so the shift is what makes
        the channel work at all), which adds a constant 1/(1-gamma) to every Q and changes
        no policy; and `w`'s sphere projection sets r_hat's scale, which auto-alpha absorbs.
        `na_rhat_corr` logs how much of the real reward survives the round trip.

        `ignore_masks` (2026-09-15) bootstraps with mask = 1 on every row, the continuing
        formulation: a synthetic reward phi(s')^T w is a per-w constant offset away from any
        reward the task actually pays, and an offset is harmless only without termination
        (with the dataset masks the 09-14 raw-r_hat arm valued staying alive above finishing).
        `reward_scale` multiplies the synthetic reward so its per-w std can be matched to the
        real -1/0 reward's, the scale the critic widths and alpha were tuned at.
        """
        c, na = self.config, self.config["dsrl_na"]
        obs, next_obs = batch["observations"], batch["next_observations"]
        w = self._actor_w(sampled.task_w)
        u_next = self._deploy_latent(next_obs, w, sampled.flow_noise)
        a_next = jax.lax.stop_gradient(self.decode(next_obs, u_next))
        q_next = self.qa(next_obs, self._na_in(a_next, w), params=self.target_qa).min(0)
        mask = jnp.ones_like(batch["masks"]) if na["ignore_masks"] else batch["masks"]

        r_true = batch["rewards"]
        info = {}
        if na["reward_source"] == "synthetic_w":
            # Arm D2. The reward IS the task vector read through the basis, so no real
            # reward enters training and the arm is zero-shot by construction. w is drawn
            # FB-style by `sample_step_inputs` (mix_ratio of phi(next_obs)[perm], the rest
            # Gaussian, both projected to the sphere and stop-gradded), which is the same
            # distribution the measure branch trains against.
            ph = jax.lax.stop_gradient(self.phi(next_obs))
            r_used = float(na["reward_scale"]) * (ph * w).sum(-1)
            dr = r_used - r_used.mean()
            dt = (r_true + na["reward_shift"]) - (r_true + na["reward_shift"]).mean()
            # How much the synthetic task happens to resemble the REAL one this batch. Not
            # used for anything -- it is the sanity trace that the arm is not accidentally
            # training on the deployed task.
            info["na_rw_corr_to_real"] = (dr * dt).mean() / (dr.std() * dt.std() + 1e-8)
            info["na_rw_std"] = r_used.std()
        elif na["reward_source"] == "phi_readout_fixed":
            # Arm D1b. w and the scale come from `refit_na_reward` on a relabel batch and
            # are constants here, so Q_A sees ONE reward function between refits instead of
            # a new one each step. No gradient reaches phi through the reward.
            ph = jax.lax.stop_gradient(self.phi(next_obs))
            r_used = (ph @ self.na_rw) * self.na_rw_scale
            r_shift = r_true + na["reward_shift"]
            dr, dt = r_used - r_used.mean(), r_shift - r_shift.mean()
            info["na_rhat_corr_inbatch"] = (dr * dt).mean() / (dr.std() * dt.std() + 1e-8)
            info["na_rhat_std"] = r_used.std()
            info["na_reward_std"] = r_shift.std()
            info["na_rw_norm"] = jnp.linalg.norm(self.na_rw)
            info["na_rw_scale"] = self.na_rw_scale
        elif na["reward_source"] == "phi_readout":
            r_shift = r_true + na["reward_shift"]
            ph = jax.lax.stop_gradient(self.phi(next_obs))                    # (B, z)
            w_hat = project_z((r_shift.reshape(1, -1) @ ph).reshape(-1) / ph.shape[0],
                              c["norm_z"])
            r_used = ph @ w_hat
            dr, dt = r_used - r_used.mean(), r_shift - r_shift.mean()
            info["na_rhat_corr"] = (dr * dt).mean() / (dr.std() * dt.std() + 1e-8)
            info["na_rhat_mean"] = r_used.mean()
            info["na_rhat_std"] = r_used.std()
            info["na_reward_std"] = r_shift.std()
        else:
            r_used = r_true

        target = jax.lax.stop_gradient(
            r_used + na["discount"] * mask * q_next)
        q = self.qa(obs, self._na_in(batch["actions"], w), params=qa_params)  # (E, B)
        loss = jnp.square(q - target[None]).mean()
        return loss, {"na_qa_loss": loss, "na_qa_mean": q.mean(),
                      "na_qa_target": target.mean(),
                      "na_qa_spread": (q.max(0) - q.min(0)).mean(), **info}

    def dsrl_qw_loss(self, batch, qw_params, key, w=None):
        """`dsrl_na`: the LATENT critic, regressed onto Q_A at the decode of prior latents.

            L = E_{s ~ D, u ~ p0} [ (Q_W(s, u) - stopgrad min_e Q_A(s, G(s, u)))^2 ]

        The point of the distillation: the actor then climbs a function of u that was never
        itself argmaxed over a learned function of u -- Q_W inherits Q_A's ordering by
        regression, on the prior's own support. No gradient reaches Q_A or the flow.
        """
        c = self.config
        na = c["dsrl_na"]
        obs = batch["observations"]
        B, d_a, n = obs.shape[0], c["action_dim"], int(na["n_latent"])
        u = jnp.clip(jax.random.normal(key, (n, B, d_a)), -c["u_clip"], c["u_clip"])
        obs_r = jnp.broadcast_to(obs[None], (n, *obs.shape)).reshape(n * B, -1)
        u_r = u.reshape(n * B, d_a)
        a = jax.lax.stop_gradient(self.decode(obs_r, u_r))
        if w is None:
            w = jnp.zeros((B, c["z_dim"]), obs.dtype)
        w_r = jnp.broadcast_to(w[None], (n, *w.shape)).reshape(n * B, -1)
        target = jax.lax.stop_gradient(self.qa(obs_r, self._na_in(a, w_r)).min(0))
        pred = self.qw(obs_r, self._na_in(u_r, w_r), params=qw_params)       # (n*B,)
        loss = jnp.square(pred - target).mean()
        return loss, {"na_qw_loss": loss, "na_qw_pred": pred.mean(),
                      "na_qw_target": target.mean()}

    def na_spread(self, batch, key, wv=None):
        """`dsrl_na`: how much Q_W actually varies over u at a FIXED state.

        This is the quantity the whole campaign is about. psi^T w reads 57 here against an
        ensemble disagreement of 124 (`docs/design/2026-09-08-critic-signal-and-dsrl-na.md`
        1), i.e. the actor's gradient is smaller than the critics' argument. If Q_W is flat
        too, DSRL-NA has the same problem and the frozen flow is the binding constraint.
        Reported relative to |Q| so it is comparable across steps, and beside qa's own
        ensemble disagreement, which is the noise it has to beat.
        """
        c = self.config
        n = min(64, batch["observations"].shape[0])
        obs = batch["observations"][:n]
        cand = 16
        u = jnp.clip(jax.random.normal(key, (cand, n, c["action_dim"])),
                     -c["u_clip"], c["u_clip"])
        w = (jnp.zeros((n, c["z_dim"]), obs.dtype) if wv is None else wv[:n])
        Q = jax.vmap(lambda u_k: self.qw(obs, self._na_in(u_k, w)))(u)       # (cand, n)
        obs_r = jnp.broadcast_to(obs[None], (cand, n, obs.shape[-1])).reshape(-1, obs.shape[-1])
        w_r = jnp.broadcast_to(w[None], (cand, *w.shape)).reshape(cand * n, -1)
        a = self.decode(obs_r, u.reshape(-1, c["action_dim"]))
        qa_ens = self.qa(obs_r, self._na_in(a, w_r))                         # (E, cand*n)
        scale = jnp.abs(Q).mean() + 1e-8
        return {"na_qw_spread_over_u": Q.std(0).mean(),
                "na_qw_spread_over_u_rel": Q.std(0).mean() / scale,
                "na_qw_range_over_u_rel": (Q.max(0) - Q.min(0)).mean() / scale,
                "na_qa_disagreement": (qa_ens.max(0) - qa_ens.min(0)).mean(),
                # > 1 means the signal over u is larger than the critics' own argument.
                "na_signal_over_disagreement": Q.std(0).mean()
                / ((qa_ens.max(0) - qa_ens.min(0)).mean() + 1e-8)}

    # ------------------------------------------------------------------ action branch
    def execute(self, observations, w, u, residual_params=None):
        """Executed action a = clip(G(s, u) + eps * delta(s, w, u)).

        NoiseConditionedActor tanh-bounds delta, so `residual_eps` is a hard per-dimension
        budget on the distance from the decode. Only used when action_critic.enabled.
        """
        a = self.decode(observations, u)
        eps = self.config["residual_eps"]
        if eps > 0.0:
            p = self.residual.params if residual_params is None else residual_params
            a = a + eps * self.residual(observations, w, u, params=p)
        return jnp.clip(a, -1.0, 1.0)

    def action_critic_loss(self, batch, sampled, psi_a_params):
        """Vector TD for psi_a(s, w, a): successor features of the SHARED phi under pi_w.

        target = phi(s') + gamma_ac * psi_a_bar(s', w, a'_exec). Reward-free, so the branch
        stays zero-shot: Q_a for any reward is psi_a^T w.
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
        # -pessimism * unc^T w, whose sign flips where w < 0. Blend the ensemble mean
        # toward the member this task values least instead.
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
        """`action_critic.fb_graft`: train (psi_a, B_a) as a self-contained FB pair.

        B_a(s') is the branch's own backward map, trained by the FB measure loss, and w_a is
        inferred from it, so delta and Q_a stop depending on the shared basis phi. No
        gradient crosses between this branch and the shared measure in either direction.
        """
        c = self.config
        ac = c["action_critic"]
        obs, next_obs = batch["observations"], batch["next_observations"]
        off, off_sum = off_diagonal_mask(obs.shape[0])
        w = sampled.task_w_a
        a_next = jax.lax.stop_gradient(self.execute(next_obs, w, sampled.u_next))

        target_B_a = self.phi_a(next_obs, params=self.target_phi_a)
        M_boot = self.psi_a(next_obs, w, a_next, params=self.target_psi_a) @ target_B_a.T
        M_mean, _ = targets_uncertainty(M_boot, c["num_parallel"])
        B_a = self.phi_a(next_obs, params=phi_a_params)
        M = self.psi_a(obs, w, batch["actions"], params=psi_a_params) @ B_a.T
        sm, sm_diag, sm_offdiag = contrastive_loss(M, jax.lax.stop_gradient(M_mean),
                                                   ac["discount"], off, off_sum)
        ortho, ortho_diag, _ = ortho_loss(B_a, off, off_sum)
        loss = sm + ac["ortho_coef"] * ortho
        q_data = (self.psi_a(obs, w, batch["actions"], params=psi_a_params)
                  * w[None]).sum(-1).mean()
        return loss, {"ac_loss": loss, "ac_fb_diag": sm_diag, "ac_fb_offdiag": sm_offdiag,
                      "ac_orth_loss": ortho, "ac_orth_diag": ortho_diag, "ac_q_data": q_data}

    def residual_loss(self, batch, sampled, residual_params):
        """-Q_a on the executed action; the value-directed step the decoder cannot take.

        The latent actor is read at stop-grad, so only the residual head chases Q_a, and
        only within its eps budget.
        """
        c = self.config
        obs, w = batch["observations"], sampled.task_w_a
        # The latent actor stays on the shared w; only the residual and Q_a move to the
        # action branch's w_a, which differs under the graft.
        u_a = jax.lax.stop_gradient(
            self._deploy_latent(obs, sampled.task_w, sampled.flow_noise))
        a_exec = self.execute(obs, w, u_a, residual_params=residual_params)
        q_ens = (self.psi_a(obs, w, a_exec) * w[None]).sum(-1)                # (P, B)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        Q = q_mean - c["action_critic"]["pessimism"] * q_unc
        loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(q_ens).mean() + 1e-8)
        spend = jnp.mean(jnp.abs(a_exec - self.decode(obs, u_a)))
        return loss, {"residual_loss": loss, "residual_q": Q.mean(),
                      "residual_spend": spend}

    # ------------------------------------------------------------------ in-loop diagnostics
    def q_dist_spread(self, batch, sampled, key):
        """Spread of the distilled expectile head over prior action latents at one state.

        Computed exactly like `action_critic_spread`, so the two numbers are comparable.
        A spread at the ~1% band of |Q| means the head cannot rank the latents the decoder
        can produce.
        """
        c = self.config
        n = min(64, batch["observations"].shape[0])
        obs = batch["observations"][:n]
        cand = c["action_critic"]["spread_candidates"]
        u = jnp.clip(jax.random.normal(key, (cand, n, c["action_dim"])),
                     -c["u_clip"], c["u_clip"])
        w = sampled.task_w[:n]
        Q = jax.vmap(lambda u_k: self.q_dist(obs, jnp.concatenate([w, u_k], -1)))(u)
        scale = jnp.abs(Q).mean() + 1e-8
        return {"q_dist_spread": Q.std(axis=0).mean(),
                "q_dist_spread_rel": Q.std(axis=0).mean() / scale}

    def index_spread(self, batch, sampled, key):
        """Can psi rank anything under policy_index='latent'? Four numbers, all relative to |Q|.

          psi_q_spread_rel        std over prior ACTION latents u at this batch's index u'
          psi_q_range_rel         the same, as a max-min range
          psi_q_index_spread_rel  std over prior POLICY indices u' at the dataset latent
          w_enc_spread            (affine only) mean pairwise ||w(u'_i) - w(u'_j)||; under
                                  norm_w random directions sit near sqrt(2), a collapsed
                                  encoder near 0

        n is deliberately small: this runs inside the jitted update every step.
        """
        c = self.config
        n = min(16, batch["observations"].shape[0])
        obs = batch["observations"][:n]
        cand = c["action_critic"]["spread_candidates"]
        k_u, k_i, k_w = jax.random.split(key, 3)
        d_a = c["action_dim"]
        ic = self._index_clip()
        u = jnp.clip(jax.random.normal(k_u, (cand, n, d_a)), -c["u_clip"], c["u_clip"])
        idx = jnp.clip(jax.random.normal(k_i, (cand, n, d_a)), -ic, ic)
        w = sampled.task_w[:n]

        def q_of(index, act):
            q_ens = (self.psi_b(obs, index, act, u_prior=sampled.u_duel) * w).sum(-1)  # (P, n)
            q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
            return q_mean - c["actor_pessimism_penalty"] * q_unc            # (n,)

        idx0, u0 = sampled.u_index[:n], sampled.u_data[:n]
        Q_act = jax.vmap(lambda uk: q_of(idx0, uk))(u)                      # (cand, n)
        Q_idx = jax.vmap(lambda ik: q_of(ik, u0))(idx)                      # (cand, n)
        s_act = jnp.abs(Q_act).mean() + 1e-8
        s_idx = jnp.abs(Q_idx).mean() + 1e-8
        out = {"psi_q_spread": Q_act.std(0).mean(),
               "psi_q_spread_rel": Q_act.std(0).mean() / s_act,
               "psi_q_range_rel": (Q_act.max(0) - Q_act.min(0)).mean() / s_act,
               "psi_q_index_spread_rel": Q_idx.std(0).mean() / s_idx}
        if c["psi_form"] == "affine":
            u_w = jnp.clip(jax.random.normal(k_w, (64, d_a)), -ic, ic)
            w_index = self.psi(u_w, method="encode_index")                  # (64, d_w)
            dist = jnp.linalg.norm(w_index[:, None] - w_index[None], axis=-1)
            m = 1.0 - jnp.eye(64)
            out["w_enc_spread"] = (dist * m).sum() / m.sum()
        return out

    def action_critic_spread(self, batch, sampled, key):
        """Spread of Q_a(s, a) = psi_a(s, w, a)^T w over `spread_candidates` decodes at one state.

        If it is ~0 the critic cannot rank the actions the decoder can produce, the -Q_a
        gradient carries no direction, and the residual head is climbing noise. Reported
        relative to |Q| so it is comparable across tasks and training steps.
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

    # ------------------------------------------------------------------ update
    def apply_update(self, batch, sampled):
        """One gradient step of every enabled branch, at the pre-update psi (PSM's convention)."""
        tau = self.config["tau"]
        proto_on = self.config["proto"]["enabled"]
        p_info = {}
        if proto_on:
            # Reference PSM order: the proto stage steps phi + proto_psi, then the SF stage
            # fits psi on the phi THAT step produced (phi frozen there, target read at the
            # online phi), then the actor. Targets polyak after each stage's step.
            (_, p_info), (g_phi, g_proto) = jax.value_and_grad(
                self.proto_measure_loss, argnums=(2, 3), has_aux=True)(
                    batch, sampled, self.phi.params, self.proto_psi.params)
            phi = self.phi.apply_gradients(grads=g_phi)
            proto_psi = self.proto_psi.apply_gradients(grads=g_proto)
            target_phi = polyak_update(phi.params, self.target_phi, tau)
            target_proto_psi = polyak_update(proto_psi.params, self.target_proto_psi, tau)
            (_, info), g_psi = jax.value_and_grad(
                self.measure_loss, argnums=3, has_aux=True)(
                    batch, sampled, phi.params, self.psi.params)
        elif self.config["train_phi"]:
            (_, info), (g_phi, g_psi) = jax.value_and_grad(
                self.measure_loss, argnums=(2, 3), has_aux=True)(
                    batch, sampled, self.phi.params, self.psi.params)
            phi = self.phi.apply_gradients(grads=g_phi)
            target_phi = polyak_update(phi.params, self.target_phi, tau)
        else:
            # Differentiate only psi: phi's params, Adam moments and step remain bitwise
            # unchanged, and the fixed online basis is also the exact target basis.
            (_, info), g_psi = jax.value_and_grad(
                self.measure_loss, argnums=3, has_aux=True)(
                    batch, sampled, self.phi.params, self.psi.params)
            phi = self.phi
            target_phi = phi.params
        psi = self.psi.apply_gradients(grads=g_psi)
        target_psi = polyak_update(psi.params, self.target_psi, tau)
        new = self.replace(phi=phi, psi=psi, target_phi=target_phi, target_psi=target_psi)
        if proto_on:
            new = new.replace(proto_psi=proto_psi, target_proto_psi=target_proto_psi)
        # Under the task-vector index the backup bootstraps the actor's latent at s'.
        # Under policy_index='latent' it uses u', so train_actor=false drops the actor.
        a_info = {}
        # Kept separate: the actor branch below ASSIGNS a_info from its own loss rather
        # than merging into it, so writing the critic diagnostics there would lose them.
        na_info = {}
        if self.config["dsrl_na"]["enabled"]:
            # Critics first, as in SAC. The actor branch below reads the PRE-update qw, the
            # same one-step lag `flow_actor_loss` has against psi -- negligible against the
            # `inner_steps` regression steps qw takes here.
            na = self.config["dsrl_na"]
            (_, qa_info), g_qa = jax.value_and_grad(
                self.dsrl_qa_loss, argnums=2, has_aux=True)(batch, sampled, new.qa.params)
            qa = new.qa.apply_gradients(grads=g_qa)
            new = new.replace(qa=qa,
                              target_qa=polyak_update(qa.params, new.target_qa, na["tau"]))
            # DSRL refits Q_W several times per outer step: it is chasing a target that
            # moves only as fast as Q_A, and the regression is cheap.
            w_na = self._actor_w(sampled.task_w)
            qw, key = new.qw, jax.random.fold_in(self.rng, 114)
            for _i in range(int(na["inner_steps"])):
                (_, qw_info), g_qw = jax.value_and_grad(
                    self.dsrl_qw_loss, argnums=1, has_aux=True)(
                    batch, qw.params, jax.random.fold_in(key, _i), w_na)
                qw = qw.apply_gradients(grads=g_qw)
            new = new.replace(qw=qw)
            na_info = {**qa_info, **qw_info,
                       **new.na_spread(batch, jax.random.fold_in(self.rng, 115), w_na)}
        if self.config["train_actor"]:
            if self.config["actor_mode"] == "ddpg":
                (_, a_info), (g_a, g_vf) = jax.value_and_grad(
                    self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
                    batch, sampled, self.actor.params, self.actor_vf.params)
                new = new.replace(actor=new.actor.apply_gradients(grads=g_a),
                                  actor_vf=new.actor_vf.apply_gradients(grads=g_vf))
            else:
                (_, a_info), (g_a, g_vf) = jax.value_and_grad(
                    self.dsrl_actor_loss, argnums=(2, 3), has_aux=True)(
                    batch, sampled, self.sac_actor.params, self.actor_vf.params)
                new = new.replace(sac_actor=new.sac_actor.apply_gradients(grads=g_a),
                                  actor_vf=new.actor_vf.apply_gradients(grads=g_vf))
                if self.config["actor"]["entropy"] == "auto":
                    (_, al_info), g_al = jax.value_and_grad(
                        self.alpha_loss, argnums=2, has_aux=True)(
                        batch, sampled, new.log_alpha.params)
                    new = new.replace(log_alpha=new.log_alpha.apply_gradients(grads=g_al))
                    a_info = {**a_info, **al_info}
        if self.config["index_agg"] == "expectile":
            (_, qd_info), g_qd = jax.value_and_grad(
                self.q_dist_loss, argnums=2, has_aux=True)(batch, sampled, new.q_dist.params)
            new = new.replace(q_dist=new.q_dist.apply_gradients(grads=g_qd))
            a_info = {**a_info, **qd_info,
                      **new.q_dist_spread(batch, sampled, jax.random.fold_in(self.rng, 109))}
        if self.config["policy_index"] == "latent":
            a_info = {**a_info,
                      **new.index_spread(batch, sampled, jax.random.fold_in(self.rng, 111))}
        if self.config["action_critic"]["enabled"]:
            ac_cfg = self.config["action_critic"]
            if ac_cfg["fb_graft"]:
                # psi_a and its own backward map B_a step together on the FB measure loss;
                # B_a's target follows FB's separate tau.
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
            s_info = new.action_critic_spread(batch, sampled,
                                              jax.random.fold_in(self.rng, 103))
            a_info = {**a_info, **ac_info, **r_info, **s_info}
        return new, {**p_info, **info, **a_info, **na_info}

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
        if self.config["proto"]["enabled"]:
            p_loss, p_info = self.proto_measure_loss(batch, sampled, self.phi.params,
                                                     self.proto_psi.params)
            loss, info = loss + p_loss, {**p_info, **info}
        return loss, info

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

    @jax.jit
    def gpi_select(self, observations, seed=None):
        """Alg. Rung 1: per-step latent argmax over K clipped prior draws, for ONE observation.

        Under policy_index='latent' this is the pair scan argmax_{i,j} [mean_P - kappa*unc]
        psi(s, u_i, u'_j)^T w over K action latents and K policy indices, redrawn every
        step. index_agg='expectile' collapses it to K candidates, q_dist having already
        marginalized u'.
        """
        c = self.config
        assert observations.ndim == 1, "gpi_select acts on a single observation"
        K, d_a = c["gpi_num_u"], c["action_dim"]
        seed = self.rng if seed is None else seed
        u_duel = self._duel_panel(jax.random.fold_in(seed, 5))
        if c["policy_index"] == "latent" and c["index_agg"] == "expectile":
            u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
            obs = jnp.broadcast_to(observations, (K, *observations.shape))
            wq = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
            return u_cand[jnp.argmax(self.q_dist(obs, jnp.concatenate([wq, u_cand], -1)))]
        if c["policy_index"] == "latent" and c["gpi_select"] != "argmax":
            return self._gpi_select_ablation(observations, seed, u_duel)
        if c["policy_index"] == "latent":
            r_u, r_up = jax.random.split(seed)
            ic = self._index_clip()
            u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -c["u_clip"], c["u_clip"])
            u_index = jnp.clip(jax.random.normal(r_up, (K, d_a)), -ic, ic)
            if (c["measure_action_input"] == "action"
                    and c["psi_form"] == "affine" and c["psi_bound"] != "tanh"):
                # Exact affine factorisation: decode and build A(s,a), beta(s,a) once for
                # each of the K current-action candidates, and encode each policy index
                # once. Flatten action-major/index-minor to preserve the explicit K*K
                # scan's ensemble reduction and first-argmax tie order.
                obs_k = jnp.broadcast_to(observations, (K, *observations.shape))
                w_k = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
                # Singleton current-action axis: einsum broadcasts it over K actions,
                # while w_enc itself sees only the K distinct policy indices.
                index_panel = u_index[:, None, :]
                q_panel = self._psi_q_over_indices(obs_k, u_cand, w_k, index_panel,
                                                   u_prior=u_duel)
                q_ens = jnp.swapaxes(q_panel, 1, 2).reshape(c["num_parallel"], K * K)
                q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
                Q = q_mean - c["actor_pessimism_penalty"] * q_unc
                return u_cand[jnp.argmax(Q) // K]
            u_pairs = jnp.repeat(u_cand, K, axis=0)         # (K*K, d_a), i-major
            index_pairs = jnp.tile(u_index, (K, 1))         # (K*K, d_a)
            obs = jnp.broadcast_to(observations, (K * K, *observations.shape))
            psi_out = self.psi_b(obs, index_pairs, u_pairs, u_prior=u_duel)  # (P, K*K, z_dim)
            q_ens = (psi_out * self.task_z).sum(-1)         # (P, K*K)
            q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
            Q = q_mean - c["actor_pessimism_penalty"] * q_unc
            return u_pairs[jnp.argmax(Q)]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        psi_out = self.psi_b(obs, w, u_cand, u_prior=u_duel)  # (P, K, z_dim)
        q_ens = (psi_out * self.task_z).sum(-1)             # (P, K)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        Q = q_mean - c["actor_pessimism_penalty"] * q_unc
        return u_cand[jnp.argmax(Q)]

    def _gpi_select_ablation(self, observations, seed, u_duel=None):
        """`gpi_select != 'argmax'`: eval-time variants of the SELECTION RULE only.

        Reached only under policy_index=latent, index_agg=max. Every mode draws from the
        same clipped prior and decodes through the same frozen flow; they differ only in
        which of the K draws is handed to `decode`.

          argmax               the shipped rule (handled by the caller, not here)
          max_norm             CRITIC-FREE: largest ||u|| among the K draws
          top_quartile_random  CRITIC-FREE: uniform among draws in the top norm quartile
          small_ball           shipped rule, candidates rejected above the prior median ||u||
          soft_topm            uniform among the top `gpi_topm` candidates by GPI score
          mean                 shipped rule with the ensemble MEAN (pessimism dropped)
          fixed_index          shipped rule with u' pinned to ONE draw, so only u varies
          prior_shrunk         CRITIC-FREE: one prior draw scaled by `gpi_prior_shrink`
        """
        c = self.config
        K, d_a = c["gpi_num_u"], c["action_dim"]
        mode = c["gpi_select"]
        r_u, r_up, r_ball, r_pick = jax.random.split(seed, 4)

        if mode == "small_ball":
            # Fixed-shape rejection sampling: oversample, keep the draws at or below the
            # sample median norm, then Gumbel top-k for K survivors without replacement.
            M = _SMALL_BALL_OVERSAMPLE * K
            draws = jnp.clip(jax.random.normal(r_ball, (M, d_a)), -c["u_clip"], c["u_clip"])
            nrm = jnp.linalg.norm(draws, axis=-1)
            keep = nrm <= jnp.median(nrm)
            g = jnp.where(keep, jax.random.gumbel(r_u, (M,)), -jnp.inf)
            u_cand = draws[jnp.argsort(-g)[:K]]
        else:
            u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -c["u_clip"], c["u_clip"])

        if mode == "prior_shrunk":
            # Critic-free control for the DSRL-NA arm: one prior draw scaled to that
            # actor's measured operating norm. Touches neither psi nor task_z, so it
            # isolates the shrinkage from the argmax the arm was built to distill. At
            # scale 1.0 it is the BC control exactly.
            return jnp.clip(c["gpi_prior_shrink"] * u_cand[0], -c["u_clip"], c["u_clip"])

        if mode in ("max_norm", "top_quartile_random"):
            nrm = jnp.linalg.norm(u_cand, axis=-1)
            if mode == "max_norm":
                return u_cand[jnp.argmax(nrm)]
            keep = nrm >= jnp.quantile(nrm, 0.75)
            return u_cand[jnp.argmax(jnp.where(keep, jax.random.gumbel(r_pick, (K,)), -jnp.inf))]

        # Scored modes. n_idx = 1 under fixed_index: one policy index for every step of
        # every episode, so the K x K pair scan collapses to K pairs.
        ic = self._index_clip()
        if mode == "fixed_index":
            n_idx = 1
            u_index = jnp.clip(jax.random.normal(jax.random.PRNGKey(c["gpi_index_seed"]), (1, d_a)),
                               -ic, ic)
        else:
            n_idx = K
            u_index = jnp.clip(jax.random.normal(r_up, (K, d_a)), -ic, ic)
        u_pairs = jnp.repeat(u_cand, n_idx, axis=0)         # (K*n_idx, d_a), i-major
        index_pairs = jnp.tile(u_index, (K, 1))             # (K*n_idx, d_a)
        obs = jnp.broadcast_to(observations, (K * n_idx, *observations.shape))
        q_ens = (self.psi_b(obs, index_pairs, u_pairs, u_prior=u_duel)
                 * self.task_z).sum(-1)                     # (P, K*n_idx)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        pess = 0.0 if mode == "mean" else c["actor_pessimism_penalty"]
        # Per-u GPI score, max over the index panel -- the ranking the shipped argmax
        # consumes, since max over pairs == max over u of max over u'.
        score = (q_mean - pess * q_unc).reshape(K, n_idx).max(-1)
        if mode == "soft_topm":
            top = jnp.argsort(-score)[:c["gpi_topm"]]
            return u_cand[top[jax.random.randint(r_pick, (), 0, c["gpi_topm"])]]
        return u_cand[jnp.argmax(score)]                    # small_ball | mean | fixed_index

    def _acting_w_a(self):
        """Task vector the ACTION branch acts on: its own w_a under the FB graft, else w.

        Reading task_z_a unconditionally would silently zero Q_a for any caller that sets
        task_z directly instead of going through `infer_eval_z`.
        """
        return self.task_z_a if self.config["action_critic"]["fb_graft"] else self.task_z

    def qa_rank_select(self, observations, seed):
        """Eval-time ablation: RANK decoded candidates by Q_a instead of pushing on them.

        K actor draws -> K decodes (no residual) -> argmax_a psi_a(s, w, a)^T w, so every
        action taken is something the frozen flow could have produced. This separates
        "Q_a ranks better" from "Q_a pushes better". Off by default (eval_rank_k = 0).
        """
        c = self.config
        K, d_a = c["action_critic"]["eval_rank_k"], c["action_dim"]
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        noise = jax.random.normal(seed, (K, d_a))
        w_a = jnp.broadcast_to(self._acting_w_a(), (K, *self.task_z.shape))
        u_cand = self._deploy_latent(obs, w, noise)
        a_cand = self.decode(obs, u_cand)
        Q = (self.psi_a(obs, w_a, a_cand) * w_a).sum(-1).mean(0)          # (K,)
        return a_cand[jnp.argmax(Q)]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """The deployed action: select a latent, then decode it through the frozen flow."""
        seed = self.rng if seed is None else seed
        ac = self.config["action_critic"]
        if ac["enabled"] and ac["eval_rank_k"] > 0:
            return self.qa_rank_select(observations, seed)
        assert not (self.config["acting"] == "actor" and not self.config["train_actor"]), (
            "acting=actor with train_actor=false would deploy an untrained actor; "
            "the write-up's LatentFlowPSM has no actor -- use acting=gpi")
        if self.config["acting"] == "actor":
            noise = jax.random.normal(seed, (1, self.config["action_dim"]))
            if self.config["actor_mode"] == "ddpg":
                u_star = self.config["u_clip"] * self.actor(
                    observations[None], self.task_z[None], noise)[0]
            else:
                # temperature=0 deploys the MODE, which is how SAC is normally evaluated
                # and what `utils.evaluation` asks for; anything else deploys the
                # stochastic policy, with `noise` as the reparameterisation draw so the
                # eval stream stays on the ddpg arm's seed. Selected with `where` rather
                # than a Python `if`: `sample_actions` is jitted and `temperature` is a
                # traced argument, so branching on it raises TracerBoolConversionError.
                w_act = self._actor_w(self.task_z)
                mu, log_std = self.sac_actor(observations[None], w_act[None])
                u_stoch, _ = tanh_gaussian_sample(mu, log_std, noise, self.config["u_clip"])
                u_mode = self.config["u_clip"] * jnp.tanh(mu)
                u_star = jnp.where(jnp.asarray(temperature) == 0, u_mode, u_stoch)[0]
        elif self.config["acting"] == "fixed_coeff":
            u_star = self.fixed_coeff_select(observations, seed=seed)
        else:
            u_star = self.gpi_select(observations, seed=seed)
        if ac["enabled"]:
            return self.execute(observations[None], self._acting_w_a()[None], u_star[None])[0]
        return self.decode(observations[None], u_star[None])[0]

    def task_vector_report(self, next_observations, w_eval, key=None):
        """Is the INFERRED eval task vector inside the distribution the arm trained on?

        Arm D2 trains on `w` drawn FB-style: `mix_ratio` of them are `project(phi(s'))` for
        an `s'` in the batch, the rest `project(N(0, I))`. At eval the deployed `w` comes
        from `infer_z`, which is neither -- it is a reward-weighted average of `phi`. If that
        vector sits outside the training mixture, the arm is being asked to generalise
        somewhere it never saw, and a collapse is a coverage failure rather than a critic
        failure. This is the check that separates those two.

        Cosines, not a density: both components are projected onto the sphere of radius
        sqrt(z_dim), so a density in R^z is not the right object and a proper one on the
        sphere would need a normalising constant nobody here would read. Against the null --
        two random unit vectors in z dimensions have cosine ~ N(0, 1/z), i.e. sd 0.088 at
        z=128 -- a cosine is directly interpretable.
        """
        c = self.config
        key = self.rng if key is None else key
        phi_b = project_z(self.phi(next_observations), c["norm_z"])           # (N, z)
        w = w_eval / (jnp.linalg.norm(w_eval) + 1e-12)
        cos_phi = phi_b @ w / (jnp.linalg.norm(phi_b, axis=-1) + 1e-12)
        g = project_z(jax.random.normal(key, phi_b.shape), c["norm_z"])
        cos_gauss = g @ w / (jnp.linalg.norm(g, axis=-1) + 1e-12)
        null_sd = 1.0 / jnp.sqrt(c["z_dim"])
        return {"w_cos_phi_max": cos_phi.max(),          # nearest training phi(s') draw
                "w_cos_phi_mean": cos_phi.mean(),
                "w_cos_phi_p99": jnp.quantile(cos_phi, 0.99),
                "w_nn_dist_phi": jnp.linalg.norm(
                    phi_b - w_eval[None], axis=-1).min(),
                "w_cos_gauss_max": cos_gauss.max(),      # the other mixture component
                "w_null_sd": null_sd,                    # cosine sd of two random unit vectors
                # > ~3 means the eval w points where SOME training draw pointed; ~1 means it
                # is no closer to the training mixture than chance.
                "w_cos_phi_max_over_null": cos_phi.max() / null_sd,
                "w_cos_gauss_max_over_null": cos_gauss.max() / null_sd}

    # ------------------------------------------------------------------ reward inference
    def infer_z(self, next_observations, rewards):
        """w from a relabelling batch, projected onto the sphere.

        `reward_inference='closed_form'` (DEFAULT, every published number) is
        Cor. `reward-inference` literally: w = E_D[r(x) phi(x)].

        That expression is the least-squares readout ONLY when E[phi phi^T] = I, which is
        what the ortho term exists to enforce. Measured 2026-09-09
        (`tools/diag_reward_readout.py`), it holds on cube and fails elsewhere:

            env        Gram dev from I   cond      cos(closed form, least squares)
            cube             0.035        1.3               0.997
            antmaze          0.423       15.4               0.893
            pointmaze        2.39         2e9               0.075

        On pointmaze the Gram has collapsed (smallest eigenvalue 2e-4) and the deployed
        estimator points almost ORTHOGONALLY to the best linear readout -- while the basis
        itself is the most expressive of the three (topline R^2 0.514 against cube's 0.116).
        `whitened` solves the normal equations instead, w = (E[phi phi^T] + eps I)^-1 E[r phi],
        which is that best linear readout. It is the same object wherever the Gram is
        already the identity, so it changes nothing on cube by construction.
        """
        phi = self.phi(next_observations)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        if self.config["reward_inference"] == "whitened":
            gram = (phi.T @ phi) / phi.shape[0]
            eps = self.config["reward_inference_eps"]
            z = jnp.linalg.solve(gram + eps * jnp.eye(gram.shape[0]), z)
        return project_z(z, self.config["norm_z"])

    def infer_z_a(self, next_observations, rewards):
        """w_a = E_D[r(x) B_a(x)], the action branch's own reward inference.

        Without the FB graft B_a is untrained, so callers fall back to w.
        """
        b = self.phi_a(next_observations)
        z = (rewards.reshape(1, -1) @ b).reshape(-1) / b.shape[0]
        return project_z(z, self.config["norm_z"])

    def refit_na_reward(self, next_observations, rewards, ho_next_observations=None,
                        ho_rewards=None):
        """Arm D1b: refit the HELD reward-readout w, and the scale that matches the reward.

        `w` is the closed form `project(E[(r + shift) phi(s')])` on a relabel batch -- the
        same estimator `infer_z` uses at eval -- so the value is trained on the reward
        channel the agent actually deploys, rather than on a fresh 256-row estimate every
        step. The scale is `std(r + shift) / std(phi w)` over that batch: the sphere
        projection fixes `||w||`, which left r_hat's std at 13.6 against the reward's 0.15
        in Arm D1, and a critic trained at that scale is not comparable to one trained on
        the real reward.

        Pass a SECOND, disjoint batch to get `na_rhat_corr_heldout`. The in-batch
        correlation is upward-biased: `w` is fit on the same rows it is then scored
        against, and on cube a 256-row batch holds ~5 rewarding rows against a 128-dim
        basis. The held-out number is the one to read.

        Returns `(agent, info)`. Call it from the training loop, not from inside `update`.
        """
        c = self.config
        shift = float(_na_opt(c, "reward_shift"))
        r_shift = rewards + shift
        ph = self.phi(next_observations)
        w = project_z((r_shift.reshape(1, -1) @ ph).reshape(-1) / ph.shape[0], c["norm_z"])
        r_hat = ph @ w
        scale = r_shift.std() / (r_hat.std() + 1e-8)
        dr, dt = r_hat - r_hat.mean(), r_shift - r_shift.mean()
        info = {"na_rhat_corr_infit": float((dr * dt).mean() / (dr.std() * dt.std() + 1e-8)),
                "na_rw_norm": float(jnp.linalg.norm(w)),
                "na_rw_scale": float(scale),
                "na_rhat_std_prescale": float(r_hat.std()),
                "na_reward_std": float(r_shift.std())}
        if ho_next_observations is not None and ho_rewards is not None:
            ho_r = ho_rewards + shift
            ho_hat = self.phi(ho_next_observations) @ w
            a_, b_ = ho_hat - ho_hat.mean(), ho_r - ho_r.mean()
            info["na_rhat_corr_heldout"] = float(
                (a_ * b_).mean() / (a_.std() * b_.std() + 1e-8))
        return self.replace(na_rw=w, na_rw_scale=scale), info

    def infer_eval_z(self, next_observations, rewards):
        """Copy of this agent with `task_z` set from a relabelling batch.

        Picked up generically by main.py's eval hook. Under the FB graft the action branch
        gets its own task vector through B_a.
        """
        z = self.infer_z(next_observations, rewards)
        z_a = (self.infer_z_a(next_observations, rewards)
               if self.config["action_critic"]["fb_graft"] else z)
        return self.replace(task_z=z, task_z_a=z_a)


def load_fixed_index_coeff(path, w_dim):
    """The Eq. 10 coefficient `c` (w_dim,) from an npz written by
    tools/infer_policy_lagrangian.py. Shape-checked here so a file fitted against another
    checkpoint's head width fails at construction, not inside a jitted einsum."""
    with np.load(str(path), allow_pickle=False) as z:
        assert "c" in z.files, f"{path}: no `c` array"
        coeff = np.asarray(z["c"], np.float32).reshape(-1)
    assert coeff.shape == (int(w_dim),), (
        f"{path}: c has shape {coeff.shape}, this head's w_dim is {w_dim}")
    assert np.all(np.isfinite(coeff)), f"{path}: c is not finite"
    return coeff


def _load_flow_params(ckpt_path, ckpt_epoch, config, ex_observations, ex_actions):
    """Extract the frozen flow subtrees from a Stage-A FQL(bc_only) checkpoint.

    Builds a throwaway FQLAgent with matching shapes, restores the pickle, then pulls
    modules_actor_bc_flow / modules_actor_onestep_flow.
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

    # restore_agent replaces the param tree without checking shapes, so a foreign-env
    # checkpoint loads silently. The bc-flow trunk's first kernel pins the env.
    ob_dim = int(ex_observations.shape[-1])
    action_dim = int(ex_actions.shape[-1])
    got = int(vf["mlp"]["Dense_0"]["kernel"].shape[0])
    expected = ob_dim + action_dim + 1
    assert got == expected, (
        f'flow checkpoint {ckpt_path} expects concat[obs, action, t] of width {got}, but '
        f'this env gives {expected} (obs {ob_dim} + action {action_dim} + t 1); '
        'the Stage-A checkpoint was trained on a different environment')
    return vf, onestep


def _load_phi_params(restore_path, restore_epoch, template):
    """phi's ONLINE params from another run's `params_<epoch>.pkl` (`restore_agent`'s glob
    convention: `restore_path` must match exactly one run directory).

    Only the `agent/phi/params` subtree is read. The tree must have the template's
    structure and leaf shapes, i.e. the same `phi` block and `z_dim` and the same env.
    """
    import glob
    import os
    import pickle

    candidates = glob.glob(str(restore_path))
    assert len(candidates) == 1, (
        f"phi_restore_path must match exactly one run directory; {restore_path!r} matched "
        f"{len(candidates)}: {candidates}")
    assert restore_epoch is not None, "phi_restore_path needs phi_restore_epoch"
    path = os.path.join(candidates[0], f"params_{int(restore_epoch)}.pkl")
    with open(path, "rb") as f:
        saved = pickle.load(f)["agent"]["phi"]["params"]
    loaded = flax.serialization.from_state_dict(template, saved)
    for want, got in zip(jax.tree_util.tree_leaves(template), jax.tree_util.tree_leaves(loaded)):
        assert tuple(want.shape) == tuple(jnp.shape(got)), (
            f"{path}: phi leaf shape {tuple(jnp.shape(got))} does not match this agent's "
            f"{tuple(want.shape)}; the checkpoint's phi block, z_dim or env differs")
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), loaded)


def get_config():
    """Importable default config, mirrored by configs/agent/psmflow.yaml."""
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            agent_name="psmflow",
            batch_size=1024,
            z_dim=128,
            num_parallel=2,          # critic ensemble size P
            discount=0.98,
            tau=0.01,
            ortho_coef=1000.0,       # lambda_perp, the reference PSM sweep winner
            # --- measure-loss stabilisers (2026-09-07); both OFF reproduce the published
            # loss exactly. See docs/design/2026-09-07-td-divergence-fix.md.
            ortho_mode="fixed",      # fixed | relative
            ortho_rel_coef=1.0,      # ortho_mode=relative: weight on stopgrad(|psm_loss|)
            psi_bound="none",        # none | tanh | clip_target
            psi_bound_scale=200.0,   # per-component ceiling on psi (~1/(1-discount))
            # kappa. 0.5 with num_parallel=2 is exactly min-Q in the TD target.
            pessimism_penalty=0.5,
            actor_pessimism_penalty=0.5,
            norm_z=True,
            lr_phi=1.0e-5,
            lr_sf=1.0e-4,
            lr_actor=1.0e-4,
            lr_actor_vf=3.0e-4,
            # Diagnostic continuation switch. False freezes phi's complete TrainState;
            # psi (including the affine policy-index encoder) continues training.
            train_phi=True,
            # Take phi's params from another run's params_<epoch>.pkl (online phi into phi
            # and target_phi; optimiser state fresh). null = phi starts at its init.
            phi_restore_path=ml_collections.config_dict.placeholder(str),
            phi_restore_epoch=ml_collections.config_dict.placeholder(int),
            phi=dict(hidden_dim=256, hidden_layers=2),
            sf=dict(hidden_dim=1024, hidden_layers=1, embedding_layers=2),
            mix_ratio=0.5,           # P(w drawn as phi(next_obs[perm])) vs a random unit z
            # Fraction of TD bootstrap latents drawn from the prior instead of the actor.
            backup_explore_frac=0.0,
            # Backup action slot at s'. "index" = today's backup. "gpi_argmax" (needs
            # policy_index=task_vector) = EMaQ-style max over bootstrap_candidates prior
            # draws of the pessimistic target-psi readout under the row's w.
            bootstrap="index",            # index | gpi_argmax
            bootstrap_candidates=8,
            # Amortized latent actor pi_eta, off by default (train_actor=False).
            actor=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                       vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10,
                       bc_coeff=1.0,
                       # K prior policy indices u' the actor's Q is aggregated over;
                       # 0 reads psi at the single index draw. See `_actor_q`.
                       index_panel=0,
                       # DSRL arms (actor_mode != 'ddpg') only.
                       q_coeff=1.0,              # weight on the (alpha*logpi - Q) term
                       na_coeff=1.0,             # weight on the GPI-argmax regression
                       na_candidates=16,         # prior action latents scored per state
                       na_states=256,            # batch rows the NA target is built on
                       na_advantage_weight=False,
                       entropy="auto",           # auto | fixed (fixed pins alpha=init_alpha)
                       target_entropy=0.0,       # DSRL's target_ent, not -dim(A)
                       init_alpha=1.0, lr_alpha=3.0e-4,
                       log_std_min=-20.0, log_std_max=2.0,
                       prior_init=False, prior_init_std=0.3,
                       # LayerNorm on the tanh-Gaussian trunk's hidden stack; DSRL puts it
                       # on every hidden layer of actor and critic alike.
                       layer_norm=False),
            # Which latent actor `train_actor=true` trains and `acting=actor` deploys.
            actor_mode="ddpg",       # ddpg | dsrl_sac | gpi_distill
            acting="gpi",            # gpi (per-step latent argmax, actor-free) | actor
                                     # | fixed_coeff (PSM Eq. 10 coefficient, eval only)
            # acting=fixed_coeff: npz holding `c` (w_dim,) from tools/infer_policy_lagrangian.py
            fixed_index_coeff_path=ml_collections.config_dict.placeholder(str),
            # psi's index slot. "latent" is psi(s, u, u'): u' ~ p0 per row indexes the
            # policy and w reaches psi only via Q = psi^T w. "task_vector" is psi(s, w, u).
            policy_index="latent",        # latent | task_vector
            # "affine" is Prop. `bilinear` literally, psi = A(s,u)^T w(u') + beta(s,u); it
            # needs policy_index="latent". "free" absorbs w(u') into psi (Rem. `tradeoff`).
            psi_form="affine",            # affine | free
            # Current-action input of the affine measure head. ``latent`` reproduces every
            # existing run; ``action`` trains on recorded actions and decodes all latent
            # queries through the frozen flow.
            measure_action_input="latent",  # latent | action
            affine=dict(w_dim=128, encoder_hidden=256, encoder_layers=2, norm_w=True),
            # Fix 2 (2026-09-15). psm_scalar_coef > 0 adds the readout's projected Bellman
            # loss (psi^T z against gamma psibar^T z + phi(s')^T z, variance-normalised) to
            # the measure loss, psi-only gradient. psi_dueling decomposes the affine head
            # into V(s,z) + Adv(s,u,z) - mean over psi_dueling_samples prior latents.
            psm_scalar_coef=0.0,
            psi_dueling=False,
            psi_dueling_samples=8,
            # How psi's index slot is aggregated. "max" is the argmax over the panel;
            # "expectile" distills an upper expectile into q_dist, taking no sample argmax.
            index_agg="max",              # max | expectile
            expectile_mu=0.9,             # upper expectile; -> 1 approaches max_{u'}
            index_panel=16,               # u' draws per state forming the target spread
            # q_dist head. `embedding_layers` is carried for config symmetry with the other
            # heads and is unused by Value, which is a plain MLP.
            q_dist=dict(hidden_dim=512, hidden_layers=1, embedding_layers=2, lr=3.0e-4),
            # Train the amortized latent actor. False drops the actor/CFM branch; needs
            # acting=gpi and policy_index="latent", where the backup does not read it.
            train_actor=False,
            # DSRL-NA's dual critic (2026-09-08). NOT ZERO-SHOT: qa is fitted on the
            # task's real reward. Requires actor_mode=dsrl_sac, acting=actor, bc_coeff=0.
            dsrl_na=dict(enabled=False, discount=0.99, tau=0.005, lr=3.0e-4,
                         hidden_dim=2048, hidden_layers=3, layer_norm=True,
                         num_ensembles=2, inner_steps=10, n_latent=1,
                         # real (Item 2) | phi_readout (Arm D1) |
                         # phi_readout_fixed (Arm D1b) | synthetic_w (Arm D2).
                         reward_source="real", reward_shift=1.0,
                         # Arm D1b only: how often main.py refits the held reward-readout
                         # w from a fresh relabel batch. Ignored by every other source.
                         reward_refit_every=10000,
                         # Arm D2: critics and actor take w, so the arm generalises over
                         # tasks instead of solving one. Requires synthetic_w.
                         task_conditioned=False,
                         # Fix 1 (2026-09-15): qa's TD target with mask = 1 on every row
                         # (continuing formulation), and a multiplier on the synthetic
                         # reward phi(s')^T w (synthetic_w only).
                         ignore_masks=False, reward_scale=1.0),
            # Reference PSM critic (2026-09-14): the proto successor-measure stage on latent
            # inputs, psi as PSM's separate SF head. enabled=False is the shipped critic.
            # Requires policy_index=task_vector, train_actor=true; main.py switches on
            # dataset.return_index for it. ortho_coef null = the agent's ortho_coef.
            proto={"enabled": False, "max_log_seed": 16, "proto_seed": 0, "lr": 1.0e-4,
                   "ortho_coef": ml_collections.config_dict.placeholder(float)},
            # Action branch: successor features over executed actions plus an eps-bounded
            # residual. `pessimism` is a [0,1] blend weight, not a spread multiplier.
            action_critic=dict(enabled=False, discount=0.99, tau=0.005, pessimism=0.0,
                               lr=3.0e-4, hidden_dim=512, hidden_layers=1,
                               embedding_layers=2, spread_candidates=16,
                               eval_rank_k=0,
                               # graft: psi_a written against its OWN backward map B_a,
                               # trained by the FB measure loss; w_a inferred from B_a.
                               fb_graft=False, ortho_coef=1000.0, b_tau=0.005),
            residual_eps=0.05,       # hard per-dim budget on |a_exec - decode|
            residual=dict(hidden_dim=256, hidden_layers=2, embedding_layers=2),
            # Frozen behaviour flow (must match the Stage-A fql bc_only run).
            flow=dict(hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
                      layer_norm=False, critic_layer_norm=True),
            flow_ckpt_path=ml_collections.config_dict.placeholder(str),
            flow_ckpt_epoch=ml_collections.config_dict.placeholder(int),
            allow_untrained_flow=False,
            preimage_path=ml_collections.config_dict.placeholder(str),
            # main.py refuses a corrected-target preimage npz with false, and every
            # published number was produced with true.
            use_point_preimage=True,
            # Drop transitions whose Stage-B inversion diverged from the measure loss.
            # False = train on them, which every number before 2026-09-08 did.
            mask_invalid_preimages=False,
            # How many action latents the MEASURE head is fitted at per transition. 1 is
            # the published loss (the single point preimage). > 1 adds latents carrying the
            # SAME (s, s') target, from `measure_u_source`:
            #   mixture  the stored EM preimage posterior. Needs an npz inverted at
            #            inversion.prior_scale > 0 (main.py enforces it), and MEASURED on
            #            cube it decodes 0.205 from the recorded action against the point
            #            inverse's 0.089 -- 60% of the way to an uninverted prior draw
            #            (tools/diag_mixture_decode.py). Those are not other preimages.
            #   jitter   u_data + measure_u_jitter_std * eps. The width is picked from the
            #            same probe's ladder: sigma=0.3 decodes at 0.103, inside the point
            #            inverse's own p90 of 0.128, i.e. still the same action.
            measure_u_samples=1,
            measure_u_source="mixture",     # mixture | jitter
            measure_u_jitter_std=0.3,
            # How `infer_z` reads w off a relabelling batch. "closed_form" is E[r phi],
            # every published number; "whitened" solves the normal equations, which is the
            # same thing wherever E[phi phi^T] = I and is not on antmaze or pointmaze.
            reward_inference="closed_form",   # closed_form | whitened
            reward_inference_eps=1e-3,        # ridge on the Gram, whitened only
            # Arm C. Covariance scale c on the mixture draws (cov -> c^2 * cov). NO
            # default: per-env, and picked from tools/diag_mixture_decode.py --shrink.
            measure_u_mixture_shrink=ml_collections.config_dict.placeholder(float),
            u_clip=3.0,              # typical-set clamp on all latent draws
            # Tighter box for the u' draws alone (psi's index slot, the bootstrap action,
            # GPI's u' roster); None = u_clip. u_data and the candidates u stay on u_clip.
            index_clip=ml_collections.config_dict.placeholder(float),
            # acting=gpi (per-step latent argmax) inference
            gpi_num_u=64,            # K, the size of Lambda_K
            # Selection rule over the K draws. "argmax" is the shipped rule; the rest are
            # eval-time ablations (docs/design/2026-09-05-gpi-selection-diag.md).
            gpi_select="argmax",     # argmax | max_norm | top_quartile_random | small_ball
                                     # | soft_topm | mean | fixed_index
            gpi_topm=8,              # gpi_select=soft_topm: pick uniformly among the top m
            gpi_index_seed=0,        # gpi_select=fixed_index: the pinned u' draw
            # gpi_select=prior_shrunk: scale on the single prior draw. 0.8 * 2.13 (the
            # clipped prior median at d_a=5) = 1.70, the gpi_distill norm. 1.0 is the BC control.
            gpi_prior_shrink=0.8,
            gpi_decode="onestep",    # onestep | ode
            flow_decode_steps=10,    # Euler steps for gpi_decode=ode
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
