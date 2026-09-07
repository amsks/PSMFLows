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
"""

import copy
import math
from typing import Any

import flax
import jax
import jax.numpy as jnp
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

# gpi_select="small_ball" oversample factor: keeps the accepted pool >= K after
# the median-norm cut.
_SMALL_BALL_OVERSAMPLE = 32

#: Eval-time selection rules over the K prior draws; see `_gpi_select_ablation`.
GPI_SELECT_MODES = ("argmax", "max_norm", "top_quartile_random", "small_ball",
                    "soft_topm", "mean", "fixed_index", "prior_shrunk")

#: Latent-actor heads. "ddpg" is the NoiseConditionedActor + flow-BC recipe; the two DSRL
#: modes share the tanh-Gaussian head.
ACTOR_MODES = ("ddpg", "dsrl_sac", "dsrl_na")

#: Defaults for the `actor` sub-keys, applied in `create` so a partial dict -- a test
#: passing only the widths, or a flags.json written before a key existed -- still builds.
ACTOR_DEFAULTS = dict(index_panel=0, q_coeff=1.0, na_coeff=1.0, na_candidates=16,
                      na_states=256, na_advantage_weight=False, entropy="auto",
                      target_entropy=0.0, init_alpha=1.0, lr_alpha=3.0e-4,
                      log_std_min=-20.0, log_std_max=2.0)


def _actor_opt(config, key):
    """`config.actor[key]`, falling back to ACTOR_DEFAULTS when the dict lacks it."""
    try:
        return config["actor"][key]
    except (KeyError, AttributeError):
        return ACTOR_DEFAULTS[key]


#: Loss-stabiliser keys (2026-09-07). Every default is the OFF value -- the loss the
#: published numbers used. `fill_stability_defaults` backfills an older flags.json.
STABILITY_DEFAULTS = dict(ortho_mode="fixed", ortho_rel_coef=1.0,
                          psi_bound="none", psi_bound_scale=200.0)

#: Measure/acting keys `create` guards with `.get(<legacy default>)` but the runtime then
#: reads unguarded. Backfilled with the SAME legacy values, so a flags.json predating any
#: of them restores onto the old behaviour instead of dying at the first update or eval.
MEASURE_DEFAULTS = dict(psi_form="free", index_agg="max", gpi_select="argmax",
                        index_clip=None)

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
    """
    u_data: Any
    u_next: Any
    task_w: Any
    u_index: Any
    flow_x0: Any
    flow_t: Any
    flow_noise: Any
    task_w_a: Any


class PSMFlowAgent(flax.struct.PyTreeNode):
    """LatentFlowPSM. See `docs/reference/psmflow-symbols.md` for the symbol map."""

    rng: Any
    phi: TrainState             # basis phi(x) over future states
    psi: TrainState             # successor features psi(s, index, u), P-fold ensemble
    target_phi: Any
    target_psi: Any
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
    task_z: Any                 # (z_dim,) eval task vector w, set by infer_eval_z
    task_z_a: Any               # (z_dim,) eval task vector of the action branch
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
        assert not (gpi_select != "argmax"
                    and (config["policy_index"] != "latent"
                         or config.get("index_agg", "max") != "max")), (
            "gpi_select != argmax requires policy_index=latent and index_agg=max: the "
            "selection ablations act on the (u, u') pair scan, which the other arms do "
            "not run.")
        actor_mode = config.get("actor_mode", "ddpg")
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
        assert not (actor_mode == "dsrl_na" and panel == 0), (
            "actor_mode=dsrl_na regresses onto max over the index panel, so panel=0 would "
            "silently distil a ONE-index argmax while the deployed rule scans gpi_num_u; "
            "set actor.index_panel > 0.")
        psi_form = config.get("psi_form", "free")
        assert psi_form in ("free", "affine"), "psi_form: free | affine"
        assert not (psi_form == "affine" and config["policy_index"] != "latent"), (
            "psi_form=affine requires policy_index=latent: the affine head reads its index "
            "slot as the POLICY latent u' and encodes it into w(u'), while A and beta are "
            "by Assumption `affine` independent of the policy index. A z_dim task vector "
            "in that slot would make A(s,u), beta(s,u) functions of the task.")
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
                                   encoder_layers=af["encoder_layers"], norm_w=af["norm_w"])
        else:
            psi_def = PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                             num_parallel=config["num_parallel"],
                             embedding_layers=config["sf"]["embedding_layers"],
                             hidden_layers=config["sf"]["hidden_layers"])
        phi = TrainState.create(phi_def, phi_def.init(rphi, ex_observations)["params"],
                                tx=optax.adam(config["lr_phi"]))
        # psi's index slot: the policy latent u' (d_a wide) under policy_index='latent',
        # the task vector w (z_dim wide) otherwise. The action slot is the latent either way.
        ex_index = ex_u if config["policy_index"] == "latent" else ex_w
        psi = TrainState.create(psi_def, psi_def.init(rpsi, ex_observations, ex_index, ex_u)["params"],
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
            log_std_max=_actor_opt(config, "log_std_max"))
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

        config = _plain_config(config)
        # Backfill the `actor` sub-keys so the runtime never sees a partial dict; every
        # default reproduces the behaviour of a config written before that key existed.
        config.setdefault("actor_mode", "ddpg")
        for _k, _v in ACTOR_DEFAULTS.items():
            config["actor"].setdefault(_k, _v)
        for _k, _v in STABILITY_DEFAULTS.items():
            config.setdefault(_k, _v)
        for _k, _v in MEASURE_DEFAULTS.items():
            config.setdefault(_k, _v)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        return cls(rng=rng, phi=phi, psi=psi,
                   target_phi=copy.deepcopy(phi.params), target_psi=copy.deepcopy(psi.params),
                   actor=actor, actor_vf=actor_vf,
                   sac_actor=sac_actor, log_alpha=log_alpha, q_dist=q_dist,
                   psi_a=psi_a, target_psi_a=copy.deepcopy(psi_a.params), residual=residual,
                   phi_a=phi_a, target_phi_a=copy.deepcopy(phi_a.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   task_z=jnp.zeros((z_dim,), jnp.float32),
                   task_z_a=jnp.zeros((z_dim,), jnp.float32),
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
        # u_clip clamps every latent to the typical set: the bootstrap is a tanh-bounded
        # actor latent, so an unclipped u_data feeds the online branch unreachable inputs.
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
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
        if c["policy_index"] == "latent":
            # The continuation at s' is pi_{u'}, the same index the online side carries --
            # that makes G(s', u') a p0 decode (Prop. `insample`), so explore_frac is inert.
            u_next = u_index
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
        return StepInputs(
            u_data=u_data, u_next=u_next, task_w=task_w, u_index=u_index,
            flow_x0=jax.random.normal(r_x0, (B, adim)),
            flow_t=jax.random.uniform(r_t, (B, 1)),
            flow_noise=jax.random.normal(r_noise, (B, adim)),
            task_w_a=task_w_a)

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

    def psi_b(self, *args, **kwargs):
        """`self.psi(...)` with the magnitude bound applied. Every psi READ goes through it.

        Only for calls that return psi itself; `method='sa_terms'` / `'encode_index'`
        return the factors A, beta, w(u'), which are not the bounded object.
        """
        return self.bound_psi(self.psi(*args, **kwargs))

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
        psi_raw = self.psi(obs, index, u, params=psi_params)
        M = self.bound_psi(psi_raw) @ phi_next.T
        target_phi_next = self.phi(next_obs, params=self.target_phi)
        M_boot = self.psi_b(next_obs, index, u_next, params=self.target_psi) @ target_phi_next.T
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
        sm, sm_diag, sm_offdiag = contrastive_loss(
            M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ortho, ortho_diag, ortho_offdiag = ortho_loss(phi_next, off, off_sum)
        # `ortho_mode='relative'`: weight the geometry regulariser BY the term it holds
        # down, ortho_coef + ortho_rel_coef * stopgrad(|psm_loss|), so the two gradient
        # directions scale together and ortho cannot be outgrown. The weight is
        # stop-gradded, so only phi's gradient direction changes.
        ortho_weight = c["ortho_coef"]
        if c["ortho_mode"] == "relative":
            ortho_weight = ortho_weight + c["ortho_rel_coef"] * jnp.abs(
                jax.lax.stop_gradient(sm))
        loss = sm + ortho_weight * ortho
        return loss, {"psm_loss": sm, "psm_diag": sm_diag, "psm_offdiag": sm_offdiag,
                      "orth_loss": ortho, "orth_diag": ortho_diag, "orth_offdiag": ortho_offdiag,
                      # Stabiliser telemetry, logged in every arm so the fixed and relative
                      # runs share a CSV schema.
                      "ortho_weight": jnp.asarray(ortho_weight, jnp.float32),
                      "ortho_term_abs": jnp.abs(ortho_weight * ortho),
                      "psi_absmean": jnp.mean(jnp.abs(psi_raw)),
                      "psi_absmax": jnp.max(jnp.abs(psi_raw)),
                      "td_target_absmean": jnp.mean(jnp.abs(target_M)),
                      "psi_bound_frac": bound_frac}

    # ------------------------------------------------------------------ latent-actor losses
    def _psi_q_over_indices(self, obs, u, w, u_index, params=None):
        """Q(s, u, u'_k) = psi(s, u, u'_k)^T w for K policy indices at once. Returns (P, K, B).

        Under psi_form=affine, A and beta do not depend on the index, so an index panel
        costs one einsum in w-space instead of K towers. psi_form=free and psi_bound='tanh'
        have no such factorisation and fall back to K full psi calls.
        """
        if self.config["psi_form"] == "affine" and self.config["psi_bound"] != "tanh":
            A, beta = self.psi(obs, u, params=params, method="sa_terms")   # (P,B,z,d_w),(P,B,z)
            w_index = self.psi(u_index, params=params, method="encode_index")   # (K, B, d_w)
            Aw = jnp.einsum("pbzw,bz->pbw", A, w)
            beta_w = jnp.einsum("pbz,bz->pb", beta, w)
            return jnp.einsum("pbw,kbw->pkb", Aw, w_index) + beta_w[:, None, :]   # (P, K, B)
        return jax.vmap(lambda idx: (self.psi_b(obs, idx, u, params=params) * w).sum(-1),
                        out_axes=1)(u_index)                               # (P, K, B)

    def _actor_q(self, obs, u_a, sampled):
        """The Q the latent actor climbs, and the raw ensemble readout it is normalised by.

        `actor.index_panel = 0` reads psi at the ONE prior index this row drew; K > 0
        replaces it with the max over K clipped prior draws -- the object `gpi_select`
        maximises. Redundant under index_agg='expectile', which `create` rejects.
        """
        c = self.config
        w = sampled.task_w
        if c["index_agg"] == "expectile":
            Q = q_ens = self.q_dist(obs, jnp.concatenate([w, u_a], -1))    # (B,)
            return Q, q_ens
        K = int(c["actor"]["index_panel"])
        if K <= 0:
            q_ens = (self.psi_b(obs, self._index(sampled), u_a) * w).sum(-1)  # (P, B)
            q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
            return q_mean - c["actor_pessimism_penalty"] * q_unc, q_ens
        B, d_a = obs.shape[0], c["action_dim"]
        ic = self._index_clip()
        u_index = jnp.clip(jax.random.normal(jax.random.fold_in(self.rng, 112), (K, B, d_a)),
                           -ic, ic)
        q_panel = self._psi_q_over_indices(obs, u_a, w, u_index)           # (P, K, B)
        q_mean, q_unc = targets_uncertainty(q_panel, c["num_parallel"])    # (K, B)
        Q = (q_mean - c["actor_pessimism_penalty"] * q_unc).max(0)         # (B,)
        return Q, q_panel.reshape(q_panel.shape[0], -1)

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

        def score(u_m):
            q_panel = self._psi_q_over_indices(obs, u_m, w, u_index)        # (P, K, B)
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
                      "actor_bc_flow_loss": bc_flow_loss, "actor_bc_error": distill}

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
        obs, w = batch["observations"], sampled.task_w
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
        scale = jax.lax.stop_gradient(jnp.abs(q_ens).mean() + 1e-8)
        alpha = jax.lax.stop_gradient(jnp.exp(self.log_alpha()))
        q_term = -Q.mean() / scale
        # The entropy term sits OUTSIDE q_coeff: at q_coeff=0 it is the only thing stopping
        # the tanh-Gaussian from shrinking its scale onto the regression target's mean.
        ent_term = alpha * logp.mean()

        info = {"actor_q": Q.mean(), "actor_logp": logp.mean(), "actor_alpha": alpha,
                "actor_entropy": -logp.mean(), "actor_u_norm": jnp.linalg.norm(u_a, axis=-1).mean()}
        loss = c["actor"]["q_coeff"] * q_term + ent_term
        if mode == "dsrl_na":
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
        obs, w = batch["observations"], sampled.task_w
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
        q_ens = (self.psi_b(obs_rep, u_index.reshape(panel * B, d_a), u_rep) * w_rep).sum(-1)
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
            q_ens = (self.psi_b(obs, index, act) * w).sum(-1)               # (P, n)
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
        (_, info), (g_phi, g_psi) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.psi.params)
        phi = self.phi.apply_gradients(grads=g_phi)
        psi = self.psi.apply_gradients(grads=g_psi)
        target_phi = polyak_update(phi.params, self.target_phi, tau)
        target_psi = polyak_update(psi.params, self.target_psi, tau)
        new = self.replace(phi=phi, psi=psi, target_phi=target_phi, target_psi=target_psi)
        # Under the task-vector index the backup bootstraps the actor's latent at s'.
        # Under policy_index='latent' it uses u', so train_actor=false drops the actor.
        a_info = {}
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
        if c["policy_index"] == "latent" and c["index_agg"] == "expectile":
            u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
            obs = jnp.broadcast_to(observations, (K, *observations.shape))
            wq = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
            return u_cand[jnp.argmax(self.q_dist(obs, jnp.concatenate([wq, u_cand], -1)))]
        if c["policy_index"] == "latent" and c["gpi_select"] != "argmax":
            return self._gpi_select_ablation(observations, seed)
        if c["policy_index"] == "latent":
            r_u, r_up = jax.random.split(seed)
            ic = self._index_clip()
            u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -c["u_clip"], c["u_clip"])
            u_index = jnp.clip(jax.random.normal(r_up, (K, d_a)), -ic, ic)
            u_pairs = jnp.repeat(u_cand, K, axis=0)         # (K*K, d_a), i-major
            index_pairs = jnp.tile(u_index, (K, 1))         # (K*K, d_a)
            obs = jnp.broadcast_to(observations, (K * K, *observations.shape))
            psi_out = self.psi_b(obs, index_pairs, u_pairs)  # (P, K*K, z_dim)
            q_ens = (psi_out * self.task_z).sum(-1)         # (P, K*K)
            q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
            Q = q_mean - c["actor_pessimism_penalty"] * q_unc
            return u_pairs[jnp.argmax(Q)]
        u_cand = jnp.clip(jax.random.normal(seed, (K, d_a)), -c["u_clip"], c["u_clip"])
        obs = jnp.broadcast_to(observations, (K, *observations.shape))
        w = jnp.broadcast_to(self.task_z, (K, *self.task_z.shape))
        psi_out = self.psi_b(obs, w, u_cand)                # (P, K, z_dim)
        q_ens = (psi_out * self.task_z).sum(-1)             # (P, K)
        q_mean, q_unc = targets_uncertainty(q_ens, c["num_parallel"])
        Q = q_mean - c["actor_pessimism_penalty"] * q_unc
        return u_cand[jnp.argmax(Q)]

    def _gpi_select_ablation(self, observations, seed):
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
        q_ens = (self.psi_b(obs, index_pairs, u_pairs) * self.task_z).sum(-1)  # (P, K*n_idx)
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
                # DSRL deploys the stochastic policy, not its mode. `noise` is the
                # reparameterisation draw, keeping the eval stream on the ddpg arm's seed.
                u_star = self._sac_latent(observations[None], self.task_z[None], noise)[0][0]
        else:
            u_star = self.gpi_select(observations, seed=seed)
        if ac["enabled"]:
            return self.execute(observations[None], self._acting_w_a()[None], u_star[None])[0]
        return self.decode(observations[None], u_star[None])[0]

    # ------------------------------------------------------------------ reward inference
    def infer_z(self, next_observations, rewards):
        """w = E_D[r(x) phi(x)] (Cor. `reward-inference`), projected onto the sphere."""
        phi = self.phi(next_observations)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_z_a(self, next_observations, rewards):
        """w_a = E_D[r(x) B_a(x)], the action branch's own reward inference.

        Without the FB graft B_a is untrained, so callers fall back to w.
        """
        b = self.phi_a(next_observations)
        z = (rewards.reshape(1, -1) @ b).reshape(-1) / b.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_eval_z(self, next_observations, rewards):
        """Copy of this agent with `task_z` set from a relabelling batch.

        Picked up generically by main.py's eval hook. Under the FB graft the action branch
        gets its own task vector through B_a.
        """
        z = self.infer_z(next_observations, rewards)
        z_a = (self.infer_z_a(next_observations, rewards)
               if self.config["action_critic"]["fb_graft"] else z)
        return self.replace(task_z=z, task_z_a=z_a)


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
            phi=dict(hidden_dim=256, hidden_layers=2),
            sf=dict(hidden_dim=1024, hidden_layers=1, embedding_layers=2),
            mix_ratio=0.5,           # P(w drawn as phi(next_obs[perm])) vs a random unit z
            # Fraction of TD bootstrap latents drawn from the prior instead of the actor.
            backup_explore_frac=0.0,
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
                       log_std_min=-20.0, log_std_max=2.0),
            # Which latent actor `train_actor=true` trains and `acting=actor` deploys.
            actor_mode="ddpg",       # ddpg | dsrl_sac | dsrl_na
            acting="gpi",            # gpi (per-step latent argmax, actor-free) | actor
            # psi's index slot. "latent" is psi(s, u, u'): u' ~ p0 per row indexes the
            # policy and w reaches psi only via Q = psi^T w. "task_vector" is psi(s, w, u).
            policy_index="latent",        # latent | task_vector
            # "affine" is Prop. `bilinear` literally, psi = A(s,u)^T w(u') + beta(s,u); it
            # needs policy_index="latent". "free" absorbs w(u') into psi (Rem. `tradeoff`).
            psi_form="affine",            # affine | free
            affine=dict(w_dim=128, encoder_hidden=256, encoder_layers=2, norm_w=True),
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
            # clipped prior median at d_a=5) = 1.70, the dsrl_na norm. 1.0 is the BC control.
            gpi_prior_shrink=0.8,
            gpi_decode="onestep",    # onestep | ode
            flow_decode_steps=10,    # Euler steps for gpi_decode=ode
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
