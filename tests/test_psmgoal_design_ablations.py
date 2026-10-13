"""psmgoal design-decision ablations (2026-10-09, mailbox/tasks/2026-10-09-design-decisions.md).

Three switches, all OFF by default:

- actor_goal_source=measure: the goal actor is trained at the row's measure goal batch['goals']
  instead of the uniform-future goal batch['fb_goals'] (task 4c).
- bootstrap_switch (actor_kind=flowbc): the bootstrap latent is one clipped prior draw (a BC
  draw) before (prior_then_actor) or from (actor_then_prior) bootstrap_switch_step training
  steps, and the flowbc actor's latent otherwise (task 3b).
- acting=flowbc_gpi (eval only): acting=gpi with the gpi_num_u candidates drawn from the flowbc
  actor, candidate k fed eval_goals[k mod K_g], scored by `_hgoal_each_score` (task 1a).

Checks: defaults in get_config and the yaml; default bootstrap is the actor's latent; each switch
mode gives the BC draw or the actor latent on the right side of the step; the actor step reads
the selected goal key; flowbc_gpi returns the argmax candidate and equals a hand computation;
the asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_design_ablations.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

NEW_KEYS = {"actor_goal_source": "future", "bootstrap_switch": "none", "bootstrap_switch_step": 100000}

# the best config, psmgoal_ja_fbc3_pt_cube
BEST = {"policy_index": "goal", "bootstrap_source": "actor", "train_actor": True, "actor_input": "goal",
        "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "measure_loss": "softmax",
        "acting": "distill", "eval_redistill": False,
        "actor_kind": "flowbc", "fb_bc_coeff": 3.0, "actor_value_kind": "point"}


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.w_star.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
        "fb_goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _relabel(n=32, seed=5, n_rewarding=6):
    b = _batch(n=n, seed=seed)
    rew = np.zeros((n,), np.float32)
    rew[:n_rewarding] = 1.0
    return b, rew


def _perturbed_delta(ag, bias=0.5):
    """delta with its output layer moved off zero, so the actor latent differs from eps."""
    params = jax.tree_util.tree_map(lambda x: x, ag.delta.params)
    name = pg.DELTA_OUT_LAYER
    params[name]["bias"] = jnp.full_like(params[name]["bias"], bias)
    params[name]["kernel"] = 0.1 * jax.random.normal(jax.random.PRNGKey(9), params[name]["kernel"].shape)
    return ag.replace(delta=ag.delta.replace(params=params))


def _boot_parts(ag, b):
    """(actor latent, BC draw) at (s', g) for the bootstrap's own eps."""
    eps = jax.random.normal(jax.random.fold_in(ag.rng, pg.FLOWBC_BOOT_KEY), (N, DA))
    u_actor = ag.flowbc_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]), eps)[0]
    return u_actor, jnp.clip(eps, -ag.config["u_clip"], ag.config["u_clip"])


# ------------------------------------------------------------------ defaults

def test_new_keys_default_off_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in NEW_KEYS.items():
        assert c[k] == v, k
        assert y[k] == v, k


def test_default_bootstrap_is_the_actor_latent():
    ag = _perturbed_delta(_agent(**BEST))
    b = _batch()
    u_actor, u_prior = _boot_parts(ag, b)
    assert not np.allclose(np.asarray(u_actor), np.asarray(u_prior))
    np.testing.assert_allclose(np.asarray(ag._select_u_next(b, None)), np.asarray(u_actor), atol=1e-6)


# ------------------------------------------------------------------ bootstrap switch

@pytest.mark.parametrize("mode,before,after", [("prior_then_actor", "prior", "actor"),
                                               ("actor_then_prior", "actor", "prior")])
def test_bootstrap_switch_picks_the_right_latent_on_each_side(mode, before, after):
    ag = _perturbed_delta(_agent(**BEST, bootstrap_switch=mode, bootstrap_switch_step=10))
    b = _batch()
    parts = dict(zip(("actor", "prior"), _boot_parts(ag, b)))
    for step, want in ((0, before), (9, before), (10, after), (50, after)):
        got = ag.replace(basis=ag.basis.replace(step=step))._select_u_next(b, None)
        np.testing.assert_allclose(np.asarray(got), np.asarray(parts[want]), atol=1e-6,
                                   err_msg=f"{mode} step {step}")


def test_bootstrap_switch_updates_and_counts_steps():
    ag = _agent(**BEST, bootstrap_switch="prior_then_actor", bootstrap_switch_step=2)
    for step in range(3):
        ag, info = ag.update(_batch(seed=step))
        assert all(np.all(np.isfinite(np.asarray(v))) for v in info.values())
    # TrainState counts from 1, so before update i (main.py's step i) basis.step == i
    assert int(ag.basis.step) == 4


def test_bootstrap_switch_asserts_flowbc():
    with pytest.raises(AssertionError, match="bootstrap_switch"):
        _agent(**{**BEST, "actor_kind": "tanh", "actor_value_kind": "shaped"}, bootstrap_switch="prior_then_actor")


# ------------------------------------------------------------------ actor goal source

@pytest.mark.parametrize("source,key", [("future", "fb_goals"), ("measure", "goals")])
def test_actor_step_reads_the_selected_goal(source, key, monkeypatch):
    ag = _agent(**BEST, actor_goal_source=source)
    seen = {}
    orig = PSMGoalAgent.actor_loss_flowbc

    def spy(self, delta_params, obs, pool, goals, eps, u_data):
        seen["goals"] = goals
        return orig(self, delta_params, obs, pool, goals, eps, u_data)

    monkeypatch.setattr(PSMGoalAgent, "actor_loss_flowbc", spy)
    b = _batch()
    ag.goal_actor_step(b)
    np.testing.assert_array_equal(np.asarray(seen["goals"]), b[key])


def test_actor_goal_source_rejects_unknown():
    with pytest.raises(AssertionError, match="actor_goal_source"):
        _agent(**BEST, actor_goal_source="random")


# ------------------------------------------------------------------ flowbc_gpi acting

def _flowbc_gpi_agent():
    ag = _perturbed_delta(_agent(**{**BEST, "acting": "flowbc_gpi", "coef_source": "hgoal_each"}), bias=0.3)
    ag, _ = ag.update(_batch(seed=1))
    rb, rew = _relabel()
    return ag.infer_eval_goals(rb, rew), rb


def test_flowbc_gpi_equals_hand_computation():
    ag, rb = _flowbc_gpi_agent()
    o = jnp.asarray(rb["observations"][0])
    seed = jax.random.PRNGKey(4)
    K, u_clip = int(ag.config["gpi_num_u"]), ag.config["u_clip"]
    eps = jnp.clip(jax.random.normal(seed, (K, DA)), -u_clip, u_clip)
    obs = jnp.broadcast_to(o, (K, OB))
    g = ag.eval_goals[jnp.arange(K) % ag.eval_goals.shape[0]]
    u_cand = ag.flowbc_latent(obs, g, eps)[0]
    q = ag._hgoal_each_score(obs, u_cand, ag.eval_goals, ag.eval_goal_w, ag.eval_ref)
    want = u_cand[jnp.argmax(q)]
    got = ag.select_latent_flowbc_gpi(o, seed)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-6)
    # candidates differ from the plain prior draws (delta is non-zero)
    assert not np.allclose(np.asarray(u_cand), np.asarray(eps))
    a = ag.sample_actions(o, seed=seed)
    np.testing.assert_allclose(np.asarray(a), np.asarray(ag.decode(o[None], want[None])[0]), atol=1e-6)


def test_flowbc_gpi_asserts():
    with pytest.raises(AssertionError, match="flowbc_gpi"):
        _agent(**{**BEST, "acting": "flowbc_gpi", "coef_source": "amortized"})
    with pytest.raises(AssertionError, match="flowbc_gpi"):
        _agent(**{**BEST, "actor_kind": "tanh", "actor_value_kind": "shaped",
                  "acting": "flowbc_gpi", "coef_source": "hgoal_each"})
