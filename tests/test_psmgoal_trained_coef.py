"""psmgoal coef_source='trained': read M with the coefficient it was trained with.

Under policy_index=code the measure is trained at w(z) on sampled binary policy codes z. The
other test-time sources compute a different w (lp, regression). coef_source=trained sets
eval_w_star = project(mean_z w(z)) over TRAINED_COEF_CODES codes drawn from a fixed key. It
reads no reward and runs no Lagrangian or regression. Acting goes through `_goal_score`:
mean of M over the goal set (measure_loss=squared) or the softmax weight on the goal columns
over [goal columns + reference columns] (measure_loss=softmax).

Checks: the default stays lp in both configs; the coefficient against a hand computation for
both losses; it does not depend on the relabel batch; no inference loop is called; acting is
the argmax of `_goal_score`; policy_index=goal raises; the softmax loss trains under the code
label with no actor; a checkpoint trained with another coef_source restores and is read
with the trained coefficient.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_trained_coef.py -q -p no:cacheprovider
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
from utils.flax_utils import restore_agent, save_agent
from utils.psm_proto import sample_z_bin

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

GC = {"policy_index": "goal", "train_actor": True, "actor_input": "goal", "goal_cur_frac": 0.2,
      "goal_random_frac": 0.3}


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
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _relabel(n=32, seed=5, n_rewarding=6):
    b = _batch(n=n, seed=seed)
    rew = np.zeros((n,), np.float32)
    rew[:n_rewarding] = 1.0
    return b, rew


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _same(t0, t1):
    return all(jnp.array_equal(a, b) for a, b in zip(_leaves(t0), _leaves(t1)))


def _np_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _trained(n_steps=3, **over):
    ag = _agent(**over)
    for step in range(n_steps):
        ag, info = ag.update(_batch(seed=step))
    return ag, info


def _want_w(ag):
    """project(mean_z w(z)) over the fixed codes, written out by hand."""
    z = sample_z_bin(jax.random.PRNGKey(pg.TRAINED_COEF_SEED), pg.TRAINED_COEF_CODES, CODE)
    mean = np.asarray(ag.w(z)).mean(axis=0)
    return np.sqrt(Z) * mean / np.linalg.norm(mean)


# ------------------------------------------------------------------ defaults and validation

def test_default_coef_source_is_still_lp_and_trained_is_listed():
    assert get_config()["coef_source"] == "lp"
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        text = f.read()
    assert yaml.safe_load(text)["coef_source"] == "lp"
    assert "trained" in pg.COEF_SOURCES
    assert set(pg.COEF_SOURCES) == {"lp", "amortized", "regression", "ridge", "trained"}
    line = next(ln for ln in text.splitlines() if ln.startswith("coef_source:"))
    assert "trained" in line, "psmgoal.yaml must list trained beside the other coefficient sources"
    assert pg.TRAINED_COEF_CODES == 256


def test_unknown_coef_source_still_asserts():
    with pytest.raises(AssertionError):
        _agent(coef_source="mean")


def test_trained_coef_raises_under_the_goal_label():
    """policy_index=goal: coef_source=amortized already is the trained coefficient."""
    with pytest.raises(AssertionError, match="amortized"):
        _agent(**GC, coef_source="trained")
    # the method itself refuses too, if it is reached with a goal-labelled agent
    ag = _agent(**GC, coef_source="amortized", acting="gpi")
    with pytest.raises(ValueError, match="amortized"):
        ag.trained_coef()


# ------------------------------------------------------------------ the coefficient

@pytest.mark.parametrize("measure_loss", ["squared", "softmax"])
def test_trained_coef_is_projected_mean_of_w_over_fixed_codes(measure_loss):
    ag, _ = _trained(measure_loss=measure_loss, coef_source="trained")
    b, rew = _relabel()
    ev = ag.infer_eval_goals(b, rew)
    np.testing.assert_allclose(np.asarray(ev.eval_w_star), _want_w(ag), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(jnp.linalg.norm(ev.eval_w_star)), np.sqrt(Z), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(ev.eval_w_star), np.asarray(ag.trained_coef()))


def test_trained_coef_reads_the_online_w_net():
    """It is w(z) at the online params: changing them changes the coefficient."""
    ag, _ = _trained(coef_source="trained")
    moved = ag.replace(w=ag.w.replace(params=jax.tree_util.tree_map(lambda p: p + 0.3, ag.w.params)))
    assert not np.allclose(np.asarray(ag.trained_coef()), np.asarray(moved.trained_coef()))
    np.testing.assert_allclose(np.asarray(moved.trained_coef()), _want_w(moved), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("measure_loss", ["squared", "softmax"])
def test_trained_coef_ignores_the_relabel_batch_and_runs_no_inference(measure_loss, monkeypatch):
    ag, _ = _trained(measure_loss=measure_loss, coef_source="trained", num_inference_steps=20000)

    def _boom(*a, **k):
        raise AssertionError("coef_source=trained must not run a coefficient inference")

    monkeypatch.setattr(PSMGoalAgent, "run_inference", _boom)
    monkeypatch.setattr(PSMGoalAgent, "run_inference_cached", _boom)
    monkeypatch.setattr(jnp.linalg, "lstsq", _boom)
    b1, rew1 = _relabel(seed=5)
    b2, rew2 = _relabel(seed=9, n_rewarding=11)
    ev1 = ag.infer_eval_goals(b1, rew1)
    ev2 = ag.infer_eval_goals(b2, rew2)
    np.testing.assert_array_equal(np.asarray(ev1.eval_w_star), np.asarray(ev2.eval_w_star))
    # the goal set still comes from the relabel batch's rewarding rows
    assert all(any(np.array_equal(g, x) for x in b1["next_observations"][:6])
               for g in np.asarray(ev1.eval_goals))


# ------------------------------------------------------------------ acting

def test_squared_acting_is_argmax_of_mean_M_at_the_trained_coef():
    ag, _ = _trained(coef_source="trained", acting="gpi")
    b, rew = _relabel()
    ev = ag.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    seed = jax.random.PRNGKey(4)
    K = int(ev.config["gpi_num_u"])
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    w = jnp.broadcast_to(jnp.asarray(_want_w(ag), jnp.float32), (K, Z))
    want = np.asarray(ev.mesh_M(obs, u_cand, ev.eval_goals, w)).mean(axis=1)
    got = ev._goal_score(obs, u_cand, ev.eval_goals, ev.eval_ref,
                         jnp.broadcast_to(ev.eval_w_star, (K, Z)))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(ev.select_latent(obs0, seed), u_cand[int(np.argmax(got))])
    a = ev.sample_actions(obs0, seed=seed)
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


def test_softmax_acting_is_argmax_of_softmax_weight_on_goal_columns():
    ag, _ = _trained(measure_loss="softmax", measure_temp=0.7, coef_source="trained", acting="gpi")
    b, rew = _relabel()
    ev = ag.infer_eval_goals(b, rew)
    refs = np.asarray(ev.eval_ref)
    assert refs.shape == (N - 4, OB)                       # batch_size - k_goals reference columns
    assert all(any(np.array_equal(r, x) for x in b["next_observations"][6:]) for r in refs)
    obs0 = jnp.asarray(b["observations"][0])
    seed = jax.random.PRNGKey(4)
    K = int(ev.config["gpi_num_u"])
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    w = jnp.broadcast_to(jnp.asarray(_want_w(ag), jnp.float32), (K, Z))
    cols = jnp.concatenate([ev.eval_goals, ev.eval_ref], axis=0)
    p = _np_softmax(np.asarray(ev.mesh_M(obs, u_cand, cols, w)) / 0.7)
    want = p[:, :ev.eval_goals.shape[0]].sum(axis=1)
    got = ev._goal_score(obs, u_cand, ev.eval_goals, ev.eval_ref,
                         jnp.broadcast_to(ev.eval_w_star, (K, Z)))
    np.testing.assert_allclose(got, want, rtol=1e-4)
    np.testing.assert_array_equal(ev.select_latent(obs0, seed), u_cand[int(np.argmax(got))])
    a = ev.sample_actions(obs0, seed=seed)
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


# ------------------------------------------------------------------ the training arm

def test_softmax_loss_trains_under_the_code_label_with_no_actor():
    """measure_loss=softmax with policy_index=code and train_actor=false: `create` accepts it,
    the bootstrap is the fixed prior latent per (z, row), w(z) is stepped, h and the actor
    are not."""
    ag0 = _agent(measure_loss="softmax")
    assert ag0.config["policy_index"] == "code" and not ag0.config["train_actor"]
    b = _batch(seed=0)
    z = sample_z_bin(jax.random.PRNGKey(1), N, CODE)
    np.testing.assert_array_equal(np.asarray(ag0._select_u_next(b, z)),
                                  np.asarray(ag0.proto_bootstrap(z, jnp.asarray(b["index"]))))
    ag, info = ag0.update(b)
    for k in ("psm_loss", "m_softmax_diag", "m_softmax_max", "m_logit_spread", "phi_eff_rank", "w_norm"):
        assert np.isfinite(float(info[k])), k
    assert not _same(ag0.w.params, ag.w.params), "w(z) must be stepped"
    assert not _same(ag0.basis.params, ag.basis.params), "phi, b must be stepped"
    assert _same(ag0.actor.params, ag.actor.params), "no actor is trained"
    assert _same(ag0.w_star.params, ag.w_star.params), "h is not trained under the code label"
    # the loss is softmax_td_loss on the mesh at w(z) (online) and the Polyak w(z) (target)
    u_next = ag0._select_u_next(b, z)
    loss, _ = ag0.measure_loss(ag0.basis.params, ag0.w.params, b, z, u_next)
    obs, nxt = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    M = ag0.mesh_M(obs, u, nxt, ag0.w(z))
    M_bar = ag0.mesh_M(nxt, u_next, nxt, ag0.w(z, params=ag0.target_w), params=ag0.target_basis)
    want = pg.softmax_td_loss(M, M_bar, ag0.config["discount"], float(ag0.config["measure_temp"]))
    np.testing.assert_allclose(float(loss), float(want), rtol=1e-5)


# ------------------------------------------------------------------ restore

@pytest.mark.parametrize("train_over", [
    {"measure_loss": "softmax"},                                       # the new arm
    {"train_goal_head": True, "coef_source": "amortized"},             # psmgoal_lift_cube_gc
])
def test_checkpoint_restores_and_is_read_with_the_trained_coef(train_over, tmp_path):
    trained, _ = _trained(n_steps=2, **train_over)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_agent(trained, str(run_dir), 2)
    ev = restore_agent(_agent(seed=3, **{**train_over, "coef_source": "trained", "acting": "gpi"}),
                       str(run_dir), 2)
    assert _same(ev.w.params, trained.w.params), "the trained w(z) must be restored"
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    np.testing.assert_allclose(np.asarray(ev.eval_w_star), _want_w(trained), rtol=1e-5, atol=1e-6)
    a = ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))
