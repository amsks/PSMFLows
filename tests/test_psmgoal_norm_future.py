"""psmgoal `norm_future_k` (2026-10-15): the softmax measure loss normalises over the batch's
next states plus K later states of each row's own trajectory (batch['future_states']).

Checks: the key defaults to 0 in get_config and the yaml; K = 0 gives the same loss, info and
update as the key being absent (bit-identical to the pre-change path); the dataset serves
future_states of shape (N, K, ob) that never leave the row's trajectory and are absent at
K = 0; with K = 2 the softmax rows sum to one over N + N*K columns, the target mass on the
row's own next state is (1 - gamma) + gamma * (bootstrap weight) by hand on a tiny mesh and
the loss is the hand cross-entropy; the three logging keys exist at K = 0 and K = 2; the
squared loss refuses K > 0.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_norm_future.py -q -p no:cacheprovider
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
from utils.datasets import Dataset, future_state_idxs, trajectory_ends

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

# The current best arm (psmgoal_ja_fbc3_pt_cube) minus the env-only keys.
FBC3_PT = {"policy_index": "goal", "bootstrap_source": "actor", "train_actor": True, "actor_input": "goal",
           "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "goal_sampling": "geometric",
           "measure_loss": "softmax", "acting": "distill", "eval_redistill": False,
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
    for k in ("measure", "w", "l", "actor", "w_star"):
        getattr(c, k).hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    c = _cfg(**over)
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), c)


def _batch(n=N, seed=0, k=0):
    rng = np.random.default_rng(seed)
    b = {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
        "fb_goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }
    if k > 0:
        b["future_states"] = rng.standard_normal((n, k, OB)).astype(np.float32)
    return b


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def _np_softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _code(seed):
    return pg.sample_z_bin(jax.random.PRNGKey(seed), N, CODE)


# ------------------------------------------------------------------ defaults

def test_norm_future_k_defaults_to_zero_in_get_config_and_yaml():
    assert get_config().norm_future_k == 0
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")) as f:
        y = yaml.safe_load(f)
    assert y["norm_future_k"] == 0


# ------------------------------------------------------------------ (i) K = 0 is the old path

@pytest.mark.parametrize("arm", ["fbc3_pt", "default"])
def test_k_zero_is_bit_identical_to_key_absent(arm):
    over = FBC3_PT if arm == "fbc3_pt" else {}
    c_absent = _cfg(**over)
    del c_absent["norm_future_k"]
    assert "norm_future_k" not in c_absent
    c_zero = _cfg(**over, norm_future_k=0)
    a0 = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), c_absent)
    a1 = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), c_zero)
    for i in range(2):
        a0, i0 = a0.update(_batch(seed=i))
        a1, i1 = a1.update(_batch(seed=i))
    for x, y in zip(_leaves((a0.basis.params, a0.w.params, a0.w_star.params, a0.target_basis)),
                    _leaves((a1.basis.params, a1.w.params, a1.w_star.params, a1.target_basis))):
        np.testing.assert_array_equal(x, y)
    for k in set(i0) | set(i1):
        assert float(i0[k]) == float(i1[k]), k
    # the key absent and 0 both run the (N, N) mesh: the softmax loss is softmax_td_loss on it
    b = _batch(seed=3)
    z = _code(0)
    u_next = a0._select_u_next(b, z)
    coef = a0.w_star.params if a0._goal_index() else a0.w.params
    l0, _ = a0.measure_loss(a0.basis.params, coef, b, z, u_next)
    l1, _ = a1.measure_loss(a1.basis.params, coef, b, z, u_next)
    assert float(l0) == float(l1)


def test_k_zero_softmax_loss_matches_softmax_td_on_the_square_mesh():
    ag = _agent(**FBC3_PT, norm_future_k=0, measure_temp=0.7)
    ag, _ = ag.update(_batch(seed=1))
    b = _batch()
    u_next = ag._select_u_next(b, _code(0))
    loss, info = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    M = ag.mesh_M(obs, u, nobs, ag.w_star(g))
    Mbar = ag.mesh_M(nobs, u_next, nobs, ag.w_star(g, params=ag.target_w_star), params=ag.target_basis)
    assert M.shape == (N, N)
    np.testing.assert_allclose(float(loss), float(pg.softmax_td_loss(M, Mbar, 0.98, 0.7)), rtol=1e-5)
    assert float(info["measure_p_future"]) == 0.0


# ------------------------------------------------------------------ (ii) the sampler

def _dataset(n=40, seed=0):
    rng = np.random.default_rng(seed)
    terminals = np.zeros((n,), np.float32)
    terminals[[9, 19, 29, 39]] = 1.0
    # next_observations[:, 0] = the row index, so a served state names its row
    nobs = rng.standard_normal((n, OB)).astype(np.float32)
    nobs[:, 0] = np.arange(n)
    ds = Dataset.create(observations=rng.standard_normal((n, OB)).astype(np.float32),
                        next_observations=nobs,
                        actions=rng.standard_normal((n, DA)).astype(np.float32),
                        terminals=terminals)
    return ds


def test_future_state_idxs_stay_in_the_trajectory_and_after_the_row():
    ds = _dataset()
    idxs = np.arange(ds.size)
    ends = trajectory_ends(idxs, ds.terminal_locs, ds.size)
    np.testing.assert_array_equal(ends, np.repeat([9, 19, 29, 39], 10))
    f = future_state_idxs(idxs, ds.terminal_locs, ds.size, 0.5, 3, rng=np.random.default_rng(1))
    assert f.shape == (ds.size, 3)
    assert np.all(f >= idxs[:, None]) and np.all(f <= ends[:, None])
    # a row at the trajectory end has only itself
    assert np.all(f[[9, 19, 29, 39]] == np.array([[9], [19], [29], [39]]))
    # offsets are geometric from 1: a low discount keeps most draws at the row itself
    assert (f == idxs[:, None]).mean() > 0.3


def test_dataset_serves_future_states_of_the_right_shape_inside_the_trajectory():
    ds = _dataset()
    ds.norm_future_k = 2
    ds.goal_discount = 0.9
    b = ds.sample(64)
    assert b["future_states"].shape == (64, 2, OB)
    rows = b["next_observations"][:, 0].astype(np.int64)
    ends = trajectory_ends(rows, ds.terminal_locs, ds.size)
    served = b["future_states"][:, :, 0].astype(np.int64)
    assert np.all(served >= rows[:, None]) and np.all(served <= ends[:, None])
    # every served state is the stored next_observations row it names
    np.testing.assert_array_equal(b["future_states"], ds["next_observations"][served])
    # the row's own s' is among the candidates at offset 1
    b2 = ds.sample(2000)
    own = (b2["future_states"][:, :, 0] == b2["next_observations"][:, :1]).mean()
    assert 0.02 < own < 0.4


def test_dataset_emits_no_future_states_at_k_zero_and_leaves_the_stream_alone():
    ds = _dataset()
    ds.return_goals = True
    ds.goal_random_frac = 0.0
    np.random.seed(7)
    b0 = ds.sample(16)
    assert "future_states" not in b0
    ds.norm_future_k = 0
    np.random.seed(7)
    b1 = ds.sample(16)
    np.testing.assert_array_equal(b0["goals"], b1["goals"])
    np.testing.assert_array_equal(b0["observations"], b1["observations"])
    # K > 0 draws its futures last: every other field is unchanged for the same seed
    ds.norm_future_k = 2
    np.random.seed(7)
    b2 = ds.sample(16)
    np.testing.assert_array_equal(b0["goals"], b2["goals"])
    np.testing.assert_array_equal(b0["observations"], b2["observations"])
    assert b2["future_states"].shape == (16, 2, OB)


# ------------------------------------------------------------------ (iii) the loss at K = 2

def test_softmax_td_loss_target_on_a_rectangular_mesh():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 12)).astype(np.float32)
    Mbar = rng.standard_normal((4, 12)).astype(np.float32)
    gamma, temp = 0.9, 0.5
    pbar = _np_softmax(Mbar / temp)
    target = (1 - gamma) * np.eye(4, 12) + gamma * pbar
    np.testing.assert_allclose(target.sum(1), 1.0, rtol=1e-6)
    for i in range(4):
        np.testing.assert_allclose(target[i, i], (1 - gamma) + gamma * pbar[i, i], rtol=1e-6)
    logp = np.log(_np_softmax(M / temp))
    np.testing.assert_allclose(float(pg.softmax_td_loss(M, Mbar, gamma, temp)),
                               -(target * logp).sum(1).mean(), rtol=1e-5)


def test_k_two_loss_normalises_over_next_and_future_states_with_the_hand_target():
    K, gamma, temp = 2, 0.98, 0.7
    ag = _agent(**FBC3_PT, norm_future_k=K, measure_temp=temp)
    ag, _ = ag.update(_batch(seed=1, k=K))
    b = _batch(seed=2, k=K)
    u_next = ag._select_u_next(b, _code(0))
    loss, info = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    cols = jnp.concatenate([nobs, jnp.asarray(b["future_states"]).reshape(-1, OB)], axis=0)
    assert cols.shape == (N + N * K, OB)
    M = np.asarray(ag.mesh_M(obs, u, cols, ag.w_star(g)))
    Mbar = np.asarray(ag.mesh_M(nobs, u_next, cols, ag.w_star(g, params=ag.target_w_star), params=ag.target_basis))
    p, pbar = _np_softmax(M / temp), _np_softmax(Mbar / temp)
    np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-5)
    target = (1 - gamma) * np.eye(N, N + N * K) + gamma * pbar
    np.testing.assert_allclose(target.sum(1), 1.0, rtol=1e-5)
    for i in range(N):
        np.testing.assert_allclose(target[i, i], (1 - gamma) + gamma * pbar[i, i], rtol=1e-6)
        assert target[i, i] > target[i, (i + 1) % N]
    np.testing.assert_allclose(float(loss), -(target * np.log(p)).sum(1).mean(), rtol=1e-4)
    # the logged own-next and own-future masses are the diagonal and the row's K future columns
    np.testing.assert_allclose(float(info["measure_p_own_next"]), np.diagonal(p).mean(), rtol=1e-4)
    own_future = np.stack([p[i, N + i * K: N + (i + 1) * K].sum() for i in range(N)])
    np.testing.assert_allclose(float(info["measure_p_future"]), own_future.mean(), rtol=1e-4)
    # a column order change (futures not first) is a different loss: the futures are not
    # interchangeable with the next states, column i must be the row's own s'
    assert not np.allclose(float(loss), float(pg.softmax_td_loss(M[:, ::-1], Mbar[:, ::-1], gamma, temp)))


def test_k_two_update_is_finite_and_moves_the_measure():
    ag = _agent(**FBC3_PT, norm_future_k=2)
    before = _leaves(ag.basis.params)
    ag, info = ag.update(_batch(seed=4, k=2))
    assert all(np.isfinite(float(v)) for v in info.values())
    assert any(not np.array_equal(x, y) for x, y in zip(before, _leaves(ag.basis.params)))


# ------------------------------------------------------------------ (iv) logging and guards

@pytest.mark.parametrize("k", [0, 2])
def test_logging_keys_exist(k):
    ag = _agent(**FBC3_PT, norm_future_k=k)
    _, info = ag.update(_batch(seed=0, k=k))
    for key in ("measure_p_own_next", "measure_p_future", "phi_eff_rank"):
        assert key in info, key
    assert 0.0 <= float(info["measure_p_own_next"]) <= 1.0
    assert 0.0 <= float(info["measure_p_future"]) <= 1.0
    assert float(info["phi_eff_rank"]) >= 1.0
    if k == 0:
        assert float(info["measure_p_future"]) == 0.0


def test_squared_loss_refuses_norm_future_k():
    with pytest.raises(AssertionError, match="measure_loss=softmax"):
        _agent(norm_future_k=2)


def test_missing_future_states_is_an_error_at_k_two():
    ag = _agent(**FBC3_PT, norm_future_k=2)
    with pytest.raises(AssertionError, match="future_states"):
        ag.update(_batch(seed=0, k=0))
