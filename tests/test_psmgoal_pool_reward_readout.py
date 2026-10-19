"""coef_source=pool_reward (docs/design/2026-10-19-pool-measure.md): value by integrating the reward
over an importance-weighted bank of relabel next states, max over the candidate coefficients; and the
eval-side pool rebuild with the pool_hash check (tools/eval_checkpoint.py). CPU, allow_untrained_flow.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal_pool_reward_readout.py -q
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import utils.xla_guard  # noqa: F401  (must precede jax)
from agents.psmgoal import REWARDING_THRESHOLD, get_config, pool_reward_bank_rows
from tests.test_psmgoal_bootstrap_delta_scale import OVER
from tests.test_psmgoal_joint_actor import DA, OB, _agent, _batch
from tests.test_psmgoal_pool_loss import _pool_agent
from tools.eval_checkpoint import LEGACY_AGENT_DEFAULTS, build_eval_pool, merge_run_config
from utils.flow_inversion import PREIMAGE_VALID_KEY
from utils.psm_common import draw_pool_indices, pool_hash

B_EVAL = 16
BANK = {"policy_index": "bank", "train_actor": False, "measure_loss": "softmax",
        "coef_source": "pool_reward", "pool_eval_size": B_EVAL}
GOAL = {**OVER, "coef_source": "pool_reward", "pool_eval_size": B_EVAL}


def _relabel(n=64, seed=5, n_rewarding=6):
    b = _batch(n=n, seed=seed)
    rew = np.zeros((n,), np.float32)
    rew[:n_rewarding] = 1.0
    return b, rew


# ------------------------------------------------------------------ defaults

def test_pool_eval_size_defaults_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    assert c["pool_eval_size"] == 512 and y["pool_eval_size"] == 512
    assert c["coef_source"] == "lp" and y["coef_source"] == "lp"


# ------------------------------------------------------------------ the bank draw

def test_pool_reward_weights_reproduce_class_counts():
    rew = np.zeros((200,), np.float32)
    rew[:30] = 1.0
    np.random.seed(0)
    rows, omega = pool_reward_bank_rows(rew, 64)
    assert rows.shape == (64,) and omega.shape == (64,)
    is_r = rew[rows] > REWARDING_THRESHOLD
    assert is_r.sum() == 16                              # n_r = min(64 // 4, 30)
    np.testing.assert_allclose(omega[is_r].sum(), 30.0, rtol=1e-6)      # class count in the relabel batch
    np.testing.assert_allclose(omega[~is_r].sum(), 170.0, rtol=1e-6)
    np.testing.assert_allclose(omega[is_r], 30.0 / 16, rtol=1e-6)
    np.testing.assert_allclose(omega[~is_r], 170.0 / 48, rtol=1e-6)
    assert len(set(rows[~is_r].tolist())) == 48          # no replacement when the class is big enough
    # fewer rewarding rows than a quarter of the bank: all of them, once
    rew2 = np.zeros((200,), np.float32)
    rew2[:3] = 1.0
    rows2, omega2 = pool_reward_bank_rows(rew2, 64)
    is_r2 = rew2[rows2] > REWARDING_THRESHOLD
    assert is_r2.sum() == 3 and sorted(rows2[is_r2].tolist()) == [0, 1, 2]
    np.testing.assert_allclose(omega2[is_r2], 1.0, rtol=1e-6)
    np.testing.assert_allclose(omega2[~is_r2].sum(), 197.0, rtol=1e-6)


def test_pool_reward_bank_with_no_rewarding_rows_is_all_non_rewarding_with_positive_weights():
    rew = np.zeros((40,), np.float32)
    rows, omega = pool_reward_bank_rows(rew, 16)
    assert rows.shape == (16,) and np.all(rew[rows] <= REWARDING_THRESHOLD)
    assert np.all(omega > 0) and np.isclose(omega.sum(), 40.0)


# ------------------------------------------------------------------ the score

def test_pool_reward_zero_rewarding_rows_gives_finite_zero():
    ag = _agent(**BANK)
    rb, rew = _relabel(n_rewarding=0)
    ag = ag.infer_eval_goals(rb, rew)                   # no goal set is needed under the bank
    assert ag.eval_bank.shape == (B_EVAL, OB) and np.all(np.asarray(ag.eval_bank_r) == 0.0)
    assert np.all(np.asarray(ag.eval_bank_w) > 0.0)
    obs = jnp.asarray(np.broadcast_to(rb["observations"][0], (5, OB)))
    u = jnp.asarray(np.random.default_rng(0).standard_normal((5, DA)).astype(np.float32))
    q = np.asarray(ag._pool_reward_score(obs, u, ag.eval_cand_w, ag.eval_bank, ag.eval_bank_r, ag.eval_bank_w))
    assert q.shape == (5,) and np.all(np.isfinite(q)) and np.all(q == 0.0)
    a = ag.sample_actions(jnp.asarray(rb["observations"][0]), seed=jax.random.PRNGKey(1))
    assert np.all(np.isfinite(np.asarray(a)))


def test_pool_reward_equals_plain_mean_when_weights_are_uniform_and_r_is_indicator():
    """omega = 1: Q(s, u; w_c) = sum_j softmax_j(M(s,u,y_j; w_c)/t) r_j, the reward-weighted shares, and the
    score is the max over the candidate coefficients."""
    ag = _agent(**GOAL)
    rng = np.random.default_rng(7)
    bank = jnp.asarray(rng.standard_normal((B_EVAL, OB)).astype(np.float32))
    r = jnp.asarray((rng.random(B_EVAL) < 0.3).astype(np.float32))
    cand_w = ag.w_star(jnp.asarray(rng.standard_normal((4, OB)).astype(np.float32)))      # (4, z)
    obs = jnp.asarray(rng.standard_normal((5, OB)).astype(np.float32))
    u = jnp.asarray(rng.standard_normal((5, DA)).astype(np.float32))
    q = np.asarray(ag._pool_reward_score(obs, u, cand_w, bank, r, jnp.ones((B_EVAL,))))
    t = float(ag.config["measure_temp"])
    expect = np.zeros((5, 4))
    for c in range(4):
        w_rows = jnp.broadcast_to(cand_w[c], (5, cand_w.shape[1]))
        m = np.asarray(ag.mesh_M(obs, u, bank, w_rows))                                    # (5, B)
        p = np.exp(m / t - m.max(1, keepdims=True) / t)
        p = p / p.sum(1, keepdims=True)
        expect[:, c] = (p * np.asarray(r)).sum(1)
    np.testing.assert_allclose(q, expect.max(1), rtol=1e-5, atol=1e-6)
    # weights scale the shares: doubling every weight changes nothing, weighting one row to zero removes it
    q2 = np.asarray(ag._pool_reward_score(obs, u, cand_w, bank, r, 2.0 * jnp.ones((B_EVAL,))))
    np.testing.assert_allclose(q2, q, rtol=1e-5, atol=1e-6)


def test_pool_reward_select_latent_runs_under_goal_and_bank():
    for over in (GOAL, BANK):
        ag = _agent(**over)
        rb, rew = _relabel()
        ag = ag.infer_eval_goals(rb, rew)
        assert ag.eval_bank.shape == (B_EVAL, OB)
        assert int((np.asarray(ag.eval_bank_r) > REWARDING_THRESHOLD).sum()) == min(B_EVAL // 4, 6)
        assert ag.eval_cand_w.shape == ((int(ag.config["k_goals"]), ag.eval_cand_w.shape[1]) if over is GOAL
                                        else (1, ag.eval_cand_w.shape[1]))
        o = jnp.asarray(rb["observations"][0])
        a1 = ag.sample_actions(o, seed=jax.random.PRNGKey(1))
        assert np.all(np.isfinite(np.asarray(a1)))
        np.testing.assert_allclose(np.asarray(a1), np.asarray(ag.sample_actions(o, seed=jax.random.PRNGKey(1))))


def test_pool_reward_on_a_pool_critic_runs():
    pool_obs = np.random.default_rng(9).standard_normal((16, OB)).astype(np.float32)
    ag = _pool_agent(pool_obs, policy_index="bank", train_actor=False, coef_source="pool_reward",
                     pool_eval_size=B_EVAL)
    ag, _ = ag.update(_batch())
    rb, rew = _relabel()
    ag = ag.infer_eval_goals(rb, rew)
    a = ag.sample_actions(jnp.asarray(rb["observations"][0]), seed=jax.random.PRNGKey(2))
    assert np.all(np.isfinite(np.asarray(a)))


@pytest.mark.parametrize("over", [
    {"coef_source": "pool_reward"},                               # policy_index=code has no w to read
    {**BANK, "pool_eval_size": 0},
    {**GOAL, "eval_goal_pool": "dataset"},
])
def test_pool_reward_asserts(over):
    with pytest.raises(AssertionError):
        _agent(**over)


# ------------------------------------------------------------------ eval: pool rebuild and hash

def _eval_dataset(n=100, n_bad=3):
    rng = np.random.default_rng(0)
    valid = np.ones((n,), np.float32)
    valid[:n_bad] = 0.0
    return {"observations": rng.standard_normal((n, OB)).astype(np.float32),
            "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
            PREIMAGE_VALID_KEY: valid}


def test_eval_pool_rebuild_matches_the_training_draw_and_skips_invalid_rows():
    ds = _eval_dataset()
    rows = draw_pool_indices(np.arange(3, 100), 8, 11)
    cfg = {"pool_size": 8, "pool_seed": 11, "pool_hash": pool_hash(rows)}
    pool_obs, pool_rows = build_eval_pool(ds, cfg)
    np.testing.assert_array_equal(pool_rows, rows)
    np.testing.assert_array_equal(pool_obs, ds["next_observations"][rows])
    assert np.all(pool_rows >= 3)
    # no recorded hash (an in-loop smoke, or a flags.json without the key): accepted
    cfg_none = {"pool_size": 8, "pool_seed": 11, "pool_hash": None}
    np.testing.assert_array_equal(build_eval_pool(ds, cfg_none)[1], rows)
    # without a validity array every row is valid
    ds2 = {k: v for k, v in ds.items() if k != PREIMAGE_VALID_KEY}
    np.testing.assert_array_equal(build_eval_pool(ds2, cfg_none)[1], draw_pool_indices(np.arange(100), 8, 11))


def test_eval_hash_mismatch_raises():
    ds = _eval_dataset()
    rows = draw_pool_indices(np.arange(3, 100), 8, 11)
    with pytest.raises(AssertionError):
        build_eval_pool(ds, {"pool_size": 8, "pool_seed": 11, "pool_hash": pool_hash(rows + 1)})
    with pytest.raises(AssertionError):      # another seed than the run's
        build_eval_pool(ds, {"pool_size": 8, "pool_seed": 12, "pool_hash": pool_hash(rows)})


def test_legacy_defaults_for_psmgoal_pool_keys(tmp_path):
    legacy = LEGACY_AGENT_DEFAULTS["psmgoal"]
    assert legacy == {"measure_loss": "squared", "pool_size": 0, "pool_seed": 0, "n_bank": 1}
    cli = {"agent_name": "psmgoal", "measure_loss": "pool", "pool_size": 1024, "pool_seed": 3, "n_bank": 1,
           "pool_hash": None, "u_clip": 3.0}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "flags.json", "w") as f:
        json.dump({"seed": 1, "agent": {"agent_name": "psmgoal", "u_clip": 1.0}}, f)
    merged, prov = merge_run_config(cli, str(run_dir), cli_keys=set())
    assert merged["measure_loss"] == "squared" and merged["pool_size"] == 0 and merged["pool_seed"] == 0
    assert merged["n_bank"] == 1 and merged["u_clip"] == 1.0 and merged["pool_hash"] is None
    assert set(prov["legacy_defaults"]) == {"measure_loss", "pool_size", "pool_seed"}
    # a run that recorded the keys keeps them
    with open(run_dir / "flags.json", "w") as f:
        json.dump({"seed": 1, "agent": {"agent_name": "psmgoal", "measure_loss": "pool", "pool_size": 512,
                                        "pool_seed": 2, "n_bank": 1, "pool_hash": "ab" * 20}}, f)
    merged, prov = merge_run_config(cli, str(run_dir), cli_keys=set())
    assert merged["measure_loss"] == "pool" and merged["pool_size"] == 512 and merged["pool_seed"] == 2
    assert merged["pool_hash"] == "ab" * 20 and prov["legacy_defaults"] == {}
