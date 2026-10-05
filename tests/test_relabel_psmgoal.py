"""psmgoal reward-relabel tool: the goal-set readout, the scale-to-real map, the npz writer,
and a guard that the psmflow relabel path is byte-identical to HEAD.

CPU, tiny, no checkpoint. Run per-file only (the whole suite SIGABRTs in XLA):
  JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_relabel_psmgoal.py -q -p no:cacheprovider
"""
import json
import os
import subprocess
import sys

import jax.numpy as jnp  # tests/conftest.py imports utils.xla_guard before any jax import
import numpy as np
import pytest

from agents import agents
from agents.psmgoal import get_config
from tools import relabel_reward_rhat_psmgoal as R
from tools.relabel_reward_rhat import scale_to_real


def _tiny_agent(seed=0, ob_dim=4, action_dim=2, z_dim=8, k_goals=3, n_ex=6):
    """A synthetic psmgoal agent with an untrained flow and small nets."""
    cfg = get_config()
    with cfg.unlocked():
        cfg.z_dim = z_dim
        cfg.k_goals = k_goals
        cfg.max_log_seed = 3
        cfg.gpi_num_u = 4
        cfg.measure = {"hidden_dim": 16, "hidden_layers": 1}
        cfg.w = {"hidden_dim": 16, "hidden_layers": 1}
        cfg.l = {"hidden_dim": 16, "hidden_layers": 1}
        cfg.actor = {"hidden_dim": 16, "hidden_layers": 1}
        cfg.w_star = {"hidden_dim": 16, "hidden_layers": 1}
        cfg.flow = {"hidden_dims": (16, 16), "value_hidden_dims": (16, 16),
                    "layer_norm": False, "critic_layer_norm": True}
        cfg.allow_untrained_flow = True
    ex_obs = np.zeros((n_ex, ob_dim), np.float32)
    ex_act = np.zeros((n_ex, action_dim), np.float32)
    agent = agents["psmgoal"].create(seed, ex_obs, ex_act, cfg)
    # Set an eval goal set and coefficient by hand (infer_eval_goals needs a rewarding batch;
    # here we just want the readout math, so plant them directly).
    rng = np.random.default_rng(0)
    goals = jnp.asarray(rng.standard_normal((k_goals, ob_dim)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((z_dim,)), jnp.float32)
    agent = agent.replace(eval_goals=goals, eval_w_star=w)
    return agent


def test_rhat_goal_set_equals_mean_over_goals():
    """The goal-set readout is exactly mean over the goal axis of mesh_M."""
    agent = _tiny_agent()
    rng = np.random.default_rng(1)
    n, ob_dim, d_a = 5, 4, 2
    next_obs = rng.standard_normal((n, ob_dim)).astype(np.float32)
    u_rows = rng.standard_normal((n, d_a)).astype(np.float32)

    r_hat = R.rhat_goal_set(agent, next_obs, u_rows, agent.eval_goals, agent.eval_w_star,
                            batch_size=2)

    z_dim = agent.eval_w_star.shape[0]
    w_rows = jnp.broadcast_to(agent.eval_w_star, (n, z_dim))
    direct = np.asarray(
        agent.mesh_M(jnp.asarray(next_obs), jnp.asarray(u_rows), agent.eval_goals, w_rows).mean(1))

    assert r_hat.shape == (n,)
    assert np.isfinite(r_hat).all()
    np.testing.assert_allclose(r_hat, direct, rtol=1e-5, atol=1e-5)


def test_rhat_goal_set_shape_matches_rows():
    """One finite value per transition, shape == dataset rows."""
    agent = _tiny_agent()
    rng = np.random.default_rng(2)
    n = 13
    next_obs = rng.standard_normal((n, 4)).astype(np.float32)
    u_rows = rng.standard_normal((n, 2)).astype(np.float32)
    r_hat = R.rhat_goal_set(agent, next_obs, u_rows, agent.eval_goals, agent.eval_w_star,
                            batch_size=4)
    assert r_hat.shape == (n,)
    assert r_hat.dtype == np.float32
    assert np.isfinite(r_hat).all()


def test_select_preimage_u_point_and_mixture():
    """Point mode returns the stored point; mixture mode the weighted component mean; both
    clipped to +-u_clip."""
    n, d_a, k = 7, 2, 2
    rng = np.random.default_rng(3)
    point = (rng.standard_normal((n, d_a)) * 5.0).astype(np.float32)
    means = rng.standard_normal((n, k, d_a)).astype(np.float32)
    weights = np.abs(rng.standard_normal((n, k))).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    aug = {"observations": np.zeros((n, 4), np.float32),
           "noise_preimage_point": point,
           "noise_preimage_mean": means,
           "noise_preimage_weights": weights}

    u_clip = 3.0
    up = R.select_preimage_u(aug, use_point_preimage=True, u_clip=u_clip)
    np.testing.assert_allclose(up, np.clip(point, -u_clip, u_clip), atol=1e-6)
    assert np.abs(up).max() <= u_clip + 1e-6

    um = R.select_preimage_u(aug, use_point_preimage=False, u_clip=u_clip)
    mix_mean = (weights[..., None] * means).sum(1)
    np.testing.assert_allclose(um, np.clip(mix_mean, -u_clip, u_clip), atol=1e-6)


def test_match_real_scale_maps_to_minus1_0():
    """scale_to_real (reused, agent-agnostic) maps r_hat onto the -1/0 convention: predicted
    successes near 0, typical rows near -1."""
    rng = np.random.default_rng(4)
    n = 400
    real = np.where(rng.random(n) < 0.1, 0.0, -1.0).astype(np.float32)  # cube -1/0
    # r_hat correlated with the real reward but on an arbitrary scale/offset.
    r_hat = (3.0 * (real + 1.0) + 7.0 + 0.05 * rng.standard_normal(n)).astype(np.float32)
    r_out, _, _ = scale_to_real(r_hat, real, reward_shift=1.0)
    assert r_out.max() <= 0.05
    assert r_out.min() >= -1.2
    # rows whose real reward is 0 should map near 0, rows at -1 near -1.
    assert r_out[real > -0.5].mean() > -0.2
    assert r_out[real < -0.5].mean() < -0.8


def test_write_reward_override_keys(tmp_path):
    """The npz carries `rewards` (float32, N) and a sibling meta.json with the expected keys."""
    out = str(tmp_path / "r.npz")
    r_out = np.linspace(-1.0, 0.0, 20).astype(np.float32)
    meta = {"kind": "reward_override", "readout": "goal_set", "env_name": "cube",
            "num_rows": 20, "output": {"kind": "scaled"}}
    R.write_reward_override(out, r_out, meta)

    with np.load(out) as z:
        assert "rewards" in z.files
        assert z["rewards"].shape == (20,)
        assert z["rewards"].dtype == np.float32
    with open(out + ".meta.json") as fh:
        m = json.load(fh)
    for k in ("kind", "readout", "env_name", "num_rows", "output"):
        assert k in m
    assert m["kind"] == "reward_override"


def test_psmflow_relabel_path_unchanged():
    """The existing psmflow relabel tool is byte-identical to HEAD (the psmgoal path is a
    sibling file, so the psmflow path cannot have been touched)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head = subprocess.check_output(
        ["git", "show", "HEAD:tools/relabel_reward_rhat.py"], cwd=repo)
    with open(os.path.join(repo, "tools", "relabel_reward_rhat.py"), "rb") as fh:
        cur = fh.read()
    assert cur == head, "tools/relabel_reward_rhat.py differs from HEAD (psmflow path modified)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
