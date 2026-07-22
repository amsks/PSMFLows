import math

import jax.numpy as jnp
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config(overrides=None):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm", overrides=overrides or [])
    return ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))


class _FakeDataset:
    """Minimal replay-buffer stand-in: .sample(n) -> dict of arrays."""

    def __init__(self, n=256, obs=8, act=2, seed=0):
        rng = np.random.default_rng(seed)
        self.obs = rng.standard_normal((n, obs)).astype(np.float32)
        self.act = np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32)
        self.nxt = rng.standard_normal((n, obs)).astype(np.float32)
        self.n = n
        self._rng = np.random.default_rng(seed + 1)

    def sample(self, k):
        idx = self._rng.integers(0, self.n, size=k)
        return dict(observations=self.obs[idx], actions=self.act[idx],
                    next_observations=self.nxt[idx], index=idx.astype(np.int64))


def _trained_agent(overrides=None):
    config = _config(overrides)
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    ds = _FakeDataset()
    for _ in range(20):
        agent, _ = agent.update(ds.sample(config["batch_size"]))
    return agent, config, ds


def _mean_violation(agent, ds, goal, k=128):
    b = ds.sample(k)
    phi, bb = agent.measure(b["observations"], b["actions"], b["next_observations"])
    M = (phi * agent.w_inf).sum(-1, keepdims=True) + bb
    return float(np.mean(np.minimum(np.asarray(M), 0.0)))


def test_full_inference_reduces_constraint_violation_dgd():
    agent, config, ds = _trained_agent(["inference.use_dgd=true", "inference.num_inference_steps=200"])
    goal = ds.sample(1)["next_observations"][0]
    before = _mean_violation(agent, ds, goal)
    agent2 = agent.infer_w_goal(ds, goal)
    after = _mean_violation(agent2, ds, goal)
    assert math.isfinite(after)
    assert after >= before - 1e-6  # violation (a negative number) moves toward 0


def test_full_inference_hinge_runs():
    agent, config, ds = _trained_agent(["inference.use_dgd=false", "inference.num_inference_steps=100"])
    goal = ds.sample(1)["next_observations"][0]
    agent2 = agent.infer_w_goal(ds, goal)
    assert np.all(np.isfinite(np.asarray(agent2.w_inf)))


def test_zero_shot_inference_normalized_and_deterministic():
    agent, config, ds = _trained_agent(["inference.mode=zero_shot"])
    goal = ds.sample(1)["next_observations"][0]
    a1 = agent.infer_eval(ds, goal)
    a2 = agent.infer_eval(ds, goal)
    w1 = np.asarray(a1.w_inf)
    assert np.all(np.isfinite(w1))
    # sqrt(d)-normalized
    assert abs(np.linalg.norm(w1) - math.sqrt(config["d_dim"])) < 1e-3
    # deterministic given same goal + dataset (full-array reduction path)
    assert np.allclose(w1, np.asarray(a2.w_inf), atol=1e-5)


def test_distill_actor_increases_q_and_bounds_actions():
    agent, config, ds = _trained_agent(["inference.mode=full",
                                         "inference.num_inference_steps=100",
                                         "inference.num_actor_inference_steps=200"])
    goal = ds.sample(1)["next_observations"][0]
    agent = agent.infer_eval(ds, goal)

    def mean_q(ag):
        b = ds.sample(128)
        obs = jnp.asarray(b["observations"])
        a = ag.actor(obs, jnp.zeros((obs.shape[0], config["z_dim"])))
        goal_rep = jnp.broadcast_to(jnp.asarray(ag.task_goal), obs.shape[:-1] + (obs.shape[-1],))
        phi, bb = ag.measure(obs, a, goal_rep)
        return float(np.mean(np.asarray((phi * ag.w_inf).sum(-1, keepdims=True) + bb)))

    q0 = mean_q(agent)
    agent2 = agent.distill_actor(ds)
    q1 = mean_q(agent2)
    assert q1 >= q0 - 1e-4
    a = np.asarray(agent2.sample_actions(ds.sample(16)["observations"]))
    assert np.all(np.abs(a) <= 1.0 + 1e-5)


def test_end_to_end_infer_distill_act():
    agent, config, ds = _trained_agent(["inference.mode=full",
                                         "inference.num_inference_steps=50",
                                         "inference.num_actor_inference_steps=50"])
    goal = ds.sample(1)["next_observations"][0]
    agent = agent.infer_eval(ds, goal)
    agent = agent.distill_actor(ds)
    obs = ds.sample(10)["observations"]
    a = np.asarray(agent.sample_actions(obs))
    assert a.shape == (10, 2)
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0 + 1e-5)
