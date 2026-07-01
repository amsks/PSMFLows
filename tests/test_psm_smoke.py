import math

import jax
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="psm")
    return ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))


def _batch(n=16, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        masks=np.ones((n,), np.float32),
    )


def test_psm_registered_and_updates():
    config = _config()
    assert config["agent_name"] == "psm"
    cls = agents["psm"]
    ex_obs = np.zeros((1, 8), np.float32)
    ex_act = np.zeros((1, 2), np.float32)
    agent = cls.create(0, ex_obs, ex_act, config)
    agent, info = agent.update(_batch())
    for k in ["psm_loss", "sf_loss", "actor_loss"]:
        assert math.isfinite(float(info[k])), (k, info[k])


def test_sample_actions_uses_inferred_z():
    """Eval must be goal-directed: infer_eval_z sets a non-zero task latent and
    changes the acted policy vs. the default zero-z."""
    config = _config()
    cls = agents["psm"]
    ex_obs = np.zeros((1, 8), np.float32)
    ex_act = np.zeros((1, 2), np.float32)
    agent = cls.create(0, ex_obs, ex_act, config)
    agent, _ = agent.update(_batch())

    b = _batch()
    obs = b["observations"]
    rewards = np.random.default_rng(1).standard_normal((obs.shape[0],)).astype(np.float32)

    a_zero = np.asarray(agent.sample_actions(obs))  # default z_eval == zeros
    agent2 = agent.infer_eval_z(b["next_observations"], rewards)
    a_inferred = np.asarray(agent2.sample_actions(obs))

    assert float(np.linalg.norm(np.asarray(agent2.z_eval))) > 0.0
    assert not np.allclose(a_zero, a_inferred), "sample_actions ignored the inferred z"
