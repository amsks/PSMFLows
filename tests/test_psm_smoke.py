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
