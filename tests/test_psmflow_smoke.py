"""Task 6: psmflow is registered, its Hydra config loads, and the agent runs end-to-end.

Guards the seam between configs/agent/psmflow.yaml and agents.psmflow.get_config(): the
YAML is what real runs use, so a drift between the two is invisible to every test that
builds its config in Python.
"""
import math

import jax
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="psmflow")
    c = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    with c.unlocked():
        c["flow"]["hidden_dims"] = tuple(c["flow"]["hidden_dims"])
        c["flow"]["value_hidden_dims"] = tuple(c["flow"]["value_hidden_dims"])
        c["allow_untrained_flow"] = True
    return c


def _batch(n=16, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        noise_preimage=rng.standard_normal((n, act)).astype(np.float32),
    )


def test_psmflow_registered_updates_and_acts():
    config = _config()
    assert config["agent_name"] == "psmflow"
    agent = agents["psmflow"].create(0, np.zeros((1, 8), np.float32), np.zeros((1, 2), np.float32), config)
    agent, info = agent.update(_batch())
    assert math.isfinite(float(info["psm_loss"]))

    b = _batch()
    rewards = np.random.default_rng(1).standard_normal((16,)).astype(np.float32)
    agent2 = agent.infer_eval_z(b["next_observations"], rewards)
    a = np.asarray(agent2.sample_actions(b["observations"][0], seed=jax.random.PRNGKey(0)))
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0 + 1e-5)


def test_yaml_and_get_config_agree():
    """configs/agent/psmflow.yaml must stay in sync with agents.psmflow.get_config().

    Real runs read the YAML; every other test reads get_config(). A key that exists in
    one and not the other, or a value that drifts, changes what actually trains while the
    whole suite stays green.
    """
    from agents.psmflow import get_config

    yaml_cfg = _config()
    py_cfg = get_config()
    with py_cfg.unlocked():
        py_cfg["allow_untrained_flow"] = True  # the one value _config() overrides

    def flat(c, prefix=""):
        out = {}
        for k, v in c.items():
            if hasattr(v, "items"):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = tuple(v) if isinstance(v, (list, tuple)) else v
        return out

    y, p = flat(yaml_cfg), flat(py_cfg)
    assert set(y) == set(p), (
        f"key drift — only in YAML: {sorted(set(y) - set(p))}, "
        f"only in get_config(): {sorted(set(p) - set(y))}")
    mismatched = {k: (y[k], p[k]) for k in y if y[k] != p[k]}
    assert not mismatched, f"value drift between YAML and get_config(): {mismatched}"
