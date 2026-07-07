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


def _make_agent(actor_type):
    config = _config()
    with config.unlocked():
        config["actor"]["type"] = actor_type
    cls = agents["psm"]
    ex_obs = np.zeros((1, 8), np.float32)
    ex_act = np.zeros((1, 2), np.float32)
    return cls.create(0, ex_obs, ex_act, config)


def test_ddpgbc_actor_trains():
    """DDPGBC actor (default): a couple of update steps produce finite losses incl. the BC term."""
    agent = _make_agent("ddpgbc")
    for _ in range(2):
        agent, info = agent.update(_batch())
    for k in ["psm_loss", "sf_loss", "actor_loss", "bc_error"]:
        assert math.isfinite(float(info[k])), (k, info[k])


def test_flow_actor_trains_and_acts():
    """Flow actor: builds actor_vf, trains (finite bc_flow_loss + actor_loss), and the
    one-step policy produces bounded actions that respond to the inferred z."""
    agent = _make_agent("flow")
    assert "actor_vf" in agent.params, "flow actor must build a velocity field"
    for _ in range(2):
        agent, info = agent.update(_batch())
    for k in ["psm_loss", "sf_loss", "actor_loss", "bc_flow_loss", "bc_error"]:
        assert math.isfinite(float(info[k])), (k, info[k])

    b = _batch()
    obs = b["observations"]
    rewards = np.random.default_rng(1).standard_normal((obs.shape[0],)).astype(np.float32)
    agent2 = agent.infer_eval_z(b["next_observations"], rewards)
    a = np.asarray(agent2.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == obs.shape[:-1] + (2,)
    assert np.all(np.abs(a) <= 1.0 + 1e-5), "flow actions must be tanh/clip bounded"


def test_flow_actor_is_noise_conditioned_architecture():
    """Regression for the flow-actor port (B1): the one-step actor must be the faithful
    NoiseConditionedActor (dual embeddings + policy trunk), NOT a flat MLP. We check the
    param tree carries the embedding + policy Dense layers of the reference topology."""
    agent = _make_agent("flow")
    actor = agent.params["actor"]
    paths = [".".join(str(getattr(k, "key", k)) for k in path)
             for path, _ in jax.tree_util.tree_flatten_with_path(actor)[0]]
    joined = "\n".join(paths)
    assert "Dense" in joined and "LayerNorm" in joined, joined
    n_dense = sum(p.endswith("kernel") and "Dense" in p for p in paths)
    # 2 embeddings x 2 Dense = 4, policy hidden_layers(=2) + head = 3 -> 7 Dense kernels.
    assert n_dense == 7, (n_dense, joined)


def test_proto_branch_consumes_batch_index():
    """Regression for B2: when the batch carries a global row index, the proto next-action
    is keyed on it (not batch position), so an update with 'index' present runs and differs
    from the arange fallback."""
    agent = _make_agent("flow")
    b = _batch()
    b_idx = dict(b, index=np.arange(1000, 1000 + b["observations"].shape[0], dtype=np.int64))
    a1, i1 = agent.update(dict(b))          # arange(B) fallback
    a2, i2 = agent.update(b_idx)            # global-index keyed
    assert math.isfinite(float(i2["psm_loss"]))
    # Different proto keys -> different proto targets -> different psm loss.
    assert not np.isclose(float(i1["psm_loss"]), float(i2["psm_loss"]))


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
