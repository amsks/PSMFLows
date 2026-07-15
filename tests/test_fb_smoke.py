"""FB agent smoke tests. The bit-exact network/agent equivalence tests live in
test_fb_networks_equiv.py and test_fb_agent_equiv.py; this file holds the fixture
schema check plus registration/update/eval smoke tests."""
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _cfg(overrides=None):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        c = compose(config_name="fb", overrides=overrides or [])
    cfg = ml_collections.ConfigDict(OmegaConf.to_container(c, resolve=True))
    cfg["ob_dims"] = (19,)
    cfg["action_dim"] = 5
    return cfg


def _ex(n=4, obs=19, act=5):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((n, obs)).astype(np.float32),
            np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32))


def _batch(o, a, n):
    return dict(observations=o, actions=a, next_observations=o,
                rewards=np.zeros((n,), np.float32), terminals=np.zeros((n,), np.float32),
                masks=np.ones((n,), np.float32))


def test_fb_registered_and_creates():
    assert "fb" in agents
    o, a = _ex()
    agent = agents["fb"].create(0, o, a, _cfg())
    for k in ["forward", "backward", "left_encoder", "actor", "actor_vf",
              "target_forward", "target_backward", "target_left_encoder"]:
        assert k in agent.params, f"missing param {k}"
    assert agent.z_eval.shape == (agent.config["z_dim"],)


def test_fb_update_finite_flow():
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, _cfg())
    agent, info = agent.update(_batch(o, a, 64))
    assert np.isfinite(info["fb_loss"]) and np.isfinite(info["actor_loss"])
    assert np.isfinite(info["bc_flow_loss"])


def test_fb_update_finite_td3():
    cfg = _cfg(overrides=["actor.type=td3"])
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, cfg)
    assert "actor_vf" not in agent.params
    agent, info = agent.update(_batch(o, a, 64))
    assert np.isfinite(info["actor_loss"]) and np.isfinite(info["fb_loss"])


def test_fb_infer_eval_z_and_act():
    import jax
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, _cfg())
    ea = agent.infer_eval_z(o, np.ones((64,), np.float32))
    assert ea.z_eval.shape == (agent.config["z_dim"],)
    act = np.asarray(ea.sample_actions(observations=o[0], seed=jax.random.PRNGKey(0), temperature=0))
    assert act.shape == (5,) and np.all(np.abs(act) <= 1.0 + 1e-5)


def test_fb_arch_regression():
    o, a = _ex()
    agent = agents["fb"].create(0, o, a, _cfg())
    # backward: 4 hidden_layers -> Dense_0..Dense_4 = 5 Dense kernels
    n_bwd = sum(1 for k in agent.params["backward"] if k.startswith("Dense"))
    assert n_bwd == 5
    # forward tower (hidden_layers=2): fs_0/fs_1/fs_2 = 3 trunk linears
    tower = agent.params["forward"]["tower"]
    assert sum(1 for k in tower if k.startswith("fs_")) == 3
    assert {"embed_z_0", "embed_z_ln", "embed_z_3", "embed_sa_0", "embed_sa_ln", "embed_sa_3"} <= set(tower)
    for k in ["target_forward", "target_backward", "target_left_encoder"]:
        assert k in agent.params


def test_fb_fixture_schema():
    fix = np.load("tests/fixtures/fb_reference.npz")
    keys = set(fix.keys())
    for k in ["in__observations", "in__actions", "in__next_observations",
              "in__z_gauss", "in__mix_mask", "in__perm", "in__z", "in__next_action",
              "out__F", "out__B", "out__left_enc", "out__fb_loss", "out__actor_loss",
              "out__actor_mu", "out__vf"]:
        assert k in keys, f"missing {k}"
    for net in ["_forward_map", "_backward_map", "_left_encoder",
                "_target_forward_map", "_target_backward_map", "_target_left_encoder",
                "_actor", "_actor_vf"]:
        assert any(k.startswith(f"w__{net}.") for k in keys), f"no params for {net}"
    for i in range(10):
        assert any(k.startswith(f"step__{i}__") for k in keys), f"no step {i}"
        assert any(k.startswith(f"step_in__{i}__") for k in keys), f"no step_in {i}"
    assert fix["out__F"].ndim == 3 and fix["out__F"].shape[0] == 2  # [P,B,z]
