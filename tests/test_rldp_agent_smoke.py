"""Smoke test for RLDPAgent — instantiate, run 10 update steps on synthetic data."""
import gymnasium
import numpy as np
import torch

from agents.rldp.agent import RLDPAgent
from buffers.transition import DictBuffer


def _make_synthetic_buffer(n_eps: int = 5, ep_len: int = 40, obs_dim: int = 20, act_dim: int = 4):
    total = n_eps * ep_len
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    next_obs = np.random.randn(total, obs_dim).astype(np.float32)
    action = np.random.randn(total, act_dim).astype(np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=bool)
    for i in range(n_eps):
        terminated[(i + 1) * ep_len - 1] = True
    timestep = np.concatenate([np.arange(ep_len) for _ in range(n_eps)]).astype(np.int64)
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_rldp_agent_constructs():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5)
    assert agent.horizon == 5
    assert hasattr(agent.model, "_predictor")


def test_rldp_agent_update_step_no_nan():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5,
                      ortho_coef=0.01, lr_b=1e-4, lr_f=1e-4, lr_actor=1e-4)
    buf = _make_synthetic_buffer()
    torch.manual_seed(0)
    for step in range(10):
        batch = buf.sample(16, horizon=5)
        metrics = agent.update(batch, step)
        for k, v in metrics.items():
            val = float(v.detach() if hasattr(v, "detach") else v)
            assert not np.isnan(val), f"NaN in metric {k!r} at step {step}: {val}"
            assert np.isfinite(val), f"non-finite metric {k!r} at step {step}: {val}"


def test_rldp_agent_backward_optimizer_includes_predictor():
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    agent = RLDPAgent(obs_space=obs_space, action_dim=4, batch_size=16, horizon=5)
    opt_params = set()
    for pg in agent.backward_optimizer.param_groups:
        for p in pg["params"]:
            opt_params.add(id(p))
    for p in agent.model._predictor.parameters():
        assert id(p) in opt_params, "predictor param missing from backward_optimizer"
