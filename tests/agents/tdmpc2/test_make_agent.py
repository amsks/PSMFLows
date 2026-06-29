import types
from omegaconf import OmegaConf
from train import make_agent


def test_make_agent_builds_tdmpc2():
    cfg = OmegaConf.create({
        "agent": "tdmpc2", "horizon": 3, "device": "cpu", "batch_size": 16,
        "tdmpc2": {"random_goal_ratio": 0.5, "success_threshold": 0.04,
                   "horizon": 3, "num_q": 5, "episode_length": 200},
    })
    obs_space = types.SimpleNamespace(shape=(19,))
    agent = make_agent(cfg, obs_space, action_dim=5)
    assert agent.__class__.__name__ == "TDMPC2Agent"
    assert agent.horizon == 3
