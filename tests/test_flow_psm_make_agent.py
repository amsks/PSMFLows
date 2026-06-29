"""make_agent must build the flow_psm agent from its YAML and run one update."""

import math

import gymnasium as gym
import torch
from hydra import compose, initialize
from omegaconf import open_dict

from agents.psm.flow_psm.agent import FlowPSMAgent
from train import make_agent


def _cfg():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="flow_psm")
    with open_dict(cfg):
        cfg.device = "cpu"
        cfg.batch_size = 8
        cfg.z_dim = 16
        cfg.max_log_seed = 6
        cfg.amp = False
        cfg.phi.hidden_dim = 32
        cfg.sf.hidden_dim = 32
        cfg.actor.hidden_dim = 32
        cfg.actor_vf.hidden_dim = 32
    return cfg


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1,
            "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs),
                     "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_make_agent_builds_flow_psm():
    cfg = _cfg()
    agent = make_agent(cfg, gym.spaces.Box(-1, 1, (40,)), action_dim=5)
    assert isinstance(agent, FlowPSMAgent)
    m = agent.update(_batch(), step=0)
    assert math.isfinite(m["bc_flow_loss"])
