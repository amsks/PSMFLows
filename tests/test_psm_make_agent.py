"""Sub-task 6.1 — make_agent must build psm / psm_flowbc agents.

Composes the agent YAMLs (configs/agent/psm.yaml, psm_flowbc.yaml) via Hydra,
shrinks them to a fast CPU config, and asserts make_agent returns the right
agent type and that one update step yields finite losses (state path)."""

import math

import gymnasium as gym
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

from agents.psm.agent import PSMAgent
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from train import make_agent


def _cfg(agent_name):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name=agent_name)
    # shrink for a fast CPU build/update
    with open_dict(cfg):
        cfg.device = "cpu"
        cfg.batch_size = 8
        cfg.z_dim = 16
        cfg.max_log_seed = 6
        cfg.amp = False
        cfg.phi.hidden_dim = 32
        cfg.sf.hidden_dim = 32
        cfg.actor.hidden_dim = 32
        if "actor_vf" in cfg:
            cfg.actor_vf.hidden_dim = 32
    return cfg


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1,
            "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs),
                     "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_make_agent_builds_psm():
    cfg = _cfg("psm")
    agent = make_agent(cfg, gym.spaces.Box(-1, 1, (40,)), action_dim=5)
    assert isinstance(agent, PSMAgent)
    m = agent.update(_batch(), step=0)
    for k in ["psm_loss", "sf_loss", "actor_loss"]:
        assert math.isfinite(m[k]), k


def test_make_agent_builds_psm_flowbc():
    cfg = _cfg("psm_flowbc")
    agent = make_agent(cfg, gym.spaces.Box(-1, 1, (40,)), action_dim=5)
    assert isinstance(agent, PSMFlowBCAgent)
    m = agent.update(_batch(), step=0)
    for k in ["psm_loss", "sf_loss", "actor_loss"]:
        assert math.isfinite(m[k]), k
