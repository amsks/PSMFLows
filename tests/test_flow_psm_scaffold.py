import math

import gymnasium as gym
import pytest
import torch

from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

from agents.psm.flow_psm.agent import FlowPSMAgent
from agents.psm.flow_psm.flow_inversion import invert_flow
from agents.psm.flow_psm.model import FlowPSMModel


def test_invert_flow_is_stub():
    with pytest.raises(NotImplementedError):
        invert_flow(None, torch.randn(8, 40), torch.rand(8, 5) * 2 - 1)


def _model():
    return FlowPSMModel(
        obs_space=gym.spaces.Box(-1, 1, (40,)),
        action_dim=5,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        z_dim=128,
        max_log_seed=16,
        batch_size=8,
        device="cpu",
    )


def test_flow_psm_net_shapes():
    m = _model()
    s = torch.randn(8, 40)
    u0 = torch.randn(8, 5)
    u0p = torch.randn(8, 5)
    phi = m.flow_phi(s, u0, u0p)
    psi = m.flow_psi(torch.randn(8, 40))
    assert phi.shape == (8, 128)
    assert psi.shape == (8, 128)
    M = psi @ phi.T  # successor-measure matrix
    assert M.shape == (8, 8)


def _agent(**kw):
    return FlowPSMAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)),
        action_dim=5,
        batch_size=8,
        z_dim=128,
        max_log_seed=16,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu",
        **kw,
    )


def _batch(n=8, obs=40, act=5):
    return {
        "observation": torch.randn(n, obs),
        "action": torch.rand(n, act) * 2 - 1,
        "index": torch.arange(n),
        "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)},
    }


def test_update_trains_behavior_flow_and_stubs_are_noops():
    a = _agent()
    m = a.update(_batch(), step=0)
    assert math.isfinite(m["bc_flow_loss"])
    # stubbed seams contribute no metrics
    assert a._update_flow_psm() == {}
    assert a._update_u0_critic() == {}
