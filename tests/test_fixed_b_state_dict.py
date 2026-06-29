"""Regression test for the fixed-B (V2) checkpoint path.

fixed_b=cube_xyz has no backward params, so backward_optimizer is None.
state_dict()/load_state_dict() must tolerate that (it crashed in the 2k-step
GPU smoke: 'NoneType' object has no attribute 'state_dict' at final.pt save).
"""
import gymnasium as gym
import torch

from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    return FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)


def test_fixed_b_state_dict_roundtrips():
    a = _agent(fixed_b="cube_xyz")
    assert a.backward_optimizer is None
    s = a.state_dict()                       # must not raise
    assert s["backward_optimizer"] is None
    b = _agent(fixed_b="cube_xyz")
    b.load_state_dict(s)                      # must not raise


def test_learned_b_state_dict_still_roundtrips():
    a = _agent()                              # learned B => optimizer present
    assert a.backward_optimizer is not None
    s = a.state_dict()
    assert s["backward_optimizer"] is not None
    _agent().load_state_dict(s)
