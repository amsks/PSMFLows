"""Bit-exact equivalence of FBAgent's update vs the torch reference: static branch
losses (atol 1e-10) and a 10-step interleaved replay (atol 1e-8), injecting the
fixture's randomness."""
import jax
jax.config.update("jax_enable_x64", True)

import ml_collections
import numpy as np
import jax.numpy as jnp

from agents.fb import FBAgent
from utils.torch_to_flax import (
    load_forward_params, load_backward_params, load_left_encoder_params,
    load_flow_actor, load_flow_vf,
)

FIX = np.load("tests/fixtures/fb_reference.npz")

EQUIV = ml_collections.ConfigDict(dict(
    agent_name="fb", batch_size=16, z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
    f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
    fb_pessimism_penalty=0.0, actor_pessimism_penalty=0.0, actor_std=0.2,
    stddev_clip=0.3, norm_z=True, actor_encode_obs=False, weight_decay=0.0,
    lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
    forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
    backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
    left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
    actor=dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
               bc_coeff=3.0, flow_steps=10, lr_actor_vf=3e-4,
               flow_actor_hidden_dim=32, flow_actor_hidden_layers=2,
               flow_actor_embedding_layers=2, flow_vf_hidden_dim=32,
               flow_vf_hidden_layers=4),
    ob_dims=(8,), action_dim=5, encoder=None,
))

_F64 = lambda k: jnp.asarray(FIX[k], jnp.float64)


def _batch(prefix="in__"):
    return {k: _F64(f"{prefix}{k}") for k in
            ["observations", "actions", "next_observations", "rewards", "terminals"]}


def _mapped_agent():
    o = _F64("in__observations")
    a = _F64("in__actions")
    agent = FBAgent.create(0, o, a, EQUIV)
    p = dict(agent.params)
    p["forward"] = load_forward_params(FIX, "_forward_map")
    p["backward"] = load_backward_params(FIX, "_backward_map")
    p["left_encoder"] = load_backward_params(FIX, "_left_encoder")
    p["actor"] = load_flow_actor(FIX, "_actor")
    p["actor_vf"] = load_flow_vf(FIX, "_actor_vf")
    p["target_forward"] = load_forward_params(FIX, "_target_forward_map")
    p["target_backward"] = load_backward_params(FIX, "_target_backward_map")
    p["target_left_encoder"] = load_backward_params(FIX, "_target_left_encoder")
    return agent.replace(params=p)


def _inj(prefix="in__"):
    return {k: _F64(f"{prefix}{k}") for k in ["z", "next_action", "flow_x0", "flow_t", "actor_noise"]}


def test_fb_static_equiv():
    agent = _mapped_agent()
    info = agent.compute_static(_batch(), _inj())
    assert np.allclose(info["fb_loss"], FIX["out__fb_loss"], atol=1e-10), \
        (float(info["fb_loss"]), float(FIX["out__fb_loss"]))
    assert np.allclose(info["actor_loss"], FIX["out__actor_loss"], atol=1e-10), \
        (float(info["actor_loss"]), float(FIX["out__actor_loss"]))
    assert np.allclose(info["ortho_loss"], FIX["out__ortho_loss"], atol=1e-10)
    assert np.allclose(info["bc_flow_loss"], FIX["out__bc_flow_loss"], atol=1e-10)


# (flax key, torch state_dict key, transpose?) representative param per stepped net.
_CHECKS = [
    (("backward", "Dense_0", "kernel"), "_backward_map.net.0.weight", True),
    (("left_encoder", "Dense_0", "kernel"), "_left_encoder.net.0.weight", True),
    (("forward", "tower", "fs_2", "kernel"), "_forward_map.Fs.4.weight", False),
    (("actor", "Dense_6", "kernel"), "_actor.policy.4.weight", True),
    (("actor_vf", "Dense_0", "kernel"), "_actor_vf.net.0.weight", True),
]


def _get(tree, path):
    for k in path:
        tree = tree[k]
    return np.asarray(tree)


def test_fb_perstep_equiv():
    agent = _mapped_agent()
    for i in range(10):
        inj = {k: _F64(f"step_in__{i}__{k}")
               for k in ["z", "next_action", "flow_x0", "flow_t", "actor_noise"]}
        agent, _ = agent.apply_update(_batch(), inj)
        for path, tkey, tr in _CHECKS:
            got = _get(agent.params, path)
            exp = FIX[f"step__{i}__{tkey}"]
            exp = exp.T if tr else exp
            assert np.allclose(got, exp, atol=1e-8), f"step {i} {path} mismatch (max {np.abs(got-exp).max():.2e})"
