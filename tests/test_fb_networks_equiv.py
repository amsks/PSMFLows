"""Per-module bit-exact equivalence of the JAX FB networks vs the torch reference
(via tests/fixtures/fb_reference.npz). x64 + atol 1e-10."""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from utils.fb_networks import ForwardMap, BackwardMap
from utils.psm_networks import NoiseConditionedActor, FlowVectorField
from utils.torch_to_flax import (
    load_forward_params, load_backward_params, load_left_encoder_params,
    load_flow_actor, load_flow_vf,
)

FIX = np.load("tests/fixtures/fb_reference.npz")
Z, L, P, ADIM, H = 8, 8, 2, 5, 32

_obs = jnp.asarray(FIX["in__observations"], jnp.float64)
_act = jnp.asarray(FIX["in__actions"], jnp.float64)
_zg = jnp.asarray(FIX["in__z_gauss"], jnp.float64)


def test_backward_equiv():
    m = BackwardMap(z_dim=Z, hidden_dim=H, hidden_layers=4, norm=True)
    out = m.apply({"params": load_backward_params(FIX)}, _obs)
    assert np.allclose(out, FIX["out__B"], atol=1e-10)


def test_left_encoder_equiv():
    m = BackwardMap(z_dim=L, hidden_dim=H, hidden_layers=4, norm=True)
    out = m.apply({"params": load_left_encoder_params(FIX)}, _obs)
    assert np.allclose(out, FIX["out__left_enc"], atol=1e-10)


def test_forward_equiv():
    m = ForwardMap(z_dim=Z, hidden_dim=H, hidden_layers=2, embedding_layers=2, num_parallel=P)
    obs_feat = jnp.asarray(FIX["out__left_enc"], jnp.float64)  # F consumes left_enc(obs)
    out = m.apply({"params": load_forward_params(FIX)}, obs_feat, _zg, _act)
    assert np.allclose(out, FIX["out__F"], atol=1e-10)


def test_flow_actor_equiv():
    m = NoiseConditionedActor(action_dim=ADIM, hidden_dim=H, hidden_layers=2, embedding_layers=2)
    noise = jnp.asarray(FIX["in__actor_noise"], jnp.float64)
    out = m.apply({"params": load_flow_actor(FIX)}, _obs, _zg, noise)
    assert np.allclose(out, FIX["out__actor_mu"], atol=1e-10)


def test_flow_vf_equiv():
    m = FlowVectorField(action_dim=ADIM, hidden_dim=H, hidden_layers=4)
    t = jnp.asarray(FIX["in__flow_t"], jnp.float64)
    out = m.apply({"params": load_flow_vf(FIX)}, _obs, _act, t)
    assert np.allclose(out, FIX["out__vf"], atol=1e-10)
