import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from utils.psm_networks import PhiMap, PsiMap, PSMActor  # noqa: E402
from utils.torch_to_flax import load_phi_params, load_psi_params, load_actor_params  # noqa: E402

FIX = np.load("tests/fixtures/psm_reference.npz")


def test_phi_equiv():
    goal = jnp.asarray(FIX["in__goal"], dtype=jnp.float64)
    phi = PhiMap(z_dim=8, hidden_dim=32, hidden_layers=2, norm=True)
    params = load_phi_params(FIX)
    out = phi.apply({"params": params}, goal)
    assert np.allclose(np.asarray(out), FIX["out__phi"], atol=1e-10, rtol=0), \
        np.abs(np.asarray(out) - FIX["out__phi"]).max()


def test_sf_psi_equiv():
    obs = jnp.asarray(FIX["in__obs"], jnp.float64)
    z = jnp.asarray(FIX["in__z_cont"], jnp.float64)
    act = jnp.asarray(FIX["in__action"], jnp.float64)
    psi = PsiMap(output_dim=8, hidden_dim=32, num_parallel=2)
    params = load_psi_params(FIX, which="sf_psi")
    out = psi.apply({"params": params}, obs, z, act)
    assert np.allclose(np.asarray(out), FIX["out__sf_psi"], atol=1e-10, rtol=0), \
        np.abs(np.asarray(out) - FIX["out__sf_psi"]).max()


def test_psm_psi_equiv():
    obs = jnp.asarray(FIX["in__obs"], jnp.float64)
    zb = jnp.asarray(FIX["in__z_psm"], jnp.float64)
    act = jnp.asarray(FIX["in__action"], jnp.float64)
    psi = PsiMap(output_dim=8, hidden_dim=32, num_parallel=2)
    params = load_psi_params(FIX, which="psm_psi")
    out = psi.apply({"params": params}, obs, zb, act)
    assert np.allclose(np.asarray(out), FIX["out__psm_psi"], atol=1e-10, rtol=0), \
        np.abs(np.asarray(out) - FIX["out__psm_psi"]).max()


def test_actor_mu_equiv():
    obs = jnp.asarray(FIX["in__obs"], jnp.float64)
    z = jnp.asarray(FIX["in__z_cont"], jnp.float64)
    actor = PSMActor(action_dim=2, hidden_dim=32)
    params = load_actor_params(FIX)
    mu = actor.apply({"params": params}, obs, z)
    assert np.allclose(np.asarray(mu), FIX["out__actor_mu"], atol=1e-10, rtol=0), \
        np.abs(np.asarray(mu) - FIX["out__actor_mu"]).max()
