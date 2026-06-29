import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from utils.psm_networks import PhiMap  # noqa: E402
from utils.torch_to_flax import load_phi_params  # noqa: E402

FIX = np.load("tests/fixtures/psm_reference.npz")


def test_phi_equiv():
    goal = jnp.asarray(FIX["in__goal"], dtype=jnp.float64)
    phi = PhiMap(z_dim=8, hidden_dim=32, hidden_layers=2, norm=True)
    params = load_phi_params(FIX)
    out = phi.apply({"params": params}, goal)
    assert np.allclose(np.asarray(out), FIX["out__phi"], atol=1e-10, rtol=0), \
        np.abs(np.asarray(out) - FIX["out__phi"]).max()
