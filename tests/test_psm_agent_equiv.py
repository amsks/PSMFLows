import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from agents.psm import contrastive_loss, ortho_loss, proto_sample  # noqa: E402

FIX = np.load("tests/fixtures/psm_reference.npz")
B = FIX["in__obs"].shape[0]
OFF = 1 - jnp.eye(B, dtype=jnp.float64)
OFF_SUM = OFF.sum()
DISC = 0.98 * jnp.ones((B, 1), jnp.float64)
MAX_SEED = 2 ** 4 + 20000  # max_log_seed=4 (proto_sampler: 2**L + 20000)


def test_contrastive_matches_fixture():
    M = jnp.asarray(FIX["out__M_psm"], jnp.float64)
    tM = jnp.asarray(FIX["out__target_M_psm"], jnp.float64)
    _, diag, offd = contrastive_loss(M, tM, DISC, OFF, OFF_SUM)
    assert np.allclose(float(offd), FIX["out__psm_offdiag"], atol=1e-10, rtol=0)
    assert np.allclose(float(diag), FIX["out__psm_diag"], atol=1e-10, rtol=0)


def test_ortho_matches_fixture():
    phi = jnp.asarray(FIX["out__phi"], jnp.float64)
    _, diag, offd = ortho_loss(phi, OFF, OFF_SUM)
    assert np.allclose(float(offd), FIX["out__orth_offdiag"], atol=1e-10, rtol=0)
    assert np.allclose(float(diag), FIX["out__orth_diag"], atol=1e-10, rtol=0)


def test_proto_sample_matches_fixture():
    table = jnp.asarray(FIX["proto_seed_to_action"], jnp.float64)
    powers = jnp.asarray(FIX["proto_powers"], jnp.float64)
    obs_hash = jnp.asarray(FIX["in__next_obs_hash"], jnp.float64)
    z = jnp.asarray(FIX["in__z_psm"], jnp.float64)
    out = proto_sample(table, powers, obs_hash, z, MAX_SEED)
    assert np.allclose(np.asarray(out), FIX["in__proto_next_action"], atol=1e-10, rtol=0)
