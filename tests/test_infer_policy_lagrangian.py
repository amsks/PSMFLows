"""Pure parts of tools/infer_policy_lagrangian.py (PSM Eq. 10 over the affine coefficient)
on tiny synthetic arrays. CPU, no checkpoint."""
import itertools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.infer_policy_lagrangian import (
    family_geometry,
    family_init,
    lagrangian_fit,
    objective_linear,
    project_c,
    violation_stats,
)

P, N, Z, W = 2, 40, 6, 4


def _synthetic(seed=0, beta_shift=0.0):
    """phi is POSITIVE, so the sign of m = psi^T phi is the sign of beta_shift while
    |A^T c| stays below it -- the two fits below then sit on one side of the constraint."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((P, N, Z, W)).astype(np.float32)
    beta = (rng.standard_normal((P, N, Z)) + beta_shift).astype(np.float32)
    phi = (np.abs(rng.standard_normal((N, Z))) + 0.5).astype(np.float32)
    w = rng.standard_normal(Z).astype(np.float32)
    return A, beta, phi, w


def test_objective_is_linear_and_family_init_is_its_argmax():
    rng = np.random.default_rng(1)
    a_bar, b_bar = rng.standard_normal(W), 0.7
    W_fam = rng.standard_normal((16, W))
    W_fam /= np.linalg.norm(W_fam, axis=1, keepdims=True)
    k = family_init(W_fam, a_bar)
    vals = [objective_linear(W_fam[i], a_bar, b_bar) for i in range(16)]
    assert vals[k] == max(vals)
    c = rng.standard_normal(W)
    assert np.isclose(objective_linear(2 * c, a_bar, b_bar) - b_bar, 2 * (objective_linear(c, a_bar, b_bar) - b_bar))


def test_violation_stats_and_geometry():
    m = np.array([[1.0, -2.0], [0.5, -0.5]])
    v = violation_stats(m)
    assert v["frac_violated"] == 0.5 and np.isclose(v["mean_violation"], (2.0 + 0.5) / 4)
    assert np.isclose(v["mean_violation_given_violated"], 1.25) and v["min_m"] == -2.0
    W_fam = np.eye(W)
    g = family_geometry(np.array([3.0, 0.0, 0.0, 0.0]), W_fam)
    assert g["nearest_family_index"] == 0 and np.isclose(g["cos_nearest_family"], 1.0)
    assert np.isclose(g["norm"], 3.0) and np.isclose(g["dist_nearest_family"], 2.0)
    assert np.isclose(np.linalg.norm(project_c(np.array([3.0, 4.0]), "sphere")), 1.0)
    np.testing.assert_array_equal(project_c(np.array([3.0, 4.0]), "free"), [3.0, 4.0])


def test_unconstrained_direction_climbs_the_objective_and_leaves_lambda_at_zero():
    """beta shifted far positive: the constraint m >= 0 never binds, lambda stays 0, and
    the free coefficient moves the objective up monotonically along a_bar."""
    A, beta, phi, w = _synthetic(0, beta_shift=50.0)
    A_j, beta_j, phi_j = jnp.asarray(A), jnp.asarray(beta), jnp.asarray(phi)
    a_bar = np.einsum("pizw,z->w", A, w) / (P * N)
    b_bar = float(np.einsum("piz,z->", beta, w) / (P * N))
    c0 = np.zeros(W, np.float32)
    c_star, lam, curve = lagrangian_fit(A_j, beta_j, phi_j, w, c0, 1.0, steps=200, lr=1e-2, lam_lr=1e-2,
                                        c_mode="free", lam_mode="row", rows_per_step=16, cols_per_step=8,
                                        key=jax.random.PRNGKey(0), log_every=100, log=lambda *_: None)
    assert lam.shape == (N,) and float(lam.max()) == 0.0
    J = [r["objective_raw"] for r in curve]
    assert J[-1] > J[0] and all(b >= a for a, b in itertools.pairwise(J))
    assert objective_linear(c_star, a_bar, b_bar) > objective_linear(c0, a_bar, b_bar)


def test_sphere_mode_keeps_unit_norm_and_scalar_lambda_activates_under_violation():
    """beta shifted far negative: every pair is violated, so the scalar multiplier grows
    and the sphere-projected coefficient stays unit norm throughout."""
    A, beta, phi, w = _synthetic(2, beta_shift=-50.0)
    c0 = np.ones(W, np.float32) / np.sqrt(W)
    c_star, lam, curve = lagrangian_fit(jnp.asarray(A), jnp.asarray(beta), jnp.asarray(phi), w, c0, 1.0,
                                        steps=50, lr=1e-2, lam_lr=1e-1, c_mode="sphere", lam_mode="scalar",
                                        rows_per_step=16, cols_per_step=8, key=jax.random.PRNGKey(3),
                                        log_every=25, log=lambda *_: None)
    assert lam.shape == () and float(lam) > 0.0
    assert np.isclose(np.linalg.norm(c_star), 1.0, atol=1e-5)
    assert all(np.isclose(r["c_norm"], 1.0, atol=1e-5) for r in curve)
    assert curve[-1]["step_violation_frac"] == 1.0


def test_as_list_accepts_every_hydra_spelling():
    from tools.infer_policy_lagrangian import as_list
    assert as_list("1,2,3", int) == [1, 2, 3]
    assert as_list("[1, 2]", int) == [1, 2]
    assert as_list([1, 2], int) == [1, 2]
    assert as_list("['free', 'sphere']") == ["free", "sphere"]
    assert as_list(["free"]) == ["free"]
