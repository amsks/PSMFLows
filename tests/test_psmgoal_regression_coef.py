"""psmgoal coef_source='regression': a least-squares eval coefficient.

A diagnostic (2026-09-21) found psmgoal's zero-shot collapse is its goal coefficient
w*(g), not its features phi. The amortized/lp coefficient gives a near-constant,
reward-uncorrelated readout (corr -0.13). A least-squares coefficient on the SAME features
carries reward (corr 0.374). These tests pin the new `coef_source='regression'` branch of
`infer_eval_goals`: it sets eval_w_star = project(lstsq(Phi, r)) with
Phi[i] = mean_g phi(s'_i, u'_i, g), and leaves the amortized/lp branches unchanged.

Run this file in its own process (JAX_PLATFORMS=cpu ... -p no:cacheprovider).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 32


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = N                 # all rows rewarding -> goal set = every next_obs (deterministic)
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N             # sub = a full permutation, so lstsq is permutation-invariant
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    obs = np.zeros((2, OB), np.float32)
    act = np.zeros((2, DA), np.float32)
    return PSMGoalAgent.create(seed, obs, act, _cfg(**over))


def _phi_mean_over_goals(ag, next_obs, u, goals):
    """Phi[i] = mean_g phi_part(basis(next_obs[i], u[i], g)) -> (N, z_dim)."""
    n, g = next_obs.shape[0], goals.shape[0]
    o_r = jnp.broadcast_to(next_obs[:, None], (n, g, next_obs.shape[-1])).reshape(n * g, -1)
    u_r = jnp.broadcast_to(u[:, None], (n, g, u.shape[-1])).reshape(n * g, -1)
    g_r = jnp.broadcast_to(goals[None], (n, g, goals.shape[-1])).reshape(n * g, -1)
    phi, _ = ag.basis(o_r, u_r, g_r, params=ag.basis.params)
    return phi.reshape(n, g, -1).mean(axis=1)


def _pearson(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _linear_reward_batch(ag, seed=0):
    """Build (batch, r) where reward is a linear function of the features the regression sees.

    All rows are made rewarding (r > REWARDING_THRESHOLD) and k_goals == N, so the goal set is
    every next_obs -> Phi is deterministic (independent of the random goal draw). r = Phi @ w_true
    plus a constant shift; a constant adds no variance, so corr(., r) is unaffected by it.
    """
    rng = np.random.default_rng(seed)
    next_obs = rng.standard_normal((N, OB)).astype(np.float32)
    u = np.clip(rng.standard_normal((N, DA)).astype(np.float32), -3.0, 3.0)
    goals = jnp.asarray(next_obs)                      # all rows rewarding -> goals = every next_obs
    Phi = _phi_mean_over_goals(ag, jnp.asarray(next_obs), jnp.asarray(u), goals)
    w_true = rng.standard_normal((Z,)).astype(np.float32)
    raw = np.asarray(Phi) @ w_true
    r = (raw - raw.min() + 1.0).astype(np.float32)     # every row > 0.5, so every row is a goal
    batch = {
        "observations": rng.standard_normal((N, OB)).astype(np.float32),
        "next_observations": next_obs,
        "noise_preimage": u.astype(np.float32),
        "index": np.arange(N).astype(np.int32),
        "rewards": r,
    }
    return batch, r, Phi


# --------------------------------------------------------------------- the regression branch

def test_regression_sets_lstsq_projected_coef():
    """eval_w_star == project(lstsq(Phi, r)); Phi = mean_g phi(s', u', g) over the goal set."""
    ag = _agent(coef_source="regression")
    batch, r, _ = _linear_reward_batch(ag, seed=1)
    ev = ag.infer_eval_goals(batch, r)
    # Recompute Phi over ALL rows against the goal set the agent actually used.
    next_obs = jnp.asarray(batch["next_observations"])
    u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -3.0, 3.0)
    Phi = _phi_mean_over_goals(ev, next_obs, u, ev.eval_goals)
    w_reg = jnp.linalg.lstsq(Phi, jnp.asarray(r), rcond=None)[0]
    expect = ev._project(w_reg)
    # sub is a full permutation of all N rows, so the least-squares solution is invariant to it.
    assert jnp.allclose(ev.eval_w_star, expect, atol=1e-4)


def test_regression_readout_beats_random_and_is_positive():
    """mean_g phi^T eval_w_star correlates with reward, and beats a random sphere coefficient."""
    ag = _agent(coef_source="regression")
    batch, r, _ = _linear_reward_batch(ag, seed=2)
    ev = ag.infer_eval_goals(batch, r)
    Phi_used = _phi_mean_over_goals(
        ev, jnp.asarray(batch["next_observations"]),
        jnp.clip(jnp.asarray(batch["noise_preimage"]), -3.0, 3.0), ev.eval_goals)
    readout_reg = np.asarray(Phi_used @ ev.eval_w_star)
    corr_reg = _pearson(readout_reg, r)
    # A typical random sphere coefficient is uncorrelated with reward; regression must beat the
    # average over many draws. (A single random draw can spike by luck in a low-dim feature
    # space, so the claim is "beats a typical random coefficient", not "beats every draw".)
    corr_rand = []
    for i in range(256):
        w_rand = ev._project(jax.random.normal(jax.random.PRNGKey(1000 + i), (Z,)))
        corr_rand.append(abs(_pearson(np.asarray(Phi_used @ w_rand), r)))
    mean_abs_rand = float(np.mean(corr_rand))
    assert corr_reg > 0.5, f"regression readout must track reward (got {corr_reg:.3f})"
    assert corr_reg > mean_abs_rand + 0.3, \
        f"regression ({corr_reg:.3f}) must beat a typical random coefficient (mean |corr| {mean_abs_rand:.3f})"


def test_regression_coef_on_sqrt_z_sphere():
    """The regression coefficient is projected onto the sqrt(z_dim) sphere like the others."""
    ag = _agent(coef_source="regression")
    batch, r, _ = _linear_reward_batch(ag, seed=4)
    ev = ag.infer_eval_goals(batch, r)
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


# --------------------------------------------------------------------- amortized/lp unchanged

def _batch(reward_rows, seed=0):
    rng = np.random.default_rng(seed)
    r = -np.ones((N,), np.float32)
    for i in reward_rows:
        r[i] = 0.0
    return {
        "observations": rng.standard_normal((N, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((N, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((N, DA)).astype(np.float32),
        "index": np.arange(N).astype(np.int32),
        "rewards": r,
    }


def test_amortized_branch_unchanged():
    """coef_source='amortized' still returns project(mean_g h(g)) -- byte-unchanged formula."""
    ag = _agent(coef_source="amortized")
    b = _batch(reward_rows=(0, 2, 5))
    ev = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    expect = ev._project(jnp.mean(ev.w_star(ev.eval_goals), axis=0))
    assert jnp.allclose(ev.eval_w_star, expect, atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


def test_lp_branch_still_runs_and_is_sphere_normed():
    """coef_source='lp' (default) still runs the Lagrangian inference and returns a sphere w."""
    ag = _agent(coef_source="lp")
    b = _batch(reward_rows=(1, 3, 7))
    ev = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    assert ev.eval_w_star.shape == (Z,)
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


def test_regression_differs_from_amortized():
    """The regression coefficient is not the amortized one (it uses phi + reward, not h(g))."""
    b = _batch(reward_rows=tuple(range(N)))   # all rewarding so both branches see a full goal set
    ev_a = _agent(coef_source="amortized").infer_eval_goals(b, b["rewards"] + 1.0)
    ev_r = _agent(coef_source="regression").infer_eval_goals(b, b["rewards"] + 1.0)
    assert not jnp.allclose(ev_a.eval_w_star, ev_r.eval_w_star, atol=1e-3)
