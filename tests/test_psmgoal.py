"""PSMGoalAgent: goal-indexed affine successor measure over the frozen flow's latents.

Spec: docs/design/2026-09-17-psmgoal.md. The measure is affine in a goal coefficient,
M(s,u,s+) = phi(s,u,s+)^T w*(g) + b(s,u,s+), with phi and b general networks on the triple
and w*(g) = h(g)/||h(g)||. phi/b are fitted by a contrastive TD on an N x (C+1) matrix whose
column 0 is the row's own s'; h climbs the goal value under a nonnegativity constraint on the
measure; an optional per-triple multiplier l does ascent on the penalty.

`_agent()` builds a tiny-dim agent on an untrained flow (`allow_untrained_flow`), the same
device tests/test_psmflow_agent.py uses.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from agents import agents
from agents.psmgoal import PSMGoalAgent, get_config
from utils.psm_common import contrastive_loss, contrastive_loss_rect, off_diagonal_mask, targets_uncertainty

OBS, ACT, B = 6, 2, 16
P = 2


def test_contrastive_loss_rect_matches_square_when_columns_are_the_batch_in_order():
    """With C+1 == N and row i's columns = [i, every other row in order], the rectangular loss
    is the square loss (column 0 the positive; the other N-1 columns the off-diagonal)."""
    rng = np.random.default_rng(0)
    N = 8
    M_sq = rng.standard_normal((P, N, N)).astype(np.float32)
    T_sq = rng.standard_normal((N, N)).astype(np.float32)
    cols = np.stack([np.concatenate([[i], [j for j in range(N) if j != i]]) for i in range(N)])  # (N, N)
    M_rect = np.stack([M_sq[:, i, cols[i]] for i in range(N)], axis=1)                      # (P, N, N)
    T_rect = np.stack([T_sq[i, cols[i]] for i in range(N)], axis=0)                          # (N, N)
    off, off_sum = off_diagonal_mask(N)
    want = contrastive_loss(M_sq, T_sq, 0.9, off, off_sum)
    got = contrastive_loss_rect(M_rect, T_rect, 0.9)
    for a, b in zip(want, got):
        np.testing.assert_allclose(float(a), float(b), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------- the agent
N_NEG, N_BACKUP, K_GOALS, Z = 4, 3, 4, 8


def _config(**overrides):
    c = get_config()
    with c.unlocked():
        c["allow_untrained_flow"] = True   # tests only; real runs require flow_ckpt_path
        c["z_dim"] = Z
        c["n_neg"] = N_NEG
        c["n_backup"] = N_BACKUP
        c["k_goals"] = K_GOALS
        c["gpi_num_u"] = 8
        c["measure"]["hidden_dim"] = 32
        c["coef"]["hidden_dim"] = 16
        c["mult"]["hidden_dim"] = 16
        for k, v in overrides.items():
            c[k] = v
    return c


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((B, OBS)).astype(np.float32),
        "actions": np.clip(rng.standard_normal((B, ACT)), -1, 1).astype(np.float32),
        "next_observations": rng.standard_normal((B, OBS)).astype(np.float32),
        "goals": rng.standard_normal((B, OBS)).astype(np.float32),
        "rewards": (rng.uniform(size=(B,)) < 0.5).astype(np.float32),
        "noise_preimage": rng.standard_normal((B, ACT)).astype(np.float32),
    }


def _agent(**overrides):
    return PSMGoalAgent.create(0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32),
                               _config(**overrides))


def _tree_absmax(tree):
    return max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(tree))


def test_agent_is_registered_under_psmgoal():
    assert agents["psmgoal"] is PSMGoalAgent
    assert get_config()["agent_name"] == "psmgoal"


def test_untrained_flow_requires_explicit_optin():
    c = _config()
    with c.unlocked():
        c["allow_untrained_flow"] = False
    try:
        PSMGoalAgent.create(0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32), c)
        assert False, "expected an assertion without flow_ckpt_path"
    except AssertionError as e:
        assert "flow_ckpt_path" in str(e)


def test_goal_coefficient_is_on_the_sqrt_d_sphere_and_measure_is_affine_in_it():
    agent = _agent()
    b = _batch()
    w = np.asarray(agent.coef(b["goals"]))
    assert w.shape == (B, Z)
    np.testing.assert_allclose(np.linalg.norm(w, axis=-1), math.sqrt(Z), rtol=1e-5)
    c1 = np.asarray(agent.coef(b["goals"]))
    c2 = np.asarray(agent.coef(-b["goals"]))
    assert not np.allclose(c1, c2), "h ignores its input"
    u = b["noise_preimage"]
    m1 = np.asarray(agent.measure(b["observations"], u, b["next_observations"], c1))
    m2 = np.asarray(agent.measure(b["observations"], u, b["next_observations"], c2))
    m12 = np.asarray(agent.measure(b["observations"], u, b["next_observations"], 0.5 * (c1 + c2)))
    assert m1.shape == (agent.config["num_parallel"], B)
    np.testing.assert_allclose(m1 + m2, 2.0 * m12, rtol=1e-5, atol=1e-5)
    assert not np.allclose(m1, m2), "the measure ignores the coefficient"


def _brute_force_backup(agent, next_obs, goals, coef, key):
    c = agent.config
    u_cand = jnp.clip(jax.random.normal(key, (N_BACKUP, next_obs.shape[0], ACT)), -c["u_clip"], c["u_clip"])

    def score(u_m):
        q = agent.measure(next_obs, u_m, goals, coef, params=agent.target_phi_b)     # (P, B)
        q_mean, q_unc = targets_uncertainty(q, c["num_parallel"])
        return q_mean - c["pessimism"] * q_unc

    Q = jax.vmap(score)(u_cand)                                                         # (n, B)
    best = jnp.argmax(Q, axis=0)
    return jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0], u_cand, Q


def test_backup_latent_is_the_pessimistic_argmax_of_the_target_measure_over_the_candidates():
    agent = _agent()
    # make online and target differ, so the test can tell which one the backup reads
    agent, _ = agent.update(_batch(1))
    b = _batch(2)
    key = jax.random.PRNGKey(7)
    coef = agent.coef(b["goals"])
    u_next, adv = agent.backup_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]), coef, key)
    u_star, u_cand, Q = _brute_force_backup(agent, jnp.asarray(b["next_observations"]),
                                            jnp.asarray(b["goals"]), coef, key)
    np.testing.assert_allclose(np.asarray(u_next), np.asarray(u_star), rtol=0, atol=1e-6)
    hit = jnp.any(jnp.all(jnp.isclose(u_cand, u_next[None]), axis=-1), axis=0)
    assert bool(jnp.all(hit)), "a backup latent is not one of its row's candidates"
    np.testing.assert_allclose(np.asarray(adv), np.asarray(Q.max(0) - Q.mean(0)), rtol=1e-5, atol=1e-6)
    assert np.all(np.abs(np.asarray(u_next)) <= agent.config["u_clip"] + 1e-6)
    # the pessimistic score is min over the ensemble at P=2, pessimism=0.5
    q = agent.measure(jnp.asarray(b["next_observations"]), u_next, jnp.asarray(b["goals"]), coef,
                      params=agent.target_phi_b)
    q_mean, q_unc = targets_uncertainty(q, 2)
    np.testing.assert_allclose(np.asarray(q_mean - 0.5 * q_unc), np.asarray(q.min(0)), rtol=1e-5, atol=1e-6)


def test_measure_loss_gradient_reaches_phi_b_only():
    agent = _agent()
    b = _batch()
    sampled = agent.sample_step_inputs(b, jax.random.PRNGKey(0))
    g_pb, g_h = jax.grad(lambda pb, h: agent.measure_loss(b, sampled, pb, h)[0], argnums=(0, 1))(
        agent.phi_b.params, agent.coef.params)
    assert _tree_absmax(g_pb) > 0.0, "loss_M does not move phi/b"
    assert _tree_absmax(g_h) == 0.0, "loss_M leaks into h (w* must be stop-gradded)"


def test_coef_loss_gradient_reaches_h_only():
    agent = _agent(constraint_coef=1.0)
    b = _batch()
    sampled = agent.sample_step_inputs(b, jax.random.PRNGKey(0))
    g_pb, g_h, g_l = jax.grad(lambda pb, h, ll: agent.coef_loss(b, sampled, pb, h, ll)[0],
                              argnums=(0, 1, 2))(agent.phi_b.params, agent.coef.params, agent.mult.params)
    assert _tree_absmax(g_h) > 0.0, "loss_h does not move h"
    assert _tree_absmax(g_pb) == 0.0, "loss_h leaks into phi/b"
    assert _tree_absmax(g_l) == 0.0, "loss_h leaks into the multiplier"


def test_constraint_coef_zero_gives_loss_h_equal_to_minus_obj():
    agent = _agent(constraint_coef=0.0)
    b = _batch()
    sampled = agent.sample_step_inputs(b, jax.random.PRNGKey(0))
    loss, (info, viol) = agent.coef_loss(b, sampled, agent.phi_b.params, agent.coef.params, agent.mult.params)
    assert float(loss) == -float(info["obj"])
    assert viol.shape == (B, N_NEG)


def test_multiplier_ascent_raises_l_on_violated_triples_and_is_zero_gradient_elsewhere():
    agent = _agent(constraint_coef=1.0)
    b = _batch()
    sampled = agent.sample_step_inputs(b, jax.random.PRNGKey(0))
    # the penalty's gradient w.r.t. the multiplier VALUES is viol / (N C): positive exactly
    # on violated triples, zero where the constraint holds
    viol = np.zeros((B, N_NEG), np.float32)
    viol[: B // 2] = 1.0
    l0 = np.asarray(agent.multipliers(b, sampled))
    assert l0.shape == (B, N_NEG) and np.all(l0 >= 0.0)
    g = np.asarray(jax.grad(agent.penalty)(jnp.asarray(l0), jnp.asarray(viol)))
    assert np.all(g[: B // 2] > 0.0) and np.all(g[B // 2:] == 0.0)
    # one ascent step raises the multipliers on the violated triples
    stepped, info = agent.multiplier_step(b, sampled, jnp.asarray(viol))
    l1 = np.asarray(stepped.multipliers(b, sampled))
    assert l1[: B // 2].mean() > l0[: B // 2].mean()
    assert math.isfinite(float(info["pen"]))
    # no violation anywhere: the multiplier parameters do not move at all
    same, _ = agent.multiplier_step(b, sampled, jnp.zeros((B, N_NEG), jnp.float32))
    for p0, p1 in zip(jax.tree_util.tree_leaves(agent.mult.params), jax.tree_util.tree_leaves(same.mult.params)):
        np.testing.assert_array_equal(np.asarray(p0), np.asarray(p1))


def test_update_runs_finite_and_logs_the_required_keys():
    for coef in (0.0, 1.0):
        agent = _agent(constraint_coef=coef)
        before = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
        t0 = agent.target_phi_b
        agent, info = agent.update(_batch())
        for k in ["psm_loss", "obj", "pen", "viol_frac", "mult_mean", "mult_max", "backup_adv",
                  "w_norm", "td_target_absmean"]:
            assert k in info, k
            assert math.isfinite(float(info[k])), (k, info[k])
        np.testing.assert_allclose(float(info["w_norm"]), math.sqrt(Z), rtol=1e-5)
        assert 0.0 <= float(info["viol_frac"]) <= 1.0
        assert float(info["backup_adv"]) >= 0.0
        # polyak target moved; the flow did not
        assert _tree_absmax(jax.tree_util.tree_map(lambda a, c: a - c, t0, agent.target_phi_b)) > 0.0
        after = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
        for x, y in zip(before, after):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_multiplier_moves_only_under_the_constraint():
    a0 = _agent(constraint_coef=0.0)
    a0, _ = a0.update(_batch())
    ref = _agent(constraint_coef=0.0)
    for p0, p1 in zip(jax.tree_util.tree_leaves(ref.mult.params), jax.tree_util.tree_leaves(a0.mult.params)):
        np.testing.assert_array_equal(np.asarray(p0), np.asarray(p1))


def test_missing_goals_key_raises():
    agent = _agent()
    bad = _batch()
    del bad["goals"]
    try:
        agent.update(bad)
        assert False, "expected KeyError for missing goals"
    except KeyError:
        pass


def test_infer_eval_goals_selects_rewarding_rows_only_and_gives_unit_w():
    agent = _agent()
    b = _batch(3)
    rew = np.zeros((B,), np.float32)
    rew[[1, 5, 9, 12, 13]] = 1.0                       # the shifted reward r + 1 of a rewarding row
    np.random.seed(0)
    ea = agent.infer_eval_goals(b["next_observations"], rew)
    goals = np.asarray(ea.eval_goals)
    assert goals.shape == (K_GOALS, OBS)
    rewarding = b["next_observations"][rew > 0.5]
    for g in goals:
        assert np.any(np.all(np.isclose(rewarding, g[None]), axis=1)), "a goal is not a rewarding next state"
    w = np.asarray(ea.eval_w)
    assert w.shape == (Z,)
    np.testing.assert_allclose(np.linalg.norm(w), math.sqrt(Z), rtol=1e-5)
    # w is the sum of the per-goal coefficients, projected back onto the sqrt(D) sphere
    ws = np.asarray(agent.coef(goals)).sum(0)
    np.testing.assert_allclose(w, math.sqrt(Z) * ws / np.linalg.norm(ws), rtol=1e-5, atol=1e-5)
    # main.py / eval_checkpoint.py dispatch on this hook, not on infer_eval_z
    assert hasattr(agent, "infer_eval_goals") and not hasattr(agent, "infer_eval_z")


def test_infer_eval_goals_refuses_a_batch_with_no_rewarding_row():
    agent = _agent()
    b = _batch(3)
    try:
        agent.infer_eval_goals(b["next_observations"], np.zeros((B,), np.float32))
        assert False, "expected a ValueError with no rewarding rows"
    except ValueError as e:
        assert "rewarding" in str(e)


def test_sample_actions_through_the_real_eval_call_path():
    from utils.evaluation import supply_rng

    agent = _agent()
    agent, _ = agent.update(_batch())
    b = _batch(4)
    rew = (np.arange(B) % 3 == 0).astype(np.float32)
    ea = agent.infer_eval_goals(b["next_observations"], rew)
    actor_fn = supply_rng(ea.sample_actions, rng=jax.random.PRNGKey(0))
    a = np.asarray(actor_fn(observations=b["observations"][0], temperature=1.0))
    assert a.shape == (ACT,)
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0 + 1e-5)
    u = np.asarray(ea.select_latent(b["observations"][0], jax.random.PRNGKey(0)))
    assert u.shape == (ACT,) and np.all(np.abs(u) <= agent.config["u_clip"] + 1e-6)
    # the selected latent is one of the gpi_num_u prior draws
    cand = np.asarray(jnp.clip(jax.random.normal(jax.random.PRNGKey(0), (8, ACT)), -3.0, 3.0))
    assert np.any(np.all(np.isclose(cand, u[None]), axis=1))


def test_different_goal_sets_change_the_action():
    """Disjoint goal sets give a different w and, over a few per-step draws, a different
    selected latent. One seed is not enough: the argmax over 8 prior draws of a tiny agent
    can coincide for two w by chance, so the check is over several seeds."""
    agent = _agent()
    for i in range(5):
        agent, _ = agent.update(_batch(i))
    b = _batch(9)
    r1 = (np.arange(B) % 2 == 0).astype(np.float32)
    e1 = agent.infer_eval_goals(b["next_observations"], r1)
    e2 = agent.infer_eval_goals(b["next_observations"], 1.0 - r1)
    assert not np.allclose(np.asarray(e1.eval_w), np.asarray(e2.eval_w))
    ob = b["observations"][0]
    u1 = [np.asarray(e1.select_latent(ob, jax.random.PRNGKey(k))) for k in range(6)]
    u2 = [np.asarray(e2.select_latent(ob, jax.random.PRNGKey(k))) for k in range(6)]
    assert any(not np.allclose(a, c) for a, c in zip(u1, u2)), "the goal set never changes the latent"


def test_phi_is_on_the_sqrt_d_sphere_and_b_is_bounded():
    """The scale anchor (2026-09-17): a general phi(s,u,s+) sets every column independently,
    so nothing couples the positive column to the negatives and the contrastive term
    -2 M(s,u,s') is unbounded below. Fix, mirroring the reference's f on the sqrt(D) sphere:
    ||phi|| = sqrt(D), |b| < D (b = D tanh(b/D))."""
    agent = _agent()
    b = _batch(5)
    phi, bias = agent.phi_b(b["observations"], b["noise_preimage"], b["next_observations"])
    phi, bias = np.asarray(phi), np.asarray(bias)
    assert phi.shape[-1] == Z and bias.shape == phi.shape[:-1]
    np.testing.assert_allclose(np.linalg.norm(phi, axis=-1), math.sqrt(Z), rtol=1e-5)
    assert np.all(np.abs(bias) < Z)


def test_measure_is_bounded_by_two_d_for_any_input():
    agent = _agent()
    b = _batch(6)
    w = np.asarray(agent.coef(b["goals"]))
    big = 50.0 * b["next_observations"]                       # far-out inputs must not escape the bound
    m = np.asarray(agent.measure(b["observations"], b["noise_preimage"], big, w))
    assert np.all(np.abs(m) <= 2.0 * Z + 1e-4)
