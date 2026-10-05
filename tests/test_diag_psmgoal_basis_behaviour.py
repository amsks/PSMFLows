"""tools/diag_psmgoal_basis_behaviour.py: next-row pairing, episode split, AUC, and the w fit
on a tiny synthetic basis. Run this file in its own process (JAX_PLATFORMS=cpu)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from tools.diag_psmgoal_basis_behaviour import (
    behaviour_eval,
    episode_ids,
    fit_w,
    future_valid,
    next_row_valid,
    roc_auc,
    split_episodes,
    td_loss,
)

# three episodes of lengths 3, 2, 4 (terminals mark the last row)
TERM = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1], np.float32)


def test_episode_ids_and_next_row_pairing():
    assert episode_ids(TERM).tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # row i pairs with i+1 only inside an episode; the last row of each episode is dropped
    assert next_row_valid(TERM).tolist() == [1, 1, 0, 1, 0, 1, 1, 1, 0]
    assert future_valid(TERM, 2).tolist() == [1, 0, 0, 0, 0, 1, 1, 0, 0]
    assert future_valid(TERM, 20).sum() == 0


def test_split_holds_out_last_episodes():
    tr, he = split_episodes(TERM, 1.0 / 3.0)
    assert tr.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert (tr ^ he).all()


def test_roc_auc():
    assert roc_auc([2, 3], [0, 1]) == 1.0
    assert roc_auc([0, 1], [2, 3]) == 0.0
    assert roc_auc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    p, n = rng.normal(1, 1, 400), rng.normal(0, 1, 500)
    brute = np.mean((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None]))
    assert abs(roc_auc(p, n) - brute) < 1e-12


def _basis(o, u, g):
    """Synthetic basis: phi = [-(o-g)^2 per dim, u], b = -|o - g|^2 / 10."""
    phi = jnp.concatenate([-(o - g) ** 2, u], -1)
    return phi, -jnp.sum((o - g) ** 2, -1) / 10.0


def test_td_loss_matches_manual_mesh():
    rng = np.random.default_rng(1)
    N, D = 5, 3
    phi, phi_t = rng.normal(size=(N, N, D)), rng.normal(size=(N, N, D))
    b, b_t = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    w, w_t = rng.normal(size=D), rng.normal(size=D)
    g = 0.9
    loss, _ = td_loss(*(jnp.asarray(x) for x in (phi, b, phi_t, b_t, w, w_t)), g)
    M, T = phi @ w + b, phi_t @ w_t + b_t
    off = [(M[i, j] - g * T[i, j]) ** 2 for i in range(N) for j in range(N) if i != j]
    ref = 0.5 * np.mean(off) - (1 - g) * np.mean(np.diag(M))
    assert abs(float(loss) - ref) < 1e-5


def test_fit_reduces_loss_and_eval_ranks_near_futures():
    # a 1-D random walk dataset: near futures are close to s_t, random states are far
    rng = np.random.default_rng(2)
    n_ep, L = 20, 60
    obs = np.concatenate([np.cumsum(rng.normal(0, 0.1, (L, 2)), 0) + rng.normal(0, 3, 2)
                          for _ in range(n_ep)]).astype(np.float32)
    u = rng.normal(size=(obs.shape[0], 2)).astype(np.float32)
    term = np.zeros(obs.shape[0], np.float32)
    term[L - 1::L] = 1
    nxt = np.concatenate([obs[1:], obs[-1:]])
    rows = np.nonzero(next_row_valid(term))[0]

    def sample(r):
        i = r.choice(rows, 16, replace=False)
        return obs[i], u[i], nxt[i], u[i + 1]

    w, hist = fit_w(_basis, sample, 4, 0.9, 0.05, 200, 1e-2, True, 0, log_every=50)
    assert w.shape == (4,) and abs(np.linalg.norm(w) - 2.0) < 1e-4
    assert all(np.isfinite(h["loss"]) for h in hist)
    _, held = split_episodes(term, 0.2)
    res = behaviour_eval(_basis, obs, u, term, held, [1, 5], 100, {"b_only": np.zeros(4)},
                         {"rand": rng.normal(size=(2, 4))}, np.random.default_rng(3))
    assert res["1"]["b_only"]["auc"] > 0.9            # b alone ranks near futures by distance
    assert res["1"]["n"] == 100 and "auc_mean" in res["1"]["rand"]
