"""Is psmgoal's learned basis (phi, b) good? Checks 2 and 3 (2026-09-30).

M(s,u,s+) = phi(s,u,s+)^T w(z) + b(s,u,s+), trained by squared TD on a state x next-state
mesh with the fixed z-indexed proto policy as the bootstrap (agents/psmgoal.py measure_loss).

CHECK 2 -- held-out policies. Every 16-bit code was seen in training, so new policies are
built with the SAME proto construction under a base key never used in training
(PRNGKey(new_proto_seed), default 12345). With phi, b (online and target) frozen, a free w is
fitted per policy by the training TD loss (off-diagonal squared TD to gamma * target mesh,
-(1-gamma) diagonal pull, Polyak target copy of w, Adam). The loss is quadratic in w given
the frozen basis, so every mesh is reduced to exact second moments of the augmented feature
[phi, b] (online side, at (s_i, u_i, s'_j)) and [phi_bar, b_bar] (target side, at
(s'_i, u+_i, s'_j) with u+_i the policy's latent at row i). Fit on `fit_meshes`, scored on
separate `eval_meshes`. Score = normalized TD error = off-diagonal MSE of (M - gamma M_bar)
divided by the variance of gamma M_bar, with the same w on both sides.
  (a) training codes, trained w(z), no refit
  (b) training codes, w refit          (c) new-seed policies, w refit
Each refit is reported three ways: Adam free w, Adam w projected to the sqrt(D) sphere (the
trained w(z) lives there), and the closed-form TD fixed point (least-squares solve).

CHECK 3 -- span. On one 256 x 256 mesh (s_i, u_i data latent, s+_j = s'_j), M for 256 random
codes: a two-way decomposition over (policy, cell) of the variance of M; participation
ratio of w(z) (256 x D) and of M (256 x cells); and, at 256 x 16 (s, s+) cells, the fraction
of the variance of M that lies across 64 clipped prior latents u at a fixed (s, s+).

  .venv/bin/python tools/diag_psmgoal_basis_policies.py \
      --run_dir '$PSM_DATA/exp/PSMFLows/psmgoal_lift_cube_gc/sd00*' --epochs 250000 500000 750000 \
      --report_dir $PSM_DATA/logs --report_out $PSM_DATA/logs/diag_psmgoal_basis_policies_summary.json
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax
import jax
import jax.numpy as jnp

from utils.psm_proto import proto_latents, proto_seed_ints


# ------------------------------------------------------------------ pure helpers (tested)
def code_bits(codes, width):
    """Integer codes -> (P, width) float32 bits, LSB-first (sample_z_bin's layout)."""
    codes = np.asarray(codes, np.int64)
    return ((codes[:, None] >> np.arange(width)) & 1).astype(np.float32)


def policy_latents(code_bin, rows, max_log_seed, action_dim, u_clip, base_seed):
    """u+ of the proto policy with code `code_bin` (width,) at dataset rows `rows` (N,),
    under base key PRNGKey(base_seed). base_seed = proto_seed reproduces a training policy;
    any other value gives a policy family never seen in training. Returns (N, d_a)."""
    rows = jnp.asarray(rows)
    z = jnp.broadcast_to(jnp.asarray(code_bin, jnp.float32), (rows.shape[0], int(max_log_seed)))
    seeds = proto_seed_ints(z, rows, int(max_log_seed))
    return proto_latents(seeds, int(action_dim), float(u_clip), jax.random.PRNGKey(base_seed))


def two_way_decomposition(M):
    """Sum-of-squares split of M (P policies x C cells): grand mean, policy main effect,
    cell main effect, interaction. Returns fractions of the total centred sum of squares."""
    M = np.asarray(M, np.float64)
    mu = M.mean()
    row = M.mean(1, keepdims=True) - mu
    col = M.mean(0, keepdims=True) - mu
    inter = M - mu - row - col
    tot = ((M - mu) ** 2).sum()
    P, C = M.shape
    ss_p, ss_c, ss_i = (row ** 2).sum() * C, (col ** 2).sum() * P, (inter ** 2).sum()
    return {"total_var": float(tot / M.size), "frac_policy": float(ss_p / tot),
            "frac_cell": float(ss_c / tot), "frac_interaction": float(ss_i / tot)}


def participation_ratio_from_gram(G):
    ev = np.clip(np.linalg.eigvalsh(np.asarray(G, np.float64)), 0.0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300)), ev[::-1]


def participation_ratio(X, center):
    """(sum sv^2)^2 / sum sv^4 of the rows of X (optionally centred across rows)."""
    X = np.asarray(X, np.float64)
    if center:
        X = X - X.mean(0, keepdims=True)
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    pr, ev = participation_ratio_from_gram(G)
    return {"pr": pr, "top1_frac": float(ev[0] / max(ev.sum(), 1e-300)),
            "top5_eigs": [float(x) for x in ev[:5]]}


def within_group_fraction(V):
    """V (cells, K): fraction of the total variance of V that lies across the K entries
    of a cell (mean within-cell variance / total variance)."""
    V = np.asarray(V, np.float64)
    tot = V.var()
    return {"total_var": float(tot), "within_var": float(V.var(1).mean()),
            "frac_within": float(V.var(1).mean() / max(tot, 1e-300))}


def spread_ratio(V):
    """V (S states, G goals, U latents): mean variance across u at fixed (s, g) over mean
    variance across s at fixed (g, u)."""
    V = np.asarray(V, np.float64)
    var_u, var_s = V.var(2).mean(), V.var(0).mean()
    return {"var_across_u": float(var_u), "var_across_s": float(var_s),
            "ratio_u_over_s": float(var_u / max(var_s, 1e-300)),
            "std_ratio_u_over_s": float(np.sqrt(var_u / max(var_s, 1e-300)))}


def argmax_disagreement(Qs):
    """Qs (K coefficients, S states, U latents) -> (K, K) fraction of states whose argmax-u
    differs between coefficient k and l."""
    am = np.asarray(Qs).argmax(-1)
    return (am[:, None, :] != am[None, :, :]).mean(-1)


def empty_stats(D1):
    return {"n_off": 0.0, "n_diag": 0.0, "Spp": np.zeros((D1, D1)), "Spq": np.zeros((D1, D1)),
            "Sqq": np.zeros((D1, D1)), "mq": np.zeros(D1), "mp_diag": np.zeros(D1)}


def add_stats(acc, s):
    """Accumulate a mesh's sums (from mesh_sums) into acc."""
    for k in acc:
        acc[k] = acc[k] + np.asarray(s[k], np.float64)
    return acc


def mesh_sums(P, Q):
    """P, Q (N, N, D+1): augmented [phi, b] online and [phi_bar, b_bar] target meshes.
    Returns SUMS over the off-diagonal cells and the diagonal cells."""
    N = P.shape[0]
    off = (1.0 - jnp.eye(N)).reshape(-1)
    Pf, Qf = P.reshape(N * N, -1), Q.reshape(N * N, -1)
    Po = Pf * off[:, None]
    diag = jnp.arange(N)
    return {"n_off": jnp.sum(off), "n_diag": jnp.asarray(float(N)),
            "Spp": Po.T @ Pf, "Spq": Po.T @ Qf, "Sqq": (Qf * off[:, None]).T @ Qf,
            "mq": (Qf * off[:, None]).sum(0), "mp_diag": P[diag, diag].sum(0)}


def finalize(acc):
    n, nd = acc["n_off"], acc["n_diag"]
    return {"Spp": acc["Spp"] / n, "Spq": acc["Spq"] / n, "Sqq": acc["Sqq"] / n,
            "mq": acc["mq"] / n, "mp_diag": acc["mp_diag"] / nd}


def normalized_td_error(st, w, gamma):
    """Off-diagonal MSE of (M - gamma M_bar) over var(gamma M_bar), same w both sides."""
    W = np.concatenate([np.asarray(w, np.float64), [1.0]])
    mse = W @ st["Spp"] @ W - 2 * gamma * W @ st["Spq"] @ W + gamma ** 2 * W @ st["Sqq"] @ W
    var_t = gamma ** 2 * (W @ st["Sqq"] @ W - (st["mq"] @ W) ** 2)
    return {"nmse": float(mse / max(var_t, 1e-300)), "mse": float(mse), "target_var": float(var_t),
            "w_norm": float(np.linalg.norm(w))}


def td_fixed_point(st, gamma):
    """Closed-form semi-gradient TD fixed point in w: (A - gamma C) w = rhs, least squares."""
    D = st["Spp"].shape[0] - 1
    A, C = st["Spp"][:D, :D], st["Spq"][:D, :D]
    rhs = -st["Spp"][:D, D] + gamma * st["Spq"][:D, D] + (1 - gamma) * st["mp_diag"][:D]
    return np.linalg.lstsq(A - gamma * C, rhs, rcond=None)[0]


def fit_w_td(st, w0, gamma, steps, lr, tau, project):
    """Adam on the training TD loss 0.5*E_off[(M - gamma M_bar)^2] - (1-gamma) E_diag[M] in w,
    with a Polyak target copy of w (rate tau). project -> w renormalised to the sqrt(D) sphere
    after every step, as w(z) is. Returns final w (D,) as numpy."""
    import optax
    D = st["Spp"].shape[0] - 1
    Spp, Spq = jnp.asarray(st["Spp"], jnp.float32), jnp.asarray(st["Spq"], jnp.float32)
    mpd = jnp.asarray(st["mp_diag"], jnp.float32)
    radius = float(np.sqrt(D))
    tx = optax.adam(lr)

    def proj(w):
        return radius * w / jnp.linalg.norm(w) if project else w

    def grad(w, wt):
        W, Wt = jnp.append(w, 1.0), jnp.append(wt, 1.0)
        return (Spp @ W - gamma * Spq @ Wt)[:D] - (1 - gamma) * mpd[:D]

    def body(carry, _):
        w, wt, opt = carry
        upd, opt = tx.update(grad(w, wt), opt, w)
        w = proj(optax.apply_updates(w, upd))
        wt = tau * w + (1 - tau) * wt
        return (w, wt, opt), None

    w0 = proj(jnp.asarray(w0, jnp.float32))
    (w, _, _), _ = jax.lax.scan(body, (w0, w0, tx.init(w0)), None, length=int(steps))
    return np.asarray(w, np.float64)


def summarize(xs):
    xs = np.asarray(xs, np.float64)
    return {"mean": float(xs.mean()), "median": float(np.median(xs)), "min": float(xs.min()),
            "max": float(xs.max())}


# ------------------------------------------------------------------ checkpoint work
def load_dataset(cfg, env, seed):
    from envs.env_utils import make_env_and_datasets
    from tools.relabel_reward_rhat_psmgoal import select_preimage_u
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(env, frame_stack=None)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(cfg["preimage_path"]))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0]
    return {"obs": np.asarray(train_dataset["observations"], np.float32),
            "next": np.asarray(train_dataset["next_observations"], np.float32),
            "actions": np.asarray(train_dataset["actions"], np.float32),
            "rewards": np.asarray(train_dataset["rewards"], np.float32),
            "u": select_preimage_u(aug, bool(cfg["use_point_preimage"]), float(cfg["u_clip"]))}


def load_cfg(run_dir):
    import ml_collections

    from agents.psmgoal import get_config
    from main import _lists_to_tuples
    from tools.eval_checkpoint import merge_run_config
    base = json.loads(json.dumps(get_config().to_dict()))
    merged, prov = merge_run_config({"agent_name": "psmgoal", **base}, run_dir, set())
    return ml_collections.ConfigDict(_lists_to_tuples(merged)), prov


def action_check(agent, cfg, data, a, obs, u_prior, Wz, rng):
    """GPI-style readout Q_w(s, u) = mean_{g in G} M(s, u, g; w) over k_goals rewarding next
    states G, at the check-3 states and the 64 prior latents. Reports the spread of Q across u
    vs across s, and how often argmax_u changes when w changes: trained w(z) of several codes,
    the coef_source=regression w (lstsq of the reward on the goal-averaged phi at s', projected
    to the sqrt(D) sphere), w = 0 (b only), and random sphere w."""
    from agents.psmgoal import REWARDING_THRESHOLD
    D, u_clip = int(cfg["z_dim"]), float(cfg["u_clip"])
    r_all = data["rewards"] + a.reward_shift
    rew_rows = np.nonzero(r_all > REWARDING_THRESHOLD)[0]
    goals = jnp.asarray(data["next"][rng.choice(rew_rows, int(cfg["k_goals"]), replace=rew_rows.size < int(cfg["k_goals"]))])
    Kg = goals.shape[0]
    on_p = agent.basis.params

    @jax.jit
    def goal_avg(o, u):
        n = o.shape[0]
        phi, b = agent.basis(jnp.repeat(o, Kg, 0), jnp.repeat(u, Kg, 0), jnp.tile(goals, (n, 1)), params=on_p)
        return phi.reshape(n, Kg, D).mean(1), b.reshape(n, Kg).mean(1)

    # coef_source=regression, on reg_rows random rows (the agent uses infer_batch rows of a relabel batch)
    idx = rng.choice(data["obs"].shape[0], a.reg_rows, replace=False)
    feats = []
    for s0 in range(0, idx.size, 2048):
        sl = idx[s0:s0 + 2048]
        feats.append(np.asarray(goal_avg(jnp.asarray(data["next"][sl]), jnp.asarray(data["u"][sl]))[0], np.float64))
    Phi = np.concatenate(feats)
    r = r_all[idx].astype(np.float64)
    w_ls = np.linalg.lstsq(Phi, r, rcond=None)[0]
    w_reg = np.sqrt(D) * w_ls / np.linalg.norm(w_ls)
    reg_fit_corr = float(np.corrcoef(Phi @ w_ls, r)[0, 1]) if r.std() > 0 else float("nan")

    # Q over (state, u): (S, U, D) goal-averaged phi and (S, U) goal-averaged b
    S, U = obs.shape[0], u_prior.shape[0]
    ph = np.zeros((S, U, D))
    bb = np.zeros((S, U))
    for k in range(U):
        p_k, b_k = goal_avg(obs, jnp.broadcast_to(u_prior[k], (S, u_prior.shape[1])))
        ph[:, k], bb[:, k] = np.asarray(p_k, np.float64), np.asarray(b_k, np.float64)
    rand = np.asarray(jax.random.normal(jax.random.PRNGKey(a.seed + 5), (a.u_codes, D)), np.float64)
    rand = np.sqrt(D) * rand / np.linalg.norm(rand, axis=1, keepdims=True)
    names = [f"wz{i}" for i in range(Wz.shape[0])] + ["w_reg", "w_zero"] + [f"rand{i}" for i in range(rand.shape[0])]
    Ws = np.concatenate([Wz, w_reg[None], np.zeros((1, D)), rand])
    Qs = np.einsum("sud,kd->ksu", ph, Ws) + bb[None]
    dis = argmax_disagreement(Qs)
    nz, ir, i0 = Wz.shape[0], Wz.shape[0], Wz.shape[0] + 1
    ra = slice(Wz.shape[0] + 2, None)
    off = ~np.eye(nz, dtype=bool)
    rr = dis[ra, ra]
    spreads = {n: spread_ratio(Q[:, None, :]) for n, Q in zip(names, Qs)}
    return {
        "k_goals": Kg, "n_states": S, "u_draws": U, "reg_rows": int(idx.size),
        "reg_rewarding_rows": int((r > REWARDING_THRESHOLD).sum()), "reg_fit_corr": reg_fit_corr,
        "w_reg_cos_to_mean_wz": float(w_reg @ Wz.mean(0) / (np.linalg.norm(w_reg) * np.linalg.norm(Wz.mean(0)))),
        "Q_u_over_s_ratio": {n: spreads[n]["ratio_u_over_s"] for n in names},
        "Q_u_over_s_ratio_wz_mean": float(np.mean([spreads[n]["ratio_u_over_s"] for n in names[:nz]])),
        "argmax_change": {
            "wz_vs_wz_mean": float(dis[:nz, :nz][off].mean()) if nz > 1 else None,
            "wreg_vs_wz_mean": float(dis[ir, :nz].mean()),
            "wzero_vs_wz_mean": float(dis[i0, :nz].mean()),
            "wreg_vs_wzero": float(dis[ir, i0]),
            "rand_vs_rand_mean": float(rr[~np.eye(rr.shape[0], dtype=bool)].mean()) if rr.shape[0] > 1 else None,
            "wreg_vs_rand_mean": float(dis[ir, ra].mean()),
            "chance": float(1 - 1 / U),
        },
        "argmax_change_matrix": {"names": names, "matrix": dis.tolist()},
    }


def run_checkpoint(run_dir, epoch, cfg, data, a):
    from agents import agents
    from utils.flax_utils import restore_agent

    agent = agents["psmgoal"].create(a.seed, data["obs"][:1], data["actions"][:1], cfg)
    agent = restore_agent(agent, run_dir, epoch)
    D, width = int(cfg["z_dim"]), int(cfg["max_log_seed"])
    d_a, u_clip, gamma = int(data["u"].shape[1]), float(cfg["u_clip"]), float(cfg["discount"])
    train_seed = int(cfg["proto_seed"])
    assert a.new_proto_seed != train_seed
    on_p, tg_p = agent.basis.params, agent.target_basis

    @jax.jit
    def aug_mesh(params, o, u, g):
        N = o.shape[0]
        oo = jnp.repeat(o, N, 0)
        uu = jnp.repeat(u, N, 0)
        gg = jnp.tile(g, (N, 1))
        phi, b = agent.basis(oo, uu, gg, params=params)
        return jnp.concatenate([phi, b[:, None]], -1).reshape(N, N, D + 1)

    @jax.jit
    def sums_for_policy(P, next_o, g, rows, code_bin, base_seed):
        u_plus = policy_latents(code_bin, rows, width, d_a, u_clip, base_seed)
        Q = aug_mesh(tg_p, next_o, u_plus, g)
        return mesh_sums(P, Q)

    rng = np.random.default_rng(a.seed)
    n_rows = data["obs"].shape[0]
    n_pol = a.n_policies
    codes_train = rng.choice(2 ** width, n_pol, replace=False)
    codes_new = rng.choice(2 ** width, n_pol, replace=False)
    bits_train, bits_new = code_bits(codes_train, width), code_bits(codes_new, width)
    pols = [("train", i, bits_train[i], train_seed) for i in range(n_pol)] + \
           [("new", i, bits_new[i], a.new_proto_seed) for i in range(n_pol)]

    # ---- check 2: accumulate exact TD second moments per policy on fit / eval meshes
    t0 = time.time()
    acc = {split: [empty_stats(D + 1) for _ in pols] for split in ("fit", "eval")}
    for split, n_mesh in (("fit", a.fit_meshes), ("eval", a.eval_meshes)):
        for _ in range(n_mesh):
            idx = rng.choice(n_rows, a.mesh_n, replace=False)
            o, u, nx = (jnp.asarray(data[k][idx]) for k in ("obs", "u", "next"))
            P = aug_mesh(on_p, o, u, nx)
            for k, (_, _, bits, bs) in enumerate(pols):
                s = sums_for_policy(P, nx, nx, jnp.asarray(idx), jnp.asarray(bits), bs)
                add_stats(acc[split][k], jax.device_get(s))
    st = {split: [finalize(x) for x in acc[split]] for split in acc}
    t_stats = time.time() - t0

    w_trained = np.asarray(agent.w(jnp.asarray(bits_train)), np.float64)      # (n_pol, D)
    key = jax.random.PRNGKey(a.seed + 7)
    rows_out = []
    for k, (kind, i, bits, bs) in enumerate(pols):
        w0 = np.asarray(jax.random.normal(jax.random.fold_in(key, i), (D,)))   # same init for (b) and (c)
        fits = {
            "adam_free": fit_w_td(st["fit"][k], w0, gamma, a.fit_steps, a.fit_lr, a.fit_tau, False),
            "adam_sphere": fit_w_td(st["fit"][k], w0, gamma, a.fit_steps, a.fit_lr, a.fit_tau, True),
            "fixed_point": td_fixed_point(st["fit"][k], gamma),
        }
        rec = {"kind": kind, "i": i, "code": int((codes_train if kind == "train" else codes_new)[i]),
               "base_seed": int(bs)}
        for name, w in fits.items():
            rec[name] = {"eval": normalized_td_error(st["eval"][k], w, gamma),
                         "fit": normalized_td_error(st["fit"][k], w, gamma)}
        if kind == "train":
            rec["trained_w"] = {"eval": normalized_td_error(st["eval"][k], w_trained[i], gamma),
                                "fit": normalized_td_error(st["fit"][k], w_trained[i], gamma)}
        rows_out.append(rec)

    def agg(kind, field, split="eval", key_="nmse"):
        return summarize([r[field][split][key_] for r in rows_out if r["kind"] == kind])

    check2 = {
        "n_policies_each": n_pol, "mesh_n": a.mesh_n, "fit_meshes": a.fit_meshes,
        "eval_meshes": a.eval_meshes, "fit_steps": a.fit_steps, "fit_lr": a.fit_lr,
        "fit_tau": a.fit_tau, "train_proto_seed": train_seed, "new_proto_seed": a.new_proto_seed,
        "gamma": gamma, "seconds_stats": t_stats,
        "a_train_trained_w": {"nmse": agg("train", "trained_w"), "w_norm": agg("train", "trained_w", key_="w_norm")},
    }
    for name in ("adam_free", "adam_sphere", "fixed_point"):
        check2[f"b_train_refit_{name}"] = {"nmse": agg("train", name), "nmse_fit": agg("train", name, "fit"),
                                           "w_norm": agg("train", name, key_="w_norm")}
        check2[f"c_new_refit_{name}"] = {"nmse": agg("new", name), "nmse_fit": agg("new", name, "fit"),
                                         "w_norm": agg("new", name, key_="w_norm")}
    check2["per_policy"] = rows_out

    # ---- check 3: span
    idx = rng.choice(n_rows, a.span_mesh_n, replace=False)
    o, u, nx = (jnp.asarray(data[k][idx]) for k in ("obs", "u", "next"))
    P = np.asarray(aug_mesh(on_p, o, u, nx), np.float64).reshape(-1, D + 1)      # (cells, D+1)
    codes = rng.choice(2 ** width, a.span_codes, replace=False)
    Wz = np.asarray(agent.w(jnp.asarray(code_bits(codes, width))), np.float64)  # (Pz, D)
    M = Wz @ P[:, :D].T + P[:, D][None, :]                                        # (Pz, cells)
    dec = two_way_decomposition(M)
    # variance across 64 prior latents at fixed (s, s+), for a few codes' w(z)
    G = a.u_goals
    o_u, g_u = o, nx[:G]
    u_prior = jnp.clip(jax.random.normal(jax.random.PRNGKey(a.seed + 99), (a.u_draws, d_a)), -u_clip, u_clip)

    @jax.jit
    def mesh_over_u(uk):
        N = o_u.shape[0]
        oo = jnp.repeat(o_u, G, 0)
        gg = jnp.tile(g_u, (N, 1))
        uu = jnp.broadcast_to(uk, (N * G, d_a))
        phi, b = agent.basis(oo, uu, gg, params=on_p)
        return phi, b

    phis, bs_ = [], []
    for k in range(a.u_draws):
        ph, bb = mesh_over_u(u_prior[k])
        phis.append(np.asarray(ph, np.float64))
        bs_.append(np.asarray(bb, np.float64))
    phis, bs_ = np.stack(phis, 1), np.stack(bs_, 1)          # (cells_u, U, D), (cells_u, U)
    wz_u = Wz[:a.u_codes]
    per_code = [within_group_fraction(phis @ w + bs_) for w in wz_u]
    # the policy index at the same cells, with the data latent: variance across codes
    Pu = P.reshape(a.span_mesh_n, a.span_mesh_n, D + 1)[:, :G].reshape(-1, D + 1)
    Mpol = Pu[:, :D] @ Wz.T + Pu[:, D][:, None]              # (cells_u, Pz)
    check3 = {
        "mesh_n": a.span_mesh_n, "n_codes": a.span_codes, **{f"M_{k}": v for k, v in dec.items()},
        "w_norm_mean": float(np.linalg.norm(Wz, axis=1).mean()),
        "w_cos_to_mean": summarize(Wz @ Wz.mean(0) / (np.linalg.norm(Wz, axis=1) * np.linalg.norm(Wz.mean(0)))),
        "w_rel_spread": float(np.linalg.norm(Wz - Wz.mean(0), axis=1).mean() / np.linalg.norm(Wz, axis=1).mean()),
        "w_pr_uncentered": participation_ratio(Wz, center=False),
        "w_pr_centered": participation_ratio(Wz, center=True),
        "M_pr_uncentered": participation_ratio(M, center=False),
        "M_pr_centered_over_policies": participation_ratio(M, center=True),
        "phi_mesh_pr": participation_ratio(P[:, :D], center=False),
        "u_variation": {"n_states": a.span_mesh_n, "n_goals": G, "u_draws": a.u_draws,
                        "n_codes": len(per_code),
                        "frac_within_state_across_u": summarize([p["frac_within"] for p in per_code]),
                        "within_var_across_u": summarize([p["within_var"] for p in per_code]),
                        "total_var": summarize([p["total_var"] for p in per_code])},
        "policy_variation_same_cells": within_group_fraction(Mpol),
        "u_vs_s_spread_goals_next_states": summarize([spread_ratio((phis @ w + bs_).reshape(
            a.span_mesh_n, G, a.u_draws))["ratio_u_over_s"] for w in wz_u]),
    }
    check3["action_check"] = action_check(agent, cfg, data, a, o, u_prior, Wz[:a.u_codes], rng)
    return {"run_dir": run_dir, "epoch": epoch, "z_dim": D, "max_log_seed": width,
            "check2_heldout_policies": check2, "check3_span": check3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="glob of psmgoal run dirs")
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--report_dir", required=True, help="per-checkpoint JSONs go here")
    ap.add_argument("--report_out", required=True, help="summary JSON")
    ap.add_argument("--tag", default="diag_psmgoal_basis_policies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new_proto_seed", type=int, default=12345)
    ap.add_argument("--n_policies", type=int, default=16)
    ap.add_argument("--mesh_n", type=int, default=256)
    ap.add_argument("--fit_meshes", type=int, default=8)
    ap.add_argument("--eval_meshes", type=int, default=4)
    ap.add_argument("--fit_steps", type=int, default=5000)
    ap.add_argument("--fit_lr", type=float, default=1e-2)
    ap.add_argument("--fit_tau", type=float, default=0.01)
    ap.add_argument("--span_mesh_n", type=int, default=256)
    ap.add_argument("--span_codes", type=int, default=256)
    ap.add_argument("--u_goals", type=int, default=16)
    ap.add_argument("--u_draws", type=int, default=64)
    ap.add_argument("--u_codes", type=int, default=8)
    ap.add_argument("--reward_shift", type=float, default=1.0, help="eval_reward_shift of the runs")
    ap.add_argument("--reg_rows", type=int, default=20000)
    a = ap.parse_args()

    runs = sorted(glob.glob(a.run_dir))
    assert runs, f"--run_dir matched nothing: {a.run_dir}"
    cfg0, _ = load_cfg(runs[0].rstrip("/"))
    data = load_dataset(cfg0, a.env, a.seed)
    os.makedirs(a.report_dir, exist_ok=True)
    summary = {"args": vars(a), "rows": []}
    for run in runs:
        run = run.rstrip("/")
        cfg, prov = load_cfg(run)
        assert cfg["preimage_path"] == cfg0["preimage_path"]
        sd = os.path.basename(run)[:5]
        for ep in a.epochs:
            t0 = time.time()
            rep = run_checkpoint(run, ep, cfg, data, a)
            rep["flags_json"], rep["args"], rep["seconds"] = prov.get("flags_json"), vars(a), time.time() - t0
            out = os.path.join(a.report_dir, f"{a.tag}_{sd}_{ep}.json")
            with open(out, "w") as f:
                json.dump(rep, f, indent=2)
            c2, c3 = rep["check2_heldout_policies"], rep["check3_span"]
            row = {"seed": sd, "epoch": ep, "json": out,
                   "a_trained": c2["a_train_trained_w"]["nmse"]["mean"]}
            for name in ("adam_free", "adam_sphere", "fixed_point"):
                row[f"b_{name}"] = c2[f"b_train_refit_{name}"]["nmse"]["mean"]
                row[f"c_{name}"] = c2[f"c_new_refit_{name}"]["nmse"]["mean"]
            row.update({"frac_policy": c3["M_frac_policy"], "frac_cell": c3["M_frac_cell"],
                        "frac_interaction": c3["M_frac_interaction"],
                        "frac_u_within": c3["u_variation"]["frac_within_state_across_u"]["mean"],
                        "w_pr": c3["w_pr_uncentered"]["pr"], "w_rel_spread": c3["w_rel_spread"], "w_pr_centered": c3["w_pr_centered"]["pr"],
                        "M_pr": c3["M_pr_uncentered"]["pr"],
                        "M_pr_centered": c3["M_pr_centered_over_policies"]["pr"],
                        "u_over_s_spread": c3["u_vs_s_spread_goals_next_states"]["mean"],
                        "Q_u_over_s_wz": c3["action_check"]["Q_u_over_s_ratio_wz_mean"],
                        "Q_u_over_s_wreg": c3["action_check"]["Q_u_over_s_ratio"]["w_reg"],
                        **{f"argmax_{k}": v for k, v in c3["action_check"]["argmax_change"].items()},
                        "seconds": rep["seconds"]})
            summary["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(a.report_out, "w") as f:
                json.dump(summary, f, indent=2)
    print("summary ->", a.report_out)


if __name__ == "__main__":
    main()
