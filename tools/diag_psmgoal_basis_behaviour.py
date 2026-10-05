"""Check 1 (behaviour): does psmgoal's frozen basis (phi, b) model where the data collector goes?

Freeze phi and b of a restored psmgoal checkpoint. Fit one free coefficient w (z_dim) by the
same loss as `PSMGoalAgent.measure_loss` -- squared TD on the off-diagonal of the state x
next-state mesh plus the -(1-gamma) diagonal pull -- with only w free. The bootstrap latent
u+ is the preimage of the dataset's NEXT action (row i+1 of the same episode), so the fitted
M(s,u,s+) = phi^T w + b is the successor measure of the data collector's policy. The target
coefficient is a Polyak copy of w (tau = the run's tau). w is renormalised to the sqrt(D)
sphere after each step (as the trained w(z) and the Lagrangian inference are); a second fit
with w left free is reported beside it.

Fit on the first (1 - holdout_frac) of the episodes. On the held-out episodes, score
M(s_t, u_t, s+) for positives s+ = s_{t+k} (same episode, k in --ks) and negatives s+ = a
random dataset state, with the same anchor (s_t, u_t). Reported per k and per coefficient:
pooled ROC-AUC, the paired win rate P[M(pos) > M(neg)] at the same anchor, and mean M for
positives and negatives. Coefficients scored:
  fit_sphere, fit_free   the fitted w
  rand_unit              random unit vectors * sqrt(D) (mean/std over --n_ctrl draws)
  wz                     w(z) of the trained PolicyCoefficient at random codes z (same)
  b_only                 w = 0
The held-out TD loss (data bootstrap) of every coefficient is reported too.

Output is JSON via --report_out.

Run (CPU smoke):
  JAX_PLATFORMS=cpu OGBENCH_DATASET_DIR=... .venv/bin/python tools/diag_psmgoal_basis_behaviour.py \
      --run_dir '$PSM_DATA/exp/PSMFLows/psmgoal_lift_cube_gc/sd000_*' --epoch 250000 \
      --fit_steps 20 --batch 32 --n_eval 256 --report_out /tmp/x.json
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


# ------------------------------------------------------------------ pure numpy helpers
def episode_ids(terminals):
    """Episode index per row; an episode ends at a row with terminals > 0.5."""
    t = np.asarray(terminals) > 0.5
    ids = np.zeros(t.shape[0], np.int64)
    ids[1:] = np.cumsum(t[:-1])
    return ids


def next_row_valid(terminals):
    """Mask of rows i whose row i+1 is in the same episode (so u_{i+1} is the next latent)."""
    ep = episode_ids(terminals)
    valid = np.zeros(ep.shape[0], bool)
    valid[:-1] = ep[1:] == ep[:-1]
    return valid


def future_valid(terminals, k):
    """Mask of rows t whose row t+k is in the same episode."""
    ep = episode_ids(terminals)
    valid = np.zeros(ep.shape[0], bool)
    if k < ep.shape[0]:
        valid[:-k] = ep[k:] == ep[:-k]
    return valid


def split_episodes(terminals, holdout_frac):
    """Row masks (train, heldout): the last holdout_frac of episodes (by order) are held out."""
    ep = episode_ids(terminals)
    n_ep = int(ep.max()) + 1
    cut = round(n_ep * (1.0 - holdout_frac))
    return ep < cut, ep >= cut


def roc_auc(pos, neg):
    """Mann-Whitney ROC-AUC of pos above neg, ties counted half."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.shape[0], np.float64)
    sv = allv[order]
    i = 0
    while i < sv.shape[0]:           # average ranks over ties
        j = i
        while j + 1 < sv.shape[0] and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_p, n_n = pos.shape[0], neg.shape[0]
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def score_stats(pos, neg):
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    return {"auc": roc_auc(pos, neg),
            "paired_win": float(np.mean((pos > neg) + 0.5 * (pos == neg))),
            "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean())}


# ------------------------------------------------------------------ jax: loss and fit
def mesh(basis_fn, obs, u, goals):
    """phi (N, G, D), b (N, G) of basis_fn over the (s_i, u_i) x g_j mesh."""
    import jax.numpy as jnp
    N, G = obs.shape[0], goals.shape[0]
    o = jnp.broadcast_to(obs[:, None], (N, G, obs.shape[-1])).reshape(N * G, -1)
    uu = jnp.broadcast_to(u[:, None], (N, G, u.shape[-1])).reshape(N * G, -1)
    g = jnp.broadcast_to(goals[None], (N, G, goals.shape[-1])).reshape(N * G, -1)
    phi, b = basis_fn(o, uu, g)
    return phi.reshape(N, G, -1), b.reshape(N, G)


def td_loss(phi, b, phi_t, b_t, w, w_t, gamma):
    """measure_loss with one shared w: off-diagonal squared TD to gamma * target plus the
    -(1-gamma) diagonal pull. phi (N,N,D) at (s_i,u_i,s'_j); phi_t at (s'_i,u+_i,s'_j)."""
    import jax
    import jax.numpy as jnp
    M = phi @ w + b
    target = jax.lax.stop_gradient(phi_t @ w_t + b_t)
    N = M.shape[0]
    off = 1.0 - jnp.eye(N)
    off_diag = 0.5 * jnp.sum(((M - gamma * target) ** 2) * off) / jnp.maximum(jnp.sum(off), 1.0)
    diag = -(1.0 - gamma) * jnp.mean(jnp.diagonal(M))
    return off_diag + diag, (off_diag, diag)


def fit_w(basis_fn, sample_batch, D, gamma, tau, steps, lr, sphere, seed, log_every=250):
    """Fit one w with the basis frozen. sample_batch(rng) -> (obs, u, next_obs, u_next).
    Returns (w, history). history rows: step, loss, off_diag, diag, w_norm."""
    import jax
    import jax.numpy as jnp
    import optax

    key = jax.random.PRNGKey(seed)
    w0 = jax.random.normal(key, (D,))
    w0 = jnp.sqrt(float(D)) * w0 / jnp.linalg.norm(w0)
    opt = optax.adam(lr)

    @jax.jit
    def step(w, w_t, opt_state, obs, u, next_obs, u_next):
        phi, b = mesh(basis_fn, obs, u, next_obs)
        phi_t, b_t = mesh(basis_fn, next_obs, u_next, next_obs)
        phi, b, phi_t, b_t = [jax.lax.stop_gradient(x) for x in (phi, b, phi_t, b_t)]
        (loss, (off, dg)), g = jax.value_and_grad(td_loss, argnums=4, has_aux=True)(
            phi, b, phi_t, b_t, w, w_t, gamma)
        upd, opt_state = opt.update(g, opt_state, w)
        w = optax.apply_updates(w, upd)
        if sphere:
            w = jnp.sqrt(float(D)) * w / (jnp.linalg.norm(w) + 1e-8)
        w_t = (1.0 - tau) * w_t + tau * w
        return w, w_t, opt_state, loss, off, dg

    rng = np.random.default_rng(seed)
    w, w_t, opt_state = w0, w0, opt.init(w0)
    hist, acc = [], []
    for s in range(1, steps + 1):
        obs, u, nxt, un = sample_batch(rng)
        w, w_t, opt_state, loss, off, dg = step(w, w_t, opt_state, jnp.asarray(obs), jnp.asarray(u),
                                                 jnp.asarray(nxt), jnp.asarray(un))
        acc.append((float(loss), float(off), float(dg)))
        if s % log_every == 0 or s == steps:
            a = np.mean(acc, 0)
            hist.append({"step": s, "loss": float(a[0]), "off_diag": float(a[1]), "diag": float(a[2]),
                         "w_norm": float(jnp.linalg.norm(w))})
            acc = []
    return np.asarray(w), hist


def eval_td(basis_fn, batches, ws, gamma):
    """Held-out TD loss (data bootstrap, target = the same w) for each named w in ws."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def phis(obs, u, nxt, un):
        return mesh(basis_fn, obs, u, nxt) + mesh(basis_fn, nxt, un, nxt)

    out = {k: [] for k in ws}
    for obs, u, nxt, un in batches:
        phi, b, phi_t, b_t = phis(*(jnp.asarray(x) for x in (obs, u, nxt, un)))
        for k, w in ws.items():
            loss, (off, _) = td_loss(phi, b, phi_t, b_t, jnp.asarray(w), jnp.asarray(w), gamma)
            out[k].append((float(loss), float(off)))
    return {k: {"loss": float(np.mean([v[0] for v in vals])), "off_diag": float(np.mean([v[1] for v in vals]))}
            for k, vals in out.items()}


def score_pairs(basis_fn, obs, u, goals, chunk=4096):
    """phi (N, D), b (N,) at row-aligned triples (s_i, u_i, g_i)."""
    import jax
    import jax.numpy as jnp
    f = jax.jit(basis_fn)
    P, B = [], []
    for s in range(0, obs.shape[0], chunk):
        p, b = f(jnp.asarray(obs[s:s + chunk]), jnp.asarray(u[s:s + chunk]), jnp.asarray(goals[s:s + chunk]))
        P.append(np.asarray(p, np.float64))
        B.append(np.asarray(b, np.float64))
    return np.concatenate(P), np.concatenate(B)


def behaviour_eval(basis_fn, obs, u, terminals, heldout, ks, n_eval, ws_single, ws_multi, rng):
    """Per k: AUC etc. for positives s_{t+k} vs random-state negatives at the same anchor.
    ws_single: {name: (D,)}; ws_multi: {name: (n, D)} reported as mean/std over the n draws."""
    res = {}
    N = obs.shape[0]
    for k in ks:
        cand = np.nonzero(heldout & future_valid(terminals, k))[0]
        t = rng.choice(cand, min(n_eval, cand.size), replace=False)
        neg_rows = rng.integers(0, N, t.size)
        pp, bp = score_pairs(basis_fn, obs[t], u[t], obs[t + k])
        pn, bn = score_pairs(basis_fn, obs[t], u[t], obs[neg_rows])
        row = {"n": int(t.size)}
        for name, w in ws_single.items():
            row[name] = score_stats(pp @ w + bp, pn @ w + bn)
        for name, W in ws_multi.items():
            per = [score_stats(pp @ w + bp, pn @ w + bn) for w in W]
            row[name] = {f"{m}_{agg}": float(fn([p[m] for p in per]))
                         for m in ("auc", "paired_win", "mean_pos", "mean_neg")
                         for agg, fn in (("mean", np.mean), ("std", np.std))}
        res[str(k)] = row
    return res


# ------------------------------------------------------------------ checkpoint driver
def load(run_glob, epoch, env, seed):
    """Restore a psmgoal checkpoint and its preimage-spliced dataset (as diag_psmgoal_gram.py)."""
    import ml_collections

    from agents import agents
    from agents.psmgoal import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from tools.relabel_reward_rhat_psmgoal import select_preimage_u
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    runs = glob.glob(run_glob)
    assert len(runs) == 1, f"--run_dir matched {len(runs)}: {runs}"
    run_dir = runs[0].rstrip("/")
    base = json.loads(json.dumps(get_config().to_dict()))
    merged, prov = merge_run_config({"agent_name": "psmgoal", **base}, run_dir, _cli_agent_keys())
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    use_point, u_clip = bool(cfg["use_point_preimage"]), float(cfg["u_clip"])

    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(env, frame_stack=None)
    train_dataset = dict(train_dataset)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(cfg["preimage_path"]))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0]
    for k in aug:
        if k.startswith("noise_preimage"):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = use_point
    ex = ds.sample(1)
    agent = agents["psmgoal"].create(seed, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, run_dir, epoch)
    data = {"obs": np.asarray(ds["observations"], np.float32),
            "next": np.asarray(ds["next_observations"], np.float32),
            "terminals": np.asarray(ds["terminals"], np.float32),
            "u": select_preimage_u(aug, use_point, u_clip)}
    return agent, cfg, data, run_dir, prov


def run_checkpoint(a, run_glob, epoch):
    import jax
    import jax.numpy as jnp

    from utils.psm_proto import sample_z_bin

    t0 = time.time()
    agent, cfg, d, run_dir, prov = load(run_glob, epoch, a.env, a.seed)
    D, gamma, tau = int(cfg["z_dim"]), float(cfg["discount"]), float(cfg["tau"])
    obs, nxt, term, u = d["obs"], d["next"], d["terminals"], d["u"]
    params = agent.basis.params

    def basis_fn(o, uu, g):
        return agent.basis(o, uu, g, params=params)

    nv = next_row_valid(term)
    # consistency of the pairing: next_observations[i] should equal observations[i+1]
    idx_nv = np.nonzero(nv)[0]
    match = float(np.mean(np.all(np.isclose(nxt[idx_nv], obs[idx_nv + 1], atol=1e-5), -1)))
    train_m, held_m = split_episodes(term, a.holdout_frac)
    fit_rows = np.nonzero(train_m & nv)[0]
    held_rows = np.nonzero(held_m & nv)[0]

    def sampler(rows):
        def f(rng):
            i = rng.choice(rows, a.batch, replace=False)
            return obs[i], u[i], nxt[i], u[i + 1]
        return f

    fits, hists = {}, {}
    for name, sphere in (("fit_sphere", True), ("fit_free", False)):
        w, h = fit_w(basis_fn, sampler(fit_rows), D, gamma, tau, a.fit_steps, a.lr, sphere, a.seed)
        fits[name], hists[name] = w, h
        print(f"[{name}] final {h[-1]}", flush=True)

    rng = np.random.default_rng(a.seed + 1)
    rand_unit = rng.standard_normal((a.n_ctrl, D))
    rand_unit = np.sqrt(D) * rand_unit / np.linalg.norm(rand_unit, axis=1, keepdims=True)
    z = sample_z_bin(jax.random.PRNGKey(a.seed + 2), a.n_ctrl, int(cfg["max_log_seed"]))
    wz = np.asarray(agent.w(z, params=agent.w.params), np.float64)
    ws_single = {"fit_sphere": fits["fit_sphere"].astype(np.float64),
                 "fit_free": fits["fit_free"].astype(np.float64), "b_only": np.zeros(D)}
    ws_multi = {"rand_unit": rand_unit, "wz": wz}

    hs = sampler(held_rows)
    held_batches = [hs(rng) for _ in range(a.td_eval_batches)]
    td_ws = dict(ws_single)
    td_ws.update({f"rand_unit_{i}": w for i, w in enumerate(rand_unit[:4])})
    td_ws.update({f"wz_{i}": w for i, w in enumerate(wz[:4])})
    td_ws = {k: jnp.asarray(v, jnp.float32) for k, v in td_ws.items()}
    td_held = eval_td(basis_fn, held_batches, td_ws, gamma)

    beh = behaviour_eval(basis_fn, obs, u, term, held_m, a.ks, a.n_eval, ws_single, ws_multi, rng)
    report = {
        "run_dir": run_dir, "epoch": epoch, "env": a.env, "z_dim": D, "gamma": gamma, "tau": tau,
        "flags_json": prov.get("flags_json"),
        "data": {"rows": int(obs.shape[0]), "episodes": int(episode_ids(term).max() + 1),
                 "fit_rows": int(fit_rows.size), "heldout_rows": int(held_rows.size),
                 "next_obs_equals_obs_next_row_frac": match},
        "fit": {"steps": a.fit_steps, "batch": a.batch, "lr": a.lr, "history": hists,
                "final_td_loss": {k: h[-1]["loss"] for k, h in hists.items()},
                "w_fit_sphere": fits["fit_sphere"].tolist()},
        "cos_fit_sphere_vs_wz_mean": float(np.mean(wz @ fits["fit_sphere"]) / D),
        "heldout_td": td_held,
        "behaviour": beh,
        "seconds": time.time() - t0,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, nargs="+", help="psmgoal run dir globs (one match each)")
    ap.add_argument("--epoch", type=int, required=True, nargs="+")
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--report_out", default=None, help="single run: JSON path")
    ap.add_argument("--out_dir", default=None, help="multi: <out_dir>/<prefix>_sd00k_<epoch>.json + summary")
    ap.add_argument("--prefix", default="diag_psmgoal_basis_behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--fit_steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--n_ctrl", type=int, default=8)
    ap.add_argument("--td_eval_batches", type=int, default=8)
    a = ap.parse_args()
    assert a.report_out or a.out_dir

    summary = []
    for rg in a.run_dir:
        for ep in a.epoch:
            rep = run_checkpoint(a, rg, ep)
            sd = os.path.basename(rep["run_dir"])[:5]
            out = a.report_out if a.report_out else os.path.join(a.out_dir, f"{a.prefix}_{sd}_{ep}.json")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump(rep, f, indent=2)
            print("report ->", out, flush=True)
            row = {"seed": sd, "epoch": ep, "json": out,
                   "final_td_loss": rep["fit"]["final_td_loss"],
                   "heldout_td": {k: v["loss"] for k, v in rep["heldout_td"].items()
                                  if k in ("fit_sphere", "fit_free", "b_only", "rand_unit_0", "wz_0")}}
            for k, r in rep["behaviour"].items():
                row[f"k{k}"] = {"fit_sphere": r["fit_sphere"]["auc"], "fit_free": r["fit_free"]["auc"],
                                "b_only": r["b_only"]["auc"], "rand_unit": r["rand_unit"]["auc_mean"],
                                "wz": r["wz"]["auc_mean"],
                                "fit_sphere_paired": r["fit_sphere"]["paired_win"],
                                "b_only_paired": r["b_only"]["paired_win"]}
            summary.append(row)
            print(json.dumps(row), flush=True)
    if a.out_dir:
        p = os.path.join(a.out_dir, f"{a.prefix}_summary.json")
        with open(p, "w") as f:
            json.dump(summary, f, indent=2)
        print("summary ->", p)


if __name__ == "__main__":
    main()
