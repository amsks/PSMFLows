"""evals/representation_profile.py — why does FB fail at transport?

Four probes (value landscape, z-decoding+sparsity, B-resolution,
coverage) over FB checkpoints. Pure logic here is unit-tested with
stubs; MuJoCo rollouts reuse evals.phase_probe. No training/eval code
is modified. See
docs/superpowers/specs/2026-05-18-fb-representation-failure-profile-design.md
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from evals._profile_core import (  # noqa: E402  (re-exported for callers)
    _region,
    _spearman,
    probe_coverage,
    transport_mask,
)
from evals.phase_probe import Thresholds  # noqa: E402


@torch.no_grad()
def _ensemble_q(F: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """F [P, B, z_dim], z [z_dim] or [B, z_dim] -> Q [B] (mean over heads)."""
    if z.dim() == 1:
        z = z.view(1, 1, -1)
    elif z.dim() == 2:
        z = z.unsqueeze(0)
    q = (F * z).sum(-1)            # [P, B]
    return q.mean(0)              # [B]


def _tile_z(z: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast one global task vector to one row per state.

    The FB nets concatenate z with obs on the feature axis (and the
    flow actor draws ``noises = randn((z.shape[0], action_dim))``), so
    z must carry the same batch dim as obs. The probes apply a single
    task z across all T states of an episode.
    """
    z = torch.as_tensor(z)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[0] == batch:
        return z
    if z.shape[0] == 1:
        return z.expand(batch, -1)
    raise ValueError(f"z batch {z.shape[0]} != obs batch {batch}")


@torch.no_grad()
def q_values(model, obs: torch.Tensor, action: torch.Tensor,
             z: torch.Tensor) -> np.ndarray:
    zb = _tile_z(z, obs.shape[0]).to(obs.device).float()
    F = model.forward_map(obs, zb, action)
    return _ensemble_q(F, zb).cpu().numpy()


@torch.no_grad()
def v_values(model, agent, obs: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    zb = _tile_z(z, obs.shape[0]).to(obs.device).float()
    action = agent.act(obs, zb)
    if isinstance(action, np.ndarray):
        action = torch.as_tensor(action, dtype=torch.float32)
    return q_values(model, obs, action.to(obs.device).float(), zb)


def rollout_for_profile(env, agent, z, n_episodes: int, thr,
                        seed_offset: int = 0) -> List[Dict[str, Any]]:
    """S0 rollouts with obs/action recorded, each tagged with outcome,
    per-step cube->goal distance, and a transport-region mask."""
    from evals.phase_probe import classify_phases, rollout_with_phase_signals

    eps = rollout_with_phase_signals(
        env, agent, z, n_episodes, thr, scenario="S0", record_obs=True)
    out = []
    for ep in eps:
        cls = classify_phases(ep, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            # split post-contact failure into maintain (cube dropped below the
            # lift threshold) vs transport (held but not delivered), matching
            # the terminal failure decomposition.
            outcome = ("maintain_fail"
                       if cls["final_cube_lift"] < thr.delta_lift
                       else "transport_fail")
        else:
            outcome = "other"
        cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
        goal = np.asarray(ep["goal"], dtype=np.float64).reshape(3)
        d = np.linalg.norm(cube - goal, axis=1)
        out.append({
            "obs": ep["obs"], "action": ep["action"],
            "cube": cube, "goal": goal, "d": d,
            "eff": np.asarray(ep["eff"], np.float64).reshape(-1, 3),
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "table_z": float(ep["table_z"]),
            "transport_mask": transport_mask(ep, thr),
            "outcome": outcome, "length": ep["length"],
        })
    return out


def probe_value_landscape(model, agent, episodes, z, thr=None):
    """Per-episode Spearman rho of V vs (-d) over the transport region
    (nan unless >=5 steps with non-degenerate d and V), plus a long-form
    (outcome, d, V) frame for binned plots."""
    import pandas as pd

    thr = thr or Thresholds()
    per_ep_rows, per_step_rows = [], []
    for i, ep in enumerate(episodes):
        obs = torch.as_tensor(np.asarray(ep["obs"]), dtype=torch.float32)
        if obs.shape[0] == 0:
            continue
        V = v_values(model, agent, obs, z)             # [T]
        d = np.asarray(ep["d"], np.float64)
        m = np.asarray(ep["transport_mask"], bool)
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        eff_xy = np.asarray(ep["eff"], np.float64).reshape(-1, 3)[:, :2]
        region = _region(ep, thr)
        for ti, (dd, vv, mm, cxy, exy, rg) in enumerate(
                zip(d, V, m, cube_xy, eff_xy, region)):
            per_step_rows.append({"episode": i, "t": int(ti),
                                  "outcome": ep["outcome"],
                                  "region": str(rg),
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm),
                                  "cube_x": float(cxy[0]),
                                  "cube_y": float(cxy[1]),
                                  "eef_x": float(exy[0]),
                                  "eef_y": float(exy[1])})
        n_t = int(m.sum())
        if (n_t >= 5 and np.ptp(d[m]) > 1e-9 and np.ptp(V[m]) > 1e-9):
            rho = _spearman(V[m], -d[m])
            v_secure = float(V[m][0])
            v_end = float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        per_ep_rows.append({"episode": i, "outcome": ep["outcome"],
                            "rho_V_negd": rho, "V_at_secure": v_secure,
                            "V_at_end": v_end,
                            "n_transport_steps": n_t})
    return pd.DataFrame(per_ep_rows), pd.DataFrame(per_step_rows)


@torch.no_grad()
def probe_z_decoding(model, next_obs: torch.Tensor, physics: np.ndarray,
                     goal: np.ndarray, z: torch.Tensor,
                     relabel_metrics: Dict[str, float],
                     topk: int = 256,
                     goal_thresh: float = 0.04) -> Dict[str, float]:
    """How sparse are goal states, and does z point at 'placed' states?"""
    B = model.backward_map(next_obs.float())              # [N, z_dim]
    scores = (B * z.view(1, -1)).sum(-1).cpu().numpy()    # B·z
    cube = np.asarray(physics, np.float64)[:, 14:17]
    d = np.linalg.norm(cube - np.asarray(goal, np.float64).reshape(1, 3),
                        axis=1)
    k = int(min(topk, len(scores)))
    idx = np.argsort(-scores)[:k]
    ns = float(relabel_metrics.get("relabel_reward#num_samples", 0.0)) or 1.0
    return {
        "relabel_pos_frac": float(
            relabel_metrics.get("relabel_reward#nonzero", 0.0)) / ns,
        "topk_mean_d": float(d[idx].mean()),
        "topk_pct_at_goal": float(np.mean(d[idx] < goal_thresh)),
        "topk": k,
    }


def _ridge_r2(X: np.ndarray, y: np.ndarray, seed: int,
              alpha: float = 1.0) -> float:
    """Closed-form ridge with a 70/30 split; returns held-out R²."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    if len(y) < 4 or np.ptp(y) < 1e-12:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1))])
    Xte = np.hstack([X[te], np.ones((len(te), 1))])
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ y[tr])
    pred = Xte @ w
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _lda_acc(X: np.ndarray, lbl: np.ndarray) -> float:
    """Fisher-LDA train=test accuracy for a 2-class problem."""
    X = np.asarray(X, np.float64)
    cls = np.unique(lbl)
    if len(cls) != 2:
        return float("nan")
    mu = [X[lbl == c].mean(0) for c in cls]
    Sw = sum(np.cov(X[lbl == c], rowvar=False) * (np.sum(lbl == c) - 1)
             for c in cls)
    Sw = np.atleast_2d(Sw) + 1e-6 * np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mu[1] - mu[0])
    proj = X @ w
    thr = 0.5 * (proj[lbl == cls[0]].mean() + proj[lbl == cls[1]].mean())
    pred = np.where(proj > thr, cls[1], cls[0])
    return float((pred == lbl).mean())


def probe_b_resolution(B: np.ndarray, d: np.ndarray,
                       seed: int = 0) -> Dict[str, float]:
    """Ridge R²(B->d) and placed-vs-near-miss linear separability."""
    if hasattr(B, "detach"):
        B = B.detach().cpu().numpy()
    B = np.asarray(B, np.float64)
    d = np.asarray(d, np.float64)
    r2 = _ridge_r2(B, d, seed)
    placed = d < 0.04
    near = (d >= 0.04) & (d < 0.12)
    sel = placed | near
    if placed.sum() >= 2 and near.sum() >= 2:
        acc = _lda_acc(B[sel], placed[sel].astype(int))
    else:
        acc = float("nan")
    return {"r2": r2, "placed_vs_near_acc": acc,
            "n_placed": int(placed.sum()), "n_near": int(near.sum())}


def run_representation_profile(*, model, agent, infer_z, make_env,
                               sample_buffer, relabel_fn_for, goal_for,
                               tasks, n_episodes, thr, buffer_sample,
                               topk, seed, coverage_feature="obs"):
    """Run the four probes per task. `infer_z(task)->(z,relabel_metrics)`,
    `make_env(task)->env`, `sample_buffer(n)->{next_obs,physics,action}`,
    `relabel_fn_for(task)->fn`, `goal_for(task)->goal[3]`. Returns a dict
    of five DataFrames (value_landscape, value_steps, z_decoding,
    b_resolution, coverage), each with a `task` column."""
    import pandas as pd

    vl, vs, zd, br, cv = [], [], [], [], []
    for task in tasks:
        z, relabel_metrics = infer_z(task)
        goal = np.asarray(goal_for(task), np.float64).reshape(3)

        env = make_env(task)
        try:
            eps = rollout_for_profile(env, agent, z, n_episodes, thr)
        finally:
            if hasattr(env, "close"):
                env.close()
        per_ep, per_step = probe_value_landscape(model, agent, eps, z)
        for df in (per_ep, per_step):
            df.insert(0, "task", task)
        vl.append(per_ep)
        vs.append(per_step)

        smp = sample_buffer(buffer_sample)
        next_obs = smp["next_obs"]
        physics = np.asarray(smp["physics"], np.float32)
        zdec = probe_z_decoding(model, next_obs, physics, goal, z,
                                relabel_metrics, topk=topk)
        zdec["task"] = task
        zd.append(zdec)

        B = model.backward_map(
            next_obs.float() if hasattr(next_obs, "float") else
            torch.as_tensor(next_obs, dtype=torch.float32))
        d_off = np.linalg.norm(
            physics[:, 14:17] - goal.reshape(1, 3), axis=1)
        bres = probe_b_resolution(B, d_off, seed=seed)
        bres["task"] = task
        br.append(bres)

        if coverage_feature == "cube":
            cov_ref = physics[:, 14:17]
        else:
            cov_ref = (smp["next_obs"].detach().cpu().numpy()
                       if hasattr(smp["next_obs"], "detach")
                       else np.asarray(smp["next_obs"]))
        cov = probe_coverage(eps, cov_ref, thr, seed=seed,
                             feature=coverage_feature)
        cov.insert(0, "task", task)
        cv.append(cov)

    return {
        "value_landscape": pd.concat(vl, ignore_index=True),
        "value_steps": pd.concat(vs, ignore_index=True),
        "z_decoding": pd.DataFrame(zd),
        "b_resolution": pd.DataFrame(br),
        "coverage": pd.concat(cv, ignore_index=True),
    }
