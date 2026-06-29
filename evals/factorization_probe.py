"""FB factorization-ablation + readout-ceiling probe — pure helpers.

Decides *why* FB fails at cube transport by measuring, on frozen embeddings,
whether terminal-placement information is (B1) present in the data, (B3-repr)
present in the learned representation, (B3-form) extractable by the bilinear
FᵀB form vs a non-separable head, or (B2) recoverable but unused downstream.

numpy in / numpy metrics out; torch used internally. No jax, no sklearn — runs
under .venv. The only cross-module dependency is a read-only import of `_auc`
from evals.training_value (no existing files are modified by this probe)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evals.training_value import _auc


# ── ground-truth value target ──────────────────────────────────────────────
def mc_return_to_go(cube_traj: np.ndarray, goal_xyz: np.ndarray,
                    gamma: float, thresh: float) -> np.ndarray:
    """Per-step discounted return-to-go under a sparse goal-reach reward for
    ONE episode: out[t] = gamma**(k*-t) where k* is the first step >= t with
    ||cube_k - goal|| < thresh, else 0.0 if the goal is never reached from t.
    `cube_traj` = [T,3] cube xyz; `goal_xyz` = [3]. This is the ground-truth
    value of the sparse-reward goal-reaching MDP along the data trajectory —
    the quantity V(s) should rank, with no Euclidean-distance confound."""
    cube_traj = np.asarray(cube_traj, np.float64).reshape(-1, 3)
    goal_xyz = np.asarray(goal_xyz, np.float64).reshape(3)
    d = np.linalg.norm(cube_traj - goal_xyz, axis=1)
    within = d < thresh
    T = len(d)
    out = np.zeros(T, dtype=np.float64)
    next_reach = -np.ones(T, dtype=np.int64)
    nxt = -1
    for t in range(T - 1, -1, -1):
        if within[t]:
            nxt = t
        next_reach[t] = nxt
    reached = next_reach >= 0
    out[reached] = gamma ** (next_reach[reached] - np.arange(T)[reached])
    return out


# ── readout ceiling (linear vs MLP, regression R² / classification AUC) ─────
def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _torch_fit(Xtr, ytr, Xte, head, *, task, epochs=3000, lr=1e-2, wd=1e-4,
               patience=60, seed=0):
    """Train `head` (nn.Module: in_dim->1) with an internal 80/20 train/val
    split and early stopping (keep best-val params), so the returned test
    predictions reflect the best *generalization* — the true readout ceiling,
    robust to both under-convergence (linear) and overfitting (MLP)."""
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)
    n = len(Xtr_t); n_val = max(1, int(0.2 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    vi, ti = perm[:n_val], perm[n_val:]
    Xt, yt, Xv, yv = Xtr_t[ti], ytr_t[ti], Xtr_t[vi], ytr_t[vi]
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        loss_fn(head(Xt), yt).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vloss = float(loss_fn(head(Xv), yv))
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    with torch.no_grad():
        return head(Xte_t).reshape(-1).numpy()


def _make_head(in_dim: int, kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_dim, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    raise ValueError(f"unknown kind {kind!r}")


def _ridge_predict(Xtr, ytr, Xte, lam=1.0):
    """Closed-form ridge regression (optimal linear, no optimizer underfit;
    matches the paper's _ridge_r2). Bias is unregularized."""
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte1 = np.hstack([Xte, np.ones((len(Xte), 1))])
    reg = lam * np.eye(Xtr1.shape[1]); reg[-1, -1] = 0.0
    w = np.linalg.solve(Xtr1.T @ Xtr1 + reg, Xtr1.T @ ytr)
    return Xte1 @ w


def fit_eval_readout(Xtr, ytr, Xte, yte, *, kind: str, task: str, seed=0) -> dict:
    """Fit a `kind` ('linear'|'mlp') readout to predict y from X; eval held-out.
    task='regression' -> score is R^2; task='classification' -> score is AUC.
    Linear regression uses closed-form ridge (the true linear ceiling); MLP and
    all classification use an early-stopped torch head."""
    Xtr, ytr = np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64)
    Xte, yte = np.asarray(Xte, np.float64), np.asarray(yte, np.float64)
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    if kind == "linear" and task == "regression":
        pred = _ridge_predict(Xtr_s, ytr, Xte_s)
    else:
        head = _make_head(Xtr_s.shape[1], kind)
        pred = _torch_fit(Xtr_s, ytr, Xte_s, head, task=task, seed=seed)
    if task == "classification":
        score = _auc(pred[yte > 0.5], pred[yte <= 0.5])
    else:
        score = r2_score(yte, pred)
    return {"kind": kind, "task": task, "score": float(score), "n_test": int(len(yte))}


# ── bilinear vs joint head ─────────────────────────────────────────────────
def _split_idx(n, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); cut = int(n * frac)
    return idx[:cut], idx[cut:]


def compare_bilinear_vs_joint(F, B, y, *, seed=0) -> dict:
    """Does the placement label y separate under the BILINEAR family
    (score = FᵀW B, i.e. linear in the outer product F⊗B) vs a JOINT MLP over
    concat([F,B])? Both fit to predict y; report held-out AUC and the gap.
    gap = auc_joint - auc_bilinear. A large positive gap => placement is NOT a
    bilinear function of (F,B) => the FᵀB form is the bottleneck (B3-form)."""
    F = np.asarray(F, np.float64); B = np.asarray(B, np.float64)
    y = np.asarray(y, np.float64)
    tr, te = _split_idx(len(F), seed=seed)
    outer = (F[:, :, None] * B[:, None, :]).reshape(len(F), -1)   # [N, dF*dB]
    bil = fit_eval_readout(outer[tr], y[tr], outer[te], y[te],
                           kind="linear", task="classification", seed=seed)
    joint_in = np.concatenate([F, B], axis=1)
    joint = fit_eval_readout(joint_in[tr], y[tr], joint_in[te], y[te],
                             kind="mlp", task="classification", seed=seed)
    return {"auc_bilinear": bil["score"], "auc_joint": joint["score"],
            "gap": float(joint["score"] - bil["score"]), "n_test": int(len(te))}


# ── per-region aggregation + verdict ───────────────────────────────────────
def readout_ceiling_table(df, feats: dict, *, targets=("placement", "mc_return"),
                          min_per_group=80, seed=0) -> "pd.DataFrame":
    """For each feature matrix in `feats` (e.g. {'B':..,'left_enc':..,'raw':..}),
    each target column in `df`, each region (+'all'), and kind in {linear,mlp},
    fit a held-out readout and record the score. `df` rows align with feats rows.
    target 'placement' (0/1) -> AUC; anything else -> R^2 regression."""
    rng = np.random.default_rng(seed)
    rows = []
    regions = ["all"] + sorted(df["region"].unique().tolist())
    for fname, X in feats.items():
        X = np.asarray(X, np.float64)
        for region in regions:
            mask = (np.ones(len(df), bool) if region == "all"
                    else (df["region"] == region).to_numpy())
            if mask.sum() < min_per_group:
                continue
            idx = np.where(mask)[0]
            perm = rng.permutation(len(idx)); cut = len(idx) // 2
            tr, te = idx[perm[:cut]], idx[perm[cut:]]
            for tgt in targets:
                y = df[tgt].to_numpy(np.float64)
                # binary target -> classification (AUC); else regression (R^2)
                task = ("classification" if np.isin(np.unique(y), [0.0, 1.0]).all()
                        else "regression")
                if task == "classification" and (y[te].sum() == 0 or y[te].sum() == len(te)):
                    continue
                for kind in ("linear", "mlp"):
                    out = fit_eval_readout(X[tr], y[tr], X[te], y[te],
                                           kind=kind, task=task, seed=seed)
                    rows.append({"feature": fname, "target": tgt, "region": region,
                                 "kind": kind, "task": task, "score": out["score"],
                                 "n_test": out["n_test"]})
    return pd.DataFrame(rows)


def classify_bucket(ceiling_df, *, bilinear_gap: float, region="transport",
                    target="placement", lo=0.65, hi=0.80, gap_thr=0.10) -> dict:
    """Map the transport-phase readout ceilings + bilinear gap to a bucket.
    lo: AUC at/below which a readout 'fails'. Decision tree (see plan)."""
    def s(feature, kind):
        m = ceiling_df[(ceiling_df.feature == feature) & (ceiling_df.kind == kind)
                       & (ceiling_df.region == region) & (ceiling_df.target == target)]
        return float(m["score"].iloc[0]) if len(m) else float("nan")
    raw_mlp, b_mlp, b_lin = s("raw", "mlp"), s("B", "mlp"), s("B", "linear")
    missing = [k for k, v in {"raw/mlp": raw_mlp, "B/mlp": b_mlp, "B/linear": b_lin}.items()
               if np.isnan(v)]
    if missing:
        return {"bucket": "insufficient-data",
                "why": f"missing {region}-phase '{target}' readouts {missing}; increase "
                       f"--n-states or use a denser target (the verdict is unreliable)"}
    gap_exceeds = (not np.isnan(bilinear_gap)) and bilinear_gap > gap_thr
    if raw_mlp < lo:
        return {"bucket": "B1",
                "why": f"raw-obs MLP AUC {raw_mlp:.2f}<{lo}: placement not recoverable "
                       f"from state (data/label problem)"}
    if b_mlp < lo or (raw_mlp - b_mlp) > 0.20:
        return {"bucket": "B3-representation",
                "why": f"B-MLP AUC {b_mlp:.2f} << raw {raw_mlp:.2f}: the representation "
                       f"discarded placement info"}
    if b_lin < lo or gap_exceeds:
        return {"bucket": "B3-form",
                "why": f"B-linear AUC {b_lin:.2f}<{lo} or bilinear gap {bilinear_gap:.2f}"
                       f">{gap_thr}: info in B is non-separable; FᵀB (linear in B) cannot use it"}
    return {"bucket": "B2",
            "why": f"placement linearly+bilinearly recoverable (B-lin {b_lin:.2f}, "
                   f"gap {bilinear_gap:.2f}); failure is downstream (ρ-sampling / z=B(g) / actor)"}


def classify_bucket_regression(ceiling_df, *, region="transport", target="d",
                               lo=0.30, repr_gap=0.30, form_gap=0.15) -> dict:
    """Verdict from a REGRESSION readout ceiling (R^2) of a static geometric
    target (default cube-to-goal distance d). Mirrors classify_bucket but with
    R^2 thresholds: lo = R^2 below which a readout 'can't predict'.
      B1            : raw-obs MLP can't predict the geometry (not in the state).
      B3-repr       : B-MLP R^2 << raw-MLP R^2 (B discarded the geometry).
      B3-form       : B-MLP R^2 - B-linear R^2 > form_gap (geometry is in B but
                      NONLINEAR; the bilinear value, linear in B, can't use it),
                      or B-linear R^2 < lo.
      B2            : geometry is linearly present in B; failure is downstream."""
    def s(feature, kind):
        m = ceiling_df[(ceiling_df.feature == feature) & (ceiling_df.kind == kind)
                       & (ceiling_df.region == region) & (ceiling_df.target == target)]
        return float(m["score"].iloc[0]) if len(m) else float("nan")
    raw_mlp, b_mlp, b_lin = s("raw", "mlp"), s("B", "mlp"), s("B", "linear")
    missing = [k for k, v in {"raw/mlp": raw_mlp, "B/mlp": b_mlp, "B/linear": b_lin}.items()
               if np.isnan(v)]
    if missing:
        return {"bucket": "insufficient-data",
                "why": f"missing {region}-phase '{target}' readouts {missing}; increase --n-states"}
    if raw_mlp < lo:
        return {"bucket": "B1",
                "why": f"raw-obs MLP R^2 {raw_mlp:.2f}<{lo}: geometry '{target}' not in the state"}
    if b_mlp < lo or (raw_mlp - b_mlp) > repr_gap:
        return {"bucket": "B3-representation",
                "why": f"B-MLP R^2 {b_mlp:.2f} << raw {raw_mlp:.2f}: B discarded the terminal geometry"}
    if b_lin < lo or (b_mlp - b_lin) > form_gap:
        return {"bucket": "B3-form",
                "why": f"B-MLP R^2 {b_mlp:.2f} but B-linear R^2 {b_lin:.2f}: geometry is in B "
                       f"but non-linear; the bilinear FᵀB value (linear in B) cannot read it"}
    return {"bucket": "B2",
            "why": f"d is linearly recoverable from B (R^2 {b_lin:.2f}); failure is downstream "
                   f"(ρ-sampling / z=B(g) / actor)"}
