"""Unit tests for evals/factorization_probe.py — the FB factorization-ablation
+ readout-ceiling probe helpers. Pure (numpy/torch), runs under .venv."""
import numpy as np
import pandas as pd

from evals.factorization_probe import (
    mc_return_to_go, r2_score, fit_eval_readout, compare_bilinear_vs_joint,
    readout_ceiling_table, classify_bucket, classify_bucket_regression,
)


# ── mc_return_to_go ────────────────────────────────────────────────────────
def test_reaches_goal_gives_gamma_powers():
    goal = np.array([1.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [0.4, 0, 0], [0.7, 0, 0], [1.0, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    np.testing.assert_allclose(out, [0.9**3, 0.9**2, 0.9**1, 0.9**0], rtol=1e-6)


def test_never_reaches_is_zero():
    goal = np.array([5.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_already_at_goal_is_one():
    goal = np.array([0.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    assert out[0] == 1.0


# ── fit_eval_readout / r2_score ────────────────────────────────────────────
def _split(X, y, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); cut = int(len(X) * frac)
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def test_linear_recovers_linear_regression_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6))
    y = X @ np.array([1.0, -2, 0.5, 0, 3, -1]) + 0.01 * rng.normal(size=800)
    Xtr, ytr, Xte, yte = _split(X, y)
    out = fit_eval_readout(Xtr, ytr, Xte, yte, kind="linear", task="regression")
    assert out["score"] > 0.95


def test_linear_fails_nonlinear_but_mlp_succeeds_classification():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.float64)
    Xtr, ytr, Xte, yte = _split(X, y)
    lin = fit_eval_readout(Xtr, ytr, Xte, yte, kind="linear", task="classification")
    mlp = fit_eval_readout(Xtr, ytr, Xte, yte, kind="mlp", task="classification")
    assert lin["score"] < 0.65
    assert mlp["score"] > 0.90


def test_r2_score_matches_definition():
    y = np.array([1.0, 2, 3, 4]); pred = np.array([1.1, 1.9, 3.2, 3.8])
    expected = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert abs(r2_score(y, pred) - expected) < 1e-9


# ── compare_bilinear_vs_joint ──────────────────────────────────────────────
def test_bilinear_target_both_succeed():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(2000, 5)); B = rng.normal(size=(2000, 5))
    W = rng.normal(size=(5, 5))
    score = np.einsum("ni,ij,nj->n", F, W, B)
    y = (score > np.median(score)).astype(np.float64)
    out = compare_bilinear_vs_joint(F, B, y, seed=0)
    assert out["auc_bilinear"] > 0.85
    assert out["auc_joint"] > 0.85
    assert out["gap"] < 0.10


def test_nonbilinear_target_joint_wins():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(3000, 5)); B = rng.normal(size=(3000, 5))
    nrm = np.linalg.norm(F - B, axis=1)
    y = (nrm > np.median(nrm)).astype(np.float64)
    out = compare_bilinear_vs_joint(F, B, y, seed=0)
    assert out["auc_joint"] - out["auc_bilinear"] > 0.10


# ── readout_ceiling_table ──────────────────────────────────────────────────
def test_readout_ceiling_table_shape_and_keys():
    rng = np.random.default_rng(4)
    n = 1200
    df = pd.DataFrame({
        "region": rng.choice(["reach", "grasp", "transport"], n),
        "placement": rng.integers(0, 2, n).astype(float),
        "mc_return": rng.random(n),
    })
    feats = {"B": rng.normal(size=(n, 8)), "raw": rng.normal(size=(n, 12))}
    tab = readout_ceiling_table(df, feats, targets=("placement", "mc_return"), seed=0)
    assert set(tab.columns) >= {"feature", "target", "region", "kind", "score", "n_test"}
    assert set(tab["region"]) <= {"reach", "grasp", "transport", "all"}
    assert set(tab["kind"]) == {"linear", "mlp"}
    assert {"B", "raw"} <= set(tab["feature"])


# ── classify_bucket ────────────────────────────────────────────────────────
def _ceiling(raw_mlp, b_mlp, b_lin, region="transport", target="placement"):
    return pd.DataFrame([
        {"feature": "raw", "target": target, "region": region, "kind": "mlp", "score": raw_mlp},
        {"feature": "B", "target": target, "region": region, "kind": "mlp", "score": b_mlp},
        {"feature": "B", "target": target, "region": region, "kind": "linear", "score": b_lin},
    ])


def test_bucket_b1_when_raw_cannot_predict():
    v = classify_bucket(_ceiling(0.55, 0.55, 0.52), bilinear_gap=0.0)
    assert v["bucket"] == "B1"


def test_bucket_b3_repr_when_b_loses_to_raw():
    v = classify_bucket(_ceiling(0.95, 0.60, 0.55), bilinear_gap=0.02)
    assert v["bucket"] == "B3-representation"


def test_bucket_b3_form_when_b_nonlinear_or_gap_large():
    v = classify_bucket(_ceiling(0.95, 0.93, 0.60), bilinear_gap=0.20)
    assert v["bucket"] == "B3-form"


def test_bucket_b2_when_b_linear_and_bilinear_ok():
    v = classify_bucket(_ceiling(0.95, 0.93, 0.90), bilinear_gap=0.03)
    assert v["bucket"] == "B2"


def test_bucket_insufficient_when_readout_missing():
    df = pd.DataFrame([
        {"feature": "raw", "target": "placement", "region": "transport", "kind": "mlp", "score": 0.95},
        {"feature": "B", "target": "placement", "region": "transport", "kind": "mlp", "score": 0.93},
    ])  # no B/linear row -> b_lin is nan
    v = classify_bucket(df, bilinear_gap=float("nan"))
    assert v["bucket"] == "insufficient-data"


def test_bucket_b2_with_nan_gap_but_linear_present():
    # gap nan must not crash or force B3-form; B-linear high -> B2
    v = classify_bucket(_ceiling(0.95, 0.93, 0.90), bilinear_gap=float("nan"))
    assert v["bucket"] == "B2"


# ── classify_bucket_regression (R^2 on geometric target d) ─────────────────
def _ceiling_r2(raw_mlp, b_mlp, b_lin, target="d"):
    return pd.DataFrame([
        {"feature": "raw", "target": target, "region": "transport", "kind": "mlp", "score": raw_mlp},
        {"feature": "B", "target": target, "region": "transport", "kind": "mlp", "score": b_mlp},
        {"feature": "B", "target": target, "region": "transport", "kind": "linear", "score": b_lin},
    ])


def test_reg_b1_when_raw_cannot_predict_geometry():
    assert classify_bucket_regression(_ceiling_r2(0.10, 0.08, 0.05))["bucket"] == "B1"


def test_reg_b3_repr_when_b_loses_geometry():
    assert classify_bucket_regression(_ceiling_r2(0.95, 0.40, 0.38))["bucket"] == "B3-representation"


def test_reg_b3_form_when_b_nonlinear():
    # B-MLP recovers geometry, B-linear lags by > form_gap -> bilinear can't read it
    assert classify_bucket_regression(_ceiling_r2(0.95, 0.80, 0.39))["bucket"] == "B3-form"


def test_reg_b2_when_linear_recovers():
    assert classify_bucket_regression(_ceiling_r2(0.95, 0.88, 0.85))["bucket"] == "B2"


def test_reg_insufficient_when_missing():
    df = pd.DataFrame([{"feature": "raw", "target": "d", "region": "transport", "kind": "mlp", "score": 0.9}])
    assert classify_bucket_regression(df)["bucket"] == "insufficient-data"
