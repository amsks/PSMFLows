"""CPU tests for the pure metrics of tools/diag_rank_agreement_psmgoal.py (synthetic arrays)."""
import numpy as np
import pytest

from tools.diag_rank_agreement_psmgoal import (
    aggregate,
    bucket_index,
    data_rank,
    desc_ranks,
    rank_metrics,
    row_spearman,
    seed_ci,
    subset_spearman,
)


def test_identical_scores_agree_fully():
    q = np.random.default_rng(0).normal(size=(50, 64))
    m = rank_metrics(q, q)
    assert np.allclose(m["spearman"], 1.0)
    assert np.all(m["top1_agree"] == 1.0)
    assert np.all(m["critic_top1_rank_in_M"] == 1.0)
    assert np.all(m["M_top1_rank_in_critic"] == 1.0)
    assert np.all(m["neg_spearman"] == 0.0)


def test_reversed_scores_disagree_fully():
    q = np.random.default_rng(1).normal(size=(20, 64))
    m = rank_metrics(q, -q)
    assert np.allclose(m["spearman"], -1.0)
    assert np.all(m["top1_agree"] == 0.0)
    assert np.all(m["critic_top1_rank_in_M"] == 64.0)
    assert np.all(m["M_top1_rank_in_critic"] == 64.0)
    assert np.all(m["neg_spearman"] == 1.0)


def test_monotone_transform_is_invisible():
    q = np.random.default_rng(2).normal(size=(10, 64))
    m = rank_metrics(q, np.exp(3 * q) + 5)
    assert np.allclose(m["spearman"], 1.0) and np.all(m["top1_agree"] == 1.0)


def test_independent_scores_are_at_chance():
    rng = np.random.default_rng(3)
    m = rank_metrics(rng.normal(size=(20000, 64)), rng.normal(size=(20000, 64)))
    assert abs(m["spearman"].mean()) < 0.01
    assert abs(m["top1_agree"].mean() - 1 / 64) < 0.005
    assert abs(m["critic_top1_rank_in_M"].mean() - 32.5) < 0.5
    assert abs(m["M_top1_rank_in_critic"].mean() - 32.5) < 0.5
    assert abs(m["neg_spearman"].mean() - 0.5) < 0.02


def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(4)
    a = rng.integers(0, 5, size=(30, 64)).astype(float)
    b = a + rng.normal(size=a.shape)
    got = row_spearman(a, b)
    want = [spearmanr(a[i], b[i]).statistic for i in range(30)]
    assert np.allclose(got, want)


def test_constant_row_gives_nan_and_is_skipped():
    q = np.random.default_rng(5).normal(size=(3, 8))
    qc = q.copy()
    qc[1] = 0.0
    m = rank_metrics(qc, q)
    assert np.isnan(m["spearman"][1]) and np.isnan(m["neg_spearman"][1])
    assert np.isfinite(m["spearman"][[0, 2]]).all()


def test_desc_ranks_and_known_cross_rank():
    qm = np.array([[3.0, 2.0, 1.0, 0.0]])
    qb = np.array([[0.0, 1.0, 5.0, 2.0]])     # critic's best is index 2, M's 3rd
    m = rank_metrics(qm, qb)
    assert np.allclose(desc_ranks(qm), [[1, 2, 3, 4]])
    assert m["critic_top1_rank_in_M"][0] == 3.0
    assert m["M_top1_rank_in_critic"][0] == 4.0     # M's best (index 0) is the critic's worst
    assert m["top1_agree"][0] == 0.0


def test_data_rank_and_subset():
    q = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 3.0, 2.0, 0.0]])
    assert np.allclose(data_rank(q), [1.0, 4.0])
    keep = np.array([[True] * 4, [True, True, False, False]])
    s = subset_spearman(q, q, keep, min_n=3)
    assert s[0] == pytest.approx(1.0) and np.isnan(s[1])


def test_bucket_index():
    assert list(bucket_index([0, 24, 25, 49, 50, 99, 100, 199, 200, 500])) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_aggregate_task_mean_then_seed_ci():
    cells = []
    for s, base in ((0, 0.1), (1, 0.2), (2, 0.3)):
        for t in range(1, 6):
            cells.append({"seed": s, "task": t, "tables": {
                "dataset/M_vs_QW": {"n_states": 10, "spearman": base + 0.01 * t}}})
    agg = aggregate(cells)
    tab = agg["tables"]["dataset/M_vs_QW"]
    assert tab["spearman"]["n_seeds"] == 3
    assert tab["spearman"]["mean"] == pytest.approx(0.23)
    assert tab["per_seed_five_task_mean"]["0"]["spearman"] == pytest.approx(0.13)
    assert tab["n_states_total"] == 150
    assert seed_ci([0.1, 0.2, 0.3])["ci95"] == pytest.approx(0.2484, abs=1e-3)
