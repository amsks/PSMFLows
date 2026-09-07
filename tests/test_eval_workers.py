"""The pure half of tools/eval_checkpoint.py's parallel-episode split.

No jax, no env, no GPU: these cover the arithmetic that decides WHICH episodes each
worker runs, WHAT seed it runs them under, and how the shards fold back into the single
counts the report is built from. The rollout itself is covered on hardware (see
docs/design/2026-09-07-gpu-packing.md); what can silently go wrong here is off-by-one --
a shard list that does not sum to `eval_episodes` reports a success rate over the wrong
denominator, and a seed plan that collides across `seed` makes two "independent" evals
share episode inits.
"""
import pytest

from tools.eval_checkpoint import (
    aggregate_worker_results,
    resolve_num_workers,
    split_episodes,
    wilson,
    worker_seed_plan,
)

# --- split ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_ep,n_w", [(500, 1), (500, 2), (500, 8), (500, 7), (500, 16),
                                      (50, 8), (13, 5), (8, 8)])
def test_split_is_a_partition(n_ep, n_w):
    shards = split_episodes(n_ep, n_w)
    assert len(shards) == n_w
    assert sum(shards) == n_ep, "the shards must cover exactly the requested episodes"
    assert min(shards) >= 1, "a zero-episode worker still builds an env and a flow"
    assert max(shards) - min(shards) <= 1, "shards must be balanced to within one episode"


def test_split_remainder_goes_to_low_indices():
    # 500 over 8 is 62 r4: workers 0-3 get 63, workers 4-7 get 62. Deterministic, because
    # `per_episode_success` is the concatenation of the shards in worker order.
    assert split_episodes(500, 8) == [63, 63, 63, 63, 62, 62, 62, 62]
    assert split_episodes(500, 1) == [500]
    assert split_episodes(500, 500) == [1] * 500


def test_split_rejects_more_workers_than_episodes():
    with pytest.raises(ValueError):
        split_episodes(4, 8)
    with pytest.raises(ValueError):
        split_episodes(500, 0)


# --- seeds ---------------------------------------------------------------------------

def test_single_worker_seed_is_the_eval_seed_itself():
    """The property that keeps every pre-2026-09-07 eval500 JSON reproducible."""
    for seed in (0, 1, 7, 12345):
        assert worker_seed_plan(seed, 1) == [seed]


def test_seed_plans_do_not_collide_across_eval_seeds():
    n_w = 8
    plans = [worker_seed_plan(s, n_w) for s in range(6)]
    flat = [s for plan in plans for s in plan]
    assert len(set(flat)) == len(flat), "two eval seeds must not share an episode-init stream"
    assert plans[0] == list(range(8))
    assert plans[1] == list(range(8, 16))


def test_seed_plan_rejects_zero_workers():
    with pytest.raises(ValueError):
        worker_seed_plan(0, 0)


# --- aggregation ---------------------------------------------------------------------

def _shard(worker, per_ep, seed=0):
    return {"worker": worker, "seed": seed, "num_episodes": len(per_ep),
            "per_episode_raw": list(per_ep), "stats_success": 0.0, "seconds": 1.0}


def test_aggregate_concatenates_in_worker_order_not_arrival_order():
    # Pool.map preserves order, but the aggregator must not depend on that: a shard list
    # that arrives shuffled has to produce the same per_episode_success sequence.
    a, b, c = _shard(0, [1.0, 0.0]), _shard(1, [0.0, 0.0]), _shard(2, [1.0, 1.0])
    per_ep, k, n, field = aggregate_worker_results([c, a, b])
    assert per_ep == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert (k, n) == (3, 6)
    assert field == pytest.approx(0.5)


def test_aggregate_single_worker_matches_the_unsplit_computation():
    """N=1 must reduce to exactly what the pre-2026-09-07 tool computed."""
    per_episode = [1.0] * 89 + [0.0] * 411          # the affine-strict antmaze sd1 @250k number
    per_ep, k, n, _ = aggregate_worker_results([_shard(0, per_episode)])
    assert (k, n) == (89, 500)
    assert per_ep == per_episode
    assert wilson(k, n) == (0.147, 0.2139)


def test_aggregate_counts_are_thresholded_not_summed():
    # `success` from OGBench is a float; the count is `> 0.5`, matching the single-process
    # tool. A shard of 0.6-valued episodes is 100% success, not 60%.
    _, k, n, field = aggregate_worker_results([_shard(0, [0.6, 0.4, 0.51])])
    assert (k, n) == (2, 3)
    assert field == pytest.approx((0.6 + 0.4 + 0.51) / 3)


def test_aggregate_rejects_a_missing_worker():
    with pytest.raises(ValueError, match="contiguous"):
        aggregate_worker_results([_shard(0, [1.0]), _shard(2, [0.0])])


def test_aggregate_rejects_a_short_shard():
    """A worker that returned fewer episodes than it was given is a silent denominator bug."""
    bad = _shard(0, [1.0, 0.0])
    bad["num_episodes"] = 5
    with pytest.raises(ValueError, match="was given 5 episodes"):
        aggregate_worker_results([bad])


def test_aggregate_falls_back_to_the_episode_weighted_mean():
    # An env with no per-step `success` field: only `evaluate`'s average survives, and it
    # has to be weighted by shard size -- 500 over 8 is not an even split.
    shards = [{"worker": 0, "seed": 0, "num_episodes": 63, "per_episode_raw": [],
               "stats_success": 1.0, "seconds": 1.0},
              {"worker": 1, "seed": 1, "num_episodes": 62, "per_episode_raw": [],
               "stats_success": 0.0, "seconds": 1.0}]
    per_ep, k, n, field = aggregate_worker_results(shards)
    assert per_ep == []
    assert n == 125
    assert k == 63          # not 62 (an unweighted 0.5 * 125 would give 62)
    assert field == pytest.approx(63 / 125)


# --- worker-count resolution ---------------------------------------------------------

def test_cli_beats_env_beats_default():
    assert resolve_num_workers(4, "8", 500) == 4          # explicit hydra override wins
    assert resolve_num_workers(1, "8", 500) == 8          # config default -> env
    assert resolve_num_workers(1, None, 500) == 1
    assert resolve_num_workers(None, None, 500) == 1
    assert resolve_num_workers(1, "", 500) == 1


def test_worker_count_is_clamped_to_the_episode_count():
    assert resolve_num_workers(1, "8", 4) == 4
    assert resolve_num_workers(64, None, 500) == 64
    assert resolve_num_workers(0, None, 500) == 1
    assert resolve_num_workers(-3, None, 500) == 1


def test_garbage_worker_counts_do_not_crash_an_eval():
    assert resolve_num_workers("nonsense", None, 500) == 1
    assert resolve_num_workers(1, "nonsense", 500) == 1
