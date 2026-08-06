"""GO/NO-GO gate for hindsight skill conditioning: does c actually steer the rollout?

Stage A can now train G(s, c, u) with c = a hindsight-window observation (skill_cond,
skill_window in configs/agent/fql.yaml; see agents/fql.py `_actor_obs` and
utils/datasets.py `add_skill_targets`). Conditioning on c is a bet: if commanding a
DIFFERENT c produces the SAME rollout as commanding the true c, the concat is decorative
and every downstream use of skill conditioning (Stage B preimages, Stage C selection)
inherits a latent that carries no signal.

This tool cannot reset OGBench envs to arbitrary dataset states, so it does not roll a
model forward from a dataset row. Instead it rolls the REAL env from its OWN reset state
under three commanded-c conditions and measures how much of the initial state-to-c gap
each one closes:
  - true:     c = a dataset skill-target plausibly reachable from this reset (nearest of
              a small pool of the K sampled skill targets, by L2 distance to the reset ob)
  - shuffled: c = the "true" c commanded for a DIFFERENT reset (the control: same
              statistics, wrong target)
  - stay:     c = the reset observation itself (a "do nothing" command; a conditioned net
              cannot be run unconditioned, so this is the second control)
Rolling `true` closer than `shuffled` is the signature that c is doing something.

Reports (JSON): per-condition mean/median gap-closed and min-dist, plus the headline gate
`coherence_ratio` = mean gap-closed(true) / max(mean gap-closed(shuffled), eps). Gate
PASSES iff coherence_ratio > 2 and mean gap-closed(true) > 0.5. Also a figure: per-
condition gap-closed distributions (matplotlib, Agg backend, style matches
tools/latent_reachability.py).

Run: JAX_PLATFORMS='' MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
    .venv/bin/python tools/validate_skill_coherence.py agent=fql \
    env_name=pointmaze-medium-navigate-singletask-task1-v0 \
    restore_path='/var/local/amsks/exp/<skill_cond_flow_run_dir>' restore_epoch=500000
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import add_skill_targets
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

K = 32               # constant: dataset rows / commanded skills sampled
NEAREST_POOL = 5      # "true" c is drawn from the this-many nearest-to-reset candidates
RATIO_GATE = 2.0       # coherence_ratio must exceed this
GAP_GATE = 0.5          # mean gap-closed(true) must exceed this
EPS = 1e-8


# --- pure functions (importable without hydra/env/jax side effects) -------------------

def episode_ends(terminals):
    """End-of-episode index for every row. MUST match utils.datasets.add_skill_targets
    exactly -- this is used only to pick rows with enough horizon left; the actual c
    values always come from add_skill_targets itself, never a local recomputation."""
    terminals = np.asarray(terminals)
    n = terminals.shape[0]
    ends = np.nonzero(terminals > 0.5)[0]
    if ends.size == 0 or ends[-1] != n - 1:
        # The dataset may not mark the final row as terminal; it still ends an episode.
        ends = np.concatenate([ends, [n - 1]])
    idx = np.arange(n)
    return ends[np.searchsorted(ends, idx, side='left')]


def rows_with_full_horizon(terminals, window):
    """Boolean mask: rows i with >= `window` steps left in THEIR OWN episode, i.e. the
    hindsight skill target at i is unclipped (add_skill_targets(i) == i + window)."""
    end = episode_ends(terminals)
    idx = np.arange(len(np.asarray(terminals)))
    return (idx + window) <= end


def state_distance(a, b):
    """L2 distance over the first min(dims) dims (matches latent_reachability's
    goal-distance convention: auxiliary when obs/c dims mismatch, exact when they match)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    nd = min(a.shape[-1], b.shape[-1])
    return float(np.linalg.norm(a[..., :nd] - b[..., :nd]))


def gap_closed_frac(initial_dist, min_dist, eps=EPS):
    """Fraction of the initial state->c gap closed by the best step of the rollout.

    1.0 = reached c exactly; 0.0 = never got closer than the start; negative = ended up
    FARTHER than the start at every step (can happen for `stay`, where initial_dist ~ 0
    and any drift away from the reset registers as strongly negative)."""
    return 1.0 - min_dist / max(float(initial_dist), eps)


def coherence_ratio(mean_gap_true, mean_gap_shuffled, eps=EPS):
    """The gate statistic: how much better `true` closes the gap than `shuffled` does.
    max(., eps) in the denominator so a shuffled mean of ~0 (or negative) doesn't blow up
    or flip sign -- a near-zero/negative shuffled baseline should read as a LARGE ratio
    (true is doing real work), not a division error."""
    return float(mean_gap_true) / max(float(mean_gap_shuffled), eps)


def gate_decision(mean_gap_true, ratio, ratio_thresh=RATIO_GATE, gap_thresh=GAP_GATE):
    """GO/NO-GO: c must both beat the shuffled control by a wide margin AND close most
    of the gap in absolute terms -- either alone can be misleading (a tiny shuffled mean
    makes the ratio huge for free; a middling true mean can still out-ratio a bad control)."""
    return bool(ratio > ratio_thresh and mean_gap_true > gap_thresh)


def nearest_indices(query, candidates, k):
    """Indices into `candidates` of the k closest to `query`, by state_distance."""
    dists = np.array([state_distance(query, c) for c in candidates])
    return np.argsort(dists)[:max(1, min(k, len(candidates)))]


# --- env-dependent rollout (not unit-tested; guarded behind main()) -------------------

def _rollout(eval_env, onestep, c, d_a, num_steps, reset_seed, rng):
    ob, _ = eval_env.reset(seed=reset_seed)
    initial_dist = state_distance(ob, c)
    min_dist, final_dist = initial_dist, initial_dist
    for _ in range(num_steps):
        u = rng.standard_normal(d_a).astype(np.float32)  # fresh u ~ N(0,I) every step
        oc = np.concatenate([np.asarray(ob), np.asarray(c)], axis=-1)[None]
        a = np.clip(np.asarray(onestep(oc, u[None]))[0], -1, 1)
        ob, _, term, trunc, _ = eval_env.step(a)
        d = state_distance(ob, c)
        min_dist, final_dist = min(min_dist, d), d
        if term or trunc:
            break
    return {
        "initial_dist": round(initial_dist, 4),
        "final_dist": round(final_dist, 4),
        "min_dist": round(min_dist, 4),
        "gap_closed": round(gap_closed_frac(initial_dist, min_dist), 4),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ex_obs = train_dataset['observations'][:1]
    ex_act = train_dataset['actions'][:1]
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow ckpt)'
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, ex_obs, ex_act, agent_cfg)
    assert cfg.restore_path is not None, 'needs a trained Stage-A flow (restore_path)'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    assert bool(agent.config.get('skill_cond', False)), (
        "validate_skill_coherence is the skill-conditioning GATE: it needs a Stage-A "
        f"checkpoint trained with skill_cond=true, but restore_path={cfg.restore_path} "
        f"has skill_cond={agent.config.get('skill_cond', None)!r}. Point restore_path at "
        "a checkpoint trained with agent.skill_cond=true (and a matching skill_window)."
    )
    skill_window = int(agent.config['skill_window'])
    onestep = jax.jit(lambda oc, u: agent.network.select('actor_onestep_flow')(oc, u))

    terminals = np.asarray(train_dataset['terminals'])
    ok_rows = np.flatnonzero(rows_with_full_horizon(terminals, skill_window))
    assert ok_rows.size > 0, f'no dataset row has {skill_window} steps left in its own episode'

    rng = np.random.default_rng(int(cfg.seed))
    chosen = rng.choice(ok_rows, size=min(K, ok_rows.size), replace=False)
    # Episode-safe c, recomputed with the EXACT SAME rule hindsight training uses.
    skills = add_skill_targets(train_dataset, skill_window)
    c_pool = np.asarray(skills)[chosen]
    d_a = ex_act.shape[-1]
    rollout_steps = 2 * skill_window

    cond_names = ["true", "shuffled", "stay"]
    per_condition = {name: [] for name in cond_names}
    for k in range(len(chosen)):
        reset_seed = int(cfg.seed) * 1000 + k
        ob0, _ = eval_env.reset(seed=reset_seed)
        nn = nearest_indices(ob0, c_pool, NEAREST_POOL)
        c_true = c_pool[rng.choice(nn)]
        c_shuffled = c_pool[(k + 1) % len(c_pool)]  # a c commanded for a DIFFERENT reset
        c_stay = ob0

        for cond_idx, (name, c) in enumerate(zip(cond_names, [c_true, c_shuffled, c_stay])):
            roll_rng = np.random.default_rng(int(cfg.seed) * 100_000 + k * 10 + cond_idx)
            per_condition[name].append(
                _rollout(eval_env, onestep, c, d_a, rollout_steps, reset_seed, roll_rng))
        if (k + 1) % 8 == 0:
            print(f"[rollouts] {k + 1}/{len(chosen)} done", flush=True)

    def _agg(rows, key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": round(float(vals.mean()), 4), "median": round(float(np.median(vals)), 4)}

    summary = {name: {"gap_closed": _agg(rows, "gap_closed"), "min_dist": _agg(rows, "min_dist")}
               for name, rows in per_condition.items()}

    mean_gap_true = summary["true"]["gap_closed"]["mean"]
    mean_gap_shuffled = summary["shuffled"]["gap_closed"]["mean"]
    ratio = coherence_ratio(mean_gap_true, mean_gap_shuffled)
    passed = gate_decision(mean_gap_true, ratio)

    report = {
        "env": cfg.env_name, "seed": int(cfg.seed),
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "skill_window": skill_window, "K": int(len(chosen)), "rollout_steps": rollout_steps,
        "gate": {"ratio_threshold": RATIO_GATE, "gap_threshold": GAP_GATE},
        "coherence_ratio": round(ratio, 4),
        "gate_pass": passed,
        "summary": summary,
        "rollouts": per_condition,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ALERT = "#1f2430", "#6b7280", "#3d6fb0", "#b3563d"
    cond_colors = {"true": BLUE, "shuffled": MUTED, "stay": ALERT}
    fig, ax_ = plt.subplots(figsize=(5.4, 4.4))
    data = [[r["gap_closed"] for r in per_condition[name]] for name in cond_names]
    bp = ax_.boxplot(data, tick_labels=cond_names, showfliers=False, patch_artist=True, widths=0.5)
    for patch, name in zip(bp["boxes"], cond_names):
        patch.set_facecolor(cond_colors[name])
        patch.set_alpha(0.35)
    jrng = np.random.default_rng(0)
    for i, (name, vals) in enumerate(zip(cond_names, data), start=1):
        xs = i + jrng.uniform(-0.08, 0.08, size=len(vals))
        ax_.scatter(xs, vals, s=24, c=cond_colors[name], edgecolors="white", lw=0.4, zorder=3)
    ax_.axhline(GAP_GATE, color=ALERT, ls="--", lw=0.8, label=f"gap gate ({GAP_GATE})")
    ax_.set_ylabel("fraction of initial gap closed", fontsize=9, color=INK)
    verdict = "PASS" if passed else "FAIL"
    ax_.set_title(f"skill-coherence gate: {verdict}  (coherence_ratio={ratio:.2f})\n"
                  f"{cfg.env_name}  skill_window={skill_window}", fontsize=9, color=INK)
    ax_.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(os.getcwd(), "validate_skill_coherence.png")
    fig.savefig(fig_path, dpi=150)
    report["figure"] = fig_path

    print(json.dumps({k: v for k, v in report.items() if k != "rollouts"}, indent=2))
    write_report(report, cfg, "validate_skill_coherence.json")


if __name__ == "__main__":
    main()
