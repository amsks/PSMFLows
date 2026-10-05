"""Does psmgoal's critic rank the 64 candidate latents badly from the first step, or does its
ranking get worse as the episode goes on? (2026-09-30)

Runs eval episodes exactly as tools/eval_checkpoint.py does with one worker (same hydra
config, same flags.json merge, same preimage splice, same np.random.seed(seed) before the
task-coefficient inference, same env seeding, same supply_rng action-key stream), and at
every step records what the gpi argmax saw:

  t, s_t, the 64 clipped prior latents u, their decodes G(s_t, u), the 64 scores
  q = mean_g phi(s,u,g)^T w + b(s,u,g) (the exact values select_latent argmaxes), the split
  q_phiw = mean_g phi^T w and q_b = mean_g b, the chosen index, the executed action, and
  the per-step PRNG key. Per episode: success, return, length.

The executed action is the agent's own `sample_actions` output. The recomputed candidate
set is checked against it at every step (the decode of the recomputed argmax must equal
it); a mismatch raises. So the npz holds the exact (s_t, 64 u) the agent scored, with no
hidden RNG: another critic can score the same pairs later.

Outputs: one npz (+npz_out=...) and a JSON summary (report_out=...) per (checkpoint, task).
The JSON gives, per step bucket (t in 0-24, 25-49, 50-99, 100-199, 200+) and separately
for successful and failed episodes: the score spread across the 64 u, best minus median,
the near-tie fraction (top-2 within 1% of the top score), and the chosen score. Spread and
best-minus-median are also given divided by the std of the chosen score across all states
of that (checkpoint, task), so they are scale-free.

Run (same arguments as scripts/eval500.sh, plus the npz path):
  .venv/bin/python tools/diag_psmgoal_rollout_ranking.py agent=psmgoal \
    env_name=cube-single-play-singletask-task1-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \
    agent.coef_source=regression restore_path=<run_dir> restore_epoch=750000 \
    eval_episodes=20 report_out=<json> +npz_out=<npz>

Pool several JSONs into the seed x task table:
  .venv/bin/python tools/diag_psmgoal_rollout_ranking.py aggregate <out.json> <in1.json> ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

BUCKETS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, None))
NEAR_TIE_REL = 0.01
TIE_TOL = 1e-4   # absolute score gap accepted as a float-rounding tie between two jit paths


# --------------------------------------------------------------------- pure statistics
def bucket_label(lo, hi):
    return f"{lo}-{hi - 1}" if hi is not None else f"{lo}+"


def bucket_index(t):
    """Bucket id per step for an int array t (BUCKETS order)."""
    t = np.asarray(t)
    out = np.full(t.shape, -1, np.int64)
    for i, (lo, hi) in enumerate(BUCKETS):
        m = (t >= lo) if hi is None else ((t >= lo) & (t < hi))
        out[m] = i
    assert (out >= 0).all(), "negative t"
    return out


def step_stats(q):
    """Per-step ranking statistics from (T, K) scores. Returns a dict of (T,) arrays."""
    q = np.asarray(q, np.float64)
    srt = np.sort(q, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    return {
        "spread": q.std(axis=1),
        "best_minus_median": top1 - np.median(q, axis=1),
        "near_tie": (top1 - top2) <= NEAR_TIE_REL * np.abs(top1),
        "chosen": top1,
    }


def summarize(t, q, ep_of_step, ep_success, q_phiw=None, q_b=None):
    """The JSON summary for one (checkpoint, task).

    t, ep_of_step: (T,) ints; q (T, K) scores; ep_success (E,) bools. The chosen score is
    the per-step max of q (the gpi argmax). `sigma_chosen` is the std of the chosen score
    across all T states; the *_norm fields divide by it. The chosen score's z-score uses
    the mean and std across all T states.
    """
    t = np.asarray(t)
    st = step_stats(q)
    chosen = st["chosen"]
    sigma = float(chosen.std())
    mu = float(chosen.mean())
    denom = sigma if sigma > 0 else np.nan
    succ_step = np.asarray(ep_success, bool)[np.asarray(ep_of_step)]
    b_id = bucket_index(t)
    extra = {}
    if q_phiw is not None and q_b is not None:
        q_phiw, q_b = np.asarray(q_phiw, np.float64), np.asarray(q_b, np.float64)
        am = np.argmax(np.asarray(q), axis=1)
        extra = {"spread_phiw_norm": q_phiw.std(axis=1) / denom,
                 "spread_b_norm": q_b.std(axis=1) / denom,
                 "argmax_eq_phiw_argmax": np.argmax(q_phiw, axis=1) == am,
                 "argmax_eq_b_argmax": np.argmax(q_b, axis=1) == am}
    table = {}
    for outcome, mask_o in (("success", succ_step), ("fail", ~succ_step)):
        rows = {}
        for i, (lo, hi) in enumerate(BUCKETS):
            m = mask_o & (b_id == i)
            n = int(m.sum())
            if n == 0:
                rows[bucket_label(lo, hi)] = {"n_steps": 0}
                continue
            row = {
                "n_steps": n,
                "n_episodes": int(np.unique(np.asarray(ep_of_step)[m]).size),
                "spread": float(st["spread"][m].mean()),
                "spread_norm": float(st["spread"][m].mean() / denom),
                "best_minus_median": float(st["best_minus_median"][m].mean()),
                "best_minus_median_norm": float(st["best_minus_median"][m].mean() / denom),
                "near_tie_frac": float(st["near_tie"][m].mean()),
                "chosen_mean": float(chosen[m].mean()),
                "chosen_z": float((chosen[m].mean() - mu) / denom),
            }
            for k, v in extra.items():
                row[k] = float(np.asarray(v, np.float64)[m].mean())
            rows[bucket_label(lo, hi)] = row
        table[outcome] = rows
    ep_success = np.asarray(ep_success, bool)
    return {"sigma_chosen": sigma, "mean_chosen": mu, "n_steps": int(t.size),
            "n_episodes": int(ep_success.size), "n_success": int(ep_success.sum()),
            "success_rate": float(ep_success.mean()) if ep_success.size else None,
            "near_tie_rel": NEAR_TIE_REL,
            "buckets": [bucket_label(lo, hi) for lo, hi in BUCKETS], "table": table}


AGG_FIELDS = ("spread_norm", "best_minus_median_norm", "near_tie_frac", "chosen_z",
              "spread", "best_minus_median", "chosen_mean",
              "spread_phiw_norm", "spread_b_norm", "argmax_eq_phiw_argmax", "argmax_eq_b_argmax")


def aggregate(summaries):
    """Average the per-(checkpoint, task) table cells over files. Each cell is the unweighted
    mean over the files that have steps in it; n_files and total n_steps are reported."""
    out = {}
    buckets = summaries[0]["buckets"]
    for outcome in ("success", "fail"):
        rows = {}
        for b in buckets:
            cells = [s["table"][outcome][b] for s in summaries if s["table"][outcome][b]["n_steps"] > 0]
            row = {"n_files": len(cells), "n_steps": int(sum(c["n_steps"] for c in cells)),
                   "n_episodes": int(sum(c.get("n_episodes", 0) for c in cells))}
            for f in AGG_FIELDS:
                vals = [c[f] for c in cells if f in c and np.isfinite(c[f])]
                row[f] = float(np.mean(vals)) if vals else None
            rows[b] = row
        out[outcome] = rows
    n_ep = sum(s["n_episodes"] for s in summaries)
    n_succ = sum(s["n_success"] for s in summaries)
    return {"n_files": len(summaries), "n_episodes": n_ep, "n_success": n_succ,
            "success_rate_pooled": n_succ / n_ep if n_ep else None,
            "success_rate_mean_of_files": float(np.mean([s["success_rate"] for s in summaries])),
            "table": out}


def _aggregate_cli(argv):
    out_path, ins = argv[0], argv[1:]
    reps = []
    for p in ins:
        with open(p) as fh:
            reps.append(json.load(fh))
    agg = aggregate([r["summary"] for r in reps])
    agg["inputs"] = ins
    known = [r.get("known_eval500_success") for r in reps]
    if all(k is not None for k in known):
        agg["known_eval500_mean_of_files"] = float(np.mean(known))
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))
    print("aggregate ->", out_path)


# --------------------------------------------------------------------- rollout (GPU half)
def _known_eval(env_name, restore_path, logs_dir):
    """The 500-episode regression eval of this checkpoint/task, if on disk (psmgoal_reg_*)."""
    import glob
    import re
    m = re.search(r"task(\d)", env_name)
    sd = os.path.basename(os.path.normpath(restore_path))[:5]
    if not m:
        return None
    dirs = [logs_dir] + ([os.path.join(os.environ["PSM_DATA"], "logs")] if os.environ.get("PSM_DATA") else [])
    paths = [p for d in dirs for p in glob.glob(os.path.join(d, f"psmgoal_reg_{sd}_task{m.group(1)}.json"))]
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if os.path.normpath(d["restore_path"]) == os.path.normpath(restore_path):
            return d
    return None


def run(cfg):
    import jax
    import jax.numpy as jnp
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.evaluation import supply_rng
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert config["agent_name"] == "psmgoal", "this diagnostic reads psmgoal's gpi path"
    assert config["acting"] == "gpi", f"acting={config['acting']!r}; the diagnostic needs gpi"
    npz_out = cfg.get("npz_out", None)
    assert npz_out, "pass +npz_out=<path>"
    base_seed = int(cfg.seed)
    n_ep = int(cfg.eval_episodes)

    # ---- identical to tools/eval_checkpoint.py::_evaluate_shard (worker 0 of 1) ----
    np.random.seed(base_seed)
    _, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    train_dataset = dict(train_dataset)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(config["preimage_path"]))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0]
    for k in aug:
        if k.startswith("noise_preimage"):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(config.get("use_point_preimage", False))
    ex = ds.sample(1)
    agent = agents["psmgoal"].create(base_seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, str(cfg.restore_path), cfg.restore_epoch)
    n_relabel = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
    zb = ds.sample(n_relabel)
    agent = agent.infer_eval_goals(zb, zb["rewards"] + float(cfg.get("eval_reward_shift", 1.0)))
    # ---------------------------------------------------------------------------------

    c = agent.config
    K, d_a, u_clip = int(c["gpi_num_u"]), int(c["action_dim"]), float(c["u_clip"])

    @jax.jit
    def score(ag, obs, key):
        # select_latent's exact candidate draw and score (same ops, same order).
        u_cand = jnp.clip(jax.random.normal(key, (K, d_a)), -u_clip, u_clip)
        o = jnp.broadcast_to(obs, (K, obs.shape[-1]))
        w = jnp.broadcast_to(ag.eval_w_star, (K, ag.eval_w_star.shape[0]))
        q = jnp.mean(ag.mesh_M(o, u_cand, ag.eval_goals, w), axis=1)
        phi, b = ag.mesh_phi_b(o, u_cand, ag.eval_goals)
        q_phiw = jnp.mean((phi * ag.eval_w_star[None, None, :]).sum(-1), axis=1)
        q_b = jnp.mean(b, axis=1)
        idx = jnp.argmax(q)
        acts = ag.decode(o, u_cand)                                   # all 64 decodes
        a_star = ag.decode(obs[None], u_cand[idx][None])[0]           # sample_actions' decode
        return u_cand, q, q_phiw, q_b, idx, acts, a_star

    @jax.jit
    def decode1(ag, obs, u):
        return ag.decode(obs[None], u[None])[0]

    # evaluate(): action keys from supply_rng(PRNGKey(seed)), env inits seeded once.
    keys = {"k": jax.random.PRNGKey(base_seed)}

    def next_key():
        keys["k"], sub = jax.random.split(keys["k"])
        return sub

    # Sanity: supply_rng hands sample_actions the same keys next_key() does.
    probe = []
    probe_fn = supply_rng(lambda seed=None: probe.append(np.asarray(seed)), rng=jax.random.PRNGKey(base_seed))
    probe_fn()
    probe_fn()
    kk = jax.random.PRNGKey(base_seed)
    kk, s1 = jax.random.split(kk)
    kk, s2 = jax.random.split(kk)
    assert np.array_equal(probe[0], np.asarray(s1)) and np.array_equal(probe[1], np.asarray(s2))

    eval_env.reset(seed=base_seed)
    rec = {k: [] for k in ("episode", "t", "obs", "u_cand", "actions_cand", "q", "q_phiw", "q_b",
                           "chosen", "action", "key", "reward")}
    ep_success, ep_return, ep_length = [], [], []
    max_dev, n_tie_swaps, max_tie_gap = 0.0, 0, 0.0
    t0 = time.time()
    for ep in range(n_ep):
        observation, info = eval_env.reset()
        done, step, ret = False, 0, 0.0
        while not done:
            key = next_key()
            action_agent = np.asarray(agent.sample_actions(observations=observation, seed=key,
                                                           temperature=0))
            u_cand, q, q_phiw, q_b, idx, acts, a_star = score(agent, jnp.asarray(observation), key)
            dev = float(np.max(np.abs(np.asarray(a_star) - action_agent)))
            idx = int(idx)
            if dev > 1e-4:
                # This jit and sample_actions' jit can round q differently in the last bits, so
                # at a near-exact tie they pick different indices. Record the index the agent
                # EXECUTED (its decode matches the executed action) and require the tie.
                qn = np.asarray(q)
                d_all = np.max(np.abs(np.asarray(acts) - action_agent[None]), axis=1)
                j = int(np.argmin(d_all))
                gap = float(qn[idx] - qn[j])
                # the batch-64 decode differs from sample_actions' batch-1 decode by up to ~1e-3,
                # so confirm candidate j with the batch-1 decode itself
                a_j = np.asarray(decode1(agent, jnp.asarray(observation), u_cand[j]))
                d_all[j] = float(np.max(np.abs(a_j - action_agent)))
                if d_all[j] > 1e-4 or gap > TIE_TOL:
                    raise RuntimeError(
                        f"recomputed argmax decode differs from sample_actions by {dev} (ep {ep}, "
                        f"t {step}); closest candidate {j} decode dev {d_all[j]}, score gap {gap}; "
                        "the acting path is not reproduced")
                n_tie_swaps += 1
                max_tie_gap = max(max_tie_gap, gap)
                idx = j
                dev = float(d_all[j])
            max_dev = max(max_dev, dev)
            action = np.clip(np.array(action_agent), -1, 1)
            next_observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            for k, v in (("episode", ep), ("t", step), ("obs", np.asarray(observation, np.float32)),
                         ("u_cand", np.asarray(u_cand)), ("actions_cand", np.asarray(acts)),
                         ("q", np.asarray(q)), ("q_phiw", np.asarray(q_phiw)), ("q_b", np.asarray(q_b)),
                         ("chosen", int(idx)), ("action", action), ("key", np.asarray(key)),
                         ("reward", float(reward))):
                rec[k].append(v)
            ret += float(reward)
            step += 1
            observation = next_observation
        ep_success.append(float(np.max(np.asarray(info.get("success", 0.0)))) > 0.5)
        ep_return.append(ret)
        ep_length.append(step)
        print(f"episode {ep}: success={ep_success[-1]} return={ret:.1f} length={step} "
              f"({time.time() - t0:.0f} s)", flush=True)

    arr = {k: np.asarray(v) for k, v in rec.items()}
    ep_success = np.asarray(ep_success, bool)
    os.makedirs(os.path.dirname(os.path.abspath(npz_out)), exist_ok=True)
    np.savez_compressed(
        npz_out, **arr,
        ep_success=ep_success, ep_return=np.asarray(ep_return), ep_length=np.asarray(ep_length),
        eval_goals=np.asarray(agent.eval_goals), eval_w=np.asarray(agent.eval_w_star),
        env_name=str(cfg.env_name), restore_path=str(cfg.restore_path),
        restore_epoch=int(cfg.restore_epoch), seed=base_seed,
        coef_source=str(c.get("coef_source")), gpi_decode=str(c["gpi_decode"]),
        u_clip=u_clip, gpi_num_u=K)

    summary = summarize(arr["t"], arr["q"], arr["episode"], ep_success, arr["q_phiw"], arr["q_b"])
    # the executed index is the argmax of the stored scores up to TIE_TOL
    gaps = arr["q"].max(axis=1) - arr["q"][np.arange(arr["q"].shape[0]), arr["chosen"]]
    assert float(gaps.max()) <= TIE_TOL, float(gaps.max())
    logs_dir = os.path.dirname(os.path.abspath(cfg.report_out)) if cfg.get("report_out") else "."
    known = _known_eval(str(cfg.env_name), str(cfg.restore_path), logs_dir)
    report = {
        "env": str(cfg.env_name), "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch), "seed": base_seed,
        "train_seed": prov.get("train_seed"), "coef_source": str(c.get("coef_source")),
        "agent_config_source": prov, "npz": os.path.abspath(npz_out),
        # max |decode of the recorded chosen u - executed action| over all steps
        "max_action_dev_vs_sample_actions": max_dev,
        # steps where the recomputed argmax and the executed index differ at a float-level tie
        "n_tie_swaps": n_tie_swaps, "max_tie_score_gap": max_tie_gap, "tie_tol": TIE_TOL,
        "per_episode_success": [int(x) for x in ep_success],
        "ep_length": [int(x) for x in ep_length], "ep_return": [float(x) for x in ep_return],
        "known_eval500_success": known["success"] if known else None,
        # worker 0 of the known 4-worker eval ran with seed 0*4+0 = 0, the same seed as here,
        # so its first 20 episodes use the same env inits and action keys.
        "known_eval500_worker0_first_n": (known["per_episode_success"][:n_ep]
                                          if known and known.get("num_workers") and
                                          known.get("worker_seeds", [None])[0] == base_seed else None),
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary,
    }
    with open(cfg.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "agent_config_source"}, indent=2))
    print("report ->", cfg.report_out, " npz ->", npz_out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        _aggregate_cli(sys.argv[2:])
    else:
        import hydra

        @hydra.main(version_base=None, config_path="../configs", config_name="config")
        def _main(cfg):
            run(cfg)

        _main()
