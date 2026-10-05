"""At one state, does psmgoal's measure order the candidate latents u the way a working scalar
critic does? (2026-09-30)

Two scorers of the SAME (s, 64 u):

  Q_M(s, u)  psmgoal's acting score: mean over the 32 rewarding eval goals of
             phi(s,u,g)^T w + b(s,u,g), with w and the goals built exactly as
             tools/eval_checkpoint.py builds them (np.random.seed(seed), preimage splice,
             ds.sample(1), create, restore, ds.sample(eval_relabel_size), infer_eval_goals)
             under the run's own flags.json plus the typed CLI overrides
             (agent.coef_source=regression).
  Q_W(s, u)  the DSRL-NA latent critic of an f_psmflow run with dsrl_na.enabled (single-task,
             no task input). Q_A(s, G(s,u)) (min over its ensemble, the target Q_W regresses
             onto) is scored too, as a check that does not depend on Q_W's latent box
             (the critic's u_clip can be smaller than psmgoal's).

States:
  dataset   n_states random dataset states with a valid preimage (same rows for every seed
            and arm: rng keyed on the task only). Candidates: 64 clipped prior latents drawn
            as select_latent draws them (clip(normal(key, (64, d_a)), -u_clip, u_clip), one key
            per state) plus the dataset preimage u_data as a 65th column.
  rollout   when +rollout_npz is given: the (s_t, 64 u) psmgoal scored while acting, recorded
            by tools/diag_psmgoal_rollout_ranking.py; Q_M is the stored score q. A slice is
            re-scored here and must match (checks goals and w are the eval's).

Per state (over the 64 prior u): Spearman(Q_M, Q_B); top-1 agreement; rank (1 = best, average
ranks on ties, chance 32.5) of the critic's argmax inside Q_M's order and of Q_M's argmax
inside the critic's order; negative-Spearman indicator. Rollout states are also bucketed by t.

Run one (seed, task) cell (GPU):
  .venv/bin/python tools/diag_rank_agreement_psmgoal.py agent=psmgoal \
    env_name=cube-single-play-singletask-task1-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \
    agent.coef_source=regression restore_path=<psmgoal run> restore_epoch=750000 \
    +critic_dir=<critic run> +critic_epoch=250000 +critic_arm=psmgoal_rhat \
    +rollout_npz=<npz or null> +n_states=2000 report_out=<cell json> +scores_out=<cell npz>

Pool cells into the five-task table (mean over tasks per seed, then mean and 95% CI over seeds):
  .venv/bin/python tools/diag_rank_agreement_psmgoal.py aggregate <out.json> <cell1.json> ...
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

BUCKETS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, None))
METRICS = ("spearman", "top1_agree", "critic_top1_rank_in_M", "M_top1_rank_in_critic", "neg_spearman")
DATASET_KEY = 20260930
RESCORE_ROWS = 256
RESCORE_TOL = 1e-3


# --------------------------------------------------------------------- pure statistics
def bucket_label(lo, hi):
    return f"{lo}-{hi - 1}" if hi is not None else f"{lo}+"


def bucket_index(t):
    t = np.asarray(t)
    out = np.full(t.shape, -1, np.int64)
    for i, (lo, hi) in enumerate(BUCKETS):
        m = (t >= lo) if hi is None else ((t >= lo) & (t < hi))
        out[m] = i
    assert (out >= 0).all(), "negative t"
    return out


def desc_ranks(q):
    """Row-wise ranks, 1 = highest score, ties get their average rank. q (N, K) -> (N, K)."""
    from scipy.stats import rankdata
    return rankdata(-np.asarray(q, np.float64), method="average", axis=1)


def row_spearman(a, b):
    """Spearman per row (average ranks on ties). A row constant in either input gives nan."""
    from scipy.stats import rankdata
    ra = rankdata(np.asarray(a, np.float64), method="average", axis=1)
    rb = rankdata(np.asarray(b, np.float64), method="average", axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def rank_metrics(qm, qb):
    """Per-state agreement of two scorers over the same K candidates. qm, qb (N, K).

    Returns a dict of (N,) arrays: spearman, top1_agree, critic_top1_rank_in_M (rank of
    argmax qb inside qm's order, 1 = best), M_top1_rank_in_critic, neg_spearman.
    """
    qm, qb = np.asarray(qm, np.float64), np.asarray(qb, np.float64)
    assert qm.shape == qb.shape and qm.ndim == 2, (qm.shape, qb.shape)
    n = np.arange(qm.shape[0])
    am, ab = qm.argmax(1), qb.argmax(1)
    rho = row_spearman(qm, qb)
    return {
        "spearman": rho,
        "top1_agree": (am == ab).astype(np.float64),
        "critic_top1_rank_in_M": desc_ranks(qm)[n, ab],
        "M_top1_rank_in_critic": desc_ranks(qb)[n, am],
        "neg_spearman": np.where(np.isnan(rho), np.nan, (rho < 0).astype(np.float64)),
    }


def mean_metrics(per_state, mask=None):
    """nan-aware means of rank_metrics output over the states in mask."""
    n_all = len(next(iter(per_state.values())))
    m = np.ones(n_all, bool) if mask is None else np.asarray(mask, bool)
    out = {"n_states": int(m.sum())}
    for k, v in per_state.items():
        x = np.asarray(v, np.float64)[m]
        x = x[np.isfinite(x)]
        out[k] = float(x.mean()) if x.size else None
    return out


def subset_spearman(qm, qb, keep, min_n=8):
    """Spearman per row over the candidates in keep (N, K) bool; nan below min_n."""
    out = np.full(qm.shape[0], np.nan)
    for i in range(qm.shape[0]):
        k = np.asarray(keep[i], bool)
        if k.sum() >= min_n:
            out[i] = row_spearman(qm[i, k][None], qb[i, k][None])[0]
    return out


def data_rank(q_with_data):
    """Rank (1 = best of K+1) of the last column (u_data) per row. (N, K+1) -> (N,)."""
    return desc_ranks(q_with_data)[:, -1]


def seed_ci(vals):
    """Mean and 95% t half-width across seeds."""
    from scipy.stats import t as student_t
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    if v.size == 0:
        return {"mean": None, "ci95": None, "n_seeds": 0}
    hw = float(student_t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
    return {"mean": float(v.mean()), "ci95": hw, "n_seeds": int(v.size)}


def aggregate(cells):
    """cells: per-(seed, task) reports. For every (scorer, block) table: mean over tasks
    within a seed, then mean and 95% CI across seeds. Per-seed and per-task values kept."""
    by_seed = {}
    for c in cells:
        by_seed.setdefault(int(c["seed"]), []).append(c)
    blocks = sorted({b for c in cells for b in c["tables"]})
    out = {"n_cells": len(cells), "seeds": sorted(by_seed),
           "tasks_per_seed": {str(s): sorted(int(c["task"]) for c in cs) for s, cs in by_seed.items()},
           "tables": {}, "per_cell": {}}
    for b in blocks:
        fields = sorted({f for c in cells if b in c["tables"] for f in c["tables"][b]
                         if f != "n_states"})
        seed_rows, tab = {}, {}
        for s, cs in by_seed.items():
            have = [c["tables"][b] for c in cs if b in c["tables"] and c["tables"][b].get("n_states", 1) > 0]
            row = {"n_tasks": len(have), "n_states": int(sum(h.get("n_states", 0) for h in have))}
            for f in fields:
                v = [h[f] for h in have if h.get(f) is not None]
                row[f] = float(np.mean(v)) if v else None
            seed_rows[str(s)] = row
        for f in fields:
            tab[f] = seed_ci([r[f] for r in seed_rows.values() if r["n_tasks"] > 0])
        tab["n_states_total"] = int(sum(r["n_states"] for r in seed_rows.values()))
        tab["n_tasks_per_seed"] = {s: r["n_tasks"] for s, r in seed_rows.items()}
        tab["per_seed_five_task_mean"] = seed_rows
        out["tables"][b] = tab
    for c in cells:
        out["per_cell"][f"sd{int(c['seed']):03d}_task{int(c['task'])}"] = c["tables"]
    return out


def _aggregate_cli(argv):
    out_path, ins = argv[0], argv[1:]
    cells = []
    for p in ins:
        with open(p) as fh:
            cells.append(json.load(fh))
    arms = {c["critic_arm"] for c in cells}
    steps = {c["critic_epoch"] for c in cells}
    assert len(arms) == 1 and len(steps) == 1, f"mixed arms/steps: {arms} {steps}"
    agg = aggregate(cells)
    agg.update({"critic_arm": arms.pop(), "critic_epoch": steps.pop(), "inputs": ins,
                "psmgoal_epoch": sorted({c["psmgoal_epoch"] for c in cells}),
                "rank_chance": 32.5, "top1_chance": 1 / 64})
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({b: {k: v for k, v in t.items() if k != "per_seed_five_task_mean"}
                      for b, t in agg["tables"].items()}, indent=2))
    print("aggregate ->", out_path)


# --------------------------------------------------------------------- scoring (GPU half)
# OGBench's bare `-singletask-v0` id is the env's default task: task 2 for cube, task 1 for the mazes.
DEFAULT_TASK = {"cube": 2, "antmaze": 1, "pointmaze": 1}


def _canon_env(name):
    """`cube-single-play-singletask-v0` -> `cube-single-play-singletask-task2-v0`."""
    if name.endswith("-singletask-v0"):
        t = DEFAULT_TASK[name.split("-")[0]]
        return name[:-len("v0")] + f"task{t}-v0"
    return name


def _critic_config(critic_dir):
    """The critic's agent config the way eval_checkpoint builds it: the psmflow hydra group
    as defaults, the run's flags.json on top, no CLI overrides."""
    from omegaconf import OmegaConf

    from tools.eval_checkpoint import merge_run_config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = OmegaConf.to_container(OmegaConf.load(os.path.join(repo, "configs/agent/psmflow.yaml")),
                                  resolve=True)
    return merge_run_config(base, critic_dir, set())


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
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    t_start = time.time()
    import re
    task = int(re.search(r"task(\d)", str(cfg.env_name)).group(1))
    critic_dir = glob.glob(str(cfg.critic_dir))
    assert len(critic_dir) == 1, f"critic_dir matches {critic_dir}"
    critic_dir = critic_dir[0]
    critic_epoch = int(cfg.critic_epoch)
    critic_arm = str(cfg.critic_arm)
    n_states = int(cfg.get("n_states", 2000))
    chunk = int(cfg.get("chunk", 64))
    rollout_npz = cfg.get("rollout_npz", None)
    rollout_npz = None if rollout_npz in (None, "", "null", "None") else str(rollout_npz)
    restore_path = glob.glob(str(cfg.restore_path))
    assert len(restore_path) == 1, restore_path
    restore_path = restore_path[0]

    # ---- psmgoal: identical to tools/eval_checkpoint.py::_evaluate_shard (worker 0) ----
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert config["agent_name"] == "psmgoal" and config["acting"] == "gpi"
    base_seed = int(cfg.seed)
    np.random.seed(base_seed)
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    train_dataset = dict(train_dataset)
    aug, valid = repair_invalid_preimages(load_augmented_dataset(config["preimage_path"]))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0]
    for k in aug:
        if k.startswith("noise_preimage"):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(config.get("use_point_preimage", False))
    ex = ds.sample(1)
    agent = agents["psmgoal"].create(base_seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, restore_path, cfg.restore_epoch)
    zb = ds.sample(min(ds.size, int(cfg.get("eval_relabel_size", 10000))))
    agent = agent.infer_eval_goals(zb, zb["rewards"] + float(cfg.get("eval_reward_shift", 1.0)))
    # ------------------------------------------------------------------------------------
    c = agent.config
    K, d_a, u_clip = int(c["gpi_num_u"]), int(c["action_dim"]), float(c["u_clip"])

    # ---- the critic ----
    cmerged, cprov = _critic_config(critic_dir)
    cconf = ml_collections.ConfigDict(_lists_to_tuples(cmerged))
    assert cconf["dsrl_na"]["enabled"] and not cconf["dsrl_na"]["task_conditioned"], critic_dir
    assert (cconf["flow_ckpt_path"] == c["flow_ckpt_path"]
            and int(cconf["flow_ckpt_epoch"]) == int(c["flow_ckpt_epoch"])
            and cconf["gpi_decode"] == c["gpi_decode"]), "the two agents decode through different flows"
    with open(os.path.join(critic_dir, "flags.json")) as fh:
        cflags = json.load(fh)
    reward_override = (cflags.get("dataset") or {}).get("reward_override_path")
    critic_seed = int(cflags["seed"])
    train_seed = prov.get("train_seed")
    sd_tag = os.path.basename(os.path.normpath(restore_path))[:5]
    pairing = {
        "critic env == this env": _canon_env(cflags["env_name"]) == _canon_env(str(cfg.env_name)),
        "critic seed == psmgoal seed": critic_seed == train_seed,
        "critic reward is this psmgoal seed/task's (or real)": (
            not reward_override or (sd_tag in reward_override and f"task{task}" in reward_override)),
    }
    # +allow_unpaired_critic=true exists only for the CPU smoke with a stand-in critic.
    unpaired = [k for k, ok in pairing.items() if not ok]
    if unpaired:
        msg = f"critic {critic_dir} fails: {unpaired} ({cflags['env_name']}, seed {critic_seed}, {reward_override})"
        assert bool(cfg.get("allow_unpaired_critic", False)), msg
        print("WARNING (smoke only):", msg, flush=True)
    critic = agents[cconf["agent_name"]].create(base_seed, ex["observations"], ex["actions"], cconf)
    critic = restore_agent(critic, critic_dir, critic_epoch)
    c_clip = float(cconf["u_clip"])

    @jax.jit
    def score_all(ag, cr, obs, u):
        """obs (N, ob), u (N, J, d_a) -> Q_M, Q_W, Q_A each (N, J)."""
        N, J = u.shape[0], u.shape[1]
        o = jnp.broadcast_to(obs[:, None], (N, J, obs.shape[-1])).reshape(N * J, -1)
        uu = u.reshape(N * J, d_a)
        w = jnp.broadcast_to(ag.eval_w_star, (N * J, ag.eval_w_star.shape[0]))
        qm = jnp.mean(ag.mesh_M(o, uu, ag.eval_goals, w), axis=1)
        qw = cr.qw(o, uu)
        qa = cr.qa(o, cr.decode(o, uu)).min(0)
        return qm.reshape(N, J), qw.reshape(N, J), qa.reshape(N, J)

    def score_chunks(obs, u):
        outs = [score_all(agent, critic, jnp.asarray(obs[i:i + chunk]), jnp.asarray(u[i:i + chunk]))
                for i in range(0, obs.shape[0], chunk)]
        return [np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(3)]

    def block_tables(qm, qw, qa, u, mask=None):
        """qm, qw, qa (N, 64) over the prior candidates; u (N, 64, d_a)."""
        tabs = {}
        pm_w, pm_a = rank_metrics(qm, qw), rank_metrics(qm, qa)
        pm_wa = rank_metrics(qw, qa)
        tabs["M_vs_QW"] = mean_metrics(pm_w, mask)
        tabs["M_vs_QA"] = mean_metrics(pm_a, mask)
        tabs["QW_vs_QA"] = mean_metrics(pm_wa, mask)
        inbox = (np.abs(u) <= c_clip + 1e-6).all(-1)
        sub = mean_metrics({"spearman_inbox": subset_spearman(qm, qw, inbox),
                            "frac_candidates_in_critic_box": inbox.mean(1)}, mask)
        tabs["M_vs_QW"].update({k: v for k, v in sub.items() if k != "n_states"})
        return tabs, pm_w, pm_a

    tables = {}
    # ---- dataset states ----
    rng = np.random.default_rng(DATASET_KEY + task)
    rows = np.sort(rng.choice(np.nonzero(valid)[0], size=n_states, replace=False))
    obs_d = np.asarray(aug["observations"][rows], np.float32)
    u_data = np.clip(np.asarray(aug["noise_preimage_point"][rows], np.float32), -u_clip, u_clip)
    keys = jax.random.split(jax.random.PRNGKey(DATASET_KEY + task), n_states)
    u_prior = np.asarray(jax.vmap(
        lambda k: jnp.clip(jax.random.normal(k, (K, d_a)), -u_clip, u_clip))(keys), np.float32)
    u_all = np.concatenate([u_prior, u_data[:, None]], axis=1)                  # (N, 65, d_a)
    qm_d, qw_d, qa_d = score_chunks(obs_d, u_all)
    tabs, _, _ = block_tables(qm_d[:, :K], qw_d[:, :K], qa_d[:, :K], u_prior)
    for sc in ("M", "QW", "QA"):
        q = {"M": qm_d, "QW": qw_d, "QA": qa_d}[sc]
        tabs.setdefault("u_data_rank_of_65", {})[sc] = float(data_rank(q).mean())
    tabs["M_u_share"] = {
        "std_over_u_mean": float(qm_d[:, :K].std(1).mean()),
        "std_over_states_of_mean_over_u": float(qm_d[:, :K].mean(1).std())}
    tables.update({f"dataset/{k}": v for k, v in tabs.items()})

    # ---- rollout states ----
    rollout_meta = None
    scores_out = {"rows": rows, "u_prior": u_prior, "u_data": u_data,
                  "qm_d": qm_d, "qw_d": qw_d, "qa_d": qa_d}
    if rollout_npz:
        r = np.load(rollout_npz)
        assert str(r["env_name"]) == str(cfg.env_name), (str(r["env_name"]), cfg.env_name)
        assert os.path.normpath(str(r["restore_path"])) == os.path.normpath(restore_path)
        assert int(r["restore_epoch"]) == int(cfg.restore_epoch)
        assert str(r["coef_source"]) == str(c["coef_source"])
        w_dev = float(np.max(np.abs(np.asarray(r["eval_w"]) - np.asarray(agent.eval_w_star))))
        g_dev = float(np.max(np.abs(np.asarray(r["eval_goals"]) - np.asarray(agent.eval_goals))))
        n_max = int(cfg.get("rollout_max_steps", 0)) or None     # smoke only
        obs_r, u_r, q_stored = r["obs"][:n_max], r["u_cand"][:n_max], np.asarray(r["q"][:n_max], np.float64)
        t_r, ep_r = r["t"][:n_max], r["episode"][:n_max]
        qm_r, qw_r, qa_r = score_chunks(obs_r, u_r)
        n_chk = min(RESCORE_ROWS, obs_r.shape[0])
        rescore_dev = float(np.max(np.abs(qm_r[:n_chk] - q_stored[:n_chk])))
        rescore_scale = float(np.abs(q_stored[:n_chk]).mean())
        print(f"rollout: w dev {w_dev:.2e}, goal dev {g_dev:.2e}, "
              f"Q_M re-score dev {rescore_dev:.2e} (|q| ~ {rescore_scale:.3g})", flush=True)
        assert rescore_dev <= RESCORE_TOL * max(1.0, rescore_scale), (
            f"re-scored Q_M differs from the recorded acting score by {rescore_dev}")
        succ = np.asarray(r["ep_success"], bool)[np.asarray(ep_r)]
        b_id = bucket_index(t_r)
        for name, m in [("all", None)] + [(bucket_label(lo, hi), b_id == i)
                                          for i, (lo, hi) in enumerate(BUCKETS)]:
            tabs, _, _ = block_tables(q_stored, qw_r, qa_r, u_r, m)
            tables.update({f"rollout_{name}/{k}": v for k, v in tabs.items()})
        for name, m in (("success_eps", succ), ("fail_eps", ~succ)):
            tabs, _, _ = block_tables(q_stored, qw_r, qa_r, u_r, m)
            tables.update({f"rollout_{name}/{k}": v for k, v in tabs.items()})
        rollout_meta = {"npz": os.path.abspath(rollout_npz), "n_steps": int(obs_r.shape[0]),
                        "n_episodes": len(r["ep_success"]),
                        "success_rate": float(np.mean(r["ep_success"])),
                        "eval_w_max_dev": w_dev, "eval_goals_max_dev": g_dev,
                        "qm_rescore_max_dev": rescore_dev, "qm_rescore_rows": n_chk}
        scores_out.update({"qw_r": qw_r, "qa_r": qa_r, "qm_r_rescored": qm_r})

    report = {
        "env": str(cfg.env_name), "task": task, "seed": train_seed, "eval_seed": base_seed,
        "psmgoal_run": restore_path, "psmgoal_epoch": int(cfg.restore_epoch),
        "coef_source": str(c["coef_source"]), "k_goals": int(agent.eval_goals.shape[0]),
        "critic_arm": critic_arm, "critic_run": critic_dir, "critic_epoch": critic_epoch,
        "critic_seed": critic_seed, "critic_reward_override": reward_override,
        "critic_reward_source": str(cconf["dsrl_na"]["reward_source"]),
        "psmgoal_u_clip": u_clip, "critic_u_clip": c_clip, "gpi_num_u": K,
        "n_dataset_states": n_states, "dataset_key": DATASET_KEY + task,
        "rollout": rollout_meta, "pairing_checks": pairing,
        "psmgoal_config_source": prov, "critic_config_source": cprov,
        "wall_seconds": round(time.time() - t_start, 1),
        "tables": tables,
    }
    if cfg.get("scores_out"):
        os.makedirs(os.path.dirname(os.path.abspath(cfg.scores_out)), exist_ok=True)
        np.savez_compressed(cfg.scores_out, **scores_out)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.report_out)), exist_ok=True)
    with open(cfg.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in tables.items() if k.endswith("M_vs_QW")}, indent=2))
    print("report ->", cfg.report_out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        _aggregate_cli(sys.argv[2:])
    else:
        import hydra

        @hydra.main(version_base=None, config_path="../configs", config_name="config")
        def _main(cfg):
            run(cfg)

        _main()
