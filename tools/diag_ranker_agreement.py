"""Do two critics AGREE about the ordering of latents, and does agreeing predict success?

The rollout-based version of this question died on 2026-09-09: on cube, one near-expert
action followed by 49 steps of behaviour cloning is indistinguishable from one random
in-support action followed by the same 49 steps, so a per-candidate return under a fixed
continuation cannot score a ranker (`docs/design/2026-09-08-critic-signal-and-dsrl-na.md` 2,
"the verdict above is WITHDRAWN"). A ranker known to be right scored worse than chance.

This probe drops the rollout entirely. Success on cube comes from small per-step edges
compounding over 50+ steps, so the quantity to measure is whether a checkpoint ORDERS the
candidates the way a known-good critic does -- and whether that agreement tracks the
checkpoint's own 500-episode success. If it does, ordering quality has a diagnostic that
costs no simulator time and can be read on any checkpoint mid-training.

Per state, over the SAME 256 onset states and the SAME `PRNGKey(12345)` roster the MC probe
uses, the checkpoint's deployed per-u GPI score is compared with:

  oracle       a frozen FQL expert's action critic read at G(s, u)
  expert_dist  -||G(s, u) - a_expert(s)||, the same expert's mode action

by Spearman and top-8 overlap, averaged over all states and over the `live` subset (the
states where the cached one-step-then-BC return was not degenerate). Then, ACROSS
checkpoints, the agreement is rank-correlated against 500-episode success.

Pre-registered (2026-09-09, with the oversight session): agreement with the expert critic
tracks success across checkpoints at rho > 0.5. If it does, this becomes the Arm B / Arm C
readout beside ladder swing. If it does not, ordering agreement with an expert is not what
success is made of, and measuring the critic by ranking ends here.

Run (CPU is fine, ~1-2 min per checkpoint):
  .venv/bin/python tools/diag_ranker_agreement.py \
      --runs $PSM_DATA/exp/PSMFLows/affine_strict_cube \
      --report $PSM_DATA/logs/diag_gpi_selection_cube_sd000_n256_250000.json \
      --oracle $PSM_DATA/exp/PSMFLows/fqlexpert_cube_a300/sd000_* \
      --flow $PSM_DATA/flow/cube-single-play \
      --out $PSM_DATA/logs/diag_ranker_agreement_cube.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

#: 500-episode success comes from the run's OWN eval500 reports, through
#: `stability_ladder.ladder_from_eval500`, which keys on the run directory and so cannot
#: attach one group's numbers to another's. The hardcoded six-cell affine_strict_cube dict
#: this replaced was keyed on (seed, epoch) alone: run on Arm B it scored Arm B's orderings
#: against the CONTROL's successes and every `_vs_success` number it printed was wrong.
#: Cells with no eval500 on disk stay absent rather than guessed, and the
#: across-checkpoint correlation reports its own n.


def _spearman(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ra, rb = np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def _topk(a, b, k):
    ia = set(np.argsort(np.asarray(a).ravel())[::-1][:k].tolist())
    ib = set(np.argsort(np.asarray(b).ravel())[::-1][:k].tolist())
    return len(ia & ib) / float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="group dir holding the seed run dirs")
    ap.add_argument("--report", required=True,
                    help="one diag_gpi_selection JSON: supplies the mc rows, K and u_clip")
    ap.add_argument("--cache", default="", help="MC cache npz, for the live-state mask")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--oracle_epoch", type=int, default=500000)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--epochs", default="250000,350000,450000,500000")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--preimages", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logs", default="",
                    help="dir holding the eval500 report JSONs "
                         "(default: the directory --report sits in)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import ml_collections

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from tools.stability_ladder import ladder_from_eval500
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.psm_common import targets_uncertainty

    if not a.logs:
        a.logs = os.path.dirname(os.path.abspath(a.report))
    with open(a.report) as f:
        rep = json.load(f)
    rows = np.asarray(rep["mc"]["rows"], np.int64)
    K, u_clip, cand_key = int(rep["K"]), float(rep["u_clip"]), 12345

    live = None
    cache = a.cache or os.path.join(os.path.dirname(a.report),
                                    "diag_gpi_selection_mc_cache_cube_n256.npz")
    if os.path.isfile(cache):
        d = np.load(cache)
        if d["onestep_bc_returns"].shape[0] == len(rows):
            # float64 FIRST: a float32 .std() of a constant row returns ~1e-7 rather than
            # exactly 0, which counts a degenerate state as live (249 of 256 instead of 59).
            live = d["onestep_bc_returns"].astype(np.float64).std(1) > 0

    np.random.seed(a.seed)
    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    obs_mc = jnp.asarray(np.asarray(ds["observations"][rows], np.float32))

    # The frozen expert, and the two checkpoint-independent rankers it defines.
    with open(os.path.join(a.oracle, "flags.json")) as f:
        ocfg = ml_collections.ConfigDict(_lists_to_tuples(json.load(f)["agent"]))
    oracle = agents[ocfg["agent_name"]].create(a.seed, ex["observations"], ex["actions"], ocfg)
    oracle = restore_agent(oracle, a.oracle, a.oracle_epoch)

    def build(run_dir, epoch):
        merged, _ = merge_run_config(
            {"agent_name": "psmflow", **_base_cfg(a)}, run_dir, _cli_agent_keys())
        cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
        ag = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
        ag = restore_agent(ag, run_dir, epoch)
        np.random.seed(a.seed)
        zb = ds.sample(min(ds.size, 10000))
        return ag.infer_eval_z(zb["next_observations"], zb["rewards"] + 1.0), cfg

    d_a = int(ex["actions"].shape[-1])
    k_u, k_i = jax.random.split(jax.random.PRNGKey(cand_key))
    fu = jnp.clip(jax.random.normal(k_u, (K, d_a)), -u_clip, u_clip)
    fi = jnp.clip(jax.random.normal(k_i, (K, d_a)), -u_clip, u_clip)

    out = {"probe": "cross-ranker ordering agreement, no rollout",
           "n_states": len(rows), "K": K, "topk": a.topk,
           "n_live": int(live.sum()) if live is not None else None,
           "live_mask": (
               "states where the cached one-step-then-BC return separates the candidates, "
               "i.e. float64(onestep_bc_returns).std(axis=1) > 0. On cube/256 that is 59. "
               "A 2026-09-09 version of this tool took .std() on the stored FLOAT32 array, "
               "where a constant row returns ~1e-7 rather than exactly 0, and so counted "
               "249 of 256 as live; the numbers reported from that run were the all-states "
               "columns and were unaffected. Read the all-states columns: the live mask "
               "derives from the one-step return, which "
               "docs/design/2026-09-08-critic-signal-and-dsrl-na.md 2 withdrew as an "
               "instrument, so restricting to it is not obviously meaningful here."),
           "runs": a.runs, "oracle": a.oracle, "checkpoints": {}}

    ranks = {}
    group = os.path.basename(os.path.normpath(a.runs))
    e500_ladder = {}
    for run_dir in sorted(glob.glob(os.path.join(a.runs, "*/"))):
        run_dir = run_dir.rstrip("/")
        sd = os.path.basename(run_dir)[:5]        # "sd000", not "sd000_"
        # This run's own 500-episode points, keyed by restore_epoch. Reads the reports on
        # disk for THIS group and run, so a group with no eval500 yet simply has no
        # success column rather than borrowing another group's.
        e500_ladder[sd] = ladder_from_eval500(a.logs, group, run_dir)
        for epoch in [int(x) for x in a.epochs.split(",")]:
            if not os.path.isfile(os.path.join(run_dir, f"params_{epoch}.pkl")):
                continue
            agent, cfg = build(run_dir, epoch)
            w, P = agent.task_z, int(cfg["num_parallel"])
            pess = float(cfg["actor_pessimism_penalty"])

            @jax.jit
            def scores(obs1, _w=w, _P=P, _pess=pess, _ag=agent):
                obs = jnp.broadcast_to(obs1, (K * K, *obs1.shape))
                up = jnp.repeat(fu, K, axis=0)
                ip = jnp.tile(fi, (K, 1))
                q = (_ag.psi_b(obs, ip, up) * _w).sum(-1)
                qm, qu = targets_uncertainty(q, _P)
                psi_q = (qm - _pess * qu).reshape(K, K).max(axis=1)          # (K,)
                o = jnp.broadcast_to(obs1, (K, *obs1.shape))
                act = _ag.decode(o, fu)
                orc = oracle.network.select("critic")(o, actions=act).mean(0)
                star = jnp.clip(oracle.network.select("actor_onestep_flow")(
                    oracle._actor_obs(obs1[None], None), jnp.zeros((1, d_a))), -1, 1)
                return psi_q, orc, -jnp.linalg.norm(act - star, axis=-1)

            trip = [jax.device_get(scores(obs_mc[i])) for i in range(len(rows))]
            q = np.stack([t[0] for t in trip])
            orc = np.stack([t[1] for t in trip])
            xd = np.stack([t[2] for t in trip])

            # _q/_orc/_xd are bound as defaults on purpose: `agg` closes over this
            # iteration's arrays, and a late-bound closure inside the checkpoint loop is
            # the classic way to silently score every checkpoint against the last one's.
            def agg(mask, _q=q, _orc=orc, _xd=xd):
                idx = np.flatnonzero(mask) if mask is not None else range(len(rows))
                r = {}
                for name, other in (("oracle", _orc), ("expert_dist", _xd)):
                    r[f"rho_{name}"] = float(np.mean([_spearman(_q[i], other[i]) for i in idx]))
                    r[f"top{a.topk}_{name}"] = float(
                        np.mean([_topk(_q[i], other[i], a.topk) for i in idx]))
                return r

            key = f"{sd}@{epoch}"
            rec = {"all": agg(None), "live": agg(live) if live is not None else None,
                   "success_e500": e500_ladder.get(sd, {}).get(epoch)}
            out["checkpoints"][key] = rec
            ranks[key] = rec
            s = rec["success_e500"]
            e5 = f"{s:.3f}" if s is not None else "  -  "
            print(f"{key:14s} e500={e5:>6s}  "
                  f"rho(q,oracle)={rec['all']['rho_oracle']:+.3f}  "
                  f"rho(q,expert_dist)={rec['all']['rho_expert_dist']:+.3f}  "
                  f"top{a.topk}(oracle)={rec['all'][f'top{a.topk}_oracle']:.3f}", flush=True)

    # Across checkpoints: does agreeing with the expert predict success?
    have = [(k, v) for k, v in ranks.items() if v["success_e500"] is not None]
    out["across_checkpoints"] = {"n": len(have), "keys": [k for k, _ in have]}
    if len(have) >= 3:
        succ = [v["success_e500"] for _, v in have]
        for scope in ("all", "live"):
            if have[0][1].get(scope) is None:
                continue
            for name in ("oracle", "expert_dist"):
                for stat in (f"rho_{name}", f"top{a.topk}_{name}"):
                    vals = [v[scope][stat] for _, v in have]
                    out["across_checkpoints"][f"{scope}/{stat}_vs_success"] = _spearman(vals, succ)
    print("\nacross checkpoints (Spearman of agreement vs 500-episode success):")
    for k, v in out["across_checkpoints"].items():
        if k.endswith("_vs_success"):
            print(f"  {k:44s}{v:+.3f}   (n={out['across_checkpoints']['n']})")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {a.out}")


def _base_cfg(a):
    """The hydra-equivalent agent dict `merge_run_config` layers the run's flags.json onto."""
    from agents.psmflow import get_config
    c = json.loads(json.dumps(get_config().to_dict()))
    c["flow_ckpt_path"], c["flow_ckpt_epoch"] = a.flow, a.flow_epoch
    c["preimage_path"] = a.preimages or None
    c["allow_untrained_flow"] = False
    return c


if __name__ == "__main__":
    main()
