"""How many steps must a selection rule be right over before it beats not selecting at all?

2026-09-09. The one-step ground truth failed its own sanity check: picking the roster
candidate nearest a frozen expert's action, then handing to behaviour cloning for the
remaining 49 steps, scored WORSE than a random in-support pick (regret 8.74 vs 8.58 over 256
onset states). Yet aiming at the expert at EVERY step scores 0.934 (E1) while never aiming
scores 0.072 (the BC control). Both anchors are solid, so the value of good selection is real
and simply is not located in one step.

This probe finds where it is located. At each state, aim for `m` steps and then hand to
behaviour cloning, sweeping m:

  oracle arm   for t < m: decode the whole K-roster and execute the candidate closest to the
               expert's mode action at s_t. This is E1's oracle-aim, restricted to the same
               K the deployed rule scans, so it is an upper bound on what ANY critic
               choosing from this roster could achieve.
  random arm   for t < m: execute a uniformly random member of the same roster. That is the
               BC control by construction, and running `--repeats` of it per state is where
               the per-state noise band comes from.
  both         for t >= m: the frozen flow on the per-state prior latent stream, the same
               common-random-number tail `diag_gpi_selection.py` uses, regenerated from the
               same seed so the two probes share ground truth.

The answer is the smallest m at which the oracle arm clears the random arm's own spread.
That m is the horizon a critic has to be right over, and it prices what a fair ranking
diagnostic costs. Pre-registered with the oversight session: if the gap only opens at
m >= 20, per-step ranking is the wrong quantity and the ranking-diagnostic line ends there.

The learned critics are deliberately NOT run here. The point is to learn what a fair ground
truth costs before spending on it.

Run (needs the simulator; ~1 h on one GPU for the defaults):
  MUJOCO_GL=egl .venv/bin/python tools/diag_commitment_horizon.py \
      --report $PSM_DATA/logs/diag_gpi_selection_cube_sd000_n256_250000.json \
      --oracle $PSM_DATA/exp/PSMFLows/fqlexpert_cube_a300/sd000_* \
      --flow $PSM_DATA/flow/cube-single-play \
      --out $PSM_DATA/logs/diag_commitment_horizon_cube.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

PROBE_ROW_SEED = 7      # diag_gpi_selection.py's; the BC tail must be the SAME stream
CAND_KEY = 12345        # ... and so must the roster


def _stats(x):
    x = np.asarray(x, np.float64).ravel()
    return {"mean": round(float(x.mean()), 4), "median": round(float(np.median(x)), 4),
            "p10": round(float(np.quantile(x, 0.1)), 4),
            "p90": round(float(np.quantile(x, 0.9)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--oracle_epoch", type=int, default=500000)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--ms", default="1,3,5,10,20,50")
    ap.add_argument("--repeats", type=int, default=8, help="random-arm draws per state")
    ap.add_argument("--horizon", type=int, default=0, help="0 = the report's own")
    ap.add_argument("--n_states", type=int, default=0, help="0 = all in the report")
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import ml_collections

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    with open(a.report) as f:
        rep = json.load(f)
    rows = np.asarray(rep["mc"]["rows"], np.int64)
    if a.n_states:
        rows = rows[:a.n_states]
    K, u_clip = int(rep["K"]), float(rep["u_clip"])
    H = a.horizon or int(rep["mc"]["horizon"])
    gamma = float(rep["mc"]["discount"])
    disc = gamma ** np.arange(H, dtype=np.float64)
    ms = [int(x) for x in a.ms.split(",")]

    np.random.seed(a.seed)
    env, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    sim = {k: np.asarray(ds[k]) for k in ("qpos", "qvel", "button_states") if k in ds}
    assert sim, "dataset carries no qpos/qvel (add_info) -- no simulator restore"

    cfg = get_config()
    with cfg.unlocked():
        cfg["flow_ckpt_path"], cfg["flow_ckpt_epoch"] = a.flow, a.flow_epoch
        cfg["u_clip"] = u_clip
    agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"],
                                     ml_collections.ConfigDict(cfg))
    with open(os.path.join(a.oracle, "flags.json")) as f:
        ocfg = ml_collections.ConfigDict(_lists_to_tuples(json.load(f)["agent"]))
    oracle = agents[ocfg["agent_name"]].create(a.seed, ex["observations"], ex["actions"], ocfg)
    oracle = restore_agent(oracle, a.oracle, a.oracle_epoch)

    d_a = int(agent.config["action_dim"])
    fu = jnp.clip(jax.random.normal(jax.random.split(jax.random.PRNGKey(CAND_KEY))[0],
                                    (K, d_a)), -u_clip, u_clip)
    # The SAME per-state BC tail diag_gpi_selection.py draws, regenerated from its rule so
    # the two probes share a ground truth rather than merely resembling one.
    tails = np.clip(np.random.default_rng(PROBE_ROW_SEED + 2).standard_normal(
        (len(rep["mc"]["rows"]), H, d_a)), -u_clip, u_clip).astype(np.float32)[:len(rows)]

    @jax.jit
    def aim(obs1):
        """The roster candidate whose decode is closest to the expert's mode action."""
        obs = jnp.broadcast_to(obs1, (K, *obs1.shape))
        acts = agent.decode(obs, fu)
        star = jnp.clip(oracle.network.select("actor_onestep_flow")(
            oracle._actor_obs(obs1[None], None), jnp.zeros((1, d_a))), -1, 1)
        i = jnp.argmin(jnp.linalg.norm(acts - star, axis=-1))
        return acts[i], jnp.linalg.norm(acts[i] - star)

    @jax.jit
    def dec1(obs1, u1):
        return agent.decode(obs1[None], u1[None])[0]

    def restore(r):
        env.reset()
        if "button_states" in sim:
            env.unwrapped.set_state(sim["qpos"][r], sim["qvel"][r], sim["button_states"][r])
        else:
            env.unwrapped.set_state(sim["qpos"][r], sim["qvel"][r])

    def roll(r, tail, m, mode, rng):
        """Aim for m steps (mode 'oracle' | 'random'), then the BC tail. Returns (ret, succ)."""
        restore(r)
        ob = np.asarray(ds["observations"][int(r)], np.float32)
        ret, sc = 0.0, 0.0
        for t in range(H):
            if t < m and mode == "oracle":
                act = np.clip(np.asarray(jax.device_get(aim(jnp.asarray(ob, jnp.float32)))[0]), -1, 1)
            elif t < m:
                u = fu[int(rng.integers(K))]
                act = np.clip(np.asarray(jax.device_get(dec1(jnp.asarray(ob, jnp.float32), u))), -1, 1)
            else:
                act = np.clip(np.asarray(jax.device_get(
                    dec1(jnp.asarray(ob, jnp.float32), tail[t]))), -1, 1)
            ob, rw, tm, tr, info = env.step(act)
            ret += disc[t] * float(rw)
            sc = max(sc, float(np.max(np.asarray(info.get("success", 0.0)))))
            if tm or tr:
                break
        return ret, sc

    out = {"probe": "commitment horizon: aim for m steps, then behaviour cloning",
           "env": a.env, "n_states": len(rows), "K": K, "horizon": H, "discount": gamma,
           "repeats": a.repeats, "oracle": a.oracle,
           "note": ("oracle = the roster candidate nearest the expert's mode action, "
                    "recomputed each step (E1's oracle-aim at this K). random = a uniform "
                    "roster draw each step, which IS the BC control. Both hand to the same "
                    "per-state prior latent tail at t >= m."),
           "by_m": {}}
    rng = np.random.default_rng(a.seed)
    for m in ms:
        o_ret, o_suc, r_ret, r_suc, r_sd = [], [], [], [], []
        for si, r in enumerate(rows):
            orr, osc = roll(r, tails[si], m, "oracle", rng)
            reps = [roll(r, tails[si], m, "random", rng) for _ in range(a.repeats)]
            rr = np.array([x[0] for x in reps])
            o_ret.append(orr); o_suc.append(osc)
            r_ret.append(rr.mean()); r_suc.append(np.mean([x[1] for x in reps]))
            r_sd.append(rr.std())
            if si % 32 == 0:
                print(f"  m={m:2d} state {si}/{len(rows)}", flush=True)
        o_ret, r_ret, r_sd = np.asarray(o_ret), np.asarray(r_ret), np.asarray(r_sd)
        # The noise band is POOLED, not per-state. Most states are degenerate -- every
        # random draw there fails identically, giving a within-state sd of exactly 0 -- and
        # dividing by that produces a z-score of 1e9 that means nothing. The pooled
        # within-state sd, sqrt(mean of the per-state variances), is the honest scale.
        pooled = float(np.sqrt(np.mean(r_sd ** 2)))
        gap = float((o_ret - r_ret).mean())
        out["by_m"][str(m)] = {
            "oracle_return": _stats(o_ret), "random_return": _stats(r_ret),
            "gap_mean": gap,
            "random_pooled_sd": round(pooled, 4),
            "gap_over_pooled_sd": (round(gap / pooled, 3) if pooled > 1e-9 else None),
            "frac_states_random_degenerate": float((r_sd <= 1e-9).mean()),
            "frac_states_oracle_beats_random": float((o_ret > r_ret).mean()),
            "oracle_success": float(np.mean(o_suc)), "random_success": float(np.mean(r_suc))}
        st = out["by_m"][str(m)]
        z = st["gap_over_pooled_sd"]
        print(f"m={m:2d}  gap {gap:+7.3f}  ({'n/a' if z is None else f'{z:+.2f}'} pooled-sd)  "
              f"beats {st['frac_states_oracle_beats_random']:.2f}  "
              f"succ {st['oracle_success']:.3f} vs {st['random_success']:.3f}", flush=True)

    # The smallest m whose mean gap clears one standard deviation of the random arm.
    firing = [m for m in ms
              if (out["by_m"][str(m)]["gap_over_pooled_sd"] or 0.0) >= 1.0]
    out["commitment_horizon"] = min(firing) if firing else None
    out["commitment_horizon_note"] = (
        "smallest m whose mean oracle-minus-random gap is >= 1 POOLED sd of the random "
        "arm's own within-state spread" if firing else
        f"NO m in {ms} separates the oracle from random by 1 sd -- aiming from the roster "
        "does not pay at any horizon tested, which would end the ranking-diagnostic line")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
