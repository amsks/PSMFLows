"""Does the first latent u change whether the goal is reached when behaviour cloning (BC)
continues afterwards? (2026-10-01)

psmgoal's measure is the measure of the BC policy: take latent u now, then follow BC. This
tool measures the true effect of that first u in the simulator, with no learned critic.

Per task t and prefix length T0:
  1. Anchor. Reset the task with a fixed seed and run BC for T0 steps. If that prefix episode
     ended (success or time limit) within the first T0 steps, the next prefix seed is used.
  2. K candidate first latents, drawn once from the clipped prior (N(0, I) clipped to u_clip).
  3. For each candidate, R rollouts from the SAME simulator state: step 0 decodes the
     candidate, every later step decodes a fresh clipped prior draw (BC), until success or
     the env's episode limit. The limit counts from the task reset, so a rollout has
     limit - T0 steps.
  4. Outcomes per rollout: success (0/1) and gamma^(steps to success), 0 on failure, where
     steps counts env steps from the anchor (the candidate's step is step 1).

BC step: the frozen flow's one-step decode `PSMGoalAgent.decode` (gpi_decode=onestep) at
batch 1, then a clip to [-1, 1]. This is the same network `agent=fql agent.bc_only=true`
acts with (actor_onestep_flow from the Stage-A checkpoint). The one difference from the BC
control eval is that the latent is clipped to +-u_clip here, as in psmgoal's acting path.
The flow is loaded directly from the Stage-A checkpoint; no psmgoal checkpoint is read.

Identical start state. Every rollout re-reaches its anchor by `env.reset(seed=...)`, a
restore of the reset's saved MuJoCo integration state, and a replay of the stored prefix
actions. The restore is needed because OGBench's cube reset takes two unseeded
`action_space.sample()` steps in the goal scene before it restores qpos/qvel (`action_space`
is a property that builds a new Box, so it cannot be seeded): time, qpos, qvel, mocap and the
observation come back identical, the solver warm start and ctrl do not. The tool asserts the
first group is reproduced by the reset itself and overwrites the second with `mj_setState`.
After the replay the observation and the full integration state (time, qpos, qvel, act,
warmstart, ctrl, mocap, ...) are compared with the anchor's and the tool raises if the max
abs difference is >= 1e-6. For every anchor, before any rollout, a fixed 20-step action
sequence is also run twice from the anchor and its end states compared. The replay check
runs again on every rollout; the largest difference is written to the JSON.

Statistics per anchor (`candidate_effect_stats`, pure numpy/scipy): per-candidate success
rate and mean discounted outcome, pooled mean, range across candidates, chi-square test of
equal success rates, share of outcome variance explained by the candidate (one-way ANOVA
eta-squared and omega-squared), and the between-candidate standard deviation observed vs
expected from sampling noise alone.

Envs (`--env`, table `ENVS`). cube is the original setup: prefix lengths 0,50,100, discount
0.98, limit 200. antmaze (`antmaze-medium-navigate-singletask-task{t}-v0`): prefix lengths
0,250,500, discount 0.99, limit 1000. OGBench's maze reset draws the start-position noise from
the GLOBAL numpy RNG, which `env.reset(seed=...)` does not seed, so for antmaze the tool calls
`np.random.seed(reset_seed)` right before every reset; the rest of the procedure (assert on
the reset, `mj_setState`, replay, per-rollout assert) is the same. The maze goal position
(`cur_goal_xy`, outside the MuJoCo state, read by the success test) is also asserted identical
at every reset.

Run one task:
  .venv/bin/python tools/diag_first_u_effect_bc.py --task 1 \
    --flow $PSM_DATA/flow/cube-single-play --flow_epoch 500000
  .venv/bin/python tools/diag_first_u_effect_bc.py --env antmaze --task 1
Pool the per-task JSONs:
  .venv/bin/python tools/diag_first_u_effect_bc.py --aggregate <task1.json> ... --out <agg.json>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

GAMMA = 0.98
IDENTITY_TOL = 1e-6
ALPHA = 0.05

# Per-env settings. file_tag goes into the default JSON names; seed_global_numpy: the env's
# reset reads the global numpy RNG; bc_eval / default_task: the BC control eval500 JSON under
# $PSM_DATA/evals and the task OGBench's un-numbered singletask env stands for.
ENVS = {
    "cube": {"env": "cube-single-play-singletask-task{task}-v0",
             "env_default_task": "cube-single-play-singletask-v0", "default_task": 2,
             "flow": "cube-single-play", "flow_epoch": 500000, "prefix_lens": "0,50,100",
             "gamma": GAMMA, "file_tag": "", "seed_global_numpy": False, "bc_eval": "bc_cube"},
    "antmaze": {"env": "antmaze-medium-navigate-singletask-task{task}-v0",
                "env_default_task": "antmaze-medium-navigate-singletask-v0", "default_task": 1,
                "flow": "antmaze-medium-navigate", "flow_epoch": 500000, "prefix_lens": "0,250,500",
                "gamma": 0.99, "file_tag": "antmaze_", "seed_global_numpy": True,
                "bc_eval": "bc_antmaze"},
}


def default_out_name(env_key, task=None, smoke=False):
    """Default JSON file name: per-task report (task given) or the aggregate (task None)."""
    tag = ENVS[env_key]["file_tag"]
    if task is None:
        return f"diag_first_u_effect_bc_{tag}aggregate.json"
    return f"diag_first_u_effect_bc_{tag}task{task}{'_smoke' if smoke else ''}.json"


# --------------------------------------------------------------------- pure statistics
def _one_way(y):
    """One-way ANOVA of y (K, R) on the row index. Returns plain floats (None if undefined)."""
    y = np.asarray(y, np.float64)
    K, R = y.shape
    means = y.mean(axis=1)
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    ss_between = float(R * ((means - grand) ** 2).sum())
    ss_within = max(ss_total - ss_between, 0.0)
    df_b, df_w = K - 1, K * (R - 1)
    ms_within = ss_within / df_w if df_w > 0 else None
    out = {
        "per_candidate_mean": [float(m) for m in means],
        "pooled_mean": grand,
        "range": float(means.max() - means.min()),
        "sd_between_observed": float(means.std(ddof=1)) if K > 1 else None,
        "sd_between_expected_from_noise": (float(np.sqrt(ms_within / R))
                                           if ms_within is not None else None),
        "eta2": None, "omega2": None, "f_statistic": None, "f_p": None,
        "f_dof": [df_b, df_w],
    }
    if ss_total <= 0.0 or ms_within is None or df_b < 1:
        return out                                     # no variance: the share is undefined
    out["eta2"] = ss_between / ss_total
    out["omega2"] = (ss_between - df_b * ms_within) / (ss_total + ms_within)
    if ms_within > 0.0:
        from scipy.stats import f as f_dist
        f_stat = (ss_between / df_b) / ms_within
        out["f_statistic"] = float(f_stat)
        out["f_p"] = float(f_dist.sf(f_stat, df_b, df_w))
    else:
        out["f_p"] = 0.0                               # candidates separate the outcome exactly
    return out


def candidate_effect_stats(success, steps, gamma=GAMMA):
    """Effect of the candidate (row) on the rollout outcome.

    success: (K, R) 0/1. steps: (K, R) env steps from the anchor to the end of the rollout
    (read only where success is 1). Returns a JSON-able dict:
      success / discounted: per-candidate mean, pooled mean, range, observed and
        noise-expected between-candidate sd, eta2, omega2, F test.
      chi2: test of equal success rates across the K candidates (statistic, dof, p); None
        when the pooled rate is 0 or 1.
    For the success outcome the noise-expected sd is sqrt(p (1 - p) / R) at the pooled p.
    """
    s = (np.asarray(success, np.float64) > 0.5).astype(np.float64)
    assert s.ndim == 2 and s.shape == np.asarray(steps).shape
    K, R = s.shape
    d = np.where(s > 0.5, gamma ** np.asarray(steps, np.float64), 0.0)
    out = {"num_u": int(K), "num_rollouts": int(R), "gamma": float(gamma),
           "success": _one_way(s), "discounted": _one_way(d)}
    p = float(s.mean())
    out["success"]["sd_between_expected_from_noise"] = float(np.sqrt(p * (1.0 - p) / R))
    chi = {"statistic": None, "dof": int(K - 1), "p": None}
    if 0.0 < p < 1.0 and K > 1:
        from scipy.stats import chi2 as chi2_dist
        stat = float(R * ((s.mean(axis=1) - p) ** 2).sum() / (p * (1.0 - p)))
        chi["statistic"] = stat
        chi["p"] = float(chi2_dist.sf(stat, K - 1))
    out["chi2"] = chi
    return out


def _mean_defined(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)) if vals else None, "n_defined": len(vals)}


def aggregate_anchors(anchors):
    """Pool anchors (dicts with task, prefix_len, stats). Shares are averaged over the
    anchors where they are defined (pooled outcome variance > 0)."""
    st = [a["stats"] for a in anchors]
    chi_p = [s["chi2"]["p"] for s in st if s["chi2"]["p"] is not None]
    f_p = [s["discounted"]["f_p"] for s in st if s["discounted"]["f_p"] is not None]
    prefix0 = {}
    for a in anchors:
        if int(a["prefix_len"]) == 0:
            prefix0[str(a["task"])] = a["stats"]["success"]["pooled_mean"]
    rows = [{
        "task": a["task"], "prefix_len": a["prefix_len"],
        "success_pooled": a["stats"]["success"]["pooled_mean"],
        "success_range": a["stats"]["success"]["range"],
        "success_sd_observed": a["stats"]["success"]["sd_between_observed"],
        "success_sd_expected_from_noise": a["stats"]["success"]["sd_between_expected_from_noise"],
        "chi2_p": a["stats"]["chi2"]["p"],
        "success_eta2": a["stats"]["success"]["eta2"],
        "success_omega2": a["stats"]["success"]["omega2"],
        "discounted_pooled": a["stats"]["discounted"]["pooled_mean"],
        "discounted_range": a["stats"]["discounted"]["range"],
        "discounted_eta2": a["stats"]["discounted"]["eta2"],
        "discounted_omega2": a["stats"]["discounted"]["omega2"],
        "discounted_f_p": a["stats"]["discounted"]["f_p"],
    } for a in anchors]
    return {
        "n_anchors": len(anchors),
        "alpha": ALPHA,
        "share_explained_success_eta2": _mean_defined([s["success"]["eta2"] for s in st]),
        "share_explained_success_omega2": _mean_defined([s["success"]["omega2"] for s in st]),
        "share_explained_discounted_eta2": _mean_defined([s["discounted"]["eta2"] for s in st]),
        "share_explained_discounted_omega2": _mean_defined([s["discounted"]["omega2"] for s in st]),
        "chi2_n_tested": len(chi_p),
        "chi2_n_p_below_alpha": int(sum(p < ALPHA for p in chi_p)),
        "chi2_expected_count_if_no_effect": ALPHA * len(chi_p),
        "discounted_f_n_tested": len(f_p),
        "discounted_f_n_p_below_alpha": int(sum(p < ALPHA for p in f_p)),
        "bc_success_at_prefix0_per_task": prefix0,
        "bc_success_at_prefix0_task_mean": (float(np.mean(list(prefix0.values())))
                                            if prefix0 else None),
        "per_anchor": rows,
    }


def _aggregate_cli(paths, out_path):
    from utils.log_utils import write_report
    anchors, known = [], {}
    for p in paths:
        with open(p) as fh:
            rep = json.load(fh)
        anchors.extend(rep["anchors"])
        if rep.get("known_bc_eval500_success") is not None:
            known[str(rep["task"])] = rep["known_bc_eval500_success"]
    agg = aggregate_anchors(anchors)
    agg["inputs"] = list(paths)
    agg["known_bc_eval500_success_per_task"] = known
    agg["known_bc_eval500_task_mean"] = float(np.mean(list(known.values()))) if known else None
    print(json.dumps(agg, indent=2))
    write_report(agg, {"report_out": out_path}, "diag_first_u_effect_bc_aggregate.json")


# --------------------------------------------------------------------- rollouts
def _mj(env):
    """(MjModel, MjData) of the wrapped OGBench env: manipspace keeps them as _model/_data,
    the gymnasium MujocoEnv behind the locomotion mazes as model/data."""
    u = env.unwrapped
    if hasattr(u, "_model"):
        return u._model, u._data
    return u.model, u.data


def _goal_xy(env):
    """The maze env's goal position (read by its success test, not part of the MuJoCo state);
    None for envs without one."""
    g = getattr(env.unwrapped, "cur_goal_xy", None)
    return None if g is None else np.array(g, np.float64)


def _physics_state(env, core=False):
    """MuJoCo integration state of the wrapped OGBench env as a float64 vector. core=True
    leaves out the warm start and ctrl (time, qpos, qvel, act, mocap only)."""
    import mujoco
    model, data = _mj(env)
    st = mujoco.mjtState
    spec = int(st.mjSTATE_INTEGRATION)
    if core:
        spec = (int(st.mjSTATE_TIME) | int(st.mjSTATE_QPOS) | int(st.mjSTATE_QVEL)
                | int(st.mjSTATE_ACT) | int(st.mjSTATE_MOCAP_POS) | int(st.mjSTATE_MOCAP_QUAT))
    out = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
    mujoco.mj_getState(model, data, out, spec)
    return out


def _set_physics_state(env, state):
    """Overwrite the full MuJoCo integration state and recompute the derived quantities."""
    import mujoco
    model, data = _mj(env)
    mujoco.mj_setState(model, data, np.asarray(state, np.float64),
                       int(mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_forward(model, data)


def _known_bc(task, env_key="cube"):
    """The 500-episode BC control eval of this task, if on disk (unclipped latents, random resets)."""
    root = os.environ.get("PSM_DATA")
    if not root:
        return None
    e = ENVS[env_key]
    for name in (f"{e['bc_eval']}_task{task}.json", f"{e['bc_eval']}.json"):
        p = os.path.join(root, "evals", name)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        want = {e["env"].format(task=task)}
        if task == e["default_task"]:
            want.add(e["env_default_task"])     # OGBench's default task: cube 2, antmaze 1
        if d.get("env") in want:
            return float(d["success"])
    return None


def run(a):
    import jax
    import ml_collections

    from agents import agents
    from agents.psmgoal import get_config
    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset
    from utils.log_utils import write_report

    task, seed = int(a.task), int(a.seed)
    env_key = a.env
    e_cfg = ENVS[env_key]
    gamma = float(a.gamma)
    seed_global = bool(e_cfg["seed_global_numpy"])
    all_lens = [int(x) for x in str(a.prefix_lens).split(",")]
    K, R = int(a.num_u), int(a.num_rollouts)
    # smoke: the identity check still runs for every prefix length; rollouts for one anchor
    prefix_lens = [max(all_lens)] if a.smoke else all_lens
    if a.smoke:
        K, R = 2, 2
    assert K >= 2 and R >= 2 and all(t0 >= 0 for t0 in all_lens)
    env_name = e_cfg["env"].format(task=task)
    out_path = a.out or os.path.join(
        os.environ["PSM_DATA"], "logs", default_out_name(env_key, task, a.smoke))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    np.random.seed(seed)
    _, env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=None)
    ex = Dataset.create(**dict(train_dataset)).sample(1)
    cfg = get_config()
    with cfg.unlocked():
        cfg["flow_ckpt_path"], cfg["flow_ckpt_epoch"] = a.flow, int(a.flow_epoch)
    agent = agents["psmgoal"].create(seed, ex["observations"], ex["actions"],
                                     ml_collections.ConfigDict(cfg))
    c = agent.config
    d_a, u_clip = int(c["action_dim"]), float(c["u_clip"])
    assert c["gpi_decode"] == "onestep", c["gpi_decode"]
    limit = env.spec.max_episode_steps
    assert limit is not None and max(all_lens) < limit, (limit, all_lens)
    identity_method = "reset(seed) + restore of that reset's MuJoCo state + replay"
    if seed_global:
        identity_method = "np.random.seed(seed) + " + identity_method + " + goal xy check"

    hp = [("env", env_name), ("task", task), ("prefix_lens (T0), rollouts", prefix_lens),
          ("prefix_lens (T0), identity check", all_lens),
          ("num_u (K candidates per anchor)", K), ("num_rollouts (R per candidate)", R),
          ("episodes this job", len(prefix_lens) * K * R), ("seed", seed),
          ("gamma (discounted outcome)", gamma), ("u_clip", u_clip),
          ("gpi_decode", c["gpi_decode"]), ("max_episode_steps (from reset)", limit),
          ("flow source", "Stage-A checkpoint (no psmgoal checkpoint read)"),
          ("flow_ckpt_path", a.flow), ("flow_ckpt_epoch", int(a.flow_epoch)),
          ("state identity", f"{identity_method}, tol {IDENTITY_TOL}"),
          ("smoke", bool(a.smoke)), ("jax devices", str(jax.devices())), ("out", out_path)]
    print("=== diag_first_u_effect_bc hyperparameters")
    for k, v in hp:
        print(f"  {k:36s} {v}")
    print(flush=True)

    @jax.jit
    def dec1(ag, obs1, u1):
        return ag.decode(obs1[None], u1[None])[0]

    def act(obs, u):
        """One BC step's action: the agent's decode at batch 1, clipped to [-1, 1]."""
        return np.clip(np.array(dec1(agent, np.asarray(obs, np.float32), u)), -1, 1)

    def prior(rng, shape):
        return np.clip(rng.standard_normal(shape), -u_clip, u_clip).astype(np.float32)

    def ended(info, terminated, truncated):
        success = float(np.max(np.asarray(info.get("success", 0.0)))) > 0.5
        return success, (success or terminated or truncated)

    n_env_steps = 0
    max_t0 = max(all_lens)

    PROBE_STEPS = 20
    reset_diag = {"max_abs_core_diff": 0.0, "max_abs_obs_diff": 0.0,
                  "max_abs_warmstart_ctrl_diff_before_restore": 0.0, "n_resets": 0}
    if seed_global:
        reset_diag["max_abs_goal_xy_diff"] = 0.0

    def seeded_reset(reset_seed, px=None):
        """env.reset(seed), then the reset's full MuJoCo state set to the one saved when the
        prefix was built (px=None: this call builds it). The reset itself must reproduce time,
        qpos, qvel, act, mocap and the observation; only warm start and ctrl are overwritten."""
        if seed_global:
            np.random.seed(reset_seed)     # the maze reset draws its start-position noise here
        ob, _ = env.reset(seed=reset_seed)
        if px is None:
            return ob, _physics_state(env), _physics_state(env, core=True)
        if seed_global:
            d_goal = float(np.max(np.abs(_goal_xy(env) - px["goal_xy0"])))
            if not d_goal < IDENTITY_TOL:
                raise RuntimeError(f"reset(seed={reset_seed}) not reproduced: max abs diff of "
                                   f"the goal xy {d_goal}, tolerance {IDENTITY_TOL}")
            reset_diag["max_abs_goal_xy_diff"] = max(reset_diag["max_abs_goal_xy_diff"], d_goal)
        d_core = float(np.max(np.abs(_physics_state(env, core=True) - px["core0"])))
        d_obs = float(np.max(np.abs(np.asarray(ob, np.float64) - px["obs"][0])))
        d_full = float(np.max(np.abs(_physics_state(env) - px["state"][0])))
        if not (d_core < IDENTITY_TOL and d_obs < IDENTITY_TOL):
            raise RuntimeError(
                f"reset(seed={reset_seed}) not reproduced: max abs diff time/qpos/qvel/act/mocap "
                f"{d_core}, observation {d_obs}, tolerance {IDENTITY_TOL}")
        reset_diag["max_abs_core_diff"] = max(reset_diag["max_abs_core_diff"], d_core)
        reset_diag["max_abs_obs_diff"] = max(reset_diag["max_abs_obs_diff"], d_obs)
        reset_diag["max_abs_warmstart_ctrl_diff_before_restore"] = max(
            reset_diag["max_abs_warmstart_ctrl_diff_before_restore"], d_full)
        reset_diag["n_resets"] += 1
        _set_physics_state(env, px["state"][0])
        return ob, None, None

    # ---- prefixes: BC from the task reset, one per prefix seed, built on demand ----
    prefixes = {}

    def build_prefix(j):
        nonlocal n_env_steps
        reset_seed = 1_000_000 * seed + 1000 * task + j
        lat = prior(np.random.default_rng([seed, task, 0, j]), (max_t0, d_a))
        ob, state0, core0 = seeded_reset(reset_seed)
        px = {"reset_seed": reset_seed, "actions": [], "n_alive": 0, "core0": core0,
              "obs": {0: np.array(ob, np.float64)}, "state": {0: state0}}
        if seed_global:
            px["goal_xy0"] = _goal_xy(env)
        _set_physics_state(env, state0)            # the same call every later reset makes
        for t in range(max_t0):
            action = act(ob, lat[t])
            ob, _, terminated, truncated, info = env.step(action)
            n_env_steps += 1
            px["actions"].append(action)
            if ended(info, terminated, truncated)[1]:
                break
            px["n_alive"] = t + 1
            if (t + 1) in all_lens:
                px["obs"][t + 1] = np.array(ob, np.float64)
                px["state"][t + 1] = _physics_state(env)
        return px

    def to_anchor(px, t0):
        """reset + replay of the stored prefix actions; returns (obs, |obs diff|, |state diff|)."""
        nonlocal n_env_steps
        ob, _, _ = seeded_reset(px["reset_seed"], px)
        for t in range(t0):
            ob, _, terminated, truncated, info = env.step(px["actions"][t])
            n_env_steps += 1
            assert not ended(info, terminated, truncated)[1], "prefix replay ended early"
        d_obs = float(np.max(np.abs(np.asarray(ob, np.float64) - px["obs"][t0])))
        d_state = float(np.max(np.abs(_physics_state(env) - px["state"][t0])))
        if not (d_obs < IDENTITY_TOL and d_state < IDENTITY_TOL):
            raise RuntimeError(
                f"anchor state not reproduced (task {task}, T0 {t0}, reset seed "
                f"{px['reset_seed']}): max abs obs diff {d_obs}, physics state diff {d_state}, "
                f"tolerance {IDENTITY_TOL}")
        return ob, d_obs, d_state

    t_start = time.time()

    # ---- pick the prefix of every anchor and check that it is reproduced, before any rollout ----
    chosen = {}
    for t0 in all_lens:
        j, skipped = 0, []
        while True:
            if j not in prefixes:
                prefixes[j] = build_prefix(j)
            if prefixes[j]["n_alive"] >= t0:
                break
            skipped.append(j)
            j += 1
            assert j < 1000, "no BC prefix survives to T0 in 1000 prefix seeds"
        px = prefixes[j]
        # Two more runs of reset + restore + replay against the state recorded when the prefix
        # was built, each followed by the same fixed action sequence; the two end states must agree.
        probe_actions = np.random.default_rng([seed, task, 3, t0]).uniform(
            -1, 1, (min(PROBE_STEPS, limit - t0), d_a)).astype(np.float32)
        ends, diffs = [], []
        for _ in range(2):
            ob, d_o, d_s = to_anchor(px, t0)
            diffs.append((d_o, d_s))
            for pa in probe_actions:
                ob, _, terminated, truncated, info = env.step(pa)
                n_env_steps += 1
                if ended(info, terminated, truncated)[1]:
                    break
            ends.append((np.asarray(ob, np.float64), _physics_state(env)))
        ident = {"max_abs_obs_diff": max(d[0] for d in diffs),
                 "max_abs_physics_state_diff": max(d[1] for d in diffs), "n_checks": 2,
                 "probe_steps": len(probe_actions),
                 "probe_end_max_abs_obs_diff": float(np.max(np.abs(ends[0][0] - ends[1][0]))),
                 "probe_end_max_abs_physics_state_diff": float(np.max(np.abs(ends[0][1] - ends[1][1])))}
        if not (ident["probe_end_max_abs_obs_diff"] < IDENTITY_TOL
                and ident["probe_end_max_abs_physics_state_diff"] < IDENTITY_TOL):
            raise RuntimeError(f"the same actions from the anchor (task {task}, T0 {t0}) gave "
                               f"different end states: {ident}")
        chosen[t0] = (j, skipped, ident)
        print(f"identity check task {task} T0 {t0} (prefix {j}, skipped {skipped}): max abs diff at "
              f"the anchor obs {ident['max_abs_obs_diff']:.3g}, physics state "
              f"{ident['max_abs_physics_state_diff']:.3g}; after {ident['probe_steps']} fixed actions "
              f"obs {ident['probe_end_max_abs_obs_diff']:.3g}, physics state "
              f"{ident['probe_end_max_abs_physics_state_diff']:.3g} (tol {IDENTITY_TOL})", flush=True)

    anchors = []
    for t0 in prefix_lens:
        t_anchor = time.time()
        j, skipped, ident = chosen[t0]
        ident = dict(ident)
        px = prefixes[j]
        anchor_ob, _, _ = to_anchor(px, t0)

        cand = prior(np.random.default_rng([seed, task, 1, t0]), (K, d_a))
        cand_actions = np.stack([act(anchor_ob, cand[k]) for k in range(K)])
        success = np.zeros((K, R), np.int64)
        steps = np.zeros((K, R), np.int64)
        budget = limit - t0
        for k in range(K):
            for r in range(R):
                ob, d_obs, d_state = to_anchor(px, t0)
                ident["max_abs_obs_diff"] = max(ident["max_abs_obs_diff"], d_obs)
                ident["max_abs_physics_state_diff"] = max(ident["max_abs_physics_state_diff"], d_state)
                ident["n_checks"] += 1
                lat = prior(np.random.default_rng([seed, task, 2, t0, k, r]), (budget, d_a))
                u, n = cand[k], 0
                while True:
                    ob, _, terminated, truncated, info = env.step(act(ob, u))
                    n += 1
                    n_env_steps += 1
                    succ, done = ended(info, terminated, truncated)
                    if done:
                        break
                    u = lat[n]                     # a fresh clipped prior draw for step n >= 1
                assert succ or n == budget, (succ, n, budget)
                success[k, r], steps[k, r] = int(succ), n
            print(f"task {task} T0 {t0} candidate {k}: success {success[k].mean():.3f} "
                  f"({time.time() - t_start:.0f} s, {n_env_steps} env steps)", flush=True)

        stats = candidate_effect_stats(success, steps, gamma)
        anchors.append({
            "task": task, "prefix_len": t0, "prefix_index": j, "prefix_reset_seed": px["reset_seed"],
            "skipped_prefix_indices": skipped, "rollout_step_budget": budget,
            "state_identity": {**ident, "tol": IDENTITY_TOL, "passed": True},
            "anchor_obs": [float(x) for x in px["obs"][t0]],
            "candidates_u": cand.tolist(), "candidate_first_actions": cand_actions.tolist(),
            "success": success.tolist(), "steps": steps.tolist(),
            "stats": stats, "wall_seconds": round(time.time() - t_anchor, 1),
        })
        s = stats["success"]
        print(f"=== task {task} T0 {t0}: pooled success {s['pooled_mean']:.4f}, range {s['range']:.4f}, "
              f"sd observed {s['sd_between_observed']:.4f} vs noise {s['sd_between_expected_from_noise']:.4f}, "
              f"chi2 p {stats['chi2']['p']}, eta2 {s['eta2']}, omega2 {s['omega2']}; "
              f"identity max diff obs {ident['max_abs_obs_diff']:.3g} "
              f"state {ident['max_abs_physics_state_diff']:.3g} over {ident['n_checks']} checks",
              flush=True)

    wall = time.time() - t_start
    report = {
        "tool": "diag_first_u_effect_bc", "env": env_name, "task": task, "seed": seed,
        "prefix_lens": prefix_lens, "identity_checked_prefix_lens": all_lens, "num_u": K, "num_rollouts": R, "gamma": gamma,
        "u_clip": u_clip, "gpi_decode": str(c["gpi_decode"]), "max_episode_steps": int(limit),
        "flow_source": "stage_a_checkpoint", "flow_ckpt_path": str(a.flow),
        "flow_ckpt_epoch": int(a.flow_epoch), "smoke": bool(a.smoke),
        "bc_step": "u ~ clip(N(0, I), +-u_clip); action = clip(PSMGoalAgent.decode(s, u), -1, 1), batch 1",
        "state_identity": {
            "method": (("np.random.seed(s) (the maze reset draws its start-position noise from "
                        "the global numpy RNG; the goal xy is asserted identical at every "
                        "reset); " if seed_global else "") +
                       "env.reset(seed=s); mj_setState to the saved integration state of that reset "
                       "(overwrites warm start and ctrl; time/qpos/qvel/act/mocap/observation are "
                       "asserted to be reproduced by the reset itself); replay of the stored prefix "
                       "actions. Observation and full MuJoCo integration state compared with the "
                       "anchor's on every rollout."),
            "reset": dict(reset_diag),
            "pre_rollout_check": {str(t0): chosen[t0][2] for t0 in all_lens},
            "tol": IDENTITY_TOL,
            "max_abs_obs_diff": max(x["state_identity"]["max_abs_obs_diff"] for x in anchors),
            "max_abs_physics_state_diff": max(x["state_identity"]["max_abs_physics_state_diff"]
                                              for x in anchors),
            "n_checks": int(sum(x["state_identity"]["n_checks"] for x in anchors)),
            "passed": True,
        },
        # BC control eval500 of this task: unclipped latents, 500 random resets. The prefix-0
        # anchor here is ONE reset state, so the two numbers are comparable only roughly.
        "known_bc_eval500_success": _known_bc(task, env_key),
        "anchors": anchors,
        "aggregate": aggregate_anchors(anchors),
        "n_env_steps": int(n_env_steps), "wall_seconds": round(wall, 1),
        "env_steps_per_second": round(n_env_steps / max(wall, 1e-9), 1),
    }
    print(json.dumps(report["aggregate"], indent=2))
    print(f"{n_env_steps} env steps in {wall:.0f} s ({report['env_steps_per_second']} steps/s)")
    write_report(report, {"report_out": out_path}, "diag_first_u_effect_bc.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube", choices=sorted(ENVS), help="env family (table ENVS)")
    ap.add_argument("--task", type=int, default=1, help="eval task, 1..5")
    ap.add_argument("--prefix_lens", default="",
                    help="BC prefix lengths T0, comma-separated; default per env (cube 0,50,100; "
                         "antmaze 0,250,500)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="discount of the discounted outcome; default per env (cube 0.98, antmaze 0.99)")
    ap.add_argument("--num_u", type=int, default=8, help="K candidate first latents per anchor")
    ap.add_argument("--num_rollouts", type=int, default=100, help="R rollouts per candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="",
                    help="default $PSM_DATA/logs/diag_first_u_effect_bc_task{t}.json "
                         "(antmaze: ..._antmaze_task{t}.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="rollouts for 1 anchor (the largest T0), K=2, R=2; identity check for all T0")
    ap.add_argument("--flow", default="", help="Stage-A flow dir; default $PSM_DATA/flow/<the env's dataset>")
    ap.add_argument("--flow_epoch", type=int, default=None, help="default per env (500000)")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-task JSONs to pool into --out")
    a = ap.parse_args()
    if a.aggregate:
        out = a.out or os.path.join(os.environ["PSM_DATA"], "logs", default_out_name(a.env))
        _aggregate_cli(a.aggregate, out)
        return
    assert 1 <= a.task <= 5, a.task
    e = ENVS[a.env]
    a.prefix_lens = a.prefix_lens or e["prefix_lens"]
    a.gamma = e["gamma"] if a.gamma is None else a.gamma
    a.flow = a.flow or os.path.join(os.environ["PSM_DATA"], "flow", e["flow"])
    a.flow_epoch = e["flow_epoch"] if a.flow_epoch is None else a.flow_epoch
    run(a)


if __name__ == "__main__":
    main()
