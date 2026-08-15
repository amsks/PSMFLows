"""Calibration: does the critic's Q at the start state predict the realized return?

Built for the W4/l1stab collapse forensics, but agent-agnostic: works for any agent
exposing `critic(obs, actions)` (latentrl, fql). At N seeded eval episodes, record
  Q(s0, a0)      the agent's own value estimate of its first action (ensemble mean and
                 the pessimistic value it actually optimizes), and
  G(s0)          the realized discounted return of the rollout under the env reward,
and report bias = mean(Q - G), the ratio of means, and the Spearman rank correlation
across episodes. An overestimation spiral shows up as a large positive bias on the
collapsed checkpoint that is absent (or small) at the same run's peak checkpoint.

Also runs the pre-registered 2a Q-GAP probe at dataset states:

  gap(s) = Q(s, a_agent(s)) - Q(s, a_data(s))

with a_data the logged action at s (and, as a geometry-robust variant, the best action
among the k=32 nearest dataset states under the shared `utils.geometry` protocol). This
is the half of the calibration verdict the paper cites but the tool never computed: a
large positive gap on states where the agent's action is far from the data says the
critic prefers actions it has no in-sample evidence for, which is over-estimation on
off-support actions rather than the on-policy return bias the rollout half measures.

Run:
  MUJOCO_GL=egl .venv/bin/python tools/diag_fql_calibration.py agent=latentrl \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=<npz> agent.use_point_preimage=true \
      agent.residual_eps=<eps> \
      restore_path=<run_dir> restore_epoch=<step> \
      report_out=/data-local/amsks/PSMFLows/logs/<name>.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf
from scipy import stats as sstats

from agents import agents
from agents.psm import targets_uncertainty
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.geometry import INDEX_ROWS, K_NEIGHBOURS, NeighbourIndex
from utils.log_utils import write_report

QGAP_STATES = 512       # dataset states for the 2a Q-gap probe


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    n_episodes = int(cfg.get("calib_episodes", 64))
    np.random.seed(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))

    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    if hasattr(agent, "critic"):
        critic_fn = agent.critic
    else:
        # FQL keeps its critic inside the ModuleDict (network.select), not as an attribute.
        assert hasattr(agent, "network"), "agent exposes no critic; nothing to calibrate"
        critic_fn = lambda o, a: agent.network.select("critic")(o, actions=a)

    gamma = float(config["discount"])
    num_parallel = int(config.get("num_parallel", 2))
    # Two separate pessimism knobs exist and they are NOT interchangeable: the critic's
    # TD target uses `pessimism_penalty`, while the value the policy is actually pushed
    # up -- the one whose calibration this tool is about -- uses `actor_pessimism_penalty`
    # (agents/latentrl.py:actor_loss). The tool previously applied the target knob, and
    # computed the spread as |q0-q1|/2 while the agent's own `targets_uncertainty` gives
    # |q0-q1| at P=2. Both errors shrank the penalty, so every earlier report UNDERSTATED
    # how pessimistic the optimized value is: the true pessimistic value sits LOWER, so
    # the measured over-estimation is at least as large as reported. Direction of every
    # prior collapse verdict is unchanged; magnitudes were conservative.
    pess_actor = float(config.get("actor_pessimism_penalty",
                                  config.get("pessimism_penalty", 0.0)))
    pess_target = float(config.get("pessimism_penalty", 0.0))

    def q_ensemble(obs, act):
        """(P, B) critic values, one row per ensemble member."""
        qs = np.asarray(critic_fn(np.atleast_2d(np.asarray(obs)),
                                  np.atleast_2d(np.asarray(act))))
        return qs.reshape(num_parallel, -1)

    def q_batch(obs, act):
        return q_ensemble(obs, act).mean(0)

    def q_stats(obs, act):
        qs = q_ensemble(obs, act)[:, 0]                       # (P,)
        q_mean, q_unc = targets_uncertainty(jnp.asarray(qs), num_parallel)
        q_mean, q_unc = float(q_mean), float(q_unc)
        return (q_mean, q_mean - pess_actor * q_unc, q_mean - pess_target * q_unc, q_unc)

    # Same env seeding convention as evaluate(): one seed pin, unseeded resets after.
    eval_env.reset(seed=int(cfg.seed))
    rng = jax.random.PRNGKey(int(cfg.seed))

    rows = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        done, t, g, disc = False, 0, 0.0, 1.0
        q_m = q_p = q_pt = q_u = None
        while not done:
            rng, key = jax.random.split(rng)
            act = np.asarray(agent.sample_actions(obs, seed=key))
            if t == 0:
                q_m, q_p, q_pt, q_u = q_stats(np.asarray(obs), act)
            obs, r, terminated, truncated, info = eval_env.step(act)
            g += disc * float(r)
            disc *= gamma
            t += 1
            done = terminated or truncated
        rows.append({"q_mean": q_m, "q_pessimistic": q_p,
                     "q_pessimistic_target_knob": q_pt, "q_ensemble_spread": q_u,
                     "realized_return": g, "episode_length": t,
                     "success": float(np.max(info.get("success", 0.0)))})

    # ---- 2a Q-gap probe: does the critic prefer the agent's action to the data's? -----
    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])
    index = NeighbourIndex(obs_all, act_all, index_rows=INDEX_ROWS, seed=int(cfg.seed))
    qrows = np.random.default_rng(int(cfg.seed)).choice(
        index.rows, size=min(QGAP_STATES, len(index.rows)), replace=False)
    s_q, a_data = obs_all[qrows], act_all[qrows]

    gkeys = jax.random.split(jax.random.PRNGKey(int(cfg.seed) + 77), len(qrows))
    a_agent = np.stack([np.asarray(agent.sample_actions(s_q[i], seed=gkeys[i]))
                        for i in range(len(qrows))])
    q_agent, q_data = q_batch(s_q, a_agent), q_batch(s_q, a_data)

    # Best action the critic can find among the k=32 nearest dataset states' actions --
    # the strongest IN-SAMPLE competitor, so the gap is not merely "better than one
    # arbitrary logged action".
    neigh = index.neighbour_actions(s_q, k=K_NEIGHBOURS, exclude_rows=qrows)
    q_nn_best = np.array([
        q_batch(np.repeat(s_q[i][None], K_NEIGHBOURS, axis=0), neigh[i]).max()
        for i in range(len(qrows))])

    # How far off the local action cloud the agent's action sits, and what "close" means
    # for real data (the same statistic on the logged actions, self excluded).
    d_agent = index.min_action_dist(s_q, a_agent, exclude_rows=qrows)
    d_data = index.min_action_dist(s_q, a_data, exclude_rows=qrows)
    d95 = float(np.percentile(d_data, 95))
    off = d_agent > d95
    gap_data = q_agent - q_data
    gap_nn = q_agent - q_nn_best
    q_scale = float(np.abs(q_data).mean()) + 1e-8
    sp_gap = sstats.spearmanr(d_agent, gap_data)

    qgap = {
        "n_states": int(len(qrows)),
        "geometry": index.protocol(),
        "q_agent_mean": float(q_agent.mean()),
        "q_data_mean": float(q_data.mean()),
        "q_knn_best_mean": float(q_nn_best.mean()),
        "gap_vs_data_mean": float(gap_data.mean()),
        "gap_vs_data_median": float(np.median(gap_data)),
        "gap_vs_data_normalised": float(gap_data.mean() / q_scale),
        "gap_vs_knn_best_mean": float(gap_nn.mean()),
        "gap_vs_knn_best_normalised": float(gap_nn.mean() / q_scale),
        "frac_states_agent_preferred": float((gap_data > 0).mean()),
        "agent_action_dist_mean": float(d_agent.mean()),
        "data_self_match_p95": d95,
        "frac_agent_actions_offsupport": float(off.mean()),
        "gap_on_offsupport_states": float(gap_data[off].mean()) if off.any() else None,
        "gap_on_insupport_states": float(gap_data[~off].mean()) if (~off).any() else None,
        "spearman_gap_vs_action_distance": float(sp_gap.statistic),
        "spearman_pvalue": float(sp_gap.pvalue),
    }
    qgap["verdict"] = (
        f"critic prefers its own action to the data's on {100 * qgap['frac_states_agent_preferred']:.0f}% "
        f"of states (mean gap {qgap['gap_vs_data_mean']:+.3f} = "
        f"{100 * qgap['gap_vs_data_normalised']:+.0f}% of |Q_data|; vs the best in-sample "
        f"k-NN action {qgap['gap_vs_knn_best_mean']:+.3f}); "
        f"{100 * qgap['frac_agent_actions_offsupport']:.0f}% of its actions are beyond the "
        f"data's own p95 match distance, and gap-vs-distance rank correlation is "
        f"{qgap['spearman_gap_vs_action_distance']:+.2f}")

    qm = np.array([r["q_mean"] for r in rows])
    qp = np.array([r["q_pessimistic"] for r in rows])
    G = np.array([r["realized_return"] for r in rows])
    spear = sstats.spearmanr(qm, G)
    bias = float((qm - G).mean())
    report = {
        "env": cfg.env_name,
        "agent": config["agent_name"],
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch),
        "seed": int(cfg.seed),
        "n_episodes": n_episodes,
        "gamma": gamma,
        "pessimism_knob_actor": pess_actor,
        "pessimism_knob_target": pess_target,
        "penalty_note": (
            "08-14 fix: the pessimistic value now uses the ACTOR's pessimism knob and the "
            "agent's own targets_uncertainty (|q0-q1| at P=2, not |q0-q1|/2). Reports "
            "written before this understated the penalty, i.e. understated how far the "
            "optimized value sits above the realized return; every prior verdict's "
            "direction is unchanged."),
        "q_mean_avg": float(qm.mean()),
        "q_pessimistic_avg": float(qp.mean()),
        "realized_return_avg": float(G.mean()),
        "bias_q_minus_realized": bias,
        "bias_pessimistic": float((qp - G).mean()),
        "abs_ratio_q_over_realized": float(np.abs(qm.mean()) / (np.abs(G.mean()) + 1e-8)),
        "spearman_q_vs_realized": float(spear.statistic),
        "spearman_pvalue": float(spear.pvalue),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "qgap_2a": qgap,
        "per_episode": rows,
    }
    print(f"\n[2a Q-gap] {qgap['verdict']}")
    write_report(report, cfg, "diag_fql_calibration.json")


if __name__ == "__main__":
    main()
