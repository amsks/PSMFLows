"""Roll a trained psmflow policy and record what it actually does, step by step.

`evaluate` returns success rates; this returns the trajectory. Per step it stores the
observation, the latent the actor emitted, the decoded action, and the reward, so the
support claim ("every executed action is a flow decode of an in-typical-set latent")
becomes something measured rather than asserted, and maze trajectories can be drawn.

Also records the predicted diagonal score psi(s, w, u)^T w at the FIRST state of each
episode next to the realised discounted return of that episode -- the calibration of the
value function for the policy that is actually deployed. The existing
tools/calibration_check.py scores fixed-u candidates, which the latent-PSM redesign made
meaningless: nothing rolls a fixed u any more.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/viz_policy_rollouts.py agent=psmflow \
    env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    restore_path=<run_dir> restore_epoch=500000 eval_episodes=20 \
    report_out=/data-local/amsks/PSMFLows/logs/rollouts_cube_sd0.json
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

from agents import agents
from agents.psm import targets_uncertainty
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report


def actor_latent(agent, obs, seed):
    """The latent the actor emits at `obs` -- the quantity sample_actions decodes."""
    noise = jax.random.normal(seed, (1, agent.config["action_dim"]))
    return agent.config["u_clip"] * agent.actor(
        obs[None], agent.task_z[None], noise)[0]


def diagonal_score(agent, obs, u):
    """psi(s, w, u)^T w with the same ensemble pessimism the actor was trained against."""
    qpsi = agent.psi(obs[None], agent.task_z[None], u[None])       # (P, 1, z_dim)
    Qs = (qpsi * agent.task_z).sum(-1)                             # (P, 1)
    qmean, qunc = targets_uncertainty(Qs, agent.config["num_parallel"])
    return float(qmean[0] - agent.config["actor_pessimism_penalty"] * qunc[0])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    name = config["agent_name"]
    assert name == "psmflow", "this tool records the latent actor's rollouts"

    ex = ds.sample(1)
    agent = agents[name].create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Task vector exactly as main.py's eval block and eval_checkpoint.py infer it.
    n_relabel = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
    zb = ds.sample(n_relabel)
    shift = float(cfg.get("eval_reward_shift", 1.0))
    agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + shift)

    n_ep = int(cfg.eval_episodes)
    discount = float(config["discount"])
    rng = jax.random.PRNGKey(int(cfg.seed))

    episodes, obs_all, u_all, act_all = [], [], [], []
    for ep in range(n_ep):
        observation, _ = eval_env.reset(seed=int(cfg.seed) + ep)
        done, t, ret, disc_ret = False, 0, 0.0, 0.0
        success = 0.0
        xy, us, acts = [], [], []
        first_score = None
        while not done:
            rng, key = jax.random.split(rng)
            ob = jnp.asarray(observation, dtype=jnp.float32)
            u = actor_latent(agent, ob, key)
            if first_score is None:
                first_score = diagonal_score(agent, ob, u)
            action = np.asarray(agent.decode(ob[None], u[None])[0])
            action = np.clip(action, -1.0, 1.0)

            xy.append(np.asarray(observation, dtype=np.float32)[:2])
            us.append(np.asarray(u, dtype=np.float32))
            acts.append(action.astype(np.float32))

            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ret += float(reward)
            disc_ret += (discount ** t) * float(reward)
            success = max(success, float(info.get("success", 0.0)))
            t += 1

        episodes.append({
            "episode": ep, "steps": t, "return": ret, "discounted_return": disc_ret,
            "success": success, "predicted_score_at_s0": first_score,
        })
        obs_all.append(np.stack(xy))
        u_all.append(np.stack(us))
        act_all.append(np.stack(acts))
        print(f"ep {ep:3d}  steps {t:4d}  success {success:.0f}  "
              f"disc.return {disc_ret:8.2f}  predicted {first_score:8.2f}", flush=True)

    u_cat = np.concatenate(u_all)
    u_clip = float(config["u_clip"])
    d_a = u_cat.shape[1]
    norm2 = (u_cat.astype(np.float64) ** 2).sum(1)

    # Spearman without scipy: Pearson correlation of the ranks.
    pred = np.array([e["predicted_score_at_s0"] for e in episodes], dtype=np.float64)
    real = np.array([e["discounted_return"] for e in episodes], dtype=np.float64)
    # Average ranks for ties. This matters here, not as a nicety: every failed episode
    # scores the identical return, so most of the sample is one big tie group and
    # arbitrary tie-breaking would manufacture a correlation out of array order.
    def _rank(a):
        order = a.argsort(kind="stable")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        srt = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and srt[j + 1] == srt[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    if len(episodes) > 2 and pred.std() > 0 and real.std() > 0:
        rp, rr = _rank(pred), _rank(real)
        spearman = float(np.corrcoef(rp, rr)[0, 1]) if rr.std() > 0 else None
    else:
        spearman = None

    # With returns dominated by ties, whether the score separates success from failure is
    # the more honest question. AUC = P(score of a success > score of a failure), 0.5 =
    # chance, computed from the rank sum so ties count as half.
    succ = np.array([e["success"] > 0.5 for e in episodes])
    n_pos, n_neg = int(succ.sum()), int((~succ).sum())
    if n_pos and n_neg:
        rk = _rank(pred) + 1.0
        auc = float((rk[succ].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = None

    npz_out = str(cfg.get("report_out", "rollouts.json")).replace(".json", ".npz")
    np.savez_compressed(
        npz_out,
        latents=u_cat.astype(np.float32),
        actions=np.concatenate(act_all).astype(np.float32),
        xy=np.concatenate(obs_all).astype(np.float32),
        episode_lengths=np.array([e["steps"] for e in episodes]),
        episode_success=np.array([e["success"] for e in episodes]),
    )

    report = {
        "env": cfg.env_name, "agent": name, "acting": config.get("acting"),
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "seed": int(cfg.seed), "num_episodes": n_ep, "discount": discount,
        "action_dim": d_a, "u_clip": u_clip,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "actor_latent": {
            "n_steps": int(u_cat.shape[0]),
            "mean_norm2": float(norm2.mean()), "expected_norm2_under_prior": d_a,
            "median_norm2": float(np.median(norm2)),
            # The typical set of the prior: ||u||^2 within the central 99% of chi^2_{d_a}.
            "frac_within_prior_99": float(np.mean(norm2 <= _chi2_q(0.99, d_a))),
            "frac_at_clip_boundary": float(np.mean(np.abs(u_cat) >= 0.999 * u_clip)),
            "per_dim_std": u_cat.std(0).tolist(),
        },
        "calibration": {
            "spearman_predicted_vs_realised": spearman,
            "auc_predicted_vs_success": auc,
            "n_success": n_pos, "n_failure": n_neg,
            "note": ("returns are heavily tied (every failure scores the same), so AUC "
                     "against binary success is the load-bearing statistic; 0.5 = chance"),
            "predicted_at_s0": pred.tolist(),
            "discounted_return": real.tolist(),
        },
        "episodes": episodes,
        "npz": npz_out,
    }
    print(f"\nsuccess {report['success_rate']:.3f} | actor latent mean ||u||^2 "
          f"{norm2.mean():.2f} (prior {d_a}) | Spearman {spearman} | AUC {auc}")
    write_report(report, cfg, "viz_policy_rollouts.json")


def _chi2_q(p, k, hi=200.0, steps=20000):
    """Quantile of chi^2_k by trapezoid CDF -- avoids a scipy dependency."""
    from math import lgamma, log
    x = np.linspace(1e-9, hi, steps)
    logpdf = ((k / 2 - 1) * np.log(x) - x / 2 - (k / 2) * log(2.0) - lgamma(k / 2))
    cdf = np.cumsum(np.exp(logpdf)) * (x[1] - x[0])
    return float(x[np.searchsorted(cdf, p * cdf[-1])])


if __name__ == "__main__":
    main()
