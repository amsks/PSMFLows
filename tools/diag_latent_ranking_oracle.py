"""D1-analog for actor-free arms: does psi rank latent candidates the way an oracle does?

The original D1 (diag_policy_ranking.py) ranks a fixed roster of trained actor policies
and needs their eval JSONs; it does not transfer to a paper-faithful arm with no actor.
This probe asks the deployment-relevant version of the same question directly: at each
state, score the SAME K clipped prior latents the GPI acting path draws, once with the
checkpoint's critic (psi^T w, ensemble pessimism, max over a fixed policy-index panel
under policy_index=latent) and once with ground truth (negative distance of the decoded
action to a frozen FQL expert's action, as in E1). Report the per-state Spearman between
the two, and the oracle-distance regret of the critic's argmax pick vs the oracle's.

Spearman ~0 -> no ranking signal (the D1/D3 failure mode); high Spearman + low regret ->
the critic ranks, and deployed GPI should approach the E1 oracle-aim number.

Run (paper-faithful Arm B):
  MUJOCO_GL=egl .venv/bin/python tools/diag_latent_ranking_oracle.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=<npz> agent.use_point_preimage=true \
      agent.policy_index=latent agent.train_actor=false agent.acting=gpi \
      restore_path=<run_dir> restore_epoch=<ep> \
      +oracle_path=<fql_run_dir> +oracle_epoch=500000 \
      report_out=/data-local/amsks/PSMFLows/logs/d1a_latent_ranking_<tag>.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

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

N_STATES = 64          # states harvested from oracle rollouts (on-path, where it matters)
K = 128                # candidates per state, matching gpi_num_u order of magnitude
N_IDX = 16             # fixed policy-index panel (policy_index=latent only)
SEED = 0


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)

    agent = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, cfg.restore_path, int(cfg.restore_epoch))
    zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
    agent = agent.infer_eval_z(zb['next_observations'],
                               zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))
    w = agent.task_z
    d_a = int(agent.config['action_dim'])
    u_clip = float(agent.config['u_clip'])
    latent_index = agent.config.get('policy_index') == 'latent'

    oracle_path = cfg.get('oracle_path', None)
    assert oracle_path, 'needs +oracle_path=<frozen FQL expert run dir>'
    with open(os.path.join(str(oracle_path), 'flags.json')) as f:
        oracle_cfg = ml_collections.ConfigDict(_lists_to_tuples(json.load(f)['agent']))
    oracle = agents[oracle_cfg['agent_name']].create(
        cfg.seed, ex['observations'], ex['actions'], oracle_cfg)
    oracle = restore_agent(oracle, str(oracle_path), int(cfg.get('oracle_epoch', 500000)))

    idx_panel = jnp.clip(jax.random.normal(jax.random.PRNGKey(SEED + 1), (N_IDX, d_a)),
                         -u_clip, u_clip)

    @jax.jit
    def critic_scores(obs, u_b):
        if latent_index:
            obs_r = jnp.broadcast_to(obs, (K * N_IDX, *obs.shape))
            u_r = jnp.repeat(u_b, N_IDX, axis=0)
            idx_r = jnp.tile(idx_panel, (K, 1))
            qpsi = agent.psi(obs_r, idx_r, u_r)
            Qs = (qpsi * w).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, agent.config['num_parallel'])
            q = qmean - agent.config['actor_pessimism_penalty'] * qunc
            return q.reshape(K, N_IDX).max(axis=-1)
        obs_b = jnp.broadcast_to(obs, (K, *obs.shape))
        wb = jnp.broadcast_to(w, (K, *w.shape))
        qpsi = agent.psi(obs_b, wb, u_b)
        Qs = (qpsi * w).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, agent.config['num_parallel'])
        return qmean - agent.config['actor_pessimism_penalty'] * qunc

    @jax.jit
    def oracle_distances(obs, u_b, key):
        a_star = jnp.clip(oracle.sample_actions(obs, seed=key), -1.0, 1.0)
        obs_b = jnp.broadcast_to(obs, (K, *obs.shape))
        a = agent.decode(obs_b, u_b)
        d = jnp.linalg.norm(a - a_star[None], axis=-1)
        return jnp.where(jnp.isfinite(d), d, jnp.inf)

    # Harvest on-path states: roll the oracle, snapshot every few steps.
    states, rng = [], jax.random.PRNGKey(SEED)
    ep = 0
    while len(states) < N_STATES:
        ob, _ = eval_env.reset(seed=SEED + ep)
        done, t = False, 0
        while not done and len(states) < N_STATES:
            if t % 10 == 0:
                states.append(np.asarray(ob, dtype=np.float32))
            rng, k = jax.random.split(rng)
            a = np.asarray(jnp.clip(oracle.sample_actions(jnp.asarray(ob), seed=k),
                                    -1.0, 1.0))
            ob, _, term, trunc, _ = eval_env.step(a)
            done, t = bool(term or trunc), t + 1
        ep += 1

    rhos, regrets, top1_d, best_d = [], [], [], []
    for i, s in enumerate(states):
        rng, k_u, k_or = jax.random.split(rng, 3)
        u = jnp.clip(jax.random.normal(k_u, (K, d_a)), -u_clip, u_clip)
        obs = jnp.asarray(s)
        q = np.asarray(critic_scores(obs, u))
        d = np.asarray(oracle_distances(obs, u, k_or))
        rhos.append(_spearman(q, -d))
        i_q, i_d = int(np.argmax(q)), int(np.argmin(d))
        top1_d.append(float(d[i_q]))
        best_d.append(float(d[i_d]))
        regrets.append(float(d[i_q] - d[i_d]))

    report = {
        'probe': 'D1-analog: latent candidate ranking vs FQL oracle',
        'restore_path': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch),
        'policy_index': agent.config.get('policy_index'),
        'n_states': len(states), 'K': K, 'n_idx_panel': N_IDX if latent_index else None,
        'spearman_mean': float(np.mean(rhos)), 'spearman_median': float(np.median(rhos)),
        'spearman_frac_above_0.3': float(np.mean(np.asarray(rhos) > 0.3)),
        'regret_mean': float(np.mean(regrets)),
        'critic_pick_dist_mean': float(np.mean(top1_d)),
        'oracle_pick_dist_mean': float(np.mean(best_d)),
        'old_D1_baseline': {'spearman': 0.10, 'p': 0.78,
                            'note': 'policy-level D1 on the shipped agent'},
    }
    out = cfg.get('report_out', None)
    if out:
        os.makedirs(os.path.dirname(str(out)), exist_ok=True)
        with open(str(out), 'w') as f:
            json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)},
                     indent=2))


if __name__ == '__main__':
    main()
