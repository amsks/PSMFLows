"""P0 -- does FB's backward map carry the reward, and does its Q have relief?

This is the deciding probe of docs/plans/2026-08-10-psm-fix-roadmap.md. Our psmflow basis
reads R^2 ~ 0.13 for the best linear reward read-out and its Q varies by ~1% of |Q| over
the latent; FB, on the same env, dataset and budget, scores 0.62-0.82. Two possibilities:

  Branch A  FB's B reads a much higher R^2 and its Q has real relief. Then the property we
            lack is one FB demonstrably has, and our measure objective is the defect ->
            fix the PSM loss (P1).
  Branch B  FB's B also reads ~0.13 yet FB still wins. Then a linear reward read-out is
            simply not the lever, and what we lack is the improvement loop, not the
            representation -> LatentFB is promoted.

Measured identically to tools/diag_task_projection.py (closed form vs ridge topline on
held-out transitions) so the two numbers are directly comparable, plus a relief probe
matching tools/diag_q_landscape.py: spread of Q(s,a,z) = F(s,z,a)^T z over dataset actions
at fixed (s, z), relative to |Q|, at the same 64 evaluation start states.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_fb_basis_probe.py agent=fb agent.actor.type=flow \
    agent.ortho_coef=1000 env_name=cube-single-play-singletask-v0 \
    report_out=/data-local/amsks/PSMFLows/logs/p0_fb_basis_probe.json
"""
import glob
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
from utils.log_utils import write_report

RUNS = sorted(glob.glob(
    '/data-local/amsks/PSMFLows/exp/PSMFLows/fb_cube_ortho1000_20260810/sd*'))
EPOCH = 500000
N_FIT = 20000
N_HELDOUT = 20000
N_STATES = 64
N_ACTIONS = 512
RIDGE_LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
SEED = 0

#: our psmflow numbers, for the branch decision (logs/d2_*, logs/d3_*)
PSMFLOW_R2 = 0.129
PSMFLOW_Q_SPREAD = 0.011


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)

    states = []
    for i in range(N_STATES):
        ob, _ = eval_env.reset(seed=SEED + i)
        states.append(np.asarray(ob, dtype=np.float32))
    states = jnp.asarray(np.stack(states))

    shift = float(cfg.get('eval_reward_shift', 1.0))
    rng_np = np.random.default_rng(SEED)

    def batch(n):
        idx = rng_np.integers(0, ds.size, n)
        return (np.asarray(ds['next_observations'])[idx],
                np.asarray(ds['rewards'])[idx] + shift,
                np.asarray(ds['actions'])[idx])

    per_seed, rng = [], jax.random.PRNGKey(SEED)
    for run in RUNS:
        agent = agents['fb'].create(cfg.seed, ex['observations'], ex['actions'], config)
        agent = restore_agent(agent, run, EPOCH)

        # ---- does B carry the reward? (same protocol as diag_task_projection) ------
        fit_obs, fit_r, _ = batch(N_FIT)
        ho_obs, ho_r, _ = batch(N_HELDOUT)
        B_fit = np.asarray(agent._apply('backward', agent.params['backward'],
                                        jnp.asarray(fit_obs)))
        B_ho = np.asarray(agent._apply('backward', agent.params['backward'],
                                       jnp.asarray(ho_obs)))
        w_closed = B_fit.T @ fit_r / len(fit_r)
        G = B_fit.T @ B_fit / len(fit_r)
        b = B_fit.T @ fit_r / len(fit_r)
        best = {'r2': -np.inf}
        for lam in RIDGE_LAMBDAS:
            w_r = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
            s = r2(ho_r, B_ho @ w_r)
            if s > best['r2']:
                best = {'r2': s, 'lambda': lam}

        # ---- does Q have relief over actions at fixed (s, z)? ---------------------
        z = agent.infer_z(jnp.asarray(fit_obs), jnp.asarray(fit_r))
        spreads = []
        for i in range(N_STATES):
            rng, k = jax.random.split(rng)
            # Dataset actions: the in-support set FB's actor picks among, matching how
            # diag_q_landscape probes our latent prior draws.
            idx = rng_np.integers(0, ds.size, N_ACTIONS)
            acts = jnp.asarray(np.asarray(ds['actions'])[idx])
            obs_b = jnp.broadcast_to(states[i], (N_ACTIONS, states.shape[1]))
            zb = jnp.broadcast_to(z, (N_ACTIONS, *z.shape))
            left = agent._apply('left_encoder', agent.params['left_encoder'], obs_b)
            Fs = agent._apply('forward', agent.params['forward'], left, zb, acts)
            Qs = (Fs * zb).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, config['num_parallel'])
            Q = np.asarray(qmean - config['actor_pessimism_penalty'] * qunc)
            spreads.append(float(Q.std()) / (abs(float(Q.mean())) + 1e-8))

        per_seed.append({
            'run': os.path.basename(run),
            'closed_form_r2_heldout': r2(ho_r, B_ho @ w_closed),
            'ridge_topline_r2_heldout': best['r2'], 'ridge_lambda': best['lambda'],
            'q_relative_spread_mean': float(np.mean(spreads)),
            'q_relative_spread_median': float(np.median(spreads)),
        })
        print(f"{per_seed[-1]['run']}: B closed-form R^2 "
              f"{per_seed[-1]['closed_form_r2_heldout']:.4f} | ridge "
              f"{per_seed[-1]['ridge_topline_r2_heldout']:.4f} | Q spread "
              f"{per_seed[-1]['q_relative_spread_mean']:.4f}", flush=True)

    fb_r2 = float(np.mean([p['ridge_topline_r2_heldout'] for p in per_seed]))
    fb_spread = float(np.mean([p['q_relative_spread_mean'] for p in per_seed]))

    if fb_r2 >= 0.4 and fb_spread >= 0.05:
        branch, verdict = 'A', (
            f"Branch A: FB's B carries the reward (ridge R^2 {fb_r2:.3f} vs our "
            f"{PSMFLOW_R2:.3f}) and its Q has relief ({fb_spread:.3f} vs our "
            f"{PSMFLOW_Q_SPREAD:.3f}). The property we lack is one FB has, so the measure "
            f"objective is the defect -> P1 (fix the PSM loss). LatentFB stays parked.")
    elif fb_r2 <= 0.2:
        branch, verdict = 'B', (
            f"Branch B: FB's B reads R^2 {fb_r2:.3f}, no better than ours "
            f"({PSMFLOW_R2:.3f}), yet FB scores 0.62-0.82. A linear reward read-out is NOT "
            f"the lever -- FB wins through its improvement loop. Promote LatentFB (P3); "
            f"the PSM-internal fix narrows to P1b.")
    else:
        branch, verdict = 'middle', (
            f"Ambiguous: FB ridge R^2 {fb_r2:.3f} (ours {PSMFLOW_R2:.3f}), Q spread "
            f"{fb_spread:.3f} (ours {PSMFLOW_Q_SPREAD:.3f}). Run P1 but gate it hard.")

    report = {'env': cfg.env_name, 'runs': RUNS, 'epoch': EPOCH,
              'n_fit': N_FIT, 'n_heldout': N_HELDOUT, 'n_states': N_STATES,
              'n_actions': N_ACTIONS, 'seed': SEED,
              'per_seed': per_seed,
              'fb_ridge_r2_mean': fb_r2, 'fb_q_relative_spread_mean': fb_spread,
              'psmflow_reference': {'ridge_r2': PSMFLOW_R2,
                                    'q_relative_spread': PSMFLOW_Q_SPREAD},
              'branch': branch, 'verdict': verdict}
    print(f'\nFB ridge R^2 {fb_r2:.4f} (ours {PSMFLOW_R2}) | FB Q spread {fb_spread:.4f} '
          f'(ours {PSMFLOW_Q_SPREAD})\nBRANCH {branch}: {verdict}')
    write_report(report, cfg, 'p0_fb_basis_probe.json')


if __name__ == '__main__':
    main()
