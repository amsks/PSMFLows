"""D2 -- is the task vector the problem? (hypothesis H1)

Eval uses w = E[r . phi(s')] (agents/psmflow.py:infer_z), the least-squares projection of
the reward onto phi -- but only if phi is orthonormal under rho, and only if the reward is
actually in span(phi). phi is trained reward-free, so neither is guaranteed, and with ~2%
nonzero rewards the estimate is noisy on top. If psi is a fine measure but w points at the
wrong function, psi^T w is uninformative and the actor's -Q term is noise: the flat-curve,
chance-AUC picture, with nothing wrong in the actor at all.

Three measurements plus one comparison:
  (i)   R^2 of r ~ phi(s')^T w_inf on held-out transitions, against a ridge-regression
        topline on the same features. The gap is how much reward is in span(phi) that the
        closed form fails to pick up; a low topline means the features cannot express the
        reward at all, which is a different (worse) problem.
  (ii)  Where w_inf sits relative to the w distribution the measure was TRAINED on
        (sample_mixed_z: Gaussian, or phi(next_obs) projected). If eval queries the critic
        far outside its training support, psi^T w is extrapolation.
  (iii) Stability of w_inf across disjoint relabel batches -- the noise floor of the
        inference itself.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_task_projection.py agent=psmflow env_name=<env> \
    agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    restore_path=<run_dir> restore_epoch=500000 \
    report_out=/data-local/amsks/PSMFLows/logs/d2_task_projection_cube.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

import hydra
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from agents.psm import project_z
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_FIT = 20000       # transitions for the projection fit
N_HELDOUT = 20000
N_BATCHES = 5       # disjoint relabel batches for the stability check
RIDGE_LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 10.0]


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
    agent = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
    assert cfg.restore_path is not None, 'needs a trained checkpoint'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    shift = float(cfg.get('eval_reward_shift', 1.0))
    rng = np.random.default_rng(int(cfg.seed))

    def batch(n):
        idx = rng.integers(0, ds.size, n)
        obs = np.asarray(ds['next_observations'])[idx]
        rew = np.asarray(ds['rewards'])[idx] + shift
        return obs, rew

    # ---- (i) does the closed form represent the reward? --------------------------
    fit_obs, fit_r = batch(N_FIT)
    ho_obs, ho_r = batch(N_HELDOUT)
    phi_fit = np.asarray(agent.phi(jnp.asarray(fit_obs)))
    phi_ho = np.asarray(agent.phi(jnp.asarray(ho_obs)))

    w_closed = phi_fit.T @ fit_r / len(fit_r)          # the agent's own estimator
    w_closed_proj = np.asarray(project_z(jnp.asarray(w_closed)[None],
                                         config['norm_z']))[0]

    # Ridge topline on the same features: the best any linear read-out of phi can do.
    G = phi_fit.T @ phi_fit / len(fit_r)
    b = phi_fit.T @ fit_r / len(fit_r)
    best = {'r2': -np.inf}
    for lam in RIDGE_LAMBDAS:
        w_r = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
        s = r2(ho_r, phi_ho @ w_r)
        if s > best['r2']:
            best = {'r2': s, 'lambda': lam, 'w': w_r}

    fits = {
        'closed_form_r2_heldout': r2(ho_r, phi_ho @ w_closed),
        'closed_form_projected_r2_heldout': r2(ho_r, phi_ho @ w_closed_proj),
        'ridge_topline_r2_heldout': best['r2'],
        'ridge_lambda': best['lambda'],
        'reward_mean': float(ho_r.mean()), 'reward_std': float(ho_r.std()),
        'frac_nonzero_reward': float((np.abs(ho_r) > 1e-8).mean()),
        # How far phi is from the orthonormality the closed form assumes.
        'gram_offdiag_rms': float(np.sqrt((G - np.diag(np.diag(G))) ** 2).mean()),
        'gram_diag_mean': float(np.diag(G).mean()),
    }

    # ---- (ii) is w_inf inside the training-w distribution? -----------------------
    n = 4096
    idx = rng.integers(0, ds.size, n)
    goal_w = np.asarray(project_z(agent.phi(jnp.asarray(
        np.asarray(ds['next_observations'])[idx])), config['norm_z']))
    gauss_w = np.asarray(project_z(jnp.asarray(
        rng.standard_normal((n, config['z_dim']))), config['norm_z']))

    w_eval = np.asarray(agent.infer_eval_z(
        jnp.asarray(fit_obs), jnp.asarray(fit_r)).task_z)

    def cos_to(pool):
        wn = w_eval / (np.linalg.norm(w_eval) + 1e-12)
        pn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
        c = pn @ wn
        return {'max': float(c.max()), 'mean': float(c.mean()),
                'p95': float(np.percentile(c, 95))}

    placement = {
        'w_eval_norm': float(np.linalg.norm(w_eval)),
        'training_goal_w_norm_mean': float(np.linalg.norm(goal_w, axis=1).mean()),
        'training_gauss_w_norm_mean': float(np.linalg.norm(gauss_w, axis=1).mean()),
        'cosine_to_goal_component': cos_to(goal_w),
        'cosine_to_gaussian_component': cos_to(gauss_w),
    }

    # ---- (iii) how stable is the inference? --------------------------------------
    ws = []
    for _ in range(N_BATCHES):
        o, r = batch(N_FIT)
        ws.append(np.asarray(agent.infer_eval_z(jnp.asarray(o), jnp.asarray(r)).task_z))
    ws = np.stack(ws)
    wn = ws / (np.linalg.norm(ws, axis=1, keepdims=True) + 1e-12)
    cross = wn @ wn.T
    off = cross[~np.eye(N_BATCHES, dtype=bool)]
    stability = {'n_batches': N_BATCHES, 'batch_size': N_FIT,
                 'pairwise_cosine_mean': float(off.mean()),
                 'pairwise_cosine_min': float(off.min())}

    closed, top = fits['closed_form_r2_heldout'], fits['ridge_topline_r2_heldout']
    # Two independent questions, so two verdicts. Collapsing them hides the case that
    # actually obtains here: inference is not lossy, but the basis it projects onto
    # explains little of the reward. "Not the bottleneck" would be the wrong summary of
    # a low topline with a small gap.
    lossy = closed < 0.8 * top
    expressive = top >= 0.5
    inference_verdict = (
        f'lossy: the closed form recovers R^2 {closed:.3f} of an available {top:.3f}'
        if lossy else
        f'not lossy: closed form R^2 {closed:.3f} matches the ridge topline {top:.3f}, so '
        f'the estimator extracts what the features contain')
    basis_verdict = (
        f'expressive: phi explains R^2 {top:.3f} of the reward'
        if expressive else
        f'weak: the best linear read-out of phi explains only R^2 {top:.3f} of the reward '
        f'(rewards are {100 * fits["frac_nonzero_reward"]:.1f}% nonzero, so this is a '
        f'sparse target -- but psi^T w can be no better than the w it is handed)')
    stable = stability['pairwise_cosine_min'] >= 0.9
    if lossy:
        verdict = f'H1 supported (task inference) -- {inference_verdict}'
    elif not expressive:
        verdict = (f'H1 relocated: inference is fine, the basis is not. {basis_verdict}. '
                   f'The fix is a phi-grounding auxiliary, not a better estimator.')
    elif not stable:
        verdict = (f'H1 partial: {inference_verdict}, but w_inf is unstable across relabel '
                   f'batches (min pairwise cosine {stability["pairwise_cosine_min"]:.3f})')
    else:
        verdict = f'H1 rejected: {inference_verdict}; {basis_verdict}; w_inf stable'

    report = {'env': cfg.env_name, 'restore_path': str(cfg.restore_path),
              'restore_epoch': int(cfg.restore_epoch), 'seed': int(cfg.seed),
              'reward_fit': fits, 'w_placement': placement, 'w_stability': stability,
              'inference_verdict': inference_verdict, 'basis_verdict': basis_verdict,
              'w_stable': stable, 'verdict': verdict}
    print(f"\nclosed-form R^2 {closed:.4f} | ridge topline {top:.4f} | "
          f"w stability {stability['pairwise_cosine_min']:.4f}\nVERDICT: {verdict}")
    write_report(report, cfg, 'diag_task_projection.json')


if __name__ == '__main__':
    main()
