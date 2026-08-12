"""L0 -- does the 0.577-vs-0.187 ordering survive matched statistics and a different
neighbourhood geometry?

The two-layer support argument (decoder subset local behaviour subset task-useful actions)
rests entirely on C1's comparison, and C1 compared two quantities that were not matched:

  d_flow  min over 512 decoded candidates of ||a_FQL(s) - G(s, u_j)||
  d_data  min over k=32 nearest-neighbour dataset actions of ||a_FQL(s) - a_i||

A minimum over 512 draws is smaller than a minimum over 32 for purely combinatorial
reasons, so part of C1's gap could be candidate count rather than geometry. This tool
reports both at MATCHED candidate counts, and re-runs the whole comparison under a
different notion of "nearby state" (per-dimension standardized observations) and over
k in {16, 32, 64}.

Conditioning asymmetry, stated because it does not disappear under any of the above:
d_flow conditions on the exact state s (the decoder is queried at s), while d_data
conditions on a NEIGHBOURHOOD of s. The data comparison is therefore handicapped by
however fast the behaviour policy varies in state space. Matched counts remove the
combinatorial advantage; they do not remove this one, and a surviving ordering should be
read as "the decoder interpolates at least as close as nearby data does", not as a
statement about the conditional at s alone.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_geometry_robustness.py agent=psmflow \
    env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    report_out=/data-local/amsks/PSMFLows/logs/diag_action_coverage_cube_robust.json
"""
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
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from tools.diag_action_coverage import (FQL_ALPHA, FQL_EPOCH, FQL_RUN, N_LATENTS,
                                        N_STATES, SEED, flow_trained_steps)
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

KS = [16, 32, 64]
POOL = 200000


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
    ode_steps = flow_trained_steps(config['flow_ckpt_path'])

    fql_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(
        OmegaConf.load(os.path.join('configs', 'agent', 'fql.yaml')), resolve=True)))
    fql_cfg['alpha'] = FQL_ALPHA
    fql = agents['fql'].create(cfg.seed, ex['observations'], ex['actions'], fql_cfg)
    fql = restore_agent(fql, FQL_RUN, FQL_EPOCH)

    rng_np = np.random.default_rng(SEED)
    obs_all, act_all = np.asarray(ds['observations']), np.asarray(ds['actions'])
    idx = rng_np.integers(0, ds.size, N_STATES)
    states = obs_all[idx]
    pool_idx = rng_np.integers(0, ds.size, POOL)
    pool_obs, pool_act = obs_all[pool_idx], act_all[pool_idx]
    a_scale = float(np.abs(act_all).mean())

    states_j = jnp.asarray(states, jnp.float32)
    a_fql = np.asarray(jax.vmap(lambda s, k: fql.sample_actions(s, seed=k))(
        states_j, jax.random.split(jax.random.PRNGKey(SEED + 1), N_STATES)))

    # Decoded candidates once per state; every setting below reuses them.
    a2 = agent.replace(config=agent.config.copy(
        {'gpi_decode': 'onestep', 'flow_decode_steps': ode_steps}))
    decodes = []
    for i in range(N_STATES):
        u = jnp.clip(jax.random.normal(jax.random.PRNGKey(SEED + 900 + i),
                                       (N_LATENTS, agent.config['action_dim'])), -3.0, 3.0)
        ob_b = jnp.broadcast_to(states_j[i], (N_LATENTS, states_j.shape[1]))
        decodes.append(np.asarray(a2.decode(ob_b, u)))

    # Per-dim standardization: raw Euclidean distance in observation space lets
    # large-scale dimensions dominate which states count as "nearby".
    ob_sd = obs_all.std(0) + 1e-8
    geoms = {'raw': (pool_obs, states), 'standardized': (pool_obs / ob_sd, states / ob_sd)}

    settings, rng_sub = [], np.random.default_rng(SEED + 1)
    for geom, (p_obs, st) in geoms.items():
        for k in KS:
            d_flow_full, d_flow_matched, d_data, selfref, cover = [], [], [], [], []
            for i in range(N_STATES):
                d = np.linalg.norm(p_obs - st[i], axis=1)
                nn = np.argpartition(d, k)[:k]
                nn_acts = pool_act[nn]
                d_data.append(float(np.min(np.linalg.norm(nn_acts - a_fql[i], axis=1))))
                dec = decodes[i]
                d_flow_full.append(float(np.min(np.linalg.norm(dec - a_fql[i], axis=1))))
                # Matched candidate count: k decodes against k dataset actions.
                sub = rng_sub.choice(len(dec), size=k, replace=False)
                d_flow_matched.append(
                    float(np.min(np.linalg.norm(dec[sub] - a_fql[i], axis=1))))
                h = k // 2
                selfref.append(float(np.mean(nn_acts[:h].std(0) /
                                             (nn_acts[h:].std(0) + 1e-8))))
                cover.append(float(np.mean(dec.std(0) / (nn_acts.std(0) + 1e-8))))
            settings.append({
                'geometry': geom, 'k': k,
                'd_flow_512_norm': float(np.mean(d_flow_full)) / a_scale,
                'd_flow_matched_k_norm': float(np.mean(d_flow_matched)) / a_scale,
                'd_data_k_norm': float(np.mean(d_data)) / a_scale,
                'coverage_ratio_mean': float(np.mean(cover)),
                'split_half_self_reference': float(np.mean(selfref)),
                'ordering_holds_matched': bool(np.mean(d_flow_matched) < np.mean(d_data)),
                'ratio_matched': float(np.mean(d_data) / (np.mean(d_flow_matched) + 1e-12)),
            })
            print(f"{geom:12s} k={k:3d}  d_flow(512) {settings[-1]['d_flow_512_norm']:.3f}  "
                  f"d_flow(k) {settings[-1]['d_flow_matched_k_norm']:.3f}  "
                  f"d_data(k) {settings[-1]['d_data_k_norm']:.3f}  "
                  f"ratio {settings[-1]['ratio_matched']:.2f}", flush=True)

    holds = [s for s in settings if s['ordering_holds_matched']]
    ratios = [s['ratio_matched'] for s in settings]
    survives = len(holds) == len(settings)
    if survives and min(ratios) >= 1.5:
        verdict = (f'ORDERING SURVIVES: at matched candidate counts the decoder is closer '
                   f'to the FQL action than nearby data is, in all {len(settings)} '
                   f'geometry x k settings (ratio {min(ratios):.2f}-{max(ratios):.2f}). '
                   f'The two-layer framing stands; propagate to note.tex.')
    elif survives:
        verdict = (f'ORDERING SURVIVES BUT NARROWLY: holds in all settings, smallest '
                   f'margin {min(ratios):.2f}x. Report the matched numbers, not C1\'s '
                   f'512-vs-32 pair, and soften the claim to "at least as close".')
    else:
        verdict = (f'ORDERING DOES NOT SURVIVE: it fails in '
                   f'{len(settings) - len(holds)}/{len(settings)} settings '
                   f'(ratios {min(ratios):.2f}-{max(ratios):.2f}). C1\'s gap was partly '
                   f'candidate count. Update note.tex to the matched numbers and revisit '
                   f'the lever ranking before L2/L3.')

    report = {
        'env': cfg.env_name, 'flow_ckpt_path': str(config['flow_ckpt_path']),
        'n_states': N_STATES, 'n_decodes': N_LATENTS, 'ks': KS, 'seed': SEED,
        'action_scale_mean_abs': a_scale, 'settings': settings,
        'c1_reference': {'d_flow_512_norm': 0.187, 'd_data_k32_norm': 0.577},
        'notes': (
            'd_flow conditions on the exact state; d_data conditions on a NEIGHBOURHOOD '
            'of it, so the data side is handicapped by how fast the behaviour policy '
            'varies in state space. Matched candidate counts remove the combinatorial '
            'advantage of 512-vs-k but not this conditioning asymmetry. A surviving '
            'ordering means "the decoder interpolates at least as close to the '
            'task-optimal action as nearby data does", not a claim about the behaviour '
            'conditional at s alone.'),
        'ordering_survives': survives, 'verdict': verdict}
    print(f'\nVERDICT: {verdict}')
    write_report(report, cfg, 'diag_geometry_robustness.json')


if __name__ == '__main__':
    main()
