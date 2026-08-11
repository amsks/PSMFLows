"""W1 -- what actions can the frozen flow actually reach, and does the winning policy's
action live in that set?

Every method that acts through the frozen flow caps near 0.22 on cube; every method in raw
action space clears 0.7 (P2 settled this: a per-task critic with the true reward caps
there too). So the binding constraint is the latent->action interface. This probe measures
the reachable action set directly, and sweeps the two knobs that could widen it.

Per state:
  1. Reachable spread -- decode N latents, per-dim action sd, against the behaviour
     conditional's sd estimated from the k nearest dataset states. Ratio < 1 means the
     decoder cannot reproduce the data's own action variability.
  2. The killer stat -- min over decoded candidates of ||G(s,u) - a_FQL(s)||, where a_FQL
     is what a trained FQL policy (0.949 on this task) does at the same state. If no
     latent decodes near the working policy's action, no critic can select it, and no
     amount of representation work helps.

Swept over decoder (`onestep` as deployed vs the ODE) and latent radius. The ODE step
count is READ FROM THE FLOW'S flags.json, not taken from the config: the two must match
the value the flow was trained at, and trusting the config default is the known
`gpi_decode=ode` step-mismatch defect.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_action_coverage.py agent=psmflow \
    env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    report_out=/data-local/amsks/PSMFLows/logs/diag_action_coverage_cube.json
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
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_STATES = 64
N_LATENTS = 512
K_NEIGHBOURS = 32       # same protocol as the 07-29 conditional-spread measurement
RADII = [3.0, 4.5, 6.0]
SEED = 0
FQL_RUN = '/data-local/amsks/PSMFLows/exp/PSMFLows/fqlbaseline_cube_a300_20260810/sd000_20260810_042228'
FQL_EPOCH = 500000
FQL_ALPHA = 300


def flow_trained_steps(flow_ckpt_path):
    """The step count the flow was TRAINED at. The ODE decode must use exactly this."""
    p = os.path.join(str(flow_ckpt_path), 'flags.json')
    assert os.path.exists(p), f'no flags.json beside the flow checkpoint: {p}'
    with open(p) as f:
        flags = json.load(f)
    steps = flags['agent'].get('flow_steps')
    assert steps, f'{p} does not record agent.flow_steps'
    return int(steps)


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
    print(f'flow was trained at flow_steps={ode_steps}; ODE decode pinned to that')

    # The reference policy: what a method that actually works does at these states.
    fql_cfg = ml_collections.ConfigDict(_lists_to_tuples(
        OmegaConf.to_container(
            OmegaConf.load(os.path.join('configs', 'agent', 'fql.yaml')), resolve=True)))
    fql_cfg['alpha'] = FQL_ALPHA
    fql = agents['fql'].create(cfg.seed, ex['observations'], ex['actions'], fql_cfg)
    fql = restore_agent(fql, FQL_RUN, FQL_EPOCH)

    rng_np = np.random.default_rng(SEED)
    obs_all = np.asarray(ds['observations'])
    act_all = np.asarray(ds['actions'])
    idx = rng_np.integers(0, ds.size, N_STATES)
    states = obs_all[idx]

    # Behaviour conditional at each state, estimated from its k nearest dataset states.
    # Subsample the pool: exact k-NN over 1M rows is not needed for an sd estimate.
    pool_idx = rng_np.integers(0, ds.size, 200000)
    pool_obs, pool_act = obs_all[pool_idx], act_all[pool_idx]
    data_sd = []
    for s in states:
        d = np.linalg.norm(pool_obs - s, axis=1)
        nn = np.argpartition(d, K_NEIGHBOURS)[:K_NEIGHBOURS]
        data_sd.append(pool_act[nn].std(0))
    data_sd = np.stack(data_sd)                      # (N_STATES, d_a)

    rng = jax.random.PRNGKey(SEED)
    states_j = jnp.asarray(states, jnp.float32)
    a_fql = np.asarray(jax.vmap(lambda s, k: fql.sample_actions(s, seed=k))(
        states_j, jax.random.split(jax.random.PRNGKey(SEED + 1), N_STATES)))
    a_scale = float(np.abs(act_all).mean())

    results = []
    for decode_mode in ('onestep', 'ode'):
        for radius in RADII:
            a2 = agent.replace(config=agent.config.copy(
                {'gpi_decode': decode_mode, 'flow_decode_steps': ode_steps}))
            cover, mindist = [], []
            for i in range(N_STATES):
                rng, k = jax.random.split(rng)
                u = jnp.clip(jax.random.normal(k, (N_LATENTS, agent.config['action_dim'])),
                             -radius, radius)
                obs_b = jnp.broadcast_to(states_j[i], (N_LATENTS, states_j.shape[1]))
                acts = np.asarray(a2.decode(obs_b, u))
                cover.append(acts.std(0) / (data_sd[i] + 1e-8))
                mindist.append(float(np.min(np.linalg.norm(acts - a_fql[i], axis=1))))
            cover = np.stack(cover)
            results.append({
                'decode': decode_mode, 'radius': radius, 'ode_steps': ode_steps,
                'coverage_ratio_mean': float(cover.mean()),
                'coverage_ratio_median': float(np.median(cover)),
                'min_dist_to_fql_action_mean': float(np.mean(mindist)),
                'min_dist_to_fql_action_median': float(np.median(mindist)),
                'min_dist_normalised_by_action_scale': float(np.mean(mindist) / a_scale),
            })
            print(f"{decode_mode:8s} r={radius:4.1f}  coverage {cover.mean():.3f}  "
                  f"min||G(s,u)-a_FQL|| {np.mean(mindist):.4f} "
                  f"({np.mean(mindist) / a_scale:.3f} of action scale)", flush=True)

    base = next(r for r in results if r['decode'] == 'onestep' and r['radius'] == 3.0)
    best = max(results, key=lambda r: r['coverage_ratio_mean'])
    best_fql = min(results, key=lambda r: r['min_dist_to_fql_action_mean'])
    ode_gain = (max(r['coverage_ratio_mean'] for r in results if r['decode'] == 'ode')
                - base['coverage_ratio_mean'])
    rad_gain = (max(r['coverage_ratio_mean'] for r in results
                    if r['decode'] == 'onestep' and r['radius'] > 3.0)
                - base['coverage_ratio_mean'])

    if best['coverage_ratio_mean'] < 0.55 and best_fql['min_dist_normalised_by_action_scale'] > 0.5:
        verdict = (f"no knob recovers coverage (best {best['coverage_ratio_mean']:.3f}, "
                   f"best FQL-action distance {best_fql['min_dist_normalised_by_action_scale']:.3f} "
                   f"of action scale) -- the flow itself collapsed the conditional; skip W2, go to W3")
    elif ode_gain >= 0.1 and rad_gain >= 0.1:
        verdict = (f'both knobs contribute (ODE +{ode_gain:.3f}, radius +{rad_gain:.3f}) '
                   f'-> run W2 arms A, B and A+B')
    elif ode_gain >= 0.1:
        verdict = f'ODE decode recovers coverage (+{ode_gain:.3f}) -> W2 arm A only'
    elif rad_gain >= 0.1:
        verdict = f'latent radius recovers coverage (+{rad_gain:.3f}) -> W2 arm B only'
    else:
        verdict = (f'neither knob moves coverage (ODE +{ode_gain:.3f}, radius +{rad_gain:.3f}) '
                   f'-> W3 (retrain the flow)')

    report = {'env': cfg.env_name, 'flow_ckpt_path': str(config['flow_ckpt_path']),
              'ode_steps_from_flags': ode_steps, 'n_states': N_STATES,
              'n_latents': N_LATENTS, 'k_neighbours': K_NEIGHBOURS, 'seed': SEED,
              'fql_reference': {'run': FQL_RUN, 'epoch': FQL_EPOCH, 'alpha': FQL_ALPHA},
              'action_scale_mean_abs': a_scale,
              'deployed_setting': {'decode': 'onestep', 'radius': 3.0, **base},
              'sweep': results, 'ode_coverage_gain': ode_gain,
              'radius_coverage_gain': rad_gain, 'verdict': verdict}
    print(f'\nVERDICT: {verdict}')
    write_report(report, cfg, 'diag_action_coverage.json')


if __name__ == '__main__':
    main()
