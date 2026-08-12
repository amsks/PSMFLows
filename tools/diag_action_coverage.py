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

    # ---- C1 / C3, computed once from the same neighbourhoods ----------------------
    # C1 is the discriminator: if a_FQL sits as far from the dataset's OWN nearest
    # neighbour actions as our decodes do, the task needs actions the behaviour
    # distribution does not contain, and no behaviour flow can decode them.
    c1_dist, c2_resid, c3_ceiling, c3_k8 = [], [], [], []
    for i, s_i in enumerate(states):
        d = np.linalg.norm(pool_obs - s_i, axis=1)
        nn = np.argpartition(d, K_NEIGHBOURS)[:K_NEIGHBOURS]
        nn_acts = pool_act[nn]
        c1_dist.append(float(np.min(np.linalg.norm(nn_acts - a_fql[i], axis=1))))
        # C3: the aleatoric ceiling. Half the neighbourhood's action spread against the
        # other half's -- coverage cannot exceed this, so 1.0 was never the reference.
        h = K_NEIGHBOURS // 2
        c3_ceiling.append(float(np.mean(nn_acts[:h].std(0) / (nn_acts[h:].std(0) + 1e-8))))
        nn8 = np.argpartition(d, 8)[:8]
        c3_k8.append(pool_act[nn8].std(0))
    c1 = {'min_dist_fql_to_knn_actions_mean': float(np.mean(c1_dist)),
          'min_dist_fql_to_knn_actions_median': float(np.median(c1_dist)),
          'normalised_by_action_scale': float(np.mean(c1_dist) / a_scale),
          'k': K_NEIGHBOURS}
    c3 = {'aleatoric_coverage_ceiling_mean': float(np.mean(c3_ceiling)),
          'aleatoric_coverage_ceiling_median': float(np.median(c3_ceiling)),
          'knn_action_sd_k8_mean': float(np.mean(np.stack(c3_k8))),
          'knn_action_sd_k32_mean': float(np.mean(data_sd))}

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
            if decode_mode == 'onestep' and radius == RADII[0]:
                # C2: WHERE the residual lives. Cube's gripper dim is quasi-discrete and a
                # BC flow smooths it; a residual concentrated there calls for a per-dim
                # epsilon rather than a blanket one.
                per_dim = []
                for i in range(N_STATES):
                    k2 = jax.random.PRNGKey(SEED + 500 + i)
                    u2 = jnp.clip(jax.random.normal(k2, (N_LATENTS, agent.config['action_dim'])),
                                  -radius, radius)
                    ob_b = jnp.broadcast_to(states_j[i], (N_LATENTS, states_j.shape[1]))
                    acts2 = np.asarray(a2.decode(ob_b, u2))
                    j = int(np.argmin(np.linalg.norm(acts2 - a_fql[i], axis=1)))
                    per_dim.append(np.abs(acts2[j] - a_fql[i]))
                per_dim = np.stack(per_dim)
                c2_resid.append({
                    'per_dim_abs_residual_mean': per_dim.mean(0).tolist(),
                    'per_dim_share_of_total': (per_dim.mean(0) / per_dim.mean(0).sum()).tolist(),
                    'per_dim_data_sd': data_sd.mean(0).tolist()})
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
    ode_gain = (max(r['coverage_ratio_mean'] for r in results if r['decode'] == 'ode')
                - base['coverage_ratio_mean'])
    rad_gain = (max(r['coverage_ratio_mean'] for r in results
                    if r['decode'] == 'onestep' and r['radius'] > 3.0)
                - base['coverage_ratio_mean'])

    # Scoped to THIS flow, so it can no longer be mistaken for a cross-flow conclusion:
    # the fs100 report reprinted the fs10 recommendation verbatim.
    knob_verdict = (
        f"on this flow, neither knob moves coverage (ODE +{ode_gain:.3f}, radius "
        f"+{rad_gain:.3f}); deployed coverage {base['coverage_ratio_mean']:.3f}, best "
        f"{best['coverage_ratio_mean']:.3f}"
        if max(ode_gain, rad_gain) < 0.1 else
        f"on this flow, ODE +{ode_gain:.3f} / radius +{rad_gain:.3f} over the deployed "
        f"setting {base['coverage_ratio_mean']:.3f}")

    ours = base['min_dist_normalised_by_action_scale']
    theirs = c1['normalised_by_action_scale']
    if theirs >= 0.6 * ours:
        capacity_arm = 'cancelled'
        c1_verdict = (
            f"FQL is OFF-SUPPORT: its action sits {theirs:.3f} of action scale from the "
            f"dataset's own {K_NEIGHBOURS}-NN actions, against {ours:.3f} for our best "
            f"decode. The task needs actions the behaviour distribution does not contain, "
            f"so no behaviour flow can decode them -- capacity/retrain arm CANCELLED, a "
            f"residual budget (W4) is the only fix.")
    else:
        capacity_arm = 'live'
        c1_verdict = (
            f"FQL is IN-SUPPORT: {theirs:.3f} of action scale from the dataset's own "
            f"neighbours, well inside our decode gap {ours:.3f}. The actions exist in the "
            f"data and the flow fails to reach them -- capacity/retrain arm stays LIVE.")

    if c2_resid:
        share = np.array(c2_resid[0]['per_dim_share_of_total'])
        top = int(np.argmax(share))
        c2_verdict = (
            f"residual concentrates in action dim {top} ({100 * share[top]:.0f}% of total) "
            f"-> shape epsilon per-dim"
            if share[top] >= 0.4 else
            f"residual spread across dims (max {100 * share.max():.0f}% in dim {top}) "
            f"-> a blanket epsilon is appropriate")
    else:
        c2_verdict = 'not computed'

    ceil = c3['aleatoric_coverage_ceiling_mean']
    c3_verdict = (
        f"coverage ceiling from the data's own {K_NEIGHBOURS}-NN split is {ceil:.3f}, not "
        f"1.0; deployed coverage {base['coverage_ratio_mean']:.3f} is "
        f"{100 * base['coverage_ratio_mean'] / (ceil + 1e-8):.0f}% of achievable")

    verdict = f"C1: {c1_verdict} | C2: {c2_verdict} | C3: {c3_verdict}"

    report = {'env': cfg.env_name, 'flow_ckpt_path': str(config['flow_ckpt_path']),
              'ode_steps_from_flags': ode_steps, 'n_states': N_STATES,
              'n_latents': N_LATENTS, 'k_neighbours': K_NEIGHBOURS, 'seed': SEED,
              'fql_reference': {'run': FQL_RUN, 'epoch': FQL_EPOCH, 'alpha': FQL_ALPHA},
              'action_scale_mean_abs': a_scale,
              'deployed_setting': {'decode': 'onestep', 'radius': 3.0, **base},
              'sweep': results, 'ode_coverage_gain': ode_gain,
              'radius_coverage_gain': rad_gain,
              'knob_sweep_verdict': knob_verdict,
              'c1_fql_offsupport': {**c1, 'our_best_decode_normalised': ours,
                                    'capacity_arm': capacity_arm, 'verdict': c1_verdict},
              'c2_residual_by_dim': (c2_resid[0] if c2_resid else None),
              'c2_verdict': c2_verdict,
              'c3_aleatoric_ceiling': {**c3, 'verdict': c3_verdict},
              'verdict': verdict}
    print(f'\n{knob_verdict}\n\nC1: {c1_verdict}\n\nC2: {c2_verdict}\n\nC3: {c3_verdict}')
    write_report(report, cfg, 'diag_action_coverage.json')


if __name__ == '__main__':
    main()
