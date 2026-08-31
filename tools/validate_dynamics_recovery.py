"""Does a latent-decoded action move the simulator where the recorded action did?

The decode tests score actions against actions. This one scores them against the
environment: for a sampled transition it puts MuJoCo back at the recorded state, applies
the action decoded from that row's preimage, and measures where the next state lands.

  restore  set_state(qpos[i], qvel[i]) from the OGBench dataset (row-aligned with
           `observations`; kept only when the dataset is loaded with add_info=True)
  step     one env.step per latent source from the SAME restored state, which is
           exactly reproducible -- restore+step twice gives bit-identical results

The headline is `next_obs_vs_replay`: the generated action's next state against the
RECORDED action's next state, both stepped from the same restored state in the same
process. Anything that makes the local simulator differ from the one that generated the
dataset (MuJoCo version, contact solver) cancels there. `next_obs_vs_dataset` compares
against the recorded next observation instead and is only as tight as `replay_vs_dataset`,
the floor reported beside it -- measured 1.4e-6 on pointmaze but 0.022 on cube, against a
mean step size of 0.47.

Two scales come with every source, because a raw state deviation means nothing on its
own: `data_step` (how far one recorded step moves the state at all) and `random` (a
uniform action from the same state).

Needs the env, unlike tools/validate_decode_recovery.py:

  MUJOCO_GL=egl .venv/bin/python tools/validate_dynamics_recovery.py agent=fql \
      env_name=cube-single-play-singletask-v0 \
      restore_path=$PSM_DATA/flow/cube-single-play restore_epoch=500000 \
      agent.flow_steps=100 +preimage_npz=$PSM_DATA/preimages/cube-single-play.npz \
      +n_states=512

`+latent_sources=[mixture,point,prior]` picks where the latents come from (default
`mixture`), and
`+u_clip=3.0` clamps them the way psmflow does before it decodes them. Keys absent from
configs/config.yaml need a leading `+`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import numpy as np
from tqdm import trange

from envs.env_utils import make_env_and_datasets
from utils.log_utils import write_report
from utils.preimage_eval import (LATENT_SOURCES, decode_in_batches, latents_from,
                                 load_flow_and_preimages, stats)


def _observe(env):
    """The env's observation at the current simulator state, without stepping."""
    unwrapped = env.unwrapped
    for name in ('compute_observation', 'get_ob'):
        fn = getattr(unwrapped, name, None)
        if fn is not None:
            return np.asarray(fn(), np.float32)
    raise AttributeError(f'{type(unwrapped).__name__} exposes neither compute_observation nor get_ob')


def _restore(env, sim_state, row):
    """Put the simulator at the recorded state of `row`.

    The reset is what keeps the episode clock away from the time limit; `set_state`
    overwrites the state the reset randomized. Puzzle/scene envs carry button latches
    outside qpos/qvel, hence the third argument when the dataset has it.
    """
    env.reset()
    if 'button_states' in sim_state:
        env.unwrapped.set_state(sim_state['qpos'][row], sim_state['qvel'][row],
                                sim_state['button_states'][row])
    else:
        env.unwrapped.set_state(sim_state['qpos'][row], sim_state['qvel'][row])


def _step_from(env, sim_state, row, action):
    _restore(env, sim_state, row)
    ob, _, _, _, _ = env.step(np.asarray(action, np.float32))
    return np.asarray(ob, np.float32)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    assert 'singletask' in cfg.env_name, (
        'simulator-state restore needs OGBench qpos/qvel (add_info); '
        f'{cfg.env_name!r} is not an OGBench singletask dataset')
    assert cfg.frame_stack is None, 'frame stacking would make a single restored step ambiguous'
    sources = [str(a) for a in cfg.get('latent_sources', None) or ['mixture']]
    assert set(sources) <= set(LATENT_SOURCES), (
        f'+latent_sources must be a subset of {LATENT_SOURCES}')

    agent, data, valid_rows, skills, meta = load_flow_and_preimages(cfg)

    # The npz carries the transitions but not the simulator state, so the dataset is
    # reloaded with add_info=True. OGBench masks qpos/qvel with the same trajectory-end
    # mask as observations, so the rows line up -- asserted below rather than assumed.
    env, _, ds, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack, add_info=True)
    assert 'qpos' in ds, f'{cfg.env_name} exposes no qpos; cannot restore the simulator'
    # A short npz is the `preimage_limit` prefix or a `preimage_sample` batch; longer than
    # the dataset means the two are not the same env at all. Row alignment itself is
    # checked against the sampled observations below.
    assert len(data['observations']) <= len(ds['observations']), (
        'the npz has more rows than the env dataset -- wrong env or a stale npz')
    # A sampled batch (tools/precompute_preimages.py `+preimage_sample`) carries the buffer
    # row each of its rows came from; without that mapping npz row i is not dataset row i
    # and the restore would put the simulator at an unrelated transition.
    to_buffer = (np.asarray(data['source_index']) if 'source_index' in data
                 else np.arange(len(data['observations'])))

    rng = np.random.default_rng(int(cfg.seed))
    n = min(int(cfg.get('n_states', 256)), len(valid_rows))
    rows = np.sort(rng.choice(valid_rows, size=n, replace=False))
    buffer_rows = to_buffer[rows]
    assert np.allclose(ds['observations'][buffer_rows], data['observations'][rows], atol=1e-5), (
        'npz rows do not match the env dataset rows -- the qpos/qvel restore would be '
        'applied to the wrong transition')

    sim_state = {k: np.asarray(ds[k]) for k in ('qpos', 'qvel', 'button_states') if k in ds}
    obs = data['observations'][rows]
    recorded = np.clip(data['actions'][rows], -1, 1)
    next_obs = np.asarray(ds['next_observations'][buffer_rows], np.float32)
    sk = None if skills is None else skills[rows]
    batch_size = int(cfg.inversion.get('batch_size', 256))

    # Every source's actions up front: one batched flow decode instead of n small ones.
    u_clip = cfg.get('u_clip', None)
    actions = {src: decode_in_batches(agent, obs, latents_from(src, data, rows, rng, u_clip=u_clip),
                                      batch_size, skills=sk)
               for src in sources}
    actions['random'] = rng.uniform(-1, 1, recorded.shape).astype(np.float32)

    restore_err = np.zeros((n,), np.float32)
    restore_diff = np.zeros_like(obs)
    replayed = np.zeros_like(next_obs)
    stepped = {src: np.zeros_like(next_obs) for src in actions}
    for j in trange(n, desc='Stepping the simulator'):
        row = int(buffer_rows[j])
        _restore(env, sim_state, row)
        restore_diff[j] = _observe(env) - obs[j]
        restore_err[j] = np.linalg.norm(restore_diff[j])
        replayed[j] = _step_from(env, sim_state, row, recorded[j])
        for src, act in actions.items():
            # The flow decode diverges on a few (state, latent) pairs (see
            # tools/validate_decode_recovery.py). MuJoCo must not be stepped with that: the
            # row is recorded as a failed decode and left out of that source's statistics.
            stepped[src][j] = (_step_from(env, sim_state, row, act[j])
                               if np.isfinite(act[j]).all() else np.nan)

    data_step = np.linalg.norm(next_obs - obs, axis=-1)
    replay_vs_dataset = np.linalg.norm(replayed - next_obs, axis=-1)
    report = {
        'env': cfg.env_name,
        'seed': int(cfg.seed),
        'npz': os.path.abspath(str(cfg.preimage_npz)),
        'flow': 'TRAINED' if cfg.restore_path is not None else 'RANDOM (control)',
        'flow_steps': int(cfg.agent.flow_steps),
        'n_states': int(n),
        'u_clip': None if u_clip is None else float(u_clip),
        'n_invalid_excluded': int(len(data['actions']) - len(valid_rows)),
        'meta': meta,
        # Is the restored state the recorded state? Zero on pointmaze; on cube the local
        # MuJoCo differs from the one that wrote the dataset, so this is nonzero even
        # though qpos/qvel are set exactly -- the per-dim vector says in which coordinates.
        'restore_obs_err': stats(restore_err),
        'restore_err_per_dim_mean': [round(float(v), 6) for v in np.abs(restore_diff).mean(0)],
        # The floor on every `next_obs_vs_dataset` below.
        'replay_vs_dataset': stats(replay_vs_dataset),
        # The scale everything is read against: one recorded step's own displacement.
        'data_step': stats(data_step),
    }

    mean_step = float(data_step.mean())
    for src, act in actions.items():
        ok = np.isfinite(act).all(-1) & np.isfinite(stepped[src]).all(-1)
        vs_replay = np.linalg.norm(stepped[src][ok] - replayed[ok], axis=-1)
        action_err = np.linalg.norm(act[ok] - recorded[ok], axis=-1)
        report[src] = {'nan_decode_frac': round(float(1.0 - ok.mean()), 6)}
        if not ok.any():
            report[src]['note'] = 'every decode was non-finite; no statistics computed'
            continue
        report[src].update({
            'action_l2_to_recorded': stats(action_err),
            'next_obs_vs_replay': stats(vs_replay),
            'next_obs_vs_dataset': stats(np.linalg.norm(stepped[src][ok] - next_obs[ok], axis=-1)),
            'next_obs_vs_replay_over_data_step': round(float(vs_replay.mean() / mean_step), 4),
            'corr_action_err_vs_state_dev': (
                round(float(np.corrcoef(action_err, vs_replay)[0, 1]), 4)
                if action_err.std() > 0 and vs_replay.std() > 0 else None),
        })

    print(json.dumps(report, indent=2))
    write_report(report, cfg, 'dynamics_recovery.json')


if __name__ == '__main__':
    main()
