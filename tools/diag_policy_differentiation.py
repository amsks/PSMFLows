"""Do different task vectors actually produce different policies?

The loop hypothesis says FB wins because its actor optimizes F^T z from step one, so its
z-conditioned policies genuinely differentiate and the measure is trained on that
differentiation. Ours may be a single behaviour-cloned policy wearing different task
vectors -- which is what D3 implies (actor gradient 84% behaviour cloning) but has never
been observed directly.

The measurement isolates the task vector from every other source of variation:

  d_task   trajectories from the SAME start state under DIFFERENT task vectors
  d_noise  trajectories from the SAME start state under the SAME task vector, differing
           only in the actor's own sampling noise

  ratio = d_task / d_noise

Ratio ~ 1 means the task vector moves the policy no more than resampling its noise does,
i.e. the policies are the same policy. Ratio >> 1 means the task vector is a real control
input. This is a ratio of like to like -- both terms are distances between trajectories
from an identical start -- so it is comparable across agents with different action scales.

Task vectors are drawn the way training draws them (the goal component of the mixed-z
sampler): z_k = project(B(goal_k)) for FB, w_k = project(phi(goal_k)) for psmflow, with
goals sampled from the dataset. That keeps every task vector inside the training
distribution, so a low ratio cannot be blamed on querying the critic out of support.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_policy_differentiation.py agent=fb agent.actor.type=flow \
    agent.ortho_coef=1000 env_name=cube-single-play-singletask-v0 \
    restore_path=<fb_run> restore_epoch=500000 \
    report_out=/data-local/amsks/PSMFLows/logs/diff_fb_cube.json
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
from agents.psm import project_z
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_TASKS = 8         # task vectors
N_STARTS = 6        # start states, shared across every task vector
N_NOISE = 2         # action-noise replicates per (start, task) -- the floor
MAX_STEPS = 400     # cap: this probe is about where the policy goes, not whether it wins
T_SAMPLES = 40      # timesteps a trajectory is summarised at
SEED = 0


def set_task(agent, name, vec):
    """Install a task vector on either agent type."""
    return agent.replace(z_eval=vec) if name == 'fb' else agent.replace(task_z=vec)


def task_vectors(agent, name, goals, norm_z):
    """project(B(goal)) for FB, project(phi(goal)) for psmflow -- the training-time form."""
    if name == 'fb':
        feats = agent._apply('backward', agent.params['backward'], goals)
    else:
        feats = agent.phi(goals)
    return np.asarray(project_z(feats, norm_z))


def summarise(traj):
    """Fixed-length summary of a trajectory so distances are comparable across lengths."""
    idx = np.linspace(0, len(traj) - 1, T_SAMPLES).astype(int)
    return traj[idx]


def dist(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    name = config['agent_name']
    assert name in ('fb', 'psmflow'), 'this probe compares fb against psmflow'

    ex = ds.sample(1)
    agent = agents[name].create(cfg.seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    rng_np = np.random.default_rng(SEED)
    goals = jnp.asarray(np.asarray(ds['next_observations'])[
        rng_np.integers(0, ds.size, N_TASKS)])
    vecs = task_vectors(agent, name, goals, config['norm_z'])

    # trajectories[start][task][noise] -> (T_SAMPLES, ob_dim)
    trajectories = [[[None] * N_NOISE for _ in range(N_TASKS)] for _ in range(N_STARTS)]
    successes = np.zeros((N_STARTS, N_TASKS, N_NOISE))
    for si in range(N_STARTS):
        for ti in range(N_TASKS):
            a = set_task(agent, name, jnp.asarray(vecs[ti]))
            for ni in range(N_NOISE):
                ob, _ = eval_env.reset(seed=SEED + si)
                key = jax.random.PRNGKey(1000 * ni + 7)
                obs_seq, done, t, succ = [np.asarray(ob, np.float32)], False, 0, 0.0
                while not done and t < MAX_STEPS:
                    key, k = jax.random.split(key)
                    act = np.asarray(a.sample_actions(jnp.asarray(ob, jnp.float32), seed=k))
                    ob, _, term, trunc, info = eval_env.step(np.clip(act, -1, 1))
                    obs_seq.append(np.asarray(ob, np.float32))
                    succ = max(succ, float(info.get('success', 0.0)))
                    done, t = term or trunc, t + 1
                trajectories[si][ti][ni] = summarise(np.stack(obs_seq))
                successes[si, ti, ni] = succ
        print(f'start {si + 1}/{N_STARTS} done', flush=True)

    # d_task: same start, different task vector, same noise replicate.
    d_task = [dist(trajectories[si][i][0], trajectories[si][j][0])
              for si in range(N_STARTS)
              for i in range(N_TASKS) for j in range(i + 1, N_TASKS)]
    # d_noise: same start, same task vector, different noise.
    d_noise = [dist(trajectories[si][ti][0], trajectories[si][ti][1])
               for si in range(N_STARTS) for ti in range(N_TASKS)]

    mt, mn = float(np.mean(d_task)), float(np.mean(d_noise))
    ratio = mt / (mn + 1e-12)

    # Spread of where the episodes END, per start: the coarse occupancy read.
    end_spread = float(np.mean([
        np.mean(np.std(np.stack([trajectories[si][ti][0][-1] for ti in range(N_TASKS)]),
                       axis=0))
        for si in range(N_STARTS)]))

    if ratio >= 2.0:
        verdict = (f'differentiated: changing the task vector moves the trajectory '
                   f'{ratio:.2f}x more than resampling the actor noise -- the task vector '
                   f'is a real control input')
    elif ratio <= 1.2:
        verdict = (f'NOT differentiated: changing the task vector moves the trajectory '
                   f'{ratio:.2f}x the noise floor, i.e. barely at all -- these are one '
                   f'policy wearing different task vectors')
    else:
        verdict = (f'weakly differentiated: ratio {ratio:.2f} over the noise floor')

    report = {
        'env': cfg.env_name, 'agent': name, 'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch), 'seed': SEED,
        'n_tasks': N_TASKS, 'n_starts': N_STARTS, 'n_noise': N_NOISE,
        'max_steps': MAX_STEPS, 't_samples': T_SAMPLES,
        'd_task_mean': mt, 'd_task_std': float(np.std(d_task)),
        'd_noise_mean': mn, 'd_noise_std': float(np.std(d_noise)),
        'differentiation_ratio': ratio,
        'final_state_spread_across_tasks': end_spread,
        'success_rate_over_probe': float(successes.mean()),
        'verdict': verdict,
    }
    print(f'\nd_task {mt:.4f} | d_noise {mn:.4f} | ratio {ratio:.3f}\nVERDICT: {verdict}')
    write_report(report, cfg, 'diag_policy_differentiation.json')


if __name__ == '__main__':
    main()
