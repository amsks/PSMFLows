"""Evaluate one psmgoal checkpoint on cube under a NAMED reward on states (2026-10-15).

The task is a reward r(s) on observations, not a goal state. Three readouts of the same
checkpoint, all with acting=gpi (argmax over prior draws):

  reward_gpi   coef_source=reward_gpi: r over the relabel batch's next states is handed to
               infer_eval_goals; k_goals candidate goals (agent.reward_gpi_cands), each read
               as the expected r over the reference states at its own h(g_c); the acting
               score is the max over candidates.
  hgoal_topk   the retrieval baseline: coef_source=hgoal_each with a binary reward, 1 for
               the k_goals batch rows with the highest r and 0 elsewhere, so the existing
               branch takes exactly those as the goal set and the rest as reference columns.
  bc           coef_source=hgoal_each, gpi_num_u=1: the argmax over one prior draw is a
               behaviour-cloning action. The binary reward is passed but not used.

The agent config comes from the run's own flags.json with typed agent.* overrides on top
(tools/eval_checkpoint.py merge_run_config), then the readout's keys are forced. The
relabel batch and the candidates are inferred ONCE (np.random.seed(seed), as eval_checkpoint
does) and reused on every task: r does not depend on the task, only the reset distribution
does. Per episode: the mean per-step r over the next states visited, the discounted return
of r (gamma = the agent's discount), r at the final step, the max over the episode, and the
env's own success flag. Mean and standard error over episodes per task and the mean over
tasks are printed and written to JSON (report_out) with the config, the readout, the reward
name, the per-episode arrays and the observation index layout.

Run (GPU):
  .venv/bin/python tools/eval_state_reward.py agent=psmgoal env_name=cube-single-play \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \\
    restore_path=<run dir> restore_epoch=500000 \\
    +reward=cube_height +readout=reward_gpi +tasks="1 2 3 4 5" eval_episodes=50 seed=0 \\
    report_out=$PSM_DATA/logs/state_reward/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported by the agents below)

# isort: split
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

# cube-single-play observation layout, 28 columns. Verified 2026-10-15 against the installed
# ogbench (manipspace/envs/cube_env.py compute_observation, lines 618-646; manipspace_env.py
# compute_ob_info, lines 386-414) by resetting the env and comparing the slices with
# compute_ob_info(): positions are (pos - [0.425, 0, 0]) * 10, so 1 unit = 10 cm.
#   0:6    arm joint positions (6 UR5e joints)
#   6:12   arm joint velocities
#   12:15  end-effector position, scaled
#   15, 16 cos, sin of the end-effector yaw
#   17     gripper opening * 3
#   18     gripper contact
#   19:22  cube position, scaled
#   22:26  cube quaternion (wxyz)
#   26, 27 cos, sin of the cube yaw
OBS_LAYOUT = {
    'joint_pos': [0, 6], 'joint_vel': [6, 12], 'effector_pos': [12, 15], 'effector_yaw_cos_sin': [15, 17],
    'gripper_opening': [17, 18], 'gripper_contact': [18, 19], 'cube_pos': [19, 22], 'cube_quat': [22, 26],
    'cube_yaw_cos_sin': [26, 28],
}
CUBE_POS, EFFECTOR_POS = slice(19, 22), slice(12, 15)

#: name -> fn(obs (N, ob)) -> r (N,). Every reward is a plain function of the observation.
STATE_REWARDS = {
    'cube_height': lambda ob: np.asarray(ob, np.float32)[:, CUBE_POS][:, 2],
    'cube_x': lambda ob: np.asarray(ob, np.float32)[:, CUBE_POS][:, 0],
    'cube_y': lambda ob: np.asarray(ob, np.float32)[:, CUBE_POS][:, 1],
    'effector_height': lambda ob: np.asarray(ob, np.float32)[:, EFFECTOR_POS][:, 2],
    'neg_cube_height': lambda ob: -np.asarray(ob, np.float32)[:, CUBE_POS][:, 2],
}

READOUTS = {
    'reward_gpi': {'coef_source': 'reward_gpi', 'acting': 'gpi'},
    'hgoal_topk': {'coef_source': 'hgoal_each', 'acting': 'gpi'},
    'bc': {'coef_source': 'hgoal_each', 'acting': 'gpi', 'gpi_num_u': 1},
}

ENV_PREFIX_DEFAULT = 'cube-single-play'


def task_env_name(prefix, task):
    """The singletask env id of one task, as scripts/eval500.sh forms it."""
    return f'{prefix}-singletask-task{int(task)}-v0'


def binary_topk_rewards(r, k):
    """1.0 on the k rows with the highest r, 0.0 elsewhere (hgoal_topk and bc readouts)."""
    out = np.zeros_like(r, dtype=np.float32)
    out[np.argsort(-r, kind='stable')[:k]] = 1.0
    return out


def episode_metrics(traj, reward_fn, gamma):
    """Per-episode readouts of r over the visited next states."""
    next_obs = np.asarray(traj['next_observation'], np.float32)
    r = np.asarray(reward_fn(next_obs), np.float64).reshape(-1)
    disc = gamma ** np.arange(r.shape[0])
    last_info = traj['info'][-1] if 'info' in traj and len(traj['info']) else {}
    return {
        'mean': float(r.mean()),
        'discounted_return': float((disc * r).sum()),
        'final': float(r[-1]),
        'max': float(r.max()),
        'success': float(np.max(np.asarray(last_info.get('success', 0.0)))),
        'length': int(r.shape[0]),
    }


def mean_sem(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {'mean': None, 'sem': None, 'n': 0}
    sem = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {'mean': float(x.mean()), 'sem': sem, 'n': int(x.size)}


def summarise(per_episode):
    keys = ('mean', 'discounted_return', 'final', 'max', 'success')
    return {k: mean_sem([e[k] for e in per_episode]) for k in keys}


def parse_tasks(v):
    if isinstance(v, (list, tuple)):
        return [int(t) for t in v]
    return [int(t) for t in str(v).strip('[]()').replace(',', ' ').split()]


def build_dataset(train_dataset, config):
    """The relabel dataset with the preimage arrays spliced in, as tools/eval_checkpoint.py."""
    train_dataset = dict(train_dataset)
    if config.get('preimage_path'):
        from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
        aug, _ = repair_invalid_preimages(load_augmented_dataset(config['preimage_path']))
        assert aug['observations'].shape[0] == train_dataset['observations'].shape[0], \
            'preimage npz row count != eval dataset (wrong env or stale npz)'
        for k in aug:
            if k.startswith('noise_preimage'):
                train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    if config.get('preimage_path'):
        ds.return_preimage_noise = True
        ds.preimage_point_mode = bool(config.get('use_point_preimage', False))
    return ds


def run(cfg):
    t_start = time.time()
    reward_name = str(cfg.get('reward', 'cube_height'))
    readout = str(cfg.get('readout', 'reward_gpi'))
    assert reward_name in STATE_REWARDS, f'reward must be one of {sorted(STATE_REWARDS)}'
    assert readout in READOUTS, f'readout must be one of {sorted(READOUTS)}'
    reward_fn = STATE_REWARDS[reward_name]
    tasks = parse_tasks(cfg.get('tasks', '1 2 3 4 5'))
    prefix = str(cfg.env_name) if cfg.env_name else ENV_PREFIX_DEFAULT
    assert '-singletask' not in prefix, 'env_name is the dataset prefix (cube-single-play); tasks come from +tasks'
    assert cfg.restore_path is not None, 'needs a trained checkpoint (restore_path)'

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    assert merged.get('agent_name') == 'psmgoal', 'eval_state_reward evaluates psmgoal checkpoints'
    forced = dict(READOUTS[readout])
    for k, v in forced.items():
        merged[k] = v
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    k_goals = int(config['k_goals'])
    gamma = float(config['discount'])
    base_seed = int(cfg.seed)
    n_ep = int(cfg.eval_episodes)
    print(f'=== eval_state_reward: reward={reward_name} readout={readout} tasks={tasks} episodes/task={n_ep}')
    print(f'    restore={cfg.restore_path} epoch={cfg.restore_epoch} seed={base_seed}')
    print(f'    forced agent keys: {forced}; coef_source={config["coef_source"]} acting={config["acting"]} '
          f'gpi_num_u={int(config["gpi_num_u"])} k_goals={k_goals} measure_loss={config.get("measure_loss")} '
          f'reward_gpi_cands={config.get("reward_gpi_cands")}', flush=True)

    # Dataset, agent and the task-independent inference, once, from the first task's env.
    np.random.seed(base_seed)
    _, eval_env, train_dataset, _ = make_env_and_datasets(task_env_name(prefix, tasks[0]), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, config)
    ex = ds.sample(1)
    agent = agents['psmgoal'].create(base_seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, str(cfg.restore_path), cfg.restore_epoch)

    n_relabel = min(ds.size, int(cfg.get('eval_relabel_size', 10000)))
    zb = ds.sample(n_relabel)
    r_batch = np.asarray(reward_fn(zb['next_observations']), np.float32).reshape(-1)
    if readout == 'reward_gpi':
        rewards_in = r_batch
    else:
        rewards_in = binary_topk_rewards(r_batch, k_goals)
    t_infer = time.time()
    agent = agent.infer_eval_goals(zb, rewards_in)
    infer_seconds = round(time.time() - t_infer, 1)
    goal_reward = np.asarray(reward_fn(np.asarray(agent.eval_goals)), np.float64)
    ref_reward = np.asarray(reward_fn(np.asarray(agent.eval_ref)), np.float64)
    print(f'    relabel batch r: mean {r_batch.mean():.4f} std {r_batch.std():.4f} min {r_batch.min():.4f} '
          f'max {r_batch.max():.4f}; top-{k_goals} mean {np.sort(r_batch)[-k_goals:].mean():.4f}')
    print(f'    goal set r: mean {goal_reward.mean():.4f} min {goal_reward.min():.4f} max {goal_reward.max():.4f}; '
          f'reference r: mean {ref_reward.mean():.4f} (R={ref_reward.shape[0]}); infer {infer_seconds}s', flush=True)

    per_task = {}
    episodes = {}
    for i, t in enumerate(tasks):
        env_name = task_env_name(prefix, t)
        if i > 0:
            _, eval_env, _, _ = make_env_and_datasets(env_name, frame_stack=cfg.frame_stack)
        t0 = time.time()
        _, trajs, _ = evaluate(agent=agent, env=eval_env, config=config, num_eval_episodes=n_ep,
                               num_video_episodes=0, seed=base_seed)
        eps = [episode_metrics(tr, reward_fn, gamma) for tr in trajs]
        episodes[str(t)] = eps
        per_task[str(t)] = {'env_name': env_name, 'seconds': round(time.time() - t0, 1), **summarise(eps)}
        s = per_task[str(t)]
        print(f'task {t}: r_mean {s["mean"]["mean"]:.4f} +- {s["mean"]["sem"]:.4f}  '
              f'disc {s["discounted_return"]["mean"]:.3f} +- {s["discounted_return"]["sem"]:.3f}  '
              f'final {s["final"]["mean"]:.4f}  max {s["max"]["mean"]:.4f}  '
              f'success {s["success"]["mean"]:.3f}  ({s["seconds"]}s)', flush=True)

    all_eps = [e for t in tasks for e in episodes[str(t)]]
    keys = ('mean', 'discounted_return', 'final', 'max', 'success')
    five_task = {k: {'mean': float(np.mean([per_task[str(t)][k]['mean'] for t in tasks])),
                     'sem_over_tasks': (float(np.std([per_task[str(t)][k]['mean'] for t in tasks], ddof=1)
                                              / np.sqrt(len(tasks))) if len(tasks) > 1 else 0.0),
                     'pooled': mean_sem([e[k] for e in all_eps])} for k in keys}
    print(f'{readout} {reward_name} | tasks {tasks} | r_mean {five_task["mean"]["mean"]:.4f} '
          f'disc {five_task["discounted_return"]["mean"]:.3f} final {five_task["final"]["mean"]:.4f} '
          f'max {five_task["max"]["mean"]:.4f} success {five_task["success"]["mean"]:.3f} '
          f'(pooled sem r_mean {five_task["mean"]["pooled"]["sem"]:.4f})', flush=True)

    report = {
        'tool': 'eval_state_reward',
        'reward': reward_name,
        'readout': readout,
        'readout_forced_keys': forced,
        'env_prefix': prefix,
        'tasks': tasks,
        'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch),
        'seed': base_seed,
        'train_seed': prov.get('train_seed'),
        'episodes_per_task': n_ep,
        'relabel_size': int(n_relabel),
        'gamma': gamma,
        'k_goals': k_goals,
        'gpi_num_u': int(config['gpi_num_u']),
        'infer_seconds': infer_seconds,
        'relabel_reward': {'mean': float(r_batch.mean()), 'std': float(r_batch.std()),
                           'min': float(r_batch.min()), 'max': float(r_batch.max()),
                           'topk_mean': float(np.sort(r_batch)[-k_goals:].mean())},
        'goal_set_reward': goal_reward.tolist(),
        'reference_reward_mean': float(ref_reward.mean()),
        'reference_rows': int(ref_reward.shape[0]),
        'per_task': per_task,
        'five_task': five_task,
        'per_episode': episodes,
        'obs_layout': OBS_LAYOUT,
        'agent_config': json.loads(json.dumps(merged, default=str)),
        'config_provenance': prov,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, f'eval_state_reward_{readout}_{reward_name}.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
