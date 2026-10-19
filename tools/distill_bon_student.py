"""Distil the best-of-64 teacher into a one-sample Gaussian student (2026-10-17, kisski's
mailbox/kisski/2026-10-17T01-21-14Z.md, ask 2).

Teacher: a psmgoal checkpoint read out as eval builds it (coef_source=hgoal_each, acting=gpi). Per
task, exactly as tools/eval_checkpoint.py: np.random.seed(seed), the task's relabel batch of
eval_relabel_size rows, rewards + eval_reward_shift, `infer_eval_goals` (32 rewarding next states G_c
with h(g), batch_size - k_goals non-rewarding next states as refs). At a state s the teacher draws
gpi_num_u RAW latents v_k ~ N(0, I), scores clip(v_k) with `_hgoal_each_score` and keeps the winning
RAW v* (not its clip, so the student fits a continuous density without the clip atoms).

Task descriptor c: [mean over G_c of h(g), mean over G_c of g], fixed per task.

Student: q(v | s, c) = N(mu(s, c), L L^T), L lower-triangular with a softplus diagonal, an MLP on the
standardised [s, c]. Loss: -mean log q(v* | s, c).

Training states, per task: +n_data uniform dataset observations (one teacher label each) and every
state of +n_rollout_eps teacher rollouts (the label is the teacher's own pick at that step, plus
+n_extra_labels fresh draws of the 64-set). Rollout episodes are split by episode into train and
validation (+val_frac); the validation NLL is reported beside the NLL of the prior N(0, I).

Students: `in_task` (unless +in_task=false) trains on +train_tasks and is evaluated on +train_tasks;
`holdout` (if +holdout_task is set) trains on +train_tasks minus that task and is evaluated on it;
+loo=true adds one such student per training task (leave-one-out). Each student acts with a sample
a = G(s, clip(mu + L xi)) and with the mode a = G(s, clip(mu)); eval is utils.evaluation
`evaluate` with seed=+eval_seed, eval_episodes episodes per task, as eval500. Pass eval_episodes=500:
the repo config sets 10, which overrides this tool's 500 (the 10-17 run evaluated 10 episodes).

Options added 2026-10-19 (kisski 2026-10-19T00-42-34Z): +weight_decay (AdamW) and +early_stop (keep the
parameters at the lowest validation NLL); +descriptor=set replaces the mean descriptor by a
permutation-invariant encoder over the 32 goal states [g, h(g)] (per-goal MLP, then the mean);
+labels_in=a.npz:b.npz reuses previous runs' labels (for the tasks they hold; first file wins) instead of
re-running the teacher. +student_seed (default seed) changes only the student's init and batch order.
+exclude_near_real=d drops the random regions whose anchor is within d metres of a real task's goal centroid;
+exclude_sparse drops the random regions whose radius had to grow past the tolerance.
+n_random_tasks=N (with +descriptor=set) adds N random goal-region training tasks (kisski 2026-10-19T01-54-05Z):
anchor = the cube position of a random dataset next state (seed +random_seed0 + i); the relabel batch's
rewarding rows are the next states with the cube within the env's 0.04 m tolerance of it (the radius grows to
the k_goals-th nearest row if fewer qualify, and is recorded); goals and refs then come from
`infer_eval_goals` as for a real task. Teacher rollouts on a random task end when the cube reaches the region
(its success). Students: `random_only` trains on the random tasks and is evaluated on the real ones; with
+loo the leave-one-out students also train on the random tasks. +student_out=<dir>
saves each student under <dir>/<student name>/ (params, standardisation, config, NLL curve, eval JSONs).

Run (GPU):
  .venv/bin/python tools/distill_bon_student.py agent=psmgoal env_name=cube-single-play-singletask-v0 \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \\
    restore_path=<teacher run dir> restore_epoch=500000 seed=0 \\
    +train_tasks=1:2:3:4:5 +holdout_task=5 eval_episodes=500 report_out=$PSM_DATA/logs/distill/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside the functions)

FORCED_KEYS = {'acting': 'gpi', 'coef_source': 'hgoal_each'}
# Random goal-region tasks (+n_random_tasks): ids RANDOM_ID0 + i. Cube observation layout and success
# tolerance from ogbench manipspace/envs/cube_env.py: the cube position is (block_pos - [0.425, 0, 0]) * 10
# at observation dims 19:22, and a cube counts as at its target within 0.04 m, i.e. 0.4 in observation units.
# `check_cube_layout` asserts the layout on the live env before any random task is built.
RANDOM_ID0 = 100
CUBE_POS = slice(19, 22)
CUBE_XYZ_CENTER = np.array([0.425, 0.0, 0.0])
CUBE_XYZ_SCALE = 10.0
CUBE_TOL_M = 0.04
LOG_2PI = float(np.log(2.0 * np.pi))


# ------------------------------------------------------------------ pure helpers (CPU-tested)
def parse_tasks(s):
    return [int(t) for t in str(s).replace(',', ' ').replace(':', ' ').split()]


def task_env_name(env_name, task):
    """cube-single-play-singletask-v0 or -task3-v0 -> cube-single-play-singletask-task{task}-v0."""
    base = env_name.split('-singletask')[0]
    return f'{base}-singletask-task{task}-v0'


def tril_from_raw(raw, d):
    """raw (..., d*(d+1)/2) -> lower-triangular L (..., d, d), softplus diagonal + 1e-3."""
    import jax
    import jax.numpy as jnp
    rows, cols = np.tril_indices(d)
    L = jnp.zeros(raw.shape[:-1] + (d, d), raw.dtype).at[..., rows, cols].set(raw)
    diag = jax.nn.softplus(jnp.diagonal(L, axis1=-2, axis2=-1)) + 1e-3
    eye = jnp.eye(d, dtype=bool)
    return jnp.where(eye, diag[..., None, :] * jnp.eye(d, dtype=raw.dtype), L)


def gaussian_nll(v, mu, L):
    """-log N(v; mu, L L^T) per row. v, mu (N, d); L (N, d, d) lower-triangular."""
    import jax
    import jax.numpy as jnp
    d = v.shape[-1]
    z = jax.scipy.linalg.solve_triangular(L, (v - mu)[..., None], lower=True)[..., 0]
    logdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    return 0.5 * jnp.sum(z ** 2, axis=-1) + logdet + 0.5 * d * LOG_2PI


def region_rewards(next_obs, anchor, k, tol=CUBE_TOL_M * CUBE_XYZ_SCALE):
    """Rewards {0, 1} for a random goal-region task: next states whose cube position is within tol of
    anchor (observation units). If fewer than k rows qualify, the radius grows to the k-th nearest row's
    distance. Returns (rewards (N,), radius used, rows within tol)."""
    dist = np.linalg.norm(np.asarray(next_obs)[:, CUBE_POS] - np.asarray(anchor)[None], axis=1)
    n_in = int((dist <= tol).sum())
    radius = float(tol) if n_in >= k else float(np.sort(dist)[k - 1])
    return (dist <= radius).astype(np.float32), radius, n_in


def check_cube_layout(env):
    """Assert that observation dims CUBE_POS are the scaled cube position (ogbench cube_env)."""
    ob, _ = env.reset(seed=0)
    pos = np.asarray(env.unwrapped.compute_ob_info()['privileged/block_0_pos'], np.float64)
    want = (pos - CUBE_XYZ_CENTER) * CUBE_XYZ_SCALE
    assert np.allclose(np.asarray(ob, np.float64)[CUBE_POS], want, atol=1e-4), (ob[CUBE_POS], want)


def split_episodes(ep_ids, val_frac, seed):
    """Boolean train mask over rows, holding out a val_frac share of whole episodes."""
    ids = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    n_val = int(round(val_frac * len(ids)))
    val = set(rng.choice(ids, size=n_val, replace=False).tolist()) if n_val else set()
    return np.array([e not in val for e in ep_ids])


# ------------------------------------------------------------------ student
def make_student(d, hidden, n_layers, ob_dim=None, set_table=None, set_hidden=256):
    """The student head. set_table=None: the input is the standardised [s, c] (mean descriptor).
    set_table (T, K_g, f), standardised: the input is [standardised s, task index]; the task's goal
    set is embedded per goal by an MLP and averaged (permutation-invariant), then joined with s."""
    import flax.linen as nn
    import jax.numpy as jnp

    class Student(nn.Module):
        @nn.compact
        def __call__(self, x):
            if set_table is not None:
                s, t = x[..., :ob_dim], x[..., ob_dim].astype(jnp.int32)
                e = jnp.asarray(set_table)[t]                                   # (B, K_g, f)
                for _ in range(2):
                    e = nn.gelu(nn.Dense(set_hidden)(e))
                x = jnp.concatenate([s, e.mean(axis=-2)], axis=-1)
            for _ in range(n_layers):
                x = nn.gelu(nn.Dense(hidden)(x))
            out = nn.Dense(d + d * (d + 1) // 2, kernel_init=nn.initializers.zeros)(x)
            mu, raw = out[..., :d], out[..., d:]
            # zero init: mu = 0 and L = softplus(0) + 1e-3 on the diagonal, close to the prior
            return mu, raw

    return Student()


def train_student(x, v, x_val, v_val, cfg, key, ob_dim=None, set_table=None, set_train_rows=None):
    """Fit the student by NLL. +weight_decay > 0 uses AdamW. +early_stop keeps the parameters at the
    lowest validation NLL (checked +val_checks times over training). set_table: see make_student;
    its features are standardised over set_train_rows (the training tasks' rows of the table)."""
    import jax
    import jax.numpy as jnp
    import optax

    d = v.shape[-1]
    # floor the std: a descriptor dim constant over the training tasks must not blow up on a held-out task
    x_mean, x_std = x.mean(0), np.maximum(x.std(0), 1e-2)
    if set_table is not None:
        x_mean[ob_dim:], x_std[ob_dim:] = 0.0, 1.0                    # the task index passes through
        g = set_table[set_train_rows].reshape(-1, set_table.shape[-1])
        set_table = (set_table - g.mean(0)) / np.maximum(g.std(0), 1e-2)
    norm = lambda a: (a - x_mean) / x_std  # noqa: E731
    net = make_student(d, int(cfg.get('hidden', 512)), int(cfg.get('n_layers', 2)), ob_dim=ob_dim,
                       set_table=set_table)
    params = net.init(key, jnp.zeros((1, x.shape[-1])))
    wd = float(cfg.get('weight_decay', 0.0))
    lr = float(cfg.get('lr', 3e-4))
    tx = optax.adamw(lr, weight_decay=wd) if wd > 0 else optax.adam(lr)
    opt = tx.init(params)

    def loss_fn(p, xb, vb):
        mu, raw = net.apply(p, norm(xb))
        return jnp.mean(gaussian_nll(vb, mu, tril_from_raw(raw, d)))

    @jax.jit
    def step(p, o, xb, vb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, vb)
        upd, o = tx.update(g, o, p)
        return optax.apply_updates(p, upd), o, loss

    eval_loss = jax.jit(loss_fn)
    steps, bs = int(cfg.get('train_steps', 50000)), int(cfg.get('train_batch', 1024))
    early = bool(cfg.get('early_stop', False))
    n_checks = int(cfg.get('val_checks', 50 if early else 20))
    has_val = x_val is not None and len(x_val) > 0
    rng = np.random.default_rng(int(cfg.get('student_seed', cfg.seed)))
    xj, vj = jnp.asarray(x), jnp.asarray(v)
    curve, best = [], (float('inf'), params, 0)
    for i in range(steps):
        idx = jnp.asarray(rng.integers(0, x.shape[0], bs))
        params, opt, loss = step(params, opt, xj[idx], vj[idx])
        if (i + 1) % max(steps // n_checks, 1) == 0 or i == 0:
            row = {'step': i + 1, 'train_nll': float(loss)}
            if has_val:
                row['val_nll'] = float(eval_loss(params, jnp.asarray(x_val), jnp.asarray(v_val)))
                if row['val_nll'] < best[0]:
                    best = (row['val_nll'], params, i + 1)
            curve.append(row)
            print(f'  student step {i + 1}: {row}', flush=True)
    stop = {'early_stop': early, 'weight_decay': wd, 'final_step': steps}
    if early and has_val:
        params = best[1]
        stop.update({'best_step': best[2], 'best_val_nll': best[0]})
        print(f'  early stop: keeping step {best[2]} (val NLL {best[0]:.3f})', flush=True)

    def apply(xb):
        mu, raw = net.apply(params, norm(xb))
        return mu, tril_from_raw(raw, d)

    saved = {'params': params, 'x_mean': x_mean, 'x_std': x_std,
             'set_table': None if set_table is None else np.asarray(set_table)}
    return jax.jit(apply), curve, stop, saved


class StudentActor:
    """The deployed student at one task: a = G(s, clip(mu + L xi)) (mode=False) or G(s, clip(mu))."""

    def __init__(self, teacher, apply_fn, desc, mode):
        import jax
        import jax.numpy as jnp
        u_clip = float(teacher.config['u_clip'])
        d = int(teacher.config['action_dim'])
        desc = jnp.asarray(desc, jnp.float32)

        def act(obs, seed):
            mu, L = apply_fn(jnp.concatenate([obs, desc])[None])
            v = mu[0] if mode else mu[0] + L[0] @ jax.random.normal(seed, (d,))
            return teacher.decode(obs[None], jnp.clip(v, -u_clip, u_clip)[None])[0]

        self._act = jax.jit(act)

    def sample_actions(self, observations, seed=None, temperature=0):
        return self._act(observations, seed)


def save_student(out_dir, saved, report, name, curve, episodes, descs, cfg):
    """Persist one student so its numbers reproduce from files: params (flax msgpack), the input
    standardisation and descriptors (npz), the config and command line (json), the NLL curve (csv) and
    one eval JSON per (task, acting mode) with per-episode success."""
    import csv

    from flax import serialization
    from omegaconf import OmegaConf

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'params.msgpack'), 'wb') as f:
        f.write(serialization.to_bytes(saved['params']))
    arrays = {'x_mean': saved['x_mean'], 'x_std': saved['x_std']}
    arrays.update({f'desc_task{t}': d for t, d in descs.items()})
    if saved['set_table'] is not None:
        arrays['set_table'] = saved['set_table']
    np.savez(os.path.join(out_dir, 'standardisation.npz'), **arrays)
    st = report['students'][name]
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'student': name, 'argv': sys.argv, 'cfg': OmegaConf.to_container(cfg, resolve=True),
                   'teacher': report['teacher'], 'restore_epoch': report['restore_epoch'],
                   'settings': report['settings'], 'train_tasks': st['train_tasks'], 'eval_tasks': st['eval_tasks'],
                   'stop': st['stop'], 'success': st['success'], 'success_mean': st['success_mean']}, f, indent=2)
    keys = sorted({k for r in curve for k in r})
    with open(os.path.join(out_dir, 'nll_curve.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(curve)
    for (t, mode), per_ep in episodes.items():
        with open(os.path.join(out_dir, f'eval_task{t}_{mode}.json'), 'w') as f:
            json.dump({'student': name, 'task': t, 'acting': mode, 'num_episodes': len(per_ep),
                       'success': float(np.mean(per_ep)), 'eval_seed': report['settings']['eval_seed'],
                       'per_episode_success': per_ep}, f)
    print(f'student {name} -> {out_dir}', flush=True)


# ------------------------------------------------------------------ teacher
def load_teacher(cfg):
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, _ = merge_run_config(cli_agent, str(cfg.restore_path), _cli_agent_keys())
    assert merged.get('agent_name') == 'psmgoal' and str(merged.get('policy_index')) == 'goal'
    merged.update(FORCED_KEYS)
    return ml_collections.ConfigDict(_lists_to_tuples(merged)), merged, agents


def make_label_fn(teacher_like):
    """(agent, obs (B, ob), key) -> raw winners v* (B, d), the teacher's pick at each state."""
    import jax
    import jax.numpy as jnp
    K, d = int(teacher_like.config['gpi_num_u']), int(teacher_like.config['action_dim'])
    u_clip = float(teacher_like.config['u_clip'])

    def label(agent, obs, key):
        B = obs.shape[0]
        V = jax.random.normal(key, (B, K, d))
        o = jnp.repeat(obs, K, axis=0)
        q = agent._hgoal_each_score(o, jnp.clip(V, -u_clip, u_clip).reshape(B * K, d),
                                    agent.eval_goals, agent.eval_goal_w, agent.eval_ref).reshape(B, K)
        return V[jnp.arange(B), jnp.argmax(q, axis=1)]

    return jax.jit(label)


def run(cfg):
    import jax
    import jax.numpy as jnp

    from envs.env_utils import make_env_and_datasets
    from tools.eval_state_reward import build_dataset
    from utils.evaluation import evaluate
    from utils.flax_utils import restore_agent
    from utils.log_utils import write_report

    t_start = time.time()
    seed = int(cfg.seed)
    student_seed = int(cfg.get('student_seed', seed))   # student init and batch order only; readout keeps seed
    tasks = parse_tasks(cfg.get('train_tasks', '1 2 3 4 5'))
    holdout = cfg.get('holdout_task', None)
    holdout = None if holdout in (None, '', 'none') else int(holdout)
    n_data = int(cfg.get('n_data', 20000))
    n_roll = int(cfg.get('n_rollout_eps', 50))
    n_extra = int(cfg.get('n_extra_labels', 1))
    val_frac = float(cfg.get('val_frac', 0.1))
    label_bs = int(cfg.get('label_batch', 32))
    n_eval = int(cfg.get('eval_episodes', 500))
    eval_seed = int(cfg.get('eval_seed', 0))
    relabel_size = int(cfg.get('eval_relabel_size', 10000))
    reward_shift = float(cfg.get('eval_reward_shift', 1.0))
    labels_out = cfg.get('labels_out', None)
    labels_in = cfg.get('labels_in', None)
    descriptor = str(cfg.get('descriptor', 'mean'))
    assert descriptor in ('mean', 'set'), 'descriptor must be mean | set'
    in_task, loo = bool(cfg.get('in_task', True)), bool(cfg.get('loo', False))
    student_out = cfg.get('student_out', None)
    n_random = int(cfg.get('n_random_tasks', 0))
    random_seed0 = int(cfg.get('random_seed0', 0))
    random_tasks = [RANDOM_ID0 + i for i in range(n_random)]
    assert descriptor == 'set' or not n_random, 'random goal-region tasks need +descriptor=set'
    all_tasks = sorted(set(tasks) | ({holdout} if holdout else set())) + random_tasks

    config, merged, agents = load_teacher(cfg)
    print(f'=== distill_bon_student teacher={cfg.restore_path}@{cfg.restore_epoch} tasks={tasks} holdout={holdout} '
          f'n_data={n_data} n_rollout_eps={n_roll} n_extra={n_extra} K={config["gpi_num_u"]} '
          f'norm_future_k={merged.get("norm_future_k")} forced={FORCED_KEYS}', flush=True)

    base, per_task, random_info, rand_env = None, {}, {}, None
    for t in all_tasks:
        if t >= RANDOM_ID0:
            # a goal region around the cube position of a random dataset next state, built the eval way:
            # rewarding rows of a relabel batch = next states inside the region, refs = rows outside it
            seed_c = random_seed0 + t - RANDOM_ID0
            if rand_env is None:
                _, rand_env, _, _ = make_env_and_datasets(task_env_name(str(cfg.env_name), tasks[0]),
                                                          frame_stack=cfg.frame_stack)
                check_cube_layout(rand_env)
                rand_env.unwrapped._terminate_at_goal = False    # the env's own task must not end the episode
            row = int(np.random.default_rng(seed_c).integers(0, train_dataset['next_observations'].shape[0]))
            anchor = np.asarray(train_dataset['next_observations'][row], np.float32)[CUBE_POS]
            np.random.seed(seed_c)
            zb = ds.sample(min(ds.size, relabel_size))
            r_c, radius, n_in = region_rewards(zb['next_observations'], anchor, int(config['k_goals']))
            ag = base.infer_eval_goals(zb, r_c)
            goal_set = np.concatenate([np.asarray(ag.eval_goals), np.asarray(ag.eval_goal_w)], 1).astype(np.float32)
            per_task[t] = {'agent': ag, 'env': rand_env, 'desc': np.asarray([all_tasks.index(t)], np.float32),
                           'goal_set': goal_set, 'anchor': anchor,
                           'obs_pool': np.asarray(train_dataset['observations'], np.float32)}
            random_info[t] = {'seed': seed_c, 'anchor_row': row, 'anchor_cube_obs': anchor.tolist(),
                              'radius_obs_units': radius, 'rows_within_tol': n_in, 'n_rewarding': int(r_c.sum())}
            print(f'random task {t}: seed {seed_c} anchor {np.round(anchor, 3)} radius {radius:.3f} '
                  f'({n_in} of {r_c.shape[0]} relabel rows within tol)', flush=True)
            continue
        env_name = task_env_name(str(cfg.env_name), t)
        np.random.seed(seed)                      # eval_checkpoint: the base seed before the dataset
        _, eval_env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=cfg.frame_stack)
        ds = build_dataset(train_dataset, config)
        ex = ds.sample(1)
        if base is None:
            base = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], config),
                                 str(cfg.restore_path), cfg.restore_epoch)
            label_fn = make_label_fn(base)
        zb = ds.sample(min(ds.size, relabel_size))
        ag = base.infer_eval_goals(zb, zb['rewards'] + reward_shift)
        goal_set = np.concatenate([np.asarray(ag.eval_goals), np.asarray(ag.eval_goal_w)], 1).astype(np.float32)
        if descriptor == 'mean':
            desc = np.concatenate([np.asarray(ag.eval_goal_w).mean(0),
                                   np.asarray(ag.eval_goals).mean(0)]).astype(np.float32)
        else:                                     # the task's row in the goal-set table
            desc = np.asarray([all_tasks.index(t)], np.float32)
        per_task[t] = {'agent': ag, 'env': eval_env, 'desc': desc, 'goal_set': goal_set,
                       'obs_pool': np.asarray(train_dataset['observations'], np.float32)}
        print(f'task {t}: goals {np.asarray(ag.eval_goals).shape} refs {np.asarray(ag.eval_ref).shape} '
              f'desc dim {desc.shape[0]}', flush=True)

    # ---------------- labels
    key = jax.random.PRNGKey(seed + 1)
    rng = np.random.default_rng(seed)
    data = {t: {} for t in all_tasks}
    teacher_success = {}
    have = set()
    for path in (str(labels_in).split(':') if labels_in else []):   # reuse previous runs' labels (same teacher)
        z = np.load(path)
        for t in all_tasks:
            if f't{t}_data_obs' not in z.files or t in have:
                continue
            data[t] = {k: z[f't{t}_{k}']
                       for k in ('data_obs', 'data_v', 'roll_obs', 'roll_v', 'roll_ep', 'roll_extra_v')}
            teacher_success[t] = None
            have.add(t)
        n_extra = int(data[min(have)]['roll_extra_v'].shape[0])
        print(f'labels <- {path} for tasks {sorted(have)} (n_extra_labels {n_extra})', flush=True)
    for t in [t for t in all_tasks if t not in have]:
        pt = per_task[t]
        ag, env = pt['agent'], pt['env']
        # dataset states
        obs_d = pt['obs_pool'][rng.integers(0, pt['obs_pool'].shape[0], n_data)]
        vs = []
        for lo in range(0, n_data, label_bs):
            key, sub = jax.random.split(key)
            vs.append(np.asarray(label_fn(ag, jnp.asarray(obs_d[lo:lo + label_bs]), sub)))
        data[t]['data_obs'], data[t]['data_v'] = obs_d, np.concatenate(vs) if vs else np.zeros((0, 5), np.float32)
        # teacher rollouts: the teacher acts with its own pick, every step is a labelled state
        env.reset(seed=seed + 1000 * t)
        r_obs, r_v, r_ep, succ = [], [], [], []
        u_clip = float(ag.config['u_clip'])
        decode = jax.jit(lambda o, u: ag.decode(o[None], u[None])[0])
        for ep in range(n_roll):
            ob, _ = env.reset()
            done, info = False, {}
            while not done:
                key, sub = jax.random.split(key)
                v = np.asarray(label_fn(ag, jnp.asarray(ob, jnp.float32)[None], sub))[0]
                r_obs.append(np.asarray(ob, np.float32))
                r_v.append(v)
                r_ep.append(ep)
                u = jnp.clip(jnp.asarray(v), -u_clip, u_clip)
                a = np.clip(np.asarray(decode(jnp.asarray(ob, jnp.float32), u)), -1, 1)
                ob, _, term, trunc, info = env.step(a)
                done = term or trunc
                if t >= RANDOM_ID0:
                    # random task: success = cube within the env's tolerance of the anchor; ends the episode
                    # as terminate_at_goal does for a real task
                    reached = bool(np.linalg.norm(np.asarray(ob)[CUBE_POS] - pt['anchor'])
                                   <= CUBE_TOL_M * CUBE_XYZ_SCALE)
                    info = {'success': float(reached)}
                    done = done or reached
            succ.append(float(info.get('success', 0.0)))
        r_obs, r_v, r_ep = np.stack(r_obs), np.stack(r_v), np.asarray(r_ep)
        extra = []
        for _ in range(n_extra):
            ve = []
            for lo in range(0, r_obs.shape[0], label_bs):
                key, sub = jax.random.split(key)
                ve.append(np.asarray(label_fn(ag, jnp.asarray(r_obs[lo:lo + label_bs]), sub)))
            extra.append(np.concatenate(ve))
        data[t].update({'roll_obs': r_obs, 'roll_v': r_v, 'roll_ep': r_ep,
                        'roll_extra_v': np.stack(extra) if extra else np.zeros((0,) + r_v.shape, np.float32)})
        teacher_success[t] = float(np.mean(succ))
        print(f'task {t}: {n_data} dataset labels, {r_obs.shape[0]} rollout states over {n_roll} episodes, '
              f'teacher rollout success {teacher_success[t]:.3f} ({time.time() - t_start:.0f} s)', flush=True)
    if labels_out and set(all_tasks) - have:
        os.makedirs(os.path.dirname(os.path.abspath(labels_out)), exist_ok=True)
        np.savez_compressed(labels_out, **{f't{t}_{k}': v for t in all_tasks for k, v in data[t].items()
                                           if t not in have},
                            **{f't{t}_desc': per_task[t]['desc'] for t in all_tasks if t not in have})
        print('labels ->', labels_out, flush=True)

    # ---------------- train + eval
    def build_xy(task_list):
        xs, vs, xv, vv = [], [], [], []
        for t in task_list:
            dt, desc = data[t], per_task[t]['desc']
            tr = split_episodes(dt['roll_ep'], val_frac, seed + t)
            obs_tr = np.concatenate([dt['data_obs'], dt['roll_obs'][tr]] + [dt['roll_obs'][tr]] * n_extra)
            v_tr = np.concatenate([dt['data_v'], dt['roll_v'][tr]] + [e[tr] for e in dt['roll_extra_v']])
            xs.append(np.concatenate([obs_tr, np.broadcast_to(desc, (obs_tr.shape[0], desc.shape[0]))], 1))
            vs.append(v_tr)
            obs_va = dt['roll_obs'][~tr]
            xv.append(np.concatenate([obs_va, np.broadcast_to(desc, (obs_va.shape[0], desc.shape[0]))], 1))
            vv.append(dt['roll_v'][~tr])
        return (np.concatenate(xs).astype(np.float32), np.concatenate(vs).astype(np.float32),
                np.concatenate(xv).astype(np.float32), np.concatenate(vv).astype(np.float32))

    def prior_nll(v):
        return float(np.mean(0.5 * (v ** 2).sum(-1) + 0.5 * v.shape[-1] * LOG_2PI)) if len(v) else float('nan')

    def held_nll(apply_fn, task_list):
        """NLL on the validation rollout states of task_list (held-out episodes)."""
        out = {}
        for t in task_list:
            dt, desc = data[t], per_task[t]['desc']
            tr = split_episodes(dt['roll_ep'], val_frac, seed + t)
            obs = dt['roll_obs'][~tr] if (~tr).any() else dt['roll_obs']
            v = dt['roll_v'][~tr] if (~tr).any() else dt['roll_v']
            x = np.concatenate([obs, np.broadcast_to(desc, (obs.shape[0], desc.shape[0]))], 1).astype(np.float32)
            mu, L = apply_fn(jnp.asarray(x))
            out[t] = {'student_nll': float(jnp.mean(gaussian_nll(jnp.asarray(v), mu, L))), 'prior_nll': prior_nll(v),
                      'n': int(obs.shape[0])}
        return out

    def eval_student(apply_fn, task_list, episodes):
        res = {}
        for t in task_list:
            res[t] = {}
            for mode in ('sample', 'mode'):
                actor = StudentActor(per_task[t]['agent'], apply_fn, per_task[t]['desc'], mode == 'mode')
                info, trajs, _ = evaluate(agent=actor, env=per_task[t]['env'], config=per_task[t]['agent'].config,
                                          num_eval_episodes=n_eval, num_video_episodes=0, seed=eval_seed)
                per_ep = [float(np.max(np.asarray(tr['info'][-1].get('success', 0.0)))) for tr in trajs]
                res[t][mode] = float(np.mean(per_ep))
                episodes[(t, mode)] = per_ep
                print(f'  eval task {t} {mode}: {res[t][mode]:.3f} over {len(per_ep)} episodes '
                      f'({time.time() - t_start:.0f} s)', flush=True)
        return res

    report = {'teacher': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch), 'seed': seed,
              'teacher_config': {k: merged.get(k) for k in ('norm_future_k', 'gpi_num_u', 'u_clip', 'k_goals',
                                                             'batch_size', 'measure_temp', 'actor_kind')},
              'settings': {'n_data': n_data, 'n_rollout_eps': n_roll, 'n_extra_labels': n_extra, 'val_frac': val_frac,
                           'train_steps': int(cfg.get('train_steps', 50000)),
                           'train_batch': int(cfg.get('train_batch', 1024)),
                           'lr': float(cfg.get('lr', 3e-4)), 'hidden': int(cfg.get('hidden', 512)),
                           'n_layers': int(cfg.get('n_layers', 2)), 'eval_episodes': n_eval, 'eval_seed': eval_seed,
                           'descriptor': descriptor, 'weight_decay': float(cfg.get('weight_decay', 0.0)),
                           'early_stop': bool(cfg.get('early_stop', False)), 'labels_in': labels_in,
                           'student_seed': student_seed},
              'teacher_rollout_success': teacher_success, 'random_tasks': random_info, 'students': {}}
    # +exclude_near_real=d (metres): drop the random regions whose anchor lies within d of any real task's goal
    # centroid (cube position), so the real goals are not covered by a nearby training region
    exclude = cfg.get('exclude_near_real', None)
    if random_tasks and exclude not in (None, '', 'none'):
        cents = {t: per_task[t]['goal_set'][:, CUBE_POS].mean(0) / CUBE_XYZ_SCALE for t in tasks}
        kept = []
        for t in random_tasks:
            d = {u: float(np.linalg.norm(per_task[t]['anchor'] / CUBE_XYZ_SCALE - c)) for u, c in cents.items()}
            random_info[t]['dist_to_real_m'] = d
            if min(d.values()) > float(exclude):
                kept.append(t)
        print(f'exclude_near_real={exclude} m: {len(kept)} of {len(random_tasks)} random regions kept: {kept}',
              flush=True)
        report['exclude_near_real_m'] = float(exclude)
        report['random_tasks_kept'] = kept
        random_tasks = kept
    # +exclude_sparse: also drop the random regions whose radius had to grow (fewer than k_goals relabel rows
    # within the tolerance), whose labels come from a teacher that rarely reaches them
    if random_tasks and bool(cfg.get('exclude_sparse', False)):
        tol = CUBE_TOL_M * CUBE_XYZ_SCALE
        kept = [t for t in random_tasks if random_info[t]['radius_obs_units'] <= tol + 1e-6]
        print(f'exclude_sparse: {len(kept)} of {len(random_tasks)} random regions kept: {kept}', flush=True)
        report['exclude_sparse'] = True
        report['random_tasks_kept'] = kept
        random_tasks = kept
    plans = [('in_task', tasks, tasks)] if in_task else []
    if random_tasks:                              # train on the random regions, test on the real tasks
        plans.append(('random_only', random_tasks, tasks))
    if holdout is not None and not (loo and holdout in tasks):
        plans.append(('holdout', [t for t in tasks if t != holdout], [holdout]))
    if loo:                                       # leave-one-out over the training tasks (+ the random ones)
        plans += [(f'loo_{t}', random_tasks + [u for u in tasks if u != t], [t]) for t in tasks]
    ob_dim = per_task[all_tasks[0]]['obs_pool'].shape[1]
    table = np.stack([per_task[t]['goal_set'] for t in all_tasks]) if descriptor == 'set' else None
    for name, tr_tasks, ev_tasks in plans:
        x, v, xv, vv = build_xy(tr_tasks)
        print(f'=== student {name}: train tasks {tr_tasks} ({x.shape[0]} rows), eval tasks {ev_tasks}', flush=True)
        apply_fn, curve, stop, saved = train_student(x, v, xv, vv, cfg, jax.random.PRNGKey(student_seed + 7),
                                                     ob_dim=ob_dim, set_table=table,
                                                     set_train_rows=[all_tasks.index(t) for t in tr_tasks])
        nll = held_nll(apply_fn, ev_tasks)
        episodes = {}
        succ = eval_student(apply_fn, ev_tasks, episodes)
        report['students'][name] = {
            'train_tasks': tr_tasks, 'eval_tasks': ev_tasks, 'n_train_rows': int(x.shape[0]), 'curve': curve,
            'stop': stop, 'val_nll': nll, 'success': succ,
            'success_mean': {m: float(np.mean([succ[t][m] for t in ev_tasks])) for m in ('sample', 'mode')},
        }
        print(f'student {name}: success {report["students"][name]["success_mean"]}', flush=True)
        if student_out:
            save_student(os.path.join(str(student_out), name), saved, report, name, curve, episodes,
                         {t: per_task[t]['desc'] for t in all_tasks}, cfg)
        with open(cfg.report_out, 'w') as f:          # persist after each student
            json.dump(report, f, indent=2)
    report['wall_seconds'] = round(time.time() - t_start, 1)
    write_report(report, cfg, 'distill_bon_student.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
