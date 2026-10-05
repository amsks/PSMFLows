"""How fast does a future state stop identifying the first latent? (2026-10-02)

The proposed "softmax over actions" loss trains a score M(s, u, s+) so that, for a state s and
a state s+ taken k steps later on the same trajectory, the latent u_0 the data used at s scores
higher than K latents drawn from the prior:

    q_0 = exp(M(s, u_0, s+)) / sum_{m=0..K} exp(M(s, u_m, s+)),    loss = -log q_0.

If s+ is far ahead, the first action hardly changes the chance of reaching it and the loss
carries no signal. This tool measures that from the dataset alone, per env:

  1. Trajectories are split 90/10 into train / held-out (by whole trajectory).
  2. A score MLP on [s, u, s+] (3 hidden layers of 512, one scalar output; inputs standardised
     per coordinate with train-split statistics; u clipped to +-u_clip) is trained with the loss
     above. u_0 is the row's point preimage (`noise_preimage_point`, what the agents read as
     `noise_preimage`), the other K = 63 candidates are fresh clipped prior draws. s+ is
     `next_observations` at row min(i + k - 1, trajectory end), k ~ Geometric(1 - discount),
     k >= 1 (the convention of `utils.datasets.hindsight_goal_idxs`).
  3. Two more models with the same network and optimiser: `k1` (s+ is always the next state)
     and `floor` (s+ is a random dataset state, so only [s, u] can carry signal: it measures how
     far the data latents can be told from prior draws without any future state).
  4. On held-out trajectories, for each k in --ks, with rows that have k steps left: top-1
     accuracy of picking u_0 among the 64 candidates, information = log(64) - mean loss (nats),
     and the standard deviation of the score over the 64 candidates (mean over rows).
     Control: the same with s+ replaced by a random held-out state.
  5. The share of training pairs per k bucket under the geometric draw.

Rows whose inversion diverged (`preimage_valid` = 0) are never used as (s, u_0) rows.
Standard errors are computed over held-out trajectories (rows of one trajectory are correlated);
the plain binomial one is stored beside it.

Run:
  .venv/bin/python tools/diag_action_from_future.py --env cube
  .venv/bin/python tools/diag_action_from_future.py --env antmaze
  .venv/bin/python tools/diag_action_from_future.py --figure \
      $PSM_DATA/logs/diag_action_from_future_cube-single-play.json \
      $PSM_DATA/logs/diag_action_from_future_antmaze-medium-navigate.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

ENVS = {
    'cube': {'name': 'cube-single-play', 'discount': 0.98},
    'antmaze': {'name': 'antmaze-medium-navigate', 'discount': 0.99},
}
KS = (1, 2, 5, 10, 20, 50, 100, 200)
# (label, lowest k, highest k) of each training-pair bucket.
BUCKETS = (('<=1', 1, 1), ('2-5', 2, 5), ('6-10', 6, 10), ('11-20', 11, 20), ('21-50', 21, 50),
           ('>50', 51, None))


# ------------------------------------------------------------------------ pure helpers
def trajectory_ends(terminals):
    """Per row, the last row of its trajectory. A final row not marked terminal still ends one."""
    terminals = np.asarray(terminals)
    n = terminals.shape[0]
    ends = np.nonzero(terminals > 0.5)[0]
    if ends.size == 0 or ends[-1] != n - 1:
        ends = np.concatenate([ends, [n - 1]])
    return ends[np.searchsorted(ends, np.arange(n), side='left')].astype(np.int64)


def split_trajectories(end_of_row, heldout_frac, seed):
    """Split whole trajectories. Returns (traj_id per row, bool mask of held-out rows)."""
    ends = np.unique(end_of_row)
    traj_id = np.searchsorted(ends, end_of_row).astype(np.int64)
    n_held = max(1, round(heldout_frac * ends.size))
    held = np.random.default_rng(seed).permutation(ends.size)[:n_held]
    is_held = np.zeros(ends.size, bool)
    is_held[held] = True
    return traj_id, is_held[traj_id]


def geometric_k(uniform, discount):
    """k >= 1 with P(k > n) = discount^n, from uniform draws in (0, 1]."""
    return np.floor(np.log(uniform) / np.log(discount)).astype(np.int64) + 1


def geometric_bucket_shares(discount):
    """Share of Geometric(1 - discount) draws in each bucket (no trajectory truncation)."""
    out = {}
    for label, lo, hi in BUCKETS:
        upper = 0.0 if hi is None else discount ** hi
        out[label] = float(discount ** (lo - 1) - upper)
    return out


def bucket_shares(k):
    """Empirical share of the integer steps `k` in each bucket."""
    k = np.asarray(k)
    out = {}
    for label, lo, hi in BUCKETS:
        m = (k >= lo) if hi is None else ((k >= lo) & (k <= hi))
        out[label] = float(m.mean())
    return out


def eligible_rows(rows, end_of_row, k):
    """Rows with at least k steps left, so `next_observations[row + k - 1]` is k steps ahead."""
    rows = np.asarray(rows)
    return rows[rows + k - 1 <= end_of_row[rows]]


def _cluster_se(x, cluster):
    """Standard error of mean(x) when rows of one cluster are correlated."""
    x = np.asarray(x, np.float64)
    _, inv = np.unique(cluster, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return None
    sums = np.bincount(inv, weights=x - x.mean(), minlength=g)
    return float(np.sqrt(g / (g - 1) * (sums ** 2).sum()) / x.size)


def summarise_scores(scores, cluster):
    """Metrics of a (N, C) score table whose column 0 is the data latent.

    A tie with the best other candidate counts as a miss, so a constant score gives accuracy 0.
    """
    scores = np.asarray(scores, np.float64)
    n, c = scores.shape
    correct = (scores[:, 0] > scores[:, 1:].max(axis=1)).astype(np.float64)
    top = scores.max(axis=1, keepdims=True)
    loss = (top[:, 0] + np.log(np.exp(scores - top).sum(axis=1))) - scores[:, 0]
    acc = float(correct.mean())
    return {
        'n_pairs': int(n),
        'n_trajectories': int(np.unique(cluster).size),
        'top1_accuracy': acc,
        'top1_se': _cluster_se(correct, cluster),
        'top1_se_binomial': float(np.sqrt(acc * (1.0 - acc) / n)),
        'mean_loss': float(loss.mean()),
        'information_nats': float(np.log(c) - loss.mean()),
        'information_se': _cluster_se(loss, cluster),
        'score_spread': float(scores.std(axis=1).mean()),
        'score_spread_prior_only': float(scores[:, 1:].std(axis=1).mean()),
        'chance_accuracy': 1.0 / c,
        'max_information_nats': float(np.log(c)),
    }


# ------------------------------------------------------------------------ data
def load_rows(preimage_path, ogbench_path):
    """Observations, next observations, terminals and the point preimage, row-aligned."""
    from utils.flow_inversion import PREIMAGE_VALID_KEY, load_augmented_dataset, repair_invalid_preimages
    keys = ('observations', 'next_observations', 'terminals', 'noise_preimage_point')
    with np.load(preimage_path, allow_pickle=False) as z:
        if PREIMAGE_VALID_KEY in z.files:
            data = {k: z[k] for k in keys + (PREIMAGE_VALID_KEY,)}
        else:
            data = None
    if data is None:  # validity has to be recomputed from every preimage product
        data = load_augmented_dataset(preimage_path)
    data, valid = repair_invalid_preimages(data)
    n = data['observations'].shape[0]
    end_of_row = trajectory_ends(data['terminals'])
    # Inside a trajectory the next state of row i is the state of row i + 1.
    inner = np.nonzero(end_of_row[:-1] != np.arange(n - 1))[0][:200000]
    assert np.array_equal(data['next_observations'][inner], data['observations'][inner + 1]), (
        'next_observations[i] != observations[i + 1] inside a trajectory: terminals do not mark the ends')
    # Row alignment with the OGBench file (the layout ogbench.utils.load_dataset produces).
    with np.load(ogbench_path) as raw:
        ob_mask = raw['terminals'] < 0.5
        raw_obs = raw['observations'][ob_mask]
    assert raw_obs.shape[0] == n, f'preimage npz has {n} rows, the OGBench dataset {raw_obs.shape[0]}'
    m = min(1000, n)
    assert (np.array_equal(raw_obs[:m], data['observations'][:m])
            and np.array_equal(raw_obs[-m:], data['observations'][-m:])), (
        'preimage npz observations differ from the OGBench dataset: latents belong to other rows')
    return data, np.asarray(valid) > 0.5, end_of_row


# ------------------------------------------------------------------------ model
def build(args, data, valid, end_of_row, discount):
    """Everything jitted: the score function, one training chunk per model kind, the scorer."""
    import jax
    import jax.numpy as jnp
    import optax

    from utils.networks import MLP

    u_clip = float(args.u_clip)
    obs = jnp.asarray(data['observations'])
    next_obs = jnp.asarray(data['next_observations'])
    u_all = jnp.clip(jnp.asarray(data['noise_preimage_point']), -u_clip, u_clip)
    ends = jnp.asarray(end_of_row.astype(np.int32))
    train_rows = jnp.asarray(args.train_rows.astype(np.int32))

    tr = args.train_rows
    mu_s = jnp.asarray(data['observations'][tr].mean(0))
    sd_s = jnp.asarray(data['observations'][tr].std(0) + 1e-6)
    u_tr = np.clip(data['noise_preimage_point'][tr], -u_clip, u_clip)
    mu_u = jnp.asarray(u_tr.mean(0))
    sd_u = jnp.asarray(u_tr.std(0) + 1e-6)

    ob_dim, act_dim = obs.shape[1], u_all.shape[1]
    model = MLP(hidden_dims=(args.width,) * args.layers + (1,))
    tx = optax.adam(args.lr)
    n_cand = args.num_prior + 1

    def score(params, s, cands, sp):
        """s (B, ds), cands (B, C, da), sp (B, ds) -> scores (B, C)."""
        c = cands.shape[1]
        s_n = jnp.broadcast_to(((s - mu_s) / sd_s)[:, None], (s.shape[0], c, ob_dim))
        sp_n = jnp.broadcast_to(((sp - mu_s) / sd_s)[:, None], (s.shape[0], c, ob_dim))
        x = jnp.concatenate([s_n, (cands - mu_u) / sd_u, sp_n], axis=-1)
        return model.apply(params, x)[..., 0]

    def loss_fn(params, s, cands, sp):
        return -jax.nn.log_softmax(score(params, s, cands, sp), axis=-1)[:, 0].mean()

    def candidates(key, rows):
        prior = jnp.clip(jax.random.normal(key, (rows.shape[0], args.num_prior, act_dim)), -u_clip, u_clip)
        return jnp.concatenate([u_all[rows][:, None], prior], axis=1)

    def sample(key, kind):
        k_row, k_goal, k_prior = jax.random.split(key, 3)
        rows = train_rows[jax.random.randint(k_row, (args.batch_size,), 0, train_rows.shape[0])]
        if kind == 'geometric':
            unif = jax.random.uniform(k_goal, (args.batch_size,), minval=1e-12, maxval=1.0)
            k = jnp.floor(jnp.log(unif) / jnp.log(discount)).astype(jnp.int32) + 1
            goal = jnp.minimum(rows + k - 1, ends[rows])
        elif kind == 'k1':
            goal = rows
        else:  # floor: a random training row's next state
            goal = train_rows[jax.random.randint(k_goal, (args.batch_size,), 0, train_rows.shape[0])]
        return obs[rows], candidates(k_prior, rows), next_obs[goal]

    def make_chunk(kind):
        def step(carry, key):
            params, opt_state = carry
            s, cands, sp = sample(key, kind)
            loss, grads = jax.value_and_grad(loss_fn)(params, s, cands, sp)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        @jax.jit
        def chunk(params, opt_state, keys):
            (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), keys)
            return params, opt_state, losses.mean()
        return chunk

    @jax.jit
    def score_rows(params, key, rows, goals):
        return score(params, obs[rows], candidates(key, rows), next_obs[goals])

    def score_table(params, key, rows, goals, chunk=4096):
        """(N, C) numpy scores for held-out (row, goal row) pairs; fresh prior draws per row."""
        n = rows.shape[0]
        pad = (-n) % chunk
        rows_p = np.concatenate([rows, rows[:1].repeat(pad)]).astype(np.int32)
        goals_p = np.concatenate([goals, goals[:1].repeat(pad)]).astype(np.int32)
        out = []
        for j, lo in enumerate(range(0, n + pad, chunk)):
            out.append(np.asarray(score_rows(params, jax.random.fold_in(key, j),
                                             jnp.asarray(rows_p[lo:lo + chunk]),
                                             jnp.asarray(goals_p[lo:lo + chunk]))))
        return np.concatenate(out)[:n]

    def init(key):
        params = model.init(key, jnp.zeros((1, 2 * ob_dim + act_dim)))
        return params, tx.init(params)

    info = {'ob_dim': int(ob_dim), 'action_dim': int(act_dim), 'num_candidates': int(n_cand),
            'u_train_std_after_clip': [float(v) for v in np.asarray(sd_u)],
            'u_train_share_at_clip': float((np.abs(u_tr) >= u_clip).mean())}
    return init, make_chunk, score_table, info


def run_env(args):
    import jax

    from utils.log_utils import write_report

    env = ENVS[args.env]
    name, discount = env['name'], float(env['discount'])
    psm_data = os.environ.get('PSM_DATA', '/mnt/home/amohan/psm-data')
    preimage_path = args.preimages or os.path.join(psm_data, 'preimages', f'{name}.npz')
    ogbench_dir = os.environ.get('OGBENCH_DATASET_DIR', os.path.expanduser('~/.ogbench/data'))
    ogbench_path = os.path.join(ogbench_dir, f'{name}-v0.npz')
    out_path = args.report_out or os.path.join(
        psm_data, 'logs', f"diag_action_from_future_{name}{'_smoke' if args.smoke else ''}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if args.smoke:
        args.steps, args.eval_every, args.eval_pairs, args.curve_pairs = 300, 100, 2000, 1024
    ks = [int(k) for k in args.ks.split(',')]
    print(f'=== diag_action_from_future env={name} discount={discount} steps={args.steps} '
          f'batch={args.batch_size} K={args.num_prior} lr={args.lr} layers={args.layers}x{args.width} '
          f'u_clip={args.u_clip} eval_pairs={args.eval_pairs} ks={ks} seed={args.seed} smoke={args.smoke}')
    print('jax devices:', jax.devices())

    t0 = time.time()
    data, valid, end_of_row = load_rows(preimage_path, ogbench_path)
    n = valid.shape[0]
    traj_id, held_mask = split_trajectories(end_of_row, args.heldout_frac, args.seed)
    args.train_rows = np.nonzero(valid & ~held_mask)[0]
    held_rows = np.nonzero(valid & held_mask)[0]
    print(f'rows {n}, invalid preimages {int((~valid).sum())}, trajectories {int(traj_id.max()) + 1}, '
          f'train rows {args.train_rows.size}, held-out rows {held_rows.size} '
          f'({np.unique(traj_id[held_rows]).size} trajectories); load {time.time() - t0:.0f}s')

    rng = np.random.default_rng(args.seed + 1)
    # Training-pair shares per k bucket: the geometric draw, and the same after the cut at the
    # trajectory end (steps actually between s and s+).
    draw_rows = args.train_rows[rng.integers(0, args.train_rows.size, size=2_000_000)]
    k_drawn = geometric_k(1.0 - rng.random(draw_rows.size), discount)
    k_eff = np.minimum(k_drawn, end_of_row[draw_rows] - draw_rows + 1)
    shares = {'geometric_analytic': geometric_bucket_shares(discount),
              'after_truncation_at_trajectory_end': bucket_shares(k_eff),
              'share_of_pairs_truncated': float((k_eff < k_drawn).mean()),
              'mean_k_after_truncation': float(k_eff.mean()),
              'cumulative_share_k_le': {str(k): float((k_eff <= k).mean()) for k in ks}}
    print('training-pair shares:', json.dumps(shares['after_truncation_at_trajectory_end']))

    init, make_chunk, score_table, info = build(args, data, valid, end_of_row, discount)

    # Fixed held-out evaluation sets, shared by the three models.
    eval_sets = {}
    for k in ks:
        rows = eligible_rows(held_rows, end_of_row, k)
        if rows.size > args.eval_pairs:
            rows = np.sort(rng.choice(rows, size=args.eval_pairs, replace=False))
        eval_sets[k] = (rows, rows + k - 1)
    ctrl_rows = eval_sets[ks[0]][0] if ks[0] == 1 else held_rows[:args.eval_pairs]
    ctrl_goals = held_rows[rng.integers(0, held_rows.size, size=ctrl_rows.size)]
    # Held-out loss curves: pairs drawn the way each model's training pairs are drawn.
    c_rows = held_rows[rng.integers(0, held_rows.size, size=args.curve_pairs)]
    c_k = geometric_k(1.0 - rng.random(c_rows.size), discount)
    curve_sets = {'geometric': (c_rows, np.minimum(c_rows + c_k - 1, end_of_row[c_rows])),
                  'k1': (c_rows, c_rows),
                  'floor': (c_rows, held_rows[rng.integers(0, held_rows.size, size=c_rows.size)])}

    report = {
        'tool': 'diag_action_from_future', 'env': name, 'discount': discount, 'smoke': bool(args.smoke),
        'config': {'steps': args.steps, 'batch_size': args.batch_size, 'num_prior_candidates': args.num_prior,
                   'lr': args.lr, 'layers': args.layers, 'width': args.width, 'u_clip': args.u_clip,
                   'heldout_frac': args.heldout_frac, 'eval_pairs': args.eval_pairs,
                   'curve_pairs': args.curve_pairs, 'eval_every': args.eval_every, 'seed': args.seed,
                   'ks': ks, 'preimage_path': preimage_path, 'ogbench_path': ogbench_path,
                   'u0': 'noise_preimage_point clipped to +-u_clip',
                   'goal_row': 'min(row + k - 1, trajectory end); s+ = next_observations[goal_row]'},
        'data': {'rows': int(n), 'invalid_preimages': int((~valid).sum()),
                 'trajectories': int(traj_id.max()) + 1, 'train_rows': int(args.train_rows.size),
                 'heldout_rows': int(held_rows.size),
                 'heldout_trajectories': int(np.unique(traj_id[held_rows]).size), **info},
        'training_pair_shares': shares,
        'models': {},
    }

    key = jax.random.PRNGKey(args.seed)
    eval_key = jax.random.PRNGKey(args.seed + 12345)
    for m_idx, kind in enumerate(('geometric', 'k1', 'floor')):
        t1 = time.time()
        k_init, k_train = jax.random.split(jax.random.fold_in(key, m_idx))
        params, opt_state = init(k_init)
        chunk = make_chunk(kind)
        cr, cg = curve_sets[kind]

        def heldout_loss(p, cr=cr, cg=cg):
            return summarise_scores(score_table(p, jax.random.fold_in(eval_key, 999), cr, cg),
                                    traj_id[cr])['mean_loss']

        curve = [{'step': 0, 'heldout_loss': heldout_loss(params), 'train_loss': None}]
        best_loss, best_step, best_params = curve[0]['heldout_loss'], 0, params
        step = 0
        while step < args.steps:
            n_steps = min(args.eval_every, args.steps - step)
            keys = jax.random.split(jax.random.fold_in(k_train, step), n_steps)
            params, opt_state, train_loss = chunk(params, opt_state, keys)
            step += n_steps
            curve.append({'step': step, 'heldout_loss': heldout_loss(params), 'train_loss': float(train_loss)})
            if curve[-1]['heldout_loss'] < best_loss:   # the held-out loss can rise late (overfit)
                best_loss, best_step, best_params = curve[-1]['heldout_loss'], step, params
            if step % (10 * args.eval_every) == 0 or step == args.steps:
                print(f'[{kind}] step {step} train {curve[-1]["train_loss"]:.4f} '
                      f'held-out {curve[-1]["heldout_loss"]:.4f} ({time.time() - t1:.0f}s)', flush=True)

        def evaluate(p):
            out = {}
            for j, k in enumerate(ks):
                rows, goals = eval_sets[k]
                out[str(k)] = summarise_scores(score_table(p, jax.random.fold_in(eval_key, j), rows, goals),
                                               traj_id[rows])
            ctrl = summarise_scores(score_table(p, jax.random.fold_in(eval_key, 500), ctrl_rows, ctrl_goals),
                                    traj_id[ctrl_rows])
            return out, ctrl

        by_k, control = evaluate(params)
        best_by_k, best_control = (by_k, control) if best_step == step else evaluate(best_params)
        held = [c['heldout_loss'] for c in curve]
        report['models'][kind] = {
            'trained_on': {'geometric': 's+ k steps ahead, k geometric with the env discount',
                           'k1': 's+ = the next state', 'floor': 's+ = a random dataset state'}[kind],
            'loss_curve': curve,
            'heldout_loss_final': held[-1], 'heldout_loss_min': float(min(held)),
            'heldout_loss_min_step': int(curve[int(np.argmin(held))]['step']),
            'heldout_loss_change_last_20pct': float(held[-1] - held[int(0.8 * (len(held) - 1))]),
            'eval_by_k': by_k, 'control_random_splus': control,
            # The same evaluation at the checkpoint (every eval_every steps) with the lowest held-out loss.
            'best_heldout_checkpoint': {'step': int(best_step), 'heldout_loss': float(best_loss),
                                        'eval_by_k': best_by_k, 'control_random_splus': best_control},
            'train_seconds': time.time() - t1,
        }
        print(f'[{kind}] final      ' + '  '.join(f'k={k}: {by_k[str(k)]["top1_accuracy"]:.4f}' for k in ks)
              + f'  random-s+: {control["top1_accuracy"]:.4f}', flush=True)
        print(f'[{kind}] best@{best_step} ' + '  '.join(f'k={k}: {best_by_k[str(k)]["top1_accuracy"]:.4f}' for k in ks)
              + f'  random-s+: {best_control["top1_accuracy"]:.4f}', flush=True)
        # Written after every model, so a later failure keeps the finished ones.
        write_report(report, {'report_out': out_path}, os.path.basename(out_path))
    report['seconds'] = time.time() - t0
    write_report(report, {'report_out': out_path}, os.path.basename(out_path))
    print_table(report)


def print_table(report):
    """The geometric model at the final step, per k. The JSON also has every model at its
    lowest-held-out-loss checkpoint (`best_heldout_checkpoint`)."""
    g = report['models']['geometric']
    print(f"\n{report['env']}  (chance {1 / report['data']['num_candidates']:.4f}); geometric model, final step")
    print('   k   top-1 acc   s.e.    info (nats)   score spread')
    for k, r in g['eval_by_k'].items():
        print(f"{k:>4}   {r['top1_accuracy']:.4f}    {r['top1_se']:.4f}   {r['information_nats']:+.3f}        "
              f"{r['score_spread']:.3f}")
    c = g['control_random_splus']
    print(f"random-s+ control: acc {c['top1_accuracy']:.4f} +- {c['top1_se']:.4f}, info {c['information_nats']:+.3f}")
    for kind in ('k1', 'floor'):
        if kind in report['models']:
            m = report['models'][kind]
            r = m['eval_by_k']['1'] if kind == 'k1' else m['control_random_splus']
            where = 'at k=1' if kind == 'k1' else 'on random s+'
            print(f"{kind} model {where}: acc {r['top1_accuracy']:.4f} +- {r['top1_se']:.4f}, "
                  f"info {r['information_nats']:+.3f}")


# ------------------------------------------------------------------------ figure
def make_figure(json_paths, out_path):
    """Top-1 accuracy against k (log axis), one curve per env (geometric model, final step), with the
    chance line and, dotted, each env's level without a future state (the model trained on a random s+)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    styles = [('#2a78d6', 'o'), ('#eb6834', 's')]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    chance, all_k = None, []
    for (color, marker), path in zip(styles, json_paths):
        with open(path) as f:
            rep = json.load(f)
        by_k = rep['models']['geometric']['eval_by_k']
        ks = sorted(int(k) for k in by_k)
        acc = [by_k[str(k)]['top1_accuracy'] for k in ks]
        se = [by_k[str(k)]['top1_se'] or 0.0 for k in ks]
        chance = by_k[str(ks[0])]['chance_accuracy']
        all_k = sorted(set(all_k) | set(ks))
        ax.errorbar(ks, acc, yerr=se, color=color, marker=marker, markersize=6, linewidth=2,
                    capsize=3, markeredgecolor=surface, markeredgewidth=1,
                    label=f"{rep['env']} (discount {rep['discount']})")
        if 'floor' in rep['models']:
            level = rep['models']['floor']['control_random_splus']['top1_accuracy']
            ax.axhline(level, color=color, linewidth=1.2, linestyle=':')
            ax.text(ks[0], level * 0.95, f"{rep['env']}, no future state: {level:.3f}", color=muted,
                    fontsize=8, va='top', ha='left')
    ax.axhline(chance, color=muted, linewidth=1, linestyle='--')
    ax.text(all_k[0], chance * 1.08, f'chance 1/64 = {chance:.4f}', color=muted, fontsize=9, va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k])
    ax.minorticks_off()
    yticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t:g}' for t in yticks])
    ax.set_ylim(0.009, 1.15)
    ax.set_xlabel('k: steps between s and s+ (held-out trajectories)', color=ink)
    ax.set_ylabel('top-1 accuracy of picking the data latent\namong 64 candidates (log scale)', color=ink)
    ax.set_title('Identifying the first latent from a state k steps ahead', color=ink, fontsize=11, loc='left')
    ax.grid(True, which='major', color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, loc='upper right', labelcolor=ink)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, facecolor=surface)
    print(f'figure -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='cube', choices=sorted(ENVS))
    ap.add_argument('--preimages', default='', help='default $PSM_DATA/preimages/<name>.npz')
    ap.add_argument('--report_out', default='',
                    help='default $PSM_DATA/logs/diag_action_from_future_<name>.json')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_prior', type=int, default=63, help='K prior candidates next to the data latent')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--u_clip', type=float, default=3.0)
    ap.add_argument('--heldout_frac', type=float, default=0.1)
    ap.add_argument('--eval_pairs', type=int, default=50000, help='held-out pairs per k')
    ap.add_argument('--curve_pairs', type=int, default=16384, help='held-out pairs for the loss curve')
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--ks', default=','.join(str(k) for k in KS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='300 steps, 2000 pairs per k, *_smoke.json')
    ap.add_argument('--figure', nargs='+', default=None, help='per-env JSONs; draw the figure and exit')
    ap.add_argument('--fig_out', default='', help='default $PSM_DATA/logs/figures/diag_action_from_future.png')
    args = ap.parse_args()
    if args.figure:
        psm_data = os.environ.get('PSM_DATA', '/mnt/home/amohan/psm-data')
        make_figure(args.figure, args.fig_out or os.path.join(psm_data, 'logs', 'figures',
                                                              'diag_action_from_future.png'))
        return
    run_env(args)


if __name__ == '__main__':
    main()
