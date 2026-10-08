"""Does the goal-fed actor make M rank first latents better? (2026-10-08)

For each psmgoal checkpoint (restored with its own flags.json on top of
configs/agent/psmgoal.yaml, as tools/eval_checkpoint.py builds it), on a fixed sample of
dataset rows (s, u_data) with a hindsight goal g = the state k steps ahead on the same
trajectory (`next_observations[row + k - 1]`, rows whose trajectory ends sooner are skipped):

  1. Candidates at s: the data's own latent u_data (column 0) and 63 clipped prior draws
     (one fixed table shared by every checkpoint). Each candidate u is scored by the share
     M puts on g among {g} + 255 fixed reference next states, read at w = h(g):
     exp(M(s,u,g)) / (exp(M(s,u,g)) + sum_r exp(M(s,u,ref_r))), temperature measure_temp.
     This is the per-goal score of `_hgoal_each_score` for one goal. measure_loss=squared
     uses M(s,u,g) itself (as `_goal_score` does).
  2. Per k: rank of u_data among the 64 (1 = best; mean rank and top-8 rate; chance 32.5 and
     0.125); spread = mean over states of the std of the 64 scores, divided by the std over
     states of the best score; mean score of u_data and mean score of the prior draws.
  3. Runs with a goal-fed actor (train_actor and actor_input=goal): the actor's own latent at
     (s, g) is scored too and ranked among the 64. flowbc: u = clip(eps + delta(s, g, eps))
     with eps = candidate 1 (a fixed prior draw), so the actor's rank can be compared with the
     rank of its own eps. tanh actor: its mode u_clip * tanh(mu(s, g)).

The rows are training rows: every checkpoint was trained on the whole dataset and only the
training rows carry a preimage, so there is no held-out row with a u_data.

Run (GPU):
  .venv/bin/python tools/diag_actor_ranking.py --report-out $PSM_DATA/logs/diag_actor_ranking.json
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

PSM_DATA = os.environ.get('PSM_DATA', '/mnt/home/amohan/psm-data')
GROUPS = ('psmgoal_db_sm_cube', 'psmgoal_ja_fbc3_sh_cube', 'psmgoal_ja_fbc03_sh_cube',
          'psmgoal_ja_fbc3_pt_cube', 'psmgoal_gc_sm_cube')
KS = (5, 20, 50, 100)
N_PRIOR = 63
N_REF = 255
TOP = 8


# ------------------------------------------------------------------------ pure helpers
def trajectory_ends(terminals):
    """Per row, the last row of its trajectory. A final row not marked terminal still ends one."""
    terminals = np.asarray(terminals)
    n = terminals.shape[0]
    ends = np.nonzero(terminals > 0.5)[0]
    if ends.size == 0 or ends[-1] != n - 1:
        ends = np.concatenate([ends, [n - 1]])
    return ends[np.searchsorted(ends, np.arange(n), side='left')].astype(np.int64)


def goal_rows(rows, end_of_row, k):
    """(mask, goal row) for the state k steps ahead of each row: `next_observations[row + k - 1]`.
    mask is False where the trajectory ends before that."""
    rows = np.asarray(rows)
    g = rows + k - 1
    return g <= end_of_row[rows], g


def rank_among(scores, x):
    """1-based rank of x[i] among scores[i, :] (1 = best), ties counted half. scores (N, C), x (N,)."""
    scores = np.asarray(scores, np.float64)
    x = np.asarray(x, np.float64)[:, None]
    return 1.0 + (scores > x).sum(axis=1) + 0.5 * (scores == x).sum(axis=1)


def rank_of_first(scores):
    """Rank of column 0 among the other columns (1 = best, ties half). scores (N, C)."""
    scores = np.asarray(scores, np.float64)
    return rank_among(scores[:, 1:], scores[:, 0])


def spread_ratio(scores):
    """Mean over rows of the within-row std of the scores, divided by the std over rows of the
    row's best score. scores (N, C)."""
    scores = np.asarray(scores, np.float64)
    across = scores.max(axis=1).std()
    return float(scores.std(axis=1).mean() / across) if across > 0 else float('nan')


def summarise(scores, actor=None, top=TOP):
    """Metrics of a (N, 64) score table: column 0 = u_data, columns 1.. = prior draws, column 1
    = the actor's eps. actor (N,) is the actor latent's score, or None."""
    scores = np.asarray(scores, np.float64)
    c = scores.shape[1]
    r = rank_of_first(scores)
    out = {
        'n': int(scores.shape[0]),
        'data_mean_rank': float(r.mean()),
        'data_top8_rate': float((r <= top).mean()),
        'chance_mean_rank': (c + 1) / 2.0,
        'chance_top8_rate': top / c,
        'spread_ratio': spread_ratio(scores),
        'within_state_std': float(scores.std(axis=1).mean()),
        'best_score_std_across_states': float(scores.max(axis=1).std()),
        'data_mean_score': float(scores[:, 0].mean()),
        'prior_mean_score': float(scores[:, 1:].mean()),
        'best_mean_score': float(scores.max(axis=1).mean()),
    }
    if actor is not None:
        actor = np.asarray(actor, np.float64)
        ra = rank_among(scores, actor)
        re = rank_among(np.delete(scores, 1, axis=1), scores[:, 1])
        out.update({
            'actor_mean_rank': float(ra.mean()),
            'actor_top8_rate': float((ra <= top).mean()),
            'actor_mean_score': float(actor.mean()),
            'eps_mean_rank': float(re.mean()),
            'actor_beats_eps_rate': float((actor > scores[:, 1]).mean()),
        })
    return out


# ------------------------------------------------------------------------ data
def load_sample(preimage_path, n_states, seed):
    """Fixed rows (valid preimage), reference next states and prior draws, from the npz."""
    from utils.flow_inversion import PREIMAGE_VALID_KEY, repair_invalid_preimages
    keys = ('observations', 'next_observations', 'terminals', 'actions', 'noise_preimage_point',
            PREIMAGE_VALID_KEY)
    with np.load(preimage_path, allow_pickle=False) as z:
        data = {k: z[k] for k in keys}
    data, valid = repair_invalid_preimages(data)
    valid = np.asarray(valid) > 0.5
    end_of_row = trajectory_ends(data['terminals'])
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(np.nonzero(valid)[0], size=n_states, replace=False))
    ref_rows = rng.choice(data['observations'].shape[0], size=N_REF, replace=False)
    prior = rng.standard_normal((n_states, N_PRIOR, data['noise_preimage_point'].shape[-1]))
    return data, end_of_row, rows, ref_rows, prior.astype(np.float32)


# ------------------------------------------------------------------------ model
def restore(run_dir, epoch, ex_obs, ex_act):
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import merge_run_config
    from utils.flax_utils import restore_agent
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = OmegaConf.to_container(OmegaConf.load(os.path.join(repo, 'configs/agent/psmgoal.yaml')),
                                  resolve=True)
    merged, prov = merge_run_config(base, run_dir, set())
    assert prov['flags_json'], f'no flags.json under {run_dir}'
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    agent = agents['psmgoal'].create(0, ex_obs, ex_act, config)
    agent = restore_agent(agent, run_dir, epoch)
    return agent, config


def make_scorer(agent, config, chunk):
    import jax
    import jax.numpy as jnp

    assert str(config.get('measure_form', 'joint')) != 'factorized', 'joint basis only'
    assert str(config.get('policy_index')) == 'goal', 'goal-indexed runs only'
    softmax = str(config.get('measure_loss', 'squared')) == 'softmax'
    temp = float(config['measure_temp'])
    actor_kind = str(config.get('actor_kind', 'tanh'))
    has_actor = bool(config.get('train_actor', False)) and str(config.get('actor_input', 'w')) == 'goal'

    @jax.jit
    def score(ag, obs, u, g, refs):
        """obs (n, ob), u (n, C, da), g (n, ob), refs (R, ob) -> (n, C)."""
        n, C, da = u.shape
        R = refs.shape[0]
        cols = jnp.concatenate([g[:, None], jnp.broadcast_to(refs[None], (n, R, refs.shape[-1]))], axis=1)
        O = jnp.broadcast_to(obs[:, None, None], (n, C, R + 1, obs.shape[-1])).reshape(-1, obs.shape[-1])
        U = jnp.broadcast_to(u[:, :, None], (n, C, R + 1, da)).reshape(-1, da)
        S = jnp.broadcast_to(cols[:, None], (n, C, R + 1, cols.shape[-1])).reshape(-1, cols.shape[-1])
        phi, b = ag.basis(O, U, S)
        w = ag.w_star(g)                                                      # (n, z) = h(g)
        M = jnp.einsum('ncrz,nz->ncr', phi.reshape(n, C, R + 1, -1), w) + b.reshape(n, C, R + 1)
        if softmax:
            return jax.nn.softmax(M / temp, axis=-1)[..., 0]
        return M[..., 0]

    @jax.jit
    def actor_latent(ag, obs, g, eps):
        if actor_kind == 'flowbc':
            return ag.flowbc_latent(obs, g, eps)[0]
        return ag.actor_backup(obs, g)

    def run(obs, u, g, refs, eps):
        out, lat = [], []
        for i in range(0, obs.shape[0], chunk):
            o, uu, gg, ee = obs[i:i + chunk], u[i:i + chunk], g[i:i + chunk], eps[i:i + chunk]
            pad = chunk - o.shape[0]
            if pad:
                o, uu, gg, ee = (np.concatenate([x, np.repeat(x[-1:], pad, axis=0)]) for x in (o, uu, gg, ee))
            if has_actor:
                ua = actor_latent(agent, o, gg, ee)
                lat.append(np.asarray(ua)[:chunk - pad])
                uu = np.concatenate([uu, np.asarray(ua)[:, None]], axis=1)
            out.append(np.asarray(score(agent, o, uu, gg, refs))[:chunk - pad])
        s = np.concatenate(out)
        if has_actor:
            return s[:, :-1], s[:, -1], np.concatenate(lat)
        return s, None, None

    return run, has_actor, actor_kind


def find_run(group, seed):
    hits = sorted(glob.glob(os.path.join(PSM_DATA, 'exp/PSMFLows', group, f'sd{seed:03d}_*')))
    assert len(hits) == 1, f'{group} seed {seed}: {hits}'
    return hits[0]


def table(cells):
    """Mean over seeds per (group, step, k)."""
    agg = {}
    for c in cells:
        for k, m in c['per_k'].items():
            agg.setdefault((c['group'], c['step'], k), []).append(m)
    rows = []
    for (g, s, k), ms in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))):
        row = {'group': g, 'step': s, 'k': int(k), 'n_seeds': len(ms)}
        for key in ms[0]:
            vals = [m[key] for m in ms if key in m]
            row[key] = float(np.mean(vals))
            if len(vals) > 1:
                row[key + '_seed_std'] = float(np.std(vals, ddof=1))
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--groups', nargs='+', default=list(GROUPS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', nargs='+', type=int, default=[250000, 500000])
    p.add_argument('--ks', nargs='+', type=int, default=list(KS))
    p.add_argument('--n-states', type=int, default=2000)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--preimages', default=os.path.join(PSM_DATA, 'preimages/cube-single-play.npz'))
    p.add_argument('--report-out', default=os.path.join(PSM_DATA, 'logs/diag_actor_ranking.json'))
    args = p.parse_args(argv)

    t0 = time.time()
    data, end_of_row, rows, ref_rows, prior = load_sample(args.preimages, args.n_states, args.sample_seed)
    refs = data['next_observations'][ref_rows].astype(np.float32)
    report = {'what': __doc__.split('\n\n')[0], 'args': vars(args), 'n_prior': N_PRIOR, 'n_ref': N_REF,
              'rows_sha': int(np.sum(rows) % 1000003), 'n_rows_per_k': {}, 'cells': []}
    for k in args.ks:
        report['n_rows_per_k'][str(k)] = int(goal_rows(rows, end_of_row, k)[0].sum())
    ex_obs, ex_act = data['observations'][:1], data['actions'][:1]
    for group in args.groups:
        for seed in args.seeds:
            run_dir = find_run(group, seed)
            for step in args.steps:
                if not os.path.exists(os.path.join(run_dir, f'params_{step}.pkl')):
                    print(f'SKIP {group} sd{seed} {step}: no checkpoint', flush=True)
                    continue
                agent, config = restore(run_dir, step, ex_obs, ex_act)
                u_clip = float(config['u_clip'])
                assert bool(config.get('use_point_preimage', False)), 'u_data is the point preimage'
                u_data = np.clip(data['noise_preimage_point'][rows], -u_clip, u_clip).astype(np.float32)
                u_prior = np.clip(prior, -u_clip, u_clip)
                cand = np.concatenate([u_data[:, None], u_prior], axis=1)                   # (N, 64, da)
                scorer, has_actor, actor_kind = make_scorer(agent, config, args.chunk)
                obs = data['observations'][rows].astype(np.float32)
                cell = {'group': group, 'seed': seed, 'step': step, 'run_dir': run_dir,
                        'has_actor': has_actor, 'actor_kind': actor_kind if has_actor else None,
                        'measure_loss': str(config['measure_loss']), 'fb_bc_coeff': float(config['fb_bc_coeff']),
                        'per_k': {}}
                for k in args.ks:
                    ok, grow = goal_rows(rows, end_of_row, k)
                    g = data['next_observations'][grow[ok]].astype(np.float32)
                    s, a, lat = scorer(obs[ok], cand[ok], g, refs, u_prior[ok, 0])
                    m = summarise(s, a)
                    if lat is not None and actor_kind == 'flowbc':
                        m['delta_norm_mean'] = float(np.linalg.norm(lat - u_prior[ok, 0], axis=-1).mean())
                    cell['per_k'][str(k)] = m
                    print(f'{group} sd{seed} {step} k={k}: top8 {m["data_top8_rate"]:.3f} rank '
                          f'{m["data_mean_rank"]:.1f} spread {m["spread_ratio"]:.3f} '
                          f'data {m["data_mean_score"]:.4f} prior {m["prior_mean_score"]:.4f}'
                          + (f' actor rank {m["actor_mean_rank"]:.1f} eps rank {m["eps_mean_rank"]:.1f}'
                             if a is not None else ''), flush=True)
                report['cells'].append(cell)
                report['table'] = table(report['cells'])
                report['elapsed_s'] = time.time() - t0
                os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
                with open(args.report_out, 'w') as f:
                    json.dump(report, f, indent=2)
    print('report ->', args.report_out, f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
