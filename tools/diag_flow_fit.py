"""Does the frozen behaviour flow FIT the dataset, or MEMORIZE it?

For an anchor state s0 we draw N prior latents, decode them through the frozen Stage-A
flow, and ask how close that generated action cloud is to the recorded actions inside an
epsilon-ball around s0 -- swept over epsilon. Five curves, because a bare "lowest MSE per
ball" answers a weaker question than it appears to:

  recall_own    min_j |G(s0,u_j) - a0|                 can the flow produce THIS state's
                                                       recorded action (epsilon-free)
  recall_ball   mean_i min_j |G(s0,u_j) - a_i|         does it cover the LOCAL action cloud
  precision     mean_j min_i |G(s0,u_j) - a_i|         does it emit actions the data never takes
  data_self     mean_i min_{i'!=i} |a_i - a_i'|        CALIBRATION: what "close" means for
                                                       real data in this same ball
  null          recall_ball against a DIFFERENT ball   does conditioning on s carry
                of equal population                    information at this radius

Why each control is load-bearing:

  * min-over-N falls monotonically with N, so the statistic is meaningless without a fixed,
    stated N. `min_vs_n` reports the whole curve at one radius so the sensitivity is visible.
  * Without `data_self` there is no scale. C1's original "FQL is off-support" reading was
    retracted for exactly this reason once the data-matches-itself baseline was computed.
  * As epsilon grows the ball holds more actions, so a min-over-targets falls mechanically.
    `null` holds the ball POPULATION fixed and moves its location, so the real-vs-null gap
    isolates what conditioning on s buys.
  * recall and precision separate the two failure modes a single number cannot: a flow that
    memorizes one action per state has perfect precision and no recall; a flow emitting
    noise has the reverse.

OVERFITTING PROPER is the train-vs-val contrast. The ball is always built from TRAINING
data; anchors come from either split. A flow that memorized its training set reproduces a
train anchor's own action far better than a held-out anchor's own action, i.e.
`recall_own` splits while `recall_ball` does not.

Latents are drawn UNCLIPPED from N(0, I): u_clip=3.0 is a deployment choice of the Stage-C
actor, not a property of the behaviour flow's fit. Decoding is the exact ODE at
`agent.flow_steps` (pass 100); the one-step distilled head is reported alongside, since
E1 measured the two 6x apart on untrained latents.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=... .venv/bin/python tools/diag_flow_fit.py agent=fql \
      env_name=antmaze-medium-navigate-singletask-v0 \
      restore_path=$PSM_DATA/flow/antmaze-medium-navigate restore_epoch=500000 \
      agent.flow_steps=100 \
      +fit_preimage=$PSM_DATA/preimages_<name>.npz \
      report_out=$PSM_DATA/logs/flow_fit_antmaze.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

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
from utils.geometry import INDEX_ROWS, NeighbourIndex
from utils.log_utils import write_report

#: epsilon grid, in multiples of the median anchor-to-1NN distance. Reported in absolute
#: standardized units too; the multiples are what make cube and antmaze comparable.
EPS_MULTIPLES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
ANCHOR_CHUNK = 8           # anchors per pairwise-distance block (k_max^2 per anchor)


def _bootstrap_ci(x, n_boot=1000, seed=0):
    """Percentile bootstrap over anchors (the independent unit here)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _summary(x, seed=0):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'n': 0}
    lo, hi = _bootstrap_ci(x, seed=seed)
    return {'n': int(x.size), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p90': float(np.percentile(x, 90)), 'ci95': [lo, hi]}


def _decode(agent, obs, noises, chunk=64):
    """Exact ODE decode of `noises` at `obs`; both (A, N, ·). Chunked over anchors."""
    out = []
    for s in range(0, obs.shape[0], chunk):
        o = jnp.asarray(obs[s:s + chunk])[:, None, :]
        n = jnp.asarray(noises[s:s + chunk])
        o = jnp.broadcast_to(o, (o.shape[0], n.shape[1], o.shape[-1]))
        out.append(np.asarray(agent.compute_flow_actions(o, n)))
    return np.concatenate(out, axis=0)


def _onestep(agent, obs, noises, chunk=64):
    """The distilled one-step head on the same noises, for the ODE-vs-onestep contrast."""
    out = []
    net = agent.network.select('actor_onestep_flow')
    for s in range(0, obs.shape[0], chunk):
        o = jnp.asarray(obs[s:s + chunk])[:, None, :]
        n = jnp.asarray(noises[s:s + chunk])
        o = jnp.broadcast_to(o, (o.shape[0], n.shape[1], o.shape[-1]))
        out.append(np.asarray(jnp.clip(net(o, n), -1, 1)))
    return np.concatenate(out, axis=0)


def _panel(index, anchor_obs, anchor_act, anchor_rows, flow_acts, k_max, eps_list,
           null_perm, seed=0):
    """All five statistics for one (split, decode-mode) panel.

    anchor_rows: dataset row id per anchor for self-exclusion, or None when the anchors
    come from a split that is not in the index (val), where no self-match can occur.
    """
    a_scale = index.action_scale
    z = index.standardize(anchor_obs)
    d_state, pos = index.tree.query(z, k=k_max)
    d_state, pos = np.atleast_2d(d_state), np.atleast_2d(pos)
    is_self = (index.rows[pos] == np.asarray(anchor_rows).reshape(-1, 1)
               if anchor_rows is not None else np.zeros(pos.shape, bool))

    n_anchor = anchor_obs.shape[0]
    n_eps = len(eps_list)
    recall_ball = np.full((n_anchor, n_eps), np.nan)
    precision = np.full((n_anchor, n_eps), np.nan)
    data_self = np.full((n_anchor, n_eps), np.nan)
    null = np.full((n_anchor, n_eps), np.nan)
    recall_matched = np.full((n_anchor, n_eps), np.nan)
    # "lowest MSE per ball": min over BOTH flow samples and ball members, squared, and the
    # same statistic computed data-to-data so the ratio has a scale.
    min_mse_ball = np.full((n_anchor, n_eps), np.nan)
    min_mse_self = np.full((n_anchor, n_eps), np.nan)
    ball_n = np.zeros((n_anchor, n_eps), np.int64)
    # random rank per flow sample, so "keep the first m" is an unbiased m-subsample.
    flow_rank = np.argsort(np.random.default_rng(seed).random(flow_acts.shape[:2]), axis=1)
    # epsilon-free: the memorization line.
    recall_own = np.linalg.norm(flow_acts - anchor_act[:, None, :], axis=-1).min(1) / a_scale

    for s in range(0, n_anchor, ANCHOR_CHUNK):
        e = min(s + ANCHOR_CHUNK, n_anchor)
        ball_a = index.act[pos[s:e]]                                   # (c, k, d_a)
        f = flow_acts[s:e]                                             # (c, N, d_a)
        # (c, N, k) flow-to-data and (c, k, k) data-to-data, computed once per chunk and
        # re-masked per epsilon rather than recomputed.
        d_fd = np.linalg.norm(f[:, :, None, :] - ball_a[:, None, :, :], axis=-1) / a_scale
        d_dd = np.linalg.norm(ball_a[:, :, None, :] - ball_a[:, None, :, :], axis=-1) / a_scale
        null_a = index.act[pos[null_perm[s:e]]]                        # (c, k, d_a)
        d_fn = np.linalg.norm(f[:, :, None, :] - null_a[:, None, :, :], axis=-1) / a_scale
        eye = np.eye(d_dd.shape[1], dtype=bool)[None]

        for j, eps in enumerate(eps_list):
            m = (d_state[s:e] <= eps) & ~is_self[s:e]                  # (c, k)
            cnt = m.sum(1)
            ball_n[s:e, j] = cnt
            ok = cnt > 0
            # recall: per data action in the ball, nearest flow sample.
            r = np.where(m[:, None, :], d_fd, np.inf).min(axis=1)      # (c, k)
            recall_ball[s:e, j] = np.where(ok, np.where(m, r, 0.0).sum(1) / np.maximum(cnt, 1),
                                           np.nan)
            # lowest squared error anywhere in the ball (the user-facing "min MSE per ball")
            min_mse_ball[s:e, j] = np.where(ok, np.where(m, r, np.inf).min(1) ** 2, np.nan)
            # precision: per flow sample, nearest data action in the ball.
            p = np.where(m[:, None, :], d_fd, np.inf).min(axis=2)      # (c, N)
            precision[s:e, j] = np.where(ok, p.mean(axis=1), np.nan)
            # data_self needs >= 2 members: each ball action to its nearest OTHER one.
            mm = m[:, :, None] & m[:, None, :] & ~eye
            ds = np.where(mm, d_dd, np.inf).min(axis=2)                # (c, k)
            has = mm.any(axis=2)
            nds = has.sum(1)
            data_self[s:e, j] = np.where(nds > 0,
                                         np.where(has, ds, 0.0).sum(1) / np.maximum(nds, 1),
                                         np.nan)
            min_mse_self[s:e, j] = np.where(nds > 0, np.where(has, ds, np.inf).min(1) ** 2,
                                            np.nan)
            # null: same POPULATION, different location -- take the permuted anchor's cnt
            # nearest neighbours (its ball ordered by distance), so counts match exactly.
            rank = np.arange(d_fn.shape[2])[None, :]
            mn = rank < cnt[:, None]
            rn = np.where(mn[:, None, :], d_fn, np.inf).min(axis=1)
            null[s:e, j] = np.where(ok, np.where(mn, rn, 0.0).sum(1) / np.maximum(cnt, 1),
                                    np.nan)
            # matched-N recall: the flow cloud cut down to the ball's own population, so
            # this is directly comparable to data_self (same number of candidate minima).
            mf = flow_rank[s:e] < np.minimum(cnt, flow_acts.shape[1])[:, None]
            rm = np.where(mf[:, :, None] & m[:, None, :], d_fd, np.inf).min(axis=1)
            recall_matched[s:e, j] = np.where(ok, np.where(m, rm, 0.0).sum(1) / np.maximum(cnt, 1),
                                              np.nan)

    saturated = (ball_n >= k_max - (1 if anchor_rows is not None else 0)).mean(axis=0)
    return {
        'recall_own': _summary(recall_own, seed),
        # the literal request: per anchor, min_j |G(s0,u_j) - a0|^2 (epsilon-free).
        'min_mse_own': _summary(recall_own ** 2, seed),
        'per_eps': [
            {'eps': float(eps),
             'ball_size': {'mean': float(ball_n[:, j].mean()),
                           'median': float(np.median(ball_n[:, j]))},
             'saturated_frac': float(saturated[j]),
             'recall_ball': _summary(recall_ball[:, j], seed),
             'recall_ball_matched_n': _summary(recall_matched[:, j], seed),
             'precision': _summary(precision[:, j], seed),
             'data_self': _summary(data_self[:, j], seed),
             'min_mse_ball': _summary(min_mse_ball[:, j], seed),
             'min_mse_data_self': _summary(min_mse_self[:, j], seed),
             'null': _summary(null[:, j], seed)}
            for j, eps in enumerate(eps_list)
        ],
        '_raw': {'recall_own': recall_own, 'recall_ball': recall_ball,
                 'd_state': d_state, 'pos': pos, 'is_self': is_self},
    }


def _min_vs_n(index, flow_acts, panel_raw, eps, seed=0):
    """recall_ball as a function of sample budget N, at one radius.

    The statistic is a minimum over N draws, so it falls with N by construction. This is
    the curve that says how much of any reported number is the model and how much is N.
    """
    a_scale = index.action_scale
    d_state, pos, is_self = panel_raw['d_state'], panel_raw['pos'], panel_raw['is_self']
    n_total = flow_acts.shape[1]
    budgets = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n <= n_total]
    out = []
    for n in budgets:
        vals = np.full(flow_acts.shape[0], np.nan)
        for s in range(0, flow_acts.shape[0], ANCHOR_CHUNK):
            e = min(s + ANCHOR_CHUNK, flow_acts.shape[0])
            ball_a = index.act[pos[s:e]]
            f = flow_acts[s:e, :n]
            d_fd = np.linalg.norm(f[:, :, None, :] - ball_a[:, None, :, :], axis=-1) / a_scale
            m = (d_state[s:e] <= eps) & ~is_self[s:e]
            cnt = m.sum(1)
            r = np.where(m[:, None, :], d_fd, np.inf).min(axis=1)
            vals[s:e] = np.where(cnt > 0, np.where(m, r, 0.0).sum(1) / np.maximum(cnt, 1),
                                 np.nan)
        out.append({'n_samples': int(n), 'recall_ball': _summary(vals, seed)})
    return {'eps': float(eps), 'curve': out}


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    n_anchor = int(cfg.get('fit_anchors', 512))
    n_samples = int(cfg.get('fit_samples', 256))
    k_max = int(cfg.get('fit_kmax', 512))
    seed = int(cfg.seed)
    rng_np = np.random.default_rng(seed)

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    assert config['agent_name'] == 'fql', 'run with agent=fql (Stage-A flow shapes)'
    assert int(config['flow_steps']) >= 100, (
        f"flow_steps={config['flow_steps']}: the exact decode needs >= 100 (D3: NaN at 10, "
        'KS 0.217 at 30, 1.2e-4 at 100). Pass agent.flow_steps=100.')

    ex = ds.sample(1)
    agent = agents[config['agent_name']].create(seed, ex['observations'], ex['actions'], config)
    assert cfg.restore_path is not None, 'a TRAINED Stage-A flow is required'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    obs_all = np.asarray(ds['observations'])
    act_all = np.asarray(ds['actions'])
    index = NeighbourIndex(obs_all, act_all, index_rows=int(cfg.get('fit_index_rows', INDEX_ROWS)),
                           seed=seed)

    # epsilon grid from the data's own scale: median anchor-to-1NN distance.
    probe = rng_np.choice(len(obs_all), size=min(4096, len(obs_all)), replace=False)
    d1, _ = index.tree.query(index.standardize(obs_all[probe]), k=2)
    d1_med = float(np.median(np.atleast_2d(d1)[:, 1]))
    eps_list = [m * d1_med for m in EPS_MULTIPLES]

    panels = {}
    raws = {}
    for split, obs_src, act_src, rows_src in (
            ('train', obs_all, act_all, None),
            ('val', np.asarray(val_dataset['observations']), np.asarray(val_dataset['actions']),
             'none')):
        n_pick = min(n_anchor, len(obs_src))
        pick = rng_np.choice(len(obs_src), size=n_pick, replace=False)
        a_obs, a_act = obs_src[pick], act_src[pick]
        # train anchors live inside the index and must be excluded from their own ball;
        # val anchors are not in it, so no self-match is possible.
        a_rows = pick if split == 'train' else None
        noises = np.asarray(jax.random.normal(
            jax.random.PRNGKey(seed + (0 if split == 'train' else 1)),
            (n_pick, n_samples, int(act_all.shape[-1]))))
        null_perm = rng_np.permutation(n_pick)
        for mode, fn in (('ode', _decode), ('onestep', _onestep)):
            acts = fn(agent, a_obs, noises)
            p = _panel(index, a_obs, a_act, a_rows, acts, k_max, eps_list, null_perm, seed)
            raws[(split, mode)] = (p.pop('_raw'), acts)
            panels[f'{split}_{mode}'] = p

    # Headline table: lowest MSE per anchor / per ball, train vs val, with the
    # data-to-data value in the same ball as the scale ("is this ratio ~1?").
    def _ratio(a, b):
        return float(a / b) if (b and np.isfinite(b) and b > 0) else float('nan')

    j1 = EPS_MULTIPLES.index(1.0)
    min_mse = {}
    for mode in ('ode', 'onestep'):
        rows = {}
        for split in ('train', 'val'):
            p = panels[f'{split}_{mode}']
            own, ball = p['min_mse_own'], p['per_eps'][j1]['min_mse_ball']
            ref = p['per_eps'][j1]['min_mse_data_self']
            rows[split] = {
                'own': {k: own.get(k) for k in ('n', 'mean', 'median', 'p90', 'ci95')},
                'ball_at_1x': {k: ball.get(k) for k in ('n', 'mean', 'median', 'p90', 'ci95')},
                'data_self_at_1x': {k: ref.get(k) for k in ('n', 'mean', 'median', 'p90')},
                'ratio_own_over_data_self': _ratio(own.get('mean'), ref.get('mean')),
                'ratio_ball_over_data_self': _ratio(ball.get('mean'), ref.get('mean')),
            }
        rows['val_over_train_own_mean'] = _ratio(rows['val']['own'].get('mean'),
                                                 rows['train']['own'].get('mean'))
        min_mse[mode] = rows
    min_mse['_units'] = ('squared distance in units of (mean|a|)^2; own = min over flow '
                         'samples to the anchor\'s own action; ball = min over flow samples '
                         'and over ball members at eps = 1x median 1-NN state distance')

    # Sample-budget sensitivity, at the 4x radius on the primary (train, ode) panel.
    raw_ode, acts_ode = raws[('train', 'ode')]
    mvn = _min_vs_n(index, acts_ode, raw_ode, eps_list[EPS_MULTIPLES.index(1.0)], seed)

    # Ceiling: the stored point preimage is the exact inverse, so its decode error is the
    # best any latent could do -- what "fit" could possibly mean for this flow.
    ceiling = None
    pre_path = cfg.get('fit_preimage', None)
    if pre_path and os.path.exists(str(pre_path)):
        with np.load(str(pre_path)) as z:
            n_pick = min(256, z['noise_preimage_point'].shape[0])
            idx = np.sort(rng_np.choice(z['noise_preimage_point'].shape[0], n_pick, replace=False))
            u = z['noise_preimage_point'][idx]
            o = z['observations'][idx]
            a = z['actions'][idx]
        dec = np.asarray(agent.compute_flow_actions(jnp.asarray(o), jnp.asarray(u)))
        e = np.linalg.norm(dec - a, axis=-1) / index.action_scale
        ceiling = {'source': str(pre_path), 'rows': int(n_pick), **_summary(e, seed)}

    report = {
        'env': cfg.env_name,
        'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch),
        'flow_steps': int(config['flow_steps']),
        'seed': seed,
        'n_anchors': n_anchor,
        'n_samples_per_anchor': n_samples,
        'k_max': k_max,
        'latents': 'unclipped N(0, I)',
        'geometry': index.protocol(),
        'eps_grid': {'median_1nn_dist': d1_med, 'multiples': list(EPS_MULTIPLES),
                     'absolute': eps_list},
        'train_rows': int(len(obs_all)),
        'val_rows': int(len(val_dataset['observations'])),
        'panels': panels,
        'min_mse': min_mse,
        'min_vs_n': mvn,
        'point_preimage_ceiling': ceiling,
    }
    write_report(report, cfg, 'diag_flow_fit.json')


if __name__ == '__main__':
    main()
