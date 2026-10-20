"""What does the flowbc actor's bootstrap latent give the critic, at ONE psmgoal checkpoint?
(2026-10-20). Run on a ladder of checkpoints (every 25k) of two critics and plot the curves
with tools/fig_bootstrap_ladder.py.

The critic is M(s, u, y) = phi(s, u, y)^T h(g) + b(s, u, y) over the frozen flow's latents u,
trained by the softmax TD loss over the batch's next states. The bootstrap latent at the next
state is u' = clip(eps + delta(s', g, eps)) from the flowbc actor delta. Critic A (fb_bc_coeff 3)
has a step delta of about 0.09 on a draw eps of about 2.1; critic B (fb_bc_coeff 1000) has a
step of about 0. The point value is q(s, u; g), the softmax share of g among [g + n_pool random
dataset next states] at w = h(g): the deployed coef_source=hgoal_each readout on one goal
(tools/diag_actor_mechanism.make_fns).

Probe set, the SAME for every checkpoint (np.random.seed(seed) before sampling): n_states
dataset transitions with their point preimage u_data, their recorded action a_data, their next
state and a hindsight goal g drawn by the agent's own goal sampling (goal_sampling,
goal_discount, goal_random_frac, goal_cur_frac from the run's flags.json). The probe state is
the row's own state s, as in tools/diag_actor_shift_direction.py, so u_data and a_data belong
to the same row; the bootstrap latent itself is taken at s', which is another dataset state
drawn from the same distribution. Per probe state:

  1. direction (tools/diag_actor_shift_direction.shift_direction_stats on n_draws prior draws):
     dbar = mean over eps of delta(s, g, eps); cos(dbar, gbar) with gbar the mean critic
     gradient d q / d u at the BC draws; cos(dbar, c_top) with c_top the centroid of the top
     top_frac draws under q; cos(dbar, u_data); cos(delta_i, u_bc_i) per draw (outward);
     ||dbar||, mean ||delta_i||, and the share of delta's second moment that is the per-(s, g)
     constant dbar (mean_explained).
  2. self-rating on the first n_sel (64) draws, under the ONLINE params (the deployed value) and
     under the TARGET params (what the TD target sees): q at the BC draw u_bc = clip(eps), at the
     actor's latent u' = clip(eps + delta), at the best of 2 BC draws (disjoint pairs, mean over
     the 32 picks) and at the best of all 64 BC draws; the differences actor - bc, bo2 - bc,
     bo64 - bc; and the percentile rank of q(u') among the 64 q(u_bc) (0.5 = a typical draw,
     1 = above every BC draw).
  3. coverage: u_bc, u', the best-of-2 picks and the best-of-64 pick are decoded through the
     frozen flow (the run's own gpi_decode). For each action the L2 distance to the nearest
     dataset action among the knn_k (32) dataset states nearest to s in observation space,
     within a fixed knn_rows (50k) dataset sample (`knn_action_coverage`); and the L2 distance to
     the recorded action a_data. Plus the per-dimension mean |a' - a_bc| (cube: pos_x, pos_y,
     pos_z, ori, gripper) from tools/diag_actor_shift_direction.action_shift_stats.
  4. TD: on n_td_batches (8) probe batches of batch_size (256) transitions (fixed seed), the
     agent's own measure loss (`PSMGoalAgent.total_loss`, the validation entry point: the same
     measure_loss and _select_u_next as `update`, no gradient step) with its td_target_absmean,
     m_softmax_diag and phi_eff_rank (5.) under the checkpoint's own bootstrap; and the same
     `measure_loss` with the bootstrap latent replaced by the plain BC draw clip(eps), the same
     eps stream (FLOWBC_BOOT_KEY) with delta zeroed: loss_own vs loss_bcboot. The mean
     ||u'_own - u'_bcboot|| over the batches is logged as a check (B: about 0).

Everything is persisted via report_out: `metrics` (flat name -> mean over states / batches),
`metrics_sem`, `summary` (mean +- SE per key), and `arrays` (per-state values of 1-3) so the
curves can be re-aggregated. The checkpoint is restored as diag_actor_mechanism does (flags.json
defaults, typed agent.* overrides on top, acting=gpi and coef_source=hgoal_each forced).

Run (GPU):
  .venv/bin/python tools/diag_bootstrap_ladder.py agent=psmgoal \\
    env_name=cube-single-play-singletask-task1-v0 \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
    restore_path=<run dir> restore_epoch=250000 +tag=A_sd0 \\
    +n_states=512 +n_draws=2000 seed=0 report_out=$PSM_DATA/logs/actor_ladder/A_sd0_250000.json
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside run())
from tools.diag_actor_mechanism import (
    FORCED_KEYS,
    chunked,
    load_critic,
    make_fns,
    summarise_rows,
)
from tools.diag_actor_shift_direction import ACTION_DIM_NAMES, action_shift_stats, shift_direction_stats

#: The latent variants compared in the self-rating and coverage blocks.
VARIANTS = ('bc', 'actor', 'bo2', 'bo64')

#: (panel title, metric names) -- the flat `metrics` keys the figure script plots, one panel per group.
METRIC_GROUPS = [
    ('direction: cos(dbar, .)', ['cos_dbar_gbar', 'cos_dbar_ctop', 'cos_dbar_udata', 'cos_delta_outward']),
    ('step size', ['dbar_norm', 'delta_norm', 'mean_explained']),
    ('self-rating, online: q(.) - q(bc)', ['dq_actor_online', 'dq_bo2_online', 'dq_bo64_online']),
    ('self-rating, target: q(.) - q(bc)', ['dq_actor_target', 'dq_bo2_target', 'dq_bo64_target']),
    ("rank of u' among 64 BC draws", ['rank_actor_online', 'rank_actor_target']),
    ('coverage: NN action distance', ['nn_bc', 'nn_actor', 'nn_bo2', 'nn_bo64']),
    ('distance to recorded action', ['dist_data_bc', 'dist_data_actor', 'dist_data_bo2', 'dist_data_bo64']),
    ('per-dim |a_actor - a_bc|', [f'shift_abs_{n}' for n in ACTION_DIM_NAMES]),
    ('TD loss', ['loss_own', 'loss_bcboot']),
    ('TD target |M| mean', ['td_target_absmean_own', 'td_target_absmean_bcboot']),
    ('softmax mass on own next state', ['m_softmax_diag_own', 'm_softmax_diag_bcboot']),
    ('phi effective rank', ['phi_eff_rank']),
]


# ------------------------------------------------------------------ pure helpers (CPU-tested)
def percentile_rank(q_new, q_ref):
    """Percentile of each q_new among the reference values q_ref: the share of q_ref strictly
    below plus half the share equal. (K,) in [0, 1]; 0.5 = a typical reference value."""
    q_new, q_ref = np.asarray(q_new, np.float64).reshape(-1), np.asarray(q_ref, np.float64).reshape(-1)
    below = (q_ref[None, :] < q_new[:, None]).mean(1)
    equal = (q_ref[None, :] == q_new[:, None]).mean(1)
    return below + 0.5 * equal


def best_of_pairs(q):
    """Indices of the argmax within consecutive disjoint pairs (0,1), (2,3), ... of q (K,); (K // 2,)."""
    q = np.asarray(q, np.float64).reshape(-1)
    K = q.shape[0] // 2
    pairs = q[:2 * K].reshape(K, 2)
    return 2 * np.arange(K) + pairs.argmax(1)


def knn_action_coverage(obs_q, acts_q, ds_obs, ds_act, k=32, chunk=64):
    """For each probe state s_i (obs_q (S, ob)) and each of its candidate actions acts_q (S, K, A):
    the L2 distance to the nearest dataset action among the k dataset states nearest to s_i in
    observation space (ds_obs (D, ob), ds_act (D, A)). Returns (S, K)."""
    obs_q, acts_q = np.asarray(obs_q, np.float64), np.asarray(acts_q, np.float64)
    ds_obs, ds_act = np.asarray(ds_obs, np.float64), np.asarray(ds_act, np.float64)
    S, K, _ = acts_q.shape
    k = min(int(k), ds_obs.shape[0])
    out = np.zeros((S, K), np.float64)
    ds_sq = (ds_obs ** 2).sum(1)
    for i0 in range(0, S, chunk):
        o = obs_q[i0:i0 + chunk]                                                   # (c, ob)
        d2 = (o ** 2).sum(1)[:, None] - 2.0 * o @ ds_obs.T + ds_sq[None, :]         # (c, D)
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k]                              # (c, k)
        a_nn = ds_act[nn]                                                           # (c, k, A)
        diff = acts_q[i0:i0 + chunk][:, :, None, :] - a_nn[:, None, :, :]           # (c, K, k, A)
        out[i0:i0 + chunk] = np.sqrt((diff ** 2).sum(-1)).min(-1)
    return out


def self_rating_stats(q_bc, q_act):
    """q at the BC draws (K,) and at the actor's latents on the same eps (K,): means of the four
    variants (bc, actor, best-of-2 over disjoint pairs, best-of-64 = best of all K), the
    differences to bc, and the mean percentile rank of q_act among q_bc."""
    q_bc, q_act = np.asarray(q_bc, np.float64), np.asarray(q_act, np.float64)
    picks2 = best_of_pairs(q_bc)
    means = {'bc': float(q_bc.mean()), 'actor': float(q_act.mean()),
             'bo2': float(q_bc[picks2].mean()), 'bo64': float(q_bc.max())}
    out = {f'q_{v}': means[v] for v in VARIANTS}
    for v in ('actor', 'bo2', 'bo64'):
        out[f'dq_{v}'] = means[v] - means['bc']
    out['rank_actor'] = float(percentile_rank(q_act, q_bc).mean())
    out['frac_actor_above_bc_mean'] = float(np.mean(q_act > q_bc.mean()))
    return out


def ladder_series(reports):
    """reports: iterable of report dicts (each with 'tag', 'restore_epoch', 'metrics',
    'metrics_sem'). Returns {tag: {metric: {'steps': [...], 'mean': [...], 'sem': [...]}}} with
    the steps sorted; a report missing a metric contributes nothing to that series."""
    out = {}
    for r in reports:
        tag, step = str(r['tag']), int(r['restore_epoch'])
        sems = r.get('metrics_sem', {})
        for k, v in r['metrics'].items():
            if v is None:
                continue
            s = out.setdefault(tag, {}).setdefault(k, {'steps': [], 'mean': [], 'sem': []})
            s['steps'].append(step)
            s['mean'].append(float(v))
            s['sem'].append(float(sems.get(k, 0.0) or 0.0))
    for series in out.values():
        for s in series.values():
            order = np.argsort(s['steps'], kind='stable')
            for key in ('steps', 'mean', 'sem'):
                s[key] = [s[key][j] for j in order]
    return out


# ------------------------------------------------------------------ main
def run(cfg):
    import jax
    import jax.numpy as jnp

    from agents.psmgoal import FLOWBC_BOOT_KEY
    from envs.env_utils import make_env_and_datasets
    from tools.eval_state_reward import build_dataset
    from utils.log_utils import write_report
    from utils.psm_proto import sample_z_bin

    t_start = time.time()
    assert cfg.restore_path is not None, 'restore_path (a psmgoal run dir with the flowbc actor) is required'
    tag = str(cfg.get('tag', 'ladder'))
    n_states = int(cfg.get('n_states', 512))
    n_draws = int(cfg.get('n_draws', 2000))
    n_sel = int(cfg.get('n_sel', 64))
    n_pool = int(cfg.get('n_pool', 224))
    n_grad = int(cfg.get('n_grad_draws', 256))
    top_frac = float(cfg.get('top_frac', 0.05))
    chunk = int(cfg.get('draw_chunk', 500))
    knn_rows = int(cfg.get('knn_rows', 50000))
    knn_k = int(cfg.get('knn_k', 32))
    n_td = int(cfg.get('n_td_batches', 8))
    seed = int(cfg.seed)
    n_top = max(int(np.ceil(top_frac * n_draws)), 2)
    n_grad, n_sel = min(n_grad, n_draws), min(n_sel, n_draws)
    assert n_sel % 2 == 0, 'n_sel must be even (best-of-2 over disjoint pairs)'

    cfg_a, merged, prov, agents, restore_agent = load_critic(cfg, str(cfg.restore_path), tag)
    assert str(merged.get('bootstrap_source', 'actor')) == 'actor' and bool(merged.get('train_actor', False)) \
        and str(merged.get('actor_input', 'w')) == 'goal', \
        f'{tag}: the bcboot counterfactual assumes bootstrap_source=actor, train_actor=true, actor_input=goal'
    assert str(merged.get('bootstrap_switch', 'none')) == 'none' and float(merged.get('bootstrap_delta_scale', 1.0)) == 1.0, \
        f'{tag}: bootstrap_switch / bootstrap_delta_scale change the bootstrap; not handled here'
    batch_size = int(merged['batch_size'])
    print(f'=== diag_bootstrap_ladder: tag={tag} epoch={cfg.restore_epoch} n_states={n_states} n_draws={n_draws} '
          f'n_sel={n_sel} n_pool={n_pool} top={n_top} knn={knn_k}/{knn_rows} td_batches={n_td}x{batch_size} seed={seed}')
    print(f'    run={cfg.restore_path} forced={FORCED_KEYS}')
    print(f'    fb_bc_coeff={merged.get("fb_bc_coeff")} bootstrap_source={merged.get("bootstrap_source")} '
          f'measure_loss={merged.get("measure_loss")} measure_temp={merged.get("measure_temp")} '
          f'goal_sampling={merged.get("goal_sampling")} goal_discount={merged.get("goal_discount")} '
          f'goal_random_frac={merged.get("goal_random_frac")} goal_cur_frac={merged.get("goal_cur_frac")} '
          f'u_clip={merged.get("u_clip")} gpi_decode={merged.get("gpi_decode")}', flush=True)

    # ---- the fixed probe set: same seed -> same rows, goals, pool, kNN sample at every checkpoint
    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(str(cfg.env_name), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, cfg_a)
    ds.return_goals = True
    ds.return_index = True
    ds.goal_sampling = str(merged.get('goal_sampling', 'geometric'))
    ds.goal_discount = float(merged.get('goal_discount', 0.98))
    ds.goal_random_frac = float(merged.get('goal_random_frac', 0.3))
    ds.goal_cur_frac = float(merged.get('goal_cur_frac', 0.0))
    ex = ds.sample(1)
    batch = ds.sample(n_states)
    pool_np = np.asarray(ds.sample(n_pool)['next_observations'], np.float32)
    td_batches = [ds.sample(batch_size) for _ in range(n_td)]
    knn_idx = np.random.default_rng(seed + 7).choice(ds.size, size=min(knn_rows, ds.size), replace=False)
    knn_obs = np.asarray(ds['observations'][knn_idx], np.float32)
    knn_act = np.asarray(ds['actions'][knn_idx], np.float32)
    obs_np = np.asarray(batch['observations'], np.float32)
    goals_np = np.asarray(batch['goals'], np.float32)
    a_data_np = np.asarray(batch['actions'], np.float32)
    u_clip = float(cfg_a['u_clip'])
    u_data_np = np.clip(np.asarray(batch['noise_preimage'], np.float32), -u_clip, u_clip)
    d_a = u_data_np.shape[-1]
    probe_sha = hashlib.sha1(np.ascontiguousarray(batch['index']).tobytes()).hexdigest()[:12]

    agent = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], cfg_a),
                          str(cfg.restore_path), cfg.restore_epoch)
    q_fn, grad_fn, actor_fn, clip_fn = make_fns(agent)
    # the parameters the TD bootstrap reads: the Polyak copy under the advantage-guided arm, else delta
    delta_params = agent.delta_target if agent._ag() else agent.delta.params
    online = (agent.basis.params, agent.w_star.params)
    target = (agent.target_basis, agent.target_w_star)

    @jax.jit
    def decode_fn(obs1, u):
        obs = jnp.broadcast_to(obs1, (u.shape[0], obs1.shape[-1]))
        return agent.decode(obs, u)

    # TD: the agent's own validation loss (own bootstrap) and the same loss with the BC draw as the bootstrap.
    max_log_seed = int(agent.config['max_log_seed'])

    @jax.jit
    def td_own(b):
        return agent.total_loss(b)

    @jax.jit
    def td_bcboot(b):
        N = jnp.asarray(b['observations']).shape[0]
        z = sample_z_bin(agent.rng, N, max_log_seed)                          # unused under policy_index=goal
        eps = jax.random.normal(jax.random.fold_in(agent.rng, FLOWBC_BOOT_KEY), (N, d_a))
        u_bc = jnp.clip(eps, -u_clip, u_clip)
        u_own = agent._select_u_next(b, z)
        loss, info = agent.measure_loss(agent.basis.params, agent.w_star.params, b, z, u_bc)
        return loss, info, jnp.mean(jnp.linalg.norm(u_own - u_bc, axis=-1))

    pool = jnp.asarray(pool_np)
    key = jax.random.PRNGKey(seed)
    dir_rows, act_rows, rate_rows = [], [], {'online': [], 'target': []}
    acts = {v: [] for v in VARIANTS}                      # decoded actions per variant, per state
    arrays = {k: [] for k in ('cos_dbar_gbar', 'cos_dbar_ctop', 'cos_dbar_udata', 'cos_delta_outward',
                              'dbar_norm', 'delta_norm', 'mean_explained')}
    t_loop = time.time()
    for i in range(n_states):
        key, k_eps = jax.random.split(key)
        obs1, goal1 = jnp.asarray(obs_np[i]), jnp.asarray(goals_np[i])
        eps = jax.random.normal(k_eps, (n_draws, d_a))
        u_bc = clip_fn(eps)
        u_bc_np = np.asarray(u_bc)
        # 1. direction
        bp, wp = online
        q_all = chunked(q_fn, u_bc, chunk, bp, wp, obs1, goal1, pool)
        top_idx = np.argsort(-q_all, kind='stable')[:n_top]
        c_top = u_bc_np[top_idx].mean(0)
        grad = np.asarray(grad_fn(bp, wp, obs1, goal1, u_bc[:n_grad], pool))
        u_act, delta = actor_fn(delta_params, obs1, goal1, eps)
        row = shift_direction_stats(np.asarray(delta), np.asarray(eps), u_bc_np, c_top, u_data_np[i], grad)
        row['q_top_mean'] = float(q_all[top_idx].mean())
        row['q_mean'] = float(q_all.mean())
        dir_rows.append(row)
        # 2. self-rating on the first n_sel draws, online and target
        q_bc_on = q_all[:n_sel]
        q_act_on = np.asarray(q_fn(bp, wp, obs1, goal1, u_act[:n_sel], pool))
        tbp, twp = target
        q_bc_tg = np.asarray(q_fn(tbp, twp, obs1, goal1, u_bc[:n_sel], pool))
        q_act_tg = np.asarray(q_fn(tbp, twp, obs1, goal1, u_act[:n_sel], pool))
        rate_rows['online'].append(self_rating_stats(q_bc_on, q_act_on))
        rate_rows['target'].append(self_rating_stats(q_bc_tg, q_act_tg))
        # 3. decoded actions: the picks are subsets of the BC decodes (online q, as deployed)
        a_bc = np.asarray(decode_fn(obs1, u_bc[:n_sel]))
        a_act = np.asarray(decode_fn(obs1, u_act[:n_sel]))
        acts['bc'].append(a_bc)
        acts['actor'].append(a_act)
        acts['bo2'].append(a_bc[best_of_pairs(q_bc_on)])
        acts['bo64'].append(a_bc[int(np.argmax(q_bc_on))][None, :])
        a_top = np.asarray(decode_fn(obs1, jnp.asarray(u_bc_np[top_idx]))).mean(0)
        act_rows.append(action_shift_stats(a_bc, a_act, a_data_np[i], a_top))
        for k, col in arrays.items():
            col.append(row[k])
        if i == 0 or (i + 1) % 64 == 0 or i + 1 == n_states:
            el = time.time() - t_loop
            print(f'  state {i + 1}/{n_states}  {el:.0f}s  ({el / (i + 1):.2f}s/state)', flush=True)

    # ---- coverage (kNN in observation space, min action distance within the neighbours)
    cov_rows = []
    cov_arrays = {}
    for v in VARIANTS:
        a_v = np.stack(acts[v])                                                   # (S, K_v, A)
        nn = knn_action_coverage(obs_np, a_v, knn_obs, knn_act, k=knn_k).mean(1)  # (S,)
        dist = np.linalg.norm(a_v - a_data_np[:, None, :], axis=-1).mean(1)       # (S,)
        cov_arrays[f'nn_{v}'] = nn
        cov_arrays[f'dist_data_{v}'] = dist
    for i in range(n_states):
        cov_rows.append({k: float(col[i]) for k, col in cov_arrays.items()})

    # ---- TD on the probe batches
    td_rows = []
    for b in td_batches:
        loss_o, info_o = td_own(b)
        loss_b, info_b, boot_gap = td_bcboot(b)
        td_rows.append({
            'loss_own': float(loss_o), 'loss_bcboot': float(loss_b),
            'td_target_absmean_own': float(info_o['td_target_absmean']),
            'td_target_absmean_bcboot': float(info_b['td_target_absmean']),
            'm_softmax_diag_own': float(info_o.get('m_softmax_diag', np.nan)),
            'm_softmax_diag_bcboot': float(info_b.get('m_softmax_diag', np.nan)),
            'phi_eff_rank': float(info_o['phi_eff_rank']),
            'm_logit_spread_own': float(info_o.get('m_logit_spread', np.nan)),
            'w_norm': float(info_o['w_norm']),
            'boot_gap_norm': float(boot_gap),
        })

    # ---- summaries
    summary = {
        'direction': summarise_rows(dir_rows),
        'rating_online': summarise_rows(rate_rows['online']),
        'rating_target': summarise_rows(rate_rows['target']),
        'coverage': summarise_rows(cov_rows),
        'action': summarise_rows(act_rows),
        'td': summarise_rows(td_rows),
    }
    metrics, metrics_sem = {}, {}

    def put(sec, k, name=None):
        v = summary[sec].get(k)
        if v is not None:
            metrics[name or k] = v['mean']
            metrics_sem[name or k] = v['sem']

    for k in ('cos_dbar_gbar', 'cos_dbar_ctop', 'cos_dbar_udata', 'cos_delta_outward', 'dbar_norm', 'delta_norm',
              'mean_explained', 'cos_delta_grad', 'cos_delta_to_ctop', 'cos_delta_to_udata', 'cos_gbar_ctop',
              'cos_ctop_udata', 'q_top_mean', 'q_mean'):
        put('direction', k)
    for which in ('online', 'target'):
        for k in list(summary[f'rating_{which}']):
            put(f'rating_{which}', k, f'{k}_{which}')
    for k in list(summary['coverage']):
        put('coverage', k)
    for k in list(summary['action']):
        put('action', k)
    for k in list(summary['td']):
        put('td', k)

    def ms(sec, k, nd=4):
        v = summary[sec].get(k)
        return 'n/a' if v is None or v['mean'] is None else f'{v["mean"]:.{nd}f}±{v["sem"]:.{nd}f}'

    print(f'\n=== {tag} @ {cfg.restore_epoch} (mean ± SE over {n_states} states / {n_td} batches)')
    print(f'direction: cos(dbar,gbar) {ms("direction", "cos_dbar_gbar", 3)}  cos(dbar,c_top) '
          f'{ms("direction", "cos_dbar_ctop", 3)}  cos(dbar,u_data) {ms("direction", "cos_dbar_udata", 3)}  '
          f'cos(delta,outward) {ms("direction", "cos_delta_outward", 3)}')
    print(f'step: |dbar| {ms("direction", "dbar_norm")}  |delta| {ms("direction", "delta_norm")}  '
          f'mean_explained {ms("direction", "mean_explained", 3)}')
    for which in ('online', 'target'):
        sec = f'rating_{which}'
        print(f'self-rating {which}: q_bc {ms(sec, "q_bc")}  q_actor {ms(sec, "q_actor")}  q_bo2 {ms(sec, "q_bo2")}  '
              f'q_bo64 {ms(sec, "q_bo64")}  | actor-bc {ms(sec, "dq_actor", 5)}  bo2-bc {ms(sec, "dq_bo2", 5)}  '
              f'bo64-bc {ms(sec, "dq_bo64", 5)}  rank(u\') {ms(sec, "rank_actor", 3)}')
    print('coverage: NN action dist ' + '  '.join(f'{v} {ms("coverage", f"nn_{v}")}' for v in VARIANTS))
    print('          to a_data      ' + '  '.join(f'{v} {ms("coverage", f"dist_data_{v}")}' for v in VARIANTS))
    print('per-dim |a_actor - a_bc|: ' + '  '.join(f'{n} {ms("action", f"shift_abs_{n}")}' for n in ACTION_DIM_NAMES[:d_a]))
    print(f'TD: loss own {ms("td", "loss_own")}  bcboot {ms("td", "loss_bcboot")}  | target|M| own '
          f'{ms("td", "td_target_absmean_own")}  bcboot {ms("td", "td_target_absmean_bcboot")}  | p_own own '
          f'{ms("td", "m_softmax_diag_own")}  bcboot {ms("td", "m_softmax_diag_bcboot")}  | phi_eff_rank '
          f'{ms("td", "phi_eff_rank", 2)}  boot gap |u_own - u_bc| {ms("td", "boot_gap_norm")}')

    report = {
        'tool': 'diag_bootstrap_ladder',
        'what': __doc__.split('\n\n')[0],
        'tag': tag, 'restore_path': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch), 'seed': seed,
        'env_name': str(cfg.env_name), 'probe_index_sha1': probe_sha,
        'n_states': n_states, 'n_draws': n_draws, 'n_sel': n_sel, 'n_pool': n_pool, 'n_top': n_top,
        'top_frac': top_frac, 'n_grad_draws': n_grad, 'knn_rows': int(knn_idx.shape[0]), 'knn_k': knn_k,
        'n_td_batches': n_td, 'batch_size': batch_size,
        'action_dim': int(d_a), 'u_clip': u_clip, 'action_dim_names': list(ACTION_DIM_NAMES),
        'variants': list(VARIANTS), 'forced_keys': FORCED_KEYS, 'metric_groups': METRIC_GROUPS,
        'goal_sampling': {'goal_sampling': ds.goal_sampling, 'goal_discount': ds.goal_discount,
                          'goal_random_frac': ds.goal_random_frac, 'goal_cur_frac': ds.goal_cur_frac},
        'metrics': metrics, 'metrics_sem': metrics_sem,
        'summary': summary,
        'per_batch_td': td_rows,
        'arrays': {**{k: [float(x) for x in v] for k, v in arrays.items()},
                   **{k: [float(x) for x in v] for k, v in cov_arrays.items()},
                   **{f'{k}_online': [r[k] for r in rate_rows['online']] for k in rate_rows['online'][0]},
                   **{f'{k}_target': [r[k] for r in rate_rows['target']] for k in rate_rows['target'][0]},
                   **{k: [r[k] for r in act_rows] for k in act_rows[0]}},
        'agent_config': json.loads(json.dumps(merged, default=str)),
        'config_provenance': prov,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, 'diag_bootstrap_ladder.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
