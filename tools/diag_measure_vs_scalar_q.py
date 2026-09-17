"""Measure readout psi^T w against a scalar TD critic fitted on the SAME inferred reward.

Eval-only. Two checkpoints on cube-single-play, one reward:

  measure   the affine paper-strict agent (`affine_strict_cube`): psi(s, u, u'), phi, and
            the eval task vector w = project(E[(r + 1) phi(s')]) over 10k relabel rows.
            Its value readout is V_m(s, u) = max_{u'} [mean_P - kappa * unc] psi(s,u,u')^T w,
            the object `gpi_select` maximises over u.
  scalar    a DSRL-NA run (`cube_dsrlna_rhat_scaled`) trained by scalar TD on
            r_scaled(s') = a * phi(s')^T w + b - 1, the SAME w, affinely mapped onto the
            dataset's -1/0 convention. Q_s(s, u) = min_e qa(s, G(s, u)) through the frozen
            flow, and the distilled latent critic qw(s, u).

Both agents decode through the same frozen flow, so V_m and Q_s are two values of the
same (s, u) under the same reward up to the affine map. Six blocks, one JSON:

  A  the measure's own Bellman residual along w on dataset transitions,
         lhs = psi(s, u_data, u')^T w,
         rhs = gamma * [mean_P - kappa * unc](psi_bar(s', u', u')^T w) + phi(s')^T w,
     u' ~ clipped N(0, I) one per row (index_clip). RMS relative to std(lhs), plus the
     FULL z-dim vector residual projected on w/|w| against 20 random unit directions --
     is the w direction fitted worse or better than an average direction?
  B  global agreement: Pearson / Spearman of V_m against Q_s on (s, u_data) and pooled
     over (s, K random u), raw and after per-state centring.
  C  per-state ranking over the K random u: Spearman(V_m(s,.), Q_s(s,.)), its
     distribution, the measure argmax's hit rate in Q_s's top-8, and Q_s's regret at the
     measure argmax normalised by the per-state Q_s range (a random pick scores
     (max - mean) / range).
  D  every block split by whether s' is a success state (reward == 0). Success rows are
     ~2% of the data, so the pool is `n_rows` uniform rows plus extra success rows up to
     `n_success_min`; `all` is the uniform rows alone, `success` every success row in
     the pool, `nonsuccess` the uniform non-success rows.
  E  coverage: |u - u_data(s)| for the K random u; per-state Spearman on the near half
     against the far half (median split per state), pooled per-state-centred agreement
     in global distance quartiles, and the regret split by where the measure's argmax
     sits. Also the subset inside the scalar run's own u box (`u_clip` 1.5 against the
     panel's 3.0), since qw never saw the rest.
  F  sanity: the scalar critic's own Bellman residual on r_scaled with the actor's
     sampled bootstrap latent (should be small), and corr(qa via decode, qw) (should be
     high).

Run (one GPU job; scripts/slurm/diag_measure_vs_scalar_q.sbatch):
  MUJOCO_GL=egl .venv/bin/python tools/diag_measure_vs_scalar_q.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
      restore_path=<measure run dir> restore_epoch=500000 \
      +na_path=<dsrl-na run dir> +na_epoch=500000 \
      +reward_raw_path=$PSM_DATA/rewards/<raw>.npz \
      +reward_scaled_path=$PSM_DATA/rewards/<scaled>.npz \
      report_out=$PSM_DATA/logs/diag_measure_vs_scalar_q_cube_sd001.json

Both agent configs come from their run's own flags.json via `merge_run_config`, so the
objects scored are the ones `tools/eval_checkpoint.py` deploys.

Task-vector-indexed measures (`policy_index=task_vector`, the Section 10 agent): the
index slot carries the inferred w itself, so the value is V_m(s, u) = [mean_P - kappa *
unc] psi(s, u, w)^T w with no index panel (the 64-latent u panel of C/E stays), and the
Bellman bootstrap is the actor's own latent at s', psi_bar(s', pi_eta(s', w, eps), w), as
that run trains it. Runs with a trained actor also get an `actor` block per split: the
scalar critic's regret at the actor's MODE latent pi_eta(s, w, 0), normalised as in C, and
the measure's own regret at that latent.

Policy-side blocks (2026-09-15):

  G  policy-slot information. At u = u_data, the readout over the 64-INDEX panel (u' draws
     for latent-indexed runs; for task-indexed runs 64 task vectors z drawn as the run
     samples them, half project(N(0,I)) and half project(phi(s'_j)) at random pool rows,
     read through the fixed w and, as the run itself reads it, through z); against the
     readout over the 64 ACTION latents at one fixed index. Within-state std along each
     axis pooled over states, their ratio, and the ensemble disagreement along each axis.
     On a 64 x 64 (u, index) grid for `n_grid` states: the two-way split of the
     within-state variance into the u main effect, the index main effect and the
     interaction; and the mean pairwise Spearman across u between index columns (near 1:
     the ranking of action latents does not depend on which policy is asked about).
     Reading: index/action ratio << 1 means the policy slot carries almost nothing.
  H  box exploitation. Panel latents binned into global quartiles of |u|_2 and |u|_inf;
     per bin the state-z-scored measure readout, the state-z-scored scalar Q, the decoded
     action's |G(s,u)|_inf and its clip fraction (> 0.999); the fraction of states whose
     measure argmax (and scalar argmax) lies in the top norm quartile. Runs with an actor
     add the actor latent's norm distribution, its decode's clip fraction, and its
     z-scored measure / scalar values against the panel.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py)

ROW_SEED = 20260915     # dataset rows (a dedicated Generator, never the global stream)
PANEL_KEY = 4242        # the (u, u') panels, one jax key
DIR_SEED = 11           # the random unit directions of block A


# ---------------------------------------------------------------- pure helpers (tested)
def spearman(a, b):
    """Rank correlation; ties broken by position (Q values are continuous floats)."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def pearson(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    da, db = a - a.mean(), b - b.mean()
    return float((da * db).sum() / (np.sqrt((da ** 2).sum() * (db ** 2).sum()) + 1e-12))


def reduce_pess(x, kappa):
    """numpy twin of `targets_uncertainty` folded into mean - kappa * unc. x (P, ...)."""
    x = np.asarray(x, np.float64)
    P = x.shape[0]
    mean = x.mean(0)
    if P == 1:
        return mean
    unc = sum(np.abs(x[p] - x[q]) for p in range(P) for q in range(P) if p != q) / (P * P - P)
    return mean - kappa * unc


def residual_stats(lhs, rhs):
    """mean / RMS of lhs - rhs, and RMS over std(lhs)."""
    lhs, rhs = np.asarray(lhs, np.float64).ravel(), np.asarray(rhs, np.float64).ravel()
    r = lhs - rhs
    return {'n': int(r.size), 'mean': float(r.mean()), 'rms': float(np.sqrt((r ** 2).mean())),
            'std_lhs': float(lhs.std()), 'rms_over_std_lhs': float(np.sqrt((r ** 2).mean())
                                                                    / (lhs.std() + 1e-12)),
            'mean_lhs': float(lhs.mean()), 'mean_rhs': float(rhs.mean())}


def measure_bellman(lhs, boot, phi_next, w, gamma, kappa):
    """Block A on arrays. lhs, boot (P, N, z): online psi(s,u_data,u') and target
    psi_bar(s',u',u'); phi_next (N, z); w (z,).

    Returns the scalar readouts along w with the ensemble reduced the way the acting
    rule reduces it (mean - kappa * unc on the SCALAR psi^T w), the plain-mean variant,
    and the full-vector residual with the reduction taken per component.
    """
    lhs, boot = np.asarray(lhs, np.float64), np.asarray(boot, np.float64)
    phi_next, w = np.asarray(phi_next, np.float64), np.asarray(w, np.float64)
    r_hat = phi_next @ w                                            # (N,)
    lhs_w_p = lhs @ w                                               # (P, N)
    boot_w_p = boot @ w                                             # (P, N)
    lhs_w = lhs_w_p.mean(0)
    rhs_w = gamma * reduce_pess(boot_w_p, kappa) + r_hat
    rhs_w_mean = gamma * boot_w_p.mean(0) + r_hat
    lhs_vec = lhs.mean(0)                                           # (N, z)
    rhs_vec = phi_next + gamma * reduce_pess(boot, kappa)           # (N, z)
    return {'lhs_w': lhs_w, 'rhs_w': rhs_w, 'rhs_w_mean': rhs_w_mean, 'r_hat': r_hat,
            'lhs_w_p': lhs_w_p, 'lhs_vec': lhs_vec, 'rhs_vec': rhs_vec}


def direction_residuals(lhs_vec, rhs_vec, w, n_dirs=20, seed=DIR_SEED):
    """RMS(residual . d) / std(lhs . d) for d = w/|w| and n random unit directions."""
    lhs_vec, rhs_vec = np.asarray(lhs_vec, np.float64), np.asarray(rhs_vec, np.float64)
    res = lhs_vec - rhs_vec
    z = lhs_vec.shape[-1]

    def rel(d):
        d = d / (np.linalg.norm(d) + 1e-12)
        r, lo = res @ d, lhs_vec @ d
        return float(np.sqrt((r ** 2).mean()) / (lo.std() + 1e-12)), float(np.sqrt((r ** 2).mean()))

    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, z))
    rnd = [rel(d) for d in dirs]
    along_rel, along_rms = rel(np.asarray(w, np.float64))
    full = np.linalg.norm(res, axis=-1)
    return {'along_w_rel': along_rel, 'along_w_rms_unit': along_rms,
            'random_dirs_rel_mean': float(np.mean([r[0] for r in rnd])),
            'random_dirs_rel_min': float(np.min([r[0] for r in rnd])),
            'random_dirs_rel_max': float(np.max([r[0] for r in rnd])),
            'random_dirs_rel': [r[0] for r in rnd],
            'full_vector_rms_norm': float(np.sqrt((full ** 2).mean())),
            'full_vector_rms_norm_over_rms_lhs_norm': float(
                np.sqrt((full ** 2).mean()) / (np.sqrt((np.linalg.norm(lhs_vec, axis=-1) ** 2).mean()) + 1e-12)),
            'n_dirs': int(n_dirs), 'z_dim': int(z)}


def per_state_ranking(Vm, Qs, topk=8):
    """Block C on (S, K) arrays: per-state Spearman, top-k hit of the Vm argmax in Qs,
    normalised regret of Qs at the Vm argmax, and the random-pick regret."""
    Vm, Qs = np.asarray(Vm, np.float64), np.asarray(Qs, np.float64)
    S = Vm.shape[0]
    rho = np.array([spearman(Vm[s], Qs[s]) for s in range(S)])
    arg = Vm.argmax(1)
    order = np.argsort(-Qs, axis=1)[:, :topk]
    hit = np.array([arg[s] in set(order[s].tolist()) for s in range(S)], np.float64)
    rng_q = Qs.max(1) - Qs.min(1)
    regret = (Qs.max(1) - Qs[np.arange(S), arg]) / np.maximum(rng_q, 1e-12)
    regret_random = (Qs.max(1) - Qs.mean(1)) / np.maximum(rng_q, 1e-12)
    return {'rho': rho, 'hit_topk': hit, 'regret_norm': regret,
            'regret_random_norm': regret_random, 'argmax': arg, 'topk': topk}


def summarize_rho(rho):
    rho = np.asarray(rho, np.float64)
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        return {'n': 0}
    return {'n': int(rho.size), 'mean': float(rho.mean()), 'median': float(np.median(rho)),
            'q25': float(np.quantile(rho, 0.25)), 'q75': float(np.quantile(rho, 0.75)),
            'frac_gt_0.3': float((rho > 0.3).mean()), 'frac_lt_0': float((rho < 0).mean())}


def center_rows(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)


def near_far_split(Vm, Qs, dist):
    """Per-state Spearman on the K/2 candidates nearest u_data and on the far half."""
    Vm, Qs, dist = (np.asarray(a, np.float64) for a in (Vm, Qs, dist))
    S = Vm.shape[0]
    near, far = [], []
    for s in range(S):
        o = np.argsort(dist[s])
        h = len(o) // 2
        near.append(spearman(Vm[s, o[:h]], Qs[s, o[:h]]))
        far.append(spearman(Vm[s, o[h:]], Qs[s, o[h:]]))
    return np.asarray(near), np.asarray(far)


def dist_bins_pooled(Vm, Qs, dist, n_bins=4):
    """Pooled per-state-centred agreement inside global distance quantile bins."""
    vz, qz = center_rows(Vm).ravel(), center_rows(Qs).ravel()
    d = np.asarray(dist, np.float64).ravel()
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d <= edges[i + 1] if i == n_bins - 1 else d < edges[i + 1])
        out.append({'bin': i, 'dist_lo': float(edges[i]), 'dist_hi': float(edges[i + 1]),
                    'n': int(m.sum()), 'pearson': pearson(vz[m], qz[m]),
                    'spearman': spearman(vz[m], qz[m])})
    return out


def subset_ranking(Vm, Qs, keep, min_n=8):
    """Per-state Spearman restricted to a boolean (S, K) subset; states with fewer than
    `min_n` kept candidates are skipped."""
    Vm, Qs, keep = np.asarray(Vm, np.float64), np.asarray(Qs, np.float64), np.asarray(keep, bool)
    rho = [spearman(Vm[s, keep[s]], Qs[s, keep[s]]) for s in range(Vm.shape[0])
           if keep[s].sum() >= min_n]
    return np.asarray(rho)


def select_rows(rewards, valid, n_rows, n_success_min, seed=ROW_SEED):
    """`n_rows` uniform valid rows, plus extra success rows until the pool holds at least
    `n_success_min` of them. Returns (rows, is_uniform, is_success)."""
    rewards, valid = np.asarray(rewards, np.float64).ravel(), np.asarray(valid, np.float64).ravel()
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(valid > 0.5)
    uniform = rng.choice(ok, size=min(n_rows, ok.size), replace=False)
    succ_all = ok[rewards[ok] >= rewards.max() - 1e-6]
    in_pool = set(uniform.tolist())
    have = int(sum(1 for r in uniform if rewards[r] >= rewards.max() - 1e-6))
    need = max(0, n_success_min - have)
    cand = np.array([r for r in succ_all if r not in in_pool], np.int64)
    extra = rng.choice(cand, size=min(need, cand.size), replace=False) if need and cand.size else np.zeros(0, np.int64)
    rows = np.concatenate([uniform, extra]).astype(np.int64)
    is_uniform = np.concatenate([np.ones(len(uniform), bool), np.zeros(len(extra), bool)])
    is_success = rewards[rows] >= rewards.max() - 1e-6
    return rows, is_uniform, is_success


def agreement(x, y):
    return {'pearson': pearson(x, y), 'spearman': spearman(x, y), 'n': int(np.asarray(x).size)}


def summarize_dist(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            'p10': float(np.quantile(x, 0.1)), 'p50': float(np.quantile(x, 0.5)),
            'p90': float(np.quantile(x, 0.9)), 'max': float(x.max())}


def two_way_decomposition(Q):
    """Block G. Q (S, Ku, Kidx): per state, the within-state variance split into the u
    main effect, the index main effect and the interaction (they sum to the total).
    Returns the mean fraction of each over states and the mean std of each."""
    Q = np.asarray(Q, np.float64)
    m = Q.mean((1, 2), keepdims=True)
    a = Q.mean(2, keepdims=True) - m                     # (S, Ku, 1)
    b = Q.mean(1, keepdims=True) - m                     # (S, 1, Kidx)
    x = Q - m - a - b
    var_u, var_i, var_x = (a ** 2).mean((1, 2)), (b ** 2).mean((1, 2)), (x ** 2).mean((1, 2))
    tot = np.maximum(var_u + var_i + var_x, 1e-24)
    return {'n_states': int(Q.shape[0]), 'Ku': int(Q.shape[1]), 'Kidx': int(Q.shape[2]),
            'frac_u_mean': float((var_u / tot).mean()), 'frac_index_mean': float((var_i / tot).mean()),
            'frac_interaction_mean': float((var_x / tot).mean()),
            'frac_index_median': float(np.median(var_i / tot)),
            'std_u_mean': float(np.sqrt(var_u).mean()), 'std_index_mean': float(np.sqrt(var_i).mean()),
            'std_interaction_mean': float(np.sqrt(var_x).mean()), 'std_total_mean': float(np.sqrt(tot).mean())}


def pairwise_rank_corr(Q, axis):
    """Block G. Mean pairwise Spearman between the slices along `axis` of Q (S, Ku, Kidx),
    each slice ranked over the OTHER axis. axis=2: between index columns, ranking the
    action latents (near 1: every policy ranks the actions alike); axis=1: between action
    rows, ranking the indices."""
    Q = np.asarray(Q, np.float64)
    if axis == 1:
        Q = np.swapaxes(Q, 1, 2)
    S, C = Q.shape[0], Q.shape[2]
    r = np.argsort(np.argsort(Q, axis=1), axis=1).astype(np.float64)   # ranks over rows, per column
    r -= r.mean(1, keepdims=True)
    r /= np.sqrt((r ** 2).sum(1, keepdims=True)) + 1e-12
    corr = np.einsum('src,srd->scd', r, r)                            # (S, C, C)
    off = ~np.eye(C, dtype=bool)
    per_state = corr[:, off].mean(1)
    return {'mean': float(per_state.mean()), 'median': float(np.median(per_state)),
            'p10': float(np.quantile(per_state, 0.1)), 'n_states': int(S), 'n_slices': int(C)}


def norm_bin_table(norms, values, n_bins=4):
    """Block H. Global quantile bins of `norms` (S, K); per bin the mean of every array in
    `values` (each (S, K)) and the count. Returns (edges, rows)."""
    n = np.asarray(norms, np.float64).ravel()
    edges = np.quantile(n, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        m = (n >= edges[i]) & ((n <= edges[i + 1]) if i == n_bins - 1 else (n < edges[i + 1]))
        rows.append({'bin': i, 'lo': float(edges[i]), 'hi': float(edges[i + 1]), 'n': int(m.sum()),
                     **{k: float(np.asarray(v, np.float64).ravel()[m].mean()) for k, v in values.items()}})
    return [float(e) for e in edges], rows


def top_bin_frac(norms, idx, edge):
    """Block H. Fraction of states whose selected latent (one index per state) has norm >= edge."""
    norms, idx = np.asarray(norms, np.float64), np.asarray(idx)
    return float((norms[np.arange(len(idx)), idx] >= edge).mean())


def task_vector_readout(psi_out, w, kappa):
    """The task-vector-indexed value: [mean_P - kappa * unc](psi(s,u,w)^T w). psi_out (P, N, z)."""
    return reduce_pess(np.asarray(psi_out, np.float64) @ np.asarray(w, np.float64), kappa)


def actor_regret(Vm, Qs, Vm_pi, Qs_pi, topk=8):
    """Regret of one latent per state (the actor's) against the (S, K) panel, normalised by
    the per-state panel range, for the scalar critic and for the measure itself. Negative
    means the actor's latent beats every panel latent."""
    Vm, Qs = np.asarray(Vm, np.float64), np.asarray(Qs, np.float64)
    Vm_pi, Qs_pi = np.asarray(Vm_pi, np.float64).ravel(), np.asarray(Qs_pi, np.float64).ravel()
    out = {}
    for name, panel, at in (('scalar', Qs, Qs_pi), ('measure', Vm, Vm_pi)):
        rng_ = np.maximum(panel.max(1) - panel.min(1), 1e-12)
        reg = (panel.max(1) - at) / rng_
        kth = np.sort(panel, axis=1)[:, -topk]
        out[name] = {'regret_norm_mean': float(reg.mean()), 'regret_norm_median': float(np.median(reg)),
                     'frac_beats_panel_max': float((reg < 0).mean()),
                     f'frac_in_panel_top{topk}': float((at >= kth).mean()),
                     'regret_random_norm_mean': float(((panel.max(1) - panel.mean(1)) / rng_).mean()),
                     'value_at_actor_mean': float(at.mean()), 'panel_max_mean': float(panel.max(1).mean())}
    return out


# ------------------------------------------------------------------------- the GPU probe
def _main(cfg):
    import jax
    import jax.numpy as jnp
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
    from utils.log_utils import write_report
    from utils.psm_common import targets_uncertainty

    seed = int(cfg.seed)
    n_rows = int(cfg.get('n_rows', 10000))
    n_success_min = int(cfg.get('n_success_min', 1000))
    K = int(cfg.get('n_panel', 64))
    n_dirs = int(cfg.get('n_random_dirs', 20))
    chunk = int(cfg.get('chunk', 256))
    topk = int(cfg.get('topk', 8))
    n_grid = int(cfg.get('n_grid', 500))
    grid_chunk = int(cfg.get('grid_chunk', 50))
    na_path, na_epoch = str(cfg.na_path), int(cfg.get('na_epoch', 500000))
    reward_raw_path = str(cfg.reward_raw_path)
    reward_scaled_path = str(cfg.reward_scaled_path)

    # ---- the measure agent and its eval w, seeded exactly as tools/eval_checkpoint.py ----
    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    m_conf = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert m_conf['policy_index'] in ('latent', 'task_vector'), m_conf['policy_index']
    m_agent = agents[m_conf['agent_name']].create(seed, ex['observations'], ex['actions'], m_conf)
    m_agent = restore_agent(m_agent, cfg.restore_path, int(cfg.restore_epoch))
    zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
    shift = float(cfg.get('eval_reward_shift', 1.0))
    w_inferred = np.asarray(m_agent.infer_z(zb['next_observations'], zb['rewards'] + shift), np.float64)

    # The w that produced the reward files (stored in the raw file's sidecar). It is the
    # reward the scalar critic was trained on, so it is the one used here; the fresh
    # inference above is a check that the two agree.
    with open(reward_raw_path + '.meta.json') as fh:
        raw_meta = json.load(fh)
    with open(reward_scaled_path + '.meta.json') as fh:
        scaled_meta = json.load(fh)
    w_file = np.asarray(raw_meta['w_inference']['w'], np.float64)
    w_cos = float(w_file @ w_inferred / (np.linalg.norm(w_file) * np.linalg.norm(w_inferred) + 1e-12))
    print(f'w: cos(file, re-inferred) = {w_cos:.6f}  |w_file| {np.linalg.norm(w_file):.4f}')
    # `w_source=file` (default): the sidecar's w, i.e. the reward the scalar critic trained
    # on -- valid only when the measure checkpoint IS the one that wrote the file (its phi).
    # `w_source=infer`: the checkpoint's OWN eval w on its own phi (a different phi gives a
    # different readout of the same true reward); the scalar critic stays on the file's
    # reward, and `r_hat_check.corr_vs_raw_file` says how the two readouts agree.
    w_source = str(cfg.get('w_source', 'file'))
    assert w_source in ('file', 'infer'), 'w_source: file | infer'
    if w_source == 'file':
        assert w_cos > 0.999, (
            f'cos(w_file, w re-inferred) = {w_cos:.4f}: this checkpoint did not write '
            f'{reward_raw_path}; pass +w_source=infer to use its own eval w')
    w = w_file if w_source == 'file' else w_inferred
    m_agent = m_agent.replace(task_z=jnp.asarray(w, jnp.float32))
    a_scale = float(scaled_meta['output']['affine_scale'])
    b_offset = float(scaled_meta['output']['affine_offset'])
    shift_back = float(scaled_meta['output']['shift_back'])
    c_off = b_offset - shift_back                     # r_scaled = a * r_hat + c_off

    # ---- the scalar critic ----
    na_merged, na_prov = merge_run_config(cli_agent, na_path, _cli_agent_keys())
    na_conf = ml_collections.ConfigDict(_lists_to_tuples(na_merged))
    assert na_conf['dsrl_na']['enabled'], f'{na_path} was not trained with dsrl_na.enabled'
    na_agent = agents[na_conf['agent_name']].create(seed, ex['observations'], ex['actions'], na_conf)
    na_agent = restore_agent(na_agent, na_path, na_epoch)
    with open(os.path.join(na_path, 'flags.json')) as fh:
        na_reward_override = str((json.load(fh).get('dataset') or {}).get('reward_override_path'))
    assert os.path.abspath(na_reward_override) == os.path.abspath(reward_scaled_path), (
        f'the scalar critic trained on {na_reward_override!r}, not on {reward_scaled_path!r}')

    mc, nc = m_agent.config, na_agent.config
    d_a, z_dim = int(mc['action_dim']), int(mc['z_dim'])
    u_clip, ic = float(mc['u_clip']), float(m_agent._index_clip())
    P, kappa_td, kappa_act = int(mc['num_parallel']), float(mc['pessimism_penalty']), float(mc['actor_pessimism_penalty'])
    gamma_m, gamma_na = float(mc['discount']), float(nc['dsrl_na']['discount'])
    na_u_clip = float(nc['u_clip'])
    assert mc['flow_ckpt_path'] == nc['flow_ckpt_path'] and int(mc['flow_ckpt_epoch']) == int(nc['flow_ckpt_epoch']), (
        'the two runs decode through different flows')

    # ---- rows: the preimage-augmented dataset the measure trained on ----
    aug = load_augmented_dataset(mc['preimage_path'])
    aug, valid = repair_invalid_preimages(aug)
    assert aug['observations'].shape[0] == ds.size
    assert np.allclose(aug['observations'][:1000], np.asarray(ds['observations'][:1000])), (
        'preimage npz observations differ from the env dataset')
    r_raw_file = np.load(reward_raw_path)['rewards'].astype(np.float64)
    r_scaled_file = np.load(reward_scaled_path)['rewards'].astype(np.float64)
    assert len(r_raw_file) == ds.size and len(r_scaled_file) == ds.size
    rows, is_uniform, is_success = select_rows(aug['rewards'], valid, n_rows, n_success_min)
    N = len(rows)
    obs = np.asarray(aug['observations'][rows], np.float32)
    next_obs = np.asarray(aug['next_observations'][rows], np.float32)
    a_data = np.asarray(aug['actions'][rows], np.float32)
    u_data = np.clip(np.asarray(aug['noise_preimage_point'][rows], np.float32), -u_clip, u_clip)
    masks = np.asarray(aug['masks'][rows], np.float32)
    print(f'rows: {N} = {int(is_uniform.sum())} uniform + {int((~is_uniform).sum())} extra success; '
          f'{int(is_success.sum())} success rows in the pool ({int((is_success & is_uniform).sum())} in the uniform set)')

    # ---- panels: K action latents and K policy indices per row, one key ----
    k_u, k_i, k_n = jax.random.split(jax.random.PRNGKey(PANEL_KEY), 3)
    u_panel = np.asarray(jnp.clip(jax.random.normal(k_u, (N, K, d_a)), -u_clip, u_clip), np.float32)
    u_index = np.asarray(jnp.clip(jax.random.normal(k_i, (N, K, d_a)), -ic, ic), np.float32)
    u_prime = u_index[:, 0]                                   # block A's one u' per row
    noise_next = np.asarray(jax.random.normal(k_n, (N, d_a)), np.float32)

    w_j = jnp.asarray(w, jnp.float32)
    tv = mc['policy_index'] == 'task_vector'
    has_actor = bool(mc['train_actor'])

    def _wb(o):
        return jnp.broadcast_to(w_j, (o.shape[0], z_dim))

    def _pess_q(o, u, uidx):
        """[mean_P - kappa_act * unc] psi(s, u, u'_k)^T w for K indices: (K, B)."""
        q_panel = m_agent._psi_q_over_indices(o, u, _wb(o), uidx)     # (P, K, B)
        qm, qu = targets_uncertainty(q_panel, P)
        return qm - kappa_act * qu

    def _pess_readout(o, u):
        """task_vector: [mean_P - kappa_act * unc] psi(s, u, w)^T w, the index slot IS w: (B,)."""
        q = (m_agent.psi_b(o, _wb(o), u) * w_j).sum(-1)               # (P, B)
        qm, qu = targets_uncertainty(q, P)
        return qm - kappa_act * qu

    def _value(o, u, uidx):
        """The deployed measure value of one latent per row, both index modes: (B,) x2
        (max over the index panel, mean over it; identical under task_vector)."""
        if tv:
            v = _pess_readout(o, u)
            return v, v
        Q = _pess_q(o, u, uidx)
        return Q.max(0), Q.mean(0)

    @jax.jit
    def measure_at_data(o, u, uidx):                                  # uidx (K, B, d_a)
        return _value(o, u, uidx)

    @jax.jit
    def measure_panel(o, up, uidx):                                   # up (K, B, d_a)
        return jax.lax.map(lambda u_k: _value(o, u_k, uidx), up)      # (K, B) x2

    @jax.jit
    def actor_at_state(o, uidx):
        """The trained actor's MODE latent pi_eta(s, w, 0), its decode's scalar values and
        its measure value."""
        u_pi = m_agent._deploy_latent(o, _wb(o), jnp.zeros((o.shape[0], d_a), jnp.float32))
        a = na_agent.decode(o, u_pi)
        return (u_pi, na_agent.qa(o, a).min(0), na_agent.qw(o, u_pi), _value(o, u_pi, uidx)[0],
                jnp.abs(a).max(-1), (jnp.abs(a) > 0.999).mean(-1))

    @jax.jit
    def scalar_panel(o, up):
        def f(u_k):
            a = na_agent.decode(o, u_k)
            return (na_agent.qa(o, a).min(0), na_agent.qw(o, u_k),
                    jnp.abs(a).max(-1), (jnp.abs(a) > 0.999).mean(-1))
        return jax.lax.map(f, up)                                     # (K, B) x4

    # Block G: the affine head factorises psi(s, u, index) = A(s,u) w_enc(index) + beta(s,u),
    # so a (u, index) grid costs one A per u and one encoding per index.
    affine_grid = mc['psi_form'] == 'affine' and mc['psi_bound'] != 'tanh'

    def _affine_terms(o, u):
        return m_agent.psi(o, m_agent._measure_input(o, u), method='sa_terms')   # (P,B,z,dw),(P,B,z)

    @jax.jit
    def grid_readout(o, up, idxp):
        """[mean_P - kappa_act*unc] psi(s, u_i, index_k)^T w over the (u_i, index_k) grid,
        the ensemble disagreement of that readout, and the same readout through the index
        itself (task_vector: psi^T z, the run's own read; latent: a copy of psi^T w).
        up (Ku, B, d_a), idxp (Kidx, B, d_idx) -> three (Ku, Kidx, B)."""
        wenc = m_agent.psi(idxp, method='encode_index')                        # (Kidx, B, dw)

        def f(u_i):
            A, beta = _affine_terms(o, u_i)
            psi_k = jnp.einsum('pbzw,kbw->pkbz', A, wenc) + beta[:, None]      # (P, Kidx, B, z)
            q_w = jnp.einsum('pkbz,z->pkb', psi_k, w_j)
            qm, qu = targets_uncertainty(q_w, P)
            if tv:
                q_s = jnp.einsum('pkbz,kbz->pkb', psi_k, idxp)
                sm, su = targets_uncertainty(q_s, P)
                return qm - kappa_act * qu, qu, sm - kappa_act * su
            return qm - kappa_act * qu, qu, qm - kappa_act * qu
        return jax.lax.map(f, up)

    @jax.jit
    def scalar_at_data(o, u, a):
        a_dec = na_agent.decode(o, u)
        return (na_agent.qa(o, a_dec).min(0), na_agent.qw(o, u), na_agent.qa(o, a).min(0),
                jnp.linalg.norm(a_dec - a, axis=-1))

    @jax.jit
    def bellman_measure_arrays(o, u, o2, upr, nz):
        """Online psi at (s, u_data), the target psi at s' under the run's OWN bootstrap,
        phi(s'), and the bootstrap latent. latent: index u' and continuation u'
        (`sample_step_inputs`: u_next = u_index). task_vector: index w and the actor's
        sampled latent at s' (u_next = pi_eta(s', w, eps)); `nz` is that eps."""
        if tv:
            wb = _wb(o)
            lhs = m_agent.psi_b(o, wb, u)                                     # (P, B, z)
            u_next = m_agent._deploy_latent(o2, wb, nz)
            boot = m_agent.psi_b(o2, wb, u_next, params=m_agent.target_psi)   # (P, B, z)
            return lhs, boot, m_agent.phi(o2), u_next
        lhs = m_agent.psi_b(o, upr, u)                                        # (P, B, z)
        boot = m_agent.psi_b(o2, upr, upr, params=m_agent.target_psi)         # (P, B, z)
        return lhs, boot, m_agent.phi(o2), upr

    @jax.jit
    def bellman_scalar_arrays(o, a, o2, r, m, nz):
        w0 = na_agent._actor_w(jnp.zeros((o.shape[0], z_dim), jnp.float32))
        u_next = na_agent._deploy_latent(o2, w0, nz)
        a_next = na_agent.decode(o2, u_next)
        q_next = na_agent.qa(o2, na_agent._na_in(a_next, w0), params=na_agent.target_qa).min(0)
        target = r + gamma_na * m * q_next
        q = na_agent.qa(o, na_agent._na_in(a, w0))                            # (E, B)
        return q, target

    # Panels are stored (N, K, d_a); the jitted functions take them (K, B, d_a).
    def batched_mixed(fn, row_arrays, panel_arrays):
        outs = []
        for lo in range(0, N, chunk):
            sl = [jnp.asarray(a[lo:lo + chunk]) for a in row_arrays]
            sl += [jnp.asarray(np.swapaxes(a[lo:lo + chunk], 0, 1)) for a in panel_arrays]
            outs.append(jax.device_get(fn(*sl)))
            if (lo // chunk) % 10 == 0:
                print(f'  {fn.__name__ if hasattr(fn, "__name__") else "fn"} rows {lo}/{N}', flush=True)
        return outs

    def cat(outs, i, axis):
        return np.concatenate([np.asarray(o[i]) for o in outs], axis=axis)

    print('measure at (s, u_data)', flush=True)
    o = batched_mixed(measure_at_data, [obs, u_data], [u_index])
    Vm_data_max, Vm_data_mean = cat(o, 0, 0), cat(o, 1, 0)                    # (N,)
    print('measure over the u panel', flush=True)
    o = batched_mixed(measure_panel, [obs], [u_panel, u_index])
    Vm_max, Vm_mean = cat(o, 0, 1).T, cat(o, 1, 1).T                          # (N, K)
    print('scalar over the u panel', flush=True)
    o = batched_mixed(scalar_panel, [obs], [u_panel])
    Qa, Qw = cat(o, 0, 1).T, cat(o, 1, 1).T                                   # (N, K)
    A_inf, A_clipc = cat(o, 2, 1).T, cat(o, 3, 1).T                           # (N, K)
    o = batched_mixed(scalar_at_data, [obs, u_data, a_data], [])
    Qa_data, Qw_data, Qa_adata, dec_err = (cat(o, i, 0) for i in range(4))     # (N,)
    print('measure Bellman arrays', flush=True)
    o = batched_mixed(bellman_measure_arrays, [obs, u_data, next_obs, u_prime, noise_next], [])
    lhs, boot, phi_next = cat(o, 0, 1), cat(o, 1, 1), cat(o, 2, 0)            # (P,N,z),(P,N,z),(N,z)
    u_boot = cat(o, 3, 0)                                                     # (N, d_a)
    actor_arrays = None
    if has_actor:
        print('actor latent', flush=True)
        o = batched_mixed(actor_at_state, [obs], [u_index])
        actor_arrays = {'u_pi': cat(o, 0, 0), 'Qa_pi': cat(o, 1, 0), 'Qw_pi': cat(o, 2, 0),
                        'Vm_pi': cat(o, 3, 0), 'a_inf': cat(o, 4, 0), 'a_clipc': cat(o, 5, 0)}

    # ---- block G arrays ----
    # The index panel: the u' draws (latent), or 64 task vectors per state drawn as
    # `sample_step_inputs` draws them, half project(N(0, I)) and half project(phi(s'_j))
    # at random pool rows (mix_ratio 0.5 taken as exact halves), on the online phi.
    if tv:
        rng_z = np.random.default_rng(PANEL_KEY + 1)
        half = K // 2
        g = rng_z.standard_normal((N, half, z_dim))
        goal = phi_next[rng_z.integers(N, size=(N, K - half))].astype(np.float64)

        def _proj(x):
            return (np.sqrt(z_dim) * x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
                    if bool(mc['norm_z']) else x)
        idx_panel = np.concatenate([_proj(g), _proj(goal)], axis=1).astype(np.float32)   # (N, K, z)
    else:
        idx_panel = u_index
    G_arrays = grid = None
    if affine_grid:
        print('G: readout along the index axis at u_data, and along the u axis at one index', flush=True)
        o = batched_mixed(grid_readout, [obs], [u_data[:, None, :], idx_panel])
        q_idx, unc_idx, qself_idx = cat(o, 0, 2)[0].T, cat(o, 1, 2)[0].T, cat(o, 2, 2)[0].T   # (N, K)
        o = batched_mixed(grid_readout, [obs], [u_panel, idx_panel[:, :1]])
        q_act, unc_act, qself_act = cat(o, 0, 2)[:, 0].T, cat(o, 1, 2)[:, 0].T, cat(o, 2, 2)[:, 0].T
        G_arrays = {'q_idx': q_idx, 'unc_idx': unc_idx, 'qself_idx': qself_idx,
                    'q_act': q_act, 'unc_act': unc_act, 'qself_act': qself_act}
        print(f'G: {n_grid}-state 64x64 grid', flush=True)
        n_grid = min(n_grid, int(is_uniform.sum()))
        gq, gu, gs = [], [], []
        for lo in range(0, n_grid, grid_chunk):
            hi = min(lo + grid_chunk, n_grid)
            r = jax.device_get(grid_readout(jnp.asarray(obs[lo:hi]),
                                            jnp.asarray(np.swapaxes(u_panel[lo:hi], 0, 1)),
                                            jnp.asarray(np.swapaxes(idx_panel[lo:hi], 0, 1))))
            gq.append(np.transpose(np.asarray(r[0]), (2, 0, 1)))                 # (B, Ku, Kidx)
            gu.append(np.transpose(np.asarray(r[1]), (2, 0, 1)))
            gs.append(np.transpose(np.asarray(r[2]), (2, 0, 1)))
        grid = {'q_w': np.concatenate(gq), 'unc': np.concatenate(gu), 'q_self': np.concatenate(gs)}
    else:
        print('G skipped: the grid needs psi_form=affine with psi_bound != tanh', flush=True)
    o = batched_mixed(bellman_scalar_arrays, [obs, a_data, next_obs, r_scaled_file[rows].astype(np.float32),
                                              masks, noise_next], [])
    q_s, target_s = cat(o, 0, 1), cat(o, 1, 0)                                 # (E, N), (N,)

    # ---- r_hat check against the file ----
    r_hat_here = phi_next.astype(np.float64) @ w
    rhat_check = {'max_abs_diff_vs_raw_file': float(np.abs(r_hat_here - r_raw_file[rows]).max()),
                  'corr_vs_raw_file': pearson(r_hat_here, r_raw_file[rows]),
                  'scaled_file_recon_max_abs_diff': float(np.abs(a_scale * r_raw_file[rows] + c_off - r_scaled_file[rows]).max()),
                  'r_hat_mean': float(r_hat_here.mean()), 'r_hat_std': float(r_hat_here.std()),
                  'r_scaled_mean': float(r_scaled_file[rows].mean()), 'r_scaled_std': float(r_scaled_file[rows].std())}

    # ---- block A arrays ----
    A = measure_bellman(lhs, boot, phi_next, w, gamma_m, kappa_td)
    dist = np.linalg.norm(u_panel.astype(np.float64) - u_data[:, None].astype(np.float64), axis=-1)  # (N, K)
    inbox = np.abs(u_panel).max(-1) <= na_u_clip                                                     # (N, K)

    def block(sel):
        """Every number of A-F on the rows selected by the boolean mask `sel`."""
        sel = np.asarray(sel, bool)
        n = int(sel.sum())
        out = {'n_rows': n}
        if n < 3:
            return out
        # A
        lhs_w, rhs_w, rhs_w_mean = A['lhs_w'][sel], A['rhs_w'][sel], A['rhs_w_mean'][sel]
        rres = residual_stats(lhs_w, rhs_w)
        per_member = [residual_stats(A['lhs_w_p'][p][sel], rhs_w)['rms'] for p in range(P)]
        lhs_s = a_scale * lhs_w + c_off / (1 - gamma_m)
        boot_s = a_scale * (rhs_w - A['r_hat'][sel]) / gamma_m + c_off / (1 - gamma_m)
        rhs_s = gamma_m * boot_s + (a_scale * A['r_hat'][sel] + c_off)
        out['A_bellman_measure'] = {
            'along_w_raw_pess_target': rres,
            'along_w_raw_pess_target_rms_per_member': per_member,
            'along_w_raw_mean_target': residual_stats(lhs_w, rhs_w_mean),
            'along_w_scaled': residual_stats(lhs_s, rhs_s),
            'r_hat_raw_mean': float(A['r_hat'][sel].mean()),
            'r_hat_raw_std': float(A['r_hat'][sel].std()),
            'directions': direction_residuals(A['lhs_vec'][sel], A['rhs_vec'][sel], w, n_dirs)}
        # B
        out['B_global'] = {
            'data_u': {f'{vn}_vs_{qn}': agreement(v[sel], q[sel])
                       for vn, v in (('Vm_max', Vm_data_max), ('Vm_mean', Vm_data_mean))
                       for qn, q in (('Qa_decode', Qa_data), ('Qw', Qw_data), ('Qa_recorded_action', Qa_adata))},
            'panel_pooled_raw': {f'{vn}_vs_{qn}': agreement(v[sel].ravel(), q[sel].ravel())
                                 for vn, v in (('Vm_max', Vm_max), ('Vm_mean', Vm_mean))
                                 for qn, q in (('Qa_decode', Qa), ('Qw', Qw))},
            'panel_pooled_state_centred': {f'{vn}_vs_{qn}': agreement(center_rows(v[sel]).ravel(),
                                                                       center_rows(q[sel]).ravel())
                                           for vn, v in (('Vm_max', Vm_max), ('Vm_mean', Vm_mean))
                                           for qn, q in (('Qa_decode', Qa), ('Qw', Qw))},
            'levels': {'Vm_max_data_mean': float(Vm_data_max[sel].mean()),
                       'Vm_max_panel_mean': float(Vm_max[sel].mean()),
                       'Vm_max_panel_within_state_std': float(Vm_max[sel].std(1).mean()),
                       'Qa_decode_data_mean': float(Qa_data[sel].mean()),
                       'Qa_decode_panel_mean': float(Qa[sel].mean()),
                       'Qa_decode_panel_within_state_std': float(Qa[sel].std(1).mean()),
                       'Qw_panel_within_state_std': float(Qw[sel].std(1).mean()),
                       'decode_err_u_data': float(dec_err[sel].mean())}}
        # C
        C = {}
        for vn, v in (('Vm_max', Vm_max), ('Vm_mean', Vm_mean)):
            for qn, q in (('Qa_decode', Qa), ('Qw', Qw)):
                r = per_state_ranking(v[sel], q[sel], topk)
                C[f'{vn}_vs_{qn}'] = {
                    'rho': summarize_rho(r['rho']),
                    f'argmax_in_top{topk}_frac': float(r['hit_topk'].mean()),
                    f'argmax_in_top{topk}_random': float(topk / K),
                    'regret_norm_mean': float(r['regret_norm'].mean()),
                    'regret_norm_median': float(np.median(r['regret_norm'])),
                    'regret_random_norm_mean': float(r['regret_random_norm'].mean()),
                    'inbox_rho': summarize_rho(subset_ranking(v[sel], q[sel], inbox[sel]))}
        C['Qa_decode_vs_Qw'] = {'rho': summarize_rho(per_state_ranking(Qa[sel], Qw[sel], topk)['rho'])}
        C['inbox_frac_of_panel'] = float(inbox[sel].mean())
        out['C_per_state'] = C
        # E
        E = {}
        for vn, v in (('Vm_max', Vm_max),):
            for qn, q in (('Qa_decode', Qa), ('Qw', Qw)):
                near, far = near_far_split(v[sel], q[sel], dist[sel])
                r = per_state_ranking(v[sel], q[sel], topk)
                d_arg = dist[sel][np.arange(int(sel.sum())), r['argmax']]
                med = np.median(dist[sel], axis=1)
                arg_near = d_arg <= med
                E[f'{vn}_vs_{qn}'] = {
                    'rho_near_half': summarize_rho(near), 'rho_far_half': summarize_rho(far),
                    'dist_bins': dist_bins_pooled(v[sel], q[sel], dist[sel], 4),
                    'argmax_dist_to_u_data_mean': float(d_arg.mean()),
                    'panel_dist_to_u_data_mean': float(dist[sel].mean()),
                    'argmax_near_frac': float(arg_near.mean()),
                    'regret_norm_when_argmax_near': float(r['regret_norm'][arg_near].mean()) if arg_near.any() else None,
                    'regret_norm_when_argmax_far': float(r['regret_norm'][~arg_near].mean()) if (~arg_near).any() else None}
        out['E_coverage'] = E
        # G: policy-slot information (per-state axis stds; the grid is top-level)
        if G_arrays is not None:
            ga = G_arrays
            s_idx, s_act = ga['q_idx'][sel].std(1), ga['q_act'][sel].std(1)
            u_idx, u_act = ga['unc_idx'][sel].mean(1), ga['unc_act'][sel].mean(1)
            G = {'std_across_index_at_u_data': summarize_dist(s_idx),
                 'std_across_u_at_fixed_index': summarize_dist(s_act),
                 'ratio_index_over_action_pooled': float(s_idx.mean() / (s_act.mean() + 1e-12)),
                 'ratio_index_over_action_per_state_median': float(np.median(s_idx / (s_act + 1e-12))),
                 'unc_along_index_mean': float(u_idx.mean()), 'unc_along_u_mean': float(u_act.mean()),
                 'std_index_over_unc_index': float(s_idx.mean() / (u_idx.mean() + 1e-12)),
                 'std_u_over_unc_u': float(s_act.mean() / (u_act.mean() + 1e-12)),
                 'index_panel': ('64 task vectors z per state: 32 project(N(0,I)) + 32 project(phi(s\'_j)), '
                                 'read through the fixed eval w' if tv else '64 prior draws u\' (index_clip)')}
            if tv:
                ss_idx = ga['qself_idx'][sel].std(1)
                ss_act = ga['qself_act'][sel].std(1)
                G['self_readout_psiTz'] = {
                    'std_across_z_at_u_data': summarize_dist(ss_idx),
                    'std_across_u_at_z_eq_first': summarize_dist(ss_act),
                    'ratio_index_over_action_pooled': float(ss_idx.mean() / (ss_act.mean() + 1e-12))}
            out['G_policy_slot'] = G
        # H: box exploitation
        n2 = np.linalg.norm(u_panel[sel].astype(np.float64), axis=-1)
        ninf = np.abs(u_panel[sel].astype(np.float64)).max(-1)
        vals = {'Vm_z': center_rows(Vm_max[sel]), 'Qa_z': center_rows(Qa[sel]), 'Qw_z': center_rows(Qw[sel]),
                'a_inf': A_inf[sel], 'a_clip_any': (A_inf[sel] > 0.999).astype(np.float64),
                'a_clip_comp': A_clipc[sel]}
        arg_m, arg_q = Vm_max[sel].argmax(1), Qa[sel].argmax(1)
        ar = np.arange(n)
        H = {}
        for nm_name, nm in (('l2', n2), ('linf', ninf)):
            edges, rows_ = norm_bin_table(nm, vals)
            H[nm_name] = {'edges': edges, 'bins': rows_,
                          'measure_argmax_in_top_quartile': top_bin_frac(nm, arg_m, edges[-2]),
                          'scalar_argmax_in_top_quartile': top_bin_frac(nm, arg_q, edges[-2]),
                          'random_in_top_quartile': 0.25,
                          'measure_argmax_norm_mean': float(nm[ar, arg_m].mean()),
                          'scalar_argmax_norm_mean': float(nm[ar, arg_q].mean()),
                          'panel_norm_mean': float(nm.mean())}
        if actor_arrays is not None:
            aa = actor_arrays
            up_ = aa['u_pi'][sel].astype(np.float64)
            p2, pinf = np.linalg.norm(up_, axis=-1), np.abs(up_).max(-1)
            e2, einf = H['l2']['edges'], H['linf']['edges']

            def _hist(x, e):
                return [float(((x >= e[i]) & ((x <= e[i + 1]) if i == 3 else (x < e[i + 1]))).mean())
                        for i in range(4)]
            H['actor'] = {
                'u_pi_l2': summarize_dist(p2), 'u_pi_linf': summarize_dist(pinf),
                'u_pi_l2_bin_frac': _hist(p2, e2), 'u_pi_linf_bin_frac': _hist(pinf, einf),
                'u_pi_in_top_l2_quartile': float((p2 >= e2[-2]).mean()),
                'u_pi_at_u_clip_comp_frac': float((np.abs(up_) >= u_clip - 1e-6).mean()),
                'a_inf_mean': float(aa['a_inf'][sel].mean()),
                'a_clip_any_frac': float((aa['a_inf'][sel] > 0.999).mean()),
                'a_clip_comp_frac': float(aa['a_clipc'][sel].mean()),
                'Vm_z_at_actor_mean': float(((aa['Vm_pi'][sel] - Vm_max[sel].mean(1))
                                             / (Vm_max[sel].std(1) + 1e-12)).mean()),
                'Qa_z_at_actor_mean': float(((aa['Qa_pi'][sel] - Qa[sel].mean(1))
                                             / (Qa[sel].std(1) + 1e-12)).mean()),
                'Qw_z_at_actor_mean': float(((aa['Qw_pi'][sel] - Qw[sel].mean(1))
                                             / (Qw[sel].std(1) + 1e-12)).mean()),
                'panel_a_inf_mean': float(A_inf[sel].mean()),
                'panel_a_clip_any_frac': float((A_inf[sel] > 0.999).mean())}
        out['H_box'] = H
        # actor (runs with a trained latent actor only)
        if actor_arrays is not None:
            aa = actor_arrays
            act = {'vs_Qa_decode': actor_regret(Vm_max[sel], Qa[sel], aa['Vm_pi'][sel], aa['Qa_pi'][sel], topk),
                   'vs_Qw': actor_regret(Vm_max[sel], Qw[sel], aa['Vm_pi'][sel], aa['Qw_pi'][sel], topk)['scalar'],
                   'u_pi_norm_mean': float(np.linalg.norm(aa['u_pi'][sel], axis=-1).mean()),
                   'u_pi_dist_to_u_data_mean': float(np.linalg.norm(aa['u_pi'][sel] - u_data[sel], axis=-1).mean()),
                   'u_pi_clipfrac': float((np.abs(aa['u_pi'][sel]) >= u_clip - 1e-6).mean()),
                   'Qa_at_actor_vs_Qa_at_u_data': agreement(aa['Qa_pi'][sel], Qa_data[sel]),
                   'Vm_at_actor_vs_Qa_at_actor': agreement(aa['Vm_pi'][sel], aa['Qa_pi'][sel])}
            out['actor'] = act
        # F
        q_min = q_s[:, sel].min(0)
        out['F_sanity'] = {
            'scalar_bellman_min_ensemble': residual_stats(q_min, target_s[sel]),
            'scalar_bellman_per_member_rms': [residual_stats(q_s[e][sel], target_s[sel])['rms'] for e in range(q_s.shape[0])],
            'scalar_q_mean': float(q_min.mean()), 'scalar_target_mean': float(target_s[sel].mean()),
            'qa_decode_vs_qw_pooled': agreement(Qa[sel].ravel(), Qw[sel].ravel()),
            'qa_decode_vs_qw_state_centred': agreement(center_rows(Qa[sel]).ravel(), center_rows(Qw[sel]).ravel()),
            'qa_decode_vs_qw_per_state_rho': summarize_rho(per_state_ranking(Qa[sel], Qw[sel], topk)['rho']),
            'qa_decode_vs_qw_data_u': agreement(Qa_data[sel], Qw_data[sel])}
        return out

    splits = {'all': is_uniform, 'success': is_success, 'nonsuccess': is_uniform & ~is_success}
    G_grid = None
    if grid is not None:
        G_grid = {'n_states': int(grid['q_w'].shape[0]), 'states': 'the first n_grid uniform pool rows',
                  'decomposition_psiTw': two_way_decomposition(grid['q_w']),
                  'rank_corr_across_u_between_indices': pairwise_rank_corr(grid['q_w'], 2),
                  'rank_corr_across_indices_between_u': pairwise_rank_corr(grid['q_w'], 1),
                  'unc_grid_mean': float(grid['unc'].mean()),
                  'unc_decomposition': two_way_decomposition(grid['unc'])}
        if tv:
            G_grid['decomposition_psiTz_self_readout'] = two_way_decomposition(grid['q_self'])
            G_grid['rank_corr_across_u_between_z_self_readout'] = pairwise_rank_corr(grid['q_self'], 2)
    report = {
        'probe': 'measure readout psi^T w vs scalar TD critic on the same inferred reward',
        'env': cfg.env_name,
        'measure': {'restore_path': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch),
                    'psi_form': mc['psi_form'], 'policy_index': mc['policy_index'],
                    'train_actor': has_actor, 'actor_mode': mc['actor_mode'], 'acting': mc['acting'],
                    'index_slot': 'inferred w' if tv else "prior draw u' (K-panel)",
                    'bellman_bootstrap': ('actor sample pi_eta(s\', w, eps), eps from the panel key'
                                          if tv else "u' as both index and continuation"),
                    'bootstrap_latent_norm_mean': float(np.linalg.norm(u_boot, axis=-1).mean()),
                    'discount': gamma_m, 'num_parallel': P, 'pessimism_penalty': kappa_td,
                    'actor_pessimism_penalty': kappa_act, 'u_clip': u_clip, 'index_clip': ic,
                    'gpi_decode': mc['gpi_decode'], 'agent_config_source': prov},
        'scalar': {'restore_path': na_path, 'restore_epoch': na_epoch,
                   'dsrl_na_discount': gamma_na, 'u_clip': na_u_clip,
                   'num_ensembles': int(nc['dsrl_na']['num_ensembles']),
                   'gpi_decode': nc['gpi_decode'], 'reward_override': na_reward_override,
                   'bootstrap_latent': 'sac_actor sample (training target), not the mode',
                   'agent_config_source': na_prov},
        'reward': {'raw_path': reward_raw_path, 'scaled_path': reward_scaled_path,
                   'affine_scale': a_scale, 'affine_offset': b_offset, 'shift_back': shift_back,
                   'scaled_definition': 'r_scaled = affine_scale * r_hat + affine_offset - shift_back',
                   'w_cos_file_vs_reinferred': w_cos, 'w_norm': float(np.linalg.norm(w)),
                   'w_source': ('raw file sidecar (the w the scalar critic trained on)' if w_source == 'file'
                                else 'this checkpoint\'s own eval w on its own phi (infer_z, 10k rows, shift 1.0); '
                                     'the scalar critic\'s reward is the file\'s'),
                   'r_hat_check': rhat_check},
        'data': {'n_rows_uniform': int(is_uniform.sum()), 'n_rows_pool': N,
                 'n_success_pool': int(is_success.sum()),
                 'n_success_uniform': int((is_success & is_uniform).sum()),
                 'success_rule': 'dataset reward == max (0 in the -1/0 convention) at s\'',
                 'row_seed': ROW_SEED, 'panel_key': PANEL_KEY, 'dir_seed': DIR_SEED,
                 'K': K, 'panel_clip': u_clip, 'index_clip': ic, 'topk': topk,
                 'preimage_path': str(mc['preimage_path']),
                 'n_invalid_preimages_excluded': int((valid < 0.5).sum())},
        'definitions': {
            'Vm_max': ('[mean_P - kappa_act*unc] psi(s,u,w)^T w, the index slot carrying the inferred w' if tv else
                       'max over K policy indices u\' of [mean_P - kappa_act*unc] psi(s,u,u\')^T w (the GPI value)'),
            'Vm_mean': ('identical to Vm_max (no index panel under task_vector)' if tv else
                        'mean over the K indices of the same'),
            'actor': 'regret of the actor MODE latent pi_eta(s,w,0) against the 64-panel, per C\'s normalisation; '
                     'measure = the measure\'s own regret at that latent',
            'Qa_decode': 'min over the qa ensemble of qa(s, G(s,u)), G the agent\'s decode',
            'Qw': 'the distilled latent critic qw(s,u)',
            'A.lhs': ('mean_P psi(s,u_data,w)^T w, online params' if tv else
                      'mean_P psi(s,u_data,u\')^T w, online params'),
            'A.rhs_pess': ('gamma*[mean_P - kappa_td*unc](psi_bar(s\',pi_eta(s\',w,eps),w)^T w) + phi(s\')^T w' if tv else
                           'gamma*[mean_P - kappa_td*unc](psi_bar(s\',u\',u\')^T w) + phi(s\')^T w, target psi, online phi'),
            'A.directions': 'full z-dim residual with the per-component pessimistic target, projected on unit w vs random unit directions; rel = RMS(res.d)/std(lhs.d)',
            'A.scaled': 'both sides mapped through the affine reward map; the relative residual is the raw one by construction',
            'C.regret_norm': '(max_u Qs - Qs(argmax_u Vm)) / (max_u Qs - min_u Qs) per state',
            'E.near_half': 'per state, the K/2 panel latents closest to u_data(s) in L2',
            'F.scalar_bellman': 'min_e qa(s,a_data) - [r_scaled(s\') + gamma_na*mask*min_e qa_bar(s\', G(s\', pi(s\')))]',
            'G.axes': 'index axis: readout at u_data over the 64-index panel; u axis: readout over the 64 panel '
                      'latents at the first index of the panel; std within state, pooled over states',
            'G.grid': 'two-way split of the within-state variance of [mean-kappa*unc] psi(s,u_i,idx_k)^T w on a '
                      '64x64 grid; rank_corr_across_u_between_indices = mean pairwise Spearman of the 64 index '
                      'columns, each ranking the 64 u',
            'H.bins': 'global quartiles of the panel latent norm; Vm_z/Qa_z/Qw_z are state-z-scored; a_clip_any = '
                      'max_j |G(s,u)_j| > 0.999; a_clip_comp = fraction of components > 0.999',
            'H.actor': 'the actor MODE latent against the same panel and bins'},
        'G_grid': G_grid,
        'splits': {name: block(sel) for name, sel in splits.items()},
    }
    out = write_report(report, cfg, 'diag_measure_vs_scalar_q.json')
    npz = str(out)[:-5] + '.npz' if str(out).endswith('.json') else str(out) + '.npz'
    np.savez_compressed(npz, rows=rows, is_uniform=is_uniform, is_success=is_success,
                        Vm_max=Vm_max.astype(np.float32), Vm_mean=Vm_mean.astype(np.float32),
                        Qa=Qa.astype(np.float32), Qw=Qw.astype(np.float32),
                        Vm_data_max=Vm_data_max, Qa_data=Qa_data, Qw_data=Qw_data,
                        lhs_w=A['lhs_w'], rhs_w=A['rhs_w'], r_hat=A['r_hat'],
                        dist=dist.astype(np.float32), u_panel=u_panel, u_data=u_data,
                        q_scalar=q_s, target_scalar=target_s, w=w, u_boot=u_boot,
                        A_inf=A_inf.astype(np.float32), A_clipc=A_clipc.astype(np.float32),
                        **({f'actor_{k}': v for k, v in actor_arrays.items()} if actor_arrays else {}),
                        **({f'G_{k}': v.astype(np.float32) for k, v in G_arrays.items()} if G_arrays else {}),
                        **({f'grid_{k}': v.astype(np.float32) for k, v in grid.items()} if grid else {}))
    print(f'npz -> {npz}')
    for name in splits:
        b = report['splits'][name]
        if 'A_bellman_measure' not in b:
            continue
        a_ = b['A_bellman_measure']['along_w_raw_pess_target']
        d_ = b['A_bellman_measure']['directions']
        c_ = b['C_per_state']['Vm_max_vs_Qa_decode']
        print(f"[{name}] n={b['n_rows']}  A rel-RMS along w {a_['rms_over_std_lhs']:.3f} "
              f"(unit-w {d_['along_w_rel']:.3f} vs random dirs {d_['random_dirs_rel_mean']:.3f})  "
              f"B data-u Spearman Vm_max/Qa {b['B_global']['data_u']['Vm_max_vs_Qa_decode']['spearman']:+.3f}  "
              f"C rho mean {c_['rho']['mean']:+.3f} top{topk} {c_[f'argmax_in_top{topk}_frac']:.3f} "
              f"regret {c_['regret_norm_mean']:.3f} (random {c_['regret_random_norm_mean']:.3f})  "
              f"F scalar rel-RMS {b['F_sanity']['scalar_bellman_min_ensemble']['rms_over_std_lhs']:.3f}"
              + (f"  actor regret scalar {b['actor']['vs_Qa_decode']['scalar']['regret_norm_mean']:.3f} "
                 f"measure {b['actor']['vs_Qa_decode']['measure']['regret_norm_mean']:.3f}" if 'actor' in b else '')
              + (f"  G index/u std ratio {b['G_policy_slot']['ratio_index_over_action_pooled']:.3f}"
                 if 'G_policy_slot' in b else '')
              + f"  H measure-argmax top-l2-quartile {b['H_box']['l2']['measure_argmax_in_top_quartile']:.3f} "
                f"scalar {b['H_box']['l2']['scalar_argmax_in_top_quartile']:.3f}")
    if G_grid is not None:
        d_ = G_grid['decomposition_psiTw']
        print(f"G grid ({G_grid['n_states']} states): var frac u {d_['frac_u_mean']:.3f} index "
              f"{d_['frac_index_mean']:.3f} interaction {d_['frac_interaction_mean']:.3f}; rank corr across u "
              f"between indices {G_grid['rank_corr_across_u_between_indices']['mean']:.3f}")


def main():
    import hydra
    hydra.main(version_base=None, config_path='../configs', config_name='config')(_main)()


if __name__ == '__main__':
    main()
