"""Does the critic ensemble disagree more on TD targets whose policy index u' is far from the data?

The measure loss (`agents/psmflow.py::measure_loss`) bootstraps through

    M_boot[p, i, j] = psibar_p(s'_i, u'_i, u^+_i)^T phibar(s'_j),      p = 1..P

with u'_i ~ p0 = N(0, I) clipped to the u box and, under policy_index='latent', u^+ = u'.
The target is mean_p M_boot - kappa * spread_p M_boot. The spread is the only pessimism
the algorithm has, so the question this probe answers is whether that spread is LARGER
where the index u' sits far from the latents the data actually occupies (the preimages
u_data of recorded actions) -- i.e. whether the ensemble's disagreement is a distance-
to-data signal at all, or just noise.

Per batch row i (one dataset transition), with eps a small constant:

    rel_disagreement_i = mean_j |M_1[i,j] - M_2[i,j]| / (mean_j |mean_p M_p[i,j]| + eps)
    rel_diag_i         = the same restricted to the diagonal entry j = i
    norm_index_i       = ||u'_i||_2                 (the policy index the loss drew)
    norm_udata_i       = ||u_i||_2                  (the transition's own preimage)
    dist_to_data_i     = min_k ||u'_i - u_data[k]||  over a random 20k-row subset of the
                         dataset's preimage latents (nearest-neighbour distance to data)

The ensemble spread is `targets_uncertainty`'s mean pairwise |M_p - M_q|, which for P=2
is exactly |M_1 - M_2| -- the same quantity the pessimism term subtracts.

Two conditions per checkpoint, same batches, same target networks:

    'prior'      u' drawn exactly as `sample_step_inputs` draws it (the training target)
    'insupport'  u' replaced by the row's own data preimage u_i (control: an index that is
                 by construction on the data manifold; its dist_to_data excludes itself)

Rows are pooled over >= 8 batches of 1024, binned into quintiles of ||u'|| and of
dist_to_data, and Spearman rank correlations of rel_disagreement against ||u'||,
dist_to_data and ||u|| are reported.

Read-only: no parameter is updated. The agent config is inherited from the run's own
flags.json via `tools.eval_checkpoint.merge_run_config`, so the restored networks are the
ones that produced the run's numbers.

Run (see scripts/slurm/diag_ensemble_disagreement.sbatch):
  MUJOCO_GL=egl .venv/bin/python tools/diag_ensemble_disagreement.py agent=psmflow \
      env_name=antmaze-medium-navigate-singletask-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/antmaze-medium-navigate agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/antmaze-medium-navigate.npz \
      restore_path=<run_dir> restore_epoch=100000 \
      +n_batches=8 +batch_size=1024 +knn_subset=20000 \
      report_out=$PSM_DATA/logs/diag_disagreement/<name>.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py)

EPS = 1e-6
N_BINS = 5


# ---------------------------------------------------------------- pure helpers (tested)
def _avg_rank(x):
    """Average ranks (ties share the mean rank), 1-based, float64."""
    x = np.asarray(x, np.float64).ravel()
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(x.size, np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation with tie-averaged ranks. NaN if either side is constant."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    assert a.shape == b.shape
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return float('nan')
    ra, rb = _avg_rank(a) - (a.size + 1) / 2.0, _avg_rank(b) - (b.size + 1) / 2.0
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if den == 0:
        return float('nan')
    return float((ra * rb).sum() / den)


def bin_by_quantiles(x, y, n_bins=N_BINS):
    """Mean of `y` inside `n_bins` quantile bins of `x`.

    Returns a list of dicts {lo, hi, n, mean, median} in ascending `x`. Bins are
    equal-count by construction (quantile edges); the last bin is closed on the right.
    """
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    assert x.shape == y.shape and x.size >= n_bins
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    out = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (x >= lo) & ((x < hi) if k < n_bins - 1 else (x <= hi))
        yy = y[m]
        out.append({'lo': float(lo), 'hi': float(hi), 'n': int(m.sum()),
                    'mean': float(yy.mean()) if yy.size else float('nan'),
                    'median': float(np.median(yy)) if yy.size else float('nan')})
    return out


def summarize_condition(rows):
    """Bin tables + rank correlations for one condition's pooled per-row arrays."""
    rel, diag = rows['rel_disagreement'], rows['rel_diag']
    ni, nu, dist = rows['norm_index'], rows['norm_udata'], rows['dist_to_data']
    return {
        'n_rows': int(rel.size),
        'rel_disagreement_mean': float(rel.mean()),
        'rel_disagreement_median': float(np.median(rel)),
        'rel_diag_mean': float(diag.mean()),
        'abs_disagreement_mean': float(rows['abs_disagreement'].mean()),
        'abs_target_mean': float(rows['abs_target'].mean()),
        'by_norm_index': bin_by_quantiles(ni, rel),
        'by_dist_to_data': bin_by_quantiles(dist, rel),
        'by_norm_udata': bin_by_quantiles(nu, rel),
        'diag_by_dist_to_data': bin_by_quantiles(dist, diag),
        'spearman': {
            'rel_vs_norm_index': spearman(rel, ni),
            'rel_vs_dist_to_data': spearman(rel, dist),
            'rel_vs_norm_udata': spearman(rel, nu),
            'diag_vs_dist_to_data': spearman(diag, dist),
            'norm_index_vs_dist_to_data': spearman(ni, dist),
        },
    }


def _fmt_bins(bins):
    return ' | '.join(f'[{b["lo"]:.2f},{b["hi"]:.2f}] {b["mean"]:.4f}' for b in bins)


def print_table(report):
    for cond in ('prior', 'insupport'):
        s = report['conditions'][cond]
        sp = s['spearman']
        print(f'--- {cond}: n={s["n_rows"]} rel={s["rel_disagreement_mean"]:.4f} '
              f'(median {s["rel_disagreement_median"]:.4f}) diag={s["rel_diag_mean"]:.4f} '
              f'|target|={s["abs_target_mean"]:.3f}')
        print(f'    by ||u\'||     : {_fmt_bins(s["by_norm_index"])}')
        print(f'    by dist2data  : {_fmt_bins(s["by_dist_to_data"])}')
        print(f'    spearman rel~||u\'|| {sp["rel_vs_norm_index"]:+.3f}  rel~dist '
              f'{sp["rel_vs_dist_to_data"]:+.3f}  rel~||u|| {sp["rel_vs_norm_udata"]:+.3f}  '
              f'diag~dist {sp["diag_vs_dist_to_data"]:+.3f}')


# ------------------------------------------------------------------------- the GPU probe
def _main(cfg):
    import jax
    import jax.numpy as jnp
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
    from utils.log_utils import write_report
    from utils.psm_common import targets_uncertainty

    assert cfg.restore_path is not None and cfg.restore_epoch is not None, \
        'needs a trained checkpoint (restore_path, restore_epoch)'
    seed = int(cfg.seed)
    n_batches = int(cfg.get('n_batches', 8))
    batch_size = int(cfg.get('batch_size', 1024))
    knn_subset = int(cfg.get('knn_subset', 20000))
    np.random.seed(seed)

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert config['policy_index'] == 'latent', (
        f"this probe instruments the u' index of policy_index='latent'; got "
        f"{config['policy_index']!r}")

    aug = load_augmented_dataset(config['preimage_path'])
    aug, _ = repair_invalid_preimages(aug)
    ds = Dataset.create(**aug)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(config.get('use_point_preimage', False))

    ex = ds.sample(1)
    agent = agents[config['agent_name']].create(seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, cfg.restore_path, int(cfg.restore_epoch))
    ac = agent.config
    P, u_clip = int(ac['num_parallel']), float(ac['u_clip'])

    # Data latents for the nearest-neighbour distance: a fixed random 20k-row subset of
    # the stored point preimages, clipped to the box the loss clips u_data to.
    gen = np.random.default_rng(seed + 1000)
    key_lat = 'noise_preimage_point' if ds.preimage_point_mode else 'noise_preimage_mean'
    all_lat = np.asarray(aug[key_lat], np.float32)
    if all_lat.ndim == 3:            # mixture means (N, n_clusters, d_a): take the first
        all_lat = all_lat[:, 0]
    sub_idx = gen.choice(all_lat.shape[0], size=min(knn_subset, all_lat.shape[0]), replace=False)
    lat_sub = jnp.clip(jnp.asarray(all_lat[sub_idx]), -u_clip, u_clip)
    lat_sub_sq = jnp.sum(lat_sub ** 2, axis=1)

    @jax.jit
    def knn_dist(q):
        """min_k ||q_i - lat_sub_k||, excluding exact coincidences (the row itself)."""
        d2 = jnp.sum(q ** 2, axis=1)[:, None] + lat_sub_sq[None, :] - 2.0 * q @ lat_sub.T
        d2 = jnp.maximum(d2, 0.0)
        d2 = jnp.where(d2 < 1e-10, jnp.inf, d2)
        return jnp.sqrt(jnp.min(d2, axis=1))

    @jax.jit
    def boot_stats(agent, next_obs, index, u_next):
        """M_boot exactly as measure_loss forms it, reduced to per-row disagreement."""
        target_phi_next = agent.phi(next_obs, params=agent.target_phi)
        M_boot = agent.psi_b(next_obs, index, u_next, params=agent.target_psi) @ target_phi_next.T
        M_mean, M_unc = targets_uncertainty(M_boot, P)          # (B, B) each
        rel = jnp.mean(M_unc, axis=1) / (jnp.mean(jnp.abs(M_mean), axis=1) + EPS)
        d = jnp.diagonal(M_unc) / (jnp.abs(jnp.diagonal(M_mean)) + EPS)
        return {'rel_disagreement': rel, 'rel_diag': d,
                'abs_disagreement': jnp.mean(M_unc, axis=1),
                'abs_target': jnp.mean(jnp.abs(M_mean), axis=1)}

    keys = ('rel_disagreement', 'rel_diag', 'abs_disagreement', 'abs_target',
            'norm_index', 'norm_udata', 'dist_to_data')
    pooled = {c: {k: [] for k in keys} for c in ('prior', 'insupport')}
    rng = jax.random.PRNGKey(seed)
    for b in range(n_batches):
        batch = ds.sample(batch_size)
        sampled = agent.sample_step_inputs(batch, jax.random.fold_in(rng, b))
        u_data, u_index = sampled.u_data, sampled.u_index
        next_obs = jnp.asarray(batch['next_observations'])
        nu = jnp.linalg.norm(u_data, axis=1)
        # (index, u_next) pairs: the loss's own draw, and the in-support control
        for cond, idx in (('prior', u_index), ('insupport', u_data)):
            st = boot_stats(agent, next_obs, idx, idx)
            st['norm_index'] = jnp.linalg.norm(idx, axis=1)
            st['norm_udata'] = nu
            st['dist_to_data'] = knn_dist(idx)
            for k in keys:
                pooled[cond][k].append(np.asarray(jax.device_get(st[k]), np.float64))
        print(f'batch {b + 1}/{n_batches} done', flush=True)

    conditions = {}
    for cond, acc in pooled.items():
        rows = {k: np.concatenate(v) for k, v in acc.items()}
        conditions[cond] = summarize_condition(rows)

    report = {
        'probe': 'critic-ensemble disagreement on the measure TD target vs distance of the '
                 "policy index u' from the data latents",
        'env': cfg.env_name, 'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch), 'seed': seed,
        'n_batches': n_batches, 'batch_size': batch_size, 'knn_subset': int(lat_sub.shape[0]),
        'num_parallel': P, 'u_clip': u_clip, 'eps': EPS,
        'discount': float(ac['discount']), 'pessimism_penalty': float(ac['pessimism_penalty']),
        'psi_bound': str(ac.get('psi_bound', 'none')), 'ortho_mode': str(ac.get('ortho_mode', 'fixed')),
        'psi_form': config.get('psi_form'), 'policy_index': config.get('policy_index'),
        'agent_config_source': prov,
        'definitions': {
            'rel_disagreement': 'mean_j spread_p M_boot[p,i,j] / (mean_j |mean_p M_boot[p,i,j]| + eps); '
                                'spread = targets_uncertainty = |M1 - M2| for P=2',
            'rel_diag': 'the same at j = i',
            'dist_to_data': "min over a random knn_subset of clipped data preimages of ||u'_i - u_k||, "
                            'excluding exact coincidences',
            'prior': "u' ~ N(0,I) clipped to u_clip, as sample_step_inputs draws it; u_next = u'",
            'insupport': "u' := u_data_i (the row's own preimage); u_next = u_data_i",
        },
        'conditions': conditions,
    }
    print_table(report)
    write_report(report, cfg, 'diag_ensemble_disagreement.json')


def main():
    import hydra
    hydra.main(version_base=None, config_path='../configs', config_name='config')(_main)()


if __name__ == '__main__':
    main()
