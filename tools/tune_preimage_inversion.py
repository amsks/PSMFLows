"""Tune the inversion hyperparameters on one sampled batch, scored the way the tests score.

Inverting the full buffer costs 4-19 h, so it is the wrong loop to tune in. This inverts a
single sampled batch (a few thousand rows, minutes) once per setting and reports, per
setting, what decides whether the latents are usable:

  coverage     draw u ~ p0 at a state and ask whether it is captured under ANY mixture
               component of that state or a related one (inside the component's 95%
               ellipsoid). This is the densification claim stated as a test: a latent the
               actor could emit at deployment must have been trained somewhere.
               `covered_at_k` is the curve over how many related states may supply it,
               k=0 being the row's own mixture; `covered_by_prior_quantile` says WHERE the
               misses are, since a miss beyond u_clip costs nothing while a miss in the
               bulk is fatal. Coverage is trivially maximised by inflating the
               covariances, so it is only readable against the decode error below: the
               useful output is the frontier of the two, and the setting to pick is the
               widest one still inside a decode-error budget.
  typicality   mean ||u||^2 of a mixture draw against E[chi^2_{d_a}], the KS statistic
               against that CDF, and the fraction past its 99th percentile. Assumption A2
               (Lemma 6.1) says an exact flow's preimages are distributed as the prior, so
               this is the falsifiable version of "the latents seen at train time look like
               the ones drawn at test time".
  decode       ||G(s, u) - a|| for a mixture draw (`decode_mix`) and for the exact
               backward-ODE latent (`decode_pt`, the floor, unaffected by these knobs).
  nan          fraction of mixture draws whose decode diverges -- latents so far off the
               flow's training support that the forward integration blows up.
  ess          effective sample size of the final EM iterate, the D3 health scalar.

Sweeps one knob at a time from the `inversion` config baseline:

  MUJOCO_GL=egl .venv/bin/python tools/tune_preimage_inversion.py agent=fql \
      env_name=cube-single-play-singletask-v0 \
      restore_path=$PSM_DATA/flow/cube-single-play restore_epoch=500000 \
      agent.flow_steps=100 +n_rows=2000 \
      '+sweep={prior_scale: [0.0, 0.5, 1.0], alpha: [5.0, 20.0, 50.0]}'

The batch is sampled as whole neighbourhoods by default (`+sample_mode=uniform` for
independent rows, `+sample_k` rows per anchor): coverage asks what NEARBY transitions
share, and a uniform sample of a 1M-row buffer has no neighbourhoods left to ask about.

Every combination of a swept knob is run against the SAME batch and the same draw seed, so
differences between rows are the setting and nothing else. With no `+sweep` it scores the
configured baseline alone. Writes a JSON with one record per setting.

To then run the full recovery tests on a promising setting, persist that batch with
tools/precompute_preimages.py `+preimage_sample=<n>` (which records `source_index`, so the
dynamics test can still find the simulator state) and point the tools at the npz.
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset, get_size
from utils.flax_utils import restore_agent
from utils.flow_inversion import (
    augment_dataset_with_point_preimage,
    augment_dataset_with_preimage_distribution,
    compute_preimage_validity,
    sample_preimage_noise,
)
from utils.log_utils import write_report
from utils.nn_search import knn_indices, neighborhood_sample
from utils.preimage_eval import decode_in_batches


def _typicality(sq_norms, d_a):
    """||u||^2 against chi^2_{d_a}: what A2 predicts if the flow is exact."""
    out = {'mean_sq_norm': round(float(np.mean(sq_norms)), 4), 'expected_mean': float(d_a)}
    try:
        from scipy.stats import chi2, kstest
        out['frac_beyond_chi2_99'] = round(float(np.mean(sq_norms > chi2.ppf(0.99, d_a))), 4)
        out['ks_stat'] = round(float(kstest(sq_norms, lambda v: chi2.cdf(v, d_a)).statistic), 4)
    except ImportError:
        out['note'] = 'scipy unavailable; KS/quantile diagnostics skipped'
    return out


def _coverage(means, covs, weights, obs, d_a, n_prior=8, k=16, seed=0):
    """Is a prior draw captured under any mixture component of this or a related state?

    The densification claim is that a latent the actor could emit at deployment was
    trained somewhere. So: draw u ~ p0 at a state, and test u against every component of
    that state's mixture and of its k nearest states', counting it covered if it falls
    inside any component's 95% ellipsoid.

    Returned as a curve over k (k=0 being the state's own mixture) rather than one number,
    because how far the measure generalises across states is not pinned down; plus the
    breakdown over prior quantiles, since a miss beyond u_clip costs nothing while a miss
    in the bulk is the failure this is meant to catch.

    Note the degenerate direction: coverage is trivially 1.0 for wide enough covariances,
    so it is only meaningful read against the decode error that width costs.
    """
    rng = np.random.default_rng(seed)
    n = len(obs)
    u = rng.standard_normal((n, n_prior, d_a)).astype(np.float32)

    scale = np.maximum(np.asarray(obs, np.float32).std(0), 1e-6)
    kk = min(k + 1, n)
    nb_idx, _ = knn_indices(obs, obs, kk, scale=scale)      # column 0 is the row itself

    try:
        from scipy.stats import chi2
        thresh = float(chi2.ppf(0.95, d_a))
    except ImportError:
        thresh = float(d_a + 3.0 * np.sqrt(2 * d_a))

    # Mahalanobis of every draw under every component of every neighbour's mixture.
    # Near-singular EM covariances (ESS-collapsed rows) can fail Cholesky at float
    # tolerance; escalate a diagonal jitter far below any covariance scale that matters.
    covs64 = covs.astype(np.float64)
    covs64 = 0.5 * (covs64 + np.swapaxes(covs64, -1, -2))
    jitter = 1e-9
    while True:
        try:
            chol = np.linalg.cholesky(covs64 + jitter * np.eye(d_a))  # (n, K, d, d)
            break
        except np.linalg.LinAlgError:
            if jitter > 1e-3:
                raise
            jitter *= 100.0
    inside = np.zeros((n, n_prior, kk), bool)
    for j in range(kk):
        nb = nb_idx[:, j]
        delta = u[:, :, None, :] - means[nb][:, None, :, :]  # (n, n_prior, K, d)
        sol = np.linalg.solve(chol[nb][:, None], delta[..., None])[..., 0]
        maha = (sol ** 2).sum(-1)                            # (n, n_prior, K)
        # A component with negligible weight is not part of the region it would cover.
        alive = weights[nb][:, None, :] > 1e-3
        inside[:, :, j] = ((maha <= thresh) & alive).any(-1)

    count_knn = inside[:, :, 1:].sum(-1)
    # Coverage as a function of how many related states are allowed to supply it -- "this
    # OR a related state", so k=0 is the row's own mixture alone. A single number would be
    # a number at one arbitrary k: how far the measure generalises across states is not
    # pinned down, so report the curve and let the reader pick where to stand on it.
    # Cumulative over the neighbour axis, so the whole curve is free.
    covered_at_k = {'0': round(float(inside[:, :, 0].mean()), 4)}
    j = 1
    while j < kk:
        covered_at_k[str(j)] = round(float(inside[:, :, :j + 1].any(-1).mean()), 4)
        j *= 2
    covered_at_k[str(kk - 1)] = round(float(inside.any(-1).mean()), 4)

    # WHERE in the prior the misses are. A miss at ||u|| > u_clip costs nothing -- psmflow
    # clamps every latent draw to that box before it decodes one -- while a miss in the
    # bulk is the failure this metric exists to catch.
    sq = (u.astype(np.float64) ** 2).sum(-1)
    covered_any = inside.any(-1)
    by_radius = {}
    try:
        from scipy.stats import chi2
        edges = [0.0] + [float(chi2.ppf(q, d_a)) for q in (0.25, 0.5, 0.75, 0.95)] + [np.inf]
        labels = ['q0-25', 'q25-50', 'q50-75', 'q75-95', 'q95-100']
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            m = (sq >= lo) & (sq < hi)
            by_radius[lab] = round(float(covered_any[m].mean()), 4) if m.any() else None
    except ImportError:
        by_radius['note'] = 'scipy unavailable; radial breakdown skipped'

    return {
        'covered_at_k': covered_at_k,
        'covered_by_prior_quantile': by_radius,
        # How redundantly, rather than whether: the same measurement read as density.
        'mean_count_knn': round(float(count_knn.mean()), 3),
        'k_max': int(kk - 1),
        'n_prior_draws': int(n_prior),
    }


def _score(agent, batch, inv, batch_size, draw_seed=0, cov_k=16, cov_draws=8):
    """Invert `batch` under `inv`, then score the latents it produces."""
    t0 = time.time()
    out = augment_dataset_with_preimage_distribution(agent, dict(batch), inv)
    out = augment_dataset_with_point_preimage(agent, out, inv)
    elapsed = time.time() - t0

    valid = compute_preimage_validity(out) >= 0.5
    obs = np.asarray(batch['observations'])[valid]
    act = np.clip(np.asarray(batch['actions'])[valid], -1, 1)
    d_a = act.shape[-1]

    # One draw per row from the fitted mixture, with a seed shared across settings.
    rng = np.random.default_rng(draw_seed)
    u = sample_preimage_noise(out['noise_preimage_mean'][valid], out['noise_preimage_cov'][valid],
                              out['noise_preimage_weights'][valid], rng=rng)
    recon = decode_in_batches(agent, obs, u, batch_size)
    finite = np.isfinite(recon).all(-1)
    decode_mix = np.linalg.norm((recon - act)[finite], axis=-1)
    ess = np.asarray(out['preimage_ess'])[valid]

    return {
        'n_rows': int(len(obs)),
        'invalid_rows': int((~valid).sum()),
        'seconds': round(elapsed, 1),
        'coverage': _coverage(out['noise_preimage_mean'][valid], out['noise_preimage_cov'][valid],
                              out['noise_preimage_weights'][valid], obs, d_a,
                              n_prior=cov_draws, k=cov_k, seed=draw_seed),
        'typicality': _typicality((u.astype(np.float64) ** 2).sum(-1), d_a),
        'nan_decode_frac': round(float(1.0 - finite.mean()), 6),
        'decode_mix_mean': round(float(decode_mix.mean()), 6) if finite.any() else None,
        'decode_pt_mean': round(float(np.asarray(out['preimage_roundtrip'])[valid].mean()), 6),
        'ess_mean': round(float(ess.mean()), 2),
        'ess_frac_above_20': round(float((ess > 20).mean()), 4),
    }


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow shapes)'
    n_rows = int(cfg.get('n_rows', 2000))

    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = dict(Dataset.create(**train_dataset))
    size = get_size(ds)
    # The same batch for every setting: differences between rows of the table must be the
    # setting, not the sample. Seeded off cfg.seed, and the same rows
    # tools/precompute_preimages.py `+preimage_sample` would take at that seed.
    sample_mode = str(cfg.get('sample_mode', 'neighborhoods'))
    sample_k = int(cfg.get('sample_k', 20))
    if sample_mode == 'neighborhoods':
        # Coverage is a statement about what NEARBY transitions share, so the batch has to
        # have neighbourhoods; a uniform sample of a 1M-row buffer does not.
        obs_all = np.asarray(ds['observations'], np.float32)
        rows = neighborhood_sample(obs_all, n_rows, sample_k, seed=int(cfg.seed),
                                   scale=np.maximum(obs_all.std(0), 1e-6))
    elif sample_mode == 'uniform':
        rows = np.sort(np.random.default_rng(int(cfg.seed)).choice(size, n_rows, replace=False))
    else:
        raise ValueError(f'+sample_mode must be neighborhoods|uniform, got {sample_mode!r}')
    batch = {k: v[rows] for k, v in ds.items() if k in ('observations', 'actions')}

    agent_cfg = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(int(cfg.seed), batch['observations'][:1], batch['actions'][:1], agent_cfg)
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    else:
        assert cfg.inversion.get('allow_untrained', False), (
            'tuning against an UNTRAINED flow measures nothing; set restore_path=<ckpt dir> '
            '(or inversion.allow_untrained=true for a plumbing smoke)')
    assert int(agent_cfg['flow_steps']) == int(cfg.inversion.n_initial_steps), (
        f"agent.flow_steps ({agent_cfg['flow_steps']}) must equal inversion.n_initial_steps "
        f'({cfg.inversion.n_initial_steps}); the inverse and forward maps must share a '
        'discretization or every decode number below measures the mismatch')

    base = OmegaConf.to_container(cfg.inversion, resolve=True)
    sweep = OmegaConf.to_container(cfg.get('sweep', None) or {}, resolve=True)
    for key in sweep:
        assert key in base, f'+sweep key {key!r} is not an inversion setting ({sorted(base)})'
    keys = sorted(sweep)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(sweep[k] for k in keys))] or [{}]

    records = []
    for i, override in enumerate(combos, 1):
        inv = {**base, **override}
        print(f'\n[{i}/{len(combos)}] ' + (', '.join(f'{k}={v}' for k, v in override.items()) or 'baseline'))
        record = {'override': override,
                  **_score(agent, batch, inv, int(base.get('batch_size', 256)),
                           cov_k=int(cfg.get('cov_k', 16)), cov_draws=int(cfg.get('cov_draws', 8)))}
        records.append(record)
        t, c = record['typicality'], record['coverage']
        # Coverage and the decode error it is bought with, together: neither means
        # anything alone. The curve is over k = how many related states may supply it.
        curve = '  '.join(f'k{k}:{v:.2f}' for k, v in c['covered_at_k'].items())
        bulk = c['covered_by_prior_quantile']
        print(f"    covered  {curve}")
        print(f"    by prior quantile (at k={c['k_max']})  "
              + '  '.join(f'{k}:{v:.2f}' for k, v in bulk.items() if isinstance(v, float)))
        print(f"    decode_mix {record['decode_mix_mean']}  E||u||^2 {t['mean_sq_norm']:7.2f} "
              f"(want {t['expected_mean']:.0f})  NaN {record['nan_decode_frac']:.4f}  "
              f"ESS {record['ess_mean']:.1f}  [{record['seconds']}s]")

    report = {
        'env': cfg.env_name,
        'seed': int(cfg.seed),
        'n_rows': int(len(rows)),
        'sample_mode': sample_mode,
        'sample_k': sample_k,
        'buffer_size': int(size),
        'flow': 'TRAINED' if cfg.restore_path is not None else 'RANDOM (control)',
        'flow_steps': int(agent_cfg['flow_steps']),
        'baseline_inversion': base,
        'sweep': sweep,
        'results': records,
    }
    write_report(report, cfg, 'preimage_tuning.json')


if __name__ == '__main__':
    main()
