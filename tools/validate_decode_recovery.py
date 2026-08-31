"""Do the stored latents recover the data? Two tests over a preimage npz + its flow.

Test 1 (decode recovery). Draw `u` from each row's stored preimage mixture, decode it
through the frozen flow and compare with the recorded action. The npz stores the round
trip of the POINT preimage only (`preimage_roundtrip`), so the distribution that
`use_point_preimage=false` trains on has no measured decode error anywhere.
`+t1_sources` selects where the latent comes from: `mixture`
(default), `point` (the exact backward-ODE latent -- reproduces the stored roundtrip, so
it is also the floor) and `prior` (a fresh N(0, I) latent at the same state: what the
error looks like when the latent carries no information about the action).

Test 2 (buffer recovery). At each of `t2_states` states, decode `t2_latents` prior latents
and search the WHOLE buffer for the nearest states, then report the smallest distance from
each generated action to those states' recorded actions. Two baselines are computed
against the same neighbours, because nearby states carry different actions in the data
regardless of the flow: the query row's own recorded action, and a uniform random action.
Without them the generated number has no scale.

Reads observations/actions out of the npz, so no env, no MuJoCo, no OGBench download:

  .venv/bin/python tools/validate_decode_recovery.py agent=fql \
      env_name=cube-single-play-singletask-v0 \
      restore_path=$PSM_DATA/flow/cube-single-play restore_epoch=500000 \
      agent.flow_steps=100 +preimage_npz=$PSM_DATA/preimages/cube-single-play.npz

`agent.flow_steps` must equal the sidecar's `flow_steps` (asserted): the forward and
inverse maps have to share a discretization, else every number below measures that
mismatch. Sizes: `preimage_limit` (test 1 rows), `+t2_states`, `+t2_latents`, `+t2_k`;
`+tests=[1]` runs one of the two. Keys absent from configs/config.yaml need a leading `+`.

`+u_clip=3.0` clamps every latent before decoding, which is what psmflow trains on
(`agents/psmflow.py sample_step_inputs`). Unclamped is the default because it measures the
stored posterior as it is -- and on cube that posterior puts 60% of its mixture draws
outside ||u||=3, far enough off the flow's N(0, I) training support that ~0.3% of decodes
diverge to NaN (`nan_decode_frac`, reported per latent source; surviving rows still get
statistics). Clamping removes those without moving the healthy rows much.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import numpy as np

from utils.log_utils import write_report
from utils.nn_search import knn_indices
from utils.preimage_eval import (LATENT_SOURCES, decode_in_batches, latents_from,
                                 load_flow_and_preimages, stats)


def _test1(cfg, agent, data, valid_rows, rng, batch_size, skills=None):
    """Decode error of a latent drawn per row, one report block per latent source."""
    sources = [str(a) for a in cfg.get('t1_sources', None) or ['mixture']]
    assert set(sources) <= set(LATENT_SOURCES), f'+t1_sources must be a subset of {LATENT_SOURCES}'
    draws = int(cfg.get('t1_draws', 1))
    u_clip = cfg.get('u_clip', None)
    n_rows = min(int(cfg.get('preimage_limit') or 50000), len(valid_rows))
    rows = rng.choice(valid_rows, size=n_rows, replace=False)
    rows = np.sort(np.tile(rows, draws))

    obs = data['observations'][rows]
    # compute_flow_actions clips its output, so an unclipped target would charge the flow
    # for the clip (the convention augment_dataset_with_point_preimage uses).
    act = np.clip(data['actions'][rows], -1, 1)
    sk = None if skills is None else skills[rows]

    out = {'n_rows': int(n_rows), 'draws': draws, 'n_decodes': int(len(rows)),
           'u_clip': None if u_clip is None else float(u_clip)}
    for source in sources:
        noises = latents_from(source, data, rows, rng, u_clip=u_clip)
        recon = decode_in_batches(agent, obs, noises, batch_size, skills=sk)
        # The forward Euler unroll diverges on a few (state, latent) pairs even when the
        # latent itself is finite and typical -- 0.3% of mixture draws on the published
        # cube npz, at ||u|| 6-10 while other rows decode fine past 13. One such row
        # would otherwise turn every statistic below into NaN and hide the rest.
        finite = np.isfinite(recon).all(-1)
        diff = (recon - act)[finite]
        l2 = np.linalg.norm(diff, axis=-1)
        out[source] = {
            'nan_decode_frac': round(float(1.0 - finite.mean()), 6),
            'latent_sq_norm_mean': round(float((noises ** 2).sum(-1).mean()), 4),
        }
        if not finite.any():
            out[source]['note'] = 'every decode was non-finite; no statistics computed'
            continue
        out[source].update({
            'mse': round(float((diff ** 2).mean()), 8),
            'rmse': round(float(np.sqrt((diff ** 2).mean())), 6),
            'l2': stats(l2),
            'max_abs_per_dim': [round(float(v), 6) for v in np.abs(diff).max(0)],
            'frac_above': {str(t): round(float((l2 > t).mean()), 4) for t in (0.05, 0.1, 0.2)},
        })
    if 'point' in sources and 'preimage_roundtrip' in data:
        # Same quantity the npz stores, up to the device: the implicit-Euler inverse is
        # float32-sensitive, so re-running it elsewhere shifts the latent enough for the
        # decode to land ~1e-3 from the recorded action where the npz says ~1e-5. A gap
        # beyond that means the flow or the discretization here is not the npz's.
        out['stored_roundtrip_mean_same_rows'] = round(
            float(np.asarray(data['preimage_roundtrip'])[rows].mean()), 6)
    return out


def _test2(cfg, agent, data, valid_rows, rng, batch_size, skills=None):
    """Nearest recorded (state, action) pair to each action generated from a prior latent."""
    m = min(int(cfg.get('t2_states', 512)), len(valid_rows))
    n = int(cfg.get('t2_latents', 32))
    k = int(cfg.get('t2_k', 32))
    u_clip = cfg.get('u_clip', None)
    chunk = int(cfg.get('t2_chunk', 100_000))

    all_obs, all_act = data['observations'], data['actions']
    d_a = all_act.shape[-1]
    rows = np.sort(rng.choice(valid_rows, size=m, replace=False))

    # Prior latents, decoded at each query state (the fql bc_only control's own draw).
    noises = rng.standard_normal((m * n, d_a)).astype(np.float32)
    if u_clip is not None:
        noises = np.clip(noises, -float(u_clip), float(u_clip))
    rep = np.repeat(rows, n)
    gen = decode_in_batches(agent, all_obs[rep], noises, batch_size,
                            skills=None if skills is None else skills[rep])
    gen = gen.reshape(m, n, d_a)

    # Per-dimension standardization: state dimensions carry different units, and an
    # unscaled L2 would let the widest one decide what "nearest state" means.
    scale = np.maximum(np.asarray(all_obs, np.float32).std(0), 1e-6)
    nb_idx, nb_dist = knn_indices(all_obs[rows], all_obs, k, chunk=chunk,
                                  scale=scale, exclude=rows)
    nb_act = np.clip(all_act[nb_idx], -1, 1)                      # (m, k, d_a)

    def _min_to_neighbors(actions):
        """(m, n, d_a) -> (m, n) distance to the closest neighbour action."""
        return np.linalg.norm(actions[:, :, None, :] - nb_act[:, None, :, :], axis=-1).min(-1)

    gen_min = _min_to_neighbors(gen)
    data_min = _min_to_neighbors(np.clip(all_act[rows], -1, 1)[:, None, :])
    rand_min = _min_to_neighbors(rng.uniform(-1, 1, (m, n, d_a)).astype(np.float32))

    data_p90 = float(np.percentile(data_min, 90))
    return {
        'n_states': int(m), 'n_latents': int(n), 'k': int(k),
        'buffer_size': int(len(all_obs)),
        'u_clip': None if u_clip is None else float(u_clip),
        'generated': stats(gen_min),
        'data_baseline': stats(data_min),
        'random_baseline': stats(rand_min),
        'frac_gen_above_data_p90': round(float((gen_min > data_p90).mean()), 4),
        # In standardized state units: how tight the neighbourhood the actions come from is.
        'neighbor_state_dist_nearest': stats(nb_dist[:, 0]),
        'neighbor_state_dist_kth': stats(nb_dist[:, -1]),
    }


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    tests = {int(t) for t in (cfg.get('tests', None) or [1, 2])}
    agent, data, valid_rows, skills, meta = load_flow_and_preimages(cfg)

    rng = np.random.default_rng(int(cfg.seed))
    batch_size = int(cfg.inversion.get('batch_size', 256))
    report = {
        'env': cfg.env_name,
        'seed': int(cfg.seed),
        'npz': os.path.abspath(str(cfg.preimage_npz)),
        'flow': 'TRAINED' if cfg.restore_path is not None else 'RANDOM (control)',
        'flow_steps': int(cfg.agent.flow_steps),
        'n_total': int(len(data['actions'])),
        'n_invalid_excluded': int(len(data['actions']) - len(valid_rows)),
        'meta': meta,
    }
    if 1 in tests:
        report['test1_decode_recovery'] = _test1(
            cfg, agent, data, valid_rows, rng, batch_size, skills=skills)
    if 2 in tests:
        report['test2_buffer_recovery'] = _test2(
            cfg, agent, data, valid_rows, rng, batch_size, skills=skills)

    print(json.dumps(report, indent=2))
    write_report(report, cfg, 'decode_recovery.json')


if __name__ == '__main__':
    main()
