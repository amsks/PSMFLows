"""S1 -- is the value landscape over `u` at fixed `s` NAVIGABLE, or scrambled?

The gate in `docs/plans/2026-09-03-latent-coherence-and-infom.md`. Forward passes only:
no training, no Stage-C checkpoint, no `psi`. The only learned objects are the FROZEN
Stage-A flow (for `decode`) and a frozen FQL expert (for the oracle score), so the answer
is valid for every Stage-C arm at once and cannot be invalidated by a retrain.

The contradiction it resolves: near-expert actions exist in the decode family at nearly
every step (E1 oracle best-of-512 = 0.934 against a 0.086 random floor), yet every LEARNED
function of `u` reads that relief as ~1% noise (D1 0.9%, D3 1.1%, Arm B 0.9%, E4b 0.86%,
DSRL-SAC 1.1-1.5%). Either (a) the landscape over `u` is rough and no smooth policy can
navigate it, or (b) it is navigable and every attempt so far died to an argmax over
samples of a noisy learned function (E4a). This tool distinguishes (a) from (b).

Per state `s` (harvested on-path from oracle rollouts): draw K clipped prior latents,
decode them through the frozen flow, score each by `v = -||a - a*||` with `a*` the frozen
FQL expert's action -- E1's oracle, exactly. Then measure whether `v` is a smooth function
of `u`:

  knn_r2_u              leave-one-out k-NN regression of v on u. THE HEADLINE.
  knn_r2_a              the same with neighbours in ACTION space. v is smooth in `a` by
                        construction, so this must come out ~1: it is the estimator's
                        calibration control, and if it fails nothing else here is readable.
  basin_spearman        Spearman(v_i, -||u_i - u_best||). High = one basin to descend.
  topdecile_dispersion  pairwise spread of the top decile by v, over the same for a random
                        decile. ~1 scattered, <1 clustered. Reported in u AND in a.
  local_lipschitz       median/p95 of |dv|/||du|| over random pairs, plus ||da||/||du||
                        and corr(||da||, ||du||) -- a value-monotone transport preserves
                        local geometry, so that correlation is the mechanism.

Pre-registered decision rule (plan S1): knn_r2_u(k=10) < 0.05 with basin_spearman < 0.1
and dispersion ~1.0 => SCRAMBLED, and S2/S3 are not worth building as specified. > 0.5
with basin_spearman > 0.4 => NAVIGABLE, the argmax is the killer, build S2.

Run (one per env x decoder; the one-step head is what deploys, the ODE is the true map):
  MUJOCO_GL=egl .venv/bin/python tools/diag_latent_smoothness.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$FLOW agent.flow_ckpt_epoch=500000 \
      agent.gpi_decode=ode agent.flow_decode_steps=100 \
      +oracle_path=$FQL_RUN +oracle_epoch=500000 \
      report_out=$PSM_DATA/logs/d5_latent_smoothness_cube_ode.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ruff: noqa: I001 -- import ORDER is load-bearing here; see the next line. Do not let
# `ruff --fix` sort this block: it moves xla_guard below jax and silently breaks the ODE.
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

N_STATES = 64          # on-path states, where the deployed policy actually has to choose
K = 512                # candidates per state -- E1's K, not the ranking tool's 128
KNN_KS = (5, 10, 20)
N_PAIRS = 4096         # random pairs for the Lipschitz statistics
N_DECILE_DRAWS = 20    # random-decile draws averaged in the dispersion denominator
SEED = 0


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def _pairdist(X):
    d2 = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2.0 * (X @ X.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _knn_loo_r2(X, v, ks=KNN_KS):
    """Leave-one-out k-NN regression R^2 of v on position in X.

    Self is excluded by construction (the diagonal is +inf), so this is an honest
    out-of-sample estimate of how much of the value is predictable from where you are.
    """
    D = _pairdist(X).copy()
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    denom = ((v - v.mean()) ** 2).sum() + 1e-12
    out = {}
    for k in ks:
        vhat = v[order[:, :k]].mean(axis=1)
        out[f'k{k}'] = float(1.0 - ((v - vhat) ** 2).sum() / denom)
    return out


def _mean_pairdist(X):
    if len(X) < 2:
        return float('nan')
    D = _pairdist(X)
    iu = np.triu_indices(len(X), k=1)
    return float(D[iu].mean())


def _topdecile_dispersion(X, v, rng):
    """Spread of the best decile over the spread of a random decile of the same size.

    The random denominator is averaged over N_DECILE_DRAWS draws rather than taken from a
    single one -- same estimator, less variance, and the draw count is recorded.
    """
    n = max(2, round(0.1 * len(v)))
    top = np.argsort(v)[::-1][:n]                      # v = -distance, so higher is better
    num = _mean_pairdist(X[top])
    den = np.mean([_mean_pairdist(X[rng.choice(len(v), n, replace=False)])
                   for _ in range(N_DECILE_DRAWS)])
    return float(num / (den + 1e-12))


def _local_lipschitz(u, a, v, rng):
    i = rng.integers(0, len(v), N_PAIRS)
    j = rng.integers(0, len(v), N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    du = np.linalg.norm(u[i] - u[j], axis=1)
    da = np.linalg.norm(a[i] - a[j], axis=1)
    dv = np.abs(v[i] - v[j])
    ok = du > 1e-9
    du, da, dv = du[ok], da[ok], dv[ok]
    rv, ra = dv / du, da / du
    return {
        'dv_du_median': float(np.median(rv)), 'dv_du_p95': float(np.percentile(rv, 95)),
        'da_du_median': float(np.median(ra)), 'da_du_p95': float(np.percentile(ra, 95)),
        'corr_da_du': float(np.corrcoef(da, du)[0, 1]),
    }


def _agg(per_state, key):
    x = np.asarray([s[key] for s in per_state], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p10': float(np.percentile(x, 10)), 'p90': float(np.percentile(x, 90))}


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    assert config['agent_name'] == 'psmflow', 'run with agent=psmflow (frozen-flow decode)'
    ex = ds.sample(1)

    # Frozen Stage-A flow ONLY. No restore_agent, no infer_eval_z, no psi -- this probe is
    # about the flow's own latent geometry and must not depend on any Stage-C checkpoint.
    agent = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
    d_a = int(agent.config['action_dim'])
    u_clip = float(agent.config['u_clip'])

    oracle_path = cfg.get('oracle_path', None)
    assert oracle_path, 'needs +oracle_path=<frozen FQL expert run dir>'
    with open(os.path.join(str(oracle_path), 'flags.json')) as f:
        oracle_cfg = ml_collections.ConfigDict(_lists_to_tuples(json.load(f)['agent']))
    oracle = agents[oracle_cfg['agent_name']].create(
        cfg.seed, ex['observations'], ex['actions'], oracle_cfg)
    oracle = restore_agent(oracle, str(oracle_path), int(cfg.get('oracle_epoch', 500000)))

    @jax.jit
    def decode_and_score(obs, u_b, key):
        a_star = jnp.clip(oracle.sample_actions(obs, seed=key), -1.0, 1.0)
        obs_b = jnp.broadcast_to(obs, (u_b.shape[0], *obs.shape))
        a = agent.decode(obs_b, u_b)
        return a, -jnp.linalg.norm(a - a_star[None], axis=-1), a_star

    # On-path states: roll the oracle, snapshot every 10 steps (the ranking tool's harness).
    states, rng = [], jax.random.PRNGKey(SEED)
    ep = 0
    while len(states) < N_STATES:
        ob, _ = eval_env.reset(seed=SEED + ep)
        done, t = False, 0
        while not done and len(states) < N_STATES:
            if t % 10 == 0:
                states.append(np.asarray(ob, dtype=np.float32))
            rng, k = jax.random.split(rng)
            a = np.asarray(jnp.clip(oracle.sample_actions(jnp.asarray(ob), seed=k), -1.0, 1.0))
            ob, _, term, trunc, _ = eval_env.step(a)
            done, t = bool(term or trunc), t + 1
        ep += 1

    nprng = np.random.default_rng(SEED)
    per_state, raw_u, raw_a, raw_v = [], [], [], []
    for s in states:
        rng, k_u, k_or = jax.random.split(rng, 3)
        u = jnp.clip(jax.random.normal(k_u, (K, d_a)), -u_clip, u_clip)
        obs = jnp.asarray(s)
        a_j, v_j, _ = decode_and_score(obs, u, k_or)
        u_n, a_n, v_n = np.asarray(u), np.asarray(a_j), np.asarray(v_j)
        finite = np.isfinite(v_n) & np.isfinite(a_n).all(1)
        u_n, a_n, v_n = u_n[finite], a_n[finite], v_n[finite]
        if len(v_n) < max(KNN_KS) + 2:
            continue
        raw_u.append(u_n), raw_a.append(a_n), raw_v.append(v_n)

        i_best = int(np.argmax(v_n))
        r2_u = _knn_loo_r2(u_n, v_n)
        r2_a = _knn_loo_r2(a_n, v_n)
        rec = {
            'basin_spearman': _spearman(v_n, -np.linalg.norm(u_n - u_n[i_best], axis=1)),
            'basin_spearman_a': _spearman(v_n, -np.linalg.norm(a_n - a_n[i_best], axis=1)),
            'topdecile_dispersion_u': _topdecile_dispersion(u_n, v_n, nprng),
            'topdecile_dispersion_a': _topdecile_dispersion(a_n, v_n, nprng),
            'best_dist': float(-v_n.max()), 'mean_dist': float(-v_n.mean()),
        }
        rec.update({f'knn_r2_u_{k}': val for k, val in r2_u.items()})
        rec.update({f'knn_r2_a_{k}': val for k, val in r2_a.items()})
        rec.update(_local_lipschitz(u_n, a_n, v_n, nprng))
        per_state.append(rec)

    keys = sorted(per_state[0].keys())
    report = {
        'probe': 'S1 -- navigability of the value landscape over u at fixed s',
        'env': cfg.env_name,
        'flow_ckpt_path': str(config.get('flow_ckpt_path')),
        'flow_ckpt_epoch': int(config.get('flow_ckpt_epoch')),
        'decoder': str(config.get('gpi_decode')),
        'flow_decode_steps': int(config.get('flow_decode_steps')),
        'oracle_path': str(oracle_path), 'oracle_epoch': int(cfg.get('oracle_epoch', 500000)),
        'oracle_agent': str(oracle_cfg['agent_name']),
        'n_states': len(per_state), 'K': K, 'u_clip': u_clip,
        'latents': 'clipped N(0, I)', 'seed': SEED,
        'n_decile_draws': N_DECILE_DRAWS, 'n_pairs': N_PAIRS,
        'stats': {k: _agg(per_state, k) for k in keys},
        'predictions': {
            'scrambled': {'knn_r2_u_k10': '< 0.05', 'basin_spearman': '< 0.1',
                          'topdecile_dispersion_u': '0.95-1.05', 'corr_da_du': '< 0.3'},
            'navigable': {'knn_r2_u_k10': '> 0.5', 'basin_spearman': '> 0.4',
                          'topdecile_dispersion_u': '< 0.8', 'corr_da_du': '> 0.7'},
            'control': {'knn_r2_a_k10': '> 0.9 either way'},
        },
    }
    out = cfg.get('report_out', None)
    if out:
        out = str(out)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        # Raw arrays beside the JSON so the statistics can be recomputed without re-rolling.
        np.savez_compressed(out.replace('.json', '') + '_raw.npz',
                            u=np.stack(raw_u), a=np.stack(raw_a), v=np.stack(raw_v),
                            states=np.stack(states[:len(raw_u)]))
        print(f'report -> {out}')
    print(json.dumps({'n_states': report['n_states'],
                      'knn_r2_u_k10': report['stats']['knn_r2_u_k10'],
                      'knn_r2_a_k10': report['stats']['knn_r2_a_k10'],
                      'basin_spearman': report['stats']['basin_spearman'],
                      'topdecile_dispersion_u': report['stats']['topdecile_dispersion_u'],
                      'corr_da_du': report['stats']['corr_da_du']}, indent=2))


if __name__ == '__main__':
    main()
