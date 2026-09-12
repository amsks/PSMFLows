"""What GPI selects, and how the critic scores it, checkpoint by checkpoint.

The affine psmflow agent on cube swings between 0.70 and 0.09 across consecutive 50k
checkpoints of one seed (500 episodes: 250k .532, 300k .286, 350k .704, 400k .282,
450k .272, 500k .086, BC control .072) and nothing logged in training separates them.
Acting is `gpi_select`: per step, draw K = gpi_num_u action latents u and K policy indices
u' from the clipped prior, score every pair by Q = psi(s,u,u')^T w with ensemble
pessimism, decode the argmax u through the frozen flow. A checkpoint can fail in three
distinguishable ways, and this probe separates them:

  (a) the ARGMAX moves to the edge of the prior box -- selection prefers extreme latents
      the flow was never asked to decode, and the decoded action saturates the action clip;
  (b) the SCORE goes flat -- top-1 and top-2 are within noise of each other, so the pick
      is effectively a uniform prior draw (= the BC control, .072);
  (c) the RANKING drifts wholesale -- the critic still discriminates, but its ordering over
      a FIXED candidate set is uncorrelated with the neighbouring checkpoint's, i.e. the
      measure is spinning rather than converging.

Three sections, one JSON + one npz sidecar per checkpoint:

  1. rollout        N eval episodes with the eval's own seeding, recording per step the K^2
                    Q values' summary, the argmax gap, the selected (u, u'), the decoded
                    action, and the observation.
  2. fixed-state    256 dataset states x ONE candidate set (u, u') fixed by a hardcoded
                    key, identical in every checkpoint's job. The (256, K, K) Q tensor goes
                    to the npz; `compare` turns four of them into cross-checkpoint rank
                    correlations and top-5 overlaps.
  3. mc             ground truth on `n_mc` states drawn `lead` steps before a success
                    ONSET: per candidate u, restore the simulator and score it two ways --
                    `held` (roll the frozen flow H steps with u fixed) and `onestep`
                    (decode u once, then replay the dataset's recorded actions). Both
                    discounted at the training gamma. The roster is ranked by the
                    checkpoint's psi^T w, by a frozen FQL expert's action critic read at
                    G(s, u) (`+oracle_path`), and by -||u||.

Run (one sbatch per epoch; see scripts/slurm/diag_gpi_selection.sbatch):
  MUJOCO_GL=egl .venv/bin/python tools/diag_gpi_selection.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
      restore_path=<run_dir> restore_epoch=350000 \
      report_out=$PSM_DATA/logs/diag_gpi_selection_cube_sd0_350000.json

Compare (CPU, no hydra):
  .venv/bin/python tools/diag_gpi_selection.py compare out.json a.npz b.npz c.npz d.npz

Ranker table across checkpoints (CPU, `-` for no JSON output):
  .venv/bin/python tools/diag_gpi_selection.py table out.json report1.json report2.json ...

The agent config is inherited from the run's own flags.json by `eval_checkpoint`'s
`merge_run_config`, imported not copied, so this probe builds exactly the policy
`tools/eval_checkpoint.py` scored.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py)

CAND_KEY = 12345        # the ONE candidate set every checkpoint's fixed-state probe uses
PROBE_ROW_SEED = 7      # dataset rows for the fixed-state probe (a dedicated Generator)


# ---------------------------------------------------------------- pure helpers (tested)
def _spearman(a, b):
    """Rank correlation. Ties are broken by position, not averaged -- Q values are
    continuous floats, so no pair of candidates ever ties in practice."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ra, rb = np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def _topk_overlap(a, b, k=5):
    """|top-k(a) ∩ top-k(b)| / k -- the part of the ranking that GPI can ever act on."""
    ia = set(np.argsort(np.asarray(a).ravel())[::-1][:k].tolist())
    ib = set(np.argsort(np.asarray(b).ravel())[::-1][:k].tolist())
    return len(ia & ib) / float(k)


def _q(x):
    """mean/std/deciles of a per-step quantity, the shape every row of the table takes."""
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': None}
    return {'mean': float(x.mean()), 'std': float(x.std()),
            'p10': float(np.quantile(x, 0.1)), 'p50': float(np.quantile(x, 0.5)),
            'p90': float(np.quantile(x, 0.9))}


def compare(npz_paths, out_path):
    """Cross-checkpoint drift on the SHARED fixed candidate set. CPU only."""
    blobs = {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        ep = int(d['restore_epoch'])
        blobs[ep] = {'Q': np.asarray(d['fixed_Q'], np.float64),          # (S, K, K)
                     'Qt': (np.asarray(d['fixed_Qtarget'], np.float64)
                            if 'fixed_Qtarget' in d else None),
                     'mc_su': (np.asarray(d['mc_su'], np.float64) if 'mc_su' in d else None),
                     'mc_ret': (np.asarray(d['mc_returns'], np.float64)
                                if 'mc_returns' in d else None),
                     'path': p}
    eps = sorted(blobs)
    pairs = {}
    for i, e1 in enumerate(eps):
        for e2 in eps[i + 1:]:
            Q1, Q2 = blobs[e1]['Q'], blobs[e2]['Q']
            n = min(len(Q1), len(Q2))
            rho_pair, rho_u, ov5, ov5p, same_arg = [], [], [], [], []
            for s in range(n):
                a, b = Q1[s], Q2[s]
                rho_pair.append(_spearman(a, b))
                sa, sb = a.max(axis=1), b.max(axis=1)      # per-u GPI score
                rho_u.append(_spearman(sa, sb))
                ov5.append(_topk_overlap(sa, sb, 5))
                ov5p.append(_topk_overlap(a.ravel(), b.ravel(), 5))
                same_arg.append(float(np.argmax(sa) == np.argmax(sb)))
            pairs[f'{e1}_vs_{e2}'] = {
                'spearman_pairs_mean': float(np.mean(rho_pair)),
                'spearman_per_u_mean': float(np.mean(rho_u)),
                'spearman_per_u_p10': float(np.quantile(rho_u, 0.1)),
                'top5_overlap_per_u': float(np.mean(ov5)),
                'top5_overlap_pairs': float(np.mean(ov5p)),
                'argmax_u_agreement': float(np.mean(same_arg)),
                'n_states': int(n)}
            T1, T2 = blobs[e1]['Qt'], blobs[e2]['Qt']
            if T1 is not None and T2 is not None:
                # the target critic is Polyak-averaged, so if IT drifts as fast as the
                # online head, averaging is not the missing stabiliser
                pairs[f'{e1}_vs_{e2}']['spearman_per_u_target'] = float(np.mean(
                    [_spearman(T1[s].max(1), T2[s].max(1)) for s in range(n)]))
    # Checkpoint-averaged selection. The MC rollouts use only the FROZEN flow and the
    # simulator, so every checkpoint's job produced the same ground truth; averaging the
    # (per-state z-scored) critic scores over checkpoints tests whether the drift between
    # them is independent noise on top of a shared, correct signal.
    mc_ens = {}
    have = [e for e in eps if blobs[e]['mc_su'] is not None]
    if len(have) >= 2:
        rets = blobs[have[0]]['mc_ret']
        live = [i for i in range(len(rets)) if rets[i].std() > 0]
        def _z(a):
            return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
        per = {e: float(np.mean([_spearman(blobs[e]['mc_su'][i], rets[i]) for i in live]))
               for e in have}
        ens = np.mean([_z(blobs[e]['mc_su']) for e in have], axis=0)
        mc_ens = {'n_states_non_degenerate': len(live),
                  'spearman_per_checkpoint': per,
                  'spearman_single_mean': float(np.mean(list(per.values()))),
                  'spearman_checkpoint_averaged': float(
                      np.mean([_spearman(ens[i], rets[i]) for i in live])),
                  'regret_single_mean': float(np.mean([
                      np.mean([rets[i].max() - rets[i][int(blobs[e]['mc_su'][i].argmax())]
                               for i in live]) for e in have])),
                  'regret_checkpoint_averaged': float(np.mean(
                      [rets[i].max() - rets[i][int(ens[i].argmax())] for i in live])),
                  'regret_random': float(np.mean([rets[i].max() - rets[i].mean() for i in live]))}
    report = {'probe': 'gpi-selection cross-checkpoint ranking drift',
              'mc_ensemble': mc_ens,
              'inputs': {str(e): blobs[e]['path'] for e in eps},
              'note': ('same 256 dataset states and the SAME (u, u\') candidate set '
                       f'(PRNGKey({CAND_KEY})) in every checkpoint; per-u score is '
                       'max over the K policy indices, which is what GPI acts on'),
              'pairs': pairs}
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(pairs, indent=2))
    print(f'report -> {out_path}')
    return report


def table(json_paths, out_path=None):
    """The ranker table: mean per-state Spearman over live states, per ground truth. CPU.

    One row per RANKER, averaged over the checkpoints whose reports are passed in, with the
    live-state count and the spread beside it so a row is never read without its n. The
    `_inbox` columns repeat the same Spearman on the roster subset inside `inbox_clip`.
    """
    blobs = []
    for p in json_paths:
        with open(p) as f:
            blobs.append((os.path.basename(p), json.load(f)['mc']))
    tags = [t for t in ('onestep_bc', 'held', 'onestep')
            if f'spread_{t}_mean' in blobs[0][1]]
    names, seen = [], set()
    for _, m in blobs:
        for k in m:
            if k.startswith('rho_') and k.endswith(f'_{tags[0]}_mean'):
                n = k[len('rho_'):-len(f'_{tags[0]}_mean')]
                if n not in seen:
                    seen.add(n)
                    names.append(n)
    def avg(key):
        v = [m[key] for _, m in blobs if key in m]
        return float(np.mean(v)) if v else None
    rep = {'probe': 'ranker table over the shared onset roster',
           'n_checkpoints': len(blobs), 'inputs': [b[0] for b in blobs],
           'n_states': blobs[0][1].get('n_states'),
           'inbox_clip': blobs[0][1].get('inbox_clip'),
           'inbox_n': blobs[0][1].get('inbox_n'),
           'ground_truth': {t: {'live_states': avg(f'n_states_non_degenerate_{t}'),
                                'spread_mean': avg(f'spread_{t}_mean'),
                                'regret_random': avg(f'regret_random_{t}')} for t in tags},
           'rankers': {}}
    for n in names:
        row = {}
        for t in tags:
            row[t] = avg(f'rho_{n}_{t}_mean')
            row[f'{t}_regret'] = avg(f'regret_{n}_{t}_mean')
            if f'rho_{n}_{t}_inbox_mean' in blobs[0][1]:
                row[f'{t}_inbox'] = avg(f'rho_{n}_{t}_inbox_mean')
        rep['rankers'][n] = row
    hdr = f"{'ranker':14s}" + ''.join(f'{t[:10]:>12s}' + (f"{'inbox':>9s}" if f'{t}_inbox' in
                                     next(iter(rep['rankers'].values())) else '') for t in tags)
    print(f"{rep['n_checkpoints']} checkpoints, {rep['n_states']} states, "
          f"inbox_clip {rep['inbox_clip']} ({rep['inbox_n']} of the roster)")
    for t in tags:
        g = rep['ground_truth'][t]
        print(f"  {t:12s} live {g['live_states']:.0f}  spread {g['spread_mean']:.2f}  "
              f"regret(random) {g['regret_random']:.2f}")
    print(hdr)
    for n, row in rep['rankers'].items():
        line = f'{n:14s}'
        for t in tags:
            line += f"{row[t]:+12.3f}"
            if f'{t}_inbox' in row:
                line += f"{row[f'{t}_inbox']:+9.3f}"
        print(line)
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(rep, f, indent=2)
        print(f'report -> {out_path}')
    return rep


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
    from utils.log_utils import write_report
    from utils.psm_common import targets_uncertainty

    np.random.seed(int(cfg.seed))            # eval_checkpoint's pinning of the relabel batch
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack, add_info=True)
    ds = Dataset.create(**train_dataset)
    merged, prov = merge_run_config(OmegaConf.to_container(cfg.agent, resolve=True),
                                    cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert config['acting'] == 'gpi' and config['policy_index'] == 'latent', (
        f"this probe instruments gpi_select's (u, u') pair scan; got acting="
        f"{config['acting']!r} policy_index={config['policy_index']!r}")
    assert config.get('index_agg', 'max') == 'max', 'index_agg=expectile takes a different path'

    ex = ds.sample(1)
    agent = agents[config['agent_name']].create(cfg.seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, cfg.restore_path, int(cfg.restore_epoch))
    zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
    agent = agent.infer_eval_z(zb['next_observations'],
                               zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))

    # `create` fills ob_dims/action_dim into the agent's OWN config; the pre-create dict
    # still has them at None (main.py sets them from the example batch).
    ac = agent.config
    K, d_a = int(ac['gpi_num_u']), int(ac['action_dim'])
    u_clip, P = float(ac['u_clip']), int(ac['num_parallel'])
    pess = float(ac['actor_pessimism_penalty'])
    w = agent.task_z

    def _score(obs, u_pairs, index_pairs, params=None):
        """Exactly gpi_select's scoring of a (u, u') roster. obs (M, ob), u_pairs / index_pairs (M, d_a).

        `params=agent.target_psi` gives the same rule read off the Polyak-averaged critic;
        the returned `qmean` is the pessimism-free readout. Both are candidate
        stabilisations of the selection rule.
        """
        Qs = (agent.psi(obs, index_pairs, u_pairs, params=params) * w).sum(-1)   # (P, M)
        qmean, qunc = targets_uncertainty(Qs, P)
        return qmean - pess * qunc, qmean, qunc

    @jax.jit
    def probe(obs1, seed):
        """gpi_select's latent branch, instrumented. obs1 (ob,) -> per-step record."""
        r_u, r_up = jax.random.split(seed)
        u_cand = jnp.clip(jax.random.normal(r_u, (K, d_a)), -u_clip, u_clip)
        u_index = jnp.clip(jax.random.normal(r_up, (K, d_a)), -u_clip, u_clip)
        u_pairs = jnp.repeat(u_cand, K, axis=0)                      # i-major, as gpi_select
        index_pairs = jnp.tile(u_index, (K, 1))
        obs = jnp.broadcast_to(obs1, (K * K, *obs1.shape))
        Q, qmean, qunc = _score(obs, u_pairs, index_pairs)
        best = jnp.argmax(Q)
        srt = jnp.sort(Q)
        su = Q.reshape(K, K).max(axis=1)                        # per-u GPI score
        su_srt = jnp.sort(su)
        u_star, i_star = u_pairs[best], index_pairs[best]
        act = agent.decode(obs1[None], u_star[None])[0]
        def clip_of(x, c):
            return (jnp.abs(x) >= c - 1e-6).mean()
        return dict(  # noqa: C408 -- the keyword form keeps the record's field names readable
            q_max=srt[-1], q_top2=srt[-2], q_mean=Q.mean(), q_std=Q.std(), q_min=srt[0],
            su_max=su_srt[-1], su_top2=su_srt[-2], su_std=su.std(),
            qunc_at_best=qunc[best], qmean_at_best=qmean[best],
            arg_u=best // K, arg_idx=best % K,
            sel_u=u_star, sel_idx=i_star, act=act,
            sel_u_norm=jnp.linalg.norm(u_star), sel_idx_norm=jnp.linalg.norm(i_star),
            sel_u_clip=clip_of(u_star, u_clip), sel_idx_clip=clip_of(i_star, u_clip),
            cand_u_norm=jnp.linalg.norm(u_cand, axis=-1).mean(),
            cand_u_clip=clip_of(u_cand, u_clip),
            act_norm=jnp.linalg.norm(act), act_clip=clip_of(act, 1.0))

    @jax.jit
    def fixed_scores(obs1, u_cand, u_index):
        """The deployed score, the pessimism-free score and the target critic's score on
        one fixed roster -- the three selection rules compared in section 2."""
        u_pairs = jnp.repeat(u_cand, K, axis=0)
        index_pairs = jnp.tile(u_index, (K, 1))
        obs = jnp.broadcast_to(obs1, (K * K, *obs1.shape))
        Q, qm, _ = _score(obs, u_pairs, index_pairs)
        Qt, _, _ = _score(obs, u_pairs, index_pairs, agent.target_psi)
        return Q.reshape(K, K), qm.reshape(K, K), Qt.reshape(K, K)

    @jax.jit
    def decode1(obs1, u1):
        return agent.decode(obs1[None], u1[None])[0]

    # ---- 1. instrumented rollout, seeded exactly as utils.evaluation.evaluate ----
    n_ep, seed = int(cfg.get('n_episodes', 20)), int(cfg.seed)
    # 0 = run to the env's own time limit (what an eval does). A small cap is the CPU
    # smoke path: it exercises every line of the step loop without a 200-step episode.
    max_ep_steps = int(cfg.get('max_ep_steps', 0))
    rng = jax.random.PRNGKey(seed)
    eval_env.reset(seed=seed)
    recs, ep_success, ep_len, ep_id_of_step, obs_log = [], [], [], [], []
    for ep in range(n_ep):
        ob, _ = eval_env.reset()
        done, t, succ = False, 0, 0.0
        while not done:
            rng, k = jax.random.split(rng)
            obs_log.append(np.asarray(ob, np.float32))
            r = jax.device_get(probe(jnp.asarray(ob, jnp.float32), k))
            a = np.clip(np.asarray(r['act']), -1, 1)
            ob, _, term, trunc, info = eval_env.step(a)
            succ = max(succ, float(np.max(np.asarray(info.get('success', 0.0)))))
            recs.append(r)
            ep_id_of_step.append(ep)
            done, t = bool(term or trunc), t + 1
            if max_ep_steps and t >= max_ep_steps:
                done = True
        ep_success.append(succ)
        ep_len.append(t)
        print(f'  ep {ep:3d}  len {t:4d}  success {succ:.0f}', flush=True)

    col = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
    ep_id = np.asarray(ep_id_of_step)
    # persistence: consecutive steps redraw the candidate roster, so an INDEX match is a
    # 1/K coincidence; the meaningful quantity is whether the selected latent VECTOR is
    # reselected in the same direction. Both are reported, within episodes only.
    same_ep = ep_id[1:] == ep_id[:-1]
    def cos(x):
        return ((x[1:] * x[:-1]).sum(-1) /
                (np.linalg.norm(x[1:], axis=-1) * np.linalg.norm(x[:-1], axis=-1) + 1e-12))
    cos_u, cos_i = cos(col['sel_u'])[same_ep], cos(col['sel_idx'])[same_ep]
    idx_same = ((col['arg_idx'][1:] == col['arg_idx'][:-1]) & same_ep).sum() / max(same_ep.sum(), 1)

    denom = np.maximum(np.abs(col['q_mean']), 1e-9)
    rollout = {
        'n_episodes': n_ep, 'n_steps': len(recs),
        'success': float(np.mean(ep_success)), 'episode_length': float(np.mean(ep_len)),
        'q_spread_rel': _q(col['q_std'] / denom),
        'q_range_rel': _q((col['q_max'] - col['q_min']) / denom),
        'q_mean_level': _q(col['q_mean']),
        'gap_pairs_over_std': _q((col['q_max'] - col['q_top2']) / np.maximum(col['q_std'], 1e-9)),
        'gap_per_u_over_std': _q((col['su_max'] - col['su_top2']) / np.maximum(col['su_std'], 1e-9)),
        'gap_per_u_rel': _q((col['su_max'] - col['su_top2']) / denom),
        'ensemble_unc_at_best_rel': _q(col['qunc_at_best'] / denom),
        'sel_u_norm': _q(col['sel_u_norm']), 'cand_u_norm': _q(col['cand_u_norm']),
        'sel_u_norm_over_cand': _q(col['sel_u_norm'] / np.maximum(col['cand_u_norm'], 1e-9)),
        'sel_idx_norm': _q(col['sel_idx_norm']),
        'sel_u_clipfrac': _q(col['sel_u_clip']), 'cand_u_clipfrac': _q(col['cand_u_clip']),
        'sel_idx_clipfrac': _q(col['sel_idx_clip']),
        'act_norm': _q(col['act_norm']), 'act_clipfrac': _q(col['act_clip']),
        'persistence_cos_u': _q(cos_u), 'persistence_cos_idx': _q(cos_i),
        'persistence_argidx_same': float(idx_same),
        'persistence_random_baseline': {'cos': 0.0, 'argidx_same': 1.0 / K},
    }

    # ---- 2. fixed-state probe: the SAME states and the SAME candidate set everywhere ----
    n_fixed = int(cfg.get('n_fixed_states', 256))
    rows = np.sort(np.random.default_rng(PROBE_ROW_SEED).choice(ds.size, n_fixed, replace=False))
    fobs = np.asarray(ds['observations'][rows], np.float32)
    ck = jax.random.PRNGKey(CAND_KEY)
    k_u, k_i = jax.random.split(ck)
    fu = jnp.clip(jax.random.normal(k_u, (K, d_a)), -u_clip, u_clip)
    fi = jnp.clip(jax.random.normal(k_i, (K, d_a)), -u_clip, u_clip)
    triples = [jax.device_get(fixed_scores(jnp.asarray(o), fu, fi)) for o in fobs]
    fixed_Q = np.stack([t[0] for t in triples]).astype(np.float32)        # (S, K, K)
    fixed_Qmean = np.stack([t[1] for t in triples]).astype(np.float32)    # no pessimism
    fixed_Qtarget = np.stack([t[2] for t in triples]).astype(np.float32)  # target critic
    su = fixed_Q.max(axis=2)                                    # (S, K)
    su_mean, su_tgt = fixed_Qmean.max(axis=2), fixed_Qtarget.max(axis=2)
    fu_np = np.asarray(jax.device_get(fu))
    sel_i = su.argmax(axis=1)
    fixed = {
        'n_states': n_fixed, 'K': K, 'cand_key': CAND_KEY, 'row_seed': PROBE_ROW_SEED,
        'q_spread_rel': _q(fixed_Q.reshape(n_fixed, -1).std(1)
                           / np.maximum(np.abs(fixed_Q.reshape(n_fixed, -1).mean(1)), 1e-9)),
        'gap_per_u_over_std': _q((np.sort(su, 1)[:, -1] - np.sort(su, 1)[:, -2])
                                 / np.maximum(su.std(1), 1e-9)),
        'sel_u_norm': _q(np.linalg.norm(fu_np[sel_i], axis=-1)),
        'cand_u_norm': float(np.linalg.norm(fu_np, axis=-1).mean()),
        'sel_u_clipfrac': _q((np.abs(fu_np[sel_i]) >= u_clip - 1e-6).mean(-1)),
        'argmax_u_histogram_top5': np.bincount(sel_i, minlength=K).argsort()[::-1][:5].tolist(),
        'argmax_u_concentration': float(np.bincount(sel_i, minlength=K).max() / n_fixed),
        'index_matters': _q(fixed_Q.max(2).mean(1) - fixed_Q.min(2).mean(1)),
        # Would a different selection RULE pick differently off the same weights?
        'pess_vs_mean': {
            'spearman_per_u': float(np.mean([_spearman(su[i], su_mean[i]) for i in range(n_fixed)])),
            'top5_overlap': float(np.mean([_topk_overlap(su[i], su_mean[i], 5) for i in range(n_fixed)]))},
        'online_vs_target': {
            'spearman_per_u': float(np.mean([_spearman(su[i], su_tgt[i]) for i in range(n_fixed)])),
            'top5_overlap': float(np.mean([_topk_overlap(su[i], su_tgt[i], 5) for i in range(n_fixed)]))},
    }

    # ---- 3. Monte-Carlo ground truth, on success ONSET states ----
    # Section 2's ranking is only worth anything against a return, and getting that return
    # right cost this probe its first answer. The 2026-09-08 reading (Spearman 0.06-0.16)
    # was a mean over FIVE live states: the pool was every dataset row with reward above
    # the minimum, stepped back `lead`, and on cube a success RUN is ~44 rows long (477
    # onsets against 20810 rewarding rows), so ten of sixteen probe states were already
    # inside a success, where every candidate scores the same and the rank correlation is a
    # tie-break. The pool is now the ONSETS alone -- r[i] > min and r[i-1] == min -- which
    # is the only place a 50-step rollout can separate candidates. That number is retired.
    #
    # Three ground truths, because they answer different questions:
    #   held      a_t = G(s_t, u) for H steps, u held fixed. What the old probe measured,
    #             and NOT what GPI deploys (GPI redraws u every step), so a flat `held`
    #             ranking does not on its own convict the critic.
    #   onestep   execute G(s, u) ONCE, then replay the dataset's own recorded actions for
    #             the remaining H-1 steps. REPORTED, NOT THE TEST: once u knocks the state
    #             off the recorded trajectory the replayed actions no longer fit the state,
    #             so this return largely measures how far u pushed the state off the
    #             recorded path -- which is what -||u|| measures too. Confounded.
    #   onestep_bc  execute G(s, u) ONCE, then run the frozen behaviour flow CLOSED LOOP
    #             for H-1 steps on a prior latent stream FIXED PER STATE, so every
    #             candidate at a state gets the same continuation noise (common random
    #             numbers: the comparison is paired and the BC policy's own variance does
    #             not enter the spread). This is the one that matches what a one-step Q
    #             means -- u once, then an in-support behaviour continuation -- and it is
    #             what the pre-registration is read on. If ITS spread is flat, no critic
    #             can rank u at this state and the argmax is a max over noise.
    # All are discounted at the TRAINING gamma; the undiscounted sum the old probe used
    # prices a reward 49 steps out like the next one, which no critic does.
    #
    # Three rankers on one roster (the same K candidates section 2 scores):
    #   (i)   the checkpoint's own psi^T w, deployed rule (pessimism, max over the index)
    #   (ii)  a frozen FQL expert's ACTION-space critic read at G(s, u) -- `+oracle_path`
    #   (iii) -||u||, critic-free
    #   (iv)  -||G(s, u) - a_expert(s)||, the same expert's own MODE action. This one is a
    #         check on the GROUND TRUTH, not on a critic: E1 measured that executing the
    #         roster candidate closest to the expert's action scores 0.934, so if (iv) does
    #         not rank the returns either, the per-candidate return under a fixed
    #         continuation is not a stable target and no conclusion may be drawn from the
    #         other three. Deterministic (the one-step head at zero noise), so it is a
    #         shared column like the returns themselves.
    # `+na_path` adds two more: a trained DSRL-NA arm's LATENT critic Q_W(s, u) and its
    # ACTION critic Q_A(s, G(s, u)). Those are reward-specific -- fitted on the task's real
    # reward -- so they answer a different question from (i): whether a critic that is
    # allowed to know the reward ranks the roster where the zero-shot measure does not.
    # Caveat when reading Q_W: the DSRL-NA arm trains at u_clip=1.5 and this roster is
    # drawn at the checkpoint's u_clip (3.0), so about half of it is outside the box Q_W
    # ever saw. Q_A carries no such caveat -- it reads actions, and G maps the whole box
    # into [-1, 1]. So every ranker also gets an `_inbox` column, the same Spearman
    # restricted to the candidates inside that smaller box: without it a low Q_W number is
    # ambiguous between "does not rank" and "never saw half the roster". The subset is a
    # property of the shared roster, so the columns stay comparable across rankers.
    #
    # The MC returns depend only on the frozen flow, the dataset and the simulator, so they
    # are identical in every checkpoint's job; `+mc_cache=<npz>` computes them once.
    n_mc, H = int(cfg.get('n_mc_states', 64)), int(cfg.get('mc_horizon', 50))
    lead = int(cfg.get('mc_lead', max(1, H // 2)))
    gamma = float(config['discount'])
    disc = gamma ** np.arange(H, dtype=np.float64)

    rew = np.asarray(ds['rewards']).ravel()
    term = np.asarray(ds['terminals']).ravel() if 'terminals' in ds else np.zeros_like(rew)
    hit = rew > rew.min()
    onset = np.flatnonzero(hit[1:] & ~hit[:-1]) + 1          # first rewarding row of a run
    onset = onset[onset >= lead]
    onset = np.array([h for h in onset if term[h - lead:h].sum() == 0], dtype=np.int64)
    mc = {'n_states': n_mc, 'horizon': H, 'lead': lead, 'discount': gamma,
          'n_onsets_available': len(onset), 'per_state': [], 'note': (
              'states are `lead` steps before a success ONSET (r[i] > min, r[i-1] == min). '
              'held: a_t = G(s_t, u) for H steps. onestep: G(s, u) once, then the recorded '
              'dataset actions (open loop, confounded with ||u||; reported only). '
              'onestep_bc: G(s, u) once, then the frozen flow closed-loop on a per-state '
              'FIXED prior latent stream -- the in-support continuation, and the one the '
              'pre-registration is read on. All discounted at the training gamma.')}
    if len(onset) >= n_mc:
        mc_rows = np.sort(np.random.default_rng(PROBE_ROW_SEED + 1).choice(
            onset - lead, size=n_mc, replace=False))
        mc['state_source'] = f'{len(onset)} success onsets, {lead} steps back'
    else:
        mc_rows = rows[:n_mc]
        mc['state_source'] = 'FALLBACK: uniform fixed-probe rows (too few success onsets)'
    mc['rows'] = [int(r) for r in mc_rows]
    sim = {k: np.asarray(ds[k]) for k in ('qpos', 'qvel', 'button_states') if k in ds}

    # (ii) the oracle ranker: a frozen FQL expert's own action-space critic, read at the
    # decode of each candidate. Optional -- without `+oracle_path` the roster is scored
    # by (i) and (iii) only and `oracle_*` keys are absent, not zero.
    oracle_q = oracle_dist = None
    oracle_path = cfg.get('oracle_path', None)
    if oracle_path:
        with open(os.path.join(str(oracle_path), 'flags.json')) as fh:
            ocfg = ml_collections.ConfigDict(_lists_to_tuples(json.load(fh)['agent']))
        oracle = agents[ocfg['agent_name']].create(cfg.seed, ex['observations'], ex['actions'], ocfg)
        oracle = restore_agent(oracle, str(oracle_path), int(cfg.get('oracle_epoch', 500000)))

        @jax.jit
        def oracle_q(obs1, u_cand):   # the jitted closure IS the value bound above
            obs = jnp.broadcast_to(obs1, (K, *obs1.shape))
            a = agent.decode(obs, u_cand)
            return oracle.network.select('critic')(obs, actions=a).mean(0)     # (K,)

        @jax.jit
        def oracle_dist(obs1, u_cand):
            """-||G(s, u) - a_expert(s)||, the expert's one-step head at ZERO noise.

            `sample_actions` ignores its temperature argument and always draws, so the mode
            is taken directly off the head. Deterministic on purpose: this column is part
            of the shared ground truth, not a per-checkpoint reading. (E1's oracle-aim used
            a SAMPLED expert action; the two differ by the head's own noise.)
            """
            obs = jnp.broadcast_to(obs1, (K, *obs1.shape))
            a = agent.decode(obs, u_cand)
            a_star = jnp.clip(oracle.network.select('actor_onestep_flow')(
                oracle._actor_obs(obs1[None], None), jnp.zeros((1, d_a))), -1.0, 1.0)
            return -jnp.linalg.norm(a - a_star, axis=-1)                       # (K,)

    # A trained DSRL-NA arm's two critics, as two more rankers on the same roster.
    na_scores = None
    na_path = cfg.get('na_path', None)
    if na_path:
        na_merged, na_prov = merge_run_config(
            OmegaConf.to_container(cfg.agent, resolve=True), str(na_path), _cli_agent_keys())
        na_conf = ml_collections.ConfigDict(_lists_to_tuples(na_merged))
        assert na_conf['dsrl_na']['enabled'], (
            f'{na_path} was not trained with dsrl_na.enabled; its qa/qw are the freshly '
            'initialised heads and would rank noise')
        na_agent = agents[na_conf['agent_name']].create(
            cfg.seed, ex['observations'], ex['actions'], na_conf)
        na_agent = restore_agent(na_agent, str(na_path), int(cfg.get('na_epoch', 500000)))

        @jax.jit
        def na_scores(obs1, u_cand):    # the jitted closure IS the value bound above
            obs = jnp.broadcast_to(obs1, (K, *obs1.shape))
            return (na_agent.qw(obs, u_cand),
                    na_agent.qa(obs, na_agent.decode(obs, u_cand)).min(0))

    if sim and n_mc > 0:
        def _restore(r):
            env.reset()
            if 'button_states' in sim:
                env.unwrapped.set_state(sim['qpos'][r], sim['qvel'][r], sim['button_states'][r])
            else:
                env.unwrapped.set_state(sim['qpos'][r], sim['qvel'][r])

        def _dec(ob, u):
            return np.clip(np.asarray(jax.device_get(
                decode1(jnp.asarray(ob, jnp.float32), u))), -1, 1)

        def _roll(r, u, mode, tail):
            """One H-step rollout from dataset row r, discounted at the training gamma.

            mode        step 0            steps 1..H-1
            held        G(s, u)           G(s_t, u), the same u
            onestep     G(s, u)           the dataset's recorded actions (open loop)
            onestep_bc  G(s, u)           G(s_t, tail[t]), the frozen flow closed loop
            replay      recorded action   the dataset's recorded actions
            bc_ref      G(s, tail[0])     G(s_t, tail[t])

            `tail` is the per-state prior latent stream and is the SAME array for every
            candidate at that state, so the BC continuation's own randomness is common to
            all of them and cancels out of the spread.
            """
            _restore(r)
            ob = np.asarray(ds['observations'][int(r)], np.float32)
            ret, sc = 0.0, 0.0
            for k in range(H):
                if mode == 'held' or (k == 0 and mode in ('onestep', 'onestep_bc')):
                    a = _dec(ob, u)
                elif mode in ('onestep_bc', 'bc_ref'):
                    a = _dec(ob, tail[k])
                else:
                    a = np.clip(np.asarray(ds['actions'][min(int(r) + k, ds.size - 1)]), -1, 1)
                ob, rw, tm, tr, info = env.step(a)
                ret += disc[k] * float(rw)
                sc = max(sc, float(np.max(np.asarray(info.get('success', 0.0)))))
                if tm or tr:
                    break
            return ret, sc

        # The ground truth is checkpoint-independent: `+mc_cache=<npz>` pays for it once.
        cache = str(cfg.get('mc_cache', '') or '')
        # The per-state BC continuation stream. Deterministic, so every job and every
        # candidate at a state sees the same tail.
        tails = np.clip(np.random.default_rng(PROBE_ROW_SEED + 2).standard_normal(
            (len(mc_rows), H, d_a)), -u_clip, u_clip).astype(np.float32)
        sig = np.array([H, lead, gamma, len(mc_rows), float(mc_rows.sum()),
                        float(np.asarray(fu_np).sum()), float(tails.sum())], np.float64)
        blob = None
        if cache and os.path.isfile(cache):
            d = np.load(cache)
            if d['sig'].shape == sig.shape and np.allclose(d['sig'], sig):
                blob = {k: d[k] for k in d.files}
                mc['cache'] = f'reused {cache}'
            else:
                mc['cache'] = f'IGNORED {cache} (signature mismatch)'
        if blob is None:
            acc = {m: ([], []) for m in ('held', 'onestep', 'onestep_bc')}
            ref_r, bc_r = [], []
            for si, r in enumerate(mc_rows):
                t = tails[si]
                for m in acc:
                    rr, ss = zip(*[_roll(r, fu[i], m, t) for i in range(K)])
                    acc[m][0].append(rr)
                    acc[m][1].append(ss)
                ref_r.append(_roll(r, None, 'replay', t)[0])
                bc_r.append(_roll(r, None, 'bc_ref', t)[0])
                print(f'  mc {si}/{len(mc_rows)} row {int(r)}  spreads  '
                      + '  '.join(f'{m} {max(acc[m][0][-1]) - min(acc[m][0][-1]):.2f}'
                                  for m in acc), flush=True)
            blob = {'sig': sig, 'replay_returns': np.asarray(ref_r, np.float32),
                    'bc_ref_returns': np.asarray(bc_r, np.float32)}
            for m, (rr, ss) in acc.items():
                blob[f'{m}_returns'] = np.asarray(rr, np.float32)
                blob[f'{m}_success'] = np.asarray(ss, np.float32)
            if cache:
                os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
                np.savez_compressed(cache, **blob)
                mc['cache'] = f'wrote {cache}'
        gts = {m: blob[f'{m}_returns'] for m in ('held', 'onestep', 'onestep_bc')}
        one = gts['onestep']
        mc_rets = gts['held']                             # npz/`compare` compatibility

        mc_trip = [jax.device_get(fixed_scores(jnp.asarray(ds['observations'][int(r)], np.float32),
                                               fu, fi)) for r in mc_rows]
        mc_su = np.stack([t[0].max(axis=1) for t in mc_trip])          # deployed rule
        mc_su_mean = np.stack([t[1].max(axis=1) for t in mc_trip])     # no pessimism
        mc_su_tgt = np.stack([t[2].max(axis=1) for t in mc_trip])      # target critic
        mc_su_or = (np.stack([np.asarray(jax.device_get(oracle_q(
            jnp.asarray(ds['observations'][int(r)], np.float32), fu))) for r in mc_rows])
            if oracle_q is not None else None)
        nrm = np.linalg.norm(fu_np, axis=-1)
        rankers = {'q': mc_su, 'qmean': mc_su_mean, 'qtarget': mc_su_tgt, 'negnorm':
                   np.broadcast_to(-nrm, mc_su.shape)}
        # The smaller box every ranker is ALSO scored inside. Defaults to the DSRL-NA arm's
        # own u_clip when one is loaded; `+inbox_clip=<x>` sets it without an na run.
        ib_clip = cfg.get('inbox_clip', None)
        if ib_clip is None and na_path:
            ib_clip = float(na_conf['u_clip'])
        inbox = (np.abs(fu_np).max(-1) <= float(ib_clip)) if ib_clip else None
        if inbox is not None:
            mc['inbox_clip'] = float(ib_clip)
            mc['inbox_n'] = int(inbox.sum())
        if mc_su_or is not None:
            rankers['oracle'] = mc_su_or
            rankers['expert_dist'] = np.stack([
                np.asarray(jax.device_get(oracle_dist(
                    jnp.asarray(ds['observations'][int(r)], np.float32), fu)))
                for r in mc_rows])
            mc['oracle_path'] = str(oracle_path)
        if na_scores is not None:
            pairs = [jax.device_get(na_scores(
                jnp.asarray(ds['observations'][int(r)], np.float32), fu)) for r in mc_rows]
            rankers['na_qw'] = np.stack([np.asarray(p[0]) for p in pairs])
            rankers['na_qa'] = np.stack([np.asarray(p[1]) for p in pairs])
            mc['na_path'] = str(na_path)
            mc['na_epoch'] = int(cfg.get('na_epoch', 500000))
            mc['na_u_clip'] = float(na_conf['u_clip'])
            mc['na_roster_inbox_frac'] = float(
                (np.abs(fu_np).max(-1) <= float(na_conf['u_clip'])).mean())
            mc['na_config_source'] = na_prov

        for si in range(len(mc_rows)):
            d = {'row': int(mc_rows[si]),
                 'replay_return': float(blob['replay_returns'][si]),
                 'bc_ref_return': float(blob['bc_ref_returns'][si])}
            for tag, g in gts.items():
                v = np.asarray(g[si], np.float64)
                d[f'{tag}_spread'] = float(v.max() - v.min())
                d[f'{tag}_std'] = float(v.std())
                d[f'{tag}_best'] = float(v.max())
                d[f'{tag}_mean'] = float(v.mean())
                d[f'{tag}_success_rate'] = float(np.mean(blob[f'{tag}_success'][si]))
                d[f'degenerate_{tag}'] = bool(v.std() == 0.0)
                for name, sc in rankers.items():
                    q = np.asarray(sc[si], np.float64)
                    d[f'rho_{name}_{tag}'] = _spearman(q, v)
                    d[f'regret_{name}_{tag}'] = float(v.max() - v[int(q.argmax())])
                    if inbox is not None:
                        qi, vi = q[inbox], v[inbox]
                        d[f'rho_{name}_{tag}_inbox'] = _spearman(qi, vi)
                        d[f'regret_{name}_{tag}_inbox'] = float(
                            vi.max() - vi[int(qi.argmax())])
            # kept under its old key so the 09-08 entry can be read against this one
            d['spearman_q_vs_mc'] = d['rho_q_held']
            mc['per_state'].append(d)

        for tag in gts:
            live = [d for d in mc['per_state'] if not d[f'degenerate_{tag}']]
            mc[f'n_states_non_degenerate_{tag}'] = len(live)
            live = live or mc['per_state']
            mc[f'spread_{tag}_mean'] = float(np.mean([d[f'{tag}_spread'] for d in live]))
            mc[f'spread_{tag}_p10'] = float(np.quantile([d[f'{tag}_spread'] for d in live], 0.1))
            mc[f'success_rate_{tag}'] = float(np.mean(
                [d[f'{tag}_success_rate'] for d in mc['per_state']]))
            mc[f'regret_random_{tag}'] = float(np.mean(
                [d[f'{tag}_best'] - d[f'{tag}_mean'] for d in live]))
            suffixes = ('', '_inbox') if inbox is not None else ('',)
            for name in rankers:
                for sfx in suffixes:
                    r = [d[f'rho_{name}_{tag}{sfx}'] for d in live]
                    mc[f'rho_{name}_{tag}{sfx}_mean'] = float(np.mean(r))
                    mc[f'rho_{name}_{tag}{sfx}_p10'] = float(np.quantile(r, 0.1))
                    mc[f'regret_{name}_{tag}{sfx}_mean'] = float(np.mean(
                        [d[f'regret_{name}_{tag}{sfx}'] for d in live]))
        mc['replay_return_mean'] = float(np.mean(blob['replay_returns']))
        mc['bc_ref_return_mean'] = float(np.mean(blob['bc_ref_returns']))
        # the retired aggregate, recomputed on the corrected pool so the two are comparable
        mc['spearman_mean'] = mc['rho_q_held_mean']
        mc['n_states_non_degenerate'] = mc['n_states_non_degenerate_held']
    else:
        mc['skipped'] = 'dataset carries no qpos/qvel (add_info) -- no simulator restore'

    report = {
        'probe': 'gpi selection anatomy (rollout / fixed-state / MC ground truth)',
        'env': cfg.env_name, 'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch), 'seed': seed,
        'K': K, 'u_clip': u_clip, 'psi_form': config.get('psi_form'),
        'policy_index': config.get('policy_index'), 'acting': config.get('acting'),
        'agent_config_source': prov,
        'rollout': rollout, 'fixed_state': fixed, 'mc': mc,
    }
    out = write_report(report, cfg, 'diag_gpi_selection.json')
    npz = str(out)[:-5] + '.npz' if str(out).endswith('.json') else str(out) + '.npz'
    mc_arrays = ({'mc_rows': np.asarray(mc_rows), 'mc_su': mc_su, 'mc_su_target': mc_su_tgt,
                  'mc_su_mean': mc_su_mean, 'mc_returns': np.asarray(mc_rets, np.float32),
                  'mc_returns_onestep': np.asarray(one, np.float32),
                  'mc_returns_onestep_bc': np.asarray(gts['onestep_bc'], np.float32),
                  'mc_replay_returns': blob['replay_returns'],
                  **({'mc_su_oracle': mc_su_or} if mc_su_or is not None else {})}
                 if sim and n_mc > 0 else {})
    np.savez_compressed(npz, restore_epoch=int(cfg.restore_epoch), fixed_Q=fixed_Q,
                        fixed_Qmean=fixed_Qmean, fixed_Qtarget=fixed_Qtarget, **mc_arrays,
                        fixed_rows=rows, cand_u=fu_np,
                        cand_idx=np.asarray(jax.device_get(fi)),
                        step_obs=np.stack(obs_log),
                        sel_u=col['sel_u'], sel_idx=col['sel_idx'], act=col['act'],
                        q_max=col['q_max'], q_top2=col['q_top2'], q_mean=col['q_mean'],
                        q_std=col['q_std'], ep_id=ep_id)
    print(f'npz -> {npz}')


def main():
    import hydra
    hydra.main(version_base=None, config_path='../configs', config_name='config')(_main)()


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'compare':
        compare(sys.argv[3:], sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == 'table':
        table(sys.argv[3:], sys.argv[2] if sys.argv[2] != '-' else None)
    else:
        main()
