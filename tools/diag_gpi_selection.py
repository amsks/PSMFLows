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
  3. mc             ground truth for the first `n_mc` of those states: for each candidate u,
                    restore the simulator, roll the frozen flow forward H steps holding u
                    fixed, sum reward. Spearman(Q-rank, MC-rank) per checkpoint.

Run (one sbatch per epoch; see scripts/slurm/diag_gpi_selection.sbatch):
  MUJOCO_GL=egl .venv/bin/python tools/diag_gpi_selection.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
      restore_path=<run_dir> restore_epoch=350000 \
      report_out=$PSM_DATA/logs/diag_gpi_selection_cube_sd0_350000.json

Compare (CPU, no hydra):
  .venv/bin/python tools/diag_gpi_selection.py compare out.json a.npz b.npz c.npz d.npz

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

    # ---- 3. Monte-Carlo ground truth, on states where the task is WINNABLE in H steps ----
    # A uniformly sampled dataset state is hopeless within H=50 for every candidate, so
    # every fixed-u policy returns -H and the ranking is all ties -- a Spearman against a
    # constant is meaningless. The MC states are therefore drawn `mc_lead` steps BEFORE a
    # rewarding transition in the SAME episode, which is where a 50-step rollout can
    # actually separate the candidates. Deterministic, so all checkpoints share them.
    n_mc, H = int(cfg.get('n_mc_states', 12)), int(cfg.get('mc_horizon', 50))
    lead = int(cfg.get('mc_lead', max(1, H // 2)))
    mc = {'n_states': n_mc, 'horizon': H, 'lead': lead, 'per_state': [], 'note': (
        'per candidate u: restore the simulator at the dataset row, then roll the frozen '
        'flow with u HELD FIXED for H steps (a_t = G(s_t, u)); score = summed reward. '
        'States sit `lead` steps before a rewarding dataset transition, so the horizon is '
        'winnable; a degenerate (all-tied) MC set is reported rather than silently ranked.')}
    rew = np.asarray(ds['rewards']).ravel()
    term = np.asarray(ds['terminals']).ravel() if 'terminals' in ds else np.zeros_like(rew)
    hot = np.flatnonzero(rew > rew.min())
    hot = hot[hot >= lead]
    hot = np.array([h for h in hot if term[h - lead:h].sum() == 0], dtype=np.int64)
    if len(hot) >= n_mc:
        mc_rows = np.sort(np.random.default_rng(PROBE_ROW_SEED + 1).choice(
            hot - lead, size=n_mc, replace=False))
        mc['state_source'] = f'{len(hot)} dataset rows {lead} steps before a reward'
    else:
        mc_rows = rows[:n_mc]
        mc['state_source'] = 'FALLBACK: uniform fixed-probe rows (no rewarding transition found)'
    mc['rows'] = [int(r) for r in mc_rows]
    sim = {k: np.asarray(ds[k]) for k in ('qpos', 'qvel', 'button_states') if k in ds}
    if sim and n_mc > 0:
        mc_trip = [jax.device_get(fixed_scores(jnp.asarray(ds['observations'][int(r)], np.float32),
                                               fu, fi)) for r in mc_rows]
        mc_su = np.stack([t[0].max(axis=1) for t in mc_trip])
        mc_su_mean = np.stack([t[1].max(axis=1) for t in mc_trip])
        mc_su_tgt = np.stack([t[2].max(axis=1) for t in mc_trip])
        mc_rets = []
        for si in range(len(mc_rows)):
            r = int(mc_rows[si])
            rets, succs = [], []
            for i in range(K):
                env.reset()
                if 'button_states' in sim:
                    env.unwrapped.set_state(sim['qpos'][r], sim['qvel'][r], sim['button_states'][r])
                else:
                    env.unwrapped.set_state(sim['qpos'][r], sim['qvel'][r])
                ob, ret, sc = np.asarray(ds['observations'][r], np.float32), 0.0, 0.0
                for _ in range(H):
                    a = np.clip(np.asarray(jax.device_get(
                        decode1(jnp.asarray(ob, jnp.float32), fu[i]))), -1, 1)
                    ob, rew, term, trunc, info = env.step(a)
                    ret += float(rew)
                    sc = max(sc, float(np.max(np.asarray(info.get('success', 0.0)))))
                    if term or trunc:
                        break
                rets.append(ret)
                succs.append(sc)
            mc_rets.append(rets)
            nrm = np.linalg.norm(fu_np, axis=-1)
            lo = nrm <= np.median(nrm)
            mc['per_state'].append({
                'row': r, 'spearman_q_vs_mc': _spearman(mc_su[si], rets),
                'spearman_qmean_vs_mc': _spearman(mc_su_mean[si], rets),
                'spearman_qtarget_vs_mc': _spearman(mc_su_tgt[si], rets),
                # does the frozen flow simply do better from SMALL latents? -- the evidence
                # for or against 'restrict the candidates to a smaller prior ball'
                'spearman_negnorm_vs_mc': _spearman(-nrm, rets),
                'mc_return_small_half': float(np.mean(np.asarray(rets)[lo])),
                'mc_return_large_half': float(np.mean(np.asarray(rets)[~lo])),
                'top5_overlap': _topk_overlap(mc_su[si], rets, 5),
                'mc_return_of_q_argmax': float(rets[int(mc_su[si].argmax())]),
                'mc_return_of_qtarget_argmax': float(rets[int(mc_su_tgt[si].argmax())]),
                'mc_return_best': float(np.max(rets)), 'mc_return_mean': float(np.mean(rets)),
                'mc_return_std': float(np.std(rets)), 'degenerate': bool(np.std(rets) == 0.0),
                'mc_success_rate': float(np.mean(succs))})
            print(f'  mc state {si}: rho={mc["per_state"][-1]["spearman_q_vs_mc"]:+.3f}', flush=True)
        live = [d for d in mc['per_state'] if not d['degenerate']]
        mc['n_states_non_degenerate'] = len(live)
        mc['mc_success_rate_mean'] = float(np.mean([d['mc_success_rate'] for d in mc['per_state']]))
        mc['per_state'] = live or mc['per_state']
        rho = [d['spearman_q_vs_mc'] for d in mc['per_state']]
        mc['spearman_mean'] = float(np.mean(rho))
        mc['spearman_p10'] = float(np.quantile(rho, 0.1))
        for k in ('spearman_qmean_vs_mc', 'spearman_qtarget_vs_mc', 'spearman_negnorm_vs_mc'):
            mc[k + '_mean'] = float(np.mean([d[k] for d in mc['per_state']]))
        mc['regret_target_mean'] = float(np.mean([d['mc_return_best'] - d['mc_return_of_qtarget_argmax']
                                                  for d in mc['per_state']]))
        mc['regret_mean'] = float(np.mean([d['mc_return_best'] - d['mc_return_of_q_argmax']
                                           for d in mc['per_state']]))
        mc['regret_vs_random'] = float(np.mean([d['mc_return_best'] - d['mc_return_mean']
                                                for d in mc['per_state']]))
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
                  'mc_returns': np.asarray(mc_rets, np.float32)}
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
    else:
        main()
