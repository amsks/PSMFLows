"""D1 -- can psi^T w rank POLICIES? (hypothesis H0)

Figure F4 panel (c) asks whether the critic predicts the outcome of individual episodes of
ONE policy under ONE task vector. The only thing varying there is the start state, so a
perfectly good critic reads AUC ~ 0.5 whenever success is decided by factors the features
cannot resolve at t=0. That is not the critic's job in this method: the actor's -Q gradient
and GPI both use it to rank *policies* (equivalently, latents) at a state. This tool asks
that question instead.

Design: ONE frozen representation judges every policy -- that is the GPI use-case, one
critic comparing candidates. For each policy we take the latent it would emit at a fixed,
seeded set of start states, score it with the judge's psi(s, w, u)^T w, and compare the
resulting ranking against the policies' measured 500-episode success.

  predicted(pi) = mean_s  psi_judge(s, w_judge, u_pi(s)) . w_judge
  realised(pi)  = success from the eval500 JSON already on disk

Verdict: Spearman >= 0.6 supports H0 (the critic ranks fine; the per-episode diagnostic was
confounded). Spearman ~ 0 means the critic is genuinely uninformative -> H1/H2/H3.

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_policy_ranking.py agent=psmflow \
    env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    report_out=/data-local/amsks/PSMFLows/logs/d1_policy_ranking.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from agents.psm import targets_uncertainty
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

EXP = '/data-local/amsks/PSMFLows/exp/PSMFLows'
GROUP = f'{EXP}/psmflow_latentpsm_cube_021456'
LOGS = '/data-local/amsks/PSMFLows/logs'

#: The judge. One representation scores every candidate.
JUDGE = (f'{GROUP}/sd000_20260805_021459', 500000)

#: (name, run_dir, epoch, eval-json basename or None). Spread of quality is the point:
#: five final seeds plus one seed's training trajectory, which is quality variation with
#: architecture held exactly fixed.
POLICIES = [
    ('actor_sd0_500k', f'{GROUP}/sd000_20260805_021459', 500000, 'eval500_latentpsm_cube_sd0'),
    ('actor_sd1_500k', f'{GROUP}/sd001_20260805_100228', 500000, 'eval500_latentpsm_cube_sd1'),
    ('actor_sd2_500k', f'{GROUP}/sd002_20260805_211248', 500000, 'eval500_latentpsm_cube_sd2'),
    ('actor_sd3_500k', f'{GROUP}/sd003_20260805_211248', 500000, 'eval500_latentpsm_cube_sd3'),
    ('actor_sd4_500k', f'{GROUP}/sd004_20260806_000039', 500000, 'eval500_latentpsm_cube_sd4'),
    ('actor_sd2_50k', f'{GROUP}/sd002_20260805_211248', 50000, 'eval100_latentpsm_cube_sd2_50k'),
    ('actor_sd2_150k', f'{GROUP}/sd002_20260805_211248', 150000, 'eval100_latentpsm_cube_sd2_150k'),
    ('actor_sd2_250k', f'{GROUP}/sd002_20260805_211248', 250000, 'eval100_latentpsm_cube_sd2_250k'),
    ('actor_sd2_400k', f'{GROUP}/sd002_20260805_211248', 400000, 'eval100_latentpsm_cube_sd2_400k'),
    # The behaviour prior, expressed in latent space: a fresh prior draw every step is
    # exactly what the BC control does.
    ('bc_prior', None, None, 'eval500_bcflow_cube'),
]
N_STATES = 64
SEED = 0


def spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind='stable')
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        srt = np.asarray(x)[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and srt[j + 1] == srt[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def permutation_p(a, b, n_perm=20000, seed=0):
    """Two-sided permutation p for Spearman. n~10 policies, so the asymptotic p is a lie."""
    obs = spearman(a, b)
    if obs is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b)
    count = 0
    for _ in range(n_perm):
        if abs(spearman(a, rng.permutation(b))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)

    def build(path, epoch):
        a = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
        a = restore_agent(a, path, epoch)
        zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
        return a.infer_eval_z(zb['next_observations'],
                              zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))

    judge = build(*JUDGE)
    w = judge.task_z

    # Fixed, seeded start states -- every policy is scored at the same states.
    states = []
    for i in range(N_STATES):
        ob, _ = eval_env.reset(seed=SEED + i)
        states.append(np.asarray(ob, dtype=np.float32))
    states = jnp.asarray(np.stack(states))

    def score(u_batch):
        """mean_s psi_judge(s, w, u_s)^T w, with the actor's ensemble pessimism."""
        qpsi = judge.psi(states, jnp.broadcast_to(w, (states.shape[0], *w.shape)), u_batch)
        Qs = (qpsi * w).sum(-1)                                   # (P, N)
        qmean, qunc = targets_uncertainty(Qs, judge.config['num_parallel'])
        Q = qmean - judge.config['actor_pessimism_penalty'] * qunc
        return float(jnp.mean(Q)), float(jnp.std(Q))

    # action_dim/u_clip are only concrete on a built agent: the raw hydra config leaves
    # action_dim null until create() reads it off the example actions.
    d_a = int(judge.config['action_dim'])
    u_clip = float(judge.config['u_clip'])

    rng = jax.random.PRNGKey(SEED)
    rows, missing = [], []
    for name, path, epoch, eval_name in POLICIES:
        rng, key = jax.random.split(rng)
        if path is None:
            # Behaviour prior in latent space: a clipped prior draw per state.
            u = jnp.clip(jax.random.normal(key, (N_STATES, d_a)),
                         -u_clip, u_clip)
        else:
            pol = build(path, epoch)
            noise = jax.random.normal(key, (N_STATES, d_a))
            u = u_clip * pol.actor(
                states, jnp.broadcast_to(w, (N_STATES, *w.shape)), noise)

        pred, pred_sd = score(u)
        p = f'{LOGS}/{eval_name}.json' if eval_name else None
        realised = None
        if p and os.path.exists(p):
            with open(p) as f:
                realised = json.load(f)['success']
        else:
            missing.append(eval_name)
        rows.append({'policy': name, 'restore_epoch': epoch,
                     'predicted': pred, 'predicted_sd_over_states': pred_sd,
                     'realised_success': realised,
                     'mean_latent_norm2': float(jnp.mean((u ** 2).sum(-1)))})
        print(f'{name:20s} predicted {pred:12.2f}  realised '
              f'{"n/a" if realised is None else f"{realised:.3f}"}', flush=True)

    usable = [r for r in rows if r['realised_success'] is not None]
    rho, pval = (None, None)
    if len(usable) >= 4:
        rho, pval = permutation_p([r['predicted'] for r in usable],
                                  [r['realised_success'] for r in usable], seed=SEED)

    if rho is None:
        verdict = 'inconclusive: too few policies with a measured success rate'
    elif rho >= 0.6:
        verdict = (f'H0 supported: the critic ranks policies (Spearman {rho:.2f}, '
                   f'p={pval:.3f}); the per-episode AUC diagnostic was confounded')
    elif rho <= 0.2:
        verdict = (f'H0 rejected: the critic does not rank policies either '
                   f'(Spearman {rho:.2f}, p={pval:.3f}) -> investigate H1/H2/H3')
    else:
        verdict = (f'ambiguous: Spearman {rho:.2f} (p={pval:.3f}) between the 0.2 and 0.6 '
                   f'decision thresholds')

    report = {
        'env': cfg.env_name, 'judge': {'path': JUDGE[0], 'epoch': JUDGE[1]},
        'n_start_states': N_STATES, 'seed': SEED,
        'policies': rows, 'n_policies_ranked': len(usable),
        'spearman_predicted_vs_realised': rho, 'permutation_p': pval,
        'missing_eval_reports': missing,
        'verdict': verdict,
    }
    print(f'\nSpearman {rho} (p={pval}) over {len(usable)} policies\nVERDICT: {verdict}')
    if missing:
        print(f'missing evals (policy excluded from the correlation): {missing}')
    write_report(report, cfg, 'diag_policy_ranking.json')


if __name__ == '__main__':
    main()
