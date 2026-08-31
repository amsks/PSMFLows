"""D3 -- does Q have usable relief over the latent, and does the actor's loss feel it?

D1 showed psi^T w barely separates policies of very different quality. Two ways that can
happen, with different fixes:

  H2  Q is flat over u everywhere, so the actor loss -Q/|Q| + bc*distill + bc_flow is owned
      by its behaviour-cloning terms and the actor converges to latent BC.
  H3  Q has relief in general but not around the actor's own mode: the backup only ever
      evaluates psi(s', w, u_actor(s',w)), so psi is trained on a shrinking slice of latent
      space, which flattens Q exactly where the actor lives -- a self-confirming
      equilibrium.

Measurements:
  (i)   Q over 512 prior draws at each of N states, w fixed: spread relative to |Q|, and
        the actor's own percentile within that distribution. Repeated in a small ball
        around the actor's latent, which is where H3 predicts extra flatness.
  (ii)  Per-term actor gradient norms (q_loss vs distill vs bc_flow) over real batches --
        is the q-term even competitive at bc_coeff=1.0?
  (iii) Actor latent dispersion across training checkpoints -- does the actor collapse?

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_q_landscape.py agent=psmflow env_name=<env> \
    agent.flow_ckpt_path=<flow> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> agent.use_point_preimage=true \
    restore_path=<run_dir> restore_epoch=500000 \
    report_out=/data-local/amsks/PSMFLows/logs/d3_q_landscape_cube.json
"""
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
from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
from utils.log_utils import write_report

N_STATES = 64
N_DRAWS = 512
BALL_SIGMA = 0.3      # local perturbation scale around the actor's latent
N_GRAD_BATCHES = 40
SEED = 0
#: checkpoints to check dispersion over, if present in the run dir
DISPERSION_EPOCHS = [50000, 150000, 250000, 350000, 450000, 500000]


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)

    def build(epoch):
        a = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
        a = restore_agent(a, cfg.restore_path, epoch)
        zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
        return a.infer_eval_z(zb['next_observations'],
                              zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))

    agent = build(cfg.restore_epoch)
    w = agent.task_z
    d_a = int(agent.config['action_dim'])
    u_clip = float(agent.config['u_clip'])

    states = []
    for i in range(N_STATES):
        ob, _ = eval_env.reset(seed=SEED + i)
        states.append(np.asarray(ob, dtype=np.float32))
    states = jnp.asarray(np.stack(states))

    latent_index = agent.config.get('policy_index') == 'latent'
    has_actor = bool(agent.config.get('train_actor', True))

    if latent_index:
        # psi(s, u', u): score a candidate u as GPI does -- max over a fixed panel of
        # policy-index draws u' ~ p0 (pessimism inside, max outside, mirroring
        # gpi_select's latent branch). The panel is fixed per probe run so Q is a
        # deterministic function of (s, u).
        N_IDX = 16
        u_idx_panel = jnp.clip(
            jax.random.normal(jax.random.PRNGKey(SEED + 1), (N_IDX, d_a)),
            -u_clip, u_clip)

        def Q(obs_b, u_b):
            n = obs_b.shape[0]
            obs_r = jnp.repeat(obs_b, N_IDX, axis=0)
            u_r = jnp.repeat(u_b, N_IDX, axis=0)
            idx_r = jnp.tile(u_idx_panel, (n, 1))
            qpsi = agent.psi(obs_r, idx_r, u_r)
            Qs = (qpsi * w).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, agent.config['num_parallel'])
            q = qmean - agent.config['actor_pessimism_penalty'] * qunc
            return q.reshape(n, N_IDX).max(axis=-1)
    else:
        def Q(obs_b, u_b):
            wb = jnp.broadcast_to(w, (obs_b.shape[0], *w.shape))
            qpsi = agent.psi(obs_b, wb, u_b)
            Qs = (qpsi * w).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, agent.config['num_parallel'])
            return qmean - agent.config['actor_pessimism_penalty'] * qunc

    rng = jax.random.PRNGKey(SEED)

    # ---- (i) Q landscape over the latent, per state -------------------------------
    glob, local, pct, act_q = [], [], [], []
    for i in range(N_STATES):
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        s_rep = jnp.broadcast_to(states[i], (N_DRAWS, states.shape[1]))
        u_prior = jnp.clip(jax.random.normal(k1, (N_DRAWS, d_a)), -u_clip, u_clip)
        q_prior = np.asarray(Q(s_rep, u_prior))

        if has_actor:
            u_a = u_clip * agent.actor(states[i][None],
                                       jnp.broadcast_to(w, (1, *w.shape)),
                                       jax.random.normal(k2, (1, d_a)))
        else:
            # No actor (paper-faithful arm): the deployed pick is gpi_select. Scored
            # against FRESH prior draws (q_prior above), so its percentile measures
            # whether the ranking transfers across draws rather than being 1.0 by
            # construction.
            u_a = agent.gpi_select(states[i], seed=k2)[None]
        q_a = float(np.asarray(Q(states[i][None], u_a))[0])

        u_ball = jnp.clip(u_a + BALL_SIGMA * jax.random.normal(k3, (N_DRAWS, d_a)),
                          -u_clip, u_clip)
        q_ball = np.asarray(Q(s_rep, u_ball))

        scale = abs(float(q_prior.mean())) + 1e-8
        glob.append(float(q_prior.std()) / scale)
        local.append(float(q_ball.std()) / scale)
        pct.append(float((q_prior < q_a).mean()))
        act_q.append(q_a)

    landscape = {
        'n_states': N_STATES, 'n_draws': N_DRAWS, 'ball_sigma': BALL_SIGMA,
        'q_relative_spread_prior_mean': float(np.mean(glob)),
        'q_relative_spread_prior_median': float(np.median(glob)),
        'q_relative_spread_local_mean': float(np.mean(local)),
        'actor_percentile_in_prior_Q_mean': float(np.mean(pct)),
        'actor_percentile_in_prior_Q_median': float(np.median(pct)),
    }

    # ---- (ii) which actor-loss term actually drives the gradient? -----------------
    aug = load_augmented_dataset(config['preimage_path'])
    aug, _ = repair_invalid_preimages(aug)
    train = Dataset.create(**aug)
    # main.py sets these on the training dataset; without them the batch has no
    # 'noise_preimage' key and the actor loss cannot be evaluated.
    train.return_preimage_noise = True
    train.preimage_point_mode = bool(config.get('use_point_preimage', False))

    def term_grads(batch, sampled, which):
        def loss_fn(actor_params, vf_params):
            total, info = agent.flow_actor_loss(batch, sampled, actor_params, vf_params)
            if which == 'total':
                return total
            if which == 'bc_flow':
                return info['actor_bc_flow_loss']
            if which == 'distill':
                return agent.config['actor']['bc_coeff'] * info['actor_bc_error']
            # q term = total - the two bc terms
            return (total - info['actor_bc_flow_loss']
                    - agent.config['actor']['bc_coeff'] * info['actor_bc_error'])
        g = jax.grad(loss_fn, argnums=0)(agent.actor.params, agent.actor_vf.params)
        leaves = jax.tree_util.tree_leaves(g)
        return float(jnp.sqrt(sum((jnp.sum(x ** 2) for x in leaves))))

    if has_actor:
        norms = {k: [] for k in ('q', 'distill', 'bc_flow', 'total')}
        for _ in range(N_GRAD_BATCHES):
            rng, k = jax.random.split(rng)
            b = train.sample(config['batch_size'])
            sampled = agent.sample_step_inputs(b, k)
            for key in norms:
                norms[key].append(term_grads(b, sampled, key))
        grads = {f'grad_norm_{k}': float(np.mean(v)) for k, v in norms.items()}
        grads['q_share_of_total'] = grads['grad_norm_q'] / (
            grads['grad_norm_q'] + grads['grad_norm_distill'] + grads['grad_norm_bc_flow'])
        grads['n_batches'] = N_GRAD_BATCHES
    else:
        grads = {'skipped': 'train_actor=false -- no actor loss exists on this arm',
                 'q_share_of_total': None}

    # ---- (iii) does the actor's dispersion shrink over training? -------------------
    dispersion = {}
    for ep in DISPERSION_EPOCHS if has_actor else []:
        if not os.path.exists(os.path.join(str(cfg.restore_path), f'params_{ep}.pkl')):
            continue
        a = build(ep)
        rng, k = jax.random.split(rng)
        u = u_clip * a.actor(states, jnp.broadcast_to(w, (N_STATES, *w.shape)),
                             jax.random.normal(k, (N_STATES, d_a)))
        dispersion[str(ep)] = float(jnp.mean((u ** 2).sum(-1)))

    spread = landscape['q_relative_spread_prior_mean']
    ratio = landscape['q_relative_spread_local_mean'] / (spread + 1e-12)
    qshare = grads.get('q_share_of_total')
    qshare_txt = f'{qshare:.3f}' if qshare is not None else 'n/a (no actor)'
    if spread < 0.05:
        verdict = (f'H2 supported: Q is flat over the latent -- relative spread {spread:.4f} '
                   f'of |Q| across 512 prior draws. The actor loss is owned by its BC terms '
                   f'(q-term gradient share {qshare_txt}), so the policy '
                   f'is latent behaviour cloning with pessimism, not value improvement.')
    elif ratio < 0.3:
        verdict = (f'H3 supported: Q has relief globally ({spread:.3f}) but is flat around '
                   f'the actor mode ({landscape["q_relative_spread_local_mean"]:.3f}) -- '
                   f'self-confirming equilibrium; try mixing prior latents into the backup.')
    else:
        verdict = (f'H2/H3 not supported: Q has usable relief ({spread:.3f}) including near '
                   f'the actor mode; the actor sits at percentile '
                   f'{landscape["actor_percentile_in_prior_Q_mean"]:.2f} of it.')

    report = {'env': cfg.env_name, 'restore_path': str(cfg.restore_path),
              'restore_epoch': int(cfg.restore_epoch), 'seed': SEED,
              'q_landscape': landscape, 'actor_grad_terms': grads,
              'actor_latent_norm2_by_epoch': dispersion,
              'bc_coeff': float(agent.config['actor']['bc_coeff']),
              'verdict': verdict}
    print(f"\nQ relative spread {spread:.4f} (local {landscape['q_relative_spread_local_mean']:.4f}) | "
          f"actor percentile {landscape['actor_percentile_in_prior_Q_mean']:.2f} | "
          f"q grad share {grads['q_share_of_total']:.3f}\nVERDICT: {verdict}")
    write_report(report, cfg, 'diag_q_landscape.json')


if __name__ == '__main__':
    main()
