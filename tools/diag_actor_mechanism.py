"""Why does the measure trained with the flowbc actor's latent in the TD bootstrap (critic A,
fb_bc_coeff=3, 0.805) beat the one trained with a prior draw (critic B, fb_bc_coeff=1000,
0.657/0.708)? Three hypotheses, each with its own measurement, on the same sampled (s, g) pairs
(2026-10-15).

  H1 (shape)     the set of latents the critic rates highly at a state is non-Gaussian
                 (multimodal or on a shell), so a Gaussian actor cannot cover it and the flowbc
                 actor only nudges the prior.
  H2 (optimism)  the actor's latent u' = clip(eps + delta) is rated above its own base draw
                 clip(eps) by the same critic's TARGET network, so A's bootstrap is a
                 best-of-(1 + eps) over the prior.
  H3 (sharpen)   the actor's step follows the measure's gradient in u, so A's measure separates
                 latents at a fixed state more than B's.

The point value is q_C(s, u; g) = the softmax share of g among [g + n_pool random dataset next
states] at w = h_C(g), on critic C's basis: exactly the deployed coef_source=hgoal_each readout
for one goal (`PSMGoalAgent._hgoal_each_score` with one column), and `actor_value` with
actor_value_kind=point. measure_loss=squared falls back to M(s, u, g).

Per sampled (s, g) -- n_states dataset rows with their point preimage u_data and a hindsight goal
drawn by the agent's own goal sampling (goal_sampling, goal_discount, goal_random_frac,
goal_cur_frac from critic A's flags.json) -- and per critic C in {A, B}:

  1. sensitivity (H3): 64 clipped prior draws; spread_within = std over the 64 of q_C;
     spread_across = std over states of the per-state mean q_C; the gradient norm
     ||d q_C / d u|| at 16 of the draws.
  2. top-set shape (H1): of n_draws clipped prior draws, the top top_frac under q_C against a
     random subset of the same size: mean norm and norm std relative to all draws (shell test),
     mean per-dimension excess kurtosis (Gaussian = 0), 2-means separation (centroid distance /
     pooled within-cluster std, as tools/diag_pick_modes.py) and minor-cluster share, mean
     log-likelihood under a Gaussian fit to the set vs under the prior N(0, I) (the clip at
     +-u_clip is ignored: it moves 0.3% of the mass per dimension), and the fraction of the set
     inside the Gaussian fit's Mahalanobis ball of squared radius d (reference: P(chi2_d <= d)).
  3. actor coverage (H1/H2/H3), critic A's delta: eps ~ N(0, I) (the same draws, unclipped),
     u' = clip(eps + delta_A(s, g, eps)), u_bc = clip(eps). mean ||delta||; the share of the
     delta's second moment carried by its mean over eps (1 = a constant shift, 0 = all
     eps-dependent); cosine(delta, d q_A/d u at u_bc); the share of u' whose nearest u_bc draw is
     in A's top set and the share of u' scoring above A's top-set threshold (control: top_frac);
     q(u') - q(u_bc) mean, relative mean and the fraction > 0 under A online, A target
     (target_basis + target_w_star), B online and B target. B's own delta norm is logged.

Everything is persisted (per-state arrays included) via report_out; the H1/H2/H3 lines print
the deciding numbers. Both checkpoints are restored as tools/eval_state_reward.py does
(flags.json defaults, typed agent.* overrides on top) with acting=gpi, coef_source=hgoal_each
forced; neither key changes a parameter shape.

Run (GPU):
  .venv/bin/python tools/diag_actor_mechanism.py agent=psmgoal \\
    env_name=cube-single-play-singletask-task1-v0 \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
    restore_path=<run dir A> +restore_path_b=<run dir B> restore_epoch=500000 \\
    +n_states=512 +n_draws=2000 seed=0 report_out=$PSM_DATA/logs/actor_mechanism/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside run())
from tools.diag_pick_modes import two_means

FORCED_KEYS = {'acting': 'gpi', 'coef_source': 'hgoal_each'}
CHI2_REFERENCE_SAMPLES = 200_000


# --------------------------------------------------------------------- pure statistics
def mean_sem(x):
    x = np.asarray(x, np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': None, 'sem': None, 'n': 0}
    sem = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {'mean': float(x.mean()), 'sem': sem, 'n': int(x.size)}


def excess_kurtosis_per_dim(x):
    """Mean over dimensions of the excess kurtosis m4 / m2^2 - 3 of x (N, d). Gaussian -> 0."""
    x = np.asarray(x, np.float64)
    c = x - x.mean(0)
    m2 = (c ** 2).mean(0)
    m4 = (c ** 4).mean(0)
    return float(np.mean(m4 / np.maximum(m2, 1e-12) ** 2 - 3.0))


def bimodality(x, rng, inits=5, iters=20):
    """2-means on x (N, d): separation = distance between the two centroids / pooled
    within-cluster std (tools/diag_pick_modes.py step_metrics), minor_share = the smaller
    cluster's share. Near 1-2 is one blob; well above 3 is two groups."""
    x = np.asarray(x, np.float64)
    lab = two_means(x, rng, inits=inits, iters=iters)
    sizes = np.bincount(lab, minlength=2)
    if sizes.min() < 2:
        return {'separation': 0.0, 'minor_share': float(sizes.min() / x.shape[0])}
    cs = np.stack([x[lab == j].mean(0) for j in range(2)])
    within = np.sqrt(np.mean([((x[lab == j] - cs[j]) ** 2).sum(-1).mean() for j in range(2)]))
    return {'separation': float(np.linalg.norm(cs[0] - cs[1]) / max(within, 1e-8)),
            'minor_share': float(sizes.min() / x.shape[0])}


def gaussian_fit(x, ridge=1e-6):
    x = np.asarray(x, np.float64)
    mu = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1) + ridge * np.eye(x.shape[1])
    return mu, cov


def gaussian_loglik(x, mu, cov):
    """Per-point log density of N(mu, cov) at the rows of x, (N,)."""
    x = np.asarray(x, np.float64)
    d = x.shape[1]
    c = x - mu
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0, 'covariance is not positive definite'
    maha = np.einsum('nd,nd->n', c @ np.linalg.inv(cov), c)
    return -0.5 * (maha + logdet + d * np.log(2.0 * np.pi))


def prior_loglik(x):
    """Per-point log density of N(0, I) at the rows of x, (N,)."""
    x = np.asarray(x, np.float64)
    return -0.5 * ((x ** 2).sum(1) + x.shape[1] * np.log(2.0 * np.pi))


def within_std_ball_fraction(x, mu, cov):
    """Fraction of rows of x with squared Mahalanobis distance to N(mu, cov) at most d (the
    expected squared radius). Gaussian reference: P(chi2_d <= d), `chi2_ball_reference`."""
    x = np.asarray(x, np.float64)
    c = x - mu
    maha = np.einsum('nd,nd->n', c @ np.linalg.inv(cov), c)
    return float(np.mean(maha <= x.shape[1]))


def chi2_ball_reference(d, rng, n=CHI2_REFERENCE_SAMPLES):
    """P(chi2_d <= d) by Monte Carlo: the within-ball fraction of an exact Gaussian sample."""
    z = rng.standard_normal((n, d))
    return float(np.mean((z ** 2).sum(1) <= d))


def shape_stats(subset, all_draws, rng):
    """The top-set shape statistics of `subset` (n, d) against the full draw set (N, d)."""
    subset = np.asarray(subset, np.float64)
    all_draws = np.asarray(all_draws, np.float64)
    norms, norms_all = np.linalg.norm(subset, axis=1), np.linalg.norm(all_draws, axis=1)
    mu, cov = gaussian_fit(subset)
    ll_fit, ll_prior = gaussian_loglik(subset, mu, cov), prior_loglik(subset)
    out = {
        'norm_mean_ratio': float(norms.mean() / max(norms_all.mean(), 1e-12)),
        'norm_std_ratio': float(norms.std() / max(norms_all.std(), 1e-12)),
        'kurtosis': excess_kurtosis_per_dim(subset),
        'loglik_gaussian_fit': float(ll_fit.mean()),
        'loglik_prior': float(ll_prior.mean()),
        'loglik_fit_minus_prior': float((ll_fit - ll_prior).mean()),
        'within_std_ball': within_std_ball_fraction(subset, mu, cov),
        'centroid_norm': float(np.linalg.norm(mu)),
    }
    out.update(bimodality(subset, rng))
    return out


def mean_explained_fraction(delta):
    """Share of the second moment of delta (N, d) carried by its mean over rows:
    ||mean_k delta_k||^2 / mean_k ||delta_k||^2. 1 = a constant shift, 0 = zero-mean."""
    delta = np.asarray(delta, np.float64)
    second = float((delta ** 2).sum(1).mean())
    if second <= 1e-20:
        return 0.0
    return float((delta.mean(0) ** 2).sum() / second)


def cosine_rows(a, b):
    """Row-wise cosine between a and b (N, d), (N,)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(den, 1e-12)


def nn_membership(query, ref, member):
    """Fraction of query rows whose nearest ref row is a member (member: bool (R,))."""
    query, ref = np.asarray(query, np.float64), np.asarray(ref, np.float64)
    d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    return float(np.mean(np.asarray(member, bool)[d2.argmin(1)]))


def nn_membership_loo(ref, member):
    """Leave-one-out `nn_membership` of ref against itself: the control rate."""
    ref = np.asarray(ref, np.float64)
    d2 = ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.mean(np.asarray(member, bool)[d2.argmin(1)]))


def summarise_rows(rows):
    """mean +- SE over states of every scalar key of a list of per-state dicts."""
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating, np.integer))]
    return {k: mean_sem([r[k] for r in rows]) for k in keys}


def diff_stats(q_new, q_base):
    """q(u') - q(u_bc) over the draws of one state."""
    d = np.asarray(q_new, np.float64) - np.asarray(q_base, np.float64)
    base = float(np.mean(np.abs(q_base)))
    return {'diff_mean': float(d.mean()), 'diff_rel': float(d.mean() / max(base, 1e-12)),
            'frac_pos': float(np.mean(d > 0)), 'q_base_mean': base, 'q_new_mean': float(np.mean(q_new))}


# --------------------------------------------------------------------- the agents
def load_critic(cfg, restore_path, label):
    """Restore one psmgoal checkpoint as tools/eval_state_reward.py does, FORCED_KEYS on top."""
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.flax_utils import restore_agent

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, restore_path, _cli_agent_keys())
    assert merged.get('agent_name') == 'psmgoal', 'diag_actor_mechanism reads psmgoal checkpoints'
    assert str(merged.get('policy_index')) == 'goal', f'{label}: needs policy_index=goal (h(g) coefficient)'
    assert str(merged.get('actor_kind')) == 'flowbc', f'{label}: needs actor_kind=flowbc (the delta actor)'
    for k, v in FORCED_KEYS.items():
        merged[k] = v
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    return config, merged, prov, agents, restore_agent


def make_fns(agent):
    """Per-state jitted functions on one critic. Params are arguments so online and target share
    one compile. q_fn: (bp, wp, obs (ob,), goal (ob,), u (K, d), pool (P, ob)) -> (K,)."""
    import jax
    import jax.numpy as jnp

    temp = float(agent.config['measure_temp'])
    softmax = agent._softmax()
    u_clip = float(agent.config['u_clip'])

    def q_fn(bp, wp, obs1, goal1, u, pool):
        K = u.shape[0]
        obs = jnp.broadcast_to(obs1, (K, obs1.shape[-1]))
        goals = jnp.broadcast_to(goal1, (K, goal1.shape[-1]))
        w = agent.w_star(goals, params=wp)                                    # (K, z) = h(g)
        m_g = agent.M(obs, u, goals, w, params=bp)                            # (K,)
        if not softmax:
            return m_g
        M = agent.mesh_M(obs, u, pool, w, params=bp)                          # (K, P)
        lse = jax.nn.logsumexp(jnp.concatenate([m_g[:, None], M], axis=1) / temp, axis=1)
        return jnp.exp(m_g / temp - lse)

    def grad_fn(bp, wp, obs1, goal1, u, pool):
        # each row's q depends on its own u only, so d(sum q)/du is the per-row gradient
        return jax.grad(lambda uu: q_fn(bp, wp, obs1, goal1, uu, pool).sum())(u)

    def actor_fn(dp, obs1, goal1, eps):
        K = eps.shape[0]
        obs = jnp.broadcast_to(obs1, (K, obs1.shape[-1]))
        goals = jnp.broadcast_to(goal1, (K, goal1.shape[-1]))
        u, d = agent.flowbc_latent(obs, goals, eps, params=dp)
        return u, d

    def clip_fn(eps):
        return jnp.clip(eps, -u_clip, u_clip)

    return jax.jit(q_fn), jax.jit(grad_fn), jax.jit(actor_fn), jax.jit(clip_fn)


def chunked(fn, u, chunk, *args):
    """fn(*args_before, u_chunk, *args_after) over row chunks of u; fn takes (bp, wp, obs, goal, u, pool)."""
    bp, wp, obs1, goal1, pool = args
    outs = [np.asarray(fn(bp, wp, obs1, goal1, u[i:i + chunk], pool)) for i in range(0, u.shape[0], chunk)]
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------- main
def run(cfg):
    import jax
    import jax.numpy as jnp

    from envs.env_utils import make_env_and_datasets
    from tools.eval_state_reward import build_dataset
    from utils.log_utils import write_report

    t_start = time.time()
    assert cfg.restore_path is not None, 'restore_path (critic A) is required'
    restore_b = cfg.get('restore_path_b', None)
    assert restore_b is not None, '+restore_path_b=<run dir> (critic B) is required'
    n_states = int(cfg.get('n_states', 512))
    n_draws = int(cfg.get('n_draws', 2000))
    n_pool = int(cfg.get('n_pool', 224))
    n_sens = int(cfg.get('n_sens_draws', 64))
    n_grad_sens = int(cfg.get('n_grad_draws', 16))
    n_grad_cov = int(cfg.get('n_grad_cov_draws', 256))
    top_frac = float(cfg.get('top_frac', 0.05))
    chunk = int(cfg.get('draw_chunk', 500))
    seed = int(cfg.seed)
    n_top = max(int(np.ceil(top_frac * n_draws)), 2)
    n_grad_sens, n_grad_cov = min(n_grad_sens, n_sens), min(n_grad_cov, n_draws)

    cfg_a, merged_a, prov_a, agents, restore_agent = load_critic(cfg, str(cfg.restore_path), 'A')
    cfg_b, merged_b, prov_b, _, _ = load_critic(cfg, str(restore_b), 'B')
    print(f'=== diag_actor_mechanism: n_states={n_states} n_draws={n_draws} n_pool={n_pool} top={n_top} seed={seed}')
    print(f'    A={cfg.restore_path}\n    B={restore_b}\n    epoch={cfg.restore_epoch} forced={FORCED_KEYS}')
    for lab, m in (('A', merged_a), ('B', merged_b)):
        print(f'    {lab}: fb_bc_coeff={m.get("fb_bc_coeff")} bootstrap_source={m.get("bootstrap_source")} '
              f'measure_loss={m.get("measure_loss")} measure_temp={m.get("measure_temp")} '
              f'goal_sampling={m.get("goal_sampling")} goal_discount={m.get("goal_discount")} '
              f'goal_random_frac={m.get("goal_random_frac")} goal_cur_frac={m.get("goal_cur_frac")} '
              f'u_clip={m.get("u_clip")}', flush=True)

    # Dataset rows (s, u_data, g) with the agent's own goal sampling; one reference pool.
    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(str(cfg.env_name), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, cfg_a)
    ds.return_goals = True
    ds.goal_sampling = str(merged_a.get('goal_sampling', 'geometric'))
    ds.goal_discount = float(merged_a.get('goal_discount', 0.98))
    ds.goal_random_frac = float(merged_a.get('goal_random_frac', 0.3))
    ds.goal_cur_frac = float(merged_a.get('goal_cur_frac', 0.0))
    ex = ds.sample(1)
    batch = ds.sample(n_states)
    pool_np = np.asarray(ds.sample(n_pool)['next_observations'], np.float32)
    obs_np = np.asarray(batch['observations'], np.float32)
    goals_np = np.asarray(batch['goals'], np.float32)
    u_clip = float(cfg_a['u_clip'])
    u_data_np = np.clip(np.asarray(batch['noise_preimage'], np.float32), -u_clip, u_clip)
    d_a = u_data_np.shape[-1]

    agent_a = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], cfg_a),
                            str(cfg.restore_path), cfg.restore_epoch)
    agent_b = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], cfg_b),
                            str(restore_b), cfg.restore_epoch)
    critics = {}
    for lab, ag in (('A', agent_a), ('B', agent_b)):
        q_fn, grad_fn, actor_fn, clip_fn = make_fns(ag)
        critics[lab] = {
            'agent': ag, 'q': q_fn, 'grad': grad_fn, 'actor': actor_fn, 'clip': clip_fn,
            'online': (ag.basis.params, ag.w_star.params),
            'target': (ag.target_basis, ag.target_w_star),
        }
    pool = jnp.asarray(pool_np)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    chi2_ref = chi2_ball_reference(d_a, np.random.default_rng(seed + 1))

    per_state = {lab: {'sens': [], 'top': [], 'ctrl': []} for lab in critics}
    cov_rows = {'A_online': [], 'A_target': [], 'B_online': [], 'B_target': []}
    actor_rows = []
    arrays = {'q_data_A': [], 'q_data_B': [], 'mean_q_A': [], 'mean_q_B': [], 'spread_A': [], 'spread_B': [],
              'grad_norm_A': [], 'grad_norm_B': [], 'delta_norm_A': [], 'delta_norm_B': [], 'cos_A': [],
              'diff_A_online': [], 'diff_A_target': [], 'diff_B_online': [], 'diff_B_target': [],
              'top_sep_A': [], 'top_sep_B': [], 'top_kurt_A': [], 'top_kurt_B': []}
    t_loop = time.time()
    for i in range(n_states):
        key, k_eps, k_sens = jax.random.split(key, 3)
        obs1, goal1 = jnp.asarray(obs_np[i]), jnp.asarray(goals_np[i])
        eps = jax.random.normal(k_eps, (n_draws, d_a))                              # unclipped prior draws
        u_bc = critics['A']['clip'](eps)                                            # the BC draws = GPI candidates
        u_sens = critics['A']['clip'](jax.random.normal(k_sens, (n_sens, d_a)))
        u_bc_np = np.asarray(u_bc)
        q_draws = {}
        for lab, c in critics.items():
            bp, wp = c['online']
            # 1. sensitivity
            q_s = np.asarray(c['q'](bp, wp, obs1, goal1, u_sens, pool))
            g_s = np.asarray(c['grad'](bp, wp, obs1, goal1, u_sens[:n_grad_sens], pool))
            q_d = float(np.asarray(c['q'](bp, wp, obs1, goal1, jnp.asarray(u_data_np[i:i + 1]), pool))[0])
            gn = np.linalg.norm(g_s, axis=1)
            per_state[lab]['sens'].append({'mean_q': float(q_s.mean()), 'spread_within': float(q_s.std()),
                                           'grad_norm': float(gn.mean()), 'q_data': q_d,
                                           'grad_norm_over_q': float(gn.mean() / max(q_s.mean(), 1e-12))})
            # 2. top set
            q_all = chunked(c['q'], u_bc, chunk, bp, wp, obs1, goal1, pool)
            q_draws[lab] = q_all
            order = np.argsort(-q_all, kind='stable')
            top_idx = order[:n_top]
            ctrl_idx = rng.choice(n_draws, size=n_top, replace=False)
            top = shape_stats(u_bc_np[top_idx], u_bc_np, rng)
            top['q_threshold'] = float(q_all[top_idx[-1]])
            top['q_top_mean'] = float(q_all[top_idx].mean())
            per_state[lab]['top'].append(top)
            per_state[lab]['ctrl'].append(shape_stats(u_bc_np[ctrl_idx], u_bc_np, rng))
            arrays[f'q_data_{lab}'].append(q_d)
            arrays[f'mean_q_{lab}'].append(float(q_s.mean()))
            arrays[f'spread_{lab}'].append(float(q_s.std()))
            arrays[f'grad_norm_{lab}'].append(float(gn.mean()))
            arrays[f'top_sep_{lab}'].append(top['separation'])
            arrays[f'top_kurt_{lab}'].append(top['kurtosis'])
        # 3. actor coverage with critic A's delta
        ca, cb = critics['A'], critics['B']
        u_act, delta = ca['actor'](ca['agent'].delta.params, obs1, goal1, eps)
        u_act_np, delta_np = np.asarray(u_act), np.asarray(delta)
        _, delta_b = cb['actor'](cb['agent'].delta.params, obs1, goal1, eps)
        bp_a, wp_a = ca['online']
        grad_bc = np.asarray(ca['grad'](bp_a, wp_a, obs1, goal1, u_bc[:n_grad_cov], pool))
        cos = cosine_rows(delta_np[:n_grad_cov], grad_bc)
        top_mask = np.zeros(n_draws, bool)
        top_mask[np.argsort(-q_draws['A'], kind='stable')[:n_top]] = True
        thr = float(np.sort(q_draws['A'])[-n_top])
        q_act = {}
        for name, c, which in (('A_online', ca, 'online'), ('A_target', ca, 'target'),
                               ('B_online', cb, 'online'), ('B_target', cb, 'target')):
            bp, wp = c[which]
            q_new = chunked(c['q'], u_act, chunk, bp, wp, obs1, goal1, pool)
            q_base = q_draws[name[0]] if which == 'online' else chunked(c['q'], u_bc, chunk, bp, wp, obs1, goal1, pool)
            q_act[name] = q_new
            cov_rows[name].append(diff_stats(q_new, q_base))
            arrays[f'diff_{name}'].append(cov_rows[name][-1]['diff_mean'])
        dn = np.linalg.norm(delta_np, axis=1)
        actor_rows.append({
            'delta_norm': float(dn.mean()), 'delta_norm_B': float(np.linalg.norm(np.asarray(delta_b), axis=1).mean()),
            'eps_norm': float(np.linalg.norm(np.asarray(eps), axis=1).mean()),
            'mean_explained': mean_explained_fraction(delta_np),
            'cos_delta_grad': float(cos.mean()), 'cos_pos_frac': float(np.mean(cos > 0)),
            'actor_in_top_nn': nn_membership(u_act_np, u_bc_np, top_mask),
            'bc_in_top_nn_loo': nn_membership_loo(u_bc_np, top_mask),
            'actor_above_threshold': float(np.mean(q_act['A_online'] >= thr)),
            'clip_frac_actor': float(np.mean(np.any(np.abs(u_act_np) >= u_clip - 1e-6, axis=1))),
            'clip_frac_bc': float(np.mean(np.any(np.abs(u_bc_np) >= u_clip - 1e-6, axis=1))),
        })
        arrays['delta_norm_A'].append(actor_rows[-1]['delta_norm'])
        arrays['delta_norm_B'].append(actor_rows[-1]['delta_norm_B'])
        arrays['cos_A'].append(actor_rows[-1]['cos_delta_grad'])
        if i == 0 or (i + 1) % 32 == 0 or i + 1 == n_states:
            el = time.time() - t_loop
            print(f'  state {i + 1}/{n_states}  {el:.0f}s  ({el / (i + 1):.2f}s/state)', flush=True)

    # ------------------------------------------------------------- summaries
    summary = {}
    for lab in critics:
        s = summarise_rows(per_state[lab]['sens'])
        means = np.array([r['mean_q'] for r in per_state[lab]['sens']])
        spread_across = float(means.std(ddof=1)) if means.size > 1 else 0.0
        s['spread_across'] = spread_across
        s['within_over_across'] = float(s['spread_within']['mean'] / max(spread_across, 1e-12))
        summary[lab] = {'sensitivity': s, 'top': summarise_rows(per_state[lab]['top']),
                        'control': summarise_rows(per_state[lab]['ctrl'])}
    summary['actor'] = summarise_rows(actor_rows)
    summary['coverage'] = {k: summarise_rows(v) for k, v in cov_rows.items()}
    summary['chi2_ball_reference'] = chi2_ref

    def f(x, nd=4):
        return 'nan' if x is None else f'{x:.{nd}f}'

    def ms(d, k, nd=4):
        return f'{f(d[k]["mean"], nd)} +- {f(d[k]["sem"], nd)}'

    def sci(d, k):
        m, e = d[k]['mean'], d[k]['sem']
        return 'nan' if m is None else f'{m:.2e} +- {e:.1e}'

    print('\n==== deciding numbers (mean +- SE over states) ====')
    for lab in ('A', 'B'):
        sn, tp, ct = summary[lab]['sensitivity'], summary[lab]['top'], summary[lab]['control']
        print(f'H3 {lab}: spread_within {ms(sn, "spread_within")}  spread_across {f(sn["spread_across"])}  '
              f'within/across {f(sn["within_over_across"], 3)}  grad_norm {ms(sn, "grad_norm")}  '
              f'grad/q {ms(sn, "grad_norm_over_q", 3)}  mean_q {ms(sn, "mean_q")}  q_data {ms(sn, "q_data")}')
        print(f'H1 {lab} top : norm_ratio {ms(tp, "norm_mean_ratio", 3)}  norm_std_ratio {ms(tp, "norm_std_ratio", 3)}  '
              f'kurtosis {ms(tp, "kurtosis", 3)}  separation {ms(tp, "separation", 3)}  '
              f'minor {ms(tp, "minor_share", 3)}  ll_fit-prior {ms(tp, "loglik_fit_minus_prior", 3)}  '
              f'in_ball {ms(tp, "within_std_ball", 3)}  centroid {ms(tp, "centroid_norm", 3)}')
        print(f'H1 {lab} ctrl: norm_ratio {ms(ct, "norm_mean_ratio", 3)}  norm_std_ratio {ms(ct, "norm_std_ratio", 3)}  '
              f'kurtosis {ms(ct, "kurtosis", 3)}  separation {ms(ct, "separation", 3)}  '
              f'minor {ms(ct, "minor_share", 3)}  ll_fit-prior {ms(ct, "loglik_fit_minus_prior", 3)}  '
              f'in_ball {ms(ct, "within_std_ball", 3)}  centroid {ms(ct, "centroid_norm", 3)}')
    ac = summary['actor']
    print(f'chi2 in-ball reference (exact Gaussian): {chi2_ref:.3f}; top_frac control: {top_frac:.3f}')
    print(f'ACTOR A: |delta| {ms(ac, "delta_norm")} (B {ms(ac, "delta_norm_B")}; |eps| {ms(ac, "eps_norm", 3)})  '
          f'mean-explained {ms(ac, "mean_explained", 3)}  clip_frac actor {ms(ac, "clip_frac_actor", 3)} '
          f'bc {ms(ac, "clip_frac_bc", 3)}')
    print(f'H3 link: cos(delta, dq_A/du) {ms(ac, "cos_delta_grad", 3)}  frac>0 {ms(ac, "cos_pos_frac", 3)}')
    print(f'H1 cover: actor in A-top (NN) {ms(ac, "actor_in_top_nn", 3)} vs bc LOO {ms(ac, "bc_in_top_nn_loo", 3)}; '
          f'actor above top threshold {ms(ac, "actor_above_threshold", 3)} vs {top_frac:.3f}')
    for name in ('A_online', 'A_target', 'B_online', 'B_target'):
        cv = summary['coverage'][name]
        print(f'H2 {name}: q(u\')-q(u_bc) {sci(cv, "diff_mean")}  rel {sci(cv, "diff_rel")}  '
              f'frac>0 {ms(cv, "frac_pos", 3)}  q_bc {ms(cv, "q_base_mean")}  q_actor {ms(cv, "q_new_mean")}')

    report = {
        'tool': 'diag_actor_mechanism',
        'what': __doc__.split('\n\n')[0],
        'restore_path_a': str(cfg.restore_path), 'restore_path_b': str(restore_b),
        'restore_epoch': int(cfg.restore_epoch), 'seed': seed, 'env_name': str(cfg.env_name),
        'n_states': n_states, 'n_draws': n_draws, 'n_pool': n_pool, 'n_top': n_top, 'top_frac': top_frac,
        'n_sens_draws': n_sens, 'n_grad_draws': n_grad_sens, 'n_grad_cov_draws': n_grad_cov,
        'action_dim': int(d_a), 'u_clip': u_clip, 'forced_keys': FORCED_KEYS,
        'goal_sampling': {'goal_sampling': ds.goal_sampling, 'goal_discount': ds.goal_discount,
                          'goal_random_frac': ds.goal_random_frac, 'goal_cur_frac': ds.goal_cur_frac},
        'summary': summary,
        'per_state': {lab: per_state[lab] for lab in critics},
        'per_state_actor': actor_rows,
        'per_state_coverage': cov_rows,
        'arrays': {k: [float(x) for x in v] for k, v in arrays.items()},
        'agent_config_a': json.loads(json.dumps(merged_a, default=str)),
        'agent_config_b': json.loads(json.dumps(merged_b, default=str)),
        'config_provenance': {'A': prov_a, 'B': prov_b},
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, 'diag_actor_mechanism.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
