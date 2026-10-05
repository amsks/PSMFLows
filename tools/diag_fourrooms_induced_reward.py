"""Four-rooms check of the cube "induced reward" effect with the real psmgoal measure.

Cube measurement being replicated: acting greedily on the measure M(s,u,s+) = phi^T w + b at
the goal states scores near behaviour cloning, while a scalar critic trained by TD on a reward
read out of the same frozen M scores like a scalar critic on the real reward.

Stages (--stage):
  exact    no learning. Exact successor measure of the uniform-random policy; greedy on it at
           the goal, exact Q-iteration on the reward read out of it, on the true reward, random.
  learned  one (--enc, --seed) run. Trains the REAL agents.psmgoal.PSMGoalAgent measure
           (apply_update: squared TD on the state x next-state mesh, default policy codes
           keyed on the dataset row, default RMSNorm head) on uniform-random four-rooms data,
           then for each of 8 goals compares M as critic against scalar critics.
  aggregate  collects the per-run JSONs into one report.
  figures    per-cell heat maps and the shortest-path line plot for one fixed goal.

Encodings: onehot (states repeat) and xyphase (cell coordinates plus a phase that advances
every step: every state is unique and s' is an exact function of (s, a), as on cube).

Definitions mirrored from the cube code, not re-invented:
  coefficient   PSMGoalAgent.infer_eval_goals with coef_source=regression and coef_source=lp
                (eval_goal_pool=dataset: 32 rewarding dataset next states, the Lagrangian on the
                cached mesh). Called as is.
  M as critic   select_latent's score mean_g M(s, a, g; w) with the 4 one-hot actions in place
                of the 64 prior draws; greedy = argmax over the 4 actions.
  induced reward, goal-set readout     tools.relabel_reward_rhat_psmgoal.rhat_goal_set (called):
                r_hat(row) = mean_g M(s'_row, a_row, g; w).
  induced reward, feature readout      that tool's --readout regression (the readout the cube
                0.876 critic was trained on): r_hat = f(s')^T w, f = PsmgoalProjectedPhi (phi
                averaged over actions and 32 dataset states, sphere-normalised), w = least
                squares on the real reward over all rows. U is the 4 one-hot actions here.
  rescaling     tools.relabel_reward_rhat.scale_to_real (called), reward_shift 1: the reward
                the critic sees is a*r_hat + b - 1 on the -1/0 scale.
Scalar critics (offline Q-learning, max over the 4 next actions; tabular Q-iteration on the
dataset for onehot, a small Q-network for xyphase):
  true          real reward (-1, 0 on entering the goal), true episode-end mask
  const_mask    reward -1 on every row, true episode-end mask (control: what the mask alone gives)
  <r>_mask      induced reward on the -1/0 scale, true episode-end mask (the cube pipeline: the
                reward file replaces only the rewards)
  <r>_nomask    induced reward alone: z-scored, sign from the affine fit, no episode end
with <r> in gs_reg, gs_lp (goal-set readout at the regression / Lagrangian w) and feat.

Run:
  JAX_PLATFORMS=cpu .venv/bin/python tools/diag_fourrooms_induced_reward.py --stage exact \
      --report_out $PSM_DATA/logs/fourrooms_induced_reward/exact.json
  .venv/bin/python tools/diag_fourrooms_induced_reward.py --stage learned --enc xyphase --seed 0 \
      --report_out $PSM_DATA/logs/fourrooms_induced_reward/xyphase_sd0.json
  .venv/bin/python tools/diag_fourrooms_induced_reward.py --stage aggregate \
      --run_dir $PSM_DATA/logs/fourrooms_induced_reward \
      --report_out $PSM_DATA/logs/diag_fourrooms_induced_reward.json
  .venv/bin/python tools/diag_fourrooms_induced_reward.py --stage figures \
      --run_dir $PSM_DATA/logs/fourrooms_induced_reward \
      --fig_dir $PSM_DATA/logs/figures/fourrooms_induced_reward
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import utils.xla_guard  # noqa: F401,E402  -- MUST precede jax (jax is imported inside run_learned)

W = 11                                                  # grid side
A = 4
MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])    # right, left, down, up as (d_row, d_col)
DOORS = ((2, 5), (8, 5), (5, 2), (5, 8))
# two goal cells per room: top-left, top-right, bottom-left, bottom-right
GOAL_RC = ((1, 1), (3, 3), (1, 9), (3, 7), (7, 1), (9, 3), (8, 8), (10, 10))
FIG_GOAL = 6                                            # (8, 8), bottom-right room
FIG_START_RC = (0, 0)                                   # top-left room
ALPHA = 0.6180339887498949                              # phase advance per step (xyphase)
N_PHASE = 16                                            # phases per cell for the per-cell tables
MAX_STEPS = 100
N_STARTS = 200
NEAR, FAR = 3, 10


# ------------------------------------------------------------------ pure helpers (numpy)
def fourrooms_layout():
    """11x11 four-rooms mask (True = free): a wall row and a wall column through the middle,
    four single-cell doorways. 104 free cells (Factored-FB docs/notes/2026-09-09)."""
    free = np.ones((W, W), bool)
    free[W // 2, :] = False
    free[:, W // 2] = False
    for r, c in DOORS:
        free[r, c] = True
    return free


def build_grid():
    """(free, cells_rc (S,2), idx (W,W; -1 on walls), T (S,4) next cell; a blocked move stays)."""
    free = fourrooms_layout()
    cells = np.argwhere(free)
    idx = -np.ones((W, W), np.int64)
    idx[free] = np.arange(len(cells))
    T = np.zeros((len(cells), A), np.int64)
    for s, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(MOVES):
            r2, c2 = r + dr, c + dc
            ok = 0 <= r2 < W and 0 <= c2 < W and free[r2, c2]
            T[s, a] = idx[r2, c2] if ok else s
    return free, cells, idx, T


def bfs_dist(T, goal):
    """Steps to the goal cell under the best moves, (S,)."""
    S = T.shape[0]
    dist = np.full(S, np.inf)
    dist[goal] = 0
    frontier = [goal]
    while frontier:
        new = []
        for s in range(S):
            if np.isinf(dist[s]) and any(T[s, a] in frontier for a in range(A)):
                new.append(s)
        d = dist[frontier[0]] + 1
        for s in new:
            dist[s] = d
        frontier = new
    return dist


def exact_measure(T, pi, gamma):
    """Exact M(s, a, s+) = (1-gamma) sum_t gamma^t P(s_{t+1} = s+ | s, a, then pi). Rows sum to 1."""
    S = T.shape[0]
    P = np.zeros((S, S))
    for a in range(A):
        np.add.at(P, (np.arange(S), T[:, a]), pi[:, a])
    Mpi = (1 - gamma) * np.linalg.inv(np.eye(S) - gamma * P)
    return (1 - gamma) * np.eye(S)[T] + gamma * Mpi[T]


def q_iteration(T, r_sa, mask_sa, gamma, iters=3000):
    """Tabular Q-learning fixed point: Q(s,a) = r(s,a) + gamma * mask(s,a) * max_a' Q(T(s,a), a')."""
    Q = np.zeros_like(r_sa, dtype=np.float64)
    for _ in range(iters):
        Q = r_sa + gamma * mask_sa * Q[T].max(-1)
    return Q


def rows_to_table(st, ac, values, S):
    """Mean of a per-row value over the dataset rows of each (cell, action), (S, A)."""
    tot = np.zeros((S, A))
    cnt = np.zeros((S, A))
    np.add.at(tot, (st, ac), values)
    np.add.at(cnt, (st, ac), 1.0)
    assert (cnt > 0).all(), 'a (cell, action) pair has no dataset row'
    return tot / cnt


def cell_mean(nx, values, S):
    """Mean of a per-row value over the rows whose NEXT cell is each cell, (S,)."""
    tot = np.bincount(nx, weights=values, minlength=S)
    cnt = np.bincount(nx, minlength=S)
    return tot / np.maximum(cnt, 1)


def affine_fit(r_hat, r_true, reward_shift):
    """Least-squares (a, b) of the shifted reward on r_hat, and the correlation; the numbers
    tools.relabel_reward_rhat.relabel_stats reports (called there for the learned stage)."""
    from tools.relabel_reward_rhat import relabel_stats
    s = relabel_stats(r_hat, r_true, reward_shift)
    return s['affine_scale'], s['affine_offset'], s['corr']


def reward_variants(readouts, r_true, done):
    """Per-row (reward, mask) of every scalar critic. `readouts`: {name: r_hat rows};
    r_true rows in {-1, 0}; done rows = 1 where the row enters the goal cell. Also returns the
    affine fit and correlation of each readout."""
    from tools.relabel_reward_rhat import scale_to_real
    keep = 1.0 - done
    out = {'true': (r_true.astype(np.float64), keep),
           'const_mask': (-np.ones_like(r_true, dtype=np.float64), keep)}
    fits = {}
    for name, r_hat in readouts.items():
        r_scaled, a, b = scale_to_real(r_hat, r_true, 1.0)
        _, _, corr = affine_fit(r_hat, r_true, 1.0)
        fits[name] = {'affine_scale': float(a), 'affine_offset': float(b), 'corr': float(corr)}
        out[name + '_mask'] = (np.asarray(r_scaled, np.float64), keep)
        z = (r_hat - r_hat.mean()) / (r_hat.std() + 1e-12) * np.sign(a)
        out[name + '_nomask'] = (np.asarray(z, np.float64), np.ones_like(keep))
    return out, fits


def rollout(T, act_fn, starts, phases, goal, max_steps=MAX_STEPS, dist=None):
    """Batched episodes. act_fn(cells, phases) -> actions. Success = entered the goal cell.
    Returns (success bool (n,), steps (n,) with max_steps on failure, the smallest distance to
    the goal seen along each episode (n,) or None without `dist`)."""
    cells = np.array(starts)
    phases = np.array(phases, dtype=np.float64)
    done = np.zeros(len(cells), bool)
    steps = np.full(len(cells), max_steps)
    min_dist = None if dist is None else dist[cells].copy()
    for t in range(max_steps):
        a = np.asarray(act_fn(cells, phases))
        cells = np.where(done, cells, T[cells, a])
        phases = (phases + ALPHA) % 1.0
        if dist is not None:
            min_dist = np.minimum(min_dist, dist[cells])
        newly = (~done) & (cells == goal)
        steps[newly] = t + 1
        done |= newly
        if done.all():
            break
    return done, steps, min_dist


def rollout_summary(done, steps, min_dist=None):
    out = {'success': float(done.mean()),
           'steps_success_mean': float(steps[done].mean()) if done.any() else None,
           'steps_all_mean': float(steps.mean())}
    if min_dist is not None:
        out['reach_within1'] = float((min_dist <= 1).mean())   # came next to the goal (or entered it)
    return out


def shortest_share(T, dist, greedy, goal):
    """Share of non-goal (cell[, phase]) entries whose greedy move lowers the distance to the
    goal by one. greedy: (S,) or (S, P)."""
    g2 = greedy.reshape(len(greedy), -1)
    nxt = T[np.arange(T.shape[0])[:, None], g2]
    ok = dist[nxt] == (dist[:, None] - 1)
    keep = np.arange(T.shape[0]) != goal
    return float(ok[keep].mean())


def argmax_agree(g1, g2, goal):
    a = g1.reshape(len(g1), -1) == g2.reshape(len(g2), -1)
    keep = np.arange(len(g1)) != goal
    return float(a[keep].mean())


def gap_over_spread(score, dist, goal):
    """Best minus second-best action value, divided by the spread (std over non-goal entries)
    of the best action value; mean over cells within NEAR steps of the goal and over cells
    more than FAR steps away. score: (S, A) or (S, P, A)."""
    sc = score.reshape(score.shape[0], -1, A)
    top = np.sort(sc, -1)
    gap = top[..., -1] - top[..., -2]
    keep = np.arange(len(sc)) != goal
    spread = top[..., -1][keep].std() + 1e-12
    near = keep & (dist <= NEAR)
    far = dist > FAR
    return {'near': float(gap[near].mean() / spread),
            'far': float(gap[far].mean() / spread) if far.any() else None,
            'n_near_cells': int(near.sum()), 'n_far_cells': int(far.sum())}


def sample_starts(seed, goal_i, goal, S):
    r = np.random.default_rng(100000 + 1000 * seed + goal_i)
    return r.choice(np.setdiff1d(np.arange(S), [goal]), N_STARTS), r.random(N_STARTS)


def shortest_path_cells(T, dist, start, goal):
    """Cells of one shortest path start -> goal (first best move at each cell)."""
    path, s = [start], start
    while s != goal:
        s = T[s, int(np.argmin(dist[T[s]]))]
        path.append(s)
    return np.array(path)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1)
    print('report ->', path, flush=True)


# ------------------------------------------------------------------ stage: exact
def run_exact(args):
    free, cells, idx, T = build_grid()
    S = len(cells)
    gamma = args.gamma
    Msa = exact_measure(T, np.full((S, A), 0.25), gamma)       # (S, A, S+), random policy
    rho = np.full(S, 1.0 / S)        # stationary state distribution of the random walk (uniform)
    s_rows = np.repeat(np.arange(S), A)                         # every (cell, action) once
    a_rows = np.tile(np.arange(A), S)
    nx_rows = T[s_rows, a_rows]
    F = Msa.mean(1) / rho[None]                                 # (S, S) action-averaged feature
    goals_out, maps = [], {}
    for gi, rc in enumerate(GOAL_RC):
        g = int(idx[rc])
        dist = bfs_dist(T, g)
        score = Msa[:, :, g] / rho[g]                           # M read at the goal, (S, A)
        r_true = (nx_rows == g).astype(np.float64) - 1.0
        done = (nx_rows == g).astype(np.float64)
        f_rows = F[nx_rows]
        w_feat = np.linalg.lstsq(f_rows, r_true + 1.0, rcond=None)[0]
        readouts = {'gs': score[nx_rows, a_rows], 'feat': f_rows @ w_feat}
        variants, fits = reward_variants(readouts, r_true, done)
        Q = {k: q_iteration(T, r.reshape(S, A), m.reshape(S, A), gamma) for k, (r, m) in variants.items()}
        greedy = {'m_exact': score.argmax(1), **{k: q.argmax(1) for k, q in Q.items()}}
        pol = {}
        for name, gr in greedy.items():
            runs = []
            for sd in range(args.n_seeds):
                starts, phases = sample_starts(sd, gi, g, S)
                runs.append(rollout_summary(*rollout(T, lambda c, p, gr=gr: gr[c], starts, phases, g, dist=dist)))
            pol[name] = {'per_seed': runs, 'shortest_share': shortest_share(T, dist, gr, g),
                         'agree_true_argmax': argmax_agree(gr, greedy['true'], g)}
        runs = []
        for sd in range(args.n_seeds):
            starts, phases = sample_starts(sd, gi, g, S)
            rr = np.random.default_rng(555 + 1000 * sd + gi)
            runs.append(rollout_summary(*rollout(T, lambda c, p: rr.integers(0, A, len(c)), starts, phases, g, dist=dist)))
        pol['random'] = {'per_seed': runs}
        goals_out.append({'goal_rc': list(rc), 'policies': pol, 'reward_fit': fits,
                          'm_gap': {'m_exact': gap_over_spread(score, dist, g)},
                          'bfs_mean_steps_from_starts': float(np.mean(
                              [dist[sample_starts(sd, gi, g, S)[0]].mean() for sd in range(args.n_seeds)]))})
        maps[f'score_m_exact_{gi}'] = score
        for k, (r, _) in variants.items():
            maps[f'rmap_{k}_{gi}'] = cell_mean(nx_rows, r, S)
        for k, q in Q.items():
            maps[f'q_{k}_{gi}'] = q
    res = {'stage': 'exact', 'gamma': gamma, 'n_cells': int(S), 'goals_rc': [list(g) for g in GOAL_RC],
           'n_starts': N_STARTS, 'max_steps': MAX_STEPS, 'n_seeds': args.n_seeds,
           'note': 'rho is uniform; each (cell, action) pair counted once in the reward fit and the Q-iteration. '
                   'feat readout: least squares of the 0/1 reward on the action-averaged exact measure '
                   '(104 features for 104 cells), so it is close to the true reward.',
           'goals': goals_out}
    write_json(args.report_out, res)
    np.savez(args.report_out[:-5] + '_maps.npz', **maps)
    return res


# ------------------------------------------------------------------ stage: learned
def run_learned(args):
    import flax
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax
    import yaml

    from agents.psmgoal import PSMGoalAgent
    from tools.relabel_reward_rhat import relabel_stats
    from tools.relabel_reward_rhat_psmgoal import rhat_goal_set
    from utils.psm_networks import PsmgoalProjectedPhi
    from utils.psm_proto import proto_latents, proto_seed_ints, sample_z_bin

    t_start = time.time()
    free, cells_rc, idx, T = build_grid()
    S = len(cells_rc)
    gamma, D, LOG_SEED, U_CLIP = args.gamma, args.D, args.log_seed, 3.0
    enc = args.enc
    print('devices:', jax.devices(), flush=True)

    # ---- data: uniform-random behaviour, episodes from uniform start cells
    rng = np.random.default_rng(args.seed)
    st, ac, nx, ph = [], [], [], []
    for _ in range(args.n_ep):
        s, th = rng.integers(S), rng.random()
        for _t in range(args.ep_len):
            a = rng.integers(A)
            st.append(s)
            ac.append(a)
            ph.append(th)
            s = T[s, a]
            nx.append(s)
            th = (th + ALPHA) % 1.0
    st, ac, nx, ph = map(np.asarray, (st, ac, nx, ph))
    nph = (ph + ALPHA) % 1.0
    Nd = len(st)
    rho = np.bincount(nx, minlength=S) / Nd

    def encode(cells, phases):
        cells = np.asarray(cells)
        if enc == 'onehot':
            return np.eye(S, dtype=np.float32)[cells]
        rc = cells_rc[cells].astype(np.float32) / (W - 1) * 2 - 1
        ang = 2 * np.pi * np.asarray(phases)
        return np.concatenate([rc, np.cos(ang)[:, None], np.sin(ang)[:, None]], -1).astype(np.float32)

    obs_d, nobs_d = encode(st, ph), encode(nx, nph)
    act_d = np.eye(A, dtype=np.float32)[ac]
    OB = obs_d.shape[1]
    base_key = jax.random.PRNGKey(0)                     # psmgoal proto_seed default 0

    def proto_action(z_bits, key_index):
        """psmgoal's fixed z-indexed bootstrap policy, discrete: argmax of the 4-dim proto latent."""
        seeds = proto_seed_ints(z_bits, key_index, LOG_SEED)
        return jnp.argmax(proto_latents(seeds, A, U_CLIP, base_key), -1)

    # ---- the real agent
    cfg = yaml.safe_load(open(f'{REPO}/configs/agent/psmgoal.yaml'))
    cfg.update(max_log_seed=LOG_SEED, z_dim=D, discount=gamma, batch_size=args.batch,
               allow_untrained_flow=True, flow_ckpt_path=None, eval_goal_pool='dataset')
    if args.lp_steps > 0:
        cfg['num_inference_steps'] = args.lp_steps
    cfg['measure'] = {'hidden_dim': args.hidden, 'hidden_layers': 2}
    cfg['flow'] = {'hidden_dims': [32, 32], 'value_hidden_dims': [32, 32], 'layer_norm': False,
                   'critic_layer_norm': True}
    assert cfg['measure_loss'] == 'squared' and cfg['policy_index'] == 'code' and cfg['measure_form'] == 'joint'
    agent = PSMGoalAgent.create(args.seed, jnp.asarray(obs_d[:1]), jnp.asarray(act_d[:1]), cfg)
    hp = {k: cfg[k] for k in ('z_dim', 'discount', 'tau', 'lr_measure', 'lr_w', 'ortho_coef', 'max_log_seed',
                              'u_clip', 'batch_size', 'k_goals', 'infer_batch', 'num_inference_steps',
                              'lr_infer', 'lr_l', 'use_dgd', 'eval_goal_pool', 'measure_loss', 'policy_index',
                              'measure_form')}
    hp.update(measure=cfg['measure'], w=cfg['w'], l=cfg['l'], keying=args.keying, dataset_rows=int(Nd),
              q_steps=args.q_steps, q_lr=3e-4, q_tau=0.005, q_hidden=256, q_batch=256, relabel_size=args.relabel_size)
    print('hyperparameters:', json.dumps({**vars(args), **hp}), flush=True)

    @jax.jit
    def train_step(agent, obs, nobs, act, key_index, k):
        z = sample_z_bin(k, obs.shape[0], LOG_SEED)
        u_next = jax.nn.one_hot(proto_action(z, key_index), A)
        batch = {'observations': obs, 'next_observations': nobs, 'noise_preimage': act}
        return agent.apply_update(batch, z, u_next)

    # ---- train the measure (or reuse a saved one)
    ckpt = args.report_out[:-5] + '_measure.pkl'
    row_key = nx if args.keying == 'state' else np.arange(Nd)
    key = jax.random.PRNGKey(args.seed)
    log = []
    if args.reuse_measure and os.path.exists(ckpt):
        with open(ckpt, 'rb') as fh:
            saved = pickle.load(fh)
        agent = agent.replace(basis=agent.basis.replace(params=saved['basis']),
                              w=agent.w.replace(params=saved['w']))
        log = saved['log']
        print('reused measure', ckpt, flush=True)
    else:
        t0 = time.time()
        for it in range(args.steps):
            bi = rng.integers(0, Nd, args.batch)
            key, k = jax.random.split(key)
            agent, info = train_step(agent, jnp.asarray(obs_d[bi]), jnp.asarray(nobs_d[bi]),
                                     jnp.asarray(act_d[bi]), jnp.asarray(row_key[bi]), k)
            if it % 500 == 0 or it == args.steps - 1:
                rec = {k_: float(v) for k_, v in info.items() if np.ndim(v) == 0}
                rec.update(step=it, sec=time.time() - t0)
                log.append(rec)
                print(json.dumps(rec), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
        with open(ckpt, 'wb') as fh:
            pickle.dump({'basis': jax.tree_util.tree_map(np.asarray, agent.basis.params),
                         'w': jax.tree_util.tree_map(np.asarray, agent.w.params), 'log': log}, fh)
    if args.stop_after_train:
        print('measure saved, stopping (--stop_after_train):', ckpt, flush=True)
        return None
    bp = agent.basis.params

    # ---- learned M against the exact M of the bootstrap policy actually used
    n_codes = 16
    eval_codes = np.random.default_rng(1234).integers(0, 2 ** LOG_SEED, n_codes)
    eval_bits = ((eval_codes[:, None] >> np.arange(LOG_SEED)) & 1).astype(np.float32)
    R = []
    for zb in eval_bits:
        zb_j = jnp.asarray(zb)[None]
        if args.keying == 'state':
            pi = np.eye(A)[np.asarray(proto_action(jnp.repeat(zb_j, S, 0), jnp.arange(S)))]
        else:                                    # rows landing in a cell each carry their own action
            acts = np.asarray(proto_action(jnp.repeat(zb_j, Nd, 0), jnp.arange(Nd)))
            pi = np.zeros((S, A))
            np.add.at(pi, (nx, acts), 1.0)
            pi = pi / np.maximum(pi.sum(1, keepdims=True), 1)
        R.append(exact_measure(T, pi, gamma) / rho[None, None, :])
    R = np.stack(R)                                              # (Z, S, A, S+) loss fixed point
    wz = np.asarray(agent.w(jnp.asarray(eval_bits)))             # (Z, D)
    ss, aa, gg = [x.ravel() for x in np.meshgrid(np.arange(S), np.arange(A), np.arange(S), indexing='ij')]
    mrng = np.random.default_rng(4321)

    @jax.jit
    def phi_b(o, a, g):
        return agent.basis(o, a, g, params=bp)

    def model_M(o, a, g, chunk=65536):
        out = []
        for lo in range(0, len(o), chunk):
            p_, b_ = phi_b(jnp.asarray(o[lo:lo + chunk]), jnp.asarray(a[lo:lo + chunk]), jnp.asarray(g[lo:lo + chunk]))
            out.append(np.asarray(p_) @ wz.T + np.asarray(b_)[:, None])
        return np.concatenate(out)                               # (n, Z)

    L = model_M(encode(ss, mrng.random(len(ss))), np.eye(A, dtype=np.float32)[aa],
                encode(gg, mrng.random(len(gg)))).T.reshape(n_codes, S, A, S)
    Rf, Lf = R.ravel(), L.ravel()
    own = (gg.reshape(S, A, S) == T[:, :, None])[None].repeat(n_codes, 0)
    ri = mrng.integers(0, Nd, 2000)
    by_cell = {c: np.flatnonzero(nx == c) for c in range(S)}
    rk = np.array([mrng.choice(by_cell[nx[i]]) for i in ri])
    rj = mrng.integers(0, Nd, 2000)
    m_own = model_M(obs_d[ri], act_d[ri], nobs_d[ri])
    m_same = model_M(obs_d[ri], act_d[ri], nobs_d[rk])
    m_rand = model_M(obs_d[ri], act_d[ri], nobs_d[rj])
    phi_all = np.asarray(phi_b(jnp.asarray(obs_d[ri]), jnp.asarray(act_d[ri]), jnp.asarray(nobs_d[rj]))[0])
    gram = phi_all.T @ phi_all / len(phi_all)
    wn = wz / np.linalg.norm(wz, axis=1, keepdims=True)
    m_diag = {
        'corr_learned_vs_exact': float(np.corrcoef(Rf, Lf)[0, 1]),
        'slope_learned_on_exact': float((Lf @ Rf) / (Rf @ Rf)),
        'cell_mesh': {'learned_own_next_cell_mean': float(L[own].mean()), 'learned_other_cell_mean': float(L[~own].mean()),
                      'exact_own_next_cell_mean': float(R[own].mean()), 'exact_other_cell_mean': float(R[~own].mean())},
        'dataset_rows': {'M_at_own_next_state': float(m_own.mean()),
                         'M_at_other_row_same_cell': float(m_same.mean()),
                         'M_at_random_row_next_state': float(m_rand.mean()),
                         'exact_at_own_next_cell': float(np.mean([R[:, st[i], ac[i], nx[i]].mean() for i in ri]))},
        'head_bound': D + 1.0, 'learned_max': float(L.max()), 'learned_min': float(L.min()),
        'phi_eff_rank': float(np.trace(gram) ** 2 / max((gram ** 2).sum(), 1e-12)),
        'w_mean_pairwise_cos': float((wn @ wn.T)[np.triu_indices(n_codes, 1)].mean()),
    }
    print('M diagnostics:', json.dumps(m_diag), flush=True)

    # ---- per-cell query states and the M-as-critic score
    if enc == 'onehot':
        P = 1
        q_cells, q_phase = np.arange(S), np.zeros(S)
    else:
        P = N_PHASE
        q_cells = np.repeat(np.arange(S), P)
        q_phase = np.tile((np.arange(P) + 0.5) / P, S)
    q_obs = jnp.asarray(encode(q_cells, q_phase))                # (S*P, OB)
    eyeA = jnp.eye(A)

    @jax.jit
    def goal_mean_phi_b(obs, goals):
        B = obs.shape[0]
        phi, b = agent.mesh_phi_b(jnp.repeat(obs, A, axis=0), jnp.tile(eyeA, (B, 1)), goals, params=bp)
        return phi.mean(axis=1).reshape(B, A, -1), b.mean(axis=1).reshape(B, A)

    def score_fn(obs, goals, w):
        """select_latent's score for the 4 actions, (B, A): mean_g M(s, a, g; w) =
        (mean_g phi)^T w + mean_g b, the goal mean of mesh_M."""
        pm, bm = goal_mean_phi_b(obs, goals)
        return np.asarray(pm) @ np.asarray(w) + np.asarray(bm)

    wbar_train = wz.mean(0)

    def score_trainw(obs, goals):
        """The same score averaged over the 16 held-out training codes' w(z) (M is linear in w)."""
        return score_fn(obs, goals, wbar_train)

    def table(x):                                                # (S*P, A) -> (S, P, A)
        return np.asarray(x).reshape(S, P, A)

    def modal(greedy_sp):                                        # (S, P) -> most frequent action per cell
        return np.array([np.bincount(r, minlength=A).argmax() for r in greedy_sp])

    # ---- feature readout (goal independent): f(s') over all rows
    f_def = PsmgoalProjectedPhi(z_dim=D, measure_hidden_dim=args.hidden, measure_hidden_layers=2,
                                U=eyeA, G=jnp.asarray(obs_d[:32]))
    f_apply = jax.jit(lambda o: f_def.apply({'params': {'measure': bp}}, o))
    f_all = np.concatenate([np.asarray(f_apply(jnp.asarray(nobs_d[lo:lo + 2048])))
                            for lo in range(0, Nd, 2048)]).astype(np.float64)

    # ---- per goal: coefficients, M as critic, induced rewards
    coef_agents = {c: agent.replace(config=flax.core.FrozenDict({**agent.config, 'coef_source': c}))
                   for c in ('regression', 'lp')}
    goal_cells = [int(idx[rc]) for rc in GOAL_RC]
    per_goal, maps, critic_rows = [], {}, {}
    m_tables, m_act = {}, {}
    for gi, g in enumerate(goal_cells):
        tg = time.time()
        dist = bfs_dist(T, g)
        r01 = (nx == g).astype(np.float32)
        r_true = r01 - 1.0
        np.random.seed(7919 * args.seed + gi)
        zb_idx = np.random.choice(Nd, size=min(Nd, args.relabel_size), replace=False)
        zb = {'observations': obs_d[zb_idx], 'next_observations': nobs_d[zb_idx], 'noise_preimage': act_d[zb_idx]}
        pool = {'next_observations': nobs_d, 'rewards': r01}
        readouts, rec = {}, {'goal_rc': list(GOAL_RC[gi]), 'n_goal_rows': int(r01.sum()), 'w': {}, 'm_gap': {},
                             'policies': {}}
        ws = {}
        for coef, short in (('regression', 'reg'), ('lp', 'lp')):
            np.random.seed(104729 * args.seed + gi)              # same goal set and rows for both paths
            ag = coef_agents[coef].infer_eval_goals(zb, r01[zb_idx], goal_pool=pool)
            goals, w = ag.eval_goals, ag.eval_w_star
            ws[short] = np.asarray(w)
            m_tables[(f'm_{short}', gi)] = table(score_fn(q_obs, goals, w))
            m_act[(f'm_{short}', gi)] = (lambda c, p, goals=goals, w=w: np.asarray(
                score_fn(jnp.asarray(encode(c, p)), goals, w)).argmax(1))
            readouts[f'gs_{short}'] = np.asarray(
                rhat_goal_set(ag, nobs_d, act_d, goals, w, batch_size=4096), np.float64)
            st_ = relabel_stats(readouts[f'gs_{short}'], r_true, 1.0)
            rec.setdefault('reward_stats', {})[f'gs_{short}'] = {k: st_[k] for k in (
                'corr', 'r2_affine', 'affine_scale', 'affine_offset', 'top_precision', 'base_rate')}
        m_tables[('m_trainw', gi)] = table(score_trainw(q_obs, goals))
        m_act[('m_trainw', gi)] = (lambda c, p, goals=goals: np.asarray(
            score_trainw(jnp.asarray(encode(c, p)), goals)).argmax(1))
        w_feat = np.linalg.lstsq(f_all, r01.astype(np.float64), rcond=None)[0]
        readouts['feat'] = f_all @ w_feat
        st_ = relabel_stats(readouts['feat'], r_true, 1.0)
        rec['reward_stats']['feat'] = {k: st_[k] for k in (
            'corr', 'r2_affine', 'affine_scale', 'affine_offset', 'top_precision', 'base_rate')}
        wbar = wz.mean(0)

        def cos(x, y):
            return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

        rec['w'] = {'cos_reg_lp': cos(ws['reg'], ws['lp']), 'cos_reg_trainmean': cos(ws['reg'], wbar),
                    'cos_lp_trainmean': cos(ws['lp'], wbar)}
        variants, fits = reward_variants(readouts, r_true, r01.astype(np.float64))
        rec['reward_fit'] = fits
        critic_rows[gi] = variants
        for k, (r, _) in variants.items():
            maps[f'rmap_{k}_{gi}'] = cell_mean(nx, r, S)
        per_goal.append(rec)
        print(f'goal {gi} {GOAL_RC[gi]} coefficients+readouts {time.time() - tg:.1f}s '
              f'corr {json.dumps({k: round(v["corr"], 3) for k, v in rec["reward_stats"].items()})}', flush=True)
    var_names = list(critic_rows[0].keys())

    # ---- scalar critics
    q_tables, q_act, q_log = {}, {}, None
    if enc == 'onehot':
        for gi in range(len(goal_cells)):
            for k, (r, m) in critic_rows[gi].items():
                q = q_iteration(T, rows_to_table(st, ac, r, S), rows_to_table(st, ac, m, S), gamma)
                q_tables[(k, gi)] = q[:, None, :]
                q_act[(k, gi)] = (lambda c, p, gr=q.argmax(1): gr[c])
    else:
        inst = [(k, gi) for gi in range(len(goal_cells)) for k in var_names]
        Rw = jnp.asarray(np.stack([critic_rows[gi][k][0] for k, gi in inst]), jnp.float32)   # (I, Nd)
        Mk = jnp.asarray(np.stack([critic_rows[gi][k][1] for k, gi in inst]), jnp.float32)
        O, NO, AI = jnp.asarray(obs_d), jnp.asarray(nobs_d), jnp.asarray(ac)

        class QNet(nn.Module):
            @nn.compact
            def __call__(self, x):
                for _ in range(2):
                    x = nn.relu(nn.Dense(256)(x))
                return nn.Dense(A)(x)

        qdef = QNet()
        tx = optax.adam(3e-4)
        qkey = jax.random.PRNGKey(10_000 + args.seed)
        params = jax.vmap(lambda k: qdef.init(k, jnp.zeros((1, OB))))(jax.random.split(qkey, len(inst)))
        carry = (params, jax.tree_util.tree_map(jnp.array, params), jax.vmap(tx.init)(params))

        def step_one(p, tp, os_, o, a, r, m, no):
            def loss(pp):
                q = jnp.take_along_axis(qdef.apply(pp, o), a[:, None], 1)[:, 0]
                tq = r + gamma * m * qdef.apply(tp, no).max(-1)
                return jnp.mean((q - jax.lax.stop_gradient(tq)) ** 2)
            l, g = jax.value_and_grad(loss)(p)
            upd, os_ = tx.update(g, os_, p)
            p = optax.apply_updates(p, upd)
            tp = jax.tree_util.tree_map(lambda t, q_: t * (1 - 0.005) + q_ * 0.005, tp, p)
            return p, tp, os_, l

        @jax.jit
        def chunk(carry, key):
            def body(c, k):
                bi = jax.random.randint(k, (256,), 0, Nd)
                p, tp, os_, l = jax.vmap(step_one, in_axes=(0, 0, 0, None, None, 0, 0, None))(
                    c[0], c[1], c[2], O[bi], AI[bi], Rw[:, bi], Mk[:, bi], NO[bi])
                return (p, tp, os_), l
            return jax.lax.scan(body, carry, jax.random.split(key, 500))

        q_log = []
        tq0 = time.time()
        for ci in range(max(1, args.q_steps // 500)):
            carry, losses = chunk(carry, jax.random.fold_in(qkey, ci))
            lm = np.asarray(losses).mean(0).reshape(len(goal_cells), len(var_names)).mean(0)
            q_log.append({'step': (ci + 1) * 500, 'sec': time.time() - tq0,
                          **{f'td_loss_{k}': float(v) for k, v in zip(var_names, lm)}})
            if ci % 10 == 0:
                print(json.dumps(q_log[-1]), flush=True)
        params = carry[0]
        q_all = np.asarray(jax.jit(jax.vmap(lambda p: qdef.apply(p, q_obs)))(params))       # (I, S*P, A)
        q_one = jax.jit(lambda i, o: qdef.apply(jax.tree_util.tree_map(lambda x: x[i], params), o))
        for i, (k, gi) in enumerate(inst):
            q_tables[(k, gi)] = q_all[i].reshape(S, P, A)
            q_act[(k, gi)] = (lambda c, p, i=i: np.asarray(q_one(i, jnp.asarray(encode(c, p)))).argmax(1))

    # ---- rollouts and per-cell metrics
    for gi, g in enumerate(goal_cells):
        tg = time.time()
        dist = bfs_dist(T, g)
        starts, phases = sample_starts(args.seed, gi, g, S)
        rec = per_goal[gi]
        true_greedy = q_tables[('true', gi)].argmax(-1)                      # (S, P)
        allp = {**{k: v for (k, j), v in m_tables.items() if j == gi},
                **{k: v for (k, j), v in q_tables.items() if j == gi}}
        acts = {**{k: v for (k, j), v in m_act.items() if j == gi},
                **{k: v for (k, j), v in q_act.items() if j == gi}}
        for name, tab in allp.items():
            gr = tab.argmax(-1)
            if enc == 'onehot':
                fn = (lambda c, p, gr=gr[:, 0]: gr[c])
            else:
                fn = acts[name]
            rec['policies'][name] = {**rollout_summary(*rollout(T, fn, starts, phases, g, dist=dist)),
                                     'shortest_share': shortest_share(T, dist, gr, g),
                                     'agree_true_argmax': argmax_agree(gr, true_greedy, g)}
            if name.startswith('m_'):
                rec['m_gap'][name] = gap_over_spread(tab, dist, g)
            maps[f'{"score" if name.startswith("m_") else "q"}_{name}_{gi}'] = tab.mean(1)
            maps[f'greedy_{name}_{gi}'] = modal(gr)
        rr = np.random.default_rng(555 + 1000 * args.seed + gi)
        rec['policies']['random'] = rollout_summary(
            *rollout(T, lambda c, p: rr.integers(0, A, len(c)), starts, phases, g, dist=dist))
        rec['bfs_mean_steps_from_starts'] = float(dist[starts].mean())
        print(f'goal {gi} rollouts {time.time() - tg:.1f}s ' + json.dumps(
            {k: round(v['success'], 3) for k, v in rec['policies'].items()}), flush=True)

    res = {'stage': 'learned', 'enc': enc, 'seed': args.seed, 'args': vars(args), 'hparams': hp,
           'n_cells': int(S), 'n_phase_per_cell': P, 'n_starts': N_STARTS, 'max_steps': MAX_STEPS,
           'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
           'M': m_diag, 'goals': per_goal, 'train_log': log, 'q_train_log': q_log,
           'wall_sec': time.time() - t_start}
    write_json(args.report_out, res)
    np.savez(args.report_out[:-5] + '_maps.npz', **maps)
    return res


# ------------------------------------------------------------------ stage: aggregate
def _seed_stats(per_seed):
    v = np.asarray(per_seed, np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {'mean': None, 'std_across_seeds': None, 'n_seeds': 0}
    return {'mean': float(v.mean()), 'std_across_seeds': float(v.std(ddof=1)) if len(v) > 1 else None,
            'n_seeds': int(len(v)), 'per_seed': [float(x) for x in v]}


def _goal_mean(goals, fn):
    vals = [fn(g) for g in goals]
    vals = [x for x in vals if x is not None]
    return float(np.mean(vals)) if vals else np.nan


def _fill_reach_within1(run_dir, run):
    """A onehot run written before `reach_within1` existed: replay its saved greedy tables (the
    policy is a table over cells) from the same start cells and add the number. The replay
    must reproduce the stored success."""
    if run['enc'] != 'onehot' or 'reach_within1' in run['goals'][0]['policies']['true']:
        return
    maps = np.load(os.path.join(run_dir, f"onehot_sd{run['seed']}_maps.npz"))
    _, cells, idx, T = build_grid()
    S = len(cells)
    for gi, rec in enumerate(run['goals']):
        g = int(idx[tuple(rec['goal_rc'])])
        dist = bfs_dist(T, g)
        starts, phases = sample_starts(run['seed'], gi, g, S)
        for name, pol in rec['policies'].items():
            if name == 'random':
                rr = np.random.default_rng(555 + 1000 * run['seed'] + gi)
                fn = (lambda c, p, rr=rr: rr.integers(0, A, len(c)))
            else:
                fn = (lambda c, p, gr=maps[f'greedy_{name}_{gi}']: gr[c])
            new = rollout_summary(*rollout(T, fn, starts, phases, g, dist=dist))
            assert abs(new['success'] - pol['success']) < 1e-9, (name, gi, new['success'], pol['success'])
            pol['reach_within1'] = new['reach_within1']


def run_aggregate(args):
    out = {'description': 'Four-rooms check of the cube induced-reward effect; see the docstring of '
                          'tools/diag_fourrooms_induced_reward.py for every definition. Means are over 8 goals, '
                          'then mean and std over seeds.',
           'blocks': {}}
    exact_path = os.path.join(args.run_dir, 'exact.json')
    if os.path.exists(exact_path):
        ex = json.load(open(exact_path))
        blk = {'policies': {}, 'm_gap': {}, 'reward_corr': {}}
        for name in ex['goals'][0]['policies']:
            pol = {}
            for key in ('success', 'reach_within1', 'steps_success_mean', 'steps_all_mean'):
                pol[key] = _seed_stats([_goal_mean(ex['goals'], lambda g: g['policies'][name]['per_seed'][sd][key])
                                        for sd in range(ex['n_seeds'])])
            for key in ('shortest_share', 'agree_true_argmax'):
                if key in ex['goals'][0]['policies'][name]:
                    pol[key] = _goal_mean(ex['goals'], lambda g: g['policies'][name][key])
            blk['policies'][name] = pol
        blk['m_gap']['m_exact'] = {k: _goal_mean(ex['goals'], lambda g: g['m_gap']['m_exact'][k]) for k in ('near', 'far')}
        blk['reward_corr'] = {k: _goal_mean(ex['goals'], lambda g: g['reward_fit'][k]['corr']) for k in ex['goals'][0]['reward_fit']}
        blk['bfs_mean_steps_from_starts'] = _goal_mean(ex['goals'], lambda g: g['bfs_mean_steps_from_starts'])
        blk['note'] = ex['note']
        out['blocks']['exact'] = blk
    for enc in ('onehot', 'xyphase'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.run_dir, f'{enc}_sd*.json')))]
        if not runs:
            continue
        for r in runs:
            _fill_reach_within1(args.run_dir, r)
        blk = {'seeds': [r['seed'] for r in runs], 'policies': {}, 'm_gap': {}, 'reward_corr': {}, 'M': {}, 'w': {}}
        for name in runs[0]['goals'][0]['policies']:
            pol = {}
            for key in ('success', 'reach_within1', 'steps_success_mean', 'steps_all_mean', 'shortest_share', 'agree_true_argmax'):
                if key in runs[0]['goals'][0]['policies'][name]:
                    pol[key] = _seed_stats([_goal_mean(r['goals'], lambda g: g['policies'][name][key]) for r in runs])
            blk['policies'][name] = pol
        for name in runs[0]['goals'][0]['m_gap']:
            blk['m_gap'][name] = {k: _seed_stats([_goal_mean(r['goals'], lambda g: g['m_gap'][name][k]) for r in runs])
                                  for k in ('near', 'far')}
        for k in runs[0]['goals'][0]['reward_stats']:
            blk['reward_corr'][k] = _seed_stats([_goal_mean(r['goals'], lambda g: g['reward_stats'][k]['corr']) for r in runs])
        for k in runs[0]['goals'][0]['w']:
            blk['w'][k] = _seed_stats([_goal_mean(r['goals'], lambda g: g['w'][k]) for r in runs])
        flat = {'corr_learned_vs_exact': lambda r: r['M']['corr_learned_vs_exact'],
                'slope_learned_on_exact': lambda r: r['M']['slope_learned_on_exact'],
                'M_at_own_next_state': lambda r: r['M']['dataset_rows']['M_at_own_next_state'],
                'M_at_other_row_same_cell': lambda r: r['M']['dataset_rows']['M_at_other_row_same_cell'],
                'M_at_random_row_next_state': lambda r: r['M']['dataset_rows']['M_at_random_row_next_state'],
                'exact_at_own_next_cell': lambda r: r['M']['dataset_rows']['exact_at_own_next_cell'],
                'phi_eff_rank': lambda r: r['M']['phi_eff_rank'],
                'w_mean_pairwise_cos': lambda r: r['M']['w_mean_pairwise_cos']}
        blk['M'] = {k: _seed_stats([fn(r) for r in runs]) for k, fn in flat.items()}
        blk['M']['head_bound'] = runs[0]['M']['head_bound']
        blk['bfs_mean_steps_from_starts'] = float(np.mean([_goal_mean(r['goals'], lambda g: g['bfs_mean_steps_from_starts']) for r in runs]))
        blk['hparams'] = runs[0]['hparams']
        blk['train_loss_curve'] = [[{k: rec[k] for k in ('step', 'psm_loss', 'psm_offdiag', 'psm_diag', 'm_diag_mean',
                                                         'm_offdiag_mean', 'phi_eff_rank') if k in rec}
                                    for rec in r['train_log'] if rec['step'] % 2500 == 0 or rec is r['train_log'][-1]]
                                   for r in runs]
        blk['wall_sec'] = [r['wall_sec'] for r in runs]
        out['blocks'][enc] = blk
    write_json(args.report_out, out)

    def cell(p, key='success'):
        s = p.get(key)
        if s is None:
            return '   -   '
        if isinstance(s, dict):
            if s['mean'] is None:
                return '   -   '
            sd = s['std_across_seeds']
            return f"{s['mean']:.3f}±{sd:.3f}" if sd is not None else f"{s['mean']:.3f}"
        return f'{s:.3f}'

    names = []
    for b in out['blocks'].values():
        names += [n for n in b['policies'] if n not in names]
    for key in ('success', 'reach_within1', 'steps_all_mean', 'shortest_share', 'agree_true_argmax'):
        print(f'\n{key}')
        print(f"{'policy':16s} " + ' '.join(f'{b:>14s}' for b in out['blocks']))
        for n in names:
            print(f'{n:16s} ' + ' '.join(f"{cell(b['policies'].get(n, {}), key):>14s}" for b in out['blocks'].values()))
    for b, blk in out['blocks'].items():
        print(f'\n[{b}] reward_corr', {k: cell({'x': v}, 'x') for k, v in blk['reward_corr'].items()})
        print(f'[{b}] m_gap', {k: {kk: cell({'x': vv}, 'x') for kk, vv in v.items()} for k, v in blk['m_gap'].items()})
        if 'M' in blk:
            print(f'[{b}] M', {k: (cell({'x': v}, 'x') if isinstance(v, dict) else v) for k, v in blk['M'].items()})
            print(f'[{b}] w', {k: cell({'x': v}, 'x') for k, v in blk['w'].items()})
    return out


# ------------------------------------------------------------------ stage: figures
def run_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    free, cells, idx, T = build_grid()
    S = len(cells)
    gi = FIG_GOAL
    g = int(idx[GOAL_RC[gi]])
    dist = bfs_dist(T, g)
    os.makedirs(args.fig_dir, exist_ok=True)
    files = []

    def heat(ax, values, title, greedy=None):
        img = np.full((W, W), np.nan)
        img[cells[:, 0], cells[:, 1]] = values
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#3a3a3a')                               # walls
        im = ax.imshow(img, cmap=cmap, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if greedy is not None:
            keep = np.arange(S) != g
            d = MOVES[greedy]
            ax.quiver(cells[keep, 1], cells[keep, 0], d[keep, 1], -d[keep, 0], color='white', scale=28,
                      width=0.006, headwidth=4, pivot='middle', edgecolor='black', linewidth=0.4)
        ax.plot(GOAL_RC[gi][1], GOAL_RC[gi][0], marker='*', color='red', markersize=15, markeredgecolor='white')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def block_figure(maps, m_name, r_name, title, fname):
        panels = [
            (maps[f'score_{m_name}_{gi}'].max(-1), 'M read at the goal\n(best action; arrows = greedy move)',
             maps[f'score_{m_name}_{gi}'].argmax(-1) if f'greedy_{m_name}_{gi}' not in maps else maps[f'greedy_{m_name}_{gi}']),
            (maps[f'rmap_{r_name}_mask_{gi}'] + 1.0, 'reward read out of M\n(per cell entered, fit to the 0/1 scale)', None),
            (maps[f'q_{r_name}_mask_{gi}'].max(-1), 'critic trained on that reward\n(true episode end kept, as on cube)',
             maps.get(f'greedy_{r_name}_mask_{gi}', maps[f'q_{r_name}_mask_{gi}'].argmax(-1))),
            (maps[f'q_{r_name}_nomask_{gi}'].max(-1), 'critic trained on that reward\n(no episode end)',
             maps.get(f'greedy_{r_name}_nomask_{gi}', maps[f'q_{r_name}_nomask_{gi}'].argmax(-1))),
            (maps[f'q_true_{gi}'].max(-1), 'critic trained on the true reward',
             maps.get(f'greedy_true_{gi}', maps[f'q_true_{gi}'].argmax(-1))),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.3), facecolor='white')
        for ax, (v, t, gr) in zip(axes, panels):
            heat(ax, v, t, gr)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        path = os.path.join(args.fig_dir, fname)
        fig.savefig(path, dpi=130, facecolor='white')
        plt.close(fig)
        files.append(path)

    blocks = {}
    ex = os.path.join(args.run_dir, 'exact_maps.npz')
    if os.path.exists(ex):
        blocks['exact'] = dict(np.load(ex))
        block_figure(blocks['exact'], 'm_exact', 'gs', f'Exact M of the random policy, goal {GOAL_RC[gi]}', 'exact.png')
    for enc, label in (('onehot', 'Learned M, repeating states (onehot)'), ('xyphase', 'Learned M, unique states (xyphase)')):
        p = os.path.join(args.run_dir, f'{enc}_sd{args.fig_seed}_maps.npz')
        if not os.path.exists(p):
            continue
        blocks[enc] = dict(np.load(p))
        for m_name, r_name, tag in (('m_reg', 'gs_reg', 'M read with the regression coefficient'),
                                    ('m_lp', 'gs_lp', 'M read with the Lagrangian coefficient'),
                                    ('m_reg', 'feat', 'reward = least squares of the true reward on M features'),
                                    ('m_trainw', 'gs_reg', 'panel 1 = M read with the training coefficients w(z); '
                                                           'other panels as in the regression-coefficient figure')):
            block_figure(blocks[enc], m_name, r_name, f'{label}, goal {GOAL_RC[gi]}, seed {args.fig_seed}: {tag}',
                         f'{enc}_trainw.png' if m_name == 'm_trainw' else f'{enc}_{r_name}.png')

    # line plot along one shortest path from the far start cell to the goal
    path_cells = shortest_path_cells(T, dist, int(idx[FIG_START_RC]), g)
    x = dist[path_cells]
    rows = [('M read at the goal (best action)', lambda m, mn, rn: m[f'score_{mn}_{gi}'].max(-1)),
            ('reward read out of M (0/1 scale)', lambda m, mn, rn: m[f'rmap_{rn}_mask_{gi}'] + 1.0),
            ('critic on that reward, true episode end', lambda m, mn, rn: m[f'q_{rn}_mask_{gi}'].max(-1)),
            ('critic on that reward, no episode end', lambda m, mn, rn: m[f'q_{rn}_nomask_{gi}'].max(-1)),
            ('critic on the true reward', lambda m, mn, rn: m[f'q_true_{gi}'].max(-1))]
    cols = [(b, mn, rn, lab) for b, mn, rn, lab in (
        ('exact', 'm_exact', 'gs', 'exact M'), ('onehot', 'm_reg', 'gs_reg', 'learned M, onehot'),
        ('xyphase', 'm_reg', 'gs_reg', 'learned M, xyphase')) if b in blocks]
    if cols:
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.2 * len(cols), 2.2 * len(rows)),
                                 facecolor='white', squeeze=False, sharex=True)
        for j, (b, mn, rn, lab) in enumerate(cols):
            for i, (rt, fn) in enumerate(rows):
                ax = axes[i, j]
                ax.plot(x, fn(blocks[b], mn, rn)[path_cells], marker='o', markersize=3, color='#1f77b4')
                ax.set_title(f'{lab}: {rt}', fontsize=8)
                ax.grid(alpha=0.3)
                if i == len(rows) - 1:
                    ax.set_xlabel('steps from the goal')
        axes[0, 0].invert_xaxis()
        fig.suptitle(f'Along the shortest path from cell {FIG_START_RC} to goal {GOAL_RC[gi]} '
                     '(regression coefficient, goal-set readout)', fontsize=11)
        fig.tight_layout()
        path = os.path.join(args.fig_dir, 'path_profiles.png')
        fig.savefig(path, dpi=130, facecolor='white')
        plt.close(fig)
        files.append(path)
    print('\n'.join(files))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['exact', 'learned', 'aggregate', 'figures'])
    p.add_argument('--enc', default='onehot', choices=['onehot', 'xyphase'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_seeds', type=int, default=3)              # exact stage: start-cell seeds
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--steps', type=int, default=30000)           # measure training steps
    p.add_argument('--D', type=int, default=64)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--log_seed', type=int, default=16)           # psmgoal max_log_seed default
    p.add_argument('--keying', default='row', choices=['row', 'state'])   # row = what psmgoal does
    p.add_argument('--n_ep', type=int, default=500)
    p.add_argument('--ep_len', type=int, default=200)
    p.add_argument('--relabel_size', type=int, default=10000)    # configs/config.yaml eval_relabel_size
    p.add_argument('--lp_steps', type=int, default=0)            # 0 = psmgoal.yaml num_inference_steps
    p.add_argument('--q_steps', type=int, default=50000)         # xyphase Q-network steps
    p.add_argument('--reuse_measure', action='store_true')
    p.add_argument('--stop_after_train', action='store_true')   # train and save the measure only
    p.add_argument('--run_dir', default=None)
    p.add_argument('--fig_dir', default=None)
    p.add_argument('--fig_seed', type=int, default=0)
    p.add_argument('--report_out', default=None)
    args = p.parse_args()
    if args.stage in ('exact', 'learned', 'aggregate'):
        assert args.report_out and args.report_out.endswith('.json'), '--report_out <file>.json is required'
    if args.stage == 'exact':
        run_exact(args)
    elif args.stage == 'learned':
        run_learned(args)
    elif args.stage == 'aggregate':
        run_aggregate(args)
    else:
        run_figures(args)


if __name__ == '__main__':
    main()
