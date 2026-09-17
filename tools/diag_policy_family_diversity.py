"""How different are the futures of the policies a frozen behaviour flow can index?

LatentFlowPSM's policy family is the noise-indexed set {pi_{u'} : a = G(s, u')} for a fixed
latent u'. The measure psi(s, u, u') is a successor measure over that family, and GPI over
it can only ever be as good as the family is diverse. This probe puts a number on that
diversity for cube-single-play, against two references that bound it from below and above:

  bc              the SAME policy (fresh u every step) resampled with K seeds -- the floor.
                  Any two members differ only by sampling noise.
  random_action   a_t ~ U[-1, 1]^d_a with K seeds -- the ceiling. Off-support, so its clouds
                  sit as far from the data as an action policy on this env can go.
  noise_index     member k holds ONE latent u'_k ~ N(0, I) clipped to u_clip for the whole
                  episode. This is the family the measure indexes.
  tilted_bc       member k has a fixed random reward r_k(s, a) (a 2-hidden-layer, 64-unit
                  tanh MLP seeded by k on dataset-standardised (s, a)); per step start at
                  u ~ N(0, I), take M ascent steps of size eta on r_k(s, G(s, u)) in u, clip,
                  decode. A family of policies that each push the behaviour flow in ONE
                  consistent direction -- what an index that carried a reward would do.
  tilted_bc_strong  the same with M = 20.
  noise_index_consistent  `noise_index` decoded through the OTHER decoder path (the ODE
                  when the run acts through the one-step head, and vice versa), to check
                  that the answer does not depend on the decoder.
  goal_directed   a trained DSRL-NA latent actor (real reward, task 2): its mode, two
                  stochastic seeds of the same weights, and a second training seed's mode
                  if given. The stochastic pair is that family's own resampling floor.

Every member of every family runs N episodes from the SAME N initial environment seeds,
for the env's own horizon (200 on cube). The K members of a family run in lockstep across
K environment copies so the flow decodes are batched.

Measurements (JSON via `report_out`, clouds in an .npz sidecar):
  A  MMD^2 between members' visited-state clouds (Gaussian kernel; bandwidth = median
     pairwise distance over a subsample of the union of ALL families, once per feature
     space) in two feature spaces: the raw observation (dataset-standardised per dim) and
     phi(s) from the affine checkpoint. Mean over member pairs, plus each member vs a
     pooled `bc` cloud. Normalised distinguishability =
     (mean pairwise MMD^2 - bc floor) / (random_action ceiling - bc floor), clipped to [0, 1].
  B  A 5-fold softmax-regression two-sample test: per-trajectory mean and std of phi(s)
     (and of raw obs) -> member label; accuracy against chance 1/K.
  C  Index coherence: on n_states dataset states x a 64-latent panel, the fraction of the
     within-state action variance G(s, u_k) - mean_k G(s, u_k) that the latent index k
     explains across states (the k main effect of the two-way split in
     tools/diag_measure_vs_scalar_q.py's block G; the state effect is removed by centring).
     Near 0: a fixed latent does not pick a consistent side of the action distribution.
  D  Per-member task-2 success and return over the N rollouts, and their spread across
     members.
  E  Mean per-step action change ||a_t - a_{t-1}|| per family, and the mean final-state
     distance from the initial state.

Run (GPU, ~30 min; see scripts/slurm/diag_policy_family_diversity.sbatch):
  MUJOCO_GL=egl .venv/bin/python tools/diag_policy_family_diversity.py agent=psmflow \
      env_name=cube-single-play-singletask-task2-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
      restore_path=<affine run dir> restore_epoch=500000 \
      +na_path=<dsrl-na run dir> +na_epoch=500000 [+na_path2=<second seed's run dir>] \
      report_out=$PSM_DATA/logs/diag_policy_family_diversity_cube.json

CPU smoke: `+n_members=2 +n_episodes=1 +max_ep_steps=10 +n_coherence_states=8`.

The agent config is inherited from the run's own flags.json by `eval_checkpoint`'s
`merge_run_config`, so the flow, phi and the decoder path are the ones the run acted with.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py)

PANEL_KEY = 4321        # the 64-latent panel of the coherence probe, and the noise_index latents
ROW_SEED = 11           # dataset rows for the coherence probe (a dedicated Generator)
BW_SEED = 13            # subsample for the median-distance bandwidth and the pooled bc cloud
TILT_HIDDEN = 64


# ---------------------------------------------------------------- pure helpers (tested)
def kernel_mean(X, Y, bandwidth, chunk=1024):
    """Mean of exp(-||x - y||^2 / (2 bw^2)) over all (x, y) in X x Y. Chunked over X."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    ynorm = (Y ** 2).sum(1)
    total = 0.0
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        d2 = (xb ** 2).sum(1)[:, None] + ynorm[None, :] - 2.0 * xb @ Y.T
        total += np.exp(-np.maximum(d2, 0.0) / (2.0 * bandwidth ** 2)).sum()
    return float(total / (len(X) * len(Y)))


def mmd2_from_means(mxx, myy, mxy, m, n):
    """Biased (V-statistic) and unbiased (U-statistic) MMD^2 from the three kernel means.

    k(x, x) = 1 for the Gaussian kernel, so the off-diagonal self mean is (m*mxx - 1)/(m - 1).
    The biased form is exactly 0 for identical clouds; the unbiased one is what a two-sample
    test would use. Both are reported; the normalisation uses the biased one, whose O(1/m)
    positive bias is the same for every member cloud of equal size.
    """
    biased = mxx + myy - 2.0 * mxy
    uxx = (m * mxx - 1.0) / (m - 1.0) if m > 1 else mxx
    uyy = (n * myy - 1.0) / (n - 1.0) if n > 1 else myy
    return float(biased), float(uxx + uyy - 2.0 * mxy)


def gaussian_mmd2(X, Y, bandwidth):
    """(biased, unbiased) MMD^2 between clouds X (m, d) and Y (n, d)."""
    return mmd2_from_means(kernel_mean(X, X, bandwidth), kernel_mean(Y, Y, bandwidth),
                           kernel_mean(X, Y, bandwidth), len(X), len(Y))


def pairwise_mmd2(clouds, bandwidth):
    """Symmetric (K, K) matrices of biased and unbiased MMD^2 over a list of clouds, and
    the per-cloud self kernel means (reused by the vs-reference column)."""
    K = len(clouds)
    self_means = [kernel_mean(c, c, bandwidth) for c in clouds]
    B, U = np.zeros((K, K)), np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            b, u = mmd2_from_means(self_means[i], self_means[j],
                                   kernel_mean(clouds[i], clouds[j], bandwidth),
                                   len(clouds[i]), len(clouds[j]))
            B[i, j] = B[j, i] = b
            U[i, j] = U[j, i] = u
    return B, U, self_means


def upper_mean(M):
    """Mean of the strict upper triangle of a square matrix (None for K < 2)."""
    M = np.asarray(M, np.float64)
    K = M.shape[0]
    if K < 2:
        return None
    return float(M[np.triu_indices(K, 1)].mean())


def median_bandwidth(Z, rng, n_max=4000):
    """Median pairwise Euclidean distance over a subsample of Z (n, d). The median
    heuristic: one bandwidth per feature space, shared by every family."""
    Z = np.asarray(Z, np.float64)
    if len(Z) > n_max:
        Z = Z[rng.choice(len(Z), n_max, replace=False)]
    sq = (Z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.sqrt(np.median(d2[iu])))


def normalised_distinguishability(value, floor, ceiling):
    """(value - floor) / (ceiling - floor), clipped to [0, 1]; None when the span is not
    positive."""
    if value is None or floor is None or ceiling is None or ceiling - floor <= 0:
        return None
    return float(np.clip((value - floor) / (ceiling - floor), 0.0, 1.0))


def cv_softmax_accuracy(X, y, n_folds=5, seed=0, l2=1e-3, iters=500, lr=0.05):
    """Stratified k-fold accuracy of an L2-regularised softmax regression (numpy, Adam).

    The linear two-sample test: if a linear read-out of per-trajectory summary features
    cannot tell the members apart above chance, their futures are not linearly separable in
    that feature space. Features are standardised on the training fold.
    """
    X, y = np.asarray(X, np.float64), np.asarray(y)
    labels = np.unique(y)
    K = len(labels)
    yi = np.searchsorted(labels, y)
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), np.int64)
    for c in range(K):
        idx = np.flatnonzero(yi == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    correct, n_eval = 0, 0
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        n_eval += int(te.sum())
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
        Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
        W = np.zeros((Xtr.shape[1], K))
        m1 = np.zeros_like(W)
        v1 = np.zeros_like(W)
        Y1 = np.eye(K)[yi[tr]]
        for t in range(1, iters + 1):
            logits = Xtr @ W
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            g = Xtr.T @ (p - Y1) / len(Xtr) + l2 * W
            m1 = 0.9 * m1 + 0.1 * g
            v1 = 0.999 * v1 + 0.001 * g ** 2
            W -= lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(v1 / (1 - 0.999 ** t)) + 1e-8)
        correct += int(((Xte @ W).argmax(1) == yi[te]).sum())
    return (float(correct / n_eval) if n_eval else None), float(1.0 / K)


def index_coherence(A, rng=None):
    """A (S, K, d): the decoded panel per state. Returns the fraction of the within-state
    variance of D = A - mean_k A explained by the latent index k across states (pooled over
    action dims; also per dim), and the same fraction after shuffling k independently within
    each state (the chance level for this S and K)."""
    A = np.asarray(A, np.float64)
    S, K, d = A.shape
    D = A - A.mean(1, keepdims=True)
    tot = (D ** 2).mean()
    M = D.mean(0)                                        # (K, d) index main effect
    frac = float((M ** 2).mean() / max(tot, 1e-24))
    per_dim = ((M ** 2).mean(0) / np.maximum((D ** 2).mean((0, 1)), 1e-24)).tolist()
    rng = np.random.default_rng(0) if rng is None else rng
    Dp = np.stack([D[s][rng.permutation(K)] for s in range(S)])
    null = float((Dp.mean(0) ** 2).mean() / max(tot, 1e-24))
    return {'n_states': int(S), 'K': int(K), 'action_dim': int(d),
            'frac_index': frac, 'frac_index_per_dim': per_dim, 'frac_index_shuffled': null,
            'std_total': float(np.sqrt(tot)), 'std_index': float(np.sqrt((M ** 2).mean()))}


def traj_features(clouds_per_traj):
    """Per-trajectory [mean, std] over time of a feature -> (n_traj, 2 d)."""
    return np.stack([np.concatenate([t.mean(0), t.std(0)]) for t in clouds_per_traj])


# ------------------------------------------------------------------------- the GPU probe
def _make_eval_envs(env_name, n):
    """n independent copies of the eval env, built as envs.env_utils does (OGBench
    env_only + EpisodeMonitor with the same info filter)."""
    import ogbench

    from envs.env_utils import EpisodeMonitor
    envs = []
    for _ in range(n):
        e = ogbench.make_env_and_datasets(env_name, env_only=True)
        envs.append(EpisodeMonitor(e, filter_regexes=['.*privileged.*', '.*proprio.*']))
    return envs


def rollout_family(envs, policy, K, init_seeds, max_ep_steps=0):
    """K members in lockstep over K env copies; every member starts episode j from
    `init_seeds[j]`. `policy(obs (K, ob), key) -> (actions (K, d_a), extras dict)`.

    Returns per-member lists of (T, ob) obs arrays and (T, d_a) action arrays, success
    (K, N), return (K, N), episode length (K, N) and the concatenated extras.
    """
    import jax
    obs_tr = [[] for _ in range(K)]
    act_tr = [[] for _ in range(K)]
    succ, ret, length = np.zeros((K, len(init_seeds))), np.zeros((K, len(init_seeds))), \
        np.zeros((K, len(init_seeds)), np.int64)
    extras = {}
    rng = jax.random.PRNGKey(int(init_seeds[0]) * 7919 + K)
    for j, s in enumerate(init_seeds):
        obs = np.stack([envs[k].reset(seed=int(s))[0] for k in range(K)]).astype(np.float32)
        assert np.allclose(obs, obs[0:1]), 'the K env copies did not reset to the same state'
        done = np.zeros(K, bool)
        o_buf, a_buf = [[] for _ in range(K)], [[] for _ in range(K)]
        t = 0
        while not done.all():
            rng, key = jax.random.split(rng)
            act, ex = policy(obs, key)
            act = np.clip(np.asarray(act, np.float32), -1.0, 1.0)
            for name, v in ex.items():
                extras.setdefault(name, []).append(np.asarray(v))
            for k in range(K):
                if done[k]:
                    continue
                o_buf[k].append(obs[k].copy())
                a_buf[k].append(act[k].copy())
                nobs, r, term, trunc, info = envs[k].step(act[k])
                obs[k] = nobs
                ret[k, j] += float(r)
                succ[k, j] = max(succ[k, j], float(np.max(np.asarray(info.get('success', 0.0)))))
                done[k] = bool(term or trunc)
            t += 1
            if max_ep_steps and t >= max_ep_steps:
                done[:] = True
        for k in range(K):
            obs_tr[k].append(np.stack(o_buf[k]))
            act_tr[k].append(np.stack(a_buf[k]))
            length[k, j] = len(o_buf[k])
    extras = {n: np.stack(vs) for n, vs in extras.items()}        # (n_steps_total, K)
    return obs_tr, act_tr, succ, ret, length, extras


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

    t_start = time.time()
    np.random.seed(int(cfg.seed))            # eval_checkpoint's pinning of the relabel batch
    _, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    merged, prov = merge_run_config(OmegaConf.to_container(cfg.agent, resolve=True),
                                    cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    ex = ds.sample(1)
    agent = agents[config['agent_name']].create(cfg.seed, ex['observations'], ex['actions'], config)
    agent = restore_agent(agent, cfg.restore_path, int(cfg.restore_epoch))
    zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
    agent = agent.infer_eval_z(zb['next_observations'],
                               zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))
    ac = agent.config
    d_a, u_clip = int(ac['action_dim']), float(ac['u_clip'])
    run_path, ode_steps = str(ac['gpi_decode']), int(ac['flow_decode_steps'])
    other_path = 'ode' if run_path == 'onestep' else 'onestep'

    K = int(cfg.get('n_members', 16))
    N = int(cfg.get('n_episodes', 20))
    max_ep_steps = int(cfg.get('max_ep_steps', 0))     # 0 = the env's own horizon
    n_coh_states = int(cfg.get('n_coherence_states', 2000))
    n_panel = int(cfg.get('n_panel', 64))
    tilt_eta = float(cfg.get('tilt_eta', 0.5))
    tilt_steps = int(cfg.get('tilt_steps', 5))
    tilt_steps_strong = int(cfg.get('tilt_steps_strong', 20))
    seed = int(cfg.seed)
    init_seeds = [seed * 1000 + j for j in range(N)]

    # ---- the two decoder paths, both local so the OTHER one is available too ----
    def decode_local(observations, u, path):
        if path == 'onestep':
            a = agent.flow_onestep_def.apply({'params': agent.flow_onestep}, observations, u)
        else:
            a = u
            for i in range(ode_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / ode_steps)
                a = a + agent.flow_vf_def.apply({'params': agent.flow_vf}, observations, a, t) / ode_steps
        return jnp.clip(a, -1.0, 1.0)

    dec_run = jax.jit(agent.decode)                                  # the run's own path
    dec_other = jax.jit(lambda o, u: decode_local(o, u, other_path))
    dec_same = jax.jit(lambda o, u: decode_local(o, u, run_path))
    phi_fn = jax.jit(lambda o: agent.phi(o))
    chk_o = jnp.asarray(ds['observations'][:64], jnp.float32)
    chk_u = jnp.clip(jax.random.normal(jax.random.PRNGKey(1), (64, d_a)), -u_clip, u_clip)
    decode_check = float(jnp.abs(dec_run(chk_o, chk_u) - dec_same(chk_o, chk_u)).max())
    assert decode_check < 1e-5, f'local {run_path} decoder differs from agent.decode by {decode_check}'

    obs_mu = np.asarray(ds['observations']).mean(0).astype(np.float32)
    obs_sd = (np.asarray(ds['observations']).std(0) + 1e-6).astype(np.float32)
    act_mu = np.asarray(ds['actions']).mean(0).astype(np.float32)
    act_sd = (np.asarray(ds['actions']).std(0) + 1e-6).astype(np.float32)

    # ---- the tilt: a random reward per member, ascended in u through the flow ----
    def tilt_init(k):
        ks = jax.random.split(jax.random.PRNGKey(100 + k), 3)
        d_in = int(obs_mu.shape[0]) + d_a
        return {'W1': jax.random.normal(ks[0], (d_in, TILT_HIDDEN)) / np.sqrt(d_in),
                'b1': jnp.zeros(TILT_HIDDEN),
                'W2': jax.random.normal(ks[1], (TILT_HIDDEN, TILT_HIDDEN)) / np.sqrt(TILT_HIDDEN),
                'b2': jnp.zeros(TILT_HIDDEN),
                'W3': jax.random.normal(ks[2], (TILT_HIDDEN, 1)) / np.sqrt(TILT_HIDDEN),
                'b3': jnp.zeros(1)}

    tilt_params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[tilt_init(k) for k in range(K)])

    def tilt_reward(p, o, a):
        x = jnp.concatenate([(o - obs_mu) / obs_sd, (a - act_mu) / act_sd], -1)
        h = jnp.tanh(x @ p['W1'] + p['b1'])
        h = jnp.tanh(h @ p['W2'] + p['b2'])
        return (h @ p['W3'] + p['b3'])[0]

    def make_tilt_policy(n_steps):
        @jax.jit
        def step(obs, key):
            u0 = jax.random.normal(key, (K, d_a))

            def objective(U):
                A = dec_run(obs, U)
                return jax.vmap(tilt_reward)(tilt_params, obs, A).sum()

            u = u0
            for _ in range(n_steps):
                u = u + tilt_eta * jax.grad(objective)(u)
            u = jnp.clip(u, -u_clip, u_clip)
            a = dec_run(obs, u)
            return a, {'displacement': jnp.linalg.norm(u - u0, axis=-1),
                       'clip_frac_coord': (jnp.abs(u) >= u_clip - 1e-6).mean(-1),
                       'clip_any': (jnp.abs(u) >= u_clip - 1e-6).any(-1).astype(jnp.float32),
                       'reward_gain': jax.vmap(tilt_reward)(tilt_params, obs, a)
                       - jax.vmap(tilt_reward)(tilt_params, obs, dec_run(obs, jnp.clip(u0, -u_clip, u_clip)))}
        return lambda obs, key: tuple(jax.device_get(step(jnp.asarray(obs), key)))

    # ---- the families ----
    fixed_u = np.asarray(jax.device_get(jnp.clip(
        jax.random.normal(jax.random.PRNGKey(PANEL_KEY), (K, d_a)), -u_clip, u_clip)))

    def noise_index_policy(dec):
        U = jnp.asarray(fixed_u)
        return lambda obs, key: (np.asarray(jax.device_get(dec(jnp.asarray(obs), U))), {})

    def bc_policy(obs, key):
        u = jnp.clip(jax.random.normal(key, (K, d_a)), -u_clip, u_clip)
        return np.asarray(jax.device_get(dec_run(jnp.asarray(obs), u))), {}

    def random_policy(obs, key):
        return np.asarray(jax.device_get(jax.random.uniform(key, (K, d_a), minval=-1.0, maxval=1.0))), {}

    families = {
        'noise_index': (K, noise_index_policy(dec_run)),
        'bc': (K, bc_policy),
        'random_action': (K, random_policy),
        'tilted_bc': (K, make_tilt_policy(tilt_steps)),
        'tilted_bc_strong': (K, make_tilt_policy(tilt_steps_strong)),
        'noise_index_consistent': (K, noise_index_policy(dec_other)),
    }

    # The goal-directed reference: a DSRL-NA latent actor restored into the same process.
    goal_members, na_prov = [], None
    na_path = cfg.get('na_path', None)
    if na_path:
        na_agents = []
        for p in [str(na_path)] + ([str(cfg.na_path2)] if cfg.get('na_path2', None) else []):
            na_merged, na_prov = merge_run_config(
                OmegaConf.to_container(cfg.agent, resolve=True), p, _cli_agent_keys())
            na_conf = ml_collections.ConfigDict(_lists_to_tuples(na_merged))
            assert na_conf['dsrl_na']['enabled'] and na_conf['acting'] == 'actor', (
                f'{p} is not a DSRL-NA actor run (dsrl_na.enabled={na_conf["dsrl_na"]["enabled"]}, '
                f'acting={na_conf["acting"]})')
            na = agents[na_conf['agent_name']].create(cfg.seed, ex['observations'], ex['actions'], na_conf)
            na = restore_agent(na, p, int(cfg.get('na_epoch', 500000)))
            na = na.infer_eval_z(zb['next_observations'],
                                 zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))
            na_agents.append((p, na))
        # members: mode, two stochastic streams of the same weights, second seed's mode
        goal_members = [('mode', na_agents[0][1], 0.0, 0), ('stoch_a', na_agents[0][1], 1.0, 1),
                        ('stoch_b', na_agents[0][1], 1.0, 2)]
        if len(na_agents) > 1:
            goal_members.append(('mode_seed2', na_agents[1][1], 0.0, 3))
        acts = [jax.jit(lambda o, s, temp, ag=m[1]: ag.sample_actions(o, seed=s, temperature=temp))
                for m in goal_members]

        def goal_policy(obs, key):
            keys = jax.random.split(key, len(goal_members))
            out = [acts[i](jnp.asarray(obs[i]), keys[i], jnp.asarray(m[2]))
                   for i, m in enumerate(goal_members)]
            return np.stack([np.asarray(jax.device_get(a)) for a in out]), {}
        families['goal_directed'] = (len(goal_members), goal_policy)

    only = cfg.get('families', None)
    if only:
        keep = [f.strip() for f in str(only).split(',')]
        families = {n: v for n, v in families.items() if n in keep}

    n_env = max(v[0] for v in families.values())
    envs = [eval_env] + _make_eval_envs(cfg.env_name, n_env - 1)
    raw = {}       # family -> member -> list of (T, ob)
    acts_all = {}
    stats = {}
    for name, (Kf, pol) in families.items():
        t0 = time.time()
        o_tr, a_tr, succ, ret, length, extras = rollout_family(envs[:Kf], pol, Kf, init_seeds, max_ep_steps)
        raw[name], acts_all[name] = o_tr, a_tr
        dact = [np.linalg.norm(a[1:] - a[:-1], axis=-1) for tr in a_tr for a in tr if len(a) > 1]
        final = [float(np.linalg.norm(o[-1] - o[0])) for tr in o_tr for o in tr]
        stats[name] = {
            'K': Kf, 'n_episodes': N, 'wall_seconds': round(time.time() - t0, 1),
            'success': {'per_member': succ.mean(1).tolist(), 'mean': float(succ.mean()),
                        'std_across_members': float(succ.mean(1).std()),
                        'min_member': float(succ.mean(1).min()), 'max_member': float(succ.mean(1).max())},
            'return': {'per_member': ret.mean(1).tolist(), 'mean': float(ret.mean()),
                       'std_across_members': float(ret.mean(1).std())},
            'episode_length_mean': float(length.mean()),
            'action_change_per_step': float(np.concatenate(dact).mean()) if dact else None,
            'action_norm': float(np.mean([np.linalg.norm(a, axis=-1).mean() for tr in a_tr for a in tr])),
            'final_state_distance': float(np.mean(final)),
        }
        if extras:
            stats[name]['tilt'] = {k: {'mean': float(v.mean()), 'std': float(v.std()),
                                       'p90': float(np.quantile(v, 0.9))}
                                   for k, v in extras.items()}
        if name == 'goal_directed':
            stats[name]['members'] = [m[0] for m in goal_members]
        print(f'  {name:24s} K={Kf}  success {succ.mean():.3f} (members {succ.mean(1).min():.2f}-'
              f'{succ.mean(1).max():.2f})  {stats[name]["wall_seconds"]}s', flush=True)

    # ---- feature spaces: standardised raw obs, and phi(s) ----
    def phi_of(o):
        out = []
        for i in range(0, len(o), 8192):
            out.append(np.asarray(jax.device_get(phi_fn(jnp.asarray(o[i:i + 8192], jnp.float32)))))
        return np.concatenate(out)

    spaces = {'raw': {}, 'phi': {}}
    for name, members in raw.items():
        spaces['raw'][name] = [[(o - obs_mu) / obs_sd for o in tr] for tr in members]
        spaces['phi'][name] = [[phi_of(o) for o in tr] for tr in members]

    bw_rng = np.random.default_rng(BW_SEED)
    bandwidth = {}
    for sp, fam in spaces.items():
        union = np.concatenate([o for members in fam.values() for tr in members for o in tr])
        bandwidth[sp] = median_bandwidth(union, bw_rng)
    print(f'bandwidths: {bandwidth}', flush=True)

    # ---- A: MMD^2 ----
    mmd = {sp: {} for sp in spaces}
    mmd_mats = {}
    for sp, fam in spaces.items():
        bc_ref = None
        if 'bc' in fam:
            pooled = np.concatenate([o for tr in fam['bc'] for o in tr])
            n_ref = min(len(pooled), int(np.mean([sum(len(o) for o in tr) for tr in fam['bc']])))
            bc_ref = pooled[bw_rng.choice(len(pooled), n_ref, replace=False)]
            ref_self = kernel_mean(bc_ref, bc_ref, bandwidth[sp])
        for name, members in fam.items():
            clouds = [np.concatenate(tr) for tr in members]
            B, U, selfm = pairwise_mmd2(clouds, bandwidth[sp])
            mmd_mats[f'{sp}_{name}_biased'], mmd_mats[f'{sp}_{name}_unbiased'] = B, U
            row = {'pairwise_mean': upper_mean(B), 'pairwise_mean_unbiased': upper_mean(U),
                   'pairwise_max': float(B.max()), 'cloud_size': int(np.mean([len(c) for c in clouds]))}
            if bc_ref is not None:
                vs = [mmd2_from_means(selfm[i], ref_self, kernel_mean(clouds[i], bc_ref, bandwidth[sp]),
                                      len(clouds[i]), len(bc_ref))[0] for i in range(len(clouds))]
                row['vs_bc_pooled_mean'] = float(np.mean(vs))
                row['vs_bc_pooled_per_member'] = [float(v) for v in vs]
            if name == 'goal_directed' and len(clouds) >= 3:
                row['stoch_pair_floor'] = float(B[1, 2])      # the two stochastic streams
                row['mode_vs_stoch'] = float(np.mean([B[0, 1], B[0, 2]]))
            mmd[sp][name] = row
        floor = mmd[sp].get('bc', {}).get('pairwise_mean')
        ceil = mmd[sp].get('random_action', {}).get('pairwise_mean')
        for name in fam:
            mmd[sp][name]['normalised'] = normalised_distinguishability(
                mmd[sp][name]['pairwise_mean'], floor, ceil)
        mmd[sp]['_floor_bc'] = floor
        mmd[sp]['_ceiling_random_action'] = ceil

    # ---- B: linear two-sample test on per-trajectory summaries ----
    clf = {sp: {} for sp in spaces}
    for sp, fam in spaces.items():
        for name, members in fam.items():
            Kf = len(members)
            if Kf < 2:
                clf[sp][name] = {'accuracy': None, 'chance': 1.0}
                continue
            X = traj_features([o for tr in members for o in tr])
            y = np.concatenate([[k] * len(tr) for k, tr in enumerate(members)])
            acc, chance = cv_softmax_accuracy(X, y, seed=seed)
            clf[sp][name] = {'accuracy': acc, 'chance': chance, 'n_traj': len(y)}

    # ---- C: index coherence on dataset states ----
    rows = np.sort(np.random.default_rng(ROW_SEED).choice(ds.size, min(n_coh_states, ds.size), replace=False))
    panel = jnp.clip(jax.random.normal(jax.random.PRNGKey(PANEL_KEY + 1), (n_panel, d_a)), -u_clip, u_clip)
    A = []
    for i in range(0, len(rows), 128):
        o = jnp.asarray(ds['observations'][rows[i:i + 128]], jnp.float32)
        S = o.shape[0]
        oo = jnp.repeat(o, n_panel, axis=0)
        uu = jnp.tile(panel, (S, 1))
        A.append(np.asarray(jax.device_get(dec_run(oo, uu))).reshape(S, n_panel, d_a))
    A = np.concatenate(A)
    coherence = index_coherence(A, np.random.default_rng(ROW_SEED))
    coherence['decode_path'] = run_path
    coherence['panel_key'] = PANEL_KEY + 1
    if run_path != other_path:
        A2 = []
        for i in range(0, len(rows), 128):
            o = jnp.asarray(ds['observations'][rows[i:i + 128]], jnp.float32)
            S = o.shape[0]
            A2.append(np.asarray(jax.device_get(dec_other(jnp.repeat(o, n_panel, axis=0),
                                                            jnp.tile(panel, (S, 1))))).reshape(S, n_panel, d_a))
        coherence['other_path'] = index_coherence(np.concatenate(A2), np.random.default_rng(ROW_SEED))
        coherence['other_path']['decode_path'] = other_path

    # ---- the paper-facing table ----
    table = []
    for name in families:
        s = stats[name]
        table.append({
            'family': name, 'K': s['K'],
            'distinguishability_phi': mmd['phi'][name]['normalised'],
            'distinguishability_raw': mmd['raw'][name]['normalised'],
            'mmd2_phi': mmd['phi'][name]['pairwise_mean'], 'mmd2_raw': mmd['raw'][name]['pairwise_mean'],
            'classifier_acc_phi': clf['phi'][name]['accuracy'], 'classifier_acc_raw': clf['raw'][name]['accuracy'],
            'classifier_chance': clf['phi'][name]['chance'],
            'index_coherence': coherence['frac_index'] if name == 'noise_index' else None,
            'success_mean': s['success']['mean'], 'success_std_members': s['success']['std_across_members'],
            'success_min_member': s['success']['min_member'], 'success_max_member': s['success']['max_member'],
            'return_mean': s['return']['mean'],
            'action_change_per_step': s['action_change_per_step'],
            'final_state_distance': s['final_state_distance'],
        })
    hdr = (f"{'family':24s}{'dist_phi':>9s}{'dist_raw':>9s}{'acc_phi':>8s}{'acc_raw':>8s}"
           f"{'chance':>7s}{'succ':>6s}{'sd':>6s}{'dact':>7s}")
    print(hdr)

    def f(v, w=9, p=3):
        return f'{v:{w}.{p}f}' if v is not None else ' ' * (w - 1) + '-'
    for r in table:
        print(f"{r['family']:24s}{f(r['distinguishability_phi'])}{f(r['distinguishability_raw'])}"
              f"{f(r['classifier_acc_phi'], 8)}{f(r['classifier_acc_raw'], 8)}{f(r['classifier_chance'], 7)}"
              f"{f(r['success_mean'], 6, 2)}{f(r['success_std_members'], 6, 2)}{f(r['action_change_per_step'], 7)}")
    print(f"index coherence (noise_index): {coherence['frac_index']:.4f} "
          f"(shuffled {coherence['frac_index_shuffled']:.4f})")

    report = {
        'probe': 'policy-family diversity through the frozen behaviour flow',
        'env_name': str(cfg.env_name), 'rollout_task': 'the env named above only (task 2 on cube)',
        'restore_path': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch),
        'na_path': str(na_path) if na_path else None, 'na_path2': str(cfg.get('na_path2', None) or '') or None,
        'na_epoch': int(cfg.get('na_epoch', 500000)) if na_path else None,
        'decode_path': run_path, 'other_decode_path': other_path, 'ode_steps': ode_steps,
        'decode_path_check_max_abs_diff': decode_check,
        'u_clip': u_clip, 'action_dim': d_a, 'obs_dim': int(obs_mu.shape[0]),
        'n_members': K, 'n_episodes': N, 'init_seeds': init_seeds, 'seed': seed,
        'max_ep_steps': max_ep_steps or 'env default',
        'tilt': {'hidden': TILT_HIDDEN, 'layers': 2, 'eta': tilt_eta, 'steps': tilt_steps,
                 'steps_strong': tilt_steps_strong, 'init': 'W ~ N(0, 1/fan_in), b = 0, seed 100+k'},
        'feature_spaces': {'raw': 'observation standardised per dim by dataset mean/std',
                           'phi': "the affine checkpoint's phi(s) network output"},
        'bandwidth': bandwidth,
        'bandwidth_note': f'median pairwise distance over a {4000}-point subsample of the union of all families',
        'mmd_note': 'biased V-statistic (0 for identical clouds); unbiased beside it. vs_bc_pooled: '
                    'each member against one seeded subsample of the bc union, sized like a member cloud',
        'normalisation': '(pairwise_mean - bc pairwise_mean) / (random_action pairwise_mean - bc pairwise_mean), clipped [0, 1]',
        'classifier': '5-fold stratified softmax regression, L2 1e-3, Adam 500 it; features = per-trajectory mean & std',
        'config_source': prov, 'na_config_source': na_prov,
        'families': stats, 'mmd2': mmd, 'classifier_acc': clf, 'index_coherence': coherence,
        'table': table, 'wall_seconds_total': round(time.time() - t_start, 1),
    }
    out = write_report(report, cfg, 'diag_policy_family_diversity.json')
    npz = os.path.splitext(out)[0] + '.npz'
    arrays = {k: v for k, v in mmd_mats.items()}
    for name, members in raw.items():
        for k, tr in enumerate(members):
            arrays[f'obs_{name}_{k}'] = np.concatenate(tr).astype(np.float32)
            arrays[f'act_{name}_{k}'] = np.concatenate(acts_all[name][k]).astype(np.float32)
    arrays['fixed_u'] = fixed_u
    arrays['coherence_actions'] = A.astype(np.float32)
    arrays['coherence_rows'] = rows
    np.savez_compressed(npz, **arrays)
    print(f'npz -> {npz}')
    print(f'total wall {report["wall_seconds_total"]}s')


def main():
    import hydra
    hydra.main(version_base=None, config_path='../configs', config_name='config')(_main)()


if __name__ == '__main__':
    main()
