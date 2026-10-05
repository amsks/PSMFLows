"""Gridworld recovery test for psmgoal's successor-measure loss.

psmgoal arm: builds the REAL agents.psmgoal.PSMGoalAgent via .create (allow_untrained_flow,
flow unused) and trains it with the REAL PSMGoalAgent.apply_update(batch, z, u_next), which
runs measure_loss -> Adam on basis (RLUMeasure) and w (PolicyCoefficient) -> Polyak tau.
The only bypass: agent.update() is not called, because it computes u_next with continuous
proto latents keyed on batch['index']. Here u_next is the discrete proto action: the same
utils.psm_proto.proto_seed_ints / proto_latents seed arithmetic, argmax over the 4 latent
components -> one-hot action. batch['noise_preimage'] carries the one-hot data action.
Keying: 'state' = seed on the grid-cell index of s'_i (a Markov policy per z);
        'row'   = seed on the dataset row i (what psmgoal does, psmgoal.py:288).
head 'plain' swaps RLUMeasure for the same trunk + Dense(D+1) without the RMSNorm.

fb arm: M = psi(s,a,z)^T phi(s+), phi = PhiMap(norm=True), psi = PhiMap(norm=False) on
[s, a, z_bits]; loss = utils.psm_common.contrastive_loss (one head) + ortho_coef*ortho_loss.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import utils.xla_guard  # noqa: F401,E402  (must precede jax)
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402

from agents.psmgoal import PSMGoalAgent  # noqa: E402
from utils.flax_utils import TrainState  # noqa: E402
from utils.psm_common import contrastive_loss, ortho_loss, polyak_update  # noqa: E402
from utils.psm_networks import PhiMap, _ORTH1, _triple_trunk  # noqa: E402
from utils.psm_proto import proto_latents, proto_seed_ints, sample_z_bin  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--loss', default='psmgoal', choices=['psmgoal', 'fb'])
p.add_argument('--enc', default='onehot', choices=['onehot', 'xy', 'xyphase'])
p.add_argument('--noise', type=float, default=0.05)
p.add_argument('--keying', default='state', choices=['state', 'row'])
p.add_argument('--head', default='rms', choices=['rms', 'plain'])
p.add_argument('--gamma', type=float, default=0.98)
p.add_argument('--steps', type=int, default=30000)
p.add_argument('--D', type=int, default=64)
p.add_argument('--hidden', type=int, default=256)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--ortho_coef', type=float, default=1.0)   # fb arm only
p.add_argument('--seed', type=int, default=0)
p.add_argument('--log_seed', type=int, default=16)   # policy-code width (psmgoal max_log_seed)
p.add_argument('--out', required=True)
args = p.parse_args()

W = 7
S = W * W
A = 4
LOG_SEED = args.log_seed
U_CLIP = 3.0
MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])


def cell_xy(s):
    return np.stack([s // W, s % W], -1)


def step_table():
    T = np.zeros((S, A), np.int64)
    for s in range(S):
        for a in range(A):
            x, y = cell_xy(np.array(s)) + MOVES[a]
            T[s, a] = s if not (0 <= x < W and 0 <= y < W) else x * W + y
    return T


T = step_table()
rng = np.random.default_rng(args.seed)
n_ep, ep_len = 500, 200
st, ac, nx, ph = [], [], [], []
ALPHA = 0.6180339887498949          # deterministic phase advance (xyphase encoding)
for _ in range(n_ep):
    s = rng.integers(S)
    th = rng.random()
    for _t in range(ep_len):
        a = rng.integers(A)
        ph.append(th)
        th = (th + ALPHA) % 1.0
        st.append(s)
        ac.append(a)
        s2 = T[s, a]
        nx.append(s2)
        s = s2
st, ac, nx, ph = map(np.asarray, (st, ac, nx, ph))
nph = (ph + ALPHA) % 1.0
Nd = len(st)
rho = np.bincount(nx, minlength=S) / Nd          # next-state marginal = column distribution


def encode(s, noisy, phase=None):
    s = np.asarray(s)
    if args.enc == 'onehot':
        return np.eye(S, dtype=np.float32)[s]
    xy = cell_xy(s).astype(np.float32) / (W - 1) * 2 - 1
    if args.enc == 'xyphase':
        # deterministic continuous state: cell coords + a phase that advances by ALPHA each
        # step. Every state is unique and s' is an exact function of (s, a), as on cube.
        if phase is None:
            phase = rng.random(len(s))
        return np.concatenate([xy, np.cos(2 * np.pi * phase)[:, None], np.sin(2 * np.pi * phase)[:, None]],
                              -1).astype(np.float32)
    if noisy:
        xy = xy + rng.normal(0, args.noise, xy.shape).astype(np.float32)
    return xy


obs_d = encode(st, True, ph)
nobs_d = encode(nx, True, nph)
act_d = np.eye(A, dtype=np.float32)[ac]
OB = obs_d.shape[1]
base_key = jax.random.PRNGKey(0)                 # psmgoal proto_seed default 0


def proto_action(z_bits, key_index):
    """Discrete proto action: psm_proto seed arithmetic, argmax of the 4-dim clipped latent."""
    seeds = proto_seed_ints(z_bits, key_index, LOG_SEED)
    return jnp.argmax(proto_latents(seeds, A, U_CLIP, base_key), -1)


# held-out evaluation codes
eval_codes = (np.random.default_rng(1234).integers(0, 2 ** LOG_SEED, 16) if 2 ** LOG_SEED > 16
              else np.resize(np.arange(2 ** LOG_SEED), 16))
eval_bits = ((eval_codes[:, None] >> np.arange(LOG_SEED)) & 1).astype(np.float32)


def true_measures(gamma):
    """Mtrue(z, s, a, s+) for each eval code under the bootstrap policy actually used."""
    out, pols = [], []
    for zb in eval_bits:
        zb_j = jnp.asarray(zb)[None]
        if args.keying == 'state':
            acts = np.asarray(proto_action(jnp.repeat(zb_j, S, 0), jnp.arange(S)))
            pi = np.eye(A)[acts]
        else:
            # effective policy at state s: the row-keyed actions of all rows whose s' = s
            acts = np.asarray(proto_action(jnp.repeat(zb_j, Nd, 0), jnp.arange(Nd)))
            pi = np.zeros((S, A))
            np.add.at(pi, (nx, acts), 1.0)
            pi = pi / np.maximum(pi.sum(1, keepdims=True), 1)
        P = np.zeros((S, S))
        for a in range(A):
            P[np.arange(S), T[:, a]] += pi[:, a]
        Mpi = (1 - gamma) * np.linalg.inv(np.eye(S) - gamma * P)
        Msa = (1 - gamma) * np.eye(S)[T] + gamma * Mpi[T]           # (S, A, S+)
        out.append(Msa)
        pols.append(pi)
    return np.stack(out), np.stack(pols)


# ------------------------------------------------------------------ models
class PlainMeasure(nn.Module):
    """RLUMeasure with the RMSNorm removed: same trunk, Dense(D+1) head."""
    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs, u, g):
        x = _triple_trunk(obs, u, g, self.hidden_dim, self.hidden_layers)
        head = nn.Dense(self.z_dim + 1, kernel_init=_ORTH1, name='head_out')(x)
        return head[..., :self.z_dim], head[..., self.z_dim]


gamma = args.gamma
D = args.D
key = jax.random.PRNGKey(args.seed)

if args.loss == 'psmgoal':
    cfg = yaml.safe_load(open(f'{REPO}/configs/agent/psmgoal.yaml'))
    cfg.update(max_log_seed=LOG_SEED, z_dim=D, discount=gamma, batch_size=args.batch, allow_untrained_flow=True,
               flow_ckpt_path=None)
    cfg['measure'] = {'hidden_dim': args.hidden, 'hidden_layers': 2}
    cfg['flow'] = {'hidden_dims': [32, 32], 'value_hidden_dims': [32, 32], 'layer_norm': False,
                   'critic_layer_norm': True}
    agent = PSMGoalAgent.create(args.seed, jnp.asarray(obs_d[:1]), jnp.asarray(act_d[:1]), cfg)
    if args.head == 'plain':
        mdef = PlainMeasure(z_dim=D, hidden_dim=args.hidden, hidden_layers=2)
        mp = mdef.init(jax.random.PRNGKey(args.seed + 7), obs_d[:1], act_d[:1], obs_d[:1])['params']
        basis = TrainState.create(mdef, mp, tx=optax.adam(cfg['lr_measure']))
        agent = agent.replace(basis=basis, target_basis=jax.tree_util.tree_map(jnp.array, mp))
    hp = {k: cfg[k] for k in ('z_dim', 'discount', 'tau', 'lr_measure', 'lr_w', 'ortho_coef',
                              'max_log_seed', 'u_clip')}
    hp.update(measure=cfg['measure'], w=cfg['w'], head=args.head)

    @jax.jit
    def train_step(agent, obs, nobs, act, key_index, k):
        z = sample_z_bin(k, obs.shape[0], LOG_SEED)
        u_next = jax.nn.one_hot(proto_action(z, key_index), A)
        batch = {'observations': obs, 'next_observations': nobs, 'noise_preimage': act}
        return agent.apply_update(batch, z, u_next)

    state = agent
else:
    psi_def = PhiMap(z_dim=D, hidden_dim=args.hidden, hidden_layers=2, norm=False)
    phi_def = PhiMap(z_dim=D, hidden_dim=args.hidden, hidden_layers=2, norm=True)
    k1, k2 = jax.random.split(key)
    params = {'psi': psi_def.init(k1, jnp.zeros((1, OB + A + LOG_SEED)))['params'],
              'phi': phi_def.init(k2, jnp.zeros((1, OB)))['params']}
    tx = optax.adam(1e-4)
    state = {'p': params, 't': jax.tree_util.tree_map(jnp.array, params), 'o': tx.init(params)}
    hp = {'z_dim': D, 'discount': gamma, 'tau': 0.01, 'lr': 1e-4, 'ortho_coef': args.ortho_coef,
          'hidden': args.hidden}

    def fb_M(pr, obs, act, zb, goals):
        psi = psi_def.apply({'params': pr['psi']}, jnp.concatenate([obs, act, zb], -1))
        phi = phi_def.apply({'params': pr['phi']}, goals)
        return psi @ phi.T, phi

    def fb_loss(pr, tp, obs, nobs, act, anext, zb):
        M, phi = fb_M(pr, obs, act, zb, nobs)
        Mt, _ = fb_M(tp, nobs, anext, zb, nobs)
        N = M.shape[0]
        off = 1.0 - jnp.eye(N)
        meas, dg, od = contrastive_loss(M[None], jax.lax.stop_gradient(Mt), gamma, off, off.sum())
        orth, _, _ = ortho_loss(phi, off, off.sum())
        return meas + args.ortho_coef * orth, {'psm_loss': meas, 'ortho': orth,
                                               'm_diag_mean': jnp.mean(jnp.diagonal(M)),
                                               'm_offdiag_mean': jnp.sum(M * off) / off.sum()}

    @jax.jit
    def train_step(state, obs, nobs, act, key_index, k):
        z = sample_z_bin(k, obs.shape[0], LOG_SEED)
        anext = jax.nn.one_hot(proto_action(z, key_index), A)
        (_, info), g = jax.value_and_grad(fb_loss, has_aux=True)(state['p'], state['t'], obs, nobs,
                                                                 act, anext, z)
        upd, o = tx.update(g, state['o'], state['p'])
        p_new = optax.apply_updates(state['p'], upd)
        return {'p': p_new, 't': polyak_update(p_new, state['t'], 0.01), 'o': o}, info

print('hyperparameters:', json.dumps({**vars(args), **hp, 'dataset_rows': Nd}), flush=True)

# ------------------------------------------------------------------ train
row_key = nx if args.keying == 'state' else np.arange(Nd)   # key of the policy at s'_i
log = []
t0 = time.time()
for it in range(args.steps):
    idx = rng.integers(0, Nd, args.batch)
    key, k = jax.random.split(key)
    state, info = train_step(state, jnp.asarray(obs_d[idx]), jnp.asarray(nobs_d[idx]),
                             jnp.asarray(act_d[idx]), jnp.asarray(row_key[idx]), k)
    if it % 1000 == 0 or it == args.steps - 1:
        rec = {k_: float(v) for k_, v in info.items() if np.ndim(v) == 0}
        rec['step'] = it
        rec['sec'] = time.time() - t0
        log.append(rec)
        print(json.dumps(rec), flush=True)

# ------------------------------------------------------------------ evaluate
Mtrue, pols = true_measures(gamma)                           # (Z, S, A, S+)
scale = 1.0 if args.loss == 'psmgoal' else 1.0 / (1 - gamma)
R = scale * Mtrue / rho[None, None, None, :]                 # expected fixed point
ss, aa, gg = np.meshgrid(np.arange(S), np.arange(A), np.arange(S), indexing='ij')
ss, aa, gg = ss.ravel(), aa.ravel(), gg.ravel()
o_e = jnp.asarray(encode(ss, False))
a_e = jnp.asarray(np.eye(A, dtype=np.float32)[aa])
g_e = jnp.asarray(encode(gg, False))
if args.loss == 'psmgoal':
    phi, b = state.basis(o_e, a_e, g_e)
    wz = np.asarray(state.w(jnp.asarray(eval_bits)))          # (Z, D)
    phi, b = np.asarray(phi), np.asarray(b)
    L = np.einsum('nd,zd->zn', phi, wz) + b[None]
    feats = phi
    wn = wz / np.linalg.norm(wz, axis=1, keepdims=True)
    cosw = wn @ wn.T
    w_cos = float(cosw[np.triu_indices(16, 1)].mean())
    cap = D + 1.0 if args.head == 'rms' else None
else:
    Ls = []
    for zb in eval_bits:
        zz = jnp.repeat(jnp.asarray(zb)[None], len(ss), 0)
        psi = psi_def.apply({'params': state['p']['psi']}, jnp.concatenate([o_e, a_e, zz], -1))
        phi_g = phi_def.apply({'params': state['p']['phi']}, g_e)
        Ls.append(np.asarray((psi * phi_g).sum(-1)))
    L = np.stack(Ls)
    feats = np.asarray(phi_def.apply({'params': state['p']['phi']}, jnp.asarray(encode(np.arange(S), False))))
    w_cos = None
    cap = None
L = L.reshape(16, S, A, S)
Rf, Lf = R.ravel(), L.ravel()
corr = float(np.corrcoef(Rf, Lf)[0, 1])
slope = float((Lf @ Rf) / (Rf @ Rf))
rel_l2_raw = float(np.linalg.norm(Lf - Rf) / np.linalg.norm(Rf))
rel_l2_fit = float(np.linalg.norm(slope ** -1 * Lf - Rf) / np.linalg.norm(Rf)) if slope != 0 else None
diag_mask = (gg.reshape(S, A, S) == T[:, :, None])[None].repeat(16, 0)


def summ(X):
    return {'diag_mean': float(X[diag_mask].mean()), 'offdiag_mean': float(X[~diag_mask].mean()),
            'rho_avg_mean': float((X * rho[None, None, None]).sum(-1).mean()),
            'max': float(X.max()), 'min': float(X.min())}


def within_z_corr(X):
    """per-z correlation, averaged: does M track the measure for each policy separately."""
    return float(np.mean([np.corrcoef(R[z].ravel(), X[z].ravel())[0, 1] for z in range(16)]))


def z_distinct(X):
    """mean pairwise correlation across z of the centred M_z vectors (1 = identical)."""
    V = X.reshape(16, -1)
    V = V - V.mean(0, keepdims=True)                  # remove the z-shared part
    share = float((V ** 2).sum() / ((X.reshape(16, -1) - X.mean()) ** 2).sum())
    C = np.corrcoef(X.reshape(16, -1))
    return {'pairwise_corr': float(C[np.triu_indices(16, 1)].mean()), 'z_variance_share': share}


def action_share(X):
    """variance over a at fixed (z, s, s+), as a share of total variance."""
    return float(X.var(axis=2).mean() / X.var())


def eff_rank(F):
    G = F.T @ F / F.shape[0]
    return float(np.trace(G) ** 2 / max((G ** 2).sum(), 1e-12))


# spike test on dataset rows: M at the exact recorded (s_i, a_i, s'_i) versus M at
# (s_i, a_i, s'_k) where s'_k is another row's next state in the SAME cell. Tabular one-hot: equal.
srng = np.random.default_rng(7)
ri = srng.integers(0, Nd, 2000)
by_cell = {c: np.flatnonzero(nx == c) for c in range(S)}
rk = np.array([srng.choice(by_cell[nx[i]]) for i in ri])
rj = srng.integers(0, Nd, 2000)


def model_M(o, a, g):
    o, a, g = jnp.asarray(o), jnp.asarray(a), jnp.asarray(g)
    if args.loss == 'psmgoal':
        ph_, b_ = state.basis(o, a, g)
        return np.asarray(ph_) @ wz.T + np.asarray(b_)[:, None]          # (n, Z)
    out = []
    for zb in eval_bits:
        zz = jnp.repeat(jnp.asarray(zb)[None], o.shape[0], 0)
        psi_ = psi_def.apply({'params': state['p']['psi']}, jnp.concatenate([o, a, zz], -1))
        phg = phi_def.apply({'params': state['p']['phi']}, g)
        out.append(np.asarray((psi_ * phg).sum(-1)))
    return np.stack(out, -1)


M_exact = model_M(obs_d[ri], act_d[ri], nobs_d[ri])
M_same = model_M(obs_d[ri], act_d[ri], nobs_d[rk])
M_rand = model_M(obs_d[ri], act_d[ri], nobs_d[rj])
spike = {'M_exact_next_mean': float(M_exact.mean()), 'M_same_cell_other_row_mean': float(M_same.mean()),
         'M_random_row_mean': float(M_rand.mean()),
         'exact_over_same_cell': float(M_exact.mean() / M_same.mean()) if M_same.mean() != 0 else None,
         'true_same_cell_ratio_mean': float(np.mean([R[:, st[i], ac[i], nx[i]].mean() for i in ri]))}

# truth for the z-AVERAGED policy (pi_bar(a|s) = mean over all codes); what M is if it ignores z
if 2 ** LOG_SEED <= 4096:
    all_codes = np.arange(2 ** LOG_SEED)
else:
    all_codes = np.random.default_rng(99).integers(0, 2 ** LOG_SEED, 4096)
all_bits = jnp.asarray(((all_codes[:, None] >> np.arange(LOG_SEED)) & 1).astype(np.float32))
if args.keying == 'state':
    acts_all = np.asarray(jax.vmap(lambda zb: proto_action(jnp.repeat(zb[None], S, 0), jnp.arange(S)))(all_bits))
    pibar = np.eye(A)[acts_all].mean(0)
else:
    pibar = pols.mean(0)
Pb = np.zeros((S, S))
for a_ in range(A):
    Pb[np.arange(S), T[:, a_]] += pibar[:, a_]
Mb = (1 - gamma) * np.linalg.inv(np.eye(S) - gamma * Pb)
Rbar = scale * ((1 - gamma) * np.eye(S)[T] + gamma * Mb[T]) / rho[None, None, :]
Lm = L.mean(0)
zavg = {'corr_learned_vs_zavg_truth': float(np.corrcoef(Lm.ravel(), Rbar.ravel())[0, 1]),
        'slope_learned_vs_zavg_truth': float((Lm.ravel() @ Rbar.ravel()) / (Rbar.ravel() @ Rbar.ravel())),
        'corr_true_z_vs_zavg_truth': float(np.mean([np.corrcoef(R[z].ravel(), Rbar.ravel())[0, 1] for z in range(16)]))}

# z-specific part: corr of (M_z - mean_z M) with (R_z - mean_z R)
Lc = L - L.mean(0, keepdims=True)
Rc = R - R.mean(0, keepdims=True)
zpart_corr = float(np.corrcoef(Lc.ravel(), Rc.ravel())[0, 1]) if Rc.std() > 0 else None
res = {
    'args': vars(args), 'hparams': hp, 'dataset_rows': int(Nd),
    'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
    'expected_scale': scale,
    'corr': corr, 'within_z_corr': within_z_corr(L), 'z_specific_corr': zpart_corr,
    'slope_learned_on_true': slope, 'rel_l2_raw': rel_l2_raw, 'rel_l2_after_slope': rel_l2_fit,
    'learned': summ(L), 'true': summ(R),
    'cap': cap, 'frac_at_cap': float((L > 0.95 * cap).mean()) if cap else None,
    'diag_frac_at_cap': float((L[diag_mask] > 0.95 * cap).mean()) if cap else None,
    'w_mean_pairwise_cos': w_cos,
    'z_distinct_learned': z_distinct(L), 'z_distinct_true': z_distinct(R),
    'phi_eff_rank': eff_rank(feats), 'D': D,
    'action_share_learned': action_share(L), 'action_share_true': action_share(R),
    'spike_rows': spike, 'zavg_policy': zavg, 'train_log_last': log[-1], 'train_log': log,
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('train_log',)}, indent=1))
