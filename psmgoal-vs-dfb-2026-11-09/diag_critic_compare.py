"""Same measurements on a psmgoal checkpoint and a density_fb_latent checkpoint (2026-11-09).

Question: both critics fit a successor measure over the frozen flow's latents. Under the same
recipe and anchor the factorised critic's actor reaches 0.89-0.95 and the joint psmgoal critic's
actor stays at BC (0.10-0.13). Which of these can the psmgoal checkpoint not do?

Per task (the task index w / z inferred exactly as at eval: ridge on a 10,000-row relabel batch):

1. INDEX FIT. kg(x) = (feature(x) - mu)^T w / sqrt(D), the per-state reward view the readout uses.
   Pearson / AUC vs the shifted relabel reward on a held-out relabel batch.
2. MEASURE vs TRUE FUTURES. At N dataset rows with >= 100 steps left in their episode, the critic's
   logits over a pool [own future state at h, 256 random dataset next states], at the dataset latent
   u_data. Per h in HORIZONS: percentile of the true future among the 256 random states (0.5 = chance),
   and its softmax share x 257 (1 = chance). Head 0.
   ACTION SENSITIVITY of the measure: the same percentile at h = 1 and 5 with u replaced by 8 random
   prior draws; u_data minus random says how much the measure knows that the recorded action leads to
   the recorded next states.
3. RANKING over 64 clipped prior draws at the acting score (pool-softmax readout at w, min over heads):
   within-state std / across-state std of the mean score; Spearman vs one simulator step of cube
   progress (tools/diag_index_ranking.py definitions, imported).
4. SLOPE. At the actor's latent u_a (actor at (s, w, eps)) and the BC rollout latent u_bc (the actor's
   own flow at the same eps):
     grad_norm        ||dQ/du|| at u_a
     slope_over_spread  ||dQ/du|| / within-state std of Q over the 64 draws (1 / latent units)
     q_term / bc_term per-row gradient norms of the two actor-loss terms at u_a:
                      q_coeff ||dQ/du|| / q_scale   vs   bc_coeff * 2 ||u_a - u_bc|| / d_u
                      (q_scale = mean |Q - mean Q| over the N rows, the reward_q_center normaliser;
                      both terms per row, before the 1/N batch mean that both share)
     ascent_gain      Q(u_bc + 0.25 g/|g|) - Q(u_bc), in units of the within-state std
     actor_shift      ||u_a - u_bc||; actor percentile among the 64 draws; one-step cube progress of u_a vs u_bc.

Run (GPU or CPU):
  MUJOCO_GL=egl .venv/bin/python psmgoal-vs-dfb-2026-11-09/diag_critic_compare.py --kind psmgoal \
      --run_dir $PSM_DATA/exp/PSMFLows/psmgoal_recipe_bc03_cube/sd000_... --epoch 500000 --report_out <json>
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import utils.xla_guard  # noqa: F401  -- MUST precede jax

# isort: split

import jax
import jax.numpy as jnp
import numpy as np

from tools.diag_index_ranking import auc, pearson, per_state_spearman, task_env_name
from tools.eval_measure_vs_rollouts import restore_row
from tools.successor_flow import load_latent_dataset, parse_blocks, rewarding_goal_set
from utils.log_utils import write_report

HORIZONS = (1, 2, 5, 10, 25, 50, 100)
N_RANDOM_POOL = 256
N_U_SENS = 8


# ---------------------------------------------------------------------------------------------- adapters
class PsmgoalAdapter:
    def __init__(self, agent):
        c = agent.config
        assert str(c.get("policy_index")) == "reward" and str(c.get("coef_source")) == "ridge"
        assert str(c.get("actor_kind")) == "flowbc_vf" and str(c.get("ridge_score")) == "pool_softmax", \
            "this adapter reads the recipe's readout (flowbc_vf actor, pool_softmax score)"
        self.a, self.c = agent, c
        self.w = jnp.asarray(agent.eval_w_star)
        self.H = agent._critic_heads()
        self.temp = float(c["measure_temp"])
        self.d = float(c["z_dim"])
        fa = c["fbc_actor"]
        self.q_coeff, self.bc_coeff = float(fa["q_coeff"]), float(fa["bc_coeff"])
        self.du, self.u_clip = int(c["action_dim"]), float(c["u_clip"])

    def _w(self, n):
        return jnp.broadcast_to(self.w, (n, self.w.shape[0]))

    def score_heads(self, obs, u):
        a = self.a
        qs = []
        for h in range(self.H):
            M = a.mesh_M(obs, u, a.index_pool, self._w(obs.shape[0]), head=None if self.H == 1 else h)
            qs.append(jnp.sum(jax.nn.softmax(M / self.temp, axis=1) * a.eval_pool_reward[None], axis=1))
        q_heads = jnp.stack(qs)
        return jnp.min(q_heads, 0), q_heads

    def logits_rowpool(self, obs, u, pools):
        """obs (N, ob), u (N, du), pools (N, P, ob) -> head-0 logits (N, P)."""
        a, w = self.a, self.w

        def one(o, uu, g):
            return a.mesh_M(o[None], uu[None], g, w[None], head=None if self.H == 1 else 0)[0]
        return jax.vmap(one)(obs, u, pools) / self.temp

    def kg(self, x):
        a = self.a
        f = a.fb_feature(x, a.fb_zenc, a.fb_anchor)
        return (f - a.fb_mu[None]) @ self.w / jnp.sqrt(self.d)

    def actor_u(self, obs, noise):
        return jnp.clip(self.a.fbc_actor_u(obs, self._w(obs.shape[0]), noise), -self.u_clip, self.u_clip)

    def bc_u(self, obs, noise):
        return self.a.fbc_rollout(self.a.fbc_vf.params, obs, noise)


class PsmgoalBaseAdapter(PsmgoalAdapter):
    """The reward-index BASE line (actor_kind=flowbc, ridge_score=goal), joint or factorized. It has no frozen
    readout pool, so the score is the flowbc actor's own value (actor_loss_flowbc_reward) at the task w: the
    softmax over a fixed pool of 256 relabel-batch next states of M(s,u,y;w), weighted by the reward view
    kg(y) = (f(y) - mu)^T w / sqrt(D). One head. u = clip(eps + delta(s, w, eps)); the BC latent is eps."""

    def __init__(self, agent, pool):
        c = agent.config
        assert str(c.get("policy_index")) == "reward" and str(c.get("coef_source")) == "ridge"
        assert str(c.get("actor_kind")) == "flowbc", "base adapter reads the flowbc (delta) actor"
        self.a, self.c = agent, c
        self.w = jnp.asarray(agent.eval_w_star)
        self.H = 1
        self.temp = float(c["measure_temp"])
        self.d = float(c["z_dim"])
        self.q_coeff, self.bc_coeff = float(c.get("q_coeff", 1.0)), float(c["fb_bc_coeff"])
        self.du, self.u_clip = int(c["action_dim"]), float(c["u_clip"])
        self.q_center = False
        self.pool = jnp.asarray(pool, jnp.float32)
        self.pool_r = jnp.asarray(self.kg(self.pool))

    def score_heads(self, obs, u):
        M = self.a.mesh_M(obs, u, self.pool, self._w(obs.shape[0]))
        q = jnp.sum(jax.nn.softmax(M / self.temp, axis=1) * self.pool_r[None], axis=1)
        return q, q[None]

    def logits_rowpool(self, obs, u, pools):
        a, w = self.a, self.w

        def one(o, uu, g):
            return a.mesh_M(o[None], uu[None], g, w[None])[0]
        return jax.vmap(one)(obs, u, pools) / self.temp

    def actor_u(self, obs, noise):
        return self.a.flowbc_latent(obs, self._w(obs.shape[0]), noise)[0]

    def bc_u(self, obs, noise):
        return jnp.clip(noise, -self.u_clip, self.u_clip)


class DfbAdapter:
    def __init__(self, agent):
        c = agent.config
        self.a, self.c = agent, c
        self.z = jnp.asarray(agent.z_eval)
        self.x_pool, self.basis = agent.eval_support()
        self.temp = float(c["temp"])
        self.d = float(c["z_dim"])
        self.q_coeff, self.bc_coeff = float(c["actor"]["q_coeff"]), float(c["actor"]["bc_coeff"])
        self.du, self.u_clip = int(c["action_dim"]), float(c["u_clip"])

    def _z(self, n):
        return jnp.broadcast_to(self.z, (n, self.z.shape[0]))

    def score_heads(self, obs, u):
        a = self.a
        q, _ = a.q_fn(a.params, obs, self._z(obs.shape[0]), u, self.x_pool, self.basis)
        _, heads = a._softmax_value(a.forward(obs, u, self._z(obs.shape[0])), a.backward(self.x_pool),
                                    self._z(obs.shape[0]), self.basis)
        return q, heads

    def logits_rowpool(self, obs, u, pools):
        a = self.a
        f = a.forward(obs, u, self._z(obs.shape[0]))[0]                       # head 0, (N, d)
        n, p, ob = pools.shape
        b = a.backward(a._b_in(pools.reshape(n * p, ob))).reshape(n, p, -1)
        return jnp.einsum("nd,npd->np", f, b) / self.temp

    def kg(self, x):
        a = self.a
        f = a.backward(a._b_in(x), a.params["density_b_zenc"])
        return (f - a.z_mu[None]) @ self.z / jnp.sqrt(self.d)

    def actor_u(self, obs, noise):
        return jnp.clip(self.a.actor_u(obs, self._z(obs.shape[0]), noise), -self.u_clip, self.u_clip)

    def bc_u(self, obs, noise):
        return self.a._rollout(self.a.params["actor_vf"], obs, noise)


# ---------------------------------------------------------------------------------------------- loading
def load_agent(kind, run_dir, epoch, flow_ckpt, flow_epoch, preimage_path, plain_ds, seed, relabel_size, reward_shift):
    """Restore as tools/eval_checkpoint.py does (flags.json defaults, path keys on top) and infer the eval index
    on a relabel batch drawn with np.random seeded by `seed`. Returns (adapter, relabel batch)."""
    import ml_collections

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import merge_run_config
    from utils.flax_utils import restore_agent

    if kind == "dfb":
        from agents.density_fb_latent import get_config
    else:
        from agents.psmgoal import get_config
    cli = get_config().to_dict()
    cli["flow_ckpt_path"], cli["flow_ckpt_epoch"], cli["preimage_path"] = str(flow_ckpt), int(flow_epoch), str(preimage_path)
    keys = {"flow_ckpt_path", "flow_ckpt_epoch", "preimage_path"}
    if kind == "psmgoal":
        # psmz (2026-11-10) is the same class under another registry name; build what the run recorded
        with open(os.path.join(str(run_dir), "flags.json")) as fh:
            _run_name = json.load(fh)["agent"]["agent_name"]
        assert _run_name in ("psmgoal", "psmz"), _run_name
        cli["agent_name"] = _run_name
    merged, _ = merge_run_config(cli, str(run_dir), keys)
    name = "density_fb_latent" if kind == "dfb" else cli["agent_name"]
    assert merged["agent_name"] == name, f"{run_dir} is a {merged['agent_name']} run, not {name}"
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    np.random.seed(int(seed))
    ex = plain_ds.sample(1)
    agent = agents[name].create(int(seed), ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, str(run_dir), int(epoch))
    zb = plain_ds.sample(min(plain_ds.size, int(relabel_size)))
    if kind == "dfb":
        agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + float(reward_shift))
        return DfbAdapter(agent), zb
    agent = agent.infer_eval_goals(zb, zb["rewards"] + float(reward_shift))
    if str(config.get("actor_kind")) == "flowbc":
        return PsmgoalBaseAdapter(agent, np.asarray(zb["next_observations"])[:256]), zb
    assert float(np.abs(np.asarray(agent.index_pool)).sum()) > 0, "psmgoal index_pool is empty (never filled)"
    return PsmgoalAdapter(agent), zb


def future_rows(next_obs, obs, h_max):
    """Rows i whose episode continues for h_max steps: next_obs[i + k] == obs[i + k + 1] for k < h_max - 1."""
    cont = np.zeros(obs.shape[0], bool)
    cont[:-1] = np.all(np.abs(next_obs[:-1] - obs[1:]) < 1e-6, axis=-1)
    # run length of consecutive continuations starting at i
    run = np.zeros(obs.shape[0], np.int64)
    for i in range(obs.shape[0] - 2, -1, -1):
        run[i] = run[i + 1] + 1 if cont[i] else 0
    return np.nonzero(run >= h_max - 1)[0]


# ---------------------------------------------------------------------------------------------- metrics
def measure_vs_futures(ad, ds, rows_f, rand_pool, u_data, key):
    obs = np.asarray(ds["observations"], np.float32)
    nxt = np.asarray(ds["next_observations"], np.float32)
    s = jnp.asarray(obs[rows_f])
    futs = np.stack([nxt[rows_f + h - 1] for h in HORIZONS], 1)                        # (N, nh, ob)
    n = len(rows_f)
    pools = jnp.asarray(np.concatenate([futs, np.broadcast_to(rand_pool[None], (n,) + rand_pool.shape)], 1))
    lg = jax.jit(ad.logits_rowpool)
    out = {}

    def stats(u):
        L = np.asarray(lg(s, jnp.asarray(u), pools))                                   # (N, nh + R)
        Lf, Lr = L[:, :len(HORIZONS)], L[:, len(HORIZONS):]
        pct = (Lr[:, None, :] < Lf[:, :, None]).mean(-1)                               # (N, nh)
        lse_r = np.logaddexp.reduce(Lr, axis=1, keepdims=True)                         # (N, 1)
        share = np.exp(Lf - np.logaddexp(Lf, lse_r))                                   # true future vs the R random
        return pct, share * (Lr.shape[1] + 1), L
    pct_d, share_d, L_d = stats(u_data)
    pct_r = []
    for j in range(N_U_SENS):
        u_r = np.asarray(jnp.clip(jax.random.normal(jax.random.fold_in(key, j), u_data.shape), -ad.u_clip, ad.u_clip))
        pct_r.append(stats(u_r)[0])
    pct_r = np.mean(pct_r, 0)
    for k, h in enumerate(HORIZONS):
        out[f"h{h}"] = {"pct_true_future_u_data": float(pct_d[:, k].mean()),
                        "share_x257_u_data": float(share_d[:, k].mean()),
                        "pct_true_future_u_random": float(pct_r[:, k].mean()),
                        "pct_u_data_minus_random": float((pct_d[:, k] - pct_r[:, k]).mean())}
    out["logit_std_over_pool"] = float(L_d[:, len(HORIZONS):].std(1).mean())
    return out


def one_step_progress_blocks(env, sim, rows, s_raw, actions, goal_blocks, blocks):
    """tools.diag_index_ranking.one_step_progress over every hit-spec block: progress = sum over blocks of the
    mean (over goals) decrease of the block's distance to the goal block; moved = any block moved > 1e-4."""
    N, C = actions.shape[:2]
    prog, moved = np.zeros((N, C)), np.zeros((N, C))
    for i in range(N):
        d0 = sum(np.linalg.norm(goal_blocks[:, lo:hi] - s_raw[i, a:b][None], axis=-1).mean()
                 for (a, b, _), (lo, hi) in zip(blocks, _block_cols(blocks)))
        for j in range(C):
            restore_row(env, sim, int(rows[i]))
            ob, _, _, _, _ = env.step(np.asarray(actions[i, j], np.float32))
            ob = np.asarray(ob, np.float32)
            d1 = sum(np.linalg.norm(goal_blocks[:, lo:hi] - ob[a:b][None], axis=-1).mean()
                     for (a, b, _), (lo, hi) in zip(blocks, _block_cols(blocks)))
            prog[i, j] = d0 - d1
            moved[i, j] = float(max(np.linalg.norm(ob[a:b] - s_raw[i, a:b]) for a, b, _ in blocks) > 1e-4)
    return prog, moved


def _block_cols(blocks):
    cols, lo = [], 0
    for a, b, _ in blocks:
        cols.append((lo, lo + b - a)); lo += b - a
    return cols


def training_reward_density(nxt, obs, blocks, rng, n_batches=64, batch=256, horizons=(5, 25)):
    """How many of a 256-row batch's next states fall inside the hit region of a row's hindsight goal (the
    training ridge's positives), mean over rows and batches, per goal horizon. 0 = the goal's reward is 1
    only on states the batch does not hold (no positive besides, at most, the row's own later states)."""
    cand = future_rows(nxt, obs, max(horizons) + 1)
    out = {}
    for h in horizons:
        counts = []
        for _ in range(n_batches):
            rows = rng.choice(cand, size=batch, replace=False)
            pool = nxt[rows]
            goals = nxt[rows + h - 1]
            hit = np.ones((batch, batch), bool)
            for a, b, eps in blocks:
                d = np.linalg.norm(pool[None, :, a:b] - goals[:, None, a:b], axis=-1)
                hit &= d < eps
            counts.append(hit.sum(1).mean())
        out[f"h{h}_positives_per_batch"] = float(np.mean(counts))
    return out


def run_task(a, kind, task, rng_seed):
    env_name = task_env_name(a.env, task)
    env, _, train_dataset, _ = load_latent_dataset(env_name, a.preimage_path, add_info=True)
    sim = {k: np.asarray(train_dataset[k]) for k in ("qpos", "qvel", "button_states") if k in train_dataset}
    env.unwrapped._terminate_at_goal = False
    from utils.datasets import Dataset
    if kind == "dfb":
        plain = Dataset.create(**{k: v for k, v in train_dataset.items()
                                  if not (k.startswith("noise_preimage") or k == "preimage_valid")})
    else:
        # psmgoal eval: preimage arrays spliced in, validity array kept out (tools/eval_checkpoint.py)
        plain = Dataset.create(**{k: v for k, v in train_dataset.items() if k != "preimage_valid"})
        plain.return_preimage_noise = True
        plain.preimage_point_mode = True
    ad, zb_fit = load_agent(kind, a.run_dir, a.epoch, a.flow_ckpt, a.flow_epoch, a.preimage_path, plain, a.seed,
                            a.relabel_size, a.reward_shift)
    zb_out = plain.sample(min(plain.size, int(a.relabel_size)))
    res = {"env_name": env_name}

    # 1. index fit
    kgf = jax.jit(ad.kg)
    for nm, zb in (("fit", zb_fit), ("heldout", zb_out)):
        x = np.asarray(zb["next_observations"], np.float32)
        kg = np.concatenate([np.asarray(kgf(jnp.asarray(x[i:i + 512]))) for i in range(0, len(x), 512)])
        r = np.asarray(zb["rewards"], np.float64).reshape(-1) + a.reward_shift
        res[f"index_{nm}"] = {"pearson": pearson(kg, r), "auc": auc(kg, r > 0.5), "n_rewarding": int((r > 0.5).sum())}

    rng = np.random.default_rng(int(a.seed) + 100 * task)
    obs_all = np.asarray(train_dataset["observations"], np.float32)
    nxt_all = np.asarray(train_dataset["next_observations"], np.float32)
    valid = np.asarray(train_dataset["preimage_valid"]) > 0.5
    u_pre = np.asarray(train_dataset["noise_preimage_point"], np.float32)
    key = jax.random.PRNGKey(int(a.seed) + 11 + task)
    k_u, k_act, k_sens = jax.random.split(key, 3)

    # 2. measure vs true futures
    cand = future_rows(nxt_all, obs_all, max(HORIZONS))
    cand = cand[valid[cand]]
    rows_f = np.sort(rng.choice(cand, size=min(int(a.n_states), len(cand)), replace=False))
    rand_pool = nxt_all[rng.choice(obs_all.shape[0], size=N_RANDOM_POOL, replace=False)]
    u_data_f = np.clip(u_pre[rows_f], -ad.u_clip, ad.u_clip)
    res["measure"] = measure_vs_futures(ad, train_dataset, rows_f, rand_pool, u_data_f, k_sens)
    res["measure"]["n_rows"] = len(rows_f)

    # 3. ranking over prior draws, 4. slope
    vrows = np.nonzero(valid)[0]
    rows = np.sort(rng.choice(vrows, size=int(a.n_states), replace=False))
    s_raw = obs_all[rows]
    N, K, du = len(rows), int(a.n_u), ad.du
    u_draw = np.asarray(jnp.clip(jax.random.normal(k_u, (N, K, du)), -ad.u_clip, ad.u_clip))
    score = jax.jit(lambda o, u: ad.score_heads(o, u)[0])
    q = np.asarray(score(jnp.asarray(np.repeat(s_raw, K, 0)), jnp.asarray(u_draw.reshape(-1, du)))).reshape(N, K)
    noise = jax.random.normal(k_act, (N, du))
    sj = jnp.asarray(s_raw)
    u_a = np.asarray(jax.jit(ad.actor_u)(sj, noise))
    u_bc = np.asarray(jax.jit(ad.bc_u)(sj, noise))
    grad = jax.jit(jax.grad(lambda u, o: score(o, u).sum()))
    g_a = np.asarray(grad(jnp.asarray(u_a), sj))
    g_bc = np.asarray(grad(jnp.asarray(u_bc), sj))
    q_a = np.asarray(score(sj, jnp.asarray(u_a)))
    q_bc = np.asarray(score(sj, jnp.asarray(u_bc)))
    gn = np.linalg.norm(g_bc, axis=-1, keepdims=True)
    u_step = np.clip(u_bc + 0.25 * g_bc / np.maximum(gn, 1e-12), -ad.u_clip, ad.u_clip)
    q_step = np.asarray(score(sj, jnp.asarray(u_step)))
    within = q.std(1)
    across = q.mean(1).std()
    centred = getattr(ad, "q_center", True)
    q_scale = max(float(np.abs(q_a - q_a.mean() if centred else q_a).mean()), 1e-6)
    q_term = ad.q_coeff * np.linalg.norm(g_a, axis=-1) / q_scale
    bc_term = ad.bc_coeff * 2.0 * np.linalg.norm(u_a - u_bc, axis=-1) / du
    # simulator: draws, actor, bc
    goals, n_rew = rewarding_goal_set(zb_fit["next_observations"], np.asarray(zb_fit["rewards"]) + a.reward_shift,
                                      a.k_goals, rng)
    blocks = parse_blocks(ad.c["fb_hit_spec"])
    goal_blocks = np.concatenate([goals[:, b0:b1] for b0, b1, _ in blocks], 1)
    u_all = np.concatenate([u_draw, u_a[:, None], u_bc[:, None]], 1)
    dec = jax.jit(ad.a.decode)
    acts = np.asarray(dec(jnp.asarray(np.repeat(s_raw, K + 2, 0)), jnp.asarray(u_all.reshape(-1, du)))).reshape(N, K + 2, -1)
    prog_c, moved = one_step_progress_blocks(env, sim, rows, s_raw, acts, goal_blocks, blocks)
    res["training_reward"] = training_reward_density(nxt_all, obs_all, blocks, rng)
    res["training_reward"]["relabel_rewarding_per_10k"] = int(n_rew)
    per_c, pooled_c = per_state_spearman(q, prog_c[:, :K])
    ar = np.arange(N)
    pick = q.argmax(1)
    res["n_rewarding_goals"] = int(n_rew)
    res["ranking"] = {
        "within_state_std": float(within.mean()), "across_state_std_of_mean": float(across),
        "within_over_across": float(within.mean() / max(across, 1e-12)),
        "spearman_q_vs_progress_cube": float(np.nanmean(per_c)), "pooled_spearman_q_vs_progress_cube": float(pooled_c),
        "pick_progress_cube": float(prog_c[ar, pick].mean()), "rand_progress_cube": float(prog_c[:, 0].mean()),
        "best_progress_cube": float(prog_c[:, :K].max(1).mean()), "frac_draws_cube_moved": float(moved[:, :K].mean())}
    res["slope"] = {
        "grad_norm_at_actor": float(np.linalg.norm(g_a, axis=-1).mean()),
        "grad_norm_at_bc": float(gn.mean()),
        "slope_over_spread": float(np.mean(gn[:, 0] / np.maximum(within, 1e-12))),
        "q_scale": q_scale, "q_term_grad": float(q_term.mean()), "bc_term_grad": float(bc_term.mean()),
        "q_over_bc_term": float(np.median(q_term / np.maximum(bc_term, 1e-12))),
        "ascent_gain_over_spread": float(np.mean((q_step - q_bc) / np.maximum(within, 1e-12))),
        "actor_minus_bc_q_over_spread": float(np.mean((q_a - q_bc) / np.maximum(within, 1e-12))),
        "actor_shift": float(np.linalg.norm(u_a - u_bc, axis=-1).mean()),
        "actor_percentile": float((q < q_a[:, None]).mean(1).mean()),
        "bc_percentile": float((q < q_bc[:, None]).mean(1).mean()),
        "actor_progress_cube": float(prog_c[:, K].mean()), "bc_progress_cube": float(prog_c[:, K + 1].mean()),
        "q_coeff": ad.q_coeff, "bc_coeff": ad.bc_coeff}
    env.close()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("psmgoal", "dfb"), required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--epoch", type=int, default=500000)
    p.add_argument("--tasks", default="1,2,3,4,5")
    p.add_argument("--n_states", type=int, default=256)
    p.add_argument("--n_u", type=int, default=64)
    p.add_argument("--env", default="cube-single-play-singletask-v0")
    p.add_argument("--flow_ckpt", default=os.path.join(os.environ.get("PSM_DATA", ""), "flow", "cube-single-play"))
    p.add_argument("--flow_epoch", type=int, default=500000)
    p.add_argument("--preimage_path",
                   default=os.path.join(os.environ.get("PSM_DATA", ""), "preimages", "cube-single-play.npz"))
    p.add_argument("--relabel_size", type=int, default=10000)
    p.add_argument("--reward_shift", type=float, default=1.0)
    p.add_argument("--k_goals", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_out", required=True)
    a = p.parse_args()
    t0 = time.time()
    tasks = [int(t) for t in a.tasks.split(",")]
    report = {"args": vars(a), "definitions": __doc__, "per_task": {}}
    for t in tasks:
        tt = time.time()
        report["per_task"][str(t)] = run_task(a, a.kind, t, a.seed)
        print(f"task {t} ({time.time() - tt:.0f} s): " + json.dumps(report["per_task"][str(t)]), flush=True)

    def mean_tree(xs):
        if isinstance(xs[0], dict):
            return {k: mean_tree([x[k] for x in xs]) for k in xs[0] if k != "env_name"}
        return float(np.nanmean(xs))
    report["mean_over_tasks"] = mean_tree([report["per_task"][str(t)] for t in tasks])
    report["seconds"] = round(time.time() - t0, 1)
    print("mean over tasks: " + json.dumps(report["mean_over_tasks"], indent=1), flush=True)
    write_report(report, {"report_out": a.report_out}, "diag_critic_compare.json")


if __name__ == "__main__":
    main()
