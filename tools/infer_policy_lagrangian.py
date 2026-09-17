"""PSM's test-time policy inference (arXiv 2411.19418 Sec. 5.3, Eq. 10) over the affine
coefficient of a trained LatentFlowPSM checkpoint. Eval-only; writes the coefficient npz
that `acting=fixed_coeff` deploys.

Paper -> ours
-------------
PSM writes the successor measure of ANY policy as M = Phi w + b, affine in an unconstrained
coefficient w, and at test time solves

    max_w  E[(Phi w + b) r]   s.t.  Phi w + b >= 0 on every (s, a)

as the Lagrangian  max_{lambda >= 0} min_w  -(Phi w) r - sum lambda(s,a) min(Phi w + b, 0)
by gradient descent-ascent, then acts on Q* = M* r.

Here the affine head is psi(s, u, u') = A(s,u)^T c + beta(s,u) with c = w(u') in R^{w_dim}
(the encoder's output, unit-norm by `norm_w`). The measure density of the policy with
coefficient c is m_c(s, u, x) = psi_c(s,u)^T phi(x) and its value is Q_c(s, u) = psi_c^T w
with w the closed-form task vector `infer_z` reads off 10k relabel rows (shift 1.0, sphere
projected), exactly as tools/eval_checkpoint.py infers it. The decision variable is c:

    Phi w + b   ->  A(s_i, u_i)^T c + beta(s_i, u_i)            (per dataset transition i)
    (Phi w + b) r   ->  psi_c(s_i, u_i)^T w_task               (objective, mean over i)
    Phi w + b >= 0  ->  psi_c(s_i, u_i)^T phi(s'_j) >= 0        (constraint on pairs (i, j))

The constraint samples x_j = s'_j are the same rows' next states, the negatives the measure
loss trains against. `free` leaves c unconstrained like the paper's w; `sphere` renormalises
it to the encoder's unit sphere after every step (is a better member of the trained family
enough, or does the optimum need to leave it?).

Procedure (per checkpoint, per task)
------------------------------------
  * N = 10,000 dataset transitions (s_i, u_i = point preimage, s'_i), one fixed draw shared
    by every task and variant; A_i, beta_i, phi(s'_i) precomputed once. A disjoint held-out
    set of H rows carries every reported number.
  * scale S = batch std of the objective term at the init coefficient; objective and
    constraint are divided by S so the learning rates mean something across tasks.
  * init c0 = the family member with the highest batch-mean value: among 4,096 clipped prior
    draws u', the w(u') maximising mean_i Q(s_i, u_i, u') (GPI's inner max averaged over
    states rather than taken per state).
  * L(c, lambda) = -J(c) + sum_i lambda_i v_i(c), v_i = mean_j relu(-m_c(i, j)) over a
    fresh (rows x cols) subsample per step. Adam on c; projected gradient ascent on lambda
    (`row`: one multiplier per dataset row, updated on the rows sampled that step; `scalar`:
    one multiplier on the pair mean). J is linear in c, so its gradient is exact from the
    full-batch means and only the constraint is stochastic.

Diagnostics per (task, variant), JSON via `report_out`
------------------------------------------------------
  objective (raw units; ensemble mean and the acting rule's mean - kappa * unc) at c0 and
  c* on train and held-out rows; constraint violation fraction and mean magnitude on
  held-out pairs at c0 and c*; ||c*||, cos(c*, c0), cos to the nearest of the 4,096 family
  coefficients and to their mean (did c* leave the trained family?); Q_{c*} against
  Q_GPI = max_{u' in 64} [mean - kappa*unc] psi^T w on 2,000 held-out states x 64 latents
  (pooled Pearson/Spearman, per-state Spearman, argmax agreement, Q_GPI's regret at the
  Q_{c*} argmax); and, when `na_path` names a DSRL-NA scalar critic and the task is
  `na_task`, per-state Spearman of Q_{c*} with Q_s(s, u) = min_e qa(s, G(s, u)) and the
  scalar critic's regret at the Q_{c*} argmax (tools/diag_measure_vs_scalar_q.py block C).

Outputs: `report_out` JSON (+ .npz sidecar with curves and coefficients) and, per
(task, variant), `<out_dir>/eq10_<variant>_<tag>_task<t>.npz` holding `c`, `w_task`, the
env id and the checkpoint it was fitted on -- what `agent.fixed_index_coeff_path` takes.

Run (one checkpoint, all five cube tasks):
  .venv/bin/python tools/infer_policy_lagrangian.py agent=psmflow \\
      env_name=cube-single-play-singletask-v0 \\
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
      restore_path=<affine_strict_cube run dir> restore_epoch=500000 \\
      +tasks=[1,2,3,4,5] +out_dir=$PSM_DATA/logs/eq10 +tag=sd001 \\
      +na_path=<cube_dsrlna_rhat_scaled sd001 run dir> +na_task=2 \\
      report_out=$PSM_DATA/logs/eq10/infer_policy_lagrangian_cube_sd001.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py)

ROW_SEED = 20260915      # the dataset rows (dedicated Generator, never the global stream)
FAMILY_KEY = 4242        # the 4,096 prior draws u' that define the family and the init
PANEL_KEY = 4343         # the held-out (u, u') panels of the Q_{c*} vs Q_GPI block
STEP_KEY = 4444          # per-step row/column subsamples of the constraint

#: (name, c_mode, lam_mode, lr, lam_lr). `free` is the variant the evals deploy.
VARIANTS = (
    ("free", "free", "row", 1e-3, 1e-3),
    ("free_lr1e-4", "free", "row", 1e-4, 1e-3),
    ("free_scalar", "free", "scalar", 1e-3, 1e-3),
    ("free_fastdual", "free", "row", 1e-3, 1e-1),
    ("sphere", "sphere", "row", 1e-3, 1e-3),
    ("sphere_lr1e-4", "sphere", "row", 1e-4, 1e-3),
)


# ---------------------------------------------------------------- pure helpers (tested)
def objective_linear(c, a_bar, b_bar):
    """J(c) = mean_i psi_c(s_i,u_i)^T w = a_bar . c + b_bar, with a_bar = mean_{P,i} A_i w
    and b_bar = mean_{P,i} beta_i^T w. Exact: J is linear in c."""
    return float(np.asarray(a_bar, np.float64) @ np.asarray(c, np.float64) + float(b_bar))


def family_init(W_fam, a_bar):
    """Index of the family coefficient with the largest batch-mean value (b_bar is common)."""
    return int(np.argmax(np.asarray(W_fam, np.float64) @ np.asarray(a_bar, np.float64)))


def violation_stats(m):
    """Fraction of pairs with m < 0 and the mean of relu(-m) over ALL pairs. m any shape."""
    m = np.asarray(m, np.float64)
    neg = np.maximum(-m, 0.0)
    return {"frac_violated": float((m < 0).mean()), "mean_violation": float(neg.mean()),
            "mean_violation_given_violated": float(neg[m < 0].mean()) if (m < 0).any() else 0.0,
            "min_m": float(m.min()), "mean_m": float(m.mean()), "n_pairs": int(m.size)}


def family_geometry(c, W_fam):
    """||c||, cos to the nearest family coefficient, to the family mean, and the distance
    to the nearest member. W_fam (K, d) rows are unit-norm when `norm_w`."""
    c = np.asarray(c, np.float64)
    W = np.asarray(W_fam, np.float64)
    nc = np.linalg.norm(c) + 1e-12
    cosines = (W @ c) / (np.linalg.norm(W, axis=1) * nc + 1e-12)
    mean_w = W.mean(0)
    k = int(np.argmax(cosines))
    return {"norm": float(np.linalg.norm(c)), "cos_nearest_family": float(cosines[k]),
            "nearest_family_index": k, "dist_nearest_family": float(np.linalg.norm(W - c, axis=1).min()),
            "cos_family_mean": float(mean_w @ c / (np.linalg.norm(mean_w) * nc + 1e-12)),
            "family_mean_norm": float(np.linalg.norm(mean_w)),
            "cos_family_quantiles": [float(q) for q in np.quantile(cosines, [0.0, 0.5, 0.9, 0.99, 1.0])]}


def project_c(c, c_mode):
    if c_mode == "sphere":
        return c / (np.linalg.norm(c) + 1e-12)
    return c


def lagrangian_fit(A, beta, phi, w, c0, scale, *, steps, lr, lam_lr, c_mode, lam_mode,
                   rows_per_step, cols_per_step, key, log_every=500, log=print):
    """Descent-ascent on L(c, lambda) = -J(c)/S + sum_i lambda_i v_i(c)/S.

    A (P, N, z, d_w), beta (P, N, z), phi (N, z) jax arrays on device; w (z,); c0 (d_w,);
    `scale` S. Returns (c*, lambda*, curve) with curve a list of dicts every `log_every`
    steps. Every stochastic quantity comes off `key`.
    """
    import jax
    import jax.numpy as jnp
    import optax

    P, N = A.shape[0], A.shape[1]
    w = jnp.asarray(w, jnp.float32)
    a_bar = jnp.einsum("pizw,z->w", A, w) / (P * N)                    # (d_w,)
    b_bar = jnp.einsum("piz,z->", beta, w) / (P * N)                   # ()
    inv_s = 1.0 / float(scale)
    opt = optax.adam(lr)
    c = jnp.asarray(c0, jnp.float32)
    opt_state = opt.init(c)
    lam = jnp.zeros((N,), jnp.float32) if lam_mode == "row" else jnp.zeros((), jnp.float32)

    def constraint_rows(c_, rows, cols):
        psi = jnp.einsum("pizw,w->piz", A[:, rows], c_) + beta[:, rows]   # (P, R, z)
        m = jnp.einsum("iz,jz->ij", psi.mean(0), phi[cols]) * inv_s    # (R, C)
        return jnp.maximum(-m, 0.0).mean(1), m                          # v_i (R,), m

    def lagrangian(c_, lam_rows, rows, cols):
        v, _ = constraint_rows(c_, rows, cols)
        J = (a_bar @ c_ + b_bar) * inv_s
        return -J + (lam_rows * v).mean() if lam_mode == "row" else -J + lam_rows * v.mean(), v

    @jax.jit
    def step(c_, opt_state_, lam_, k):
        k_r, k_c = jax.random.split(k)
        rows = jax.random.permutation(k_r, N)[:rows_per_step]
        cols = jax.random.permutation(k_c, N)[:cols_per_step]
        lam_rows = lam_[rows] if lam_mode == "row" else lam_
        (loss, v), g = jax.value_and_grad(lagrangian, has_aux=True)(c_, lam_rows, rows, cols)
        upd, opt_state_ = opt.update(g, opt_state_, c_)
        c_new = optax.apply_updates(c_, upd)
        if c_mode == "sphere":
            c_new = c_new / (jnp.linalg.norm(c_new) + 1e-12)
        if lam_mode == "row":
            lam_ = lam_.at[rows].set(jnp.maximum(lam_rows + lam_lr * v, 0.0))
        else:
            lam_ = jnp.maximum(lam_ + lam_lr * v.mean(), 0.0)
        return c_new, opt_state_, lam_, loss, v.mean(), (v > 0).mean(), jnp.linalg.norm(g)

    curve = []
    t0 = time.time()
    stats = None                       # (loss, mean violation, violated frac, grad norm) of the last step
    for it in range(steps + 1):
        if it % log_every == 0 or it == steps:
            J = float(a_bar @ c + b_bar)
            rec = {"step": it, "objective_raw": J, "objective_scaled": J * inv_s,
                   "c_norm": float(jnp.linalg.norm(c)),
                   "lambda_mean": float(jnp.mean(lam)), "lambda_max": float(jnp.max(lam)),
                   "lambda_frac_active": float(jnp.mean(lam > 0)) if lam_mode == "row" else float(lam > 0),
                   "seconds": round(time.time() - t0, 1)}
            if stats is not None:
                rec.update({"loss": float(stats[0]), "step_violation_mean": float(stats[1]),
                            "step_violation_frac": float(stats[2]), "grad_norm": float(stats[3])})
            curve.append(rec)
            log(f"    step {it:5d}  J {J:+.4f} (scaled {J * inv_s:+.4f})  |c| {rec['c_norm']:.3f}  "
                f"lambda mean {rec['lambda_mean']:.4f} max {rec['lambda_max']:.4f}"
                + (f"  viol frac {float(stats[2]):.3f} mean {float(stats[1]):.4f}" if stats is not None else ""))
        if it == steps:
            break
        c, opt_state, lam, *stats = step(c, opt_state, lam, jax.random.fold_in(key, it))
        if not bool(jnp.isfinite(c).all()):
            log(f"    step {it}: c is not finite; stopping")
            curve.append({"step": it, "diverged": True})
            break
    return np.asarray(c, np.float64), np.asarray(lam, np.float64), curve


def as_list(v, cast=str):
    """A hydra override typed as `key=[a,b]` (a ListConfig), a python list, or a string
    'a,b' / '[a,b]' -> [cast(a), cast(b)]."""
    if isinstance(v, str):
        items = v.strip("[]").split(",")
    else:
        items = list(v)
    return [cast(str(x).strip().strip("'\"")) for x in items if str(x).strip()]


def ci95(xs):
    xs = np.asarray(xs, np.float64)
    if xs.size < 2:
        return 0.0
    return float(1.96 * xs.std(ddof=1) / np.sqrt(xs.size))


# ------------------------------------------------------------------------- the GPU probe
def _main(cfg):
    import jax
    import jax.numpy as jnp
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.diag_measure_vs_scalar_q import agreement, center_rows, per_state_ranking, summarize_rho
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
    from utils.log_utils import write_report
    from utils.psm_common import targets_uncertainty

    seed = int(cfg.seed)
    tasks = as_list(cfg.get("tasks", "1,2,3,4,5"), int)
    tag = str(cfg.get("tag", "sd"))
    out_dir = str(cfg.get("out_dir", os.path.join(os.getcwd(), "eq10")))
    os.makedirs(out_dir, exist_ok=True)
    N = int(cfg.get("n_rows", 10000))
    H = int(cfg.get("n_holdout", 2000))
    K_fam = int(cfg.get("n_family", 4096))
    K = int(cfg.get("n_panel", 64))
    steps = int(cfg.get("steps", 5000))
    rows_per_step = int(cfg.get("rows_per_step", 1024))
    cols_per_step = int(cfg.get("cols_per_step", 256))
    chunk = int(cfg.get("chunk", 500))
    wanted = set(as_list(cfg.get("variants", [x[0] for x in VARIANTS])))
    variants = [v for v in VARIANTS if v[0] in wanted]
    assert variants, f"no known variant among {sorted(wanted)}; known: {[x[0] for x in VARIANTS]}"
    na_path = cfg.get("na_path", None)
    na_epoch = int(cfg.get("na_epoch", 500000))
    na_task = int(cfg.get("na_task", 2))
    reward_raw_path = cfg.get("reward_raw_path", None)
    env_base = str(cfg.env_name).split("-singletask")[0]

    def env_of(t):
        return f"{env_base}-singletask-task{t}-v0"

    # ---- the measure agent, restored under the run's own config ----
    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(env_of(tasks[0]), frame_stack=cfg.frame_stack)
    ds0 = Dataset.create(**train_dataset)
    ex = ds0.sample(1)
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    m_conf = ml_collections.ConfigDict(_lists_to_tuples(merged))
    assert m_conf["psi_form"] == "affine" and m_conf["policy_index"] == "latent", (
        "Eq. 10 over the affine coefficient needs psi_form=affine, policy_index=latent")
    assert m_conf.get("psi_bound", "none") != "tanh", "the tanh bound breaks the affine factorisation"
    agent = agents[m_conf["agent_name"]].create(seed, ex["observations"], ex["actions"], m_conf)
    agent = restore_agent(agent, cfg.restore_path, int(cfg.restore_epoch))
    mc = agent.config
    d_a, z_dim, P = int(mc["action_dim"]), int(mc["z_dim"]), int(mc["num_parallel"])
    w_dim = int(mc["affine"]["w_dim"])
    u_clip, ic, kappa = float(mc["u_clip"]), float(agent._index_clip()), float(mc["actor_pessimism_penalty"])
    run_dir = os.path.normpath(str(cfg.restore_path))

    na_agent = None
    if na_path:
        na_merged, _ = merge_run_config(cli_agent, str(na_path), _cli_agent_keys())
        na_conf = ml_collections.ConfigDict(_lists_to_tuples(na_merged))
        assert na_conf["dsrl_na"]["enabled"], f"{na_path} was not trained with dsrl_na.enabled"
        na_agent = agents[na_conf["agent_name"]].create(seed, ex["observations"], ex["actions"], na_conf)
        na_agent = restore_agent(na_agent, str(na_path), na_epoch)
        assert (mc["flow_ckpt_path"] == na_agent.config["flow_ckpt_path"]
                and int(mc["flow_ckpt_epoch"]) == int(na_agent.config["flow_ckpt_epoch"])), (
            "the scalar critic decodes through a different flow")

    # ---- rows ----
    aug = load_augmented_dataset(mc["preimage_path"])
    aug, valid = repair_invalid_preimages(aug)
    assert aug["observations"].shape[0] == ds0.size
    rng = np.random.default_rng(ROW_SEED)
    ok = np.flatnonzero(np.asarray(valid) > 0.5)
    pick = rng.choice(ok, size=N + H, replace=False)
    rows, rows_h = pick[:N], pick[N:]
    obs = np.asarray(aug["observations"][rows], np.float32)
    next_obs = np.asarray(aug["next_observations"][rows], np.float32)
    u_data = np.clip(np.asarray(aug["noise_preimage_point"][rows], np.float32), -u_clip, u_clip)
    obs_h = np.asarray(aug["observations"][rows_h], np.float32)
    next_obs_h = np.asarray(aug["next_observations"][rows_h], np.float32)
    u_data_h = np.clip(np.asarray(aug["noise_preimage_point"][rows_h], np.float32), -u_clip, u_clip)
    print(f"rows: {N} train + {H} held-out of {ok.size} valid", flush=True)

    @jax.jit
    def sa_terms(o, u):
        return agent.psi(o, agent._measure_input(o, u), method="sa_terms")   # (P,B,z,w),(P,B,z)

    @jax.jit
    def phi_of(x):
        return agent.phi(x)

    def precompute(o, u, x):
        As, Bs = [], []
        for lo in range(0, o.shape[0], chunk):
            A_, b_ = sa_terms(jnp.asarray(o[lo:lo + chunk]), jnp.asarray(u[lo:lo + chunk]))
            As.append(A_)
            Bs.append(b_)
        return jnp.concatenate(As, 1), jnp.concatenate(Bs, 1), phi_of(jnp.asarray(x))

    t0 = time.time()
    A, beta, phi_next = precompute(obs, u_data, next_obs)             # (P,N,z,w),(P,N,z),(N,z)
    A_h, beta_h, phi_next_h = precompute(obs_h, u_data_h, next_obs_h)
    print(f"precompute: A {tuple(A.shape)} in {time.time() - t0:.0f}s", flush=True)

    # ---- the family: 4,096 prior draws u' and their coefficients w(u') ----
    k_fam = jax.random.PRNGKey(FAMILY_KEY)
    u_fam = jnp.clip(jax.random.normal(k_fam, (K_fam, d_a)), -ic, ic)
    W_fam = np.asarray(agent.psi(u_fam, method="encode_index"), np.float64)     # (K_fam, w_dim)
    fam_pair_cos = float((W_fam @ W_fam.T)[np.triu_indices(K_fam, 1)].mean())
    print(f"family: {K_fam} coefficients, mean pairwise cos {fam_pair_cos:.4f}", flush=True)

    # ---- held-out panels for Q_{c} vs Q_GPI: H states x K latents, K indices per state ----
    kp_u, kp_i = jax.random.split(jax.random.PRNGKey(PANEL_KEY))
    u_panel = jnp.clip(jax.random.normal(kp_u, (H, K, d_a)), -u_clip, u_clip)     # (H, K, d_a)
    u_index = jnp.clip(jax.random.normal(kp_i, (H, K, d_a)), -ic, ic)             # (H, K, d_a)

    @jax.jit
    def panel_values(o, up, uidx, w, c_list):
        """For one chunk of states: Q_GPI (B, K) = max over the K indices of the
        pessimistic readout, and Q_c (B, K) at each coefficient in c_list."""
        B = o.shape[0]
        wb = jnp.broadcast_to(w, (B, z_dim))
        w_idx = agent.psi(uidx, method="encode_index")                           # (B, K, w_dim)

        def per_u(u_k):                                                          # u_k (B, d_a)
            A_, b_ = sa_terms(o, u_k)                                            # (P,B,z,w),(P,B,z)
            Aw = jnp.einsum("pbzw,bz->pbw", A_, wb)                              # (P, B, w)
            bw = jnp.einsum("pbz,bz->pb", b_, wb)                                # (P, B)
            q_idx = jnp.einsum("pbw,bkw->pkb", Aw, w_idx) + bw[:, None, :]       # (P, K, B)
            qm, qu = targets_uncertainty(q_idx, P)
            q_gpi = (qm - kappa * qu).max(0)                                     # (B,)
            outs = [q_gpi]
            for c_ in c_list:
                q_c = jnp.einsum("pbw,w->pb", Aw, c_) + bw                       # (P, B)
                cm, cu = targets_uncertainty(q_c, P)
                outs.append(cm - kappa * cu)
            return jnp.stack(outs)                                               # (1+n_c, B)
        return jax.lax.map(per_u, jnp.swapaxes(up, 0, 1))                       # (K, 1+n_c, B)

    @jax.jit
    def scalar_panel(o, up):
        def f(u_k):
            a = na_agent.decode(o, u_k)
            return na_agent.qa(o, a).min(0)
        return jax.lax.map(f, jnp.swapaxes(up, 0, 1))                           # (K, B)

    def readouts(Aa, bb, w, c, pess):
        """Objective terms on precomputed (A, beta): ensemble mean and pessimistic."""
        q = jnp.einsum("pizw,w->piz", Aa, jnp.asarray(c, jnp.float32)) + bb     # (P, N, z)
        q = jnp.einsum("piz,z->pi", q, jnp.asarray(w, jnp.float32))              # (P, N)
        qm, qu = targets_uncertainty(q, P)
        return np.asarray(qm, np.float64), np.asarray(qm - pess * qu, np.float64)

    def holdout_violation(c):
        psi = jnp.einsum("pizw,w->piz", A_h, jnp.asarray(c, jnp.float32)) + beta_h
        m = jnp.einsum("iz,jz->ij", psi.mean(0), phi_next_h)                    # (H, H)
        return violation_stats(np.asarray(m))

    report = {"restore_path": run_dir, "restore_epoch": int(cfg.restore_epoch), "tag": tag,
              "env_base": env_base, "tasks": tasks, "seed": seed, "n_rows": N, "n_holdout": H,
              "n_family": K_fam, "family_mean_pairwise_cos": fam_pair_cos, "n_panel": K,
              "steps": steps, "rows_per_step": rows_per_step, "cols_per_step": cols_per_step,
              "variants": [dict(zip(("name", "c_mode", "lam_mode", "lr", "lam_lr"), v)) for v in variants],
              "w_dim": w_dim, "z_dim": z_dim, "kappa": kappa, "u_clip": u_clip, "index_clip": ic,
              "na_path": str(na_path) if na_path else None, "na_task": na_task,
              "agent_config_source": prov, "per_task": {}}
    sidecar = {"W_fam": W_fam, "rows": rows, "rows_h": rows_h}

    for t in tasks:
        env_name = env_of(t)
        print(f"\n=== task {t}: {env_name}", flush=True)
        # w exactly as tools/eval_checkpoint.py: seed the global stream, one ex draw, then
        # the relabel batch.
        np.random.seed(seed)
        _, _, td, _ = make_env_and_datasets(env_name, frame_stack=cfg.frame_stack)
        ds_t = Dataset.create(**td)
        ds_t.sample(1)
        zb = ds_t.sample(min(ds_t.size, int(cfg.get("eval_relabel_size", 10000))))
        w = np.asarray(agent.infer_z(zb["next_observations"], zb["rewards"] + float(cfg.get("eval_reward_shift", 1.0))),
                       np.float64)
        w_j = jnp.asarray(w, jnp.float32)
        task_rec = {"env_name": env_name, "w_norm": float(np.linalg.norm(w))}
        if reward_raw_path and t == na_task:
            with open(str(reward_raw_path) + ".meta.json") as fh:
                w_file = np.asarray(json.load(fh)["w_inference"]["w"], np.float64)
            task_rec["cos_w_vs_reward_file"] = float(w_file @ w / (np.linalg.norm(w_file) * np.linalg.norm(w) + 1e-12))

        # init from the family, and the scale
        a_bar = np.asarray(jnp.einsum("pizw,z->w", A, w_j) / (P * N), np.float64)
        b_bar = float(jnp.einsum("piz,z->", beta, w_j) / (P * N))
        k0 = family_init(W_fam, a_bar)
        c0 = W_fam[k0]
        q0_mean, q0_pess = readouts(A, beta, w, c0, kappa)
        scale = float(q0_mean.std()) + 1e-12
        fam_values = W_fam @ a_bar + b_bar
        task_rec.update({"init": {"family_index": k0, "objective_raw": objective_linear(c0, a_bar, b_bar),
                                  "family_value_quantiles": [float(q) for q in np.quantile(fam_values, [0, .5, .9, 1])],
                                  "scale_std_objective_term": scale}})
        print(f"  init: family member {k0}, J {task_rec['init']['objective_raw']:+.4f}, scale {scale:.4f}", flush=True)

        # held-out reference values at c0 and GPI
        viol0 = holdout_violation(c0)
        q0h_mean, q0h_pess = readouts(A_h, beta_h, w, c0, kappa)

        task_rec["variants"] = {}
        c_stars = {}
        for name, c_mode, lam_mode, lr, lam_lr in variants:
            print(f"  -- variant {name}: c_mode={c_mode} lam_mode={lam_mode} lr={lr} lam_lr={lam_lr}", flush=True)
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(STEP_KEY), t), hash(name) % (2 ** 31))
            c_star, lam_star, curve = lagrangian_fit(
                A, beta, phi_next, w_j, c0, scale, steps=steps, lr=lr, lam_lr=lam_lr, c_mode=c_mode,
                lam_mode=lam_mode, rows_per_step=rows_per_step, cols_per_step=cols_per_step, key=key)
            diverged = not np.all(np.isfinite(c_star))
            q_mean, q_pess = readouts(A, beta, w, c_star, kappa) if not diverged else (q0_mean * np.nan,) * 2
            qh_mean, qh_pess = readouts(A_h, beta_h, w, c_star, kappa) if not diverged else (q0h_mean * np.nan,) * 2
            rec = {"c_mode": c_mode, "lam_mode": lam_mode, "lr": lr, "lam_lr": lam_lr, "diverged": diverged,
                   "curve": curve,
                   "objective": {"init_train_mean": float(q0_mean.mean()), "end_train_mean": float(q_mean.mean()),
                                 "init_train_pess": float(q0_pess.mean()), "end_train_pess": float(q_pess.mean()),
                                 "init_holdout_mean": float(q0h_mean.mean()), "end_holdout_mean": float(qh_mean.mean()),
                                 "init_holdout_pess": float(q0h_pess.mean()), "end_holdout_pess": float(qh_pess.mean()),
                                 "scale": scale},
                   "violation_holdout": {"init": viol0, "end": holdout_violation(c_star) if not diverged else None},
                   "geometry": {**family_geometry(c_star, W_fam),
                                "cos_c0": float(c0 @ c_star / (np.linalg.norm(c0) * np.linalg.norm(c_star) + 1e-12)),
                                "lambda_mean": float(lam_star.mean()), "lambda_max": float(lam_star.max()),
                                "lambda_frac_active": float((lam_star > 0).mean())} if not diverged else None}
            task_rec["variants"][name] = rec
            c_stars[name] = c_star
            sidecar[f"c_task{t}_{name}"] = c_star
            sidecar[f"lambda_task{t}_{name}"] = lam_star
            if not diverged:
                npz = os.path.join(out_dir, f"eq10_{name}_{tag}_task{t}.npz")
                np.savez(npz, c=c_star.astype(np.float32), c0=c0.astype(np.float32), w_task=w.astype(np.float32),
                         env_name=env_name, restore_path=run_dir, restore_epoch=int(cfg.restore_epoch),
                         variant=name, c_mode=c_mode, lam_mode=lam_mode, lr=lr, lam_lr=lam_lr, steps=steps,
                         objective_init=task_rec["init"]["objective_raw"],
                         objective_end=objective_linear(c_star, a_bar, b_bar))
                rec["npz"] = npz
            g = rec["geometry"] or {}
            print(f"     end: J train {rec['objective']['end_train_mean']:+.4f} (init {rec['objective']['init_train_mean']:+.4f})"
                  f"  holdout viol frac {rec['violation_holdout']['end']['frac_violated'] if rec['violation_holdout']['end'] else float('nan'):.3f}"
                  f" (init {viol0['frac_violated']:.3f})  |c*| {g.get('norm', float('nan')):.3f}"
                  f"  cos nearest fam {g.get('cos_nearest_family', float('nan')):.3f}  cos c0 {g.get('cos_c0', float('nan')):.3f}",
                  flush=True)

        # ---- Q_c vs Q_GPI on held-out states x K latents ----
        names = [n for n in c_stars if np.all(np.isfinite(c_stars[n]))]
        c_list = [jnp.asarray(c0, jnp.float32)] + [jnp.asarray(c_stars[n], jnp.float32) for n in names]
        vals = []
        for lo in range(0, H, chunk):
            vals.append(np.asarray(panel_values(jnp.asarray(obs_h[lo:lo + chunk]), u_panel[lo:lo + chunk],
                                                u_index[lo:lo + chunk], w_j, c_list)))       # (K, 1+n, B)
        vals = np.concatenate(vals, axis=2)                                                   # (K, 1+n, H)
        Q_gpi = vals[:, 0].T                                                                  # (H, K)
        Qs = None
        if na_agent is not None and t == na_task:
            Qs = np.concatenate([np.asarray(scalar_panel(jnp.asarray(obs_h[lo:lo + chunk]), u_panel[lo:lo + chunk]))
                                 for lo in range(0, H, chunk)], axis=1).T                     # (H, K)
        sidecar[f"Q_gpi_task{t}"] = Q_gpi
        if Qs is not None:
            sidecar[f"Q_scalar_task{t}"] = Qs

        def compare(Qc, ref):
            r = per_state_ranking(Qc, ref)
            out = {"pooled_raw": agreement(Qc, ref),
                   "pooled_state_centred": agreement(center_rows(Qc), center_rows(ref)),
                   "per_state_rho": summarize_rho(r["rho"]),
                   "argmax_agreement": float((Qc.argmax(1) == ref.argmax(1)).mean()),
                   "argmax_in_ref_top8": float(r["hit_topk"].mean()),
                   "ref_regret_norm_at_Qc_argmax_mean": float(r["regret_norm"].mean()),
                   "ref_regret_norm_random_mean": float(r["regret_random_norm"].mean()),
                   "within_state_std_Qc": float(Qc.std(1).mean()), "within_state_std_ref": float(ref.std(1).mean())}
            return out

        # Second axis of `vals`: 0 = Q_GPI, 1 = Q at c0, 2.. = the fitted variants in `names` order.
        panel = {"n_states": H, "n_latents": K}
        Q_c0 = vals[:, 1].T
        panel["c0_vs_gpi"] = compare(Q_c0, Q_gpi)
        for j, n in enumerate(names):
            Qc = vals[:, 2 + j].T
            panel[f"{n}_vs_gpi"] = compare(Qc, Q_gpi)
            panel[f"{n}_vs_c0"] = compare(Qc, Q_c0)
            sidecar[f"Q_{n}_task{t}"] = Qc
        if Qs is not None:
            panel["gpi_vs_scalar"] = compare(Q_gpi, Qs)
            panel["c0_vs_scalar"] = compare(Q_c0, Qs)
            for j, n in enumerate(names):
                panel[f"{n}_vs_scalar"] = compare(vals[:, 2 + j].T, Qs)
        task_rec["panel"] = panel
        for n in names:
            p = panel[f"{n}_vs_gpi"]
            print(f"  panel {n}: vs GPI pooled Spearman {p['pooled_raw']['spearman']:+.3f}, per-state rho "
                  f"{p['per_state_rho']['mean']:+.3f}, argmax agree {p['argmax_agreement']:.3f}"
                  + (f"; vs scalar per-state rho {panel[f'{n}_vs_scalar']['per_state_rho']['mean']:+.3f} regret "
                     f"{panel[f'{n}_vs_scalar']['ref_regret_norm_at_Qc_argmax_mean']:.3f}" if Qs is not None else ""),
                  flush=True)
        report["per_task"][str(t)] = task_rec

    out = write_report(report, cfg, "infer_policy_lagrangian.json")
    npz = os.path.splitext(out)[0] + ".npz"
    np.savez_compressed(npz, **sidecar)
    print(f"arrays -> {npz}")


def main():
    import hydra
    hydra.main(version_base=None, config_path="../configs", config_name="config")(_main)()


if __name__ == "__main__":
    main()
