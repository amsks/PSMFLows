"""Read-only numeric check (2026-11-09 port audit): does the direction the actor's Q pushes u depend on the
task index? Per row i, g_i(w) = dQ_i/du_i at the actor's u, under the row's own index and under another row's
index (a permutation of the rows' indices, reward view recomputed for that index). Reports the mean cosine
between the two per-row gradients (1 = the push ignores the task) and the same for two unrelated sphere draws.
Run: JAX_PLATFORMS=cpu .venv/bin/python psmgoal-vs-dfb-2026-11-09/task_dependence_check.py <out.json>"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actor_grad_check as A  # noqa: E402  (imports utils.xla_guard first)
import jax, jax.numpy as jnp, numpy as np  # noqa: E402


def cos(a, b):
    return float(jnp.mean(jnp.sum(a * b, -1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-12)))


def per_row_grad(qfun, u):
    return jax.grad(lambda uu: qfun(uu).sum())(u)


def run_L(agent, batch):
    c = agent.config
    inj = agent._draw_injection(batch, jax.random.PRNGKey(1))
    obs, x_in, basis = jnp.asarray(batch["observations"]), inj["x_in"], inj["zenc_u"]
    z = inj["z_a"]
    u = agent.actor_u(obs, z, inj["actor_noise"])
    perm = jax.random.permutation(jax.random.PRNGKey(5), z.shape[0])
    d = int(c["z_dim"])
    s1 = A.__dict__.get("sphere", None)
    r1 = jax.random.normal(jax.random.PRNGKey(6), z.shape); r1 = r1 * jnp.sqrt(d) / jnp.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = jax.random.normal(jax.random.PRNGKey(7), z.shape); r2 = r2 * jnp.sqrt(d) / jnp.linalg.norm(r2, axis=-1, keepdims=True)
    q = lambda zz: (lambda uu: agent.q_fn(agent.params, obs, zz, uu, x_in, basis)[0])
    g0, gp = per_row_grad(q(z), u), per_row_grad(q(z[perm]), u)
    ga, gb = per_row_grad(q(r1), u), per_row_grad(q(r2), u)
    return {"cos_own_vs_permuted_index": cos(g0, gp), "cos_two_sphere_indices": cos(ga, gb)}


def run_P(agent, batch):
    c = agent.config
    agent = agent.replace(rng=jax.random.PRNGKey(1))
    batch = {**batch, **agent.reward_index_rows(batch)}
    obs, pool = jnp.asarray(batch["observations"]), jnp.asarray(batch["next_observations"])
    w = jnp.asarray(batch["index_w_actor"])
    f_c = jnp.asarray(batch["index_f"]) - agent.fb_mu[None]
    noise = jax.random.normal(jax.random.PRNGKey(3), (obs.shape[0], int(c["action_dim"])))
    u = agent.fbc_actor_u(obs, w, noise)
    perm = jax.random.permutation(jax.random.PRNGKey(5), w.shape[0])
    d = int(c["z_dim"])
    r1 = jax.random.normal(jax.random.PRNGKey(6), w.shape); r1 = r1 * jnp.sqrt(d) / jnp.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = jax.random.normal(jax.random.PRNGKey(7), w.shape); r2 = r2 * jnp.sqrt(d) / jnp.linalg.norm(r2, axis=-1, keepdims=True)
    bp = agent.basis.params
    q = lambda ww: (lambda uu: agent.reward_softmax_q(obs, uu, pool, ww, f_c, bp)[0])
    g0, gp = per_row_grad(q(w), u), per_row_grad(q(w[perm]), u)
    ga, gb = per_row_grad(q(r1), u), per_row_grad(q(r2), u)
    return {"cos_own_vs_permuted_index": cos(g0, gp), "cos_two_sphere_indices": cos(ga, gb)}


def main():
    _, _, train, _ = A.make_env_and_datasets("cube-single-play-singletask-v0")
    base = dict(train)
    aug, _ = A.repair_invalid_preimages(A.load_augmented_dataset(A.PRE))
    for k in aug:
        if k.startswith("noise_preimage"):
            base[k] = aug[k]
    res = {}
    for name, d, fn in [("L_bc0p3", sorted(glob.glob(f"{A.EXP}/dfb_latent_oc_bc0p3_cube/*"))[0], run_L),
                        ("P_bc03_sd000", sorted(glob.glob(f"{A.EXP}/psmgoal_recipe_bc03_cube/*"))[0], run_P)]:
        agent, ds, cfg = A.load(d, 500000, base)
        np.random.seed(0)
        batch = {k: jnp.asarray(v) for k, v in ds.sample(512).items()}
        res[name] = fn(agent, batch)
        res[name]["run"] = d
        print(name, res[name], flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
