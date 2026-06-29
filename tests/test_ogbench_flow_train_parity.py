import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import jax_python, SHIM_DIR, OGBENCH_IMPLS

B, OBS, Z, ACT, H, N, LR = 4, 11, 7, 5, 512, 100, 3e-4


def test_flow_training_dynamics_parity_float64():
    import gymnasium
    import torch
    from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    vf = SimpleVectorFieldArchiConfig(hidden_dim=H, hidden_layers=4).build(obs_space, ACT).double()
    ac = NoiseConditionedActorArchiConfig(hidden_dim=H, hidden_layers=2, embedding_layers=2).build(
        obs_space, Z, ACT
    ).double()

    # Pre-generated, framework-shared data for N steps.
    obs = rng.standard_normal((N, B, OBS))
    action = rng.standard_normal((N, B, ACT))     # x_1 for vf
    x0 = rng.standard_normal((N, B, ACT))
    tt = rng.uniform(size=(N, B, 1))
    z = rng.standard_normal((N, B, Z))
    noise = rng.standard_normal((N, B, ACT))
    actor_target = np.tanh(rng.standard_normal((N, B, ACT)))  # fixed supervised targets in (-1,1)
    eval_obs = rng.standard_normal((B, OBS)); eval_a = rng.standard_normal((B, ACT))
    eval_t = rng.uniform(size=(B, 1)); eval_z = rng.standard_normal((B, Z)); eval_n = rng.standard_normal((B, ACT))

    def torch_init(mod):
        return {k: v.detach().numpy().copy() for k, v in mod.state_dict().items()}

    init_vf, init_ac = torch_init(vf), torch_init(ac)

    opt_vf = torch.optim.Adam(vf.parameters(), lr=LR)
    losses_vf = []
    for i in range(N):
        o = torch.tensor(obs[i]); x1 = torch.tensor(action[i])
        xt = (1 - torch.tensor(tt[i])) * torch.tensor(x0[i]) + torch.tensor(tt[i]) * x1
        vel = x1 - torch.tensor(x0[i])
        loss = ((vf(o, xt, torch.tensor(tt[i])) - vel) ** 2).mean()
        opt_vf.zero_grad(); loss.backward(); opt_vf.step()
        losses_vf.append(float(loss.detach()))

    opt_ac = torch.optim.Adam(ac.parameters(), lr=LR)
    losses_ac = []
    for i in range(N):
        o = torch.tensor(obs[i]); zz = torch.tensor(z[i]); nn_ = torch.tensor(noise[i]); tgt = torch.tensor(actor_target[i])
        loss = ((ac(o, zz, nn_) - tgt) ** 2).mean()
        opt_ac.zero_grad(); loss.backward(); opt_ac.step()
        losses_ac.append(float(loss.detach()))

    with torch.no_grad():
        fout_vf = vf(torch.tensor(eval_obs), torch.tensor(eval_a), torch.tensor(eval_t)).numpy()
        fout_ac = ac(torch.tensor(eval_obs), torch.tensor(eval_z), torch.tensor(eval_n)).numpy()
    torch.set_default_dtype(torch.float32)

    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "train_parity.npz"
        np.savez(
            npz, obs=obs, action=action, x0=x0, tt=tt, z=z, noise=noise, actor_target=actor_target,
            eval_obs=eval_obs, eval_a=eval_a, eval_t=eval_t, eval_z=eval_z, eval_n=eval_n,
            losses_vf=np.array(losses_vf), losses_ac=np.array(losses_ac),
            fout_vf=fout_vf, fout_ac=fout_ac,
            **{f"ivf__{k}": v for k, v in init_vf.items()},
            **{f"iac__{k}": v for k, v in init_ac.items()},
        )
        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, optax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from ogbench_flow import VectorField, NoiseConditionedActor
            d = np.load(sys.argv[1])
            def T(w): return jnp.asarray(w, dtype=jnp.float64).T
            def V(w): return jnp.asarray(w, dtype=jnp.float64)
            g = lambda p: d[p]
            vfp = {
                "l0": {"kernel": T(g("ivf__net.0.weight")), "bias": V(g("ivf__net.0.bias"))},
                "l1": {"kernel": T(g("ivf__net.2.weight")), "bias": V(g("ivf__net.2.bias"))},
                "l2": {"kernel": T(g("ivf__net.4.weight")), "bias": V(g("ivf__net.4.bias"))},
                "l3": {"kernel": T(g("ivf__net.6.weight")), "bias": V(g("ivf__net.6.bias"))},
                "out": {"kernel": T(g("ivf__net.8.weight")), "bias": V(g("ivf__net.8.bias"))},
            }
            acp = {
                "embed_z_l0": {"kernel": T(g("iac__embed_z.0.weight")), "bias": V(g("iac__embed_z.0.bias"))},
                "embed_z_ln": {"scale": V(g("iac__embed_z.1.weight")), "bias": V(g("iac__embed_z.1.bias"))},
                "embed_z_lout": {"kernel": T(g("iac__embed_z.3.weight")), "bias": V(g("iac__embed_z.3.bias"))},
                "embed_s_l0": {"kernel": T(g("iac__embed_s.0.weight")), "bias": V(g("iac__embed_s.0.bias"))},
                "embed_s_ln": {"scale": V(g("iac__embed_s.1.weight")), "bias": V(g("iac__embed_s.1.bias"))},
                "embed_s_lout": {"kernel": T(g("iac__embed_s.3.weight")), "bias": V(g("iac__embed_s.3.bias"))},
                "policy_l0": {"kernel": T(g("iac__policy.0.weight")), "bias": V(g("iac__policy.0.bias"))},
                "policy_l1": {"kernel": T(g("iac__policy.2.weight")), "bias": V(g("iac__policy.2.bias"))},
                "policy_out": {"kernel": T(g("iac__policy.4.weight")), "bias": V(g("iac__policy.4.bias"))},
            }
            vf = VectorField(hidden_dim=512, hidden_layers=4)
            ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
            N = d["losses_vf"].shape[0]
            obs, action, x0, tt = V(d["obs"]), V(d["action"]), V(d["x0"]), V(d["tt"])
            z, noise, tgt = V(d["z"]), V(d["noise"]), V(d["actor_target"])

            def run(params, loss_at):
                opt = optax.adam(3e-4); state = opt.init(params); curve = []
                for i in range(N):
                    l, grads = jax.value_and_grad(loss_at)(params, i)
                    upd, state = opt.update(grads, state, params)
                    params = optax.apply_updates(params, upd); curve.append(float(l))
                return params, np.array(curve)

            def vf_loss(p, i):
                xt = (1 - tt[i]) * x0[i] + tt[i] * action[i]; vel = action[i] - x0[i]
                return jnp.mean((vf.apply({"params": p}, obs[i], xt, tt[i]) - vel) ** 2)
            def ac_loss(p, i):
                return jnp.mean((ac.apply({"params": p}, obs[i], z[i], noise[i]) - tgt[i]) ** 2)

            vfp2, cv = run(vfp, vf_loss)
            acp2, ca = run(acp, ac_loss)
            fvf = np.asarray(vf.apply({"params": vfp2}, V(d["eval_obs"]), V(d["eval_a"]), V(d["eval_t"])))
            fac = np.asarray(ac.apply({"params": acp2}, V(d["eval_obs"]), V(d["eval_z"]), V(d["eval_n"])))
            diffs = {
                "curve_vf": float(np.max(np.abs(cv - d["losses_vf"]))),
                "curve_ac": float(np.max(np.abs(ca - d["losses_ac"]))),
                "fout_vf": float(np.max(np.abs(fvf - d["fout_vf"]))),
                "fout_ac": float(np.max(np.abs(fac - d["fout_ac"]))),
            }
            print("DIFFS", diffs)
            sys.exit(0 if max(diffs.values()) < 1e-6 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")])
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run([jax_python(), "-c", jax_code, str(npz)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"train parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DIFFS" in proc.stdout
