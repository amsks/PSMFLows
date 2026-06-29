import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import jax_python, SHIM_DIR, OGBENCH_IMPLS

B, OBS, Z, ACT, H = 4, 11, 7, 5, 512
FLOW_STEPS = 10


def _np(x):
    return x.detach().numpy()


def test_flow_nets_forward_parity_under_transplant():
    import gymnasium
    import torch
    from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

    torch.manual_seed(0)
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    vf = SimpleVectorFieldArchiConfig(hidden_dim=H, hidden_layers=4).build(obs_space, ACT).eval()
    ac = NoiseConditionedActorArchiConfig(hidden_dim=H, hidden_layers=2, embedding_layers=2).build(
        obs_space, Z, ACT
    ).eval()

    obs = torch.randn(B, OBS)
    action = torch.randn(B, ACT)
    t = torch.rand(B, 1)
    z = torch.randn(B, Z)
    noise = torch.randn(B, ACT)
    with torch.no_grad():
        out_vf = _np(vf(obs, action, t))
        out_ac = _np(ac(obs, z, noise))
        # inline torch rollout == FB compute_flow_actions
        roll = noise.clone()
        for i in range(FLOW_STEPS):
            tt = torch.ones(B, 1) * i / FLOW_STEPS
            roll = roll + vf(obs, roll, tt) / FLOW_STEPS
        out_roll = _np(torch.clamp(roll, -1, 1))

    vfsd = vf.net  # Sequential; Linears at 0,2,4,6,8
    acsd = ac
    arrs = dict(
        obs=_np(obs), action=_np(action), t=_np(t), z=_np(z), noise=_np(noise),
        out_vf=out_vf, out_ac=out_ac, out_roll=out_roll,
        # VectorField linears
        vf_l0_w=_np(vfsd[0].weight), vf_l0_b=_np(vfsd[0].bias),
        vf_l1_w=_np(vfsd[2].weight), vf_l1_b=_np(vfsd[2].bias),
        vf_l2_w=_np(vfsd[4].weight), vf_l2_b=_np(vfsd[4].bias),
        vf_l3_w=_np(vfsd[6].weight), vf_l3_b=_np(vfsd[6].bias),
        vf_out_w=_np(vfsd[8].weight), vf_out_b=_np(vfsd[8].bias),
        # embed_z (Linear@0, LN@1, Linear@3)
        ez_l0_w=_np(acsd.embed_z[0].weight), ez_l0_b=_np(acsd.embed_z[0].bias),
        ez_ln_w=_np(acsd.embed_z[1].weight), ez_ln_b=_np(acsd.embed_z[1].bias),
        ez_lout_w=_np(acsd.embed_z[3].weight), ez_lout_b=_np(acsd.embed_z[3].bias),
        # embed_s
        es_l0_w=_np(acsd.embed_s[0].weight), es_l0_b=_np(acsd.embed_s[0].bias),
        es_ln_w=_np(acsd.embed_s[1].weight), es_ln_b=_np(acsd.embed_s[1].bias),
        es_lout_w=_np(acsd.embed_s[3].weight), es_lout_b=_np(acsd.embed_s[3].bias),
        # policy (Linear@0,2,4)
        p_l0_w=_np(acsd.policy[0].weight), p_l0_b=_np(acsd.policy[0].bias),
        p_l1_w=_np(acsd.policy[2].weight), p_l1_b=_np(acsd.policy[2].bias),
        p_out_w=_np(acsd.policy[4].weight), p_out_b=_np(acsd.policy[4].bias),
    )

    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "flow_parity.npz"
        np.savez(npz, **arrs)
        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, jax.numpy as jnp
            from ogbench_flow import VectorField, NoiseConditionedActor, compute_flow_actions
            d = np.load(sys.argv[1])
            def T(w):  # torch Linear [out,in] -> flax Dense [in,out]
                return jnp.asarray(w).T
            obs, action, t = jnp.asarray(d["obs"]), jnp.asarray(d["action"]), jnp.asarray(d["t"])
            z, noise = jnp.asarray(d["z"]), jnp.asarray(d["noise"])
            vf = VectorField(hidden_dim=512, hidden_layers=4)
            vfp = {
                "l0": {"kernel": T(d["vf_l0_w"]), "bias": jnp.asarray(d["vf_l0_b"])},
                "l1": {"kernel": T(d["vf_l1_w"]), "bias": jnp.asarray(d["vf_l1_b"])},
                "l2": {"kernel": T(d["vf_l2_w"]), "bias": jnp.asarray(d["vf_l2_b"])},
                "l3": {"kernel": T(d["vf_l3_w"]), "bias": jnp.asarray(d["vf_l3_b"])},
                "out": {"kernel": T(d["vf_out_w"]), "bias": jnp.asarray(d["vf_out_b"])},
            }
            ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
            acp = {
                "embed_z_l0": {"kernel": T(d["ez_l0_w"]), "bias": jnp.asarray(d["ez_l0_b"])},
                "embed_z_ln": {"scale": jnp.asarray(d["ez_ln_w"]), "bias": jnp.asarray(d["ez_ln_b"])},
                "embed_z_lout": {"kernel": T(d["ez_lout_w"]), "bias": jnp.asarray(d["ez_lout_b"])},
                "embed_s_l0": {"kernel": T(d["es_l0_w"]), "bias": jnp.asarray(d["es_l0_b"])},
                "embed_s_ln": {"scale": jnp.asarray(d["es_ln_w"]), "bias": jnp.asarray(d["es_ln_b"])},
                "embed_s_lout": {"kernel": T(d["es_lout_w"]), "bias": jnp.asarray(d["es_lout_b"])},
                "policy_l0": {"kernel": T(d["p_l0_w"]), "bias": jnp.asarray(d["p_l0_b"])},
                "policy_l1": {"kernel": T(d["p_l1_w"]), "bias": jnp.asarray(d["p_l1_b"])},
                "policy_out": {"kernel": T(d["p_out_w"]), "bias": jnp.asarray(d["p_out_b"])},
            }
            o_vf = np.asarray(vf.apply({"params": vfp}, obs, action, t))
            o_ac = np.asarray(ac.apply({"params": acp}, obs, z, noise))
            o_roll = np.asarray(compute_flow_actions(vf, vfp, obs, noise, 10))
            diffs = {
                "vf": float(np.max(np.abs(o_vf - d["out_vf"]))),
                "ac": float(np.max(np.abs(o_ac - d["out_ac"]))),
                "roll": float(np.max(np.abs(o_roll - d["out_roll"]))),
            }
            print("DIFFS", diffs)
            sys.exit(0 if max(diffs.values()) < 1e-5 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")])
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run([jax_python(), "-c", jax_code, str(npz)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DIFFS" in proc.stdout
