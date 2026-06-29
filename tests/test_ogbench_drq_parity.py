import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import OGBENCH_IMPLS, SHIM_DIR, jax_python


def test_drq_jax_matches_fb_torch_under_transplant():
    import gymnasium
    import torch
    from nn_models import DrQEncoderArchiConfig

    torch.manual_seed(0)
    obs_space = gymnasium.spaces.Box(0, 255, (9, 64, 64), np.uint8)  # CHW, obs-alone
    enc = DrQEncoderArchiConfig(feature_dim=256).build(obs_space).eval()

    x_raw = torch.rand(2, 9, 64, 64) * 255.0          # [0,255] float, NCHW
    with torch.no_grad():
        out_t = enc(x_raw / 255.0 - 0.5).numpy()      # FB applies RGBNorm outside
    sd = enc.state_dict()
    # trunk = Sequential(Conv,ReLU,Conv,ReLU,Conv,ReLU,Conv,ReLU,Flatten) -> convs at 0,2,4,6
    # proj  = Sequential(Linear, LayerNorm, Tanh) -> Linear=proj.0, LayerNorm=proj.1
    arrs = dict(
        conv0_w=sd["trunk.0.weight"].numpy(), conv0_b=sd["trunk.0.bias"].numpy(),
        conv1_w=sd["trunk.2.weight"].numpy(), conv1_b=sd["trunk.2.bias"].numpy(),
        conv2_w=sd["trunk.4.weight"].numpy(), conv2_b=sd["trunk.4.bias"].numpy(),
        conv3_w=sd["trunk.6.weight"].numpy(), conv3_b=sd["trunk.6.bias"].numpy(),
        proj_w=sd["proj.0.weight"].numpy(),   proj_b=sd["proj.0.bias"].numpy(),
        ln_w=sd["proj.1.weight"].numpy(),     ln_b=sd["proj.1.bias"].numpy(),
        x_raw=x_raw.numpy(), out_t=out_t,
    )
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "drq_parity.npz"
        np.savez(npz, **arrs)

        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, jax.numpy as jnp
            from ogbench_drq import DrQEncoder
            d = np.load(sys.argv[1])
            x = jnp.asarray(d["x_raw"]).transpose(0, 2, 3, 1)        # NCHW -> NHWC
            enc = DrQEncoder(feature_dim=256)
            enc.init(jax.random.PRNGKey(0), x)                       # shape check only
            def ck(w):  # torch [out,in,kh,kw] -> flax [kh,kw,in,out]
                return jnp.asarray(w).transpose(2, 3, 1, 0)
            # proj: torch flatten order (C,H,W); flax reshape order (H,W,C).
            # reorder torch Linear weight columns C,H,W -> H,W,C, then to [in,out].
            pw = d["proj_w"].reshape(256, 32, 25, 25)                # [out,C,H,W]
            pw = np.transpose(pw, (2, 3, 1, 0)).reshape(20000, 256)  # [H*W*C, out]
            params = {
                "conv0": {"kernel": ck(d["conv0_w"]), "bias": jnp.asarray(d["conv0_b"])},
                "conv1": {"kernel": ck(d["conv1_w"]), "bias": jnp.asarray(d["conv1_b"])},
                "conv2": {"kernel": ck(d["conv2_w"]), "bias": jnp.asarray(d["conv2_b"])},
                "conv3": {"kernel": ck(d["conv3_w"]), "bias": jnp.asarray(d["conv3_b"])},
                "proj": {"kernel": jnp.asarray(pw), "bias": jnp.asarray(d["proj_b"])},
                "proj_ln": {"scale": jnp.asarray(d["ln_w"]), "bias": jnp.asarray(d["ln_b"])},
            }
            out_j = np.asarray(enc.apply({"params": params}, x))
            out_t = d["out_t"]
            err = np.abs(out_j - out_t)
            max_abs = float(np.max(err))
            mean_abs = float(np.mean(err))
            # per-element relative error: a diagnostic only (tanh outputs near 0
            # make this large for individual elements), so it is printed, not gated.
            max_rel = float(np.max(err / (np.abs(out_t) + 1e-8)))
            cos_sim = float(
                np.dot(out_j.ravel(), out_t.ravel())
                / (np.linalg.norm(out_j.ravel()) * np.linalg.norm(out_t.ravel()) + 1e-8)
            )
            print(
                f"METRICS max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
                f"max_rel={max_rel:.6e} cos_sim={cos_sim:.10f}"
            )
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")]
        )
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [jax_python(), "-c", jax_code, str(npz)],
            env=env, capture_output=True, text=True,
        )
    assert proc.returncode == 0, (
        f"jax subprocess failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("METRICS")
    )
    print(line)  # visible under `pytest -s` and on failure
    m = dict(kv.split("=") for kv in line.split()[1:])
    max_abs = float(m["max_abs"])
    mean_abs = float(m["mean_abs"])
    cos_sim = float(m["cos_sim"])
    # Not bit-exact across PyTorch (NCHW/oneDNN) and JAX/XLA (NHWC): float32 conv
    # accumulation order differs. Gate the whole representation, not one element:
    assert max_abs < 1e-5, f"max_abs={max_abs:.3e} (>=1e-5): layout/transpose/padding bug?"
    assert mean_abs < 1e-6, f"mean_abs={mean_abs:.3e} (>=1e-6): broad misalignment"
    assert cos_sim > 0.999999, f"cos_sim={cos_sim:.10f} (<=0.999999): representations not aligned"
