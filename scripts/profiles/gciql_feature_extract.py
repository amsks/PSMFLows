"""Extract GCIQL/GCIVL value-network PENULTIMATE features phi(s,g) for the
representation readout probe (the positive control for the FB factorization
probe). Run in .venv-jax-cpu (JAX). Dumps one npz per task with
{phi, cube, grip, lift, table_z, goal_xyz}; the torch consumer
scripts/profiles/gciql_feature_readout.py computes region/d and the R^2 readout ceiling.

phi = the last hidden activation of GCValue.value_net (MLP [512,512,512,1]),
i.e. what the final value Dense reads from — the analog of FB's backward map B.

Example:
  .venv-jax-cpu/bin/python -m scripts.profiles.gciql_feature_extract \
    --run-dir results/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd001_20260518_201036 \
    --step 1000000 --data-path datasets --out analysis/features_raw/gciql_feature/state_sd001 \
    --n-states 6000 --tasks 1,2,3,4,5
Use --inspect once to print the intermediates tree and confirm the penultimate.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stack_episode(raw, k):
    """Frame-stack one episode's obs along the channel axis, oldest-first with
    edge-padding at the start — matches FrameStackWrapper (concat axis=-1; reset
    fills the deque with the first frame)."""
    T = len(raw)
    out = np.empty((T,) + raw.shape[1:-1] + (raw.shape[-1] * k,), raw.dtype)
    for t in range(T):
        idxs = [max(0, t - k + 1 + j) for j in range(k)]
        out[t] = np.concatenate([raw[i] for i in idxs], axis=-1)
    return out


def _sample_states(data_path, domain, n, seed, obs_key, max_files=0, frame_stack=1):
    files = sorted(glob.glob(str(Path(data_path) / domain / "buffer" / "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episodes under {data_path}/{domain}/buffer")
    if max_files:
        files = files[:max_files]  # cap memory (pixels are ~12MB/episode)
    rng = np.random.default_rng(seed)
    obs_l, phys_l = [], []
    for f in files:
        z = np.load(f)
        raw = np.asarray(z[obs_key])
        if frame_stack and frame_stack > 1:
            raw = _stack_episode(raw, frame_stack)
        obs_l.append(raw)
        phys_l.append(np.asarray(z["physics"], np.float32))
    obs = np.concatenate(obs_l); phys = np.concatenate(phys_l)
    if len(obs) > n:
        idx = rng.choice(len(obs), n, replace=False)
        obs, phys = obs[idx], phys[idx]
    return obs, phys


def _collect_arrays(d, path, out):
    if isinstance(d, dict):
        for k, v in d.items():
            _collect_arrays(v, path + [str(k)], out)
    elif isinstance(d, (tuple, list)):
        for v in d:
            _collect_arrays(v, path, out)
    else:
        out.append(("/".join(path), np.asarray(d)))


def _find_penultimate(intermediates, width):
    arrs = []
    _collect_arrays(intermediates, [], arrs)
    cands = [(p, a) for p, a in arrs
             if a.ndim >= 2 and a.shape[-1] == width and "value" in p.lower()]
    if not cands:
        return None, None

    def dense_idx(p):
        m = re.findall(r"Dense_(\d+)", p)
        return int(m[-1]) if m else -1
    cands.sort(key=lambda pa: dense_idx(pa[0]))
    return cands[-1]


def _find_encoder_output(intermediates):
    arrs = []
    _collect_arrays(intermediates, [], arrs)
    for p, a in arrs:
        if p.endswith("gc_encoder/__call__") and np.asarray(a).ndim == 2:
            return np.asarray(a)
    return None


def _layernorm(x, scale, bias, eps=1e-6):
    import jax.numpy as jnp
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def _reconstruct_ensemble_phi(e, vp, penult_idx, full=False):
    """Manually run the ensembled GCValue.value_net MLP (Dense->gelu->LayerNorm)
    from the encoder output e, since flax vmap hides its intermediates. Returns
    the penultimate Dense_{penult_idx} pre-activation (matching GCIQL's captured
    Dense_2), meaned over the ensemble. full=True continues to the scalar value
    (for a sanity check vs the real output)."""
    import jax
    import jax.numpy as jnp
    e = jnp.asarray(e)
    E = np.asarray(vp["Dense_0"]["kernel"]).shape[0]
    outs = []
    for k in range(E):
        h = e
        for i in range(penult_idx):
            h = h @ vp[f"Dense_{i}"]["kernel"][k] + vp[f"Dense_{i}"]["bias"][k]
            h = jax.nn.gelu(h)
            h = _layernorm(h, vp[f"LayerNorm_{i}"]["scale"][k], vp[f"LayerNorm_{i}"]["bias"][k])
        phi_k = h @ vp[f"Dense_{penult_idx}"]["kernel"][k] + vp[f"Dense_{penult_idx}"]["bias"][k]
        if full:
            g = jax.nn.gelu(phi_k)
            g = _layernorm(g, vp[f"LayerNorm_{penult_idx}"]["scale"][k],
                           vp[f"LayerNorm_{penult_idx}"]["bias"][k])
            phi_k = g @ vp[f"Dense_{penult_idx + 1}"]["kernel"][k] + vp[f"Dense_{penult_idx + 1}"]["bias"][k]
        outs.append(np.asarray(phi_k))
    return np.stack(outs, 0).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--load-episodes", type=int, default=0, help="cap episodes loaded (0=all); use ~50 for pixels (memory)")
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--inspect", action="store_true", help="print intermediates tree and exit")
    ap.add_argument("--inspect-params", action="store_true", help="print value param tree and exit")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    if ogb in sys.path:
        sys.path.remove(ogb)
    sys.path.insert(0, ogb)
    import ogbench  # noqa: F401
    import jax.numpy as jnp
    from utils.flax_utils import restore_agent

    flags = json.loads((Path(args.run_dir) / "flags.json").read_text())
    saved = flags.get("agent")
    name = (saved.get("agent_name") if isinstance(saved, dict) else None) or "gciql"
    if name == "gcivl":
        from agents.gcivl import GCIVLAgent as Cls, get_config
    else:
        from agents.gciql import GCIQLAgent as Cls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v
    width = int(config["value_hidden_dims"][-1])
    fs = config.get("frame_stack")
    env_name = flags["env_name"]
    is_pixel = env_name.startswith("visual")
    obs_key = "pixels" if is_pixel else "observation"
    domain = env_name.replace("visual-", "")

    obs, phys = _sample_states(args.data_path, domain, args.n_states,
                               int(flags.get("seed", 0)), obs_key, args.load_episodes,
                               int(fs or 1))
    cube = phys[:, 14:17].astype(np.float64)
    grip = phys[:, 6].astype(np.float64)
    lift = phys[:, 16].astype(np.float64)

    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    for t in task_ids:
        env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if fs:
            from utils.env_utils import FrameStackWrapper
            env = FrameStackWrapper(env, fs)
        ex_obs, info = env.reset(options=dict(task_id=t))
        ex_act = env.action_space.sample()
        agent = Cls.create(int(flags["seed"]),
                           np.asarray(ex_obs, np.float32)[None],
                           np.asarray(ex_act, np.float32)[None], config)
        agent = restore_agent(agent, str(args.run_dir), args.step)
        if args.inspect_params:
            import jax
            for kp, v in jax.tree_util.tree_flatten_with_path(agent.network.params)[0]:
                path = "/".join(getattr(k, "key", str(k)) for k in kp)
                if "value_net" in path and "target" not in path.lower():
                    print(path, np.asarray(v).shape)
            return
        goal = np.asarray(info["goal"])
        u = env.unwrapped
        tb = int(getattr(u, "_target_block", 0) or 0)
        gxyz = np.asarray(u.cur_task_info["goal_xyzs"][tb], np.float64)
        table_z = float(u.cur_task_info["init_xyzs"][tb][2])
        env.close()

        N = len(obs); bs = 256 if is_pixel else 4096
        penult_idx = len(config["value_hidden_dims"]) - 1
        vp = (agent.network.params["modules_value"]["value_net"]
              if name == "gcivl" else None)
        phis = []
        used_path = None
        for i in range(0, N, bs):
            ob = jnp.asarray(obs[i:i + bs], jnp.float32)
            g = jnp.broadcast_to(jnp.asarray(goal, jnp.float32)[None],
                                 (ob.shape[0],) + goal.shape)
            vout, state = agent.network.select("value")(
                ob, g, capture_intermediates=True, mutable=["intermediates"])
            inter = state["intermediates"]
            if args.inspect and i == 0:
                arrs = []
                _collect_arrays(inter, [], arrs)
                for p, a in arrs:
                    print(f"  {p}  {a.shape}")
                return
            if name == "gcivl":
                # vmap hides value_net intermediates -> reconstruct from params.
                e = _find_encoder_output(inter)
                if e is None:
                    raise SystemExit("gcivl: encoder output not captured")
                a = _reconstruct_ensemble_phi(e, vp, penult_idx)
                if i == 0:
                    fv = _reconstruct_ensemble_phi(e, vp, penult_idx, full=True)
                    av = np.asarray(vout); av = av.mean(0) if av.ndim == 2 else av
                    d = float(np.max(np.abs(fv.reshape(-1) - av.reshape(-1))))
                    used_path = f"manual recon value_net/Dense_{penult_idx} (sanity |dv|={d:.3g})"
            else:
                p, a = _find_penultimate(inter, width)
                if a is None:
                    raise SystemExit("penultimate not found; rerun with --inspect")
                a = np.asarray(a)
                if a.ndim == 3:
                    a = a.mean(0)
                used_path = p
            phis.append(a)
        phi = np.concatenate(phis, 0)
        np.savez(out / f"task{t}.npz", phi=phi, cube=cube, grip=grip, lift=lift,
                 table_z=table_z, goal_xyz=gxyz, agent=name, is_pixel=is_pixel)
        print(f"[extract:{name}] task{t}: phi {phi.shape} via {used_path}")


if __name__ == "__main__":
    main()
