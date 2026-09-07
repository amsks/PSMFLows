"""Does the behaviour flow reproduce the dataset's own actions inside a local ball?

For each anchor state, the ball is every dataset row within `eps` of it in per-dimension
standardized observation space (utils/geometry's protocol, radius instead of k). The flow
is decoded K times at the anchor, and the ball's score is the LOWEST MSE between any decode
and any ball action: how close the flow can get to an action the data actually took nearby.

The number is meaningless alone, so two references are computed on the same balls:
  data    the anchor's own recorded action against the rest of its ball (self-excluded).
          This is what "close" means for real data at this radius.
  prior   K draws from the action box, same statistic. This is what the metric scores when
          the decode carries no state information at all.

Also reported per ball: the flow's median MSE over its K decodes (min over ball actions per
decode, then median), which separates "one lucky sample landed on a data action" from "the
whole decode sits in the ball".

Run:
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \\
  JAX_PLATFORMS='' .venv/bin/python tools/diag_eps_ball_fit.py \\
      agent=psmflow env_name=cube-single-play-singletask-v0 \\
      agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \\
      report_out=/data-local/amsks/PSMFLows/logs/eps_ball_cube.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.geometry import INDEX_ROWS, NeighbourIndex
from utils.log_utils import write_report

N_ANCHORS = 256          # anchor states drawn from the dataset
N_LATENTS = 512          # flow decodes per anchor
EPS = [0.25, 0.5, 1.0, 2.0]   # ball radii, in standardized-observation units
MIN_BALL = 4             # balls smaller than this are dropped, not scored
LATENT_CLIP = 3.0        # typical-set clamp on the prior draw, as the deployed decode uses
SEED = 0


def flow_trained_steps(flow_ckpt_path):
    """The step count the flow was TRAINED at. The ODE decode must use exactly this."""
    p = os.path.join(str(flow_ckpt_path), "flags.json")
    assert os.path.exists(p), f"no flags.json beside the flow checkpoint: {p}"
    with open(p) as f:
        flags = json.load(f)
    steps = flags["agent"].get("flow_steps")
    assert steps, f"{p} does not record agent.flow_steps"
    return int(steps)


def mse_matrix(a, b):
    """(n, m) per-dimension mean squared error between action sets a (n, d) and b (m, d)."""
    return ((np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :]) ** 2).mean(-1)


def summarize(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "n": int(v.size)}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents["psmflow"].create(cfg.seed, ex["observations"], ex["actions"], config)
    ode_steps = flow_trained_steps(config["flow_ckpt_path"])
    print(f"flow trained at flow_steps={ode_steps}; ODE decode pinned to that")

    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])
    d_a = act_all.shape[-1]

    # Ball geometry: the shared standardized-observation protocol, queried by RADIUS rather
    # than by k. NeighbourIndex owns the standardization and the row bookkeeping, so the
    # balls here select neighbours on the same scale every other support probe uses.
    index = NeighbourIndex(obs_all, act_all, index_rows=INDEX_ROWS, seed=SEED)
    tree = cKDTree(index.standardize(index.obs))

    rng_np = np.random.default_rng(SEED)
    anchor_pos = rng_np.choice(len(index.obs), N_ANCHORS, replace=False)  # index-local rows
    anchors = index.obs[anchor_pos]
    anchor_act = index.act[anchor_pos]
    anchors_std = index.standardize(anchors)

    # Decode once per anchor and reuse across radii: the decode does not depend on eps, and
    # re-drawing per radius would make the eps comparison a comparison of latent draws too.
    decodes = {}
    for mode in ("onestep", "ode"):
        a2 = agent.replace(config=agent.config.copy(
            {"gpi_decode": mode, "flow_decode_steps": ode_steps}))
        out = []
        for i in range(N_ANCHORS):
            k = jax.random.PRNGKey(SEED + 1000 + i)
            u = jnp.clip(jax.random.normal(k, (N_LATENTS, d_a)), -LATENT_CLIP, LATENT_CLIP)
            obs_b = jnp.broadcast_to(jnp.asarray(anchors[i], jnp.float32), (N_LATENTS, anchors.shape[1]))
            out.append(np.asarray(a2.decode(obs_b, u)))
        decodes[mode] = np.stack(out)                      # (N_ANCHORS, N_LATENTS, d_a)

    # Uniform draws over the action box: the score a state-blind proposal gets on these
    # same balls. Without it a small MSE could just mean the ball's actions are everywhere.
    prior_acts = rng_np.uniform(-1.0, 1.0, (N_ANCHORS, N_LATENTS, d_a))

    report = {
        "env": cfg.env_name,
        "flow_ckpt_path": str(config["flow_ckpt_path"]),
        "flow_ckpt_epoch": int(config.get("flow_ckpt_epoch") or -1),
        "ode_steps_from_flags": ode_steps,
        "n_anchors": N_ANCHORS,
        "n_latents": N_LATENTS,
        "latent_clip": LATENT_CLIP,
        "min_ball": MIN_BALL,
        "seed": SEED,
        "metric": "per-dimension MSE between actions; ball = dataset rows within eps "
                  "of the anchor in per-dim standardized observation space",
        "geometry": index.protocol(),
        "action_var_dataset": float(act_all.var(axis=0).mean()),
        "by_eps": [],
    }

    for eps in EPS:
        members = tree.query_ball_point(anchors_std, r=float(eps))
        flow_min = {"onestep": [], "ode": []}
        flow_med = {"onestep": [], "ode": []}
        data_min, prior_min, sizes, ball_var = [], [], [], []
        for i, mem in enumerate(members):
            mem = np.asarray(mem, np.int64)
            # Self-exclusion: the anchor's own row would otherwise give the data reference a
            # free exact match, and it is the row the flow is being asked to predict.
            mem = mem[mem != anchor_pos[i]]
            if mem.size < MIN_BALL:
                continue
            ball = index.act[mem]                                  # (m, d_a)
            sizes.append(mem.size)
            ball_var.append(float(ball.var(axis=0).mean()))
            data_min.append(float(mse_matrix(anchor_act[i][None], ball).min()))
            prior_min.append(float(mse_matrix(prior_acts[i], ball).min()))
            for mode in ("onestep", "ode"):
                m = mse_matrix(decodes[mode][i], ball)             # (N_LATENTS, m)
                flow_min[mode].append(float(m.min()))
                flow_med[mode].append(float(np.median(m.min(axis=1))))

        if not sizes:
            report["by_eps"].append({"eps": eps, "balls_scored": 0,
                                     "note": f"no ball reached MIN_BALL={MIN_BALL}"})
            continue

        entry = {
            "eps": eps,
            "balls_scored": len(sizes),
            "ball_size": summarize(sizes),
            "ball_action_var_mean": float(np.mean(ball_var)),
            "min_mse": {
                "flow_onestep": summarize(flow_min["onestep"]),
                "flow_ode": summarize(flow_min["ode"]),
                "data_self": summarize(data_min),
                "uniform_prior": summarize(prior_min),
            },
            "median_decode_mse": {
                "flow_onestep": summarize(flow_med["onestep"]),
                "flow_ode": summarize(flow_med["ode"]),
            },
        }
        # Ratio to the data reference: 1.0 means the flow gets as close to a ball action as
        # a real held-out action from the same ball does.
        d_med = entry["min_mse"]["data_self"]["median"]
        entry["min_mse_ratio_to_data_median"] = {
            "flow_onestep": float(entry["min_mse"]["flow_onestep"]["median"] / (d_med + 1e-12)),
            "flow_ode": float(entry["min_mse"]["flow_ode"]["median"] / (d_med + 1e-12)),
            "uniform_prior": float(entry["min_mse"]["uniform_prior"]["median"] / (d_med + 1e-12)),
        }
        entry["frac_flow_below_data_self"] = {
            mode: float(np.mean(np.asarray(flow_min[mode]) < np.asarray(data_min)))
            for mode in ("onestep", "ode")
        }
        report["by_eps"].append(entry)
        print(f"eps={eps}: {len(sizes)} balls, median size {np.median(sizes):.0f}, "
              f"min-MSE flow(onestep)={entry['min_mse']['flow_onestep']['median']:.5f} "
              f"data={d_med:.5f} prior={entry['min_mse']['uniform_prior']['median']:.5f}")

    write_report(report, cfg, "diag_eps_ball_fit.json")


if __name__ == "__main__":
    main()
