"""T2: do the flow's generated (state, action) pairs exist anywhere in the buffer?

The coverage probe compared decodes to the k-NN actions at the SAME state. This asks the
global version: sample N prior latents at M dataset states, decode through the deployed
one-step path, and for each generated pair search the whole buffer (state-first: k-NN
states in standardized observation space, then the closest action among those states'
actions). The required calibration baseline is the data matching itself -- each of the M
states' recorded actions searched the same way with itself excluded -- which defines
what "close" means for real pairs. Verdict: generated ~= baseline means the flow stays
within what the buffer contains somewhere; generated >> baseline means it emits pairs
with no counterpart in the data.

Run (GPU, minutes):
  MUJOCO_GL=egl .venv/bin/python tools/diag_generated_pair_support.py agent=latentrl \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=<npz> agent.use_point_preimage=true \
      report_out=/data-local/amsks/PSMFLows/logs/diag_generated_pair_support_cube.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf
from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.geometry import INDEX_ROWS, NeighbourIndex
from utils.log_utils import write_report

M_STATES = 1024
N_LATENTS = 64
K_STATES = 64          # nearest states retrieved per query (this probe's global search
                       # is deliberately wider than the shared k=32 neighbourhood)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    rng_np = np.random.default_rng(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        cfg.seed, ex["observations"], ex["actions"], config)
    # No restore needed: only the FROZEN flow (loaded in create) is used.

    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])

    # Shared geometry protocol (utils/geometry): standardized obs, seeded index.
    index = NeighbourIndex(obs_all, act_all, index_rows=INDEX_ROWS, seed=int(cfg.seed))
    a_scale = index.action_scale

    qrows = rng_np.choice(index.rows, size=M_STATES, replace=False)
    states = obs_all[qrows]

    # Generated pairs: N prior latents decoded at each state via the deployed path.
    rng = jax.random.PRNGKey(int(cfg.seed))
    gen_min, gen_state_d = [], []
    d_a = act_all.shape[-1]
    for i in range(M_STATES):
        rng, k = jax.random.split(rng)
        u = np.clip(np.asarray(jax.random.normal(k, (N_LATENTS, d_a))),
                    -float(config["u_clip"]), float(config["u_clip"]))
        obs_b = np.broadcast_to(states[i], (N_LATENTS, states.shape[1]))
        acts = np.asarray(agent.decode(obs_b, u))
        sd_state, pos = index.query(states[i], k=K_STATES)
        sd_state, pos = sd_state[0], pos[0]
        neigh_a = index.act[pos]                                  # (K, d_a)
        d = np.linalg.norm(acts[:, None, :] - neigh_a[None], axis=-1)  # (N, K)
        j = d.argmin(axis=1)
        gen_min.append(d.min(axis=1) / a_scale)
        gen_state_d.append(np.asarray(sd_state)[j])               # state-dist of argmin
    gen_min = np.concatenate(gen_min)
    gen_state_d = np.concatenate(gen_state_d)

    # Baseline: the data matching itself, self excluded.
    base_min = index.min_action_dist(obs_all[qrows], act_all[qrows], k=K_STATES,
                                     exclude_rows=qrows)
    b95 = float(np.percentile(base_min, 95))

    agg = lambda x: {"mean": float(np.mean(x)), "median": float(np.median(x)),
                     "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99))}
    frac = float((gen_min > b95).mean())
    report = {
        "env": cfg.env_name,
        "flow_ckpt": str(config["flow_ckpt_path"]),
        "decode": "onestep (deployed)",
        "m_states": M_STATES, "n_latents": N_LATENTS, "k_states": K_STATES,
        "geometry": index.protocol(), "seed": int(cfg.seed),
        "action_scale_mean_abs": a_scale,
        "generated_min_action_dist": agg(gen_min),
        "generated_argmin_state_dist": agg(gen_state_d),
        "baseline_data_min_action_dist": agg(base_min),
        "baseline_p95": b95,
        "frac_generated_beyond_baseline_p95": frac,
        "verdict": ("generated pairs match the buffer about as well as data matches "
                    f"itself (frac beyond baseline p95 = {frac:.3f})" if frac < 0.2 else
                    f"flow emits pairs with weak counterparts in data: {frac:.1%} of "
                    "generated actions exceed the baseline's p95 match distance"),
    }
    write_report(report, cfg, "diag_generated_pair_support.json")


if __name__ == "__main__":
    main()
