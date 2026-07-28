"""D2: are fixed-u flow policies coherent and diverse?

The method indexes policies by a latent u. That only means something if holding u fixed
produces a REPRODUCIBLE behaviour (coherence) and different u produce DIFFERENT ones
(diversity). If every u collapses to the same rollout, psi(s, u', u) has nothing to
discriminate and GPI is choosing among identical policies.

Rolls pi_u (one-step decode of a FIXED latent u each step) in the real env for a grid
of u draws. Reports (stdout JSON): mean within-u pairwise final-state distance vs
across-u; consistency ratio < 1 means u indexes reproducible behaviours.

Run: JAX_PLATFORMS='' .venv/bin/python tools/eval_fixed_u_rollouts.py \
    agent=fql env_name=cube-single-play-singletask-v0 \
    restore_path='/var/local/amsks/exp/<flow_run_dir>' restore_epoch=1000000

On midi-01 add MUJOCO_GL=egl (no DISPLAY) and OGBENCH_DATASET_DIR=/var/local/amsks/ogbench.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.flax_utils import restore_agent

NUM_U = 16
EPISODES_PER_U = 4
MAX_STEPS = 200


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ex_obs = train_dataset['observations'][:1]
    ex_act = train_dataset['actions'][:1]
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql'
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, ex_obs, ex_act, agent_cfg)
    assert cfg.restore_path is not None, 'D2 needs a trained flow (restore_path)'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    rng = np.random.default_rng(cfg.seed)
    d_a = ex_act.shape[-1]
    us = np.clip(rng.standard_normal((NUM_U, d_a)), -3, 3).astype(np.float32)

    finals = np.zeros((NUM_U, EPISODES_PER_U, ex_obs.shape[-1]), np.float32)
    onestep = jax.jit(lambda o, u: agent.network.select('actor_onestep_flow')(o, u))
    for i, u in enumerate(us):
        for ep in range(EPISODES_PER_U):
            # Same start states across u, so a difference in final state is attributable
            # to the latent rather than to the reset.
            ob, _ = eval_env.reset(seed=int(cfg.seed) * 1000 + ep)
            for _ in range(MAX_STEPS):
                a = np.clip(np.asarray(onestep(ob[None], u[None]))[0], -1, 1)
                ob, _, term, trunc, _ = eval_env.step(a)
                if term or trunc:
                    break
            finals[i, ep] = ob

    within = np.mean([np.linalg.norm(finals[i, a] - finals[i, b])
                      for i in range(NUM_U) for a in range(EPISODES_PER_U) for b in range(a + 1, EPISODES_PER_U)])
    flat = finals.reshape(-1, finals.shape[-1])
    across = np.mean(np.linalg.norm(flat[:, None] - flat[None], axis=-1))
    print(json.dumps({
        "env": cfg.env_name,
        "seed": int(cfg.seed),
        "num_u": NUM_U, "episodes_per_u": EPISODES_PER_U,
        "within_u_final_dist": float(within),
        "across_final_dist": float(across),
        "consistency_ratio": float(within / (across + 1e-8)),
    }, indent=2))


if __name__ == "__main__":
    main()
