"""Report flow-inversion round-trip error and ESS on a real OGBench dataset.

Run:
  JAX_PLATFORMS=cpu .venv-jax/bin/python tools/validate_flow_inversion.py \
      env_name=pointmaze-medium-navigate-singletask-task1-v0

The round-trip number is training-independent (it characterizes the inverter vs. the
network). ESS comes from the EM proposal and indicates how informative the preimage
distribution is. Low round-trip + reasonable ESS => inversion is trustworthy enough
to build the PSMFlows representation on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import jax
import jax.numpy as jnp

from agents.fql import FQLAgent, get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    flow_steps = 100
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    batch = ds.sample(256)
    obs = jnp.asarray(batch['observations'])
    act = jnp.asarray(batch['actions'])

    # Build a real FQL agent at the dataset's dims (create derives ob_dims/action_dim).
    agent_cfg = get_config()
    agent_cfg['flow_steps'] = flow_steps
    agent = FQLAgent.create(0, obs[:1], act[:1], agent_cfg)

    # Round-trip: action -> preimage -> action, with matched step discretization.
    preimage = jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, flow_steps)[0]
    )(obs, act)
    recon = agent.compute_flow_actions(obs, noises=preimage)
    rt = float(jnp.mean(jnp.linalg.norm(recon - act, axis=-1)))

    # ESS from the EM proposal distribution.
    keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
    _, _, _, ess = jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0, n_components=3
        )
    )(obs, act, keys)

    print({
        "roundtrip_l2": rt,
        "mean_ess": float(jnp.mean(ess)),
        "min_ess": float(jnp.min(ess)),
        "n": int(obs.shape[0]),
        "flow_steps": flow_steps,
    })


if __name__ == "__main__":
    main()
