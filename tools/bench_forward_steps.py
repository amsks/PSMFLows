"""Does the EM's forward decode need the same resolution as the inversion?

Stage B costs ~20 h per 1M rows, and essentially all of it is one term: every EM
iteration re-decodes all `num_samples` proposal draws through the full `flow_steps` ODE
to score ||G(s,u) - a||. That is 10 x 200 x 100 = 200k network evals per row. The 100
steps are a hard requirement for the INVERSE map (D3: NaN round-trip at 10 steps, KS
0.217 at 30, 1.2e-4 at 100) -- but nobody has checked whether the forward SCORING pass
needs them. If it holds up at 20, Stage B drops ~5x.

`precompute_preimages.py` asserts agent.flow_steps == inversion.n_initial_steps, so this
deliberately bypasses that tool and drives the EM directly: inversion stays pinned at 100
while the forward decode varies.

The honest quality metric is NOT the EM's own ESS (a coarse decoder can report a happy
ESS about a target it is itself computing wrongly). It is: take the posterior mean the
cheap EM produced, decode it with the ACCURATE 100-step decoder, and measure
||G_100(s, mu) - a||. A cheaper EM that still lands on a good preimage is a real win; one
whose ESS looks fine while its preimage decodes badly is the failure mode to catch.

Also reports the diverged-sample fraction, because the code notes the flow blows up for
tail draws only at >=100 steps -- 10 and 30 under-resolve it and stay deceptively finite,
which would silently disarm `preimage_valid`.

Run: MUJOCO_GL=egl .venv/bin/python tools/bench_forward_steps.py agent=fql \
    env_name=cube-single-play-singletask-v0 restore_path=<flow_dir> restore_epoch=500000
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_ROWS = 256          # one Stage-B batch
FORWARD_STEPS = [20, 50, 100]
REFERENCE_STEPS = 100  # the accurate decoder every variant is judged against


def _decode_at(agent, steps, states, noises):
    """Decode with an explicit step count, independent of agent.config['flow_steps']."""
    cfg = dict(agent.config)
    cfg["flow_steps"] = steps
    probe = agent.replace(config=ml_collections.ConfigDict(cfg))
    return np.asarray(probe.compute_flow_actions(states, noises=noises))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents_create(cfg, ex, config)

    inv = cfg.inversion
    alpha = float(inv.alpha)
    num_samples = int(inv.num_samples)
    n_steps = int(inv.n_steps)
    n_initial_steps = int(inv.n_initial_steps)   # inversion resolution, PINNED at 100

    states = np.asarray(ds["observations"][:N_ROWS])
    actions = np.asarray(ds["actions"][:N_ROWS])

    results = []
    for fsteps in FORWARD_STEPS:
        # Rebuild the agent with this forward resolution; the EM's internal
        # compute_flow_actions reads flow_steps off the agent config.
        c = dict(config)
        c["flow_steps"] = fsteps
        probe = agent.replace(config=ml_collections.ConfigDict(c))

        em = jax.jit(jax.vmap(
            lambda s, a, r: probe.compute_full_proposal_distribution_em(
                s, a, r, num_samples=num_samples, n_steps=n_steps,
                n_initial_steps=n_initial_steps, alpha=alpha,
                n_components=int(inv.num_clusters))))

        keys = jax.random.split(jax.random.PRNGKey(int(inv.seed)), N_ROWS)
        means, covs, weights, ess = em(jnp.asarray(states), jnp.asarray(actions), keys)
        means, ess = np.asarray(means), np.asarray(ess)
        jax.block_until_ready(means)

        t0 = time.time()
        means2, _, _, _ = em(jnp.asarray(states), jnp.asarray(actions), keys)
        jax.block_until_ready(means2)
        elapsed = time.time() - t0

        # Judge the fitted posterior mean with the ACCURATE decoder, not the cheap one.
        mu = means[:, 0, :] if means.ndim == 3 else means
        finite = np.isfinite(mu).all(-1)
        mu_safe = np.where(finite[:, None], mu, 0.0).astype(np.float32)
        decoded = _decode_at(agent, REFERENCE_STEPS, jnp.asarray(states), jnp.asarray(mu_safe))
        err = np.linalg.norm(decoded - actions, axis=-1)
        err = np.where(finite, err, np.inf)

        final_ess = ess[:, -1]
        ok = np.isfinite(err)
        results.append({
            "forward_steps": fsteps,
            "seconds_per_256_rows": round(elapsed, 2),
            "speedup_vs_100": None,
            "final_ess_mean": round(float(np.mean(final_ess)), 2),
            "final_ess_median": round(float(np.median(final_ess)), 2),
            "frac_ess_gt_20": round(float((final_ess > 20).mean()), 3),
            "nonfinite_mean_frac": round(float((~finite).mean()), 4),
            "decode_err_at_100_mean": round(float(np.mean(err[ok])), 5),
            "decode_err_at_100_median": round(float(np.median(err[ok])), 5),
            "decode_err_at_100_p95": round(float(np.percentile(err[ok], 95)), 5),
            "frac_decode_err_gt_0.1": round(float((err[ok] > 0.1).mean()), 3),
        })
        print(f"forward_steps={fsteps:3d}  {elapsed:6.2f}s/256  ESS={results[-1]['final_ess_mean']:6.2f}  "
              f"decode_err@100 median={results[-1]['decode_err_at_100_median']:.5f}  "
              f"p95={results[-1]['decode_err_at_100_p95']:.5f}")

    base = next(r["seconds_per_256_rows"] for r in results if r["forward_steps"] == REFERENCE_STEPS)
    for r in results:
        r["speedup_vs_100"] = round(base / r["seconds_per_256_rows"], 2)

    report = {
        "env": cfg.env_name, "restore_path": str(cfg.restore_path),
        "n_rows": N_ROWS, "alpha": alpha, "num_samples": num_samples,
        "em_iterations": n_steps, "inversion_steps_pinned": n_initial_steps,
        "reference_decoder_steps": REFERENCE_STEPS,
        "results": results,
    }
    write_report(report, cfg, "bench_forward_steps.json")


def agents_create(cfg, ex, config):
    agent = FQLAgent.create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained Stage-A flow (restore_path)"
    return restore_agent(agent, cfg.restore_path, cfg.restore_epoch)


if __name__ == "__main__":
    main()
