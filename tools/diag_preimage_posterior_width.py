"""Is the preimage genuinely a DISTRIBUTION, or is the point inverse the whole story?

This is the measurement that decides whether Stage B should ship mixtures or points.

An exact ODE flow is invertible, so the preimage of an action is a single point and a
"distribution" would be an artifact of the epsilon-relaxed target. But the DEPLOYED decode
is not the exact ODE — and the coverage probe says decodes are ~3x tighter than the local
data spread, which is what a many-to-one decoder looks like from the action side. If the
decoder really is many-to-one, then a large set of latents decodes to the same action, the
point inverse is one arbitrary member of that set, and every latent-space label in Stage C
(the TD anchor u_data, the actor's BC target) is an arbitrary pick from a set the mixture
would have represented properly.

So: fit the posterior on the fly (no npz needed, so the legacy and prior-corrected targets
can be compared without regenerating 1M rows), draw K latents from it, decode all of them,
and report the JOINT statistic —

    how far apart are the latents,  AND  how close are their decodes to the same action?

  far apart in u AND close in a  -> the preimage is a SET; the mixture carries information
                                    the point inverse discards, and mixtures are the right
                                    pipeline choice;
  tight in u                     -> the action pins the latent; the point inverse is the
                                    whole story and mixtures buy nothing.

Two calibrations make the numbers readable: prior draws (latents with no relation to the
action) bound the decode error from above, and the point inverse's own round-trip bounds it
from below.

Run (GPU, minutes):
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/diag_preimage_posterior_width.py agent=fql \
      env_name=cube-single-play-singletask-v0 agent.flow_steps=100 \
      restore_path=<stage_a_flow_dir> restore_epoch=500000 \
      report_out=/data-local/amsks/PSMFLows/logs/diag_preimage_posterior_width_cube.json
"""
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

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_ROWS = 256        # transitions to fit posteriors for
K_SAMPLES = 32      # latents drawn per row from the fitted posterior
ALPHAS = (20.0, 100.0, 500.0)   # inverse temperature of the relaxed target
# "Faithful" must mean faithful, not merely better than a random latent. A sample counts
# only if its decode lands within 5% of the typical action magnitude — two orders of
# magnitude looser than the point inverse's round-trip (1.2e-4) and still far tighter than
# the local data spread. The first version of this tool used "better than the 5th
# percentile of prior draws", which 23% of samples cleared while decoding half an action
# scale away; that bar cannot distinguish a genuine preimage set from a badly-sited blob.
FAITHFUL_FRAC_OF_ACTION_SCALE = 0.05


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    np.random.seed(int(cfg.seed))
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs the Stage-A flow checkpoint"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    rows = np.random.default_rng(int(cfg.seed)).choice(ds.size, N_ROWS, replace=False)
    obs = jnp.asarray(np.asarray(ds["observations"][rows]))
    act = jnp.asarray(np.asarray(ds["actions"][rows]))
    a_scale = float(np.abs(np.asarray(ds["actions"])).mean())
    d_a = act.shape[-1]
    inv = dict(cfg.inversion)
    n_init = int(inv.get("n_initial_steps", 100))

    # The point inverse, and its round-trip: the lower bound on decode error.
    u_point = jax.jit(jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, n_init)[0]))(obs, act)
    a_point = agent.compute_flow_actions(obs, noises=u_point)
    err_point = np.asarray(jnp.linalg.norm(a_point - jnp.clip(act, -1, 1), axis=-1))

    # Upper bound: latents drawn from the prior, i.e. unrelated to this action.
    u_prior = jax.random.normal(jax.random.PRNGKey(int(cfg.seed) + 1), (N_ROWS, K_SAMPLES, d_a))
    obs_k = jnp.broadcast_to(obs[:, None, :], (N_ROWS, K_SAMPLES, obs.shape[-1]))
    a_prior = agent.compute_flow_actions(obs_k.reshape(-1, obs.shape[-1]),
                                         noises=u_prior.reshape(-1, d_a))
    err_prior = np.asarray(jnp.linalg.norm(
        a_prior.reshape(N_ROWS, K_SAMPLES, d_a) - jnp.clip(act, -1, 1)[:, None, :], axis=-1))

    out = {"env": cfg.env_name, "flow_ckpt": str(cfg.restore_path),
           "restore_epoch": int(cfg.restore_epoch), "n_rows": N_ROWS,
           "k_samples": K_SAMPLES, "seed": int(cfg.seed),
           "action_scale_mean_abs": a_scale,
           "inversion": {k: inv.get(k) for k in ("alpha", "num_samples", "n_steps",
                                                 "n_initial_steps", "num_clusters")},
           "point_inverse_roundtrip": {"mean": float(err_point.mean()),
                                       "p95": float(np.percentile(err_point, 95))},
           "prior_draw_decode_error": {"mean": float(err_prior.mean()),
                                       "p05": float(np.percentile(err_prior, 5))},
           "targets": {}}

    # Sweep the temperature as well as the prior. If the posterior stays WIDE in u while
    # its decodes converge on the action as alpha rises, the decoder has genuine flat
    # directions and the preimage is a real set. If instead the width shrinks toward the
    # point inverse as alpha rises, the width was only ever the temperature and the mixture
    # is a blurred point, not a set.
    for ps, alpha in [(0.0, 20.0)] + [(1.0, a) for a in ALPHAS]:
        fit = jax.jit(jax.vmap(lambda s, a_, r: agent.compute_full_proposal_distribution_em(
            s, a_, r, num_samples=int(inv.get("num_samples", 200)),
            n_steps=int(inv.get("n_steps", 10)), n_initial_steps=n_init,
            alpha=alpha, n_components=int(inv.get("num_clusters", 1)),
            prior_scale=ps)))
        means, covs, weights, ess = fit(
            obs, act, jax.random.split(jax.random.PRNGKey(int(cfg.seed)), N_ROWS))

        # Draw K latents from component 0 (num_clusters=1 in every shipped config; with
        # K>1 this is the dominant component, which is what the report says it is).
        mu, cov = means[:, 0], covs[:, 0]
        L = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(d_a))
        z = jax.random.normal(jax.random.PRNGKey(int(cfg.seed) + 2), (N_ROWS, K_SAMPLES, d_a))
        u = mu[:, None, :] + jnp.einsum("bij,bkj->bki", L, z)
        a_dec = agent.compute_flow_actions(obs_k.reshape(-1, obs.shape[-1]),
                                           noises=u.reshape(-1, d_a))
        err = np.asarray(jnp.linalg.norm(
            a_dec.reshape(N_ROWS, K_SAMPLES, d_a) - jnp.clip(act, -1, 1)[:, None, :], axis=-1))
        u_np = np.asarray(u)
        dist_to_point = np.linalg.norm(u_np - np.asarray(u_point)[:, None, :], axis=-1)
        pairwise = np.linalg.norm(u_np[:, :, None, :] - u_np[:, None, :, :], axis=-1)
        # THE discriminator: latents genuinely elsewhere in the box (>1 away from the point
        # inverse, i.e. a real move in a unit-variance prior) whose decode still lands
        # closer to the action than an unrelated prior draw's 5th percentile.
        faithful_thresh = FAITHFUL_FRAC_OF_ACTION_SCALE * a_scale
        far_and_faithful = (dist_to_point > 1.0) & (err < faithful_thresh)
        out["targets"][f"prior_scale={ps},alpha={alpha}"] = {
            "faithful_threshold": float(faithful_thresh),
            "decode_error_median": float(np.median(err)),
            "frac_faithful_any_distance": float((err < faithful_thresh).mean()),
            "fitted_per_dim_variance_mean": float(np.mean(np.trace(np.asarray(cov), axis1=-2, axis2=-1) / d_a)),
            "final_ess_mean": float(np.asarray(ess)[:, -1].mean()),
            "latent_spread_per_dim_sd": float(u_np.std(axis=1).mean()),
            "latent_dist_to_point_inverse_mean": float(dist_to_point.mean()),
            "latent_pairwise_dist_mean": float(pairwise.mean()),
            "decode_error_mean": float(err.mean()),
            "decode_error_normalised": float(err.mean() / a_scale),
            "frac_far_in_u_and_faithful_in_a": float(far_and_faithful.mean()),
        }

    fixed = out["targets"][f"prior_scale=1.0,alpha={ALPHAS[-1]}"]
    out["verdict"] = (
        f"posterior latents sit {fixed['latent_dist_to_point_inverse_mean']:.2f} from the "
        f"point inverse on average (pairwise {fixed['latent_pairwise_dist_mean']:.2f}) and "
        f"decode {fixed['decode_error_mean']:.4f} from the recorded action against "
        f"{out['prior_draw_decode_error']['mean']:.4f} for an unrelated prior draw; "
        f"at the sharpest alpha, {100 * fixed['frac_far_in_u_and_faithful_in_a']:.0f}% of "
        f"drawn latents are BOTH >1 away in u and within "
        f"{FAITHFUL_FRAC_OF_ACTION_SCALE:.0%} of action scale in a -> " + (
            "the decoder has genuine flat directions: the preimage is a SET and the point "
            "inverse discards it"
            if fixed["frac_far_in_u_and_faithful_in_a"] > 0.2 else
            "the posterior width is the TEMPERATURE, not the decoder; the mixture is a "
            "blurred point and the point inverse loses little"))
    print("\n" + out["verdict"])
    write_report(out, cfg, "diag_preimage_posterior_width.json")


if __name__ == "__main__":
    main()
