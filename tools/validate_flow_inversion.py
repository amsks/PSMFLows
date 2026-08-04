"""Flow-inversion health check (PSMFlows diagnostic D3) on a real OGBench dataset.

Run against a PRETRAINED behaviour flow (scripts/pretrain_behavior_flow.sh):

  .venv/bin/python tools/validate_flow_inversion.py \
      env_name=cube-single-play-singletask-v0 \
      restore_path='/var/local/amsks/exp/PSMFLows/bcflow_*/sd000_*' restore_epoch=500000

Omit restore_path to characterize a RANDOM flow — useful only as a control, since the
typicality test below is a statement about a *fitted* flow.

Reports three things (RESEARCH_NOTE.md §3.2, write-up §6):

  roundtrip   ||G(s, E(s,a)) - a||. Purely a property of the inverter vs. the network;
              large values mean the implicit-Euler inversion is not finding the preimage.
  typicality  the load-bearing one. Under an exact flow (assumption A2), Lemma 6.1 gives
              E(s,a) ~ p0 for (s,a) ~ D, so ||E(s,a)||^2 ~ chi^2_{d_a}. If the flow
              underfits, inverted latents land in the tail of p0 and the latents seen at
              TRAIN time do not match the u ~ p0 drawn at TEST time. This chi^2 check is
              therefore a direct, falsifiable test of A2.
  ess         effective sample size of the EM preimage posterior: how informative the
              epsilon-relaxed preimage distribution is.

Gate (spec §1): chi2 typicality >= 0.95, round-trip < 0.1, mean ESS > 20 => inversion is
trustworthy enough to build the representation on. Otherwise the flow needs capacity /
more ODE steps. `gate_pass` in the report is the conjunction.

The sample is SEEDED off cfg.seed. It did not used to be (Dataset.sample draws from
global np.random), which made the typicality numbers move run to run and invited reading
a different batch as a change in the flow.
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

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent


def _chi2_report(sq_norms, d_a):
    """||u||^2 vs chi^2_{d_a}: mean, median, coverage of the 95% ball, and a KS test."""
    out = {
        "mean_sq_norm": round(float(np.mean(sq_norms)), 4),
        "expected_mean": float(d_a),                  # E[chi^2_k] = k
        "median_sq_norm": round(float(np.median(sq_norms)), 4),
    }
    try:
        from scipy.stats import chi2, kstest
        out["expected_median"] = round(float(chi2.ppf(0.5, d_a)), 4)
        out["frac_within_95pct_ball"] = round(float(np.mean(sq_norms <= chi2.ppf(0.95, d_a))), 4)
        out["expected_frac"] = 0.95
        # A KS test against the exact chi^2_{d_a} CDF is the sharpest single number.
        ks = kstest(np.asarray(sq_norms), lambda v: chi2.cdf(v, d_a))
        out["ks_stat"] = round(float(ks.statistic), 4)
        out["ks_pvalue"] = float(ks.pvalue)
    except ImportError:
        out["note"] = "scipy unavailable; quantile/KS diagnostics skipped"
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    flow_steps = int(cfg.get("flow_steps", 100))
    n = int(cfg.get("n_samples", 256))

    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    # Seeded: Dataset.sample() otherwise draws from global np.random, so every invocation
    # scored a different batch and the typicality numbers wandered between runs.
    idxs = np.random.default_rng(int(cfg.seed)).integers(0, ds.size, n)
    batch = ds.sample(n, idxs=idxs)
    obs = jnp.asarray(batch["observations"])
    act = jnp.asarray(batch["actions"])

    # Build the agent from the Hydra `agent` group, NOT fql.get_config(): the net shapes
    # must match the Stage-A run whose checkpoint we restore below.
    assert cfg.agent.agent_name == "fql", "run with agent=fql (Stage-A flow shapes)"
    agent_cfg = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent_cfg["flow_steps"] = flow_steps
    agent = FQLAgent.create(int(cfg.seed), obs[:1], act[:1], agent_cfg)

    trained = cfg.get("restore_path", None) is not None
    if trained:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    else:
        assert cfg.inversion.get("allow_untrained", False), (
            "a RANDOM flow is only a control; the typicality test is a statement about a "
            "FITTED flow. Set restore_path=<stage-A ckpt dir> "
            "(or inversion.allow_untrained=true to characterize the control).")

    # --- round trip: a -> E(s,a) -> G(s,·), with matched step discretization ---
    # jit(vmap), NOT bare vmap: on GPU, un-jitted vmap of this function (lax.scan with an
    # inner fori_loop, plus jacfwd) returns all-NaN, reproducibly, while the same call
    # jitted, looped in python, or run on CPU is correct. utils/flow_inversion.py already
    # jits its batched EM for this reason; this tool did not, which made every number it
    # printed for a trained flow NaN.
    preimage = jax.jit(jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, flow_steps)[0]
    ))(obs, act)
    recon = agent.compute_flow_actions(obs, noises=preimage)
    rt = jnp.linalg.norm(recon - act, axis=-1)

    # --- D3 typicality: ||E(s,a)||^2 should be chi^2_{d_a} under A2 (Lemma 6.1) ---
    d_a = int(act.shape[-1])
    sq_norms = np.asarray(jnp.sum(preimage ** 2, axis=-1))

    # --- ESS of the EM preimage posterior ---
    # Read from `cfg.inversion`, the same group Stage B uses (mapped exactly as
    # utils/flow_inversion.py:augment_dataset_with_preimage_distribution does, including
    # num_clusters -> n_components). These were hardcoded to alpha=1.0 / n_components=3,
    # so the gate scored a posterior the precompute never computes: it was blind to
    # `inversion.alpha` — the one knob that decides whether ESS passes — and fit 3
    # components against the configured 1.
    inv = cfg.inversion
    keys = jax.random.split(jax.random.PRNGKey(int(inv.get("seed", 0))), obs.shape[0])
    _, _, _, ess = jax.jit(jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k,
            num_samples=int(inv.get("num_samples", 100)),
            n_steps=int(inv.get("n_steps", 10)),
            n_initial_steps=int(inv.get("n_initial_steps", 100)),
            alpha=float(inv.get("alpha", 1.0)),
            n_components=int(inv.get("num_clusters", 1)),
        )
    ))(obs, act, keys)

    # `ess` is (B, n_steps): one value per EM iteration, from the lax.scan in
    # compute_full_proposal_distribution_em. The posterior Stage B stores is the LAST
    # iterate, and precompute_preimages.py persists ess[:, -1] — gate on that same
    # statistic. Averaging the whole trace (as this tool previously did) understates it,
    # because ESS improves across EM iterations as the proposal tightens onto the target.
    ess_final = ess[:, -1]
    rt_mean = float(jnp.mean(rt))
    mean_ess = float(jnp.mean(ess_final))
    report = {
        "flow": "TRAINED" if trained else "RANDOM (control)",
        "env": cfg.env_name,
        "seed": int(cfg.seed),
        "n": int(obs.shape[0]),
        "action_dim": d_a,
        "flow_steps": flow_steps,
        "roundtrip_l2_mean": round(rt_mean, 6),
        "roundtrip_l2_median": round(float(jnp.median(rt)), 6),
        "roundtrip_l2_max": round(float(jnp.max(rt)), 6),
        "typicality": _chi2_report(sq_norms, d_a),
        "mean_ess": round(mean_ess, 3),
        "min_ess": round(float(jnp.min(ess_final)), 3),
        # Trace mean over all EM iterations — comparable with numbers this tool printed
        # before 2026-08-04, which gated on this (lower) statistic by mistake.
        "mean_ess_trace": round(float(jnp.mean(ess)), 3),
        # ESS is only interpretable against the settings that produced it (alpha above all).
        "inversion": OmegaConf.to_container(inv, resolve=True),
    }

    # Gate (spec §1). The typicality term is the fraction inside the central 99% chi^2
    # band — a coverage statement, deliberately weaker than the KS test above, which
    # rejects on shape differences too small to matter for training.
    try:
        from scipy import stats
        lo, hi = stats.chi2.ppf(0.005, d_a), stats.chi2.ppf(0.995, d_a)
        typicality = float(np.mean((sq_norms >= lo) & (sq_norms <= hi)))
        report["chi2_typicality_99pct_band"] = round(typicality, 4)
        report["gate_pass"] = bool(typicality >= 0.95 and rt_mean < 0.1 and mean_ess > 20)
        report["gate_detail"] = {
            "typicality>=0.95": bool(typicality >= 0.95),
            "roundtrip<0.1": bool(rt_mean < 0.1),
            "mean_ess>20": bool(mean_ess > 20),
        }
    except ImportError:
        report["note_gate"] = "scipy unavailable; gate not evaluated"

    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
