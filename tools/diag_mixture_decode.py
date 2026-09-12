"""Do samples from the stored preimage posterior decode to the recorded action?

Stage B stores, per transition, a point inverse `u_pt` and an EM Gaussian posterior over
`u`. Everything that trains on more than one latent per transition -- `use_point_preimage
=false`, and `measure_u_samples > 1` (2026-09-08) -- relies on samples from that posterior
being *other preimages of the same action*. `preimage_roundtrip` in the npz answers this
for the POINT only; nothing measured it for a sample.

The distinction decides whether such an arm is trainable at all. If `G(s, u_sample)` is far
from `a`, the extra rows carry the point inverse's `(s, s')` target while decoding somewhere
else -- a fiction, at whatever fraction of the batch the extra samples occupy.

Four latents per row, one decode each, against the same recorded action:

  point     the stored backward-ODE inverse            (the floor)
  mean      the posterior's mean                        (COMPENDIUM 4.9 measured this one)
  sample    a draw from the posterior                   (what training would actually use)
  prior     a clipped N(0, I) draw                      (the ceiling: no inversion at all)

and two ladders:

  --jitter   isotropic Gaussian balls of radius sigma around the point inverse. Picks a
             jitter WIDTH from evidence rather than from the inversion's temperature: take
             the largest sigma whose mean decode error still sits inside the point's p90.
  --shrink   the stored posterior with every component's covariance scaled by c^2, i.e.
             `sample_preimage_noise(..., scale=c)`. Same criterion, but the draw keeps the
             posterior's SHAPE: the EM covariance is stretched along the directions in
             which the decode barely moves, so at a matched decode error a shrunk
             anisotropic draw should sit FURTHER from u_data than a round ball -- more of
             the preimage set for the same fidelity. `dist_to_point` at the chosen c is the
             number that settles whether the shape is worth anything.

With --shrink the report also carries a per-COMPONENT breakdown at the recommended c and the
mass of the component that owns u_data, so a component whose samples decode badly can be
dropped and the weights renormalised rather than shrinking the whole mixture harder. Every
published npz has num_clusters=1, which makes that breakdown a single row and the drop rule
vacuous -- it is computed anyway so a future multi-component inversion is covered.

Run (CPU is fine; the decode is the one-step distilled head unless --ode):
  .venv/bin/python tools/diag_mixture_decode.py \
      --npz $PSM_DATA/preimages/cube-single-play.npz \
      --flow $PSM_DATA/flow/cube-single-play --epoch 500000 \
      --out $PSM_DATA/logs/diag_mixture_decode_cube.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def _stats(x):
    x = np.asarray(x, np.float64).ravel()
    return {"mean": round(float(x.mean()), 4), "median": round(float(np.median(x)), 4),
            "p90": round(float(np.quantile(x, 0.9)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--flow", required=True, help="Stage-A fql bc_only checkpoint dir")
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--u_clip", type=float, default=3.0)
    ap.add_argument("--ode", action="store_true", help="exact decode instead of one-step")
    ap.add_argument("--jitter", default="",
                    help="comma-separated sigmas: add jitter_<sigma> rows, u_pt + sigma*eps")
    ap.add_argument("--shrink", default="",
                    help="comma-separated c: sample the posterior with cov scaled by c^2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import jax.numpy as jnp
    import ml_collections

    from agents.psmflow import PSMFlowAgent, get_config
    from utils.flow_inversion import sample_preimage_noise

    d = np.load(a.npz, mmap_mode="r")
    n = min(a.rows, d["observations"].shape[0])
    obs = np.asarray(d["observations"][:n], np.float32)
    act = np.asarray(d["actions"][:n], np.float32)
    pt = np.asarray(d["noise_preimage_point"][:n], np.float32)
    mean = np.asarray(d["noise_preimage_mean"][:n, 0], np.float32)

    # Full (B, K, ...) posterior, for the per-component pass; the rows above are its K=0
    # slice, which is the whole thing on every npz published so far.
    mean_k = np.asarray(d["noise_preimage_mean"][:n], np.float32)
    cov_k = np.asarray(d["noise_preimage_cov"][:n], np.float32)
    w_k = np.asarray(d["noise_preimage_weights"][:n], np.float32)
    n_comp = mean_k.shape[1]

    rng = np.random.default_rng(a.seed)

    def draw(scale, weights=None):
        """One posterior sample per row at covariance scale `scale`, clipped to the box."""
        w = w_k if weights is None else weights
        return np.clip(sample_preimage_noise(mean_k, cov_k, w, rng=rng, scale=scale),
                       -a.u_clip, a.u_clip)

    latents = {"point": pt, "mean": mean, "sample": draw(1.0),
               "prior": np.clip(rng.standard_normal(mean.shape), -a.u_clip, a.u_clip)}
    for sig in [float(x) for x in a.jitter.split(",") if x.strip()]:
        latents[f"jitter_{sig:g}"] = np.clip(
            pt + sig * rng.standard_normal(pt.shape), -a.u_clip, a.u_clip)
    shrinks = [float(x) for x in a.shrink.split(",") if x.strip()]
    for c in shrinks:
        latents[f"shrink_{c:g}"] = draw(c)

    cfg = get_config()
    with cfg.unlocked():
        cfg["flow_ckpt_path"] = a.flow
        cfg["flow_ckpt_epoch"] = a.epoch
        cfg["u_clip"] = a.u_clip
        cfg["gpi_decode"] = "ode" if a.ode else "onestep"
    cfg = ml_collections.ConfigDict(cfg)
    agent = PSMFlowAgent.create(a.seed, obs[:1], act[:1], cfg)

    report = {"probe": "decode distance of stored-posterior latents to the recorded action",
              "npz": a.npz, "flow": a.flow, "epoch": a.epoch, "n_rows": n,
              "decode": cfg["gpi_decode"], "u_clip": a.u_clip, "n_components": n_comp,
              "action_norm": _stats(np.linalg.norm(act, axis=-1)),
              "latents": {}}
    for name, u in latents.items():
        dec = np.asarray(agent.decode(jnp.asarray(obs), jnp.asarray(u, jnp.float32)))
        report["latents"][name] = {
            "decode_error": _stats(np.linalg.norm(dec - act, axis=-1)),
            "dist_to_point": _stats(np.linalg.norm(u - pt, axis=-1)),
            "u_norm": _stats(np.linalg.norm(u, axis=-1))}
    ref = report["latents"]["point"]["decode_error"]["mean"]
    span = report["latents"]["prior"]["decode_error"]["mean"] - ref
    for name, r in report["latents"].items():
        # 0 = as good as the point inverse, 1 = no better than an uninverted prior draw.
        r["fraction_of_the_way_to_prior"] = (
            round((r["decode_error"]["mean"] - ref) / span, 3) if span > 1e-9 else None)
    # The selection rule, applied to whichever ladder was swept: the largest setting whose
    # POOLED mean decode error still sits inside the point inverse's own p90.
    gate = report["latents"]["point"]["decode_error"]["p90"]
    report["gate_point_p90"] = gate
    for tag, vals in (("shrink", shrinks),
                      ("jitter", [float(x) for x in a.jitter.split(",") if x.strip()])):
        ok = [v for v in vals
              if report["latents"][f"{tag}_{v:g}"]["decode_error"]["mean"] <= gate]
        if vals:
            report[f"recommended_{tag}"] = max(ok) if ok else None
            report[f"recommended_{tag}_note"] = (
                f"largest {tag} whose mean decode error <= the point inverse's p90 ({gate})"
                if ok else f"NO {tag} in the swept range stays inside the point's p90")
    # Per-component, at the recommended shrink. A component whose samples decode worse than
    # the gate is a DROP candidate: renormalise the weights over the survivors rather than
    # shrinking the whole mixture harder. Vacuous while num_clusters=1.
    c_star = report.get("recommended_shrink")
    if shrinks and c_star:
        eye = np.eye(n_comp, dtype=np.float32)
        comps = {}
        # Which component owns u_data: the one whose Gaussian is densest at the point
        # inverse (a Mahalanobis argmin, the log-det term dropped -- it is a per-component
        # constant and only the ranking is used).
        dif = pt[:, None] - mean_k
        prec = np.linalg.pinv(cov_k + 1e-6 * np.eye(cov_k.shape[-1], dtype=cov_k.dtype))
        maha = np.einsum("bki,bkij,bkj->bk", dif, prec, dif)
        owner = maha.argmin(1)
        for k in range(n_comp):
            u = draw(c_star, weights=np.broadcast_to(eye[k], (n, n_comp)))
            dec = np.asarray(agent.decode(jnp.asarray(obs), jnp.asarray(u, jnp.float32)))
            comps[str(k)] = {
                "weight_mean": round(float(w_k[:, k].mean()), 4),
                "decode_error": _stats(np.linalg.norm(dec - act, axis=-1)),
                "dist_to_point": _stats(np.linalg.norm(u - pt, axis=-1)),
                "owns_u_data_frac": round(float((owner == k).mean()), 4),
                "drop": bool(np.linalg.norm(dec - act, axis=-1).mean() > gate)}
        report["components_at_recommended_shrink"] = comps
        report["owner_component_mass_mean"] = round(
            float(w_k[np.arange(n), owner].mean()), 4)
        drops = [k for k, v in comps.items() if v["drop"]]
        # Dropping every component leaves nothing to sample. At num_clusters=1 the
        # per-component draw IS the pooled draw, so a marginal c can flag the only
        # component on sampling noise alone -- that is a MARGINAL verdict on c, not a
        # reason to empty the mixture.
        if len(drops) >= n_comp:
            report["drop_candidates"] = []
            report["drop_note"] = (
                f"all {n_comp} components flagged at c={c_star}; refusing to empty the "
                "mixture. The pooled draw sits at the gate, so read c as MARGINAL and "
                "prefer the next c down.")
        else:
            report["drop_candidates"] = drops

    out = a.out or (a.npz + ".mixture_decode.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["latents"], indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
