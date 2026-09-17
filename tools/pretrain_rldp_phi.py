"""Policy-free basis phi by latent dynamics prediction (RLDP, arXiv 2603.15857, Eq. 6-8).

PSMFlow's phi is fitted jointly with the measure psi, so its geometry is whatever the
contrastive TD loss over the flow-indexed policy family asks for. This tool fits a phi on
the SAME dataset with NO policy in the loop: a latent predictor rolls phi(s_0) forward
through the recorded actions and is asked to land on phi_bar(s_t) at every step, with an
orthonormality term keeping the basis from collapsing. The result plugs into a stage-C run
unchanged through `agent.phi_restore_path` (`agents/psmflow.py::_load_phi_params`) with
`agent.train_phi=false`, so the only thing that changes in that run is the basis.

Losses, as implemented (B segments of length H+1 per batch, d = z_dim):

    h_0     = phi(s_0)
    h_{t+1} = g(h_t, a_t)                    an MLP on (h, a), output on the sqrt(d) sphere
                                             like phi (the paper writes g(h, a)^T w)
    L_dyn   = sum_{t=1..H} mean_B || h_t - stopgrad(phi_bar(s_t)) ||^2
    L_orth  = || (1/B) sum_B phi(s_0) phi(s_0)^T - I_d ||_F^2
    L       = L_dyn + lambda L_orth

phi_bar is a Polyak target of phi (tau). `utils.psm_common.ortho_loss` is the B x B
sample-Gram form used by the measure loss (off-diagonal penalty minus the diagonal mean),
not the d x d covariance-to-identity form the paper writes, so L_orth is implemented here
as the Frobenius form directly.

phi is `utils.psm_networks.PhiMap` with the affine run's kwargs (z_dim 128, hidden 256,
2 layers, sqrt(d)-sphere output), and the checkpoint is written in `save_agent`'s layout,
so `_load_phi_params` reads `pickle["agent"]["phi"]["params"]` exactly as it would from
a stage-C run directory.

Deviations from the paper: 1M steps (paper 2M), d = 128 (ours, so psi's shapes are
unchanged), batch 1024, Adam 3e-4, H = 5 and lambda = 1 (the paper's best), and the
orthonormality term is taken on phi(s_0) of the batch only.

Run (GPU, see scripts/slurm/pretrain_rldp_phi.sbatch):

    python tools/pretrain_rldp_phi.py --dataset_npz $PSM_DATA/preimages/cube-single-play.npz \
        --out_dir $PSM_DATA/exp/PSMFLows/rldp_phi_cube/H5_sd0 --steps 1000000

Writes `params_<step>.pkl` every `--save_interval`, `flags.json`, and the loss curve as
`rldp_curve.json` (`report_out`).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the XLA guard MUST precede every jax import (see agents/psmflow.py)
# isort: off
import utils.xla_guard  # noqa: F401
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.psm_networks import PhiMap, psm_norm
# isort: on

PHI_KWARGS = {"z_dim": 128, "hidden_dim": 256, "hidden_layers": 2, "norm": True}  # the affine run's phi


# ------------------------------------------------------------------ networks
class LatentPredictor(nn.Module):
    """g(h, a) -> h': MLP on the concatenated (latent, action), output on the sqrt(d) sphere."""

    z_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, h, a):
        x = jnp.concatenate([h, a], axis=-1)
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.z_dim)(x)
        return psm_norm(x)


# ------------------------------------------------------------------ data
def load_transitions(dataset_npz=None, env_name=None):
    """(observations, actions, next_observations, terminals) as float32/float32/float32/float32.

    `dataset_npz` is a preimage-augmented npz (utils.flow_inversion.load_augmented_dataset),
    i.e. the exact rows the stage-C run trains on; `env_name` goes through
    envs.env_utils.make_env_and_datasets, the loader main.py uses.
    """
    if dataset_npz:
        from utils.flow_inversion import load_augmented_dataset
        data = load_augmented_dataset(dataset_npz)
    else:
        assert env_name, "one of --dataset_npz / --env_name is required"
        from envs.env_utils import make_env_and_datasets
        _, _, data, _ = make_env_and_datasets(env_name)
        data = dict(data)
    return tuple(np.asarray(data[k], np.float32)
                 for k in ("observations", "actions", "next_observations", "terminals"))


def episode_ends(terminals):
    """Index of the last row of the episode containing each row (terminals > 0.5 end one;
    the final row always does). Same rule as utils.datasets.add_skill_targets."""
    terminals = np.asarray(terminals)
    n = terminals.shape[0]
    ends = np.nonzero(terminals > 0.5)[0]
    if ends.size == 0 or ends[-1] != n - 1:
        ends = np.concatenate([ends, [n - 1]])
    return ends[np.searchsorted(ends, np.arange(n), side="left")]


class SegmentSampler:
    """Uniform length-(H+1) segments (s_0..s_H, a_0..a_{H-1}) that never cross an episode end.

    Row i starts a segment iff rows i..i+H-1 share one episode; s_{t+1} is
    next_observations[i+t], so the terminal row's own successor is used as s_H and the
    segment needs H transitions, not H+1 observation rows.
    """

    def __init__(self, observations, actions, next_observations, terminals, horizon, seed=0):
        self.obs, self.act, self.next_obs = observations, actions, next_observations
        self.horizon = int(horizon)
        n = observations.shape[0]
        end = episode_ends(terminals)
        self.starts = np.nonzero(end >= np.arange(n) + self.horizon - 1)[0]
        assert self.starts.size > 0, f"no episode has {self.horizon} transitions"
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        i = self.rng.choice(self.starts, size=batch_size)          # (B,)
        idx = i[:, None] + np.arange(self.horizon)[None, :]        # (B, H)
        return {
            "s0": self.obs[i],                                        # (B, obs)
            "actions": self.act[idx],                                 # (B, H, act)
            "targets": self.next_obs[idx],                            # (B, H, obs): s_1..s_H
        }


# ------------------------------------------------------------------ loss
def rldp_loss(phi_def, pred_def, params, target_phi_params, batch, lam):
    """Returns (L_dyn + lam * L_orth, info)."""
    phi0 = phi_def.apply({"params": params["phi"]}, batch["s0"])       # (B, d)
    h = phi0
    tgt_all = phi_def.apply({"params": target_phi_params}, batch["targets"])  # (B, H, d)
    tgt_all = jax.lax.stop_gradient(tgt_all)
    dyn = 0.0
    per_step = []
    for t in range(batch["targets"].shape[1]):
        h = pred_def.apply({"params": params["pred"]}, h, batch["actions"][:, t])
        step_loss = jnp.mean(jnp.sum((h - tgt_all[:, t]) ** 2, axis=-1))
        per_step.append(step_loss)
        dyn = dyn + step_loss
    gram = phi0.T @ phi0 / phi0.shape[0]                                  # (d, d)
    orth = jnp.sum((gram - jnp.eye(gram.shape[0])) ** 2)
    total = dyn + lam * orth
    info = {"loss": total, "dyn_loss": dyn, "orth_loss": orth, "batch_gram_dev": jnp.sqrt(orth),
                "dyn_step1": per_step[0], "dyn_stepH": per_step[-1]}
    return total, info


def gram_deviation(phi_def, phi_params, observations):
    """|| E[phi phi^T] - I ||_F on a probe batch, and the same divided by sqrt(d)."""
    phi = phi_def.apply({"params": phi_params}, observations)
    d = phi.shape[-1]
    dev = float(jnp.linalg.norm(phi.T @ phi / phi.shape[0] - jnp.eye(d)))
    return dev, dev / float(np.sqrt(d))


# ------------------------------------------------------------------ state
def init_state(obs_dim, act_dim, z_dim, phi_hidden_dim, phi_hidden_layers,
               pred_hidden_dim, pred_hidden_layers, lr, seed):
    phi_def = PhiMap(z_dim=z_dim, hidden_dim=phi_hidden_dim, hidden_layers=phi_hidden_layers,
                     norm=True)
    pred_def = LatentPredictor(z_dim=z_dim, hidden_dim=pred_hidden_dim,
                               hidden_layers=pred_hidden_layers)
    k_phi, k_pred = jax.random.split(jax.random.PRNGKey(seed))
    params = {
        "phi": phi_def.init(k_phi, jnp.zeros((1, obs_dim)))["params"],
        "pred": pred_def.init(k_pred, jnp.zeros((1, z_dim)), jnp.zeros((1, act_dim)))["params"],
    }
    tx = optax.adam(lr)
    return {"phi_def": phi_def, "pred_def": pred_def, "params": params,
                "target_phi": jax.tree_util.tree_map(lambda x: x, params["phi"]),
                "opt_state": tx.init(params), "tx": tx, "step": 0}


def make_update(phi_def, pred_def, tx, lam, tau):
    @jax.jit
    def update(params, target_phi, opt_state, batch):
        (_, info), grads = jax.value_and_grad(
            lambda p: rldp_loss(phi_def, pred_def, p, target_phi, batch, lam), has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_phi = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p,
                                            target_phi, params["phi"])
        return params, target_phi, opt_state, info
    return update


def save_checkpoint(out_dir, step, params, target_phi):
    """`save_agent`'s layout: pickle["agent"]["phi"]["params"] is what `_load_phi_params` reads."""
    state = {"agent": {
        "phi": {"params": flax.serialization.to_state_dict(params["phi"])},
        "target_phi": flax.serialization.to_state_dict(target_phi),
        "predictor": {"params": flax.serialization.to_state_dict(params["pred"])},
    }, "step": int(step)}
    path = os.path.join(out_dir, f"params_{int(step)}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return path


# ------------------------------------------------------------------ main
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    obs, act, next_obs, term = load_transitions(args.dataset_npz, args.env_name)
    n_rows = obs.shape[0]
    sampler = SegmentSampler(obs, act, next_obs, term, args.horizon, seed=args.seed)
    state = init_state(obs.shape[1], act.shape[1], args.z_dim, args.phi_hidden_dim,
                       args.phi_hidden_layers, args.pred_hidden_dim, args.pred_hidden_layers,
                       args.lr, args.seed)
    update = make_update(state["phi_def"], state["pred_def"], state["tx"], args.lam, args.tau)
    probe = obs[np.random.default_rng(args.seed + 1).choice(n_rows, size=min(4096, n_rows),
                                                            replace=False)]

    flags = dict(vars(args), phi={"z_dim": args.z_dim, "hidden_dim": args.phi_hidden_dim,
                                      "hidden_layers": args.phi_hidden_layers, "norm": True},
                 predictor={"z_dim": args.z_dim, "hidden_dim": args.pred_hidden_dim,
                                "hidden_layers": args.pred_hidden_layers},
                 n_rows=int(n_rows), n_segment_starts=int(sampler.starts.size),
                 obs_dim=int(obs.shape[1]), act_dim=int(act.shape[1]))
    with open(os.path.join(args.out_dir, "flags.json"), "w") as f:
        json.dump(flags, f, indent=2)
    print("hyperparameters:")
    for k, v in flags.items():
        print(f"  {k:20s} {v}")

    params, target_phi, opt_state = state["params"], state["target_phi"], state["opt_state"]
    dev0, dev0n = gram_deviation(state["phi_def"], params["phi"], probe)
    curve = [{"step": 0, "loss": None, "dyn_loss": None, "orth_loss": None, "gram_dev": dev0,
                  "gram_dev_norm": dev0n}]
    print(f"step 0: gram_dev={dev0:.4f} (/sqrt(d) {dev0n:.4f})", flush=True)
    t0 = time.time()
    last_info = None
    for step in range(1, args.steps + 1):
        batch = {k: jnp.asarray(v) for k, v in sampler.sample(args.batch_size).items()}
        params, target_phi, opt_state, info = update(params, target_phi, opt_state, batch)
        if step % args.log_interval == 0 or step == args.steps or step == 1:
            info = {k: float(v) for k, v in info.items()}
            dev, devn = gram_deviation(state["phi_def"], params["phi"], probe)
            row = dict(step=step, gram_dev=dev, gram_dev_norm=devn, **info)
            curve.append(row)
            last_info = row
            rate = step / max(time.time() - t0, 1e-9)
            print(f"step {step}: loss={info['loss']:.4f} dyn={info['dyn_loss']:.4f} "
                  f"(t=1 {info['dyn_step1']:.4f}, t=H {info['dyn_stepH']:.4f}) "
                  f"orth={info['orth_loss']:.4f} gram_dev={dev:.4f} (/sqrt(d) {devn:.4f}) "
                  f"[{rate:.1f} it/s]", flush=True)
            assert np.isfinite(info["loss"]), f"non-finite loss at step {step}: {info}"
        if step % args.save_interval == 0 or step == args.steps:
            path = save_checkpoint(args.out_dir, step, params, target_phi)
            print(f"saved {path}", flush=True)
            with open(args.report_out, "w") as f:
                json.dump({"flags": flags, "curve": curve}, f, indent=1)
    with open(args.report_out, "w") as f:
        json.dump({"flags": flags, "curve": curve, "final": last_info}, f, indent=1)
    print(f"wrote {args.report_out}")
    return params, target_phi, curve


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset_npz", default=None,
                   help="preimage-augmented npz; the rows the stage-C run trains on")
    p.add_argument("--env_name", default=None,
                   help="OGBench env id, loaded via envs.env_utils.make_env_and_datasets")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--report_out", default=None, help="default <out_dir>/rldp_curve.json")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=5, help="H: predicted steps per segment")
    p.add_argument("--lam", type=float, default=1.0, help="weight on the orthonormality term")
    p.add_argument("--z_dim", type=int, default=PHI_KWARGS["z_dim"])
    p.add_argument("--phi_hidden_dim", type=int, default=PHI_KWARGS["hidden_dim"])
    p.add_argument("--phi_hidden_layers", type=int, default=PHI_KWARGS["hidden_layers"])
    p.add_argument("--pred_hidden_dim", type=int, default=256)
    p.add_argument("--pred_hidden_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.01, help="Polyak rate of the target phi")
    p.add_argument("--save_interval", type=int, default=250_000)
    p.add_argument("--log_interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.report_out is None:
        args.report_out = os.path.join(args.out_dir, "rldp_curve.json")
    return args


if __name__ == "__main__":
    train(parse_args())
