"""scripts/dev/diag_rng_tdjepa.py — RNG-state fingerprint diagnostic for td_jepa.

Mirror of scripts/dev/diag_rng.py, driving td_jepa's pipeline.

Two-phase diagnostic:
  Phase 1 (init): fingerprint torch RNG state after env.build, set_seed,
    data.build, and agent build. Hash the first 4 model parameters.

  Phase 2 (train): run 5 update steps exactly as td_jepa's
    Workspace.train_offline does (agent.update(replay_buffer, t)),
    fingerprint RNG state after each, hash key metrics.

Output is line-for-line comparable with scripts/dev/diag_rng.py.

Run with: python scripts/dev/diag_rng_tdjepa.py > tdjepa.diag
Then:    diff -u ours.diag tdjepa.diag
The first mismatching line localizes the divergence point.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

OUR_REPO = Path(__file__).resolve().parents[2]
TD_JEPA = Path(os.environ.get("TD_JEPA_ROOT", "/home/mclovin/git/Austin/td_jepa"))

if not TD_JEPA.exists():
    sys.exit(f"[diag_rng_tdjepa] td_jepa not found at {TD_JEPA}. Set TD_JEPA_ROOT.")

sys.path.insert(0, str(TD_JEPA))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

torch.set_float32_matmul_precision("high")

from train import TrainConfig                          # noqa: E402
from metamotivo.envs.ogbench import ALL_TASKS          # noqa: E402
from metamotivo.utils import set_seed_everywhere       # noqa: E402


DOMAIN = "antmaze-medium-navigate-v0"

# Identical to scripts/train/run_td_jepa_50k.py CONFIG (the 50k run that produced
# the wandb data we're comparing against), with eval/wandb stripped for
# diagnostic purity.
CONFIG: dict = {
    "num_train_steps": 5,
    "log_every_updates": 1,
    "checkpoint_every_steps": 60_000,
    "eval_every_steps": 60_000,
    "seed": 1,
    "work_dir": str(OUR_REPO / "outputs" / "diag_td_jepa"),
    "use_wandb": False,
    "data": {
        "name": "ogbench",
        "domain": DOMAIN,
        "obs_type": "state",
        "dataset_root": str(OUR_REPO / "datasets"),
        "load_n_episodes": 1_000,
    },
    "env": {
        "name": "ogbench",
        "obs_type": "state",
        "domain": DOMAIN,
        "task": ALL_TASKS[DOMAIN][0],
    },
    "agent": {
        "name": "FBFlowBCAgent",
        "compile": False,
        "cudagraphs": False,
        "model": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "obs_normalizer": {"name": "IdentityNormalizerConfig"},
            "archi": {
                "f": {"name": "ForwardArchi", "hidden_dim": 512, "hidden_layers": 2},
                "actor": {"name": "noise_conditioned_actor", "hidden_dim": 512, "hidden_layers": 2},
                "actor_vf": {"hidden_dim": 512, "hidden_layers": 4},
                "b": {"name": "BackwardArchi", "hidden_dim": 512, "hidden_layers": 4, "norm": True},
                "left_encoder": {"name": "BackwardArchi", "hidden_dim": 512, "hidden_layers": 4, "norm": True},
                "L_dim": 50,
                "z_dim": 50,
                "norm_z": True,
            },
            "actor_encode_obs": False,
        },
        "train": {
            "batch_size": 256,
            "discount": 0.99,
            "ortho_coef": 100.0,
            "lr_b": 1.0e-4,
            "lr_f": 1.0e-4,
            "lr_actor": 1.0e-4,
            "lr_actor_vf": 3.0e-4,
            "f_target_tau": 0.005,
            "b_target_tau": 0.005,
            "bc_coeff": 0.3,
            "clip_grad_norm": 1.0,
            "flow_steps": 10,
        },
    },
    "evaluations": [],
}


def fp(label: str) -> None:
    cpu_state = torch.get_rng_state().numpy().tobytes()
    cpu_digest = hashlib.sha1(cpu_state).hexdigest()[:16]
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state().cpu().numpy().tobytes()
        cuda_digest = hashlib.sha1(cuda_state).hexdigest()[:16]
    else:
        cuda_digest = "n/a             "
    print(f"  [rng] {label:42s}  cpu={cpu_digest}  cuda={cuda_digest}")


def hash_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy().tobytes()
    return hashlib.sha1(arr).hexdigest()[:16]


def main():
    print("=" * 72)
    print("td_jepa RNG diagnostic")
    print("=" * 72)

    Path(CONFIG["work_dir"]).mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(**CONFIG)

    # ─── Phase 1: Init (mirrors Workspace.__init__ ordering) ──────────────
    print("\n--- Phase 1: init ---")
    fp("start (pre-seed)")

    sample_env, _ = cfg.env.build()
    fp("after env.build (pre-seed)")
    obs_space = sample_env.observation_space
    action_dim = sample_env.action_space.shape[0]

    set_seed_everywhere(cfg.seed)
    fp("after set_seed_everywhere(1)")

    agent = cfg.agent.build(obs_space=obs_space, action_dim=action_dim)
    fp("after agent.build (weight_init done)")

    for name, p in list(agent._model.named_parameters())[:4]:
        print(f"  [param] {name:42s}  sum={p.detach().sum().item():+.6f}  hash={hash_tensor(p)}")

    # Evaluations build is lazy (RNG-free); skipped here.
    # Data build (DictBuffer):
    buffer_device = agent.device
    relabel_fn = None
    batch_size = cfg.agent.train.batch_size
    replay_buffer = cfg.data.build(buffer_device, batch_size, cfg.env.frame_stack, relabel_fn)
    fp("after data.build (DictBuffer ready)")
    print(f"  [info] buffer_size={len(replay_buffer['train']):,}")

    # ─── Phase 2: Train (5 steps, mirroring train_offline loop) ───────────
    print("\n--- Phase 2: train (5 steps) ---")
    agent._model.train()

    for step in range(1, 6):
        # td_jepa's agent.update internally calls replay_buffer["train"].sample.
        # To match scripts/dev/diag_rng.py's fingerprint AFTER sample, we replicate
        # the same sequence: sample first, then "the rest of update".
        # But we cannot break into the middle of agent.update without forking
        # the method, so we just fingerprint AFTER each full update step.
        metrics = agent.update(replay_buffer, step)
        fp(f"step {step}: after agent.update")

        # Print key metrics (td_jepa key names differ slightly from ours)
        # ours -> tdjepa:  loss/fb_offdiag -> fb_offdiag, train/M1 -> M1, etc.
        key_metrics = [
            "M1", "B_norm", "z_norm",
            "fb_offdiag", "fb_diag", "orth_loss",
            "q", "actor_loss",
        ]
        for k in key_metrics:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, torch.Tensor):
                    v = float(v.mean().item())
                print(f"  [metric] step {step:>2d} {k:30s}  {float(v):+.6e}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
