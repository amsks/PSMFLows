"""scripts/train/run_td_jepa_50k.py — Run td_jepa's OFFICIAL training for 50k steps
on the SAME data and SAME hyperparameters as scripts/dev/verify_ortho_fix.py,
so we can compare side-by-side whether collapse is a bug in our port or
present in td_jepa itself.

Hyperparameters (mirror our verify_ortho_fix.py + td_jepa's
launch_fb_ogbench.py BASE_CFG + sweep_antmaze first trial):

    agent           = FBFlowBCAgent  (noise-conditioned actor + actor_vf)
    bc_coeff        = 0.3            (sweep_antmaze)
    ortho_coef      = 100.0          (sweep_antmaze first value)
    lr_b            = 1.0e-4         (sweep_antmaze first value)
    lr_f, lr_actor  = 1.0e-4
    clip_grad_norm  = 1.0            (matches our verify_ortho_fix; td_jepa default 0)
    f_target_tau    = 0.005
    b_target_tau    = 0.005
    batch_size      = 256
    discount        = 0.99
    num_train_steps = 50_000         (override; td_jepa default 1M)
    log_every       = 2_500          (override)

Reads data from <repo>/datasets/<domain>/buffer/*.npz — same files our
training uses.

Logs to wandb project `factored-fb`, group `td-jepa-50k`.

Must be run with td_jepa's uv environment (it needs safetensors, tyro, etc.):

    uv run --directory /home/mclovin/git/Austin/td_jepa \\
        python /home/mclovin/git/Austin/Factored-FB/scripts/train/run_td_jepa_50k.py

Override the td_jepa path via env var if needed:

    TD_JEPA_ROOT=/path/to/td_jepa uv run --directory /path/to/td_jepa \\
        python scripts/train/run_td_jepa_50k.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

OUR_REPO = Path(__file__).resolve().parent.parent.parent
TD_JEPA = Path(os.environ.get("TD_JEPA_ROOT", "/home/mclovin/git/Austin/td_jepa"))

if not TD_JEPA.exists():
    sys.exit(f"[run_td_jepa_50k] td_jepa not found at {TD_JEPA}. Set TD_JEPA_ROOT.")

# td_jepa's modules use absolute `from train import TrainConfig`, so its repo
# root must be importable and we need to run with cwd=td_jepa or have train.py
# on the path.
sys.path.insert(0, str(TD_JEPA))

# Env flags td_jepa's train.py sets at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch

torch.set_float32_matmul_precision("high")

from train import TrainConfig                          # noqa: E402
from metamotivo.envs.ogbench import ALL_TASKS          # noqa: E402


DOMAIN = "antmaze-medium-navigate-v0"


CONFIG: dict = {
    # ── Training loop ─────────────────────────────────────────────────── #
    "num_train_steps": 50_000,
    "log_every_updates": 2_500,
    "checkpoint_every_steps": 60_000,     # skip mid-run checkpointing
    "eval_every_steps": 60_000,           # skip mid-run eval (matches our verifier)
    "seed": 1,
    "work_dir": str(OUR_REPO / "outputs" / "td_jepa_50k"),

    # ── wandb ─────────────────────────────────────────────────────────── #
    "use_wandb": True,
    "wandb_pname": "factored-fb",
    "wandb_gname": "td-jepa-50k",
    "wandb_ename": "amsks",

    # ── Data ─────────────────────────────────────────────────────────── #
    "data": {
        "name": "ogbench",
        "domain": DOMAIN,
        "obs_type": "state",
        "dataset_root": str(OUR_REPO / "datasets"),
        "load_n_episodes": 1_000,
    },

    # ── Env ──────────────────────────────────────────────────────────── #
    "env": {
        "name": "ogbench",
        "obs_type": "state",
        "domain": DOMAIN,
        "task": ALL_TASKS[DOMAIN][0],
    },

    # ── Agent ────────────────────────────────────────────────────────── #
    "agent": {
        "name": "FBFlowBCAgent",
        "compile": False,                  # match our verifier (eager)
        "cudagraphs": False,
        "model": {
            "device": "cuda",
            "obs_normalizer": {"name": "IdentityNormalizerConfig"},
            "archi": {
                "f": {
                    "name": "ForwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 2,
                },
                "actor": {
                    "name": "noise_conditioned_actor",
                    "hidden_dim": 512,
                    "hidden_layers": 2,
                },
                "actor_vf": {
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                },
                "b": {
                    "name": "BackwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                    "norm": True,
                },
                "left_encoder": {
                    "name": "BackwardArchi",
                    "hidden_dim": 512,
                    "hidden_layers": 4,
                    "norm": True,
                },
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

    # ── Eval (skipped; we only care about train metrics for now) ─────── #
    "evaluations": [],
}


def main() -> None:
    Path(CONFIG["work_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"[run_td_jepa_50k] data: {CONFIG['data']['dataset_root']}/{DOMAIN}/buffer")
    print(f"[run_td_jepa_50k] workdir: {CONFIG['work_dir']}")
    print(f"[run_td_jepa_50k] wandb group: {CONFIG['wandb_gname']}")

    cfg = TrainConfig(**CONFIG)
    workspace = cfg.build()
    workspace.train()
    print("[run_td_jepa_50k] DONE")


if __name__ == "__main__":
    main()
