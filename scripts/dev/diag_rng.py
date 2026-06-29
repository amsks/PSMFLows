"""scripts/dev/diag_rng.py — RNG-state fingerprint diagnostic for our codebase.

Goal: localize WHERE the RNG state and/or computation diverges from td_jepa.

Two-phase diagnostic:
  Phase 1 (init): fingerprint torch RNG state after set_seed, env build,
    data load, and agent build. Also hash the first 4 model parameters.
    Compare against scripts/dev/diag_rng_tdjepa.py's Phase 1 to find any RNG-shifting
    step in the pipeline setup.

  Phase 2 (train): run 5 update steps exactly as train.py does (sample +
    agent.update), fingerprint RNG state after each, and hash a few key
    metric values. Compare against td_jepa's Phase 2 to see whether step-
    level computation diverges (gradients, intermediate tensors).

Output format mirrors scripts/dev/diag_rng_tdjepa.py line-for-line, so a plain
`diff -u ours.diag tdjepa.diag` localizes the first point of divergence.

Run with: python scripts/dev/diag_rng.py > ours.diag
(uses antmaze-medium + fb_flowbc, seed=1, matching the 50k comparison runs)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from train import make_agent, set_seed, resolve_device  # noqa: E402
from envs.ogbench import create_ogbench_env  # noqa: E402
from data.ogbench import load_ogbench_dataset  # noqa: E402


def fp(label: str) -> None:
    """Print SHA1 fingerprint of current torch CPU+CUDA RNG state."""
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
    print("Factored-FB RNG diagnostic")
    print("=" * 72)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "domain=antmaze_medium",
                "seed=1",
                "use_wandb=false",
                "num_train_steps=5",
                "log_every=1",
                # Match diag_rng_tdjepa.py CONFIG exactly
                "ortho_coef=100.0",
                "clip_grad_norm=1.0",
            ],
        )
    cfg.device = resolve_device(cfg.device)

    # ─── Phase 1: Init ────────────────────────────────────────────────────
    print("\n--- Phase 1: init ---")
    fp("start (pre-seed)")

    set_seed(cfg.seed)
    fp("after set_seed(1)")

    torch.set_float32_matmul_precision("high")
    fp("after TF32-high")

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    fp("after create_ogbench_env")
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]

    buffer = load_ogbench_dataset(
        domain=cfg.domain,
        data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes,
        device=cfg.device,
        n_transitions=cfg.n_transitions,
    )
    fp("after load_ogbench_dataset")
    print(f"  [info] buffer_size={len(buffer):,}")

    agent = make_agent(cfg, obs_space, action_dim)
    fp("after make_agent (weight_init done)")

    for name, p in list(agent.model.named_parameters())[:4]:
        print(f"  [param] {name:42s}  sum={p.detach().sum().item():+.6f}  hash={hash_tensor(p)}")

    # ─── Phase 2: Train (5 steps, mirroring train.py main loop) ───────────
    print("\n--- Phase 2: train (5 steps) ---")

    for step in range(1, 6):
        batch = buffer.sample(cfg.batch_size)
        # fingerprint right after sampling (captures the buffer.sample RNG draw)
        fp(f"step {step}: after buffer.sample")

        metrics = agent.update(batch, step)
        fp(f"step {step}: after agent.update")

        # Print key metrics that we know diverge between runs (M1, fb_offdiag)
        key_metrics = [
            "M1", "B_norm", "z_norm",
            "fb_offdiag", "fb_diag", "orth_loss",
            "q", "actor_loss",
        ]
        for k in key_metrics:
            if k in metrics:
                print(f"  [metric] step {step:>2d} {k:30s}  {float(metrics[k]):+.6e}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
