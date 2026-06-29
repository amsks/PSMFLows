"""Thin Hydra launcher for the vendored OGBench (JAX) GCIQL track.

This module is PURE ORCHESTRATION and MUST NOT import jax/flax/the OGBench
training code. It translates a Hydra config into OGBench's exact absl CLI
flags and runs `third_party/ogbench/impls/main.py` in the isolated .venv-jax
via subprocess. The two tracks never share a process or environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
OGBENCH_IMPLS = REPO_ROOT / "third_party" / "ogbench" / "impls"


def _resolve_agent_file(agent_file: str) -> str:
    """Make a repo-relative agent file absolute (so custom agents outside the
    vendored tree work as --agent from cwd=impls). Vendored 'agents/*.py' do not
    exist under REPO_ROOT, so they pass through unchanged."""
    candidate = REPO_ROOT / str(agent_file)
    return str(candidate.resolve()) if candidate.is_file() else str(agent_file)


def build_ogbench_argv(cfg: Mapping[str, Any]) -> list[str]:
    """Translate launcher config -> OGBench main.py absl argv (deterministic)."""
    argv = [
        f"--env_name={cfg['env_name']}",
        f"--agent={_resolve_agent_file(cfg['agent_file'])}",
        f"--seed={cfg['seed']}",
        f"--train_steps={cfg['train_steps']}",
        f"--log_interval={cfg['log_interval']}",
        f"--eval_interval={cfg['eval_interval']}",
        f"--save_interval={cfg['save_interval']}",
        f"--eval_episodes={cfg['eval_episodes']}",
        f"--eval_on_cpu={cfg['eval_on_cpu']}",
        f"--run_group={cfg['run_group']}",
        f"--save_dir={cfg['output_root']}",
    ]
    # Resume passthrough (only when set): vendored main.py restores the full
    # agent state (params + opt state) from params_<restore_epoch>.pkl.
    if cfg.get("restore_path"):
        argv.append(f"--restore_path={cfg['restore_path']}")
    if cfg.get("restore_epoch") is not None:
        argv.append(f"--restore_epoch={cfg['restore_epoch']}")
    overrides = cfg.get("agent_overrides") or {}
    for key in sorted(overrides):
        argv.append(f"--agent.{key}={overrides[key]}")
    return argv


def _preflight(cfg: Mapping[str, Any]) -> Path:
    """Fail fast with actionable messages if the JAX track isn't provisioned."""
    if not OGBENCH_IMPLS.is_dir():
        sys.exit(
            f"[run_gciql] vendored OGBench not found: {OGBENCH_IMPLS}\n"
            "Re-vendor it (see third_party/ogbench/PROVENANCE.md)."
        )
    jax_python = Path(str(cfg["jax_python"])).expanduser()
    if not jax_python.exists():
        sys.exit(
            f"[run_gciql] JAX interpreter not found: {jax_python}\n"
            "Mount the NVMe and run `make install-jax` first "
            "(see the spec's Prerequisites section)."
        )
    symlink = Path(str(cfg["ogbench_home_symlink"])).expanduser()
    if not symlink.exists():
        sys.exit(
            f"[run_gciql] {symlink} missing. Create it so OGBench writes "
            "datasets to the NVMe:\n"
            f"  ln -s /mnt/scratch1/ogbench_data {symlink}"
        )
    out_root = Path(str(cfg["output_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    return jax_python


def build_child_env(plain: Mapping[str, Any], base_env: Mapping[str, str]) -> dict:
    """Build the OGBench child env from the resolved config + a base environment.

    Pure (no os.environ access) so it is unit-testable. Mirrors the wandb-shim
    contract documented in this module's docstring and sets
    OGBENCH_DRQ_FRONTEND=1 when the config selects the DrQ visual encoder.
    """
    env = dict(base_env)
    env["MUJOCO_GL"] = "egl"
    cvd = base_env.get("CUDA_VISIBLE_DEVICES", "0")
    env["CUDA_VISIBLE_DEVICES"] = cvd
    # CUDA_VISIBLE_DEVICES masks CUDA/JAX but NOT MuJoCo's EGL device
    # enumeration. With MUJOCO_EGL_DEVICE_ID unset, every process defaults to
    # EGL device 0, so N concurrent runs pile all offscreen render contexts
    # (OGBench's eval renders the goal image on every episode reset) onto
    # GPU 0 -> "Offscreen framebuffer is not complete" (GL 0x8cdd) for most.
    # Pin EGL to the same physical GPU JAX uses so contexts spread out.
    egl_dev = cvd.split(",")[0].strip() or "0"
    env.setdefault("MUJOCO_EGL_DEVICE_ID", egl_dev)
    # OGBench's vendored setup_wandb hardcodes wandb.init(mode='online'), which
    # (wandb >= 0.27) overrides WANDB_MODE/WANDB_DISABLED env vars. Since the
    # vendored code must not be edited, force the mode via an auto-imported
    # sitecustomize shim on PYTHONPATH (no-op unless the flag below is set).
    # The wandb shim (tools/wandb_mode_shim/sitecustomize.py) is ALWAYS on the
    # child PYTHONPATH; it is inert unless its env flags are set. GCIQL_FB_ENV
    # enables mirroring GCIQL eval metrics into FB's wandb key scheme (so FB and
    # GCIQL overlay on the same panels). GCIQL_WANDB_FORCE_MODE forces wandb
    # offline/disabled (OGBench hardcodes mode='online'; wandb >= 0.27 lets that
    # kwarg override the env vars). Vendored OGBench is never edited.
    # Put the shim dir FIRST (so Python imports OUR sitecustomize) and the
    # OGBench impls dir SECOND. The impls dir is required on PYTHONPATH -- not
    # just via main.py's cwd -- because sitecustomize is imported at interpreter
    # startup, BEFORE the script directory is added to sys.path; without it the
    # shim's `import utils.encoders` fails and encoder='drq' is never registered.
    shim_dir = str(REPO_ROOT / "tools" / "wandb_mode_shim")
    impls_dir = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    existing = base_env.get("PYTHONPATH")
    parts = [shim_dir, impls_dir] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["GCIQL_FB_ENV"] = str(plain["env_name"])
    # Repo-wide default: all GCIQL runs log to FB's wandb entity/project so
    # they sit alongside FB runs for comparison. Only run_group/tags vary.
    env["GCIQL_WANDB_ENTITY"] = str(plain.get("wandb_entity", "amsks"))
    env["GCIQL_WANDB_PROJECT"] = str(plain.get("wandb_project", "factored-fb"))
    mode = str(plain.get("wandb_mode", "online"))
    # don't inherit a stale GCIQL_WANDB_FORCE_MODE from the parent shell:
    # only this launcher should set it, and only for offline/disabled.
    env.pop("GCIQL_WANDB_FORCE_MODE", None)
    if mode in ("offline", "disabled"):
        env["GCIQL_WANDB_FORCE_MODE"] = mode
        # belt-and-suspenders; "disabled" is not a valid WANDB_MODE value
        env["WANDB_MODE"] = "offline" if mode == "disabled" else mode
    # DrQ visual front-end: the sitecustomize shim swaps GCDataset.augment for
    # FB's random_shifts only when this flag is set; gate it on the encoder so
    # impala runs keep OGBench's native random_crop aug untouched.
    overrides = plain.get("agent_overrides") or {}
    if str(overrides.get("encoder")) == "drq":
        env["OGBENCH_DRQ_FRONTEND"] = "1"
    # Resume: the sitecustomize shim reads these to reproduce a single-go layout
    # (vendored OGBench is never edited). GCIQL_EXP_NAME pins the run dir / wandb
    # name; GCIQL_STEP_OFFSET shifts saved ckpt epochs and wandb/CSV steps by the
    # restore epoch; GCIQL_CSV_APPEND continues train.csv/eval.csv. See
    # tools/wandb_mode_shim/sitecustomize.py.
    if plain.get("exp_name"):
        env["GCIQL_EXP_NAME"] = str(plain["exp_name"])
    if plain.get("restore_epoch"):
        env["GCIQL_STEP_OFFSET"] = str(int(plain["restore_epoch"]))
        env["GCIQL_CSV_APPEND"] = "1"
    return env


@hydra.main(version_base="1.3", config_path="configs/gciql",
            config_name="cube_single_state")
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    argv = build_ogbench_argv(plain)
    jax_python = _preflight(plain)

    env = build_child_env(plain, os.environ.copy())

    run_meta = {
        "argv": argv,
        # best-effort provenance; empty string if git is unavailable
        "factored_fb_sha": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "ogbench_pinned": "1d4140997f60c52c6fb0702ec100dc988b18c548",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(str(plain["output_root"])) / "last_run_gciql.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"[run_gciql] {jax_python} main.py {' '.join(argv)}")

    proc = subprocess.run(
        [str(jax_python), "main.py", *argv],
        cwd=str(OGBENCH_IMPLS), env=env,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
