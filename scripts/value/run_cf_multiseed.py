#!/usr/bin/env python
"""scripts/value/run_cf_multiseed.py — drive the counterfactual value probe across all
10 training seeds of one method, writing per-seed tagged outputs.

  fb, rldp  -> torch probe (counterfactual_value_probe.py)      .venv
  crl, gciql -> jax probe  (counterfactual_value_probe_jax.py)  .venv-jax-cpu

Per-seed outputs land at analysis/value/repsep/cf_value_<method>_ms<NN>.json (+ parquet);
existing outputs are skipped so the run resumes. Usage:
    python scripts/value/run_cf_multiseed.py <fb|rldp|crl|gciql>
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
TORCH_PY = str(REPO / ".venv" / "bin" / "python")
JAX_PY = str(REPO / ".venv-jax-cpu" / "bin" / "python")


def seed_dirs(method):
    if method == "fb":
        return sorted(glob.glob(str(REPO / "results/Factored-FB-cube-run/*__fb_flowbc__s*")))
    if method == "rldp":
        return sorted(glob.glob(str(REPO / "results/factored-fb-rldp-flowbc/*__s*")))
    if method == "crl":
        return sorted(glob.glob(str(REPO / "results/factored-fb-crl-flowbc/sd*")))
    if method == "gciql":
        return sorted(glob.glob(str(
            REPO / "results/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd*")))
    raise SystemExit(f"unknown method {method}")


def torch_ckpt(seed_dir):
    ck = Path(seed_dir) / "checkpoints"
    if (ck / "final.pt").exists():
        return ck / "final.pt"
    steps = sorted(ck.glob("step_*.pt"),
                   key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {ck}")
    return steps[-1]


def build_cmd(method, seed_dir, tag):
    if method in ("fb", "rldp"):
        cfg = Path(seed_dir) / ".hydra" / "config.yaml"
        ck = torch_ckpt(seed_dir)
        return TORCH_PY, [
            TORCH_PY, "scripts/value/counterfactual_value_probe.py",
            "--config", str(cfg), "--checkpoint", str(ck),
            "--method", method, "--data-path", "datasets", "--out-tag", f"_ms{tag}"]
    if method == "crl":
        return JAX_PY, [
            JAX_PY, "scripts/value/counterfactual_value_probe_jax.py",
            "--method", "crl", "--checkpoint-dir", str(seed_dir), "--step", "1000000",
            "--data-path", "datasets", "--out-tag", f"_ms{tag}"]
    if method == "gciql":
        return JAX_PY, [
            JAX_PY, "scripts/value/counterfactual_value_probe_jax.py",
            "--method", "gciql", "--run-dir", str(seed_dir), "--step", "1000000",
            "--data-path", "datasets", "--out-tag", f"_ms{tag}"]


def main():
    method = sys.argv[1]
    dirs = seed_dirs(method)
    print(f"[{method}] {len(dirs)} seed dirs", flush=True)
    # phase_probe_crl.py does a bare `import rldp_shared` (lives in scripts/),
    # so scripts/ must be importable.
    pp = os.pathsep.join([str(REPO / "scripts"), str(REPO), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, MUJOCO_GL="glfw", PYTHONPATH=pp)
    ok = 0
    for i, d in enumerate(dirs):
        tag = f"{i:02d}"
        out = REPSEP / f"cf_value_{method}_ms{tag}.json"
        if out.exists():
            print(f"[{method}] seed {tag} exists, skip", flush=True); ok += 1; continue
        _py, cmd = build_cmd(method, d, tag)
        print(f"[{method}] seed {tag} <- {Path(d).name}", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO), env=env)
        if r.returncode == 0 and out.exists():
            ok += 1
        else:
            print(f"[{method}] seed {tag} FAILED (rc={r.returncode})", flush=True)
    print(f"[{method}] done: {ok}/{len(dirs)} seeds succeeded", flush=True)


if __name__ == "__main__":
    main()
