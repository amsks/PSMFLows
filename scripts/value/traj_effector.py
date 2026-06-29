"""Recover the end-effector position + phase along play trajectories (run in .venv).

The effector is FK-derived (mujoco site ur5e/robotiq/pinch), not in obs/physics, so
we replay each play qpos through the env (set_state + mj_forward) and read the site.
Model-independent — computed ONCE; joined onto FB/GCIQL value rows by (traj, t).
Uses the SAME stride/goal_offset as traj_value_profile so (traj, t) align.

Per (traj, t): effector xyz, cube xyz, grip, lift, phase region (reach/grasp/
transport), and d_eff_cube = ||effector - cube|| (the approach-phase target).

  MUJOCO_GL=glfw .venv/bin/python -m scripts.value.traj_effector \
    --data-path datasets --out analysis/value/traj_value/effector.parquet --n-traj 1000
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evals.training_value import region_labels
from evals.phase_probe import Thresholds

CUBE_SLICE = slice(14, 17)
GRIP_IDX, LIFT_IDX = 6, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-traj", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=5)
    args = ap.parse_args()

    import mujoco
    from envs.ogbench import create_ogbench_env
    env, _ = create_ogbench_env("cube-single-play-singletask-task1-v0", seed=0, obs_type="state")
    u = env.unwrapped
    _, _ = env.reset(options=dict(task_id=1))
    tb = int(getattr(u, "_target_block", 0) or 0)
    table_z = float(u.cur_task_info["init_xyzs"][tb][2])
    pinch = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_SITE, "ur5e/robotiq/pinch")
    nv = u.model.nv
    thr = Thresholds()

    files = sorted(glob.glob(str(Path(args.data_path) / args.domain / "buffer" / "episode_*.npz")))[:args.n_traj]
    rows = []
    for ti, f in enumerate(files):
        phys = np.asarray(np.load(f)["physics"], np.float64)
        T = len(phys); tg = T - 1 - args.goal_offset
        if tg < args.stride * 3:
            continue
        for t in np.arange(0, tg, args.stride):
            u.set_state(phys[t], np.zeros(nv))
            mujoco.mj_forward(u.model, u.data)
            eff = np.asarray(u.data.site_xpos[pinch]).copy()
            cube = phys[t, CUBE_SLICE]
            rows.append(dict(traj=ti, t=int(t),
                             eff_x=float(eff[0]), eff_y=float(eff[1]), eff_z=float(eff[2]),
                             cube_z=float(cube[2]),
                             d_eff_cube=float(np.linalg.norm(eff - cube)),
                             grip=float(phys[t, GRIP_IDX]), lift=float(phys[t, LIFT_IDX])))
    env.close()
    df = pd.DataFrame(rows)
    df["region"] = [str(r) for r in region_labels(
        np.clip(df["grip"].to_numpy() / 0.8, 0, 1),
        df["lift"].to_numpy() - table_z, thr)]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    vc = df["region"].value_counts()
    print(f"[traj_effector] {df['traj'].nunique()} trajs, {len(df)} steps -> {out}")
    print("  phase counts: " + ", ".join(f"{k}={int(v)}" for k, v in vc.items()))


if __name__ == "__main__":
    main()
