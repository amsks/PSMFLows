"""scripts/eval/training_data_failure_modes.py — apply the 4-phase classification
from eval to the cube-single-play training trajectories, with hindsight goal
= each trajectory's own final cube position.

Under this hindsight relabelling, "transport to the goal" is trivially passed
(by construction the goal IS the final cube position), so we drop that phase
from the analysis. What's left is a coverage question — does the training
data contain examples of each of the three remaining phases:

  - approach: did the gripper ever come within 6 cm of the cube?
  - grasp:    did the cube ever come off the table with the gripper closed
              for >= 5 consecutive steps?
  - maintain: is the cube still held at the final step (cube z > 3 cm)?

Operates on state obs only. Uses obs[12:15] (effector), obs[17:18] (gripper
signals) and obs[19:22] (scaled cube xyz) from the saved episode_*.npz files.

  .venv/bin/python -m scripts.eval.training_data_failure_modes
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# obs layout from cube_env.compute_observation (state, 1 cube):
#   [0:6]   joint_pos
#   [6:12]  joint_vel
#   [12:15] effector_pos (×10 from [0.425,0,0])
#   [15:17] cos/sin(effector_yaw)
#   [17]    gripper_opening ×3   (in [0, 3])
#   [18]    gripper_contact      (in [0, 1])
#   [19:22] block_0_pos ×10 from [0.425,0,0]
#   [22:26] block_0_quat
#   [26:28] cos/sin(block_yaw)
XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0
GRIPPER_SCALER = 3.0
TABLE_Z = 0.02
DELTA_LIFT = 0.03   # cube above table by this -> lifted
TAU_CONTACT = 0.5   # gripper_contact above this -> closed on something
EPS_REACH = 0.06    # effector close to cube by this -> approached


def _unpack(obs: np.ndarray):
    eff = obs[:, 12:15] / XYZ_SCALER + XYZ_CENTER
    cube = obs[:, 19:22] / XYZ_SCALER + XYZ_CENTER
    grip_open = obs[:, 17] / GRIPPER_SCALER     # [0, 1]; high = closed
    contact = obs[:, 18]                         # [0, 1]
    return eff, cube, grip_open, contact


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] for contiguous True runs."""
    out, n = [], len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


K_STEPS = 5            # consecutive held-steps required to count as a pickup
EPS_GOAL = 0.05        # released within this xy of trajectory-final = "at goal"


def _max_true_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _classify_segment(post_lifted: bool, post_closed: bool,
                      release_xy: np.ndarray, goal_xy: np.ndarray) -> str:
    """Classify the release event at step b (FIRST non-held step).

    The held condition (lifted AND closed) ended at step b. Diagnose what
    broke it:
      - cube no longer lifted (post_lifted=False) => cube reached the table
        while the gripper was still closed => CONTROLLED placement.
      - cube still lifted (post_lifted=True) but gripper opened (post_closed
        =False) => gripper released while cube was still in the air => DROP.
    """
    if post_lifted and not post_closed:
        return "dropped"
    # placement (controlled). Sort into at-goal vs not-at-goal by xy.
    dist = float(np.linalg.norm(release_xy - goal_xy))
    return "placed_at_goal" if dist <= EPS_GOAL else "placed_not_at_goal"


def analyse_episode(obs: np.ndarray) -> tuple[dict, list[dict]]:
    """Per-trajectory analysis + per-held-segment classification.

    Hindsight goal = trajectory's own final cube xy.
    For each held-segment >= K_STEPS, the release event is bucketed into
    {dropped, placed_not_at_goal, placed_at_goal} based on what broke the
    held condition (gripper opening vs cube reaching the table).
    """
    eff, cube, _, contact = _unpack(obs)
    lift = cube[:, 2] - TABLE_Z
    lifted = lift > DELTA_LIFT
    closed = contact > TAU_CONTACT
    held = lifted & closed
    eff_cube_dist = np.linalg.norm(eff - cube, axis=1)
    goal_xy = cube[-1, :2]
    T = len(obs)

    segs = []
    for (a, b) in _runs(held):
        if b - a < K_STEPS:
            continue
        # Release-side state. If b == T the trajectory ended while still
        # holding the cube (rare); treat as "placed_at_goal" because the
        # cube ended at its own final position with gripper still closed.
        if b >= T:
            post_lifted, post_closed = True, True
            release_xy = cube[b - 1, :2]
        else:
            post_lifted = bool(lifted[b])
            post_closed = bool(closed[b])
            release_xy = cube[b, :2]   # cube xy when held condition first broke
        sxy = cube[a, :2]
        bucket = _classify_segment(post_lifted, post_closed, release_xy, goal_xy)
        segs.append(dict(
            seg_start=int(a), seg_end=int(b), length=int(b - a),
            start_xy=tuple(map(float, sxy)),
            release_xy=tuple(map(float, release_xy)),
            release_z=float(cube[min(b, T-1), 2]),
            release_lift=float(cube[min(b, T-1), 2] - TABLE_Z),
            post_lifted=post_lifted, post_closed=post_closed,
            xy_displacement=float(np.linalg.norm(release_xy - sxy)),
            dist_to_goal=float(np.linalg.norm(release_xy - goal_xy)),
            bucket=bucket,
        ))

    n_drop = sum(s["bucket"] == "dropped" for s in segs)
    n_placed_not = sum(s["bucket"] == "placed_not_at_goal" for s in segs)
    n_placed_at = sum(s["bucket"] == "placed_at_goal" for s in segs)

    return dict(
        T=int(len(obs)),
        approached=bool(eff_cube_dist.min() < EPS_REACH),
        grasped=bool(_max_true_run(held) >= K_STEPS),
        n_segments=len(segs),
        n_dropped=int(n_drop),
        n_placed_not_at_goal=int(n_placed_not),
        n_placed_at_goal=int(n_placed_at),
        has_drop=bool(n_drop > 0),
        has_placed_not_at_goal=bool(n_placed_not > 0),
        has_placed_at_goal=bool(n_placed_at > 0),
        goal_xy_x=float(goal_xy[0]),
        goal_xy_y=float(goal_xy[1]),
        held_steps=int(held.sum()),
        max_held_run=int(_max_true_run(held)),
    ), segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="datasets/cube-single-play-v0/buffer")
    ap.add_argument("--out", default="analysis/value/training_data_failure_modes")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="0 = all available episodes")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(args.root) / "episode_*.npz")))
    if args.max_episodes > 0:
        files = files[:args.max_episodes]
    rows, all_segs = [], []
    for i, f in enumerate(files):
        with np.load(f) as z:
            obs = np.asarray(z["observation"], np.float32)
        r, segs = analyse_episode(obs)
        r["episode"] = i
        rows.append(r)
        for s in segs:
            s["episode"] = i
            all_segs.append(s)
    df = pd.DataFrame(rows)
    seg = pd.DataFrame(all_segs)
    df.to_parquet(out / "per_episode.parquet")
    seg.to_parquet(out / "per_segment.parquet")

    n_ep = len(df)
    n_seg = len(seg)
    n_drop_seg = int((seg["bucket"] == "dropped").sum())
    n_placed_not_seg = int((seg["bucket"] == "placed_not_at_goal").sum())
    n_placed_at_seg = int((seg["bucket"] == "placed_at_goal").sum())

    n_drop_traj = int(df["has_drop"].sum())
    n_placed_not_traj = int(df["has_placed_not_at_goal"].sum())
    n_placed_at_traj = int(df["has_placed_at_goal"].sum())

    summary = {
        "n_episodes": n_ep,
        "n_held_segments": n_seg,
        "thresholds": {
            "delta_lift_m": DELTA_LIFT, "eps_goal_m": EPS_GOAL,
            "eps_reach_m": EPS_REACH, "tau_contact": TAU_CONTACT,
            "k_steps": K_STEPS,
        },
        "segments_by_bucket_pct": {
            "dropped":            100.0 * n_drop_seg / max(n_seg, 1),
            "placed_not_at_goal": 100.0 * n_placed_not_seg / max(n_seg, 1),
            "placed_at_goal":     100.0 * n_placed_at_seg / max(n_seg, 1),
        },
        "trajectories_with_at_least_one_pct": {
            "dropped":            100.0 * n_drop_traj / n_ep,
            "placed_not_at_goal": 100.0 * n_placed_not_traj / n_ep,
            "placed_at_goal":     100.0 * n_placed_at_traj / n_ep,
        },
        "segments_per_trajectory": {
            "mean": float(df["n_segments"].mean()),
            "median": float(df["n_segments"].median()),
            "max": int(df["n_segments"].max()),
            "min": int(df["n_segments"].min()),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    def q(s, x):
        return float(np.quantile(s, x)) if len(s) else float("nan")
    placed = seg[seg["bucket"] == "placed_at_goal"]["xy_displacement"]
    placed_not = seg[seg["bucket"] == "placed_not_at_goal"]["xy_displacement"]
    dropped = seg[seg["bucket"] == "dropped"]["release_lift"]

    lines = [
        "# Cube-single-play training data — pickup-event outcomes (hindsight goal)",
        "",
        f"- trajectories scanned: **{n_ep}**",
        f"- held-segments (>= {K_STEPS} steps each): **{n_seg}**  "
        f"(mean {df['n_segments'].mean():.2f} per traj, max {int(df['n_segments'].max())})",
        f"- thresholds: delta_lift={DELTA_LIFT} m, eps_goal={EPS_GOAL} m, "
        f"tau_contact={TAU_CONTACT}, k_steps={K_STEPS}",
        "- Each trajectory's hindsight goal xy = the cube xy at the FINAL step.",
        "",
        "## Pickup-event outcomes (% of all held-segments)",
        "",
        "| bucket | count | % |",
        "| :--- | ---: | ---: |",
        f"| **dropped** (cube z > 3 cm at release — fell uncontrolled) | {n_drop_seg} | {100*n_drop_seg/max(n_seg,1):.1f} |",
        f"| **placed but not at goal** (released on table, >5 cm from goal xy) | {n_placed_not_seg} | {100*n_placed_not_seg/max(n_seg,1):.1f} |",
        f"| **placed at goal** (released on table, within 5 cm of goal xy) | {n_placed_at_seg} | {100*n_placed_at_seg/max(n_seg,1):.1f} |",
        "",
        "## Per-trajectory: did the trajectory contain at least one of each?",
        "",
        "| bucket | trajectories | % |",
        "| :--- | ---: | ---: |",
        f"| >=1 dropped            | {n_drop_traj} | {100*n_drop_traj/n_ep:.1f} |",
        f"| >=1 placed-not-at-goal | {n_placed_not_traj} | {100*n_placed_not_traj/n_ep:.1f} |",
        f"| >=1 placed-at-goal     | {n_placed_at_traj} | {100*n_placed_at_traj/n_ep:.1f} |",
        "",
        "## Transport distance for placement events (xy_displacement at release, m)",
        "",
        "| quantile | placed at goal | placed not at goal |",
        "| :--- | ---: | ---: |",
    ]
    for x in [0.10, 0.25, 0.50, 0.75, 0.90]:
        lines.append(f"| {int(x*100):2d}% | {q(placed, x):.3f} | {q(placed_not, x):.3f} |")
    lines += [
        f"| max  | {placed.max() if len(placed) else float('nan'):.3f} | "
        f"{placed_not.max() if len(placed_not) else float('nan'):.3f} |",
        "",
        "## Release-height for dropped segments (release_lift = z above table, m)",
        "",
        "| quantile | release_lift |",
        "| :--- | ---: |",
    ]
    for x in [0.10, 0.25, 0.50, 0.75, 0.90]:
        lines.append(f"| {int(x*100):2d}% | {q(dropped, x):.3f} |")
    lines += [
        f"| max | {dropped.max() if len(dropped) else float('nan'):.3f} |",
        "",
        "## Read-off",
        "",
        f"- {100*n_drop_seg/max(n_seg,1):.1f}% of pickup events are drops "
        f"(released while cube was still in the air).",
        f"- {100*n_placed_at_seg/max(n_seg,1):.1f}% are clean placements at goal (within "
        f"{int(EPS_GOAL*100)} cm of the trajectory's final cube position).",
        f"- {100*n_placed_not_seg/max(n_seg,1):.1f}% are intermediate placements "
        "(controlled release on the table, but somewhere other than where the cube "
        "ends up at the trajectory end).",
        f"- {100*n_placed_at_traj/n_ep:.1f}% of trajectories contain at least one "
        f"placed-at-goal example — this is the coverage of "
        "the transport-and-place behaviour the eval-time policy needs.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"[done] -> {out}/summary.md, per_episode.parquet, per_segment.parquet")


if __name__ == "__main__":
    main()
