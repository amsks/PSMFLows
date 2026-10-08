"""Row mask that removes every direct start->goal stretch of the five cube tasks (2026-10-08).

Stitching test for cube-single-play. The play data holds, for each eval task, trajectories
that carry the cube from the task's start position to its goal position in one go, so a
score on the full data does not show that the agent combines pieces of different
trajectories. This tool writes a row mask for `dataset.drop_rows_path` (main.py):

  for every trajectory and task, at each first goal-visit y (cube within `r_goal` of the
  goal) that follows a start-visit (cube within `r_start` of the start), let x be the last
  start-visit before y; rows x+1 .. y-1 are dropped.

After the drop no remaining contiguous piece of a trajectory moves the cube from a start
region to the same task's goal region (checked below). Start and goal states stay in the
data. main.py marks the last kept row before each cut as a trajectory end, so hindsight
goals never span a cut. Cube xyz is qpos[:, 14:17] (the observation slice is normalised).
The mask is computed on the raw rows and written in the training set's transition row order.

`--control` writes the matched control instead: the same number of stretches with the same
lengths, dropped at random trajectories and offsets, so the two arms differ in which rows are
removed and not in how many.

Run (CPU):
  .venv/bin/python tools/make_stitch_mask.py --out $PSM_DATA/masks/cube_nostitch.npz
  .venv/bin/python tools/make_stitch_mask.py --control --out $PSM_DATA/masks/cube_randdrop.npz
"""
import argparse
import json
import os

import numpy as np

CUBE_TASKS = {  # ogbench manipspace/envs/cube_env.py, env_type=single
    1: ((0.425, 0.1, 0.02), (0.425, -0.1, 0.02)),
    2: ((0.35, 0.0, 0.02), (0.50, 0.0, 0.02)),
    3: ((0.50, 0.0, 0.02), (0.35, 0.0, 0.02)),
    4: ((0.35, -0.2, 0.02), (0.50, 0.2, 0.02)),
    5: ((0.35, 0.2, 0.02), (0.50, -0.2, 0.02)),
}


def trajectory_bounds(terminals):
    ends = np.nonzero(np.asarray(terminals) > 0.5)[0]
    n = len(terminals)
    if ends.size == 0 or ends[-1] != n - 1:
        ends = np.concatenate([ends, [n - 1]])
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def direct_stretches(near_s, near_g):
    """(x, y) pairs within one trajectory: y a goal-visit, x the last start-visit before it,
    with no goal-visit in x+1..y-1. near_s, near_g: bool arrays over the trajectory's rows."""
    out = []
    last_s = -1
    for i in range(len(near_s)):
        if near_g[i] and last_s >= 0:
            out.append((last_s, i))
            last_s = -1
        if near_s[i]:
            last_s = i
    return out


def build_keep(cube, terminals, tasks, r_start, r_goal, lengths=None):
    keep = np.ones(len(cube), bool)
    per_task = {}
    starts, ends = trajectory_bounds(terminals)
    for k, (s, g) in tasks.items():
        ns = np.linalg.norm(cube - np.asarray(s), axis=1) < r_start
        ng = np.linalg.norm(cube - np.asarray(g), axis=1) < r_goal
        n_str, n_rows = 0, 0
        for a, b in zip(starts, ends):
            for x, y in direct_stretches(ns[a:b + 1], ng[a:b + 1]):
                keep[a + x + 1:a + y] = False
                if lengths is not None:
                    lengths.append(max(y - x - 1, 0))
                n_str += 1
                n_rows += max(y - x - 1, 0)
        per_task[k] = {'stretches': n_str, 'rows_in_stretches': n_rows}
    return keep, per_task


def random_keep(terminals, lengths, seed):
    """Control mask: one stretch per entry of `lengths`, same length, at a uniform random
    trajectory and offset. Stretches may overlap each other and the direct stretches."""
    rng = np.random.default_rng(seed)
    starts, ends = trajectory_bounds(terminals)
    keep = np.ones(len(terminals), bool)
    for L in lengths:
        t = rng.integers(len(starts))
        a, b = starts[t], ends[t]
        L = min(L, b - a)
        o = a + rng.integers(0, b - a - L + 1)
        keep[o:o + L] = False
    return keep


def remaining_direct(cube, terminals, keep, tasks, r_start, r_goal):
    """Direct stretches left in the kept data, with a cut ending a trajectory piece."""
    term = np.asarray(terminals) > 0.5
    nxt_dropped = np.r_[~keep[1:], False]
    term = (term | nxt_dropped)[keep]
    c = cube[keep]
    starts, ends = trajectory_bounds(term)
    out = {}
    for k, (s, g) in tasks.items():
        ns = np.linalg.norm(c - np.asarray(s), axis=1) < r_start
        ng = np.linalg.norm(c - np.asarray(g), axis=1) < r_goal
        out[k] = int(sum(len(direct_stretches(ns[a:b + 1], ng[a:b + 1])) for a, b in zip(starts, ends)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=os.path.join(
        os.environ.get('OGBENCH_DATASET_DIR', os.path.expanduser('~/.ogbench/data')), 'cube-single-play-v0.npz'))
    p.add_argument('--r-start', type=float, default=0.06)
    p.add_argument('--r-goal', type=float, default=0.06)
    p.add_argument('--control', action='store_true',
                   help='write the matched random-drop control instead (same stretch lengths)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with np.load(a.dataset) as d:
        cube = np.asarray(d['qpos'][:, 14:17], np.float64)
        terminals = np.asarray(d['terminals'])
    lengths = []
    keep, per_task = build_keep(cube, terminals, CUBE_TASKS, a.r_start, a.r_goal, lengths)
    if a.control:
        keep = random_keep(terminals, lengths, a.seed)
    left = remaining_direct(cube, terminals, keep, CUBE_TASKS, a.r_start, a.r_goal)
    if not a.control:
        assert all(v == 0 for v in left.values()), f'direct stretches left after the drop: {left}'
    # The training set has one row per transition: the raw rows whose terminal flag is 0
    # (ogbench.utils.load_dataset drops each trajectory's last state). Row i is s_i -> s_{i+1}.
    keep = keep[np.asarray(terminals) < 0.5]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, keep=keep)
    report = {'dataset': a.dataset, 'control': a.control, 'seed': a.seed, 'r_start': a.r_start,
              'r_goal': a.r_goal, 'rows': len(keep), 'kept': int(keep.sum()), 'dropped': int((~keep).sum()),
              'per_task': per_task, 'direct_stretches_left': left}
    with open(a.out + '.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
