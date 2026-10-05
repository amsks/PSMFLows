# Gridworld tests of the psmgoal measure (2026-09-30 .. 2026-10-02)

Two small tests that train the real `PSMGoalAgent` measure on a gridworld where the exact
successor measure is known. Both use uniform-random behaviour data and 4 deterministic moves.
Encodings: `onehot` (states repeat across rows) and `xyphase` (cell coordinates plus a phase
that advances every step, so every state is unique and the next state is an exact function of
state and action, as on OGBench cube).

## fourrooms/ (`tools/diag_fourrooms_induced_reward.py`)

Four-rooms layout, 8 goals, 3 seeds. Compares acting greedily on M at the goal with a scalar
critic trained on a reward read out of M.

| figure | content |
|---|---|
| `exact.png` | exact M of the random policy |
| `onehot_*.png` | learned M, repeating states |
| `xyphase_*.png` | learned M, unique states |
| `*_trainw.png` | M read with the coefficient it was trained with |
| `*_gs_reg.png`, `*_gs_lp.png` | M read with the regression / Lagrangian coefficient |
| `*_feat.png` | reward read out of M's features |
| `path_profiles.png` | each quantity along the shortest path from cell (0,0) to goal (8,8) |

Panels in each grid figure, left to right: M read at the goal with greedy arrows; reward read
out of M; scalar critic on that reward with the true episode end; same without episode end;
scalar critic on the true reward.

| policy | exact M | onehot | xyphase |
|---|---|---|---|
| greedy on M, trained w | 1.000 | 0.731 | 0.565 |
| greedy on M, regression w | | 0.084 | 0.239 |
| greedy on M, Lagrangian w | | 0.044 | 0.134 |
| scalar critic, reward -1 everywhere + true episode end | 1.000 | 1.000 | 1.000 |
| random | 0.251 | 0.251 | 0.251 |

`results.json` holds every number; `EXPECTED.txt` was written before the runs.

## gridworld_7x7/ (`tools/diag_gridworld_psm.py`)

7x7 open room, 30k steps, 1 seed per arm. Measures how well the learned M matches the exact M,
M at the own next state vs elsewhere, the cosine between policy coefficients w(z), and phi's
effective rank, across encodings, policy-code keying (state vs dataset row), number of policy
codes (4 vs 2^16), head bound, discount, and an FB-loss comparator. No figures; numbers are in
`results.json`, summarised by `aggregate.py`. Key rows: one-hot, 4 policies, discount 0.9:
correlation 0.980; unique states (`Bdet`): M at the own next state 64 (bound 65), elsewhere 0.15,
correlation 0.157.
