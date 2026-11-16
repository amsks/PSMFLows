# Density FB (normalized FB) paper benchmark — raw paper numbers vs latent Density FB vs psmz (flow PSM)

Started 2026-11-16 (user order). The goal is to run exactly the environments the Density FB paper reports and put
three numbers side by side per environment: the paper's raw-action Density FB, our latent Density FB port on the
frozen behaviour flow, and psmz (behaviour flow + z-conditioned factorised PSM measure). Protocol for our rows:
actor alone, five tasks (or the paper's task set), 500 episodes per task, 3 seeds, mean ± 95% CI (4.303·SE), BC
control beside it. Paper rows are the paper's protocol (aws to confirm: episodes, seeds, checkpoint rule).
Rows marked `?` are pending aws (mailbox request 2026-11-16).

| env (dataset) | task set | state input | raw Density FB (paper) | latent Density FB (ours) | psmz (ours) | BC |
|---|---|---|---|---|---|---|
| cube-single-play (OGBench) | 5 single tasks | coords in B | 0.829 ± 0.040 (z variant) / 0.917 (full state, z) / 0.931 (goal) | 0.948 ± 0.005 (coords, anchor 0.1, 1M) / 0.844 (full state, 500k) | 0.884 ± 0.025 (full state, batch 1024, 500k); 0.897 ± 0.050 (+2 heads) | 0.111 |
| cube-double-play (OGBench) | 5 single tasks | coords in B | 0.624 (? variant; permuting env 0.158 full state) | 0.516 ± 0.073 (coords, 1M) / 0.052 (full state, 500k) | 0.319 ± 0.027 (coords, 250k, 2 heads + w input) / full state ? (training) | 0.019 |
| scene-play (OGBench) | 5 single tasks | coords (5 blocks) | 0.527 (?) | 0.543 ± 0.062 graded / 0.856 ± 0.111 pass-fail index (1M) | 0.248 ± 0.050 (old config, 1M) / winning config ? (training) | ? |
| puzzle-3x3-play (OGBench) | ? | ? | ? | — | — | — |
| antmaze / pointmaze (OGBench) | ? | ? | ? | antmaze 0.085 (psmgoal line, BC 0.072) | — | 0.072 |
| MetaWorld (which tasks / dataset?) | ? | ? | ? | — (dataset access pending) | — | — |
| ManiSkill (which tasks / dataset?) | ? | ? | ? | — (dataset access pending) | — | — |

Sources for the filled cells: docs/HANDOFF.md 2026-11-04..06 (aws rows), psmgoal-vs-dfb-2026-11-09/README.md (psmz rows),
memory of TD-JEPA Table 1 context (raw Density FB 81.3 / 62.4 / 52.7 — aws to confirm exact variant and protocol).
Stage A (behaviour flow) and Stage B (preimages) have to be produced for every new dataset before latent rows exist.
