# 2026-09-15 — Policy-family diversity through the frozen flow (cube)

Source: `tools/diag_policy_family_diversity.py` (commit 28fac72), one run on
`cube-single-play-singletask-task2-v0`. JSON:
`$PSM_DATA/logs/diag_policy_family_diversity_cube.json` (PSM_DATA=/mnt/home/amohan/psm-data),
wall clock 975 s. Clouds are in the `.npz` sidecar next to it.

Question. The measure `psi(s,u,u')` indexes the family `{pi_{u'} : a = G(s,u')}`, one fixed
latent per policy. GPI over that family can only pick among futures the family actually
produces. This run measures how different those futures are, against a resampling floor
(`bc`) and an off-support reference (`random_action`), and against families that carry a
consistent direction (`tilted_bc*`, `goal_directed`).

## 1. Table

MMD² is the biased Gaussian-kernel V-statistic between visited-state clouds; 0 for identical
clouds. "raw" = observation standardised per dimension by dataset mean/std (28 dims).
"phi" = the affine checkpoint's `phi(s)` output. "pairwise" = mean over member pairs.
"vs pooled bc" = each member against one seeded subsample of the `bc` union sized like a
member cloud, then the mean over members. Classifier = 5-fold stratified softmax regression
(L2 1e-3, Adam 500 it) on per-trajectory mean and std of the features, label = member.
Success and spread are over K members, each on N = 20 episodes.

| family | K | MMD² pairwise raw | MMD² pairwise phi | norm. dist. raw | norm. dist. phi | MMD² vs pooled bc raw | MMD² vs pooled bc phi | clf acc raw | clf acc phi | chance | task-2 success mean ± sd (min..max) | action change / step | final-state dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noise_index | 16 | 0.2441 | 0.1108 | null | null | 0.1369 | 0.0668 | 0.859 | 0.738 | 0.0625 | 0.006 ± 0.024 (0.00..0.10) | 0.072 | 5.43 |
| bc | 16 | 0.0069 | 0.0052 | null | null | 0.0034 | 0.0026 | 0.031 | 0.025 | 0.0625 | 0.084 ± 0.052 (0.00..0.20) | 0.227 | 4.64 |
| random_action | 16 | 0.0050 | 0.0040 | null | null | 0.0530 | 0.0336 | 0.034 | 0.003 | 0.0625 | 0.000 ± 0.000 (0.00..0.00) | 1.760 | 3.84 |
| tilted_bc (M=5) | 16 | 0.0121 | 0.0071 | null | null | 0.0065 | 0.0036 | 0.125 | 0.044 | 0.0625 | 0.094 ± 0.079 (0.00..0.25) | 0.229 | 4.70 |
| tilted_bc_strong (M=20) | 16 | 0.0536 | 0.0171 | null | null | 0.0299 | 0.0096 | 0.409 | 0.253 | 0.0625 | 0.087 ± 0.091 (0.00..0.30) | 0.234 | 5.10 |
| noise_index_consistent (ODE) | 16 | 0.2719 | 0.1236 | null | null | 0.1544 | 0.0733 | 0.834 | 0.666 | 0.0625 | 0.000 ± 0.000 (0.00..0.00) | 0.076 | 5.83 |
| goal_directed (DSRL-NA) | 4 | 0.0283 | 0.0247 | null | null | 0.0236 | 0.0199 | 0.163 | 0.087 | 0.2500 | 0.887 ± 0.074 (0.80..1.00) | 0.173 | 4.35 |

Mean return per family (task 2, horizon 200, reward -1 per step until success):
noise_index -199.7, bc -193.4, random_action -200.0, tilted_bc -191.8, tilted_bc_strong -192.7,
noise_index_consistent -200.0, goal_directed -62.5.

### The normalised-distinguishability column is null in the JSON

The tool defines the column as
`(pairwise_mean − bc pairwise_mean) / (random_action pairwise_mean − bc pairwise_mean)`,
clipped to [0, 1], and writes `null` when the span is not positive. The span is negative in
both spaces:

| space | floor = bc pairwise | ceiling = random_action pairwise | span |
|---|---|---|---|
| raw | 0.006912 | 0.005004 | −0.0019 |
| phi | 0.005233 | 0.003994 | −0.0012 |

Reason. The 16 `random_action` members are 16 seeds of the same distribution `U[-1,1]^5`,
so their clouds differ only by sampling noise, the same way the 16 `bc` members do. The
pairwise MMD² of `random_action` is therefore a second resampling floor, not an
off-support ceiling. The off-support distance the docstring describes is the
`vs pooled bc` column (random_action 0.0530 raw, 0.0336 phi), which is 8x and 13x the bc
floor of that column. The column as specified is not available from this run. The tool's
normalisation would have to use `random_action` vs pooled bc as the ceiling to give a
positive span; that is a code change and was not made here.

Derived in this document, not written by the tool. Normalising the `vs pooled bc` column
with floor = bc vs pooled bc and ceiling = random_action vs pooled bc, clipped to [0, 1]:

| family | raw | phi |
|---|---|---|
| noise_index | 1.00 (unclipped 2.69) | 1.00 (unclipped 2.07) |
| bc | 0.00 | 0.00 |
| random_action | 1.00 | 1.00 |
| tilted_bc | 0.06 | 0.04 |
| tilted_bc_strong | 0.54 | 0.23 |
| noise_index_consistent | 1.00 (unclipped 3.04) | 1.00 (unclipped 2.28) |
| goal_directed | 0.41 | 0.56 |

Floor and ceiling for this derived column: raw 0.003363 / 0.052994, phi 0.002551 / 0.033569.

## 2. Protocol and constants

- Checkpoint: affine paper-strict run
  `$PSM_DATA/exp/PSMFLows/affine_strict_cube/sd001_s_2491601.0.20260904_181115`, epoch 500000.
  Agent config inherited from that run's `flags.json`; CLI overrides were only
  `flow_ckpt_path`, `flow_ckpt_epoch`, `preimage_path`, `use_point_preimage`.
- Decoder path: the run acts through the one-step distilled head (`decode_path = onestep`).
  `noise_index_consistent` decodes through the 10-step Euler ODE. The local re-implementation
  of the onestep path matched `agent.decode` on 64 dataset states to max abs diff 0.0.
- `u_clip = 3.0`, `action_dim = 5`, `obs_dim = 28`.
- Rollouts: K = 16 members per family (K = 4 for goal_directed), N = 20 episodes per member,
  horizon = the env default (200 steps on cube). Initial env seeds are `seed*1000 + j`,
  j = 0..19, with `seed = 0`, so seeds 0..19, the same 20 for every member of every family.
  The K members run in lockstep over K env copies.
- noise_index latents: `N(0, I)` clipped to `u_clip`, `PRNGKey(4321)`, the same 16 for both
  decoder paths.
- Kernel bandwidths: median pairwise distance over a 4000-point subsample of the union of all
  families, once per space. raw 7.3457, phi 15.9104.
- Cloud sizes (mean points per member): 3838-4000 for the K = 16 families, 1268 for
  goal_directed (its episodes end early on success; mean length 63 steps).
- Classifier: 320 trajectories per K = 16 family, 80 for goal_directed. Chance 1/K.
- goal_directed members: `mode`, `stoch_a`, `stoch_b` (the same weights, two sampling
  streams), `mode_seed2` (a second training seed). Run dirs
  `$PSM_DATA/exp/PSMFLows/dsrlna_cube/sd001_s_2492499.0.20260908_231056` and
  `.../sd000_s_2492498.0.20260908_231056`, epoch 500000. That run trained with
  `u_clip 1.5`, `actor.bc_coeff 0`, `actor_mode dsrl_sac`, `dsrl_na.enabled true`.
  Its own resampling floor (stoch_a vs stoch_b) is 0.0129 raw / 0.0144 phi; mode vs a
  stochastic stream is 0.0252 raw / 0.0147 phi.
- Tilted BC reward: 2-hidden-layer, 64-unit tanh MLP on dataset-standardised (s, a),
  `W ~ N(0, 1/fan_in)`, `b = 0`, seed 100+k. Ascent in `u` with step 0.5, M = 5 or 20 steps,
  then clip to `u_clip`.

## 3. Index coherence (block C)

Present in the JSON. 2000 dataset states (row seed 11) × a 64-latent panel
(`PRNGKey(4322)`, a different panel from the 16 rollout latents). Fraction of the
within-state action-deviation variance `G(s,u_k) − mean_k G(s,u_k)` explained by the latent
main effect across states:

| decoder | frac_index | per action dim | shuffled control | std total | std index |
|---|---|---|---|---|---|
| onestep (the run's path) | 0.510 | 0.638, 0.515, 0.479, 0.406, 0.687 | 0.0005 | 0.0885 | 0.0632 |
| ode (10 steps) | 0.466 | 0.599, 0.469, 0.453, 0.362, 0.656 | 0.0005 | 0.0920 | 0.0628 |

Half of the action deviation a fixed latent produces is the same across states. The
shuffled control is 0.0005, so 0.51 is not a small-sample artefact. The total within-state
action std is 0.089 on actions clipped to [-1, 1].

## 4. Tilted-BC latent statistics (per decision, over all members and steps)

| family | displacement ‖u − u0‖ mean / std / p90 | any coord clipped | frac coords clipped | reward gain mean / p90 |
|---|---|---|---|---|
| tilted_bc (M=5) | 0.103 / 0.079 / 0.170 | 0.0145 | 0.0029 | 0.0061 / 0.0110 |
| tilted_bc_strong (M=20) | 0.388 / 0.230 / 0.646 | 0.0180 | 0.0036 | 0.0220 / 0.0420 |

The latent moves 0.10 (M = 5) or 0.39 (M = 20) from its `N(0, I)` draw, against a draw norm
of about 2.2 in 5 dims. Fewer than 2% of decisions touch the clip.

## 5. Reading per row

noise_index. Sixteen fixed latents give sixteen clouds that a linear read-out separates
86% of the time (chance 6%), and their pairwise MMD² is 35x the bc resampling floor in raw
space and 21x in phi space. Those clouds succeed on task 2 at 0.006, so the family is
distinguishable but every member sits below the bc control; the per-step action change of
0.07 (bc 0.23) and the lowest action norm (0.62) say a fixed latent produces a slower, more
repetitive action stream.

bc. Sixteen resamplings of the same policy have pairwise MMD² 0.0069 raw / 0.0052 phi and
the classifier scores 3% (below chance); this is the floor every other row is read against.
Task-2 success is 0.084 ± 0.052 across the 16 seeds, which is the seed spread of the BC
control at N = 20.

random_action. Uniform random actions give pairwise MMD² at the bc floor (0.0050 raw), because
16 seeds of one distribution are one distribution; the off-support distance is in the
vs-pooled-bc column, 0.053 raw / 0.034 phi. Success is 0.000 with the largest per-step action
change (1.76) and the smallest final-state distance (3.84).

tilted_bc (M=5). Five ascent steps on a random reward move the latent by 0.10 and leave the
clouds at 1.8x the bc floor pairwise and 1.9x in the vs-pooled column; the classifier scores
12.5% raw and 4% phi. Success 0.094 ± 0.079 is the bc control within its spread.

tilted_bc_strong (M=20). Twenty steps move the latent by 0.39 and give clouds 7.8x the bc
floor pairwise raw, 3.3x phi, with classifier accuracy 41% raw / 25% phi. Success 0.087 ± 0.091
is the bc control within its spread, so a consistent latent push that is visible in the state
distribution does not change task-2 outcome for random reward directions.

noise_index_consistent (ODE). Decoding the same 16 latents through the 10-step ODE instead of
the one-step head gives pairwise MMD² 0.272 raw / 0.124 phi and classifier 83% / 67%, within
10% of the one-step numbers. Success is 0.000 on all 16 members, so the noise_index result does
not depend on the decoder path.

goal_directed (DSRL-NA). Four members (a mode, two stochastic streams of the same weights, and
a second training seed's mode) have pairwise MMD² 0.028 raw / 0.025 phi, 4x the bc floor,
and the classifier is at or below chance (16% raw, 9% phi against 25%). They succeed at
0.887 ± 0.074, so a family whose members are barely separable in state space is the one that
solves the task, and the within-family separability numbers do not track task success.

## 6. What this does and does not show

- The noise-indexed family is diverse in state space (35x floor raw, 21x floor phi, 86%
  linear identification) and coherent in action space (0.51 of within-state action
  deviation is latent-explained). The measure has distinct futures to represent.
- None of the 16 noise-index members reaches the bc control on task 2 (0.006 vs 0.084).
  GPI over the 16 sampled here has an argmax whose success is at most 0.10 (one member).
  The GPI runs use K = 64 latents and a per-state argmax, which this probe does not
  reproduce.
- Separability and task success are unrelated across families: the most separable family
  (noise_index) has the lowest success, the least separable non-floor family
  (goal_directed) has the highest.
- The normalised-distinguishability column as defined by the tool does not exist for this
  run (negative span in both spaces). The derived column in section 1 uses a different
  ceiling and is not the tool's output.
- One run, one checkpoint per model, N = 20 episodes per member. The success spreads
  (± 0.05 to 0.09 across members at N = 20) are the resolution of the success column.
