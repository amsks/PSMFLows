# What we have as of 2026-10-20 15:30 CEST

All numbers: cube-single, five-task mean success, 500 episodes per task, mean ± 95% CI over seeds. BC 0.111.

## 0. Headline: the zero-shot results on cube (five tasks, 500 episodes per task)

| method | five-task success |
|---|---|
| critic with the future-state measure (nf2), best-of-64 by the critic, 3 seeds, 250k and 500k | 0.866 |
| critic A (goal-indexed measure), best-of-64 by the critic, 5 seeds | 0.805 |
| student distilled from the critic's best-of-64 picks on 20 random goal regions, acting alone on the five real tasks, 3 seeds | 0.925 ± 0.011 |
| BC (the frozen behaviour flow alone) | 0.111 |

Best-of-64 uses the critic at test time (64 BC draws, the critic picks one). The student acts with no critic at
test time. Everything below is the investigation into why the actor matters and whether one co-trained network
can replace best-of-64 plus the offline student; none of it is a new top number.

## 1. The co-trained actor (advantage-guided actor, arXiv 2610.19783 ported onto critic A)

One network trained next to the critic supplies the TD bootstrap and acts at test time.
Three seeds, bootstrap weight 0.5, on aws (250k landed 04:15 on 10-21).

| | 50k | 100k | 250k |
|---|---|---|---|
| best-of-64 by the critic (critic at test time) | 0.523 ± 0.103 | 0.694 ± 0.182 | 0.635 ± 0.175 |
| the actor alone (no critic at test time) | 0.431 ± 0.131 | 0.434 ± 0.240 | 0.608 ± 0.291 |
| BC | 0.111 | 0.111 | 0.111 |
| critic A's own actor, for comparison | 0.11 | 0.11 | 0.11 |
| critic A's best-of-64 (five seeds) | | | 0.805 |

- 250k: the actor alone is at its own critic's best-of-64 level (0.61 against 0.64); that best-of-64 is 0.17
  below critic A's. The actor rose on every seed from 100k; best-of-64 fell on every seed from 100k.

- First actor trained next to this critic that acts above BC on its own (four times BC).
- The critic's pick keeps improving; the actor does not follow. Gap 0.09 at 50k, 0.26 at 100k.
- A larger tilt at test time does not close it: tilt 1 is the peak on both runs tested; tilt 4 drops below BC.
- Training curves: the actor's supervised loss is 0.92 from 5k on (the score explains 8% of a draw's variance);
  the critic's spread over draws shrinks 25% and gets more peaked while best-of-64 rises. The actor learns a
  smooth shift along the average score; the critic's gain is in which few draws it ranks on top.
- Bootstrap weight: 0.5 / 0.25 / 0.1 give best-of-64 0.70 / 0.69 / 0.58 at 100k (one seed each for 0.25 and 0.1).

## 2. Where the bootstrap action sits over training (ladder, checkpoint every 25k, critic A and its BC-bootstrap control)

Figure: `docs/figures/2026-10-20-actor-ladder/ladder_A_s25_B_s25.png`; with the guided arm:
`docs/figures/2026-10-20-actor-ladder/ladder_A_s25_A_sd0_ag_s0_wb025_s0_wb01_s0.png`.

| bootstrap action, seed 0 | critic A nudge, 25k | critic A nudge, 250k | control, any step | guided actor w_boot 0.5, 100k |
|---|---|---|---|---|
| cosine to the critic's gradient | 0.58 | 0.32 | 0.1 to 0.4 | 0.72 |
| size (latent) | 0.057 | 0.083 | 0.0002 | 0.68 |
| percentile among BC draws by the critic | 52 | 52 | 50 | 64 |
| nearest-data distance (BC draw 0.2127, best-of-64 pick 0.224) | 0.2127 | 0.2122 | 0.2127 | 0.2149 |
| TD loss with vs without it | equal | equal | equal | equal to 3rd decimal |

- Every bootstrap we have tried stays inside the data. None changes the critic's TD loss.
- Critic A's nudge starts small and aligned with the critic's gradient, ends larger and less aligned; its rank
  among BC draws never moves. Its score gain stays below a best-of-2 pick throughout.
- The gain of critic A (0.79) over the control (0.66) comes from a small, consistent, in-support shift in the
  bootstrap; the same step applied as a random best-of-2 pick gave nothing (earlier arms).

## 3. The z-conditioned critic (Density FB's reward index in critic A's recipe), 250k, three seeds

| readout | plain reward | kernel actor reward (Density FB's kernel mode) |
|---|---|---|
| best-of-64 by the critic at the fitted vector | 0.389 ± 0.052 | 0.408 ± 0.211 |
| best-of-64 by expected reward | 0.214 ± 0.073 | 0.265 ± 0.142 |
| the actor conditioned on the vector | 0.107 ± 0.009 | 0.104 ± 0.005 |
| goal-indexed critic A, best-of-64 | 0.789 | |

- Half of the goal index on the same critic; the actor at BC with or without the kernel reward (the actor's
  step is 1/13 of critic A's; the BC term holds it).
- One seed's readout rose 0.26 to 0.42 from 50k to 250k; the 500k line is running on aws (slow GPUs, about 36 h)
  and queued on kisski.

## 4. Distillation (luis): a student trained on best-of-64 picks

- Student trained on 20 random goal regions, acting on the five real tasks, three seeds: 0.925 ± 0.011.
- This is offline distillation after training, with the critic at label time only.

## 5. Antmaze

- 500k, three seeds: 0.034 ± 0.069 (BC 0.072). Closed at BC for both arms.

## 6. Cube-double

- Behaviour flow and preimages published, critic A recipe smoked; no full run launched yet.

## Tonight

- 250k three-seed lines of the co-trained actor, both readouts and a tilt sweep, about 23:30 CEST.
- If the gap to best-of-64 stands, next recipe: the same co-trained actor regressing the critic's best-of-64
  pick during training instead of 16 scored draws.
