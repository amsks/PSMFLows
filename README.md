# PSMFlows

Zero-shot reinforcement learning from offline data, where every policy the model trains on
is one the data could have produced.

FB and PSM learn a reward-free representation by fitting successor measures over a family
of policies, then answer any reward at test time. Both define that family in action space,
and both include policies the dataset cannot execute: FB maximises over the whole action
set, PSM's codebook policies act uniformly at random. On narrow offline data every
temporal-difference backup is then evaluated where there is no data, and the error lands in
a representation shared by every downstream task.

PSMFlows replaces the policy family. A behaviour-cloned conditional flow `G(s, u)` maps
Gaussian noise to dataset actions and is then frozen; every action the model evaluates,
bootstraps or executes is a flow decode `G(s, u)` of a latent `u`.

The main implementation is `agent=psmgoal` ([`agents/psmgoal.py`](agents/psmgoal.py),
design [`docs/design/2026-09-17-psmgoal.md`](docs/design/2026-09-17-psmgoal.md)). Its
successor measure is

```
M^{pi_u'}(s, u, s+) = phi(s, u, s+)^T w(u') + b(s, u, s+)
```

fitted by squared TD on a state x goal mesh (goals = the batch's next states), bootstrapping
from a fixed latent policy. A reward is answered by the Lagrangian

```
max_{l >= 0} min_w  - E_{(s,u)~D} sum_{s+} (phi^T w + b) r(s+)
                    - sum_{(s,u,s+)} l(s,u,s+) min(phi^T w + b, 0)
```

with `w` renormalised to the sqrt(D) sphere each step, and acting is generalised policy
improvement over 64 prior draws: decode `argmax_u sum_{s+} (phi(s,u,s+)^T w + b) r(s+)`.

The code keeps three simplifications of that spec: `phi` and `b` are one MLP on
`[s, u, s+]` rather than `A(s,u)^T f(s+)` and `beta(s,u)^T f(s+)`; the policy index is a
binary code `z` with its own `w(z)` network (the RLU proto family) rather than `u'` through
`f`; and inference sums over 32 sampled rewarding goal states rather than all `s+`.

The earlier agent, `agent=psmflow` (= `f_psmflow`, the affine form
`psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)` with `w = E_D[r phi]`), is kept as the
comparator; its symbol map is at the top of [`agents/f_psmflow.py`](agents/f_psmflow.py).
The theory is in `PAPER/` (the technical report is
kept outside the working tree; read it with `git show 5249267:PAPER/main.tex`) and
transcribed in [`docs/COMPENDIUM.md`](docs/COMPENDIUM.md) §2.

## Pipeline

| stage | what it does | how | cost |
|---|---|---|---|
| A behaviour flow | fits `G(s, u)` by flow matching on the dataset | `main.py agent=fql agent.bc_only=true` | ~1 h |
| B inversion | finds, per transition, the `u` that decodes to the recorded action | `tools/precompute_preimages.py` | 4-19 h |
| C representation | fits the measure `phi`, `b`, `w` | `main.py agent=psmgoal` | ~3 h |

Stages A and B are published for `cube-single-play`, `antmaze-medium-navigate` and
`pointmaze-medium-navigate` on Hugging Face as `amsks/psmflows-preimages` (private; ask for
access), so normal work is stage C. [`docs/PREIMAGES.md`](docs/PREIMAGES.md) is the
operational manual: download, sidecar repair, training, eval, regeneration.

## Commands

```bash
pip install -r requirements.txt
```

Stage-C training, once the artifacts are in `$PSM_DATA`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py \
  agent=psmgoal env_name=cube-single-play-singletask-v0 \
  agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play \
  agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
  agent.use_point_preimage=true \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 \
  save_dir=$PSM_DATA/exp seed=0
```

Evaluation (500 episodes; nothing else is reportable) and table regeneration:

```bash
PSM_REPO=$PWD GPU=0 bash scripts/eval500.sh psmgoal cube <run_dir> <out_name>
GPU=0 bash scripts/eval500.sh psmflow cube <run_dir> <out_name>   # comparator
GPU=0 bash scripts/eval500.sh bc      cube -         bc_control
.venv/bin/python tools/make_tables.py [--logs DIR]
```

`eval500.sh` needs no arm flags: `tools/eval_checkpoint.py` takes the agent config from the
run's own `flags.json` and lets only typed CLI overrides sit on top, falling back to
`LEGACY_AGENT_DEFAULTS` for keys that did not exist when the run was written.

`scripts/launch_psmflow.sh` and `scripts/slurm/train_psmflow.sbatch` default `agent.discount`
per environment rather than using the yaml's `0.98` everywhere: antmaze-medium's goal is
200-400 steps away, past `0.98`'s `1/(1-gamma)=50`-step effective horizon, so it trains at
`0.99` (100 steps) while cube/pointmaze/scene stay at `0.98`, whose reward or terminal
horizon fits within 50 steps (`docs/design/2026-09-06-antmaze-failure-tests.md` H2). Override
with `DISCOUNT=<value>` on either launcher; `configs/agent/psmflow.yaml` itself keeps `0.98`
so existing checkpoints still restore against it.

Tests and lint:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
```

Environment: `OGBENCH_DATASET_DIR`, `MUJOCO_GL=egl` (headless),
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `PSM_DATA` (artifacts), `HF_TOKEN` (the dataset repo
is private).

## Config seams

`configs/agent/psmgoal.yaml` (the main agent) documents each of its keys inline. The seams
that change the method: `coef_source` (`lp` Lagrangian inference, default | `amortized`
goal head `w*(g) = h(g)/||h(g)||` | `regression` least-squares `w`), `acting` (`gpi` |
`distill` | `sfbc`), `train_goal_head`, `train_actor` + `actor_objective`, `use_dgd`
(learned multiplier `l` vs fixed hinge).

Everything below is a key of `configs/agent/psmflow.yaml`, the comparator agent. Its
defaults are the affine form of the write-up; the other values are ablations.

| key | default | what it switches |
|---|---|---|
| `discount` | `0.98` | value-function horizon `1/(1-discount)`. The yaml default is fixed so existing checkpoints restore against it, but `scripts/launch_psmflow.sh` / `scripts/slurm/train_psmflow.sbatch` set it per environment (antmaze `0.99`, others `0.98` for now; `DISCOUNT=<value>` overrides) since antmaze-medium's goal is past `0.98`'s 50-step horizon. |
| `psi_form` | `affine` | `affine`: `psi = A(s,u)^T w(u') + beta(s,u)`. `free`: `w(u')` absorbed into the network. |
| `policy_index` | `latent` | what fills psi's index slot: a prior draw `u'`, or the task vector `w`. |
| `acting` | `gpi` | per-step argmax over `(u_i, u'_j)` pairs, or one shot through the amortized actor. |
| `train_actor` | `false` | whether the amortized latent actor is trained at all. |
| `actor_mode` | `ddpg` | which latent actor: the flow-BC head, or the tanh-Gaussian `dsrl_sac` / `gpi_distill` heads (`dsrl_na` is the old alias for `gpi_distill`). |
| `gpi_select` | `argmax` | eval-time selection rule over the `K` prior draws (`max_norm`, `small_ball`, `soft_topm`, `mean`, `fixed_index`, ...). |
| `index_agg` | `max` | how the index slot is aggregated: argmax over the panel, or an upper expectile distilled into `q_dist`. |

`psi_form=affine`, `index_agg=expectile` and `actor.index_panel > 0` all require
`policy_index=latent`; `create` refuses the other combinations rather than mis-typing the
head.

## Layout

```
agents/psmgoal.py        the algorithm (main): measure, TD loss, inference, acting
agents/f_psmflow.py      the comparator affine agent (agents/psmflow.py is its alias)
agents/fql.py            the behaviour flow and its inverse (stage B's core)
utils/psm_networks.py    nn.Modules only: PhiMap, PsiMap, AffinePsiMap, the actors
utils/psm_common.py      pure loss/ensemble/projection helpers
utils/flow_inversion.py  preimage validity, repair, augmented-dataset IO
tools/                   preimage precompute, checkpoint evaluation, diagnostics, figures
scripts/                 launchers; scripts/slurm/ for a scheduler, one seed per job
configs/                 Hydra config tree; agent/psmgoal.yaml holds the main agent's knobs
```

## Results

500 episodes per number ([`tools/eval_checkpoint.py`](tools/eval_checkpoint.py) /
`scripts/eval500.sh`); the in-loop 50-episode evals swing by about ±0.15 between consecutive
points. Report mean and 95% CI across seeds, never a peak or a best seed, and always quote
the behaviour-cloning control beside it — the frozen flow acting alone
(`agent=fql agent.bc_only=true`) from the same checkpoint the agent decodes through, since
that is what the method has to beat.

- [`docs/tables/results.md`](docs/tables/results.md) — generated by `tools/make_tables.py`;
  not hand-edited.
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — the dated record of what was run and found, newest
  entry first.
- [`docs/COMPENDIUM.md`](docs/COMPENDIUM.md) — theory, code seams, every verified number and
  the live hypotheses, including which are already settled negative.
- `docs/design/` and `docs/plans/` — dated design notes and pre-registrations,
  `YYYY-MM-DD-slug.md`.

## The `archive` branch

Every agent, config, test, tool and plan that is not on the `fql` -> preimages -> `psmgoal` / `psmflow`
path lives on the `archive` branch: the peer baselines, the raw-action measure agents, the
settled-negative experiments and the `scripts/baselines/` launchers. It is kept because it is
the negative result of record. Nothing on this branch imports it.

```bash
git show archive:archive/README.md          # what moved and why
git show archive:archive/agents/fb.py       # read one file
git checkout archive -- archive/agents/fb.py   # bring one back, deliberately
```

## Acknowledgments

Built on [Flow Q-Learning](https://github.com/seohongpark/fql) and
[OGBench](https://github.com/seohongpark/ogbench). The FQL README, including its full
baseline and reproduction instructions, is preserved at
[`docs/README-fql.md`](docs/README-fql.md).
