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
Gaussian noise to dataset actions and is then frozen; policies are indexed by the latent
`u`, so `pi_{u'} = G(., u')` and every action the model evaluates, bootstraps or executes is
a flow decode. The successor measure of that family is fitted in the affine form the
write-up derives,

```
m(s, u, u', x) = psi(s, u, u')^T phi(x),    psi(s, u, u') = A(s, u)^T w(u') + beta(s, u),
```

with `phi` a basis over future states, `A` and `beta` independent of the policy index `u'`,
and `w(u')` the finite-dimensional coordinate of the policy it indexes. A reward is answered
in closed form by `w = E_D[r(x) phi(x)]`, and acting is generalised policy improvement over
`K` prior draws: decode `argmax_u max_{u'} psi(s, u, u')^T w`.

The symbol map from these letters to the code is at the top of
[`agents/psmflow.py`](agents/psmflow.py). The theory is in `PAPER/` (the technical report is
kept outside the working tree; read it with `git show 5249267:PAPER/main.tex`) and
transcribed in [`docs/COMPENDIUM.md`](docs/COMPENDIUM.md) §2.

## Pipeline

| stage | what it does | how | cost |
|---|---|---|---|
| A behaviour flow | fits `G(s, u)` by flow matching on the dataset | `main.py agent=fql agent.bc_only=true` | ~1 h |
| B inversion | finds, per transition, the `u` that decodes to the recorded action | `tools/precompute_preimages.py` | 4-19 h |
| C representation | fits `phi`, `psi` and (optionally) the latent actor | `main.py agent=psmflow` | ~3 h |

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
  agent=psmflow env_name=cube-single-play-singletask-v0 \
  agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play \
  agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
  agent.use_point_preimage=true \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 \
  save_dir=$PSM_DATA/exp seed=0
```

Evaluation (500 episodes; nothing else is reportable) and table regeneration:

```bash
GPU=0 bash scripts/eval500.sh psmflow cube <run_dir> <out_name>
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

Everything below is a key of `configs/agent/psmflow.yaml`. The defaults are the affine form
of the write-up; the other values are ablations.

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
agents/psmflow.py        the algorithm: construction, losses, update, acting, inference
agents/fql.py            the behaviour flow and its inverse (stage B's core)
utils/psm_networks.py    nn.Modules only: PhiMap, PsiMap, AffinePsiMap, the actors
utils/psm_common.py      pure loss/ensemble/projection helpers shared with archive/
utils/flow_inversion.py  preimage validity, repair, augmented-dataset IO
tools/                   preimage precompute, checkpoint evaluation, diagnostics, figures
scripts/                 launchers; scripts/slurm/ for a scheduler, one seed per job
configs/                 Hydra config tree; agent/psmflow.yaml holds the knobs
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

## archive/

`archive/` holds every agent, config, test, tool and plan that is not on the
`fql` -> preimages -> `psmflow` path: the peer baselines, the raw-action measure agents and
the settled-negative experiments. It is excluded from `ruff check .` and from pytest
collection, nothing in the live tree imports it, and it is kept because it is the negative
result of record. Do not add to it or extend it; see
[`archive/README.md`](archive/README.md) for how to revive something.

## Acknowledgments

Built on [Flow Q-Learning](https://github.com/seohongpark/fql) and
[OGBench](https://github.com/seohongpark/ogbench). The FQL README, including its full
baseline and reproduction instructions, is preserved at
[`docs/README-fql.md`](docs/README-fql.md).
