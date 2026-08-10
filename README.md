# PSMFlows

Zero-shot reinforcement learning from offline data, where every policy the model trains on
is one the data could have produced.

Methods like FB and PSM learn a reward-free representation by training successor measures
over a family of policies, then answer any reward at test time. Both define that family in
action space, and both include policies the dataset cannot execute: FB maximises over the
whole action set, PSM's codebook policies act uniformly at random. On narrow offline data
every temporal-difference backup is then evaluated where there is no data, and the error
lands in a representation shared by every downstream task.

PSMFlows replaces the policy family. A behaviour-cloned conditional flow `G(s, u)` maps
Gaussian noise to dataset actions, and policies are indexed in that latent space, so every
action the model evaluates, bootstraps, or executes is a flow decode.

The algorithm is written up in [`PAPER/main.tex`](PAPER/main.tex) (section
*LatentFlowPSM*).

## Pipeline

| step | what it does | cost | agent |
|---|---|---|---|
| behaviour flow | fits `G(s, u)` by flow matching on the dataset | ~1 h | `agent=fql agent.bc_only=true` |
| inversion | finds, per transition, the `u` that decodes to the recorded action | 4-19 h | `tools/precompute_preimages.py` |
| representation | trains the features, successor features and latent actor | ~3 h | `agent=psmflow` |

The first two steps are **published, so you can skip them** — checkpoints and preimages are
on Hugging Face as `amsks/psmflows-preimages` (private; ask for access).

## Quickstart

```bash
pip install -r requirements.txt
```

Then follow [`docs/PREIMAGES.md`](docs/PREIMAGES.md): it covers downloading the preimages
and flow checkpoints, pointing the pairing guard at your copy, the environment variables
that matter, training, evaluating against the behaviour-cloning control, and regenerating
preimages if you retrain the flow.

Training, once the artifacts are in `$PSM_DATA`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py \
  agent=psmflow env_name=cube-single-play-singletask-v0 \
  agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play \
  agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
  agent.use_point_preimage=true \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 seed=0
```

Environments run so far: `pointmaze-medium-navigate-singletask-task1-v0`,
`cube-single-play-singletask-v0`, `antmaze-medium-navigate-singletask-v0`.

## Layout

```
agents/psmflow.py      the algorithm: measure loss, latent actor, inference
agents/fql.py          the behaviour flow, and its inverse
agents/psm.py          PSM in action space (the comparison this builds on)
utils/flow_inversion.py  preimage computation, validity, augmented-dataset IO
tools/                 preimage precompute, checkpoint evaluation, inspection
scripts/               launchers, one seed per GPU
configs/               Hydra config tree; agent/psmflow.yaml holds the knobs
docs/PREIMAGES.md      how to get the artifacts and run the pipeline
docs/HANDOFF.md        session-by-session record of what was run and found
PAPER/                 the write-up
```

## Reporting results

Evaluate with 500 episodes (`tools/eval_checkpoint.py`); the 50-episode evaluations logged
during training move by ±0.15 between consecutive points. Report the mean and 95%
confidence interval across seeds, not a peak or a best seed. Always report the
behaviour-cloning control — the frozen flow acting alone, from the same checkpoint the
agent decodes through — beside any number, since that is what the method has to beat.

## Acknowledgments

Built on [Flow Q-Learning](https://github.com/seohongpark/fql) and
[OGBench](https://github.com/seohongpark/ogbench)'s reference implementations. The FQL
README, including its full baseline and reproduction instructions, is preserved at
[`docs/README-fql.md`](docs/README-fql.md).
