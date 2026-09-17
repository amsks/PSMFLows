import os

import json
import random
import time

import utils.xla_guard  # noqa: F401  -- MUST precede jax: disables the miscompiling XLA:GPU autotuner

import hydra
import jax
import ml_collections
import numpy as np
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer, add_skill_targets, apply_reward_override
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb


def _lists_to_tuples(x):
    """Recursively convert lists to tuples (e.g. *_hidden_dims) so the agent
    config matches the original ml_collections shape and stays hashable for jax."""
    if isinstance(x, list):
        return tuple(_lists_to_tuples(v) for v in x)
    if isinstance(x, dict):
        return {k: _lists_to_tuples(v) for k, v in x.items()}
    return x


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    # Set up logger.
    exp_name = get_exp_name(cfg.seed)
    setup_wandb(
        entity=cfg.wandb_entity, project=cfg.wandb_project, group=cfg.run_group, name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    save_dir = os.path.join(cfg.save_dir, wandb.run.project, cfg.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, default=str)

    # Build the agent config (ml_collections.ConfigDict) from the Hydra agent group.
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))

    # Make environment and datasets.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack,
        dataset_fraction=cfg.get('dataset_fraction', 1.0),
        dataset_fraction_seed=cfg.get('dataset_fraction_seed', 0))
    if cfg.video_episodes > 0:
        assert 'singletask' in cfg.env_name, 'Rendering is currently only supported for OGBench environments.'
    if cfg.online_steps > 0:
        assert 'visual' not in cfg.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    if config.get('skill_cond', False):
        # Hindsight-window skill conditioning (fql only): attach batch['skills'] before
        # Dataset.create freezes the arrays. make_env_and_datasets may already hand back a
        # frozen Dataset (OGBench path), so unfreeze to a plain dict first.
        train_dataset = dict(train_dataset)
        train_dataset['skills'] = add_skill_targets(train_dataset, config['skill_window'])
        if val_dataset is not None:
            val_dataset = dict(val_dataset)
            val_dataset['skills'] = add_skill_targets(val_dataset, config['skill_window'])
            # train_dataset is re-frozen by Dataset.create below; val_dataset is not, so
            # re-wrap it here or it stays a plain dict when p_aug/frame_stack are set.
            val_dataset = Dataset.create(**val_dataset)

    if config['agent_name'] in ('psmflow', 'psmgoal'):
        # These train on the preimage-augmented dataset (latents per transition).
        from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
        assert config.get('preimage_path'), (
            f"{config['agent_name']} requires agent.preimage_path "
            "(tools/precompute_preimages.py)")
        aug = load_augmented_dataset(config['preimage_path'])
        # Read the sidecar before the size check so a sampled tuning batch reports its own
        # cause. `+preimage_sample` npz files cover a random subset of the buffer and are
        # scoring artifacts for tools/tune_preimage_inversion.py, never training inputs;
        # without this they fail the row count below and read as "wrong env or stale file".
        _meta_path = str(config['preimage_path']) + '.meta.json'
        _meta = {}
        if os.path.exists(_meta_path):
            with open(_meta_path) as f:
                _meta = json.load(f)
        assert not _meta.get('sampled_batch'), (
            f"preimage npz {config['preimage_path']!r} is a SAMPLED BATCH "
            f"(sample_mode={_meta.get('sample_mode')!r}, k={_meta.get('sample_k')}, "
            f"{_meta.get('num_transitions')} rows): a tuning artifact covering a random "
            'subset of the buffer, not a training input. Regenerate the chosen setting '
            'without +preimage_sample (use dataset_fraction to shrink the run instead).')
        assert aug['observations'].shape[0] == train_dataset['observations'].shape[0], (
            'preimage npz size mismatch vs env dataset — wrong env or stale file?')
        # Pairing guard: latents are only meaningful for the EXACT flow that produced them.
        # The row count above catches wrong-env; the .meta.json sidecar catches the silent
        # case — same env, different Stage-A seed/epoch or inversion vintage — which would
        # otherwise train the whole representation on another flow's latents with no
        # symptom except bad results.
        import glob as _glob
        meta_path = _meta_path
        if os.path.exists(meta_path):
            meta = _meta
            assert meta.get('env_name') == cfg.env_name, (
                f"preimage npz was computed on {meta.get('env_name')!r}, not {cfg.env_name!r}")
            ckpt = config.get('flow_ckpt_path')
            if ckpt and meta.get('restore_path'):
                ours = {os.path.realpath(p) for p in _glob.glob(str(ckpt))}
                theirs = {os.path.realpath(p) for p in _glob.glob(str(meta['restore_path']))}
                assert ours & theirs, (
                    f"preimage npz was inverted from {meta['restore_path']!r} but "
                    f"agent.flow_ckpt_path={ckpt!r} resolves elsewhere — mismatched flow")
                assert int(meta.get('restore_epoch') or 0) == int(config.get('flow_ckpt_epoch') or 0), (
                    f"preimage npz used restore_epoch={meta.get('restore_epoch')} but "
                    f"agent.flow_ckpt_epoch={config.get('flow_ckpt_epoch')}")
            # WHICH SUBSET. At dataset_fraction < 1 the row count is identical for every
            # fraction_seed, so the size check above cannot see a wrong-subset pairing:
            # the latents would belong to different transitions than the ones trained on,
            # with no symptom but bad numbers. Sidecars written before 08-14 lack these
            # keys; the content spot-check below is the guard that covers those.
            if 'dataset_fraction' in meta:
                assert float(meta['dataset_fraction']) == float(cfg.get('dataset_fraction', 1.0)), (
                    f"preimage npz was computed at dataset_fraction="
                    f"{meta['dataset_fraction']} but this run uses "
                    f"{cfg.get('dataset_fraction', 1.0)}")
                assert int(meta['dataset_fraction_seed']) == int(cfg.get('dataset_fraction_seed', 0)), (
                    f"preimage npz used dataset_fraction_seed="
                    f"{meta['dataset_fraction_seed']} but this run uses "
                    f"{cfg.get('dataset_fraction_seed', 0)}")
            elif float(cfg.get('dataset_fraction', 1.0)) != 1.0:
                print('WARNING: preimage sidecar predates dataset_fraction recording and '
                      'this run subsamples; relying on the content spot-check below.')
            # WHICH TARGET the mixture was fitted to. prior_scale=0 is the pre-2026-08-14
            # likelihood-only target, whose fits leave the prior's typical set (measured:
            # per-dim variance 3.68 and latents 3.10 from the point inverse on pointmaze,
            # against 0.096 / 0.428 at prior_scale=1). The point arm is the backward ODE
            # and is unaffected, so this only bites when the run actually reads the
            # mixture — which is exactly the case nothing else here checks.
            _ps = (meta.get('inversion') or {}).get('prior_scale')
            if not config.get('use_point_preimage', False):
                assert _ps is not None and float(_ps) > 0.0, (
                    f'preimage npz was inverted at prior_scale={_ps} (legacy '
                    'likelihood-only target) but this run reads the MIXTURE '
                    '(use_point_preimage=false). Those fits sit outside the prior the '
                    'latent actor samples from. Regenerate at inversion.prior_scale=1.0, '
                    'or set agent.use_point_preimage=true.')
            # `measure_u_samples > 1` with the mixture source reads the same arrays, but for
            # a different purpose and under a different criterion, so it does NOT take the
            # assertion above. That gate asks whether the fit sits inside the prior the
            # latent ACTOR samples from. This path never samples the actor from the mixture:
            # it fits the measure head at extra latents, where the only thing that matters is
            # whether those latents decode to the transition's recorded action. That is now
            # measured directly (tools/diag_mixture_decode.py) and the prior_scale proxy is
            # actively misleading for it -- on cube the prior_scale=0.691 npz's samples
            # decode WORSE (0.207) than the legacy npz's (0.167) against a point inverse at
            # 0.089, and on pointmaze no prior_scale>0 npz exists at all. The gate here is
            # instead that `measure_u_mixture_shrink` was chosen deliberately, which the
            # agent's `create` asserts; this prints what the sidecar says so a run that
            # picked the shrink on a different npz is visible in the log.
            elif int(config.get('measure_u_samples', 1)) > 1 and (
                    config.get('measure_u_source', 'mixture') == 'mixture'):
                print(f'NOTE: reading the preimage MIXTURE for measure_u_samples='
                      f'{config.get("measure_u_samples")} at shrink='
                      f'{config.get("measure_u_mixture_shrink")}; npz prior_scale={_ps}. '
                      'The shrink must come from tools/diag_mixture_decode.py --shrink on '
                      'THIS npz -- see docs/design/2026-09-08-critic-signal-and-dsrl-na.md.')
            if (_ps is None or float(_ps) == 0.0) and config.get(
                    'use_point_preimage', False) and int(
                    config.get('measure_u_samples', 1)) == 1:
                # Sidecars written before 08-14 have no prior_scale key at all; absent
                # means the legacy target, same as an explicit 0.0.
                print(f'NOTE: preimage npz uses the legacy prior_scale={_ps} mixture, but '
                      'this run reads point preimages only — the mixture arrays are unread.')
        else:
            print('WARNING: preimage npz has no .meta.json sidecar; cannot verify it '
                  'matches agent.flow_ckpt_path — proceed only if you are sure.')
        # Content spot-check: the rows themselves must be the SAME transitions, not just
        # the same count. First and last 1k observations settle it — a different subset
        # (or a different episode ordering) diverges immediately at both ends, and a full
        # 1M-row comparison costs seconds of startup for no extra certainty.
        n_rows = aug['observations'].shape[0]
        m = min(1000, n_rows)
        for lo, hi, where in ((0, m, 'first'), (n_rows - m, n_rows, 'last')):
            assert np.array_equal(np.asarray(aug['observations'][lo:hi]),
                                  np.asarray(train_dataset['observations'][lo:hi])), (
                f'preimage npz observations differ from the env dataset in the {where} '
                f'{m} rows: the latents belong to different transitions (wrong subset, '
                f'wrong dataset_fraction_seed, or a stale npz)')
        # Size check FIRST (it is the wrong-env guard), then neutralize rows whose inversion
        # diverged. Applied at load, not only in the precompute, so npz files written before
        # `preimage_valid` existed are covered too: a single NaN latent otherwise NaNs the
        # whole update, and at batch 1024 over 1M rows a poisoned row lands within ~100 steps.
        aug, _ = repair_invalid_preimages(aug)
        train_dataset = aug
        val_dataset = None  # val split has no preimages; skip validation logging at v1

    # Row-aligned reward relabel (dataset.reward_override_path). Applied last, after every
    # other dataset transform, so the row order it was written against is the one trained on.
    _reward_override = (cfg.get('dataset') or {}).get('reward_override_path')
    if _reward_override:
        _r_before = np.asarray(train_dataset['rewards'], np.float64)
        train_dataset = apply_reward_override(train_dataset, _reward_override)
        _r_after = np.asarray(train_dataset['rewards'], np.float64)
        print(f'reward override from {_reward_override}: rewards mean/std '
              f'{_r_before.mean():.4f}/{_r_before.std():.4f} -> {_r_after.mean():.4f}/{_r_after.std():.4f}')

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if cfg.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=cfg.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(cfg.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = cfg.p_aug
            dataset.frame_stack = cfg.frame_stack
            if config['agent_name'] in ('psmflow', 'psmgoal'):
                # Emit u_0 / u_0' per transition: either a draw from the stored EM mixture
                # or the exact backward-ODE point, per the point-vs-mixture ablation.
                dataset.return_preimage_noise = True
                dataset.preimage_point_mode = bool(config.get('use_point_preimage', False))
            if config['agent_name'] == 'psmgoal':
                # Hindsight goal per row as batch['goals'] (utils.datasets.hindsight_goal_idxs):
                # geometric horizon at the agent's discount, goal_random_frac random states.
                dataset.return_goals = True
                dataset.goal_discount = float(config['discount'])
                dataset.goal_random_frac = float(config['goal_random_frac'])
                # Reference PSM proto stage (agent.proto.enabled): the proto policy is keyed
                # on the dataset ROW, so the batch must carry it as batch['index'].
                _proto_cfg = config.get('proto', None)
                dataset.return_index = bool(_proto_cfg is not None and _proto_cfg.get('enabled', False))

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    na_refit_info = {}          # Arm D1b: latest `refit_na_reward` values, logged every row
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(cfg.seed)
    for i in tqdm.tqdm(range(1, cfg.offline_steps + cfg.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= cfg.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in cfg.env_name and (
                'diverse' in cfg.env_name or 'play' in cfg.env_name or 'umaze' in cfg.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if cfg.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            # Arm D1b: the refit runs on its own schedule, but CsvLogger fixes the header
            # from the FIRST row it writes, so a key present only on refit steps would
            # never reach train.csv. Carry the latest values on every row instead. The
            # first refit is at i == 1, before any log step, so the header always has them.
            train_metrics.update({f'training/{k}': v for k, v in na_refit_info.items()})
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Arm D1b (`dsrl_na.reward_source=phi_readout_fixed`): refit the HELD reward
        # readout w from a fresh relabel batch, and rescale r_hat onto the real reward's
        # scale. Driven from here rather than from inside `update` so that Q_A sees ONE
        # reward function between refits -- refitting per 256-row batch (Arm D1) trained
        # the critic on a reward that was redrawn every step. Two disjoint batches: one
        # to fit on, one to score the fit on.
        _na_cfg = config.get('dsrl_na') or {}
        _refit = int(_na_cfg.get('reward_refit_every') or 0)
        if (_na_cfg.get('enabled') and _na_cfg.get('reward_source') == 'phi_readout_fixed'
                and _refit > 0 and (i == 1 or i % _refit == 0)):
            n_relabel = min(train_dataset.size, int(cfg.get('eval_relabel_size', 10000)))
            fit_b = train_dataset.sample(n_relabel)
            ho_b = train_dataset.sample(n_relabel)
            agent, na_refit_info = agent.refit_na_reward(
                fit_b['next_observations'], fit_b['rewards'],
                ho_b['next_observations'], ho_b['rewards'])
            wandb.log({f'training/{k}': v for k, v in na_refit_info.items()}, step=i)

        # Evaluate agent.
        if cfg.eval_interval != 0 and (i == 1 or i % cfg.eval_interval == 0):
            renders = []
            eval_metrics = {}
            # PSM acts on a reward-inferred task latent; infer it from the (trained)
            # agent over a dataset sample so eval is goal-directed. No-op for agents
            # without infer_eval_z (they act directly on observations).
            eval_agent = agent
            if hasattr(agent, 'infer_eval_z'):
                # Match the reference eval z-inference (evals/ogbench.py): sample
                # `eval_relabel_size` transitions and shift rewards by `eval_reward_shift`
                # (=1.0) so cube-single's {-1,0} task reward becomes {0,1} => z points at
                # goal-reaching states. Without the shift z is inverted (round-2 audit #1).
                n_relabel = min(train_dataset.size, int(cfg.get('eval_relabel_size', 10000)))
                z_batch = train_dataset.sample(n_relabel)
                rew = z_batch['rewards'] + float(cfg.get('eval_reward_shift', 1.0))
                eval_agent = agent.infer_eval_z(z_batch['next_observations'], rew)
            elif hasattr(agent, 'infer_eval_goals'):
                # psmgoal: the same relabel batch, read as a GOAL SET (its rewarding next
                # states) instead of a task vector.
                n_relabel = min(train_dataset.size, int(cfg.get('eval_relabel_size', 10000)))
                z_batch = train_dataset.sample(n_relabel)
                rew = z_batch['rewards'] + float(cfg.get('eval_reward_shift', 1.0))
                eval_agent = agent.infer_eval_goals(z_batch['next_observations'], rew)
            eval_info, trajs, cur_renders = evaluate(
                agent=eval_agent,
                env=eval_env,
                config=config,
                num_eval_episodes=cfg.eval_episodes,
                num_video_episodes=cfg.video_episodes,
                video_frame_skip=cfg.video_frame_skip,
                # Re-seed the eval env's init-state RNG each eval so success is a
                # reproducible function of the weights (matches reference
                # evals/ogbench.py, which re-seeds with cfg.seed every eval).
                seed=cfg.seed,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    main()
