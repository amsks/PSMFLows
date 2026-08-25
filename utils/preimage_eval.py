"""Shared plumbing for the tools that score a preimage npz against its flow.

The npz carries the dataset it was inverted from, so a diagnostic needs only the npz and
the Stage-A checkpoint. What every such tool repeats is the same four steps: check the npz
really belongs to that checkpoint, drop the rows whose inversion diverged, rebuild the
flow, and draw a latent per row from one of the available latent sources.
"""

import glob
import json
import os

import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from main import _lists_to_tuples
from utils.flax_utils import restore_agent
from utils.flow_inversion import (
    load_augmented_dataset,
    repair_invalid_preimages,
    sample_preimage_noise,
)

#: Where a row's latent can come from. `mixture` is the stored EM preimage distribution,
#: `point` the exact backward-ODE preimage, `prior` an uninformed N(0, I) draw.
LATENT_SOURCES = ('mixture', 'point', 'prior')


def stats(x):
    """Six-number summary of a 1-D error array."""
    x = np.asarray(x, np.float64).ravel()
    return {
        'mean': round(float(x.mean()), 6),
        'median': round(float(np.median(x)), 6),
        'p90': round(float(np.percentile(x, 90)), 6),
        'p99': round(float(np.percentile(x, 99)), 6),
        'min': round(float(x.min()), 6),
        'max': round(float(x.max()), 6),
    }


def latents_from(source, data, rows, rng, u_clip=None):
    """One latent per row from `source` (see LATENT_SOURCES), optionally clamped.

    `u_clip` is what psmflow applies to every latent draw before use
    (agents/psmflow.py `sample_step_inputs`), so pass its config value (3.0) to score the
    latent training actually consumes; leave it None to score the stored latents as they
    are. The difference is not cosmetic for the mixture: its draws run well outside the
    N(0, I) support the flow was fitted on, and the decode of a far-tail latent can
    diverge outright (see tools/validate_decode_recovery.py `nan_decode_frac`).
    """
    assert source in LATENT_SOURCES, f'unknown latent source {source!r}, expected one of {LATENT_SOURCES}'
    if source == 'mixture':
        latents = sample_preimage_noise(
            data['noise_preimage_mean'][rows], data['noise_preimage_cov'][rows],
            data['noise_preimage_weights'][rows], rng=rng)
    elif source == 'point':
        latents = np.asarray(data['noise_preimage_point'][rows], np.float32)
    else:
        latents = rng.standard_normal((len(rows), data['actions'].shape[-1])).astype(np.float32)
    if u_clip is not None:
        latents = np.clip(latents, -float(u_clip), float(u_clip))
    return latents


def decode_in_batches(agent, obs, noises, batch_size, skills=None):
    """Flow-decode `noises` at `obs`, in batches, as numpy."""
    out = np.empty_like(np.asarray(noises, np.float32))
    for start in range(0, len(obs), batch_size):
        end = start + batch_size
        sk = None if skills is None else jnp.asarray(skills[start:end])
        out[start:end] = np.asarray(agent.compute_flow_actions(
            jnp.asarray(obs[start:end]), noises=jnp.asarray(noises[start:end]), skills=sk))
    return out


def check_pairing(cfg, agent_cfg, npz_path):
    """Does this npz belong to the checkpoint being restored, at the same discretization?

    Mirrors the guard in main.py -- same env, same checkpoint, same epoch -- plus a
    flow_steps check. That one matters more in a diagnostic than at training time: a
    decode under a different discretization than the inverse would be reported as the
    flow's error rather than as the mismatch it is.
    """
    meta_path = str(npz_path) + '.meta.json'
    if not os.path.exists(meta_path):
        print(f'WARNING: no {meta_path}; cannot verify the npz matches restore_path')
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get('env_name') == cfg.env_name, (
        f"npz was computed on {meta.get('env_name')!r}, not {cfg.env_name!r}")
    if cfg.restore_path is not None and meta.get('restore_path'):
        ours = {os.path.realpath(p) for p in glob.glob(str(cfg.restore_path))}
        theirs = {os.path.realpath(p) for p in glob.glob(str(meta['restore_path']))}
        assert ours & theirs, (
            f"npz was inverted from {meta['restore_path']!r} but restore_path="
            f'{cfg.restore_path!r} resolves elsewhere -- mismatched flow')
        assert int(meta.get('restore_epoch') or 0) == int(cfg.restore_epoch or 0), (
            f"npz used restore_epoch={meta.get('restore_epoch')} but "
            f'restore_epoch={cfg.restore_epoch}')
    if meta.get('flow_steps') is not None:
        assert int(meta['flow_steps']) == int(agent_cfg['flow_steps']), (
            f"npz was inverted at flow_steps={meta['flow_steps']} but "
            f"agent.flow_steps={agent_cfg['flow_steps']}; the decode must use the same "
            'discretization as the inverse or this measures the mismatch')
    return meta


def load_flow_and_preimages(cfg):
    """Restore the Stage-A flow and its preimage npz, paired and validity-filtered.

    Returns:
        (agent, data, valid_rows, skills, meta): `data` is the npz as a plain dict,
        `valid_rows` the row indices whose inversion did not diverge (the repaired ones
        were reset to the N(0, I) prior, so their latent no longer describes their action).
    """
    npz_path = cfg.get('preimage_npz', None)
    assert npz_path, 'set +preimage_npz=<preimages .npz> (tools/precompute_preimages.py output)'
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow shapes)'

    agent_cfg = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    meta = check_pairing(cfg, agent_cfg, npz_path)

    data = load_augmented_dataset(str(npz_path))
    data, valid = repair_invalid_preimages(data)
    valid_rows = np.nonzero(np.asarray(valid) >= 0.5)[0]

    skill_cond = bool(agent_cfg.get('skill_cond', False))
    skills = data.get('skills') if skill_cond else None
    assert not (skill_cond and skills is None), (
        'agent.skill_cond=true but the npz carries no `skills`; the flow would be decoded '
        'with a different input width than it was inverted with')

    agent = FQLAgent.create(int(cfg.seed), data['observations'][:1], data['actions'][:1], agent_cfg)
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    else:
        assert cfg.inversion.get('allow_untrained', False), (
            'an UNTRAINED flow decodes nothing; set restore_path=<stage-A ckpt dir> '
            '(or inversion.allow_untrained=true for a plumbing smoke)')
    return agent, data, valid_rows, skills, meta
