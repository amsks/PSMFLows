"""Guard test: the training data path wires `with_index` for PSM agents.

The PSM proto-behavior sampler must be keyed by the GLOBAL buffer row index of
each transition (reference baselines/PSM/agent/psm.py:255 uses
next_observation_hash = np.arange(0, N)). DictBuffer.sample emits these rows
under batch["index"] only when the buffer was built with with_index=True.

train.py builds its buffer via data.ogbench.load_ogbench_dataset(...), so the
flag must thread through that loader and be turned on iff the agent is a PSM
variant. For FB / every other agent it stays OFF (byte-identical: no "index").

We exercise the SAME loader train.py calls, on a tiny synthetic on-disk OGBench
episode, so the test is cheap but covers the real wiring.
"""
import inspect

import numpy as np
import torch

from data.ogbench import load_ogbench_dataset


# Mirror train.py's decision: which agents enable with_index.
def _wants_index(agent: str) -> bool:
    return agent in ("psm", "psm_flowbc")


def _write_episode(tmp_path, domain="synthetic-navigate-v0", total=64):
    """Write one tiny .npz OGBench episode load_transitions can read.

    load_transitions reads singular keys 'observation'/'action' and applies the
    EXORL shift: transition row r gets observation = obs_full[r]. We set
    obs_full[t, 0] = 4*t so that, after the shift, observation row r has
    obs[r, 0] == 4*r == 4*(global buffer row), letting the test assert that
    batch['index'] indexes the FULL stored arrays (global rows).
    """
    buf_dir = tmp_path / domain / "buffer"
    buf_dir.mkdir(parents=True)
    obs_dim, act_dim = 4, 2
    observation = (
        np.arange(total, dtype=np.float32)[:, None] * 4.0
        + np.arange(obs_dim, dtype=np.float32)[None, :]
    )  # obs[t, 0] = 4*t
    action = np.zeros((total, act_dim), dtype=np.float32)
    np.savez(
        buf_dir / "episode_000.npz",
        observation=observation,
        action=action,
    )
    return str(tmp_path)


def test_loader_accepts_with_index_param():
    """load_ogbench_dataset must expose a with_index param (defaults off)."""
    sig = inspect.signature(load_ogbench_dataset)
    assert "with_index" in sig.parameters, (
        "load_ogbench_dataset must accept with_index to wire the PSM proto-sampler"
    )
    assert sig.parameters["with_index"].default is False, (
        "with_index must default to False so non-PSM agents stay byte-identical"
    )


def test_psm_path_emits_global_row_index(tmp_path):
    """When the loader is configured for a PSM agent (with_index=True), the
    sampled batch carries `index` as a [B] long tensor of valid GLOBAL rows."""
    data_path = _write_episode(tmp_path)
    buffer = load_ogbench_dataset(
        domain="synthetic-navigate-v0",
        data_path=data_path,
        with_index=_wants_index("psm"),
    )
    assert buffer.with_index is True
    B = 16
    torch.manual_seed(0)
    batch = buffer.sample(B)
    assert "index" in batch, "PSM training path must emit batch['index']"
    idx = batch["index"]
    assert idx.dtype == torch.long
    assert idx.shape == (B,)
    assert int(idx.min()) >= 0 and int(idx.max()) < len(buffer)
    # GLOBAL rows: observation == global row index (obs[:,0] == 4*row).
    obs0 = batch["observation"][:, 0].cpu().numpy()
    assert np.array_equal(obs0, idx.cpu().numpy().astype(np.float32) * 4.0), (
        "batch['index'] does not index the full stored arrays (not global rows)"
    )


def test_default_agent_path_has_no_index(tmp_path):
    """For FB / default agents (with_index=False) the batch must NOT carry
    an 'index' key — keeps every non-PSM caller byte-identical."""
    data_path = _write_episode(tmp_path)
    buffer = load_ogbench_dataset(
        domain="synthetic-navigate-v0",
        data_path=data_path,
        with_index=_wants_index("fb"),
    )
    assert buffer.with_index is False
    torch.manual_seed(0)
    batch = buffer.sample(8)
    assert batch.get("index") is None, "default (FB) path must not emit 'index'"
