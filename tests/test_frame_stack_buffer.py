import numpy as np
import torch

from buffers.transition import DictBuffer


def _pixel_storage(eps):
    """eps: list of episode lengths (#transitions). Frame value = global row id
    in channel 0 so we can assert exact stacking/clamping."""
    obs, nobs, ts = [], [], []
    rid = 0
    for L in eps:
        for t in range(L):
            frame = np.full((1, 2, 2), rid, dtype=np.uint8)
            nframe = np.full((1, 2, 2), rid + 100, dtype=np.uint8)
            obs.append(frame)
            nobs.append(nframe)
            ts.append(t)
            rid += 1
    return {
        "observation": np.stack(obs).astype(np.uint8),       # (N,1,2,2)
        "action": np.zeros((len(ts), 2), np.float32),
        "next": {
            "observation": np.stack(nobs).astype(np.uint8),
            "terminated": np.zeros((len(ts), 1), bool),
        },
        "timestep": np.array(ts, dtype=np.int32),
    }


def test_frame_stack_shapes_and_channel_concat():
    data = _pixel_storage([5, 3])
    n = len(data["timestep"])
    buf = DictBuffer(capacity=n, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)
    b = buf.sample(16)
    assert b["observation"].shape == (16, 3, 2, 2)       # C*frame_stack
    assert b["next"]["observation"].shape == (16, 3, 2, 2)
    assert b["action"].shape == (16, 2)                  # non-pixel untouched


def test_frame_stack_clamps_at_episode_start():
    # episode 0: rows 0..4 (ts 0..4); episode 1: rows 5..7 (ts 0..2)
    data = _pixel_storage([5, 3])
    buf = DictBuffer(capacity=8, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)

    # row 0 (episode start, ts=0): all 3 frames clamp to row 0
    stacked = buf._stack_obs("observation", np.array([0]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [0, 0, 0])
    # row 6 (episode 1, ts=1): frames clamp to rows [5, 5, 6]
    stacked = buf._stack_obs("observation", np.array([6]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [5, 5, 6])
    # row 4 (episode 0, ts=4): frames are rows [2, 3, 4] (oldest->newest)
    stacked = buf._stack_obs("observation", np.array([4]))
    np.testing.assert_array_equal(stacked[0, :, 0, 0].numpy(), [2, 3, 4])


def test_next_obs_drops_oldest_appends_next_frame():
    data = _pixel_storage([5, 3])
    buf = DictBuffer(capacity=8, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)
    # row 4: s = [2,3,4]; s' = drop oldest -> [3,4] + next_frame(=104)
    nxt = buf._stack_next_obs(np.array([4]))
    np.testing.assert_array_equal(nxt[0, :, 0, 0].numpy(), [3, 4, 104])


def test_frame_stack_1_is_passthrough():
    data = _pixel_storage([4])
    buf = DictBuffer(capacity=4, device="cpu", frame_stack=1, obs_type="pixels")
    buf.extend(data)
    b = buf.sample(4)
    assert b["observation"].shape == (4, 1, 2, 2)  # no channel multiplication


def test_state_buffer_unaffected():
    n = 10
    data = {
        "observation": np.random.rand(n, 4).astype(np.float32),
        "action": np.random.rand(n, 2).astype(np.float32),
        "next": {
            "observation": np.random.rand(n, 4).astype(np.float32),
            "terminated": np.zeros((n, 1), bool),
        },
    }
    buf = DictBuffer(capacity=n, device="cpu")  # defaults: frame_stack=1, state
    buf.extend(data)
    b = buf.sample(5)
    assert b["observation"].shape == (5, 4)


def test_pixel_stack_requires_timestep():
    import numpy as np
    data = {
        "observation": np.zeros((4, 1, 2, 2), np.uint8),
        "action": np.zeros((4, 2), np.float32),
        "next": {
            "observation": np.zeros((4, 1, 2, 2), np.uint8),
            "terminated": np.zeros((4, 1), bool),
        },
        # intentionally NO "timestep"
    }
    buf = DictBuffer(capacity=4, device="cpu", frame_stack=3, obs_type="pixels")
    buf.extend(data)
    try:
        buf.sample(2)
        assert False, "expected KeyError for missing timestep"
    except KeyError as e:
        assert "timestep" in str(e)
