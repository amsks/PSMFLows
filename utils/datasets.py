from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


def add_skill_targets(dataset_dict, window):
    """Hindsight-window skill target: skill[i] = observations[min(i + window, end_of_episode(i))].

    end_of_episode(i) is the last index of the episode containing i, from episode
    boundaries in `terminals` (> 0.5). Vectorized via searchsorted over episode-end
    indices -- no python loop over the dataset. Returns the (N, *ob_dims) array; the
    caller stores it as dataset['skills'] so sampled batches carry batch['skills'].
    """
    observations = np.asarray(dataset_dict['observations'])
    terminals = np.asarray(dataset_dict['terminals'])
    n = terminals.shape[0]
    ends = np.nonzero(terminals > 0.5)[0]
    if ends.size == 0 or ends[-1] != n - 1:
        # The dataset may not mark the final row as terminal; it still ends an episode.
        ends = np.concatenate([ends, [n - 1]])
    idx = np.arange(n)
    end_of_episode = ends[np.searchsorted(ends, idx, side='left')]
    skill_idx = np.minimum(idx + window, end_of_episode)
    return observations[skill_idx]


def get_noise_preimage_dataset(dataset, num_clusters=1):
    """Return a dataset with placeholders for the noise preimage."""
    size = get_size(dataset)
    noise_preimage_mean = jax.tree_util.tree_map(lambda arr: np.zeros((size, num_clusters, *arr.shape[1:]), dtype=np.float32), dataset['actions'])
    noise_preimage_cov = jax.tree_util.tree_map(lambda arr: np.eye(*arr.shape[1:], dtype=np.float32)[None, None].repeat(size, axis=0).repeat(num_clusters, axis=1), dataset['actions'])
    noise_preimage_weights = np.ones((size, num_clusters), dtype=np.float32) / num_clusters
    dataset = {
        **dataset,
        'noise_preimage_mean': noise_preimage_mean,
        'noise_preimage_cov': noise_preimage_cov,
        'noise_preimage_weights': noise_preimage_weights
    }
    return dataset

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.return_preimage_noise = False  # Whether to sample preimage noise from the EM mixture; set outside the class.
        self.return_index = False  # Whether to emit the global row index as batch['index'] (PSM proto sampler); set outside the class.
        self.preimage_point_mode = False  # Serve the stored point preimage instead of mixture draws; set outside the class.
        # Cache for _valid_preimage_rows, keyed on the size it was computed at (a
        # ReplayBuffer grows). -1 means "not computed yet".
        self._preimage_valid_rows = None
        self._preimage_rows_size = -1

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def _valid_preimage_rows(self):
        """Rows whose stored preimage is real, or None when every row's is.

        `repair_invalid_preimages` neutralizes rows whose backward ODE diverged (point
        preimage -> u = 0, mixture -> the N(0, I) prior) so that one NaN latent cannot
        poison an update. But u = 0 is not a missing value, it is a WRONG one:
        G(s, 0) != a for those transitions, so training on them teaches the measure a
        (state, latent, action) triple the frozen flow never produces. The rows are few
        (cube 13/1M, antmaze 881/1M) and they are wrong rather than noisy, so they are
        dropped from sampling rather than down-weighted.

        Returns None when nothing is invalid, which keeps the common path untouched.
        """
        if self._preimage_rows_size != self.size:
            from utils.flow_inversion import PREIMAGE_VALID_KEY  # local: avoids a cycle
            valid = self._dict.get(PREIMAGE_VALID_KEY)
            rows = None
            if valid is not None:
                ok = np.asarray(valid[:self.size]) > 0.5
                if not ok.all():
                    rows = np.nonzero(ok)[0]
            self._preimage_valid_rows = rows
            self._preimage_rows_size = self.size
        return self._preimage_valid_rows

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if self.return_preimage_noise:
            rows = self._valid_preimage_rows()
            if rows is not None:
                return rows[np.random.randint(rows.size, size=num_idxs)]
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_index:
            # Global replay-buffer row index of each sampled transition. PSM's proto
            # behavior sampler keys its deterministic next-action on this.
            result['index'] = np.asarray(idxs)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        if self.return_preimage_noise:
            # u for this transition: the exact backward-ODE preimage, or a draw from its
            # stored EM mixture (the point-vs-mixture ablation).
            if self.preimage_point_mode:
                result['noise_preimage'] = self._dict['noise_preimage_point'][idxs]
            else:
                from utils.flow_inversion import sample_preimage_noise  # local: avoids a cycle
                result['noise_preimage'] = sample_preimage_noise(
                    result['noise_preimage_mean'],
                    result['noise_preimage_cov'],
                    result['noise_preimage_weights'],
                )
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
