"""One k-NN neighbourhood protocol for every support/coverage diagnostic.

Before this module the three probes that quote "the local data" each built their own
neighbourhood: `diag_action_coverage` searched RAW observation distances over a
with-replacement 200k pool, while `diag_action_distance` and
`diag_generated_pair_support` searched PER-DIMENSION STANDARDIZED observations over a
250k without-replacement cKDTree index. Cube observations mix positions (~1e-1) with
velocities (~1e0), so raw distances are dominated by whichever block happens to have the
largest scale, and the two geometries do not select the same neighbours. The coverage
ratio, the C1 distance and the C3 aleatoric ceiling therefore could not be quoted on the
same figure as the support curve without mixing protocols.

The protocol here is the standardized one (it is the defensible one: no dimension gets a
vote proportional to its unit), fixed at k=32, over a seeded subsample of the buffer.
Every report that uses it should embed `NeighbourIndex.protocol()` so the JSON says which
geometry produced the numbers; the tools that previously published raw-geometry numbers
keep them in their JSONs under a `*_raw_geometry` key, labeled, for continuity.
"""
import numpy as np
from scipy.spatial import cKDTree

K_NEIGHBOURS = 32
INDEX_ROWS = 250_000
PROTOCOL = "standardized-obs k-NN (per-dim z-scored observations, cKDTree, k=32)"


class NeighbourIndex:
    """k-NN over per-dimension standardized observations, with the actions alongside.

    Args:
        observations, actions: the full dataset arrays (N, d_s) / (N, d_a).
        index_rows: buffer subsample size for the tree (the full 1M rows are not needed
            for a neighbourhood estimate, and the subsample size is recorded, never silent).
        seed: seeds the subsample draw.

    Row bookkeeping: `rows` maps a position in the index back to its dataset row, which
    is what makes exact self-exclusion possible in the data-matches-itself baselines.
    """

    def __init__(self, observations, actions, index_rows=INDEX_ROWS, seed=0):
        obs = np.asarray(observations)
        act = np.asarray(actions)
        n = obs.shape[0]
        rng = np.random.default_rng(int(seed))
        if index_rows is None or index_rows >= n:
            rows = np.arange(n)
        else:
            rows = rng.choice(n, size=int(index_rows), replace=False)
        self.rows = np.sort(rows)
        self.mu = obs.mean(axis=0)
        self.sd = obs.std(axis=0) + 1e-8
        self.obs = obs[self.rows]
        self.act = act[self.rows]
        self.tree = cKDTree(self.standardize(self.obs))
        self.action_scale = float(np.abs(act).mean())
        self.seed = int(seed)

    def standardize(self, obs):
        return (np.asarray(obs) - self.mu) / self.sd

    def query(self, obs, k=K_NEIGHBOURS, exclude_rows=None):
        """Nearest neighbours of `obs`.

        Returns (state_dists, positions), both (B, k); `positions` index into this
        index's own `obs`/`act`/`rows` arrays.

        exclude_rows: dataset row ids (B,) to drop from their own neighbourhood -- the
        self-exclusion the data-matches-itself baselines need. At most one neighbour can
        be excluded per query, so k+1 are retrieved and the offending entry dropped.
        """
        obs = np.atleast_2d(np.asarray(obs))
        kq = int(k) + (1 if exclude_rows is not None else 0)
        d, pos = self.tree.query(self.standardize(obs), k=kq)
        d, pos = np.atleast_2d(d), np.atleast_2d(pos)
        if exclude_rows is not None:
            keep = self.rows[pos] != np.asarray(exclude_rows).reshape(-1, 1)
            # stable argsort puts the kept neighbours first in their original order
            order = np.argsort(~keep, axis=1, kind="stable")[:, :int(k)]
            d = np.take_along_axis(d, order, axis=1)
            pos = np.take_along_axis(pos, order, axis=1)
        return d[:, : int(k)], pos[:, : int(k)]

    def neighbour_actions(self, obs, k=K_NEIGHBOURS, exclude_rows=None):
        """(B, k, d_a) actions of the k nearest dataset states."""
        _, pos = self.query(obs, k=k, exclude_rows=exclude_rows)
        return self.act[pos]

    def min_action_dist(self, obs, actions, k=K_NEIGHBOURS, exclude_rows=None,
                        normalize=True):
        """Min distance from each action to the neighbourhood's actions, /mean|a|."""
        neigh = self.neighbour_actions(obs, k=k, exclude_rows=exclude_rows)
        d = np.linalg.norm(neigh - np.atleast_2d(np.asarray(actions))[:, None, :], axis=-1)
        d = d.min(axis=1)
        return d / self.action_scale if normalize else d

    def action_sd(self, obs, k=K_NEIGHBOURS, exclude_rows=None):
        """(B, d_a) per-dimension action spread of the behaviour conditional."""
        return self.neighbour_actions(obs, k=k, exclude_rows=exclude_rows).std(axis=1)

    def protocol(self):
        """Embed this in every report that uses the index."""
        return {
            "protocol": PROTOCOL,
            "k_neighbours": K_NEIGHBOURS,
            "index_rows": int(len(self.rows)),
            "index_seed": self.seed,
            "standardized": True,
            "action_scale_mean_abs": self.action_scale,
        }
