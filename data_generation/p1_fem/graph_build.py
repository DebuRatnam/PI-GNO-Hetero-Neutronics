"""Build the MESSAGE graph (neural communication graph) and its edge features.

IMPORTANT: this is the NEURAL graph, kept separate from the PHYSICS graph (the P1
triangulation `elements`) that defines A and F (operators.py). The message graph
is a kNN graph by default: every mesh node connects to its `knn_k` nearest
neighbors, with symmetric closure. Fixed degree gives clean batching + good GPU
utilization on variable-N samples and avoids the ragged-degree / radius-tuning
sensitivity of a radius graph on the structured hex mesh (dense assembly interior
vs. thin sodium gaps). There is NO hardcoded 8-neighbor / diagonal offset list:
connectivity emerges from the geometry. It emits a directed `edge_index [2,E]` /
`edge_features [E,8]`.

Edge feature order [E, 8] (directed src -> dst; see datagen_config.metadata):
    [0] distance        = ||x_dst - x_src||
    [1] dx              = x_dst - x_src           (relative position)
    [2] dy              = y_dst - y_src
    [3] interface_flag  = 1.0 if material differs across the edge, else 0.0
    [4] harmonic_D1     = harmonic mean of the two cells' group-1 D
    [5] dD1             = |D1_src - D1_dst|
    [6] dSigma_r1       = |Sigma_r1_src - Sigma_r1_dst|
    [7] dSigma_s12      = |Sigma_s12_src - Sigma_s12_dst|

Group-1 XS are used as the representative scalars for the interface/diff features
(the PDE itself uses per-group D in operators.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from geometry import CoreGeometry
from datagen_config import GraphConfig
from xs_common import d_slice, sr_slice, scatter_slice, groups_from_n_xs_cols


def _knn_graph(coords: np.ndarray, k: int) -> np.ndarray:
    """kNN message graph: each node -> its `k` nearest neighbors (symmetric).

    Uses a KD-tree; queries k+1 and drops the self-match. Adds the reverse of
    every edge and de-duplicates, so the result is an undirected graph stored as
    directed edge_index [2, E] (both directions), no self-loops. Degree is ~k
    (slightly higher where mutual-kNN is asymmetric), which keeps batching clean.
    """
    n = coords.shape[0]
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k_eff + 1)  # [N, k+1]; col 0 is self
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n), k_eff)
    dst = idx[:, 1:].reshape(-1)              # drop self column
    # symmetric closure + de-dupe (undirected graph as directed both-ways)
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    key = s.astype(np.int64) * n + d.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    keep = np.ones(key_sorted.shape[0], dtype=bool)
    keep[1:] = key_sorted[1:] != key_sorted[:-1]
    sel = order[keep]
    return np.stack([s[sel], d[sel]], axis=0).astype(np.int64)


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise harmonic mean, 0 where either side is non-positive."""
    out = np.zeros_like(a, dtype=np.float64)
    m = (a > 0) & (b > 0)
    out[m] = 2.0 * a[m] * b[m] / (a[m] + b[m])
    return out


def build_message_graph(geom: CoreGeometry, cfg: GraphConfig
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edge_index [2, E], edge_features [E, 8])."""
    coords = geom.coordinates
    edge_index = _knn_graph(coords, cfg.knn_k)

    src, dst = edge_index[0], edge_index[1]
    # Group-1 (fastest group) scalars as the representative interface features,
    # located via the multigroup XS layout so the indices hold for any G.
    G = groups_from_n_xs_cols(geom.cross_sections.shape[1])
    D1 = geom.cross_sections[:, d_slice(G).start]
    Sr1 = geom.cross_sections[:, sr_slice(G).start]
    Ss12 = geom.cross_sections[:, scatter_slice(G).start]
    mat = geom.material_state

    diff = coords[dst] - coords[src]
    distance = np.linalg.norm(diff, axis=1)
    dx, dy = diff[:, 0], diff[:, 1]
    interface_flag = (mat[src] != mat[dst]).astype(np.float64)
    harmonic_D1 = _harmonic(D1[src], D1[dst])
    dD1 = np.abs(D1[src] - D1[dst])
    dSr1 = np.abs(Sr1[src] - Sr1[dst])
    dSs12 = np.abs(Ss12[src] - Ss12[dst])

    edge_features = np.stack(
        [distance, dx, dy, interface_flag, harmonic_D1, dD1, dSr1, dSs12], axis=1)
    return edge_index.astype(np.int64), edge_features.astype(np.float64)
