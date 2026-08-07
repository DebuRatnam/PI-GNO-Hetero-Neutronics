"""Fixed physical domain and sensor lattice, shared by DeepONet and FNO.

Both grid-based baselines need a domain that is FIXED across the dataset. That
is not a detail -- it is the limitation the benchmark is measuring. A DeepONet
branch net reads the input function at a fixed set of sensor locations, and an
FNO reads it on a fixed Cartesian grid; neither can be re-sited per sample
without changing the operator being learned. PI-GNO and MeshGraphNet have no
such object because they read the mesh directly.

The domain is derived once from the train split and then frozen into the run
config, so val and test are probed with exactly the sensors the branch was
trained on. Deriving it per sample would quietly hand the baselines a geometry
adaptation they do not actually have.

Nearest-neighbour is used to sample the input field, not linear interpolation:
material and cross-section fields are piecewise constant per material, and
linear blending across a duct/coolant interface would invent cross sections that
exist in no material.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

import _paths  # noqa: F401


Domain = Tuple[float, float, float, float]   # (x0, x1, y0, y1)


def domain_from_dataset(ds, n_probe: int = 128, pad: float = 0.05) -> Domain:
    """Global bounding box over the first `n_probe` samples, padded.

    Deterministic (the split order is sorted), and `n_probe` samples is enough
    to see every core size in a split: hex core extent is a function of
    reflector_rings, which takes two values.
    """
    n = min(n_probe, len(ds))
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for i in range(n):
        c = ds[i].coords
        x0 = min(x0, float(c[:, 0].min())); x1 = max(x1, float(c[:, 0].max()))
        y0 = min(y0, float(c[:, 1].min())); y1 = max(y1, float(c[:, 1].max()))
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx, y0 - dy, y1 + dy)


def lattice(domain: Domain, m_side: int, device=None,
            dtype=torch.float32) -> torch.Tensor:
    """[m_side**2, 2] Cartesian lattice of sensor / grid points, row-major in y."""
    x0, x1, y0, y1 = domain
    xs = torch.linspace(x0, x1, m_side, device=device, dtype=dtype)
    ys = torch.linspace(y0, y1, m_side, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def to_unit(coords: torch.Tensor, domain: Domain) -> torch.Tensor:
    """Map physical coordinates into [-1, 1]^2 on the fixed domain."""
    x0, x1, y0, y1 = domain
    sx = coords.new_tensor([(x1 + x0) / 2.0, (y1 + y0) / 2.0])
    hw = coords.new_tensor([max((x1 - x0) / 2.0, 1e-9), max((y1 - y0) / 2.0, 1e-9)])
    return (coords - sx) / hw


def nearest_index(query: torch.Tensor, coords: torch.Tensor,
                  chunk: int = 4096) -> torch.Tensor:
    """[M] index into `coords` of the nearest mesh node to each query point.

    Chunked over queries so the M x N distance block never materializes in full
    (fhr is N ~ 7100 and a 128^2 grid is M = 16384).
    """
    out = torch.empty(query.shape[0], dtype=torch.long, device=coords.device)
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d = torch.cdist(q.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [c, N]
        out[i:i + chunk] = d.argmin(dim=1)
    return out


def gather_per_graph(batch, points: torch.Tensor, values: torch.Tensor
                     ) -> torch.Tensor:
    """Sample a node field at `points` for every graph in a batch.

    `values` is [sumN, C] over the batched node set; returns [n_graphs, M, C].
    The nearest-node search is done per graph, because a node of a DIFFERENT
    core in the same batch must never be a candidate.
    """
    outs: List[torch.Tensor] = []
    for i in range(batch.n_graphs):
        lo, hi = int(batch.ptr[i]), int(batch.ptr[i + 1])
        idx = nearest_index(points, batch.coords[lo:hi])
        outs.append(values[lo:hi][idx])
    return torch.stack(outs, dim=0)
