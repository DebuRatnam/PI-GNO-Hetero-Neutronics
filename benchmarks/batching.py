"""Graph batching for the benchmark suite: collate variable-N cores into one
disconnected graph, and remap the sparse physics operators to match.

WHY THIS EXISTS
    src/train.py trains at batch size 1 and re-reads every .npz from disk each
    epoch. Over 5000 samples x hundreds of epochs x four models x three seeds
    that is not a runnable benchmark on a shared GPU. Everything else in
    benchmarks/ depends on this module being right, so it is the first thing
    tested (see test_batching.py).

THE ONLY SUBTLE PART: OPERATOR INDEXING
    Each sample's A/F is [G*N_i, G*N_i] in GROUP-MAJOR local order, i.e. local
    row  L = g * N_i + n   for group g and local node n.

    Node features are concatenated across the batch, so batched node n of sample
    i lives at global row  ptr[i] + n,  and the batched flux vector packed by
    physics.pack_group_major is group-major over the CONCATENATED node set:

        batched row  =  g * sumN + ptr[i] + n

    That is NOT the same as scipy.sparse.block_diag of the per-sample operators,
    whose row for the same entry is  G*ptr[i] + g*N_i + n. Stacking the blocks
    naively therefore multiplies group-1 flux against group-2 rows for every
    sample after the first -- a silent, plausible-looking wrongness. We remap
    each sample's COO indices directly (no block_diag, no permutation matmul):

        g = L // N_i ;  n = L % N_i ;  L' = g*sumN + ptr[i] + n

    applied to rows and columns alike. Exact, one pass, no CSR slicing.

k_eff IS PER GRAPH
    Every core in a batch has its own eigenvalue, so the residual
    R = A phi - (1/k) F phi needs k expanded per node, not one k for the batch.
    See expand_k_group_major.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class BatchedSample:
    """A batch of cores as one disconnected graph. Field-compatible with
    dataio.Sample wherever the benchmark harness touches it, so metrics and
    losses do not need a second code path: a batch of ONE is a valid input to
    metrics.sample_metrics."""
    node_feats: torch.Tensor      # [sumN, D]
    edge_index: torch.Tensor      # [2, sumE] (offset into the batch)
    edge_feats: torch.Tensor      # [sumE, 8]
    flux: torch.Tensor            # [sumN, G]
    k_eff: torch.Tensor           # [B]
    power: torch.Tensor           # [sumN]
    boundary_mask: torch.Tensor   # [sumN] bool
    A: torch.Tensor               # sparse [G*sumN, G*sumN]
    F: torch.Tensor               # sparse [G*sumN, G*sumN]
    coords: torch.Tensor          # [sumN, 2]
    nodal_volume: torch.Tensor    # [sumN]
    material_state: torch.Tensor  # [sumN] long
    batch: torch.Tensor           # [sumN] long, graph id per node
    ptr: torch.Tensor             # [B+1] long, node offset per graph
    n_groups: int
    n_graphs: int
    meta: dict                    # metadata of the FIRST sample (schema is
                                  # identical across a dataset by construction)

    @property
    def n_nodes(self) -> int:
        return self.node_feats.shape[0]

    def counts(self) -> torch.Tensor:
        """[B] nodes per graph."""
        return self.ptr[1:] - self.ptr[:-1]

    def to(self, device) -> "BatchedSample":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return BatchedSample(
            **{k: mv(v) for k, v in self.__dict__.items()})


def expand_k_group_major(k: torch.Tensor, batch: torch.Tensor,
                         n_groups: int) -> torch.Tensor:
    """[B] per-graph k -> [G*sumN] aligned with pack_group_major node ordering.

    The batched flux vector is [phi_g0 over all nodes, phi_g1 over all nodes, ...],
    so the per-node k simply repeats G times: k[batch] tiled group-wise.
    """
    if k.dim() == 0:                      # single graph: leave as a scalar
        return k
    per_node = k[batch]                   # [sumN]
    return per_node.repeat(n_groups)      # [G*sumN]


def _remap_sparse(op: torch.Tensor, n_i: int, ptr_i: int, sum_n: int,
                  n_groups: int):
    """Map one sample's [G*N_i, G*N_i] operator indices into batched group-major
    coordinates. Returns (indices [2, nnz], values [nnz])."""
    op = op.coalesce()
    idx = op.indices()                    # [2, nnz], local group-major
    g = torch.div(idx, n_i, rounding_mode="floor")   # group of each row/col
    n = idx - g * n_i                                 # local node of each row/col
    return g * sum_n + ptr_i + n, op.values()


def collate_graphs(samples: Sequence, device=None) -> BatchedSample:
    """Collate dataio.Sample objects into one BatchedSample.

    Works for a single sample too (B=1), which is what the evaluation path uses
    so that train and eval share exactly one code path.
    """
    if len(samples) == 0:
        raise ValueError("collate_graphs got an empty sample list")

    n_groups = samples[0].flux.shape[1]
    for s in samples:
        if s.flux.shape[1] != n_groups:
            raise ValueError(
                "cannot batch samples with different group counts "
                f"({s.flux.shape[1]} vs {n_groups}); G is a dataset-level schema "
                "property, so this means two datasets got mixed")

    counts = [int(s.node_feats.shape[0]) for s in samples]
    ptr = torch.zeros(len(samples) + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    sum_n = int(ptr[-1])

    batch = torch.repeat_interleave(
        torch.arange(len(samples), dtype=torch.long),
        torch.tensor(counts, dtype=torch.long))

    edge_index = torch.cat(
        [s.edge_index + int(ptr[i]) for i, s in enumerate(samples)], dim=1)

    a_idx, a_val, f_idx, f_val = [], [], [], []
    for i, s in enumerate(samples):
        ai, av = _remap_sparse(s.A, counts[i], int(ptr[i]), sum_n, n_groups)
        fi, fv = _remap_sparse(s.F, counts[i], int(ptr[i]), sum_n, n_groups)
        a_idx.append(ai); a_val.append(av)
        f_idx.append(fi); f_val.append(fv)

    gn = n_groups * sum_n
    A = torch.sparse_coo_tensor(torch.cat(a_idx, dim=1), torch.cat(a_val),
                                size=(gn, gn)).coalesce()
    F = torch.sparse_coo_tensor(torch.cat(f_idx, dim=1), torch.cat(f_val),
                                size=(gn, gn)).coalesce()

    out = BatchedSample(
        node_feats=torch.cat([s.node_feats for s in samples], dim=0),
        edge_index=edge_index,
        edge_feats=torch.cat([s.edge_feats for s in samples], dim=0),
        flux=torch.cat([s.flux for s in samples], dim=0),
        k_eff=torch.stack([s.k_eff.reshape(()) for s in samples]),
        power=torch.cat([s.power for s in samples], dim=0),
        boundary_mask=torch.cat([s.boundary_mask for s in samples], dim=0),
        A=A, F=F,
        coords=torch.cat([s.coords for s in samples], dim=0),
        nodal_volume=torch.cat([s.nodal_volume for s in samples], dim=0),
        material_state=torch.cat([s.material_state for s in samples], dim=0),
        batch=batch, ptr=ptr,
        n_groups=n_groups, n_graphs=len(samples),
        meta=samples[0].meta,
    )
    return out.to(device) if device is not None else out
