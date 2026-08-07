"""Normalization layers for message-passing stability (LayerNorm / GraphNorm).

All norms here take the SAME call signature `(x, batch=None, n_graphs=None)` so
message_passing.MPLayer does not have to branch on which one it holds. `batch`
is the [N] long graph-id vector from benchmarks.batching.collate_graphs; passing
None means "one graph", which is what src/train.py does.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    """GraphNorm: normalize across a graph's node set per feature, with a
    learnable mean-subtraction weight (Cai et al. 2021).

    The statistics are PER GRAPH by definition. Under batching, taking them over
    all rows would mix cores with different rod insertions and burnups into one
    mean/variance -- the normalization would then leak information between
    samples and change a graph's prediction depending on what it was batched
    with. So `batch` must be threaded through; with it None we keep the exact
    single-graph behaviour.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None,
                n_graphs: int = None) -> torch.Tensor:  # x: [N, dim]
        if batch is None:
            mean = x.mean(0, keepdim=True)
            out = x - self.alpha * mean
            var = out.var(0, unbiased=False, keepdim=True)
        else:
            n_graphs = int(batch.max()) + 1 if n_graphs is None else n_graphs
            cnt = x.new_zeros((n_graphs, 1)).index_add_(
                0, batch, x.new_ones((x.shape[0], 1))).clamp_min(1.0)
            mean = x.new_zeros((n_graphs, x.shape[1])).index_add_(0, batch, x) / cnt
            out = x - self.alpha * mean[batch]
            var = (x.new_zeros((n_graphs, x.shape[1]))
                   .index_add_(0, batch, out * out) / cnt)[batch]
        out = out / torch.sqrt(var + self.eps)
        return out * self.weight + self.bias


class _IgnoreBatch(nn.Module):
    """Adapt a node-wise norm (LayerNorm, Identity) to the common signature.
    These are per-row, so batching cannot affect them and `batch` is ignored."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None,
                n_graphs: int = None) -> torch.Tensor:
        return self.inner(x)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return _IgnoreBatch(nn.LayerNorm(dim))
    if kind == "graphnorm":
        return GraphNorm(dim)
    if kind == "none":
        return _IgnoreBatch(nn.Identity())
    raise ValueError(f"unknown norm: {kind}")
