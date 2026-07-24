"""Normalization layers for message-passing stability (LayerNorm / GraphNorm)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    """GraphNorm over a single graph's node set: normalize across nodes per
    feature with a learnable mean-subtraction weight (Cai et al. 2021)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, dim]
        mean = x.mean(0, keepdim=True)
        out = x - self.alpha * mean
        var = out.var(0, unbiased=False, keepdim=True)
        out = out / torch.sqrt(var + self.eps)
        return out * self.weight + self.bias


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "graphnorm":
        return GraphNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")
