"""Lifting (encoder) and projection helpers — pure PyTorch MLPs.

Separate lifting networks for node features and edge features map both into the
shared latent space (CLAUDE: "separate lifting networks"). These are dense GEMMs
-> cuBLAS handles them; no custom CUDA needed (see README rationale).

SiLU is used as the nonlinearity to stay consistent with the message MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, *, depth: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class NodeLift(nn.Module):
    """Lift raw node features -> latent."""
    def __init__(self, in_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(in_dim, hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeLift(nn.Module):
    """Lift raw edge/interface features -> latent."""
    def __init__(self, in_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(in_dim, hidden, latent_dim)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)
