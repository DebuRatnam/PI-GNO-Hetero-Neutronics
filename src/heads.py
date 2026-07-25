"""Projection heads: flux, k_eff, power density.

  - FluxHead:  latent -> [N, 2]  (normalized flux; group ordering [phi1, phi2]).
  - KHead:     pool node latents (mean | sum | attention) -> MLP -> scalar k_hat.
  - PowerHead: computes power from PREDICTED flux using the SAME physical relation
               as the labels (power.py), not an unconstrained head. Takes physical
               (de-normalized) flux and nuSigma_f from node features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lifting import mlp


class FluxHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, n_groups: int = 2):
        super().__init__()
        self.net = mlp(latent_dim, hidden, n_groups)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # -> [N, n_groups] normalized
        return self.net(h)


class KHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        if pool == "attention":
            self.attn = nn.Linear(latent_dim, 1)
        self.net = mlp(latent_dim, hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # -> scalar (normalized k)
        if self.pool == "mean":
            g = h.mean(0, keepdim=True)
        elif self.pool == "sum":
            g = h.sum(0, keepdim=True)
        elif self.pool == "attention":
            w = torch.softmax(self.attn(h), dim=0)        # [N,1]
            g = (w * h).sum(0, keepdim=True)
        else:
            raise ValueError(self.pool)
        return self.net(g).squeeze()                       # scalar


class PowerHead(nn.Module):
    """Physics-consistent power: power = E_f * sum_g nuSf_g * phi_g / nu_bar.
    nuSigma_f columns (one per group) are read from the raw (un-normalized) node
    feature tensor. Matches data_generation/power.py exactly for any group count G.
    `nusf_cols` are the node-feature column indices of nuSf_0..nuSf_{G-1} (see
    features.NodeLayout.nusf_cols)."""
    def __init__(self, energy_per_fission_j: float, nusf_cols, nu_bar: float = 1.0):
        super().__init__()
        self.ef = energy_per_fission_j
        self.nu_bar = nu_bar
        self.register_buffer("nusf_cols",
                             torch.as_tensor(list(nusf_cols), dtype=torch.long))

    def forward(self, phi_phys: torch.Tensor, raw_node_feats: torch.Tensor
                ) -> torch.Tensor:
        nuSf = raw_node_feats[:, self.nusf_cols]              # [N, G]
        fission_rate = (nuSf * phi_phys).sum(dim=1)          # sum_g nuSf_g * phi_g
        return self.ef * fission_rate / self.nu_bar
