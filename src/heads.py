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
    """Pool node latents to one graph vector, then MLP -> normalized k_eff.

    k_eff is a GRAPH-wise quantity, so pooling must respect graph boundaries. With
    `batch=None` this pools over every row and returns a 0-d scalar (the original
    single-graph behaviour, kept bit-for-bit). With `batch` given -- a [N] long
    tensor of sample ids, as produced by benchmarks.batching.collate_graphs -- it
    pools per graph and returns [n_graphs]. Pooling over a whole batch as if it
    were one core would blend cores with different rod insertions into a single
    k, which is silently wrong rather than merely inaccurate.
    """

    def __init__(self, latent_dim: int, hidden: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        if pool == "attention":
            self.attn = nn.Linear(latent_dim, 1)
        self.net = mlp(latent_dim, hidden, 1)

    def _pool_single(self, h: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return h.mean(0, keepdim=True)
        if self.pool == "sum":
            return h.sum(0, keepdim=True)
        if self.pool == "attention":
            w = torch.softmax(self.attn(h), dim=0)         # [N,1]
            return (w * h).sum(0, keepdim=True)
        raise ValueError(self.pool)

    def _pool_batched(self, h: torch.Tensor, batch: torch.Tensor,
                      n_graphs: int) -> torch.Tensor:
        g = h.new_zeros((n_graphs, h.shape[1]))
        if self.pool == "attention":
            s = self.attn(h)                                # [N,1]
            # segment softmax: subtract the per-graph max before exp for stability
            mx = h.new_full((n_graphs, 1), float("-inf"))
            mx = mx.scatter_reduce(0, batch.unsqueeze(1), s, reduce="amax",
                                   include_self=True)
            e = torch.exp(s - mx[batch])                    # [N,1]
            den = h.new_zeros((n_graphs, 1)).index_add_(0, batch, e)
            w = e / den[batch].clamp_min(1e-12)
            return g.index_add_(0, batch, w * h)
        g = g.index_add_(0, batch, h)
        if self.pool == "sum":
            return g
        if self.pool == "mean":
            cnt = h.new_zeros((n_graphs, 1)).index_add_(
                0, batch, h.new_ones((h.shape[0], 1)))
            return g / cnt.clamp_min(1.0)
        raise ValueError(self.pool)

    def forward(self, h: torch.Tensor, batch: torch.Tensor = None,
                n_graphs: int = None) -> torch.Tensor:
        if batch is None:
            return self.net(self._pool_single(h)).squeeze()          # scalar
        n_graphs = int(batch.max()) + 1 if n_graphs is None else n_graphs
        return self.net(self._pool_batched(h, batch, n_graphs)).squeeze(-1)  # [B]


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
