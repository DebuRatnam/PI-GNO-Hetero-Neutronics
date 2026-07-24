"""Kernel-integration / message-passing layer with residual update.

Update rule (CLAUDE):
    h_i <- h_i + sum_{j in N(i)} message(h_i, h_j, edge_ij)

The message depends on the source/target node states AND the edge/interface
physics (distance, interface diffusion, material transition, geometry), per the
guide — NOT on material id alone. SiLU is the required nonlinearity.

Compute path per layer:
    1. gather h_src, h_dst over edges                (PyTorch indexing)
    2. m = MessageMLP([h_src, h_dst, e_latent])      (cuBLAS GEMM + SiLU)
    3. aggregate: h_agg = scatter_add(m, dst)        (custom CUDA kernel, src/scatter.py)
    4. h <- h + UpdateMLP(h_agg)  then Norm           (residual + LayerNorm/GraphNorm)

Only step 3 is the custom CUDA kernel; the rest are dense ops on cuBLAS.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lifting import mlp
from norm import make_norm
from scatter import scatter_add_messages


class MPLayer(nn.Module):
    def __init__(self, latent_dim: int, edge_dim: int, hidden: int,
                 norm: str = "layernorm"):
        super().__init__()
        # message sees [h_src, h_dst, edge_latent]
        self.message = mlp(2 * latent_dim + edge_dim, hidden, latent_dim)
        self.update = mlp(latent_dim, hidden, latent_dim)
        self.norm = make_norm(norm, latent_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                e_latent: torch.Tensor, *, use_cuda_scatter: bool = True
                ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h[src], h[dst]
        m = self.message(torch.cat([h_src, h_dst, e_latent], dim=-1))  # [E, latent], SiLU inside
        agg = scatter_add_messages(m, dst, h.shape[0],
                                   use_cuda_scatter=use_cuda_scatter)   # [N, latent]
        h = h + self.update(agg)                                        # residual
        return self.norm(h)


class MessagePassingStack(nn.Module):
    def __init__(self, n_layers: int, latent_dim: int, edge_dim: int,
                 hidden: int, norm: str = "layernorm"):
        super().__init__()
        self.layers = nn.ModuleList([
            MPLayer(latent_dim, edge_dim, hidden, norm) for _ in range(n_layers)
        ])

    def forward(self, h, edge_index, e_latent, *, use_cuda_scatter: bool = True):
        for layer in self.layers:
            h = layer(h, edge_index, e_latent, use_cuda_scatter=use_cuda_scatter)
        return h
