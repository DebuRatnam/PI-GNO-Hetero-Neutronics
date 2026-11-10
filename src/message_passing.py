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


AGGREGATIONS = ("sum", "volume", "volume_raw")


class MPLayer(nn.Module):
    """One kernel-integration layer.

    AGGREGATION is the difference between a GNN and a GNO, and it is selectable
    here so the benchmark can measure the difference rather than assert it:

      "sum"        h_agg_i = sum_j m_ij
                   Unweighted scatter-add. This is a GNN aggregation. It is what
                   the model did originally, kept as the honest GNN-side control.

      "volume"     h_agg_i = sum_j V_j m_ij / sum_j V_j
                   Volume-weighted average, V = nodal_volume, the FEM quadrature
                   weight already stored in every sample and previously unused by
                   the model. The weight belongs to the SOURCE node j, because
                   the integral being approximated is over y:
                       (K phi)(x_i) = int K(x_i, y) phi(y) dy
                                    ~ sum_j V_j K(x_i, y_j) phi_j
                   This is the default for the GNO claim.

      "volume_raw" h_agg_i = sum_j V_j m_ij
                   The strict Nystrom estimate, unnormalized. Included because it
                   shows WHY normalization is needed: with a fixed-k kNN graph the
                   neighbourhood shrinks as the mesh refines, so the raw sum's
                   magnitude collapses with resolution while the normalized form
                   stays stable.

    THE LIMITATION, stated plainly rather than overclaimed: discretization
    invariance in the GNO sense needs (i) a fixed physical integration domain and
    (ii) quadrature weights converging to the measure. Volume weighting supplies
    (ii). It does NOT supply (i) -- a fixed-k neighbourhood still shrinks as
    N grows -- so this is not the theorem, it is the part of the theorem that
    matters over the 2-4x refinement range the resolution study spans.
    """

    def __init__(self, latent_dim: int, edge_dim: int, hidden: int,
                 norm: str = "layernorm", aggregation: str = "sum"):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, "
                             f"got {aggregation!r}")
        self.aggregation = aggregation
        # message sees [h_src, h_dst, edge_latent]
        self.message = mlp(2 * latent_dim + edge_dim, hidden, latent_dim)
        self.update = mlp(latent_dim, hidden, latent_dim)
        self.norm = make_norm(norm, latent_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                e_latent: torch.Tensor, *, use_cuda_scatter: bool = True,
                batch: torch.Tensor = None, n_graphs: int = None,
                node_weight: torch.Tensor = None) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = h.shape[0]
        h_src, h_dst = h[src], h[dst]
        m = self.message(torch.cat([h_src, h_dst, e_latent], dim=-1))  # [E, latent], SiLU inside

        if self.aggregation == "sum":
            agg = scatter_add_messages(m, dst, n,
                                       use_cuda_scatter=use_cuda_scatter)
        else:
            if node_weight is None:
                raise ValueError(
                    f"aggregation={self.aggregation!r} needs node_weight "
                    "(nodal_volume); it is stored in every sample -- pass it "
                    "through from the batch")
            w = node_weight[src].unsqueeze(1)                  # [E,1], weight of y_j
            agg = scatter_add_messages(m * w, dst, n,
                                       use_cuda_scatter=use_cuda_scatter)
            if self.aggregation == "volume":
                den = scatter_add_messages(w, dst, n,
                                           use_cuda_scatter=use_cuda_scatter)
                agg = agg / den.clamp_min(1e-12)

        h = h + self.update(agg)                                        # residual
        # `batch` matters only for GraphNorm, whose statistics are per graph;
        # LayerNorm/Identity ignore it (see norm._IgnoreBatch).
        return self.norm(h, batch, n_graphs)


class MessagePassingStack(nn.Module):
    def __init__(self, n_layers: int, latent_dim: int, edge_dim: int,
                 hidden: int, norm: str = "layernorm",
                 aggregation: str = "sum"):
        super().__init__()
        self.aggregation = aggregation
        self.layers = nn.ModuleList([
            MPLayer(latent_dim, edge_dim, hidden, norm, aggregation)
            for _ in range(n_layers)
        ])

    def forward(self, h, edge_index, e_latent, *, use_cuda_scatter: bool = True,
                batch=None, n_graphs=None, node_weight=None):
        for layer in self.layers:
            h = layer(h, edge_index, e_latent, use_cuda_scatter=use_cuda_scatter,
                      batch=batch, n_graphs=n_graphs, node_weight=node_weight)
        return h
