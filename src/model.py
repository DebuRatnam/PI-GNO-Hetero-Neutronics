"""PI-GNO model: lift -> stacked message passing -> projection heads.

Forward returns predictions in BOTH normalized (for L_flux/L_k) and physical (for
power + PDE residual) units. The caller supplies the NormBundle so de-norm is
exact and inverse-consistent.

Pipeline:
    node_feats(raw) --NodeLift--> h0
    edge_feats(raw) --EdgeLift--> e_latent
    h = MessagePassingStack(h0, edge_index, e_latent)     # SiLU messages
    flux_norm = FluxHead(h)                               # [N,2]
    k_norm    = KHead(h)                                  # scalar
    flux_phys = FluxScaler.inverse(flux_norm)
    k_phys    = k_norm * k_std + k_mean
    power     = PowerHead(flux_phys, raw_node_feats)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from lifting import NodeLift, EdgeLift
from message_passing import MessagePassingStack
from heads import FluxHead, KHead, PowerHead
from features import NodeLayout
from config import ModelConfig


@dataclass
class PIGNOOutput:
    flux_norm: torch.Tensor
    k_norm: torch.Tensor
    flux_phys: torch.Tensor
    k_phys: torch.Tensor
    power: torch.Tensor


class PIGNO(nn.Module):
    def __init__(self, cfg: ModelConfig, energy_per_fission_j: float):
        super().__init__()
        self.cfg = cfg
        self.node_lift = NodeLift(cfg.node_in_dim, cfg.latent_dim, cfg.message_hidden)
        self.edge_lift = EdgeLift(cfg.edge_in_dim, cfg.latent_dim, cfg.message_hidden)
        self.mp = MessagePassingStack(cfg.n_mp_layers, cfg.latent_dim,
                                      cfg.latent_dim, cfg.message_hidden, cfg.norm)
        self.flux_head = FluxHead(cfg.latent_dim, cfg.message_hidden, cfg.n_groups)
        self.k_head = KHead(cfg.latent_dim, cfg.message_hidden, cfg.k_pool)
        # nuSf column indices depend on material count + group count (schema-driven)
        layout = NodeLayout(cfg.n_materials, cfg.n_groups)
        self.power_head = PowerHead(energy_per_fission_j, layout.nusf_cols)

    def forward(self, *, node_feats_norm, edge_feats_norm, raw_node_feats,
                edge_index, flux_scaler, k_mean, k_std,
                use_cuda_scatter: bool = True,
                batch=None, n_graphs=None) -> PIGNOOutput:
        """`batch` [N] long / `n_graphs` are the batched-graph descriptors from
        benchmarks.batching.collate_graphs. Left None (the default) the model
        behaves exactly as before on a single graph and k is a 0-d scalar; given
        them, k is [n_graphs] and pooling respects graph boundaries."""
        h = self.node_lift(node_feats_norm)
        e = self.edge_lift(edge_feats_norm)
        h = self.mp(h, edge_index, e, use_cuda_scatter=use_cuda_scatter,
                    batch=batch, n_graphs=n_graphs)

        flux_norm = self.flux_head(h)               # [N,G]
        k_norm = self.k_head(h, batch, n_graphs)    # scalar, or [n_graphs]

        flux_phys = flux_scaler.inverse(flux_norm)
        k_phys = k_norm * k_std + k_mean
        power = self.power_head(flux_phys, raw_node_feats)
        return PIGNOOutput(flux_norm=flux_norm, k_norm=k_norm,
                           flux_phys=flux_phys, k_phys=k_phys, power=power)
