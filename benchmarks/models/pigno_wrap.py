"""PI-GNO behind the BenchModel interface.

This is a thin adapter, not a reimplementation: it holds the real src/model.py
PIGNO so the benchmarked model is literally the model under study. It supplies
its own flux/k path and inherits the shared PowerHead from BenchModel, which is
the same parameter-free head every baseline gets.
"""

from __future__ import annotations

from typing import Tuple

import torch

import _paths  # noqa: F401
from config import ModelConfig
from features import NodeLayout, NormBundle
from lifting import NodeLift, EdgeLift
from message_passing import MessagePassingStack
from heads import FluxHead, KHead

from batching import BatchedSample
from interface import BenchModel, register


class PIGNOBench(BenchModel):
    def __init__(self, meta: dict, layout: NodeLayout, *,
                 latent_dim: int = 64, n_mp_layers: int = 6,
                 message_hidden: int = 128, norm: str = "layernorm",
                 k_pool: str = "mean", use_cuda_scatter: bool = True,
                 aggregation: str = "sum", **_):
        super().__init__(layout)
        cfg = ModelConfig.from_metadata(
            meta, latent_dim=latent_dim, n_mp_layers=n_mp_layers,
            message_hidden=message_hidden, norm=norm, k_pool=k_pool,
            aggregation=aggregation)
        self.cfg = cfg
        self.use_cuda_scatter = use_cuda_scatter
        self.aggregation = aggregation
        self.node_lift = NodeLift(cfg.node_in_dim, cfg.latent_dim, cfg.message_hidden)
        self.edge_lift = EdgeLift(cfg.edge_in_dim, cfg.latent_dim, cfg.message_hidden)
        self.mp = MessagePassingStack(cfg.n_mp_layers, cfg.latent_dim,
                                      cfg.latent_dim, cfg.message_hidden, cfg.norm,
                                      cfg.aggregation)
        self.flux_head = FluxHead(cfg.latent_dim, cfg.message_hidden, cfg.n_groups)
        self.k_head = KHead(cfg.latent_dim, cfg.message_hidden, cfg.k_pool)

    def predict_norm(self, batch: BatchedSample, norm: NormBundle
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_lift(norm.node.transform(batch.node_feats))
        e = self.edge_lift(norm.edge.transform(batch.edge_feats))
        h = self.mp(h, batch.edge_index, e,
                    use_cuda_scatter=self.use_cuda_scatter,
                    batch=batch.batch, n_graphs=batch.n_graphs,
                    node_weight=batch.nodal_volume)
        return (self.flux_head(h),
                self.k_head(h, batch.batch, batch.n_graphs))

    def describe(self) -> dict:
        d = super().describe()
        d.update(latent_dim=self.cfg.latent_dim, n_mp_layers=self.cfg.n_mp_layers,
                 message_hidden=self.cfg.message_hidden, norm=self.cfg.norm,
                 k_pool=self.cfg.k_pool, aggregation=self.aggregation)
        return d


@register("pigno")
def _build(meta, layout, **hp):
    return PIGNOBench(meta, layout, **hp)
