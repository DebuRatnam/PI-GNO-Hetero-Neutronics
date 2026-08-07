"""MeshGraphNet baseline, using NVIDIA PhysicsNeMo's implementation.

    physicsnemo.models.meshgraphnet.MeshGraphNet(
        input_dim_nodes, input_dim_edges, output_dim,
        processor_size=15, hidden_dim_processor=128, ...)
    forward(node_features [N, Dn], edge_features [E, De], graph) -> [N, output_dim]

`graph` carries TOPOLOGY ONLY and is a torch_geometric.data.Data.

This is the most important baseline in the suite. The claim under test is that
GNNs and GNOs both handle irregular geometry but differ on resolution and
boundary-condition transfer, and MeshGraphNet is the canonical GNN for irregular
meshes (Pfaff et al., encode-process-decode). Using NVIDIA's implementation
rather than a hand-rolled GNN means the GNO-vs-GNN result is not self-refereed.

FAIRNESS, deliberately:

  * MeshGraphNet is given the SAME kNN message graph and the SAME 8-dimensional
    edge features PI-GNO gets. Not the FEM triangulation -- if the two models
    saw different graphs, a difference in accuracy would be attributable to the
    graph rather than to the architecture.
  * Its decoder emits a LATENT vector, and the same FluxHead / KHead PI-GNO uses
    are attached on top. So the comparison is processor-vs-processor with the
    heads held fixed, and MeshGraphNet is not penalised for lacking a graph-level
    k_eff readout, which it has no native notion of.
  * Power comes from the shared parameter-free PowerHead, as for every model.

What it structurally cannot do is transfer across resolution: its edge features
encode relative position at the training mesh spacing, and its aggregation is an
unweighted sum over a fixed-degree neighbourhood with no quadrature weight. That
is the prediction E3 tests.
"""

from __future__ import annotations

from typing import Tuple

import torch

import _paths  # noqa: F401
from features import NodeLayout, NormBundle
from heads import FluxHead, KHead

from batching import BatchedSample
from interface import BenchModel, register

try:
    from physicsnemo.models.meshgraphnet import MeshGraphNet as _MGN
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "the MeshGraphNet baseline needs NVIDIA PhysicsNeMo: pip install "
        "'nvidia-physicsnemo[cu12]' (python 3.11-3.13). Original error: {}"
        .format(e))

try:
    from torch_geometric.data import Data as _PyGData
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "PhysicsNeMo's MeshGraphNet takes torch_geometric.data.Data for "
        f"topology: pip install torch_geometric. Original error: {e}")


class MGNBench(BenchModel):
    def __init__(self, meta: dict, layout: NodeLayout, *,
                 hidden_dim: int = 128, processor_size: int = 6,
                 latent_dim: int = 64, head_hidden: int = 128,
                 k_pool: str = "mean", aggregation: str = "sum", **_):
        super().__init__(layout)
        self.G = layout.n_groups
        self.hidden_dim = hidden_dim
        self.processor_size = processor_size
        self.mgn = _MGN(
            input_dim_nodes=layout.total_dim,
            input_dim_edges=8,                 # the shared kNN edge features
            output_dim=latent_dim,             # latent, so shared heads apply
            processor_size=processor_size,
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            hidden_dim_edge_encoder=hidden_dim,
            hidden_dim_node_decoder=hidden_dim,
            aggregation=aggregation,
            mlp_activation_fn="silu",
        )
        self.flux_head = FluxHead(latent_dim, head_hidden, self.G)
        self.k_head = KHead(latent_dim, head_hidden, k_pool)

    def predict_norm(self, batch: BatchedSample, norm: NormBundle
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        # topology only; features are passed separately, per the API contract
        graph = _PyGData(edge_index=batch.edge_index, num_nodes=batch.n_nodes)
        h = self.mgn(norm.node.transform(batch.node_feats),
                     norm.edge.transform(batch.edge_feats),
                     graph)                                  # [sumN, latent]
        return (self.flux_head(h),
                self.k_head(h, batch.batch, batch.n_graphs))

    def describe(self) -> dict:
        d = super().describe()
        d.update(hidden_dim=self.hidden_dim, processor_size=self.processor_size)
        return d


@register("mgn")
def _build(meta, layout, **hp):
    return MGNBench(meta, layout, **hp)
