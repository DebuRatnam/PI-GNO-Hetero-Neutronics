"""DeepONet baseline (Lu et al. 2021), written here rather than imported.

PhysicsNeMo core does not ship DeepONet -- it lives in physicsnemo-sym, whose
constraint/geometry stack this benchmark does not use. The architecture is a
branch net, a trunk net and a dot product, so vendoring the dependency to get
~80 lines is the wrong trade.

    flux_g(x) = sum_p b_p^{(g)}(a) * t_p(x) + bias_g

  branch b(a): reads the input field at M FIXED sensor locations -> G x p
  trunk  t(x): reads a query coordinate -> p

WHAT THIS BASELINE CAN AND CANNOT DO, which is the point of including it:

  * the TRUNK takes arbitrary query points, so DeepONet evaluates directly at
    mesh nodes with no interpolation error, and transfers to a refined mesh for
    free on the output side. FNO cannot do the first; MeshGraphNet cannot do
    the second.
  * the BRANCH is tied to a fixed sensor set on a fixed domain. It cannot read a
    different geometry, a different resolution, or a different core extent
    except through whatever those sensors happen to capture. That is the honest
    limitation, and it is why DeepONet is "partial" rather than "yes" on the
    resolution axis in the results table.

Sensors use nearest-neighbour, not interpolation: cross sections are piecewise
constant per material and linear blending across an interface would invent
materials that do not exist.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import _paths  # noqa: F401
from features import NodeLayout, NormBundle
from lifting import mlp

from batching import BatchedSample
from interface import BenchModel, register
from sensors import Domain, lattice, to_unit, gather_per_graph


def _deep_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Module:
    """SiLU MLP of arbitrary depth. `lifting.mlp` is fixed at one hidden layer;
    DeepONet needs deeper branch/trunk nets to be a fair baseline."""
    if depth <= 1:
        return mlp(in_dim, hidden, out_dim)
    layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class DeepONetBench(BenchModel):
    def __init__(self, meta: dict, layout: NodeLayout, *,
                 domain: Domain = None, sensor_side: int = 24,
                 basis: int = 96, branch_hidden: int = 256,
                 trunk_hidden: int = 128, branch_depth: int = 3,
                 trunk_depth: int = 4, **_):
        super().__init__(layout)
        if domain is None:
            raise ValueError(
                "DeepONet needs a fixed domain; the harness derives it from the "
                "train split (sensors.domain_from_dataset) and freezes it into "
                "the run config. Re-siting sensors per sample would hand the "
                "baseline a geometry adaptation it does not have.")
        self.domain = tuple(domain)
        self.sensor_side = sensor_side
        self.G = layout.n_groups
        self.basis = basis

        # sensors are a fixed buffer: they move with the model to the device and
        # are saved in the checkpoint, so an evaluation cannot silently use a
        # different sensor set than the one trained on
        self.register_buffer("sensors", lattice(self.domain, sensor_side))
        m = sensor_side * sensor_side

        # the branch reads the FIELD, not position: x and y are the trunk's job
        self.field_cols = [c for c in range(layout.total_dim) if c not in (0, 1)]
        c_in = len(self.field_cols)

        self.branch = _deep_mlp(m * c_in, branch_hidden,
                                self.G * basis, branch_depth)
        self.trunk = _deep_mlp(2, trunk_hidden, basis, trunk_depth)
        self.bias = nn.Parameter(torch.zeros(self.G))
        # k_eff is graph-wise; it reads the branch code, which is the only
        # whole-core representation DeepONet forms
        self.k_net = _deep_mlp(self.G * basis, branch_hidden, 1, 2)

    def predict_norm(self, batch: BatchedSample, norm: NormBundle
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        nf = norm.node.transform(batch.node_feats)             # [sumN, D]
        field = nf[:, self.field_cols]                          # [sumN, C]

        sens = gather_per_graph(batch, self.sensors, field)     # [B, M, C]
        code = self.branch(sens.reshape(batch.n_graphs, -1))    # [B, G*p]

        t = self.trunk(to_unit(batch.coords, self.domain))      # [sumN, p]
        # pair each node with its own graph's branch code
        b = code.reshape(batch.n_graphs, self.G, self.basis)[batch.batch]
        flux_norm = (b * t.unsqueeze(1)).sum(-1) + self.bias    # [sumN, G]

        k_norm = self.k_net(code).squeeze(-1)                   # [B]
        return flux_norm, k_norm

    def describe(self) -> dict:
        d = super().describe()
        d.update(sensor_side=self.sensor_side,
                 n_sensors=self.sensor_side ** 2, basis=self.basis,
                 domain=list(self.domain))
        return d


@register("deeponet")
def _build(meta, layout, **hp):
    return DeepONetBench(meta, layout, **hp)
