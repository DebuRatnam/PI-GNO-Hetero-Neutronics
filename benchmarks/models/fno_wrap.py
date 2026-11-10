"""FNO baseline, using NVIDIA PhysicsNeMo's implementation.

    physicsnemo.models.fno.FNO(in_channels, out_channels, dimension=2,
                               latent_channels=32, num_fno_layers=4,
                               num_fno_modes=16, ...)
    forward(x: [B, C_in, H, W]) -> [B, C_out, H, W]

Imported rather than reimplemented so the baseline is NVIDIA-maintained and
citable, and so "you wrote a weak FNO" is not an available objection.

THE HONEST FRAMING. It is often said that FNO cannot handle varying resolution.
That is false and a reviewer will say so: FNO learns a kernel in Fourier space
and is genuinely resolution-flexible in function space, which is its central
claim. Its real limitation on this problem is different and narrower: it needs a
uniform Cartesian grid, and a hex-duct lattice or an RSA-packed pebble bed is
not one. So the input is rasterized and the output interpolated back, and the
cost of that round trip is measured separately as the interpolation floor
(rasterize.interpolation_floor) and reported next to every FNO result.

k_eff is graph-wise, so it is pooled from the final grid feature map. Flux is
returned at MESH NODES, like every other model in the suite -- scoring FNO on
its own grid would hide precisely the error this baseline is here to expose.
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
from rasterize import grid_to_mesh, mesh_to_grid
from sensors import Domain

try:
    from physicsnemo.models.fno import FNO as _FNO
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "the FNO baseline needs NVIDIA PhysicsNeMo: pip install "
        "'nvidia-physicsnemo[cu12]' (python 3.11-3.13; it cannot be installed "
        f"in the local 3.9 training env). Original error: {e}")


class FNOBench(BenchModel):
    def __init__(self, meta: dict, layout: NodeLayout, *,
                 domain: Domain = None, side: int = 96,
                 latent_channels: int = 32, num_fno_layers: int = 4,
                 num_fno_modes: int = 16, decoder_layers: int = 2,
                 decoder_layer_size: int = 64, k_hidden: int = 128, **_):
        super().__init__(layout)
        if domain is None:
            raise ValueError(
                "FNO needs a fixed domain; the harness derives it from the "
                "train split and freezes it into the run config.")
        self.domain = tuple(domain)
        self.side = side
        self.G = layout.n_groups

        # the grid carries the FIELD; FNO adds its own coordinate features
        # (coord_features=True), so x and y are dropped from the input channels
        self.field_cols = [c for c in range(layout.total_dim) if c not in (0, 1)]

        self.fno = _FNO(
            in_channels=len(self.field_cols),
            out_channels=self.G,
            dimension=2,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            activation_fn="gelu",
            coord_features=True,
        )
        self.k_net = mlp(self.G, k_hidden, 1)
        self.latent_channels = latent_channels
        self.num_fno_modes = num_fno_modes

    def predict_norm(self, batch: BatchedSample, norm: NormBundle
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        nf = norm.node.transform(batch.node_feats)
        x = mesh_to_grid(batch, nf[:, self.field_cols], self.domain, self.side)
        y = self.fno(x)                                      # [B, G, H, W]
        flux_norm = grid_to_mesh(batch, y, self.domain)      # [sumN, G]
        k_norm = self.k_net(y.mean(dim=(2, 3))).squeeze(-1)  # [B]
        return flux_norm, k_norm

    def describe(self) -> dict:
        d = super().describe()
        d.update(side=self.side, latent_channels=self.latent_channels,
                 num_fno_modes=self.num_fno_modes, domain=list(self.domain))
        return d


@register("fno")
def _build(meta, layout, **hp):
    return FNOBench(meta, layout, **hp)
