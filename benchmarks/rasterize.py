"""Mesh <-> Cartesian grid operators for the FNO baseline, and the error floor
that comes with them.

FNO needs a uniform grid. A hex-duct lattice and an RSA-packed pebble bed are
not one, so the input field must be rasterized onto a grid and the prediction
interpolated back to mesh nodes before it can be scored. That round trip loses
information no matter how good the model is, and the loss is a property of the
GEOMETRY, not of FNO's fitting ability.

`interpolation_floor` measures exactly that: it pushes the REFERENCE flux to the
grid and pulls it back, and reports the resulting error. Every FNO number in the
benchmark is bounded below by it. Reporting the floor alongside the FNO result
is what makes "FNO does worse on irregular geometry" a measurement instead of a
rigged comparison -- without it a reader cannot tell whether FNO lost because of
its architecture or because of the resampling we imposed on it.

Directions use different schemes, on purpose:

  mesh -> grid   NEAREST. Material and cross-section fields are piecewise
                 constant per material; linearly blending across a duct/coolant
                 or pebble/FLiBe interface would invent cross sections belonging
                 to no material, destroying exactly the heterogeneity the model
                 is supposed to resolve.
  grid -> mesh   BILINEAR. Flux is a smooth field, so linear interpolation is
                 the right (and most generous to FNO) choice.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as Fn

import _paths  # noqa: F401
from sensors import Domain, lattice, to_unit, gather_per_graph


def mesh_to_grid(batch, values: torch.Tensor, domain: Domain, side: int
                 ) -> torch.Tensor:
    """[sumN, C] node field -> [B, C, side, side], nearest node per cell.

    The grid is indexed [H=y, W=x] to match torch's convention, consistent with
    sensors.lattice, which varies x fastest.
    """
    pts = lattice(domain, side, device=values.device, dtype=values.dtype)
    g = gather_per_graph(batch, pts, values)              # [B, side*side, C]
    return g.reshape(batch.n_graphs, side, side, -1).permute(0, 3, 1, 2)


def grid_to_mesh(batch, grid: torch.Tensor, domain: Domain) -> torch.Tensor:
    """[B, C, H, W] -> [sumN, C], bilinear sample at each node's own coordinate.

    Each graph samples only from its OWN grid slice; a node must never read a
    different core's field.
    """
    outs = []
    for i in range(batch.n_graphs):
        lo, hi = int(batch.ptr[i]), int(batch.ptr[i + 1])
        uv = to_unit(batch.coords[lo:hi], domain)          # [N_i, 2] in [-1,1]
        # grid_sample wants [B, H_out, W_out, 2] with (x, y) ordering
        samp = Fn.grid_sample(grid[i:i + 1], uv.view(1, -1, 1, 2),
                              mode="bilinear", padding_mode="border",
                              align_corners=True)          # [1, C, N_i, 1]
        outs.append(samp[0, :, :, 0].t())                  # [N_i, C]
    return torch.cat(outs, dim=0)


def interpolation_floor(batch, domain: Domain, side: int) -> dict:
    """Round-trip the REFERENCE flux through the grid and report the error.

    This is the best any FNO at this grid resolution could possibly do. Returns
    per-group relative L2, so it drops straight into the results table next to
    the model's own flux_rel_l2_g*.
    """
    ref = batch.flux
    back = grid_to_mesh(batch, mesh_to_grid(batch, ref, domain, side), domain)
    out = {}
    for g in range(ref.shape[1]):
        num = torch.linalg.vector_norm(back[:, g] - ref[:, g])
        den = torch.linalg.vector_norm(ref[:, g]).clamp_min(1e-12)
        out[f"floor_flux_rel_l2_g{g + 1}"] = float(num / den)
    out["floor_side"] = side
    return out
