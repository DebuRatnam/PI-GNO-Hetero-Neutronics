"""Shared multigroup cross-section layout + container for the diffusion generator.

Generalizes the original fixed two-group (7-scalar) cross-section layout to a
configurable number of energy groups G, while staying BYTE-COMPATIBLE at G=2 so
the existing Natrium data / operators are unchanged.

Approximation: DOWN-SCATTER ONLY (no up-scatter) -- the standard multigroup
diffusion approximation. The scatter block stores the strictly-lower-triangular
(higher-energy g_from -> lower-energy g_to) transfer terms. Up-scatter (a small
correction that appears only deep in a thermal spectrum at fine group structure)
is neglected; document in metadata if a finer treatment is ever needed.

Per-node XS row layout for G groups
(length n_xs_cols(G) = 3G + G(G-1)/2):

    [ D_0..D_{G-1},          # group diffusion coefficients        [cm]
      Sr_0..Sr_{G-1},        # REMOVAL xs = absorption + out-scatter [1/cm]
      Ss(down-scatter pairs),# g_from -> g_to, g_from < g_to        [1/cm]
      nuSf_0..nuSf_{G-1} ]   # fission production                   [1/cm]

Down-scatter pairs are ordered (0->1, 0->2, ..., 0->G-1, 1->2, ..., G-2->G-1).

At G=2 this is exactly [D1, D2, Sr1, Sr2, Ss12, nuSf1, nuSf2] -- identical to the
original TwoGroupXS.as_row(). chi (fission spectrum) is NOT stored here: it is
per-fuel nuclear data carried globally in PhysicsConfig and applied in F.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import List, Tuple


def scatter_pairs(G: int) -> List[Tuple[int, int]]:
    """Ordered down-scatter (g_from, g_to) pairs with g_from < g_to."""
    return [(gf, gt) for gf in range(G) for gt in range(gf + 1, G)]


def n_scatter(G: int) -> int:
    return G * (G - 1) // 2


def n_xs_cols(G: int) -> int:
    """Total per-node XS columns for G groups."""
    return 3 * G + n_scatter(G)


def d_slice(G: int) -> slice:
    return slice(0, G)


def sr_slice(G: int) -> slice:
    return slice(G, 2 * G)


def scatter_slice(G: int) -> slice:
    return slice(2 * G, 2 * G + n_scatter(G))


def nusf_slice(G: int) -> slice:
    return slice(2 * G + n_scatter(G), 3 * G + n_scatter(G))


def xs_col_names(G: int) -> List[str]:
    """Human-readable names for the per-node XS columns, in row order. At G=2 this
    is [D1,D2,Sigma_r1,Sigma_r2,Sigma_s12,nuSigma_f1,nuSigma_f2] (unchanged)."""
    names = [f"D{g + 1}" for g in range(G)]
    names += [f"Sigma_r{g + 1}" for g in range(G)]
    names += [f"Sigma_s{gf + 1}{gt + 1}" for gf, gt in scatter_pairs(G)]
    names += [f"nuSigma_f{g + 1}" for g in range(G)]
    return names


def groups_from_n_xs_cols(ncols: int) -> int:
    """Invert n_xs_cols: recover G from a per-node XS width.

    Solves 3G + G(G-1)/2 = ncols  =>  G^2 + 5G - 2*ncols = 0.
    Used where G is not otherwise available (e.g. graph_build, which only holds
    the geometry). Raises if ncols is not a valid multigroup width.
    """
    G = (isqrt(25 + 8 * ncols) - 5) // 2
    if G < 1 or n_xs_cols(G) != ncols:
        raise ValueError(f"{ncols} is not a valid multigroup XS width (3G+G(G-1)/2)")
    return G


@dataclass(frozen=True)
class MultiGroupXS:
    """G-group cross sections for one material. Down-scatter only.

    D, Sr, nuSf are length-G tuples; scatter is length G(G-1)/2 in scatter_pairs
    order. Sr_g is the removal xs (absorption + total out-scatter from group g),
    matching the original convention so operators.py puts Sr on the diagonal and
    the scatter terms as off-diagonal in-scatter sources.
    """
    D: Tuple[float, ...]
    Sr: Tuple[float, ...]
    scatter: Tuple[float, ...]
    nuSf: Tuple[float, ...]

    def __post_init__(self):
        G = len(self.D)
        if len(self.Sr) != G or len(self.nuSf) != G:
            raise ValueError("D, Sr, nuSf must all have length G")
        if len(self.scatter) != n_scatter(G):
            raise ValueError(f"scatter must have length {n_scatter(G)} for G={G}")

    @property
    def n_groups(self) -> int:
        return len(self.D)

    def as_row(self) -> tuple:
        """Flatten to the canonical per-node XS row (see module docstring)."""
        return (*self.D, *self.Sr, *self.scatter, *self.nuSf)


def multigroupxs_to_dict(x: "MultiGroupXS") -> dict:
    return {"D": list(x.D), "Sr": list(x.Sr),
            "scatter": list(x.scatter), "nuSf": list(x.nuSf)}


def multigroupxs_from_dict(d: dict) -> "MultiGroupXS":
    return MultiGroupXS(D=tuple(d["D"]), Sr=tuple(d["Sr"]),
                        scatter=tuple(d["scatter"]), nuSf=tuple(d["nuSf"]))


def load_cached_library(path: str):
    """Load an OpenMC-generated XS cache (see xs_openmc.py) if present.

    Returns (library_dict{name -> MultiGroupXS}, blob_meta) or (None, None) when the
    file is absent -- so the material modules fall back to their committed defaults.
    """
    import os
    import json
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        blob = json.load(f)
    lib = {name: multigroupxs_from_dict(d) for name, d in blob["materials"].items()}
    return lib, blob


def two_group(D1, D2, Sigma_r1, Sigma_r2, Sigma_s12, nuSigma_f1, nuSigma_f2
              ) -> MultiGroupXS:
    """Backward-compatible constructor for the original 7-scalar two-group data.

    Produces a MultiGroupXS whose as_row() is exactly
    [D1, D2, Sr1, Sr2, Ss12, nuSf1, nuSf2].
    """
    return MultiGroupXS(
        D=(D1, D2), Sr=(Sigma_r1, Sigma_r2),
        scatter=(Sigma_s12,), nuSf=(nuSigma_f1, nuSigma_f2),
    )
