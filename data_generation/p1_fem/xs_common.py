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


def up_scatter_pairs(G: int) -> List[Tuple[int, int]]:
    """Ordered UP-scatter (g_from, g_to) pairs with g_from > g_to.

    Mirror image of scatter_pairs (same length G(G-1)/2), ordered
    (1->0, 2->0, 2->1, ...). Up-scatter is NOT part of the per-node XS row (that
    stays down-scatter-only, see module docstring); it is carried alongside and
    assembled at operator level for thermal cores (see materials_fhr.up_scatter_for
    / operators.assemble_AF), sourced from the upper triangle of the tallied
    nu-scatter matrix in the branch table.
    """
    return [(gf, gt) for gf in range(G) for gt in range(gf)]


def n_scatter(G: int) -> int:
    return G * (G - 1) // 2


def n_xs_cols(G: int) -> int:
    """Total per-node XS columns for G groups."""
    return 3 * G + n_scatter(G)


# --- group structure (SINGLE SOURCE OF TRUTH) --------------------------------
# Ascending group boundaries [eV] per reactor type at G=2. Everything that needs to
# know where the groups are split reads THIS: the OpenMC collapse (xs_openmc.py),
# the fission spectra CHI in materials.py / materials_fhr.py, the docstrings, and
# the per-sample metadata. Changing a boundary here changes the physics -- chi and
# the hand libraries must be re-derived with it (see the CHI comments).
#
#   hex (Natrium, FAST): split at 0.1 MeV. Both groups are fast; the cut sits near
#     the SFR flux peak (~100-200 keV), so both groups carry comparable flux, and
#     it is above the U-238 inelastic threshold (~45 keV) that dominates slowing
#     down in a sodium fast reactor. g2 therefore captures the slowing-down tail
#     where the capture-to-fission ratio climbs and control worth actually lives.
#     A cut near the U-238 fast-fission threshold (~0.8-1 MeV) would leave ~10-15%
#     of the flux in g1 and put the peak, most fission, and most slowing-down all
#     inside one enormous g2 -- a one-group model with a correction term.
#   fhr (KP-FHR, THERMAL): split at 0.625 eV. This is the classical cadmium cutoff,
#     and it is deliberately LOW for this reactor -- state that plainly rather than
#     citing the LWR convention as if it transferred. The cadmium edge suits water at
#     ~570 K; KP-FHR is graphite + FLiBe over 823-1100 K, where kT = 0.071-0.095 eV,
#     so 0.625 eV sits at only ~6.6-8.8 kT and cuts INSIDE the Maxwellian rather than
#     above its tail (~20-30 kT, i.e. ~1.9-2.5 eV).
#
#     The consequence is measured, not hypothetical: in xs_fhr.json the thermal->fast
#     up-scatter runs at 55-63% of down-scatter for the pebble and coolant materials,
#     and for graphite_pebble Ss21 is 93% of thermal absorption. A cut near 2 eV would
#     make cross-group up-scatter nearly vanish.
#
#     This is ACCEPTED, not overlooked, because the fhr model does not assume
#     down-scatter-only: the tallied Sigma_r already counts Ss21 as thermal
#     out-scatter (xs_openmc: removal = absorption + ALL out-scatter, down and up),
#     and geometry_pebble supplies Ss21 separately so operators.assemble_AF can add
#     the matching fast-group IN-scatter source to A. The full 2x2 scattering matrix
#     is solved exactly; the up-scatter is transported, not neglected. (The removal
#     side lives in Sr ONLY -- assemble_AF must not add Ss21 to removal again; that
#     double-count once made every fhr state subcritical by ~21,000 pcm. The node XS
#     ROW remains down-scatter-only -- up-scatter is operator-level; see
#     geometry_pebble's upscatter_note.)
#
#     If you ever raise this cut, chi is unaffected: the Watt spectrum below a few eV
#     is ~1e-10 of births, so materials_fhr.CHI stays (1.0, 0.0). What DOES change is
#     every collapsed constant, so xs_fhr.json must be regenerated in the same change.
GROUP_BOUNDARIES_EV = {
    "hex": [1.0e-5, 1.0e5, 2.0e7],
    "fhr": [1.0e-5, 0.625, 2.0e7],
}

# Physics-chosen FOUR-group structure for the hex (Natrium, FAST) reactor. Chosen to
# NEST inside the G=2 structure -- the 0.1 MeV cut is kept, so G=4 groups {1,2} sum
# to the old g1 and {3,4} to the old g2, and any G=2 vs G=4 bias comparison is a
# refinement of the same partition rather than a different one.
#
#   1.353 MeV  (10 MeV * e^-2, standard lethargy grid): U-238 fast-fission threshold
#              region. g1 isolates threshold fission + high-energy inelastic
#              scattering, where nu and chi are hardest.
#   0.1 MeV    the G=2 cut, retained (see the G=2 rationale above): SFR flux peak,
#              above the U-238 inelastic threshold.
#   9.1188 keV (10 MeV * e^-7): splits the slowing-down tail. g3 carries the
#              unresolved-resonance capture range; g4 the resolved resonances
#              (Na 2.85 keV, Fe/Cr structural) where the capture-to-fission ratio
#              climbs steepest and rod/reflector worth concentrates.
#
# chi for this structure is materials.CHI_BY_G[4], derived from the same Watt
# spectrum as the G=2 value (the two collapse consistently: g1+g2 = 0.9878 -> 0.99).
# fhr has no G=4 structure on purpose -- its +380 pcm G=2 bias needs no refinement
# (that contrast is the point of the study).
GROUP_BOUNDARIES_EV_G4 = {
    "hex": [1.0e-5, 9.1188e3, 1.0e5, 1.353e6, 2.0e7],
}


def group_boundaries_ev(reactor_type: str, G: int) -> List[float]:
    """Ascending group boundaries (length G+1). Uses the documented per-reactor
    cutoffs where a physics-chosen structure exists (G=2 both reactors, G=4 hex);
    log-spaced fill otherwise (a placeholder, not a physics choice -- add a
    structure above before collapsing a production table on it)."""
    if G == 2:
        return list(GROUP_BOUNDARIES_EV[reactor_type])
    if G == 4 and reactor_type in GROUP_BOUNDARIES_EV_G4:
        return list(GROUP_BOUNDARIES_EV_G4[reactor_type])
    import math
    lo, hi = 1.0e-5, 2.0e7
    lg = [math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / G for i in range(G + 1)]
    return [10.0 ** v for v in lg]


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


def blend_xs(follower: MultiGroupXS, absorber: MultiGroupXS, frac: float
             ) -> MultiGroupXS:
    """Gray-rod blend of a withdrawn (follower) and inserted (absorber) control XS.

    `frac` in [0,1] is the partial-insertion / rod-worth fraction: frac=0 -> pure
    follower, frac=1 -> pure absorber, between -> a partially-inserted control
    element represented as an axially-averaged gray absorber (the standard way to
    carry an axial insertion depth into a 2D radial model).

    Macroscopic cross sections (Sr, scatter, nuSf) blend LINEARLY -- volume/worth
    weighted homogenization of a partially-present absorber. The diffusion
    coefficient blends via its transport cross section Sigma_tr = 1/(3D): linear in
    Sigma_tr then inverted, which reduces to the worth-weighted HARMONIC mean of D.
    This is physically consistent (averaging D directly is not).

    The endpoints frac<=0 / frac>=1 short-circuit to the exact follower / absorber
    objects, so binary insertion stays byte-identical to the pre-gray-rod pipeline.
    """
    G = follower.n_groups
    if absorber.n_groups != G:
        raise ValueError("follower and absorber must share group count")
    t = float(min(max(frac, 0.0), 1.0))

    def lin(a, b):
        return tuple((1.0 - t) * ai + t * bi for ai, bi in zip(a, b))

    def blend_D(a, b):  # worth-weighted harmonic mean (linear in transport XS)
        return tuple(1.0 / ((1.0 - t) / ai + t / bi) for ai, bi in zip(a, b))

    return MultiGroupXS(
        D=blend_D(follower.D, absorber.D),
        Sr=lin(follower.Sr, absorber.Sr),
        scatter=lin(follower.scatter, absorber.scatter),
        nuSf=lin(follower.nuSf, absorber.nuSf),
    )
