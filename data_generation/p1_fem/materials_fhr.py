"""Material schema and cross-section lookup for a Kairos KP-FHR pebble bed (THERMAL).

Parallel to materials.py (Natrium fast core). Exposes the SAME public symbols
(MATERIAL_IDS, N_MATERIALS, one_hot_batch, xs_for, xs_for_id, CHI,
AXIAL_BUCKLING_CM2, BRANCH*) so the generator swaps reactor type by importing this
module instead.

THERE IS NO HAND-TUNED LIBRARY. Every constant is flux-weighted from
continuous-energy ENDF/B data by xs_openmc.py, tallied in situ in the assembled full
core with explicit TRISO inside explicit pebbles (double heterogeneity preserved), at
a (burnup, temperature, rod) branch. Absent table -> import fails loudly; no
representative constants are substituted. See materials.py for why.

KP-FHR is a graphite + FLiBe thermally-moderated system, so group 2 is a genuine
THERMAL group (unlike the Natrium slow-fast group), split at 0.625 eV
(xs_common.GROUP_BOUNDARIES_EV["fhr"]). The collapsed constants reflect this
directly: fission dominated by the thermal group (nuSf2 >> nuSf1, opposite of the
fast core); strong fast->thermal down-scatter Ss12 in graphite, reflector, and FLiBe;
B4C elements as strong thermal absorbers because B-10 (n,alpha) rises steeply toward
thermal energies; and a modest FLiBe thermal absorption from Li-6 capture.

Per material the table supplies, per group:
  D_g diffusion, 1/(3*Sigma_tr_g) [cm]; Sigma_r_g removal (absorption + total
  out-scatter) [1/cm]; Sigma_s12 fast->thermal down-scatter [1/cm]; nuSigma_f_g
  fission production [1/cm]; plus the UPPER triangle of the scatter matrix, which is
  where thermal up-scatter comes from (see UPSCATTER / up_scatter_for below).

What still lives here rather than in the table: CHI (analytic for this group
boundary, superseded per-sample by the tallied chi) and AXIAL_BUCKLING_CM2
(geometric; the transport models are axially reflective by design).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from xs_branch import require_branch_table
from xs_common import GROUP_BOUNDARIES_EV, MultiGroupXS


# Canonical material id mapping for the KP-FHR core (7-way one-hot).
#   fuel_pebble       TRISO-in-graphite fuel pebble (homogenized), thermal-fissile
#   graphite_pebble   moderator-only pebble (no fission)
#   control_element   B4C, 4 cylinders on the OUTER edge (insertion toggles XS)
#   shutdown_element  B4C, 3 X-shapes in the INNER bed (insertion toggles XS)
#   reflector         graphite reflector annulus (60 cm, gFHR)
#   coolant           FLiBe salt (interstitial + gaps)
#   vessel            SS316H core barrel + FLiBe downcomer + SS316H reactor vessel,
#                     homogenized into one ring (parasitic capture, no fission).
#                     316H is the alloy Kairos qualified for the KP-FHR -- NOT
#                     Hastelloy-N, whose Ni/Mo content captures very differently.
MATERIAL_IDS: Dict[str, int] = {
    "fuel_pebble": 0,
    "graphite_pebble": 1,
    "control_element": 2,
    "shutdown_element": 3,
    "reflector": 4,
    "coolant": 5,
    "vessel": 6,
}
ID_TO_MATERIAL = {v: k for k, v in MATERIAL_IDS.items()}
MATERIAL_ORDER: List[str] = [ID_TO_MATERIAL[i] for i in range(len(MATERIAL_IDS))]
N_MATERIALS: int = len(MATERIAL_IDS)

# Convenience groupings.
FUEL_IDS = (MATERIAL_IDS["fuel_pebble"],)
CONTROL_IDS = (MATERIAL_IDS["control_element"], MATERIAL_IDS["shutdown_element"])


def one_hot_batch(material_state: np.ndarray) -> np.ndarray:
    """[N] int material ids -> [N, N_MATERIALS] one-hot float matrix."""
    ids = np.asarray(material_state, dtype=np.int64)
    oh = np.zeros((ids.shape[0], N_MATERIALS), dtype=np.float64)
    oh[np.arange(ids.shape[0]), ids] = 1.0
    return oh


# Number of energy groups this reactor's schema is built for; verified against the
# branch table at load time (require_branch_table).
N_GROUPS: int = 2

# Fission spectrum chi for the KP-FHR THERMAL core (G=2). The group boundary is
# xs_common.GROUP_BOUNDARIES_EV["fhr"][1] = 0.625 eV -- the SAME cut the OpenMC
# collapse uses. The Watt spectrum integrated below 0.625 eV is ~2e-10 of the total,
# so chi_2 is zero to every digit that matters: fission neutrons are born ~2 MeV and
# reach the thermal group only by moderating down (Sigma_s12), never by birth.
#
# Contrast materials.CHI = (0.99, 0.01) for the Natrium fast core: the same physics
# (births are fast) lands on a different split purely because that cut sits at
# 0.1 MeV, inside the tail of the birth spectrum, instead of 12 decades below it.
# Do not copy one reactor's chi to the other.
CHI = (1.0, 0.0)

# Transverse (axial) leakage buckling Bz^2 [1/cm^2]. Bz^2 = (pi / H_extrap)^2 with the
# published gFHR active bed height 309.47 cm plus a short extrapolation distance ->
# ~(pi/314)^2. Smaller than the fast core: a taller core leaks less axially per unit
# height. (Axially the real core is reflected by graphite, so the true extrapolated
# height is longer and this is a conservative upper bound on axial leakage.)
AXIAL_BUCKLING_CM2 = 1.0e-4


# --- OpenMC branch table (REQUIRED) ------------------------------------------
# The only source of cross sections; see materials.py for the rationale and for why
# the load is LAZY rather than at import (xs_openmc.py must import this module to
# build the table, so an import-time requirement would deadlock the bootstrap).
# Control and shutdown insertion blend between the table's rod-out
# ("control_follower", a FLiBe-filled channel) and rod-in (B4C absorber) branches,
# both measured.

BRANCH_TABLE_PATH = os.path.join(os.path.dirname(__file__), "xs_fhr.json")

_BRANCH = None


def branch():
    """The loaded branch table, or raise xs_branch.MissingBranchTable."""
    global _BRANCH
    if _BRANCH is None:
        _BRANCH = require_branch_table(
            BRANCH_TABLE_PATH, reactor_type="fhr", expect_n_groups=N_GROUPS,
            expect_boundaries_ev=GROUP_BOUNDARIES_EV["fhr"])
    return _BRANCH


def branch_metadata() -> dict:
    """Provenance block for geometry_metadata (transport code, data library, branch
    grid, per-branch k_eff, Monte Carlo uncertainty)."""
    return branch().metadata()


def branch_chi(*, burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
               core_rod_frac: float = 0.0):
    """TALLIED core-average fission spectrum at this state, or None if the table
    carries no chi (callers then use the analytic CHI above)."""
    return branch().chi(burnup_mwd_kg=burnup_mwd_kg, temperature_k=temperature_k,
                        core_rod_frac=core_rod_frac)


def xs_for(material: str, *, inserted: bool = True,
           insert_frac: float | None = None,
           burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
           core_rod_frac: float = 0.0) -> MultiGroupXS:
    """Transport-collapsed cross sections for a material at a core state.

    (burnup_mwd_kg, temperature_k, core_rod_frac) select an interpolated branch case:
    burnt isotopics (including the Xe/Sm poison the old `burnup_poison_coeff` stood
    in for), Doppler + S(alpha,beta) temperature effects, and the rodded/unrodded
    spectrum are all measured.

    For control/shutdown labels, insertion is a gray-rod depth in [0,1]: pass
    `insert_frac` for a partially-inserted element (0 = withdrawn FLiBe follower,
    1 = full B4C absorber, between = axially-averaged blend between the two branch
    endpoints; see xs_common.blend_xs). `inserted` is the legacy binary switch, used
    only when `insert_frac` is None. Non-control materials ignore both and instead
    see `core_rod_frac`.

    Raises KeyError if the table has no record for `material`.
    """
    frac = (1.0 if inserted else 0.0) if insert_frac is None else float(insert_frac)

    table = branch()
    got = table.lookup(material, burnup_mwd_kg=burnup_mwd_kg,
                       temperature_k=temperature_k, insert_frac=frac,
                       core_rod_frac=core_rod_frac)
    if got is None:
        raise KeyError(
            f"{material!r} has no record in {os.path.basename(BRANCH_TABLE_PATH)} "
            f"(tabulated: {', '.join(table.materials)}). Every FEM material must be "
            f"tallied as an OpenMC domain -- check the homogenize map in "
            f"openmc_models.fhr_model.")
    return got[0]


def xs_for_id(material_id: int, *, inserted: bool = True,
              insert_frac: float | None = None,
              burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
              core_rod_frac: float = 0.0) -> MultiGroupXS:
    return xs_for(ID_TO_MATERIAL[material_id], inserted=inserted,
                  insert_frac=insert_frac, burnup_mwd_kg=burnup_mwd_kg,
                  temperature_k=temperature_k, core_rod_frac=core_rod_frac)


# Thermal UP-scatter Ss_{g2->g1} [1/cm] (thermal -> epithermal) is a real effect in
# graphite/FLiBe at KP-FHR temperature (~650 C) and negligible in
# fuel/absorber/vessel. It comes from the UPPER TRIANGLE of the tallied nu-scatter
# matrix -- a measured transfer rate at the branch temperature, which is the only
# defensible way to get it, since up-scatter is precisely a thermal-motion
# (S(alpha,beta)) effect and therefore cannot be a temperature-independent constant.
# The former hand-tuned UPSCATTER_21 table was removed with the rest of the fallback.
#
# Applied at the OPERATOR level (assembled into A: raises thermal removal + adds an
# in-scatter source to the fast group) rather than stored in the per-node XS row,
# which stays down-scatter-only (schema unchanged).


def up_scatter_for(material: str, *, insert_frac: float = 1.0,
                   burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
                   core_rod_frac: float = 0.0) -> float:
    """Tallied thermal->fast up-scatter Ss21 [1/cm] for a material at a core state.

    Returns the G=2 up-scatter scalar. For G>2 the full up-scatter block lives in
    the branch table; operators.assemble_AF consumes the g2->g1 term only, matching
    the two-group operator-level treatment documented above.

    A material whose tallied upper triangle is genuinely zero (absorbers, vessel)
    returns 0.0. A material MISSING from the table raises, via xs_for.
    """
    got = branch().lookup(material, burnup_mwd_kg=burnup_mwd_kg,
                          temperature_k=temperature_k, insert_frac=insert_frac,
                          core_rod_frac=core_rod_frac)
    if got is None:
        raise KeyError(
            f"{material!r} has no record in {os.path.basename(BRANCH_TABLE_PATH)}; "
            f"cannot resolve thermal up-scatter.")
    return float(got[1][0]) if len(got[1]) else 0.0


def up_scatter_for_id(material_id: int, *, insert_frac: float = 1.0,
                      burnup_mwd_kg: float = 0.0,
                      temperature_k: float | None = None,
                      core_rod_frac: float = 0.0) -> float:
    """Thermal->fast up-scatter Ss21 [1/cm] for a material id (0 if none)."""
    return up_scatter_for(ID_TO_MATERIAL[material_id], insert_frac=insert_frac,
                          burnup_mwd_kg=burnup_mwd_kg, temperature_k=temperature_k,
                          core_rod_frac=core_rod_frac)
