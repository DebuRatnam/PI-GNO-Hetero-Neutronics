"""Material schema and cross-section lookup for a Natrium-inspired sodium fast reactor.

This module owns the material identity of the `hex` core -- the id mapping, the
one-hot encoding, and the fixed nuclear data that transport cannot supply -- and
resolves per-node cross sections out of the OpenMC branch table `xs_natrium.json`.

THERE IS NO HAND-TUNED LIBRARY. Every D, Sigma_r, scatter, and nuSigma_f is
flux-weighted from continuous-energy ENDF/B data by xs_openmc.py, tallied in situ in
the assembled full core, at a (burnup, temperature, rod) branch. If the table is
missing the module raises at import (see xs_branch.require_branch_table); it does
not substitute representative constants. That is the point: hand-tuned numbers are
indistinguishable from physics once they are in a dataset, and a reviewer cannot
audit what is not traceable to an evaluation.

FAST-SPECTRUM data: BOTH groups are fast, split at 0.1 MeV
(xs_common.GROUP_BOUNDARIES_EV["hex"]) -- g1 = high-fast 0.1-20 MeV,
g2 = slow-fast 1e-5 eV-0.1 MeV. g2 spans the slowing-down tail, but a sodium fast
reactor has no thermalized population in it: there is no moderator, so the flux
there is a steep 1/E-like tail, not a Maxwellian. The collapsed constants show this
directly -- similar D in both groups (long fast mean free path), removal dominated
by inelastic/elastic down-scatter rather than capture, fission in both groups, and
absorbers far weaker than in a moderated spectrum because the B-10 (n,alpha) cross
section falls steeply with energy.

Per material, per group g in {1=high-E fast, 2=low-E fast}, the table supplies:
  D_g        diffusion coefficient, 1/(3*Sigma_tr_g)   [cm]
  Sigma_r_g  removal = absorption + total out-scatter  [1/cm]
  Sigma_s12  g1->g2 down-scatter                       [1/cm]
  nuSigma_f_g  fission production                      [1/cm]

Sigma_r1 INCLUDES Sigma_s12 (removal = absorption + out-scatter). The assembly in
operators.py uses Sigma_r1 on the group-1 diagonal and Sigma_s12 as the off-diagonal
down-scatter source into group 2.

What still lives here rather than in the table: CHI (derived analytically from the
fission spectrum for this group boundary, and superseded per-sample by the TALLIED
chi whenever the table carries one) and AXIAL_BUCKLING_CM2 (a geometric quantity --
the transport models are axially reflective by design, so Bz^2 cannot come from
them; see CLAUDE.md on axial leakage entering exactly once).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from xs_branch import require_branch_table
from xs_common import GROUP_BOUNDARIES_EV, MultiGroupXS


# Canonical material id mapping (kept stable; embedded in metadata).
# 8-way set matching the Natrium core map + explicit HT9 duct steel. material_state
# is stored as these integer ids per node, but NODE FEATURES use a ONE-HOT encoding
# (see one_hot_batch) so the model sees no artificial ordinal relationship.
#   fuel_inner / fuel_outer      -> inner vs outer enrichment zones (OUTER is the
#                                   HIGHER-enrichment zone; see openmc_models.py)
#   primary_control / secondary_control -> 9 primary + 4 secondary B4C control
#                                     assemblies (NRC-docketed Natrium counts); the
#                                     37/19 absorber-pin lattices in openmc_models are
#                                     representative, not vendor numbers. XS toggle
#                                     absorber (inserted) vs sodium follower (out)
#   reflector / shield           -> outer radial rings
#   duct                         -> HT9 duct wall + inter-assembly Na gap (homogenized)
#   coolant                      -> sodium (assembly interiors / gaps)
MATERIAL_IDS: Dict[str, int] = {
    "fuel_inner": 0,
    "fuel_outer": 1,
    "primary_control": 2,
    "secondary_control": 3,
    "reflector": 4,
    "shield": 5,
    "duct": 6,
    "coolant": 7,
}
ID_TO_MATERIAL = {v: k for k, v in MATERIAL_IDS.items()}

# Fixed column order of the one-hot material block (index == material id).
MATERIAL_ORDER: List[str] = [ID_TO_MATERIAL[i] for i in range(len(MATERIAL_IDS))]
N_MATERIALS: int = len(MATERIAL_IDS)

# Convenience groupings.
FUEL_IDS = (MATERIAL_IDS["fuel_inner"], MATERIAL_IDS["fuel_outer"])
CONTROL_IDS = (MATERIAL_IDS["primary_control"], MATERIAL_IDS["secondary_control"])


def one_hot_batch(material_state: np.ndarray) -> np.ndarray:
    """[N] int material ids -> [N, N_MATERIALS] one-hot float matrix.

    Column j is 1 where material_state == j (j follows MATERIAL_ORDER). Removes the
    spurious ordinal signal between unrelated materials.
    """
    ids = np.asarray(material_state, dtype=np.int64)
    oh = np.zeros((ids.shape[0], N_MATERIALS), dtype=np.float64)
    oh[np.arange(ids.shape[0]), ids] = 1.0
    return oh


# Number of energy groups this reactor's schema is built for. The branch table is
# verified against it at load time (require_branch_table), so it is also the width
# the node-feature XS block and the fission spectrum below must agree on.
N_GROUPS: int = 2

# Fission spectrum chi for the Natrium FAST core (G=2). The group boundary is
# xs_common.GROUP_BOUNDARIES_EV["hex"][1] = 1.0e5 eV (0.1 MeV) -- the SAME cut the
# OpenMC collapse uses, so chi and the collapsed constants describe one structure.
#
# Integrating a Watt spectrum exp(-E/a) sinh(sqrt(bE)) with the fast-induced
# U-235/Pu-239 parameters (a=0.966 MeV, b=2.842 /MeV) gives 98.78% of births above
# 0.1 MeV; U-238 fast fission is marginally harder still, and delayed neutrons
# (beta_eff ~ 0.0035) shift <0.05% into g2. Rounded: (0.99, 0.01).
#
# This is NOT the same statement as "chi ~ (0.95,0.05) because the spectrum is
# fast". At 0.1 MeV essentially nothing is BORN in g2 -- g2 is populated by
# down-scatter (Sigma_s12: U-238/Fe inelastic + sodium elastic), which is exactly
# the decomposition two-group diffusion assumes. An earlier revision carried
# (0.60,0.40), correct for a 0.8 MeV cut that this code never actually used.
CHI = (0.99, 0.01)

# Transverse (axial) leakage buckling Bz^2 [1/cm^2]. Bz^2 = (pi / H_extrap)^2 with an
# SFR active height ~100 cm plus a few-cm extrapolation length -> ~(pi/104)^2.
AXIAL_BUCKLING_CM2 = 9.1e-4


# --- OpenMC branch table (REQUIRED) ------------------------------------------
# The only source of cross sections. Constants are looked up per (burnup,
# temperature, control insertion) from transport-collapsed group constants; there is
# no hand library to fall back to and no ad-hoc burnup/temperature multiplier. The
# loader verifies reactor type, group count, and group boundaries, and raises
# xs_branch.MissingBranchTable with the generating command if any of that is wrong.
#
# Control insertion needs no follower constant here: the table stores rod-out
# ("control_follower") and rod-in (B4C absorber) endpoints as separate branches, and
# BranchTable.lookup blends between them on the insertion axis. Both endpoints are
# measured, which is the whole reason the gray-rod blend is now defensible.
#
# The load is LAZY, on first cross-section access, and deliberately so. xs_openmc.py
# has to import openmc_models -> geometry -> this module in order to BUILD the table,
# so requiring it at import time would deadlock the bootstrap: the table could never
# be generated because generating it needs the table. Everything above this line (the
# material id schema, the one-hot encoding, CHI, Bz^2) is table-independent and stays
# importable; the first xs_for/xs_for_id/branch_metadata call is where a missing
# table becomes an error. Nothing can silently proceed without one.

BRANCH_TABLE_PATH = os.path.join(os.path.dirname(__file__), "xs_natrium.json")

_BRANCH = None


def branch():
    """The loaded branch table, or raise xs_branch.MissingBranchTable."""
    global _BRANCH
    if _BRANCH is None:
        _BRANCH = require_branch_table(
            BRANCH_TABLE_PATH, reactor_type="hex", expect_n_groups=N_GROUPS,
            expect_boundaries_ev=GROUP_BOUNDARIES_EV["hex"])
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

    (burnup_mwd_kg, temperature_k, core_rod_frac) select an interpolated branch
    case -- burnt isotopics, Doppler-broadened data, and the rodded/unrodded spectrum
    are all measured quantities, not multipliers on a nominal library.

    For control labels, insertion is a gray-rod depth in [0,1]: pass `insert_frac`
    for a partially-inserted rod (0 = withdrawn sodium follower, 1 = full absorber,
    between = axially-averaged blend between the two branch endpoints; see
    xs_common.blend_xs). `inserted` is the legacy binary switch, used only when
    `insert_frac` is None (True -> 1, False -> 0). Non-control materials ignore both
    and instead see `core_rod_frac`, the spectrum shift absorbers elsewhere impose.

    Raises KeyError if the table has no record for `material` -- previously this fell
    through to the hand library, which is exactly the silent downgrade this module no
    longer permits.
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
            f"openmc_models.natrium_model.")
    return got[0]


def xs_for_id(material_id: int, *, inserted: bool = True,
              insert_frac: float | None = None,
              burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
              core_rod_frac: float = 0.0) -> MultiGroupXS:
    return xs_for(ID_TO_MATERIAL[material_id], inserted=inserted,
                  insert_frac=insert_frac, burnup_mwd_kg=burnup_mwd_kg,
                  temperature_k=temperature_k, core_rod_frac=core_rod_frac)
