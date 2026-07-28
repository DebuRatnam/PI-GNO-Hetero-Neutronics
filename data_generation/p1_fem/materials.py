"""Two-group cross-section library for a Natrium-inspired sodium fast reactor.

Materials: fuel, control_rod, reflector, shield, coolant.

FAST-SPECTRUM data: BOTH groups are fast (g1 = high-fast ~0.8-10 MeV,
g2 = slow-fast ~1 keV-0.8 MeV). Both groups sit in the fast spectrum — a sodium
fast reactor has no moderated/slowed-down neutron population. Consequences baked
into these numbers:

  - Large, similar diffusion coefficients in both groups (fast neutrons stream
    far; D2 is close to D1, not much smaller as in a moderated lower group).
  - Small absorption/removal cross sections; removal is dominated by elastic
    DOWN-SCATTER (slowing within the fast range), not capture.
  - Fission occurs in BOTH groups at the SAME order of magnitude (fast fission).
    nuSf1 ~ nuSf2 — emphatically NOT nuSf2 >> nuSf1 (that would be a moderated
    spectrum).
  - Absorbers (B4C control rod / shield) are far weaker than in a moderated
    spectrum: the B-10 (n,alpha) cross section falls steeply with energy.

Values are representative order-of-magnitude fast constants, NOT benchmarked
against an evaluation. Replace with lattice-code data when available; assembly
and solver do not depend on these specific numbers.

Per material we store, per group g in {1=high-E fast, 2=low-E fast}:
  D_g        diffusion coefficient            [cm]
  Sigma_r_g  removal cross section            [1/cm]
             (g1: absorption + downscatter;  g2: absorption, g2 is last group)
  Sigma_s12  fast(g1)->fast(g2) elastic slowing-down scatter  [1/cm]
  nuSigma_f_g  fission production             [1/cm]

Sigma_r1 already INCLUDES Sigma_s12 (removal = absorption + out-scatter). The
assembly in operators.py uses Sigma_r1 on the group-1 diagonal and Sigma_s12 as
the off-diagonal down-scatter source into group 2.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from xs_branch import load_branch_table
from xs_common import MultiGroupXS, blend_xs, two_group as TwoGroupXS


# Canonical material id mapping (kept stable; embedded in metadata).
# 8-way set matching the Natrium core map + explicit HT9 duct steel. material_state
# is stored as these integer ids per node, but NODE FEATURES use a ONE-HOT encoding
# (see one_hot_batch) so the model sees no artificial ordinal relationship.
#   fuel_inner / fuel_outer      -> inner vs outer enrichment zones (OUTER is the
#                                   HIGHER-enrichment zone; see LIBRARY)
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


# Cross sections are stored as MultiGroupXS (see xs_common). The TwoGroupXS(...)
# factory keeps the original 7-scalar call signature and produces the identical
# [D1,D2,Sr1,Sr2,Ss12,nuSf1,nuSf2] row at G=2 -- so this fast library is unchanged.

# Representative FAST-spectrum two-group data (both groups fast; see module note).
# Control labels store the INSERTED (absorber) XS; when a control assembly is
# withdrawn the geometry substitutes CONTROL_FOLLOWER (sodium follower) instead.
LIBRARY: Dict[str, MultiGroupXS] = {
    # SFR radial enrichment zoning: the OUTER zone carries the HIGHER enrichment.
    # Fast neutrons have a long mean free path, so the core periphery leaks hard and
    # runs a steep flux gradient; loading more fissile there is what flattens the
    # radial power profile (standard practice -- ABR-1000, S-PRISM, BN-800 all zone
    # this way, and Natrium's public description says enrichment "varies by core
    # position" with the peak below 20 wt%). nuSf_outer > nuSf_inner accordingly.
    "fuel_inner": TwoGroupXS(   # lower-enrichment U-10Zr metal fuel (inner zone)
        D1=2.00, D2=1.40,
        Sigma_r1=0.028, Sigma_r2=0.020,
        Sigma_s12=0.022,
        nuSigma_f1=0.014, nuSigma_f2=0.028,
    ),
    "fuel_outer": TwoGroupXS(   # higher-enrichment U-10Zr metal fuel (outer zone)
        D1=2.00, D2=1.40,
        Sigma_r1=0.028, Sigma_r2=0.020,
        Sigma_s12=0.022,
        nuSigma_f1=0.018, nuSigma_f2=0.038,   # more fissile than the inner zone
    ),
    "primary_control": TwoGroupXS(   # B4C, 37 absorber pins (inserted) -> strongest
        D1=1.50, D2=1.00,
        Sigma_r1=0.055, Sigma_r2=0.130,
        Sigma_s12=0.020,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "secondary_control": TwoGroupXS(   # B4C, 19 absorber pins (inserted) -> weaker
        D1=1.55, D2=1.05,
        Sigma_r1=0.048, Sigma_r2=0.095,
        Sigma_s12=0.020,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "reflector": TwoGroupXS(   # steel/sodium reflector: scatters fast neutrons back
        D1=2.20, D2=1.60,
        Sigma_r1=0.032, Sigma_r2=0.012,
        Sigma_s12=0.030,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "shield": TwoGroupXS(   # B4C/steel shield: highest fast absorption
        D1=1.20, D2=0.70,
        Sigma_r1=0.055, Sigma_r2=0.110,
        Sigma_s12=0.025,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "duct": TwoGroupXS(   # HT9 duct wall + inter-assembly Na gap (volume-homogenized)
        D1=1.10, D2=0.70,                 # steel: smaller D than bulk sodium
        Sigma_r1=0.045, Sigma_r2=0.035,   # parasitic Fe capture + scatter, no fission
        Sigma_s12=0.028,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "coolant": TwoGroupXS(   # sodium: very transparent to fast neutrons (large D)
        D1=2.60, D2=2.00,
        Sigma_r1=0.014, Sigma_r2=0.005,
        Sigma_s12=0.011,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
}

# Fission spectrum chi for the Natrium FAST core (G=2). Group boundary ~0.8 MeV;
# a prompt-fission (Watt) spectrum places ~60% of births above 0.8 MeV (g1 high-fast)
# and ~40% in g2 (slow-fast) -- NOT ~all in g1, because both groups are fast and the
# boundary sits inside the birth spectrum. Replaces the old shared CHI=(0.95,0.05).
CHI = (0.60, 0.40)

# Transverse (axial) leakage buckling Bz^2 [1/cm^2]. Bz^2 = (pi / H_extrap)^2 with an
# SFR active height ~100 cm plus a few-cm extrapolation length -> ~(pi/104)^2.
AXIAL_BUCKLING_CM2 = 9.1e-4

# Withdrawn control assembly = sodium follower (nearly transparent, like coolant).
CONTROL_FOLLOWER: MultiGroupXS = TwoGroupXS(
    D1=2.55, D2=1.95,
    Sigma_r1=0.015, Sigma_r2=0.006,
    Sigma_s12=0.011,
    nuSigma_f1=0.0, nuSigma_f2=0.0,
)


# --- OpenMC branch table -----------------------------------------------------
# If an OpenMC branch table (xs_natrium.json, see xs_openmc.py) sits next to this
# module, it becomes the source of truth: constants are looked up per (burnup,
# temperature, control insertion) instead of being read off the hand library and
# multiplied by ad-hoc burnup/temperature factors. Absent -> the committed defaults
# below are used unchanged, so the pipeline still runs without OpenMC.

BRANCH_TABLE_PATH = os.path.join(os.path.dirname(__file__), "xs_natrium.json")
BRANCH = load_branch_table(BRANCH_TABLE_PATH)


def branch_metadata() -> dict | None:
    """Provenance block for geometry_metadata, or None when running hand data."""
    return BRANCH.metadata() if BRANCH is not None else None


def branch_chi(*, burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
               core_rod_frac: float = 0.0):
    """Tallied core-average fission spectrum, or None (falls back to CHI)."""
    if BRANCH is None:
        return None
    return BRANCH.chi(burnup_mwd_kg=burnup_mwd_kg, temperature_k=temperature_k,
                      core_rod_frac=core_rod_frac)


def xs_for(material: str, *, inserted: bool = True,
           insert_frac: float | None = None,
           burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
           core_rod_frac: float = 0.0) -> MultiGroupXS:
    """Cross sections for a material at a core state.

    With an OpenMC branch table loaded, (burnup_mwd_kg, temperature_k,
    core_rod_frac) select an interpolated branch case -- burnt isotopics, Doppler-
    broadened data, and the rodded/unrodded spectrum are all measured quantities.
    Without one, the hand library is returned and those arguments are ignored (the
    caller then applies its documented legacy perturbation instead).

    For control labels, insertion is a gray-rod depth in [0,1]: pass `insert_frac`
    for a partially-inserted rod (0 = withdrawn sodium follower, 1 = full absorber,
    between = axially-averaged blend; see xs_common.blend_xs). `inserted` is the
    legacy binary switch, used only when `insert_frac` is None (True -> 1, False ->
    0), so existing callers are unchanged. Non-control materials ignore both.
    """
    is_control = material in ("primary_control", "secondary_control")
    frac = (1.0 if inserted else 0.0) if insert_frac is None else float(insert_frac)

    if BRANCH is not None and BRANCH.has(material):
        got = BRANCH.lookup(material, burnup_mwd_kg=burnup_mwd_kg,
                            temperature_k=temperature_k, insert_frac=frac,
                            core_rod_frac=core_rod_frac)
        if got is not None:
            return got[0]

    if not is_control:
        return LIBRARY[material]
    if frac >= 1.0:
        return LIBRARY[material]              # exact absorber (byte-compatible)
    if frac <= 0.0:
        return CONTROL_FOLLOWER               # exact follower (byte-compatible)
    return blend_xs(CONTROL_FOLLOWER, LIBRARY[material], frac)


def xs_for_id(material_id: int, *, inserted: bool = True,
              insert_frac: float | None = None,
              burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
              core_rod_frac: float = 0.0) -> MultiGroupXS:
    return xs_for(ID_TO_MATERIAL[material_id], inserted=inserted,
                  insert_frac=insert_frac, burnup_mwd_kg=burnup_mwd_kg,
                  temperature_k=temperature_k, core_rod_frac=core_rod_frac)
