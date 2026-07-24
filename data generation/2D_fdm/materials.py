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

from typing import Dict, List

import numpy as np

from xs_common import MultiGroupXS, two_group as TwoGroupXS


# Canonical material id mapping (kept stable; embedded in metadata).
# 8-way set matching the Natrium core map + explicit HT9 duct steel. material_state
# is stored as these integer ids per node, but NODE FEATURES use a ONE-HOT encoding
# (see one_hot_batch) so the model sees no artificial ordinal relationship.
#   fuel_inner / fuel_outer      -> inner vs outer enrichment zones
#   primary_control / secondary_control -> 9 primary (37 pins) + 4 secondary (19 pins);
#                                     XS toggle absorber (inserted) vs follower (out)
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
    "fuel_inner": TwoGroupXS(   # higher-enrichment U-10Zr metal fuel (inner zone)
        D1=2.00, D2=1.40,
        Sigma_r1=0.028, Sigma_r2=0.020,
        Sigma_s12=0.022,
        nuSigma_f1=0.018, nuSigma_f2=0.038,   # hotter than the outer zone
    ),
    "fuel_outer": TwoGroupXS(   # lower-enrichment U-10Zr metal fuel (outer zone)
        D1=2.00, D2=1.40,
        Sigma_r1=0.028, Sigma_r2=0.020,
        Sigma_s12=0.022,
        nuSigma_f1=0.014, nuSigma_f2=0.028,
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

# Withdrawn control assembly = sodium follower (nearly transparent, like coolant).
CONTROL_FOLLOWER: MultiGroupXS = TwoGroupXS(
    D1=2.55, D2=1.95,
    Sigma_r1=0.015, Sigma_r2=0.006,
    Sigma_s12=0.011,
    nuSigma_f1=0.0, nuSigma_f2=0.0,
)


# If an OpenMC-generated cache (xs_natrium.json, see xs_openmc.py) sits next to this
# module, override the matching hand-tuned entries with the traceable constants.
# Absent -> keep the committed defaults so the pipeline runs without OpenMC.
def _load_openmc_cache() -> None:
    import os
    from xs_common import load_cached_library
    cache = os.path.join(os.path.dirname(__file__), "xs_natrium.json")
    lib, _ = load_cached_library(cache)
    if lib:
        for name, xs in lib.items():
            if name in LIBRARY:
                LIBRARY[name] = xs


_load_openmc_cache()


def xs_for(material: str, *, inserted: bool = True) -> MultiGroupXS:
    """Cross sections for a material. For control labels, `inserted=False` returns
    the sodium follower XS (withdrawn rod)."""
    if material in ("primary_control", "secondary_control") and not inserted:
        return CONTROL_FOLLOWER
    return LIBRARY[material]


def xs_for_id(material_id: int, *, inserted: bool = True) -> MultiGroupXS:
    return xs_for(ID_TO_MATERIAL[material_id], inserted=inserted)
