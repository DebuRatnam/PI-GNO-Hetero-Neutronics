"""Cross-section library for a Kairos KP-FHR pebble-bed core (THERMAL spectrum).

Parallel to materials.py (Natrium fast core). Exposes the SAME public symbols
(MATERIAL_IDS, N_MATERIALS, one_hot_batch, LIBRARY, xs_for, xs_for_id) so the
generator swaps reactor type by importing this module instead. KP-FHR is a
graphite + FLiBe thermally-moderated system, so group 2 is a genuine THERMAL group
(unlike the Natrium slow-fast group). Consequences baked into the numbers:

  - Group 1 = fast (born ~2 MeV), Group 2 = thermal (~0.025 eV). The fast/thermal
    boundary for the collapsed constants is ~0.625 eV (recorded in metadata).
  - Fission is DOMINATED by the thermal group: nuSf2 >> nuSf1 (opposite of the
    fast Natrium library) -- neutrons fission after moderation.
  - Moderators (graphite pebble, reflector, FLiBe) have strong fast->thermal
    down-scatter Ss12 (that IS moderation) and low absorption.
  - B4C control / shutdown elements are strong THERMAL absorbers (large Sr2): B-10
    (n,alpha) rises steeply toward thermal energies.
  - FLiBe carries a modest thermal absorption (Sr2) from Li-6 capture.

Values are representative order-of-magnitude THERMAL two-group constants, NOT
benchmarked -- they are replaced by OpenMC-generated group constants
(xs_openmc.py) when the cached library is present. Group count is configurable
(xs_common); this hand default is G=2.

Two-group storage per material (via the MultiGroupXS G=2 constructor):
  D_g diffusion [cm]; Sigma_r_g removal (absorption + out-scatter) [1/cm];
  Sigma_s12 fast->thermal down-scatter [1/cm]; nuSigma_f_g fission prod [1/cm].
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from xs_branch import load_branch_table
from xs_common import MultiGroupXS, blend_xs, two_group as TwoGroupXS


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


# Representative THERMAL-spectrum two-group data (g1 fast, g2 thermal).
# Control/shutdown labels store the INSERTED (absorber) XS; when withdrawn the
# geometry substitutes FLIBE_FOLLOWER (salt-filled channel) instead.
LIBRARY: Dict[str, MultiGroupXS] = {
    "fuel_pebble": TwoGroupXS(    # TRISO-in-graphite, homogenized: thermal-fissile
        D1=1.30, D2=0.90,
        Sigma_r1=0.032, Sigma_r2=0.095,        # g1 removal ~ downscatter; g2 absorption
        Sigma_s12=0.026,                        # fast -> thermal moderation
        nuSigma_f1=0.004, nuSigma_f2=0.150,     # fission dominated by thermal group
    ),
    "graphite_pebble": TwoGroupXS(  # moderator-only pebble: strong scatter, ~no absorption
        D1=1.20, D2=0.85,
        Sigma_r1=0.028, Sigma_r2=0.0006,
        Sigma_s12=0.027,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    # Control/shutdown absorber is B4C enriched to 100 at% B-10 at 1.76 g/cm3 (the
    # published gFHR rod), ~3.5x the B-10 atom density of natural full-density B4C.
    # These elements are BLACK to thermal neutrons either way, so the homogenized
    # removal is not the true macroscopic Sigma_a (~300 /cm) but the diffusion-
    # equivalent value for a black cylinder, order 1/a with a = 2.6 cm rod radius.
    "control_element": TwoGroupXS(  # B4C cylinder (inserted): strong thermal absorber
        D1=1.00, D2=0.50,
        Sigma_r1=0.020, Sigma_r2=0.750,
        Sigma_s12=0.010,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "shutdown_element": TwoGroupXS(  # B4C X-element (inserted): strongest absorber
        D1=0.95, D2=0.45,
        Sigma_r1=0.020, Sigma_r2=0.900,
        Sigma_s12=0.010,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "reflector": TwoGroupXS(    # graphite reflector: scatters neutrons back, low absorption
        D1=1.20, D2=0.90,
        Sigma_r1=0.025, Sigma_r2=0.0005,
        Sigma_s12=0.024,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "coolant": TwoGroupXS(      # FLiBe salt: moderate moderation, small Li-6 thermal capture
        D1=1.40, D2=1.00,
        Sigma_r1=0.020, Sigma_r2=0.008,
        Sigma_s12=0.018,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
    "vessel": TwoGroupXS(       # structural steel barrel/shield: parasitic capture
        D1=1.00, D2=0.60,
        Sigma_r1=0.030, Sigma_r2=0.050,
        Sigma_s12=0.020,
        nuSigma_f1=0.0, nuSigma_f2=0.0,
    ),
}

# Fission spectrum chi for the KP-FHR THERMAL core (G=2). Thermal cut ~0.625 eV;
# essentially every fission neutron is born fast (~2 MeV), so ALL birth goes to g1 and
# none to the thermal group g2. chi_2 must be ~0 (the old shared 0.05 was unphysical
# for a thermal group).
CHI = (1.0, 0.0)

# Transverse (axial) leakage buckling Bz^2 [1/cm^2]. Bz^2 = (pi / H_extrap)^2 with the
# published gFHR active bed height 309.47 cm plus a short extrapolation distance ->
# ~(pi/314)^2. Smaller than the fast core: a taller core leaks less axially per unit
# height. (Axially the real core is reflected by graphite, so the true extrapolated
# height is longer and this is a conservative upper bound on axial leakage.)
AXIAL_BUCKLING_CM2 = 1.0e-4

# Withdrawn control/shutdown element = FLiBe-filled channel (like coolant).
FLIBE_FOLLOWER: MultiGroupXS = TwoGroupXS(
    D1=1.40, D2=1.00,
    Sigma_r1=0.020, Sigma_r2=0.008,
    Sigma_s12=0.018,
    nuSigma_f1=0.0, nuSigma_f2=0.0,
)


# --- OpenMC branch table -----------------------------------------------------
# If an OpenMC branch table (xs_fhr.json, see xs_openmc.py) sits next to this module,
# it becomes the source of truth: constants are looked up per (burnup, temperature,
# insertion) instead of being read off the hand library and multiplied by ad-hoc
# factors. Absent -> the committed defaults below are used unchanged.

BRANCH_TABLE_PATH = os.path.join(os.path.dirname(__file__), "xs_fhr.json")
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
    core_rod_frac) select an interpolated branch case: burnt isotopics (including
    the Xe/Sm poison that the old `burnup_poison_coeff` stood in for), Doppler +
    S(alpha,beta) temperature effects, and the rodded/unrodded spectrum are all
    measured. Without a table the hand library is returned and those arguments are
    ignored.

    For control/shutdown labels, insertion is a gray-rod depth in [0,1]: pass
    `insert_frac` for a partially-inserted element (0 = withdrawn FLiBe follower,
    1 = full B4C absorber, between = axially-averaged blend; see xs_common.blend_xs).
    `inserted` is the legacy binary switch, used only when `insert_frac` is None.
    Non-control materials ignore both.
    """
    is_control = material in ("control_element", "shutdown_element")
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
        return FLIBE_FOLLOWER                 # exact follower (byte-compatible)
    return blend_xs(FLIBE_FOLLOWER, LIBRARY[material], frac)


def xs_for_id(material_id: int, *, inserted: bool = True,
              insert_frac: float | None = None,
              burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
              core_rod_frac: float = 0.0) -> MultiGroupXS:
    return xs_for(ID_TO_MATERIAL[material_id], inserted=inserted,
                  insert_frac=insert_frac, burnup_mwd_kg=burnup_mwd_kg,
                  temperature_k=temperature_k, core_rod_frac=core_rod_frac)


# Thermal UP-scatter Ss_{g2->g1} [1/cm] (thermal -> epithermal): a real effect in
# graphite/FLiBe at KP-FHR temperature (~650 C), negligible in fuel/absorber/vessel.
# Applied at the OPERATOR level (assembled into A: raises thermal removal + adds an
# in-scatter source to the fast group) rather than stored in the per-node XS row,
# which stays down-scatter-only (schema unchanged).
#
# These committed values are the hand fallback. With an OpenMC branch table loaded
# the up-scatter comes from the UPPER TRIANGLE of the tallied nu-scatter matrix --
# a measured transfer rate at the branch temperature, which is the right way to get
# it since up-scatter is precisely a thermal-motion (S(alpha,beta)) effect.
UPSCATTER_21: Dict[str, float] = {
    "fuel_pebble": 0.0008, "graphite_pebble": 0.0025, "control_element": 0.0,
    "shutdown_element": 0.0, "reflector": 0.0022, "coolant": 0.0015, "vessel": 0.0,
}


def up_scatter_for(material: str, *, insert_frac: float = 1.0,
                   burnup_mwd_kg: float = 0.0, temperature_k: float | None = None,
                   core_rod_frac: float = 0.0) -> float:
    """Thermal->fast up-scatter Ss21 [1/cm] for a material at a core state.

    Returns the G=2 up-scatter scalar. For G>2 the full up-scatter block lives in
    the branch table; operators.assemble_AF consumes the g2->g1 term only, matching
    the two-group operator-level treatment documented above.
    """
    if BRANCH is not None and BRANCH.has(material):
        got = BRANCH.lookup(material, burnup_mwd_kg=burnup_mwd_kg,
                            temperature_k=temperature_k, insert_frac=insert_frac,
                            core_rod_frac=core_rod_frac)
        if got is not None and len(got[1]):
            return float(got[1][0])
    return UPSCATTER_21.get(material, 0.0)


def up_scatter_for_id(material_id: int, *, insert_frac: float = 1.0,
                      burnup_mwd_kg: float = 0.0,
                      temperature_k: float | None = None,
                      core_rod_frac: float = 0.0) -> float:
    """Thermal->fast up-scatter Ss21 [1/cm] for a material id (0 if none)."""
    return up_scatter_for(ID_TO_MATERIAL[material_id], insert_frac=insert_frac,
                          burnup_mwd_kg=burnup_mwd_kg, temperature_k=temperature_k,
                          core_rod_frac=core_rod_frac)
