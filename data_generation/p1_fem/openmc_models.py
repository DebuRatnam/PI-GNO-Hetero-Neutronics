"""Full-core OpenMC transport models for group-constant generation.

These models exist for ONE purpose: to produce the *in-situ* neutron spectrum that
weights the multigroup constants consumed by the P1 diffusion generator. They are
therefore built to mirror the FEM cores in geometry.py / geometry_pebble.py as
closely as continuous-energy transport allows, NOT to be standalone design models.

Why full-core rather than per-material infinite media
-----------------------------------------------------
The group-collapse equation Sigma_g = int Sigma(E) phi(E) dE / int phi(E) dE is only
as good as phi(E). An infinite reflective box of pure B4C driven by a Watt source has
a spectrum with no relationship to the spectrum a reflector-adjacent control element
actually sees, and the resulting absorption constants are wrong by a large factor in
a thermal system. Every material here is tallied with `domain_type="material"` inside
the assembled core, so each material is collapsed against the flux it really sees.

Double heterogeneity
--------------------
Resonance self-shielding is a geometric effect: smearing fuel into its moderator
destroys it and biases nuSigma_f and absorption.

  - `fhr` : every fuel pebble contains an explicit TRISO particle lattice (kernel /
    buffer / IPyC / SiC / OPyC) in a graphite matrix, inside a graphite shell. All
    pebbles of a kind share ONE universe, so the 10^4-particle lattice is defined
    once and instantiated per pebble -- the standard OpenMC pebble-bed construction.
  - `hex` : every fuel / control assembly contains an explicit hexagonal pin lattice
    (U-10Zr slug + HT9 clad + sodium bond, or B4C absorber pins) inside the
    homogenized HT9 duct ring.

Axial treatment
---------------
The FEM model is a 2D radial slice with axial leakage carried separately as a
transverse buckling (PhysicsConfig.axial_buckling). The OpenMC models match that
convention: a short axial slab with REFLECTIVE top/bottom (axially infinite, i.e.
zero axial leakage) and a vacuum radial boundary. Axial leakage must therefore NOT
be double counted -- it enters once, via Bz^2 in operators.assemble_AF.

Provenance of the numbers
-------------------------
`fhr` follows the PUBLISHED gFHR benchmark -- Kairos Power's own non-proprietary
KP-FHR surrogate (Satvat et al., Nucl. Eng. Des. 384 (2021) 111461; INL Virtual Test
Bed gFHR description; Duchnowski et al. 2023 Table 1): 120 cm bed radius, 60 cm
graphite reflector, SS316H barrel / FLiBe downcomer / SS316H vessel, 4 cm pebbles
with a 1.38 cm low-density buoyancy core and an annular TRISO-bearing fuel shell,
19.55 wt% UCO kernels at 0.22 TRISO packing, 0.60 bed packing, 100 at% B-10 B4C
control absorber at 2.6 cm radius, 7.9 cm out from the bed edge. Control-element
COUNT follows gFHR too (10 reflector rods), since the bed radius does; the 3 in-bed
shutdown elements are a KP-FHR feature gFHR omits entirely, taken from the licensed
design (NRC KP-FHR Core Design and Analysis Methodology, KP-TR-024-NP Rev 0,
ML24095A258: the reactivity control system inserts into side-reflector channels, the
reactivity shutdown system inserts directly into the pebble bed). Hermes as licensed
runs 4 control + 3 shutdown in a ~2 m^3 core -- set n_control=4 if you rescale.

`hex` is REPRESENTATIVE, not a vendor spec. Natrium's public docket fixes the fuel
form (U-10wt%Zr, sodium-bonded, HT9 clad, peak enrichment < 20 wt%), the B4C
absorber, and the 9 primary + 4 secondary control assembly counts -- but not the
pitch, the pin lattices, the ring layout or the enrichment split. Those are literature
values for a HALEU metal-fuel SFR of this class. Review them against your own spec
before treating the tallied constants as benchmark grade.

openmc is imported lazily inside the builders so this module (and everything that
imports it) works without OpenMC installed.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from datagen_config import HexCoreConfig, PebbleCoreConfig


# --- branch state ------------------------------------------------------------

@dataclass(frozen=True)
class BranchState:
    """One lattice-physics branch case.

    burnup_mwd_kg : fuel burnup [MWd/kgHM]. Fuel compositions come from the
                    depletion table (xs_depletion.py); 0 = fresh.
    temperature_k : bulk material temperature [K] (Doppler + S(alpha,beta)).
    rod           : "out" | "in" -- control/shutdown channels filled with the
                    follower (coolant) or with the B4C absorber. Both control
                    families switch together; each is tallied as its own material
                    domain, so one "in" branch yields constants for all of them.
    """
    burnup_mwd_kg: float = 0.0
    temperature_k: float = 900.0
    rod: str = "out"

    @property
    def key(self) -> str:
        return f"bu{self.burnup_mwd_kg:g}_T{self.temperature_k:g}_rod-{self.rod}"


# --- geometric constants (documented, auditable) -----------------------------

# hex / Natrium pin cell. Radii are expressed as fractions of the pin pitch so the
# lattice scales with the assembly. Fuel volume fraction ~ pi*r_fuel^2/(sqrt3/2 p^2).
HEX_PIN_RINGS_FUEL = 9        # 3n(n-1)+1 = 217 pins per fuel assembly
HEX_PIN_RINGS_PRIMARY = 4     # 37 B4C pins  (primary control)
HEX_PIN_RINGS_SECONDARY = 3   # 19 B4C pins  (secondary control)
HEX_R_FUEL_FRAC = 0.32        # fuel slug radius / pin pitch  -> VF ~ 0.37
HEX_R_CLAD_IN_FRAC = 0.36     # sodium bond gap outer radius / pitch
HEX_R_CLAD_OUT_FRAC = 0.40    # HT9 clad outer radius / pitch

# fhr / KP-FHR pebble + TRISO. These are the PUBLISHED gFHR benchmark values (Kairos
# Power's non-proprietary KP-FHR surrogate; Satvat et al., Nucl. Eng. Des. 384 (2021)
# 111461; INL Virtual Test Bed gFHR reactor description; Duchnowski et al. 2023
# Table 1). The pebble is NOT a solid fuel sphere: it is a LOW-DENSITY graphite core
# for buoyancy in FLiBe (the pebbles float up through the bed), a TRISO-bearing fuel
# SHELL around it, and a fuel-free outer graphite shell.
PEBBLE_CORE_R = 1.38          # [cm] low-density graphite buoyancy core
PEBBLE_FUEL_ZONE_R = 1.80     # [cm] outer radius of the TRISO-bearing fuel shell
PEBBLE_SHELL_R = 2.00         # [cm] pebble outer radius (4 cm dia)
PEBBLE_CORE_DENSITY = 1.41    # [g/cm3] buoyancy core graphite
PEBBLE_MATRIX_DENSITY = 1.74  # [g/cm3] fuel-layer graphite matrix
PEBBLE_SHELL_DENSITY = 1.74   # [g/cm3] outer fuel-free graphite shell
TRISO_PACKING_FRACTION = 0.22 # TRISO volume fraction in the fuel SHELL
TRISO_R = {                   # [cm] cumulative TRISO layer radii
    "kernel": 0.02125,        # 425 um dia UCO (UC0.5O1.5) kernel, 10.5 g/cm3
    "buffer": 0.03125,        # +100 um porous carbon, 1.05 g/cm3
    "ipyc":   0.03525,        # +40 um inner pyrolytic carbon, 1.90 g/cm3
    "sic":    0.03875,        # +35 um SiC, 3.18 g/cm3
    "opyc":   0.04275,        # +40 um outer pyrolytic carbon, 1.90 g/cm3
}
BED_PACKING_FRACTION = 0.60   # 3D random sphere packing in the pebble bed
FHR_ENRICHMENT_WT_PCT = 19.55 # gFHR UCO enrichment (HALEU; Hermes quotes 19.74)
REFLECTOR_GRAPHITE_DENSITY = 1.74   # [g/cm3] gFHR side/plenum reflector graphite
SS316H_DENSITY = 8.0          # [g/cm3] gFHR core barrel + reactor vessel
B4C_CONTROL_DENSITY = 1.76    # [g/cm3] gFHR control-rod B4C
B4C_CONTROL_B10_ENRICH = 1.0  # gFHR control rods are 100 at% B-10

# Pebble packings are expensive to generate and identical across branch cases, so
# they are cached here (see _packed_bed). Safe to delete; it only costs time.
PACKING_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".packing_cache")


# --- environment resolution --------------------------------------------------

def resolve_openmc_exec() -> str:
    """Absolute path to the `openmc` binary for the active environment.

    `Model.run()` shells out to `openmc`, which is not necessarily on PATH: running
    the driver through `conda run`, or from an interpreter that is not the one the
    OpenMC package was installed against, both break the default lookup. Resolution
    order: OPENMC_EXECUTABLE, the active interpreter's own bin directory, then PATH.
    """
    env_path = os.environ.get("OPENMC_EXECUTABLE")
    if env_path:
        return env_path

    py_dir = os.path.dirname(sys.executable)
    for cand in (os.path.join(py_dir, "openmc"),
                 os.path.join(py_dir, "openmc.exe"),
                 os.path.join(py_dir, "..", "bin", "openmc"),
                 os.path.join(py_dir, "..", "bin", "openmc.exe")):
        if os.path.exists(cand):
            return cand

    found = shutil.which("openmc")
    if found:
        return found

    raise RuntimeError(
        "OpenMC executable not found. Install OpenMC into the active Python "
        "environment or set OPENMC_EXECUTABLE to the full path of the 'openmc' binary.")


def prepare_openmc_env() -> str:
    """Resolve the executable, put its directory on PATH, and locate nuclear data.

    Returns the executable path. Cross-section resolution honours, in order:
    OPENMC_CROSS_SECTIONS, openmc.config['cross_sections'], then a few conventional
    install locations. Missing data is reported here rather than as an opaque failure
    several minutes into a transport run.
    """
    exe = resolve_openmc_exec()
    bin_dir = os.path.dirname(exe)
    if bin_dir and bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    xs = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not xs:
        try:
            import openmc
            cfg = getattr(openmc, "config", {})
            xs = str(cfg["cross_sections"]) if "cross_sections" in cfg else None
        except Exception:
            xs = None
    if not xs:
        home = os.path.expanduser("~")
        for cand in (os.path.join(home, "nucdata", "endfb-viii.0-hdf5", "cross_sections.xml"),
                     os.path.join(home, "nuclear-data", "cross_sections.xml"),
                     "/usr/local/share/openmc/cross_sections.xml"):
            if os.path.exists(cand):
                xs = cand
                break
    if xs:
        os.environ["OPENMC_CROSS_SECTIONS"] = xs
    else:
        print("WARNING: no nuclear data library found (OPENMC_CROSS_SECTIONS unset, "
              "openmc.config empty, no library in the usual locations). Transport "
              "will fail until one is installed.", flush=True)
    return exe


# --- material builders -------------------------------------------------------

def _add_haleu(openmc, m, wt_pct_u235: float, u_weight_fraction: float = 1.0) -> None:
    """Add HALEU uranium by explicit isotopics.

    `Material.add_element("U", enrichment=...)` assumes a constant U234/U235 mass
    ratio of 0.008, which OpenMC itself warns is only valid below ~5 wt%. Both cores
    here run at 15.5-19.75 wt%, so the isotopics are set explicitly instead.

    U-234 follows the standard enrichment-cascade correlation
    w(U234) = 0.0089 * w(U235) (ASTM C996 range for enriched product), and U-236 is
    taken as zero: that is correct for enrichment from natural feed, and would need
    revisiting only if the fuel is specified as downblended from HEU (which carries
    ~0.1-0.5 wt% U236 and a small negative reactivity effect).
    """
    w235 = wt_pct_u235 / 100.0
    w234 = 0.0089 * w235
    w238 = 1.0 - w235 - w234
    m.add_nuclide("U234", u_weight_fraction * w234, "wo")
    m.add_nuclide("U235", u_weight_fraction * w235, "wo")
    m.add_nuclide("U238", u_weight_fraction * w238, "wo")


def _apply_composition(openmc, m, composition: Dict[str, float]) -> None:
    """Fill a material from a depleted nuclide inventory.

    `composition` is {nuclide: atom density [atom/b-cm]} straight out of the
    depletion table. `set_density("sum")` tells OpenMC to read the per-nuclide
    values as atom densities and sum them, so the burnt material carries both the
    correct isotopics AND the correct number density (swelling / burnout included)
    rather than being renormalized to the fresh density.
    """
    for nuc, dens in composition.items():
        m.add_nuclide(nuc, float(dens))
    m.set_density("sum")


def _u_metal_fuel(openmc, name: str, enrichment: float, T: float,
                  composition: Optional[Dict[str, float]] = None):
    """U-10Zr HALEU metal fuel slug. `composition` (nuclide atom densities from the
    depletion table) overrides the fresh composition when supplied."""
    m = openmc.Material(name=name)
    if composition:
        _apply_composition(openmc, m, composition)
    else:
        # U-10Zr is 10 WEIGHT percent Zr (~22 at%), so the alloy is specified by
        # weight fraction, not atom fraction.
        _add_haleu(openmc, m, enrichment, u_weight_fraction=0.90)
        m.add_element("Zr", 0.10, "wo")
        m.set_density("g/cm3", 15.5)
    m.temperature = T
    return m


def _haleu_atom_fractions(wt_pct_u235: float) -> Tuple[float, float, float]:
    """(a234, a235, a238) ATOM fractions of the uranium at a given wt% U-235.

    Same U-234 cascade correlation as _add_haleu; returned normalized to sum to 1 so
    the caller can use them directly in a stoichiometric ('ao') compound.
    """
    w235 = wt_pct_u235 / 100.0
    w234 = 0.0089 * w235
    w238 = 1.0 - w235 - w234
    n = {"U234": w234 / 234.0409, "U235": w235 / 235.0439, "U238": w238 / 238.0508}
    tot = sum(n.values())
    return n["U234"] / tot, n["U235"] / tot, n["U238"] / tot


def _uco_kernel(openmc, name: str, enrichment: float, T: float,
                composition: Optional[Dict[str, float]] = None):
    """TRISO UCO kernel, UC(0.5)O(1.5) at 10.5 g/cm3.

    The gFHR/KP-FHR kernel is uranium OXYCARBIDE, not UO2 -- the specification is the
    AGR-2 oxycarbide form with the carbon content at the upper limit of the AGR
    testing envelope. That excess carbon is what getters the CO released by fission
    and buffers the kernel-migration/SiC-corrosion failure modes; it also puts extra
    moderating carbon inside the kernel, which UO2 does not have. `composition` from
    the depletion table overrides the fresh isotopics.
    """
    m = openmc.Material(name=name)
    if composition:
        _apply_composition(openmc, m, composition)
    else:
        a234, a235, a238 = _haleu_atom_fractions(enrichment)
        m.add_nuclide("U234", a234)          # atom fractions, U total = 1
        m.add_nuclide("U235", a235)
        m.add_nuclide("U238", a238)
        m.add_element("C", 0.5)              # UC(0.5)O(1.5)
        m.add_element("O", 1.5)
        m.set_density("g/cm3", 10.5)
    m.temperature = T
    return m


def _graphite(openmc, name: str, density: float, T: float):
    m = openmc.Material(name=name)
    m.add_element("C", 1.0)
    m.set_density("g/cm3", density)
    m.add_s_alpha_beta("c_Graphite")     # bound-atom thermal scattering: essential
    m.temperature = T
    return m


def _b4c(openmc, name: str, T: float, density: float = 2.52,
         b10_enrichment: Optional[float] = None):
    """B4C absorber. `b10_enrichment` is the B-10 ATOM fraction of the boron; None
    keeps natural boron (19.9 at% B-10).

    The gFHR control rods are specified as 100% B-10 at 1.76 g/cm3 (a porous /
    matrix-bound absorber, not full-density B4C), which is a very different absorber
    from natural-boron full-density B4C: ~5x the B-10 atom density per boron atom but
    ~0.7x the bulk density. Natrium's control assemblies are docketed only as "boron
    carbide"; enrichment is not public, so those stay natural.
    """
    m = openmc.Material(name=name)
    if b10_enrichment is None:
        m.add_element("B", 4.0)
    else:
        f = float(b10_enrichment)
        m.add_nuclide("B10", 4.0 * f)
        if f < 1.0:
            m.add_nuclide("B11", 4.0 * (1.0 - f))
    m.add_element("C", 1.0)
    m.set_density("g/cm3", density)
    m.temperature = T
    return m


def _flibe(openmc, name: str, T: float):
    """FLiBe (Li2BeF4), Li-7 enriched to 99.995 at% (Li-6 is a strong thermal poison).
    Density from the Li2BeF4 correlation rho[g/cm3] = 2.28 - 4.884e-4 * T[C]."""
    m = openmc.Material(name=name)
    m.add_nuclide("Li7", 2.0 * 0.99995)
    m.add_nuclide("Li6", 2.0 * 0.00005)
    m.add_element("Be", 1.0)
    m.add_element("F", 4.0)
    m.set_density("g/cm3", 2.28 - 4.884e-4 * (T - 273.15))
    m.temperature = T
    return m


def _sodium(openmc, name: str, T: float):
    """Liquid sodium. rho[g/cm3] = 1.0117 - 2.359e-4 * T[K] (Fink & Leibowitz)."""
    m = openmc.Material(name=name)
    m.add_element("Na", 1.0)
    m.set_density("g/cm3", 1.0117 - 2.359e-4 * T)
    m.temperature = T
    return m


def _ht9(openmc, name: str, T: float, density: float = 7.8):
    """HT9 ferritic-martensitic steel (Fe-12Cr-1Mo-W-V)."""
    m = openmc.Material(name=name)
    m.add_element("Fe", 0.850)
    m.add_element("Cr", 0.120)
    m.add_element("Mo", 0.010)
    m.add_element("W", 0.005)
    m.add_element("Ni", 0.005)
    m.add_element("Mn", 0.006)
    m.add_element("V", 0.003)
    m.add_element("Si", 0.001)
    m.set_density("g/cm3", density)
    m.temperature = T
    return m


def _ss316(openmc, name: str, T: float, density: float = 7.99):
    """Type 316 / 316H stainless steel.

    Used for the Natrium radial reflector and shield structure, and -- as 316H at the
    gFHR-specified 8.0 g/cm3 -- for the KP-FHR core barrel and reactor vessel. 316H is
    the structural alloy Kairos qualified for the KP-FHR (ASME code case); the older
    FLiBe-loop reference alloy Hastelloy-N is NOT what this design uses, and its Ni/Mo
    content would give a materially different parasitic capture in the vessel ring.
    """
    m = openmc.Material(name=name)
    m.add_element("Fe", 0.655)
    m.add_element("Cr", 0.17)
    m.add_element("Ni", 0.12)
    m.add_element("Mo", 0.025)
    m.add_element("Mn", 0.02)
    m.add_element("Si", 0.01)
    m.set_density("g/cm3", density)
    m.temperature = T
    return m


# --- tally-domain bookkeeping ------------------------------------------------

@dataclass
class CoreModel:
    """An OpenMC model plus everything needed to turn its tallies into FEM constants.

    tally_domains : tag -> openmc.Material. One entry per *transport* material, at
                    the resolution the transport model actually resolves (fuel slug,
                    clad, bond, TRISO layers, ...).
    homogenize    : FEM material name -> {tag: volume [cm^3 per unit cell]}. One FEM
                    node is a homogenized region, so its constants are the
                    flux-volume-weighted homogenization of its constituents. A 1:1
                    material maps to {itself: 1.0}.

    Only materials present in this branch can be collapsed: a rod-out branch has no
    B4C domains and a rod-in branch has no follower domain. That is expected -- the
    branch table stores each endpoint where it exists.
    """
    model: object                                       # openmc.Model
    tally_domains: Dict[str, object]                    # tag -> openmc.Material
    homogenize: Dict[str, Dict[str, float]]             # FEM name -> {tag: volume}
    notes: Dict[str, object] = field(default_factory=dict)


# --- Natrium hex core --------------------------------------------------------

def n_pins(n_rings: int) -> int:
    """Pins in a hexagonal lattice of n_rings rings: 3n(n-1)+1."""
    return 3 * n_rings * (n_rings - 1) + 1


def _hex_pin_volumes(pitch: float, n_rings: int, Ri: float, center_name: str,
                     absorber: bool = False) -> Dict[str, float]:
    """Per-unit-height constituent volumes [cm^3/cm] of one hex assembly interior.

    The FEM mesh represents a whole assembly interior with ONE material label
    (`fuel_inner`, `primary_control`, ...), so the transport-resolved constituents
    must be homogenized back together with these volume weights.
    """
    r_f = HEX_R_FUEL_FRAC * pitch
    r_ci = HEX_R_CLAD_IN_FRAC * pitch
    r_co = HEX_R_CLAD_OUT_FRAC * pitch
    n = n_pins(n_rings)
    a_hex = 1.5 * np.sqrt(3.0) * Ri ** 2                 # regular hexagon, circumradius Ri
    vols = {center_name: n * np.pi * r_f ** 2}
    if not absorber:
        vols[f"{center_name}__bond"] = n * np.pi * (r_ci ** 2 - r_f ** 2)
        vols[f"{center_name}__clad"] = n * np.pi * (r_co ** 2 - r_ci ** 2)
    else:
        # absorber pins are clad directly (no sodium bond gap)
        vols[f"{center_name}__clad"] = n * np.pi * (r_co ** 2 - r_f ** 2)
    vols[f"{center_name}__na"] = a_hex - n * np.pi * r_co ** 2
    if vols[f"{center_name}__na"] <= 0.0:
        raise ValueError(f"{center_name}: pin lattice overfills the assembly "
                         f"(pitch={pitch:.4f}, rings={n_rings})")
    return {k: float(v) for k, v in vols.items()}


def _hex_assembly_universe(openmc, tag: str, center_mat, bond, clad, coolant,
                           pitch: float, n_rings: int, absorber: bool = False):
    """Hexagonal pin lattice for one assembly role.

    Fuel pins: slug / sodium bond / HT9 clad in flowing sodium.
    Absorber pins: B4C / HT9 clad in flowing sodium (no bond gap).
    """
    r_c = HEX_R_FUEL_FRAC * pitch
    r_ci = HEX_R_CLAD_IN_FRAC * pitch
    r_co = HEX_R_CLAD_OUT_FRAC * pitch

    s_c = openmc.ZCylinder(r=r_c)
    s_co = openmc.ZCylinder(r=r_co)
    if absorber:
        pin_cells = [openmc.Cell(fill=center_mat, region=-s_c),
                     openmc.Cell(fill=clad, region=+s_c & -s_co)]
    else:
        s_ci = openmc.ZCylinder(r=r_ci)
        pin_cells = [openmc.Cell(fill=center_mat, region=-s_c),
                     openmc.Cell(fill=bond, region=+s_c & -s_ci),
                     openmc.Cell(fill=clad, region=+s_ci & -s_co)]
    pin_cells.append(openmc.Cell(fill=coolant, region=+s_co))
    pin = openmc.Universe(name=f"{tag}_pin", cells=pin_cells)
    na = openmc.Universe(name=f"{tag}_na", cells=[openmc.Cell(fill=coolant)])

    lat = openmc.HexLattice(name=tag)
    lat.center = (0.0, 0.0)
    lat.pitch = (pitch,)
    lat.orientation = "y"          # matches geometry._hex_corners (vertex on +y)
    lat.outer = na
    # rings are ordered outermost -> innermost; ring r has max(1, 6r) positions
    lat.universes = [[pin] * max(1, 6 * r) for r in range(n_rings - 1, -1, -1)]
    return lat


def natrium_model(hx: HexCoreConfig, state: BranchState, *,
                  reflector_rings: Optional[int] = None,
                  shield_rings: Optional[int] = None,
                  enrichment_boundary: Optional[int] = None,
                  axial_cm: float = 40.0,
                  fuel_composition: Optional[Dict[str, Dict[str, float]]] = None,
                  ) -> CoreModel:
    """Full-core Natrium-inspired hex-duct model matching geometry.make_core.

    Same lattice, same ring roles, same control positions, same homogenized duct
    ring. Assemblies are resolved down to the pin, so resonance self-shielding is
    physical rather than smeared. Radial vacuum + axially reflective (see module
    docstring on the buckling convention).

    `fuel_composition` maps "fuel_inner"/"fuel_outer" -> {nuclide: atom_fraction}
    from the depletion table at state.burnup_mwd_kg; None = fresh fuel.
    """
    import openmc
    # role/control assignment is shared with the FEM builder so the two cores are
    # the same core -- never re-derive the layout here.
    from geometry import (SQRT3, DUCT_RING_FRAC, _hex_cells, _axial_to_xy,
                          _assign_roles, _pick_control)
    from materials import MATERIAL_IDS, ID_TO_MATERIAL

    reflector_rings = hx.reflector_rings if reflector_rings is None else reflector_rings
    shield_rings = hx.shield_rings if shield_rings is None else shield_rings
    enrichment_boundary = (hx.enrichment_boundary_ring if enrichment_boundary is None
                           else enrichment_boundary)
    total_rings = hx.fuel_rings + reflector_rings + shield_rings - 1
    Rc = (hx.pitch_cm - hx.gap_cm) / SQRT3           # hex circumradius (= edge length)
    Ri = Rc * (1.0 - DUCT_RING_FRAC)                 # inside the homogenized duct ring

    cells_qr = _hex_cells(total_rings)
    roles = _assign_roles(cells_qr, hx, enrichment_boundary, reflector_rings, shield_rings)
    roles.update(_pick_control(roles, hx))

    T = state.temperature_k
    fc = fuel_composition or {}
    rod_in = state.rod == "in"

    # Materials shared by every assembly of a role must still be tallied PER ROLE:
    # the sodium inside a fuel assembly and the sodium inside a control assembly see
    # different spectra, and each is homogenized into a different FEM node. Hence one
    # clad/bond/sodium instance per role rather than one global instance.
    mats: List[object] = []
    domains: Dict[str, object] = {}
    homogenize: Dict[str, Dict[str, float]] = {}

    def reg(tag: str, mat):
        mats.append(mat)
        domains[tag] = mat
        return mat

    coolant = reg("coolant", _sodium(openmc, "coolant", T))     # inter-assembly gap
    duct = reg("duct", _ht9(openmc, "duct", T, density=7.0))    # HT9 wall + gap, homog.
    reflector = reg("reflector", _ss316(openmc, "reflector", T))
    shield = reg("shield", _b4c(openmc, "shield", T))
    for name in ("coolant", "duct", "reflector", "shield"):
        homogenize[name] = {name: 1.0}                          # 1:1, no homogenization

    pitch_f = Ri / (HEX_PIN_RINGS_FUEL - 0.5)
    pitch_p = Ri / (HEX_PIN_RINGS_PRIMARY - 0.5)
    pitch_s = Ri / (HEX_PIN_RINGS_SECONDARY - 0.5)

    def make_assembly(fem_name: str, center_mat, pitch: float, rings: int,
                      absorber: bool):
        """Register one assembly role: constituent materials + volumes + lattice."""
        bond = (None if absorber
                else reg(f"{fem_name}__bond", _sodium(openmc, f"{fem_name}__bond", T)))
        clad = reg(f"{fem_name}__clad", _ht9(openmc, f"{fem_name}__clad", T))
        na = reg(f"{fem_name}__na", _sodium(openmc, f"{fem_name}__na", T))
        reg(fem_name, center_mat)
        homogenize[fem_name] = _hex_pin_volumes(pitch, rings, Ri, fem_name,
                                                absorber=absorber)
        return _hex_assembly_universe(openmc, fem_name, center_mat, bond, clad, na,
                                      pitch, rings, absorber=absorber)

    # SFR radial enrichment zoning: the OUTER zone is the HIGHER-enriched one, to
    # offset the hard leakage at the core periphery and flatten radial power. Natrium
    # is docketed only as "enrichment varies by core position, peak < 20 wt% U-235";
    # 15.5 / 19.75 is a representative split at that ceiling.
    lat_inner = make_assembly(
        "fuel_inner", _u_metal_fuel(openmc, "fuel_inner", 15.50, T,
                                    fc.get("fuel_inner")),
        pitch_f, HEX_PIN_RINGS_FUEL, absorber=False)
    lat_outer = make_assembly(
        "fuel_outer", _u_metal_fuel(openmc, "fuel_outer", 19.75, T,
                                    fc.get("fuel_outer")),
        pitch_f, HEX_PIN_RINGS_FUEL, absorber=False)

    follower = None
    if rod_in:
        lat_prim = make_assembly("primary_control",
                                 _b4c(openmc, "primary_control", T),
                                 pitch_p, HEX_PIN_RINGS_PRIMARY, absorber=True)
        lat_sec = make_assembly("secondary_control",
                                _b4c(openmc, "secondary_control", T),
                                pitch_s, HEX_PIN_RINGS_SECONDARY, absorber=True)
    else:
        # withdrawn control assembly = sodium follower filling the duct. Tallied as its
        # own domain so the rod-out endpoint of the gray-rod blend is MEASURED, not
        # assumed; xs_branch maps it onto both control labels.
        follower = reg("control_follower", _sodium(openmc, "control_follower", T))
        homogenize["control_follower"] = {"control_follower": 1.0}

    def interior_fill(mat_id: int):
        """Universe or material filling the inside of a duct ring for a role."""
        name = ID_TO_MATERIAL[mat_id]
        if name == "fuel_inner":
            return lat_inner
        if name == "fuel_outer":
            return lat_outer
        if name == "primary_control":
            return lat_prim if rod_in else follower
        if name == "secondary_control":
            return lat_sec if rod_in else follower
        if name == "reflector":
            return reflector
        if name == "shield":
            return shield
        raise ValueError(f"unmapped hex role {name}")

    # --- geometry: one hex prism per assembly, duct annulus + interior -------
    core_cells: List[object] = []
    hex_outlines: List[object] = []
    for (q, r), mat_id in roles.items():
        cx, cy = _axial_to_xy(q, r, hx.pitch_cm)
        outer_hex = openmc.model.HexagonalPrism(edge_length=Rc, orientation="y",
                                                origin=(cx, cy))
        inner_hex = openmc.model.HexagonalPrism(edge_length=Ri, orientation="y",
                                                origin=(cx, cy))
        fill = interior_fill(mat_id)
        holder = openmc.Cell(fill=fill, region=-inner_hex)
        if isinstance(fill, openmc.HexLattice):
            # pin lattices are built about the origin; translate onto this assembly
            holder.translation = (cx, cy, 0.0)
        core_cells.append(holder)
        core_cells.append(openmc.Cell(fill=duct, region=+inner_hex & -outer_hex))
        hex_outlines.append(outer_hex)

    # radial extent: circumscribe the assembly centres plus one pitch of sodium
    r_core = hx.pitch_cm * (total_rings + 1.2)
    outer_cyl = openmc.ZCylinder(r=r_core, boundary_type="vacuum")
    z0 = openmc.ZPlane(z0=-axial_cm / 2.0, boundary_type="reflective")
    z1 = openmc.ZPlane(z0=+axial_cm / 2.0, boundary_type="reflective")
    axial = +z0 & -z1

    for c in core_cells:
        c.region = c.region & axial
    # Everything inside the vacuum cylinder that is not an assembly is the sodium
    # that fills the inter-assembly gaps and the space out to the barrel. Written as
    # an intersection of hex complements (one per assembly) -- unambiguous, and it
    # keeps this core geometrically identical to the FEM one. It is the slowest cell
    # to track, but this is a one-shot offline calculation.
    gap_region = -outer_cyl & axial
    for h in hex_outlines:
        gap_region = gap_region & +h
    core_cells.append(openmc.Cell(fill=coolant, region=gap_region,
                                  name="inter_assembly_sodium"))

    geometry = openmc.Geometry(openmc.Universe(cells=core_cells))
    model = openmc.Model(geometry=geometry, materials=openmc.Materials(mats))
    return CoreModel(model=model, tally_domains=domains, homogenize=homogenize, notes={
        "n_assemblies": len(roles),
        "pin_pitch_cm": float(pitch_f),
        "assembly_pitch_cm": float(hx.pitch_cm),
        "n_fuel_pins": n_pins(HEX_PIN_RINGS_FUEL),
        "n_primary_absorber_pins": n_pins(HEX_PIN_RINGS_PRIMARY),
        "n_secondary_absorber_pins": n_pins(HEX_PIN_RINGS_SECONDARY),
        "axial_cm": float(axial_cm),
        "double_heterogeneity": "explicit hexagonal pin lattice per assembly",
    })


# --- KP-FHR pebble bed -------------------------------------------------------

def _triso_universe(openmc, kernel, buffer_, ipyc, sic, opyc):
    r = TRISO_R
    s = {k: openmc.Sphere(r=v) for k, v in r.items()}
    return openmc.Universe(cells=[
        openmc.Cell(fill=kernel, region=-s["kernel"]),
        openmc.Cell(fill=buffer_, region=+s["kernel"] & -s["buffer"]),
        openmc.Cell(fill=ipyc, region=+s["buffer"] & -s["ipyc"]),
        openmc.Cell(fill=sic, region=+s["ipyc"] & -s["sic"]),
        openmc.Cell(fill=opyc, region=+s["sic"] & -s["opyc"]),
    ])


def _fuel_pebble_universe(openmc, triso_u, core_graphite, matrix, shell, outside,
                          seed: int = 1):
    """One gFHR/KP-FHR fuel pebble, built ONCE and shared by every fuel pebble.

    Three concentric regions (see PEBBLE_* constants): a low-density graphite
    BUOYANCY CORE (r < 1.38 cm), an explicit TRISO packing in a graphite matrix
    SHELL (1.38 - 1.80 cm), and a fuel-free outer graphite shell (1.80 - 2.00 cm).
    Packing the TRISO into the annular shell rather than the whole interior is not
    cosmetic: it moves every kernel outward, changing the self-shielding the fuel
    sees and the moderator path a thermal neutron takes to reach it.
    """
    core_s = openmc.Sphere(r=PEBBLE_CORE_R)
    fuel_zone = openmc.Sphere(r=PEBBLE_FUEL_ZONE_R)
    shell_s = openmc.Sphere(r=PEBBLE_SHELL_R)

    centers = openmc.model.pack_spheres(
        radius=TRISO_R["opyc"], region=+core_s & -fuel_zone,
        pf=TRISO_PACKING_FRACTION, seed=seed)
    trisos = [openmc.model.TRISO(TRISO_R["opyc"], triso_u, c) for c in centers]

    # bin the particles into a background lattice so tracking stays fast
    ll, ur = np.array([-PEBBLE_FUEL_ZONE_R] * 3), np.array([PEBBLE_FUEL_ZONE_R] * 3)
    shape = (10, 10, 10)
    pitch = (ur - ll) / np.array(shape)
    lattice = openmc.model.create_triso_lattice(
        trisos, ll, pitch, shape, matrix)

    return openmc.Universe(cells=[
        openmc.Cell(fill=core_graphite, region=-core_s),
        openmc.Cell(fill=lattice, region=+core_s & -fuel_zone),
        openmc.Cell(fill=shell, region=+fuel_zone & -shell_s),
        openmc.Cell(fill=outside, region=+shell_s),
    ]), len(trisos)


def _packed_bed(openmc, region, r_bed: float, axial_cm: float, pf: float,
                seed: int) -> np.ndarray:
    """Random pebble packing, cached on disk.

    `openmc.model.pack_spheres` runs a random sequential addition followed by a
    contraction loop, and at pf near the 0.64 jamming limit that loop dominates the
    whole model build -- minutes to hours for a full bed. The packing is a pure
    function of (radius, cylinder size, pf, seed), and EVERY branch case rebuilds the
    same one, so it is computed once and reused. Delete PACKING_CACHE_DIR to force a
    regeneration.
    """
    import hashlib
    # Sphere CENTRES live in a slab of height (axial_cm - 2*r), so the packer divides
    # by zero at axial_cm == 2*r and gives a single unphysical pebble layer just above
    # it. Require real axial room; the pebble-density dip against the reflective top
    # and bottom is an artifact of the 2D-equivalent slab and shrinks as axial_cm grows.
    if axial_cm < 4.0 * PEBBLE_SHELL_R:
        raise ValueError(
            f"axial_cm={axial_cm} is too short for {2*PEBBLE_SHELL_R} cm pebbles; "
            f"use at least {4*PEBBLE_SHELL_R} cm (30 cm is the production default)")
    key = f"{PEBBLE_SHELL_R}_{r_bed}_{axial_cm}_{pf}_{seed}"
    tag = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(PACKING_CACHE_DIR, exist_ok=True)
    path = os.path.join(PACKING_CACHE_DIR, f"pebbles_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    centers = openmc.model.pack_spheres(radius=PEBBLE_SHELL_R, region=region,
                                        pf=pf, seed=seed)
    centers = np.asarray(centers, dtype=float)
    np.save(path, centers)
    return centers


def _graphite_pebble_universe(openmc, graphite, outside):
    shell_s = openmc.Sphere(r=PEBBLE_SHELL_R)
    return openmc.Universe(cells=[
        openmc.Cell(fill=graphite, region=-shell_s),
        openmc.Cell(fill=outside, region=+shell_s),
    ])


def _x_element_region(openmc, cx: float, cy: float, arm_len: float, arm_w: float,
                      angle: float):
    """Union of two crossed slabs = the X-shaped shutdown element footprint.

    Mirrors geometry_pebble._in_cross exactly (same arm_len / arm_w / rotation), so
    the transport model and the FEM mesh describe the same absorber.
    """
    def slab(theta, half_long, half_short):
        # local axes rotated by theta
        ux, uy = np.cos(theta), np.sin(theta)
        vx, vy = -np.sin(theta), np.cos(theta)
        du = ux * cx + uy * cy
        dv = vx * cx + vy * cy
        p_long_lo = openmc.Plane(a=ux, b=uy, c=0.0, d=du - half_long)
        p_long_hi = openmc.Plane(a=ux, b=uy, c=0.0, d=du + half_long)
        p_short_lo = openmc.Plane(a=vx, b=vy, c=0.0, d=dv - half_short)
        p_short_hi = openmc.Plane(a=vx, b=vy, c=0.0, d=dv + half_short)
        return +p_long_lo & -p_long_hi & +p_short_lo & -p_short_hi

    return slab(angle, arm_len, arm_w / 2.0) | slab(angle + np.pi / 2.0,
                                                    arm_len, arm_w / 2.0)


def fhr_model(pb: PebbleCoreConfig, state: BranchState, *,
              graphite_pebble_frac: Optional[float] = None,
              # 16 cm = 4 pebble diameters. Measured: packing to pf=0.60 in a slab
              # only 2 diameters thick is effectively unreachable and the contraction
              # loop thrashes for >10 min; at 16 cm it converges in ~90 s (3459
              # pebbles). Taller adds cost, not physics -- the slab is axially
              # reflective, so its height is a free parameter.
              axial_cm: float = 16.0,
              bed_packing_fraction: float = BED_PACKING_FRACTION,
              enrichment: float = FHR_ENRICHMENT_WT_PCT,
              fuel_composition: Optional[Dict[str, float]] = None,
              seed: int = 1) -> CoreModel:
    """Full-core KP-FHR pebble-bed model matching geometry_pebble.make_pebble_core.

    Same gFHR radial build (full-diameter pebble bed -> 60 cm graphite side reflector
    holding the control channels -> SS316H barrel -> FLiBe downcomer -> SS316H
    vessel), same shutdown X-elements inserted directly into the bed, same radii.
    Pebbles are explicit spheres on a random (RSA) packing; fuel pebbles carry an
    explicit TRISO lattice in an annular fuel shell. Radial vacuum + axially
    reflective.

    NOTE ON COST: at the gFHR bed radius (120 cm) a 16 cm slab holds ~13k pebbles,
    ~4x the old reduced-radius dev core. The packing is cached (see _packed_bed), but
    the first build of a new radius is the slow step.

    `fuel_composition` = {nuclide: atom_fraction} for the TRISO kernel at
    state.burnup_mwd_kg (depletion table); None = fresh.
    """
    import openmc
    from geometry_pebble import _structure_centers

    T = state.temperature_k
    gpf = pb.graphite_pebble_frac if graphite_pebble_frac is None else graphite_pebble_frac
    rod_in = state.rod == "in"

    # materials (densities are the published gFHR values; see the PEBBLE_* constants)
    flibe = _flibe(openmc, "coolant", T)
    peb_core = _graphite(openmc, "pebble_core", PEBBLE_CORE_DENSITY, T)
    matrix = _graphite(openmc, "pebble_matrix", PEBBLE_MATRIX_DENSITY, T)
    shell = _graphite(openmc, "pebble_shell", PEBBLE_SHELL_DENSITY, T)
    gpeb = _graphite(openmc, "graphite_pebble", PEBBLE_SHELL_DENSITY, T)
    refl = _graphite(openmc, "reflector", REFLECTOR_GRAPHITE_DENSITY, T)
    # barrel / downcomer / vessel are resolved separately here and homogenized back
    # into the single FEM `vessel` ring below.
    barrel = _ss316(openmc, "barrel", T, density=SS316H_DENSITY)
    downcomer = _flibe(openmc, "downcomer", T)
    vessel = _ss316(openmc, "vessel", T, density=SS316H_DENSITY)
    kernel = _uco_kernel(openmc, "fuel_pebble", enrichment, T, fuel_composition)
    buffer_ = _graphite(openmc, "triso_buffer", 1.05, T)
    ipyc = _graphite(openmc, "triso_ipyc", 1.90, T)
    opyc = _graphite(openmc, "triso_opyc", 1.90, T)
    sic = openmc.Material(name="triso_sic")
    sic.add_element("Si", 1.0)
    sic.add_element("C", 1.0)
    sic.set_density("g/cm3", 3.18)
    sic.temperature = T

    # gFHR control/shutdown absorber: 100 at% B-10 B4C at 1.76 g/cm3.
    control = (_b4c(openmc, "control_element", T, density=B4C_CONTROL_DENSITY,
                    b10_enrichment=B4C_CONTROL_B10_ENRICH) if rod_in else None)
    shutdown = (_b4c(openmc, "shutdown_element", T, density=B4C_CONTROL_DENSITY,
                     b10_enrichment=B4C_CONTROL_B10_ENRICH) if rod_in else None)
    follower = _flibe(openmc, "control_follower", T) if not rod_in else None

    mats = [flibe, peb_core, matrix, shell, gpeb, refl, barrel, downcomer, vessel,
            kernel, buffer_, ipyc, opyc, sic]
    mats += [m for m in (control, shutdown, follower) if m is not None]

    # pebble universes (defined once, instantiated per pebble)
    triso_u = _triso_universe(openmc, kernel, buffer_, ipyc, sic, opyc)
    fuel_peb_u, n_triso = _fuel_pebble_universe(openmc, triso_u, peb_core, matrix,
                                                shell, flibe, seed=seed)
    graph_peb_u = _graphite_pebble_universe(openmc, gpeb, flibe)

    # --- radial build --------------------------------------------------------
    z0 = openmc.ZPlane(z0=-axial_cm / 2.0, boundary_type="reflective")
    z1 = openmc.ZPlane(z0=+axial_cm / 2.0, boundary_type="reflective")
    axial = +z0 & -z1

    # R_center_refl is 0 for gFHR/KP-FHR (full-diameter bed, no central column); the
    # surface is only created when a central column is actually configured.
    c_center = (openmc.ZCylinder(r=pb.R_center_refl) if pb.R_center_refl > 0.0
                else None)
    c_bed = openmc.ZCylinder(r=pb.R_bed)
    c_refl = openmc.ZCylinder(r=pb.R_refl)
    c_barrel = openmc.ZCylinder(r=pb.R_barrel)
    c_down = openmc.ZCylinder(r=pb.R_downcomer)
    c_vessel = openmc.ZCylinder(r=pb.R_vessel, boundary_type="vacuum")

    ctrl_centers, shut_centers = _structure_centers(pb)

    # shutdown X footprints in the inner bed. geometry_pebble rotates EVERY X by a
    # fixed pi/4 (see its _cross_fill_nodes calls); match that exactly or the two
    # models describe different absorbers.
    shut_regions = [
        _x_element_region(openmc, float(c[0]), float(c[1]), pb.x_arm_len,
                          pb.x_arm_w, np.pi / 4.0)
        for c in shut_centers]

    # Control channels in the outer reflector. The FEM mesh draws an explicit
    # graphite channel lining (`channel_wall`) around each element; here the element
    # simply sits in the reflector, which is the SAME material, so the transport
    # geometry is unchanged. The shutdown X in the bed is the one place the two
    # differ: its FEM liner separates B4C from pebbles/FLiBe, while here the blade
    # faces the packing directly.
    ctrl_cyls = [openmc.ZCylinder(x0=float(c[0]), y0=float(c[1]), r=pb.r_ctrl)
                 for c in ctrl_centers]

    # pebble bed region: the cylinder (annulus if a central column is configured)
    # minus the shutdown footprints
    bed_region = -c_bed & axial
    if c_center is not None:
        bed_region = bed_region & +c_center
    for reg in shut_regions:
        bed_region = bed_region & ~reg

    # openmc.model.pack_spheres only accepts a simple container (cylinder, sphere,
    # spherical shell, rectangular prism), so pack the full cylinder and then discard
    # the centres that fall in a central reflector column (if any) or a shutdown
    # channel. Local packing fraction is unaffected by the discard. The exclusion
    # tests mirror geometry_pebble._rsa_pebbles exactly (same inflated arm dimensions,
    # same pi/4 rotation) so both models pack against the same obstacles.
    from geometry_pebble import _in_cross

    centers = _packed_bed(openmc, -c_bed & axial, pb.R_bed, axial_cm,
                          bed_packing_fraction, seed)
    sh_len = pb.x_arm_len + PEBBLE_SHELL_R
    sh_w = pb.x_arm_w + 2 * PEBBLE_SHELL_R
    keep = []
    for c in centers:
        if pb.R_center_refl > 0.0 and (
                np.hypot(c[0], c[1]) < pb.R_center_refl + PEBBLE_SHELL_R):
            continue                                    # central graphite column
        if any(_in_cross((c[0], c[1]), sc, sh_len, sh_w, np.pi / 4.0)
               for sc in shut_centers):
            continue                                    # shutdown channel
        keep.append(c)
    centers = np.asarray(keep, dtype=float)
    if centers.size == 0:
        raise ValueError("no pebbles placed: check R_center_refl / R_bed / axial_cm")

    rng = np.random.default_rng(seed)
    is_graphite = rng.random(len(centers)) < gpf
    # any configured unfueled radial band is graphite-only. gFHR/KP-FHR has none
    # (R_fuel_in=0, R_fuel_out=R_bed), so this is a no-op on the default core.
    rad = np.hypot(centers[:, 0], centers[:, 1])
    is_graphite |= (rad < pb.R_fuel_in) | (rad > pb.R_fuel_out)

    pebbles = [openmc.model.TRISO(PEBBLE_SHELL_R,
                                  graph_peb_u if g else fuel_peb_u, c)
               for c, g in zip(centers, is_graphite)]
    ll = np.array([-pb.R_bed, -pb.R_bed, -axial_cm / 2.0])
    ur = np.array([pb.R_bed, pb.R_bed, axial_cm / 2.0])
    shape = (24, 24, max(4, int(axial_cm / 5)))
    pitch = (ur - ll) / np.array(shape)
    bed_lattice = openmc.model.create_triso_lattice(pebbles, ll, pitch, shape, flibe)

    cells = [openmc.Cell(fill=bed_lattice, region=bed_region, name="pebble_bed")]
    if c_center is not None:
        cells.append(openmc.Cell(fill=refl, region=-c_center & axial,
                                 name="center_reflector"))
    for i, reg in enumerate(shut_regions):
        # X blade (B4C or FLiBe follower) inserted directly into the bed
        blade = reg & -c_bed & axial
        if c_center is not None:
            blade = blade & +c_center
        cells.append(openmc.Cell(fill=shutdown if rod_in else follower,
                                 region=blade, name=f"shutdown_{i}"))

    refl_region = +c_bed & -c_refl & axial
    for cyl in ctrl_cyls:
        cells.append(openmc.Cell(fill=control if rod_in else follower,
                                 region=-cyl & refl_region, name="control"))
        refl_region = refl_region & +cyl
    cells.append(openmc.Cell(fill=refl, region=refl_region, name="outer_reflector"))
    # barrel / downcomer / vessel: three thin rings, resolved separately so each is
    # collapsed against its own spectrum, then homogenized into the FEM `vessel` ring.
    cells.append(openmc.Cell(fill=barrel, region=+c_refl & -c_barrel & axial,
                             name="barrel"))
    cells.append(openmc.Cell(fill=downcomer, region=+c_barrel & -c_down & axial,
                             name="downcomer"))
    cells.append(openmc.Cell(fill=vessel, region=+c_down & -c_vessel & axial,
                             name="vessel"))

    geometry = openmc.Geometry(openmc.Universe(cells=cells))

    # A FEM `fuel_pebble` node is the WHOLE pebble, but transport resolves it into
    # kernel / buffer / IPyC / SiC / OPyC / buoyancy core / matrix / shell. Each
    # constituent is its own tally domain (that is what makes the self-shielding real)
    # and xs_openmc homogenizes them back with fuel_pebble_volumes(). The FEM `vessel`
    # node is likewise the whole barrel+downcomer+vessel stack.
    domains = {
        "fuel_pebble": kernel, "graphite_pebble": gpeb, "reflector": refl,
        "coolant": flibe,
        "_pebble_core": peb_core, "_pebble_matrix": matrix, "_pebble_shell": shell,
        "_triso_buffer": buffer_, "_triso_ipyc": ipyc, "_triso_sic": sic,
        "_triso_opyc": opyc,
        "_barrel": barrel, "_downcomer": downcomer, "_vessel": vessel,
    }
    # per-unit-height annulus areas [cm^3/cm] for the vessel-ring homogenization
    ring = lambda r_out, r_in: float(np.pi * (r_out ** 2 - r_in ** 2))
    homogenize: Dict[str, Dict[str, float]] = {
        "fuel_pebble": fuel_pebble_volumes(),
        "graphite_pebble": {"graphite_pebble": 1.0},
        "reflector": {"reflector": 1.0},
        "coolant": {"coolant": 1.0},
        "vessel": {
            "_barrel": ring(pb.R_barrel, pb.R_refl),
            "_downcomer": ring(pb.R_downcomer, pb.R_barrel),
            "_vessel": ring(pb.R_vessel, pb.R_downcomer),
        },
    }
    if rod_in:
        domains["control_element"] = control
        domains["shutdown_element"] = shutdown
        homogenize["control_element"] = {"control_element": 1.0}
        homogenize["shutdown_element"] = {"shutdown_element": 1.0}
    else:
        domains["control_follower"] = follower
        homogenize["control_follower"] = {"control_follower": 1.0}

    model = openmc.Model(geometry=geometry, materials=openmc.Materials(mats))
    return CoreModel(model=model, tally_domains=domains, homogenize=homogenize, notes={
        "n_pebbles": int(len(centers)),
        "n_fuel_pebbles": int((~is_graphite).sum()),
        "n_triso_per_pebble": int(n_triso),
        "bed_packing_fraction": float(bed_packing_fraction),
        "triso_packing_fraction": TRISO_PACKING_FRACTION,
        "enrichment_wt_pct": float(enrichment),
        "kernel_form": "UC(0.5)O(1.5) oxycarbide, 10.5 g/cm3",
        "pebble_build_cm": [PEBBLE_CORE_R, PEBBLE_FUEL_ZONE_R, PEBBLE_SHELL_R],
        "radial_build_cm": {"bed": pb.R_bed, "reflector": pb.R_refl,
                            "barrel": pb.R_barrel, "downcomer": pb.R_downcomer,
                            "vessel": pb.R_vessel},
        "control_absorber": (f"B4C, {100.0 * B4C_CONTROL_B10_ENRICH:g} at% B-10, "
                             f"{B4C_CONTROL_DENSITY} g/cm3"),
        "axial_cm": float(axial_cm),
        "geometry_source": ("gFHR benchmark (Satvat et al. 2021 / INL VTB) for "
                            "dimensions and materials; Hermes (NRC KP-TR-024-NP, "
                            "ML24095A258) for "
                            "the 4 reflector control + 3 in-bed shutdown element count"),
        "double_heterogeneity": ("explicit TRISO lattice in an annular fuel shell "
                                 "inside explicit pebbles"),
    })


# volumes [cm^3] of the constituents of one fuel pebble, used to homogenize the
# separately-tallied TRISO layers back into the single `fuel_pebble` FEM node.
def fuel_pebble_volumes() -> Dict[str, float]:
    r = TRISO_R
    v = lambda rad: 4.0 / 3.0 * np.pi * rad ** 3
    v_core = v(PEBBLE_CORE_R)                       # low-density buoyancy core
    v_zone = v(PEBBLE_FUEL_ZONE_R) - v_core         # TRISO-bearing shell
    v_triso = v(r["opyc"])
    n = TRISO_PACKING_FRACTION * v_zone / v_triso
    return {
        "fuel_pebble": n * v(r["kernel"]),                       # UCO kernels
        "_triso_buffer": n * (v(r["buffer"]) - v(r["kernel"])),
        "_triso_ipyc": n * (v(r["ipyc"]) - v(r["buffer"])),
        "_triso_sic": n * (v(r["sic"]) - v(r["ipyc"])),
        "_triso_opyc": n * (v(r["opyc"]) - v(r["sic"])),
        "_pebble_core": v_core,
        "_pebble_matrix": v_zone - n * v_triso,
        "_pebble_shell": v(PEBBLE_SHELL_R) - v(PEBBLE_FUEL_ZONE_R),
    }


def build(reactor_type: str, state: BranchState, cfg, **kw) -> CoreModel:
    """Dispatch: reactor_type -> full-core CoreModel for one branch."""
    if reactor_type == "fhr":
        return fhr_model(cfg, state, **kw)
    return natrium_model(cfg, state, **kw)
