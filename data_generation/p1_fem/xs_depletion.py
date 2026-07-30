"""Unit-cell depletion: fuel isotopics as a function of burnup.

The dataset varies fuel burnup across samples. Previously that was a hand multiplier
("nuSf *= (1 - burn)", "Sr2 *= (1 + 0.6*burn)") with no traceability. This module
replaces the assumption with computed isotopics: OpenMC depletion of a REPRESENTATIVE
UNIT CELL produces the actual nuclide inventory at each burnup point, and
xs_openmc.py then builds the full-core branch models from those compositions.

Unit cell, not full core: depleting an explicit-TRISO full core at several burnup
points is not affordable, and it is not standard practice either. Lattice physics
depletes a representative cell (here: one fuel pebble in its FLiBe cell for `fhr`,
one fuel assembly with reflective boundaries for `hex`) and applies the resulting
isotopics to the full-core spectrum calculation. The spectrum used for the group
collapse is still the full-core one -- only the ISOTOPICS come from the unit cell.
That is the approximation, and it is documented in the output JSON.

ONE CELL PER FUEL ZONE. `hex` has two enrichment zones (fuel_inner 15.50 wt%,
fuel_outer 19.75 wt%; see openmc_models.HEX_ENRICHMENT_WT_PCT) and therefore takes
two depletion runs. They are not interchangeable: an assembly at 15.50 wt% reaches a
different inventory at the same burnup than one at 19.75 -- more Pu-239 relative to
remaining U-235, different fission-product loading per unit fissile -- and because the
depleted composition OVERRIDES the fresh material in openmc_models, serving one zone's
isotopics to the other silently discards the enrichment zoning altogether. Schema v1
did exactly that (a single "fuel" key plus an implicit fallback in composition_at) and
is now rejected by load_composition_table. `fhr` has one fuel material, so one run.

Requires a depletion chain (`--chain`), e.g. the ENDF/B-VIII.0 chains from
https://openmc.org/depletion-chains/ . Run once per reactor; the result is cached:

    python xs_depletion.py --reactor fhr --chain chain_endfb80_pwr.xml \\
        --out depletion_fhr.json --burnups 0 2 20 50 90 130 160 190

    python xs_depletion.py --reactor hex --chain chain_endfb80_fast.xml \\
        --out depletion_natrium.json --burnups 0 2 20 40 60

Burnups are cumulative MWd/kgHM. The grid must COVER the sampling range in
datagen_config (fhr 2-190, i.e. the gFHR ~20% FIMA peak discharge burnup at
~9.6 MWd/kgHM per % FIMA; hex 2-60) and should include an early point near 2 so branch
cases interpolate past Xe/Sm equilibrium rather than across it. Output schema:

    { "schema_version": 2, "reactor_type", "burnups_mwd_kg": [...],
      "zone_enrichment_wt_pct": {"<material>": wt%},
      "unit_cell": {"<material>": {...}}, "unit_cell_keff": {"<material>": [...]},
      "chain", "specific_power_w_per_ghm", "temperature_k",
      "compositions": { "<material>": [ {nuclide: atom/b-cm}, ... one per burnup ] } }
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from datagen_config import DEFAULT
from openmc_models import (PEBBLE_CORE_DENSITY, PEBBLE_MATRIX_DENSITY,
                           PEBBLE_SHELL_DENSITY, PEBBLE_SHELL_R,
                           FHR_ENRICHMENT_WT_PCT, HEX_ENRICHMENT_WT_PCT,
                           HEX_PIN_RINGS_FUEL,
                           HEX_R_FUEL_FRAC, TRISO_R, n_pins,
                           _flibe, _graphite, _ht9, _sodium, _triso_universe,
                           _u_metal_fuel, _uco_kernel)


# Specific power [W/gHM] used to convert depletion time to burnup. Representative
# values; burnup (not time) is the reported axis, so this only sets the flux level
# during the depletion, which has a second-order effect on the isotopics.
SPECIFIC_POWER_W_PER_GHM = {"fhr": 30.0, "hex": 45.0}

# Nuclides below this atom density [atom/b-cm] are dropped from the exported
# composition (they carry no reactivity and may be missing from the CE library).
MIN_ATOM_DENSITY = 1.0e-12


def _available_nuclides(openmc) -> set:
    """Nuclide names present in the configured continuous-energy data library."""
    path = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not path or not os.path.exists(path):
        return set()
    lib = openmc.data.DataLibrary.from_xml(path)
    return {e["materials"][0] for e in lib.libraries
            if e["type"] == "neutron" and e["materials"]}


# --- unit cells --------------------------------------------------------------

def _fhr_unit_cell(openmc, temperature_k: float, enrichment: float):
    """One fuel pebble (explicit TRISO) in its FLiBe cell, reflective on all sides.

    Cell size is set so the pebble volume fraction equals the bed packing fraction,
    i.e. the pebble sees the same moderator-to-fuel ratio it sees in the bed.
    """
    from openmc_models import BED_PACKING_FRACTION, _fuel_pebble_universe

    T = temperature_k
    kernel = _uco_kernel(openmc, "fuel_pebble", enrichment, T)
    peb_core = _graphite(openmc, "pebble_core", PEBBLE_CORE_DENSITY, T)
    matrix = _graphite(openmc, "pebble_matrix", PEBBLE_MATRIX_DENSITY, T)
    shell = _graphite(openmc, "pebble_shell", PEBBLE_SHELL_DENSITY, T)
    buffer_ = _graphite(openmc, "triso_buffer", 1.05, T)
    ipyc = _graphite(openmc, "triso_ipyc", 1.90, T)
    opyc = _graphite(openmc, "triso_opyc", 1.90, T)
    sic = openmc.Material(name="triso_sic")
    sic.add_element("Si", 1.0)
    sic.add_element("C", 1.0)
    sic.set_density("g/cm3", 3.18)
    sic.temperature = T
    flibe = _flibe(openmc, "coolant", T)

    triso_u = _triso_universe(openmc, kernel, buffer_, ipyc, sic, opyc)
    peb_u, n_triso = _fuel_pebble_universe(openmc, triso_u, peb_core, matrix, shell,
                                           flibe)

    v_peb = 4.0 / 3.0 * np.pi * PEBBLE_SHELL_R ** 3
    a = (v_peb / BED_PACKING_FRACTION) ** (1.0 / 3.0)      # cubic cell edge
    box = openmc.model.RectangularParallelepiped(
        -a / 2, a / 2, -a / 2, a / 2, -a / 2, a / 2, boundary_type="reflective")
    cell = openmc.Cell(fill=peb_u, region=-box)
    geometry = openmc.Geometry([cell])
    mats = openmc.Materials([kernel, peb_core, matrix, shell, buffer_, ipyc, opyc,
                             sic, flibe])
    # Depletion needs the heavy-metal volume to turn power into burnup. It is known
    # exactly (n_triso identical kernels), so set it analytically rather than paying
    # for a stochastic volume calculation -- which in any case cannot infer a
    # bounding box for a material nested inside a lattice.
    v_kernel = n_triso * 4.0 / 3.0 * np.pi * TRISO_R["kernel"] ** 3
    kernel.volume = float(v_kernel)
    return geometry, mats, {"fuel_pebble": kernel}, {
        "kind": "single fuel pebble (explicit TRISO) in a reflective FLiBe cell",
        "cell_edge_cm": float(a), "n_triso": int(n_triso),
        "fuel_volume_cm3": float(v_kernel)}


def _hex_unit_cell(openmc, temperature_k: float, enrichment: float,
                   hx, material_name: str = "fuel") -> tuple:
    """One Natrium fuel assembly (explicit 217-pin lattice, HT9 duct), reflective.

    `material_name` is the FEM fuel material this cell represents ("fuel_inner" /
    "fuel_outer"). It names the depletable material so the exported table is keyed by
    zone, and openmc_models can look up each zone's own isotopics.
    """
    from geometry import SQRT3, DUCT_RING_FRAC
    from openmc_models import _hex_assembly_universe

    T = temperature_k
    Rc = (hx.pitch_cm - hx.gap_cm) / SQRT3
    Ri = Rc * (1.0 - DUCT_RING_FRAC)
    pitch = Ri / (HEX_PIN_RINGS_FUEL - 0.5)

    fuel = _u_metal_fuel(openmc, material_name, enrichment, T)
    clad = _ht9(openmc, "clad", T)
    bond = _sodium(openmc, "bond", T)
    coolant = _sodium(openmc, "coolant", T)
    duct = _ht9(openmc, "duct", T, density=7.0)

    lat = _hex_assembly_universe(openmc, material_name, fuel, bond, clad, coolant,
                                 pitch, HEX_PIN_RINGS_FUEL)
    inner = openmc.model.HexagonalPrism(edge_length=Ri, orientation="y")
    outer = openmc.model.HexagonalPrism(edge_length=Rc, orientation="y",
                                        boundary_type="reflective")
    z0 = openmc.ZPlane(z0=-10.0, boundary_type="reflective")
    z1 = openmc.ZPlane(z0=+10.0, boundary_type="reflective")
    axial = +z0 & -z1
    cells = [openmc.Cell(fill=lat, region=-inner & axial),
             openmc.Cell(fill=duct, region=+inner & -outer & axial)]
    geometry = openmc.Geometry(openmc.Universe(cells=cells))
    mats = openmc.Materials([fuel, clad, bond, coolant, duct])
    # analytic heavy-metal volume (see the pebble cell for why it is not stochastic)
    height = 20.0
    v_fuel = n_pins(HEX_PIN_RINGS_FUEL) * np.pi * (HEX_R_FUEL_FRAC * pitch) ** 2 * height
    fuel.volume = float(v_fuel)
    return geometry, mats, {material_name: fuel}, {
        "kind": f"single {material_name} assembly (explicit "
                f"{n_pins(HEX_PIN_RINGS_FUEL)}-pin lattice) with reflective duct",
        "pin_pitch_cm": float(pitch), "fuel_volume_cm3": float(v_fuel)}


# --- depletion driver --------------------------------------------------------

SCHEMA_VERSION = 2


def default_zones(reactor_type: str) -> Dict[str, float]:
    """FEM fuel material -> fresh enrichment [wt% U-235]; one unit cell each.

    Read from the same constants openmc_models uses to build the transport core, so
    the depletion isotopics and the transport zones cannot drift apart.

    hex has TWO fuel zones at different enrichments and therefore needs TWO depletion
    runs: an assembly at 15.50 wt% burns to a different inventory than one at 19.75
    (different Pu buildup rate, different fission-product yield per unit burnup), and
    since the composition OVERRIDES the fresh material in openmc_models, serving one
    zone's isotopics to the other silently discards the zoning entirely. fhr has a
    single fuel material, so one run.
    """
    if reactor_type == "fhr":
        return {"fuel_pebble": FHR_ENRICHMENT_WT_PCT}
    return dict(HEX_ENRICHMENT_WT_PCT)


def default_enrichment(reactor_type: str) -> float:
    """Highest fresh enrichment across a reactor's fuel zones [wt% U-235].

    Retained for reporting / single-cell overrides only. Prefer default_zones: for
    `hex` a single scalar cannot describe the two-zone core.
    """
    return max(default_zones(reactor_type).values())


def _deplete_one_zone(openmc, reactor_type: str, zone: str, enrichment: float,
                      bu: List[float], chain: str, temperature_k: float,
                      particles: int, batches: int, inactive: int,
                      workdir: str) -> tuple:
    """Deplete ONE zone's unit cell. Returns (compositions, k_inf, cell_notes)."""
    import openmc.deplete

    if reactor_type == "fhr":
        geometry, mats, depletable, cell_notes = _fhr_unit_cell(
            openmc, temperature_k, enrichment)
    else:
        geometry, mats, depletable, cell_notes = _hex_unit_cell(
            openmc, temperature_k, enrichment, DEFAULT.hexcore, material_name=zone)

    for m in depletable.values():
        m.depletable = True      # volumes were set analytically by the cell builder

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {"method": "interpolation", "default": temperature_k}

    model = openmc.Model(geometry=geometry, materials=mats, settings=settings)

    # one subdirectory per zone: OpenMC writes depletion_results.h5 to cwd, so two
    # zones sharing a workdir would clobber each other's results
    zone_dir = os.path.join(workdir, zone)
    os.makedirs(zone_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(zone_dir)
    try:
        op = openmc.deplete.CoupledOperator(model, chain)
        steps = [b2 - b1 for b1, b2 in zip(bu, bu[1:])]
        integrator = openmc.deplete.PredictorIntegrator(
            op, steps, power_density=SPECIFIC_POWER_W_PER_GHM[reactor_type],
            timestep_units="MWd/kg")
        integrator.integrate()
        results = openmc.deplete.Results("depletion_results.h5")
        avail = _available_nuclides(openmc)

        out: List[Dict[str, float]] = []
        for step in range(len(bu)):
            by_name = {m.name: m for m in results.export_to_materials(step)}
            dens = by_name[zone].get_nuclide_atom_densities()
            out.append({nuc: float(d) for nuc, d in dens.items()
                        if float(d) > MIN_ATOM_DENSITY and (not avail or nuc in avail)})
        keff = [float(k.n) for k in results.get_keff()[1]]
    finally:
        os.chdir(cwd)
    return out, keff, cell_notes


def run_depletion(reactor_type: str, burnups_mwd_kg: List[float], chain: str, *,
                  temperature_k: float = 900.0,
                  enrichment: Optional[float] = None,
                  particles: int = 5000, batches: int = 60, inactive: int = 15,
                  workdir: str = "depletion_run") -> dict:
    """Deplete one unit cell PER FUEL ZONE and return the composition table.

    burnups_mwd_kg must be ascending and start at 0 (fresh). Returns a table whose
    `compositions` is keyed by FEM fuel material name -> list of {nuclide: atom/b-cm},
    one entry per burnup. `hex` produces two keys (fuel_inner, fuel_outer) at their own
    enrichments; `fhr` produces one (fuel_pebble). See default_zones.

    `enrichment`, if given, overrides EVERY zone with that one value -- a deliberate
    single-cell shortcut for cheap runs, not the default, since it collapses the
    Natrium enrichment zoning.
    """
    import openmc

    zones = default_zones(reactor_type)
    if enrichment is not None:
        zones = {z: float(enrichment) for z in zones}
    bu = [float(b) for b in burnups_mwd_kg]
    if bu[0] != 0.0 or any(b2 <= b1 for b1, b2 in zip(bu, bu[1:])):
        raise ValueError("burnups must be ascending and start at 0")

    comps: Dict[str, List[Dict[str, float]]] = {}
    keffs: Dict[str, List[float]] = {}
    notes: Dict[str, dict] = {}
    for i, (zone, enr) in enumerate(sorted(zones.items()), start=1):
        print(f"[{i}/{len(zones)}] depleting {zone} at {enr:g} wt% U-235 "
              f"over {len(bu)} burnup points", flush=True)
        comps[zone], keffs[zone], notes[zone] = _deplete_one_zone(
            openmc, reactor_type, zone, enr, bu, chain, temperature_k,
            particles, batches, inactive, workdir)

    return {
        "schema_version": SCHEMA_VERSION,
        "reactor_type": reactor_type,
        "burnups_mwd_kg": bu,
        # per-zone, because the cells differ (hex: two enrichments)
        "zone_enrichment_wt_pct": zones,
        "unit_cell": notes,
        "unit_cell_keff": keffs,
        "chain": os.path.basename(chain),
        "specific_power_w_per_ghm": SPECIFIC_POWER_W_PER_GHM[reactor_type],
        "temperature_k": temperature_k,
        "compositions": comps,
    }


class StaleDepletionTable(RuntimeError):
    """Raised for a depletion table that predates per-zone compositions."""


def load_composition_table(path: str) -> Optional[dict]:
    """Load a depletion table if present, else None (fresh-fuel-only branch grid).

    Rejects schema v1 (a single "fuel" cell for the whole core). Such a table is not
    merely older: served through the old implicit fallback it gave BOTH Natrium
    enrichment zones the same isotopics, silently overriding fuel_inner's lower
    enrichment. Failing here beats reproducing that.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        table = json.load(f)

    version = int(table.get("schema_version", 1))
    if version < SCHEMA_VERSION:
        keys = ", ".join(table.get("compositions", {})) or "(none)"
        raise StaleDepletionTable(
            f"{os.path.basename(path)} is a schema-v{version} depletion table "
            f"(compositions keyed by: {keys}); v{SCHEMA_VERSION} keys compositions by "
            f"FEM fuel material so each zone carries its own isotopics.\n"
            f"A v1 hex table holds ONE cell depleted at a single enrichment, and the "
            f"composition overrides the fresh material in openmc_models -- so reusing "
            f"it would give fuel_inner (15.50 wt%) the isotopics of fuel_outer "
            f"(19.75 wt%) and discard the enrichment zoning.\n"
            f"Regenerate:  python xs_depletion.py --reactor "
            f"{table.get('reactor_type', '<hex|fhr>')} --chain <chain.xml> --out "
            f"{os.path.basename(path)}")
    return table


def composition_at(table: dict, material: str, burnup_mwd_kg: float
                   ) -> Optional[Dict[str, float]]:
    """Nuclide atom densities [atom/b-cm] at a burnup, linearly interpolated.

    Interpolation is done on atom densities, which is exact for the fresh endpoints
    and is the standard treatment between depletion points. Outside the tabulated
    range the endpoint composition is used (clamped, never extrapolated).

    `material` must be a key of the table. There is deliberately NO fallback to
    another zone's inventory: the composition overrides the fresh material, so serving
    fuel_outer's isotopics for fuel_inner would erase the enrichment zoning with no
    outward sign. A missing zone is a table that needs regenerating.
    """
    if table is None:
        return None
    comps_all = table["compositions"]
    if material not in comps_all:
        raise KeyError(
            f"depletion table has no compositions for {material!r} (has: "
            f"{', '.join(comps_all)}). Every fuel material in the transport model "
            f"needs its own depleted cell -- see xs_depletion.default_zones.")
    bu = np.asarray(table["burnups_mwd_kg"], dtype=float)
    comps = comps_all[material]
    b = float(np.clip(burnup_mwd_kg, bu[0], bu[-1]))
    j = int(np.searchsorted(bu, b, side="right") - 1)
    j = max(0, min(j, len(bu) - 2)) if len(bu) > 1 else 0
    if len(bu) == 1 or b == bu[j]:
        return dict(comps[j])
    w = (b - bu[j]) / (bu[j + 1] - bu[j])
    lo, hi = comps[j], comps[j + 1]
    out = {}
    for nuc in set(lo) | set(hi):
        out[nuc] = (1.0 - w) * lo.get(nuc, 0.0) + w * hi.get(nuc, 0.0)
    return {n: v for n, v in out.items() if v > MIN_ATOM_DENSITY}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reactor", choices=["hex", "fhr"], required=True)
    ap.add_argument("--chain", required=True, help="OpenMC depletion chain XML")
    ap.add_argument("--out", required=True)
    ap.add_argument("--burnups", type=float, nargs="+",
                    default=[0.0, 2.0, 20.0, 40.0, 60.0, 100.0],
                    help="cumulative burnup points [MWd/kgHM], ascending, first = 0. "
                         "Depletion must START fresh, but include an early point "
                         "(~2) so branch cases can be built past Xe/Sm equilibrium "
                         "instead of interpolating across that step change.")
    ap.add_argument("--temperature", type=float, default=900.0)
    ap.add_argument("--enrichment", type=float, default=None,
                    help="wt%% U-235 forced onto EVERY fuel zone (cheap single-cell "
                         "shortcut). Default depletes each zone at its own documented "
                         "enrichment: fhr fuel_pebble 19.55 (gFHR UCO); hex "
                         "fuel_inner 15.50 + fuel_outer 19.75, as two runs.")
    ap.add_argument("--particles", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--inactive", type=int, default=15)
    ap.add_argument("--workdir", default="depletion_run")
    args = ap.parse_args()

    table = run_depletion(
        args.reactor, args.burnups, args.chain, temperature_k=args.temperature,
        enrichment=args.enrichment, particles=args.particles, batches=args.batches,
        inactive=args.inactive, workdir=args.workdir)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote depletion table ({len(args.burnups)} burnup points, "
          f"{len(table['compositions'])} fuel zone(s)) -> {args.out}")
    for zone, keff in sorted(table["unit_cell_keff"].items()):
        enr = table["zone_enrichment_wt_pct"][zone]
        print(f"  {zone} ({enr:g} wt%) k_inf: "
              + ", ".join(f"{k:.5f}" for k in keff))


if __name__ == "__main__":
    main()
