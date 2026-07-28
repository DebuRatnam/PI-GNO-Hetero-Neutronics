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

Requires a depletion chain (`--chain`), e.g. the ENDF/B-VIII.0 casl/pwr chain from
https://openmc.org/depletion-chains/ . Run once per reactor; the result is cached:

    python xs_depletion.py --reactor fhr --chain chain_endfb80_pwr.xml \\
        --out depletion_fhr.json --burnups 0 20 40 60 80 100

    python xs_depletion.py --reactor hex --chain chain_endfb80_pwr.xml \\
        --out depletion_natrium.json --burnups 0 20 40 60 80

Burnups are cumulative MWd/kgHM. Output schema:

    { "reactor_type", "burnups_mwd_kg": [...], "unit_cell", "chain", "power_density",
      "compositions": { "<material>": [ {nuclide: atom/b-cm}, ... one per burnup ] } }
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from datagen_config import DEFAULT
from openmc_models import (PEBBLE_SHELL_R, HEX_PIN_RINGS_FUEL, HEX_R_FUEL_FRAC,
                           TRISO_R, n_pins,
                           _flibe, _graphite, _ht9, _sodium, _triso_universe,
                           _u_metal_fuel, _uo2_kernel)


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
    kernel = _uo2_kernel(openmc, "fuel_pebble", enrichment, T)
    matrix = _graphite(openmc, "pebble_matrix", 1.60, T)
    shell = _graphite(openmc, "pebble_shell", 1.75, T)
    buffer_ = _graphite(openmc, "triso_buffer", 1.00, T)
    ipyc = _graphite(openmc, "triso_ipyc", 1.90, T)
    opyc = _graphite(openmc, "triso_opyc", 1.90, T)
    sic = openmc.Material(name="triso_sic")
    sic.add_element("Si", 1.0)
    sic.add_element("C", 1.0)
    sic.set_density("g/cm3", 3.20)
    sic.temperature = T
    flibe = _flibe(openmc, "coolant", T)

    triso_u = _triso_universe(openmc, kernel, buffer_, ipyc, sic, opyc)
    peb_u, n_triso = _fuel_pebble_universe(openmc, triso_u, matrix, shell, flibe)

    v_peb = 4.0 / 3.0 * np.pi * PEBBLE_SHELL_R ** 3
    a = (v_peb / BED_PACKING_FRACTION) ** (1.0 / 3.0)      # cubic cell edge
    box = openmc.model.RectangularParallelepiped(
        -a / 2, a / 2, -a / 2, a / 2, -a / 2, a / 2, boundary_type="reflective")
    cell = openmc.Cell(fill=peb_u, region=-box)
    geometry = openmc.Geometry([cell])
    mats = openmc.Materials([kernel, matrix, shell, buffer_, ipyc, opyc, sic, flibe])
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
                   hx) -> tuple:
    """One Natrium fuel assembly (explicit 217-pin lattice, HT9 duct), reflective."""
    from geometry import SQRT3, DUCT_RING_FRAC
    from openmc_models import _hex_assembly_universe

    T = temperature_k
    Rc = (hx.pitch_cm - hx.gap_cm) / SQRT3
    Ri = Rc * (1.0 - DUCT_RING_FRAC)
    pitch = Ri / (HEX_PIN_RINGS_FUEL - 0.5)

    fuel = _u_metal_fuel(openmc, "fuel", enrichment, T)
    clad = _ht9(openmc, "clad", T)
    bond = _sodium(openmc, "bond", T)
    coolant = _sodium(openmc, "coolant", T)
    duct = _ht9(openmc, "duct", T, density=7.0)

    lat = _hex_assembly_universe(openmc, "fuel", fuel, bond, clad, coolant,
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
    return geometry, mats, {"fuel": fuel}, {
        "kind": "single fuel assembly (explicit 217-pin lattice) with reflective duct",
        "pin_pitch_cm": float(pitch), "fuel_volume_cm3": float(v_fuel)}


# --- depletion driver --------------------------------------------------------

def run_depletion(reactor_type: str, burnups_mwd_kg: List[float], chain: str, *,
                  temperature_k: float = 900.0, enrichment: float = 19.75,
                  particles: int = 5000, batches: int = 60, inactive: int = 15,
                  workdir: str = "depletion_run") -> dict:
    """Deplete the representative unit cell and return the composition table.

    burnups_mwd_kg must be ascending and start at 0 (fresh). Returns a dict keyed by
    depletable material name -> list of {nuclide: atom/b-cm}, one entry per burnup.
    """
    import openmc
    import openmc.deplete

    bu = [float(b) for b in burnups_mwd_kg]
    if bu[0] != 0.0 or any(b2 <= b1 for b1, b2 in zip(bu, bu[1:])):
        raise ValueError("burnups must be ascending and start at 0")

    if reactor_type == "fhr":
        geometry, mats, depletable, cell_notes = _fhr_unit_cell(
            openmc, temperature_k, enrichment)
    else:
        geometry, mats, depletable, cell_notes = _hex_unit_cell(
            openmc, temperature_k, enrichment, DEFAULT.hexcore)

    for m in depletable.values():
        m.depletable = True      # volumes were set analytically by the cell builder

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {"method": "interpolation", "default": temperature_k}

    model = openmc.Model(geometry=geometry, materials=mats, settings=settings)

    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        op = openmc.deplete.CoupledOperator(model, chain)
        steps = [b2 - b1 for b1, b2 in zip(bu, bu[1:])]
        integrator = openmc.deplete.PredictorIntegrator(
            op, steps, power_density=SPECIFIC_POWER_W_PER_GHM[reactor_type],
            timestep_units="MWd/kg")
        integrator.integrate()
        results = openmc.deplete.Results("depletion_results.h5")
        avail = _available_nuclides(openmc)

        comps: Dict[str, List[Dict[str, float]]] = {n: [] for n in depletable}
        for step in range(len(bu)):
            step_mats = results.export_to_materials(step)
            by_name = {m.name: m for m in step_mats}
            for name in depletable:
                m = by_name[name]
                dens = m.get_nuclide_atom_densities()
                comps[name].append({
                    nuc: float(d) for nuc, d in dens.items()
                    if float(d) > MIN_ATOM_DENSITY and (not avail or nuc in avail)})
        keff = [float(k.n) for k in results.get_keff()[1]]
    finally:
        os.chdir(cwd)

    return {
        "reactor_type": reactor_type,
        "burnups_mwd_kg": bu,
        "unit_cell": cell_notes,
        "chain": os.path.basename(chain),
        "specific_power_w_per_ghm": SPECIFIC_POWER_W_PER_GHM[reactor_type],
        "temperature_k": temperature_k,
        "enrichment_wt_pct": enrichment,
        "unit_cell_keff": keff,
        "compositions": comps,
    }


def load_composition_table(path: str) -> Optional[dict]:
    """Load a depletion table if present, else None (fresh-fuel-only branch grid)."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def composition_at(table: dict, material: str, burnup_mwd_kg: float
                   ) -> Optional[Dict[str, float]]:
    """Nuclide atom densities [atom/b-cm] at a burnup, linearly interpolated.

    Interpolation is done on atom densities, which is exact for the fresh endpoints
    and is the standard treatment between depletion points. Outside the tabulated
    range the endpoint composition is used (clamped, never extrapolated).
    """
    if table is None:
        return None
    key = material if material in table["compositions"] else None
    if key is None:                      # hex table stores a single "fuel" entry
        key = next(iter(table["compositions"]))
    bu = np.asarray(table["burnups_mwd_kg"], dtype=float)
    comps = table["compositions"][key]
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
    ap.add_argument("--enrichment", type=float, default=19.75)
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
    print(f"Wrote depletion table ({len(args.burnups)} burnup points) -> {args.out}")
    print("  unit-cell k_inf:", ", ".join(f"{k:.5f}" for k in table["unit_cell_keff"]))


if __name__ == "__main__":
    main()
