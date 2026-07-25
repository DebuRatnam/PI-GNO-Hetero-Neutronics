"""Offline OpenMC group-constant generation for the PI-GNO material libraries.

Replaces the hand-tuned cross sections in materials.py / materials_fhr.py with
TRACEABLE, flux-weighted multigroup constants derived from ENDF/B via OpenMC Monte
Carlo transport (the standard group-collapse equation Sigma_g = int Sigma(E) phi(E)
dE / int phi(E) dE). This is an OFFLINE pre-step: it writes a cached JSON per
reactor which the material modules load at import; the per-sample generator never
calls OpenMC.

Usage (requires OpenMC + an ENDF/B HDF5 library via OPENMC_CROSS_SECTIONS):

    python xs_openmc.py --reactor fhr --out xs_fhr.json
    python xs_openmc.py --reactor hex --out xs_natrium.json

Without OpenMC installed you can still exercise the cache plumbing:

    python xs_openmc.py --reactor fhr --out xs_fhr.json --dry-run

which serializes the CURRENT (committed default) library to the cache JSON, so the
load path is reproducible and testable. Real runs overwrite it with MC-tallied
constants. The JSON schema is self-describing:

    { "reactor_type", "n_groups", "group_boundaries_ev" (length G+1, ascending),
      "chi" (length G), "source" ("openmc" | "default-dry-run"),
      "materials": { name: { "D":[G], "Sr":[G], "scatter":[G(G-1)/2], "nuSf":[G] } } }
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from xs_common import MultiGroupXS, multigroupxs_to_dict, n_scatter, scatter_pairs


# Default two-group energy boundaries (eV), ascending [low, cut, high].
#   fhr : thermal cutoff 0.625 eV (fast | thermal).
#   hex : fast split ~0.1 MeV (high-fast | slow-fast); both groups fast.
DEFAULT_BOUNDARIES_EV = {
    "fhr": [1.0e-5, 0.625, 2.0e7],
    "hex": [1.0e-5, 1.0e5, 2.0e7],
}


def energy_boundaries_ev(reactor_type: str, G: int) -> List[float]:
    """Ascending group boundaries (length G+1). Uses the documented two-group
    cutoffs for G=2; log-spaced fill for finer group structures."""
    if G == 2:
        return DEFAULT_BOUNDARIES_EV[reactor_type]
    import math
    lo, hi = 1.0e-5, 2.0e7
    lg = [math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / G for i in range(G + 1)]
    return [10.0 ** v for v in lg]


def _default_library(reactor_type: str) -> Dict[str, MultiGroupXS]:
    """The committed hand-tuned library for a reactor (used by --dry-run)."""
    if reactor_type == "fhr":
        import materials_fhr as m
    else:
        import materials as m
    return dict(m.LIBRARY)


def _mat_module(reactor_type: str):
    import materials_fhr
    import materials
    return materials_fhr if reactor_type == "fhr" else materials


def _default_chi(reactor_type: str, G: int) -> List[float]:
    # per-reactor fission spectrum from the material module (fast core vs thermal core)
    mod = _mat_module(reactor_type)
    chi = getattr(mod, "CHI", None)
    if chi is not None and len(chi) == G:
        return list(chi)
    out = [0.0] * G
    out[0] = 1.0                      # fission neutrons born in the fastest group
    return out


# Fissile materials per reactor (run in eigenvalue mode; others get a driving source).
_FISSILE = {"hex": {"fuel_inner", "fuel_outer"}, "fhr": {"fuel_pebble"}}


def _build_material(name: str, reactor_type: str):
    """Representative openmc.Material for one library material.

    DOCUMENTED STARTING POINTS at public/literature level (HALEU U-10Zr metal fuel,
    graphite, B4C, FLiBe, sodium, HT9/steel) -- NOT a validated core specification.
    Review compositions/densities against your reactor spec before treating the
    tallied constants as benchmark-grade.
    """
    import openmc
    m = openmc.Material(name=name)
    if reactor_type == "fhr":
        if name == "fuel_pebble":                    # TRISO UO2 (HALEU) homogenized in graphite
            m.add_element("U", 1.0, enrichment=19.75)
            m.add_element("O", 2.0)
            m.add_element("C", 30.0)                  # matrix + coatings (homogenized pebble)
            m.set_density("g/cm3", 2.0)
            m.add_s_alpha_beta("c_Graphite")
        elif name in ("graphite_pebble", "reflector"):
            m.add_element("C", 1.0)
            m.set_density("g/cm3", 1.7)
            m.add_s_alpha_beta("c_Graphite")
        elif name in ("control_element", "shutdown_element"):
            m.add_element("B", 4.0); m.add_element("C", 1.0)   # B4C (natural boron)
            m.set_density("g/cm3", 2.52)
        elif name == "coolant":                      # FLiBe (Li2BeF4), Li-7 enriched
            m.add_nuclide("Li7", 2.0 * 0.99995); m.add_nuclide("Li6", 2.0 * 0.00005)
            m.add_element("Be", 1.0); m.add_element("F", 4.0)
            m.set_density("g/cm3", 1.94)
        elif name == "vessel":                       # Hastelloy-N-like Ni-Mo-Cr alloy
            m.add_element("Ni", 0.71); m.add_element("Mo", 0.16)
            m.add_element("Cr", 0.07); m.add_element("Fe", 0.06)
            m.set_density("g/cm3", 8.86)
        else:
            raise ValueError(f"unknown fhr material {name}")
    else:                                            # hex / Natrium fast
        if name in ("fuel_inner", "fuel_outer"):     # U-10Zr metal, HALEU (inner hotter)
            enr = 19.75 if name == "fuel_inner" else 14.0
            m.add_element("U", 0.9, enrichment=enr); m.add_element("Zr", 0.1)
            m.set_density("g/cm3", 15.5)
        elif name in ("primary_control", "secondary_control", "shield"):
            m.add_element("B", 4.0); m.add_element("C", 1.0)   # B4C
            m.set_density("g/cm3", 2.52)
        elif name == "reflector":                    # stainless reflector
            m.add_element("Fe", 0.70); m.add_element("Cr", 0.18); m.add_element("Ni", 0.12)
            m.set_density("g/cm3", 7.9)
        elif name == "duct":                         # HT9 (Fe-12Cr) + sodium gap, homogenized
            m.add_element("Fe", 0.85); m.add_element("Cr", 0.12); m.add_element("Na", 0.03)
            m.set_density("g/cm3", 7.0)
        elif name == "coolant":                      # sodium
            m.add_element("Na", 1.0); m.set_density("g/cm3", 0.85)
        else:
            raise ValueError(f"unknown hex material {name}")
    return m


def _collapse(mg_lib, material, G: int) -> MultiGroupXS:
    """Collapse an openmc.mgxs.Library result for one material to a MultiGroupXS row.

    Sr_g = absorption_g + total out-scatter_g ; the stored scatter block keeps only the
    DOWN-scatter pairs (g_from < g_to), matching xs_common. Up-scatter is folded into
    Sr (removal) here; the FHR generator carries it explicitly at operator level too.
    OpenMC orders groups by DECREASING energy (index 0 = fastest), matching g0=fast.
    """
    def xs(kind):
        return np.asarray(mg_lib.get_mgxs(material, kind).get_xs(), dtype=float).reshape(-1)
    D = xs("diffusion-coefficient")[:G]
    Sa = xs("absorption")[:G]
    nuSf = xs("nu-fission")[:G]
    Smat = np.asarray(mg_lib.get_mgxs(material, "nu-scatter matrix").get_xs(),
                      dtype=float).reshape(G, G)               # [g_from, g_to]
    out_scatter = Smat.sum(axis=1) - np.diag(Smat)             # total scatter out of g
    Sr = Sa + out_scatter
    down = [Smat[gf, gt] for gf, gt in scatter_pairs(G)]       # gf < gt (down-scatter)
    return MultiGroupXS(D=tuple(D), Sr=tuple(Sr), scatter=tuple(down), nuSf=tuple(nuSf))


def run_openmc_library(reactor_type: str, G: int,
                       boundaries_ev: List[float]) -> Dict[str, MultiGroupXS]:
    """Build a unit model per material, tally flux-weighted G-group constants with
    openmc.mgxs, and collapse to MultiGroupXS. Requires OpenMC + ENDF/B data.

    This is the real pipeline; it imports openmc lazily so the module (and the
    --dry-run path) work without OpenMC installed. Filled in per-reactor unit-cell
    definitions live here; the collapse below is generic.
    """
    import openmc
    import openmc.mgxs as mgxs

    groups = mgxs.EnergyGroups(np.asarray(boundaries_ev, dtype=float))
    fissile = _FISSILE[reactor_type]
    out: Dict[str, MultiGroupXS] = {}

    for name in _default_library(reactor_type):          # material names for this reactor
        mat = _build_material(name, reactor_type)

        # infinite medium: one cell of the material in a reflective box (the collapse
        # spectrum is the material's own infinite-medium spectrum; a fission source
        # drives non-fissile media). Full-core flux-weighting is higher fidelity --
        # swap this unit cell for the assembled core geometry if you have it.
        box = openmc.model.RectangularParallelepiped(
            -10, 10, -10, 10, -10, 10, boundary_type="reflective")
        cell = openmc.Cell(fill=mat, region=-box)
        geometry = openmc.Geometry([cell])

        settings = openmc.Settings()
        settings.particles = 20000
        settings.batches = 150
        settings.inactive = 30
        if name in fissile:
            settings.run_mode = "eigenvalue"
        else:
            settings.run_mode = "fixed source"
            src = openmc.IndependentSource()
            src.space = openmc.stats.Point((0.0, 0.0, 0.0))
            src.energy = openmc.stats.Watt()            # representative fission driving source
            settings.source = src

        mg_lib = mgxs.Library(geometry)
        mg_lib.energy_groups = groups
        mg_lib.mgxs_types = ["diffusion-coefficient", "absorption",
                             "nu-scatter matrix", "nu-fission"]
        mg_lib.domain_type = "material"
        mg_lib.domains = [mat]
        mg_lib.build_library()

        tallies = openmc.Tallies()
        mg_lib.add_to_tallies_file(tallies, merge=True)

        model = openmc.Model(geometry=geometry, settings=settings,
                             tallies=tallies, materials=openmc.Materials([mat]))
        sp_path = model.run()
        with openmc.StatePoint(sp_path) as sp:
            mg_lib.load_from_statepoint(sp)

        out[name] = _collapse(mg_lib, mat, G)
    return out


def write_library_json(path: str, reactor_type: str, G: int,
                       library: Dict[str, MultiGroupXS], chi: List[float],
                       boundaries_ev: List[float], source: str) -> None:
    for name, xs in library.items():
        if xs.n_groups != G:
            raise ValueError(f"{name}: {xs.n_groups} groups != requested G={G}")
        if len(xs.scatter) != n_scatter(G):
            raise ValueError(f"{name}: scatter width mismatch for G={G}")
    blob = {
        "reactor_type": reactor_type,
        "n_groups": G,
        "group_boundaries_ev": boundaries_ev,
        "chi": chi,
        "source": source,
        "materials": {name: multigroupxs_to_dict(xs) for name, xs in library.items()},
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactor", choices=["hex", "fhr"], required=True)
    ap.add_argument("--out", required=True, help="cache JSON path (xs_natrium.json / xs_fhr.json)")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="write the committed default library (no OpenMC) to test the cache path")
    args = ap.parse_args()

    G = args.groups
    boundaries = energy_boundaries_ev(args.reactor, G)
    chi = _default_chi(args.reactor, G)

    if args.dry_run:
        library = _default_library(args.reactor)
        source = "default-dry-run"
    else:
        library = run_openmc_library(args.reactor, G, boundaries)
        source = "openmc"

    write_library_json(args.out, args.reactor, G, library, chi, boundaries, source)
    print(f"Wrote {len(library)} materials ({source}, G={G}) -> {args.out}")


if __name__ == "__main__":
    main()
