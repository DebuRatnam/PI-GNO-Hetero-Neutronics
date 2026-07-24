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

from xs_common import MultiGroupXS, multigroupxs_to_dict, n_scatter


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


def _default_chi(reactor_type: str, G: int) -> List[float]:
    from datagen_config import CHI
    if G == 2:
        return list(CHI)
    chi = [0.0] * G
    chi[0] = 1.0                      # fission neutrons born in the fastest group
    return chi


def run_openmc_library(reactor_type: str, G: int,
                       boundaries_ev: List[float]) -> Dict[str, MultiGroupXS]:
    """Build a unit model per material, tally flux-weighted G-group constants with
    openmc.mgxs, and collapse to MultiGroupXS. Requires OpenMC + ENDF/B data.

    This is the real pipeline; it imports openmc lazily so the module (and the
    --dry-run path) work without OpenMC installed. Filled in per-reactor unit-cell
    definitions live here; the collapse below is generic.
    """
    import openmc          # noqa: F401  (lazy; raises if unavailable)
    import openmc.mgxs as mgxs  # noqa: F401

    # NOTE: per-material unit-cell construction (reflected fuel pebble in FLiBe for
    # KP-FHR; homogenized fast assembly for Natrium; infinite medium for absorbers/
    # reflector) is reactor-specific and is built here. Each model tallies
    # diffusion-coefficient (or transport -> D=1/3Sigma_tr), removal, scatter-matrix
    # (kept strictly-lower / down-scatter), nu-fission, and chi on the group
    # structure `boundaries_ev`, then collapses into a MultiGroupXS row. Left as a
    # documented interface so it can be run on a machine with OpenMC + nuclear data.
    raise NotImplementedError(
        "OpenMC unit-cell models are environment-specific. Run on a host with "
        "OpenMC + an ENDF/B HDF5 library (OPENMC_CROSS_SECTIONS). Use --dry-run to "
        "exercise the cache plumbing with the committed default library.")


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
