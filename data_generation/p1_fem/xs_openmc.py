"""OpenMC group-constant generation for the PI-GNO material libraries.

Replaces hand-tuned cross sections with TRACEABLE, flux-weighted multigroup
constants collapsed from ENDF/B continuous-energy data by OpenMC Monte Carlo
transport. This is an OFFLINE pre-step: it writes one cached JSON per reactor which
the material modules load at import. The per-sample data generator never calls
OpenMC.

What makes these constants defensible
-------------------------------------
1. IN-SITU WEIGHTING. Every material is tallied with `domain_type="material"` inside
   the assembled FULL CORE (openmc_models.py), so Sigma_g = int Sigma(E) phi(E) dE /
   int phi(E) dE uses the spectrum that material actually sees. Infinite-medium unit
   cells are not used anywhere.
2. DOUBLE HETEROGENEITY. Fuel is resolved explicitly (TRISO particles inside pebbles;
   pin lattices inside hex assemblies), so resonance self-shielding is geometric
   rather than assumed, and the resolved constituents are then flux-volume
   homogenized back onto the single FEM material the mesh actually carries.
3. BRANCH CASES. The dataset varies burnup, temperature, and control insertion. Each
   is a real transport branch here -- depleted isotopics from xs_depletion.py, true
   Doppler/S(alpha,beta) temperature, and rodded/unrodded cores -- rather than a
   multiplier applied to a single nominal library.
4. TRANSPORT-CORRECTED DIFFUSION. D = 1/(3*Sigma_tr) from a tallied transport cross
   section, not a raw flux-weighted 1/(3*Sigma_t).
5. UNCERTAINTIES + PROVENANCE. Every constant carries its Monte Carlo relative
   standard deviation, and the library records the OpenMC version, data library,
   particle counts, group structure, and per-branch k_eff.

Usage
-----
    # optional but recommended: isotopics vs burnup (see xs_depletion.py)
    python xs_depletion.py --reactor fhr --chain chain_endfb80_pwr.xml \\
        --out depletion_fhr.json

    python xs_openmc.py --reactor fhr --out xs_fhr.json \\
        --depletion depletion_fhr.json \\
        --burnups 0 40 80 --temperatures 900 1200 --rods out in

    python xs_openmc.py --reactor hex --out xs_natrium.json \\
        --depletion depletion_natrium.json \\
        --burnups 0 40 80 --temperatures 800 1000 --rods out in

Without OpenMC installed the cache plumbing is still exercisable:

    python xs_openmc.py --reactor fhr --out xs_fhr.json --dry-run

which serializes the CURRENT committed hand library as a single-branch table, so the
load/interpolate path is reproducible and testable. Real runs overwrite it.

Output schema (v2, self-describing) -- see xs_branch.py for the reader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from xs_common import (MultiGroupXS, multigroupxs_to_dict, n_scatter,
                       scatter_pairs, up_scatter_pairs)

SCHEMA_VERSION = 2

# Default two-group energy boundaries (eV), ascending [low, ..., high].
#   fhr : thermal cutoff 0.625 eV (fast | thermal).
#   hex : fast split ~0.1 MeV (high-fast | slow-fast); both groups fast.
DEFAULT_BOUNDARIES_EV = {
    "fhr": [1.0e-5, 0.625, 2.0e7],
    "hex": [1.0e-5, 1.0e5, 2.0e7],
}

# mgxs scores requested per material domain. "nu-scatter matrix" carries (n,2n)
# multiplication, which is what the diffusion balance in operators.py expects.
MGXS_TYPES = ["transport", "absorption", "nu-fission", "nu-scatter matrix", "fission"]


def energy_boundaries_ev(reactor_type: str, G: int) -> List[float]:
    """Ascending group boundaries (length G+1). Uses the documented two-group
    cutoffs for G=2; log-spaced fill for finer group structures."""
    if G == 2:
        return list(DEFAULT_BOUNDARIES_EV[reactor_type])
    lo, hi = 1.0e-5, 2.0e7
    lg = [math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / G for i in range(G + 1)]
    return [10.0 ** v for v in lg]


def _mat_module(reactor_type: str):
    import materials_fhr
    import materials
    return materials_fhr if reactor_type == "fhr" else materials


def _default_library(reactor_type: str) -> Dict[str, MultiGroupXS]:
    """The committed hand-tuned library for a reactor (used by --dry-run)."""
    return dict(_mat_module(reactor_type).LIBRARY)


def _default_chi(reactor_type: str, G: int) -> List[float]:
    mod = _mat_module(reactor_type)
    chi = getattr(mod, "CHI", None)
    if chi is not None and len(chi) == G:
        return list(chi)
    out = [0.0] * G
    out[0] = 1.0                      # fission neutrons born in the fastest group
    return out


# --- raw per-domain tally results -------------------------------------------

class DomainXS:
    """Raw group constants for ONE transport material, before homogenization.

    Arrays are ordered with index 0 = HIGHEST energy, matching OpenMC's mgxs group
    ordering and the PI-GNO convention (g0 = fast).
    """

    __slots__ = ("flux", "Sigma_tr", "Sigma_a", "nuSf", "Sf", "Smat", "chi",
                 "std", "volume")

    def __init__(self, flux, Sigma_tr, Sigma_a, nuSf, Sf, Smat, chi, std, volume=1.0):
        self.flux = np.asarray(flux, float)          # [G] volume-integrated flux
        self.Sigma_tr = np.asarray(Sigma_tr, float)  # [G]
        self.Sigma_a = np.asarray(Sigma_a, float)    # [G]
        self.nuSf = np.asarray(nuSf, float)          # [G]
        self.Sf = np.asarray(Sf, float)              # [G]
        self.Smat = np.asarray(Smat, float)          # [G, G] (g_from, g_to)
        self.chi = None if chi is None else np.asarray(chi, float)
        self.std: Dict[str, np.ndarray] = std        # relative std dev per score
        self.volume = float(volume)


def _rel(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Relative standard deviation, 0 where the mean is 0."""
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float)
    out = np.zeros_like(mean)
    nz = np.abs(mean) > 0.0
    out[nz] = np.abs(sd[nz] / mean[nz])
    return out


def _collapse_domain(mgxs_lib, material, G: int, flux: np.ndarray) -> DomainXS:
    """Pull one material's tallied constants out of an openmc.mgxs.Library."""
    def get(kind, value="mean"):
        x = mgxs_lib.get_mgxs(material, kind)
        return np.asarray(x.get_xs(value=value), dtype=float)

    Sigma_tr = get("transport").reshape(-1)[:G]
    Sigma_a = get("absorption").reshape(-1)[:G]
    nuSf = get("nu-fission").reshape(-1)[:G]
    Sf = get("fission").reshape(-1)[:G]
    Smat = get("nu-scatter matrix").reshape(G, G)          # [g_from, g_to]

    std = {
        "Sigma_tr": _rel(Sigma_tr, get("transport", "std_dev").reshape(-1)[:G]),
        "Sigma_a": _rel(Sigma_a, get("absorption", "std_dev").reshape(-1)[:G]),
        "nuSf": _rel(nuSf, get("nu-fission", "std_dev").reshape(-1)[:G]),
        "scatter": _rel(Smat, get("nu-scatter matrix", "std_dev").reshape(G, G)),
    }

    chi = None
    if nuSf.sum() > 0.0:
        try:
            chi = get("chi").reshape(-1)[:G]
            std["chi"] = _rel(chi, get("chi", "std_dev").reshape(-1)[:G])
        except Exception:
            chi = None
    return DomainXS(flux, Sigma_tr, Sigma_a, nuSf, Sf, Smat, chi, std)


def _homogenize(parts: Dict[str, Tuple[DomainXS, float]], G: int
                ) -> Tuple[MultiGroupXS, np.ndarray, Dict[str, List[float]]]:
    """Flux-volume homogenize resolved constituents onto one FEM material.

    Weight w_{i,g} = V_i * phi_{i,g}. Reaction rates add, so
        Sigma_hom,g = sum_i w_{i,g} Sigma_{i,g} / sum_i w_{i,g}
    and the scatter matrix is weighted by the INCOMING group flux. The diffusion
    coefficient is homogenized through the transport cross section
    (Sigma_tr adds; D = 1/(3 Sigma_tr)) -- averaging D directly is wrong.

    Returns (MultiGroupXS, up-scatter block, relative-std-dev dict).
    """
    names = list(parts)
    W = np.stack([parts[n][0].flux * parts[n][1] for n in names])      # [P, G]
    tot = W.sum(axis=0)
    tot[tot <= 0.0] = 1.0

    def avg(attr):
        return sum(W[i] * getattr(parts[n][0], attr) for i, n in enumerate(names)) / tot

    Sigma_tr = avg("Sigma_tr")
    Sigma_a = avg("Sigma_a")
    nuSf = avg("nuSf")
    Smat = sum(W[i][:, None] * parts[n][0].Smat
               for i, n in enumerate(names)) / tot[:, None]

    Sigma_tr = np.where(Sigma_tr > 0.0, Sigma_tr, 1.0e-10)
    D = 1.0 / (3.0 * Sigma_tr)

    # removal = absorption + ALL out-scatter (down AND up). Up-scatter is then added
    # back as an explicit in-scatter source at operator level for thermal cores.
    out_scatter = Smat.sum(axis=1) - np.diag(Smat)
    Sr = Sigma_a + out_scatter

    down = tuple(float(Smat[gf, gt]) for gf, gt in scatter_pairs(G))
    up = np.array([Smat[gf, gt] for gf, gt in up_scatter_pairs(G)], dtype=float)

    # propagate the dominant statistical uncertainties (weighted the same way)
    def avg_std(key):
        return sum(W[i] * parts[n][0].std[key] for i, n in enumerate(names)) / tot

    rel_std = {
        "D": [float(v) for v in avg_std("Sigma_tr")],
        "Sr": [float(v) for v in avg_std("Sigma_a")],
        "nuSf": [float(v) for v in avg_std("nuSf")],
    }
    xs = MultiGroupXS(D=tuple(float(v) for v in D),
                      Sr=tuple(float(v) for v in Sr),
                      scatter=down,
                      nuSf=tuple(float(v) for v in nuSf))
    return xs, up, rel_std


def _core_chi(domains: Dict[str, DomainXS], G: int) -> Tuple[List[float], List[float]]:
    """Core-average fission spectrum, weighted by fission-neutron production.

    chi is fixed nuclear data carried in F (never in node features), so one
    core-average spectrum per branch is what the operator assembly needs.
    """
    num = np.zeros(G)
    den = 0.0
    sd = np.zeros(G)
    for d in domains.values():
        if d.chi is None:
            continue
        prod = float((d.nuSf * d.flux).sum())
        if prod <= 0.0:
            continue
        num += prod * d.chi
        sd += prod * d.std.get("chi", np.zeros(G))
        den += prod
    if den <= 0.0:
        return [], []
    chi = num / den
    s = chi.sum()
    if s > 0:
        chi = chi / s
    return [float(v) for v in chi], [float(v) for v in (sd / den)]


# --- one branch --------------------------------------------------------------

def run_branch(reactor_type: str, cfg, state, G: int, boundaries_ev: List[float], *,
               particles: int, batches: int, inactive: int,
               depletion_table: Optional[dict], workdir: str,
               model_kwargs: Optional[dict] = None) -> dict:
    """Build, run, and collapse ONE branch case. Returns a branch dict for the table."""
    import openmc
    import openmc.mgxs as mgxs
    import openmc_models
    from xs_depletion import composition_at

    fc = None
    if depletion_table is not None:
        if reactor_type == "fhr":
            fc = composition_at(depletion_table, "fuel_pebble", state.burnup_mwd_kg)
        else:
            fc = {name: composition_at(depletion_table, name, state.burnup_mwd_kg)
                  for name in ("fuel_inner", "fuel_outer")}

    kw = dict(model_kwargs or {})
    kw["fuel_composition"] = fc
    openmc_exec = openmc_models.prepare_openmc_env()

    core = openmc_models.build(reactor_type, state, cfg, **kw)
    model = core.model

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {"method": "interpolation",
                            "default": state.temperature_k,
                            "range": (250.0, 2500.0)}
    settings.output = {"tallies": False}
    model.settings = settings

    groups = mgxs.EnergyGroups(np.asarray(boundaries_ev, dtype=float))
    lib = mgxs.Library(model.geometry)
    lib.energy_groups = groups
    lib.mgxs_types = MGXS_TYPES + ["chi"]
    lib.domain_type = "material"
    lib.domains = list(core.tally_domains.values())
    lib.build_library()

    tallies = openmc.Tallies()
    lib.add_to_tallies_file(tallies, merge=True)

    # explicit per-material group flux: the homogenization weights. EnergyFilter bins
    # are ASCENDING in energy, so they are reversed below to the g0=fast convention.
    mat_list = list(core.tally_domains.values())
    flux_tally = openmc.Tally(name="flux_by_material")
    flux_tally.filters = [openmc.MaterialFilter(mat_list),
                          openmc.EnergyFilter(np.asarray(boundaries_ev, dtype=float))]
    flux_tally.scores = ["flux"]
    tallies.append(flux_tally)
    model.tallies = tallies

    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        sp_path = model.run(openmc_exec=openmc_exec)
        with openmc.StatePoint(sp_path) as sp:
            lib.load_from_statepoint(sp)
            k = sp.keff
            ft = sp.get_tally(name="flux_by_material")
            flux = ft.mean.reshape(len(mat_list), G)[:, ::-1]   # -> g0 = fast
    finally:
        os.chdir(cwd)

    raw: Dict[str, DomainXS] = {}
    for i, (tag, mat) in enumerate(core.tally_domains.items()):
        d = _collapse_domain(lib, mat, G, flux[i])
        raw[tag] = d

    chi, chi_std = _core_chi(raw, G)

    materials_out: Dict[str, dict] = {}
    for fem_name, vols in core.homogenize.items():
        parts = {tag: (raw[tag], vol) for tag, vol in vols.items() if tag in raw}
        if not parts:
            continue
        xs, up, rel_std = _homogenize(parts, G)
        rec = multigroupxs_to_dict(xs)
        rec["upscatter"] = [float(v) for v in up]
        rec["rel_std"] = rel_std
        rec["constituents"] = {t: float(v) for t, v in vols.items()}
        materials_out[fem_name] = rec

    return {
        "burnup_mwd_kg": float(state.burnup_mwd_kg),
        "temperature_k": float(state.temperature_k),
        "rod": state.rod,
        "k_eff": float(k.n),
        "k_eff_std": float(k.s),
        "chi": chi,
        "chi_rel_std": chi_std,
        "materials": materials_out,
        "model_notes": core.notes,
    }


# --- dry run -----------------------------------------------------------------

def _dry_run_branches(reactor_type: str, G: int) -> List[dict]:
    """Rod-out / rod-in branches built from the committed hand library (no OpenMC).

    Lets the branch-table reader, the interpolator, and the generator be exercised
    end to end without a transport run, with the SAME rod-branch structure a real run
    produces -- so gray-rod insertion still interpolates between follower and
    absorber. `source` in the header marks the data as non-transport-derived.
    """
    mod = _mat_module(reactor_type)
    lib = _default_library(reactor_type)
    up_tab = getattr(mod, "UPSCATTER_21", {})
    zero_std = {"D": [0.0] * G, "Sr": [0.0] * G, "nuSf": [0.0] * G}

    def rec_for(name: str, xs) -> dict:
        rec = multigroupxs_to_dict(xs)
        up = [0.0] * n_scatter(G)
        if G == 2 and name in up_tab:
            up[0] = float(up_tab[name])
        rec["upscatter"] = up
        rec["rel_std"] = dict(zero_std)
        rec["constituents"] = {name: 1.0}
        return rec

    control = {"primary_control", "secondary_control",
               "control_element", "shutdown_element"}
    follower = getattr(mod, "FLIBE_FOLLOWER", None) or getattr(mod, "CONTROL_FOLLOWER")

    rod_in = {name: rec_for(name, xs) for name, xs in lib.items()}
    rod_out = {name: rec_for(name, xs) for name, xs in lib.items()
               if name not in control}
    rod_out["control_follower"] = rec_for("control_follower", follower)

    def branch(rod: str, materials: Dict[str, dict]) -> dict:
        return {
            "burnup_mwd_kg": 0.0, "temperature_k": 900.0, "rod": rod,
            # no transport run happened, so there is no k_eff to report
            "k_eff": None, "k_eff_std": None,
            "chi": _default_chi(reactor_type, G), "chi_rel_std": [0.0] * G,
            "materials": materials,
            "model_notes": {"source": "committed hand library"},
        }

    return [branch("out", rod_out), branch("in", rod_in)]


# --- table assembly ----------------------------------------------------------

def _provenance(reactor_type: str, particles: int, batches: int, inactive: int,
                depletion_table: Optional[dict], dry_run: bool) -> dict:
    prov = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "particles_per_batch": particles,
        "batches": batches,
        "inactive_batches": inactive,
        "weighting": ("committed hand library (NOT transport-derived)" if dry_run else
                      "in-situ full-core material-domain flux"),
        "D_definition": "1/(3*Sigma_tr), tallied transport cross section",
        "removal_definition": "Sigma_a + total out-scatter (down and up)",
        "scatter_score": "nu-scatter matrix (includes (n,2n) multiplication)",
        "axial_treatment": ("axially reflective slab; axial leakage carried "
                            "separately as Bz^2 in operators.assemble_AF"),
        "homogenization": ("flux-volume weighted over transport-resolved "
                           "constituents; Sigma_tr homogenized, then D = 1/(3 Sigma_tr)"),
    }
    if depletion_table is not None:
        prov["depletion"] = {
            "chain": depletion_table.get("chain"),
            "unit_cell": depletion_table.get("unit_cell"),
            "burnups_mwd_kg": depletion_table.get("burnups_mwd_kg"),
            "specific_power_w_per_ghm": depletion_table.get("specific_power_w_per_ghm"),
            "note": ("isotopics from a representative unit cell; the collapse "
                     "spectrum is still the full-core one"),
        }
    if not dry_run:
        try:
            import openmc
            prov["openmc_version"] = str(openmc.__version__)
        except Exception:
            pass
        xs_path = os.environ.get("OPENMC_CROSS_SECTIONS", "")
        prov["cross_sections_xml"] = xs_path
        prov["data_library"] = os.path.basename(os.path.dirname(xs_path)) or "unknown"
    return prov


def write_table(path: str, reactor_type: str, G: int, boundaries_ev: List[float],
                branches: List[dict], provenance: dict, source: str) -> None:
    for b in branches:
        for name, rec in b["materials"].items():
            if len(rec["D"]) != G or len(rec["scatter"]) != n_scatter(G):
                raise ValueError(f"{name}: group-structure mismatch for G={G}")
    axes = {
        "burnup_mwd_kg": sorted({b["burnup_mwd_kg"] for b in branches}),
        "temperature_k": sorted({b["temperature_k"] for b in branches}),
        "rod": sorted({b["rod"] for b in branches}),
    }
    blob = {
        "schema_version": SCHEMA_VERSION,
        "reactor_type": reactor_type,
        "n_groups": G,
        "group_boundaries_ev": boundaries_ev,
        "source": source,
        "axes": axes,
        "provenance": provenance,
        "branches": branches,
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reactor", choices=["hex", "fhr"], required=True)
    ap.add_argument("--out", required=True,
                    help="branch-table JSON (xs_natrium.json / xs_fhr.json)")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--burnups", type=float, nargs="+", default=[0.0],
                    help="burnup branch points [MWd/kgHM]")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[900.0],
                    help="temperature branch points [K]")
    ap.add_argument("--rods", nargs="+", default=["out", "in"], choices=["out", "in"])
    ap.add_argument("--depletion", default=None,
                    help="depletion table JSON from xs_depletion.py")
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--inactive", type=int, default=40)
    ap.add_argument("--axial-cm", type=float, default=None,
                    help="axial slab height of the reflective transport model")
    ap.add_argument("--workdir", default="openmc_run")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the committed hand library as a single branch (no OpenMC)")
    args = ap.parse_args()

    G = args.groups
    boundaries = energy_boundaries_ev(args.reactor, G)

    if args.dry_run:
        branches = _dry_run_branches(args.reactor, G)
        prov = _provenance(args.reactor, 0, 0, 0, None, True)
        write_table(args.out, args.reactor, G, boundaries, branches, prov,
                    "default-dry-run")
        print(f"Wrote {len(branches)} dry-run branches (G={G}) -> {args.out}")
        return

    from datagen_config import DEFAULT
    from openmc_models import BranchState
    from xs_depletion import load_composition_table

    cfg = DEFAULT.pebblecore if args.reactor == "fhr" else DEFAULT.hexcore
    dep = load_composition_table(args.depletion) if args.depletion else None
    if dep is None and max(args.burnups) > 0.0:
        raise SystemExit(
            "non-zero burnup branches requested but no --depletion table given; "
            "run xs_depletion.py first (burnt isotopics cannot be guessed)")
    model_kwargs = {} if args.axial_cm is None else {"axial_cm": args.axial_cm}

    branches: List[dict] = []
    total = len(args.burnups) * len(args.temperatures) * len(args.rods)
    n = 0
    for bu in args.burnups:
        for T in args.temperatures:
            for rod in args.rods:
                n += 1
                state = BranchState(burnup_mwd_kg=float(bu), temperature_k=float(T),
                                    rod=rod)
                print(f"[{n}/{total}] branch {state.key}", flush=True)
                branches.append(run_branch(
                    args.reactor, cfg, state, G, boundaries,
                    particles=args.particles, batches=args.batches,
                    inactive=args.inactive, depletion_table=dep,
                    workdir=os.path.join(args.workdir, state.key),
                    model_kwargs=model_kwargs))
                print(f"    k_eff = {branches[-1]['k_eff']:.5f} "
                      f"+/- {branches[-1]['k_eff_std']:.5f}", flush=True)

    prov = _provenance(args.reactor, args.particles, args.batches, args.inactive,
                       dep, False)
    write_table(args.out, args.reactor, G, boundaries, branches, prov, "openmc")
    print(f"Wrote {len(branches)} branches (G={G}) -> {args.out}")


if __name__ == "__main__":
    main()
