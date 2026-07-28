"""Verification: P1 diffusion vs OpenMC continuous-energy transport.

Generating group constants with OpenMC makes the labels traceable. It does NOT by
itself show that the two-group diffusion model reproduces transport -- and that is
the claim the dataset rests on, because every reference k_eff, flux, and power
density in the dataset comes from the diffusion solve, not from transport.

This harness closes that gap. For a set of core states it runs BOTH:

  * the P1-FEM two-group diffusion solve used to label the dataset, and
  * a full-core continuous-energy OpenMC eigenvalue calculation of the SAME state,

and reports, per state:

    dk        = k_diff - k_ce
    d_rho     = (1/k_ce - 1/k_diff) * 1e5     [pcm]   -- the standard reactivity bias
    rod worth = rho(rod in) - rho(rod out)             for the paired states
    radial power shape: RMS and max relative deviation between the diffusion power
    density and the OpenMC fission-rate distribution, binned in common radial rings

Each state is run at a UNIFORM burnup and temperature so the transport model can
represent exactly the same core the FEM model sees (the dataset's per-node burnup
scatter has no continuous-energy counterpart). Control insertion is held at the
rod-out / rod-in endpoints for the same reason.

    python validate_openmc.py --reactor fhr --out validation_fhr.csv \\
        --depletion depletion_fhr.json --burnups 0 60 --temperatures 900

Output is a CSV plus a printed summary; both are paper-ready.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from datagen_config import DEFAULT, DataGenConfig


N_RADIAL_RINGS = 12


def _uniform_config(cfg: DataGenConfig, burnup: float, temperature: float
                    ) -> DataGenConfig:
    """Collapse the per-node burnup/temperature draw to a single point.

    The generator samples burnup and temperature uniformly over a range; setting the
    range to a degenerate interval makes every node identical, which is the only way
    a single transport model can describe the same core.
    """
    if cfg.reactor_type == "fhr":
        return replace(cfg, pebblecore=replace(
            cfg.pebblecore, burnup_mwd_kg_range=(burnup, burnup),
            temperature_k_range=(temperature, temperature)))
    return replace(cfg, hexcore=replace(
        cfg.hexcore, burnup_mwd_kg_range=(burnup, burnup),
        temperature_k_range=(temperature, temperature)))


def _diffusion_state(cfg: DataGenConfig, rod: str, seed: int) -> dict:
    """One labelled sample at a rod endpoint (the dataset's own pipeline)."""
    from dataset import make_sample
    frac = 1.0 if rod == "in" else 0.0
    rng = np.random.default_rng(seed)
    if cfg.reactor_type == "fhr":
        return make_sample(cfg, rng=rng, insert_control=frac, insert_shutdown=frac)
    return make_sample(cfg, rng=rng, insert_fraction=frac)


def _radial_power_diffusion(sample: dict, edges: np.ndarray) -> np.ndarray:
    """Diffusion power integrated into radial rings (volume weighted)."""
    xy = sample["coordinates"]
    r = np.hypot(xy[:, 0], xy[:, 1])
    p = np.asarray(sample["power_density"], float) * np.asarray(sample["nodal_volume"], float)
    idx = np.clip(np.digitize(r, edges) - 1, 0, len(edges) - 2)
    out = np.zeros(len(edges) - 1)
    np.add.at(out, idx, p)
    return out


def _run_transport(reactor_type: str, core_cfg, state, edges: np.ndarray, *,
                   particles: int, batches: int, inactive: int,
                   depletion_table: Optional[dict], workdir: str,
                   axial_cm: Optional[float]) -> Tuple[float, float, np.ndarray]:
    """Full-core CE eigenvalue run. Returns (k, k_std, radial fission rate)."""
    import openmc
    import openmc_models
    from xs_depletion import composition_at

    openmc_exec = openmc_models.prepare_openmc_env()

    fc = None
    if depletion_table is not None:
        if reactor_type == "fhr":
            fc = composition_at(depletion_table, "fuel_pebble", state.burnup_mwd_kg)
        else:
            fc = {n: composition_at(depletion_table, n, state.burnup_mwd_kg)
                  for n in ("fuel_inner", "fuel_outer")}

    kw = {"fuel_composition": fc}
    if axial_cm is not None:
        kw["axial_cm"] = axial_cm
    core = openmc_models.build(reactor_type, state, core_cfg, **kw)
    model = core.model

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {"method": "interpolation",
                            "default": state.temperature_k,
                            "range": (250.0, 2500.0)}
    model.settings = settings

    zmax = float(core.notes.get("axial_cm", 40.0)) / 2.0
    mesh = openmc.CylindricalMesh(r_grid=edges, z_grid=np.array([-zmax, zmax]),
                                  phi_grid=np.array([0.0, 2.0 * np.pi]))
    t = openmc.Tally(name="radial_fission")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["kappa-fission"]
    model.tallies = openmc.Tallies([t])

    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        sp_path = model.run(openmc_exec=openmc_exec)
        with openmc.StatePoint(sp_path) as sp:
            k = sp.keff
            radial = sp.get_tally(name="radial_fission").mean.reshape(-1).copy()
    finally:
        os.chdir(cwd)
    return float(k.n), float(k.s), radial


def _shape_error(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(RMS, max) relative deviation of two radial distributions, each normalized
    to unit total so only the SHAPE is compared."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan"), float("nan")
    a, b = a / sa, b / sb
    keep = b > 1e-6                      # ignore rings with no fission (reflector)
    if not keep.any():
        return float("nan"), float("nan")
    rel = (a[keep] - b[keep]) / b[keep]
    return float(np.sqrt(np.mean(rel ** 2))), float(np.max(np.abs(rel)))


def run(reactor_type: str, burnups: List[float], temperatures: List[float],
        rods: List[str], *, particles: int, batches: int, inactive: int,
        depletion: Optional[str], workdir: str, seed: int,
        axial_cm: Optional[float]) -> List[dict]:
    from openmc_models import BranchState
    from xs_depletion import load_composition_table

    dep = load_composition_table(depletion) if depletion else None
    base = replace(DEFAULT, reactor_type=reactor_type)
    core_cfg = base.pebblecore if reactor_type == "fhr" else base.hexcore
    r_max = (base.pebblecore.R_vessel if reactor_type == "fhr"
             else base.hexcore.pitch_cm * (base.hexcore.fuel_rings
                                           + base.hexcore.reflector_rings
                                           + base.hexcore.shield_rings))
    edges = np.linspace(0.0, r_max, N_RADIAL_RINGS + 1)

    rows: List[dict] = []
    for bu in burnups:
        for T in temperatures:
            cfg = _uniform_config(base, bu, T)
            for rod in rods:
                sample = _diffusion_state(cfg, rod, seed)
                k_diff = float(sample["k_eff"])
                state = BranchState(burnup_mwd_kg=float(bu), temperature_k=float(T),
                                    rod=rod)
                k_ce, k_std, radial = _run_transport(
                    reactor_type, core_cfg, state, edges, particles=particles,
                    batches=batches, inactive=inactive, depletion_table=dep,
                    workdir=os.path.join(workdir, "validate_" + state.key),
                    axial_cm=axial_cm)
                p_diff = _radial_power_diffusion(sample, edges)
                rms, mx = _shape_error(p_diff, radial)
                rows.append({
                    "reactor": reactor_type,
                    "burnup_mwd_kg": bu,
                    "temperature_k": T,
                    "rod": rod,
                    "k_diffusion": k_diff,
                    "k_openmc": k_ce,
                    "k_openmc_std": k_std,
                    "dk": k_diff - k_ce,
                    "d_rho_pcm": (1.0 / k_ce - 1.0 / k_diff) * 1.0e5,
                    "power_shape_rms_rel": rms,
                    "power_shape_max_rel": mx,
                    "n_nodes": int(sample["material_state"].shape[0]),
                })
                r = rows[-1]
                print(f"  {state.key}: k_diff={k_diff:.5f} k_ce={k_ce:.5f}"
                      f" ({k_std:.5f})  d_rho={r['d_rho_pcm']:+.0f} pcm"
                      f"  shape RMS={rms:.3%}", flush=True)
    return rows


def _rod_worths(rows: List[dict]) -> List[dict]:
    """Rod worth from each rod-out / rod-in pair, both ways of computing it."""
    out = []
    by_state: Dict[tuple, Dict[str, dict]] = {}
    for r in rows:
        by_state.setdefault((r["burnup_mwd_kg"], r["temperature_k"]), {})[r["rod"]] = r
    for (bu, T), pair in sorted(by_state.items()):
        if "in" not in pair or "out" not in pair:
            continue
        def worth(key):
            # rho = 1 - 1/k, so worth = rho(out) - rho(in) = 1/k_in - 1/k_out.
            # Positive = inserting the rods removes reactivity.
            return (1.0 / pair["in"][key] - 1.0 / pair["out"][key]) * 1.0e5
        w_d, w_c = worth("k_diffusion"), worth("k_openmc")
        out.append({"burnup_mwd_kg": bu, "temperature_k": T,
                    "rod_worth_diffusion_pcm": w_d, "rod_worth_openmc_pcm": w_c,
                    "rod_worth_error_pct": 100.0 * (w_d - w_c) / w_c if w_c else float("nan")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reactor", choices=["hex", "fhr"], required=True)
    ap.add_argument("--out", required=True, help="CSV report path")
    ap.add_argument("--burnups", type=float, nargs="+", default=[0.0])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[900.0])
    ap.add_argument("--rods", nargs="+", default=["out", "in"], choices=["out", "in"])
    ap.add_argument("--depletion", default=None)
    ap.add_argument("--particles", type=int, default=40000)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--inactive", type=int, default=50)
    ap.add_argument("--axial-cm", type=float, default=None)
    ap.add_argument("--workdir", default="openmc_run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = run(args.reactor, args.burnups, args.temperatures, args.rods,
               particles=args.particles, batches=args.batches,
               inactive=args.inactive, depletion=args.depletion,
               workdir=args.workdir, seed=args.seed, axial_cm=args.axial_cm)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    d_rho = np.array([r["d_rho_pcm"] for r in rows])
    rms = np.array([r["power_shape_rms_rel"] for r in rows])
    print("\nSummary")
    print(f"  states                : {len(rows)}")
    print(f"  reactivity bias       : mean {d_rho.mean():+.0f} pcm, "
          f"max |{np.abs(d_rho).max():.0f}| pcm")
    print(f"  power shape RMS error : mean {np.nanmean(rms):.2%}, "
          f"max {np.nanmax(rms):.2%}")
    for w in _rod_worths(rows):
        print(f"  rod worth @ bu={w['burnup_mwd_kg']:g} T={w['temperature_k']:g}: "
              f"diffusion {w['rod_worth_diffusion_pcm']:.0f} pcm vs OpenMC "
              f"{w['rod_worth_openmc_pcm']:.0f} pcm "
              f"({w['rod_worth_error_pct']:+.1f}%)")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
