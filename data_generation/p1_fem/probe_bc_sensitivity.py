"""Measure how much the outer boundary condition actually reaches the solution.

    python probe_bc_sensitivity.py --reactor fhr --radii 140 135 133 --full
    python probe_bc_sensitivity.py --reactor hex --shield-rings 0 1

A boundary-condition transfer experiment (benchmark E4) only measures something if
k_eff and the flux shape actually DEPEND on the boundary. This script sweeps the
albedo beta end to end with EVERYTHING ELSE HELD FIXED (same core seed -> same
pebble packing / same lattice, same rod pattern, same enrichment) and reports:

    dk    = (k(beta_hi) - k(beta_lo)) * 1e5                        [pcm]
    drho  = (1/k(beta_lo) - 1/k(beta_hi)) * 1e5                    [pcm]
    dphi  = || phi_hi/||phi_hi|| - phi_lo/||phi_lo|| || / ||phi_lo/||phi_lo|| ||

phi is compared UNIT-NORMALIZED because an eigenvector has no absolute scale; the
question is whether the boundary changes the SHAPE. Both fluxes come off the same
mesh (beta touches only the Marshak coefficient on the boundary edges, never node
placement), so the comparison is nodewise and exact.

Measured, for the record:

    hex, reflector 1 / shield 1 (production)        0.5 pcm     0.065%
    hex, reflector 1 / shield 0 (E4 uses this)   1445   pcm    11.08%
    hex, reflector 0 / shield 0                  5371   pcm    29.15%
    fhr, full published vessel (R=191)              1.4 pcm     0.049%

The fhr number is why E4 is hex-only as it stands: at the published build (bed 120
-> 60 cm graphite reflector -> barrel -> downcomer -> vessel 191) the boundary is
optically invisible and every model would "generalize" perfectly by ignoring it.
The `--radii` sweep tests the one available fix -- truncate the meshed domain and
let the albedo boundary stand in for the reflector you did not mesh, which is the
textbook use of an albedo BC. The floor is set by the control elements and their
graphite channel lining (geometry_pebble.structure_outer_radius = 132.1 cm at the
gFHR build): cutting inside that deletes the rod-insertion variability the dataset
depends on, so it is refused rather than silently allowed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from datagen_config import DEFAULT
from dataset import make_sample


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _shape_change(phi_lo: np.ndarray, phi_hi: np.ndarray) -> float:
    """Relative L2 between the two unit-normalized flux vectors."""
    a, b = _unit(np.asarray(phi_lo, float)), _unit(np.asarray(phi_hi, float))
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def probe(cfg, knobs: dict, seed: int, beta_lo: float, beta_hi: float,
          validate: bool = False) -> dict:
    """Solve the same core at two albedos and report the boundary's reach."""
    out = {}
    for tag, beta in (("lo", beta_lo), ("hi", beta_hi)):
        s = make_sample(cfg, rng=np.random.default_rng(seed), validate=validate,
                        boundary_albedo=beta, **knobs)
        out[tag] = s
    k_lo, k_hi = float(out["lo"]["k_eff"]), float(out["hi"]["k_eff"])
    phi_lo, phi_hi = out["lo"]["flux"], out["hi"]["flux"]
    if phi_lo.shape != phi_hi.shape:                 # would mean beta moved the mesh
        raise RuntimeError(f"mesh changed with beta: {phi_lo.shape} vs {phi_hi.shape}")
    G = phi_lo.shape[1]
    return {
        "n_nodes": int(phi_lo.shape[0]),
        "k_lo": k_lo, "k_hi": k_hi,
        "dk_pcm": (k_hi - k_lo) * 1e5,
        "drho_pcm": (1.0 / k_lo - 1.0 / k_hi) * 1e5,
        "dphi": _shape_change(phi_lo.ravel(), phi_hi.ravel()),
        "dphi_g": [_shape_change(phi_lo[:, g], phi_hi[:, g]) for g in range(G)],
        "converged": bool(out["lo"]["geometry_metadata"]["solver_converged"]
                          and out["hi"]["geometry_metadata"]["solver_converged"]),
    }


def _fhr_knobs(cfg, n_control_in: int, n_shutdown_in: int, seed: int) -> dict:
    """A fixed, representative binary rod pattern (fhr insertion is per-element
    binary -- see dataset.FHRSplitPlan). Drawn from its own rng so the pattern does
    not shift when the core seed is reused for the packing."""
    r = np.random.default_rng(seed)
    nc, ns = cfg.pebblecore.n_control, cfg.pebblecore.n_shutdown
    ic, ish = np.zeros(nc), np.zeros(ns)
    ic[r.choice(nc, size=n_control_in, replace=False)] = 1.0
    ish[r.choice(ns, size=n_shutdown_in, replace=False)] = 1.0
    return {"insert_control": ic.tolist(), "insert_shutdown": ish.tolist(),
            "graphite_pebble_frac": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactor", choices=["hex", "fhr"], default="fhr")
    ap.add_argument("--beta-lo", type=float, default=0.0)
    ap.add_argument("--beta-hi", type=float, default=0.8)
    ap.add_argument("--cores", type=int, default=1,
                    help="independent core seeds per case (packing varies)")
    ap.add_argument("--seed0", type=int, default=900)
    ap.add_argument("--validate", action="store_true")
    # fhr: meshed-domain truncation radii to test
    ap.add_argument("--radii", type=float, nargs="*", default=[],
                    help="fhr: domain_outer_r values to sweep")
    ap.add_argument("--full", action="store_true",
                    help="fhr: also measure the untruncated published vessel")
    ap.add_argument("--control-in", type=int, default=3)
    ap.add_argument("--shutdown-in", type=int, default=0)
    # hex: shield-ring counts to test
    ap.add_argument("--shield-rings", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--insert-fraction", type=float, default=0.3)
    a = ap.parse_args()

    cfg = replace(DEFAULT, reactor_type=a.reactor)
    cases = []                                    # (label, knobs)
    if a.reactor == "fhr":
        base = _fhr_knobs(cfg, a.control_in, a.shutdown_in, seed=7)
        if a.full or not a.radii:
            cases.append((f"full vessel R={cfg.pebblecore.R_vessel:g}", dict(base)))
        for r in a.radii:
            cases.append((f"truncated R={r:g}", dict(base, domain_outer_r=float(r))))
    else:
        for sr in a.shield_rings:
            cases.append((f"reflector 1 / shield {sr}",
                          {"reflector_rings": 1, "shield_rings": sr,
                           "enrichment_boundary": 4,
                           "insert_fraction": a.insert_fraction}))

    print(f"{a.reactor}: beta {a.beta_lo:g} -> {a.beta_hi:g}, "
          f"{a.cores} core seed(s) per case, everything else held fixed\n")
    hdr = f"{'case':<26} {'seed':>5} {'N':>7} {'k(lo)':>9} {'k(hi)':>9} " \
          f"{'dk pcm':>9} {'drho pcm':>9} {'dphi %':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, knobs in cases:
        for c in range(a.cores):
            seed = a.seed0 + c
            r = probe(cfg, knobs, seed, a.beta_lo, a.beta_hi, validate=a.validate)
            flag = "" if r["converged"] else "  <- NOT CONVERGED"
            print(f"{label:<26} {seed:>5} {r['n_nodes']:>7} {r['k_lo']:>9.6f} "
                  f"{r['k_hi']:>9.6f} {r['dk_pcm']:>9.1f} {r['drho_pcm']:>9.1f} "
                  f"{100 * r['dphi']:>8.3f}{flag}")
            print(f"{'':<26} {'':>5} per-group dphi %: "
                  + "  ".join(f"g{g + 1}={100 * v:.3f}" for g, v in
                              enumerate(r["dphi_g"])))
    print("\nE4 needs the boundary to be a live part of the problem. hex at "
          "shield_rings=0 measures 1445 pcm / 11.1%; a case coming back at a few "
          "pcm is measuring nothing.")


if __name__ == "__main__":
    main()
