"""Generate the boundary-condition transfer dataset (benchmark experiment E4).

    python generate_bc.py --out ../../datasets/hex_bc --train 2000 --val 500 --test 500

The albedo beta is the ONLY axis held disjoint across splits. Control insertion,
enrichment boundary, reflector rings, burnup and temperature are drawn i.i.d.
across all three, so a failure to generalize is attributable to the boundary
condition and not to a rod pattern the model never saw.

    train  beta in [0.00, 0.40)
    val    beta in [0.40, 0.55)
    test   beta in [0.55, 0.80]      <- extrapolation

WHY shield_rings = 0, AND WHY THIS IS NOT OPTIONAL.

At the production hex layout (reflector_rings=1, shield_rings=1) the boundary
condition is very nearly invisible. Measured on one core, sweeping beta from 0
to 0.8 moves k_eff by 0.5 pcm and the normalized flux by 0.065%. The shield ring
is optically thick, so essentially nothing reaches the outer boundary and it
does not matter what happens there. A BC-transfer experiment on that geometry
would measure nothing: every model would "generalize" perfectly by ignoring the
BC entirely, and the result would be an artifact.

Dropping the shield ring restores a real effect:

    reflector / shield     dk(beta: 0 -> 0.8)     normalized flux change
        1 / 1                    0.5 pcm                 0.065%
        1 / 0                 1445   pcm                11.08%
        0 / 0                 5371   pcm                29.15%

shield_rings=0 is used here: it keeps the radial reflector, which is physically
the normal thing for a fast core, while making the outer boundary a live part of
the problem. This is a deliberate departure from the production layout and is
recorded in every sample's metadata, so a reader can see that E4 runs on a
reflector-only core rather than the shielded production one.

The same measurement on fhr gives 1.4 pcm / 0.049% at the full published vessel
(bed -> 60 cm graphite reflector -> barrel -> downcomer -> vessel). fhr has no
equivalent knob for thinning that stack, and inventing one would depart from the
gFHR dimensional spec the project pins, so BC transfer is a HEX-ONLY claim
unless that trade is explicitly accepted.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from datagen_config import DEFAULT, robin_alpha_from_albedo
from dataset import make_sample, save_sample

# beta ranges: DISJOINT across splits. This is the axis under test.
BETA_RANGE = {
    "train": (0.00, 0.40),
    "val": (0.40, 0.55),
    "test": (0.55, 0.80),
}
SPLIT_SEED = {"train": 201, "val": 202, "test": 203}

# Reflector-only core: see the module docstring. With the production shield ring
# the boundary condition does not measurably reach the solution.
SHIELD_RINGS = 0
REFLECTOR_RINGS = 1


def generate(out_root: str, counts: dict, validate: bool = True):
    manifest = []
    for split, n in counts.items():
        if n <= 0:
            continue
        rng = np.random.default_rng(SPLIT_SEED[split])
        d = os.path.join(out_root, split)
        os.makedirs(d, exist_ok=True)
        lo, hi = BETA_RANGE[split]
        for i in range(n):
            beta = float(rng.uniform(lo, hi))
            knobs = {
                "boundary_albedo": beta,
                "reflector_rings": REFLECTOR_RINGS,
                "shield_rings": SHIELD_RINGS,
                # i.i.d. across splits on purpose: beta is the only disjoint axis
                "insert_fraction": float(rng.uniform(0.0, 1.0)),
                "enrichment_boundary": int(rng.choice([3, 4, 5])),
            }
            sample = make_sample(DEFAULT, rng=rng, validate=validate, **knobs)
            path = os.path.join(d, f"sample_{i:05d}.npz")
            save_sample(sample, path)
            meta = sample["geometry_metadata"]
            manifest.append({
                "split": split, "path": path,
                "boundary_albedo": beta,
                "robin_alpha": float(robin_alpha_from_albedo(beta)),
                "k_eff": float(sample["k_eff"]),
                "n_nodes": int(meta["n_nodes"]),
                "residual": float(meta.get("solver_residual", float("nan"))),
                "converged": bool(meta.get("solver_converged", False)),
                **{k: v for k, v in knobs.items() if k != "boundary_albedo"},
            })
            if i % 50 == 0:
                print(f"[{split}] {i:05d}  beta={beta:.3f}  "
                      f"k={float(sample['k_eff']):.6f}", flush=True)

    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} samples to {out_root}")
    _report_sensitivity(manifest)
    return manifest


def _report_sensitivity(manifest, n_probe: int = 3):
    """k_eff must actually depend on beta, or E4 is measuring nothing.

    This is a CONTROLLED probe, not a correlation over the manifest. Rod
    insertion varies i.i.d. across these samples and moves k_eff by ~10000 pcm,
    roughly an order of magnitude more than beta does, so a raw corr(beta, k)
    over the split is dominated by insertion and says nothing about the boundary.
    Instead a few cores are re-solved with everything held fixed and only beta
    swept end to end.
    """
    print("\nbeta -> k_eff sensitivity (controlled: only beta varies):")
    lo = BETA_RANGE["train"][0]
    hi = BETA_RANGE["test"][1]
    for i in range(n_probe):
        seed = 900 + i
        ks = []
        for beta in (lo, hi):
            s = make_sample(DEFAULT, rng=np.random.default_rng(seed), validate=False,
                            boundary_albedo=beta, reflector_rings=REFLECTOR_RINGS,
                            shield_rings=SHIELD_RINGS, enrichment_boundary=4,
                            insert_fraction=0.3)
            ks.append(float(s["k_eff"]))
        print(f"  core {i}: k(beta={lo:.2f}) = {ks[0]:.6f} -> "
              f"k(beta={hi:.2f}) = {ks[1]:.6f}   dk = {(ks[1] - ks[0]) * 1e5:+8.1f} pcm")
    print("  (a few pcm here would mean the boundary is not reaching the "
          "solution and E4 would be measuring nothing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    generate(a.out, {"train": a.train, "val": a.val, "test": a.test},
             validate=not a.no_validate)


if __name__ == "__main__":
    main()
