"""CLI entry point to generate the PI-GNO dataset. NOT auto-run (no GPU needed;
pure CPU SciPy). Invoke manually:

    python generate.py --out ../../datasets/run01                 # Natrium hex (default)
    python generate.py --reactor fhr --out ../../datasets/fhr01   # KP-FHR pebble bed

Generates geometry-disjoint train/val/test splits, solves each eigenvalue problem,
validates, and writes one .npz per sample plus a manifest.json listing k_eff and
solver residuals. Splits are disjoint via per-reactor insertion ranges.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dataclasses import replace

from datagen_config import DEFAULT, SamplingConfig
from dataset import (make_sample, save_sample,
                     make_split_plans, make_split_plans_fhr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--reactor", choices=["hex", "fhr"], default="hex",
                    help="hex = Natrium hex-duct (fast); fhr = KP-FHR pebble bed (thermal)")
    ap.add_argument("--no-validate", action="store_true")
    # Dataset scaling: default to the large-study target (5000/1000/1000); these
    # override SamplingConfig so you choose how many to actually solve.
    ap.add_argument("--train-samples", type=int, default=DEFAULT.sampling.train_samples)
    ap.add_argument("--val-samples", type=int, default=DEFAULT.sampling.val_samples)
    ap.add_argument("--test-samples", type=int, default=DEFAULT.sampling.test_samples)
    args = ap.parse_args()

    sampling = SamplingConfig(train_samples=args.train_samples,
                              val_samples=args.val_samples,
                              test_samples=args.test_samples)
    cfg = replace(DEFAULT, reactor_type=args.reactor, sampling=sampling)
    is_fhr = args.reactor == "fhr"
    plans = make_split_plans_fhr(sampling) if is_fhr else make_split_plans(sampling)
    print(f"Generating {args.reactor} dataset "
          f"{sampling.train_samples}/{sampling.val_samples}/{sampling.test_samples} "
          f"(train/val/test) -> {args.out}")

    manifest = []
    for plan in plans:
        rng = np.random.default_rng(plan.seed)
        for i in range(plan.n_samples):
            if is_fhr:
                ic = float(rng.uniform(*plan.control_insert_range))
                ish = float(rng.uniform(*plan.shutdown_insert_range))
                gpf = float(rng.uniform(*plan.graphite_pebble_range))
                knobs = dict(insert_control=ic, insert_shutdown=ish,
                             graphite_pebble_frac=gpf)
            else:
                knobs = dict(
                    insert_fraction=float(rng.uniform(*plan.insert_fraction_range)),
                    enrichment_boundary=int(rng.choice(plan.enrichment_boundary_choices)),
                    reflector_rings=int(rng.choice(plan.reflector_ring_choices)))
            sample = make_sample(cfg, layout_name=f"{plan.name}-{i}", rng=rng,
                                 validate=not args.no_validate, **knobs)
            path = os.path.join(args.out, plan.name, f"sample_{i:05d}.npz")
            save_sample(sample, path)
            meta = sample["geometry_metadata"]
            row = {
                "split": plan.name, "path": path,
                "reactor_type": args.reactor,
                "k_eff": float(sample["k_eff"]),
                "residual": meta["solver_residual"],
                "converged": meta["solver_converged"],
            }
            if is_fhr:
                row.update(n_pebbles=meta["n_pebbles"],
                           packing_fraction=meta["packing_fraction_actual"],
                           n_control_inserted=meta["n_control_inserted"],
                           n_shutdown_inserted=meta["n_shutdown_inserted"],
                           **knobs)
                tag = (f"peb={meta['n_pebbles']} ctrl={meta['n_control_inserted']}/"
                       f"{meta['n_control']} shut={meta['n_shutdown_inserted']}/{meta['n_shutdown']}")
            else:
                row.update(n_assemblies=meta["n_assemblies"],
                           n_control_inserted=meta["n_control_inserted"], **knobs)
                tag = f"inserted={meta['n_control_inserted']}/13"
            manifest.append(row)
            print(f"[{plan.name}] {i:05d}  k_eff={float(sample['k_eff']):.5f}  "
                  f"res={meta['solver_residual']:.2e}  {tag}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} samples to {args.out}")


if __name__ == "__main__":
    main()
