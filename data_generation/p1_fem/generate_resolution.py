"""Generate the PAIRED multi-resolution hex dataset for the resolution-transfer
study (benchmark experiment E3).

    python generate_resolution.py --out ../../datasets/hex_res \
        --levels 1 2 3 --train 400 --eval 300

WHY PAIRED. The question is whether a model trained at one mesh density still
works at another. If each level drew its own random cores, a difference in error
across levels would confound discretization with configuration -- rod insertion
alone moves k_eff by far more than mesh refinement does. So every level here
solves THE SAME physical cores: the same control depths, the same per-assembly
burnup and temperature, the same ring counts. Only `hex_subdiv` changes.

The pairing is exact rather than approximate. `make_core` consumes its rng for
per-assembly control depths, burnup and temperature, and the NUMBER of those
draws depends only on the assembly count, which refinement does not change. So
re-seeding with the same value at each level replays the identical draw sequence
and reproduces the identical physical core. `verify_pairing` asserts this rather
than trusting it.

WHY NOT LEVEL 0. hex_subdiv=0 is the original 13-node submesh, which is a
different mesh TOPOLOGY (6 nodes per ring) from the general scheme (6k nodes on
ring k). Mixing it into the family would confound refinement with a change of
mesh style, so the family is levels 1..3 and datasets/hex01 keeps level 0 for
the accuracy experiments. Levels 1/2/3 give 19/61/127 nodes per assembly, about
4.7k/13.8k/28.2k nodes per core.

Splits keep the same disjoint control-insertion ranges as dataset.make_split_plans,
so a model trained here faces the same extrapolation regime as in E1/E2.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np

from datagen_config import DEFAULT, SamplingConfig
from dataset import make_sample, save_sample

# Same disjoint insertion ranges as the main hex splits, so E3 is measured in
# the same extrapolation regime as E1/E2 rather than an easier one.
SPLIT_RANGES = {
    "train": (0.0, 0.5),
    "val": (0.5, 0.75),
    "test": (0.75, 1.0),
}
SPLIT_SEED = {"train": 101, "val": 102, "test": 103}


def draw_config(rng: np.random.Generator, split: str) -> dict:
    """One physical core configuration, independent of mesh level."""
    lo, hi = SPLIT_RANGES[split]
    return {
        "insert_fraction": float(rng.uniform(lo, hi)),
        "enrichment_boundary": int(rng.choice([3, 4, 5])),
        "reflector_rings": int(rng.choice([1, 2])),
    }


def generate(out_root: str, levels, counts: dict, validate: bool = True):
    manifest = []
    for split, n in counts.items():
        if n <= 0:
            continue
        cfg_rng = np.random.default_rng(SPLIT_SEED[split])
        configs = [draw_config(cfg_rng, split) for _ in range(n)]
        for ci, knobs in enumerate(configs):
            # ONE seed per configuration, replayed at every level -> identical
            # physics, different mesh
            core_seed = SPLIT_SEED[split] * 1_000_000 + ci
            for s in levels:
                d = os.path.join(out_root, f"L{s}", split)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"sample_{ci:05d}.npz")
                sample = make_sample(
                    DEFAULT, rng=np.random.default_rng(core_seed),
                    validate=validate, hex_subdiv=s, **knobs)
                save_sample(sample, path)
                meta = sample["geometry_metadata"]
                manifest.append({
                    "split": split, "level": s, "config_id": ci,
                    "path": path, "core_seed": core_seed,
                    "k_eff": float(sample["k_eff"]),
                    "n_nodes": int(meta["n_nodes"]),
                    "residual": float(meta.get("solver_residual", float("nan"))),
                    "converged": bool(meta.get("solver_converged", False)),
                    **knobs,
                })
                print(f"[{split} L{s}] {ci:05d}  k={float(sample['k_eff']):.6f}  "
                      f"N={meta['n_nodes']}", flush=True)

    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} samples to {out_root}")
    _report_convergence(manifest, levels)
    return manifest


def _report_convergence(manifest, levels):
    """k_eff must converge as the mesh refines. If it does not, the refinement is
    wrong and no model result computed on it means anything."""
    by_cfg = {}
    for m in manifest:
        by_cfg.setdefault((m["split"], m["config_id"]), {})[m["level"]] = m["k_eff"]
    levels = sorted(levels)
    print("\nk_eff convergence (mean over paired configs):")
    for a, b in zip(levels[:-1], levels[1:]):
        d = [abs(v[b] - v[a]) * 1e5 for v in by_cfg.values() if a in v and b in v]
        if d:
            print(f"  L{a} -> L{b}: mean |dk| = {np.mean(d):8.1f} pcm  "
                  f"(max {np.max(d):8.1f})")


def verify_pairing(out_root: str, levels, split: str = "val", n: int = 3):
    """Assert that paired samples really are the same physical core.

    Checks the invariants refinement must preserve: assembly count, control
    depths, ring counts and enrichment boundary. If these drift, the levels are
    different reactors and the study is meaningless.
    """
    from dataset import load_sample
    levels = sorted(levels)
    ok = True
    for ci in range(n):
        metas = []
        for s in levels:
            p = os.path.join(out_root, f"L{s}", split, f"sample_{ci:05d}.npz")
            if not os.path.exists(p):
                return None
            metas.append(load_sample(p)["geometry_metadata"])
        for key in ("n_assemblies", "reflector_rings", "shield_rings",
                    "enrichment_boundary_ring", "control_depths",
                    "n_control_inserted"):
            vals = [m.get(key) for m in metas]
            if any(v != vals[0] for v in vals):
                print(f"  MISMATCH config {ci} on '{key}': {vals}")
                ok = False
        ns = [m["n_nodes"] for m in metas]
        if len(set(ns)) != len(ns):
            print(f"  config {ci}: levels did not change N: {ns}")
            ok = False
    print(f"pairing verified across levels {levels}: {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--eval", type=int, default=300, help="test-split configs")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    if a.verify_only:
        verify_pairing(a.out, a.levels)
        return
    counts = {"train": a.train, "val": a.val, "test": a.eval}
    generate(a.out, a.levels, counts, validate=not a.no_validate)
    verify_pairing(a.out, a.levels)


if __name__ == "__main__":
    main()
