"""Shard a dataset generation run across independent processes (SLURM array).

Sample generation is embarrassingly parallel -- every core is an independent
eigenvalue solve -- but only if the RANDOM DRAWS are independent too, and by
default they are not.

THE CONSTRAINT. `generate.py` seeds ONE rng per split and then draws inside the
sample loop, so sample i's configuration depends on the rng state left by
samples 0..i-1. Splitting that loop across processes changes the data. Any
generator that wants to shard must therefore derive a PER-SAMPLE seed up front:

    core_seed = SPLIT_SEED[split] * 1_000_000 + index

Then sample i is a pure function of its index, every shard reproduces exactly
the samples it owns, and the union over shards is bit-identical to the serial
run. `generate_resolution.py` was already written this way;
`generate_bc.py` was restructured to match.

USAGE (SLURM array of 40 tasks):

    python generate_bc.py --out ../../datasets/hex_bc \
        --shard $SLURM_ARRAY_TASK_ID --n-shards 40

    # then once, after the array completes:
    python -c "import sharding; sharding.merge_manifests('../../datasets/hex_bc')"

Each shard writes manifest.shard<i>.json; merging concatenates them into the
manifest.json the loaders expect. Merging is a separate step ON PURPOSE -- a
shard that writes straight to manifest.json races every other shard and silently
loses rows.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Sequence


def add_shard_args(ap):
    """Attach --shard / --n-shards to an argparse parser."""
    ap.add_argument("--shard", type=int, default=0,
                    help="index of this shard, 0-based (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                    help="total number of shards; 1 = serial, the default")
    return ap


def owns(index: int, shard: int, n_shards: int) -> bool:
    """Does this shard own sample `index`?

    Round-robin rather than contiguous blocks: cost per sample varies a lot with
    the mesh level and the core size, so contiguous blocks would leave some array
    tasks running long after the others finished.
    """
    if n_shards <= 1:
        return True
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for n_shards={n_shards}")
    return index % n_shards == shard


def manifest_path(out_root: str, shard: int, n_shards: int) -> str:
    if n_shards <= 1:
        return os.path.join(out_root, "manifest.json")
    return os.path.join(out_root, f"manifest.shard{shard:04d}.json")


def write_manifest(rows: List[dict], out_root: str, shard: int, n_shards: int):
    os.makedirs(out_root, exist_ok=True)
    p = manifest_path(out_root, shard, n_shards)
    with open(p, "w") as f:
        json.dump(rows, f, indent=2)
    return p


def merge_manifests(out_root: str, remove_parts: bool = True) -> str:
    """Concatenate manifest.shard*.json into manifest.json.

    Sorted by (split, level, config/index) so the merged manifest is identical
    regardless of the order the array tasks happened to finish in -- otherwise
    two runs of the same job produce different files and nothing downstream is
    reproducible.
    """
    parts = sorted(glob.glob(os.path.join(out_root, "manifest.shard*.json")))
    if not parts:
        raise FileNotFoundError(f"no manifest.shard*.json under {out_root}")
    rows: List[dict] = []
    for p in parts:
        with open(p) as f:
            rows.extend(json.load(f))

    def key(r):
        return (str(r.get("split", "")), str(r.get("level", "")),
                int(r.get("config_id", r.get("index", 0))))

    rows.sort(key=key)
    out = os.path.join(out_root, "manifest.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    if remove_parts:
        for p in parts:
            os.remove(p)
    print(f"merged {len(parts)} shard manifests -> {out} ({len(rows)} rows)")
    return out


def check_complete(out_root: str, expected: Optional[int] = None) -> bool:
    """Verify every .npz referenced by the merged manifest exists.

    An array task that hits the wall clock dies silently and leaves a gap; a
    training run on the survivors then reports numbers on a dataset nobody
    intended. Cheap to check, expensive to miss.
    """
    p = os.path.join(out_root, "manifest.json")
    with open(p) as f:
        rows = json.load(f)
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    ok = not missing
    if missing:
        print(f"MISSING {len(missing)} of {len(rows)} sample files, e.g. "
              f"{missing[:3]}")
    if expected is not None and len(rows) != expected:
        print(f"manifest has {len(rows)} rows, expected {expected} -- a shard "
              f"probably died before writing its manifest")
        ok = False
    if ok:
        print(f"complete: {len(rows)} samples, all files present")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="merge / verify sharded manifests")
    ap.add_argument("out_root")
    ap.add_argument("--expected", type=int, default=None)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    if not a.no_merge:
        merge_manifests(a.out_root)
    check_complete(a.out_root, a.expected)
