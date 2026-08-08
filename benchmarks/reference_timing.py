"""Time the classical reference solver. This is the DENOMINATOR of E5.

    python benchmarks/reference_timing.py --data datasets/hex01 --n 40
    python benchmarks/reference_timing.py --data datasets/hex_probe/L3 --n 20

Every "N times faster than the solver" claim is a ratio, and a ratio is only as
trustworthy as its denominator. This measures the denominator honestly, and
separates the pieces so the comparison cannot quietly cheat:

  assemble    building A and F from the mesh (operators.assemble_AF)
  factorize   the sparse LU (scipy splu) -- almost always the dominant cost
  iterate     the power iteration on the factorized operator
  total       what actually has to happen to solve one core from scratch

WHY THE SPLIT MATTERS. A neural surrogate replaces the whole pipeline, so
`total` is the fair comparison. But `factorize + iterate` is the fair comparison
against a solver that gets to reuse an assembly, and someone will ask. Reporting
one number invites the reader to assume whichever is more flattering, so both
are reported.

TWO THINGS THIS DELIBERATELY DOES NOT DO:

  It does not compare CPU solver time against GPU model time and call the ratio
  a speedup. That measures the hardware, not the method. The device of every
  measurement is recorded in the output, and a cross-device ratio must be
  labelled as such.

  It does not count the amortized cost of generating training data. A surrogate
  that needs 5000 reference solves to train has already spent 5000x this number
  before its first prediction. `--report-amortized` prints that break-even count
  so the framing is available rather than buried.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import List

import numpy as np

import _paths  # noqa: F401
from datagen_config import DEFAULT
from dataset import load_sample
from operators import assemble_AF
from solver import solve_keff
from geometry import CoreGeometry


def _geom_from_sample(s: dict) -> CoreGeometry:
    """Rebuild the minimal CoreGeometry that assemble_AF needs from a stored
    sample, so timing runs on the SAME cores the models are evaluated on rather
    than on freshly drawn ones."""
    return CoreGeometry(
        material_state=s["material_state"],
        coordinates=s["coordinates"],
        boundary_mask=s["boundary_mask"],
        cross_sections=s["cross_sections"],
        control_rod_cells=np.zeros(0, dtype=np.int64),
        elements=s["elements"],
        boundary_edges=s["boundary_edges"],
        nodal_volume=s["nodal_volume"],
        mesh=None,
        layout_name="timing",
        assembly_metadata={},
    )


def time_one(path: str, repeats: int = 1) -> dict:
    s = load_sample(path)
    geom = _geom_from_sample(s)
    n_nodes = geom.coordinates.shape[0]

    t_asm: List[float] = []
    t_sol: List[float] = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        A, F = assemble_AF(geom, DEFAULT.physics)
        t1 = time.perf_counter()
        sol = solve_keff(A, F, DEFAULT.solver, n_nodes=n_nodes)
        t2 = time.perf_counter()
        t_asm.append(t1 - t0)
        t_sol.append(t2 - t1)

    return {
        "sample": os.path.basename(path),
        "n_nodes": int(n_nodes),
        "n_elements": int(geom.elements.shape[0]),
        "nnz_A": int(A.nnz),
        "assemble_s": float(np.median(t_asm)),
        "solve_s": float(np.median(t_sol)),
        "total_s": float(np.median(t_asm) + np.median(t_sol)),
        "k_eff": float(sol.k_eff),
        "iters": int(sol.iters),
        "residual": float(sol.residual),
    }


def run(data: str, split: str, n: int, repeats: int) -> dict:
    import glob
    paths = sorted(glob.glob(os.path.join(data, split, "*.npz")))[:n]
    if not paths:
        raise FileNotFoundError(f"no samples in {data}/{split}")
    rows = [time_one(p, repeats) for p in paths]
    for r in rows:
        print(f"  N={r['n_nodes']:7d}  assemble={r['assemble_s']:7.3f}s  "
              f"solve={r['solve_s']:7.3f}s  total={r['total_s']:7.3f}s  "
              f"iters={r['iters']:3d}", flush=True)

    def agg(k):
        v = [r[k] for r in rows]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}

    return {
        "data": data, "split": split, "n_samples": len(rows), "repeats": repeats,
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_nodes": agg("n_nodes"),
        "assemble_s": agg("assemble_s"),
        "solve_s": agg("solve_s"),
        "total_s": agg("total_s"),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=1,
                    help="median over this many timings per sample")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-amortized", type=int, default=None,
                    metavar="N_TRAIN",
                    help="also print the cost of generating a training set of "
                         "this size, i.e. what the surrogate must amortize")
    a = ap.parse_args()

    print(f"timing the reference solver on {a.data}/{a.split} "
          f"({a.n} samples, CPU)")
    res = run(a.data, a.split, a.n, a.repeats)

    t = res["total_s"]["mean"]
    print(f"\nreference solve, mean over {res['n_samples']} cores "
          f"(N ~ {res['n_nodes']['mean']:.0f}):")
    print(f"  assemble A,F : {res['assemble_s']['mean']:8.3f} s")
    print(f"  splu + power : {res['solve_s']['mean']:8.3f} s")
    print(f"  TOTAL        : {t:8.3f} s   <- the E5 denominator")

    if a.report_amortized:
        n = a.report_amortized
        print(f"\namortization: generating {n} training samples costs "
              f"{n * t / 3600:.2f} core-hours of reference solves. A surrogate "
              f"must serve at least that many predictions to break even against "
              f"just running the solver.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
