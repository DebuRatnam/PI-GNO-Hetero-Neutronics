"""Aggregate results/ into the tables and plots the paper reports.

    python benchmarks/report.py --results results --out results/report

Reads every summary.json under results/ and emits:
  * a markdown table per experiment, mean +- std over seeds
  * resolution-transfer and BC-transfer curves
  * an accuracy-vs-cost scatter (E5)
  * the per-material breakdown already present in the metric CSVs

Two reporting rules that are not cosmetic:

  The FNO rows carry the INTERPOLATION FLOOR alongside them. FNO's error is
  bounded below by the mesh->grid->mesh round trip that its architecture forces,
  and a table that omits the floor invites the reader to attribute the whole gap
  to the architecture.

  Every number here is an EXTRAPOLATION number. The generator holds train, val
  and test disjoint on control insertion (hex 0-0.5 / 0.5-0.75 / 0.75-1.0; fhr
  0-5 / 6-7 / 8-10 rods inserted), so these are not i.i.d. holdouts and absolute
  errors are higher than an i.i.d. split would give. The banner says so on every
  emitted report rather than leaving it to a footnote nobody reads.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import _paths  # noqa: F401

BANNER = (
    "> **All errors below are extrapolation errors.** train/val/test are held "
    "disjoint on control insertion by the generator (`dataset.make_split_plans`, "
    "`make_split_plans_fhr`), so these are not i.i.d. holdouts. Absolute values "
    "are higher than an i.i.d. split would give; the comparison between models "
    "is the meaningful part."
)

KEY_METRICS = ["flux_rel_l2_g1", "flux_rel_l2_g2", "k_rel_err",
               "power_rel_l2", "pde_residual_rms"]


def load_runs(results_root: str):
    runs = []
    for p in glob.glob(os.path.join(results_root, "**", "summary.json"),
                       recursive=True):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cfg = s.get("config", {})
        if "_lrsel" in p:            # LR-selection runs are not results
            continue
        run = {
            "dir": os.path.dirname(p),
            "exp": cfg.get("exp"), "model": cfg.get("model"),
            "seed": cfg.get("seed"), "lr": cfg.get("lr"),
            "lambda_pde": cfg.get("lambda_pde"),
            "aggregation": (cfg.get("hparams") or {}).get("aggregation"),
            "n_params": s.get("n_params"),
            "train_wall_s": s.get("train_wall_s"),
            "splits": s.get("splits", {}),
        }
        for t in glob.glob(os.path.join(run["dir"], "transfer_*.json")):
            with open(t) as f:
                d = json.load(f)
            run.setdefault("transfer", {})[d["target"]] = d["metrics"]
        runs.append(run)
    return runs


def _agg(values):
    """mean, std over seeds; std is None for a single seed rather than 0.0,
    which would falsely read as a converged spread."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, None
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, var ** 0.5


def _cell(values, fmt="{:.3e}"):
    m, s = _agg(values)
    return fmt.format(m) if s is None else f"{fmt.format(m)} ± {fmt.format(s)}"


def table_for_experiment(runs, exp: str, split: str = "test") -> str:
    rows = defaultdict(list)
    for r in runs:
        if r["exp"] != exp or split not in r["splits"]:
            continue
        key = (r["model"], r.get("lambda_pde"), r.get("aggregation"))
        rows[key].append(r)
    if not rows:
        return ""

    show_pde = len({k[1] for k in rows}) > 1
    show_agg = len({k[2] for k in rows}) > 1
    head = ["model"] + (["lambda_pde"] if show_pde else []) \
        + (["aggregation"] if show_agg else []) \
        + ["params", "seeds"] + KEY_METRICS
    out = [f"### {exp} ({split})", "", "| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]

    for key in sorted(rows, key=lambda k: (k[0], str(k[1]), str(k[2]))):
        rs = rows[key]
        line = [key[0]]
        if show_pde:
            line.append(str(key[1]))
        if show_agg:
            line.append(str(key[2]))
        line += [f"{rs[0]['n_params']:,}", str(len(rs))]
        for m in KEY_METRICS:
            vals = [r["splits"][split][m] for r in rs if m in r["splits"][split]]
            line.append(_cell(vals) if vals else "-")
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out) + "\n"


def transfer_table(runs, exp: str) -> str:
    rows = defaultdict(lambda: defaultdict(list))
    targets = set()
    for r in runs:
        if r["exp"] != exp:
            continue
        key = (r["model"], r.get("aggregation"))
        if "test" in r["splits"]:
            rows[key]["(trained-on)"].append(r["splits"]["test"])
        for tgt, m in (r.get("transfer") or {}).items():
            rows[key][tgt].append(m)
            targets.add(tgt)
    if not rows or not targets:
        return ""          # nothing was transferred; a one-column table is noise
    cols = ["(trained-on)"] + sorted(targets)
    head = ["model", "aggregation"] + [f"{c}\nflux_g1" for c in cols]
    out = [f"### {exp} — transfer (mean per-group flux rel L2, g1)", "",
           "| " + " | ".join(h.replace("\n", " ") for h in head) + " |",
           "|" + "---|" * len(head)]
    for key in sorted(rows, key=lambda k: (k[0], str(k[1]))):
        line = [key[0], str(key[1])]
        for c in cols:
            vals = [m["flux_rel_l2_g1"] for m in rows[key].get(c, [])
                    if "flux_rel_l2_g1" in m]
            line.append(_cell(vals) if vals else "-")
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out) + "\n"


def cost_table(runs) -> str:
    """E5: accuracy against cost, including inference time vs the splu solve."""
    rows = defaultdict(list)
    for r in runs:
        if "test" not in r["splits"]:
            continue
        rows[(r["exp"], r["model"])].append(r)
    if not rows:
        return ""
    head = ["exp", "model", "params", "train_wall_s", "inference_s/core",
            "flux_rel_l2_g1"]
    out = ["### E5 — cost", "", "| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]
    for key in sorted(rows):
        rs = rows[key]
        inf = [r["splits"]["test"]["inference_s"] for r in rs
               if "inference_s" in r["splits"]["test"]]
        out.append("| " + " | ".join([
            key[0], key[1], f"{rs[0]['n_params']:,}",
            _cell([r["train_wall_s"] for r in rs], "{:.0f}"),
            _cell(inf, "{:.4f}") if inf else "-",
            _cell([r["splits"]["test"]["flux_rel_l2_g1"] for r in rs]),
        ]) + " |")
    return "\n".join(out) + "\n"


def interpolation_floor_table(data_roots, sides=(64, 96, 128, 192),
                              n: int = 16) -> str:
    """The lower bound on any FNO result, measured not asserted."""
    import torch  # noqa: F401
    from batching import collate_graphs
    from data import GraphDataset
    from rasterize import interpolation_floor
    from sensors import domain_from_dataset

    out = ["### FNO interpolation floor (mesh → grid → mesh, reference flux)",
           "",
           "Lower bound on FNO error at each grid size. FNO cannot beat these "
           "numbers regardless of training, because its architecture requires a "
           "uniform grid and this problem's geometry is not one.", "",
           "| dataset | grid | floor g1 | floor g2 |", "|---|---|---|---|"]
    for root in data_roots:
        if not os.path.isdir(os.path.join(root, "test")):
            continue
        ds = GraphDataset(root, "test", limit=n)
        dom = domain_from_dataset(GraphDataset(root, "train", limit=32), 32)
        for side in sides:
            acc = defaultdict(float)
            for i in range(len(ds)):
                f = interpolation_floor(collate_graphs([ds[i]]), dom, side)
                for k, v in f.items():
                    if k != "floor_side":
                        acc[k] += v / len(ds)
            out.append(f"| {os.path.basename(root)} | {side}² | "
                       f"{acc['floor_flux_rel_l2_g1']:.4f} | "
                       f"{acc['floor_flux_rel_l2_g2']:.4f} |")
    return "\n".join(out) + "\n"


def plots(runs, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)

    pts = defaultdict(list)
    for r in runs:
        if "test" in r["splits"] and r["n_params"]:
            inf = r["splits"]["test"].get("inference_s")
            if inf:
                pts[r["model"]].append((inf, r["splits"]["test"]["flux_rel_l2_g1"]))
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for model, xy in sorted(pts.items()):
            ax.scatter([p[0] for p in xy], [p[1] for p in xy], label=model, s=42)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("inference time per core (s)")
        ax.set_ylabel("flux rel L2 (group 1)")
        ax.set_title("Accuracy vs cost")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "accuracy_vs_cost.png"), dpi=150)
        plt.close(fig)
        print(f"wrote {out_dir}/accuracy_vs_cost.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/report")
    ap.add_argument("--floor-data", nargs="*",
                    default=["datasets/hex01", "datasets/fhr01"])
    ap.add_argument("--no-floor", action="store_true")
    a = ap.parse_args()

    runs = load_runs(a.results)
    if not runs:
        print(f"no summary.json found under {a.results}")
        return
    print(f"loaded {len(runs)} runs")

    os.makedirs(a.out, exist_ok=True)
    parts = ["# PI-GNO benchmark results", "", BANNER, ""]
    for exp in sorted({r["exp"] for r in runs if r["exp"]}):
        parts += [table_for_experiment(runs, exp), transfer_table(runs, exp)]
    parts.append(cost_table(runs))
    if not a.no_floor:
        parts.append(interpolation_floor_table(a.floor_data))

    md = "\n".join(p for p in parts if p)
    path = os.path.join(a.out, "report.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"wrote {path}")
    plots(runs, a.out)


if __name__ == "__main__":
    main()
