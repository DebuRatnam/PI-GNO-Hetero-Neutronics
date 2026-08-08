"""The experiment matrix (E1..E5) and the runner that expands it.

    python benchmarks/experiments.py --list
    python benchmarks/experiments.py --run e1_hex --seeds 0 1 2
    python benchmarks/experiments.py --run e1_hex --select-lr      # LR sweep first

DESIGN, and why each cell is shaped the way it is:

E1  ACCURACY, DATA-ONLY LOSS FOR EVERY MODEL (lambda_pde = 0).
    This is what makes the architecture comparison answerable. If PI-GNO ran
    with its physics loss while the baselines did not, any win would be
    attributable to the loss rather than the architecture, and a reader would
    say so immediately. E1 removes the physics term from PI-GNO too.

E2  PHYSICS-LOSS ABLATION, on the graph models that can take A and F directly.
    This is where the physics-informed term gets its credit, separately and
    visibly, instead of being folded into E1.

E3  RESOLUTION TRANSFER on the paired hex_res family: train at one mesh level,
    evaluate at all of them. The aggregation ablation (sum vs volume) runs here,
    because that is the axis it is supposed to affect.

E4  BOUNDARY-CONDITION TRANSFER on hex_bc: train on beta in [0, 0.4), test on
    [0.55, 0.8]. hex only -- see generate_bc.py for why fhr's published vessel
    makes its boundary condition unmeasurable.

E5  COST. Inference wall clock against the splu reference solve, parameter
    count, training time. Reads the timings the harness already records, so it
    is a reporting step rather than a separate training run.

Budget matching and the LR grid come from budget.py, so every model in a cell is
sized to the same parameter count and tuned over the same grid.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace, asdict
from typing import Dict, List

import _paths  # noqa: F401

from budget import LR_GRID, match_budget
from data import GraphDataset
from harness import RunConfig, train
from sensors import domain_from_dataset

# Common parameter budget. Every model in every cell is bisected to this, so
# "your model is just bigger" has a recorded answer.
PARAM_BUDGET = 400_000

GRAPH_MODELS = ["pigno", "mgn"]
ALL_MODELS = ["pigno", "mgn", "fno", "deeponet"]

# The paired discretization probe: the SAME physical cores meshed six ways.
# L2 is omitted because it is the training discretization, so it appears in the
# ordinary test split rather than as a transfer target. L6 (N ~ 128k) is the
# fine truth mesh used to score accuracy; it is also a legitimate transfer
# target, being ~9x the training node count.
PROBE_MESHES = [
    "datasets/hex_probe/L0",             # different meshing RULE, similar N
    "datasets/hex_probe/L1",             # coarser
    "datasets/hex_probe/L3",             # finer
    "datasets/hex_probe/Lmixed3_1_3",    # density varies WITHIN one graph
    "datasets/hex_probe/L6",             # much finer (truth mesh)
]


EXPERIMENTS: Dict[str, dict] = {
    # ---- E1: architecture, data-only loss ---------------------------------
    "e1_hex": dict(data="datasets/hex01", models=ALL_MODELS, lambda_pde=0.0),
    "e1_fhr": dict(data="datasets/fhr01", models=ALL_MODELS, lambda_pde=0.0),

    # ---- E2: physics-loss ablation ----------------------------------------
    "e2_hex": dict(data="datasets/hex01", models=GRAPH_MODELS,
                   sweep={"lambda_pde": [0.0, 0.1, 1.0]}),
    "e2_fhr": dict(data="datasets/fhr01", models=GRAPH_MODELS,
                   sweep={"lambda_pde": [0.0, 0.1, 1.0]}),

    # ---- E3: discretization transfer --------------------------------------
    # Trained at L2, the MIDDLE uniform level, so both a coarsening and a
    # refinement are tested; training at an end would only ever test one
    # direction. Evaluated on the paired 6-mesh probe, whose configurations are
    # drawn from a different seed and a disjoint control-insertion range, so
    # there is no leakage from the training set.
    "e3_res": dict(data="datasets/hex_res/L2", models=ALL_MODELS, lambda_pde=0.0,
                   transfer_to=PROBE_MESHES),
    # The aggregation ablation is the one that attributes any GNO-vs-GNN
    # difference to the MECHANISM rather than to the model's name: PI-GNO with
    # `sum` is architecturally a GNN in exactly the respect under test, so if it
    # tracks MeshGraphNet while `volume` separates from both, the volume
    # quadrature is what did the work.
    "e3_res_agg": dict(data="datasets/hex_res/L2", models=["pigno"], lambda_pde=0.0,
                       sweep={"hparams.aggregation": ["sum", "volume", "volume_raw"]},
                       transfer_to=PROBE_MESHES),

    # ---- E4: boundary-condition transfer ----------------------------------
    "e4_bc": dict(data="datasets/hex_bc", models=ALL_MODELS, lambda_pde=0.0),
    "e4_bc_pde": dict(data="datasets/hex_bc", models=GRAPH_MODELS,
                      sweep={"lambda_pde": [0.0, 0.1]}),
}


def _cells(name: str) -> List[dict]:
    """Expand one experiment into its (model, sweep-value) cells."""
    spec = EXPERIMENTS[name]
    out = []
    for model in spec["models"]:
        sweep = spec.get("sweep")
        if not sweep:
            out.append({"model": model, "overrides": {}, "tag": model})
            continue
        (key, values), = sweep.items()
        for v in values:
            tag = f"{model}__{key.split('.')[-1]}={v}"
            out.append({"model": model, "overrides": {key: v}, "tag": tag})
    return out


def _apply(cfg: RunConfig, overrides: dict) -> RunConfig:
    """Apply dotted overrides; `hparams.x` writes into the hparams dict."""
    hp = dict(cfg.hparams)
    flat = {}
    for k, v in overrides.items():
        if k.startswith("hparams."):
            hp[k.split(".", 1)[1]] = v
        else:
            flat[k] = v
    return replace(cfg, hparams=hp, **flat)


def build_config(exp: str, cell: dict, seed: int, lr: float,
                 out_root: str, limit=None, device="cuda",
                 num_workers: int = 0, epochs: int = 200) -> RunConfig:
    spec = EXPERIMENTS[exp]
    cfg = RunConfig(exp=exp, model=cell["model"], data=spec["data"],
                    lambda_pde=spec.get("lambda_pde", 0.0),
                    lr=lr, seed=seed, epochs=epochs, limit=limit,
                    device=device, num_workers=num_workers, out_root=out_root)
    cfg = _apply(cfg, cell["overrides"])

    # size to the shared budget, and freeze the domain the grid baselines need
    ds = GraphDataset(cfg.data, "train", limit=min(limit or 64, 64))
    meta = ds.metadata()
    if cfg.domain is None:
        cfg.domain = list(domain_from_dataset(ds, min(len(ds), 64)))
    hp, got = match_budget(cfg.model, meta, PARAM_BUDGET,
                           {**cfg.hparams, "domain": tuple(cfg.domain)},
                           verbose=False)
    hp.pop("domain", None)
    return replace(cfg, hparams=hp)


def select_lr(exp: str, cell: dict, out_root: str, **kw) -> float:
    """Pick the LR from the shared grid on VAL, with a shortened run.

    Every model gets the same grid and the same shortened budget, so the choice
    is equal treatment rather than per-model tuning effort.
    """
    best_lr, best = LR_GRID[0], float("inf")
    for lr in LR_GRID:
        cfg = build_config(exp, cell, seed=0, lr=lr,
                           out_root=os.path.join(out_root, "_lrsel"), **kw)
        cfg = replace(cfg, exp=f"{exp}_lrsel", epochs=max(kw.get("epochs", 200) // 5, 5),
                      eval_splits=[])
        s = train(cfg)
        score = s["best_val_flux_rel_l2"]
        print(f"  lr={lr:.0e} -> val_flux_rel_l2={score:.4e}")
        if score < best:
            best_lr, best = lr, score
    print(f"  selected lr={best_lr:.0e} for {cell['tag']}")
    return best_lr


def run(exp: str, seeds, out_root="results", select: bool = False, **kw):
    results = {}
    for cell in _cells(exp):
        lr = select_lr(exp, cell, out_root, **kw) if select else 1e-3
        for seed in seeds:
            cfg = build_config(exp, cell, seed=seed, lr=lr, out_root=out_root, **kw)
            cfg = replace(cfg, model=cell["model"])
            print(f"\n=== {exp} / {cell['tag']} / seed {seed} "
                  f"(lr={lr:.0e}) ===")
            results[f"{cell['tag']}/seed{seed}"] = train(cfg)

        # resolution / BC transfer: re-evaluate the SAME checkpoint on the other
        # datasets. Re-training per target would answer a different question.
        for tgt in EXPERIMENTS[exp].get("transfer_to", []):
            _transfer_eval(exp, cell, seeds, tgt, out_root, **kw)
    return results


def _transfer_eval(exp, cell, seeds, target_data, out_root, **kw):
    """Evaluate a trained checkpoint on a dataset it was NOT trained on."""
    import csv
    import torch
    from features import NodeLayout
    from harness import evaluate
    from interface import build_model
    from data import load_norm

    for seed in seeds:
        cfg = build_config(exp, cell, seed=seed, lr=1e-3, out_root=out_root, **kw)
        cfg = replace(cfg, model=cell["model"])
        run_dir = cfg.run_dir()
        ck_path = os.path.join(run_dir, "best.pt")
        if not os.path.exists(ck_path):
            print(f"skip transfer: no checkpoint at {ck_path}")
            continue
        ds = GraphDataset(target_data, "test", cfg.cache_root, cfg.limit)
        norm = load_norm(os.path.join(run_dir, "norm.json"))
        model = build_model(cfg.model, ds.metadata(), domain=tuple(cfg.domain),
                            **cfg.hparams).to(cfg.device)
        model.load_state_dict(torch.load(ck_path, weights_only=False)["state_dict"])
        rows, agg = evaluate(model, ds, norm, NodeLayout.from_metadata(ds.metadata()),
                             cfg.device, cfg.loss_cfg(), time_inference=True)
        tag = target_data.strip("/").replace("/", "_")
        with open(os.path.join(run_dir, f"metrics_transfer_{tag}.csv"), "w",
                  newline="") as f:
            cols = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(run_dir, f"transfer_{tag}.json"), "w") as f:
            json.dump({"target": target_data, "metrics": agg}, f, indent=2)
        print(f"  transfer {cfg.model} seed{seed} -> {target_data}: " + "  ".join(
            f"{k}={agg[k]:.4e}" for k in sorted(agg) if k.startswith("flux_rel_l2_g")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--select-lr", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()

    if a.list or not a.run:
        for name, spec in EXPERIMENTS.items():
            cells = _cells(name)
            print(f"{name:12s} data={spec['data']:24s} cells={len(cells):2d}  "
                  f"models={','.join(spec['models'])}")
        return

    run(a.run, a.seeds, out_root=a.out_root, select=a.select_lr,
        epochs=a.epochs, limit=a.limit, device=a.device,
        num_workers=a.num_workers)


if __name__ == "__main__":
    main()
