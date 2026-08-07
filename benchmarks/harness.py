"""One training / evaluation loop for every benchmarked model.

    python benchmarks/harness.py --config benchmarks/configs/e1_hex_pigno.yaml
    python benchmarks/harness.py --config ... --limit 50 --epochs 5   # smoke test

Every model in the suite runs through this file. That is the point: if PI-GNO
had its own loop, its own schedule or its own early-stopping rule, a win would
be unattributable. Per-model freedom is confined to the architecture and to the
learning rate, which is selected for EACH model from the same grid on val (see
budget.py) so "the baselines were undertuned" has an answer.

What it writes, per run, under results/<exp>/<model>/seed<k>/:
    config.json        the fully resolved config, including the LR that was used
    norm.json          train-split normalization -- without this a checkpoint
                       cannot be evaluated later with the transform it was
                       trained on
    best.pt            model state dict at best val score
    metrics_<split>.csv  per-sample metrics, one row per core
    summary.json       aggregates, timings, param count, provenance

Evaluation always runs at batch size 1 so metrics.sample_metrics sees one core
at a time and the per-material breakdown stays per core. Training batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch

import _paths  # noqa: F401
from config import LossConfig
from features import NodeLayout
from losses import compute_loss
from metrics import sample_metrics

from batching import collate_graphs, expand_k_group_major
from data import (GraphDataset, make_loader, fit_normalization_streaming,
                  save_norm, load_norm)
from interface import build_model, available, missing_reasons


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    exp: str = "adhoc"
    model: str = "pigno"
    data: str = "datasets/hex01"
    hparams: Dict[str, Any] = field(default_factory=dict)
    # loss
    lambda_k: float = 1.0
    lambda_pde: float = 0.0      # E1 is data-only for EVERY model; E2 turns it on
    lambda_bc: float = 0.0
    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    patience: int = 30           # epochs without val improvement before stopping
    warmup_epochs: int = 5
    # plumbing
    device: str = "cuda"
    num_workers: int = 0
    limit: Optional[int] = None
    cache_root: Optional[str] = None
    out_root: str = "results"
    eval_splits: List[str] = field(default_factory=lambda: ["val", "test"])
    # Fixed physical domain (x0, x1, y0, y1) for the grid-based baselines. Left
    # None it is derived ONCE from the train split and written back here, so
    # val/test are probed with exactly the sensors/grid the model was trained
    # on. Deriving it per sample would hand FNO and DeepONet a geometry
    # adaptation they do not actually have.
    domain: Optional[List[float]] = None
    domain_probe: int = 128

    def loss_cfg(self) -> LossConfig:
        return LossConfig(lambda_k=self.lambda_k, lambda_pde=self.lambda_pde,
                          lambda_bc=self.lambda_bc)

    def run_dir(self) -> str:
        return os.path.join(self.out_root, self.exp, self.model, f"seed{self.seed}")


def load_config(path: Optional[str], overrides: Dict[str, Any]) -> RunConfig:
    d: Dict[str, Any] = {}
    if path:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    d.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in RunConfig.__dataclass_fields__.values()}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown config keys {sorted(unknown)}; "
                       f"valid keys are {sorted(known)}")
    return RunConfig(**d)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _provenance(cfg: RunConfig, model, meta: dict) -> dict:
    from scatter import extension_available
    return {
        "git_rev": _git_rev(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_scatter_ext": extension_available(),
        "model": model.describe(),
        "reactor_type": meta.get("reactor_type"),
        "n_groups": meta.get("n_groups"),
        "n_materials": meta.get("n_materials"),
        "node_feature_order": meta.get("node_feature_order"),
        "xs_provenance": meta.get("xs_provenance"),
    }


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #

def _step_loss(model, batch, norm, loss_cfg):
    out = model(batch, norm)
    k_ref_norm = (batch.k_eff - norm.k_mean) / norm.k_std          # [B]
    # each core in the batch has its OWN eigenvalue -> expand per node
    k_vec = expand_k_group_major(out.k_phys, batch.batch, batch.n_groups)
    terms = compute_loss(
        flux_hat_norm=out.flux_norm,
        flux_ref_norm=norm.flux.transform(batch.flux),
        k_hat_norm=out.k_norm, k_ref_norm=k_ref_norm,
        flux_hat_phys=out.flux_phys, k_hat_phys=k_vec,
        A=batch.A, F=batch.F, boundary_mask=batch.boundary_mask, cfg=loss_cfg)
    return out, terms


@torch.no_grad()
def evaluate(model, ds: GraphDataset, norm, layout: NodeLayout, device,
             loss_cfg, time_inference: bool = False):
    """Per-sample evaluation. Returns (rows, aggregate). One core per step."""
    model.eval()
    rows: List[dict] = []
    timings: List[float] = []
    for i in range(len(ds)):
        batch = collate_graphs([ds[i]], device=device)
        if time_inference:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = model(batch, norm)
        if time_inference:
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
        # sample_metrics wants a scalar k for a single core
        out.k_phys = out.k_phys.reshape(())
        m = sample_metrics(out, batch,
                           material_onehot_cols=layout.material_onehot_cols)
        m["sample"] = os.path.basename(ds.paths[i])
        m["n_nodes"] = batch.n_nodes
        if time_inference:
            m["inference_s"] = timings[-1]
        rows.append(m)

    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return rows, agg


def train(cfg: RunConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    train_ds = GraphDataset(cfg.data, "train", cfg.cache_root, cfg.limit)
    meta = train_ds.metadata()
    layout = NodeLayout.from_metadata(meta)

    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # --- normalization: fit on TRAIN ONLY, then persist ---------------------
    norm_path = os.path.join(run_dir, "norm.json")
    if os.path.exists(norm_path):
        norm = load_norm(norm_path)
        print(f"normalization: reused {norm_path}")
    else:
        t0 = time.perf_counter()
        norm = fit_normalization_streaming(
            train_ds, node_passthrough=layout.passthrough_cols,
            num_workers=cfg.num_workers)
        save_norm(norm, norm_path)
        print(f"normalization: fitted on {len(train_ds)} train samples "
              f"in {time.perf_counter() - t0:.1f}s -> {norm_path}")

    # --- fixed domain for the grid-based baselines --------------------------
    # Derived from TRAIN only and frozen into the config. Graph models ignore it.
    if cfg.domain is None:
        from sensors import domain_from_dataset
        cfg.domain = list(domain_from_dataset(train_ds, cfg.domain_probe))
        print(f"domain (from {min(cfg.domain_probe, len(train_ds))} train "
              f"samples): {[round(v, 2) for v in cfg.domain]}")

    model = build_model(cfg.model, meta, domain=tuple(cfg.domain),
                        **cfg.hparams).to(device)
    prov = _provenance(cfg, model, meta)
    print(f"model={cfg.model}  params={model.n_params():,}  device={device}  "
          f"reactor={meta.get('reactor_type')}  N_train={len(train_ds)}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"config": asdict(cfg), "provenance": prov}, f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0,
                                           max(cfg.warmup_epochs, 1)),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             opt, max(cfg.epochs - cfg.warmup_epochs, 1))],
        milestones=[max(cfg.warmup_epochs, 1)])

    loader = make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, seed=cfg.seed)
    val_ds = GraphDataset(cfg.data, "val", cfg.cache_root, cfg.limit)
    loss_cfg = cfg.loss_cfg()

    best = float("inf")
    best_epoch = -1
    history: List[dict] = []
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        running: Dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            _, terms = _step_loss(model, batch, norm, loss_cfg)
            opt.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in terms.item_dict().items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
        sched.step()
        tr = {k: v / max(n_batches, 1) for k, v in running.items()}

        _, vagg = evaluate(model, val_ds, norm, layout, device, loss_cfg)
        # early-stopping score: mean per-group relative flux L2 on val. Reported
        # in the same units as the paper's headline error, so the model selected
        # is the one that is best at what gets reported.
        gkeys = [k for k in vagg if k.startswith("flux_rel_l2_g")]
        score = sum(vagg[k] for k in gkeys) / len(gkeys)

        history.append({"epoch": epoch, "lr": sched.get_last_lr()[0],
                        **{f"train_{k}": v for k, v in tr.items()},
                        "val_flux_rel_l2": score,
                        "val_k_rel_err": vagg["k_rel_err"]})
        if score < best - 1e-6:
            best, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_flux_rel_l2": score, "config": asdict(cfg)},
                       os.path.join(run_dir, "best.pt"))
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:4d}  " +
                  "  ".join(f"{k}={v:.3e}" for k, v in tr.items()) +
                  f"  val_flux={score:.4e}  val_k={vagg['k_rel_err']:.3e}"
                  f"  lr={sched.get_last_lr()[0]:.2e}")
        if epoch - best_epoch >= cfg.patience:
            print(f"early stop at epoch {epoch} "
                  f"(no val improvement since {best_epoch})")
            break

    train_s = time.perf_counter() - t_start

    # --- final evaluation from the BEST checkpoint --------------------------
    ck = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ck["state_dict"])
    summary = {"best_epoch": best_epoch, "best_val_flux_rel_l2": best,
               "train_wall_s": train_s, "n_params": model.n_params(),
               "config": asdict(cfg), "provenance": prov, "splits": {}}

    for split in cfg.eval_splits:
        try:
            ds = GraphDataset(cfg.data, split, cfg.cache_root, cfg.limit)
        except FileNotFoundError:
            print(f"skip split '{split}': not present under {cfg.data}")
            continue
        rows, agg = evaluate(model, ds, norm, layout, device, loss_cfg,
                             time_inference=True)
        summary["splits"][split] = agg
        p = os.path.join(run_dir, f"metrics_{split}.csv")
        cols = sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{split.upper():5s} n={len(rows)}  " + "  ".join(
            f"{k}={agg[k]:.4e}" for k in sorted(agg)
            if k.startswith(("flux_rel_l2_g", "k_", "power_"))))

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"wrote {run_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--exp")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    ap.add_argument("--num-workers", type=int, dest="num_workers")
    ap.add_argument("--lambda-pde", type=float, dest="lambda_pde")
    ap.add_argument("--patience", type=int)
    ap.add_argument("--out-root", dest="out_root")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    if a.list_models:
        print("available:", available())
        for m, why in missing_reasons().items():
            print(f"  unavailable {m}: {why}")
        return

    ov = {k: v for k, v in vars(a).items()
          if k not in ("config", "list_models")}
    cfg = load_config(a.config, ov)
    train(cfg)


if __name__ == "__main__":
    main()
