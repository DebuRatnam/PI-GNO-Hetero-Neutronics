"""Discretization-invariance probe: the GNO-vs-GNN measurement.

    python benchmarks/invariance.py --probe datasets/hex_probe \
        --run results/e3_res/pigno/seed0 --meshes L0 L1 L2 L3 mixed --truth L6

WHAT THIS MEASURES, AND WHY THE OBVIOUS METRIC IS WRONG

The claim is that a graph NEURAL OPERATOR is discretization-invariant while a
GNN is not: given two meshes of the SAME physical core, the operator should
return the same physical field, and the GNN should not.

The obvious way to score that -- error against each mesh's own FEM reference --
does not measure it. Those references are not the same field. Measured on one
paired core, the reference solutions disagree with each other by 6.4% (L1 vs L3,
group 1) after volume normalization, and k_eff moves 3124 pcm from L1 to L2.
Scoring against each level's own reference therefore charges the model for the
FEM's convergence error, and a model that perfectly fit every mesh would look
maximally mesh-DEPENDENT.

So two numbers are reported, and the first is the claim:

  INVARIANCE (no reference needed)
      inv(a,b) = || f(mesh_a) interpolated onto mesh_b  -  f(mesh_b) ||
                 / || f(mesh_b) ||
      after normalizing each field to integral(sum_g phi_g) dV = 1.
      A discretization-invariant operator scores ~0. The FEM references
      themselves score 6.4% on L1/L3, so that is the number to beat and it is
      printed alongside as `reference` for exactly that reason.

  ACCURACY vs a fixed fine truth mesh (default L6, N ~ 102k)
      Scored against ONE target for every mesh, so cross-level errors are
      comparable. Reported with each level's own discretization floor beside it,
      because the coarse references are far from converged: observed convergence
      order is p ~ 1.0 (confirmed on levels 3,4,5,6), and Richardson
      extrapolation puts even L6 about 1100 pcm from the continuum.

Accuracy is reported because invariance alone is gameable: a model that outputs
a constant is perfectly invariant. Invariance LOW and accuracy LOW together is
the only combination that supports the claim.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

import _paths  # noqa: F401
from dataio import load_torch_sample

from batching import collate_graphs
from interp import P1Interpolator, normalize_flux


def _npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def field_disagreement(field_a, sample_a, field_b, sample_b) -> Dict[str, float]:
    """Interpolate `field_a` (on sample_a's mesh) onto sample_b's nodes and
    compare, after mesh-independent normalization of BOTH."""
    fa = normalize_flux(_npy(field_a), _npy(sample_a.nodal_volume))
    fb = normalize_flux(_npy(field_b), _npy(sample_b.nodal_volume))
    interp = P1Interpolator(_npy(sample_a.coords), _npy(sample_a.elements))
    fab, fallback = interp(fa, _npy(sample_b.coords))
    out = {f"flux_rel_l2_g{g + 1}": _rel_l2(fab[:, g], fb[:, g])
           for g in range(fb.shape[1])}
    out["fallback_frac"] = fallback
    return out


class ModelField:
    """Evaluate a trained BenchModel on a sample and return physical flux + k."""

    def __init__(self, run_dir: str, device="cpu"):
        from data import load_norm
        from features import NodeLayout
        from interface import build_model

        with open(os.path.join(run_dir, "config.json")) as f:
            blob = json.load(f)
        self.cfg = blob["config"]
        self.device = device
        self.norm = load_norm(os.path.join(run_dir, "norm.json"))
        self.run_dir = run_dir
        self._build = build_model
        self._layout = NodeLayout
        self._model = None
        self._meta_key = None

    def _get(self, meta):
        # rebuilt when the schema changes (feature width can differ per dataset)
        key = (meta.get("n_materials"), meta.get("n_groups"),
               len(meta.get("node_feature_order") or []))
        if self._model is None or key != self._meta_key:
            m = self._build(self.cfg["model"], meta,
                            domain=tuple(self.cfg["domain"]),
                            **(self.cfg.get("hparams") or {}))
            ck = torch.load(os.path.join(self.run_dir, "best.pt"),
                            weights_only=False)
            m.load_state_dict(ck["state_dict"])
            self._model = m.to(self.device).eval()
            self._meta_key = key
        return self._model

    @torch.no_grad()
    def __call__(self, sample):
        b = collate_graphs([sample], device=self.device)
        out = self._get(sample.meta)(b, self.norm)
        return _npy(out.flux_phys), float(out.k_phys.reshape(()))


def probe(probe_root: str, meshes: List[str], truth: Optional[str],
          run_dir: Optional[str], split: str = "test",
          n_configs: Optional[int] = None, device="cpu") -> dict:
    """Run the discretization probe over paired configurations."""
    model = ModelField(run_dir, device) if run_dir else None

    def load(mesh, ci):
        p = os.path.join(probe_root, mesh, split, f"sample_{ci:05d}.npz")
        return load_torch_sample(p) if os.path.exists(p) else None

    # how many paired configurations are present in every mesh
    ci, configs = 0, []
    while True:
        if all(os.path.exists(os.path.join(probe_root, m, split,
                                           f"sample_{ci:05d}.npz"))
               for m in meshes):
            configs.append(ci); ci += 1
        else:
            break
        if n_configs and len(configs) >= n_configs:
            break
    if not configs:
        raise FileNotFoundError(
            f"no configuration present in all of {meshes} under "
            f"{probe_root}/*/{split}")
    print(f"probe: {len(configs)} paired configs x {len(meshes)} meshes"
          + (f" (+ truth {truth})" if truth else ""))

    inv_model = defaultdict(list)
    inv_ref = defaultdict(list)
    acc_model = defaultdict(list)
    acc_ref = defaultdict(list)
    kerr = defaultdict(list)

    ref_mesh = meshes[-1]
    for ci in configs:
        S = {m: load(m, ci) for m in meshes}
        T = load(truth, ci) if truth else None
        preds = {m: model(S[m]) for m in meshes} if model else None

        # --- invariance: every mesh against the reference mesh of the pair ----
        for m in meshes:
            if m == ref_mesh:
                continue
            d = field_disagreement(S[m].flux, S[m], S[ref_mesh].flux, S[ref_mesh])
            inv_ref[m].append(d)
            if model:
                d = field_disagreement(preds[m][0], S[m],
                                       preds[ref_mesh][0], S[ref_mesh])
                inv_model[m].append(d)

        # --- accuracy against the fine truth mesh ----------------------------
        if T is not None:
            for m in meshes:
                acc_ref[m].append(field_disagreement(S[m].flux, S[m],
                                                     T.flux, T))
                if model:
                    acc_model[m].append(field_disagreement(preds[m][0], S[m],
                                                           T.flux, T))
                    kerr[m].append(abs(preds[m][1] - float(T.k_eff)) * 1e5)

    def mean(d):
        return {m: {k: float(np.mean([r[k] for r in rows]))
                    for k in rows[0]} for m, rows in d.items() if rows}

    return {
        "configs": len(configs), "meshes": meshes, "truth": truth,
        "reference_mesh": ref_mesh,
        "invariance_reference": mean(inv_ref),
        "invariance_model": mean(inv_model),
        "accuracy_reference_vs_truth": mean(acc_ref),
        "accuracy_model_vs_truth": mean(acc_model),
        "k_err_pcm_vs_truth": {m: float(np.mean(v)) for m, v in kerr.items() if v},
    }


def format_report(res: dict) -> str:
    ref = res["reference_mesh"]
    L = [f"# Discretization probe ({res['configs']} paired configs)", "",
         f"Invariance is measured against mesh **{ref}**; lower is more "
         f"discretization-invariant.", "",
         "`reference` is the FEM solution's OWN mesh-dependence -- the number a "
         "model must beat to be more discretization-invariant than the "
         "discretization it was trained on.", "",
         "| mesh | reference g1 | model g1 | reference g2 | model g2 |",
         "|---|---|---|---|---|"]
    for m in res["meshes"]:
        if m == ref:
            continue
        r = res["invariance_reference"].get(m, {})
        mo = res["invariance_model"].get(m, {})
        L.append(f"| {m} | {r.get('flux_rel_l2_g1', float('nan')):.4f} | "
                 f"{mo.get('flux_rel_l2_g1', float('nan')):.4f} | "
                 f"{r.get('flux_rel_l2_g2', float('nan')):.4f} | "
                 f"{mo.get('flux_rel_l2_g2', float('nan')):.4f} |")

    if res.get("accuracy_reference_vs_truth"):
        L += ["", f"## Accuracy vs truth mesh ({res['truth']})", "",
              "`reference` here is the DISCRETIZATION FLOOR: how far that mesh's "
              "own FEM solution sits from the fine truth. A model cannot be "
              "faulted for error below its floor.", "",
              "| mesh | floor g1 | model g1 | model k err (pcm) |",
              "|---|---|---|---|"]
        for m in res["meshes"]:
            fl = res["accuracy_reference_vs_truth"].get(m, {})
            mo = res["accuracy_model_vs_truth"].get(m, {})
            k = res["k_err_pcm_vs_truth"].get(m)
            L.append(f"| {m} | {fl.get('flux_rel_l2_g1', float('nan')):.4f} | "
                     f"{mo.get('flux_rel_l2_g1', float('nan')):.4f} | "
                     + (f"{k:.0f} |" if k is not None else "- |"))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probe dataset root")
    ap.add_argument("--meshes", nargs="+", default=["L1", "L2", "L3"])
    ap.add_argument("--truth", default=None, help="fine truth mesh, e.g. L6")
    ap.add_argument("--run", default=None, help="results/<exp>/<model>/seed<k>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--configs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = probe(a.probe, a.meshes, a.truth, a.run, a.split, a.configs, a.device)
    txt = format_report(res)
    print()
    print(txt)
    # Default the output beside the checkpoint it describes: report.py collects
    # <run_dir>/invariance.json to build the cross-model comparison table, so
    # writing elsewhere silently leaves the probe out of the final report.
    if a.out is None and a.run:
        a.out = os.path.join(a.run, "invariance.md")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
