"""Reporting metrics (CLAUDE Reporting Minimums).

Per-group flux error, k_eff error, power-density error, PDE residual, boundary
residual, and per-material/region breakdown. All in physical units.
"""

from __future__ import annotations

import torch

from physics import pde_residual, boundary_residual
from features import MATERIAL_ONEHOT_COLS


def _rel_l2(pred, ref, eps=1e-12):
    return (torch.linalg.vector_norm(pred - ref) /
            torch.linalg.vector_norm(ref).clamp_min(eps)).item()


def sample_metrics(out, sample, *, material_onehot_cols=None) -> dict:
    """Reporting metrics for one sample. `material_onehot_cols` gives the one-hot
    material block column indices (from features.NodeLayout.material_onehot_cols);
    defaults to the Natrium hex block. Per-group flux error covers all G groups."""
    flux_p, flux_r = out.flux_phys, sample.flux
    G = flux_r.shape[1]
    m = {
        "k_abs_err": abs(float(out.k_phys) - float(sample.k_eff)),
        "k_rel_err": abs(float(out.k_phys) - float(sample.k_eff)) / float(sample.k_eff),
        "power_rel_l2": _rel_l2(out.power, sample.power),
        "pde_residual_rms": pde_residual(flux_p, out.k_phys, sample.A, sample.F)
                            .pow(2).mean().sqrt().item(),
        "bc_residual_rms": boundary_residual(flux_p, sample.boundary_mask)
                           .pow(2).mean().sqrt().item(),
    }
    for g in range(G):
        m[f"flux_rel_l2_g{g + 1}"] = _rel_l2(flux_p[:, g], flux_r[:, g])
    # per-material flux error: recover material id from the one-hot block
    cols = material_onehot_cols if material_onehot_cols is not None else MATERIAL_ONEHOT_COLS
    mat = sample.node_feats[:, cols].argmax(dim=1)
    for mid in mat.unique().tolist():
        mask = mat == mid
        m[f"flux_rel_l2_mat{mid}"] = _rel_l2(flux_p[mask], flux_r[mask])
    return m
