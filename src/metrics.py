"""Reporting metrics (CLAUDE Reporting Minimums).

Per-group flux error, k_eff error, power-density error, PDE residual, boundary
residual, and per-material/region breakdown. All in physical units.
"""

from __future__ import annotations

import torch

from physics import pde_residual, boundary_residual
from features import MATERIAL_ONEHOT_COLS


def _rel_l2(pred, ref):
    """Relative L2 error ||pred - ref|| / ||ref||.

    NO ABSOLUTE EPSILON ON THE DENOMINATOR. The previous version clamped it at
    1e-12, which is fine for flux but silently wrong for power: power density is
    E_f * sum_g nuSf_g phi_g with E_f = 3.2e-11 J, so a whole core's power vector
    has norm ~2.5e-13 -- BELOW that clamp. Every power_rel_l2 was therefore
    divided by the constant 1e-12 rather than by the true norm, reporting a
    number about 4x too small that was not a relative error at all and that moved
    with the flux normalization instead of with the model.

    Any fixed epsilon is a unit-dependent assumption about the magnitude of the
    quantity, and this codebase carries fields spanning 1e-14 to 1e+2. The
    degenerate case is handled explicitly instead:
        ref == 0 and pred == 0 -> 0.0   (exactly right)
        ref == 0 and pred != 0 -> inf   (infinitely wrong, and visible)

    Computed in float64: squaring 1e-14 gives 1e-28, which float32 accumulates
    lossily.
    """
    pred = pred.double()
    ref = ref.double()
    den = torch.linalg.vector_norm(ref)
    num = torch.linalg.vector_norm(pred - ref)
    if float(den) == 0.0:
        return 0.0 if float(num) == 0.0 else float("inf")
    return float(num / den)


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
