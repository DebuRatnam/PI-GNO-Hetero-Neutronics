"""Physics-informed objective.

    L = L_flux + lambda_k L_k + lambda_PDE L_PDE + lambda_BC L_BC
    R = A phi_hat - (1/k_hat) F phi_hat
    L_PDE = mean(R^2)
    L_BC  = mean(phi_hat[boundary_mask]^2)

L_flux is flux MSE (computed in NORMALIZED units for scale stability); L_k is
k_eff MSE. L_PDE and L_BC use PHYSICAL flux because A/F are physical (see
physics.py). All weights come from LossConfig and are logged.

BOUNDARY-CONDITION CAVEAT: with the default data generator, vacuum is imposed as
extrapolated-length leakage folded into A, so boundary cell-center flux is
NONZERO in the reference and the BC is already enforced via L_PDE. The zero-flux
L_BC = mean(phi[boundary]^2) below would fight the reference, so lambda_bc
defaults to 0.0 (see config.LossConfig). Only enable L_BC with a Dirichlet-zero
boundary discretization. The term is kept here for that case and for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as Fn

from physics import pde_residual, boundary_residual


@dataclass
class LossTerms:
    total: torch.Tensor
    flux: torch.Tensor
    k: torch.Tensor
    pde: torch.Tensor
    bc: torch.Tensor

    def item_dict(self) -> dict:
        return {k: float(v.detach()) for k, v in self.__dict__.items()}


def compute_loss(
    *,
    flux_hat_norm: torch.Tensor,    # [N,2] normalized prediction
    flux_ref_norm: torch.Tensor,    # [N,2] normalized reference
    k_hat_norm: torch.Tensor,       # scalar normalized prediction
    k_ref_norm: torch.Tensor,       # scalar normalized reference
    flux_hat_phys: torch.Tensor,    # [N,2] de-normalized prediction
    k_hat_phys: torch.Tensor,       # scalar de-normalized prediction (for residual)
    A: torch.Tensor, F: torch.Tensor,   # torch sparse [2N,2N]
    boundary_mask: torch.Tensor,    # [N] bool
    cfg,                            # LossConfig
) -> LossTerms:
    l_flux = Fn.mse_loss(flux_hat_norm, flux_ref_norm)
    l_k = Fn.mse_loss(k_hat_norm, k_ref_norm)

    R = pde_residual(flux_hat_phys, k_hat_phys, A, F)
    l_pde = R.pow(2).mean()

    bphi = boundary_residual(flux_hat_phys, boundary_mask)
    l_bc = bphi.pow(2).mean() if bphi.numel() > 0 else flux_hat_phys.new_zeros(())

    total = l_flux + cfg.lambda_k * l_k + cfg.lambda_pde * l_pde + cfg.lambda_bc * l_bc
    return LossTerms(total=total, flux=l_flux, k=l_k, pde=l_pde, bc=l_bc)
