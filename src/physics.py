"""PDE and boundary residuals computed with the assembled PHYSICS operators.

R = A phi_hat - (1 / k_hat) F phi_hat       (CLAUDE objective)

CRITICAL contract points:
  - A, F are the sparse operators from data_generation (operators.py), shape
    [2N, 2N], group-major ordering [g1(N), g2(N)]. They are NEVER reconstructed
    from the message graph.
  - phi_hat must be PHYSICAL (de-normalized) and packed group-major to match A/F.
  - Sparse mat-vec uses torch.sparse (cuSPARSE on GPU) — no custom CUDA needed.

The model predicts normalized flux; de-normalize via FluxScaler.inverse before
calling here (handled in model.forward / losses.py).
"""

from __future__ import annotations

import torch


def pack_group_major(flux_phys: torch.Tensor) -> torch.Tensor:
    """[N, 2] -> [2N] group-major [phi1(N), phi2(N)] to match A/F ordering."""
    return torch.cat([flux_phys[:, 0], flux_phys[:, 1]], dim=0)


def scipy_csr_to_torch(A_csr, device, dtype=torch.float32) -> torch.Tensor:
    """Convert a scipy CSR matrix to a torch sparse COO tensor (build-time, once
    per sample; cache the result)."""
    A = A_csr.tocoo()
    idx = torch.tensor([A.row, A.col], dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, device=device).coalesce()


def pde_residual(flux_phys: torch.Tensor, k_hat: torch.Tensor,
                 A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Return R [2N]. A, F are torch sparse [2N, 2N]; flux_phys [N, 2] physical."""
    phi = pack_group_major(flux_phys).unsqueeze(1)        # [2N, 1]
    Aphi = torch.sparse.mm(A, phi)
    Fphi = torch.sparse.mm(F, phi)
    R = Aphi - (1.0 / k_hat) * Fphi
    return R.squeeze(1)


def boundary_residual(flux_phys: torch.Tensor, boundary_mask: torch.Tensor
                      ) -> torch.Tensor:
    """Returns boundary-node flux (both groups); mean-square is the zero-flux
    L_BC term.

    NOTE: this is the DIRICHLET-ZERO boundary residual. The default generator
    uses extrapolated-length vacuum (boundary cell-center flux is nonzero), where
    the BC is enforced through A / L_PDE instead — so lambda_bc defaults to 0
    (see losses.py / config.LossConfig). Valid as a loss only with a Dirichlet-
    zero discretization; otherwise this is an informational diagnostic."""
    return flux_phys[boundary_mask]
