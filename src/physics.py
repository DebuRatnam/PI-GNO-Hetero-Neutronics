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

import numpy as np
import torch


def pack_group_major(flux_phys: torch.Tensor) -> torch.Tensor:
    """[N, G] -> [GN] group-major [phi_1(N), ..., phi_G(N)] to match A/F ordering.

    G is read from the tensor, never hardcoded: the same packing has to serve
    G=2 today and a G=4/G=8 collapse later without a second code path. For a
    BATCHED graph the same expression is still correct provided the batched A/F
    were remapped to batched group-major indexing g*sum(N) + node (see
    benchmarks/batching.py) -- that is exactly the ordering this produces when
    `flux_phys` is the concatenation of the samples' node blocks.
    """
    return flux_phys.t().reshape(-1)


def scipy_csr_to_torch(A_csr, device, dtype=torch.float32) -> torch.Tensor:
    """Convert a scipy CSR matrix to a torch sparse COO tensor (build-time, once
    per sample; cache the result)."""
    A = A_csr.tocoo()
    # np.stack first: torch.tensor on a list of ndarrays copies element-wise and
    # is very slow, and this runs once per operator per sample load.
    idx = torch.as_tensor(np.stack([A.row, A.col]), dtype=torch.long,
                          device=device)
    val = torch.as_tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, device=device).coalesce()


def pde_residual(flux_phys: torch.Tensor, k_hat: torch.Tensor,
                 A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Return R [GN]. A, F are torch sparse [GN, GN]; flux_phys [N, G] physical.

    `k_hat` is either a 0-d scalar (single graph) or a [GN] vector already
    expanded to group-major node order (batched: each graph carries its OWN
    eigenvalue, so a single k across a batch would be wrong -- use
    benchmarks.batching.expand_k_group_major to build it).
    """
    phi = pack_group_major(flux_phys).unsqueeze(1)        # [GN, 1]
    Aphi = torch.sparse.mm(A, phi)
    Fphi = torch.sparse.mm(F, phi)
    inv_k = (1.0 / k_hat)
    if inv_k.dim() > 0:
        inv_k = inv_k.reshape(-1, 1)                       # [GN, 1]
    R = Aphi - inv_k * Fphi
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
