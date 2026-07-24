"""Sparse eigenvalue solve for the dominant k_eff mode.

Solves  A phi = (1/k) F phi  <=>  F phi = k A phi  for the largest k_eff via
power iteration on  phi <- A^{-1} F phi, using a single sparse LU of A
(scipy.sparse.linalg.splu). Robust and standard for neutron diffusion.

Returns the eigenvalue k_eff, the flux eigenvector [N, G] (group-major unpacked;
G inferred from A.shape[0] / n_nodes), and the relative PDE residual for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from datagen_config import SolverConfig


@dataclass
class EigenSolution:
    k_eff: float
    flux: np.ndarray         # [N, G]
    residual: float          # ||A phi - (1/k) F phi|| / ||F phi||
    iters: int
    converged: bool


def solve_keff(A: sp.csr_matrix, F: sp.csr_matrix, cfg: SolverConfig,
               *, n_nodes: int) -> EigenSolution:
    GN = A.shape[0]
    assert GN % n_nodes == 0, "A must be [GN, GN] group-major"
    G = GN // n_nodes

    lu = splu(A.tocsc())
    phi = np.ones(GN)
    phi /= np.linalg.norm(phi)
    k = 1.0
    converged = False
    it = 0
    for it in range(1, cfg.max_outer + 1):
        s = F @ phi                      # fission source
        phi_new = lu.solve(s)            # A^{-1} F phi
        k_new = float(np.linalg.norm(F @ phi_new) / np.linalg.norm(s))
        phi_new /= np.linalg.norm(phi_new)
        dk = abs(k_new - k) / max(abs(k_new), 1e-30)
        dphi = np.linalg.norm(phi_new - phi)
        phi, k = phi_new, k_new
        if dk < cfg.tol_k and dphi < cfg.tol_flux:
            converged = True
            break

    # sign convention: dominant flux is non-negative
    if np.sum(phi) < 0:
        phi = -phi

    Fphi = F @ phi
    res = float(np.linalg.norm(A @ phi - (1.0 / k) * Fphi) / max(np.linalg.norm(Fphi), 1e-30))
    flux = phi.reshape(G, n_nodes).T           # [N, G] (group-major -> columns=groups)
    return EigenSolution(k_eff=k, flux=flux, residual=res, iters=it,
                         converged=converged and res < cfg.residual_tol)
