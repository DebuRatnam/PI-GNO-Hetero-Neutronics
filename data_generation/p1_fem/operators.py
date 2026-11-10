"""Assemble the sparse multigroup diffusion operators A (loss/removal/leakage) and
F (fission production) with LINEAR (P1) finite elements on the irregular Delaunay
mesh (geometry.py). Generalized to a configurable number of energy groups G
(physics.n_groups); at G=2 the result is identical to the original two-group
assembly, so existing Natrium data is unchanged.

Continuous problem (2D, G groups, DOWN-SCATTER ONLY):

    -div(D_g grad phi_g) + Sigma_r_g phi_g - sum_{g'<g} Ss_{g'->g} phi_{g'}
        = (1/k) chi_g sum_{g'} nuSf_{g'} phi_{g'}          for g = 0..G-1

Galerkin P1 weak form on triangles gives, per group g:
    K_g   = element STIFFNESS   sum_e integral D_g grad(phi).grad(psi)
    M(c)  = LUMPED MASS         diag(c_i * V_i),  V_i = sum_e Ae/3 (nodal volume)
Vacuum BC (Marshak partial current): a Robin term is added to the K_g diagonal on
each boundary edge, + alpha * L / 2 per endpoint (alpha=0.5, L=edge length). No D
factor (the boundary integral is integral 0.5 phi psi ds).

Block structure (each block N x N; group-major [g0(N), g1(N), ..., g_{G-1}(N)]):
    A[g][g]  = K_g + M(Sr_g) + Robin
    A[gt][gf] = -M(Ss_{gf->gt})   for each down-scatter pair gf < gt   (in-scatter source)
    F[g][g'] = M(chi_g * nuSf_g')
Assembled as  A phi = (1/k) F phi,  phi group-major -> shape [GN, GN].

XS row layout per node is [D(G), Sr(G), downscatter(G(G-1)/2), nuSf(G)] (xs_common).

P1 element stiffness (vertices p0,p1,p2, area Ae):
    b0=y1-y2, b1=y2-y0, b2=y0-y1 ;  c0=x2-x1, c1=x0-x2, c2=x1-x0
    Ke_ij = D_e * (b_i b_j + c_i c_j) / (4 Ae)
D_e per group is the arithmetic mean of the triangle's 3 vertex D_g.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from datagen_config import PhysicsConfig
from geometry import CoreGeometry
from xs_common import (d_slice, sr_slice, scatter_slice, nusf_slice,
                       scatter_pairs, n_xs_cols)


def assemble_AF(geom: CoreGeometry, physics: PhysicsConfig
                ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (A, F) as CSR matrices of shape [GN, GN] (P1 FEM, G=physics.n_groups)."""
    coords = geom.coordinates
    el = geom.elements                         # [T, 3]
    N = geom.n_nodes
    xs = geom.cross_sections
    V = geom.nodal_volume                       # [N] lumped nodal volumes
    T = el.shape[0]
    G = physics.n_groups
    assert xs.shape[1] == n_xs_cols(G), (
        f"cross_sections width {xs.shape[1]} != {n_xs_cols(G)} expected for G={G}")

    # per-element geometry
    x = coords[el, 0]                           # [T, 3]
    y = coords[el, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)  # [T,3]
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)  # [T,3]
    signed2 = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
               - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    area = 0.5 * np.abs(signed2)                # [T] positive area
    area = np.maximum(area, 1e-12)

    # local 3x3 gradient dot products (orientation-independent): b_i b_j + c_i c_j
    grad = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]  # [T,3,3]
    I = np.broadcast_to(el[:, :, None], (T, 3, 3)).ravel()
    J = np.broadcast_to(el[:, None, :], (T, 3, 3)).ravel()

    def stiffness(Dg: np.ndarray) -> sp.csr_matrix:
        De = Dg[el].mean(axis=1)                # [T] element diffusion (vertex mean)
        Ke = (De / (4.0 * area))[:, None, None] * grad
        return sp.coo_matrix((Ke.ravel(), (I, J)), shape=(N, N)).tocsr()

    # Marshak partial-current Robin term on boundary edges (same for every group
    # diagonal).
    #
    # The coefficient is PER EDGE, so a boundary condition other than pure vacuum
    # can be expressed. For an albedo beta (the fraction of the outgoing partial
    # current returned to the core, J- = beta J+):
    #
    #     alpha = (1 - beta) / (2 (1 + beta))
    #
    # beta = 0 gives alpha = 0.5, exactly the vacuum value
    # (VACUUM_ROBIN_ALPHA), so every previously generated sample is reproduced
    # bit-for-bit. beta -> 1 gives alpha -> 0, a reflective boundary.
    #
    # geom.boundary_albedo, when present, is a per-edge alpha array [B]; absent,
    # the scalar physics.vacuum_robin_alpha applies to every edge as before.
    Rd = sp.csr_matrix((N, N))
    be = geom.boundary_edges
    if be.size:
        L = np.hypot(coords[be[:, 0], 0] - coords[be[:, 1], 0],
                     coords[be[:, 0], 1] - coords[be[:, 1], 1])
        alpha = getattr(geom, "boundary_alpha", None)
        if alpha is None:
            alpha = physics.vacuum_robin_alpha
        else:
            alpha = np.asarray(alpha, dtype=float)
            if alpha.shape != (be.shape[0],):
                raise ValueError(
                    f"boundary_alpha has shape {alpha.shape}, expected one value "
                    f"per boundary edge ({be.shape[0]},)")
        contrib = alpha * L / 2.0
        robin = np.zeros(N)
        np.add.at(robin, be[:, 0], contrib)
        np.add.at(robin, be[:, 1], contrib)
        Rd = sp.diags(robin).tocsr()

    D = xs[:, d_slice(G)]                        # [N, G]
    Sr = xs[:, sr_slice(G)]                      # [N, G]
    Ssc = xs[:, scatter_slice(G)]               # [N, G(G-1)/2] down-scatter
    nuSf = xs[:, nusf_slice(G)]                  # [N, G]
    chi = np.asarray(physics.chi, dtype=float)   # [G]
    assert chi.shape[0] == G, f"chi length {chi.shape[0]} != n_groups {G}"

    # loss operator A: diagonal blocks (stiffness + removal + Robin), off-diagonal
    # blocks are the down-scatter in-scatter sources.
    # transverse (axial) leakage: a 2D radial model ignores axial leakage, so add
    # D_g*Bz^2 to each group's removal (Bz^2 = physics.axial_buckling). Bz^2=0 -> the
    # original pure-2D operator, so existing zero-buckling data is unchanged.
    Bz2 = float(getattr(physics, "axial_buckling", 0.0))
    Ablk = [[None] * G for _ in range(G)]
    for g in range(G):
        removal = (Sr[:, g] + Bz2 * D[:, g]) * V         # absorption/out-scatter + axial leak
        Ablk[g][g] = stiffness(D[:, g]) + Rd + sp.diags(removal)
    for idx, (gf, gt) in enumerate(scatter_pairs(G)):
        # neutrons scattering DOWN from gf appear as a source in group gt
        Ablk[gt][gf] = -sp.diags(Ssc[:, idx] * V)
    # optional thermal UP-scatter (FHR, operator-level): Ss_{g2->g1}, added as an
    # in-scatter SOURCE in the fast-group equation. G=2 only (the up-scatter data is
    # two-group); None (hex) -> unchanged.
    #
    # The thermal-group REMOVAL is deliberately not touched here. xs_openmc builds
    # Sigma_r = Sigma_a + ALL out-scatter, down AND up (xs_openmc._collapse:
    # `out_scatter = Smat.sum(axis=1) - diag(Smat)`), so Sr[thermal] already carries
    # Ss21 as out-scatter. Adding it again double-counts thermal loss -- and it is not
    # a small error in this reactor: for graphite_pebble Ss21 is ~93% of thermal
    # absorption, so the thermal removal came out ~1.9x too large and the fhr core was
    # subcritical at every state (k <= 0.843 rods-out) against a CE transport k of
    # 1.335 at the same branch, a ~21,000 pcm bias.
    Ssup = getattr(geom, "upscatter", None)
    if Ssup is not None and np.any(Ssup):
        if G != 2:
            raise ValueError("operator-level up-scatter is implemented for G=2 only")
        up = np.asarray(Ssup, dtype=float) * V
        src = -sp.diags(up)                                 # in-scatter source into fast group
        Ablk[0][1] = src if Ablk[0][1] is None else (Ablk[0][1] + src)
    for i in range(G):
        for j in range(G):
            if Ablk[i][j] is None:
                Ablk[i][j] = sp.csr_matrix((N, N))
    A = sp.bmat(Ablk, format="csr")

    # fission operator F: F[g][g'] = diag(chi_g * nuSf_g' * V) (lumped mass)
    Fblk = [[sp.diags(chi[g] * nuSf[:, gp] * V) for gp in range(G)] for g in range(G)]
    F = sp.bmat(Fblk, format="csr")

    return A.tocsr(), F.tocsr()
