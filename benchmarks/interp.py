"""Mesh-to-mesh interpolation and mesh-independent flux normalization.

Both are prerequisites for comparing anything across discretizations, and the
second one is a trap that silently invalidates the comparison if skipped.

THE SCALE TRAP. solver.solve_keff returns an eigenvector normalized to unit L2
norm over its GN entries, so reference flux MAGNITUDE is a pure artifact of node
count -- measured on one paired core, max(phi_1) runs 0.0375 / 0.0212 / 0.0147
at L1 / L2 / L3, which is exactly 1/sqrt(N). Comparing raw fluxes across meshes
would be dominated ~2.5x by eigensolver bookkeeping and would say nothing about
any model. `normalize_flux` divides by a physical integral instead, which
converges under refinement rather than tracking N.

INTERPOLATION uses the stored P1 elements, not a fresh triangulation. scipy's
LinearNDInterpolator would re-triangulate from scratch and fill the concavities
of the hex-lattice perimeter with elements that are not part of the domain,
inventing flux in the sodium gaps and outside the core. The FEM field is
piecewise linear on the elements the operator was assembled from, so those are
the elements to interpolate on.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

import _paths  # noqa: F401


def normalize_flux(flux, nodal_volume):
    """Scale a flux field so that integral(sum_g phi_g) dV = 1.

    ONE scalar for all groups, deliberately: the ratio between group fluxes is
    physical (it is the spectrum), so normalizing per group would destroy the
    very thing a multigroup comparison is about.
    """
    is_t = torch.is_tensor(flux)
    if is_t:
        scale = (flux.sum(dim=1) * nodal_volume).sum()
        return flux / scale.clamp_min(1e-30)
    scale = float((flux.sum(axis=1) * nodal_volume).sum())
    return flux / max(scale, 1e-30)


class P1Interpolator:
    """Barycentric interpolation on a stored P1 triangulation.

    Point location is a KD-tree over element centroids plus a barycentric
    containment test on the nearest candidates. Points outside the source mesh
    (the domains differ slightly between levels -- the Delaunay gap filter
    captures ~0.6% more area at L3 than at L1) fall back to the nearest node,
    and the fallback COUNT is reported rather than hidden, because a large
    fallback fraction would mean the two meshes are not really the same domain.
    """

    def __init__(self, coords: np.ndarray, elements: np.ndarray,
                 n_candidates: int = 24):
        self.coords = np.asarray(coords, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_candidates = n_candidates
        p = self.coords[self.elements]                       # [T,3,2]
        self.centroids = p.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.node_tree = cKDTree(self.coords)
        # cache the affine inverse per triangle for barycentric coordinates
        self.p0 = p[:, 0, :]
        v1 = p[:, 1, :] - p[:, 0, :]
        v2 = p[:, 2, :] - p[:, 0, :]
        det = v1[:, 0] * v2[:, 1] - v2[:, 0] * v1[:, 1]
        self.det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        self.v1, self.v2 = v1, v2

    def _bary(self, tri_idx: np.ndarray, pts: np.ndarray):
        d = pts - self.p0[tri_idx]
        v1, v2, det = self.v1[tri_idx], self.v2[tri_idx], self.det[tri_idx]
        b1 = (d[:, 0] * v2[:, 1] - v2[:, 0] * d[:, 1]) / det
        b2 = (v1[:, 0] * d[:, 1] - d[:, 0] * v1[:, 1]) / det
        return 1.0 - b1 - b2, b1, b2

    def __call__(self, values: np.ndarray, targets: np.ndarray,
                 tol: float = 1e-9) -> Tuple[np.ndarray, float]:
        """Interpolate `values` [N, C] onto `targets` [M, 2].
        Returns (interpolated [M, C], fallback_fraction)."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        targets = np.asarray(targets, dtype=float)
        M = targets.shape[0]
        out = np.empty((M, values.shape[1]), dtype=float)
        found = np.zeros(M, dtype=bool)

        k = min(self.n_candidates, len(self.centroids))
        _, cand = self.tree.query(targets, k=k)
        cand = np.atleast_2d(cand)
        for j in range(k):
            todo = ~found
            if not todo.any():
                break
            idx = np.where(todo)[0]
            t = cand[idx, j]
            b0, b1, b2 = self._bary(t, targets[idx])
            inside = (b0 >= -tol) & (b1 >= -tol) & (b2 >= -tol)
            if not inside.any():
                continue
            sel = idx[inside]
            tt = t[inside]
            w = np.stack([b0[inside], b1[inside], b2[inside]], axis=1)  # [n,3]
            nodes = self.elements[tt]                                    # [n,3]
            out[sel] = np.einsum("nk,nkc->nc", w, values[nodes])
            found[sel] = True

        n_fb = int((~found).sum())
        if n_fb:
            miss = np.where(~found)[0]
            _, nn = self.node_tree.query(targets[miss])
            out[miss] = values[nn]
        return out, n_fb / max(M, 1)


def interpolate_sample_to(src, dst, values: Optional[np.ndarray] = None):
    """Interpolate a field defined on `src`'s nodes onto `dst`'s node positions.

    `src` / `dst` are dataio.Sample-like (coords, elements, flux). Defaults to
    interpolating src.flux. Returns (values_on_dst [M, C], fallback_fraction).
    """
    def npy(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    if values is None:
        values = npy(src.flux)
    interp = P1Interpolator(npy(src.coords), npy(src.elements))
    return interp(npy(values), npy(dst.coords))
