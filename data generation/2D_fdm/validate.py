"""Pre-training validation checks on a generated sample.

Verifies the dataset contract: shapes, finite values, operator dimensions,
non-empty boundary mask, FEM mesh validity (triangles + boundary edges + positive
nodal volumes), group-major consistency, and a small PDE residual for the
reference solution. N is the per-sample node count (irregular FEM mesh). Raises
AssertionError on violation so bad samples never reach training.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xs_common import n_xs_cols

# Canonical edge feature width: [distance, dx, dy, interface_flag, harmonic_D1,
# dD1, dSigma_r1, dSigma_s12] (message-graph edges; see graph_build.py).
EXPECTED_EDGE_FEATURE_DIM = 8


def validate_sample(sample: dict, *, residual_tol: float = 1e-6) -> None:
    N = sample["material_state"].shape[0]
    C = sample["cross_sections"].shape[1]
    meta = sample["geometry_metadata"]
    # schema is self-describing: derive expected widths from the sample metadata so
    # both Natrium (hex, fast) and KP-FHR (pebble, thermal) samples validate.
    reactor = str(meta.get("reactor_type", "hex"))
    G = int(meta["n_groups"])
    n_mat = int(meta["n_materials"])
    assert C == n_xs_cols(G), f"cross_sections width {C} != n_xs_cols({G})={n_xs_cols(G)}"
    expected_node_dim = 2 + n_mat + n_xs_cols(G) + 1

    # shapes
    assert sample["coordinates"].shape == (N, 2)
    assert sample["cross_sections"].shape == (N, C)
    assert sample["flux"].shape == (N, G), f"flux {sample['flux'].shape}, expected {(N, G)}"
    # node features: [x,y] + one-hot material (n_mat) + G-group XS + boundary flag
    nf = sample["node_features"]
    assert nf.shape == (N, expected_node_dim), (
        f"node_features {nf.shape}, expected {(N, expected_node_dim)}")
    onehot = nf[:, 2:2 + n_mat]
    assert np.all(onehot.sum(axis=1) == 1), "material one-hot rows must sum to 1"
    assert sample["power_density"].shape == (N,)
    assert sample["boundary_mask"].shape == (N,)
    assert sample["edge_index"].shape[0] == 2
    assert sample["edge_features"].shape[0] == sample["edge_index"].shape[1]
    assert sample["edge_features"].shape[1] == EXPECTED_EDGE_FEATURE_DIM, (
        f"edge_features width {sample['edge_features'].shape[1]}, "
        f"expected {EXPECTED_EDGE_FEATURE_DIM}")

    # finiteness
    for key in ("coordinates", "cross_sections", "flux", "power_density"):
        assert np.all(np.isfinite(sample[key])), f"non-finite values in {key}"

    # operators [GN, GN]
    A, F = sample["A"], sample["F"]
    assert A.shape == (G * N, G * N), f"A is {A.shape}, expected {(G*N, G*N)}"
    assert F.shape == (G * N, G * N), f"F is {F.shape}, expected {(G*N, G*N)}"
    assert sp.issparse(A) and sp.issparse(F)

    # boundary mask non-empty and within range
    assert sample["boundary_mask"].any(), "empty boundary mask"

    # FEM mesh validity
    el = sample["elements"]
    assert el.ndim == 2 and el.shape[1] == 3 and el.shape[0] > 0, f"elements {el.shape}"
    assert el.min() >= 0 and el.max() < N, "element node index out of range"
    be = sample["boundary_edges"]
    assert be.ndim == 2 and be.shape[1] == 2 and be.shape[0] > 0, f"boundary_edges {be.shape}"
    assert be.min() >= 0 and be.max() < N, "boundary edge index out of range"
    vol = sample["nodal_volume"]
    assert vol.shape == (N,) and np.all(vol > 0), "nodal volumes must be positive"
    # lumped volumes must tile the meshed core area (partition of unity)
    core_area = float(meta["core_area_cm2"])
    assert abs(float(vol.sum()) - core_area) < 1e-3 * core_area, \
        f"sum(nodal_volume)={vol.sum():.1f} != core area {core_area:.1f}"
    # reactor-specific material sanity: the defining structural/fuel materials must
    # be explicitly present in the mesh (not lumped away).
    counts = meta["material_counts"]
    if reactor == "fhr":
        assert counts.get("fuel_pebble", 0) > 0, "fuel_pebble material missing from the mesh"
        assert counts.get("coolant", 0) > 0, "FLiBe coolant material missing from the mesh"
    else:
        assert counts.get("duct", 0) > 0, "duct (HT9) material missing from the mesh"

    # edge indices in range, no self loops
    ei = sample["edge_index"]
    assert ei.min() >= 0 and ei.max() < N, "edge index out of range"
    assert np.all(ei[0] != ei[1]), "self-loops present in message graph"

    # residual of the reference solution (group-major flux vector)
    phi = sample["flux"].T.reshape(-1)          # [g0(N), g1(N), ...]
    k = float(sample["k_eff"])
    Fphi = F @ phi
    res = np.linalg.norm(A @ phi - (1.0 / k) * Fphi) / max(np.linalg.norm(Fphi), 1e-30)
    assert res < residual_tol, f"reference residual {res:.2e} exceeds tol {residual_tol:.1e}"

    # physical sanity
    assert 0.1 < k < 3.0, f"k_eff={k} outside sane range"
