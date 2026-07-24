"""Feature ordering + normalization for PI-GNO.

Normalization statistics are fit ONLY on the training split and applied
identically to val/test (CLAUDE requirement). Inverse transforms are retained so
predictions and the PDE residual can be reported/computed in physical units.

The PDE residual MUST use physical (un-normalized) flux because A and F are in
physical units. So the model predicts normalized flux for stability, and
physics.py de-normalizes before forming R = A phi - (1/k) F phi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# The node feature layout is SCHEMA-DRIVEN so the same model code handles both
# reactors (Natrium hex: 8 materials, G=2 -> 18 dims; KP-FHR pebble: 7 materials,
# G=2 -> 17 dims) and any group count G. Layout:
#   [x, y, <one-hot material (n_materials)>, <XS block (n_xs_cols(G))>, boundary_flag]
# XS block order (see data generation/xs_common): D(G), Sr(G), downscatter(G(G-1)/2),
# nuSf(G). Build a NodeLayout from a dataset's geometry_metadata (n_materials,
# n_groups); the module-level constants below are the Natrium hex default so
# existing code keeps working unchanged.

def _n_xs_cols(G: int) -> int:
    return 3 * G + G * (G - 1) // 2


@dataclass(frozen=True)
class NodeLayout:
    """Column indices for a node feature tensor, derived from the material count
    and group count (no hardcoding). Use NodeLayout.from_metadata(meta)."""
    n_materials: int
    n_groups: int

    @classmethod
    def from_metadata(cls, meta: dict) -> "NodeLayout":
        return cls(int(meta["n_materials"]), int(meta["n_groups"]))

    @property
    def xs_width(self) -> int:
        return _n_xs_cols(self.n_groups)

    @property
    def total_dim(self) -> int:
        return 2 + self.n_materials + self.xs_width + 1

    @property
    def material_onehot_cols(self) -> list:
        return list(range(2, 2 + self.n_materials))

    @property
    def xs_start(self) -> int:
        return 2 + self.n_materials

    @property
    def d_cols(self) -> list:
        return list(range(self.xs_start, self.xs_start + self.n_groups))

    @property
    def nusf_cols(self) -> list:
        # nuSf is the LAST G columns of the XS block.
        end = self.xs_start + self.xs_width
        return list(range(end - self.n_groups, end))

    @property
    def boundary_col(self) -> int:
        return self.xs_start + self.xs_width

    @property
    def passthrough_cols(self) -> list:
        # binary one-hot material block + binary boundary flag: leave un-normalized.
        return self.material_onehot_cols + [self.boundary_col]


# Natrium hex default layout (8 materials, 2 groups -> 18 dims). Kept as
# module-level constants for backward compatibility; new code should build a
# NodeLayout from the dataset metadata instead.
DEFAULT_LAYOUT = NodeLayout(n_materials=8, n_groups=2)

NODE_COLS = {
    "x": 0, "y": 1,
    "mat_fuel_inner": 2, "mat_fuel_outer": 3, "mat_primary_control": 4,
    "mat_secondary_control": 5, "mat_reflector": 6, "mat_shield": 7,
    "mat_duct": 8, "mat_coolant": 9,
    "D1": 10, "D2": 11,
    "Sigma_r1": 12, "Sigma_r2": 13, "Sigma_s12": 14,
    "nuSigma_f1": 15, "nuSigma_f2": 16, "boundary_flag": 17,
}

# One-hot material block columns (argmax -> material id for reporting/breakdowns).
MATERIAL_ONEHOT_COLS = DEFAULT_LAYOUT.material_onehot_cols

# Columns left UN-normalized: the binary one-hot material block and the binary
# boundary flag. Z-scoring binary indicators distorts them and gains nothing.
PASSTHROUGH_COLS = DEFAULT_LAYOUT.passthrough_cols

# Edge feature columns (8 dims; see graph_build.py / datagen_config edge_feature_order):
# [0]distance [1]dx [2]dy [3]interface_flag [4]harmonic_D1 [5]dD1 [6]dSigma_r1 [7]dSigma_s12
# The interface_flag (col 3) is binary -> leave it un-normalized like the node
# passthrough columns; the rest are continuous and get z-scored.
EDGE_INTERFACE_FLAG_COL = 3
EDGE_PASSTHROUGH_COLS = [EDGE_INTERFACE_FLAG_COL]


@dataclass
class Standardizer:
    """Affine (z-score) normalizer with retained inverse. Fit on train only.

    `passthrough` column indices are left identity (mean 0, std 1) so binary /
    one-hot features are not distorted by z-scoring."""
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-8,
            passthrough=None) -> "Standardizer":
        mean = x.mean(0)
        std = x.std(0).clamp_min(eps)
        if passthrough:
            idx = torch.tensor(passthrough, dtype=torch.long)
            mean = mean.clone(); std = std.clone()
            mean[idx] = 0.0
            std[idx] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x) + self.mean.to(x)


@dataclass
class FluxScaler:
    """Per-group flux scaling (scalar magnitude per group). Eigenvector flux is
    arbitrary-scale, so we normalize by train-set RMS per group and keep the
    factor to recover physical magnitude for the PDE residual."""
    scale: torch.Tensor   # [n_groups]

    @classmethod
    def fit(cls, flux: torch.Tensor, eps: float = 1e-8) -> "FluxScaler":
        # flux: [N, n_groups]
        rms = flux.pow(2).mean(0).sqrt().clamp_min(eps)
        return cls(scale=rms)

    def transform(self, flux: torch.Tensor) -> torch.Tensor:
        return flux / self.scale.to(flux)

    def inverse(self, flux: torch.Tensor) -> torch.Tensor:
        return flux * self.scale.to(flux)


@dataclass
class NormBundle:
    node: Standardizer
    edge: Standardizer
    flux: FluxScaler
    k_mean: float
    k_std: float

    def to_dict(self) -> dict:
        return {
            "node_mean": self.node.mean.tolist(), "node_std": self.node.std.tolist(),
            "edge_mean": self.edge.mean.tolist(), "edge_std": self.edge.std.tolist(),
            "flux_scale": self.flux.scale.tolist(),
            "k_mean": self.k_mean, "k_std": self.k_std,
        }


def fit_normalization(node_feats, edge_feats, flux, k_values,
                      node_passthrough=None) -> NormBundle:
    """Fit all stats on concatenated TRAIN tensors. Inputs are torch tensors.

    `node_passthrough` is the list of node-feature columns left un-normalized
    (one-hot material + boundary flag); defaults to the Natrium hex block. Pass
    `NodeLayout.from_metadata(meta).passthrough_cols` for a KP-FHR / multigroup
    dataset so the right columns are protected."""
    k = torch.as_tensor(k_values, dtype=torch.float32)
    node_pt = PASSTHROUGH_COLS if node_passthrough is None else node_passthrough
    return NormBundle(
        node=Standardizer.fit(node_feats, passthrough=node_pt),
        edge=Standardizer.fit(edge_feats, passthrough=EDGE_PASSTHROUGH_COLS),
        flux=FluxScaler.fit(flux),
        k_mean=float(k.mean()), k_std=float(k.std().clamp_min(1e-8)),
    )
