"""Derived node-wise power density from the multigroup flux.

power_density(x) = E_f * sum_g nuSigma_f_g * phi_g / nu_bar

We use nuSigma_f as the available fission-related cross section (Sigma_f is not
tracked separately in this dataset). The result is proportional to fission rate
density; E_f folds energy-per-fission. The SAME relation is reused by the model's
power head (heads.py) so labels and predictions are consistent. nu_bar cancels in
relative comparisons and is set to 1 (documented in metadata).

Only fuel-bearing cells (nuSf > 0) produce power; everything else is ~0.
"""

from __future__ import annotations

import numpy as np

from datagen_config import PhysicsConfig
from xs_common import nusf_slice


NU_BAR = 1.0  # documented constant; folded so power ∝ fission energy deposition


def power_density(flux: np.ndarray, cross_sections: np.ndarray,
                  physics: PhysicsConfig) -> np.ndarray:
    """flux [N,G], cross_sections [N, n_xs_cols(G)] -> power_density [N]."""
    G = physics.n_groups
    nuSf = cross_sections[:, nusf_slice(G)]     # [N, G]
    fission_rate = (nuSf * flux).sum(axis=1)    # sum_g nuSf_g * phi_g
    return physics.energy_per_fission_j * fission_rate / NU_BAR
