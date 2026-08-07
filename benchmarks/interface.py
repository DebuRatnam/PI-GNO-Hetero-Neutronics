"""The contract every benchmarked model obeys.

The comparison is only meaningful if the models differ ONLY in architecture.
Everything downstream of the architecture is therefore fixed here and shared:

  * predictions are reported AT MESH NODES, always. FNO predicts on a Cartesian
    grid and must interpolate back before it returns, so its rasterization error
    is counted against it rather than hidden by evaluating on the grid.
  * flux de-normalization uses the SAME NormBundle fitted on the SAME train
    split, and k de-normalization the same k_mean/k_std.
  * power comes from the SAME parameter-free heads.PowerHead every model shares,
    so power error reflects flux error alone and no model can win power with a
    free head that quietly learns the labels.
  * outputs are PIGNOOutput, so losses.compute_loss and metrics.sample_metrics
    need no per-model branch.

A subclass implements ONE method, `predict_norm`, returning normalized flux and
normalized k. Everything else is inherited.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

import _paths  # noqa: F401
from datagen_config import ENERGY_PER_FISSION_J
from features import NodeLayout, NormBundle
from heads import PowerHead
from model import PIGNOOutput

from batching import BatchedSample


class BenchModel(nn.Module):
    """Base class for every model in the benchmark.

    Subclasses implement:
        predict_norm(batch, norm) -> (flux_norm [sumN, G], k_norm [n_graphs])

    `k_norm` must be per graph. Returning one k for a whole batch would train
    (the MSE still decreases) while making k_eff meaningless, which is the exact
    failure mode benchmarks/test_batching.py exists to catch.
    """

    def __init__(self, layout: NodeLayout,
                 energy_per_fission_j: float = ENERGY_PER_FISSION_J):
        super().__init__()
        self.layout = layout
        # nuSf column indices are schema-driven (material count + group count),
        # never hardcoded -- hex is 8 materials, fhr is 7.
        self.power_head = PowerHead(energy_per_fission_j, layout.nusf_cols)

    # -- to implement -------------------------------------------------------
    def predict_norm(self, batch: BatchedSample, norm: NormBundle
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------
    def forward(self, batch: BatchedSample, norm: NormBundle) -> PIGNOOutput:
        flux_norm, k_norm = self.predict_norm(batch, norm)
        flux_phys = norm.flux.inverse(flux_norm)
        k_phys = k_norm * norm.k_std + norm.k_mean
        power = self.power_head(flux_phys, batch.node_feats)
        return PIGNOOutput(flux_norm=flux_norm, k_norm=k_norm,
                           flux_phys=flux_phys, k_phys=k_phys, power=power)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict:
        return {"class": type(self).__name__, "n_params": self.n_params()}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Register a model builder: build(meta, layout, **hparams) -> BenchModel."""
    def deco(fn):
        if name in _REGISTRY:
            raise KeyError(f"model '{name}' registered twice")
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(name: str, meta: dict, **hparams) -> BenchModel:
    if name not in _REGISTRY:
        _import_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; have {sorted(_REGISTRY)}")
    layout = NodeLayout.from_metadata(meta)
    return _REGISTRY[name](meta, layout, **hparams)


def available() -> list:
    _import_builtins()
    return sorted(_REGISTRY)


def _import_builtins():
    """Import model modules on demand.

    Deliberately tolerant: the FNO and MeshGraphNet wrappers need PhysicsNeMo,
    which requires python 3.11-3.13 and cannot be installed in the local 3.9
    training env. A missing PhysicsNeMo must not stop PI-GNO or DeepONet from
    running -- it should only make those two names unavailable, with the reason
    reported when they are actually requested.
    """
    import importlib
    for mod in ("models.pigno_wrap", "models.deeponet",
                "models.mgn_wrap", "models.fno_wrap"):
        try:
            importlib.import_module(mod)
        except ImportError as e:            # noqa: PERF203
            _MISSING[mod] = str(e)


_MISSING: Dict[str, str] = {}


def missing_reasons() -> Dict[str, str]:
    """Why a model module failed to import (usually: PhysicsNeMo not installed)."""
    return dict(_MISSING)
