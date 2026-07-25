"""Load generated .npz samples into torch tensors for training.

Bridges data_generation (numpy/scipy) -> model (torch). Builds the torch sparse
A/F once per sample and caches the canonical node feature tensor. No training
here.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import torch

from physics import scipy_csr_to_torch

# import the loader from the data-generation package
import sys
_DG = os.path.join(os.path.dirname(__file__), "..", "data_generation", "p1_fem")
sys.path.insert(0, os.path.abspath(_DG))
from dataset import load_sample  # noqa: E402


@dataclass
class Sample:
    node_feats: torch.Tensor     # [N, 18] raw (physical; 8-way one-hot material)
    edge_index: torch.Tensor     # [2, E] long (kNN message graph)
    edge_feats: torch.Tensor     # [E, 8] raw
    flux: torch.Tensor           # [N, 2] physical reference
    k_eff: torch.Tensor          # scalar
    power: torch.Tensor          # [N] reference
    boundary_mask: torch.Tensor  # [N] bool
    A: torch.Tensor              # sparse [2N, 2N]
    F: torch.Tensor              # sparse [2N, 2N]


def load_torch_sample(path: str, device="cpu", dtype=torch.float32) -> Sample:
    s = load_sample(path)
    t = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=device)
    return Sample(
        node_feats=t(s["node_features"]),
        edge_index=t(s["edge_index"], torch.long),
        edge_feats=t(s["edge_features"]),
        flux=t(s["flux"]),
        k_eff=t(float(s["k_eff"])),
        power=t(s["power_density"]),
        boundary_mask=t(s["boundary_mask"], torch.bool),
        A=scipy_csr_to_torch(s["A"], device, dtype),
        F=scipy_csr_to_torch(s["F"], device, dtype),
    )


def list_split(root: str, split: str):
    return sorted(glob.glob(os.path.join(root, split, "*.npz")))
