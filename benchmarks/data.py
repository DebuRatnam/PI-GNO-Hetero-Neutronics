"""Dataset, loader, and normalization fitting for the benchmark suite.

Two problems with the src/train.py data path that this module fixes:

1. RE-DECODING EVERY EPOCH. `dataset.load_sample` decompresses an npz and then
   re-`eval`s a `repr`-ed metadata dict (data_generation/p1_fem/dataset.py:183)
   on every access. src/train.py does that inside the epoch loop, so the same
   5000 samples are decoded hundreds of times. `build_cache` writes each sample
   once as a flat .pt of tensors; after that a load is an mmap-friendly read.

2. FITTING NORMALIZATION BY CONCATENATION. `train.py:fit_norm_on_train` cats
   every train sample's node and edge features into one tensor. For fhr01 the
   edge block alone is 5000 x 80k x 8 x 4 B ~ 13 GB of RAM. `fit_normalization_
   streaming` accumulates count/sum/sumsq instead and reproduces the same
   statistics (unbiased std, per-group flux RMS) in constant memory.

Cache size, measured (float32 values, int32 indices): 2.4 MB/sample for hex01 and
5.3 MB/sample for fhr01, i.e. ~17 GB and ~37 GB for the full 7000-sample sets,
~54 GB for both. That is bigger than the npz for hex (1.1 MB) because npz is
compressed, and about the same for fhr (5.4 MB) because float64->float32 offsets
the lost compression. Check free space before caching fhr.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

import _paths  # noqa: F401
from dataio import Sample, list_split
from dataset import load_sample
from features import (NormBundle, Standardizer, FluxScaler, NodeLayout,
                      EDGE_PASSTHROUGH_COLS)

from batching import collate_graphs

CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# on-disk tensor cache
# --------------------------------------------------------------------------- #

def _cache_path(npz_path: str, cache_root: str, data_root: str) -> str:
    rel = os.path.relpath(npz_path, data_root)
    return os.path.join(cache_root, os.path.splitext(rel)[0] + ".pt")


def _encode(npz_path: str, dtype=torch.float32) -> dict:
    """npz -> a flat dict of tensors. Sparse operators are stored as their COO
    triplets: torch.save of a sparse tensor is fragile across versions, and the
    triplets are what we need to rebuild them anyway.

    edge_index is stored int32 (node counts are ~1e4, far under 2^31) and cast
    back to long on load -- it halves the largest tensor in the cache."""
    s = load_sample(npz_path)
    A, F = s["A"].tocoo(), s["F"].tocoo()
    import numpy as np
    t = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt)
    return {
        "version": CACHE_VERSION,
        "node_feats": t(s["node_features"]),
        "edge_index": t(s["edge_index"], torch.int32),
        "edge_feats": t(s["edge_features"]),
        "flux": t(s["flux"]),
        "k_eff": float(s["k_eff"]),
        "power": t(s["power_density"]),
        "boundary_mask": t(s["boundary_mask"], torch.bool),
        "coords": t(s["coordinates"]),
        "nodal_volume": t(s["nodal_volume"]),
        "material_state": t(s["material_state"], torch.int32),
        "A_idx": t(np.stack([A.row, A.col]), torch.int32),
        "A_val": t(A.data),
        "F_idx": t(np.stack([F.row, F.col]), torch.int32),
        "F_val": t(F.data),
        "gn": int(A.shape[0]),
        "meta": s["geometry_metadata"],
    }


def _decode(d: dict, device="cpu", dtype=torch.float32) -> Sample:
    gn = d["gn"]
    mk = lambda idx, val: torch.sparse_coo_tensor(
        idx.to(torch.long), val.to(dtype), size=(gn, gn)).coalesce()
    return Sample(
        node_feats=d["node_feats"].to(dtype),
        edge_index=d["edge_index"].to(torch.long),
        edge_feats=d["edge_feats"].to(dtype),
        flux=d["flux"].to(dtype),
        k_eff=torch.tensor(float(d["k_eff"]), dtype=dtype),
        power=d["power"].to(dtype),
        boundary_mask=d["boundary_mask"],
        A=mk(d["A_idx"], d["A_val"]),
        F=mk(d["F_idx"], d["F_val"]),
        coords=d["coords"].to(dtype),
        nodal_volume=d["nodal_volume"].to(dtype),
        material_state=d["material_state"].to(torch.long),
        meta=d["meta"],
    )


def build_cache(data_root: str, splits=("train", "val", "test"),
                cache_root: Optional[str] = None, verbose=True) -> str:
    """Convert every .npz under `data_root` to a .pt tensor cache. Idempotent:
    existing up-to-date entries are skipped, so this is safe to re-run."""
    cache_root = cache_root or os.path.join(data_root, "_cache")
    total = done = 0
    for split in splits:
        paths = list_split(data_root, split)
        total += len(paths)
        for p in paths:
            out = _cache_path(p, cache_root, data_root)
            if os.path.exists(out):
                done += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save(_encode(p), out)
            done += 1
            if verbose and done % 250 == 0:
                print(f"  cached {done}/{total}", flush=True)
    if verbose:
        print(f"cache ready: {cache_root} ({done} samples)")
    return cache_root


# --------------------------------------------------------------------------- #
# dataset / loader
# --------------------------------------------------------------------------- #

class GraphDataset(Dataset):
    """Cores as graphs. Reads the .pt cache when present, else the raw .npz.

    Falling back rather than failing keeps the smoke tests runnable on a fresh
    checkout with no cache built.
    """

    def __init__(self, data_root: str, split: str,
                 cache_root: Optional[str] = None,
                 limit: Optional[int] = None, dtype=torch.float32):
        self.data_root = data_root
        self.split = split
        self.dtype = dtype
        self.paths: List[str] = list_split(data_root, split)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no samples in {data_root}/{split}")
        self.cache_root = cache_root or os.path.join(data_root, "_cache")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Sample:
        p = self.paths[i]
        c = _cache_path(p, self.cache_root, self.data_root)
        if os.path.exists(c):
            return _decode(torch.load(c, weights_only=False), dtype=self.dtype)
        from dataio import load_torch_sample
        return load_torch_sample(p, dtype=self.dtype)

    def metadata(self) -> dict:
        return self[0].meta

    def layout(self) -> NodeLayout:
        return NodeLayout.from_metadata(self.metadata())


def make_loader(ds: GraphDataset, batch_size: int = 4, shuffle: bool = False,
                num_workers: int = 0, seed: Optional[int] = None) -> DataLoader:
    """DataLoader emitting BatchedSample.

    pin_memory stays off: the batch carries sparse tensors, which do not pin.
    The batch is moved to the device in the training loop instead.
    """
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_graphs,
                      pin_memory=False, generator=g,
                      drop_last=False)


# --------------------------------------------------------------------------- #
# streaming normalization
# --------------------------------------------------------------------------- #

class _Moments:
    """Constant-memory count / sum / sum-of-squares accumulator, column-wise."""

    def __init__(self):
        self.n = 0
        self.s = None
        self.ss = None

    def update(self, x: torch.Tensor):
        x = x.double()
        if self.s is None:
            self.s = torch.zeros(x.shape[1], dtype=torch.float64)
            self.ss = torch.zeros(x.shape[1], dtype=torch.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)

    def mean_std(self, eps=1e-8):
        mean = self.s / self.n
        # unbiased (n-1) variance, matching Standardizer.fit's tensor.std(0)
        var = (self.ss - self.n * mean * mean) / max(self.n - 1, 1)
        std = var.clamp_min(0).sqrt().clamp_min(eps)
        return mean.float(), std.float()


def fit_normalization_streaming(ds: GraphDataset, *, node_passthrough=None,
                                num_workers: int = 0, verbose=True
                                ) -> NormBundle:
    """Fit node/edge standardizers, per-group flux RMS, and k stats on a split
    WITHOUT concatenating it into memory.

    Reproduces features.fit_normalization exactly (unbiased std; passthrough
    columns forced to mean 0 / std 1) but in O(n_features) memory instead of
    O(dataset). Fit on TRAIN ONLY -- the caller is responsible for passing the
    train split, and the harness persists the result so val/test use these same
    numbers.
    """
    node_m, edge_m = _Moments(), _Moments()
    flux_sq = None
    flux_n = 0
    ks: List[float] = []

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, collate_fn=lambda b: b[0])
    for i, s in enumerate(loader):
        node_m.update(s.node_feats)
        edge_m.update(s.edge_feats)
        f = s.flux.double()
        flux_sq = (f * f).sum(0) if flux_sq is None else flux_sq + (f * f).sum(0)
        flux_n += f.shape[0]
        ks.append(float(s.k_eff))
        if verbose and (i + 1) % 500 == 0:
            print(f"  norm fit {i + 1}/{len(ds)}", flush=True)

    nmean, nstd = node_m.mean_std()
    emean, estd = edge_m.mean_std()

    pt = node_passthrough if node_passthrough is not None else ds.layout().passthrough_cols
    for mean, std, cols in ((nmean, nstd, pt),
                            (emean, estd, EDGE_PASSTHROUGH_COLS)):
        if cols:
            idx = torch.tensor(list(cols), dtype=torch.long)
            mean[idx] = 0.0
            std[idx] = 1.0

    k = torch.tensor(ks, dtype=torch.float32)
    return NormBundle(
        node=Standardizer(mean=nmean, std=nstd),
        edge=Standardizer(mean=emean, std=estd),
        flux=FluxScaler(scale=(flux_sq / flux_n).sqrt().float().clamp_min(1e-8)),
        k_mean=float(k.mean()), k_std=float(k.std().clamp_min(1e-8)),
    )


def save_norm(norm: NormBundle, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(norm.to_dict(), f, indent=2)


def load_norm(path: str) -> NormBundle:
    with open(path) as f:
        return NormBundle.from_dict(json.load(f))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build the .pt tensor cache")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cache-root", default=None)
    a = ap.parse_args()
    build_cache(a.data, tuple(a.splits), a.cache_root)
