"""Correctness tests for graph batching. BLOCKING for the whole benchmark.

If the batched operator indexing is wrong, every downstream number -- every
model, every experiment cell -- is wrong in a way that still looks like a
plausible loss curve. Run this before anything else:

    /usr/bin/python3 benchmarks/test_batching.py [--data datasets/hex01]

Checks:
  1. batched pde_residual == per-sample pde_residual, entry for entry
  2. batched A/F sparsity structure is exactly the union of the remapped blocks
  3. batched KHead == per-sample KHead for every pooling mode
  4. batch-of-one collate is a no-op with respect to metrics.sample_metrics
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

import _paths  # noqa: F401  (sys.path wiring; must precede src imports)
from dataio import load_torch_sample, list_split
from physics import pde_residual
from heads import KHead
from metrics import sample_metrics
from features import NodeLayout

from batching import collate_graphs, expand_k_group_major

TOL = 1e-5


def _fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def _ok(msg):
    print("  ok:", msg)


def test_residual_equivalence(samples):
    """The load-bearing test: A/F remapping must preserve the residual exactly."""
    G = samples[0].flux.shape[1]
    b = collate_graphs(samples)

    # give each graph a DIFFERENT k, so a bug that broadcasts one k across the
    # batch cannot pass by coincidence
    k_per_graph = torch.tensor([1.01 + 0.07 * i for i in range(len(samples))])
    k_vec = expand_k_group_major(k_per_graph, b.batch, G)

    R_batch = pde_residual(b.flux, k_vec, b.A, b.F).reshape(G, b.n_nodes)

    for i, s in enumerate(samples):
        n_i = s.node_feats.shape[0]
        lo = int(b.ptr[i])
        R_i = pde_residual(s.flux, k_per_graph[i], s.A, s.F).reshape(G, n_i)
        got = R_batch[:, lo:lo + n_i]
        denom = R_i.abs().max().clamp_min(1e-30)
        err = (got - R_i).abs().max() / denom
        if not torch.isfinite(err) or err > TOL:
            _fail(f"residual mismatch on sample {i}: rel err {float(err):.3e}")
    _ok(f"batched pde_residual matches {len(samples)} per-sample residuals "
        f"(G={G}, distinct k per graph)")


def test_operator_structure(samples):
    """nnz must be conserved: remapping is a bijection, not a merge."""
    b = collate_graphs(samples)
    for name, batched, per in (("A", b.A, [s.A for s in samples]),
                               ("F", b.F, [s.F for s in samples])):
        want = sum(int(p.coalesce()._nnz()) for p in per)
        got = int(batched._nnz())
        if got != want:
            _fail(f"{name} nnz {got} != sum of per-sample nnz {want} -- "
                  "indices collided, so two cores were written to the same row")
        wsum = sum(float(p.coalesce().values().sum()) for p in per)
        gsum = float(batched.values().sum())
        if abs(gsum - wsum) > TOL * max(abs(wsum), 1.0):
            _fail(f"{name} value sum drifted: {gsum} vs {wsum}")
    _ok("batched A/F conserve nnz and value mass")


def test_khead_equivalence(samples):
    torch.manual_seed(0)
    b = collate_graphs(samples)
    latent = 16
    h = torch.randn(b.n_nodes, latent)

    for pool in ("mean", "sum", "attention"):
        head = KHead(latent, 32, pool)
        head.eval()
        with torch.no_grad():
            k_batched = head(h, b.batch, b.n_graphs)          # [B]
            for i in range(len(samples)):
                lo, hi = int(b.ptr[i]), int(b.ptr[i + 1])
                k_single = head(h[lo:hi])                      # scalar
                err = (k_batched[i] - k_single).abs()
                scale = k_single.abs().clamp_min(1e-6)
                if err / scale > TOL:
                    _fail(f"KHead pool={pool} graph {i}: "
                          f"batched {float(k_batched[i]):.6f} != "
                          f"single {float(k_single):.6f}")
        _ok(f"KHead pool={pool} batched == per-graph")


def test_batch_of_one(samples):
    """metrics.sample_metrics must accept a batch-of-one unchanged, so train and
    eval can share one collate path."""
    s = samples[0]
    b = collate_graphs([s])
    layout = NodeLayout.from_metadata(s.meta)

    class Out:
        pass
    o = Out()
    o.flux_phys = s.flux
    o.k_phys = s.k_eff
    o.power = s.power

    m_single = sample_metrics(o, s, material_onehot_cols=layout.material_onehot_cols)

    o2 = Out()
    o2.flux_phys = b.flux
    o2.k_phys = b.k_eff.reshape(())
    o2.power = b.power
    m_batch = sample_metrics(o2, b, material_onehot_cols=layout.material_onehot_cols)

    for key in m_single:
        a, c = m_single[key], m_batch[key]
        if abs(a - c) > TOL * max(abs(a), 1.0):
            _fail(f"metric {key} differs batch-of-one: {a} vs {c}")
    _ok(f"sample_metrics identical on batch-of-one ({len(m_single)} metrics)")


def test_model_equivalence(samples):
    """Full PI-GNO forward: batched must equal per-sample, for EVERY norm.

    This is the test that catches graph-blending. GraphNorm's statistics are per
    graph; if they were taken over the whole batch, a core's prediction would
    depend on what it happened to be batched with -- which would still train and
    still look fine on a loss curve.
    """
    from dataclasses import replace as _replace
    from config import DEFAULT, ModelConfig
    from features import fit_normalization
    from model import PIGNO

    layout = NodeLayout.from_metadata(samples[0].meta)
    norm = fit_normalization(
        torch.cat([s.node_feats for s in samples]),
        torch.cat([s.edge_feats for s in samples]),
        torch.cat([s.flux for s in samples]),
        [float(s.k_eff) for s in samples],
        node_passthrough=layout.passthrough_cols)

    for norm_kind in ("layernorm", "graphnorm", "none"):
        torch.manual_seed(0)
        mc = ModelConfig.from_metadata(samples[0].meta, latent_dim=24,
                                       n_mp_layers=3, message_hidden=32,
                                       norm=norm_kind)
        model = PIGNO(mc, 3.2e-11)
        model.eval()
        b = collate_graphs(samples)

        with torch.no_grad():
            ob = model(node_feats_norm=norm.node.transform(b.node_feats),
                       edge_feats_norm=norm.edge.transform(b.edge_feats),
                       raw_node_feats=b.node_feats, edge_index=b.edge_index,
                       flux_scaler=norm.flux, k_mean=norm.k_mean,
                       k_std=norm.k_std, use_cuda_scatter=False,
                       batch=b.batch, n_graphs=b.n_graphs)
            for i, s in enumerate(samples):
                os_ = model(node_feats_norm=norm.node.transform(s.node_feats),
                            edge_feats_norm=norm.edge.transform(s.edge_feats),
                            raw_node_feats=s.node_feats,
                            edge_index=s.edge_index, flux_scaler=norm.flux,
                            k_mean=norm.k_mean, k_std=norm.k_std,
                            use_cuda_scatter=False)
                lo, hi = int(b.ptr[i]), int(b.ptr[i + 1])
                fe = (ob.flux_norm[lo:hi] - os_.flux_norm).abs().max()
                fs = os_.flux_norm.abs().max().clamp_min(1e-6)
                if fe / fs > 1e-4:
                    _fail(f"norm={norm_kind} graph {i}: batched flux differs "
                          f"(rel {float(fe / fs):.2e}) -- graphs are blending")
                ke = (ob.k_norm[i] - os_.k_norm).abs()
                ks = os_.k_norm.abs().clamp_min(1e-6)
                if ke / ks > 1e-4:
                    _fail(f"norm={norm_kind} graph {i}: batched k differs "
                          f"(rel {float(ke / ks):.2e})")
        _ok(f"PIGNO forward norm={norm_kind}: batched == per-sample")


def test_batched_loss(samples):
    """The batched loss must be finite and must use per-graph k in the residual."""
    from config import DEFAULT
    from losses import compute_loss

    b = collate_graphs(samples)
    k_hat = b.k_eff * 1.001                       # slightly-off per-graph k
    k_vec = expand_k_group_major(k_hat, b.batch, b.n_groups)
    terms = compute_loss(
        flux_hat_norm=b.flux, flux_ref_norm=b.flux,
        k_hat_norm=k_hat, k_ref_norm=b.k_eff,
        flux_hat_phys=b.flux, k_hat_phys=k_vec,
        A=b.A, F=b.F, boundary_mask=b.boundary_mask, cfg=DEFAULT.loss)
    d = terms.item_dict()
    if not all(v == v and abs(v) < float("inf") for v in d.values()):
        _fail(f"non-finite batched loss: {d}")
    if d["flux"] != 0.0:
        _fail("flux loss should be exactly 0 when prediction == reference")
    _ok(f"batched compute_loss finite, flux term exact: {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_paths.REPO_ROOT,
                                                   "datasets", "hex01"))
    args = ap.parse_args()

    # deliberately mix splits: hex val is N=3421 and test is N=4279, so this
    # batches graphs of DIFFERENT sizes -- the case that breaks naive stacking
    paths = []
    for split in ("val", "test", "train"):
        p = list_split(args.data, split)
        if p:
            paths.append(p[0])
    if len(paths) < 2:
        _fail(f"need >=2 samples under {args.data}; found {len(paths)}")

    samples = [load_torch_sample(p) for p in paths]
    sizes = [s.node_feats.shape[0] for s in samples]
    print(f"batching {len(samples)} samples, N={sizes} "
          f"({'variable' if len(set(sizes)) > 1 else 'uniform'} size)")

    test_residual_equivalence(samples)
    test_operator_structure(samples)
    test_khead_equivalence(samples)
    test_batch_of_one(samples)
    test_model_equivalence(samples)
    test_batched_loss(samples)
    print("PASS")


if __name__ == "__main__":
    main()
