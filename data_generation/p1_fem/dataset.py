"""Orchestrate generation of one sample and a full dataset; serialize to disk.

One sample = one IRREGULAR P1-FEM core state (N varies per sample):
    material_state[N], coordinates[N,2], cross_sections[N,7],
    elements[T,3], boundary_edges[B,2], nodal_volume[N],   (FEM mesh topology)
    edge_index[2,E], edge_features[E,8],                    (kNN message graph)
    A[2N,2N], F[2N,2N], boundary_mask[N], k_eff,
    flux[N,2], power_density[N], geometry_metadata.

Node feature tensor (canonical 15-dim order, one-hot material) is assembled here for
direct model consumption. Splits are kept geometrically disjoint by disjoint
reflector-thickness ranges per split (train/val/test).

NOTE: generate.py is the CLI entry point. This module performs NO training and
is not auto-run; it only builds tensors and writes .npz files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import scipy.sparse as sp

from datagen_config import DataGenConfig, DEFAULT, SamplingConfig
from geometry import make_core
from operators import assemble_AF
from solver import solve_keff
from power import power_density
from graph_build import build_message_graph
from validate import validate_sample

import materials as _mat_hex
import materials_fhr as _mat_fhr


def _mat_module(reactor_type: str):
    """Return the material module (Natrium hex vs KP-FHR pebble) for a reactor."""
    return _mat_fhr if reactor_type == "fhr" else _mat_hex


def node_features(sample: dict) -> np.ndarray:
    """Assemble the canonical node feature matrix in documented order:
    [x, y, <one-hot material>, <G-group XS>, boundary_flag].

    Material is ONE-HOT (width = the active reactor's material count) so the model
    sees no ordinal material column. The XS block width follows n_groups. The exact
    order is recorded in geometry_metadata['node_feature_order'].
    """
    rt = str(sample["geometry_metadata"].get("reactor_type", "hex"))
    onehot = _mat_module(rt).one_hot_batch(sample["material_state"])
    coords = sample["coordinates"]
    xs = sample["cross_sections"]                       # [N, n_xs_cols(G)]
    bflag = sample["boundary_mask"].astype(np.float64)[:, None]
    return np.concatenate([coords, onehot, xs, bflag], axis=1)


def make_sample(cfg: DataGenConfig = DEFAULT, *,
                layout_name: str = "default",
                rng: Optional[np.random.Generator] = None,
                validate: bool = True,
                **knobs) -> dict:
    """Generate one core sample. Dispatches on cfg.reactor_type:

      hex : Natrium hex-duct core (knobs: reflector_rings, shield_rings,
            enrichment_boundary, insert_fraction).
      fhr : KP-FHR pebble bed (knobs: insert_control, insert_shutdown).

    The physics solve / graph build / serialization are identical for both. Split
    plans keep train/val/test disjoint via the per-reactor knob ranges."""
    rng = rng or np.random.default_rng()
    if cfg.reactor_type == "fhr":
        from geometry_pebble import make_pebble_core
        geom = make_pebble_core(
            cfg.pebblecore, layout_name=layout_name,
            insert_control=knobs.get("insert_control", 1.0),
            insert_shutdown=knobs.get("insert_shutdown", 0.0),
            graphite_pebble_frac=knobs.get("graphite_pebble_frac"), rng=rng)
    else:
        geom = make_core(
            cfg.hexcore, layout_name=layout_name,
            reflector_rings=knobs.get("reflector_rings"),
            shield_rings=knobs.get("shield_rings"),
            enrichment_boundary=knobs.get("enrichment_boundary"),
            insert_fraction=knobs.get("insert_fraction"), rng=rng)

    # Per-reactor FIXED nuclear data: fission spectrum chi + axial-leakage buckling
    # come from the material module (fast core vs thermal pebble bed differ). These are
    # the two quantities the branch table cannot supply -- Bz^2 because the transport
    # models are axially reflective by design, and chi only when a table carries no
    # tallied spectrum. The module CHI is analytic for that reactor's group boundary
    # and is used when its length matches n_groups; an explicit multigroup chi in
    # cfg.physics is otherwise preserved.
    #
    # PREFERRED: the TALLIED core-average fission spectrum at this core's state, which
    # shifts with burnup (Pu-239 births harder than U-235) and with rod insertion.
    # geom was built above, so the branch table is already loaded by this point.
    mod = _mat_module(cfg.reactor_type)
    chi = (mod.CHI if getattr(mod, "CHI", None) is not None
           and len(mod.CHI) == cfg.physics.n_groups else cfg.physics.chi)
    tallied_chi = mod.branch_chi(
        core_rod_frac=float(geom.assembly_metadata.get("mean_control_depth", 0.0)))
    if tallied_chi is not None and len(tallied_chi) == cfg.physics.n_groups:
        chi = tallied_chi
    physics = replace(cfg.physics, chi=chi,
                      axial_buckling=getattr(mod, "AXIAL_BUCKLING_CM2",
                                             cfg.physics.axial_buckling))
    eff_cfg = replace(cfg, physics=physics)

    A, F = assemble_AF(geom, physics)
    sol = solve_keff(A, F, cfg.solver, n_nodes=geom.n_nodes)
    pwr = power_density(sol.flux, geom.cross_sections, physics)
    edge_index, edge_features = build_message_graph(geom, cfg.graph)

    id2mat = _mat_module(cfg.reactor_type).ID_TO_MATERIAL
    counts = {id2mat[i]: int((geom.material_state == i).sum()) for i in id2mat}

    sample = {
        "material_state": geom.material_state,
        "coordinates": geom.coordinates,
        "cross_sections": geom.cross_sections,
        # FEM mesh topology (physics graph): triangles, boundary edges, nodal volumes.
        "elements": geom.elements,
        "boundary_edges": geom.boundary_edges,
        "nodal_volume": geom.nodal_volume,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "A": A, "F": F,
        "boundary_mask": geom.boundary_mask,
        "k_eff": np.float64(sol.k_eff),
        "flux": sol.flux,
        "power_density": pwr,
        "geometry_metadata": {
            **eff_cfg.metadata(),
            "layout_name": layout_name,
            "control_rod_cells": geom.control_rod_cells.tolist(),
            # reactor-specific layout (assemblies/pebbles, insertion, rings, ...)
            **geom.assembly_metadata,
            "material_counts": counts,
            "solver_converged": bool(sol.converged),
            "solver_residual": float(sol.residual),
            "solver_iters": int(sol.iters),
        },
    }
    sample["node_features"] = node_features(sample)
    if validate:
        validate_sample(sample, residual_tol=cfg.solver.residual_tol)
    return sample


def save_sample(sample: dict, path: str) -> None:
    """Serialize to .npz; sparse A/F stored as CSR triplets."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    A, F = sample["A"].tocoo(), sample["F"].tocoo()
    flat = {k: v for k, v in sample.items() if k not in ("A", "F", "geometry_metadata")}
    np.savez_compressed(
        path,
        **flat,
        A_data=A.data, A_row=A.row, A_col=A.col, A_shape=np.array(A.shape),
        F_data=F.data, F_row=F.row, F_col=F.col, F_shape=np.array(F.shape),
        geometry_metadata=np.array(repr(sample["geometry_metadata"])),
    )


def load_sample(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    A = sp.coo_matrix((z["A_data"], (z["A_row"], z["A_col"])),
                      shape=tuple(z["A_shape"])).tocsr()
    F = sp.coo_matrix((z["F_data"], (z["F_row"], z["F_col"])),
                      shape=tuple(z["F_shape"])).tocsr()
    out = {k: z[k] for k in z.files
           if not k.startswith(("A_", "F_")) and k != "geometry_metadata"}
    out["A"], out["F"] = A, F
    # trusted local file. `nan`/`inf` can appear in the metadata (e.g. a Monte Carlo
    # statistic that never converged), and repr() emits them bare, so bind them.
    out["geometry_metadata"] = eval(
        str(z["geometry_metadata"]),
        {"__builtins__": {}, "nan": float("nan"), "inf": float("inf")})
    return out


# ---- split planning (geometry-disjoint) ------------------------------------

@dataclass
class SplitPlan:
    """Disjoint configuration generator for one split. Train/val/test draw from
    DISJOINT control-insertion ranges (and enrichment-boundary choices) so the core
    states do not overlap (generalization requirement); insertion pattern, ring
    counts, and per-assembly XS perturbation still vary within each split."""
    name: str
    n_samples: int
    insert_fraction_range: tuple           # disjoint across splits
    enrichment_boundary_choices: tuple
    reflector_ring_choices: tuple
    seed: int


def make_split_plans(sampling: SamplingConfig) -> tuple:
    """Build the 3 split plans, sized by `sampling`. Counts default to the
    large-study target (5000/1000/1000); override via the generate.py CLI. The
    control-insertion ranges are kept disjoint across splits."""
    return (
        SplitPlan("train", sampling.train_samples,
                  insert_fraction_range=(0.0, 0.5),
                  enrichment_boundary_choices=(1, 2), reflector_ring_choices=(1, 2), seed=1),
        SplitPlan("val", sampling.val_samples,
                  insert_fraction_range=(0.5, 0.75),
                  enrichment_boundary_choices=(2,), reflector_ring_choices=(1,), seed=2),
        SplitPlan("test", sampling.test_samples,
                  insert_fraction_range=(0.75, 1.0),
                  enrichment_boundary_choices=(2, 3), reflector_ring_choices=(2,), seed=3),
    )


# Default plans at the large-study target scale (NOT auto-generated).
DEFAULT_SPLITS = make_split_plans(SamplingConfig())


@dataclass
class FHRSplitPlan:
    """Disjoint configuration generator for one KP-FHR split. Train/val/test draw
    from DISJOINT control/shutdown-insertion ranges so core states do not overlap;
    the random pebble packing (seed), moderator-pebble fraction, and burnup
    perturbation still vary within each split."""
    name: str
    n_samples: int
    control_insert_range: tuple            # disjoint across splits
    shutdown_insert_range: tuple
    graphite_pebble_range: tuple           # fuel:moderator ratio (reactivity lever)
    seed: int
    # Fraction of samples whose 4 control elements insert to INDEPENDENT per-element
    # depths (asymmetric tilt / stuck-rod states) rather than a single ganged depth.
    # KP-FHR runs banked/symmetric in normal operation, so keep this a minority.
    control_independent_frac: float = 0.20


def make_split_plans_fhr(sampling: SamplingConfig) -> tuple:
    """KP-FHR split plans, sized by `sampling`, with disjoint insertion ranges. The
    moderator-pebble fraction varies within each split (a real reactivity lever)."""
    gpr = (0.08, 0.22)
    return (
        FHRSplitPlan("train", sampling.train_samples,
                     control_insert_range=(0.0, 0.5),
                     shutdown_insert_range=(0.0, 0.34),
                     graphite_pebble_range=gpr, seed=11),
        FHRSplitPlan("val", sampling.val_samples,
                     control_insert_range=(0.5, 0.75),
                     shutdown_insert_range=(0.34, 0.67),
                     graphite_pebble_range=gpr, seed=12),
        FHRSplitPlan("test", sampling.test_samples,
                     control_insert_range=(0.75, 1.0),
                     shutdown_insert_range=(0.67, 1.0),
                     graphite_pebble_range=gpr, seed=13),
    )
