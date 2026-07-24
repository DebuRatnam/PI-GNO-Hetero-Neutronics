# PI-GNO — File Map & Language Split

Physics-Informed Graph Neural Operator for a 2D **multigroup** neutron-diffusion
eigenvalue problem, discretized with **P1 finite elements**. The generator supports
**two reactor types** (selected by `reactor_type`), both flowing through the same
solver / graph / model:

- **`hex`** — a **Natrium-inspired hexagonal-duct core** (fast spectrum): identical
  structured submesh per assembly + Delaunay-stitched sodium gaps, explicit HT9 duct.
- **`fhr`** — a **Kairos KP-FHR pebble bed** (thermal spectrum): an **annular** core
  (central graphite reflector + fueled pebble annulus with fuel *and* graphite
  moderator pebbles + outer reflector), RSA-packed pebble nodes in FLiBe, **4 B4C
  control elements in the outer reflector** (NRC: KP-FHR control inserts into the side
  reflector) and **3 X-shaped shutdown elements in the inner bed**, each in a
  graphite-lined channel; steel vessel.

Group count is **configurable G** (`PhysicsConfig.n_groups`; default G=2, byte-
compatible with the original two-group data). The XS layout, node-feature width, and
model dims are **schema-driven** from each sample's metadata. This README maps every
file to its role and records the Python-vs-csrc decision. See `CLAUDE.md` for the
physics/dataset contract.

> Status: code only. Data generation runs on CPU (SciPy). FRNN is **not**
> implemented — it is yours (`csrc/custom_frnn/`); a KD-tree radius fallback stands
> in until it and the CUDA scatter kernel are compiled. Node count **N varies per
> sample** (irregular mesh); the model is N-agnostic.

## Language split (why each piece is Python or csrc)

| Aspect | Language | Rationale |
|---|---|---|
| **Data generation** | Python (NumPy/SciPy) | One-time offline; sparse assembly + eigensolve already optimized in SciPy/SuperLU. csrc buys nothing. |
| **Kernel layering / message passing** | **csrc CUDA** (+ Python fallback) | Hot path: runs every fwd/bwd × layers × epochs. Only the memory-bound **scatter-add** is fused in CUDA; the message MLP stays cuBLAS. |
| **Lifting / projecting** | Python (`nn.Linear`) | Dense GEMM → already cuBLAS. Custom csrc would reinvent it. |
| **Loss / backprop** | Python (`torch`, `torch.sparse`) | MSE + sparse matvec via cuSPARSE. The MP op's backward comes free from the CUDA kernel. |
| **FRNN** | **Pure C++/CUDA (you)** | Builds the neural message graph; left as an interface + stub. |

## `data generation/2D_fdm/` — dataset generation (Python)

| File | Role |
|---|---|
| `datagen_config.py` | Physics/solver/graph config + **`reactor_type`** dispatch + units + **reactor-aware metadata** (self-describing schema: n_groups, n_materials, node/edge feature order). `HexCoreConfig` (Natrium) and `PebbleCoreConfig` (KP-FHR) hold the per-reactor geometry knobs. `PhysicsConfig.n_groups` is the authoritative group count. Named distinctly from `src/config.py`. |
| `xs_common.py` | **Multigroup XS layout + `MultiGroupXS` container** shared by both material libraries. Row layout `[D(G), Sr(G), downscatter(G(G-1)/2), nuSf(G)]` (down-scatter-only), byte-compatible with the original 7-scalar two-group at G=2. Column-slice helpers + cache (de)serialization for the OpenMC library. |
| `materials.py` | **8-way** cross-section library for the Natrium **fast** core (fuel_inner/outer, primary/secondary_control, reflector, shield, **duct** HT9+gap, coolant) + control insert/withdraw (`CONTROL_FOLLOWER`). Loads an OpenMC cache (`xs_natrium.json`) if present, else uses committed defaults. |
| `materials_fhr.py` | **7-way** cross-section library for the KP-FHR **thermal** core (fuel_pebble, graphite_pebble, control_element, shutdown_element, reflector, coolant=FLiBe, vessel) + insert/withdraw (`FLIBE_FOLLOWER`). Same public symbols as `materials.py`; loads `xs_fhr.json` if present. |
| `geometry.py` | Builds the **Natrium hex lattice** (identical structured submesh + homogenized `duct` ring + Delaunay-stitched sodium gaps, 9+4 control). Returns the `CoreGeometry` struct (nodes, `elements[T,3]`, `boundary_edges`, `nodal_volume`, per-node XS). Reusable helpers `triangle_areas`/`nodal_volumes` shared with the pebble builder. |
| `geometry_pebble.py` | Builds the **annular KP-FHR pebble bed**: central graphite reflector column, RSA-packed fueled pebble annulus (fuel + graphite moderator pebbles) in FLiBe, **4 rigid B4C control cylinders in the outer reflector** (16-node circles in lined channels), **3 rigid X shutdown elements in the inner bed** (lined channels), steel vessel, Delaunay + **free-edge boundary**. Returns the same `CoreGeometry` struct → downstream unchanged. |
| `operators.py` | Assembles sparse **A** (leakage/removal) and **F** (fission) → `[GN, GN]`, group-major, with **P1 finite elements** (stiffness + lumped mass + Marshak Robin vacuum BC) on `elements`. **G×G block build** driven by `n_groups` (G=2 reproduces the original two-group result). |
| `solver.py` | Power iteration on `A⁻¹F` (one sparse LU) → `k_eff`, flux `[N,G]`, residual. G inferred from `A.shape / n_nodes`. |
| `power.py` | Derived node power density = `E_f · Σ_g nuSf_g·φ_g` (same relation reused by the model's power head). |
| `xs_openmc.py` | **Offline** OpenMC group-constant generation (flux-weighted collapse of ENDF/B data → cached JSON per reactor). `--dry-run` writes the committed default library to test the cache path without OpenMC. Not in the per-sample hot loop. |
| `graph_build.py` | Builds the **message graph** (a **radius/FRNN graph** on the irregular mesh nodes — no hardcoded neighbors) + **8-dim edge features** `[distance, dx, dy, interface_flag, harmonic_D1, dD1, dSigma_r1, dSigma_s12]`. Calls FRNN; KD-tree radius fallback until FRNN is wired. Separate from the FEM physics graph. |
| `frnn_interface.py` | The only touch-point to your FRNN (radius graph). Set `FRNN_AVAILABLE=True` + implement `frnn_query` to switch over. |
| `validate.py` | Pre-training contract checks, **schema-driven from metadata**: shapes, finiteness, `[GN,GN]` dims, `[N,G]` flux, non-empty boundary, **FEM mesh validity** (triangles + boundary edges in range, positive nodal volumes tiling the domain), reference residual, node/edge feature widths, reactor-gated material presence (hex→`duct`; fhr→`fuel_pebble`+`coolant`). |
| `dataset.py` | Orchestrates one sample → contract dict; **dispatches geometry on `reactor_type`** (hex `make_core` / fhr `make_pebble_core`); schema-driven node features; save/load `.npz`; `make_split_plans` (hex) / `make_split_plans_fhr` sized by `SamplingConfig`. |
| `generate.py` | CLI entry. `python generate.py [--reactor hex\|fhr] --out <dir> [--train-samples N …]`. Geometry-disjoint train/val/test via per-reactor insertion ranges. **Not auto-run.** |

## `src/` — model, physics objective, training (Python)

| File | Role |
|---|---|
| `config.py` | Model/loss/train hyperparameters (all logged). `ModelConfig.from_metadata(meta)` derives `node_in_dim`, `n_groups`, `n_materials` from a dataset so the same model fits either reactor. SiLU MP activation. |
| `features.py` | **`NodeLayout`** (schema-driven column indices from n_materials + n_groups) + normalization (fit on **train only**, retained inverses; one-hot material + boundary flag and the binary edge `interface_flag` pass through un-normalized). `NODE_COLS`/defaults are the Natrium hex layout for back-compat. |
| `dataio.py` | Loads `.npz` samples → torch tensors; builds torch-sparse A/F. |
| `lifting.py` | Separate Node/Edge lifting MLPs (Linear+SiLU). |
| `norm.py` | LayerNorm / GraphNorm. |
| `scatter.py` | Scatter-add aggregation: routes to CUDA `pigno_mp` ext, else pure-PyTorch fallback. Custom autograd `Function`. |
| `message_passing.py` | `MPLayer`/stack: gather → **SiLU** message MLP → scatter-add → residual update → norm. |
| `heads.py` | `FluxHead` `[N,2]`, `KHead` (mean/sum/attention pool→MLP), `PowerHead` (physics-consistent, matches `power.py`). |
| `physics.py` | PDE residual `R = Aφ − (1/k)Fφ` via torch.sparse (uses assembled A/F, never the FRNN graph) + BC residual. |
| `losses.py` | `L = L_flux + λk·L_k + λpde·L_PDE + λbc·L_BC`. |
| `model.py` | `PIGNO`: lift → MP stack → heads; returns normalized + physical predictions. |
| `metrics.py` | Reporting minimums: per-group flux err, k err, power err, PDE/BC residual, per-material breakdown. |
| `train.py` | Training loop scaffold. `python train.py --data <dir>`. **Not auto-run.** |

## `csrc/` — compiled kernels

| Path | Role |
|---|---|
| `message_passing/scatter_cuda.cu` | Fused scatter-add forward (atomicAdd) + backward (gather) CUDA kernels. |
| `message_passing/scatter.cpp` | torch/pybind bindings → `pigno_mp`; CPU parity fallback. |
| `message_passing/setup.py` | Builds `pigno_mp`. **Not auto-run** (needs CUDA toolkit). |
| `custom_frnn/` | **You implement.** README documents the required `edge_index [2,E]` output contract + wiring. |

## Run order (when you have a GPU)

```bash
# 0. (optional) generate traceable OpenMC group constants; without this the
#    committed synthetic libraries are used. --dry-run tests the cache path.
cd "data generation/2D_fdm"
python xs_openmc.py --reactor hex --out xs_natrium.json   # (add --dry-run to skip OpenMC)
python xs_openmc.py --reactor fhr --out xs_fhr.json

# 1. generate a dataset (CPU). Pick the reactor; default scale = 5000/1000/1000.
#    start small with overrides, scale up when ready.
python generate.py --reactor hex --out ../../../datasets/hex01 \
    --train-samples 50 --val-samples 10 --test-samples 10
python generate.py --reactor fhr --out ../../../datasets/fhr01 \
    --train-samples 50 --val-samples 10 --test-samples 10

# 2. (optional) build CUDA scatter kernel; src/scatter.py adds csrc/message_passing
#    to sys.path automatically, so the in-place .so is picked up from any cwd.
cd ../../csrc/message_passing && python setup.py build_ext --inplace

# 3. implement + wire FRNN, then train (model dims auto-detected from the dataset)
cd ../../src && python train.py --data ../../datasets/fhr01
```

## Key invariants (do not break silently)

- Both groups are FAST (sodium fast reactor). `0=high-fast` (high-energy),
  `1=slow-fast` (low-energy); DOF ordering group-major `[g1(N), g2(N)]`.
- `chi` lives in **F**, not node features.
- Material is **one-hot** (8-way: `[fuel_inner, fuel_outer, primary_control,
  secondary_control, reflector, shield, duct, coolant]`, cols 2–9); node features
  are **18-dim**. `material_state` is stored as integer ids, but the model never
  sees an ordinal material column.
- Mesh is a **Natrium hex-duct lattice** discretized with **P1 finite elements**;
  **N varies per sample**. Every assembly has an IDENTICAL structured submesh
  (spatial invariance); the **HT9 duct + sodium gap** is an explicit `duct`
  material (in A/F, not lumped into coolant); inter-assembly gaps are
  Delaunay-stitched `coolant`. Each sample stores `elements[T,3]`,
  `boundary_edges[B,2]`, `nodal_volume[N]`, hex metadata.
- Reactivity: **9 primary + 4 secondary** control positions; insertion toggles
  absorber vs sodium-follower XS (moves k_eff).
- Physics graph = the **FEM triangulation** (→ A, F). Message graph = a
  **radius/FRNN graph** on the same nodes (no hardcoded neighbors); edges carry
  **8** features `[distance, dx, dy, interface_flag, harmonic_D1, dD1, dSigma_r1,
  dSigma_s12]`. The two graphs are kept separate.
- Dataset variability = control insertion / enrichment boundary / ring counts /
  per-assembly XS perturbation. Train/val/test disjoint via non-overlapping
  **control-insertion ranges** per split.
- Dataset scale is set by `SamplingConfig` / `generate.py` CLI (default target
  5000/1000/1000); generation is never auto-run.
- PDE residual uses the **assembled FEM A/F** and **physical** (de-normalized) flux
  — never the FRNN message graph.
- Normalization fit on **train split only**; same transform to val/test.
- Vacuum BC is a **Marshak partial-current Robin term** on boundary edges
  (α=0.5), folded into **A** → enforced via L_PDE. Boundary flux is intentionally
  nonzero, so the zero-flux `L_BC` stays **off** (`lambda_bc=0.0`).
