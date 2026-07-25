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

> Status: code only. Data generation runs on CPU (SciPy). The message graph is a
> **kNN graph** (fixed degree); there is no FRNN/radius path. The CUDA scatter kernel
> is optional (pure-PyTorch fallback). Node count **N varies per sample** (irregular
> mesh); the model is N-agnostic. Cross sections are **synthetic** until traceable
> OpenMC constants are generated — see [OpenMC](#openmc--traceable-cross-sections-next-session).

## Language split (why each piece is Python or csrc)

| Aspect | Language | Rationale |
|---|---|---|
| **Data generation** | Python (NumPy/SciPy) | One-time offline; sparse assembly + eigensolve already optimized in SciPy/SuperLU. csrc buys nothing. |
| **Kernel layering / message passing** | **csrc CUDA** (+ Python fallback) | Hot path: runs every fwd/bwd × layers × epochs. Only the memory-bound **scatter-add** is fused in CUDA; the message MLP stays cuBLAS. |
| **Lifting / projecting** | Python (`nn.Linear`) | Dense GEMM → already cuBLAS. Custom csrc would reinvent it. |
| **Loss / backprop** | Python (`torch`, `torch.sparse`) | MSE + sparse matvec via cuSPARSE. The MP op's backward comes free from the CUDA kernel. |
| **Message graph** | Python (SciPy cKDTree) | kNN on mesh nodes; fixed degree → clean batching. Cheap, one-time per sample. |

## `data_generation/p1_fem/` — dataset generation (Python)

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
| `xs_openmc.py` | **Offline** OpenMC group-constant generation: per-material infinite-medium `openmc.mgxs` tally → flux-weighted collapse → cached JSON per reactor (`xs_natrium.json` / `xs_fhr.json`) that the material modules auto-load. Per-reactor `chi` from the material module. `--dry-run` writes the committed default library (no OpenMC) to test the cache path. Needs OpenMC + ENDF/B data to run — see [OpenMC](#openmc--traceable-cross-sections-next-session). Not in the per-sample hot loop. |
| `graph_build.py` | Builds the **message graph** — a **kNN graph** (fixed degree `knn_k`) on the mesh nodes, no hardcoded neighbors — + **8-dim edge features** `[distance, dx, dy, interface_flag, harmonic_D1, dD1, dSigma_r1, dSigma_s12]`. Separate from the FEM physics graph. |
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
| `physics.py` | PDE residual `R = Aφ − (1/k)Fφ` via torch.sparse (uses assembled A/F, never the message graph) + BC residual. |
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

## Run order (when you have a GPU)

```bash
# 0. (optional) generate traceable OpenMC group constants; without this the
#    committed synthetic libraries are used. --dry-run tests the cache path.
cd data_generation/p1_fem
python3 xs_openmc.py --reactor hex --out xs_natrium.json   # (add --dry-run to skip OpenMC)
python3 xs_openmc.py --reactor fhr --out xs_fhr.json

# 1. generate a dataset (CPU). Pick the reactor; default scale = 5000/1000/1000.
#    start small with overrides, scale up when ready.
python3 generate.py --reactor hex --out ../../datasets/hex01 \
    --train-samples 50 --val-samples 10 --test-samples 10
python3 generate.py --reactor fhr --out ../../datasets/fhr01 \
    --train-samples 50 --val-samples 10 --test-samples 10

# 2. (optional) build CUDA scatter kernel; src/scatter.py adds csrc/message_passing
#    to sys.path automatically, so the in-place .so is picked up from any cwd.
cd ../../csrc/message_passing && python3 setup.py build_ext --inplace

# 3. train (model dims auto-detected from the dataset metadata)
cd ../../src && python3 train.py --data ../../datasets/fhr01
```

## Key invariants (do not break silently)

- **Two reactors, one pipeline.** `reactor_type` selects geometry + material set;
  everything downstream (operators/solver/graph/model) is reactor-agnostic and
  schema-driven from metadata. Never hardcode widths — read `n_groups`,
  `n_materials`, `node_feature_order`.
- **Spectra (G=2).** `hex` (Natrium) is FAST both groups (`0=high-fast`,
  `1=slow-fast`). `fhr` (KP-FHR) is genuinely thermal (`0=fast`, `1=thermal`, ~0.625
  eV cut). Never call the Natrium reactor "thermal". DOF ordering group-major
  `[g0(N), g1(N)]`.
- **Per-reactor fixed nuclear data.** Fission spectrum `chi` and axial buckling live
  in the material module and are applied per reactor in `dataset.make_sample`:
  `materials.CHI=(0.60,0.40)` / `AXIAL_BUCKLING_CM2≈9.1e-4` (fast);
  `materials_fhr.CHI=(1.0,0.0)` / `≈1.0e-4` (thermal). `chi` lives in **F**, not node
  features.
- **Axial leakage.** The 2D operator adds `D_g·Bz²` to removal (`PhysicsConfig.axial_buckling`);
  `Bz²=0` reproduces the pure-2D operator.
- **Material one-hot**, schema-driven width: hex 8-way (`[fuel_inner, fuel_outer,
  primary_control, secondary_control, reflector, shield, duct, coolant]`) → 18-dim
  node features at G=2; fhr 7-way → 17-dim. `material_state` is stored as integer
  ids; the model never sees an ordinal material column.
- **Gray-rod insertion.** Control/shutdown insertion is a continuous depth `f∈[0,1]`
  (axially-averaged absorber, `xs_common.blend_xs`), not binary. Hex: per-rod
  independent depth. FHR: bank depth, with **80% ganged / 20% independent**
  per-element (tilt / stuck-rod). Binary endpoints stay byte-identical.
- **FHR thermal up-scatter** `Ss_{g2→g1}` is applied at the **operator level** (into
  `A`), not stored in the per-node XS row (schema stays down-scatter-only); hex has
  none (`CoreGeometry.upscatter=None`).
- **Directional burnup proxy** (fuel only): depletion lowers `nuSf`, fission-product
  poison raises removal (thermal `Sr2` in fhr); plus a symmetric temperature wiggle.
- **Mesh** = P1 finite elements; **N varies per sample**. Hex: identical structured
  submesh per assembly (spatial invariance) + explicit `duct` (HT9+gap) + Delaunay
  sodium gaps. FHR: annular RSA pebble bed (1 node/pebble) in FLiBe. Each sample
  stores `elements[T,3]`, `boundary_edges[B,2]`, `nodal_volume[N]`, reactor metadata.
- **Two graphs, kept separate.** Physics graph = the **FEM triangulation** (→ A, F).
  Message graph = a **kNN graph** (`knn_k`, fixed degree, no hardcoded neighbors) on
  the same nodes; edges carry **8** features `[distance, dx, dy, interface_flag,
  harmonic_D1, dD1, dSigma_r1, dSigma_s12]`.
- **PDE residual** uses the **assembled FEM A/F** and **physical** (de-normalized)
  flux — never the message graph.
- **Splits disjoint** via non-overlapping per-reactor insertion / (hex) ring /
  (fhr) moderator-pebble-fraction ranges. Scale set by `SamplingConfig` /
  `generate.py` (default 5000/1000/1000); generation is never auto-run.
- Normalization fit on **train split only**; same transform to val/test.
- Vacuum BC is a **Marshak partial-current Robin term** on boundary edges (α=0.5),
  folded into **A** → enforced via L_PDE. Boundary flux is intentionally nonzero, so
  the zero-flux `L_BC` stays **off** (`lambda_bc=0.0`).
- **Cross sections are synthetic** (hand-tuned, representative order-of-magnitude)
  until OpenMC constants are generated. Consequence: nominal k_eff is not tuned to a
  real reactor (hex currently sits ~0.95 rods-out). See the OpenMC section.

## OpenMC — traceable cross sections (next session)

The committed cross sections are **synthetic** (right spectrum/ordering, not
benchmarked). `data_generation/p1_fem/xs_openmc.py` replaces them with **traceable,
flux-weighted multigroup constants** tallied from ENDF/B via OpenMC. It is an
**offline pre-step** — the per-sample generator only ever reads the cached JSON.

**Current state of `xs_openmc.py`:**
- Real pipeline implemented: per-material infinite-medium unit cell → `openmc.mgxs`
  tally (`diffusion-coefficient`, `absorption`, `nu-scatter matrix`, `nu-fission`) on
  the two-group structure → collapse to `MultiGroupXS` (`Sr = absorption + out-scatter`,
  down-scatter block only). Fissile materials run `eigenvalue`; others use a Watt
  driving source. Per-reactor `chi` from the material module.
- `--dry-run` works with no OpenMC (writes the committed library to test the cache).
- **Not runnable in this repo's dev env** (OpenMC + nuclear data absent).

**To make it real (do on a machine you provision):**
1. **Install OpenMC** (prebuilt, don't compile): `conda create -n openmc -c conda-forge openmc`.
2. **Get nuclear data**: download an ENDF/B-VIII.0 HDF5 library (~10–15 GB) and
   `export OPENMC_CROSS_SECTIONS=/path/cross_sections.xml`. (~0.5–2 GB RAM at runtime;
   disk is the real cost.)
3. **Validate `_build_material` compositions** — they are documented *literature-level
   starting points* (HALEU U-10Zr, graphite, B4C, FLiBe, sodium, HT9/steel), **not**
   your reactor spec. Refine enrichment/densities/Li-7 % before trusting output.
4. **Run** (in your own Terminal, so the conda env persists):
   ```bash
   conda activate openmc
   cd data_generation/p1_fem
   python3 xs_openmc.py --reactor hex --out xs_natrium.json
   python3 xs_openmc.py --reactor fhr --out xs_fhr.json
   ```
5. **Drop the JSONs next to the material modules** → `materials.py` / `materials_fhr.py`
   auto-override the hand-tuned defaults at import (no code change).
6. **Sanity-check**: infinite-medium k-inf / group constants look physical; re-check the
   dataset k_eff band (should move toward critical once XS are realistic).
7. **Optional follow-ups**: feed the tallied thermal up-scatter into
   `materials_fhr.UPSCATTER_21` (traceable instead of representative); consider
   full-core flux-weighting instead of infinite-medium if the spectrum looks off.

**Caveat**: infinite-medium weighting + starting-point compositions mean the first
output is a scaffold result — review before treating as benchmark-grade.
