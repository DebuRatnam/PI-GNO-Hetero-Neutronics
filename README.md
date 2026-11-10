# PI-GNO — File Map & Language Split

Physics-Informed Graph Neural Operator for a 2D **multigroup** neutron-diffusion
eigenvalue problem, discretized with **P1 finite elements**. The generator supports
**two reactor types** (selected by `reactor_type`), both flowing through the same
solver / graph / model:

- **`hex`** — a **Natrium-inspired hexagonal-duct core** (fast spectrum): identical
  structured submesh per assembly + Delaunay-stitched sodium gaps, explicit HT9 duct.
- **`fhr`** — a **Kairos KP-FHR pebble bed** (thermal spectrum) at published **gFHR**
  dimensions: a **cylindrical, full-diameter** RSA-packed bed (R=120 cm) of fuel *and*
  graphite moderator pebbles in FLiBe → 60 cm graphite side reflector → SS316H barrel
  + FLiBe downcomer + SS316H vessel (one homogenized ring). **10 B4C control elements
  in the side reflector** (NRC: KP-FHR control inserts into the side reflector), each
  in a graphite-lined channel, and **3 X-shaped shutdown elements inserted directly
  into the packed bed** (NRC), no thimble. Not annular: KP-FHR pebbles are buoyant and
  rise through the whole bed, so there is no central reflector column.

Group count is **configurable G** (`PhysicsConfig.n_groups`; default G=2, byte-
compatible with the original two-group data). The XS layout, node-feature width, and
model dims are **schema-driven** from each sample's metadata. This README maps every
file to its role and records the Python-vs-csrc decision. See `CLAUDE.md` for the
physics/dataset contract.

> Status: code only. Data generation runs on CPU (SciPy). The message graph is a
> **kNN graph** (fixed degree); there is no FRNN/radius path. The CUDA scatter kernel
> is optional (pure-PyTorch fallback). Node count **N varies per sample** (irregular
> mesh); the model is N-agnostic. Cross sections come **only** from an **OpenMC branch
> table** — there is no fallback library, so `xs_openmc.py` must be run before any
> data can be generated. See [OpenMC](#openmc--traceable-cross-sections).

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
| `xs_common.py` | **Multigroup XS layout + `MultiGroupXS` container** shared by both material libraries. Row layout `[D(G), Sr(G), downscatter(G(G-1)/2), nuSf(G)]` (down-scatter-only), byte-compatible with the original 7-scalar two-group at G=2. Column-slice helpers, `blend_xs` gray-rod blending, and the up-scatter pair ordering used by the OpenMC branch table. |
| `materials.py` | **8-way** material schema for the Natrium **fast** core (fuel_inner/outer, primary/secondary_control, reflector, shield, **duct** HT9+gap, coolant) + cross-section lookup. **Requires** the OpenMC branch table `xs_natrium.json`; there is no hand-tuned library, so constants always depend on burnup / temperature / insertion. Control insertion blends the table's measured rod-out (`control_follower`) and rod-in endpoints. Also owns the two things transport cannot supply: `CHI` (analytic for the 0.1 MeV cut) and `AXIAL_BUCKLING_CM2`. |
| `materials_fhr.py` | **7-way** material schema for the KP-FHR **thermal** core (fuel_pebble, graphite_pebble, control_element, shutdown_element, reflector, coolant=FLiBe, vessel). Same public symbols as `materials.py`; **requires** `xs_fhr.json`. Thermal up-scatter `Ss21` is the tallied upper triangle of the scatter matrix at the node's own temperature — the only defensible source, since up-scatter is an S(α,β) thermal-motion effect and cannot be a temperature-independent constant. |
| `geometry.py` | Builds the **Natrium hex lattice** (identical structured submesh + homogenized `duct` ring + Delaunay-stitched sodium gaps, 9+4 control). Returns the `CoreGeometry` struct (nodes, `elements[T,3]`, `boundary_edges`, `nodal_volume`, per-node XS). Reusable helpers `triangle_areas`/`nodal_volumes` shared with the pebble builder. |
| `geometry_pebble.py` | Builds the **cylindrical KP-FHR pebble bed** at gFHR dimensions: RSA-packed full-diameter bed (fuel + graphite moderator pebbles) in FLiBe, 60 cm graphite side reflector holding **10 rigid B4C control cylinders** (16-node circles in lined channels), **3 rigid X shutdown elements inserted directly into the bed**, SS316H barrel + FLiBe downcomer + SS316H vessel as one homogenized ring, Delaunay + **free-edge boundary**. Returns the same `CoreGeometry` struct → downstream unchanged. |
| `operators.py` | Assembles sparse **A** (leakage/removal) and **F** (fission) → `[GN, GN]`, group-major, with **P1 finite elements** (stiffness + lumped mass + Marshak Robin vacuum BC) on `elements`. **G×G block build** driven by `n_groups` (G=2 reproduces the original two-group result). |
| `solver.py` | Power iteration on `A⁻¹F` (one sparse LU) → `k_eff`, flux `[N,G]`, residual. G inferred from `A.shape / n_nodes`. |
| `power.py` | Derived node power density = `E_f · Σ_g nuSf_g·φ_g` (same relation reused by the model's power head). |
| `openmc_models.py` | **Full-core OpenMC transport models** mirroring both FEM cores, with **double heterogeneity**: explicit TRISO lattices inside explicit pebbles (`fhr`), explicit hex pin lattices inside each duct (`hex`). Reuses `geometry.py`'s own role/control assignment so the transport core *is* the FEM core. Axially reflective slab + radial vacuum (axial leakage stays in `Bz²`). Also carries the constituent-volume map used to homogenize resolved materials back onto one FEM node. |
| `xs_depletion.py` | **Unit-cell depletion** (`openmc.deplete`) → fuel isotopics vs burnup [MWd/kgHM] as a cached JSON, keyed **per fuel zone**. `hex` deplete's `fuel_inner` (15.50 wt%) and `fuel_outer` (19.75 wt%) as **two separate runs** — the depleted composition overrides the fresh material, so one shared inventory would erase the enrichment zoning; `fhr` has one fuel material, so one run. Enrichments come from `openmc_models.HEX_ENRICHMENT_WT_PCT` / `FHR_ENRICHMENT_WT_PCT`, the same constants the transport core is built from. Schema v1 (single `"fuel"` key) is rejected on load. Needs a depletion chain. |
| `xs_openmc.py` | **Offline** branch-case group-constant generation. For each (burnup × temperature × rod) branch: run the full-core model, tally `openmc.mgxs` with `domain_type="material"` (**in-situ** spectrum weighting — no infinite media), collapse with `D = 1/(3Σ_tr)`, `Sr = Σ_a + total out-scatter`, full scatter matrix (down **and** up), tallied `chi`, and per-constant Monte Carlo σ. Writes the schema-v2 branch table (`xs_natrium.json` / `xs_fhr.json`). **Running this is a prerequisite for generating any data** — the material modules have no fallback. Not in the per-sample hot loop. |
| `xs_branch.py` | **Branch-table reader + interpolator.** Linear in burnup and temperature (clamped, never extrapolated), gray-rod blend on the insertion axis with `D` combined through `Σ_tr`. Supplies the `xs_provenance` block embedded in every sample. |
| `validate_openmc.py` | **Verification harness**: P1 diffusion vs full-core continuous-energy OpenMC on identical core states → Δk, reactivity bias [pcm], rod worth (both codes), and radial power-shape RMS/max error. CSV + printed summary. This is the table/figure that justifies the diffusion labels. |
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
# 0. group constants. MANDATORY -- there is no fallback library, so step 1 raises
#    xs_branch.MissingBranchTable until these exist. Needs the OpenMC conda env +
#    ENDF/B data; see the OpenMC section below.
cd data_generation/p1_fem
python3 xs_depletion.py --reactor hex --chain chain_endfb80_fast.xml \
    --out depletion_natrium.json --burnups 0 2 20 40 60
python3 xs_openmc.py --reactor hex --out xs_natrium.json \
    --depletion depletion_natrium.json \
    --burnups 2 30 60 --temperatures 630 1000 --rods out in
python3 xs_depletion.py --reactor fhr --chain chain_endfb80_pwr.xml \
    --out depletion_fhr.json --burnups 0 2 20 50 90 130 160 190
python3 xs_openmc.py --reactor fhr --out xs_fhr.json \
    --depletion depletion_fhr.json \
    --burnups 2 95 190 --temperatures 823 1100 --rods out in
#    The branch points must SPAN the generator's sampled ranges (HexCoreConfig /
#    PebbleCoreConfig .burnup_mwd_kg_range, .temperature_k_range) -- states outside
#    the grid clamp onto the outermost point and silently share cross sections.

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
- **Spectra (G=2).** Group boundaries have ONE home, `xs_common.GROUP_BOUNDARIES_EV`,
  which drives the OpenMC collapse, the per-reactor `CHI`, and every sample's
  `geometry_metadata["group_boundaries_ev"]`. `hex` (Natrium) is FAST in both groups
  (`0=high-fast`, `1=slow-fast`), split at **0.1 MeV** — near the SFR flux peak and
  above the U-238 inelastic threshold (~45 keV), so both groups carry comparable flux
  and `g2` is the slowing-down tail where capture and control worth live. `fhr`
  (KP-FHR) is genuinely thermal (`0=fast`, `1=thermal`), split at **0.625 eV**. Never
  call the Natrium reactor "thermal", and never call its `g2` "thermal" either — it
  has no Maxwellian population, just a 1/E-like tail. DOF ordering group-major
  `[g0(N), g1(N)]`.
- **Per-reactor fixed nuclear data.** Fission spectrum `chi` and axial buckling live
  in the material module and are applied per reactor in `dataset.make_sample`:
  `materials.CHI=(0.99,0.01)` / `AXIAL_BUCKLING_CM2≈9.1e-4` (fast);
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
  sodium gaps. FHR: cylindrical RSA pebble bed (1 node/pebble) in FLiBe. Each sample
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
- **Cross-section provenance is recorded per sample** in
  `geometry_metadata["xs_provenance"]`, and is never absent — a sample cannot be
  generated without a branch table. It names the transport code version, evaluated
  data library, weighting spectrum, branch grid, per-branch k_eff and Monte Carlo σ.
  Check `converged` before publishing: a short smoke-test run carries
  `source="openmc"` and a real OpenMC version but is statistically worthless, so the
  loader flags `max_rel_std > 5%` rather than letting it label a dataset quietly.

## OpenMC — traceable cross sections

`materials.py` / `materials_fhr.py` carry **no cross sections of their own**. Every
constant is a **flux-weighted multigroup value collapsed from ENDF/B
continuous-energy data**; the hand-tuned synthetic libraries that used to serve as a
fallback have been deleted, so this pipeline is not optional.
It is an **offline pre-step**: the per-sample generator only ever reads a cached JSON.

### What the pipeline does

| Step | Script | Output |
|---|---|---|
| 1. isotopics vs burnup | `xs_depletion.py` | `depletion_<reactor>.json` |
| 2. branch-case group constants | `xs_openmc.py` | `xs_natrium.json` / `xs_fhr.json` |
| 3. verification vs transport | `validate_openmc.py` | `validation_<reactor>.csv` |

Design choices, and why each one is there:

- **In-situ weighting.** Every material is tallied with `domain_type="material"`
  inside the assembled full core, so `Σ_g = ∫Σ(E)φ(E)dE / ∫φ(E)dE` uses the spectrum
  the material actually sees. No infinite-medium unit cells anywhere.
- **Double heterogeneity.** Explicit TRISO particles inside explicit pebbles (`fhr`);
  explicit pin lattices inside each duct (`hex`). Self-shielding is geometric, not
  assumed. Resolved constituents are then flux-volume homogenized back onto the single
  material the FEM node carries.
- **Branch cases.** Burnup, temperature, and control insertion are real transport
  branches — depleted isotopics, Doppler + S(α,β), rodded/unrodded cores — rather than
  multipliers on one nominal library. `xs_branch.py` interpolates between them.
- **Transport-corrected `D`** = `1/(3Σ_tr)` from a tallied transport cross section.
- **Up-scatter** from the upper triangle of the tallied ν-scatter matrix.
- **Uncertainties + provenance** on every constant, surfaced into every sample's
  `geometry_metadata["xs_provenance"]`.

### Setup (macOS, Apple Silicon)

conda-forge has **no `osx-arm64` build of OpenMC**, so the env must be `osx-64` under
Rosetta. It therefore cannot be the same env as the arm64 torch install — which is
fine, since only the offline XS step needs OpenMC.

```bash
softwareupdate --install-rosetta --agree-to-license
brew install --cask miniforge && conda init zsh && exec zsh

conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name openmc-env --platform osx-64 openmc
conda activate openmc-env

# nuclear data: ~2.5 GB download, ~10 GB extracted
mkdir -p ~/nucdata && cd ~/nucdata
curl -L -o endfb-viii.0-hdf5.tar.xz \
  https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz
tar -xJf endfb-viii.0-hdf5.tar.xz && rm endfb-viii.0-hdf5.tar.xz
conda env config vars set \
  OPENMC_CROSS_SECTIONS=$HOME/nucdata/endfb-viii.0-hdf5/cross_sections.xml
conda activate openmc-env      # re-activate to apply
```

A **depletion chain** is also needed for step 1 (`https://openmc.org/depletion-chains/`).

### Running it

```bash
conda activate openmc-env
cd data_generation/p1_fem

# 1. isotopics vs burnup, one unit cell PER FUEL ZONE (fhr: 1, hex: 2)
python xs_depletion.py --reactor fhr --chain chain_endfb80_thermal.xml \
    --out depletion_fhr.json --burnups 0 2 20 50 90 130 160 190

# 2. branch-case group constants (full core, in-situ weighting)
python xs_openmc.py --reactor fhr --out xs_fhr.json \
    --depletion depletion_fhr.json \
    --burnups 2 95 190 --temperatures 823 1100 --rods out in

# 3. verify the diffusion model against transport
python validate_openmc.py --reactor fhr --out validation_fhr.csv \
    --depletion depletion_fhr.json --burnups 2 190
```

Same three commands with `--reactor hex` and `xs_natrium.json` / `depletion_natrium.json`
(hex burnups `0 2 20 40 60`, temperatures `630 1000`). Note step 1 costs **two**
depletion runs for `hex` — one per enrichment zone.

Drop the JSONs next to the material modules and the generator picks them up
— no code change. Cost scales as (burnups × temperatures × rods) full-core runs; start
with `--particles 2000 --batches 30 --inactive 10 --axial-cm 10` to shake out geometry
errors before committing to a production run.

### Before treating the output as benchmark-grade

1. **Know which numbers are sourced.** `fhr` geometry and materials are the
   *published* gFHR benchmark (Kairos' non-proprietary KP-FHR surrogate): 19.55 wt%
   **UCO** kernels, 0.22 TRISO packing, buoyancy-core pebble, 120/180/182/187/191 cm
   radial build, SS316H (not Hastelloy-N), 100 at% B-10 B4C rods. Element counts come
   from Hermes as licensed (NRC KP-TR-024-NP Rev 0, ML24095A258) where gFHR is silent. `hex` is
   **representative, not a vendor spec**: Natrium's public docket fixes the fuel form
   (U-10Zr, sodium-bonded, HT9 clad, peak enrichment <20 wt%), the B4C absorber and
   the 9+4 control assembly counts — but not the pitch, pin lattices, ring layout or
   enrichment split, which are literature values for a HALEU metal-fuel SFR.
2. **Read `validation_<reactor>.csv`.** The reactivity bias and radial power-shape
   error there are what justify labelling the dataset with diffusion solutions.
3. **Check `max_rel_std`** in the branch table — it bounds the Monte Carlo noise the
   labels inherit. Raise `--particles`/`--batches` if it is large next to the physics
   you are resolving.
4. **Unit-cell depletion is an approximation**: isotopics come from a representative
   cell, while the collapse spectrum is the full-core one. Recorded in the JSON.
