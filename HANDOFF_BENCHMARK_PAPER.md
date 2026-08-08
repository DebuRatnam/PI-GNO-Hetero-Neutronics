# Handoff: benchmarking methodology for `main.tex`

How PI-GNO is compared against DeepONet, FNO, and a GNN — what is measured, what
controls fairness, and what may and may not be claimed.

> **STATUS: NO MODEL RESULTS EXIST YET.** All 105 training runs are pending (GPU).
> Every number below is either a *measured property of the data/solver* (safe to
> write, marked ✅) or a *pending model result* (do not write, marked ⏳). All
> previously trained checkpoints were deliberately deleted.

---

## 1. The claim — and the correction that keeps it defensible

The intuitive framing "PI-GNO beats FNO and DeepONet, and beats GNNs on
resolution" contains a statement that a reviewer will reject:

> ❌ **Do not write that FNO cannot handle varying resolution.** It is false. FNO
> learns a kernel in Fourier space and is genuinely resolution-flexible — that is
> its central claim.

FNO's real limitation here is narrower and defensible: **it requires a uniform
Cartesian grid**, and a hex-duct lattice and an RSA-packed pebble bed are not one.

The defensible contribution is a 2×2, not a one-axis win:

| | irregular geometry | discretization transfer | new BC |
|---|---|---|---|
| FNO | ✗ needs a grid (measured floor) | ✓ | ✗ |
| DeepONet | partial — trunk ✓, fixed branch sensors ✗ | partial | ✗ |
| MeshGraphNet (GNN) | ✓ | ✗ | ✗ |
| **PI-GNO** | ✓ | ✓ | ✓ (via `A` in the PDE residual) |

**A second correction, equally important.** PI-GNO with `aggregation="sum"` *is*
a GNN in exactly the respect under test. The GNO behaviour comes from
volume-weighted quadrature, not from the model's name. The paper must attribute
any difference to the **mechanism**, which is what experiment `e3_res_agg` does.

---

## 2. What is being compared

| model | source | notes |
|---|---|---|
| **PI-GNO** | this repo, `src/` | the model under study |
| **MeshGraphNet** | `physicsnemo.models.meshgraphnet` (NVIDIA PhysicsNeMo 2.1.1) | the GNN baseline; NVIDIA-maintained so the GNO-vs-GNN result is not self-refereed |
| **FNO** | `physicsnemo.models.fno` (same) | grid-based neural operator |
| **DeepONet** | written in-repo, `benchmarks/models/deeponet.py` | not in PhysicsNeMo core (only in the much heavier physicsnemo-sym); branch + trunk + dot product is ~80 lines |

Cite PhysicsNeMo 2.1.1 for FNO and MeshGraphNet. State DeepONet is implemented
following Lu et al. 2021 and say why it wasn't imported.

### Architectural fairness details worth stating in the paper

- MeshGraphNet is given the **same kNN message graph and the same 8-dimensional
  edge features** PI-GNO gets — not the FEM triangulation. If the two saw
  different graphs, an accuracy difference would be attributable to the graph.
- MeshGraphNet's decoder emits a latent vector and the **same `FluxHead`/`KHead`**
  PI-GNO uses are attached on top, so the comparison is processor-vs-processor
  with the heads held fixed.
- **Every model** computes power through the same parameter-free `PowerHead`
  (`power = E_f · Σ_g νΣf_g φ_g`), so power error reflects flux error alone and no
  model can win power with a free head that learns the labels.
- FNO predicts on a grid and **interpolates back to mesh nodes before scoring**.
  Scoring it on its own grid would hide the error it is there to expose.
- DeepONet's trunk evaluates directly at mesh nodes (no output interpolation
  error); its branch reads a fixed sensor lattice on a fixed domain. That
  asymmetry is why it is "partial" on the resolution axis.

---

## 3. Fairness controls (these belong in the methods section)

1. **Common parameter budget: 400k.** One width knob per model is bisected to
   hit it. Achieved ✅: pigno 398,605 (`latent_dim=25`), mgn 399,765
   (`hidden_dim=75`), deeponet 402,384 (`branch_hidden=35`), fno 432,199
   (`latent_channels=10`). FNO is 8% **over** — the closest reachable given its
   `latent_channels²` scaling, erring in the baseline's favour rather than
   starving it. Report actual counts.
2. **Same learning-rate grid** {3e-4, 1e-3, 3e-3}, selected on val with the same
   shortened budget, for every model. This is the only per-model tuning.
3. **One training loop, one normalization, one metric function** for all four
   models (`benchmarks/harness.py`). Normalization is fitted on **train only** and
   persisted, so val/test use the transform the model was trained with.
4. **E1 runs every model data-only (λ_pde = 0), PI-GNO included.** This is the
   single most important control: if PI-GNO kept its physics loss while the
   baselines had none, any win would be attributable to the loss rather than the
   architecture. E2 then reintroduces the physics term and credits it separately.
5. **3 seeds per cell**, reported mean ± std.

---

## 4. The experiment matrix — 35 cells × 3 seeds = 105 runs

```
experiment  →  cells (model × sweep value)  →  seeds (×3)
```

| experiment | dataset | cells | question |
|---|---|---|---|
| `e1_hex` | hex01 | 4 | architecture accuracy, fast spectrum, data-only |
| `e1_fhr` | fhr01 | 4 | same, thermal spectrum + random packing |
| `e2_hex` | hex01 | 6 | physics-loss ablation, λ_pde ∈ {0, 0.1, 1.0}, graph models |
| `e2_fhr` | fhr01 | 6 | same |
| `e3_res` | hex_res/L2 | 4 | discretization transfer, all 4 models |
| `e3_res_agg` | hex_res/L2 | 3 | **GNO-vs-GNN mechanism**: pigno × sum / volume / volume_raw |
| `e4_bc` | hex_bc | 4 | boundary-condition transfer |
| `e4_bc_pde` | hex_bc | 4 | does the physics loss help BC transfer |

E5 (cost) is a reporting pass, not extra training.

**E3 and E4 are hex-only.** State the reasons rather than omitting them:
- fhr's mesh is one node per pebble — that is physics, not discretization, so it
  cannot be refined without changing the reactor.
- fhr's boundary is unmeasurable at the published vessel (see §7).

---

## 5. Metrics

Per core, written to CSV, identical across all experiments and models:

| metric | meaning |
|---|---|
| `flux_rel_l2_g1`, `flux_rel_l2_g2` | per-group flux relative L2 — headline |
| `k_abs_err`, `k_rel_err` | eigenvalue error (**convert to pcm for the paper**) |
| `power_rel_l2` | power density, from the shared physics-consistent head |
| `flux_rel_l2_mat0…mat7` | per-material breakdown — where the error lives |
| `pde_residual_rms` | ‖Aφ̂ − (1/k̂)Fφ̂‖ — physical consistency |
| `bc_residual_rms` | boundary-flux diagnostic |
| `inference_s`, `n_nodes` | E5 cost |

Planned additions for a reactor-physics audience (see `HANDOFF_CPU_WORK.md` §4d):
power peaking factor error, L∞ nodal error, assembly-wise radial power error,
control rod worth.

---

## 6. Discretization transfer (E3) — the metric subtlety that must be explained

**The obvious metric is wrong and the paper should say why.** Scoring a model
against *each mesh's own FEM reference* does not measure discretization
invariance, because those references are not the same field. ✅ Measured: the
reference solutions disagree with each other by **6.4%** (L1 vs L3, group 1) after
normalization, and k_eff moves **3059 pcm** from L1 to L2 on the *same physical
core*. Scoring that way charges the model for the FEM's convergence error, and a
model that perfectly fit every mesh would look maximally mesh-*dependent*.

Two metrics are therefore reported:

**(a) Invariance — no reference involved.** Run the model on two meshes of the
same physical core, interpolate one prediction onto the other's nodes, measure
disagreement. A discretization-invariant operator scores ≈ 0.

**(b) Accuracy vs a fixed fine truth mesh** (L6, N ≈ 102k), with each mesh's
discretization floor beside it.

Both are needed: invariance alone is gameable — a constant output is perfectly
invariant. (Observed in a throwaway run: a barely-trained model scored *better*
invariance than the references while being uniformly wrong at every mesh.)

### A scale trap worth one sentence in the methods

`solve_keff` returns an eigenvector normalized to unit L2 norm over its GN
entries, so reference flux **magnitude is a pure artifact of node count** — ✅
measured max φ₁ = 0.0375 / 0.0212 / 0.0147 at L1/L2/L3, exactly 1/√N. All
cross-mesh comparisons normalize by ∫Σ_g φ_g dV instead (one scalar for all
groups, so the physical spectrum ratio survives).

### ✅ Reference baselines, 30 paired configs (safe to publish now)

Invariance vs L3 and floor vs L6, group 1:

| mesh | N | invariance (FEM reference) | floor vs L6 |
|---|---|---|---|
| L0 — different meshing *rule* | 3,850 | 0.0963 | 0.1131 |
| L1 — coarser | 5,314 | 0.0655 | 0.0815 |
| Lmixed3_1_3 — density varies *within* one graph | 9,310 | 0.0863 | 0.0977 |
| L2 — training mesh | 15,562 | 0.0158 | 0.0315 |
| L3 | 31,666 | — | 0.0155 |

These are what a model must beat. ⏳ Model columns pending.

Note: Lmixed has ~1.8× L1's nodes but a worse floor. ✅ Cause is measured and is
**not** mesh quality — aspect-ratio distributions are identical to L1 (p99 13.47,
max 30.16). It is domain area: the Delaunay gap filter rejects more triangles
where a fine hex abuts a coarse one, so Lmixed loses 0.94% of area vs L1's 0.76%
(both against L6). Floor tracks area deficit across the whole family. Do not
write "more nodes gave worse physics."

---

## 7. Boundary-condition transfer (E4)

Marshak Robin coefficient α = (1−β)/(2(1+β)) for albedo β; β = 0 reproduces pure
vacuum exactly. Train β ∈ [0, 0.40), test β ∈ [0.55, 0.80] — β is the **only**
axis held disjoint; rod insertion, enrichment, burnup and temperature are i.i.d.
across splits so a failure is attributable to the boundary.

**✅ A finding that changed the experiment and must be stated.** At the production
hex layout the boundary condition is essentially invisible:

| reflector / shield | Δk (β: 0→0.8) | normalized flux change |
|---|---|---|
| 1 / 1 (production) | **0.5 pcm** | 0.065% |
| 1 / 0 | 1445 pcm | 11.08% |
| 0 / 0 | 5371 pcm | 29.15% |

The shield ring is optically thick, so nothing reaches the outer boundary. E4 run
on that geometry would have measured nothing while looking like a clean pass —
every model would "generalize" by ignoring the BC. **E4 therefore uses
`shield_rings=0`** (a reflector-only core), recorded in every sample's metadata.
✅ Measured on generated data: ~1410 pcm end to end.

**fhr is excluded**: the same sweep at the published vessel (bed → 60 cm graphite
reflector → barrel → downcomer → vessel) gives **1.4 pcm / 0.049%**. State this
rather than silently omitting fhr.

**Caveat to state:** sweeping β is a *synthetic* axis. Only one β reproduces the
real full-geometry solution; the rest are boundary conditions no physical reactor
has. Legitimate for testing generalization across a family of BCs — but it is not
a physical sweep.

---

## 8. Floors — publish these next to the results

### ✅ FNO rasterization floor (measured, model-independent)

FNO needs a uniform grid, so the pipeline is mesh → grid → FNO → mesh. Pushing
the **reference** flux through that round trip bounds FNO's error from below:

| grid | hex g1 / g2 | fhr g1 / g2 |
|---|---|---|
| 64² | 2.7% / 2.4% | 3.7% / **9.1%** |
| 128² | 1.2% / 1.1% | 1.8% / **4.7%** |
| 192² | 0.7% / 0.6% | 1.1% / **3.1%** |

The fhr thermal group is the sharp result: a 128² grid over a 420 cm domain is
3.3 cm per cell against 4 cm pebbles, so no FNO beats 4.7% thermal flux error
there regardless of training. Publishing the floor turns "FNO did worse" from
something attackable as rigged into a decomposition: *X points are the grid, Y are
the model*. FNO gets a grid sweep {64², 96², 128², 192²} so it competes at its
best configuration.

Mesh→grid is nearest-neighbour (cross sections are piecewise constant per
material; blending across an interface invents materials that do not exist);
grid→mesh is bilinear (flux is smooth — the generous choice for FNO).

### ✅ FEM discretization floor

Observed convergence order **p ≈ 1.0**, confirmed on levels 3/4/5/6 (successive
k_eff gaps 541 / 329 / 221 pcm; ratio 1.49 vs the p=1 prediction of 1.50).
Richardson extrapolation puts even L6 (N=102k) about **1100 pcm** from the
continuum, and hex01's L0 mesh about **4700 pcm** from L3. Labels are internally
self-consistent per mesh — fine for the ML benchmark — but this matters for any
comparison against continuous-energy transport.

### ✅ Reference solver cost (the E5 denominator)

| dataset | N | assemble | solve (splu + power iteration) | total |
|---|---|---|---|---|
| hex01 | 4,279 | 0.005 s | 0.318 s | **0.323 s** |
| fhr01 | 7,093 | 0.012 | 0.748 | **0.760** |
| hex_probe/L2 | 15,562 | 0.019 | 1.699 | 1.718 |
| hex_probe/L3 | 31,666 | 0.048 | 4.301 | 4.348 |
| hex_probe/L6 | 115,114 | 0.138 | 22.142 | **22.280** |

Assembly is 1.5–2% of total; the sparse LU dominates. Scaling **t ~ N^1.27–1.31**,
consistent with nested-dissection sparse LU in 2D. All CPU.

**Framing caution.** hex01 solves in 0.32 s, so the entire 7000-sample dataset
cost ~38 minutes of CPU. "Faster than the solver" is not a compelling story per
core. The honest argument is the **many-query** setting (design sweeps, UQ,
optimization — 10⁴–10⁶ solves) or the **large-mesh** setting (L6 at 22 s with
superlinear scaling). Also: do **not** quote a CPU-solver vs GPU-model ratio as a
method speedup without labelling it — that measures hardware.

---

## 9. Statements that must appear (honesty controls)

1. **All errors are extrapolation errors.** The generator holds train/val/test
   disjoint on control insertion (hex: 0–0.5 / 0.5–0.75 / 0.75–1.0; fhr: 0–5 /
   6–7 / 8–10 rods inserted). These are not i.i.d. holdouts, and absolute errors
   are higher than an i.i.d. split would give. The *between-model* comparison is
   the meaningful part. `report.py` emits this as a banner automatically.
2. **The GNO limitation, stated not glossed.** Discretization invariance in the
   GNO sense needs (i) a fixed physical integration domain and (ii) quadrature
   weights converging to the measure. Volume weighting supplies (ii). It does
   **not** supply (i) — a fixed-k kNN neighbourhood still shrinks as N grows. So
   this is the part of the theorem that matters over the 2–4× refinement the
   study spans, not the theorem itself.
3. **Cross-section provenance.** Every sample carries transport provenance
   (OpenMC code + version, data library, branch grid, per-branch k_eff, Monte
   Carlo uncertainty). There is no hand-tuned fallback library.
4. **PI-GNO-sum is a GNN.** Do not present the aggregation ablation as three
   variants of "our model"; present `sum` as the GNN-side control.

---

## 10. Suggested results structure for `main.tex`

1. **Accuracy (E1)** — table, 4 models × 2 reactors, data-only. Per-group flux,
   k in pcm, power, per-material breakdown. FNO row carries its floor.
2. **Physics-loss ablation (E2)** — λ_pde ∈ {0, 0.1, 1.0} on the graph models.
   This is where the "physics-informed" claim in the title is earned.
3. **Discretization transfer (E3)** — invariance table with the FEM reference as
   the first row; accuracy-vs-L6 with floors. Plus the aggregation ablation
   attributing the effect to volume quadrature.
4. **BC transfer (E4)** — error vs held-out β, with the `shield_rings=0`
   justification and the synthetic-axis caveat.
5. **Cost (E5)** — accuracy-vs-inference-time Pareto against the measured solver
   times, with the many-query framing.

An optional high-value experiment, cheap once E4 exists: **test-time physics
refinement.** At a new BC, `A` is known exactly and needs no labels, so PI-GNO can
take a few gradient steps minimizing the PDE residual on the test sample. No
data-driven baseline has that channel. It is the strongest single argument for
the physics-informed formulation.
