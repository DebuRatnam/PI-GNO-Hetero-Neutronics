# Handoff: remaining CPU-only work on the benchmark suite

Everything here runs **without a GPU**. The only GPU-bound work is the 105
training runs themselves (see `HANDOFF_BENCHMARK_PAPER.md` for what those are).

Branch: **`benchmark-suite`** (off `openmc-group-constants`). All work below is
committed and pushed.

---

## 1. Environments — three, and they must not be mixed

| env | what | why |
|---|---|---|
| `/opt/homebrew/Caskroom/miniforge/base/envs/openmc-env/bin/python` | py 3.11.15, openmc 0.15.3, numpy, scipy, **no torch** | ALL data generation. `xs_openmc.py` must never import torch. |
| `/opt/homebrew/Caskroom/miniforge/base/envs/pigno-bench/bin/python` | py 3.11.15, torch 2.13.0, physicsnemo 2.1.1, torch_geometric 2.8.0, torch_scatter 2.1.2 | ALL benchmark/model code. Created for this work. |
| `/usr/bin/python3` | py 3.9.6, torch 2.8.0 | legacy; runs `src/` and most of `benchmarks/` but **cannot** host PhysicsNeMo (needs 3.11–3.13) |

Benchmark code is kept 3.9-compatible so the legacy env still works for
everything except the FNO/MeshGraphNet wrappers.

### Install gotchas (both cost a failed run to discover — see `requirements.txt`)

1. **MeshGraphNet needs `torch_scatter`**, which neither physicsnemo nor
   torch_geometric pulls in. It is *not* an import-time failure: the constructor
   succeeds and the **first forward** raises. A smoke test that only builds the
   model passes.
2. **`torch_scatter` needs `--no-build-isolation`.** Its `setup.py` imports torch
   at build time; pip's isolated build env has none, so a plain install dies with
   `ModuleNotFoundError: No module named 'torch'` from inside the build.

---

## 2. Datasets currently on disk (`datasets/`, gitignored)

| dataset | contents | N per core | size |
|---|---|---|---|
| `hex01` | 7000 (5000/1000/1000), Natrium hex, G=2 | 3421 or 4279 | 6.5 GB |
| `fhr01` | 7000 (5000/1000/1000), KP-FHR pebble bed, G=2 | ~7100 | 35 GB |
| `hex_probe` | 30 paired configs × 6 meshes = 180, test split only | 3850–115114 | 1.1 GB |
| `hex_res/L2` | 400/100/100, the E3 training set | ~15562 | 1.8 GB |

`hex_probe` meshes: `L0, L1, L2, L3, Lmixed3_1_3, L6`. Same physical cores at six
discretizations — pairing is asserted by `verify_pairing`.

**Not yet generated:** `hex_bc` (the E4 boundary-condition dataset).

---

## 3. Sharded generation (built and verified)

Sample generation is embarrassingly parallel, but only because the generators
were restructured to derive a **per-sample seed from the index**. `generate.py`
still seeds one rng per split and draws inside the loop, so it is NOT shardable —
do not add `--shard` to it without changing that, and note that changing it would
break `hex01`/`fhr01` reproducibility.

```bash
# SLURM array (fill in account/partition/qos in the sbatch header first)
sbatch --array=0-39 slurm_generate.sbatch bc  ../../datasets/hex_bc --train 2000 --val 500 --test 500
sbatch --array=0-29 slurm_generate.sbatch res ../../datasets/hex_probe --levels 0 1 2 3 mixed:3,1,3 6 --eval 30

# ONCE, after the array completes — merges shard manifests and verifies no gaps
python sharding.py ../../datasets/hex_bc --expected 3000
```

Design points, all deliberate:
- Shards are **round-robin**, not contiguous blocks: per-sample cost varies ~70×
  across mesh levels, so blocks would leave stragglers.
- The resolution generator shards whole **configurations**, never single levels —
  paired levels must be produced together or the invariance metric has nothing to
  compare.
- Manifest merge is a **separate step**. A shard writing into `manifest.json`
  races the others and silently drops rows. The merge also verifies every
  referenced `.npz` exists, because an array task that hits the wall clock dies
  quietly.

**Verified bit-for-bit**: 3 parallel shards reproduce the serial run exactly
(node features, flux, A, F, k_eff, metadata, manifest).

---

## 4. Remaining CPU work, in priority order

### 4a. fhr BC sensitivity sweep — **blocks a scoping decision** (~1 h)

E4 (boundary-condition transfer) is currently **hex-only**. Reason: at fhr's
published vessel (bed R=120 → 60 cm graphite reflector → barrel → downcomer →
vessel R=191), sweeping albedo β from 0 to 0.8 moves k_eff by **1.4 pcm**. The
boundary is invisible; an experiment there would measure nothing.

The same problem existed on hex and was solved with `shield_rings=0` (measured:
0.5 pcm → 1445 pcm). fhr has no equivalent knob.

**Option A** — add a `domain_outer_r` truncation knob to
`geometry_pebble.make_pebble_core`, clamping the reflector fill and skipping the
vessel fill. Then put the albedo boundary at, say, R=140. This is the *textbook*
use of an albedo BC (representing a reflector you don't mesh), not a hack.
Boundary extraction is already generic (`_free_boundary_edges` reads free edges
off the triangulation), so no boundary surgery is needed.

**Hard constraint:** control elements sit at `R_bed + control_offset` = 127.9 cm
with radius 2.6, so their outer edge is **130.5 cm**. Truncating below that
deletes the control rods and destroys the rod-insertion variability the whole
dataset depends on. R=131 is the floor.

**Do this first:** sweep candidate radii {140, 135, 131}, measure Δk(β: 0→0.8)
and normalized flux change at each. Target ~1000+ pcm (hex gets 1445). If all
three come back near 1.4 pcm, Option A is dead → E4 stays hex-only (Option B, zero
work). ~1 h to know; ~4 h more if it passes.

### 4b. Generate `hex_bc` (~1.5 h serial, ~3 min as a 40-task array)

```bash
python generate_bc.py --out ../../datasets/hex_bc --train 2000 --val 500 --test 500
```
Already implemented and smoke-tested. Uses `shield_rings=0` (documented in the
module docstring with the measurements justifying it). β is the only axis held
disjoint across splits: train [0, 0.40), val [0.40, 0.55), test [0.55, 0.80].
Everything else (rod insertion, enrichment, burnup, temperature) is i.i.d. across
splits, so a generalization failure is attributable to the boundary.

### 4c. Build tensor caches — **must live where training runs**

```bash
python benchmarks/data.py --data datasets/hex01
python benchmarks/data.py --data datasets/fhr01
```
Measured: 2.4 MB/sample (hex01) and 5.3 MB/sample (fhr01) → ~17 GB + ~37 GB.
Check free space. Put these on cluster scratch, not home.

Why: `dataset.load_sample` decompresses an npz and re-`eval`s a `repr`-ed
metadata dict on every access, and `src/train.py` does that inside the epoch loop.

### 4d. Add reactor-physics metrics to `src/metrics.py`

Current metrics are ML-flavoured (relative L2 over all nodes). A reactor-physics
reviewer will want:

- **k_eff in pcm** — `k_abs_err = 0.000954` is 95 pcm; pcm is the convention.
- **Power peaking factor error** (max/mean power density). The most
  safety-relevant number; a model can have 1% power L2 and get the peak wrong.
- **L∞ / max nodal error**, not just L2 — L2 hides local blowups.
- **Assembly-wise radial power error in %** — how C5G7 and transport benchmarks
  report.
- **Control rod worth** (Δρ between rods-in / rods-out) — tests reactivity, not
  just flux shape. Needs a paired-sample pass; the generators can produce pairs.

First three are a few lines each. Rod worth needs a small harness addition.

### 4e. G=4 hex dataset (optional, hours)

Another session landed `xs_natrium_g4.json` (commit `8e3ea42`). The benchmark
suite is already G-generic — `NodeLayout` derives width from metadata,
`pack_group_major` / `pde_residual` / the batched operator remap all read G from
the data. A G=4 hex dataset would flow through unchanged. Free capability.

### 4f. Reduced CPU training run (optional but recommended)

You *can* train on CPU at reduced scale — hex01 is N=4279 and a budget-matched
model is 400k params. A 500-sample, 30-epoch run is hours, not days. Useless as a
headline result, but it surfaces real bugs (loss plateaus, LR grid behaviour,
transfer plumbing) before GPU allocation is spent. Do it once.

---

## 5. Invariants that must not break

1. **`hex01` and `fhr01` must regenerate bit-for-bit.** Verified repeatedly after
   every generator change. `hex_subdiv` defaults to 0 for exactly this reason (it
   was declared as 2 while dead code; reading it at that default would have
   silently re-meshed both datasets).
2. **`openmc-env` stays torch-free.** The per-sample generator must never call
   OpenMC, and the XS pipeline must never import torch.
3. **`benchmarks/test_batching.py` is blocking.** Run it after any change to
   `src/` or `benchmarks/batching.py`. It checks batched vs per-sample residual
   with distinct k per graph, A/F nnz conservation, KHead across all pools, full
   PIGNO forward across all norms, and batch-of-one metric identity. If batching
   is wrong, every downstream number is wrong while still looking plausible.
4. **Reference solutions are self-consistent per mesh, not converged.** Observed
   FEM convergence order is p ≈ 1.0 (confirmed on levels 3,4,5,6); Richardson puts
   even L6 (N=102k) about **1100 pcm** from the continuum, and `hex01`'s L0 mesh
   about 4700 pcm from L3. Harmless for ML benchmarking; matters for any claim
   against continuous-energy transport.

---

## 6. Bugs found and fixed (do not reintroduce)

- `heads.KHead` pooled with `h.mean(0)` over every row → one k per *batch*
  instead of per core.
- `norm.GraphNorm` took statistics over every row → a core's prediction depended
  on what it was batched with.
- `physics.pack_group_major` hardcoded G=2; `pde_residual` assumed a scalar k.
- `budget.py` `KNOB_STEP` of 4 for FNO → params scale with `latent_channels²`, so
  it jumped 284k straight past a 400k target; the matcher correctly *refused*.
  Step is now 1 throughout.
- `metrics._rel_l2` clamped the denominator at `eps=1e-12`, but power density has
  norm ~2.5e-13 — **below the clamp**. Every `power_rel_l2` was divided by a
  constant instead of the true norm (~4× too small). No absolute epsilon is safe
  when fields span 1e-14 to 1e+2.
- `harness`/`experiments` stdout was block-buffered when redirected → a running
  job looked identical to a hang for 7 minutes. Both now line-buffer.

---

## 7. Quick verification commands

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/pigno-bench/bin/python
$PY benchmarks/test_batching.py                       # blocking correctness
$PY benchmarks/harness.py --list-models               # should list all 4
$PY benchmarks/budget.py --data datasets/hex01 --target 400000 \
     --models pigno mgn fno deeponet                  # all within tolerance
$PY benchmarks/experiments.py --list                  # 35 cells
$PY benchmarks/reference_timing.py --data datasets/hex01 --n 10
```
