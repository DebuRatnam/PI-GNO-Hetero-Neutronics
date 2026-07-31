# PI-GNO handoff — 2026-07-30, hex data generation COMPLETE

Read `CLAUDE.md` first; it is authoritative on physics conventions. This file records
what changed, what exists, and what is left.

**The pipeline is code-complete and the hex reactor is done.** The previous handoff's
open problem ("the hex core is subcritical everywhere") is RESOLVED — root cause found,
fixed, and verified against continuous-energy transport. Do not redesign anything.

---

## Environment (already set up — install nothing, download nothing)

```
/opt/homebrew/Caskroom/miniforge/base/envs/openmc-env/bin/python
  python 3.11.15, openmc 0.15.3, numpy 2.4.6, scipy 1.17.1  (no torch)
```

Not `/usr/bin/python3` (no openmc — that is the training env). Not conda `base` (empty).
Do **not** create a project-local `.venv`.

Nuclear data, auto-discovered by `openmc_models.prepare_openmc_env()`:

```
~/nucdata/endfb-viii.0-hdf5/cross_sections.xml   ENDF/B-VIII.0
~/nucdata/chain_endfb80_fast.xml                 hex
~/nucdata/chain_endfb80_thermal.xml              fhr
```

Machine: 14 cores, 36 GB. OpenMC saturates ~13 cores; do not run two transport jobs
concurrently. Pure-NumPy generation is single-core and coexists fine with transport.

---

## RESOLVED: why the hex core was subcritical

The previous handoff blamed radial leakage. **That was wrong.** Evidence:

- OpenMC leakage fraction was **1.4%**; the FEM neutron balance (closes to 4e-10) put
  radial leakage at **0.50%** of total losses. Leakage cannot cost 0.4 in k.
- Actual cause: `fuel_rings=4` was a **reduced dev core** — 91 assemblies, of which only
  **24 were fuel**. The 13 control assemblies (9 primary + 4 secondary) are a docketed
  Natrium count, so they occupied 13 of 37 central positions: **35% of the active region
  was absorber/sodium-follower instead of the ~10% a real Natrium core runs.**
- The balance at `fuel_rings=4` put only 50.6% of losses in fuel absorption, 13.8% in the
  B4C shield, 28.6% in axial leakage.

How it survived so long: `fuel_rings=4` and the docketed control count were set together
in the initial commit; commit `8e94b12` ("Align core specs with NRC/gFHR documentation")
audited the control count and certified it as docketed but never touched `fuel_rings`
three lines above; and until OpenMC constants landed, hand-tuned cross sections made any
core map produce a plausible k. The config comment was also stale and wrong — it claimed
`fuel_rings=4 -> 127 assemblies` when the code builds 91.

**Fix:** `fuel_rings` 4 -> 7 (217 assemblies, 114 fuel). Transport at the same branch
state went **0.87116 -> 1.12136**. Also `enrichment_boundary_ring` 2 -> 4, chosen on
radial power FLATTENING (peak/avg 1.674, the only monotone profile) rather than on k.

More reflector rings does NOT help — it moves the sink (shield absorption 13.8% -> 2.5%
but reflector absorption 4.0% -> 10.5%, net +0.02 in k). SS316H is a poor fast reflector.

## RESOLVED: axial buckling was a bare-slab value

`AXIAL_BUCKLING_CM2` was `(pi/104)^2 = 9.1e-4`, i.e. ~100 cm active height plus a few cm
extrapolation — the buckling of a **bare** slab. A Natrium-class SFR carries an axial
reflector and sodium plenum, and reflector savings are large in a fast core because D is
large (1.8 fuel, 4.6 sodium). Now `(pi/135)^2 = 5.4e-4` (~15 cm savings per side).

Derived from geometry, **not** fitted to k. It costs ~23% in k; with the bare value every
dataset state was subcritical (fresh/rods-out label 0.942) and the reactor could not be
brought critical at any state in the dataset.

---

## Current state — hex is DONE

| artifact | status |
|---|---|
| `xs_natrium.json` | REGENERATED at the new core map. 12 branches, `converged=True`, max σ **2.81%** (secondary_control; fuel 0.87–0.93%). Branch k spans 0.947–1.121. |
| `depletion_natrium.json` | Unchanged — unit cell per fuel zone, core size does not enter. |
| `datasets/hex01` | **7000 samples** (5000/1000/1000), all converged, residuals ≤1.09e-8, 6.5 GB. k 0.922–1.048, **40.2% supercritical**. |
| `validation_hex.csv` | 6 CE-transport comparison points. |

Dataset k by split: train 0.960–1.048 (mean 1.0026), val 0.946–0.967, test 0.922–0.965.

### Physics coherence checks (all independent, all pass)

- Rod worth **~9,350 pcm** vs burnup swing **~6,237 pcm** (2->60 MWd/kg) — the control
  system can hold the cycle with shutdown margin. On the old core rod worth was ~38,000 pcm.
- Leakage fraction 0.44–0.58%.
- Train split straddles criticality, mean 1.0026.

### Validation vs continuous-energy transport (`validation_hex.csv`)

| bu | rod | k_diffusion (Bz²=0) | k_openmc | d_rho | shape RMS | pow-wtd |
|---|---|---|---|---|---|---|
| 2 | out | 1.19907 | 1.11891±100 | +5975 | 6.58% | 4.99% |
| 2 | in | 1.08625 | 1.01127±104 | +6826 | 6.64% | 5.47% |
| 30 | out | 1.16814 | 1.08655±88 | +6428 | 6.08% | 4.60% |
| 30 | in | 1.05743 | 0.98238±95 | +7225 | 6.58% | 5.27% |
| 60 | out | 1.13312 | 1.04836±100 | +7136 | 6.59% | 4.65% |
| 60 | in | 1.02550 | 0.94711±90 | +8071 | 6.39% | 5.00% |

- **Radial power shape: 6.5% RMS (5.0% power-weighted).** Good agreement.
- **k bias: +6,943 pcm mean**, monotone +5975 -> +8071. Real and systematic — grows with
  burnup (spectrum shift from Pu buildup) and with rod insertion (diffusion cannot resolve
  flux depression at black absorbers). This is the honest cost of G=2 in a fast core.
  REPORT IT; do not bury it.
- **Rod worth −8 to −9%** vs transport, consistently. (This supersedes the "+12.1%" in the
  old handoff, which was smoke statistics on the old undersized core.)

---

## Bugs found and fixed this session

1. **`validate_openmc._radial_power_diffusion` binned NODAL power into cylindrical
   rings** while the CE side is a `CylindricalMesh` tally, i.e. a continuum integral.
   FEM nodes cluster at assembly centres, and the 14.0 cm ring width is incommensurate
   with the 18.7 cm assembly pitch, so nodal binning ALIASED against the lattice and
   produced an alternating-sign ring error (+31%, −17%, +5%, −22%, ...). It reported
   **40.2% RMS shape error that was pure sampling artifact.** Now integrates over
   triangles (area × element-mean power): same state reads **6.58%**.
2. **`_shape_error` gained a power-weighted RMS.** Plain RMS gave the near-empty fuel-edge
   ring (~1% of core power, 108% relative error) the same weight as a ring carrying 19%.
3. **`xs_openmc.py` and `validate_openmc.py` shared workdirs between reactors.** Both use
   the same branch keys with `--workdir` defaulting to `openmc_run`, and `--resume` globs
   for any statepoint, so a hex run could load an fhr statepoint. Workdirs are now
   reactor-keyed. (An fhr statepoint was in fact sitting in `openmc_run/bu0_T900_rod-in`.)

Also deleted the stale `openmc_run/` (249 MB) — every statepoint predated the core map.

## Things I got wrong mid-session (recorded so they are not re-derived)

- A Doppler coefficient computed from branch k at bu=2 (K_D ≈ −0.0043) was reading signal
  out of noise. At 6000p×120b, σ ≈ 100 pcm and the effect is ~200 pcm; bu=30 shows +0.00001
  across the same ΔT. **The branch grid cannot support a Doppler claim in the paper.**
  That needs correlated sampling / perturbation theory. The temperature AXIS is still
  valid — it changes the collapsed constants, which is what gets stored.

---

## Known, ACCEPTED limitations (decided — do not silently "fix")

**fhr thermal cutoff stays at 0.625 eV.** Decided deliberately; regeneration is ~2.5 h and
was judged not worth it. The concern is real and quantified — at 823–1100 K, kT =
0.071–0.095 eV, so 0.625 eV sits at only ~6.6–8.8 kT and cuts inside the Maxwellian
(tail runs to ~20–30 kT, i.e. ~1.9–2.5 eV). Measured in `xs_fhr.json`: thermal->fast
up-scatter is **55–63% of down-scatter** for pebble/coolant materials, and for
`graphite_pebble` Ss21 is **93% of thermal absorption**.

**Why it is nonetheless defensible:** fhr does NOT assume down-scatter-only.
`geometry_pebble` supplies Ss21 and `operators.assemble_AF` adds it to `A` as a
thermal-group removal plus a fast-group in-scatter source, so the full 2×2 scattering
matrix is solved exactly. The up-scatter is transported, not neglected. The rationale is
now written out in `xs_common.GROUP_BOUNDARIES_EV` — the old comment cited the cadmium
cutoff as if the LWR convention transferred, which it does not.

If it is ever raised: chi is unaffected (Watt spectrum below a few eV is ~1e-10 of
births, so `materials_fhr.CHI` stays `(1.0, 0.0)`), but every collapsed constant changes,
so `xs_fhr.json` must be regenerated in the same change.

**Split k-ranges are near-disjoint.** train ≥0.960, test ≤0.965. Splits are disjoint on
`insert_fraction` by design and k tracks rod insertion, so the `k_hat` head EXTRAPOLATES
on test. This is an intentional generalization test — **disclose it in the paper** rather
than let a reviewer find it. Test k_eff error will read worse than train for this reason
alone. val spread is narrow (σ=0.0043) because val pins `eb=4` and `reflector_rings=1`.

---

## Remaining work

**1. fhr validation — the biggest publication gap.** hex now has k bias, power shape and
rod worths vs CE transport; fhr has NONE. Costs like an fhr branch grid.

```
cd data_generation/p1_fem
PY=/opt/homebrew/Caskroom/miniforge/base/envs/openmc-env/bin/python
$PY validate_openmc.py --reactor fhr --out validation_fhr.csv \
    --depletion depletion_fhr.json --burnups 2 95 190 --particles 10000 --batches 200
```

Run under `nohup`, on AC power, lid open. Verify flags with `--help` first.

**2. fhr dataset.** `datasets/fhr01` does not exist yet. ~2.1 s/sample at the old node
count; the fhr core was not resized, so that estimate should still hold.

```
$PY generate.py --reactor fhr --out ../../datasets/fhr01 \
    --train-samples 1100 --val-samples 200 --test-samples 200
```

**3. A G=4 or G=8 run for one reactor.** The +6,943 pcm hex bias is the strongest
reviewer target. The schema is group-count-driven (`PhysicsConfig.n_groups`), so this is
supported; showing the bias shrink with G largely defuses the "two groups is too coarse
for a fast reactor" objection.

**4. Training + the actual paper.** Ablations of `lambda_PDE` / `lambda_BC` (the title
claim), a no-physics-loss GNN baseline, honest wall-clock inference vs the `splu` solve,
and per-group flux / power / by-material error reporting.

---

## Operational gotchas

- **Run long jobs under `nohup`.** A background shell started by Claude Code is torn down
  when that process exits. `xs_openmc.py --resume` exists because of this.
- **Check power first** (`pmset -g batt`). Never arm `caffeinate` on battery — it turns a
  graceful sleep into a dead-battery shutdown. Nothing prevents clamshell sleep; lid open.
- **Keep smoke/test tables out of the repo.** `xs_*.json` beside the material modules
  becomes the de facto library and they are deliberately NOT gitignored. To test against a
  table elsewhere, monkeypatch `materials.BRANCH_TABLE_PATH` / `materials_fhr.BRANCH_TABLE_PATH`
  before importing `generate` or `validate_openmc` (the load is lazy).
- `datasets/` IS gitignored. hex01 is 6.5 GB and regenerable in ~50 min.
- OpenMC prints thousands of benign `WARNING: Negative value(s) found on probability
  table for nuclide ...` lines for trace fission products with depleted isotopics. Filter
  them out of logs; they are not errors.

## Guardrails — never suppress

- `load_branch_table` warns when `max_rel_std > 5%`; `converged` is a strict max over every
  material and σ entry. Both current tables pass honestly.
- `check_axis_coverage` warns when a sampled range escapes the branch grid.
- `require_branch_table` rejects a table whose reactor type, group count, or group
  boundaries disagree with `xs_common.GROUP_BOUNDARIES_EV`.
- Never reintroduce hand-tuned cross sections, and never tune XS or enrichments to hit a
  target k_eff. If the core map is wrong, change it as GEOMETRY and regenerate the table —
  the transport core must remain the FEM core.
