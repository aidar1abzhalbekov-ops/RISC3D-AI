# RISC3D-AI — Round 3 Task Brief: Consistent Scale, Then Convergence

**Updated 2026-08-29.** Supersedes the round-2 brief.

## BLOCKER 1 — Stage 2 fails on every run: catalog and default grid are 13× apart

```
[FATAL] Pipeline aborted: Stage 2 placement optimization failed: Chiplet 'CPU_Core'
(4000 x 4000 um) does not fit within the 1500 x 1500 um die (60x60 voxels at
dx=25.0 dy=25.0 um).
```

**My error.** The realistic µm catalog I supplied (4000 µm CPU core, 11000 µm HBM) implies a
~20 mm package. `main.py` still defaults to 60×60 at dx=25 µm — a 1.5 mm die. The catalog and
the default grid describe different objects, so Stage 2 cannot succeed at any iteration count.

The error message itself is correct and well-formed — block name, both dimensions, die size,
voxel/dx derivation. Nothing to fix there. But it fires at Stage 2 *runtime*, after Stage 1
has already run; see the exit-code item below.

### Verified consistent defaults — 20 × 20 mm package

| | current | new |
|---|---|---|
| nx, ny | 60, 60 | **40, 40** |
| dx, dy | 25 µm | **500 µm** |
| nz | 40 | **24** |
| dz | 10 µm | 10 µm (unchanged) |
| domain | 1.5 × 1.5 × 0.4 mm | **20 × 20 mm × 240 µm** |

38,400 unknowns — the same grid size the Stage 2 runs already use.

| Block | W | µm | voxels @ dx=500 | mm² | W/cm² |
|---|---|---|---|---|---|
| CPU_Core | 10 | 4000 × 4000 | 8 × 8 | 16.0 | 62.5 |
| GPU_Core | 14 | 6000 × 5000 | 12 × 10 | 30.0 | 46.7 |
| HBM_Memory | 4 | 11000 × 11000 | 22 × 22 | 121.0 | 3.3 |
| NPU_Accelerator | 8 | 5000 × 5000 | 10 × 10 | 25.0 | 32.0 |

Die 400 mm², catalog 192 mm² = **48% fill**. Predicted peak **~115 °C**
(R_conv = 1/(h·A) = 2.5 K/W, ΔT = 90 K at 36 W).

**Stack layers: 100 / 40 / 100 µm** replacing `interposer_frac` / `bump_frac` — 100 µm thinned
Si interposer, 40 µm microbump + underfill, 100 µm thinned die. Total 240 µm = nz 24 × dz 10.

Caveats to state rather than hide: dx = 500 µm is coarse for gradients *within* a chiplet —
defensible for placement comparison since systematic error largely cancels, but it is a
tradeoff, not a free choice. And HBM alone occupies 30% of die area, so SA has less freedom
than 48% fill implies. Realistic (HBM does dominate CoWoS floorplans) but it caps achievable
improvement.

## BLOCKER 2 — stack layers still derived from grid units

Reported peak 145–154 °C, ~16% spread, `dz` dominating. **That spread is not (only)
discretization error.** Layer boundaries come from `round(nz * interposer_frac)` and
`round(nz * (interposer_frac + bump_frac))`, so realized bump thickness is a rounding artifact:

| nz | bump voxels | realized frac | vs nominal 0.15 |
|---|---|---|---|
| 12 | 2 | 0.1667 | +11.1% |
| 16 | 3 | 0.1875 | **+25.0%** |
| 20 | 3 | 0.1500 | 0% |
| 24 | 3 | 0.1250 | **−16.7%** |
| 28 | 4 | 0.1429 | −4.8% |
| 32 | 5 | 0.1562 | +4.2% |
| 40 / 60 / 80 | 6 / 9 / 12 | 0.1500 | 0% |
| 48 | 7 | 0.1458 | −2.8% |

(Executed and verified against independent hand computation.)

The composite layer carries the stack's dominant thermal resistance (`kz ≈ 20` vs silicon
`~150`, ~7.5× more resistive per unit thickness), so a −17%…+25% thickness swing moves the
peak directly. **nz=24 is the worst point in the table**, and its −16.7% sits close enough to
the observed ~16% spread to be worth ruling in or out. The 100/40/100 µm authoring above
fixes this.

**Ruled out:** the first-order Robin boundary (`coeff = h/dz`, cell-centre temperature as
surface temperature) carries a half-cell conduction resistance of `dz/(2k) ≈ 3×10⁻⁸` against
a convective `1/h = 10⁻³` — five orders down. Not the `dz` sensitivity.

## Round 3 tasks, in order

### 1. Make scale consistent

Apply the defaults table above. Locations: parser defaults and interactive-prompt defaults in
`main.py`; `GridSection` defaults and `StackSection` (fracs → µm thicknesses) in
`config_loader.py`. `DEFAULT_CHIPLET_CATALOG` is already in µm. Confirm Stage 2 inherits
`dx`/`dy`/`dz` from the same `GridSection` Stage 1 uses rather than carrying its own.

### 2. Refinement study

With 100/40/100 µm layers (gcd 20 µm) and the µm catalog, every chiplet edge and layer
interface lands exactly on a cell face at every level:

| mesh | unknowns | dx | dz | exact |
|---|---|---|---|---|
| 40×40×24 | 38,400 | 500 µm | 10 µm | ✓ |
| 80×80×48 | 307,200 | 250 µm | 5 µm | ✓ |
| 160×160×96 | 2,457,600 | 125 µm | 2.5 µm | ✓ |

r = 2 throughout. **The study is three solves of one fixed layout**, not three SA runs — take
the best layout found and re-solve it, so the 2.4M-unknown mesh is a one-off outside the
anneal.

Print realized layer thicknesses for every mesh so a geometry change cannot hide in the
result. Report observed order `p` and a Grid Convergence Index band (ASME V&V 20 / Roache),
not a min–max range. `148.3 °C, GCI 2.1%, p = 1.94` is a portfolio-grade claim; `145–154 °C`
is an admission of not knowing.

### 3. Only then, iteration count

Deferred and still correct, but pointless before the number converges: measure per-proposal
wall clock, parameterize cooling schedule α by N, re-baseline the headline against an
adversarial corner-clustered layout rather than the shelf-packed seed.

## Still open

- **`git init` — urgent.** Three sessions unversioned, concurrent editing confirmed (a stray
  `wq` broke `optimizer.py`'s import mid-session; self-corrected in ~1 min, and the 1317-line
  delivered version was correct before and after). A `SyntaxError` is the benign collision;
  one that stays valid and changes a number is the expensive one. Move out of
  `C:\Users\mimat\.vscode\extensions\` first (VS Code removes non-conforming folders there),
  `git init`, commit current state **before** any round-3 edit.
- Catalog-fit check moves to config-validation time, exit 2, before Stage 1 runs. Distinct
  exit 3 for runtime Stage 2 failure after Stage 1 succeeded. Never exit 0 on failure.
- `TimingReport.optimize_s`.
- Confirm `TestChipletPlacementOptimizer` landed and the regression-injection red/green pass
  was shown.

## Resolved earlier

- **176.89 °C — closed, my call was wrong.** I suspected a lost cell-volume conversion in
  `q_map`. Backing `dx` out gives 385 µm, reproducing 151.9 K exactly at 47–68 W/cm². Physics
  was sound.
- **`SolverSection` is not stale.** Seven fields at `config_loader.py:80-87`: `method`,
  `steady_state`, `total_time_s`, `dt`, `safety_factor`, `record_steps`, `max_steps`. Earlier
  docs (mine included) listed six and omitted `total_time_s`.
