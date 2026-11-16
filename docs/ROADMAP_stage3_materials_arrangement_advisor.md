# RISC3D-AI — Stage 3 Design: Materials, Arrangement, Advisor

**Created 2026-08-29. Updated 2026-09-09** with heterogeneous per-block composition.
Scope expansion from "simulator + optimizer" to "design advisor". Sequenced against the
round-3 convergence work — read `NEXT_TASK_stage2_wiring.md` first.

## The requested capabilities

1. **Material-aware analysis** — material becomes a variable you compare, not a fixed
   constant, and a chiplet is a *stack of materials*, not one solid block.
2. **Arrangement system** — produce a real floorplan artifact under real constraints.
3. **Improvement suggestions** — rank how to make it cooler, by impact.

## SEQUENCING — read before building any of it

**A recommendation engine inherits every error in the solver and restates it as advice.**
If peak temperature is not mesh-converged, "moving the GPU saves 8 K" may be entirely a mesh
artifact — but it is now a confident sentence rather than a number someone might question.
That is strictly worse than reporting nothing.

| Build now (no solver dependency) | Build after convergence |
|---|---|
| Material library expansion (pure data) | Any quantified recommendation ("saves X K") |
| Per-block stack representation | Sensitivity *values* |
| Floorplan export format (pure I/O) | Ranked advice output |
| Adjoint sensitivity *machinery* | Material comparison deltas |

## 0. Where the temperature actually comes from (measured 2026-09-09)

This governs every design decision below. At 20×20 mm, 36 W, h=1000:

| | over full die | over one CPU footprint (4×4 mm, 10 W) |
|---|---|---|
| BEOL (8 µm, kz=2) | 0.36 K | 2.50 K |
| bump+underfill (40 µm, kz=20) | 0.18 K | 1.25 K |
| interposer Si (100 µm) | 0.06 K | 0.42 K |
| **convective floor `1/(hA)`** | **90.0 K** | 625 K *(zero-spreading limit)* |

**Vertical conduction through every layer is negligible.** The measured peak rise of 172 K
sits between the 90 K perfect-spreading floor and the 625 K zero-spreading limit, so
**spreading physics accounts for essentially the entire result** — 82 K of the 172 K.

Spreading is governed by *lateral* conductivity. Spreading length `λ = √(kt/h)`:

| layer | λ |
|---|---|
| interposer Si | 3.87 mm |
| die Si | 3.87 mm |
| **bump layer (kxy = 2)** | **0.28 mm** ← the blocker |
| BEOL (kxy ≈ 15) | 0.35 mm |

**Consequence:** per-block material composition is not cosmetic detail. It sets the
constriction term, which is half the temperature rise. Getting materials right *is* the
physics.

## 1. Heterogeneous per-block composition

### The gap

`assign_layer()` currently paints whole z-slabs uniformly, so every chiplet is a solid
silicon brick, and `add_power_source(z_slice=slice(z_bump_end, nz))` smears power through the
die's full thickness. A real die is a stack, and devices occupy a ~1 µm layer.

### `ChipletBlock` gains its own vertical stack

```python
ChipletBlock(
    name="CPU_Core", power_w=10.0,
    width_um=4000, depth_um=4000,
    stack=[LayerSpec("beol", 20), LayerSpec("silicon", 80)],
    power_plane="beol/silicon",   # planar source, not volumetric
)
```

### Power is a plane, not a volume

Devices live in a ~1 µm layer — sub-grid at any affordable `dz`. The standard package-level
treatment is a **planar heat source at the Si/BEOL interface**. This replaces the current
volumetric smear through the whole die.

### Flip-chip orientation

CoWoS is flip-chip: BEOL faces **down**, toward the microbumps, so all heat crosses it to
reach the interposer and sink. Bottom-to-top:

| layer | thickness | dz=20 | dz=10 | dz=5 |
|---|---|---|---|---|
| interposer Si | 100 µm | 5 | 10 | 20 |
| bump + underfill | 40 µm | 2 | 4 | 8 |
| **BEOL** | **20 µm** | 1 | 2 | 4 |
| bulk Si (die) | 80 µm | 4 | 8 | 16 |
| **total** | **240 µm** | **12** | **24** | **48** |

All thicknesses are multiples of 20 µm, so every interface lands exactly on a cell face at
all three ladder levels. **The refinement ladder survives this change untouched.**

### This breaks an existing test — deliberately

`test_peak_hotspot_is_at_die_top_farthest_from_heat_sink` asserts `pk == nz - 1`. That
encodes the current assumption that heat originates at the *top* of the die. Under flip-chip
with a planar source at the BEOL interface, the source sits near the *bottom*, close to the
sink. The test is not wrong — it correctly validates a modeling choice that is about to
change. **Update it deliberately; do not let it fail and then patch the assertion.**

### HBM is not silicon

HBM is 4–12 stacked DRAM dies with TSVs and microbumps between each layer. Effective vertical
conductivity is far below bulk Si. It needs its own composite material entry, not
`"silicon"`.

## 2. Material library

`MATERIALS` already carries anisotropic k (kx/ky/kz), rho, cp from config. What is missing is
treating material as a **comparison axis**.

### Priority comparison: CoWoS microbump vs SoIC hybrid bond

| Interface | CoWoS microbump + underfill | SoIC hybrid bond |
|---|---|---|
| kz | ~20 W/mK | ~150 W/mK |
| kxy | ~2 W/mK | ~150 W/mK |

Given section 0, the kxy jump from 2 → 150 is the headline: it removes the spreading blocker
entirely. Expect this to move the constriction term far more than any placement change.
Running one floorplan through both bonding technologies is a directly TSMC-relevant result
and the strongest single portfolio figure available here.

### Library additions

| Class | Options | k (W/mK) |
|---|---|---|
| BEOL | Cu/low-k stack | kz 1–3, kxy 5–20 |
| TIM1 | polymer / indium solder / graphite | 3–5 / 30–86 / ~400 in-plane |
| Underfill | capillary / thermally enhanced | 0.3–0.8 / 1–3 |
| Microbump | SnAg solder / Cu pillar | 50–60 / ~400 |
| Interposer | Si / glass / organic | ~150 / ~1.1 / 0.3–0.5 |
| HBM composite | 8-high DRAM + TSV + µbump | effective, derive |
| Mold compound | standard | 0.8–1.5 |
| TSV | Cu | ~400 |

All values are literature-range placeholders and **must be cited to a source** before
appearing in any output. Do not present them as measured.

## 3. Arrangement system

**Non-thermal constraints.** Real floorplans are constrained, not thermally optimal:

- HBM requires edge/beachfront placement for short PHY routing
- die-to-die links want adjacency (interconnect length, energy per bit)
- assembly requires minimum inter-block spacing (underfill flow)
- keep-out zones (power delivery, C4 fields, mechanical)

**Multi-objective cost:** `w₁·T_peak + w₂·wirelength + Σ penalties`, weights exposed so the
user can trade thermal against routing rather than getting a layout nobody can build.

**Real output artifact.** Floorplan as JSON/CSV — block, x, y, width, depth, power,
peak-in-block — not only a rendered picture.

**Placement lattice.** Block origins must snap to the coarsest ladder pitch (1000 µm) or
geometry is not identical across levels and GCI is invalid.

## 4. Advisor — adjoint sensitivity

### Why not brute force

Re-solving once per parameter costs N solves. With ~15 material properties plus layer
thicknesses plus `h`, that is prohibitive inside any interactive loop.

### The method

The steady-state system is **linear**: `A·T = b`. For a scalar output `f(T)`, solve the
adjoint **once**:

```
Aᵀ·λ = ∂f/∂T
```

then for every parameter `p`:

```
df/dp = λᵀ·( ∂b/∂p − (∂A/∂p)·T )
```

**One extra solve yields sensitivities for every material property, every layer thickness,
and `h` simultaneously.**

- `A` should be symmetric by construction (harmonic-mean face conductivities + Robin), so
  `Aᵀ = A` and the adjoint reuses the cached factorization from the forward solve. Verify
  symmetry first — if `solve_steady_state` uses `M = D⁻¹A` from the transient form, rescale
  to the symmetric form.
- Peak temperature is a `max`, non-differentiable. Use a smooth surrogate (log-sum-exp /
  p-norm) as `f`; report both the smooth value and the true max.

### Recommendation categories

| Category | Lever | Source |
|---|---|---|
| Placement | separate co-located hot blocks | SA + adjoint |
| Material | higher-kxy underfill, better TIM, hybrid bond | adjoint |
| Boundary | increase `h` (heatsink/coldplate class) | adjoint |
| Stack geometry | thin the interposer, thicken the die | adjoint |

Each recommendation carries a quantified ΔT **and** the mesh-convergence status of that
figure. A recommendation derived from an unconverged solve must say so.

## Build order

1. Finish round-3 convergence work (blocking for anything quantified).
2. Per-block `stack` representation + planar power source + flip-chip orientation.
   Update `test_peak_hotspot_is_at_die_top_farthest_from_heat_sink` deliberately.
3. Material library expansion + citation of every property value. HBM composite.
4. Floorplan JSON/CSV export + constraint representation.
5. Verify operator symmetry; implement adjoint reusing the cached factorization.
6. Sensitivity ranking output.
7. Multi-objective SA cost with wirelength + constraints.
8. Advisor text generation, convergence status attached to every number.
9. CoWoS-vs-SoIC comparison study as the headline portfolio figure.
