# RISC3D-AI — Validation Plan

**Created 2026-09-09.** Answers the question "are the outputs accurate for a real chiplet?"

**Current answer: no.** The solver is *verified* (residual 1e-11, energy imbalance 2e-9 K,
superposition 6.5e-9 K — it solves the given equations correctly). It has never been
*validated* — compared against a measurement or an independent reference. Only the second
licenses any accuracy claim. This document is how to earn one, using sources already in
this project.

---

## Part 1 — Modeling gaps that must close first

Validating a model that is structurally wrong just measures the wrongness.

| gap | current | reality |
|---|---|---|
| **heat sink face** | bottom (interposer underside), top adiabatic | CoWoS cools through the **top**: die backside → TIM1 → lid → TIM2 → heatsink. The downward path through µbumps → interposer → C4 → substrate is secondary and high-resistance. **The model is inverted.** |
| **h** | 1000 W/m²K = R 2.50 K/W = passive heatsink, still air | active air ≈ 5,000; cold plate ≈ 25,000; microchannel ≈ 83,000. H100-class measures ≈ 5,100. |
| **mesh** | nz=12, level 1 of {12,24,48} | refinement study never run; ~16% spread measured, dz-dominated |
| **block composition** | uniform silicon | BEOL + device plane + bulk Si; HBM is a DRAM/TSV/µbump stack, not silicon |
| **power distribution** | volumetric through full die thickness | ~1 µm device plane at the Si/BEOL interface |
| **material values** | `bump_underfill_composite` kz=20 / kxy=2 | invented placeholders — see Part 3 |

### Note on power and cooling being a matched pair

`h=1000` with `P=12 W` is internally consistent — a 12 W passively-cooled part at ~80 °C is
a real operating point. Raising power without raising `h` produces nonsense:

| config | peak |
|---|---|
| h=1000, 12 W | ~55 °C floor — realistic |
| h=1000, 36 W | ~115 °C floor — not a real passive part |
| h=1000, 200 W | ~525 °C — garbage |
| h=5,000, 100 W | ~75 °C — realistic (active air) |
| h=25,000, 150 W | ~40 °C — realistic (cold plate) |

**Ship two named presets, not one default:** `passive-lowpower` (12 W, h=1000) and
`cowos-coldplate` (150 W, h=25000). Print W/cm² and R_ja in the header so the regime is
never implicit.

### Consequence worth stating as a finding

Spreading length is `λ = √(kt/h)` — h in the denominator, so **better cooling decouples the
chiplets**:

| cooling | h | λ | coupling at 6 mm |
|---|---|---|---|
| passive (now) | 1,000 | 3.87 mm | 21.2% |
| active air | 5,000 | 1.73 mm | 3.1% |
| cold plate | 25,000 | 0.77 mm | ~0% |

The measured 13.09 °C improvement exists *because* h=1000 is weak. Under aggressive cooling
the optimizer's benefit shrinks toward zero. That is a real result, not a defect:
**chiplet placement is a thermal lever only in coupled, weakly-cooled packages.** Report it
with the λ table behind it — it is a stronger claim than any single percentage.

---

## Part 2 — Reproducible benchmark (dac143.pdf)

Published numbers RISC3D-AI can be run against directly.

**Table 6, FFT benchmark — max temperature above room:**

| | footprint (µm) | Si area (mm²) | max ΔT (°C) |
|---|---|---|---|
| 2-tier, 3DFP | 1005 × 735 | 1.48 | 27.63 |
| 2-tier, theirs | 867 × 849 | 1.47 | **27.36** |
| 3-tier, 3DFP | 972 × 518 | 1.51 | 45.81 |
| 3-tier, theirs | 891 × 560 | 1.50 | **42.84** |

**Table 5, ind_ckt 3-tier** — non-thermal-aware 40.89 °C above room → thermal-aware
32.03 °C = **22% reduction**. Their reported temperature ranges: [47, 68] °C and
[63, 79] °C.

**Acceptance:** run the FFT case and land within a few °C of 27.36 / 42.84. Agreement
validates; disagreement localises the modeling error. Either outcome is worth having.

**Cross-check on the optimizer:** their 22% reduction against RISC3D-AI's
13.09/83.46 = **15.7%**. Same order, same mechanism — they also report the floorplanner
"avoids stacking of high power density blocks and forces such blocks to tiers closer to the
heat sink." This does not validate absolutes, but indicates the captured physics is right.

**Runtime context** (dac143 Table 4, 3-tier, 1.3×1.3 mm): their compact model 0.00022 s;
HotSpot 59.5 s; GDS-level FEA 1082 s. RISC3D-AI's 23 ms cached-LU solve and 37,000
layouts/s via superposition sit sensibly between compact-model and full-FEA territory.

---

## Part 3 — Material properties, derived rather than invented (TSCI220805061R.pdf)

`bump_underfill_composite` kz=20 / kxy=2 are placeholders. That paper derives µbump
equivalent conductivity from volume fractions of the laminate:

- out-of-plane (series): `1/k_eq,z = Σ vᵢ/kᵢ`
- in-plane (parallel): `k_eq,xy = Σ kᵢvᵢ`

It validates the same approach for TSV against FEM: in-plane analytical 116.74 vs FEM
115.3 W/mK (**1.2%** error); out-of-plane 175.2 vs 169.6 W/mK (**3.9%**), for r₁=8 µm,
r₂=9 µm, pitch 40 µm.

**Action:** recompute the composite from stated volume fractions using these formulas, cite
the paper, and replace every invented k in `MATERIALS`. Report old vs new and the resulting
temperature shift.

---

## Part 4 — The accuracy bar (CR_early_tcad_3d.pdf)

3D-ICE 3.0 validated against a **thermal test chip** — an IC of heater elements and
temperature sensors — using an open dataset of **24 traces** (half natural convection, half
forced air), each ~1 hour, preheat then transient, with a 200 W/cm² hot spot. Reported
**average error under 1 °C**, higher under natural convection due to air-model
simplifications.

That is what "accurate" means in this field: measured against silicon, with a stated error
distribution, across many traces. Note it also validates the *sink model* specifically —
their stated motivation is "the relevance of good sink modeling for an accurate
representation of chip temperatures," which is precisely the gap in Part 1's first two rows.

---

## Execution order

1. Flip the sink to the top face. Re-run; expect a significant drop.
2. Ship the two cooling presets; print W/cm² and R_ja.
3. Recompute composite material properties per Part 3; cite TSCI.
4. Run the mesh refinement study {12, 24, 48}; report GCI and observed order `p`.
5. Reproduce the dac143 FFT benchmark; compare against 27.36 / 42.84 °C.
6. Only after 1–5: attach any accuracy claim to an output, with the error stated.

Until step 5 passes, every reported temperature carries "verified, not validated" on the
report itself — alongside the four modeling limitations already listed in the roadmap.
