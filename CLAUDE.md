# CLAUDE.md — Working Agreement (canonical copy)

This is the canonical copy of the `CLAUDE.md` that belongs at the **repo root**. Claude Code
reads a root-level `CLAUDE.md` automatically at session start, so these rules apply without
the user restating them each time.

**Purpose:** Claude Code was creating and running files in its own sandbox, so the user's
actual VS Code repo never received the changes. These rules force complete-file delivery the
user can paste in, and forbid autonomous background execution.

If the repo root copy is ever lost, restore it from the content below.

---

## 1. Deliver complete files, always

Every changed file is delivered whole, top to bottom, in one fenced code block, with its exact
repo-relative path on the line above. Never a diff, never "add this method to X", never
`# ... rest unchanged ...`, never a placeholder. A 900-line file is delivered as 900 lines —
split across messages if needed, but never abbreviated. If the user cannot select-all and
replace, it is not delivered.

## 2. End every response with a replacement manifest

A table of file / line count / what changed, plus an explicit list of files NOT touched.
Verification command:

```bash
find . -name "*.py" -not -path "./.git/*" -not -path "*/__pycache__/*" -exec wc -l {} +
```

## 3. No autonomous background execution

No dev servers or watchers left running. No auto-opening browsers (respect `--no-browser`;
read HTML source instead of launching a viewer). No web UI or scaffold the user didn't ask
for — this is a Python EDA tool.

## 4. Run the code, show real output

Unchanged standing rule. Smoke-test on creation, regression-test after edits, paste actual
terminal output. Never describe expected output. Every benchmark/timing/temperature figure in
a docstring must be personally measured that session.

## 5. Never vouch for unverified code

On finding unfamiliar pre-existing code: read the whole file, trace the math by hand, execute
with adversarial test cases, fix what's broken, credit nothing unverified.

## 6. `AttributeError` on config fields = sync issue first

Check `grep -n "class SolverSection" -A 15 src/config_loader.py` before assuming a real bug.

Expected fields — **seven**, verified against `config_loader.py:80-87` on 2026-08-29:
`method`, `steady_state`, `total_time_s`, `dt`, `safety_factor`, `record_steps`, `max_steps`.

Note `total_time_s`: genuinely present and correct. Earlier project documentation listed only
six fields and omitted it. Do not treat it as unexpected and do not remove it.

Say so explicitly when it is a sync issue, harden with `getattr` fallbacks, re-deliver the
complete file.

## 7. Sanity-check absolute temperatures out loud

Absurd temperatures on toy grids are correct physics. **A plausible number where physics says
it should be absurd means something is silently wrong.** Check ΔT ≈ P · R_conv with
R_conv ≈ 1/(h·A) by hand and state which regime the result is in.

## 8. Physical dimensions are authored in micrometers, never voxels

`ChipletBlock` declares `width_um` / `depth_um`. Keep it that way. A catalog authored in voxel
counts makes chiplet size a function of mesh resolution — halve `dx` for a convergence study
and every block halves in linear size and quadruples in power density, so the physics changes
when you refine the mesh. That is backwards.

Author every footprint in µm; derive voxel extent as `ceil(width_um / dx)` at build time. The
catalog-fit check then compares physical areas against physical die size, not voxel counts
against grid counts.

Corollary: **Stage 1 and Stage 2 must share one `GridSection`.** If the optimizer carries its
own hardcoded `dx`, a single `main.py --optimize` run silently models two different chips.

## 9. Tone

Direct, technical, minimal preamble. State assumptions briefly rather than deciding silently.

---

## Appendix — resolved: the 176.89 °C question (2026-08-29)

Earlier flagged as a blocker on suspicion of a lost cell-volume conversion in `q_map`.
**That suspicion was wrong.** Backing `dx` out of the reported result:

| | dx | die | power density | ΔT |
|---|---|---|---|---|
| Stage 1 default | 25 µm | 1.00 mm² | 3600 W/cm² | 36,000 K |
| implied by Stage 2 | **385 µm** | 237 mm² | 15.2 W/cm² | **151.9 K** ✓ |

385 µm reproduces 176.89 °C exactly, and per-block densities at that scale are 47–68 W/cm² —
realistic for advanced packaging. The physics is sound.

Remaining open item: confirm where Stage 2's `dx` originates. If hardcoded in the optimizer
rather than inherited from `GridSection`, one `--optimize` run models a 1 mm² chip in Stage 1
and a 237 mm² chip in Stage 2. See rule 8 corollary.

## Appendix — suggested realistic chiplet catalog (µm)

The round-1 catalog dimensions (12×12, 14×10, 8×6, 6×14 voxels) were invented to fit a 40×40
die; only the wattages were specified. Physically-grounded replacements:

| Block | W | Footprint | Area | Density |
|---|---|---|---|---|
| CPU_Core | 10 | 4000 × 4000 µm | 16 mm² | 63 W/cm² |
| GPU_Core | 14 | 6000 × 5000 µm | 30 mm² | 47 W/cm² |
| HBM_Memory | 4 | 11000 × 11000 µm | 121 mm² | 3.3 W/cm² |
| NPU_Accelerator | 8 | 5000 × 5000 µm | 25 mm² | 32 W/cm² |

HBM dominates on area because a real HBM3 KGSD genuinely is ~11 × 11 mm. Total 192 mm² on a
20 × 20 mm die = 48% fill, leaving SA real room to move. At h = 1000 W/m²K this predicts a
~115 °C peak — a defensible headline figure.
