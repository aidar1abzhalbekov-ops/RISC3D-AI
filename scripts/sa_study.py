"""
Round 3 task 3: SA iteration count and cooling schedule, on the 40x40x24 mesh.

Usage (from the repo root):
    python -I /home/claude/refine/sa_study.py <N> <schedule> <start> <seed>

  N         iteration count (int)
  schedule  'fixed'  -> cooling_alpha = 0.995 (shipped default)
            'scaled' -> alpha = (T_FINAL / T0) ** (1 / N), so T reaches T_FINAL at iteration N
  start     'shelf'  -> optimizer's shelf-packed seed
            'corner' -> adversarial corner-clustered seed (defined below)
  seed      integer RNG seed

Prints one JSON line with the result and timing.
"""

import json
import logging
import os
import sys
import time

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd())

from src.config_loader import load_config  # noqa: E402
from src.optimizer import ChipletPlacementOptimizer  # noqa: E402

T0 = 5.0
T_FINAL = 0.01
SHIPPED_ALPHA = 0.995


def corner_seed(o):
    """Adversarial start: the two hottest blocks pushed into the same corner, adjacent.

    Catalog order is CPU_Core (10 W), GPU_Core (14 W), HBM_Memory (4 W), NPU_Accelerator (8 W).
    Hottest first: GPU (idx 1), CPU (idx 0), NPU (idx 3); HBM (idx 2) placed last in the far corner.
    Layout is built greedily from (0,0) and checked for overlap and bounds before use.
    """
    ext = o.extents
    order = [1, 0, 3, 2]  # hot blocks first, then HBM
    layout = [None] * len(ext)
    x = 0
    y = 0
    row_h = 0
    for idx in order:
        w, d = ext[idx]
        if x + w > o.nx:
            x = 0
            y += row_h
            row_h = 0
        layout[idx] = (x, y)
        x += w
        row_h = max(row_h, d)
    if any(p is None for p in layout):
        raise RuntimeError("corner seed incomplete")
    for idx, (x0, y0) in enumerate(layout):
        if not o._origin_in_bounds(idx, x0, y0):
            raise RuntimeError(f"corner seed out of bounds for block {idx}: {(x0, y0)}")
    if o.overlap_voxels(layout) != 0:
        raise RuntimeError("corner seed overlaps; adjust the packing order")
    return layout


def main(argv) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    N = int(argv[1])
    schedule = argv[2]
    start = argv[3]
    seed = int(argv[4])

    if schedule == "fixed":
        alpha = SHIPPED_ALPHA
    elif schedule == "scaled":
        alpha = (T_FINAL / T0) ** (1.0 / N)
    else:
        raise SystemExit(f"unknown schedule {schedule}")

    cfg = load_config("configs/stage2_config.json")
    o = ChipletPlacementOptimizer(cfg, seed=seed)
    init = corner_seed(o) if start == "corner" else None
    if start not in ("shelf", "corner"):
        raise SystemExit(f"unknown start {start}")

    t0 = time.perf_counter()
    res = o.optimize(
        iterations=N,
        initial_temperature=T0,
        cooling_alpha=alpha,
        min_temperature=1e-3,
        initial_layout=init,
    )
    wall = time.perf_counter() - t0
    evals = res.iterations + 1  # optimize() counts the initial evaluation too

    out = {
        "N": N,
        "schedule": schedule,
        "alpha": alpha,
        "start": start,
        "seed": seed,
        "best_peak_c": res.best_peak_temperature_c,
        "best_layout": [list(p) for p in res.best_layout],
        "accepted": res.accepted_moves,
        "overlap_candidates": res.overlap_candidates,
        "wall_s": wall,
        "s_per_proposal": wall / max(evals - 1, 1),
    }
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
