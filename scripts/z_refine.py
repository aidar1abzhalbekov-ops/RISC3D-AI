"""
Vertical-only mesh study (validation plan step 4): nz = 12, 24, 48 with the stack
thicknesses fixed in micrometres (interposer 100, bump+underfill 40, die 100 um).
Lateral mesh is fixed at 40x40 with dx = dy = 500 um.

Usage (from the repo root):  python -I /home/claude/refine/z_refine.py
"""

import json
import logging
import os
import sys
import time

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from src.config_loader import load_config, stack_layer_bounds  # noqa: E402
from src.optimizer import ChipletPlacementOptimizer  # noqa: E402

STACK_UM = 240.0  # interposer 100 + bump 40 + die 100
LEVELS = [12, 24, 48]


def run(nz: int, layout):
    cfg = load_config("configs/stage2_config.json")
    cfg.grid.nz = nz
    cfg.grid.dz = STACK_UM * 1e-6 / nz
    o = ChipletPlacementOptimizer(cfg)
    ts = o.solver
    zi, zb = stack_layer_bounds(cfg)
    q = o._power_map(layout)
    total_w = float(q.sum() * o.cell_volume)
    ts.q_map[...] = q
    t0 = time.perf_counter()
    T = ts.solve_steady_state(apply_result=False)
    elapsed = time.perf_counter() - t0
    N = T.size
    M = ts._conduction_operator.tocsr()
    f = ts._boundary_forcing + ts.q_map.reshape(N) / ts._rho_cp.reshape(N)
    res = np.linalg.norm(M @ T.reshape(N) + f) / np.linalg.norm(f)
    return {
        "nz": nz,
        "dz_um": cfg.grid.dz * 1e6,
        "layers": {"interposer": zi, "bump": zb - zi, "die": nz - zb},
        "N": N,
        "power_w_in_mesh": total_w,
        "peak_c": float(T.max()) - 273.15,
        "residual": float(res),
        "solve_s": elapsed,
    }


def main() -> int:
    rows = [json.loads(l) for l in open("/home/claude/refine/results.jsonl")]
    best = min((r for r in rows if r["start"] == "shelf" and r["schedule"] == "fixed" and r["N"] == 2000),
               key=lambda r: r["best_peak_c"])
    layout = [tuple(p) for p in best["best_layout"]]
    print("layout:", layout, flush=True)
    out = []
    for nz in LEVELS:
        r = run(nz, layout)
        out.append(r)
        print(json.dumps(r), flush=True)
    peaks = [r["peak_c"] for r in out]  # coarse -> fine
    print("peaks (nz 12, 24, 48):", ["%.6f" % p for p in peaks], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
