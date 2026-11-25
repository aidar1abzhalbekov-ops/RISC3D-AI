"""
Refinement probe for the production steady-state path.

Usage (from the repo root):
    python -I /home/claude/refine/refine_probe2.py <level>

level 1 = 40x40x24 (500 um, 10 um), level 2 = 80x80x48, level 3 = 160x160x96.
Each level halves dx/dy and dz relative to the previous one (r = 2**(level-1)).
Block edges and layer boundaries are exact integer multiples at every level.

The solve goes through ThermalSolver.solve_steady_state(), the same code path
the optimizer uses, so its residual guard (STEADY_RESIDUAL_MAX) applies. The
residual reported here is recomputed independently from the assembled operator.
"""

import logging
import os
import sys
import time

try:
    import resource  # Unix only; absent on Windows
except ImportError:
    resource = None

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from src.config_loader import load_config, stack_layer_bounds  # noqa: E402
from src.optimizer import ChipletPlacementOptimizer  # noqa: E402

# Best SA layout found at the 500 um base mesh (block origins in voxels).
LAYOUT_40 = [(6, 3), (24, 7), (2, 18), (29, 30)]


def main(level: int) -> int:
    if level not in (1, 2, 3):
        print("level must be 1, 2 or 3", file=sys.stderr)
        return 2
    r = 2 ** (level - 1)

    base = load_config("configs/stage2_config.json")
    g = base.grid
    nx, ny, nz = g.nx * r, g.ny * r, g.nz * r
    base.grid.nx, base.grid.ny, base.grid.nz = nx, ny, nz
    base.grid.dx = base.grid.dy = g.dx / r
    base.grid.dz = g.dz / r

    layout = [(x * r, y * r) for (x, y) in LAYOUT_40]
    o = ChipletPlacementOptimizer(base)
    ts = o.solver
    zi, zb = stack_layer_bounds(base)
    N = nx * ny * nz
    print(f"mesh {nx}x{ny}x{nz}  N={N}  "
          f"interposer z[0:{zi}]  composite z[{zi}:{zb}]  die z[{zb}:{nz}]", flush=True)

    ts.q_map[...] = o._power_map(layout)

    t0 = time.perf_counter()
    T = ts.solve_steady_state(apply_result=False)
    elapsed = time.perf_counter() - t0

    # Independent residual check on the assembled operator.
    M = ts._conduction_operator.tocsr()
    f = ts._boundary_forcing + ts.q_map.reshape(N) / ts._rho_cp.reshape(N)
    res = np.linalg.norm(M @ T.reshape(N) + f) / np.linalg.norm(f)
    peak_c = float(T.max()) - 273.15
    if resource is not None:
        maxrss_txt = f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB"
    else:
        maxrss_txt = "n/a (Windows)"

    print(f"solve: {elapsed:.2f}s  residual={res:.3e}  peak={peak_c:.4f} C  "
          f"maxrss={maxrss_txt}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: refine_probe2.py <level 1|2|3>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(int(sys.argv[1])))
