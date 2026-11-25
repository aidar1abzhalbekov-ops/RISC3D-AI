import json, logging, sys, os
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd())
import numpy as np
from src.config_loader import load_config
from src.optimizer import ChipletPlacementOptimizer
from src.thermal_solver import ThermalSolver

rows = [json.loads(l) for l in open('/home/claude/refine/results.jsonl')]
best = min((r for r in rows if r['start'] == 'shelf' and r['schedule'] == 'fixed' and r['N'] == 2000),
           key=lambda r: r['best_peak_c'])
layout = [tuple(p) for p in best['best_layout']]
print('best layout (seed %d, N=2000):' % best['seed'], layout, 'stored peak %.4f' % best['best_peak_c'])

cfg = load_config('configs/stage2_config.json')
o = ChipletPlacementOptimizer(cfg)
q = o._power_map(layout)
print('total power from q_map (W):', float(q.sum() * o.cell_volume))

# bottom sink (shipped)
o.solver.q_map[...] = q
Tb = o.solver.solve_steady_state(apply_result=False)
peak_b = float(Tb.max()) - 273.15

# top sink: same physics, mirrored frame
gb = o.gb
ts_top = ThermalSolver.from_grid_builder(gb, sink_side='top')
ts_top.q_map[...] = np.ascontiguousarray(q[:, :, ::-1])
Tt_mir = ts_top.solve_steady_state(apply_result=False)
Tt = Tt_mir[:, :, ::-1]  # back to physical frame
peak_t = float(Tt.max()) - 273.15

# energy checks in each frame
A = o.nx * cfg.grid.dx * o.ny * cfg.grid.dy
P = float(q.sum() * o.cell_volume)
Ta = cfg.boundary_conditions.t_ambient_c
h = cfg.boundary_conditions.h_convective
out_b = float(np.sum(h * cfg.grid.dx * cfg.grid.dy * (Tb[:, :, 0] - 273.15 - Ta)))
out_t = float(np.sum(h * cfg.grid.dx * cfg.grid.dy * (Tt_mir[:, :, 0] - 273.15 - Ta)))
print('P = %.4f W, A = %.3e m2, P/(hA) = %.2f K' % (P, A, P / (h * A)))
print('bottom sink: peak %.4f C, sink outflow %.4f W' % (peak_b, out_b))
print('top sink:    peak %.4f C, sink outflow %.4f W' % (peak_t, out_t))
print('difference (top - bottom): %.4f K' % (peak_t - peak_b))
print('mean rise over ambient (bottom): %.3f K, (top): %.3f K' % (float(Tb.mean()) - 273.15 - Ta, float(Tt.mean()) - 273.15 - Ta))
