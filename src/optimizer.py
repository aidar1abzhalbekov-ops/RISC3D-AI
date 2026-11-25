"""
src/optimizer.py
RISC3D-AI :: Stage 2 -- chiplet placement optimizer ("AI Architect")
 
Simulated-annealing search over chiplet block origins on a fixed die.
Every candidate is scored by the steady-state peak temperature from the
same ThermalSolver Stage 1 uses. Overlapping candidates are never solved;
they receive a penalty cost instead.
 
Design notes (from claude/NEXT_TASK_stage2_wiring.md and CLAUDE.md):
- Block footprints are authored in micrometres (width_um, depth_um). Voxel
  extents are derived at build time as ceil(width_um / dx).
- The solver and conduction operator are built once for the fixed material
  stack. Each candidate only overwrites solver.q_map before calling
  solve_steady_state(apply_result=False).
- Overlapping candidates get penalty = OVERLAP_BASE_PENALTY +
  OVERLAP_VOXEL_COEFF * overlap_voxels and skip the thermal solve.
- The elitist best-tracker only adopts candidates that have a real peak
  temperature. An overlap-penalty cost can be numerically lower than a very
  hot valid best, so the guard `cand_peak is not None` is required.
 
Status: rebuilt 2026-11-13 after the original src/optimizer.py was lost.
Not executed when written. Verify with `python -m pytest tests -q`.
Values marked ASSUMPTION below are my choices, not from the original file.
"""
 
from __future__ import annotations
 
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
 
import numpy as np
 
from src.config_loader import PipelineConfig, stack_layer_bounds, has_top_package
from src.grid_builder import GridBuilder
from src.thermal_solver import ThermalSolver
 
logger = logging.getLogger("RISC3D-AI.Optimizer")
 
OVERLAP_BASE_PENALTY = 500.0      # from the design notes
OVERLAP_VOXEL_COEFF = 10.0        # ASSUMPTION: per-overlapping-voxel penalty
KELVIN_OFFSET = 273.15
PLOT_MAX_POINTS = 60_000          # ASSUMPTION: point-cloud subsample cap
DEFAULT_REPORT_PATH = os.path.join("outputs", "optimized_thermal_report.html")
 
 
class OptimizerError(Exception):
    """Raised on invalid optimizer configuration or inputs."""
 
 
class PlacementFitError(OptimizerError):
    """Raised when a chiplet cannot fit on the die at the configured grid."""
 
 
@dataclass(frozen=True)
class ChipletBlock:
    """A chiplet with a physical footprint (micrometres) and power (W)."""
    name: str
    power_w: float
    width_um: float
    depth_um: float
 
    def __post_init__(self) -> None:
        if not self.name:
            raise OptimizerError("ChipletBlock.name must be non-empty.")
        if self.power_w < 0:
            raise OptimizerError(f"{self.name}: power_w must be >= 0, got {self.power_w}")
        if self.width_um <= 0 or self.depth_um <= 0:
            raise OptimizerError(
                f"{self.name}: width_um/depth_um must be > 0, got "
                f"({self.width_um}, {self.depth_um})"
            )
 
 
# Footprints from claude/CLAUDE_md_working_agreement.md (Appendix: catalog).
DEFAULT_CHIPLET_CATALOG: Tuple[ChipletBlock, ...] = (
    ChipletBlock("CPU_Core", 10.0, 4000.0, 4000.0),
    ChipletBlock("GPU_Core", 14.0, 6000.0, 5000.0),
    ChipletBlock("HBM_Memory", 4.0, 11000.0, 11000.0),
    ChipletBlock("NPU_Accelerator", 8.0, 5000.0, 5000.0),
)
 
 
@dataclass
class OptimizationResult:
    """Outcome of one optimize() call. best_peak_temperature_c is always a
    float: only valid (non-overlapping) layouts are ever recorded as best."""
    best_layout: Tuple[Tuple[int, int], ...]     # (x0, y0) voxel origin per block
    best_cost: float
    best_peak_temperature_c: float
    best_temperature_k: np.ndarray               # shape (nx, ny, nz)
    history_best_cost: List[float]
    iterations: int
    accepted_moves: int
    overlap_candidates: int
    evaluations: int
    catalog: Tuple[ChipletBlock, ...]
    extents: Tuple[Tuple[int, int], ...]         # (w_vox, d_vox) per block
 
    def placements_um(self) -> List[dict]:
        """Best layout expressed in micrometres, one dict per block."""
        out = []
        for blk, (x0, y0), (w, d) in zip(self.catalog, self.best_layout, self.extents):
            out.append({
                "name": blk.name,
                "power_w": blk.power_w,
                "x_voxel": x0, "y_voxel": y0,
                "w_voxel": w, "d_voxel": d,
                "width_um": blk.width_um, "depth_um": blk.depth_um,
            })
        return out
 
 
class ChipletPlacementOptimizer:
    """
    Simulated-annealing chiplet placement on the fixed stack described by a
    PipelineConfig. Stage 1 and Stage 2 must share one GridSection, so this
    class takes the PipelineConfig itself (CLAUDE.md rule 8 corollary).
    """
 
    def __init__(
        self,
        config: PipelineConfig,
        catalog: Sequence[ChipletBlock] = DEFAULT_CHIPLET_CATALOG,
        seed: Optional[int] = 0,
    ):
        if not isinstance(config, PipelineConfig):
            raise OptimizerError(
                f"config must be a PipelineConfig, got {type(config).__name__}"
            )
        config.validate()
        if has_top_package(config):
            # The SA evaluation builds its own heatsink-bottom stack and power map; it does not
            # yet mirror the top-side package. Refuse rather than optimize against the wrong sink.
            raise OptimizerError(
                "ChipletPlacementOptimizer does not yet support the top-side package "
                "(stack.tim1_um / lid_um / tim2_um). Set them to 0 for Stage 2, or use Stage 1."
            )
        self.config = config
        self.catalog: Tuple[ChipletBlock, ...] = tuple(catalog)
        if not self.catalog:
            raise OptimizerError("catalog must contain at least one ChipletBlock.")
        self.rng = random.Random(seed)
 
        g = config.grid
        self.nx, self.ny, self.nz = g.nx, g.ny, g.nz
        self.dx, self.dy, self.dz = g.dx, g.dy, g.dz
        self.cell_volume = g.dx * g.dy * g.dz
 
        self.extents: Tuple[Tuple[int, int], ...] = self._compute_extents()
        self.gb, self.z_die_start = self._build_stack()
        self.solver = ThermalSolver.from_grid_builder(self.gb)
        self.die_layers = self.nz - self.z_die_start
 
        logger.info(
            "Optimizer ready: die %dx%d voxels (%.0f x %.0f um), %d blocks, die layers=%d",
            self.nx, self.ny, self.nx * self.dx * 1e6, self.ny * self.dy * 1e6,
            len(self.catalog), self.die_layers,
        )
 
    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _compute_extents(self) -> Tuple[Tuple[int, int], ...]:
        extents = []
        for blk in self.catalog:
            # Subtract a tiny tolerance so exact multiples do not round up.
            w = math.ceil(blk.width_um * 1e-6 / self.dx - 1e-9)
            d = math.ceil(blk.depth_um * 1e-6 / self.dy - 1e-9)
            if w > self.nx or d > self.ny:
                raise PlacementFitError(
                    f"Chiplet '{blk.name}' ({blk.width_um:.0f} x {blk.depth_um:.0f} um, "
                    f"{w} x {d} voxels) does not fit within the {self.nx} x {self.ny} "
                    f"voxel die (dx={self.dx * 1e6:.1f} um, dy={self.dy * 1e6:.1f} um)."
                )
            extents.append((max(1, w), max(1, d)))
        return tuple(extents)
 
    def _build_stack(self) -> Tuple[GridBuilder, int]:
        """Same layer split as main.build_chiplet_stack(), without power.
        Returns (GridBuilder, z index where the die layer starts)."""
        nz = self.nz
        z_interposer_end, z_bump_end = stack_layer_bounds(self.config)

        gb = GridBuilder(config=self.config.to_dict())
        gb.assign_layer(slice(0, z_interposer_end), "silicon")
        gb.assign_layer(slice(z_interposer_end, z_bump_end), "bump_underfill_composite")
        gb.assign_layer(slice(z_bump_end, nz), "silicon")
        gb.tag_boundaries()
        return gb, z_bump_end
 
    # ------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------
    def _origin_in_bounds(self, idx: int, x0: int, y0: int) -> bool:
        w, d = self.extents[idx]
        return 0 <= x0 <= self.nx - w and 0 <= y0 <= self.ny - d
 
    def overlap_voxels(self, layout: Sequence[Tuple[int, int]]) -> int:
        """Total number of voxels shared by any pair of blocks."""
        total = 0
        n = len(layout)
        for a in range(n):
            xa, ya = layout[a]
            wa, da = self.extents[a]
            for b in range(a + 1, n):
                xb, yb = layout[b]
                wb, db = self.extents[b]
                ox = min(xa + wa, xb + wb) - max(xa, xb)
                oy = min(ya + da, yb + db) - max(ya, yb)
                if ox > 0 and oy > 0:
                    total += ox * oy
        return total
 
    def shelf_seed(self) -> List[Tuple[int, int]]:
        """Deterministic shelf-packed starting layout (left-to-right rows)."""
        layout: List[Tuple[int, int]] = []
        x = 0
        y = 0
        row_h = 0
        for idx, (w, d) in enumerate(self.extents):
            if x + w > self.nx:
                x = 0
                y += row_h
                row_h = 0
            if y + d > self.ny:
                raise PlacementFitError(
                    f"Shelf seed does not fit: block '{self.catalog[idx].name}' would start "
                    f"at y={y} with depth {d} on a {self.ny}-voxel die. Use a larger die or "
                    f"a smaller catalog."
                )
            layout.append((x, y))
            x += w
            row_h = max(row_h, d)
        return layout
 
    def _random_move(self, layout: Tuple[Tuple[int, int], ...]) -> Tuple[Tuple[int, int], ...]:
        idx = self.rng.randrange(len(layout))
        w, d = self.extents[idx]
        if self.rng.random() < 0.5:
            x = self.rng.randint(0, self.nx - w)
            y = self.rng.randint(0, self.ny - d)
        else:
            step = max(1, int(round(0.1 * max(self.nx, self.ny))))
            x0, y0 = layout[idx]
            x = min(max(0, x0 + self.rng.randint(-step, step)), self.nx - w)
            y = min(max(0, y0 + self.rng.randint(-step, step)), self.ny - d)
        new = list(layout)
        new[idx] = (x, y)
        return tuple(new)
 
    # ------------------------------------------------------------------
    # Thermal evaluation
    # ------------------------------------------------------------------
    def _power_map(self, layout: Sequence[Tuple[int, int]]) -> np.ndarray:
        q = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        for idx, blk in enumerate(self.catalog):
            if blk.power_w == 0.0:
                continue
            x0, y0 = layout[idx]
            w, d = self.extents[idx]
            n_cells = w * d * self.die_layers
            density = blk.power_w / (n_cells * self.cell_volume)  # W/m^3
            q[x0:x0 + w, y0:y0 + d, self.z_die_start:] += density
        return q
 
    def _evaluate(self, layout: Tuple[Tuple[int, int], ...]):
        """Returns (cost, peak_c_or_None, T_k_or_None).
        Overlapping layouts are not solved."""
        overlap = self.overlap_voxels(layout)
        if overlap > 0:
            return OVERLAP_BASE_PENALTY + OVERLAP_VOXEL_COEFF * overlap, None, None
        self.solver.q_map[...] = self._power_map(layout)
        T = self.solver.solve_steady_state(apply_result=False)
        peak_c = float(np.max(T)) - KELVIN_OFFSET
        return peak_c, peak_c, T
 
    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def optimize(
        self,
        iterations: int = 2000,
        initial_temperature: float = 5.0,
        cooling_alpha: float = 0.995,
        min_temperature: float = 1e-3,
        initial_layout: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> OptimizationResult:
        """
        Simulated annealing. Returns the best valid layout seen.
 
        ASSUMPTION: initial_temperature (K) and cooling_alpha are my defaults.
        They are not from the original file and need tuning against real runs.
        """
        if iterations < 1:
            raise OptimizerError(f"iterations must be >= 1, got {iterations}")
        if not (0.0 < cooling_alpha <= 1.0):
            raise OptimizerError(f"cooling_alpha must be in (0, 1], got {cooling_alpha}")
        if initial_temperature <= 0:
            raise OptimizerError(f"initial_temperature must be > 0, got {initial_temperature}")
 
        start = tuple(initial_layout) if initial_layout is not None else tuple(self.shelf_seed())
        if len(start) != len(self.catalog):
            raise OptimizerError("initial_layout must have one origin per catalog block.")
        for idx, (x0, y0) in enumerate(start):
            if not self._origin_in_bounds(idx, x0, y0):
                raise PlacementFitError(
                    f"Initial origin for '{self.catalog[idx].name}' ({x0}, {y0}) is out of bounds."
                )
 
        current = start
        cur_cost, cur_peak, cur_T = self._evaluate(current)
        if cur_peak is None:
            raise OptimizerError("Initial layout overlaps; choose a non-overlapping initial_layout.")
 
        best_layout, best_cost, best_peak, best_T = current, cur_cost, cur_peak, cur_T
        history: List[float] = [best_cost]
        temperature = initial_temperature
        accepted = 0
        overlap_candidates = 0
        evaluations = 1
 
        for _ in range(iterations):
            cand = self._random_move(current)
            cand_cost, cand_peak, cand_T = self._evaluate(cand)
            evaluations += 1
            if cand_peak is None:
                overlap_candidates += 1
 
            delta = cand_cost - cur_cost
            if delta <= 0 or self.rng.random() < math.exp(-delta / temperature):
                current, cur_cost = cand, cand_cost
                accepted += 1
 
            # Elitist guard: only a real (non-overlapping) candidate can become best.
            improved = (cand_peak is not None) and (cand_cost < best_cost)
            if improved:
                best_layout, best_cost, best_peak, best_T = cand, cand_cost, cand_peak, cand_T
 
            history.append(best_cost)
            temperature = max(temperature * cooling_alpha, min_temperature)
 
        logger.info(
            "Optimizer finished: best peak=%.2f C, accepted=%d/%d, overlap candidates=%d",
            best_peak, accepted, iterations, overlap_candidates,
        )
        return OptimizationResult(
            best_layout=best_layout,
            best_cost=best_cost,
            best_peak_temperature_c=best_peak,
            best_temperature_k=np.array(best_T, dtype=np.float64, copy=True),
            history_best_cost=history,
            iterations=iterations,
            accepted_moves=accepted,
            overlap_candidates=overlap_candidates,
            evaluations=evaluations,
            catalog=self.catalog,
            extents=self.extents,
        )
 
 
# ---------------------------------------------------------------------------
# Plotly report
# ---------------------------------------------------------------------------
 
_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
              (0, 2), (1, 3), (4, 6), (5, 7),
              (0, 4), (1, 5), (2, 6), (3, 7)]
 
 
def _block_wireframe(x0: float, x1: float, y0: float, y1: float,
                     z0: float, z1: float) -> Tuple[list, list, list]:
    """Returns x, y, z lists tracing all 12 box edges, separated by None."""
    corners = [(xx, yy, zz) for zz in (z0, z1) for yy in (y0, y1) for xx in (x0, x1)]
    xs, ys, zs = [], [], []
    for a, b in _BOX_EDGES:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs
 
 
def render_optimization_plotly_report(
    result: OptimizationResult,
    config: PipelineConfig,
    output_path: str = DEFAULT_REPORT_PATH,
) -> str:
    """
    Writes a self-contained Plotly HTML report (plotly.js inlined, so it
    works offline): 3D temperature point cloud with wireframe block overlay,
    and the best-cost convergence history.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
 
    g = config.grid
    T_c = result.best_temperature_k - KELVIN_OFFSET
    if T_c.shape != (g.nx, g.ny, g.nz):
        raise OptimizerError(
            f"best_temperature_k shape {T_c.shape} does not match grid ({g.nx}, {g.ny}, {g.nz})"
        )
 
    flat = T_c.reshape(-1)
    idx = np.indices(T_c.shape).reshape(3, -1).T
    stride = max(1, int(math.ceil(flat.size / PLOT_MAX_POINTS)))
    sel = np.arange(0, flat.size, stride)
    px = (idx[sel, 0] + 0.5) * g.dx * 1e6
    py = (idx[sel, 1] + 0.5) * g.dy * 1e6
    pz = (idx[sel, 2] + 0.5) * g.dz * 1e6
    pt = flat[sel]
 
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Steady-state temperature (best layout)", "Best-cost convergence"),
        column_widths=[0.6, 0.4],
    )
 
    fig.add_trace(
        go.Scatter3d(
            x=px, y=py, z=pz, mode="markers",
            marker=dict(size=2, color=pt, colorscale="Inferno",
                        colorbar=dict(title="°C", x=0.45)),
            name="Temperature",
        ),
        row=1, col=1,
    )
 
    z_die0 = _die_start_index(config) * g.dz * 1e6
    z_die1 = g.nz * g.dz * 1e6
    wx, wy, wz = [], [], []
    for (x0v, y0v), (w, d), blk in zip(result.best_layout, result.extents, result.catalog):
        x0 = x0v * g.dx * 1e6
        x1 = (x0v + w) * g.dx * 1e6
        y0 = y0v * g.dy * 1e6
        y1 = (y0v + d) * g.dy * 1e6
        ex, ey, ez = _block_wireframe(x0, x1, y0, y1, z_die0, z_die1)
        wx += ex; wy += ey; wz += ez
    fig.add_trace(
        go.Scatter3d(x=wx, y=wy, z=wz, mode="lines",
                     line=dict(color="cyan", width=3), name="Chiplets"),
        row=1, col=1,
    )
 
    fig.add_trace(
        go.Scatter(x=list(range(len(result.history_best_cost))),
                   y=result.history_best_cost, mode="lines", name="Best cost"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Best cost (°C)", row=1, col=2)
 
    fig.update_layout(
        title=f"RISC3D-AI Stage 2 placement -- best peak {result.best_peak_temperature_c:.2f} °C",
        scene=dict(xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)"),
        legend=dict(orientation="h"),
    )
 
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    logger.info("Wrote Plotly optimization report to %s", output_path)
    return output_path
 
 
def _die_start_index(config: PipelineConfig) -> int:
    """Die-layer start index, from the same stack_layer_bounds() used by _build_stack()."""
    _, z_bump_end = stack_layer_bounds(config)
    return z_bump_end