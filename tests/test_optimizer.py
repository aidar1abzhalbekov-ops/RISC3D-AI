"""
tests/test_optimizer.py
RISC3D-AI :: Stage 2 optimizer test suite (pytest)
 
Covers src/optimizer.py as rebuilt on 2026-11-13. Written but NOT yet
executed when created. Run: python -m pytest tests -q
 
Coverage:
  1. ChipletBlock validation.
  2. Overlap voxel counting (exact known case).
  3. Overlapping candidates are never sent to the thermal solver.
  4. Shelf seed fits and does not overlap.
  5. Fit error when a block cannot fit the die.
  6. Regression: best_peak_temperature_c is never None, even when an
     overlap-penalty candidate is cheaper than a very hot valid best.
  7. Best-cost history is monotonically non-increasing.
  8. Non-overlap invariant on the returned best layout.
  9. Plotly report writes a file (skipped if plotly is not installed).
"""
 
from __future__ import annotations
 
import math
 
import numpy as np
import pytest
 
from src.config_loader import (
    PipelineConfig,
    GridSection,
    BoundaryConditionsSection,
    StackSection,
    PowerSection,
    SolverSection,
    VisualizationSection,
)
from src.optimizer import (
    ChipletBlock,
    ChipletPlacementOptimizer,
    OptimizerError,
    PlacementFitError,
    OVERLAP_BASE_PENALTY,
    OVERLAP_VOXEL_COEFF,
    render_optimization_plotly_report,
)
 
 
def make_config(nx: int = 20, ny: int = 20, nz: int = 12,
                dx_um: float = 500.0, dy_um: float = 500.0, dz_um: float = 10.0,
                h_convective: float = 1000.0) -> PipelineConfig:
    """Small die: 20 x 20 voxels at 500 um = 10 x 10 mm, 12 layers at 10 um."""
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz,
                         dx=dx_um * 1e-6, dy=dy_um * 1e-6, dz=dz_um * 1e-6),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=25.0, h_convective=h_convective, bottom_bc="robin"),
        stack=StackSection(interposer_frac=0.40, bump_frac=0.20),
        power=PowerSection(die_power_w=0.0, die_footprint_frac=0.5),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )
 
 
def two_block_catalog(power_w: float = 5.0):
    """Two 2 mm x 2 mm blocks (4 x 4 voxels at 500 um)."""
    return (
        ChipletBlock("A", power_w, 2000.0, 2000.0),
        ChipletBlock("B", power_w, 2000.0, 2000.0),
    )
 
 
# ---------------------------------------------------------------------------
# 1. Block validation
# ---------------------------------------------------------------------------
 
class TestChipletBlock:
    @pytest.mark.parametrize("kwargs", [
        dict(name="", power_w=1.0, width_um=1.0, depth_um=1.0),
        dict(name="X", power_w=-1.0, width_um=1.0, depth_um=1.0),
        dict(name="X", power_w=1.0, width_um=0.0, depth_um=1.0),
        dict(name="X", power_w=1.0, width_um=1.0, depth_um=-2.0),
    ])
    def test_rejects_invalid_block(self, kwargs):
        with pytest.raises(OptimizerError):
            ChipletBlock(**kwargs)
 
 
# ---------------------------------------------------------------------------
# 2-5. Geometry
# ---------------------------------------------------------------------------
 
class TestGeometry:
    def test_overlap_voxels_known_case(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=0)
        # Blocks are 4x4 voxels. Origins (0,0) and (2,2) overlap in a 2x2 patch.
        assert opt.overlap_voxels([(0, 0), (2, 2)]) == 4
        # Adjacent, touching but not overlapping: zero.
        assert opt.overlap_voxels([(0, 0), (4, 0)]) == 0
 
    def test_shelf_seed_fits_and_does_not_overlap(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=0)
        seed = opt.shelf_seed()
        assert len(seed) == 2
        assert opt.overlap_voxels(seed) == 0
        for idx, (x0, y0) in enumerate(seed):
            assert opt._origin_in_bounds(idx, x0, y0)
 
    def test_block_larger_than_die_raises_fit_error(self):
        too_big = (ChipletBlock("BIG", 1.0, 20000.0, 2000.0),)  # 20 mm on a 10 mm die
        with pytest.raises(PlacementFitError):
            ChipletPlacementOptimizer(make_config(), catalog=too_big, seed=0)
 
    def test_extents_derived_from_micrometres(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=0)
        assert opt.extents == ((4, 4), (4, 4))
 
 
# ---------------------------------------------------------------------------
# 3. Overlap candidates skip the thermal solve
# ---------------------------------------------------------------------------
 
class TestOverlapHandling:
    def test_overlapping_layout_is_not_solved_and_is_penalised(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=0)
 
        calls = []
        original = opt.solver.solve_steady_state
 
        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
 
        opt.solver.solve_steady_state = spy
 
        cost, peak, T = opt._evaluate(((0, 0), (2, 2)))  # overlap = 4 voxels
        assert peak is None
        assert T is None
        assert calls == []
        assert cost == pytest.approx(OVERLAP_BASE_PENALTY + OVERLAP_VOXEL_COEFF * 4)
 
 
# ---------------------------------------------------------------------------
# 6-8. Search behaviour
# ---------------------------------------------------------------------------
 
class TestSearch:
    def test_best_peak_is_never_none_under_penalty_regression(self):
        """Regression for the elitist-tracker bug.
 
        Every proposed move is forced to an overlapping layout. With high
        block power the valid seed is hotter than the overlap penalty, so an
        unguarded tracker would adopt the penalised candidate. The guard must
        keep the seed as best, with a real float peak.
 
        Toy-scale note: 500 W per block on a 10 x 10 mm die is far above any
        real part. That is the intended adversarial regime, and the assertion
        below checks that the seed really is hotter than the penalty.
        """
        opt = ChipletPlacementOptimizer(
            make_config(), catalog=two_block_catalog(power_w=500.0), seed=0)
        seed = opt.shelf_seed()
        seed_cost, seed_peak, _ = opt._evaluate(tuple(seed))
        assert seed_peak is not None
        assert seed_peak > OVERLAP_BASE_PENALTY, (
            "Scenario not hot enough to trigger the bug; raise power_w.")
 
        opt._random_move = lambda layout: ((0, 0), (0, 0))  # always overlapping
 
        result = opt.optimize(iterations=50, initial_layout=seed)
 
        assert isinstance(result.best_peak_temperature_c, float)
        assert math.isfinite(result.best_peak_temperature_c)
        assert result.best_peak_temperature_c == pytest.approx(seed_peak, rel=1e-9)
        assert opt.overlap_voxels(result.best_layout) == 0
 
    def test_history_is_monotonically_non_increasing(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=1)
        result = opt.optimize(iterations=60)
        hist = result.history_best_cost
        assert len(hist) == 61  # initial + one entry per iteration
        assert all(b <= a + 1e-12 for a, b in zip(hist, hist[1:]))
 
    def test_best_layout_is_non_overlapping_and_in_bounds(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=2)
        result = opt.optimize(iterations=80)
        assert opt.overlap_voxels(result.best_layout) == 0
        for idx, (x0, y0) in enumerate(result.best_layout):
            assert opt._origin_in_bounds(idx, x0, y0)
        assert np.all(np.isfinite(result.best_temperature_k))
 
    def test_same_seed_gives_same_result(self):
        a = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=7)
        b = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=7)
        ra = a.optimize(iterations=30)
        rb = b.optimize(iterations=30)
        assert ra.best_layout == rb.best_layout
        assert ra.best_peak_temperature_c == pytest.approx(rb.best_peak_temperature_c)
 
    def test_rejects_overlapping_initial_layout(self):
        opt = ChipletPlacementOptimizer(make_config(), catalog=two_block_catalog(), seed=0)
        with pytest.raises(OptimizerError):
            opt.optimize(iterations=5, initial_layout=[(0, 0), (1, 1)])
 
 
# ---------------------------------------------------------------------------
# 9. Plotly report
# ---------------------------------------------------------------------------
 
class TestPlotlyReport:
    def test_report_file_is_written(self, tmp_path):
        pytest.importorskip("plotly")
        cfg = make_config()
        opt = ChipletPlacementOptimizer(cfg, catalog=two_block_catalog(), seed=0)
        result = opt.optimize(iterations=10)
        out = tmp_path / "report.html"
        path = render_optimization_plotly_report(result, cfg, output_path=str(out))
        assert out.exists()
        assert path == str(out)
        assert out.stat().st_size > 1000