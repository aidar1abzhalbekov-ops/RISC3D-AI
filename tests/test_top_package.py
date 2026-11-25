"""
tests/test_top_package.py
RISC3D-AI :: Top-side package layers (TIM1, lid, TIM2) with the heatsink on the lid.

Checks:
  1. stack_extents() places the layers in the documented order and the die is the
     remainder of nz; with no top package every key equals the original stack.
  2. Layer materials land in the right z ranges (kz_map values).
  3. Energy balance: generation equals heatsink outflow on the lid face.
  4. Lower bound: the peak rise is at least the mean sink-face rise, P / (h * A).
  5. Monotone in TIM conductivity: a better TIM lowers the peak.
  6. Stage 2 optimizer refuses a top-side package instead of optimizing the wrong sink.
  7. Invalid thicknesses (negative, non-integer voxels, no die layer) are rejected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import (  # noqa: E402
    BoundaryConditionsSection, ConfigError, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection, has_top_package, sink_side_for,
    stack_extents,
)
from src.grid_builder import MATERIALS, GridBuilder  # noqa: E402
from src.thermal_solver import ThermalSolver  # noqa: E402

DX = 500e-6
DZ = 10e-6
H = 1000.0
TA_C = 25.0


def _cfg(nz=20, nx=6, ny=6, die_power_w=10.0, **stack_kwargs) -> PipelineConfig:
    stack = dict(interposer_um=50.0, bump_um=20.0, tim1_um=20.0, lid_um=50.0, tim2_um=20.0)
    stack.update(stack_kwargs)
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz, dx=DX, dy=DX, dz=DZ),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=TA_C, h_convective=H, bottom_bc="robin",
        ),
        stack=StackSection(**stack),
        power=PowerSection(die_power_w=die_power_w, die_footprint_frac=1.0),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


def _build(cfg: PipelineConfig) -> GridBuilder:
    """Same layer assignment as main.build_chiplet_stack(), without the print."""
    from main import build_chiplet_stack  # local import: main is the single source of the stack
    return build_chiplet_stack(cfg)


class TestLayerOrder:
    def test_top_layers_stack_above_die_in_order(self):
        ext = stack_extents(_cfg())
        # interposer 5, bump 2, die 4 (remainder of nz=20 after 9 top voxels), TIM1 2, lid 5, TIM2 2
        assert ext == {
            "interposer_end": 5, "bump_end": 7, "die_end": 11,
            "tim1_end": 13, "lid_end": 18, "tim2_end": 20,
        }

    def test_no_top_package_is_the_original_stack(self):
        ext = stack_extents(_cfg(nz=24, interposer_um=100.0, bump_um=40.0,
                                 tim1_um=0.0, lid_um=0.0, tim2_um=0.0))
        assert ext == {
            "interposer_end": 10, "bump_end": 14, "die_end": 24,
            "tim1_end": 24, "lid_end": 24, "tim2_end": 24,
        }
        assert not has_top_package(_cfg(tim1_um=0.0, lid_um=0.0, tim2_um=0.0))
        assert sink_side_for(_cfg(tim1_um=0.0, lid_um=0.0, tim2_um=0.0)) == "bottom"

    def test_top_package_selects_top_sink(self):
        assert has_top_package(_cfg())
        assert sink_side_for(_cfg()) == "top"

    def test_partial_top_package_is_allowed(self):
        ext = stack_extents(_cfg(nz=16, tim1_um=0.0, lid_um=50.0, tim2_um=0.0))
        assert ext["tim1_end"] == ext["die_end"]
        assert ext["lid_end"] == ext["die_end"] + 5
        assert ext["tim2_end"] == 16


class TestLayerMaterials:
    def test_kz_values_follow_the_layer_stack(self):
        cfg = _cfg()
        gb = _build(cfg)
        ext = stack_extents(cfg)
        kz = gb.kz_map[0, 0, :]
        assert np.all(kz[0:ext["interposer_end"]] == MATERIALS["silicon"]["k"])
        assert np.all(kz[ext["bump_end"]:ext["die_end"]] == MATERIALS["silicon"]["k"])
        assert np.all(kz[ext["die_end"]:ext["tim1_end"]] == MATERIALS["tim_top"]["k"])
        assert np.all(kz[ext["tim1_end"]:ext["lid_end"]] == MATERIALS["copper"]["k"])
        assert np.all(kz[ext["lid_end"]:ext["tim2_end"]] == MATERIALS["tim_top"]["k"])

    def test_power_stays_in_the_die_not_the_package(self):
        cfg = _cfg()
        gb = _build(cfg)
        ext = stack_extents(cfg)
        q = gb.q_map
        assert q[:, :, ext["die_end"]:].sum() == 0.0
        assert q[:, :, ext["bump_end"]:ext["die_end"]].sum() > 0.0


class TestTopPackageSolve:
    def _solve(self, cfg: PipelineConfig):
        gb = _build(cfg)
        solver = ThermalSolver.from_grid_builder(gb, sink_side="top")
        T = solver.solve_steady_state(apply_result=False)
        return gb, solver, T

    def test_energy_balance_on_the_lid_face(self):
        cfg = _cfg(die_power_w=10.0)
        gb, solver, T = self._solve(cfg)
        # Sink face is index 0 after the mirror; flux per face is h * (T - Ta).
        outflow = float(np.sum(H * DX * DX * (T[:, :, 0] - solver.bc.t_ambient_k)))
        generated = float(np.sum(solver.q_map)) * DX * DX * DZ
        assert outflow == pytest.approx(generated, rel=1e-6)
        assert generated == pytest.approx(10.0, rel=1e-9)

    def test_peak_rise_bounded_below_by_sink_film(self):
        cfg = _cfg(die_power_w=10.0)
        _, solver, T = self._solve(cfg)
        footprint = cfg.grid.nx * cfg.grid.dx * cfg.grid.ny * cfg.grid.dy
        film_rise = 10.0 / (H * footprint)  # mean rise over the whole sink face
        peak_rise = float(T.max()) - solver.bc.t_ambient_k
        assert peak_rise >= film_rise

    def test_better_tim_lowers_the_peak(self, monkeypatch):
        cfg = _cfg(die_power_w=10.0)
        _, _, T_low = self._solve(cfg)
        monkeypatch.setitem(MATERIALS["tim_top"], "k", 50.0)
        _, _, T_high = self._solve(cfg)
        assert float(T_high.max()) < float(T_low.max())


class TestGuards:
    def test_optimizer_refuses_top_package(self):
        from src.optimizer import ChipletPlacementOptimizer, OptimizerError
        with pytest.raises(OptimizerError, match="top-side package"):
            ChipletPlacementOptimizer(_cfg())

    def test_negative_thickness_rejected(self):
        with pytest.raises(ConfigError, match="must be >= 0"):
            _cfg(lid_um=-5.0).validate()

    def test_non_integer_voxel_count_rejected(self):
        with pytest.raises(ConfigError, match="whole number of voxels"):
            stack_extents(_cfg(lid_um=55.0))

    def test_no_die_layer_rejected(self):
        with pytest.raises(ConfigError, match="no Si die layer"):
            stack_extents(_cfg(nz=10))
