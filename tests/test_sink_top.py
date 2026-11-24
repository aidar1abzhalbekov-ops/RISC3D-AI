"""
tests/test_sink_top.py
RISC3D-AI :: Heat sink on the top face (validation plan step 1)

The solver always applies the sink at index z=0. sink_side="top" reverses the
z axis of every 3-D array (mirror_sink_to_top) so the original top layer sits at
index 0. These tests check:
  1. the mirror is an involution and keeps the sink flag on index 0,
  2. a 1-D slab with a known analytic peak is reproduced in both orientations,
  3. mirror symmetry: sink-top with source at z=0 equals sink-bottom with source
     at z=nz-1, reversed,
  4. energy balance holds on the top-sink model,
  5. an invalid sink_side is rejected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import (  # noqa: E402
    BoundaryConditionsSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection,
)
from src.grid_builder import GridBuilder  # noqa: E402
from src.thermal_solver import (  # noqa: E402
    ThermalSolver, ThermalSolverError, mirror_sink_to_top,
)

H = 1000.0
TA_C = 25.0
TA_K = TA_C + 273.15
DX = 500e-6
DZ = 10e-6


def _slab_config(nx=4, ny=4, nz=12) -> PipelineConfig:
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz, dx=DX, dy=DX, dz=DZ),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=TA_C, h_convective=H, bottom_bc="robin",
        ),
        stack=StackSection(interposer_um=40.0, bump_um=20.0),
        power=PowerSection(die_power_w=1.0, die_footprint_frac=1.0),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


def _uniform_silicon_gb(cfg: PipelineConfig) -> GridBuilder:
    gb = GridBuilder(config=cfg.to_dict())
    gb.assign_layer(slice(0, cfg.grid.nz), "silicon")
    gb.tag_boundaries()
    gb.validate_grid()
    return gb


def _set_power_layer(gb: GridBuilder, z_index: int, total_w: float) -> None:
    """Uniform power over the full footprint in one z layer; total_w spread over nx*ny cells."""
    nx, ny = gb.config.nx, gb.config.ny
    cell_vol = DX * DX * DZ
    q = total_w / (nx * ny * cell_vol)  # W/m^3
    gb.q_map[...] = 0.0
    gb.q_map[:, :, z_index] = q


class TestMirror:
    def test_mirror_is_involution(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        arrays = gb.export_arrays()
        twice = mirror_sink_to_top(mirror_sink_to_top(arrays))
        for key in ("kx_map", "ky_map", "kz_map", "rho_map", "cp_map", "q_map", "boundary_mask"):
            assert np.array_equal(np.asarray(twice[key]), np.asarray(arrays[key])), key

    def test_sink_flag_stays_on_index_zero(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        m = mirror_sink_to_top(gb.export_arrays())["boundary_mask"]
        # index 0 plane: sink bit (0b01) set, adiabatic-top bit (0b10) not set
        assert np.all(m[:, :, 0] & 0b01)
        assert not np.any(m[:, :, 0] & 0b10)
        # index nz-1 plane: adiabatic bit set, sink bit not set
        assert np.all(m[:, :, -1] & 0b10)
        assert not np.any(m[:, :, -1] & 0b01)

    def test_mirror_reverses_power_layer(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        _set_power_layer(gb, z_index=cfg.grid.nz - 1, total_w=1.0)
        arrays = gb.export_arrays()
        mir = mirror_sink_to_top(arrays)
        assert np.array_equal(mir["q_map"][:, :, 0], arrays["q_map"][:, :, -1])

    def test_non_3d_array_rejected(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        arrays = gb.export_arrays()
        arrays["q_map"] = np.zeros(10)
        with pytest.raises(ThermalSolverError):
            mirror_sink_to_top(arrays)


class TestAnalyticSlab:
    """Uniform silicon slab, uniform power per footprint area, 1-D problem exactly.

    Sink at one face (h, Ta), the other face adiabatic. For total power P over
    area A with the source in one layer, the flux through the slab is P/A on
    the sink side, so the sink-face temperature rise is P/(h A) exactly. The
    source-side rise above the sink face is (P/A) * L_conduction / k, with
    L_conduction measured from the sink face to the source cell centre.
    """

    K_SI = 149.0

    def _peak_c(self, z_source: int, sink_side: str, cfg: PipelineConfig) -> float:
        gb = _uniform_silicon_gb(cfg)
        _set_power_layer(gb, z_index=z_source, total_w=1.0)
        solver = ThermalSolver.from_grid_builder(gb, sink_side=sink_side)
        # Re-apply the power in the solver's frame (mirror if needed).
        q = gb.q_map.copy()
        if sink_side == "top":
            q = q[:, :, ::-1]
        solver.q_map[...] = np.ascontiguousarray(q)
        T = solver.solve_steady_state(apply_result=False)
        return float(np.max(T)) - 273.15

    def test_source_at_sink_face_matches_p_over_hA(self):
        """Source next to the sink: peak = Ta + P/(hA) + conduction across half a cell."""
        cfg = _slab_config()
        nx, ny = cfg.grid.nx, cfg.grid.ny
        area = nx * DX * ny * DX
        expected_c = TA_C + 1.0 / (H * area)
        peak = self._peak_c(z_source=0, sink_side="bottom", cfg=cfg)
        assert peak == pytest.approx(expected_c, abs=0.01)

    def test_source_far_from_sink_adds_conduction_drop(self):
        """Source at the adiabatic face, sink at bottom: add (P/A) * L / k."""
        cfg = _slab_config()
        nx, ny, nz = cfg.grid.nx, cfg.grid.ny, cfg.grid.nz
        area = nx * DX * ny * DX
        L = (nz - 1) * DZ  # sink-face cell centre to source cell centre
        expected_c = TA_C + 1.0 / (H * area) + (1.0 / area) * L / self.K_SI
        peak = self._peak_c(z_source=nz - 1, sink_side="bottom", cfg=cfg)
        assert peak == pytest.approx(expected_c, abs=0.02)

    def test_top_sink_with_source_at_top_matches_bottom_sink_with_source_at_bottom(self):
        """Mirror symmetry: moving the sink and the source together gives the same peak."""
        cfg = _slab_config()
        nz = cfg.grid.nz
        peak_bottom_sink = self._peak_c(z_source=0, sink_side="bottom", cfg=cfg)
        peak_top_sink = self._peak_c(z_source=nz - 1, sink_side="top", cfg=cfg)
        assert peak_top_sink == pytest.approx(peak_bottom_sink, abs=1e-6)

    def test_source_next_to_top_sink_is_cooler_than_far(self):
        """Sink on top: source next to it must be cooler than source at the far face."""
        cfg = _slab_config()
        nz = cfg.grid.nz
        near = self._peak_c(z_source=nz - 1, sink_side="top", cfg=cfg)
        far = self._peak_c(z_source=0, sink_side="top", cfg=cfg)
        assert near < far


class TestTopSinkEnergyBalance:
    def test_boundary_outflow_equals_generation(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        _set_power_layer(gb, z_index=3, total_w=1.0)
        solver = ThermalSolver.from_grid_builder(gb, sink_side="top")
        q = gb.q_map[:, :, ::-1]
        solver.q_map[...] = np.ascontiguousarray(q)
        T = solver.solve_steady_state(apply_result=False)

        # Sink flux per unit face area is h * (T - Ta); integrate over the sink face.
        outflow = float(np.sum(H * DX * DX * (T[:, :, 0] - solver.bc.t_ambient_k)))
        generated = float(np.sum(solver.q_map)) * DX * DX * DZ
        assert outflow == pytest.approx(generated, rel=1e-6)


class TestSinkSideValidation:
    def test_invalid_sink_side_rejected(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        with pytest.raises(ThermalSolverError):
            ThermalSolver.from_grid_builder(gb, sink_side="side")

    def test_default_is_bottom(self):
        cfg = _slab_config()
        gb = _uniform_silicon_gb(cfg)
        solver = ThermalSolver.from_grid_builder(gb)
        assert solver.sink_side == "bottom"
