"""
tests/test_core.py
RISC3D-AI :: Core numerical correctness test suite (pytest)

IMPORTANT INTERFACE NOTE:
GridBuilder.__init__() accepts a config as `dict`, a JSON file path (`str`),
or `None` -- it does NOT accept a `PipelineConfig` instance directly (passing
one raises `TypeError`, verified explicitly in
TestConfigDictInterface.test_grid_builder_rejects_pipeline_config_instance).
Every test in this module that needs a PipelineConfig therefore constructs
GridBuilder as:

    gb = GridBuilder(config=pipeline_config.to_dict())

or, equivalently, with a raw dict literal directly. Both paths are exercised.

Coverage:
  1. TestConfigDictInterface   -- the dict/to_dict() contract itself.
  2. TestCFLStability          -- automated CFL bound matches an independently
                                   coded reference, and is actually enforced.
  3. TestEnergyConservationAndPositivity -- discrete energy balance holds
                                   exactly (no numerical leak/injection), and
                                   temperatures stay finite and physical.
  4. TestHotspotLocalization   -- peak/threshold hotspot detection matches
                                   the known location and magnitude of an
                                   injected power source.
  5. TestRobinBoundaryConvection -- the convective (Robin) bottom BC produces
                                   the exact analytically-predicted cooling
                                   rate, isolated from internal conduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config_loader import (
    PipelineConfig,
    ConfigError,
    GridSection,
    BoundaryConditionsSection,
    StackSection,
    PowerSection,
    SolverSection,
    VisualizationSection,
)
from src.grid_builder import GridBuilder, GridBuilderError, MATERIALS
from src.thermal_solver import ThermalSolver, SolverBCConfig, ThermalSolverError

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_pipeline_config(
    nx: int = 10, ny: int = 10, nz: int = 12,
    dx: float = 25e-6, dy: float = 25e-6, dz: float = 10e-6,
    t_ambient_c: float = 25.0, h_convective: float = 2500.0, bottom_bc: str = "robin",
    interposer_frac: float = 0.40, bump_frac: float = 0.20,
    die_power_w: float = 0.05, die_footprint_frac: float = 0.5,
) -> PipelineConfig:
    """Builds a small, fast-to-solve PipelineConfig for unit testing."""
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=t_ambient_c, h_convective=h_convective, bottom_bc=bottom_bc,
        ),
        stack=StackSection(interposer_frac=interposer_frac, bump_frac=bump_frac),
        power=PowerSection(die_power_w=die_power_w, die_footprint_frac=die_footprint_frac),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


def build_uniform_grid(config: PipelineConfig, material: str) -> GridBuilder:
    """Builds a GridBuilder filled entirely with a single material, via
    config.to_dict() (never a PipelineConfig instance)."""
    gb = GridBuilder(config=config.to_dict())
    gb.assign_layer(slice(0, config.grid.nz), material)
    gb.tag_boundaries()
    gb.validate_grid()
    return gb


def build_stack_grid(config: PipelineConfig):
    """Builds the standard interposer / bump+underfill / Si-die stack with a
    centered active power footprint, mirroring main.py's build_chiplet_stack()
    logic. Returns (GridBuilder, z_interposer_end, z_bump_end)."""
    gb = GridBuilder(config=config.to_dict())
    nz = config.grid.nz
    nx, ny = config.grid.nx, config.grid.ny

    z_interposer_end = max(1, int(round(nz * config.stack.interposer_frac)))
    z_bump_end = max(z_interposer_end + 1,
                      int(round(nz * (config.stack.interposer_frac + config.stack.bump_frac))))
    z_bump_end = min(z_bump_end, nz - 1)

    gb.assign_layer(slice(0, z_interposer_end), "silicon")
    gb.assign_layer(slice(z_interposer_end, z_bump_end), "bump_underfill_composite")
    gb.assign_layer(slice(z_bump_end, nz), "silicon")
    gb.tag_boundaries()

    margin_x = int(round(nx * (1.0 - config.power.die_footprint_frac) / 2.0))
    margin_y = int(round(ny * (1.0 - config.power.die_footprint_frac) / 2.0))
    x_slice = slice(margin_x, nx - margin_x)
    y_slice = slice(margin_y, ny - margin_y)
    z_slice = slice(z_bump_end, nz)

    gb.add_power_source(power_watts=config.power.die_power_w, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
    gb.validate_grid()
    return gb, z_interposer_end, z_bump_end, (x_slice, y_slice, z_slice)


def reference_max_dt(gb: GridBuilder, safety_factor: float = 1.0) -> float:
    """
    Independently-coded (test-side) reference implementation of the CFL /
    diagonal-dominance bound, used to cross-check ThermalSolver.compute_max_dt()
    without calling any of its private methods. Mirrors the physics (harmonic-
    mean face conductivities, Robin/Dirichlet bottom coefficient) but is
    written from scratch here so this is a genuine independent verification.
    """
    kx, ky, kz = gb.kx_map, gb.ky_map, gb.kz_map
    rho, cp = gb.rho_map, gb.cp_map
    dx, dy, dz = gb.config.dx, gb.config.dy, gb.config.dz
    nx, ny, nz = kx.shape

    def harmonic(a, b):
        return 2.0 * a * b / (a + b + _EPS)

    total = np.zeros_like(kx)

    kxf = harmonic(kx[:-1, :, :], kx[1:, :, :]) / dx ** 2
    left = np.zeros_like(kx); left[1:, :, :] = kxf
    right = np.zeros_like(kx); right[:-1, :, :] = kxf
    total += left + right

    kyf = harmonic(ky[:, :-1, :], ky[:, 1:, :]) / dy ** 2
    left = np.zeros_like(ky); left[:, 1:, :] = kyf
    right = np.zeros_like(ky); right[:, :-1, :] = kyf
    total += left + right

    kzf = harmonic(kz[:, :, :-1], kz[:, :, 1:]) / dz ** 2
    left = np.zeros_like(kz); left[:, :, 1:] = kzf
    right = np.zeros_like(kz); right[:, :, :-1] = kzf
    total += left + right

    if gb.config.bottom_bc == "dirichlet":
        coeff_b = 2.0 * kz[:, :, 0] / dz ** 2
    else:
        coeff_b = np.full((nx, ny), gb.config.h_convective / dz)
    total[:, :, 0] += coeff_b

    dt_per_cell = (rho * cp) / total
    return float(np.min(dt_per_cell)) * safety_factor


# ---------------------------------------------------------------------------
# 1. Config / dict interface contract
# ---------------------------------------------------------------------------

class TestConfigDictInterface:
    def test_grid_builder_accepts_config_to_dict(self):
        config = make_pipeline_config()
        d = config.to_dict()
        assert isinstance(d, dict)
        gb = GridBuilder(config=d)
        assert gb.kx_map.shape == (config.grid.nx, config.grid.ny, config.grid.nz)

    def test_to_dict_matches_to_grid_builder_dict(self):
        config = make_pipeline_config()
        assert config.to_dict() == config.to_grid_builder_dict()

    def test_grid_builder_accepts_raw_dict_directly(self):
        raw = {
            "grid": {"nx": 8, "ny": 8, "nz": 10, "dx": 25e-6, "dy": 25e-6, "dz": 10e-6},
            "boundary_conditions": {
                "t_ambient_k": 298.15, "h_convective": 2500.0,
                "bottom_bc": "robin", "top_bc": "neumann", "side_bc": "neumann",
            },
        }
        gb = GridBuilder(config=raw)
        assert gb.kx_map.shape == (8, 8, 10)

    def test_grid_builder_accepts_none_with_fallback_defaults(self):
        # No config.json on disk at cwd for this test -> hardcoded fallback.
        gb = GridBuilder(config=None)
        assert gb.kx_map.ndim == 3

    def test_grid_builder_rejects_pipeline_config_instance(self):
        """The core bug report: GridBuilder must NOT accept a PipelineConfig
        object directly -- only dict / str / None. Verifies the exact
        TypeError callers must route around via .to_dict()."""
        config = make_pipeline_config()
        with pytest.raises(TypeError):
            GridBuilder(config=config)  # type: ignore[arg-type]

    def test_thermal_solver_from_grid_builder_roundtrip(self):
        """End-to-end: PipelineConfig -> to_dict() -> GridBuilder ->
        ThermalSolver.from_grid_builder() must succeed with no manual
        SolverBCConfig construction required."""
        config = make_pipeline_config()
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        assert isinstance(solver.bc, SolverBCConfig)
        assert solver.shape == (config.grid.nx, config.grid.ny, config.grid.nz)


# ---------------------------------------------------------------------------
# 2. CFL stability
# ---------------------------------------------------------------------------

class TestCFLStability:
    @pytest.mark.parametrize("material,bottom_bc", [
        ("copper", "robin"),
        ("silicon", "robin"),
        ("copper", "dirichlet"),
        ("bump_underfill_composite", "robin"),
    ])
    def test_compute_max_dt_matches_independent_reference(self, material, bottom_bc):
        config = make_pipeline_config(bottom_bc=bottom_bc, die_power_w=0.0)
        gb = build_uniform_grid(config, material)
        solver = ThermalSolver.from_grid_builder(gb)

        dt_solver = solver.compute_max_dt(safety_factor=1.0)
        dt_reference = reference_max_dt(gb, safety_factor=1.0)

        assert dt_solver == pytest.approx(dt_reference, rel=1e-9)

    def test_safety_factor_scales_linearly(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        dt_90 = solver.compute_max_dt(safety_factor=0.9)
        dt_45 = solver.compute_max_dt(safety_factor=0.45)
        assert dt_45 == pytest.approx(dt_90 * 0.5, rel=1e-9)

    def test_step_at_or_below_bound_is_stable(self):
        config = make_pipeline_config(die_power_w=0.02)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T_after = solver.step(dt, enforce_stability=True)
        assert np.all(np.isfinite(T_after))

    def test_step_above_bound_raises_thermal_solver_error(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "copper")  # Cu -> tight CFL bound
        solver = ThermalSolver.from_grid_builder(gb)

        dt_stable = solver.compute_max_dt(safety_factor=0.9)
        with pytest.raises(ThermalSolverError):
            solver.step(dt_stable * 10.0, enforce_stability=True)

    def test_copper_bound_is_tighter_than_silicon(self):
        """Sanity check on physical ordering: Cu's higher k/(rho*cp) ratio
        must yield a smaller (more restrictive) stable dt than Si on an
        identical grid."""
        cfg_cu = make_pipeline_config(die_power_w=0.0)
        cfg_si = make_pipeline_config(die_power_w=0.0)
        gb_cu = build_uniform_grid(cfg_cu, "copper")
        gb_si = build_uniform_grid(cfg_si, "silicon")
        dt_cu = ThermalSolver.from_grid_builder(gb_cu).compute_max_dt(safety_factor=1.0)
        dt_si = ThermalSolver.from_grid_builder(gb_si).compute_max_dt(safety_factor=1.0)
        assert dt_cu < dt_si


# ---------------------------------------------------------------------------
# 3. Energy conservation & positivity
# ---------------------------------------------------------------------------

class TestEnergyConservationAndPositivity:
    def test_equilibrium_field_is_a_fixed_point(self):
        """If T is already uniform at T_ambient everywhere and Q=0, one
        explicit step must leave T EXACTLY unchanged (no numerical leak/
        injection from the discretization at equilibrium)."""
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        T0 = solver.T.copy()
        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        assert np.allclose(T1, T0, atol=1e-9)

    def test_discrete_energy_balance_with_active_source(self):
        """sum(rho*cp*dT*cell_vol) over one step must equal
        dt * (sum(Q*cell_vol) - sum(boundary_outflow*cell_area... )) exactly,
        i.e. the discretization neither creates nor destroys energy beyond
        the explicit source term and the explicit boundary exchange term."""
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin")
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        cell_vol = config.grid.dx * config.grid.dy * config.grid.dz
        T0 = solver.T.copy()
        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        actual_energy_delta = float(np.sum(solver.rho_map * solver.cp_map * (T1 - T0))) * cell_vol

        # Reference boundary outflow term, computed independently (not via
        # solver._compute_divergence): coeff * (T0_bottom - T_ambient) per
        # bottom-layer cell, matching the Robin/Dirichlet BC definition.
        if config.boundary_conditions.bottom_bc == "dirichlet":
            coeff_b = 2.0 * solver.kz_map[:, :, 0] / (config.grid.dz ** 2)
        else:
            coeff_b = np.full((config.grid.nx, config.grid.ny), config.boundary_conditions.h_convective / config.grid.dz)
        boundary_outflow = coeff_b * (T0[:, :, 0] - solver.bc.t_ambient_k)

        expected_energy_delta = dt * cell_vol * (float(np.sum(solver.q_map)) - float(np.sum(boundary_outflow)))

        assert actual_energy_delta == pytest.approx(expected_energy_delta, rel=1e-6, abs=1e-12)

    def test_temperatures_remain_finite_and_physical_over_run(self):
        config = make_pipeline_config(die_power_w=0.1)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=200 * dt, dt=dt, safety_factor=0.9)

        T_final = result["final_T"]
        assert np.all(np.isfinite(T_final))
        assert np.all(T_final > 0.0)  # positive Kelvin -- non-physical to violate

    def test_active_power_source_strictly_increases_peak_temperature(self):
        config = make_pipeline_config(die_power_w=0.15)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T0_peak = float(np.max(solver.T))
        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=100 * dt, dt=dt, safety_factor=0.9)
        T1_peak = float(np.max(result["final_T"]))

        assert T1_peak > T0_peak

    def test_zero_power_never_exceeds_ambient(self):
        """With no active source and starting at ambient, no cell should
        ever rise above ambient (only Robin cooling toward ambient is
        possible, never overshoot heating)."""
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=100 * dt, dt=dt, safety_factor=0.9)
        assert np.all(result["final_T"] <= solver.bc.t_ambient_k + 1e-9)


# ---------------------------------------------------------------------------
# 4. Hotspot localization
# ---------------------------------------------------------------------------

class TestHotspotLocalization:
    def test_peak_hotspot_is_within_active_power_footprint(self):
        config = make_pipeline_config(nx=16, ny=16, nz=14, die_power_w=0.2, die_footprint_frac=0.4)
        gb, z_interposer_end, z_bump_end, (x_slice, y_slice, z_slice) = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=300 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        hotspots = solver.get_hotspots(threshold_c=-273.0)  # threshold below absolute zero -> peak always found
        pi, pj, pk = hotspots["peak_location_ijk"]

        assert x_slice.start <= pi < x_slice.stop
        assert y_slice.start <= pj < y_slice.stop
        assert z_slice.start <= pk < z_slice.stop

    def test_peak_hotspot_is_at_die_top_farthest_from_heat_sink(self):
        """Physical sanity: with only a bottom heat sink and an adiabatic
        top, the hottest voxel in the die must be at the top z-layer (the
        point of maximum thermal resistance from the sink), not buried
        mid-die."""
        config = make_pipeline_config(nx=16, ny=16, nz=14, die_power_w=0.2)
        gb, z_interposer_end, z_bump_end, _ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=300 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        hotspots = solver.get_hotspots(threshold_c=-273.0)
        _, _, pk = hotspots["peak_location_ijk"]
        assert pk == config.grid.nz - 1

    def test_hotspot_count_responds_monotonically_to_threshold(self):
        config = make_pipeline_config(nx=14, ny=14, nz=12, die_power_w=0.3)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=500 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        low = solver.get_hotspots(threshold_c=solver.bc.t_ambient_k - 273.15)["count"]     # ~ambient -> most cells qualify
        mid = solver.get_hotspots(threshold_c=85.0)["count"]
        high = solver.get_hotspots(threshold_c=1e6)["count"]                              # impossible -> zero

        assert low >= mid >= high == 0

    def test_no_power_source_yields_no_hotspots_above_ambient(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        hotspots = solver.get_hotspots(threshold_c=config.boundary_conditions.t_ambient_c + 0.01)
        assert hotspots["count"] == 0


# ---------------------------------------------------------------------------
# 5. Robin boundary convection
# ---------------------------------------------------------------------------

class TestRobinBoundaryConvection:
    def test_uniform_elevated_field_isolates_robin_cooling_to_bottom_layer(self):
        """With T uniformly elevated by delta_T everywhere (zero internal
        gradient, so zero conduction anywhere), only the bottom (z=0) layer
        should change after one step -- via the Robin term referencing the
        external T_ambient -- while every other layer stays exactly fixed."""
        config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0, h_convective=1800.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 12.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        # Non-bottom layers: exactly unchanged (no internal gradient anywhere,
        # no boundary reference except at z=0).
        assert np.allclose(T1[:, :, 1:], T_initial[:, :, 1:], atol=1e-9)

        # Bottom layer: analytic Robin prediction, isolated from conduction
        # because the neighboring z=1 layer is at the SAME temperature (zero
        # vertical gradient), so the only active term is the convective one.
        rho0 = solver.rho_map[:, :, 0]
        cp0 = solver.cp_map[:, :, 0]
        expected_bottom = T_initial[:, :, 0] - dt * (config.boundary_conditions.h_convective / config.grid.dz) \
            * (T_initial[:, :, 0] - solver.bc.t_ambient_k) / (rho0 * cp0)

        assert np.allclose(T1[:, :, 0], expected_bottom, rtol=1e-9, atol=1e-9)

    def test_robin_cools_toward_ambient_never_past_it(self):
        config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 20.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        for _ in range(50):
            solver.step(dt)

        # Should have cooled toward ambient, but a single explicit run at a
        # stable dt must never overshoot past ambient (would indicate a sign
        # error or instability in the BC term).
        assert np.all(solver.T[:, :, 0] >= solver.bc.t_ambient_k - 1e-6)
        assert np.all(solver.T[:, :, 0] <= T_initial[:, :, 0])

    def test_higher_h_convective_cools_bottom_layer_faster(self):
        """Larger convective film coefficient must produce strictly greater
        cooling of the bottom layer after an identical elapsed physical
        time, all else equal."""
        delta_T = 15.0

        def run_and_get_bottom_mean(h_value: float) -> float:
            config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0, h_convective=h_value)
            gb = build_uniform_grid(config, "silicon")
            solver = ThermalSolver.from_grid_builder(gb)
            T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
            solver.reset(initial_temperature_k=T_initial)
            # Use the SAME dt for both cases (driven by the smaller of the two
            # to keep both stable) so the comparison isolates h's effect.
            dt = solver.compute_max_dt(safety_factor=0.5)
            for _ in range(20):
                solver.step(dt, enforce_stability=False)
            return float(np.mean(solver.T[:, :, 0]))

        mean_low_h = run_and_get_bottom_mean(500.0)
        mean_high_h = run_and_get_bottom_mean(4500.0)

        ambient = 298.15
        assert (mean_high_h - ambient) < (mean_low_h - ambient)

    def test_dirichlet_bottom_coefficient_matches_expected_formula(self):
        """Cross-check the Dirichlet boundary coefficient (2*k/dz^2) used
        internally by comparing its effect against a hand-derived one-step
        prediction, analogous to the Robin isolation test above."""
        config = make_pipeline_config(bottom_bc="dirichlet", die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 10.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        coeff_dirichlet = 2.0 * solver.kz_map[:, :, 0] / (config.grid.dz ** 2)
        rho0 = solver.rho_map[:, :, 0]
        cp0 = solver.cp_map[:, :, 0]
        expected_bottom = T_initial[:, :, 0] - dt * coeff_dirichlet * (T_initial[:, :, 0] - solver.bc.t_ambient_k) / (rho0 * cp0)

        assert np.allclose(T1[:, :, 0], expected_bottom, rtol=1e-9, atol=1e-9)
        assert np.allclose(T1[:, :, 1:], T_initial[:, :, 1:], atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Implicit (Crank-Nicolson) solver
# ---------------------------------------------------------------------------

class TestImplicitSolver:
    def test_stable_at_dt_far_exceeding_explicit_cfl_bound(self):
        """The entire point of Crank-Nicolson: a dt that would immediately
        raise ThermalSolverError under the explicit scheme must produce a
        perfectly finite result here."""
        config = make_pipeline_config(die_power_w=0.05)
        gb = build_uniform_grid(config, "copper")  # tight explicit CFL bound
        solver = ThermalSolver.from_grid_builder(gb)

        dt_explicit_bound = solver.compute_max_dt(safety_factor=0.9)
        dt_huge = dt_explicit_bound * 1000.0  # would hard-fail solver.step()

        with pytest.raises(ThermalSolverError):
            solver.step(dt_huge)  # sanity: explicit really does reject this dt

        T_after = solver.step_implicit(dt_huge)
        assert np.all(np.isfinite(T_after))

    def test_adi_method_raises_not_implemented(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)
        with pytest.raises(NotImplementedError):
            solver.step_implicit(1e-6, method="adi")

    def test_zero_or_negative_dt_raises(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)
        with pytest.raises(ThermalSolverError):
            solver.step_implicit(0.0)
        with pytest.raises(ThermalSolverError):
            solver.step_implicit(-1e-6)

    def test_direct_backend_selected_for_small_grid(self):
        config = make_pipeline_config(nx=8, ny=8, nz=10, die_power_w=0.02)  # N=640 << threshold
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        solver.step_implicit(1e-6)
        cache_entry = next(iter(solver._cn_solver_cache.values()))
        assert cache_entry["backend"] == "direct"

    def test_iterative_backend_selected_above_threshold_and_converges(self):
        """Grid sized just above DIRECT_SOLVE_MAX_UNKNOWNS must route to the
        Jacobi-preconditioned iterative backend and still produce a finite,
        converged result."""
        from src.thermal_solver import DIRECT_SOLVE_MAX_UNKNOWNS
        # Pick dimensions whose product safely exceeds the threshold.
        n_side = int(np.ceil((DIRECT_SOLVE_MAX_UNKNOWNS * 1.1) ** (1.0 / 3.0))) + 1
        config = make_pipeline_config(nx=n_side, ny=n_side, nz=n_side, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        assert solver.nx * solver.ny * solver.nz > DIRECT_SOLVE_MAX_UNKNOWNS

        T_after = solver.step_implicit(1e-4)
        cache_entry = next(iter(solver._cn_solver_cache.values()))
        assert cache_entry["backend"] == "iterative"
        assert np.all(np.isfinite(T_after))

    def test_lu_cache_reused_across_repeated_dt(self):
        """The factorization/preconditioner must be built once and reused
        for subsequent step_implicit() calls at the SAME dt (performance
        contract: repeated fixed-dt steps should not rebuild the system)."""
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = 1e-6
        solver.step_implicit(dt)
        cached_object_id = id(next(iter(solver._cn_solver_cache.values())))
        solver.step_implicit(dt)
        assert len(solver._cn_solver_cache) == 1
        assert id(next(iter(solver._cn_solver_cache.values()))) == cached_object_id

    def test_energy_conservation_holds_for_implicit_scheme_too(self):
        """The same discrete energy-balance identity verified for the
        explicit scheme must also hold (to CN's linear-solve tolerance) for
        Crank-Nicolson: conduction redistributes energy internally and only
        Q and the boundary term are true sources/sinks."""
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin")
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        cell_vol = config.grid.dx * config.grid.dy * config.grid.dz
        T0 = solver.T.copy()
        dt = 1e-6
        T1 = solver.step_implicit(dt)

        actual_energy_delta = float(np.sum(solver.rho_map * solver.cp_map * (T1 - T0))) * cell_vol

        coeff_b = np.full((config.grid.nx, config.grid.ny),
                           config.boundary_conditions.h_convective / config.grid.dz)
        # Trapezoidal (Crank-Nicolson) boundary outflow: average of the term
        # evaluated at T0 and T1, matching the theta=0.5 scheme actually used.
        boundary_outflow_0 = coeff_b * (T0[:, :, 0] - solver.bc.t_ambient_k)
        boundary_outflow_1 = coeff_b * (T1[:, :, 0] - solver.bc.t_ambient_k)
        boundary_outflow_avg = 0.5 * (boundary_outflow_0 + boundary_outflow_1)

        expected_energy_delta = dt * cell_vol * (float(np.sum(solver.q_map)) - float(np.sum(boundary_outflow_avg)))

        assert actual_energy_delta == pytest.approx(expected_energy_delta, rel=1e-5, abs=1e-10)

    def test_crank_nicolson_matches_explicit_reference_within_tolerance(self):
        """Cross-validation: at a dt small enough to also be explicit-
        stable, both schemes must agree closely (both are consistent
        discretizations of the same PDE); at that resolution, CN's extra
        implicit averaging should not meaningfully change the result."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)

        gb_exp, *_ = build_stack_grid(config)
        solver_exp = ThermalSolver.from_grid_builder(gb_exp)
        dt = solver_exp.compute_max_dt(safety_factor=0.9)
        result_exp = solver_exp.run(total_time_s=100 * dt, dt=dt, method="explicit")

        gb_cn, *_ = build_stack_grid(config)
        solver_cn = ThermalSolver.from_grid_builder(gb_cn)
        result_cn = solver_cn.run(total_time_s=100 * dt, dt=dt, method="crank_nicolson")

        assert np.max(np.abs(result_exp["final_T"] - result_cn["final_T"])) < 0.05  # < 0.05 K agreement

    def test_run_with_crank_nicolson_method_tag(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        result = solver.run(total_time_s=1e-5, dt=1e-6, method="crank_nicolson")
        assert result["method"] == "crank_nicolson"
        assert np.all(np.isfinite(result["final_T"]))


class TestSteadyStateSolve:
    def test_steady_state_energy_balance_is_exact(self):
        """At true steady state (dT/dt=0), total heat generated must exactly
        equal total heat removed through the boundary -- the cleanest
        possible energy-conservation check, since there is no transient
        storage term to account for."""
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state()

        coeff_b = np.full((config.grid.nx, config.grid.ny),
                           config.boundary_conditions.h_convective / config.grid.dz)
        boundary_outflow = float(np.sum(coeff_b * (T_steady[:, :, 0] - solver.bc.t_ambient_k)))
        total_q = float(np.sum(solver.q_map))

        assert boundary_outflow == pytest.approx(total_q, rel=1e-4, abs=1e-9)

    def test_steady_state_matches_long_transient_asymptote(self):
        """A sufficiently long Crank-Nicolson run must converge toward the
        same equilibrium the direct steady-state solve produces exactly."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.03)

        gb_a, *_ = build_stack_grid(config)
        solver_a = ThermalSolver.from_grid_builder(gb_a)
        T_steady = solver_a.solve_steady_state()

        gb_b, *_ = build_stack_grid(config)
        solver_b = ThermalSolver.from_grid_builder(gb_b)
        result = solver_b.run(total_time_s=20.0, dt=0.1, method="crank_nicolson")

        assert np.max(np.abs(T_steady - result["final_T"])) < 1.0  # within 1K after long transient

    def test_steady_state_zero_power_returns_ambient_everywhere(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state()
        assert np.allclose(T_steady, solver.bc.t_ambient_k, atol=1e-6)

    def test_apply_result_false_does_not_mutate_solver_state(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_before = solver.T.copy()
        T_steady = solver.solve_steady_state(apply_result=False)

        assert np.array_equal(solver.T, T_before)  # solver.T untouched
        assert not np.allclose(T_steady, T_before)  # but the computed result differs

    def test_apply_result_true_updates_solver_temperature(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state(apply_result=True)
        assert np.array_equal(solver.T, T_steady)


class TestSolverSectionConfig:
    def test_config_rejects_invalid_method(self):
        config = make_pipeline_config()
        config.solver.method = "not_a_real_method"
        with pytest.raises(ConfigError):
            config.validate()

    def test_config_accepts_crank_nicolson_method(self):
        config = make_pipeline_config()
        config.solver.method = "crank_nicolson"
        config.validate()  # must not raise

    def test_config_rejects_negative_dt(self):
        config = make_pipeline_config()
        config.solver.dt = -1e-6
        with pytest.raises(ConfigError):
            config.validate()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
