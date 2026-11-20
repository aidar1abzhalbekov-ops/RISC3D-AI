"""
tests/test_steady_iterative.py
RISC3D-AI :: Iterative steady-state path (ILU-preconditioned BiCGSTAB)

The direct LU path is used for N <= STEADY_DIRECT_MAX_UNKNOWNS. These tests
force the iterative path on small grids by patching the threshold to 0, then
check that:
  1. the iterative solution matches the direct LU solution,
  2. the ILU factor is built once and reused across solves,
  3. the relative-residual guard rejects a solve that does not meet the limit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.thermal_solver as ts  # noqa: E402
from src.thermal_solver import ThermalSolver, ThermalSolverError  # noqa: E402
from tests.test_core import make_pipeline_config, build_stack_grid  # noqa: E402


def _solver(die_power_w: float = 0.05) -> ThermalSolver:
    config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=die_power_w)
    gb, *_ = build_stack_grid(config)
    return ThermalSolver.from_grid_builder(gb)


@pytest.fixture
def force_iterative(monkeypatch):
    """Route solve_steady_state() through the iterative branch."""
    monkeypatch.setattr(ts, "STEADY_DIRECT_MAX_UNKNOWNS", 0)


class TestIterativeSteadyPath:
    def test_iterative_matches_direct_lu(self, monkeypatch):
        direct = _solver().solve_steady_state(apply_result=False)

        monkeypatch.setattr(ts, "STEADY_DIRECT_MAX_UNKNOWNS", 0)
        iterative_solver = _solver()
        iterative = iterative_solver.solve_steady_state(apply_result=False)

        assert iterative_solver._steady_ilu is not None  # iterative branch really ran
        assert np.max(np.abs(iterative - direct)) < 1e-3  # sub-millikelvin agreement

    def test_ilu_factor_is_built_once_and_reused(self, force_iterative):
        solver = _solver()
        solver.solve_steady_state(apply_result=False)
        first = solver._steady_ilu
        assert first is not None

        solver.solve_steady_state(apply_result=False)
        assert solver._steady_ilu is first  # same object, no refactorization

    def test_ilu_preconditioner_is_passed_to_krylov_solver(self, force_iterative, monkeypatch):
        """Small grids converge without a preconditioner too, so convergence
        alone cannot prove the ILU is applied. Spy on bicgstab and require a
        non-None preconditioner on every call."""
        import scipy.sparse.linalg as spla
        real_bicgstab = spla.bicgstab
        calls = []

        def spy(A, b, *args, **kwargs):
            calls.append(kwargs.get("M"))
            return real_bicgstab(A, b, *args, **kwargs)

        monkeypatch.setattr(spla, "bicgstab", spy)
        monkeypatch.setattr(ts.sparse_linalg, "bicgstab", spy)
        _solver().solve_steady_state(apply_result=False)

        assert len(calls) == 1
        assert calls[0] is not None

    def test_iterative_energy_balance_is_exact(self, force_iterative):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        T = solver.solve_steady_state(apply_result=False)

        coeff_b = np.full((config.grid.nx, config.grid.ny),
                          config.boundary_conditions.h_convective / config.grid.dz)
        boundary_outflow = float(np.sum(coeff_b * (T[:, :, 0] - solver.bc.t_ambient_k)))
        total_q = float(np.sum(solver.q_map))
        assert boundary_outflow == pytest.approx(total_q, rel=1e-4, abs=1e-9)


class TestResidualGuard:
    def test_residual_guard_rejects_solution_above_limit(self, force_iterative, monkeypatch):
        # Any real solve has a residual above 1e-30, so the guard must fire.
        monkeypatch.setattr(ts, "STEADY_RESIDUAL_MAX", 1e-30)
        solver = _solver()
        with pytest.raises(ThermalSolverError, match="relative"):
            solver.solve_steady_state(apply_result=False)

    def test_residual_guard_accepts_solution_within_limit(self, force_iterative):
        solver = _solver()
        T = solver.solve_steady_state(apply_result=False)
        assert np.all(np.isfinite(T))
