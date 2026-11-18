"""
Round 3 task 1: one consistent scale and an exact stack split.

Covers the defaults (40 x 40 x 24 at 500 um / 10 um, 100/40/100 um stack),
the shipped JSON configs, and the rejection of stack thicknesses that are not
a whole number of voxels or that leave no die layer.
"""

from __future__ import annotations

import os

import pytest

from src.config_loader import (
    BoundaryConditionsSection,
    ConfigError,
    GridSection,
    PipelineConfig,
    PowerSection,
    SolverSection,
    StackSection,
    VisualizationSection,
    load_config,
    stack_layer_bounds,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_config(grid: GridSection = None, stack: StackSection = None) -> PipelineConfig:
    return PipelineConfig(
        grid=grid or GridSection(),
        boundary_conditions=BoundaryConditionsSection(),
        stack=stack or StackSection(),
        power=PowerSection(),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


class TestRound3Defaults:
    def test_default_grid_is_20mm_package_240um_stack(self):
        g = GridSection()
        assert (g.nx, g.ny, g.nz) == (40, 40, 24)
        assert g.dx == pytest.approx(500e-6)
        assert g.dy == pytest.approx(500e-6)
        assert g.dz == pytest.approx(10e-6)
        assert g.nz * g.dz == pytest.approx(240e-6)

    def test_default_stack_bounds_are_exact_voxel_faces(self):
        # 100 um interposer = 10 voxels, 40 um composite = 4 voxels, die = 10 voxels.
        config = make_config()
        assert stack_layer_bounds(config) == (10, 14)

    def test_default_config_validates(self):
        make_config().validate()


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["configs/config.json", "configs/stage2_config.json"])
    def test_shipped_config_loads_with_expected_stack(self, name):
        config = load_config(os.path.join(REPO_ROOT, name))
        assert (config.grid.nx, config.grid.ny, config.grid.nz) == (40, 40, 24)
        assert stack_layer_bounds(config) == (10, 14)


class TestExactSplit:
    def test_non_integer_interposer_is_rejected(self):
        # 45 um / 10 um = 4.5 voxels: must not be silently rounded.
        config = make_config(stack=StackSection(interposer_um=45.0, bump_um=40.0))
        with pytest.raises(ConfigError, match="interposer_um"):
            stack_layer_bounds(config)

    def test_non_integer_bump_is_rejected_when_dz_changes(self):
        # 40 um at dz=16 um = 2.5 voxels.
        config = make_config(grid=GridSection(nx=8, ny=8, nz=24, dx=500e-6, dy=500e-6, dz=16e-6),
                             stack=StackSection(interposer_um=96.0, bump_um=40.0))
        with pytest.raises(ConfigError, match="bump_um"):
            stack_layer_bounds(config)

    def test_stack_that_fills_the_grid_leaves_no_die_and_is_rejected(self):
        # 140 um of stack at 10 um voxels = 14 voxels, nz = 14: no Si die layer.
        config = make_config(grid=GridSection(nx=8, ny=8, nz=14),
                             stack=StackSection(interposer_um=100.0, bump_um=40.0))
        with pytest.raises(ConfigError, match="no Si die layer"):
            stack_layer_bounds(config)

    def test_non_positive_thickness_is_rejected_by_validate(self):
        with pytest.raises(ConfigError, match="must be positive"):
            make_config(stack=StackSection(interposer_um=100.0, bump_um=0.0)).validate()

    def test_bounds_are_strictly_ordered(self):
        config = make_config(grid=GridSection(nx=8, ny=8, nz=12),
                             stack=StackSection(interposer_um=10.0, bump_um=10.0))
        zi, zb = stack_layer_bounds(config)
        assert 0 < zi < zb < config.grid.nz
