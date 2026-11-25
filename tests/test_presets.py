"""
tests/test_presets.py
RISC3D-AI :: Named cooling presets (validation plan step 2)

Hand-checked values, 40x40 lateral mesh at 500 um, so the domain is 20 mm x 20 mm
(A_domain = 4e-4 m^2) and the die footprint is 50% per side (10 mm x 10 mm = 1 cm^2):

  passive-lowpower: P = 12 W, h = 1000 -> floor rise P/(hA) = 30 K, R_ja floor = 1/(hA) = 2.5 K/W,
                    power density 12 W/cm^2
  cowos-coldplate:  P = 150 W, h = 25000 -> floor rise 15 K, R_ja floor = 0.1 K/W,
                    power density 150 W/cm^2
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as main_mod  # noqa: E402
from src.config_loader import (  # noqa: E402
    COOLING_PRESETS, ConfigError, PipelineConfig, GridSection, BoundaryConditionsSection,
    PowerSection, SolverSection, StackSection, VisualizationSection,
    apply_cooling_preset, cooling_metrics, die_footprint_m2,
)

A_DOMAIN = 4e-4  # m^2
T_AMB = 25.0


def _config() -> PipelineConfig:
    return PipelineConfig(
        grid=GridSection(nx=40, ny=40, nz=24, dx=500e-6, dy=500e-6, dz=10e-6),
        boundary_conditions=BoundaryConditionsSection(t_ambient_c=T_AMB, h_convective=1000.0, bottom_bc="robin"),
        stack=StackSection(),
        power=PowerSection(die_power_w=12.0, die_footprint_frac=0.5),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


class TestPresetTable:
    def test_exact_preset_values(self):
        assert COOLING_PRESETS["passive-lowpower"]["die_power_w"] == 12.0
        assert COOLING_PRESETS["passive-lowpower"]["h_convective"] == 1000.0
        assert COOLING_PRESETS["cowos-coldplate"]["die_power_w"] == 150.0
        assert COOLING_PRESETS["cowos-coldplate"]["h_convective"] == 25000.0

    def test_apply_sets_power_and_h_together(self):
        cfg = apply_cooling_preset(_config(), "cowos-coldplate")
        assert cfg.power.die_power_w == 150.0
        assert cfg.boundary_conditions.h_convective == 25000.0

    def test_apply_leaves_other_fields_alone(self):
        cfg = _config()
        before_grid = (cfg.grid.nx, cfg.grid.dx, cfg.boundary_conditions.t_ambient_c)
        apply_cooling_preset(cfg, "passive-lowpower")
        assert (cfg.grid.nx, cfg.grid.dx, cfg.boundary_conditions.t_ambient_c) == before_grid

    def test_unknown_preset_rejected(self):
        with pytest.raises(ConfigError, match="unknown cooling preset"):
            apply_cooling_preset(_config(), "liquid-nitrogen")


class TestFootprintAndMetrics:
    def test_die_footprint_is_one_cm2_at_half_fraction(self):
        assert die_footprint_m2(_config()) == pytest.approx(1e-4, rel=1e-12)

    def test_passive_power_density_and_floor_resistance(self):
        cfg = apply_cooling_preset(_config(), "passive-lowpower")
        floor_peak = T_AMB + cfg.power.die_power_w / (cfg.boundary_conditions.h_convective * A_DOMAIN)
        assert floor_peak == pytest.approx(55.0, abs=1e-9)  # 25 + 30
        m = cooling_metrics(cfg, floor_peak)
        assert m["w_per_cm2"] == pytest.approx(12.0, rel=1e-12)
        assert m["r_ja_k_per_w"] == pytest.approx(2.5, rel=1e-12)  # = 1/(hA)

    def test_coldplate_power_density_and_floor_resistance(self):
        cfg = apply_cooling_preset(_config(), "cowos-coldplate")
        floor_peak = T_AMB + cfg.power.die_power_w / (cfg.boundary_conditions.h_convective * A_DOMAIN)
        assert floor_peak == pytest.approx(40.0, abs=1e-9)  # 25 + 15
        m = cooling_metrics(cfg, floor_peak)
        assert m["w_per_cm2"] == pytest.approx(150.0, rel=1e-12)
        assert m["r_ja_k_per_w"] == pytest.approx(0.1, rel=1e-12)  # = 1/(hA)

    def test_zero_power_gives_no_resistance(self):
        cfg = _config()
        cfg.power.die_power_w = 0.0
        assert cooling_metrics(cfg, 25.0)["r_ja_k_per_w"] is None


class TestCli:
    def test_parser_accepts_known_preset(self):
        args = main_mod.build_cli_parser().parse_args(["--preset", "cowos-coldplate"])
        assert args.preset == "cowos-coldplate"

    def test_parser_default_is_no_preset(self):
        args = main_mod.build_cli_parser().parse_args([])
        assert args.preset is None

    def test_parser_rejects_unknown_preset(self):
        with pytest.raises(SystemExit):
            main_mod.build_cli_parser().parse_args(["--preset", "bogus"])


class TestCliResultsPath:
    """End-to-end: main.main() prints R_ja from the steady solve, and warns when the
    short transient is far from steady state. Values measured this session with
    --preset passive-lowpower (steady peak 80.11 C, R_ja 4.5929 K/W; after the composite update)."""

    def test_passive_preset_prints_steady_rja_and_warning(self, tmp_path, capsys):
        rc = main_mod.main(["--preset", "passive-lowpower", "--no-browser",
                            "--output-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "R_ja (steady):" in out
        assert "4.5929 K/W" in out
        assert "WARNING: the 100-step transient" in out
