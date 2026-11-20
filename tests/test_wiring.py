"""
Stage 2 wiring in main.py, its exit codes, and the GridBuilder sanity checks.

Exit-code contract (see main.py docstring):
  2  Stage 2 configuration error caught before Stage 1 starts
  3  Stage 2 runtime failure after Stage 1 completed
"""

from __future__ import annotations

import logging

import pytest

import main as cli
from src.config_loader import GridSection, PipelineConfig
from src.grid_builder import GridBuilder


# ---------------------------------------------------------------------------
# GridBuilder sanity warnings
# ---------------------------------------------------------------------------

def _grid_dict(dx: float, dy: float, dz: float) -> dict:
    return {"grid": {"nx": 8, "ny": 8, "nz": 8, "dx": dx, "dy": dy, "dz": dz}}


class TestGridBuilderWarnings:
    def test_round3_scale_emits_no_range_or_aspect_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="RISC3D-AI.GridBuilder"):
            GridBuilder(config=_grid_dict(500e-6, 500e-6, 10e-6))
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("sanity range" in m for m in messages)
        assert not any("aspect ratio" in m for m in messages)

    def test_implausible_in_plane_size_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="RISC3D-AI.GridBuilder"):
            GridBuilder(config=_grid_dict(1e-9, 500e-6, 10e-6))  # 1 nm: likely a units error
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("dx=" in m and "sanity range" in m for m in messages)

    def test_extreme_aspect_ratio_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="RISC3D-AI.GridBuilder"):
            GridBuilder(config=_grid_dict(500e-6, 500e-6, 1e-6))  # 500:1
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("aspect ratio" in m for m in messages)


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestCliFlags:
    def test_optimize_flags_default_off(self):
        args = cli.build_cli_parser().parse_args([])
        assert args.optimize is False
        assert args.optimize_iterations == 2000

    def test_optimize_flags_parse(self):
        args = cli.build_cli_parser().parse_args(["--optimize", "--optimize-iterations", "50"])
        assert args.optimize is True
        assert args.optimize_iterations == 50

    def test_build_config_from_flags_carries_run_options(self):
        args = cli.build_cli_parser().parse_args(
            ["--optimize", "--optimize-iterations", "7", "--total-steps", "3"])
        config, options = cli.build_config_from_flags(args)
        assert options.optimize is True
        assert options.optimize_iterations == 7
        assert options.total_steps == 3
        assert config.grid.nx == 40


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_catalog_that_does_not_fit_exits_2_before_stage1(self, monkeypatch, tmp_path):
        # A 2 x 2 voxel die cannot hold the default 11 mm HBM block, so the
        # optimizer's fit check must fail before Stage 1 is ever called.
        def stage1_must_not_run(*args, **kwargs):
            raise AssertionError("Stage 1 ran even though Stage 2 configuration is invalid")

        monkeypatch.setattr(cli, "run_pipeline", stage1_must_not_run)
        rc = cli.main(["--nx", "2", "--ny", "2", "--nz", "24", "--optimize",
                       "--no-browser", "--output-dir", str(tmp_path)])
        assert rc == 2

    def test_stage2_runtime_failure_exits_3_after_stage1(self, monkeypatch, tmp_path):
        ran_stage1 = {"value": False}
        real_run_pipeline = cli.run_pipeline

        def tracking_run_pipeline(*args, **kwargs):
            ran_stage1["value"] = True
            return real_run_pipeline(*args, **kwargs)

        def failing_stage2(*args, **kwargs):
            raise RuntimeError("simulated Stage 2 failure")

        monkeypatch.setattr(cli, "run_pipeline", tracking_run_pipeline)
        monkeypatch.setattr(cli, "run_stage2", failing_stage2)
        rc = cli.main(["--nx", "40", "--ny", "40", "--nz", "24", "--total-steps", "3",
                       "--optimize", "--no-browser", "--output-dir", str(tmp_path)])
        assert ran_stage1["value"] is True
        assert rc == 3

    def test_stage1_only_run_exits_0(self, tmp_path):
        rc = cli.main(["--nx", "40", "--ny", "40", "--nz", "24", "--total-steps", "3",
                       "--no-browser", "--output-dir", str(tmp_path)])
        assert rc == 0

    def test_stage2_success_path_exits_0_and_reports_result(self, monkeypatch, tmp_path, capsys):
        # Stub only the Plotly writer (Plotly is not installed in the sandbox).
        # The optimizer, Stage 1 and the wiring all run for real.
        import src.optimizer as optimizer_module

        def fake_render(result, config, output_path):
            with open(output_path, "w") as fh:
                fh.write("<html>stub</html>")
            return output_path

        monkeypatch.setattr(optimizer_module, "render_optimization_plotly_report", fake_render)
        monkeypatch.setattr(cli, "render_optimization_plotly_report", fake_render)
        rc = cli.main(["--nx", "40", "--ny", "40", "--nz", "24", "--total-steps", "3",
                       "--optimize", "--optimize-iterations", "5",
                       "--no-browser", "--output-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "CHIPLET PLACEMENT OPTIMIZATION RESULTS (Stage 2)" in out
        assert (tmp_path / "optimized_thermal_report.html").exists()
