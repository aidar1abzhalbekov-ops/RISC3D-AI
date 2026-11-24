#!/usr/bin/env python3
"""
main.py
RISC3D-AI :: End-to-End 3D IC Thermal Simulation Pipeline (Interactive CLI)

Entry point that:
  1. Prompts the user directly in the terminal for all simulation parameters
     (grid size, spacing, ambient temp, convective coefficient, die power,
     step count, safety factor, and whether to run Stage 2). NO
     configs/config.json dependency for the default interactive workflow.
     Press Enter on any prompt to accept the bracketed default.
  2. Configures a multi-layer TSMC-style advanced-packaging chiplet stack
     (Interposer -> Microbump+Underfill composite -> Silicon Die with an
     active compute-block power density Q) via src.grid_builder.GridBuilder.
  3. Constructs src.thermal_solver.ThermalSolver, computes the automated
     CFL-stable timestep, and runs the explicit solver for exactly the
     requested number of steps.
  4. Renders the result as a single interactive WebGL 3D HTML file
     (orbit/zoom/pan, Z-layer slicing, click-to-inspect voxel temperatures)
     via src.visualization.ThermalVisualizer.render_interactive_webgl(),
     and opens it automatically in the default browser.
  5. Optionally (--optimize or the Stage 2 prompt) runs the chiplet placement
     optimizer (src.optimizer) on the SAME GridSection and stack, writes
     outputs/optimized_thermal_report.html, and opens it in the browser.
  6. Reports wall-clock timing metrics for each pipeline phase.

Exit codes:
  0  success
  1  Stage 1 (thermal simulation) or configuration failure
  2  configuration failure caught BEFORE Stage 1 runs (e.g. a Stage 2 catalog
     block does not fit the die; Stage 1 is never started)
  3  Stage 2 runtime failure, after Stage 1 has already completed; Stage 1
     outputs remain valid

IMPORTANT INTERFACE NOTE (fix for a prior bug report):
This file never reads SolverSection.method or SolverSection.dt -- those are
optional fields on the PipelineConfig dataclass used elsewhere in this
codebase (e.g. for Crank-Nicolson / steady-state runs driven from a config
file), but a stale/older local copy of config_loader.py may not define them.
To make main.py robust regardless of which config_loader.py version is on
disk, this file drives solver execution entirely from plain local variables
(dt, total_steps, safety_factor) captured directly from user input, and
only ever reads SolverSection.safety_factor -- a field that has existed on
every version of this dataclass -- via getattr() with a hardcoded fallback
as an extra safety margin.

Usage:
    python main.py                     # interactive prompts (default)
    python main.py --config foo.json   # load parameters from a JSON file instead
    python main.py --nx 80 --ny 80 --nz 50 --die-power-w 20   # non-interactive, explicit flags
    python main.py --optimize --optimize-iterations 200       # also run Stage 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import Optional

# Strict, package-qualified imports only -- main.py is always run from the
# repository root (`python main.py`), so `src` must resolve as a package.
from src.config_loader import (
    PipelineConfig, ConfigError, load_config,
    GridSection, BoundaryConditionsSection, StackSection, PowerSection,
    SolverSection, VisualizationSection, stack_layer_bounds,
    COOLING_PRESETS, apply_cooling_preset, cooling_metrics,
)
from src.grid_builder import GridBuilder, GridBuilderError
from src.thermal_solver import ThermalSolver, ThermalSolverError
from src.visualization import ThermalVisualizer, VisualizationError
from src.optimizer import (
    ChipletPlacementOptimizer, OptimizerError, PlacementFitError,
    render_optimization_plotly_report,
)


class PipelineError(Exception):
    """Raised on any unrecoverable failure in the end-to-end pipeline."""


@dataclass
class RunOptions:
    """Everything the user chose that is not part of PipelineConfig."""
    total_steps: int = 100
    optimize: bool = False
    optimize_iterations: int = 2000


@dataclass
class TimingReport:
    grid_build_s: float = 0.0
    solver_init_s: float = 0.0
    cfl_check_s: float = 0.0
    solve_s: float = 0.0
    visualization_s: float = 0.0
    optimize_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.grid_build_s + self.solver_init_s + self.cfl_check_s
                + self.solve_s + self.visualization_s + self.optimize_s)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("RISC3D-AI PIPELINE TIMING SUMMARY")
        print("=" * 60)
        rows = [
            ("Grid construction (GridBuilder)", self.grid_build_s),
            ("Solver initialization", self.solver_init_s),
            ("CFL stability check", self.cfl_check_s),
            ("Transient solve (explicit, N steps)", self.solve_s),
            ("WebGL 3D export + browser launch", self.visualization_s),
            ("Stage 2 optimizer + Plotly report", self.optimize_s),
        ]
        for label, seconds in rows:
            print(f"  {label:<38s} {seconds:>10.4f} s")
        print("-" * 60)
        print(f"  {'TOTAL WALL-CLOCK TIME':<38s} {self.total_s:>10.4f} s")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Interactive terminal prompts
# ---------------------------------------------------------------------------

def prompt_float(label: str, default: float) -> float:
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"    Not a valid number -- using default {default}")
        return default


def prompt_int(label: str, default: int) -> int:
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"    Not a valid integer -- using default {default}")
        return default


def prompt_yes_no(label: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"  {label} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def run_interactive_prompts():
    """
    Prompts for every parameter listed in the interactive-CLI requirement,
    in micrometers/Celsius/Watts as the user naturally thinks in them, then
    converts into the SI-unit PipelineConfig this codebase uses internally.
    Everything not explicitly prompted for (stack layer thicknesses, die
    footprint fraction, output directory, hotspot threshold) uses a fixed,
    documented default -- kept out of the prompt list per the requested
    parameter set, but easy to expose later if needed.
    """
    print("=" * 60)
    print("RISC3D-AI :: Interactive Simulation Setup")
    print("Press Enter to accept the bracketed default for any parameter.")
    print("=" * 60)

    print("\nGrid dimensions (voxel counts):")
    nx = prompt_int("nx", 40)
    ny = prompt_int("ny", 40)
    nz = prompt_int("nz", 24)

    print("\nVoxel spacing (micrometers):")
    dx_um = prompt_float("dx (um)", 500.0)
    dy_um = prompt_float("dy (um)", 500.0)
    dz_um = prompt_float("dz (um)", 10.0)

    print("\nThermal boundary conditions:")
    t_ambient_c = prompt_float("Ambient temperature (C)", 25.0)
    h_convective = prompt_float("Heat transfer coefficient h_convective (W/m^2K)", 1000.0)

    print("\nActive power source:")
    die_power_w = prompt_float("Active die total power (W)", 12.0)

    print("\nSolve parameters:")
    total_steps = prompt_int("Total simulation steps", 100)
    safety_factor = prompt_float("Safety factor (0-1]", 0.8)

    print("\nStage 2 (chiplet placement optimizer):")
    run_optimizer = prompt_yes_no("Run Stage 2 automated chiplet placement optimizer?", False)
    optimize_iterations = 2000
    if run_optimizer:
        optimize_iterations = prompt_int("Optimizer iterations", 2000)

    print()

    grid = GridSection(nx=nx, ny=ny, nz=nz, dx=dx_um * 1e-6, dy=dy_um * 1e-6, dz=dz_um * 1e-6)
    bc = BoundaryConditionsSection(t_ambient_c=t_ambient_c, h_convective=h_convective, bottom_bc="robin")
    stack = StackSection()      # fixed defaults: interposer_um=100, bump_um=40 (die = remainder)
    power = PowerSection(die_power_w=die_power_w)  # fixed default die_footprint_frac=0.5
    solver = SolverSection(safety_factor=safety_factor)  # total_steps handled locally, not stored here
    viz = VisualizationSection()  # output_dir="outputs", hotspot_threshold_c=85.0

    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()

    # total_steps is intentionally NOT a PipelineConfig/SolverSection field
    # (avoids ever depending on a field name that might not exist on a
    # stale local config_loader.py) -- it is threaded through main()
    # separately inside RunOptions.
    options = RunOptions(total_steps=total_steps, optimize=run_optimizer,
                         optimize_iterations=optimize_iterations)
    return config, options


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RISC3D-AI end-to-end 3D IC thermal simulation pipeline. "
                     "Running with NO arguments launches interactive terminal prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Load parameters from a JSON config file instead of prompting "
                              "or using flag defaults below.")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--dx-um", type=float, default=500.0, help="Voxel size along X (micrometers).")
    parser.add_argument("--dy-um", type=float, default=500.0, help="Voxel size along Y (micrometers).")
    parser.add_argument("--dz-um", type=float, default=10.0, help="Voxel size along Z (micrometers).")
    parser.add_argument("--t-ambient-c", type=float, default=25.0, help="Ambient temperature (degC).")
    parser.add_argument("--h-convective", type=float, default=1000.0, help="Convective coefficient (W/m^2K).")
    parser.add_argument("--die-power-w", type=float, default=12.0, help="Active die total power (W).")
    parser.add_argument("--total-steps", type=int, default=100, help="Total explicit simulation steps.")
    parser.add_argument("--safety-factor", type=float, default=0.8, help="CFL safety factor in (0, 1].")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--no-browser", action="store_true",
                         help="Do not automatically open the generated HTML in a browser.")
    parser.add_argument("--optimize", action="store_true",
                         help="Also run the Stage 2 chiplet placement optimizer after Stage 1.")
    parser.add_argument("--optimize-iterations", type=int, default=2000,
                         help="Simulated-annealing iterations for Stage 2 (used with --optimize).")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(COOLING_PRESETS),
                         help="Named cooling preset: sets die power and h_convective together "
                              "(overrides --die-power-w and --h-convective).")
    return parser


def build_config_from_flags(args: argparse.Namespace):
    grid = GridSection(nx=args.nx, ny=args.ny, nz=args.nz,
                        dx=args.dx_um * 1e-6, dy=args.dy_um * 1e-6, dz=args.dz_um * 1e-6)
    bc = BoundaryConditionsSection(t_ambient_c=args.t_ambient_c, h_convective=args.h_convective, bottom_bc="robin")
    stack = StackSection()
    power = PowerSection(die_power_w=args.die_power_w)
    solver = SolverSection(safety_factor=args.safety_factor)
    viz = VisualizationSection(output_dir=args.output_dir)
    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()
    options = RunOptions(total_steps=args.total_steps, optimize=args.optimize,
                         optimize_iterations=args.optimize_iterations)
    return config, options


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def steady_peak_temperature_c(config: PipelineConfig) -> float:
    """Peak steady-state temperature (C) of the Stage 1 stack, from a direct solve."""
    gb = build_chiplet_stack(config)
    solver = ThermalSolver.from_grid_builder(gb)
    T = solver.solve_steady_state(apply_result=False)
    return float(T.max()) - 273.15


def build_chiplet_stack(config: PipelineConfig) -> GridBuilder:
    """
    Constructs a 3-region CoWoS-style vertical stack:
      z-layer 0 .. z_interposer_top      : passive Si interposer (bottom)
      z_interposer_top .. z_bump_top     : microbump + underfill composite (anisotropic)
      z_bump_top .. nz                   : Si compute die (top), with an active
                                            power-density footprint in its center
    Bottom face (z=0, interposer underside) is the heat-sink boundary
    (Robin, per this file's fixed bottom_bc="robin"); top and all sides
    are adiabatic.
    """
    try:
        gb = GridBuilder(config=config.to_dict())
    except GridBuilderError as exc:
        raise PipelineError(f"GridBuilder construction failed: {exc}") from exc

    nz = config.grid.nz
    nx, ny = config.grid.nx, config.grid.ny
    z_interposer_end, z_bump_end = stack_layer_bounds(config)

    try:
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
    except GridBuilderError as exc:
        raise PipelineError(f"Chiplet stack assembly failed: {exc}") from exc

    print(
        f"[main] Stack assembled: interposer z[0:{z_interposer_end}], "
        f"bump+underfill z[{z_interposer_end}:{z_bump_end}], "
        f"Si die z[{z_bump_end}:{nz}] with {config.power.die_power_w:.2f}W active source "
        f"over footprint x[{margin_x}:{nx - margin_x}] y[{margin_y}:{ny - margin_y}]"
    )
    return gb


def run_pipeline(config: PipelineConfig, total_steps: int, open_browser: bool) -> dict:
    timing = TimingReport()

    # --- Phase 1: Grid construction ---
    t0 = time.perf_counter()
    gb = build_chiplet_stack(config)
    timing.grid_build_s = time.perf_counter() - t0

    # --- Phase 2: Solver initialization ---
    t0 = time.perf_counter()
    try:
        solver = ThermalSolver.from_grid_builder(gb)
    except ThermalSolverError as exc:
        raise PipelineError(f"ThermalSolver initialization failed: {exc}") from exc
    timing.solver_init_s = time.perf_counter() - t0

    # --- Phase 3: Automated CFL stability check ---
    # NOTE: reads ONLY SolverSection.safety_factor (via getattr with a
    # hardcoded fallback for extra defensiveness against a stale/older
    # local config_loader.py). Never reads .method or .dt.
    safety_factor = getattr(config.solver, "safety_factor", 0.8)
    t0 = time.perf_counter()
    try:
        dt_stable = solver.compute_max_dt(safety_factor=safety_factor)
    except ThermalSolverError as exc:
        raise PipelineError(f"CFL stability check failed: {exc}") from exc
    timing.cfl_check_s = time.perf_counter() - t0

    total_time_s = dt_stable * total_steps
    print(
        f"[main] Automated CFL check: dt_stable={dt_stable:.6e}s "
        f"(safety_factor={safety_factor}) -> running exactly {total_steps} explicit steps "
        f"(total_time_s={total_time_s:.6e}s)"
    )

    # --- Phase 4: Transient solve for exactly `total_steps` explicit steps ---
    t0 = time.perf_counter()
    try:
        result = solver.run(
            total_time_s=total_time_s,
            dt=dt_stable,
            safety_factor=safety_factor,
            record_interval_s=None,
            max_steps=total_steps + 10,
        )
    except ThermalSolverError as exc:
        raise PipelineError(f"Transient solve failed: {exc}") from exc
    timing.solve_s = time.perf_counter() - t0

    print(f"[main] Solve complete: {result['steps_taken']} steps, final_time={result['final_time_s']:.6e}s")

    # --- Phase 5: Interactive WebGL 3D export (replaces static PNG output) ---
    t0 = time.perf_counter()
    try:
        viz = ThermalVisualizer(
            temperature_k=result["final_T"],
            dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
            output_dir=config.visualization.output_dir,
        )
        html_path = viz.render_interactive_webgl(filename="thermal_3d.html")
        hotspot_report = viz.get_hotspot_report(threshold_c=config.visualization.hotspot_threshold_c)

        if open_browser:
            abs_path = os.path.abspath(html_path)
            try:
                webbrowser.open(f"file://{abs_path}")
                print(f"[main] Opened {abs_path} in the default browser.")
            except Exception as browser_exc:  # noqa: BLE001
                # A missing/unavailable browser in this environment must not
                # fail the simulation -- the HTML file itself is still the
                # valid, complete deliverable; only the auto-launch failed.
                print(f"[main] Could not auto-launch a browser ({browser_exc}). "
                      f"Open manually: {abs_path}")
        else:
            print(f"[main] Browser launch skipped (--no-browser). Open manually: {html_path}")

        report = {
            "html_path": html_path,
            "hotspot_count": hotspot_report.count,
            "peak_temperature_c": hotspot_report.peak_temperature_c,
            "peak_location_um": hotspot_report.peak_location_um,
            "peak_location_ijk": hotspot_report.peak_location_ijk,
        }
    except VisualizationError as exc:
        raise PipelineError(f"WebGL visualization export failed: {exc}") from exc
    timing.visualization_s = time.perf_counter() - t0

    return {"timing": timing, "solve_result": result, "viz_report": report}


def run_stage2(optimizer: ChipletPlacementOptimizer, config: PipelineConfig,
               iterations: int, open_browser: bool) -> dict:
    """Stage 2: runs the placement search on the same GridSection and stack as
    Stage 1, writes the Plotly report under the configured output directory,
    and opens it in the browser (failure to open never fails the run)."""
    t0 = time.perf_counter()
    result = optimizer.optimize(iterations=iterations)

    report_path = os.path.join(config.visualization.output_dir, "optimized_thermal_report.html")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    html_path = render_optimization_plotly_report(result, config, output_path=report_path)

    if open_browser:
        abs_path = os.path.abspath(html_path)
        try:
            webbrowser.open(f"file://{abs_path}")
            print(f"[main] Opened {abs_path} in the default browser.")
        except Exception as browser_exc:  # noqa: BLE001
            print(f"[main] Could not auto-launch a browser ({browser_exc}). "
                  f"Open manually: {abs_path}")
    else:
        print(f"[main] Browser launch skipped (--no-browser). Open manually: {html_path}")

    return {"result": result, "html_path": html_path, "elapsed_s": time.perf_counter() - t0}


def main(argv=None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        if args.config is not None:
            config = load_config(args.config)
            options = RunOptions(total_steps=args.total_steps, optimize=args.optimize,
                                 optimize_iterations=args.optimize_iterations)
        elif argv is None and len(sys.argv) == 1:
            # `python main.py` with truly no arguments -> interactive prompts.
            config, options = run_interactive_prompts()
        else:
            # Any explicit flag supplied -> non-interactive, flag-driven run.
            config, options = build_config_from_flags(args)
    except ConfigError as exc:
        print(f"\n[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n[main] Input cancelled. Exiting.")
        return 1

    if args.preset is not None:
        try:
            config = apply_cooling_preset(config, args.preset)
        except ConfigError as exc:
            print(f"\n[FATAL] Configuration error: {exc}", file=sys.stderr)
            return 1
        print(f"[main] Cooling preset '{args.preset}': "
              f"{COOLING_PRESETS[args.preset]['description']}")

    # --- Configuration check for Stage 2, BEFORE Stage 1 starts. ---
    # Constructing the optimizer runs the catalog-fit check (PlacementFitError)
    # and builds the fixed stack. No thermal solve happens here.
    optimizer: Optional[ChipletPlacementOptimizer] = None
    if options.optimize:
        try:
            optimizer = ChipletPlacementOptimizer(config)
        except (PlacementFitError, OptimizerError, ConfigError,
                GridBuilderError, ThermalSolverError) as exc:
            print(f"\n[FATAL] Stage 2 configuration error (Stage 1 not started): {exc}", file=sys.stderr)
            return 2

    print("=" * 60)
    print("RISC3D-AI :: 3D IC Thermal Simulation Pipeline")
    print("=" * 60)
    print(f"Grid: {config.grid.nx}x{config.grid.ny}x{config.grid.nz} voxels "
          f"({config.grid.dx*1e6:.1f}x{config.grid.dy*1e6:.1f}x{config.grid.dz*1e6:.1f} um)")
    print(f"Ambient: {config.boundary_conditions.t_ambient_c:.1f}C | "
          f"h_convective: {config.boundary_conditions.h_convective:.1f} W/m^2K | "
          f"Die power: {config.power.die_power_w:.2f}W | Steps: {options.total_steps} | "
          f"Stage 2: {'on' if options.optimize else 'off'}")
    print("=" * 60)

    open_browser = not args.no_browser

    # --- Stage 1: thermal simulation ---
    try:
        outcome = run_pipeline(config, options.total_steps, open_browser=open_browser)
    except PipelineError as exc:
        print(f"\n[FATAL] Pipeline aborted: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net, always exit cleanly
        print(f"\n[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 1

    report = outcome["viz_report"]
    print("\n" + "=" * 60)
    print("THERMAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Peak temperature:      {report['peak_temperature_c']:.2f} °C")
    # R_ja is a steady-state quantity. The explicit run above covers only
    # total_steps steps and is not generally at steady state, so R_ja and the
    # steady peak come from a direct steady solve of the same stack.
    steady_peak_c = steady_peak_temperature_c(config)
    metrics = cooling_metrics(config, steady_peak_c)
    transient_rise = report['peak_temperature_c'] - config.boundary_conditions.t_ambient_c
    steady_rise = steady_peak_c - config.boundary_conditions.t_ambient_c
    print(f"  Steady-state peak:     {steady_peak_c:.2f} °C")
    if steady_rise > 0 and transient_rise < 0.5 * steady_rise:
        print(f"  [main] WARNING: the {options.total_steps}-step transient reached only "
              f"{transient_rise:.2f} K of the {steady_rise:.2f} K steady rise; the transient peak "
              f"is not a steady-state result.")
    print(f"  Die power:             {metrics['power_w']:.2f} W over "
          f"{metrics['footprint_cm2']:.3f} cm^2 = {metrics['w_per_cm2']:.2f} W/cm^2")
    if metrics['r_ja_k_per_w'] is not None:
        print(f"  R_ja (steady):         {metrics['r_ja_k_per_w']:.4f} K/W "
              f"(h = {metrics['h_convective']:.0f} W/m^2K)")
    print(f"  Peak location (i,j,k): {report['peak_location_ijk']}")
    print(f"  Peak location (um):    "
          f"({report['peak_location_um'][0]:.1f}, {report['peak_location_um'][1]:.1f}, "
          f"{report['peak_location_um'][2]:.1f})")
    print(f"  Voxels > {config.visualization.hotspot_threshold_c:.1f}°C:       {report['hotspot_count']}")
    print(f"  Interactive 3D view:   {report['html_path']}")
    print("=" * 60)

    # --- Stage 2: chiplet placement optimizer (only after Stage 1 succeeded) ---
    if optimizer is not None:
        try:
            stage2 = run_stage2(optimizer, config, options.optimize_iterations, open_browser=open_browser)
        except Exception as exc:  # noqa: BLE001 - Stage 1 outputs above remain valid
            print(f"\n[FATAL] Stage 2 optimizer failed (Stage 1 results above are valid): {exc}",
                  file=sys.stderr)
            outcome["timing"].print_summary()
            return 3

        result = stage2["result"]
        print("\n" + "=" * 60)
        print("CHIPLET PLACEMENT OPTIMIZATION RESULTS (Stage 2)")
        print("=" * 60)
        print(f"  Best peak temperature: {result.best_peak_temperature_c:.2f} °C")
        print(f"  Iterations:            {result.iterations}  "
              f"(accepted {result.accepted_moves}, overlap candidates {result.overlap_candidates})")
        print(f"  Report:                {stage2['html_path']}")
        for block in result.placements_um():
            print(f"    {block['name']:<16s} origin voxel ({block['x_voxel']}, {block['y_voxel']})  "
                  f"{block['width_um']:.0f} x {block['depth_um']:.0f} um  {block['power_w']:.1f} W")
        print("=" * 60)
        outcome["timing"].optimize_s = stage2["elapsed_s"]

    outcome["timing"].print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
