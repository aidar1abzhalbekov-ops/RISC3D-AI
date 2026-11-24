"""
config_loader.py
RISC3D-AI :: Centralized Pipeline Configuration

Single source of truth for pipeline configuration. Loads configs/config.json
(or a caller-specified path) into a validated PipelineConfig, so GridBuilder's
grid/BC dict, ThermalSolver's SolverBCConfig, and the chiplet-stack/solver/
visualization parameters main.py needs are all derived from ONE consistent
schema -- instead of being redefined ad hoc as argparse defaults scattered
across main.py.

Design goals:
  - `python main.py` with zero CLI flags must run end-to-end using
    configs/config.json defaults.
  - Missing file / malformed JSON / missing keys must never crash the
    pipeline silently -- fall back to hardcoded safe defaults per-section
    with a clear warning, exactly like GridBuilder._load_config() already
    does for its own grid/BC section.
  - main.py's optional CLI flags override individual fields of the loaded
    config without needing to duplicate the whole schema.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.ConfigLoader")

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.json")


class ConfigError(Exception):
    """Raised on unrecoverable configuration errors (invalid types/ranges
    that cannot be safely defaulted)."""


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GridSection:
    # Round 3 scale: 20 x 20 mm package, 240 um stack (see docs/NEXT_TASK_stage2_wiring.md).
    nx: int = 40
    ny: int = 40
    nz: int = 24
    dx: float = 500e-6
    dy: float = 500e-6
    dz: float = 10e-6


@dataclass
class BoundaryConditionsSection:
    t_ambient_c: float = 25.0
    h_convective: float = 2500.0
    bottom_bc: str = "robin"      # "robin" | "dirichlet"
    top_bc: str = "neumann"       # only "neumann" currently supported
    side_bc: str = "neumann"      # only "neumann" currently supported


@dataclass
class StackSection:
    """Vertical chiplet stack, authored as physical thicknesses in micrometres
    (bottom to top): interposer_um (thinned Si), bump_um (microbump + underfill
    composite), then the remainder of nz as the Si die. Each thickness must be a
    whole number of voxels at grid.dz; see stack_layer_bounds()."""
    interposer_um: float = 100.0
    bump_um: float = 40.0


@dataclass
class PowerSection:
    die_power_w: float = 12.0
    die_footprint_frac: float = 0.5   # fraction of XY extent, centered


@dataclass
class SolverSection:
    method: str = "explicit"              # "explicit" | "crank_nicolson"
    steady_state: bool = False            # if True, solve directly for equilibrium (skips time-stepping)
    total_time_s: Optional[float] = None  # None -> auto = 400 * CFL dt (explicit) or total_time_s/200 (CN)
    dt: Optional[float] = None            # None -> auto-selected per method
    safety_factor: float = 0.9
    record_steps: int = 50
    max_steps: int = 2_000_000


@dataclass
class VisualizationSection:
    output_dir: str = "outputs"
    hotspot_threshold_c: float = 85.0
    use_pyvista: bool = True


@dataclass
class PipelineConfig:
    grid: GridSection
    boundary_conditions: BoundaryConditionsSection
    stack: StackSection
    power: PowerSection
    solver: SolverSection
    visualization: VisualizationSection

    # ------------------------------------------------------------------
    def validate(self) -> None:
        g = self.grid
        if min(g.nx, g.ny, g.nz) <= 0:
            raise ConfigError(f"grid.nx/ny/nz must be positive, got ({g.nx}, {g.ny}, {g.nz})")
        if min(g.dx, g.dy, g.dz) <= 0:
            raise ConfigError(f"grid.dx/dy/dz must be positive, got ({g.dx}, {g.dy}, {g.dz})")

        bc = self.boundary_conditions
        if bc.bottom_bc not in ("robin", "dirichlet"):
            raise ConfigError(f"boundary_conditions.bottom_bc must be 'robin' or 'dirichlet', got '{bc.bottom_bc}'")
        if bc.top_bc != "neumann" or bc.side_bc != "neumann":
            raise ConfigError("boundary_conditions.top_bc/side_bc: only 'neumann' is currently supported.")
        if bc.bottom_bc == "robin" and bc.h_convective <= 0:
            raise ConfigError(f"boundary_conditions.h_convective must be > 0 for Robin BC, got {bc.h_convective}")

        s = self.stack
        if s.interposer_um <= 0 or s.bump_um <= 0:
            raise ConfigError(f"stack.interposer_um and stack.bump_um must be positive, "
                              f"got ({s.interposer_um}, {s.bump_um}).")
        stack_layer_bounds(self)  # raises ConfigError on non-integer voxel counts or no die layer

        p = self.power
        if p.die_power_w < 0:
            raise ConfigError(f"power.die_power_w must be >= 0, got {p.die_power_w}")
        if not (0.0 < p.die_footprint_frac <= 1.0):
            raise ConfigError(f"power.die_footprint_frac must be in (0, 1], got {p.die_footprint_frac}")

        sv = self.solver
        if sv.method not in ("explicit", "crank_nicolson"):
            raise ConfigError(f"solver.method must be 'explicit' or 'crank_nicolson', got '{sv.method}'")
        if sv.total_time_s is not None and sv.total_time_s <= 0:
            raise ConfigError(f"solver.total_time_s must be positive if set, got {sv.total_time_s}")
        if sv.dt is not None and sv.dt <= 0:
            raise ConfigError(f"solver.dt must be positive if set, got {sv.dt}")
        if not (0.0 < sv.safety_factor <= 1.0):
            raise ConfigError(f"solver.safety_factor must be in (0, 1], got {sv.safety_factor}")
        if sv.record_steps < 1:
            raise ConfigError(f"solver.record_steps must be >= 1, got {sv.record_steps}")
        if sv.max_steps < 1:
            raise ConfigError(f"solver.max_steps must be >= 1, got {sv.max_steps}")

        v = self.visualization
        if v.hotspot_threshold_c is None:
            raise ConfigError("visualization.hotspot_threshold_c must be set.")

    # ------------------------------------------------------------------
    def to_grid_builder_dict(self) -> Dict[str, Any]:
        """Converts to the nested dict schema GridBuilder(config=...) expects."""
        return {
            "grid": asdict(self.grid),
            "boundary_conditions": {
                "t_ambient_k": self.boundary_conditions.t_ambient_c + 273.15,
                "h_convective": self.boundary_conditions.h_convective,
                "bottom_bc": self.boundary_conditions.bottom_bc,
                "top_bc": self.boundary_conditions.top_bc,
                "side_bc": self.boundary_conditions.side_bc,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Alias for to_grid_builder_dict(). GridBuilder.__init__() only accepts
        a raw dict, a JSON file path, or None -- it does NOT accept a
        PipelineConfig instance directly. Callers (including tests) that
        hold a PipelineConfig must construct GridBuilder via:

            gb = GridBuilder(config=pipeline_config.to_dict())

        rather than `GridBuilder(config=pipeline_config)`, which raises
        TypeError. This alias exists purely for that call-site clarity.
        """
        return self.to_grid_builder_dict()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grid": asdict(self.grid),
            "boundary_conditions": asdict(self.boundary_conditions),
            "stack": asdict(self.stack),
            "power": asdict(self.power),
            "solver": asdict(self.solver),
            "visualization": asdict(self.visualization),
        }


def stack_layer_bounds(config: PipelineConfig) -> Tuple[int, int]:
    """Single source of the vertical layer split, used by main.py, the tests and
    the optimizer. Returns (z_interposer_end, z_bump_end) as voxel indices:
    interposer = [0, z_interposer_end), composite = [z_interposer_end, z_bump_end),
    Si die = [z_bump_end, nz).

    Each authored thickness must be a whole number of voxels at grid.dz. Rounding
    to the nearest voxel silently changes the layer thickness (e.g. 40 um at 10 um
    voxels is exact, but 40 um at 16 um voxels would be 2.5 voxels), so a
    non-integer count is rejected instead of rounded.
    """
    g = config.grid
    s = config.stack
    dz_um = g.dz * 1e6

    def _voxels(name: str, thickness_um: float) -> int:
        n = thickness_um / dz_um
        k = int(round(n))
        if k < 1 or abs(n - k) > 1e-6 * max(1.0, n):
            raise ConfigError(
                f"stack.{name}={thickness_um} um is not a whole number of voxels at "
                f"dz={dz_um:.4g} um (gives {n:.4f}). Choose a multiple of dz."
            )
        return k

    z_interposer_end = _voxels("interposer_um", s.interposer_um)
    z_bump_end = z_interposer_end + _voxels("bump_um", s.bump_um)
    if z_bump_end >= g.nz:
        raise ConfigError(
            f"Stack needs {z_bump_end} voxels below the die but grid.nz={g.nz} leaves no Si die layer."
        )
    return z_interposer_end, z_bump_end


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _build_section(dataclass_type, raw: Dict[str, Any], section_name: str):
    """Builds a section dataclass from a raw dict, ignoring unknown keys
    (with a warning) and falling back to the dataclass default for any
    missing key -- never raises on partial/incomplete sections."""
    valid_keys = {f.name for f in fields(dataclass_type)}
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        logger.warning("Ignoring unknown key(s) in config section '%s': %s", section_name, sorted(unknown))
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    try:
        return dataclass_type(**filtered)
    except TypeError as exc:
        raise ConfigError(f"Invalid value type in config section '{section_name}': {exc}") from exc


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """
    Loads and validates the full pipeline configuration.

    Resolution order:
      1. `path` if explicitly given.
      2. DEFAULT_CONFIG_PATH ("configs/config.json") if it exists.
      3. Hardcoded dataclass defaults for every section, with a warning,
         if neither of the above is available or the file is malformed.

    This mirrors GridBuilder._load_config()'s fallback behavior so the
    pipeline never hard-crashes on a missing/partial config file.
    """
    resolved_path = path or DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}

    if path is not None and not os.path.exists(path):
        raise ConfigError(f"Explicitly specified config path does not exist: '{path}'")

    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r") as f:
                raw = json.load(f)
            logger.info("Loaded pipeline configuration from %s", resolved_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to parse config at '%s' (%s); falling back to hardcoded defaults.",
                         resolved_path, exc)
            raw = {}
    else:
        logger.warning(
            "Config file '%s' not found; running with hardcoded pipeline defaults. "
            "Create configs/config.json (see config_loader.write_default_config()) to customize.",
            resolved_path,
        )

    if not isinstance(raw, dict):
        logger.error("Top-level config content is not a JSON object; falling back to hardcoded defaults.")
        raw = {}

    config = PipelineConfig(
        grid=_build_section(GridSection, raw.get("grid", {}), "grid"),
        boundary_conditions=_build_section(
            BoundaryConditionsSection, raw.get("boundary_conditions", {}), "boundary_conditions"
        ),
        stack=_build_section(StackSection, raw.get("stack", {}), "stack"),
        power=_build_section(PowerSection, raw.get("power", {}), "power"),
        solver=_build_section(SolverSection, raw.get("solver", {}), "solver"),
        visualization=_build_section(VisualizationSection, raw.get("visualization", {}), "visualization"),
    )
    config.validate()
    return config


def write_default_config(path: str = DEFAULT_CONFIG_PATH) -> str:
    """Writes the hardcoded default PipelineConfig to `path` as pretty JSON.
    Useful for bootstrapping a fresh configs/config.json."""
    default_config = PipelineConfig(
        grid=GridSection(), boundary_conditions=BoundaryConditionsSection(),
        stack=StackSection(), power=PowerSection(),
        solver=SolverSection(), visualization=VisualizationSection(),
    )
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(default_config.as_dict(), f, indent=2)
        logger.info("Wrote default configuration to %s", path)
        return path
    except OSError as exc:
        raise ConfigError(f"Failed to write default config to '{path}': {exc}") from exc


if __name__ == "__main__":
    # Smoke test: write and reload the default config.
    written_path = write_default_config(os.path.join("configs_smoke_test", "config.json"))
    reloaded = load_config(written_path)
    print("Reloaded config:", reloaded.as_dict())


# ---------------------------------------------------------------------------
# Named cooling presets (validation plan step 2)
# ---------------------------------------------------------------------------

# Each preset fixes one matched pair of power and convective coefficient.
# Raising power without raising h gives physically meaningless temperatures
# (see claude/VALIDATION_PLAN.md, Part 1), so the two are always set together.
COOLING_PRESETS: Dict[str, Dict[str, Any]] = {
    "passive-lowpower": {
        "die_power_w": 12.0,
        "h_convective": 1000.0,
        "description": "12 W die, passive heatsink in still air",
    },
    "cowos-coldplate": {
        "die_power_w": 150.0,
        "h_convective": 25000.0,
        "description": "150 W die, cold plate",
    },
}


def apply_cooling_preset(config: PipelineConfig, name: str) -> PipelineConfig:
    """Sets die power and h_convective to a named preset and re-validates.

    Raises ConfigError for an unknown preset name. Other fields are untouched.
    """
    if name not in COOLING_PRESETS:
        raise ConfigError(
            f"unknown cooling preset '{name}'; choose one of: {', '.join(sorted(COOLING_PRESETS))}"
        )
    preset = COOLING_PRESETS[name]
    config.power.die_power_w = float(preset["die_power_w"])
    config.boundary_conditions.h_convective = float(preset["h_convective"])
    config.validate()
    return config


def die_footprint_m2(config: PipelineConfig) -> float:
    """Active die footprint in m^2.

    ASSUMPTION: the die occupies die_footprint_frac of each lateral extent,
    centred, so its area is (frac * Lx) * (frac * Ly).
    """
    frac = config.power.die_footprint_frac
    g = config.grid
    return (frac * g.nx * g.dx) * (frac * g.ny * g.dy)


def cooling_metrics(config: PipelineConfig, peak_c: float) -> Dict[str, Any]:
    """Header metrics for a result: power density over the die footprint and the
    measured junction-to-ambient resistance R_ja = (T_peak - T_ambient) / P.

    Returns None for R_ja when the die power is zero (undefined).
    """
    power = float(config.power.die_power_w)
    area_cm2 = die_footprint_m2(config) * 1e4
    w_per_cm2 = power / area_cm2 if area_cm2 > 0 else float("nan")
    if power > 0:
        r_ja = (float(peak_c) - float(config.boundary_conditions.t_ambient_c)) / power
    else:
        r_ja = None
    return {
        "power_w": power,
        "footprint_cm2": area_cm2,
        "w_per_cm2": w_per_cm2,
        "r_ja_k_per_w": r_ja,
        "h_convective": float(config.boundary_conditions.h_convective),
    }
