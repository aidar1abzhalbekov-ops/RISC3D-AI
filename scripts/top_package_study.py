"""
scripts/top_package_study.py
Bottom-sink baseline vs top-side package (TIM1 / lid / TIM2, heatsink on the lid).

Same lateral grid (40 x 40, dx = dy = 500 um), same die footprint and power, dz = 20 um.
  baseline : interposer 100 / bump 40 / die 100 um, nz = 12, sink on the interposer underside
  top      : + TIM1 40 um / lid 1000 um Cu / TIM2 40 um, nz = 66, sink on the lid
The die is 100 um in both. Only the package above the die and the sink side differ.

Run from the repo root:  python3 scripts/top_package_study.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import (  # noqa: E402
    BoundaryConditionsSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection, apply_cooling_preset,
    cooling_metrics, stack_extents,
)
from main import steady_peak_temperature_c  # noqa: E402


def make_config(nz: int, tim1: float, lid: float, tim2: float) -> PipelineConfig:
    return PipelineConfig(
        grid=GridSection(nx=40, ny=40, nz=nz, dx=500e-6, dy=500e-6, dz=20e-6),
        boundary_conditions=BoundaryConditionsSection(t_ambient_c=25.0, h_convective=1000.0,
                                                      bottom_bc="robin"),
        stack=StackSection(interposer_um=100.0, bump_um=40.0,
                           tim1_um=tim1, lid_um=lid, tim2_um=tim2),
        power=PowerSection(die_power_w=12.0, die_footprint_frac=0.5),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


def run(label: str, cfg: PipelineConfig) -> None:
    ext = stack_extents(cfg)
    peak = steady_peak_temperature_c(cfg)
    m = cooling_metrics(cfg, peak)
    print(f"{label:<34} nz={cfg.grid.nz:<3} die={ext['die_end'] - ext['bump_end']} vox "
          f"peak={peak:.4f} C  P={m['power_w']:.1f} W  R_ja={m['r_ja_k_per_w']:.4f} K/W  "
          f"h={cfg.boundary_conditions.h_convective:.0f}", flush=True)


if __name__ == "__main__":
    presets = ("passive-lowpower", "cowos-coldplate")
    for name in presets:
        base = make_config(nz=12, tim1=0.0, lid=0.0, tim2=0.0)
        apply_cooling_preset(base, name)
        top = make_config(nz=66, tim1=40.0, lid=1000.0, tim2=40.0)
        apply_cooling_preset(top, name)
        print(f"--- preset: {name}")
        run("bottom sink, no package", base)
        run("top sink, TIM1/lid/TIM2", top)
