"""
tests/test_materials.py
RISC3D-AI :: Composite conductivity derived from volume fraction (validation plan step 3)

Source of the formulas and matrix values: Rao et al., TSCI 27(5B), 2023
  - Eq. (2): out-of-plane equivalent conductivity, parallel form.
  - Eqs. (10)-(11): series (out-of-plane) and parallel (in-plane) forms.
  - Table 2: Sn2.5Ag solder 57 W/mK, underfill 0.2 W/mK.

Hand values at d = 20 um, p = 40 um (assumed geometry):
  v_b = pi * 10^2 / 40^2 = 0.19635
  k_z  = 0.19635*57 + 0.80365*0.2                 = 11.3527
  k_xy (Maxwell-Garnett)                          = 0.3447
  series lower bound k_xy                         = 0.2487
  parallel upper bound k_xy                       = 11.3527
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.grid_builder import (  # noqa: E402
    BUMP_DIAMETER_UM, BUMP_PITCH_UM, K_SOLDER_TSCI, K_UNDERFILL_TSCI, MATERIALS,
    bump_volume_fraction, composite_conductivity,
)

KB = K_SOLDER_TSCI
KM = K_UNDERFILL_TSCI


class TestVolumeFraction:
    def test_hand_value_at_20_over_40(self):
        assert bump_volume_fraction(20.0, 40.0) == pytest.approx(0.19635, abs=1e-5)

    def test_rejects_diameter_larger_than_pitch(self):
        with pytest.raises(ValueError):
            bump_volume_fraction(50.0, 40.0)

    def test_rejects_non_positive_diameter(self):
        with pytest.raises(ValueError):
            bump_volume_fraction(0.0, 40.0)


class TestCompositeLimits:
    def test_all_matrix_gives_matrix(self):
        kxy, kz = composite_conductivity(KB, KM, 0.0)
        assert kxy == pytest.approx(KM, rel=1e-12)
        assert kz == pytest.approx(KM, rel=1e-12)

    def test_all_bump_gives_bump(self):
        kxy, kz = composite_conductivity(KB, KM, 1.0)
        assert kxy == pytest.approx(KB, rel=1e-12)
        assert kz == pytest.approx(KB, rel=1e-12)

    def test_rejects_fraction_out_of_range(self):
        with pytest.raises(ValueError):
            composite_conductivity(KB, KM, 1.2)

    def test_rejects_non_positive_conductivity(self):
        with pytest.raises(ValueError):
            composite_conductivity(0.0, KM, 0.2)


class TestCompositeHandValues:
    def test_shipped_composite_matches_hand_values(self):
        v = bump_volume_fraction(BUMP_DIAMETER_UM, BUMP_PITCH_UM)
        kxy, kz = composite_conductivity(KB, KM, v)
        assert kz == pytest.approx(11.3527, abs=1e-3)
        assert kxy == pytest.approx(0.3447, abs=1e-3)

    def test_registry_uses_derived_values(self):
        comp = MATERIALS["bump_underfill_composite"]
        v = bump_volume_fraction(BUMP_DIAMETER_UM, BUMP_PITCH_UM)
        kxy, kz = composite_conductivity(KB, KM, v)
        assert comp["kz"] == pytest.approx(kz, rel=1e-12)
        assert comp["kx"] == pytest.approx(kxy, rel=1e-12)
        assert comp["ky"] == pytest.approx(kxy, rel=1e-12)

    def test_tsci_matrix_values_in_registry(self):
        assert MATERIALS["solder_microbump"]["k"] == pytest.approx(57.0)
        assert MATERIALS["underfill"]["k"] == pytest.approx(0.2)


class TestCompositeBounds:
    @pytest.mark.parametrize("v", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    def test_lateral_between_series_and_parallel(self, v):
        kxy, _ = composite_conductivity(KB, KM, v)
        series = 1.0 / (v / KB + (1.0 - v) / KM)
        parallel = v * KB + (1.0 - v) * KM
        assert series <= kxy + 1e-12
        assert kxy <= parallel + 1e-12

    @pytest.mark.parametrize("v", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    def test_vertical_equals_parallel_bound(self, v):
        _, kz = composite_conductivity(KB, KM, v)
        assert kz == pytest.approx(v * KB + (1.0 - v) * KM, rel=1e-12)

    def test_lateral_monotone_in_fraction(self):
        vs = np.linspace(0.0, 1.0, 51)
        kxy = np.array([composite_conductivity(KB, KM, v)[0] for v in vs])
        assert np.all(np.diff(kxy) >= -1e-12)

    def test_old_placeholder_kz_is_above_parallel_bound(self):
        """The previous placeholder kz = 20 W/mK needs a bump fraction of 0.3486,
        i.e. 27 um bumps at 40 um pitch. At 20 um it is above the parallel bound."""
        v = bump_volume_fraction(BUMP_DIAMETER_UM, BUMP_PITCH_UM)
        parallel_bound_kz = v * KB + (1.0 - v) * KM
        assert 20.0 > parallel_bound_kz
