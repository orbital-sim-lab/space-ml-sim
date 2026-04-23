"""TDD tests for dose-depth curve analysis.

Models TID as a function of shielding thickness — the standard
deliverable in every radiation analysis report.
"""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig


class TestDoseDepthCurve:
    """TID vs shielding depth curve generation."""

    def test_generate_curve(self) -> None:
        from space_ml_sim.environment.dose_depth import generate_dose_depth_curve

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        curve = generate_dose_depth_curve(
            orbit=orbit,
            mission_years=5.0,
            shielding_range_mm=(0.5, 20.0),
            num_points=10,
        )
        assert len(curve.shielding_mm) == 10
        assert len(curve.dose_krad) == 10

    def test_dose_decreases_with_shielding(self) -> None:
        from space_ml_sim.environment.dose_depth import generate_dose_depth_curve

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        curve = generate_dose_depth_curve(orbit=orbit, mission_years=5.0, num_points=5)

        for i in range(len(curve.dose_krad) - 1):
            assert curve.dose_krad[i + 1] <= curve.dose_krad[i], (
                f"Dose should decrease: {curve.dose_krad[i]} -> {curve.dose_krad[i+1]} "
                f"at {curve.shielding_mm[i+1]}mm"
            )

    def test_higher_altitude_more_dose(self) -> None:
        from space_ml_sim.environment.dose_depth import generate_dose_depth_curve

        orbit_low = OrbitConfig(altitude_km=500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        orbit_high = OrbitConfig(altitude_km=1500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)

        c_low = generate_dose_depth_curve(orbit=orbit_low, mission_years=5.0, num_points=3)
        c_high = generate_dose_depth_curve(orbit=orbit_high, mission_years=5.0, num_points=3)

        # At same shielding, higher altitude should have more dose
        assert c_high.dose_krad[0] > c_low.dose_krad[0]

    def test_find_shielding_for_dose_limit(self) -> None:
        from space_ml_sim.environment.dose_depth import find_shielding_for_dose

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        shield_mm = find_shielding_for_dose(
            orbit=orbit,
            mission_years=5.0,
            target_dose_krad=5.0,
        )
        assert shield_mm >= 0

    def test_to_dataframe(self) -> None:
        from space_ml_sim.environment.dose_depth import generate_dose_depth_curve

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        curve = generate_dose_depth_curve(orbit=orbit, mission_years=5.0, num_points=5)
        df = curve.to_dataframe()

        assert len(df) == 5
        assert "shielding_mm_al" in df.columns
        assert "dose_krad" in df.columns
