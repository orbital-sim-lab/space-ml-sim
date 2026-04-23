"""TDD tests for shielding optimization recommender."""

from __future__ import annotations


from space_ml_sim.core.orbit import OrbitConfig


class TestShieldingOptimizer:
    """Recommender must find optimal shielding for given constraints."""

    def test_find_minimum_shielding_for_tid(self) -> None:
        from space_ml_sim.analysis.shielding_optimizer import find_minimum_shielding

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = find_minimum_shielding(
            orbit=orbit,
            max_tid_krad=10.0,
            mission_years=5.0,
        )

        assert result.shielding_mm_al >= 0
        assert result.achieved_tid_krad <= 10.0

    def test_higher_altitude_needs_more_shielding(self) -> None:
        from space_ml_sim.analysis.shielding_optimizer import find_minimum_shielding

        orbit_low = OrbitConfig(altitude_km=500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        orbit_high = OrbitConfig(
            altitude_km=1500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0
        )

        r_low = find_minimum_shielding(orbit=orbit_low, max_tid_krad=10.0, mission_years=5.0)
        r_high = find_minimum_shielding(orbit=orbit_high, max_tid_krad=10.0, mission_years=5.0)

        assert r_high.shielding_mm_al >= r_low.shielding_mm_al

    def test_reports_mass_penalty(self) -> None:
        from space_ml_sim.analysis.shielding_optimizer import find_minimum_shielding

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = find_minimum_shielding(
            orbit=orbit,
            max_tid_krad=10.0,
            mission_years=5.0,
        )
        assert result.mass_penalty_kg_m2 >= 0

    def test_sweep_returns_multiple_options(self) -> None:
        from space_ml_sim.analysis.shielding_optimizer import shielding_sweep

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        results = shielding_sweep(
            orbit=orbit,
            mission_years=5.0,
            shielding_range_mm=(0.5, 10.0),
            steps=5,
        )
        assert len(results) == 5
        # More shielding -> less TID
        assert results[-1].achieved_tid_krad < results[0].achieved_tid_krad
