"""TDD tests for orbital thermal cycling model."""

from __future__ import annotations


from space_ml_sim.core.orbit import OrbitConfig


class TestThermalCycling:
    """Orbital thermal cycling with eclipse/sunlit transitions."""

    def test_generate_thermal_profile(self) -> None:
        from space_ml_sim.environment.thermal_cycling import (
            generate_thermal_profile,
            ThermalProfile,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        profile = generate_thermal_profile(
            orbit=orbit,
            duration_seconds=6000.0,  # ~1 orbit
            step_seconds=30.0,
        )
        assert isinstance(profile, ThermalProfile)
        assert len(profile.times_seconds) > 0
        assert len(profile.temperatures_c) == len(profile.times_seconds)

    def test_temperature_range_realistic(self) -> None:
        """LEO thermal cycling: roughly -40°C to +60°C for typical spacecraft."""
        from space_ml_sim.environment.thermal_cycling import generate_thermal_profile

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        profile = generate_thermal_profile(orbit=orbit, duration_seconds=6000.0)

        assert min(profile.temperatures_c) > -120  # Not colder than deep space
        assert max(profile.temperatures_c) < 100  # Not hotter than direct sun

    def test_eclipse_cools_sunlit_heats(self) -> None:
        """Temperature should drop in eclipse and rise in sunlit."""
        from space_ml_sim.environment.thermal_cycling import generate_thermal_profile

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        profile = generate_thermal_profile(orbit=orbit, duration_seconds=6000.0, step_seconds=10.0)

        # Should see both warming and cooling phases
        diffs = [
            profile.temperatures_c[i + 1] - profile.temperatures_c[i]
            for i in range(len(profile.temperatures_c) - 1)
        ]
        has_heating = any(d > 0 for d in diffs)
        has_cooling = any(d < 0 for d in diffs)
        assert has_heating and has_cooling

    def test_cycles_per_orbit(self) -> None:
        from space_ml_sim.environment.thermal_cycling import generate_thermal_profile

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        # 3 orbits
        profile = generate_thermal_profile(orbit=orbit, duration_seconds=18000.0, step_seconds=30.0)
        assert profile.num_cycles >= 2  # At least 2 complete thermal cycles

    def test_component_derating(self) -> None:
        """Component performance degrades at temperature extremes."""
        from space_ml_sim.environment.thermal_cycling import derate_at_temperature

        # Normal operating temperature: no derating
        factor_25 = derate_at_temperature(25.0, max_temp_c=85.0, min_temp_c=-40.0)
        assert 0.95 <= factor_25 <= 1.0

        # Near max temp: some derating
        factor_80 = derate_at_temperature(80.0, max_temp_c=85.0, min_temp_c=-40.0)
        assert factor_80 < factor_25

        # Below min temp: significant derating
        factor_minus50 = derate_at_temperature(-50.0, max_temp_c=85.0, min_temp_c=-40.0)
        assert factor_minus50 < factor_25
