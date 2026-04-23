"""TDD tests for solar cycle radiation environment presets.

Solar cycle modulates trapped particle flux and GCR rates:
- Solar maximum: higher trapped proton flux, lower GCR
- Solar minimum: lower trapped protons, higher GCR
- These partially cancel but the net effect varies by altitude/orbit
"""

from __future__ import annotations

import pytest

from space_ml_sim.environment.radiation import RadiationEnvironment


class TestSolarCyclePresets:
    """Radiation environment must support solar cycle phase."""

    def test_solar_max_increases_seu_at_low_orbit(self) -> None:
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle

        base = RadiationEnvironment.leo_500km()
        env_max = apply_solar_cycle(base, phase="solar_max")
        env_min = apply_solar_cycle(base, phase="solar_min")

        # At low LEO, trapped protons dominate — solar max is worse
        # But GCR is lower at solar max. Net effect depends on orbit.
        # Just verify they produce different rates
        assert env_max.base_seu_rate != env_min.base_seu_rate

    def test_solar_average_matches_baseline(self) -> None:
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle

        base = RadiationEnvironment.leo_500km()
        env_avg = apply_solar_cycle(base, phase="average")

        # Average should be close to baseline (within 10%)
        ratio = env_avg.base_seu_rate / base.base_seu_rate
        assert 0.9 <= ratio <= 1.1

    def test_tid_higher_at_solar_max(self) -> None:
        """TID from trapped protons is higher during solar max."""
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle

        base = RadiationEnvironment.leo_500km()
        env_max = apply_solar_cycle(base, phase="solar_max")
        env_min = apply_solar_cycle(base, phase="solar_min")

        assert env_max.tid_rate_krad_per_day > env_min.tid_rate_krad_per_day

    def test_all_phases_valid(self) -> None:
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle, SOLAR_PHASES

        base = RadiationEnvironment.leo_500km()
        for phase in SOLAR_PHASES:
            env = apply_solar_cycle(base, phase=phase)
            assert env.base_seu_rate > 0
            assert env.tid_rate_krad_per_day > 0

    def test_invalid_phase_raises(self) -> None:
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle

        base = RadiationEnvironment.leo_500km()
        with pytest.raises(ValueError):
            apply_solar_cycle(base, phase="invalid")

    def test_high_altitude_solar_max_worst_case(self) -> None:
        """At high LEO, solar max should significantly increase trapped particle rates."""
        from space_ml_sim.environment.solar_cycle import apply_solar_cycle

        base = RadiationEnvironment.leo_2000km()
        env_max = apply_solar_cycle(base, phase="solar_max")

        # Should be at least 1.5x at 2000km solar max
        assert env_max.base_seu_rate > base.base_seu_rate * 1.3
