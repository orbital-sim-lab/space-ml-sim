"""TDD tests for Single Event Latchup (SEL) modeling.

SEL is a destructive event where a parasitic thyristor latches,
drawing excessive current. Unlike SEU (soft error), SEL can
permanently damage the chip if not detected and power-cycled.
"""

from __future__ import annotations


from space_ml_sim.environment.radiation import RadiationEnvironment


class TestSELRate:
    """SEL rate prediction from chip cross-section and environment."""

    def test_sel_rate_positive(self) -> None:
        from space_ml_sim.environment.sel_model import sel_rate_per_day

        rad_env = RadiationEnvironment.leo_500km()
        rate = sel_rate_per_day(
            rad_env=rad_env,
            sel_cross_section_cm2=1e-8,
            sel_threshold_let=20.0,
        )
        assert rate >= 0

    def test_higher_cross_section_higher_rate(self) -> None:
        from space_ml_sim.environment.sel_model import sel_rate_per_day

        rad_env = RadiationEnvironment.leo_500km()
        r_low = sel_rate_per_day(
            rad_env=rad_env, sel_cross_section_cm2=1e-10, sel_threshold_let=20.0
        )
        r_high = sel_rate_per_day(
            rad_env=rad_env, sel_cross_section_cm2=1e-7, sel_threshold_let=20.0
        )
        assert r_high > r_low

    def test_higher_threshold_lower_rate(self) -> None:
        """Higher LET threshold means fewer particles can trigger latchup."""
        from space_ml_sim.environment.sel_model import sel_rate_per_day

        rad_env = RadiationEnvironment.leo_500km()
        r_low_thresh = sel_rate_per_day(
            rad_env=rad_env, sel_cross_section_cm2=1e-8, sel_threshold_let=10.0
        )
        r_high_thresh = sel_rate_per_day(
            rad_env=rad_env, sel_cross_section_cm2=1e-8, sel_threshold_let=60.0
        )
        assert r_high_thresh < r_low_thresh

    def test_rad_hard_immune(self) -> None:
        """Chips with very high SEL threshold should be effectively immune."""
        from space_ml_sim.environment.sel_model import sel_rate_per_day

        rad_env = RadiationEnvironment.leo_500km()
        # LET threshold of 100 MeV·cm²/mg — above any practical particle
        rate = sel_rate_per_day(
            rad_env=rad_env, sel_cross_section_cm2=1e-8, sel_threshold_let=100.0
        )
        assert rate < 1e-10  # Effectively zero


class TestSELMissionRisk:
    """Mission-level SEL probability."""

    def test_mission_sel_probability(self) -> None:
        from space_ml_sim.environment.sel_model import mission_sel_probability

        rad_env = RadiationEnvironment.leo_500km()
        prob = mission_sel_probability(
            rad_env=rad_env,
            sel_cross_section_cm2=1e-8,
            sel_threshold_let=20.0,
            mission_years=5.0,
        )
        assert 0.0 <= prob <= 1.0

    def test_longer_mission_higher_probability(self) -> None:
        from space_ml_sim.environment.sel_model import mission_sel_probability

        rad_env = RadiationEnvironment.leo_500km()
        kwargs = dict(rad_env=rad_env, sel_cross_section_cm2=1e-8, sel_threshold_let=20.0)
        p1 = mission_sel_probability(**kwargs, mission_years=1.0)
        p10 = mission_sel_probability(**kwargs, mission_years=10.0)
        assert p10 >= p1

    def test_power_cycle_recovery_time(self) -> None:
        """SEL mitigation requires knowing recovery time."""
        from space_ml_sim.environment.sel_model import sel_mitigation_requirements

        result = sel_mitigation_requirements(
            sel_rate_per_day=0.01,
            power_cycle_time_seconds=5.0,
            mission_years=5.0,
        )
        assert result.expected_sel_events > 0
        assert result.total_downtime_hours >= 0
        assert result.availability_fraction <= 1.0
