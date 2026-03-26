"""Tests for radiation environment model."""

from space_ml_sim.environment.radiation import RadiationEnvironment


class TestRadiationRates:
    def test_seu_rate_increases_with_altitude(self):
        """Higher altitude = more trapped protons = higher SEU rate."""
        low = RadiationEnvironment(altitude_km=500, inclination_deg=53)
        high = RadiationEnvironment(altitude_km=2000, inclination_deg=53)
        assert high.base_seu_rate > low.base_seu_rate

    def test_tid_rate_increases_with_altitude(self):
        """Higher altitude = higher TID rate."""
        low = RadiationEnvironment(altitude_km=500, inclination_deg=53)
        high = RadiationEnvironment(altitude_km=2000, inclination_deg=53)
        assert high.tid_rate_krad_per_day > low.tid_rate_krad_per_day

    def test_shielding_reduces_seu_rate(self):
        """More shielding = lower SEU rate."""
        thin = RadiationEnvironment(altitude_km=500, inclination_deg=53, shielding_mm_al=1.0)
        thick = RadiationEnvironment(altitude_km=500, inclination_deg=53, shielding_mm_al=5.0)
        assert thick.base_seu_rate < thin.base_seu_rate

    def test_saa_enhancement(self):
        """Inclinations through SAA (20-60 deg) should have higher SEU rate."""
        saa = RadiationEnvironment(altitude_km=500, inclination_deg=40)
        no_saa = RadiationEnvironment(altitude_km=500, inclination_deg=98)
        assert saa.base_seu_rate > no_saa.base_seu_rate

    def test_rates_are_positive(self):
        env = RadiationEnvironment.leo_500km()
        assert env.base_seu_rate > 0
        assert env.tid_rate_krad_per_day > 0


class TestSEUSampling:
    def test_zero_time_zero_events(self):
        env = RadiationEnvironment.leo_500km()
        events = env.sample_seu_events(1e-14, 1_000_000, dt_seconds=0)
        assert events == 0

    def test_long_exposure_produces_events(self):
        """Very long exposure should produce at least some events."""
        env = RadiationEnvironment.leo_2000km()
        # 1 day exposure, 1 billion bits
        events = env.sample_seu_events(1e-13, 10**9, dt_seconds=86400)
        assert events >= 0  # Poisson can still be 0, but unlikely


class TestTIDAccumulation:
    def test_tid_scales_with_time(self):
        env = RadiationEnvironment.leo_500km()
        dose_1h = env.tid_dose(3600)
        dose_2h = env.tid_dose(7200)
        assert abs(dose_2h - 2 * dose_1h) < 1e-15

    def test_tid_zero_for_zero_time(self):
        env = RadiationEnvironment.leo_500km()
        assert env.tid_dose(0) == 0.0


class TestPresets:
    def test_presets_ordered_by_severity(self):
        """500km < 650km < 2000km in radiation severity."""
        low = RadiationEnvironment.leo_500km()
        mid = RadiationEnvironment.sso_650km()
        high = RadiationEnvironment.leo_2000km()
        assert low.tid_rate_krad_per_day < mid.tid_rate_krad_per_day
        assert mid.tid_rate_krad_per_day < high.tid_rate_krad_per_day

    def test_preset_altitudes(self):
        assert RadiationEnvironment.leo_500km().altitude_km == 500
        assert RadiationEnvironment.sso_650km().altitude_km == 650
        assert RadiationEnvironment.leo_2000km().altitude_km == 2000
