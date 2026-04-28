"""Tests for the Solar Particle Event statistical model."""

from __future__ import annotations

import numpy as np
import pytest

from space_ml_sim.environment.solar_particle_event import (
    SPEStatisticalModel,
    SolarParticleEvent,
    mission_spe_dose,
)


class TestEventFrequencies:
    """Magnitude ordering and solar-phase modulation."""

    def test_smaller_events_more_frequent(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        assert m.annual_event_frequency("small") > m.annual_event_frequency("medium")
        assert m.annual_event_frequency("medium") > m.annual_event_frequency("large")
        assert m.annual_event_frequency("large") > m.annual_event_frequency("extreme")

    def test_solar_min_is_quieter(self) -> None:
        max_model = SPEStatisticalModel(solar_phase="max")
        min_model = SPEStatisticalModel(solar_phase="min")
        for mag in ("small", "medium", "large", "extreme"):
            assert min_model.annual_event_frequency(mag) < max_model.annual_event_frequency(mag)  # type: ignore[arg-type]

    def test_window_scales_linearly(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        a = m.expected_events_in_window(180.0, "medium")
        b = m.expected_events_in_window(360.0, "medium")
        assert b == pytest.approx(2 * a)


class TestDoseEstimates:
    """Mean and worst-case dose."""

    def test_longer_mission_more_dose(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        assert m.expected_dose_krad(60.0) < m.expected_dose_krad(360.0)

    def test_thicker_shielding_reduces_dose(self) -> None:
        thin = SPEStatisticalModel(solar_phase="max", shielding_mm_al=1.0)
        thick = SPEStatisticalModel(solar_phase="max", shielding_mm_al=10.0)
        assert thick.expected_dose_krad(180.0) < thin.expected_dose_krad(180.0)

    def test_p95_is_at_least_the_mean(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        for days in (30.0, 180.0, 730.0):
            mean = m.expected_dose_krad(days)
            p95 = m.worst_case_dose_krad(days, 0.95)
            assert p95 >= mean

    def test_p95_percentile_must_be_in_range(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        with pytest.raises(ValueError):
            m.worst_case_dose_krad(180.0, percentile=0.4)
        with pytest.raises(ValueError):
            m.worst_case_dose_krad(180.0, percentile=1.0)

    def test_yearlong_mission_dose_in_published_band(self) -> None:
        """A 1-year mission at solar max behind 2 mm Al should land
        between 1 and 50 krad — the band reported by JPL/Xapsos for
        contemporary spacecraft mission planning.
        """
        m = SPEStatisticalModel(solar_phase="max", shielding_mm_al=2.0)
        dose = m.worst_case_dose_krad(365.0, percentile=0.95)
        assert 1.0 < dose < 50.0


class TestMonteCarlo:
    """Sampling produces realistic mission realizations."""

    def test_sample_returns_events_and_dose(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        events, dose = m.sample_mission(180.0, rng=np.random.default_rng(42))
        assert isinstance(events, list)
        assert all(isinstance(e, SolarParticleEvent) for e in events)
        assert dose >= 0.0

    def test_sample_reproducible_with_seed(self) -> None:
        m = SPEStatisticalModel(solar_phase="max")
        a_events, a_dose = m.sample_mission(365.0, rng=np.random.default_rng(7))
        b_events, b_dose = m.sample_mission(365.0, rng=np.random.default_rng(7))
        assert a_dose == b_dose
        assert len(a_events) == len(b_events)

    def test_monte_carlo_mean_converges(self) -> None:
        """Average over many trials should approach the closed-form mean."""
        m = SPEStatisticalModel(solar_phase="max")
        rng = np.random.default_rng(0)
        n_trials = 500
        doses = [m.sample_mission(365.0, rng=rng)[1] for _ in range(n_trials)]
        observed_mean = float(np.mean(doses))
        expected_mean = m.expected_dose_krad(365.0)
        assert abs(observed_mean - expected_mean) < 0.3 * expected_mean


class TestConvenienceWrapper:
    """`mission_spe_dose` matches the underlying model."""

    def test_default_returns_p95(self) -> None:
        wrapped = mission_spe_dose(180.0, solar_phase="max", shielding_mm_al=2.0)
        m = SPEStatisticalModel(solar_phase="max", shielding_mm_al=2.0)
        assert wrapped == pytest.approx(m.worst_case_dose_krad(180.0, 0.95))

    def test_mean_method(self) -> None:
        wrapped = mission_spe_dose(180.0, solar_phase="max", shielding_mm_al=2.0, method="mean")
        m = SPEStatisticalModel(solar_phase="max", shielding_mm_al=2.0)
        assert wrapped == pytest.approx(m.expected_dose_krad(180.0))
