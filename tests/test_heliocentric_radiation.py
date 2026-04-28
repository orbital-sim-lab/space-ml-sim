"""Tests for the heliocentric (interplanetary) radiation environment."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from space_ml_sim.environment.heliocentric_radiation import (
    HeliocentricEnvironment,
)
from space_ml_sim.environment.radiation import RadiationEnvironment


class TestHeliocentricEnvironmentBasics:
    """Construction, validation, and contract parity with RadiationEnvironment."""

    def test_defaults_at_1au_solar_min_match_calibration(self) -> None:
        env = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        # Calibration constants applied with all factors = 1.0
        assert env.base_seu_rate == pytest.approx(2.0e-12, rel=1e-6)
        assert env.tid_rate_krad_per_day == pytest.approx(1.0e-4, rel=1e-6)

    def test_solar_max_halves_rates(self) -> None:
        env_min = HeliocentricEnvironment(heliocentric_distance_au=1.0, solar_phase="min")
        env_max = HeliocentricEnvironment(heliocentric_distance_au=1.0, solar_phase="max")
        assert env_max.base_seu_rate == pytest.approx(env_min.base_seu_rate * 0.5)
        assert env_max.tid_rate_krad_per_day == pytest.approx(env_min.tid_rate_krad_per_day * 0.5)

    def test_distance_increases_gcr_outside_1au(self) -> None:
        inner = HeliocentricEnvironment(heliocentric_distance_au=0.5)
        earth = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        outer = HeliocentricEnvironment(heliocentric_distance_au=5.0)
        assert inner.base_seu_rate < earth.base_seu_rate < outer.base_seu_rate

    def test_distance_factor_saturates(self) -> None:
        far = HeliocentricEnvironment(heliocentric_distance_au=10.0)
        farther = HeliocentricEnvironment(heliocentric_distance_au=30.0)
        # Beyond 10 AU the model caps at 1.5x — outer must equal saturation
        assert far.base_seu_rate == pytest.approx(farther.base_seu_rate)

    def test_shielding_attenuates_rates(self) -> None:
        thin = HeliocentricEnvironment(heliocentric_distance_au=1.0, shielding_mm_al=2.0)
        thick = HeliocentricEnvironment(heliocentric_distance_au=1.0, shielding_mm_al=10.0)
        assert thick.base_seu_rate < thin.base_seu_rate
        assert thick.tid_rate_krad_per_day < thin.tid_rate_krad_per_day

    def test_distance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            HeliocentricEnvironment(heliocentric_distance_au=0.0)
        with pytest.raises(ValidationError):
            HeliocentricEnvironment(heliocentric_distance_au=-1.0)

    def test_distance_capped_at_50_au(self) -> None:
        # 50 AU is allowed (Voyager-class), 51 is rejected
        HeliocentricEnvironment(heliocentric_distance_au=50.0)
        with pytest.raises(ValidationError):
            HeliocentricEnvironment(heliocentric_distance_au=51.0)

    def test_solar_phase_must_be_valid_literal(self) -> None:
        with pytest.raises(ValidationError):
            HeliocentricEnvironment(heliocentric_distance_au=1.0, solar_phase="middle")  # type: ignore[arg-type]


class TestPresets:
    """Built-in mission presets produce sensible relative orderings."""

    def test_solar_min_is_worse_than_solar_max(self) -> None:
        worse = HeliocentricEnvironment.cruise_1au_solar_min()
        better = HeliocentricEnvironment.cruise_1au_solar_max()
        assert worse.base_seu_rate > better.base_seu_rate

    def test_venus_flyby_below_1au(self) -> None:
        venus = HeliocentricEnvironment.venus_flyby()
        assert venus.heliocentric_distance_au < 1.0
        # Closer to Sun → suppressed GCR
        assert venus.base_seu_rate < HeliocentricEnvironment.cruise_1au_solar_min().base_seu_rate

    def test_mars_transit_above_1au(self) -> None:
        mars = HeliocentricEnvironment.mars_transit()
        assert mars.heliocentric_distance_au == 1.5
        assert mars.base_seu_rate > HeliocentricEnvironment.cruise_1au_solar_min().base_seu_rate

    def test_lunar_transfer_matches_1au(self) -> None:
        lunar = HeliocentricEnvironment.lunar_transfer()
        assert lunar.heliocentric_distance_au == 1.0


class TestApiParity:
    """HeliocentricEnvironment must be a drop-in for RadiationEnvironment."""

    def test_has_same_public_rate_fields(self) -> None:
        helio = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        leo = RadiationEnvironment.leo_500km()
        assert hasattr(helio, "base_seu_rate")
        assert hasattr(helio, "tid_rate_krad_per_day")
        # Same units, same semantics — both are floats
        assert isinstance(helio.base_seu_rate, float)
        assert isinstance(leo.base_seu_rate, float)

    def test_sample_seu_events_returns_nonneg_int(self) -> None:
        env = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        rng = np.random.default_rng(42)
        n = env.sample_seu_events(
            chip_cross_section_cm2=1e-14,
            num_bits=1_000_000,
            dt_seconds=3600.0,
            rng=rng,
        )
        assert isinstance(n, int)
        assert n >= 0

    def test_sample_seu_events_reproducible_with_seed(self) -> None:
        env = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        a = env.sample_seu_events(1e-14, 10**8, 3600.0, np.random.default_rng(123))
        b = env.sample_seu_events(1e-14, 10**8, 3600.0, np.random.default_rng(123))
        assert a == b

    def test_sample_seu_mean_matches_lambda(self) -> None:
        """Many trials → sample mean ~ rate × bits × seconds."""
        env = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        bits = 10**8
        seconds = 86400.0  # 1 day
        expected = env.base_seu_rate * (1e-14 / 1e-14) * bits * seconds
        rng = np.random.default_rng(7)
        samples = [env.sample_seu_events(1e-14, bits, seconds, rng) for _ in range(200)]
        observed_mean = float(np.mean(samples))
        assert abs(observed_mean - expected) < max(0.2 * expected, 1.0)

    def test_tid_dose_scales_linearly(self) -> None:
        env = HeliocentricEnvironment(heliocentric_distance_au=1.0)
        one_day = env.tid_dose(86400.0)
        two_days = env.tid_dose(2 * 86400.0)
        assert two_days == pytest.approx(2 * one_day)


class TestPublishedOrderOfMagnitude:
    """Predictions must stay inside published order-of-magnitude envelopes.

    These ranges are intentionally wide (>10x) — the model is parametric
    and is meant to support trade studies, not flight qualification. Any
    drift outside these bands signals a calibration regression.
    """

    def test_lunar_orbit_tid_within_crater_band(self) -> None:
        """LRO/CRaTER reports ~14 rad/yr behind ~5mm Al at 1 AU solar min.

        Our 2mm Al ought to be 1x-5x higher (less shielding). Express the
        range as krad/year and check the prediction lands in a wide band.
        """
        env = HeliocentricEnvironment(
            heliocentric_distance_au=1.0,
            shielding_mm_al=2.0,
            solar_phase="min",
        )
        krad_per_year = env.tid_rate_krad_per_day * 365.25
        # Wide envelope: 5 mrad/day → 500 mrad/day
        assert 0.005 < krad_per_year < 0.5

    def test_seu_rate_in_voyager_class_band(self) -> None:
        """Voyager-class SRAM upsets in cruise: 1e-8 to 1e-6 per bit per day.

        We're calibrated near 1.7e-7 upsets/bit/day at 1 AU solar min, so
        any reasonable distance/phase combo should stay in the wide band.
        """
        env = HeliocentricEnvironment(
            heliocentric_distance_au=1.0,
            shielding_mm_al=2.0,
            solar_phase="min",
        )
        upsets_per_bit_per_day = env.base_seu_rate * 86400.0
        assert 1e-8 < upsets_per_bit_per_day < 1e-6
