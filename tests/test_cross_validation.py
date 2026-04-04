"""Cross-validation tests: Keplerian model vs SGP4 reference.

These tests validate that our simplified Keplerian model produces
physically reasonable results by comparing properties (not exact
positions) against the SGP4 reference implementation.

The two models use fundamentally different approaches:
- SGP4: Mean elements with Brouwer theory + drag
- Our model: Osculating Keplerian elements with J2 RAAN drift

They will NOT produce identical positions, but must agree on:
- Orbital altitude (within ~20 km)
- Orbital period (within ~1%)
- General trajectory shape (same inclination, similar RAAN)
"""

from __future__ import annotations

import math


from space_ml_sim.core.orbit import position_at, R_EARTH_KM
from space_ml_sim.core.tle import parse_tle, propagate_sgp4

# ISS TLE for cross-validation
ISS_L1 = "1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075"


class TestAltitudeAgreement:
    """Both models must produce the correct orbital altitude."""

    def test_sgp4_altitude_is_iss_range(self):
        """SGP4 altitude at epoch must be in ISS range (400-430 km)."""
        pos = propagate_sgp4(ISS_L1, ISS_L2, 0.0)
        alt = math.sqrt(sum(x**2 for x in pos)) - R_EARTH_KM
        assert 390 < alt < 440, f"SGP4 altitude: {alt:.1f} km"

    def test_keplerian_altitude_is_iss_range(self):
        """Keplerian altitude must be in ISS range (400-430 km)."""
        config = parse_tle(ISS_L1, ISS_L2)
        pos = position_at(config, time_seconds=0.0, use_j2=False)
        alt = math.sqrt(sum(x**2 for x in pos)) - R_EARTH_KM
        assert 390 < alt < 440, f"Keplerian altitude: {alt:.1f} km"

    def test_altitude_agreement_within_20km(self):
        """Both models must agree on altitude within 20 km."""
        config = parse_tle(ISS_L1, ISS_L2)

        for minutes in [0.0, 45.0, 90.0]:
            sgp4_pos = propagate_sgp4(ISS_L1, ISS_L2, minutes)
            kepler_pos = position_at(config, time_seconds=minutes * 60, use_j2=True)

            sgp4_alt = math.sqrt(sum(x**2 for x in sgp4_pos)) - R_EARTH_KM
            kepler_alt = math.sqrt(sum(x**2 for x in kepler_pos)) - R_EARTH_KM

            assert abs(sgp4_alt - kepler_alt) < 20, (
                f"Altitude disagreement at t={minutes}min: "
                f"SGP4={sgp4_alt:.1f}, Kepler={kepler_alt:.1f}"
            )


class TestOrbitalPeriodAgreement:
    """Both models must produce similar orbital periods."""

    def test_sgp4_returns_to_similar_altitude_after_one_period(self):
        """SGP4 altitude after one orbital period should be similar to epoch."""
        config = parse_tle(ISS_L1, ISS_L2)
        period_min = config.orbital_period_seconds / 60

        pos_0 = propagate_sgp4(ISS_L1, ISS_L2, 0.0)
        pos_T = propagate_sgp4(ISS_L1, ISS_L2, period_min)

        alt_0 = math.sqrt(sum(x**2 for x in pos_0)) - R_EARTH_KM
        alt_T = math.sqrt(sum(x**2 for x in pos_T)) - R_EARTH_KM

        # Altitude should be similar after one period (within eccentricity effects)
        assert abs(alt_T - alt_0) < 10, (
            f"Altitude change after 1 period: {abs(alt_T - alt_0):.2f} km"
        )

    def test_keplerian_period_matches_sgp4_within_1_percent(self):
        """Keplerian period must be within 1% of SGP4-derived period.

        SGP4 mean motion gives the reference period.
        """
        from sgp4.api import Satrec, WGS72

        sat = Satrec.twoline2rv(ISS_L1, ISS_L2, WGS72)
        n_rad_per_min = sat.no_kozai if hasattr(sat, "no_kozai") else sat.no
        sgp4_period_min = 2 * math.pi / n_rad_per_min

        config = parse_tle(ISS_L1, ISS_L2)
        kepler_period_min = config.orbital_period_seconds / 60

        error_pct = abs(kepler_period_min - sgp4_period_min) / sgp4_period_min * 100
        assert error_pct < 1.0, (
            f"Period error: {error_pct:.3f}% "
            f"(Kepler={kepler_period_min:.2f}, SGP4={sgp4_period_min:.2f})"
        )


class TestInclinationPreservation:
    """Inclination from TLE must be correctly extracted."""

    def test_parsed_inclination_matches_tle(self):
        """parse_tle must extract the correct inclination (51.6416 deg for ISS)."""
        config = parse_tle(ISS_L1, ISS_L2)
        assert abs(config.inclination_deg - 51.6416) < 0.001

    def test_parsed_raan_matches_tle(self):
        """parse_tle must extract the correct RAAN (247.4627 deg for ISS)."""
        config = parse_tle(ISS_L1, ISS_L2)
        assert abs(config.raan_deg - 247.4627) < 0.001


class TestSGP4Consistency:
    """SGP4 propagation must be self-consistent."""

    def test_sgp4_position_magnitude_is_orbital(self):
        """SGP4 position vector magnitude must be at orbital radius."""
        for minutes in [0, 30, 60, 90, 120]:
            pos = propagate_sgp4(ISS_L1, ISS_L2, float(minutes))
            r = math.sqrt(sum(x**2 for x in pos))
            # ISS: ~6780 km from Earth center
            assert 6700 < r < 6850, f"SGP4 radius at t={minutes}min: {r:.1f} km"

    def test_sgp4_position_changes_over_time(self):
        """SGP4 positions must differ at different times."""
        pos_0 = propagate_sgp4(ISS_L1, ISS_L2, 0.0)
        pos_30 = propagate_sgp4(ISS_L1, ISS_L2, 30.0)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_0, pos_30)))
        assert dist > 100, f"SGP4 should move >100km in 30min, got {dist:.1f}km"
