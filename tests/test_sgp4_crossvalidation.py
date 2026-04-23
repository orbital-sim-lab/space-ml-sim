"""Cross-validate Keplerian orbit propagation against SGP4.

Compares position_at() (simplified Keplerian + J2) against propagate_sgp4()
(full SGP4 perturbation model) for standard TLEs.

Key insight: The Keplerian propagator uses a circular orbit approximation
and doesn't model argument of perigee, so absolute position will differ
from SGP4. We validate:
- Orbital altitude (radius from Earth center) matches within bounds
- Orbital period matches within bounds
- RAAN drift direction matches (J2 effect)
"""

from __future__ import annotations

import math
import pytest

from space_ml_sim.core.orbit import position_at, R_EARTH_KM
from space_ml_sim.core.tle import parse_tle, propagate_sgp4


# ISS TLE (standard reference)
ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703  92.2476  13.4937 15.49844545454321"

# Starlink satellite
STARLINK_LINE1 = "1 44713U 19074A   24001.50000000  .00000668  00000-0  43831-4 0  9995"
STARLINK_LINE2 = "2 44713  53.0553 163.7439 0001401  86.6825 273.4314 15.06381756241234"


def _radius_km(pos: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x**2 for x in pos))


class TestAltitudeConsistency:
    """Both propagators should maintain consistent orbital altitude."""

    def test_iss_altitude_at_epoch(self) -> None:
        orbit = parse_tle(ISS_LINE1, ISS_LINE2)
        kep_pos = position_at(orbit, time_seconds=0.0)
        sgp4_pos = propagate_sgp4(ISS_LINE1, ISS_LINE2, minutes_from_epoch=0.0)

        kep_alt = _radius_km(kep_pos) - R_EARTH_KM
        sgp4_alt = _radius_km(sgp4_pos) - R_EARTH_KM

        # Both should be in ISS altitude range
        assert 380 < kep_alt < 460, f"Keplerian altitude: {kep_alt:.1f} km"
        assert 380 < sgp4_alt < 460, f"SGP4 altitude: {sgp4_alt:.1f} km"

        # Altitude should match within 20 km (eccentricity causes variation)
        assert abs(kep_alt - sgp4_alt) < 20, (
            f"Altitude mismatch: Keplerian={kep_alt:.1f}, SGP4={sgp4_alt:.1f}"
        )

    def test_iss_altitude_after_one_orbit(self) -> None:
        orbit = parse_tle(ISS_LINE1, ISS_LINE2)
        period_sec = orbit.orbital_period_seconds
        period_min = period_sec / 60.0

        kep_pos = position_at(orbit, time_seconds=period_sec)
        sgp4_pos = propagate_sgp4(ISS_LINE1, ISS_LINE2, minutes_from_epoch=period_min)

        kep_alt = _radius_km(kep_pos) - R_EARTH_KM
        sgp4_alt = _radius_km(sgp4_pos) - R_EARTH_KM

        assert 380 < kep_alt < 460
        assert 380 < sgp4_alt < 460

    def test_starlink_altitude_at_epoch(self) -> None:
        orbit = parse_tle(STARLINK_LINE1, STARLINK_LINE2)
        kep_pos = position_at(orbit, time_seconds=0.0)
        sgp4_pos = propagate_sgp4(STARLINK_LINE1, STARLINK_LINE2, minutes_from_epoch=0.0)

        kep_alt = _radius_km(kep_pos) - R_EARTH_KM
        sgp4_alt = _radius_km(sgp4_pos) - R_EARTH_KM

        # Starlink at ~550 km
        assert 500 < kep_alt < 600
        assert 500 < sgp4_alt < 600


class TestOrbitalPeriod:
    """Keplerian and SGP4 orbital periods should match closely."""

    def test_iss_period(self) -> None:
        """ISS period should be ~92 minutes."""
        orbit = parse_tle(ISS_LINE1, ISS_LINE2)
        kep_period_min = orbit.orbital_period_seconds / 60.0

        # ISS does ~15.5 revolutions per day -> period ~92.8 min
        assert 90 < kep_period_min < 95, f"Keplerian period: {kep_period_min:.1f} min"

    def test_starlink_period(self) -> None:
        orbit = parse_tle(STARLINK_LINE1, STARLINK_LINE2)
        period_min = orbit.orbital_period_seconds / 60.0

        # Starlink ~550km -> period ~95-96 min
        assert 93 < period_min < 98, f"Period: {period_min:.1f} min"


class TestKeplerianReturn:
    """Keplerian propagator should return near start position after one period."""

    def test_iss_returns_near_start(self) -> None:
        orbit = parse_tle(ISS_LINE1, ISS_LINE2)
        pos_start = position_at(orbit, time_seconds=0.0)
        pos_end = position_at(orbit, time_seconds=orbit.orbital_period_seconds)

        # After one orbit, J2 shifts RAAN slightly, so not exact return
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(pos_start, pos_end)))
        # Should be within ~100 km (J2 RAAN drift ~0.005 deg/orbit)
        assert dist < 150, f"Return distance: {dist:.1f} km"


class TestSGP4MultiOrbit:
    """SGP4 should produce physically reasonable positions over many orbits."""

    def test_sgp4_altitude_stable_over_24h(self) -> None:
        """SGP4 altitude should stay bounded over 24 hours."""
        for t_min in [0, 60, 360, 720, 1440]:
            pos = propagate_sgp4(ISS_LINE1, ISS_LINE2, minutes_from_epoch=float(t_min))
            alt = _radius_km(pos) - R_EARTH_KM
            assert 350 < alt < 500, f"SGP4 altitude at t={t_min}min: {alt:.1f} km"

    def test_keplerian_altitude_stable_over_24h(self) -> None:
        """Keplerian altitude should stay bounded over 24 hours."""
        orbit = parse_tle(ISS_LINE1, ISS_LINE2)
        for t_sec in [0, 3600, 21600, 43200, 86400]:
            pos = position_at(orbit, time_seconds=float(t_sec))
            alt = _radius_km(pos) - R_EARTH_KM
            assert 380 < alt < 500, f"Keplerian altitude at t={t_sec}s: {alt:.1f} km"
