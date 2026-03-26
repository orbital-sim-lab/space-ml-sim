"""Tests for orbital mechanics module."""

import math

import pytest

from space_ml_sim.core.orbit import (
    OrbitConfig,
    propagate,
    walker_delta_orbits,
    sun_synchronous_orbits,
    is_in_eclipse,
    R_EARTH_KM,
)


class TestOrbitConfig:
    def test_semi_major_axis(self):
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        assert config.semi_major_axis_km == R_EARTH_KM + 550

    def test_orbital_period_leo(self):
        """LEO at 550km should have ~96 minute period."""
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        period_min = config.orbital_period_seconds / 60
        assert 90 < period_min < 100

    def test_mean_motion_positive(self):
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        assert config.mean_motion_rad_per_sec > 0


class TestPropagate:
    def test_returns_correct_number_of_states(self):
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        states = propagate(config, duration_minutes=10, step_seconds=60)
        assert len(states) == 11  # 0 to 10 minutes inclusive

    def test_position_magnitude_is_orbital_radius(self):
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        states = propagate(config, duration_minutes=1, step_seconds=60)
        for state in states:
            r = math.sqrt(sum(x**2 for x in state.position_km))
            expected = R_EARTH_KM + 550
            assert abs(r - expected) < 1.0  # Within 1km

    def test_satellite_moves_over_time(self):
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        states = propagate(config, duration_minutes=10, step_seconds=60)
        assert states[0].position_km != states[-1].position_km


class TestWalkerDelta:
    def test_correct_number_of_orbits(self):
        orbits = walker_delta_orbits(
            num_planes=10, sats_per_plane=10, altitude_km=550, inclination_deg=53
        )
        assert len(orbits) == 100

    def test_all_same_altitude(self):
        orbits = walker_delta_orbits(
            num_planes=5, sats_per_plane=4, altitude_km=550, inclination_deg=53
        )
        for orbit in orbits:
            assert orbit.altitude_km == 550

    def test_raan_spacing(self):
        """RAAN should be evenly spaced across 360 degrees."""
        orbits = walker_delta_orbits(
            num_planes=4, sats_per_plane=2, altitude_km=550, inclination_deg=53
        )
        raans = sorted(set(o.raan_deg for o in orbits))
        assert len(raans) == 4
        for i in range(len(raans) - 1):
            assert abs(raans[i + 1] - raans[i] - 90.0) < 0.01


class TestSunSynchronous:
    def test_correct_number_of_orbits(self):
        orbits = sun_synchronous_orbits(num_sats=10, altitude_km=650)
        assert len(orbits) == 10

    def test_inclination_is_retrograde(self):
        """SSO inclination should be > 90 degrees (retrograde)."""
        orbits = sun_synchronous_orbits(num_sats=1, altitude_km=650)
        assert orbits[0].inclination_deg > 90

    def test_higher_altitude_higher_inclination(self):
        """Higher SSO altitude requires more inclination (further from 90).

        At higher altitudes, the J2 effect weakens (a is larger), so cos(i)
        must be more negative, meaning i must be further from 90 degrees.
        """
        low = sun_synchronous_orbits(num_sats=1, altitude_km=400)
        high = sun_synchronous_orbits(num_sats=1, altitude_km=800)
        assert high[0].inclination_deg > low[0].inclination_deg


class TestEclipse:
    def test_sunlit_satellite(self):
        """Satellite on the sun side should not be in eclipse."""
        pos = (R_EARTH_KM + 550, 0, 0)
        sun_dir = (1, 0, 0)
        assert not is_in_eclipse(pos, sun_dir)

    def test_eclipsed_satellite(self):
        """Satellite directly behind Earth from Sun should be in eclipse."""
        pos = (-(R_EARTH_KM + 550), 0, 0)
        sun_dir = (1, 0, 0)
        assert is_in_eclipse(pos, sun_dir)

    def test_satellite_offset_from_shadow(self):
        """Satellite far from shadow axis should not be eclipsed."""
        pos = (-(R_EARTH_KM + 550), R_EARTH_KM + 1000, 0)
        sun_dir = (1, 0, 0)
        assert not is_in_eclipse(pos, sun_dir)
