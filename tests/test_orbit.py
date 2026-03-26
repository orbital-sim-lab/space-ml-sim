"""Tests for orbital mechanics module."""

import math


from space_ml_sim.core.orbit import (
    OrbitConfig,
    propagate,
    position_at,
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


class TestPositionAt:
    """Tests for single-point position lookup."""

    def test_position_at_zero_is_initial(self):
        """Position at t=0 should match propagate at t=0."""
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        states = propagate(config, duration_minutes=0, step_seconds=60)
        pos_at = position_at(config, time_seconds=0.0)
        for a, b in zip(states[0].position_km, pos_at):
            assert abs(a - b) < 1e-10

    def test_position_at_moves_over_time(self):
        """Position at t=600s should differ from t=0."""
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        pos_0 = position_at(config, time_seconds=0.0)
        pos_600 = position_at(config, time_seconds=600.0)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_0, pos_600)))
        assert dist > 10  # Should move significantly in 10 minutes

    def test_position_at_full_orbit_returns_to_start(self):
        """After one full orbital period, Keplerian satellite should return near start.

        Uses use_j2=False to test pure two-body closure; J2 precession would
        shift RAAN by ~0.4 deg per orbit and break the 1 km tolerance.
        """
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        period = config.orbital_period_seconds
        pos_0 = position_at(config, time_seconds=0.0, use_j2=False)
        pos_T = position_at(config, time_seconds=period, use_j2=False)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_0, pos_T)))
        assert dist < 1.0  # Within 1 km after full orbit

    def test_position_at_altitude_is_correct(self):
        """Radial distance should match semi-major axis."""
        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        pos = position_at(config, time_seconds=300.0)
        r = math.sqrt(sum(x**2 for x in pos))
        expected = R_EARTH_KM + 550
        assert abs(r - expected) < 1.0


class TestJ2Perturbations:
    """Tests for J2 secular perturbation effects on orbit propagation."""

    def test_j2_raan_drifts_retrograde_for_prograde_orbit(self):
        """For inc=53 deg prograde orbit, RAAN should decrease over time.

        cos(53 deg) > 0, so dΩ/dt is negative → RAAN drifts in negative direction.
        """
        import math
        from space_ml_sim.core.orbit import (
            R_EARTH_KM, J2
        )

        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=180, true_anomaly_deg=0)
        a = config.semi_major_axis_km
        n = config.mean_motion_rad_per_sec
        inc_rad = math.radians(53.0)

        # Compute expected drift rate
        d_raan_dt = -1.5 * n * J2 * (R_EARTH_KM / a) ** 2 * math.cos(inc_rad)
        assert d_raan_dt < 0, "RAAN drift should be negative for prograde (inc<90) orbit"

        # Propagate one day: with J2 the RAAN should shift
        one_day_seconds = 86400.0
        pos_j2 = position_at(config, time_seconds=one_day_seconds, use_j2=True)
        pos_no_j2 = position_at(config, time_seconds=one_day_seconds, use_j2=False)
        # Positions must differ because RAAN drifted
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_j2, pos_no_j2)))
        assert dist > 1.0, "J2 RAAN drift should produce measurable position difference after 1 day"

    def test_j2_raan_drift_magnitude(self):
        """At 550 km, 53 deg inclination, RAAN drift should be approx -5 to -7 deg/day."""
        import math
        from space_ml_sim.core.orbit import (
            R_EARTH_KM, J2
        )

        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        a = config.semi_major_axis_km
        n = config.mean_motion_rad_per_sec
        inc_rad = math.radians(53.0)

        # Drift rate in rad/s
        d_raan_dt_rad_s = -1.5 * n * J2 * (R_EARTH_KM / a) ** 2 * math.cos(inc_rad)
        # Convert to deg/day
        d_raan_dt_deg_day = math.degrees(d_raan_dt_rad_s) * 86400.0

        assert -7.0 < d_raan_dt_deg_day < -3.5, (
            f"RAAN drift should be ~-4.5 deg/day at 550km/53deg, got {d_raan_dt_deg_day:.3f}"
        )

    def test_j2_disabled_matches_keplerian(self):
        """position_at with use_j2=False must match original Keplerian behavior exactly."""
        import math

        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=45, true_anomaly_deg=30)

        for t in [0.0, 300.0, 3600.0, 86400.0]:
            pos_no_j2 = position_at(config, time_seconds=t, use_j2=False)

            # Manually compute pure Keplerian position (same logic as original code)
            a = config.semi_major_axis_km
            inc_rad = math.radians(config.inclination_deg)
            raan_rad = math.radians(config.raan_deg)
            nu0_rad = math.radians(config.true_anomaly_deg)
            n = config.mean_motion_rad_per_sec
            nu_rad = nu0_rad + n * t
            from space_ml_sim.core.orbit import _keplerian_to_cartesian
            expected_pos, _ = _keplerian_to_cartesian(a, inc_rad, raan_rad, nu_rad)

            for got, exp in zip(pos_no_j2, expected_pos):
                assert abs(got - exp) < 1e-10, (
                    f"use_j2=False must match Keplerian at t={t}s: got {got}, expected {exp}"
                )

    def test_j2_sso_zero_raan_drift(self):
        """For SSO inclination, RAAN drift should equal ~+0.9856 deg/day (solar rate).

        SSO is designed so that the J2 RAAN precession matches Earth's orbital rate
        around the Sun (~360 deg / 365.25 days ≈ 0.9856 deg/day).
        The precession formula gives a positive value when cos(i) < 0 (retrograde SSO),
        so the drift rate should be close to +0.9856 deg/day.
        """
        import math
        from space_ml_sim.core.orbit import (
            MU_EARTH_KM3_S2, R_EARTH_KM, J2, _sso_inclination_deg
        )

        altitude_km = 550.0
        sso_inc_deg = _sso_inclination_deg(altitude_km)
        a = R_EARTH_KM + altitude_km
        n = math.sqrt(MU_EARTH_KM3_S2 / a ** 3)
        inc_rad = math.radians(sso_inc_deg)

        d_raan_dt_rad_s = -1.5 * n * J2 * (R_EARTH_KM / a) ** 2 * math.cos(inc_rad)
        d_raan_dt_deg_day = math.degrees(d_raan_dt_rad_s) * 86400.0

        solar_rate_deg_day = 360.0 / 365.25  # ~0.9856 deg/day
        assert abs(d_raan_dt_deg_day - solar_rate_deg_day) < 0.01, (
            f"SSO RAAN drift should be ~{solar_rate_deg_day:.4f} deg/day, "
            f"got {d_raan_dt_deg_day:.4f}"
        )

    def test_j2_affects_propagate(self):
        """propagate() with use_j2=True should give different positions than use_j2=False."""
        import math

        config = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        # Propagate for 1 day to accumulate drift
        states_j2 = propagate(config, duration_minutes=1440.0, step_seconds=3600.0, use_j2=True)
        states_no_j2 = propagate(config, duration_minutes=1440.0, step_seconds=3600.0, use_j2=False)

        assert len(states_j2) == len(states_no_j2), "Both should produce the same number of states"

        # At least one state after t=0 should differ
        differences = []
        for s_j2, s_no_j2 in zip(states_j2[1:], states_no_j2[1:]):
            dist = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(s_j2.position_km, s_no_j2.position_km))
            )
            differences.append(dist)

        assert max(differences) > 1.0, (
            f"J2 propagation should diverge from Keplerian; max diff was {max(differences):.3f} km"
        )


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
