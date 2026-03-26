"""Tests for constellation propagation — verifies satellites actually move."""

import math


from space_ml_sim.core.constellation import Constellation
from space_ml_sim.models.chip_profiles import TERAFAB_D3


class TestConstellationStep:
    """Verify the critical bug fix: satellites must move when step() is called."""

    def test_satellites_move_after_step(self):
        """After stepping, satellite positions must differ from initial."""
        constellation = Constellation.walker_delta(
            num_planes=2,
            sats_per_plane=2,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        initial_positions = [sat.position_km for sat in constellation.satellites]

        # Step 10 minutes (600 seconds) in 60s increments
        for _ in range(10):
            constellation.step(dt_seconds=60.0)

        final_positions = [sat.position_km for sat in constellation.satellites]

        # Every satellite should have moved
        for i, (init, final) in enumerate(zip(initial_positions, final_positions)):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(init, final)))
            assert dist > 10, f"Satellite {i} didn't move: init={init}, final={final}"

    def test_positions_change_between_consecutive_steps(self):
        """THE BUG TEST: Positions must change between step 5 and step 10.

        The bug was that propagate() was called with duration_minutes=0,
        always returning the initial true anomaly position. This test catches
        that by comparing positions across two different time windows.
        """
        constellation = Constellation.walker_delta(
            num_planes=1,
            sats_per_plane=1,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        # Run 5 steps to get past initial state
        for _ in range(5):
            constellation.step(dt_seconds=60.0)
        pos_at_5min = constellation.satellites[0].position_km

        # Run 5 more steps
        for _ in range(5):
            constellation.step(dt_seconds=60.0)
        pos_at_10min = constellation.satellites[0].position_km

        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_at_5min, pos_at_10min)))
        assert dist > 10, (
            f"Satellite position didn't change between step 5 and step 10! "
            f"dist={dist:.4f}km. This means orbit propagation is broken."
        )

    def test_positions_differ_between_satellites(self):
        """Different satellites in different planes should have different positions."""
        constellation = Constellation.walker_delta(
            num_planes=2,
            sats_per_plane=2,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        constellation.step(dt_seconds=60.0)

        positions = [sat.position_km for sat in constellation.satellites]
        # At least some positions should differ (different orbital planes)
        unique_positions = set(positions)
        assert len(unique_positions) > 1

    def test_step_returns_valid_metrics(self):
        """step() should return a metrics dict with all expected keys."""
        constellation = Constellation.walker_delta(
            num_planes=2,
            sats_per_plane=2,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        metrics = constellation.step(dt_seconds=60.0)

        assert "sim_time" in metrics
        assert "active_count" in metrics
        assert "degraded_count" in metrics
        assert "failed_count" in metrics
        assert "avg_temperature_c" in metrics
        assert "total_seus" in metrics
        assert metrics["sim_time"] == 60.0
        assert metrics["active_count"] + metrics["degraded_count"] + metrics["failed_count"] == 4

    def test_multiple_steps_accumulate_time(self):
        """sim_time should accumulate across multiple steps."""
        constellation = Constellation.walker_delta(
            num_planes=1,
            sats_per_plane=2,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        constellation.step(dt_seconds=30.0)
        metrics = constellation.step(dt_seconds=30.0)
        assert metrics["sim_time"] == 60.0

    def test_satellite_returns_near_start_after_full_orbit(self):
        """After one orbital period, satellite should be near starting position."""
        constellation = Constellation.walker_delta(
            num_planes=1,
            sats_per_plane=1,
            altitude_km=550,
            inclination_deg=53,
            chip_profile=TERAFAB_D3,
        )
        # Get initial position after first step
        constellation.step(dt_seconds=60.0)
        pos_after_1min = constellation.satellites[0].position_km

        # Step through remaining orbit (~95 min - 1 min = 94 min)
        period = constellation.satellites[0].orbit_config.orbital_period_seconds
        remaining_steps = int((period - 60) / 60)
        for _ in range(remaining_steps):
            constellation.step(dt_seconds=60.0)

        pos_after_orbit = constellation.satellites[0].position_km
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_after_1min, pos_after_orbit)))
        # Tolerance is large because discrete 60s steps don't land exactly on the period.
        # At ~7.5 km/s orbital velocity, a 60s step error = ~450km position error.
        assert dist < 1000, (
            f"Satellite didn't return near start after full orbit: dist={dist:.1f}km"
        )
