"""TDD tests for ground station visibility and downlink scheduling.

Written FIRST before implementation (RED phase).

Ground station features:
- Elevation-angle-based visibility windows
- Contact duration calculation
- Downlink data budget (bandwidth * contact time)
- Multi-station scheduling with handover
"""

from __future__ import annotations

import math

import pytest

from space_ml_sim.core.orbit import OrbitConfig, R_EARTH_KM


# ---------------------------------------------------------------------------
# Test: GroundStation construction
# ---------------------------------------------------------------------------


class TestGroundStationConstruction:
    """GroundStation must accept standard geodetic coordinates."""

    def test_creates_station(self) -> None:
        from space_ml_sim.environment.ground_station import GroundStation

        gs = GroundStation(
            name="Svalbard",
            latitude_deg=78.23,
            longitude_deg=15.39,
            min_elevation_deg=5.0,
        )
        assert gs.name == "Svalbard"
        assert gs.latitude_deg == 78.23

    def test_preset_stations(self) -> None:
        from space_ml_sim.environment.ground_station import GROUND_STATION_PRESETS

        assert "svalbard" in GROUND_STATION_PRESETS
        assert "mcmurdo" in GROUND_STATION_PRESETS
        assert len(GROUND_STATION_PRESETS) >= 3


class TestVisibilityCheck:
    """Visibility must be based on elevation angle from ground station."""

    def test_overhead_satellite_is_visible(self) -> None:
        """A satellite directly above a station at 500km is visible."""
        from space_ml_sim.environment.ground_station import GroundStation

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        # Satellite directly above (on x-axis at 500km altitude)
        sat_pos = (R_EARTH_KM + 500.0, 0.0, 0.0)
        assert gs.is_visible(sat_pos) is True

    def test_opposite_side_not_visible(self) -> None:
        """A satellite on the opposite side of Earth is not visible."""
        from space_ml_sim.environment.ground_station import GroundStation

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        sat_pos = (-(R_EARTH_KM + 500.0), 0.0, 0.0)
        assert gs.is_visible(sat_pos) is False

    def test_elevation_angle_computation(self) -> None:
        """Elevation angle must be computable."""
        from space_ml_sim.environment.ground_station import GroundStation

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        sat_pos = (R_EARTH_KM + 500.0, 0.0, 0.0)
        elev = gs.elevation_deg(sat_pos)
        assert elev == 90.0  # directly overhead


class TestContactWindows:
    """Contact window computation over an orbit pass."""

    def test_find_contacts_returns_list(self) -> None:
        from space_ml_sim.environment.ground_station import (
            GroundStation,
            find_contact_windows,
        )

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        orbit = OrbitConfig(
            altitude_km=500,
            inclination_deg=51.6,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        windows = find_contact_windows(
            orbit=orbit,
            station=gs,
            duration_seconds=6000.0,  # ~1 orbit
            step_seconds=10.0,
        )
        assert isinstance(windows, list)

    def test_contact_window_has_start_end_duration(self) -> None:
        from space_ml_sim.environment.ground_station import (
            GroundStation,
            ContactWindow,
            find_contact_windows,
        )

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        orbit = OrbitConfig(
            altitude_km=500,
            inclination_deg=51.6,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        windows = find_contact_windows(
            orbit=orbit,
            station=gs,
            duration_seconds=6000.0,
            step_seconds=10.0,
        )
        if windows:
            w = windows[0]
            assert isinstance(w, ContactWindow)
            assert w.start_seconds >= 0
            assert w.end_seconds > w.start_seconds
            assert w.duration_seconds > 0

    def test_contact_duration_is_reasonable(self) -> None:
        """LEO pass over a station should be 2-15 minutes typically."""
        from space_ml_sim.environment.ground_station import (
            GroundStation,
            find_contact_windows,
        )

        gs = GroundStation(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            min_elevation_deg=5.0,
        )
        orbit = OrbitConfig(
            altitude_km=500,
            inclination_deg=51.6,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        windows = find_contact_windows(
            orbit=orbit,
            station=gs,
            duration_seconds=6000.0,
            step_seconds=10.0,
        )
        for w in windows:
            # Typical LEO pass: 2-15 minutes
            assert 30 < w.duration_seconds < 1200, (
                f"Contact duration {w.duration_seconds}s outside expected range"
            )


class TestDownlinkBudget:
    """Downlink budget must account for contact time and bandwidth."""

    def test_downlink_bytes(self) -> None:
        from space_ml_sim.environment.ground_station import ContactWindow

        w = ContactWindow(
            start_seconds=0.0,
            end_seconds=600.0,  # 10 minutes
            max_elevation_deg=45.0,
            station_name="Test",
        )
        # At 1 Gbps for 600 seconds
        downlink = w.downlink_bytes(bandwidth_gbps=1.0)
        expected = 1e9 / 8 * 600  # bits/sec -> bytes/sec * time
        assert abs(downlink - expected) < 1.0

    def test_higher_bandwidth_more_data(self) -> None:
        from space_ml_sim.environment.ground_station import ContactWindow

        w = ContactWindow(
            start_seconds=0.0,
            end_seconds=600.0,
            max_elevation_deg=45.0,
            station_name="Test",
        )
        d1 = w.downlink_bytes(bandwidth_gbps=1.0)
        d10 = w.downlink_bytes(bandwidth_gbps=10.0)
        assert d10 == d1 * 10
