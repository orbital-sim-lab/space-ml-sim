"""Ground station visibility and downlink scheduling.

Models elevation-angle-based visibility windows between LEO satellites
and ground stations. Computes contact duration and downlink data budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from space_ml_sim.core.orbit import OrbitConfig, R_EARTH_KM, position_at


@dataclass(frozen=True)
class GroundStation:
    """A ground station at a fixed geodetic location.

    Uses a simplified spherical Earth model (no oblateness).
    """

    name: str
    latitude_deg: float  # -90 to 90
    longitude_deg: float  # -180 to 180
    min_elevation_deg: float = 5.0  # Minimum elevation for contact

    @property
    def ecef_position(self) -> tuple[float, float, float]:
        """Station position in ECEF (approximated as ECI for simplicity).

        For simulation purposes, we treat longitude as the ECI angle
        at epoch. This is a simplification — a full model would account
        for Earth rotation.
        """
        lat = math.radians(self.latitude_deg)
        lon = math.radians(self.longitude_deg)
        x = R_EARTH_KM * math.cos(lat) * math.cos(lon)
        y = R_EARTH_KM * math.cos(lat) * math.sin(lon)
        z = R_EARTH_KM * math.sin(lat)
        return (x, y, z)

    def elevation_deg(self, sat_pos: tuple[float, float, float]) -> float:
        """Compute elevation angle of a satellite from this station.

        Args:
            sat_pos: Satellite ECI position (x, y, z) in km.

        Returns:
            Elevation angle in degrees. Negative means below horizon.
        """
        gs = self.ecef_position

        # Vector from ground station to satellite
        dx = sat_pos[0] - gs[0]
        dy = sat_pos[1] - gs[1]
        dz = sat_pos[2] - gs[2]

        range_km = math.sqrt(dx**2 + dy**2 + dz**2)
        if range_km < 1e-6:
            return 90.0

        # Ground station normal (radial direction from Earth center)
        gs_mag = math.sqrt(gs[0] ** 2 + gs[1] ** 2 + gs[2] ** 2)
        nx = gs[0] / gs_mag
        ny = gs[1] / gs_mag
        nz = gs[2] / gs_mag

        # Dot product of range vector with station normal
        dot = (dx * nx + dy * ny + dz * nz) / range_km

        # Elevation = 90 - zenith angle
        # zenith_angle = acos(dot)
        # elevation = 90 - zenith_angle
        dot = max(-1.0, min(1.0, dot))
        zenith_rad = math.acos(dot)
        elevation_deg = 90.0 - math.degrees(zenith_rad)

        return elevation_deg

    def is_visible(self, sat_pos: tuple[float, float, float]) -> bool:
        """Check if a satellite is visible above minimum elevation.

        Args:
            sat_pos: Satellite ECI position (x, y, z) in km.

        Returns:
            True if satellite elevation exceeds min_elevation_deg.
        """
        return self.elevation_deg(sat_pos) >= self.min_elevation_deg


@dataclass(frozen=True)
class ContactWindow:
    """A single contact window between a satellite and ground station."""

    start_seconds: float
    end_seconds: float
    max_elevation_deg: float
    station_name: str

    @property
    def duration_seconds(self) -> float:
        """Contact duration in seconds."""
        return self.end_seconds - self.start_seconds

    def downlink_bytes(self, bandwidth_gbps: float) -> float:
        """Total data that can be downlinked during this window.

        Args:
            bandwidth_gbps: Downlink bandwidth in Gbps.

        Returns:
            Data volume in bytes.
        """
        bits_per_second = bandwidth_gbps * 1e9
        bytes_per_second = bits_per_second / 8
        return bytes_per_second * self.duration_seconds


def find_contact_windows(
    orbit: OrbitConfig,
    station: GroundStation,
    duration_seconds: float,
    step_seconds: float = 10.0,
) -> list[ContactWindow]:
    """Find all contact windows over a time period.

    Steps through the orbit and identifies continuous intervals where
    the satellite is visible from the ground station.

    Args:
        orbit: Satellite orbital configuration.
        station: Ground station to check visibility against.
        duration_seconds: Total simulation duration in seconds.
        step_seconds: Time step for visibility sampling.

    Returns:
        List of ContactWindow objects, sorted by start time.
    """
    windows: list[ContactWindow] = []
    in_contact = False
    contact_start = 0.0
    max_elev = 0.0

    t = 0.0
    while t <= duration_seconds:
        pos = position_at(orbit, t)
        visible = station.is_visible(pos)

        if visible and not in_contact:
            in_contact = True
            contact_start = t
            max_elev = station.elevation_deg(pos)
        elif visible and in_contact:
            elev = station.elevation_deg(pos)
            max_elev = max(max_elev, elev)
        elif not visible and in_contact:
            in_contact = False
            windows.append(
                ContactWindow(
                    start_seconds=contact_start,
                    end_seconds=t,
                    max_elevation_deg=max_elev,
                    station_name=station.name,
                )
            )

        t += step_seconds

    # Close any open window at end of simulation
    if in_contact:
        windows.append(
            ContactWindow(
                start_seconds=contact_start,
                end_seconds=duration_seconds,
                max_elevation_deg=max_elev,
                station_name=station.name,
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Preset ground stations
# ---------------------------------------------------------------------------

GROUND_STATION_PRESETS: dict[str, GroundStation] = {
    "svalbard": GroundStation(
        name="Svalbard SvalSat",
        latitude_deg=78.23,
        longitude_deg=15.39,
        min_elevation_deg=5.0,
    ),
    "mcmurdo": GroundStation(
        name="McMurdo Station",
        latitude_deg=-77.85,
        longitude_deg=166.67,
        min_elevation_deg=5.0,
    ),
    "fairbanks": GroundStation(
        name="Fairbanks NOAA",
        latitude_deg=64.97,
        longitude_deg=-147.52,
        min_elevation_deg=5.0,
    ),
    "troll": GroundStation(
        name="Troll Satellite Station",
        latitude_deg=-72.01,
        longitude_deg=2.53,
        min_elevation_deg=5.0,
    ),
    "singapore": GroundStation(
        name="Singapore NUS",
        latitude_deg=1.30,
        longitude_deg=103.77,
        min_elevation_deg=10.0,
    ),
}
