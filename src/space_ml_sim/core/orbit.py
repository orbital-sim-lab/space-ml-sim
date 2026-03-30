"""Keplerian orbit propagation for LEO satellites.

Uses simple two-body mechanics (no J2 perturbations in v0.1).
Accepts/returns plain floats with km/deg conventions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, Field

# Earth constants
MU_EARTH_KM3_S2 = 398600.4418  # km^3/s^2
R_EARTH_KM = 6371.0
J2 = 1.08263e-3  # For SSO inclination calculation only


class OrbitConfig(BaseModel):
    """Keplerian orbital elements for a satellite."""

    altitude_km: float = Field(gt=0, description="Altitude above Earth surface in km")
    inclination_deg: float = Field(ge=0, le=180, description="Orbital inclination in degrees")
    raan_deg: float = Field(
        ge=0, lt=360, description="Right ascension of ascending node in degrees"
    )
    true_anomaly_deg: float = Field(ge=0, lt=360, description="True anomaly in degrees")

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis = Earth radius + altitude."""
        return R_EARTH_KM + self.altitude_km

    @property
    def orbital_period_seconds(self) -> float:
        """Keplerian orbital period: T = 2*pi*sqrt(a^3/mu)."""
        a = self.semi_major_axis_km
        return 2 * math.pi * math.sqrt(a**3 / MU_EARTH_KM3_S2)

    @property
    def mean_motion_rad_per_sec(self) -> float:
        """Mean motion n = sqrt(mu/a^3)."""
        a = self.semi_major_axis_km
        return math.sqrt(MU_EARTH_KM3_S2 / a**3)


class OrbitalState(NamedTuple):
    """Propagated orbital state at a point in time."""

    time_seconds: float
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]


def _keplerian_to_cartesian(
    a: float, inc_rad: float, raan_rad: float, nu_rad: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert Keplerian elements to ECI position and velocity (circular orbit)."""
    r = a  # Circular orbit: r = a
    v_mag = math.sqrt(MU_EARTH_KM3_S2 / a)

    # Position in orbital plane
    x_orb = r * math.cos(nu_rad)
    y_orb = r * math.sin(nu_rad)

    # Velocity in orbital plane (circular)
    vx_orb = -v_mag * math.sin(nu_rad)
    vy_orb = v_mag * math.cos(nu_rad)

    # Rotation matrices: R3(-RAAN) * R1(-inc) * R3(-arg_periapsis=0 for circular)
    cos_raan = math.cos(raan_rad)
    sin_raan = math.sin(raan_rad)
    cos_inc = math.cos(inc_rad)
    sin_inc = math.sin(inc_rad)

    # ECI position
    x = cos_raan * x_orb - sin_raan * cos_inc * y_orb
    y = sin_raan * x_orb + cos_raan * cos_inc * y_orb
    z = sin_inc * y_orb

    # ECI velocity
    vx = cos_raan * vx_orb - sin_raan * cos_inc * vy_orb
    vy = sin_raan * vx_orb + cos_raan * cos_inc * vy_orb
    vz = sin_inc * vy_orb

    return (x, y, z), (vx, vy, vz)


def _j2_raan_drift(a: float, inc_rad: float, n: float) -> float:
    """Compute secular J2 drift rate for RAAN.

    For circular orbits, only RAAN drift is physically meaningful.
    Argument of perigee is undefined for circular orbits, so we
    do not include it (it would corrupt the trajectory).

    Args:
        a: Semi-major axis in km.
        inc_rad: Inclination in radians.
        n: Mean motion in rad/s.

    Returns:
        d_raan_dt in rad/s.
    """
    return -1.5 * n * J2 * (R_EARTH_KM / a) ** 2 * math.cos(inc_rad)


def position_at(
    orbit_config: OrbitConfig,
    time_seconds: float,
    use_j2: bool = True,
) -> tuple[float, float, float]:
    """Compute satellite ECI position at a given time from epoch.

    Uses Keplerian mean motion to advance the true anomaly from the
    initial value stored in orbit_config. Optionally applies secular
    J2 perturbations to RAAN and argument of perigee.

    Args:
        orbit_config: Orbital elements (true_anomaly_deg is the epoch value).
        time_seconds: Time elapsed since epoch in seconds.
        use_j2: If True (default), apply secular J2 RAAN and arg-perigee drift.

    Returns:
        (x, y, z) position in km (ECI frame).
    """
    a = orbit_config.semi_major_axis_km
    inc_rad = math.radians(orbit_config.inclination_deg)
    raan_rad = math.radians(orbit_config.raan_deg)
    nu0_rad = math.radians(orbit_config.true_anomaly_deg)
    n = orbit_config.mean_motion_rad_per_sec

    nu_rad = nu0_rad + n * time_seconds

    if use_j2:
        d_raan_dt = _j2_raan_drift(a, inc_rad, n)
        raan_rad = raan_rad + d_raan_dt * time_seconds

    pos, _ = _keplerian_to_cartesian(a, inc_rad, raan_rad, nu_rad)
    return pos


def propagate(
    orbit_config: OrbitConfig,
    start_time: float = 0.0,
    duration_minutes: float = 90.0,
    step_seconds: float = 60.0,
    use_j2: bool = True,
) -> list[OrbitalState]:
    """Propagate a circular Keplerian orbit over time.

    Args:
        orbit_config: Orbital elements.
        start_time: Start time offset in seconds.
        duration_minutes: Duration to propagate in minutes.
        step_seconds: Time step in seconds.
        use_j2: If True (default), apply secular J2 RAAN and arg-perigee drift.

    Returns:
        List of OrbitalState tuples (time, position_km, velocity_km_s).
    """
    a = orbit_config.semi_major_axis_km
    inc_rad = math.radians(orbit_config.inclination_deg)
    raan_rad_epoch = math.radians(orbit_config.raan_deg)
    nu0_rad = math.radians(orbit_config.true_anomaly_deg)
    n = orbit_config.mean_motion_rad_per_sec

    d_raan_dt = _j2_raan_drift(a, inc_rad, n) if use_j2 else 0.0

    duration_seconds = duration_minutes * 60.0
    num_steps = int(duration_seconds / step_seconds) + 1

    states: list[OrbitalState] = []
    for i in range(num_steps):
        t = start_time + i * step_seconds
        elapsed = t - start_time
        raan_rad = raan_rad_epoch + d_raan_dt * elapsed
        nu_rad = nu0_rad + n * elapsed
        pos, vel = _keplerian_to_cartesian(a, inc_rad, raan_rad, nu_rad)
        states.append(OrbitalState(time_seconds=t, position_km=pos, velocity_km_s=vel))

    return states


def walker_delta_orbits(
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    phasing: int = 1,
) -> list[OrbitConfig]:
    """Generate Walker-Delta constellation orbital elements.

    Args:
        num_planes: Number of orbital planes.
        sats_per_plane: Satellites per plane.
        altitude_km: Altitude in km.
        inclination_deg: Inclination in degrees.
        phasing: Walker phasing parameter F (0..num_planes-1).

    Returns:
        List of OrbitConfig for each satellite.
    """
    configs: list[OrbitConfig] = []
    total_sats = num_planes * sats_per_plane
    raan_spacing = 360.0 / num_planes
    in_plane_spacing = 360.0 / sats_per_plane
    phase_offset = 360.0 * phasing / total_sats

    for plane in range(num_planes):
        raan = plane * raan_spacing
        for sat in range(sats_per_plane):
            nu = (sat * in_plane_spacing + plane * phase_offset) % 360.0
            configs.append(
                OrbitConfig(
                    altitude_km=altitude_km,
                    inclination_deg=inclination_deg,
                    raan_deg=raan,
                    true_anomaly_deg=nu,
                )
            )

    return configs


def _sso_inclination_deg(altitude_km: float) -> float:
    """Compute sun-synchronous inclination for a given altitude.

    Uses the J2 nodal precession rate requirement:
    d(RAAN)/dt = -3/2 * n * J2 * (R_e/a)^2 * cos(i) = 360 deg / 365.25 days
    """
    a = R_EARTH_KM + altitude_km
    n = math.sqrt(MU_EARTH_KM3_S2 / a**3)  # rad/s
    rate_required = 2 * math.pi / (365.25 * 86400)  # rad/s for SSO
    cos_i = -rate_required / (1.5 * n * J2 * (R_EARTH_KM / a) ** 2)
    cos_i = max(-1.0, min(1.0, cos_i))  # Clamp for numerical safety
    return math.degrees(math.acos(cos_i))


def sun_synchronous_orbits(
    num_sats: int,
    altitude_km: float,
    ltan_hours: float = 6.0,
) -> list[OrbitConfig]:
    """Generate sun-synchronous orbit configs.

    Args:
        num_sats: Number of satellites.
        altitude_km: Altitude in km.
        ltan_hours: Local time of ascending node in hours (6.0 = dawn-dusk).

    Returns:
        List of OrbitConfig with SSO inclination and evenly-spaced true anomalies.
    """
    inc = _sso_inclination_deg(altitude_km)
    # RAAN derived from LTAN (simplified: LTAN=6 -> RAAN=90, LTAN=18 -> RAAN=270)
    raan = (ltan_hours / 24.0) * 360.0

    configs: list[OrbitConfig] = []
    anomaly_spacing = 360.0 / num_sats
    for i in range(num_sats):
        configs.append(
            OrbitConfig(
                altitude_km=altitude_km,
                inclination_deg=round(inc, 4),
                raan_deg=raan,
                true_anomaly_deg=i * anomaly_spacing,
            )
        )

    return configs


def is_in_eclipse(
    position_km: tuple[float, float, float], sun_direction: tuple[float, float, float]
) -> bool:
    """Check if a satellite is in Earth's shadow using cylindrical shadow model.

    Args:
        position_km: Satellite ECI position (x, y, z) in km.
        sun_direction: Unit vector pointing from Earth to Sun.

    Returns:
        True if the satellite is in Earth's cylindrical shadow.
    """
    pos = np.array(position_km)
    sun_dir = np.array(sun_direction)
    norm = np.linalg.norm(sun_dir)
    if norm == 0:
        return False  # No sun direction → assume sunlit
    sun_dir = sun_dir / norm

    # Project satellite position onto sun direction
    projection = float(np.dot(pos, sun_dir))

    # If satellite is on the sunlit side, not in eclipse
    if projection > 0:
        return False

    # Perpendicular distance from shadow axis
    perp = pos - projection * sun_dir
    perp_dist = float(np.linalg.norm(perp))

    return perp_dist < R_EARTH_KM
