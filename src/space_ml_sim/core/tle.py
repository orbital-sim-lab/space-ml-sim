"""TLE (Two-Line Element) ingestion and sgp4-based propagation.

Provides three public functions:
  - parse_tle(line1, line2) -> OrbitConfig
  - load_tle_file(path) -> list[OrbitConfig]
  - propagate_sgp4(line1, line2, minutes_from_epoch) -> tuple[float, float, float]

Uses the modern sgp4.api (Satrec / WGS72) interface.
"""

from __future__ import annotations

import math

from space_ml_sim.core.orbit import MU_EARTH_KM3_S2, R_EARTH_KM, OrbitConfig

# sgp4 constants: 1 revolution = 2*pi rad, 1 day = 86400 seconds.
_TWOPI = 2.0 * math.pi
_SEC_PER_DAY = 86400.0
_MIN_PER_DAY = 1440.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_anomaly_to_true_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    """Convert mean anomaly M to true anomaly nu using Kepler's equation.

    Iterates Newton's method to solve M = E - e*sin(E), then converts
    eccentric anomaly E to true anomaly nu.

    Args:
        mean_anomaly_rad: Mean anomaly in radians (any value; will be normalised).
        eccentricity: Orbital eccentricity [0, 1).

    Returns:
        True anomaly in radians, normalised to [0, 2*pi).
    """
    # Normalise M to [0, 2*pi)
    m = mean_anomaly_rad % _TWOPI

    # Initial guess for eccentric anomaly
    e_anom = m

    # Newton-Raphson iteration (converges in <10 steps for typical LEO eccentricities)
    for _ in range(50):
        delta = (m - (e_anom - eccentricity * math.sin(e_anom))) / (
            1.0 - eccentricity * math.cos(e_anom)
        )
        e_anom += delta
        if abs(delta) < 1e-12:
            break

    # Eccentric anomaly -> true anomaly
    cos_nu = (math.cos(e_anom) - eccentricity) / (1.0 - eccentricity * math.cos(e_anom))
    sin_nu = (math.sqrt(1.0 - eccentricity**2) * math.sin(e_anom)) / (
        1.0 - eccentricity * math.cos(e_anom)
    )
    nu_rad = math.atan2(sin_nu, cos_nu) % _TWOPI
    return nu_rad


def _build_satrec(line1: str, line2: str):
    """Create a sgp4 Satrec from TLE lines, raising ValueError on failure.

    Args:
        line1: TLE line 1 (must start with '1').
        line2: TLE line 2 (must start with '2').

    Returns:
        sgp4.api.Satrec object.

    Raises:
        ValueError: If lines are empty, malformed, or sgp4 initialisation fails.
    """
    from sgp4.api import Satrec, WGS72  # noqa: PLC0415

    line1 = line1.strip()
    line2 = line2.strip()

    if not line1 or not line2:
        raise ValueError("TLE lines must not be empty.")

    if not line1.startswith("1"):
        raise ValueError(f"TLE line 1 must start with '1', got: {line1[:10]!r}")
    if not line2.startswith("2"):
        raise ValueError(f"TLE line 2 must start with '2', got: {line2[:10]!r}")

    try:
        sat = Satrec.twoline2rv(line1, line2, WGS72)
    except Exception as exc:
        raise ValueError(f"Failed to parse TLE: {exc}") from exc

    # sgp4 error code 0 means success; non-zero indicates initialisation failure.
    if getattr(sat, "error", 0) != 0:
        raise ValueError(f"sgp4 initialisation returned error code {sat.error} for TLE: {line1!r}")

    return sat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_tle(line1: str, line2: str) -> OrbitConfig:
    """Parse a TLE two-line element set and convert to OrbitConfig.

    Extracts Keplerian elements from the TLE via sgp4's Satrec object and
    maps them to OrbitConfig fields:
      - altitude_km: derived from mean motion (n) -> semi-major axis -> altitude
      - inclination_deg: directly from TLE
      - raan_deg: directly from TLE
      - true_anomaly_deg: mean anomaly converted via Kepler's equation

    Args:
        line1: TLE line 1 string.
        line2: TLE line 2 string.

    Returns:
        OrbitConfig populated with the satellite's orbital elements.

    Raises:
        ValueError: If the TLE is invalid or cannot be parsed.
    """
    sat = _build_satrec(line1, line2)

    # sgp4's Satrec stores Kozai mean motion in rad/min as `no_kozai`.
    # Older builds may expose it as `no` (also rad/min).
    # Convert to rad/s by dividing by 60.
    n_rad_per_min: float = getattr(sat, "no_kozai", None) or getattr(sat, "no", 0.0)
    if n_rad_per_min == 0.0:
        raise ValueError("Could not retrieve mean motion from Satrec (no_kozai / no both zero).")

    n_rad_per_sec = n_rad_per_min / 60.0

    if n_rad_per_sec <= 0.0:
        raise ValueError(
            f"Computed mean motion is non-positive ({n_rad_per_sec}); TLE may be invalid."
        )

    # Semi-major axis from mean motion: n = sqrt(mu / a^3) => a = (mu / n^2)^(1/3)
    semi_major_axis_km = (MU_EARTH_KM3_S2 / n_rad_per_sec**2) ** (1.0 / 3.0)
    altitude_km = semi_major_axis_km - R_EARTH_KM

    if altitude_km <= 0.0:
        raise ValueError(
            f"Derived altitude {altitude_km:.1f} km is non-positive; TLE may be invalid."
        )

    # Inclination (deg) — stored as radians in Satrec; convert.
    inclination_deg = math.degrees(sat.inclo)

    # RAAN (deg)
    raan_deg = math.degrees(sat.nodeo) % 360.0

    # Eccentricity — stored as a decimal fraction
    eccentricity: float = sat.ecco

    # Mean anomaly (rad)
    mean_anomaly_rad: float = sat.mo

    # Convert mean anomaly to true anomaly
    true_anomaly_rad = _mean_anomaly_to_true_anomaly(mean_anomaly_rad, eccentricity)
    true_anomaly_deg = math.degrees(true_anomaly_rad) % 360.0

    return OrbitConfig(
        altitude_km=round(altitude_km, 4),
        inclination_deg=round(inclination_deg, 4),
        raan_deg=round(raan_deg, 4),
        true_anomaly_deg=round(true_anomaly_deg, 4),
    )


def load_tle_file(path: str) -> list[OrbitConfig]:
    """Read a standard 3-line TLE file and return a list of OrbitConfigs.

    The file format is the widely-used CelesTrak/Space-Track format:
        <name>
        <line 1>
        <line 2>
        [blank lines ignored]

    Args:
        path: Filesystem path to the TLE file.

    Returns:
        List of OrbitConfig, one per TLE entry.  Empty list for an empty file.

    Raises:
        OSError / FileNotFoundError: If the file does not exist or cannot be read.
        ValueError: If a TLE entry within the file is malformed.
    """
    with open(path) as fh:
        raw_lines = fh.readlines()

    # Strip and drop blank lines
    lines = [ln.rstrip("\n").rstrip() for ln in raw_lines]
    lines = [ln for ln in lines if ln]

    configs: list[OrbitConfig] = []

    # Process in groups of 3: name, line1, line2
    i = 0
    while i + 2 < len(lines):
        _name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        configs.append(parse_tle(l1, l2))
        i += 3

    return configs


def propagate_sgp4(
    line1: str,
    line2: str,
    minutes_from_epoch: float,
) -> tuple[float, float, float]:
    """Propagate a TLE using sgp4 for high-fidelity position prediction.

    Uses the sgp4 library directly (not the Keplerian approximation), providing
    J2, J3, J4, drag, and solar radiation pressure perturbations.

    The returned coordinates are in the TEME (True Equator, Mean Equinox) frame,
    which is close enough to ECI for most simulation purposes.

    Args:
        line1: TLE line 1 string.
        line2: TLE line 2 string.
        minutes_from_epoch: Minutes elapsed since the TLE epoch (may be negative).

    Returns:
        (x, y, z) position in km in the TEME frame.

    Raises:
        ValueError: If the TLE is invalid or sgp4 propagation fails.
    """
    sat = _build_satrec(line1, line2)

    # Compute target Julian date from epoch + offset.
    # sat.jdsatepoch is the integer part, sat.jdsatepochF is the fractional part.
    jd_epoch: float = sat.jdsatepoch
    jd_epoch_f: float = sat.jdsatepochF

    minutes_per_day = _MIN_PER_DAY
    day_offset = minutes_from_epoch / minutes_per_day

    # Add the offset to the fractional part to avoid precision loss.
    jd = jd_epoch
    fr = jd_epoch_f + day_offset

    try:
        e, r, _v = sat.sgp4(jd, fr)
    except Exception as exc:
        raise ValueError(f"sgp4 propagation failed: {exc}") from exc

    if e != 0:
        raise ValueError(f"sgp4 propagation returned error code {e} at t={minutes_from_epoch} min.")

    return (float(r[0]), float(r[1]), float(r[2]))
