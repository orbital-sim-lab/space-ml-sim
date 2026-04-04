"""Adapter for importing orbits from poliastro.

Converts poliastro Orbit objects to our OrbitConfig.
poliastro is an optional dependency — install with:
    pip install space-ml-sim[poliastro]

For users without poliastro, `from_elements()` provides
a direct way to create OrbitConfig from orbital parameters.
"""

from __future__ import annotations


from space_ml_sim.core.orbit import OrbitConfig, R_EARTH_KM


def from_elements(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float = 0.0,
    true_anomaly_deg: float = 0.0,
) -> OrbitConfig:
    """Create an OrbitConfig from orbital elements directly.

    This is the poliastro-free path for creating orbits from known parameters.

    Args:
        altitude_km: Altitude above Earth surface in km.
        inclination_deg: Orbital inclination in degrees.
        raan_deg: Right ascension of ascending node in degrees.
        true_anomaly_deg: True anomaly in degrees.

    Returns:
        OrbitConfig with the given elements.
    """
    return OrbitConfig(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg % 360.0,
        true_anomaly_deg=true_anomaly_deg % 360.0,
    )


def from_poliastro(orbit: object) -> OrbitConfig:
    """Convert a poliastro Orbit object to OrbitConfig.

    Extracts classical orbital elements from the poliastro Orbit
    and maps them to our simplified circular-orbit model.

    Args:
        orbit: A poliastro.twobody.Orbit instance.

    Returns:
        OrbitConfig with elements extracted from the poliastro orbit.

    Raises:
        ImportError: If poliastro/astropy are not installed.
        TypeError: If the input is not a poliastro Orbit.
    """
    try:
        from astropy import units as u
        from poliastro.twobody import Orbit as PoliastroOrbit
    except ImportError as exc:
        raise ImportError(
            "poliastro is required for this function. "
            "Install with: pip install space-ml-sim[poliastro]"
        ) from exc

    if not isinstance(orbit, PoliastroOrbit):
        raise TypeError(f"Expected poliastro Orbit, got {type(orbit).__name__}")

    # Extract semi-major axis and convert to altitude
    a_km = orbit.a.to(u.km).value
    altitude_km = a_km - R_EARTH_KM

    # Extract angular elements in degrees
    inc_deg = orbit.inc.to(u.deg).value
    raan_deg = orbit.raan.to(u.deg).value
    nu_deg = orbit.nu.to(u.deg).value

    return OrbitConfig(
        altitude_km=altitude_km,
        inclination_deg=inc_deg,
        raan_deg=raan_deg % 360.0,
        true_anomaly_deg=nu_deg % 360.0,
    )
