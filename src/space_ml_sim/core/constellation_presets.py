"""Pre-built constellation configurations for popular constellations.

Data from FCC filings and ITU filings as of 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

from space_ml_sim.core.orbit import OrbitConfig, walker_delta_orbits


@dataclass(frozen=True)
class ConstellationPreset:
    """Configuration for a known constellation."""

    name: str
    operator: str
    num_satellites: int
    num_planes: int
    altitude_km: float
    inclination_deg: float
    description: str


CONSTELLATION_PRESETS: dict[str, ConstellationPreset] = {
    "starlink_shell1": ConstellationPreset(
        name="Starlink Shell 1",
        operator="SpaceX",
        num_satellites=1584,
        num_planes=72,
        altitude_km=550,
        inclination_deg=53.0,
        description="Primary Starlink shell — 22 sats per plane",
    ),
    "starlink_shell4": ConstellationPreset(
        name="Starlink Shell 4 (polar)",
        operator="SpaceX",
        num_satellites=348,
        num_planes=6,
        altitude_km=560,
        inclination_deg=97.6,
        description="Starlink polar shell — 58 sats per plane, SSO",
    ),
    "oneweb": ConstellationPreset(
        name="OneWeb Gen 1",
        operator="OneWeb",
        num_satellites=648,
        num_planes=18,
        altitude_km=1200,
        inclination_deg=87.9,
        description="Near-polar LEO broadband — 36 sats per plane",
    ),
    "kuiper_shell1": ConstellationPreset(
        name="Kuiper Shell 1",
        operator="Amazon",
        num_satellites=784,
        num_planes=28,
        altitude_km=590,
        inclination_deg=33.0,
        description="Amazon Kuiper low-inclination shell — 28 sats per plane",
    ),
    "kuiper_shell2": ConstellationPreset(
        name="Kuiper Shell 2",
        operator="Amazon",
        num_satellites=1296,
        num_planes=36,
        altitude_km=610,
        inclination_deg=42.0,
        description="Amazon Kuiper mid-inclination shell — 36 sats per plane",
    ),
    "iridium_next": ConstellationPreset(
        name="Iridium NEXT",
        operator="Iridium",
        num_satellites=66,
        num_planes=6,
        altitude_km=780,
        inclination_deg=86.4,
        description="Iridium NEXT — 11 sats per plane, near-polar",
    ),
    "planet_flock": ConstellationPreset(
        name="Planet Flock (SSO)",
        operator="Planet Labs",
        num_satellites=130,
        num_planes=1,
        altitude_km=475,
        inclination_deg=97.4,
        description="Planet Dove imaging constellation — single SSO plane",
    ),
}


def generate_from_preset(preset_name: str) -> list[OrbitConfig]:
    """Generate Walker-Delta orbits from a constellation preset.

    Args:
        preset_name: Key in CONSTELLATION_PRESETS dict.

    Returns:
        List of OrbitConfig, one per satellite.

    Raises:
        KeyError: If preset_name is not found.
    """
    preset = CONSTELLATION_PRESETS[preset_name]
    sats_per_plane = preset.num_satellites // preset.num_planes

    return walker_delta_orbits(
        num_planes=preset.num_planes,
        sats_per_plane=sats_per_plane,
        altitude_km=preset.altitude_km,
        inclination_deg=preset.inclination_deg,
    )
