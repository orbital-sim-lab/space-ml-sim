"""Shielding optimization recommender.

Given mission radiation constraints, recommends minimum shielding
thickness to meet TID and SEU requirements.

Aluminium density: 2.7 g/cm³ = 2700 kg/m³
"""

from __future__ import annotations

from dataclasses import dataclass

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment


_AL_DENSITY_KG_M3 = 2700.0


@dataclass(frozen=True)
class ShieldingResult:
    """Result of shielding optimization."""

    shielding_mm_al: float
    achieved_tid_krad: float
    achieved_seu_rate: float
    mass_penalty_kg_m2: float


def find_minimum_shielding(
    orbit: OrbitConfig,
    max_tid_krad: float,
    mission_years: float,
    max_shielding_mm: float = 20.0,
    step_mm: float = 0.5,
) -> ShieldingResult:
    """Find minimum shielding to meet TID constraint.

    Uses a linear search over shielding thickness.

    Args:
        orbit: Orbital configuration.
        max_tid_krad: Maximum allowed TID over mission in krad(Si).
        mission_years: Mission duration in years.
        max_shielding_mm: Maximum shielding to consider in mm Al.
        step_mm: Search step size in mm.

    Returns:
        ShieldingResult with the minimum shielding that meets the constraint.
    """
    mission_days = mission_years * 365.25
    shield = 0.0

    while shield <= max_shielding_mm:
        rad_env = RadiationEnvironment(
            altitude_km=orbit.altitude_km,
            inclination_deg=orbit.inclination_deg,
            shielding_mm_al=max(shield, 0.1),  # Avoid zero shielding
        )
        tid = rad_env.tid_rate_krad_per_day * mission_days

        if tid <= max_tid_krad:
            return ShieldingResult(
                shielding_mm_al=shield,
                achieved_tid_krad=tid,
                achieved_seu_rate=rad_env.base_seu_rate,
                mass_penalty_kg_m2=_mass_per_area(shield),
            )
        shield += step_mm

    # Max shielding still doesn't meet constraint — return best effort
    rad_env = RadiationEnvironment(
        altitude_km=orbit.altitude_km,
        inclination_deg=orbit.inclination_deg,
        shielding_mm_al=max_shielding_mm,
    )
    return ShieldingResult(
        shielding_mm_al=max_shielding_mm,
        achieved_tid_krad=rad_env.tid_rate_krad_per_day * mission_days,
        achieved_seu_rate=rad_env.base_seu_rate,
        mass_penalty_kg_m2=_mass_per_area(max_shielding_mm),
    )


def shielding_sweep(
    orbit: OrbitConfig,
    mission_years: float,
    shielding_range_mm: tuple[float, float] = (0.5, 10.0),
    steps: int = 10,
) -> list[ShieldingResult]:
    """Sweep shielding thickness and report TID/SEU at each level.

    Args:
        orbit: Orbital configuration.
        mission_years: Mission duration in years.
        shielding_range_mm: (min, max) shielding thickness in mm Al.
        steps: Number of sweep points.

    Returns:
        List of ShieldingResult, one per sweep point.
    """
    mission_days = mission_years * 365.25
    low, high = shielding_range_mm
    results: list[ShieldingResult] = []

    for i in range(steps):
        shield = low + (high - low) * i / max(steps - 1, 1)
        rad_env = RadiationEnvironment(
            altitude_km=orbit.altitude_km,
            inclination_deg=orbit.inclination_deg,
            shielding_mm_al=shield,
        )
        results.append(
            ShieldingResult(
                shielding_mm_al=shield,
                achieved_tid_krad=rad_env.tid_rate_krad_per_day * mission_days,
                achieved_seu_rate=rad_env.base_seu_rate,
                mass_penalty_kg_m2=_mass_per_area(shield),
            )
        )

    return results


def _mass_per_area(thickness_mm: float) -> float:
    """Mass per unit area for aluminium shielding in kg/m²."""
    return _AL_DENSITY_KG_M3 * (thickness_mm / 1000.0)
