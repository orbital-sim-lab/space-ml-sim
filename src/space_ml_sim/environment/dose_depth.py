"""Dose-depth curve analysis.

Generates TID vs aluminium shielding thickness curves — the standard
chart in every radiation hardness assurance report. Shows how dose
decreases exponentially with shielding depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment


@dataclass(frozen=True)
class DoseDepthCurve:
    """TID vs shielding depth data."""

    shielding_mm: tuple[float, ...]
    dose_krad: tuple[float, ...]
    orbit_altitude_km: float
    mission_years: float

    def to_dataframe(self) -> pd.DataFrame:
        """Export as DataFrame."""
        return pd.DataFrame({
            "shielding_mm_al": list(self.shielding_mm),
            "dose_krad": list(self.dose_krad),
        })


def generate_dose_depth_curve(
    orbit: OrbitConfig,
    mission_years: float,
    shielding_range_mm: tuple[float, float] = (0.5, 20.0),
    num_points: int = 20,
) -> DoseDepthCurve:
    """Generate TID vs shielding depth curve.

    Args:
        orbit: Orbital configuration.
        mission_years: Mission duration in years.
        shielding_range_mm: (min, max) shielding in mm Al.
        num_points: Number of data points.

    Returns:
        DoseDepthCurve with shielding and dose arrays.
    """
    mission_days = mission_years * 365.25
    low, high = shielding_range_mm

    shieldings: list[float] = []
    doses: list[float] = []

    for i in range(num_points):
        shield = low + (high - low) * i / max(num_points - 1, 1)
        rad_env = RadiationEnvironment(
            altitude_km=orbit.altitude_km,
            inclination_deg=orbit.inclination_deg,
            shielding_mm_al=shield,
        )
        dose = rad_env.tid_rate_krad_per_day * mission_days
        shieldings.append(round(shield, 2))
        doses.append(round(dose, 4))

    return DoseDepthCurve(
        shielding_mm=tuple(shieldings),
        dose_krad=tuple(doses),
        orbit_altitude_km=orbit.altitude_km,
        mission_years=mission_years,
    )


def find_shielding_for_dose(
    orbit: OrbitConfig,
    mission_years: float,
    target_dose_krad: float,
    max_shielding_mm: float = 30.0,
    precision_mm: float = 0.1,
) -> float:
    """Find minimum shielding thickness to achieve a target dose limit.

    Uses binary search over shielding thickness.

    Args:
        orbit: Orbital configuration.
        mission_years: Mission duration in years.
        target_dose_krad: Maximum acceptable TID in krad.
        max_shielding_mm: Upper bound for search in mm Al.
        precision_mm: Search precision in mm.

    Returns:
        Required shielding thickness in mm Al.
    """
    mission_days = mission_years * 365.25
    low = 0.1
    high = max_shielding_mm

    while high - low > precision_mm:
        mid = (low + high) / 2
        rad_env = RadiationEnvironment(
            altitude_km=orbit.altitude_km,
            inclination_deg=orbit.inclination_deg,
            shielding_mm_al=mid,
        )
        dose = rad_env.tid_rate_krad_per_day * mission_days

        if dose > target_dose_krad:
            low = mid
        else:
            high = mid

    return round(high, 1)
