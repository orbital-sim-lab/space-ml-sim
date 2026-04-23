"""Orbital thermal cycling model.

Models temperature variation during eclipse/sunlit transitions for LEO
satellites. Uses a simplified single-node thermal model with solar flux
heating and radiative cooling.

Typical LEO thermal cycle: ~-40°C in eclipse to ~+60°C in sunlit,
depending on surface properties, attitude, and thermal design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


from space_ml_sim.core.orbit import OrbitConfig, position_at, is_in_eclipse


# Physical constants
_SOLAR_FLUX_W_M2 = 1361.0  # Solar constant at 1 AU
_STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴
_EARTH_IR_W_M2 = 237.0  # Earth IR emission


@dataclass(frozen=True)
class ThermalProfile:
    """Time-series thermal profile over an orbital segment."""

    times_seconds: tuple[float, ...]
    temperatures_c: tuple[float, ...]
    eclipse_flags: tuple[bool, ...]
    num_cycles: int


def generate_thermal_profile(
    orbit: OrbitConfig,
    duration_seconds: float,
    step_seconds: float = 30.0,
    absorptivity: float = 0.3,
    emissivity: float = 0.8,
    area_m2: float = 0.1,
    mass_kg: float = 5.0,
    specific_heat_j_kg_k: float = 900.0,
    internal_dissipation_w: float = 5.0,
    initial_temp_c: float = 20.0,
) -> ThermalProfile:
    """Generate thermal profile over orbital duration.

    Uses a lumped-parameter thermal model:
        m*Cp*dT/dt = Q_solar + Q_earthIR + Q_internal - Q_radiated

    Args:
        orbit: Orbital configuration.
        duration_seconds: Simulation duration in seconds.
        step_seconds: Time step for integration.
        absorptivity: Solar absorptivity (0-1).
        emissivity: IR emissivity (0-1).
        area_m2: Effective radiating/absorbing area in m².
        mass_kg: Component thermal mass in kg.
        specific_heat_j_kg_k: Specific heat capacity in J/kg/K.
        internal_dissipation_w: Internal power dissipation in W.
        initial_temp_c: Initial temperature in °C.

    Returns:
        ThermalProfile with time-series data.
    """
    thermal_capacity = mass_kg * specific_heat_j_kg_k
    temp_k = initial_temp_c + 273.15

    times: list[float] = []
    temps_c: list[float] = []
    eclipses: list[bool] = []

    t = 0.0
    prev_eclipse = False
    cycles = 0

    while t <= duration_seconds:
        pos = position_at(orbit, t)
        # Approximate sun direction: rotates ~1 deg/day, simplified as fixed for short sims
        # Sun at +x direction at epoch, rotating at ~0.0172 rad/day
        sun_angle = 0.0172 * (t / 86400.0)
        sun_dir = (math.cos(sun_angle), math.sin(sun_angle), 0.0)
        in_eclipse = is_in_eclipse(pos, sun_dir)

        # Count thermal cycles (eclipse -> sunlit transition)
        if prev_eclipse and not in_eclipse:
            cycles += 1
        prev_eclipse = in_eclipse

        times.append(t)
        temps_c.append(temp_k - 273.15)
        eclipses.append(in_eclipse)

        # Heat inputs
        q_solar = 0.0 if in_eclipse else absorptivity * _SOLAR_FLUX_W_M2 * area_m2
        q_earth_ir = emissivity * _EARTH_IR_W_M2 * area_m2 * 0.3  # View factor ~0.3
        q_internal = internal_dissipation_w

        # Radiative cooling
        q_radiated = emissivity * _STEFAN_BOLTZMANN * area_m2 * temp_k**4

        # Energy balance
        q_net = q_solar + q_earth_ir + q_internal - q_radiated
        dt_k = (q_net / thermal_capacity) * step_seconds
        temp_k += dt_k

        # Clamp to physical bounds
        temp_k = max(temp_k, 3.0)  # Don't go below ~-270°C

        t += step_seconds

    return ThermalProfile(
        times_seconds=tuple(times),
        temperatures_c=tuple(temps_c),
        eclipse_flags=tuple(eclipses),
        num_cycles=cycles,
    )


def derate_at_temperature(
    temperature_c: float,
    max_temp_c: float = 85.0,
    min_temp_c: float = -40.0,
) -> float:
    """Compute performance derating factor at a given temperature.

    Returns 1.0 at optimal temperature (center of range), decreasing
    linearly toward 0.5 at the extremes, and further beyond limits.

    Args:
        temperature_c: Current temperature in °C.
        max_temp_c: Maximum rated temperature.
        min_temp_c: Minimum rated temperature.

    Returns:
        Derating factor (0.0 to 1.0).
    """
    center = (max_temp_c + min_temp_c) / 2.0
    half_range = (max_temp_c - min_temp_c) / 2.0

    if half_range <= 0:
        return 1.0

    distance = abs(temperature_c - center) / half_range

    # Linear derating: 1.0 at center, 0.8 at edge, 0.5 at 1.5x edge
    factor = max(0.0, 1.0 - 0.2 * distance)
    return min(factor, 1.0)
