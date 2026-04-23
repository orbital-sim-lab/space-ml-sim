"""Orbital power budget calculator.

Models power generation from solar panels and consumption across orbit
phases to determine if a satellite can sustain AI inference workloads.

Key considerations:
- Solar generation only during sunlit phase
- Battery must sustain all loads during eclipse
- TMR multiplies compute power consumption
- Eclipse fraction depends on altitude and beta angle
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from space_ml_sim.core.orbit import OrbitConfig


_SOLAR_FLUX_W_M2 = 1361.0  # Solar constant at 1 AU


@dataclass(frozen=True)
class SolarArrayConfig:
    """Solar panel configuration."""

    area_m2: float
    efficiency: float = 0.30  # GaAs triple-junction typical
    degradation_per_year: float = 0.02  # 2%/year BOL to EOL


@dataclass(frozen=True)
class BatteryConfig:
    """Battery configuration."""

    capacity_wh: float
    dod_limit: float = 0.40  # Max depth of discharge (40% for Li-ion space grade)
    round_trip_efficiency: float = 0.92


@dataclass(frozen=True)
class PowerBudgetResult:
    """Complete power budget analysis result."""

    solar_generation_w: float
    total_load_sunlit_w: float
    total_load_eclipse_w: float
    sunlit_duration_minutes: float
    eclipse_duration_minutes: float
    margin_w: float
    power_positive: bool
    battery_eclipse_wh: float
    battery_dod_fraction: float
    eclipse_compute_duty_cycle: float
    orbit_average_power_w: float


class PowerBudget:
    """Analyze satellite power budget across orbital phases."""

    def __init__(
        self,
        orbit: OrbitConfig,
        solar_array: SolarArrayConfig,
        battery: BatteryConfig,
        base_load_w: float = 10.0,
        compute_load_w: float = 15.0,
        comms_load_w: float = 0.0,
        thermal_load_w: float = 0.0,
        tmr_multiplier: float = 1.0,
        mission_age_years: float = 0.0,
    ) -> None:
        self.orbit = orbit
        self.solar = solar_array
        self.battery = battery
        self.base_load_w = base_load_w
        self.compute_load_w = compute_load_w * tmr_multiplier
        self.comms_load_w = comms_load_w
        self.thermal_load_w = thermal_load_w
        self.mission_age_years = mission_age_years

    def analyze(self) -> PowerBudgetResult:
        """Run power budget analysis.

        Returns:
            PowerBudgetResult with all metrics.
        """
        # Orbital period and eclipse fraction
        period_sec = self.orbit.orbital_period_seconds
        period_min = period_sec / 60.0
        eclipse_frac = _eclipse_fraction(self.orbit.altitude_km)
        eclipse_min = period_min * eclipse_frac
        sunlit_min = period_min - eclipse_min

        # Solar generation (average over sunlit phase, accounting for cosine losses)
        degradation = (1 - self.solar.degradation_per_year) ** self.mission_age_years
        solar_w = (
            _SOLAR_FLUX_W_M2
            * self.solar.area_m2
            * self.solar.efficiency
            * degradation
            * 0.7  # Average cosine factor for body-mounted panels
        )

        # Loads
        load_sunlit = (
            self.base_load_w + self.compute_load_w + self.comms_load_w + self.thermal_load_w
        )
        # During eclipse: no comms (typically), reduced thermal
        load_eclipse = self.base_load_w + self.compute_load_w + self.thermal_load_w * 0.5

        # Battery energy needed for eclipse
        eclipse_sec = eclipse_min * 60
        battery_needed_wh = load_eclipse * (eclipse_sec / 3600)
        usable_battery_wh = self.battery.capacity_wh * self.battery.dod_limit * self.battery.round_trip_efficiency
        battery_dod = battery_needed_wh / self.battery.capacity_wh if self.battery.capacity_wh > 0 else 1.0

        # Eclipse compute duty cycle (how much of eclipse can run compute)
        base_eclipse_wh = self.base_load_w * (eclipse_sec / 3600)
        available_for_compute_wh = max(0, usable_battery_wh - base_eclipse_wh)
        compute_eclipse_wh = self.compute_load_w * (eclipse_sec / 3600)
        eclipse_duty = min(1.0, available_for_compute_wh / compute_eclipse_wh) if compute_eclipse_wh > 0 else 1.0

        # Orbit-average power balance
        sunlit_sec = sunlit_min * 60
        energy_generated_wh = solar_w * (sunlit_sec / 3600)
        energy_consumed_wh = (
            load_sunlit * (sunlit_sec / 3600)
            + load_eclipse * (eclipse_sec / 3600)
        )
        orbit_avg_power = (energy_generated_wh - energy_consumed_wh) / (period_sec / 3600)

        # Margin: surplus power averaged over orbit
        margin = solar_w * (sunlit_min / period_min) - (
            load_sunlit * (sunlit_min / period_min) + load_eclipse * (eclipse_min / period_min)
        )
        power_positive = margin > 0 and battery_dod <= self.battery.dod_limit

        return PowerBudgetResult(
            solar_generation_w=solar_w,
            total_load_sunlit_w=load_sunlit,
            total_load_eclipse_w=load_eclipse,
            sunlit_duration_minutes=sunlit_min,
            eclipse_duration_minutes=eclipse_min,
            margin_w=margin,
            power_positive=power_positive,
            battery_eclipse_wh=battery_needed_wh,
            battery_dod_fraction=min(battery_dod, 1.0),
            eclipse_compute_duty_cycle=eclipse_duty,
            orbit_average_power_w=orbit_avg_power,
        )


def _eclipse_fraction(altitude_km: float) -> float:
    """Estimate eclipse fraction for a circular LEO orbit.

    Uses a simplified geometric model (cylindrical shadow).
    Actual eclipse fraction depends on beta angle (sun-orbit angle).
    This returns the worst-case (beta=0) fraction.

    Args:
        altitude_km: Orbital altitude in km.

    Returns:
        Fraction of orbit spent in eclipse (0-1).
    """
    R_EARTH = 6371.0
    r = R_EARTH + altitude_km
    # Half-angle of Earth's shadow cone
    rho = math.asin(R_EARTH / r)
    # Eclipse fraction = rho / pi (for beta=0)
    return rho / math.pi
