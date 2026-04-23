"""Single Event Latchup (SEL) modeling.

SEL is a destructive radiation effect where a parasitic thyristor in
CMOS circuits latches, drawing excessive current that can damage or
destroy the device. Unlike SEU (which is a soft/recoverable error),
SEL requires immediate power cycling.

SEL rate depends on:
- Device SEL cross-section (from beam testing)
- LET threshold for SEL onset
- Integral particle flux above the threshold LET
- Shielding effectiveness

Radiation-hardened chips are typically SEL-immune up to 62-100 MeV·cm²/mg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from space_ml_sim.environment.radiation import RadiationEnvironment


# Simplified integral LET flux model (particles/cm²/day above LET threshold)
# Based on CREME96 GCR model for LEO, solar minimum
# Flux decreases roughly as power law with LET
_REFERENCE_FLUX_PER_CM2_DAY = 1e-2  # At LET = 1 MeV·cm²/mg at 500km


def _integral_flux_above_let(
    threshold_let: float,
    altitude_km: float,
    shielding_mm_al: float,
) -> float:
    """Estimate integral particle flux above a given LET threshold.

    Uses a simplified power-law model:
        Flux(>L) = F0 * (L/L0)^(-alpha)

    Args:
        threshold_let: LET threshold in MeV·cm²/mg.
        altitude_km: Orbital altitude in km.
        shielding_mm_al: Aluminium shielding in mm.

    Returns:
        Flux in particles/cm²/day.
    """
    # Power law index (typical for GCR integral spectrum)
    alpha = 2.5

    # Reference flux at LET = 1 MeV·cm²/mg
    base_flux = _REFERENCE_FLUX_PER_CM2_DAY

    # Altitude scaling
    alt_factor = 1.0 + 0.5 * max(0, (altitude_km - 500) / 1500)

    # Shielding attenuation
    shield_factor = math.exp(-0.2 * shielding_mm_al)

    # Power law
    flux = base_flux * (threshold_let ** (-alpha)) * alt_factor * shield_factor

    return flux


def sel_rate_per_day(
    rad_env: RadiationEnvironment,
    sel_cross_section_cm2: float,
    sel_threshold_let: float,
) -> float:
    """Predict SEL event rate.

    SEL rate = integral_flux(>L_th) × cross_section

    Args:
        rad_env: Radiation environment.
        sel_cross_section_cm2: Device SEL cross-section in cm².
        sel_threshold_let: LET threshold for SEL onset in MeV·cm²/mg.

    Returns:
        Expected SEL events per day.
    """
    flux = _integral_flux_above_let(
        threshold_let=sel_threshold_let,
        altitude_km=rad_env.altitude_km,
        shielding_mm_al=rad_env.shielding_mm_al,
    )
    return flux * sel_cross_section_cm2


def mission_sel_probability(
    rad_env: RadiationEnvironment,
    sel_cross_section_cm2: float,
    sel_threshold_let: float,
    mission_years: float,
) -> float:
    """Probability of at least one SEL event over the mission.

    P(>=1) = 1 - exp(-lambda*t) where lambda is the SEL rate.

    Args:
        rad_env: Radiation environment.
        sel_cross_section_cm2: Device SEL cross-section in cm².
        sel_threshold_let: LET threshold in MeV·cm²/mg.
        mission_years: Mission duration in years.

    Returns:
        Probability of at least one SEL (0 to 1).
    """
    rate = sel_rate_per_day(rad_env, sel_cross_section_cm2, sel_threshold_let)
    mission_days = mission_years * 365.25
    expected_events = rate * mission_days
    return 1.0 - math.exp(-expected_events)


@dataclass(frozen=True)
class SELMitigationResult:
    """SEL mitigation analysis result."""

    expected_sel_events: float
    total_downtime_hours: float
    availability_fraction: float


def sel_mitigation_requirements(
    sel_rate_per_day: float,
    power_cycle_time_seconds: float,
    mission_years: float,
) -> SELMitigationResult:
    """Analyze SEL mitigation requirements.

    Computes expected downtime from SEL events requiring power cycling.

    Args:
        sel_rate_per_day: Expected SEL events per day.
        power_cycle_time_seconds: Time to detect and power-cycle in seconds.
        mission_years: Mission duration in years.

    Returns:
        SELMitigationResult with downtime and availability.
    """
    mission_days = mission_years * 365.25
    expected_events = sel_rate_per_day * mission_days
    downtime_seconds = expected_events * power_cycle_time_seconds
    downtime_hours = downtime_seconds / 3600.0
    total_hours = mission_days * 24.0
    availability = 1.0 - (downtime_hours / total_hours) if total_hours > 0 else 1.0

    return SELMitigationResult(
        expected_sel_events=expected_events,
        total_downtime_hours=downtime_hours,
        availability_fraction=max(0.0, min(1.0, availability)),
    )
