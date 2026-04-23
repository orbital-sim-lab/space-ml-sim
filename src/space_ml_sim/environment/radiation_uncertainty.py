"""Uncertainty quantification for radiation model outputs.

Adds confidence intervals to SEU and TID rate estimates. The uncertainty
model accounts for:
- Parametric model uncertainty (~factor of 2-3 for SEU, ~50% for TID)
- Altitude-dependent uncertainty (higher orbits are less characterized)
- Solar cycle variability (factor ~2-5 for trapped particles)

Uncertainty factors are based on published comparison studies between
AP-8/AE-8, AP-9/AE-9, and flight data (e.g., Xapsos et al., Barth et al.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from space_ml_sim.environment.radiation import RadiationEnvironment


@dataclass(frozen=True)
class UncertaintyBand:
    """A value with lower and upper confidence bounds."""

    nominal: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.90


def seu_rate_with_uncertainty(
    rad_env: RadiationEnvironment,
    confidence: float = 0.90,
) -> UncertaintyBand:
    """SEU rate with confidence interval.

    Uncertainty sources:
    - Model vs flight data: factor ~2-3 (AP-8 vs AP-9 differ by 2-5x)
    - Solar cycle phase: factor ~2 for trapped protons
    - Cross-section uncertainty: factor ~1.5

    Combined uncertainty modeled as log-normal with sigma dependent on altitude.

    Args:
        rad_env: Radiation environment to compute uncertainty for.
        confidence: Confidence level (0-1) for the interval.

    Returns:
        UncertaintyBand with nominal rate and bounds in upsets/bit/second.
    """
    nominal = rad_env.base_seu_rate

    # Log-normal sigma: wider at higher altitudes (less characterized)
    base_sigma = 0.4  # ~factor of 1.5 at 500km
    altitude_factor = 1.0 + 0.3 * max(0, (rad_env.altitude_km - 500) / 1500)
    sigma = base_sigma * altitude_factor

    z = _z_score(confidence)
    lower = nominal * math.exp(-z * sigma)
    upper = nominal * math.exp(z * sigma)

    return UncertaintyBand(
        nominal=nominal,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
    )


def tid_rate_with_uncertainty(
    rad_env: RadiationEnvironment,
    confidence: float = 0.90,
) -> UncertaintyBand:
    """TID rate with confidence interval.

    TID uncertainty is typically ~30-50% (better characterized than SEU).

    Args:
        rad_env: Radiation environment.
        confidence: Confidence level (0-1).

    Returns:
        UncertaintyBand with nominal rate and bounds in krad(Si)/day.
    """
    nominal = rad_env.tid_rate_krad_per_day

    # TID is better characterized: sigma ~0.25 (factor ~1.3)
    sigma = 0.25 + 0.15 * max(0, (rad_env.altitude_km - 500) / 1500)

    z = _z_score(confidence)
    lower = nominal * math.exp(-z * sigma)
    upper = nominal * math.exp(z * sigma)

    return UncertaintyBand(
        nominal=nominal,
        lower_bound=lower,
        upper_bound=upper,
        confidence=confidence,
    )


def mission_tid_with_uncertainty(
    rad_env: RadiationEnvironment,
    mission_years: float,
    confidence: float = 0.90,
) -> UncertaintyBand:
    """Total mission TID with uncertainty.

    Args:
        rad_env: Radiation environment.
        mission_years: Mission duration in years.
        confidence: Confidence level.

    Returns:
        UncertaintyBand in krad(Si) for the full mission.
    """
    rate = tid_rate_with_uncertainty(rad_env, confidence=confidence)
    days = mission_years * 365.25
    return UncertaintyBand(
        nominal=rate.nominal * days,
        lower_bound=rate.lower_bound * days,
        upper_bound=rate.upper_bound * days,
        confidence=confidence,
    )


def mission_seus_with_uncertainty(
    rad_env: RadiationEnvironment,
    mission_years: float,
    total_bits: int,
    confidence: float = 0.90,
) -> UncertaintyBand:
    """Total mission SEU count with uncertainty.

    Args:
        rad_env: Radiation environment.
        mission_years: Mission duration in years.
        total_bits: Number of bits exposed to radiation.
        confidence: Confidence level.

    Returns:
        UncertaintyBand with expected SEU count over mission.
    """
    rate = seu_rate_with_uncertainty(rad_env, confidence=confidence)
    seconds = mission_years * 365.25 * 86400
    return UncertaintyBand(
        nominal=rate.nominal * total_bits * seconds,
        lower_bound=rate.lower_bound * total_bits * seconds,
        upper_bound=rate.upper_bound * total_bits * seconds,
        confidence=confidence,
    )


def _z_score(confidence: float) -> float:
    """Approximate z-score for a given confidence level.

    Uses a rational approximation good to ~0.01 for common confidence levels.
    """
    # Common lookup
    _table = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    if confidence in _table:
        return _table[confidence]

    # Approximate using inverse error function approximation
    p = (1 + confidence) / 2
    # Abramowitz and Stegun approximation for inverse normal CDF
    t = math.sqrt(-2 * math.log(1 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
