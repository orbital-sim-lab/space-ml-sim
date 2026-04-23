"""Solar cycle modulation of radiation environment.

Models the effect of the ~11-year solar cycle on trapped particle
flux and galactic cosmic ray (GCR) intensity. During solar maximum,
trapped proton flux increases while GCR decreases, and vice versa.

Scaling factors are based on published AP-9/AE-9 solar cycle
variation data (Ginet et al. 2013) and CREME96 GCR modulation.
"""

from __future__ import annotations

from space_ml_sim.environment.radiation import RadiationEnvironment


SOLAR_PHASES: tuple[str, ...] = (
    "solar_min",
    "solar_ascending",
    "solar_max",
    "solar_descending",
    "average",
)

# SEU scaling factors by phase (relative to average)
# At low altitude: dominated by GCR (higher at solar min)
# At high altitude: dominated by trapped protons (higher at solar max)
_SEU_GCR_FACTOR = {
    "solar_min": 1.3,  # GCR 30% higher at solar min
    "solar_ascending": 1.1,
    "solar_max": 0.7,  # GCR 30% lower at solar max
    "solar_descending": 0.9,
    "average": 1.0,
}

_SEU_TRAPPED_FACTOR = {
    "solar_min": 0.5,  # Trapped protons ~50% lower at solar min
    "solar_ascending": 0.8,
    "solar_max": 2.0,  # Trapped protons ~2x at solar max
    "solar_descending": 1.2,
    "average": 1.0,
}

# TID scaling (dominated by trapped particles)
_TID_FACTOR = {
    "solar_min": 0.6,
    "solar_ascending": 0.8,
    "solar_max": 1.8,
    "solar_descending": 1.1,
    "average": 1.0,
}


def apply_solar_cycle(
    rad_env: RadiationEnvironment,
    phase: str,
) -> RadiationEnvironment:
    """Apply solar cycle modulation to a radiation environment.

    Creates a new RadiationEnvironment with adjusted rates based on
    the solar cycle phase. The original environment is unchanged.

    Args:
        rad_env: Base radiation environment (assumed "average" solar conditions).
        phase: One of "solar_min", "solar_ascending", "solar_max",
               "solar_descending", "average".

    Returns:
        New RadiationEnvironment with modulated rates.

    Raises:
        ValueError: If phase is not recognized.
    """
    if phase not in SOLAR_PHASES:
        raise ValueError(f"Unknown solar cycle phase: {phase!r}. Must be one of {SOLAR_PHASES}")

    # Blend GCR and trapped contributions based on altitude
    # Below 800km: ~90% GCR, 10% trapped
    # Above 1500km: ~30% GCR, 70% trapped
    alt = rad_env.altitude_km
    if alt <= 800:
        gcr_weight = 0.9
    elif alt >= 1500:
        gcr_weight = 0.3
    else:
        gcr_weight = 0.9 - 0.6 * (alt - 800) / 700
    trapped_weight = 1.0 - gcr_weight

    seu_factor = gcr_weight * _SEU_GCR_FACTOR[phase] + trapped_weight * _SEU_TRAPPED_FACTOR[phase]
    tid_factor = _TID_FACTOR[phase]

    new_seu_rate = rad_env.base_seu_rate * seu_factor
    new_tid_rate = rad_env.tid_rate_krad_per_day * tid_factor

    # Create new environment with overridden rates
    env = RadiationEnvironment(
        altitude_km=rad_env.altitude_km,
        inclination_deg=rad_env.inclination_deg,
        shielding_mm_al=rad_env.shielding_mm_al,
    )
    object.__setattr__(env, "base_seu_rate", new_seu_rate)
    object.__setattr__(env, "tid_rate_krad_per_day", new_tid_rate)

    return env
