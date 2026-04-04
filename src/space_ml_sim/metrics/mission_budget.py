"""Mission radiation budget calculator.

Computes cumulative SEU count and TID dose over a mission lifetime
for a given orbit and chip profile. Uses deterministic calculations
(not Monte Carlo) for the expected values.

The SEU model treats base_seu_rate as upsets/bit/second at a reference
cross-section of 1e-14 cm^2. The expected total SEU count (Poisson mean)
is therefore:

    expected_seu = base_seu_rate
                   * (chip_cross_section / 1e-14)
                   * chip_memory_bits
                   * mission_seconds

TID accumulation is purely deterministic:

    tid_krad = tid_rate_krad_per_day * mission_days
"""

from __future__ import annotations

from dataclasses import dataclass

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile

# Cross-section reference used by RadiationEnvironment.base_seu_rate calibration.
_REFERENCE_CROSS_SECTION_CM2: float = 1e-14

# Shielding search grid: 0.5 mm steps from 0.5 mm to 49.5 mm.
_SHIELD_SEARCH_GRID: list[float] = [s * 0.5 for s in range(1, 100)]


@dataclass(frozen=True)
class MissionBudget:
    """Radiation budget for a mission.

    All values are deterministic expected values — no Monte Carlo sampling.
    """

    mission_duration_years: float
    altitude_km: float
    inclination_deg: float
    chip_name: str

    # Deterministic expected values (Poisson mean, not random)
    expected_seu_count: float  # Total expected SEUs over mission
    expected_seu_per_day: float  # Average SEUs per day (rate — independent of duration)
    tid_accumulated_krad: float  # Total TID at end of mission
    tid_margin_fraction: float  # tid_accumulated / chip_tolerance (1.0 = at limit)
    years_to_tid_limit: float  # Years until TID reaches chip tolerance

    # Mission assessment
    tid_ok: bool  # True if TID stays within tolerance
    recommended_shielding_mm: float  # Minimum shielding to survive mission


def compute_mission_budget(
    chip: ChipProfile,
    altitude_km: float,
    inclination_deg: float,
    mission_years: float,
    shielding_mm_al: float = 2.0,
) -> MissionBudget:
    """Compute deterministic radiation budget for a mission.

    Uses expected (mean) values — no randomness. Results are exact
    for the parametric model.

    Args:
        chip: Hardware chip profile.
        altitude_km: Orbital altitude in km.
        inclination_deg: Orbital inclination in degrees.
        mission_years: Mission duration in years.
        shielding_mm_al: Aluminum shielding thickness in mm.

    Returns:
        MissionBudget with expected SEU/TID values and mission assessment.
    """
    env = RadiationEnvironment(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        shielding_mm_al=shielding_mm_al,
    )

    mission_days = mission_years * 365.25
    mission_seconds = mission_days * 86400.0

    # --- Expected SEU count (Poisson mean — fully deterministic) ---
    # base_seu_rate is upsets/bit/second at reference cross-section 1e-14 cm^2.
    cross_section_factor = chip.seu_cross_section_cm2 / _REFERENCE_CROSS_SECTION_CM2
    expected_seu_total = (
        env.base_seu_rate * cross_section_factor * chip.memory_bits * mission_seconds
    )

    # SEU per-day is a rate: independent of mission duration.
    expected_seu_per_day = env.base_seu_rate * cross_section_factor * chip.memory_bits * 86400.0

    # --- TID accumulation (purely deterministic) ---
    tid_total_krad = env.tid_rate_krad_per_day * mission_days
    tid_margin = tid_total_krad / chip.tid_tolerance_krad

    # Years to TID limit from current orbital/shielding environment.
    if env.tid_rate_krad_per_day > 0:
        years_to_limit = chip.tid_tolerance_krad / (env.tid_rate_krad_per_day * 365.25)
    else:
        years_to_limit = float("inf")

    tid_ok: bool = tid_margin <= 1.0

    # --- Minimum shielding to survive mission ---
    # When current shielding is already sufficient, keep it.
    # Otherwise binary-search upward in 0.5mm steps.
    recommended_shield = shielding_mm_al
    if not tid_ok:
        for trial_shield in _SHIELD_SEARCH_GRID:
            trial_env = RadiationEnvironment(
                altitude_km=altitude_km,
                inclination_deg=inclination_deg,
                shielding_mm_al=trial_shield,
            )
            trial_tid = trial_env.tid_rate_krad_per_day * mission_days
            if trial_tid <= chip.tid_tolerance_krad:
                recommended_shield = trial_shield
                break

    return MissionBudget(
        mission_duration_years=mission_years,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        chip_name=chip.name,
        expected_seu_count=expected_seu_total,
        expected_seu_per_day=expected_seu_per_day,
        tid_accumulated_krad=tid_total_krad,
        tid_margin_fraction=tid_margin,
        years_to_tid_limit=years_to_limit,
        tid_ok=tid_ok,
        recommended_shielding_mm=recommended_shield,
    )
