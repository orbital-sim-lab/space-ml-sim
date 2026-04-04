"""Monte Carlo mission lifetime reliability estimation.

Runs N simulated mission lifetimes, sampling SEU events and tracking
TID accumulation at each time step, to produce statistical estimates
of mission survival probability and SEU count distributions.

Unlike mission_budget.py (which computes deterministic expected values),
this module captures the stochastic nature of radiation effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile

# Simulation time step: 1 day (86400 seconds).
# Fine enough for TID accumulation, coarse enough for speed.
_DT_SECONDS: float = 86400.0

# Cross-section reference matching RadiationEnvironment calibration.
_REFERENCE_CROSS_SECTION_CM2: float = 1e-14


@dataclass(frozen=True)
class MonteCarloResult:
    """Statistical results from Monte Carlo mission reliability estimation."""

    num_simulations: int
    survival_probability: float  # Fraction of sims where TID stayed within tolerance
    tid_failure_probability: float  # 1 - survival_probability
    mean_seu_count: float  # Mean total SEUs across all simulations
    seu_count_p5: float  # 5th percentile SEU count
    seu_count_p95: float  # 95th percentile SEU count
    mean_time_to_tid_failure_years: float  # Mean TTF (inf if no failures)


def estimate_mission_reliability(
    chip: ChipProfile,
    altitude_km: float,
    inclination_deg: float,
    mission_years: float,
    num_simulations: int = 1000,
    shielding_mm_al: float = 2.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """Estimate mission reliability via Monte Carlo simulation.

    Each simulation steps through the mission day-by-day, sampling
    SEU events from a Poisson distribution and accumulating TID
    deterministically. A mission "fails" when accumulated TID exceeds
    the chip's tolerance.

    Args:
        chip: Hardware chip profile.
        altitude_km: Orbital altitude in km.
        inclination_deg: Orbital inclination in degrees.
        mission_years: Mission duration in years.
        num_simulations: Number of Monte Carlo trials.
        shielding_mm_al: Aluminum shielding thickness in mm.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with survival statistics.
    """
    rng = np.random.default_rng(seed)
    env = RadiationEnvironment(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        shielding_mm_al=shielding_mm_al,
    )

    mission_days = mission_years * 365.25
    num_steps = max(1, int(mission_days))
    tid_per_step_krad = env.tid_rate_krad_per_day * (_DT_SECONDS / 86400.0)

    # SEU Poisson rate per step
    xsec_factor = chip.seu_cross_section_cm2 / _REFERENCE_CROSS_SECTION_CM2
    seu_lambda_per_step = env.base_seu_rate * xsec_factor * chip.memory_bits * _DT_SECONDS

    # Pre-sample all SEU counts at once for performance: (num_simulations, num_steps)
    all_seus = rng.poisson(lam=seu_lambda_per_step, size=(num_simulations, num_steps))

    # Track results
    total_seus = np.sum(all_seus, axis=1)  # shape: (num_simulations,)

    # TID is deterministic per step, so failure day is the same for all sims
    # (no stochastic TID in this model). But we still compute per-sim for
    # the result structure.
    tid_at_end = tid_per_step_krad * num_steps
    if tid_per_step_krad > 0:
        steps_to_failure = chip.tid_tolerance_krad / tid_per_step_krad
    else:
        steps_to_failure = float("inf")

    tid_failed = tid_at_end > chip.tid_tolerance_krad
    num_failures = num_simulations if tid_failed else 0
    survival_prob = 1.0 - (num_failures / num_simulations)

    if tid_failed and steps_to_failure != float("inf"):
        mean_ttf_years = (steps_to_failure * _DT_SECONDS) / (365.25 * 86400.0)
    else:
        mean_ttf_years = float("inf")

    return MonteCarloResult(
        num_simulations=num_simulations,
        survival_probability=survival_prob,
        tid_failure_probability=1.0 - survival_prob,
        mean_seu_count=float(np.mean(total_seus)),
        seu_count_p5=float(np.percentile(total_seus, 5)),
        seu_count_p95=float(np.percentile(total_seus, 95)),
        mean_time_to_tid_failure_years=mean_ttf_years,
    )
