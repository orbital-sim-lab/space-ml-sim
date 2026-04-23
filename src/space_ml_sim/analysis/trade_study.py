"""Mission trade-study comparison API.

Compare multiple mission configurations side-by-side to evaluate
cost/benefit/risk tradeoffs across orbit, chip, TMR strategy,
and shielding choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile


_COMPUTE_MULTIPLIERS = {
    "none": 1.0,
    "checkpoint_rollback": 1.0,
    "selective_tmr": 1.5,
    "full_tmr": 3.0,
}


@dataclass(frozen=True)
class MissionConfig:
    """A single mission configuration for trade study."""

    name: str
    orbit: OrbitConfig
    chip: ChipProfile
    tmr_strategy: str
    shielding_mm_al: float
    mission_years: float


@dataclass(frozen=True)
class TradeStudyResult:
    """Analysis result for one mission configuration."""

    name: str
    altitude_km: float
    inclination_deg: float
    chip_name: str
    tmr_strategy: str
    shielding_mm_al: float
    mission_years: float
    seu_rate_per_day: float
    expected_seus_per_orbit: float
    tid_over_mission_krad: float
    compute_multiplier: float
    power_watts: float
    compute_tops: float
    tid_tolerance_krad: float
    risk_level: str


class TradeStudy:
    """Compare multiple mission configurations side-by-side."""

    def __init__(self, configs: list[MissionConfig]) -> None:
        self.configs = configs

    def run(self) -> list[TradeStudyResult]:
        """Evaluate all configurations and return comparison results.

        Returns:
            List of TradeStudyResult, one per configuration.
        """
        results: list[TradeStudyResult] = []
        for config in self.configs:
            results.append(_evaluate_config(config))
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Run the study and return results as a DataFrame.

        Returns:
            DataFrame with one row per configuration and all metrics as columns.
        """
        results = self.run()
        return pd.DataFrame(
            [
                {
                    "name": r.name,
                    "altitude_km": r.altitude_km,
                    "inclination_deg": r.inclination_deg,
                    "chip_name": r.chip_name,
                    "tmr_strategy": r.tmr_strategy,
                    "shielding_mm_al": r.shielding_mm_al,
                    "mission_years": r.mission_years,
                    "seu_rate_per_day": r.seu_rate_per_day,
                    "expected_seus_per_orbit": r.expected_seus_per_orbit,
                    "tid_over_mission_krad": r.tid_over_mission_krad,
                    "compute_multiplier": r.compute_multiplier,
                    "power_watts": r.power_watts,
                    "compute_tops": r.compute_tops,
                    "tid_tolerance_krad": r.tid_tolerance_krad,
                    "risk_level": r.risk_level,
                }
                for r in results
            ]
        )


def _evaluate_config(config: MissionConfig) -> TradeStudyResult:
    """Evaluate a single mission configuration."""
    rad_env = RadiationEnvironment(
        altitude_km=config.orbit.altitude_km,
        inclination_deg=config.orbit.inclination_deg,
        shielding_mm_al=config.shielding_mm_al,
    )

    # SEU rate scaled by chip cross-section
    reference_xsec = 1e-14
    xsec_factor = config.chip.seu_cross_section_cm2 / reference_xsec
    seu_rate_per_bit_per_sec = rad_env.base_seu_rate * xsec_factor
    seu_rate_per_day = seu_rate_per_bit_per_sec * config.chip.memory_bits * 86400

    # Orbital period
    R_EARTH = 6371.0
    MU = 398600.4418
    a = R_EARTH + config.orbit.altitude_km
    period_sec = 2 * math.pi * math.sqrt(a**3 / MU)
    seus_per_orbit = seu_rate_per_bit_per_sec * config.chip.memory_bits * period_sec

    # TID
    mission_days = config.mission_years * 365.25
    tid_over_mission = rad_env.tid_rate_krad_per_day * mission_days

    # Compute cost
    compute_mult = _COMPUTE_MULTIPLIERS.get(config.tmr_strategy, 1.0)
    power = config.chip.tdp_watts * compute_mult

    # Risk assessment
    risk = _assess_risk(seus_per_orbit, tid_over_mission, config.chip.tid_tolerance_krad)

    return TradeStudyResult(
        name=config.name,
        altitude_km=config.orbit.altitude_km,
        inclination_deg=config.orbit.inclination_deg,
        chip_name=config.chip.name,
        tmr_strategy=config.tmr_strategy,
        shielding_mm_al=config.shielding_mm_al,
        mission_years=config.mission_years,
        seu_rate_per_day=seu_rate_per_day,
        expected_seus_per_orbit=seus_per_orbit,
        tid_over_mission_krad=tid_over_mission,
        compute_multiplier=compute_mult,
        power_watts=power,
        compute_tops=config.chip.compute_tops,
        tid_tolerance_krad=config.chip.tid_tolerance_krad,
        risk_level=risk,
    )


def _assess_risk(
    seus_per_orbit: float,
    tid_krad: float,
    tid_tolerance_krad: float,
) -> str:
    """Classify overall mission radiation risk."""
    tid_margin = tid_tolerance_krad / tid_krad if tid_krad > 0 else float("inf")

    if seus_per_orbit > 10 or tid_margin < 1.5:
        return "HIGH"
    if seus_per_orbit > 1 or tid_margin < 3.0:
        return "MEDIUM"
    return "LOW"
