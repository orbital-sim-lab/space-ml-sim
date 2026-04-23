"""End-to-end mission analysis pipeline.

Runs the full analysis chain in one call:
    orbit → radiation → thermal → link budget → TMR recommendation → report

This is the primary entry point for consultants generating deliverables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.environment.thermal_cycling import generate_thermal_profile
from space_ml_sim.environment.solar_cycle import apply_solar_cycle
from space_ml_sim.comms.link_budget import compute_link_budget
from space_ml_sim.models.chip_profiles import ChipProfile
from space_ml_sim.reports.ecss_report import generate_ecss_report


@dataclass(frozen=True)
class MissionAnalysisResult:
    """Complete mission analysis output across all domains."""

    mission_name: str
    # Radiation
    seu_rate_per_day: float
    expected_seus_per_orbit: float
    tid_over_mission_krad: float
    # Thermal
    min_temperature_c: float
    max_temperature_c: float
    thermal_cycles_per_orbit: int
    # Link budget
    downlink_margin_db: float | None
    max_data_rate_bps: float | None
    # Risk
    overall_risk: str
    risk_factors: dict[str, str]
    # Reports
    ecss_report_html: str

    def to_summary_dict(self) -> dict:
        """Export key metrics as a flat dictionary."""
        return {
            "mission_name": self.mission_name,
            "seu_rate_per_day": self.seu_rate_per_day,
            "expected_seus_per_orbit": self.expected_seus_per_orbit,
            "tid_over_mission_krad": self.tid_over_mission_krad,
            "min_temperature_c": self.min_temperature_c,
            "max_temperature_c": self.max_temperature_c,
            "thermal_cycles_per_orbit": self.thermal_cycles_per_orbit,
            "downlink_margin_db": self.downlink_margin_db,
            "max_data_rate_bps": self.max_data_rate_bps,
            "overall_risk": self.overall_risk,
        }


def run_mission_analysis(
    mission_name: str,
    orbit: OrbitConfig,
    chip: ChipProfile,
    mission_years: float,
    shielding_mm_al: float = 2.0,
    tmr_strategy: str = "none",
    solar_cycle_phase: str | None = None,
    # Link budget (optional)
    downlink_freq_hz: float = 2.2e9,
    tx_power_dbw: float = 5.0,
    tx_antenna_gain_dbi: float = 6.0,
    rx_antenna_gain_dbi: float = 30.0,
    system_noise_temp_k: float = 300.0,
    downlink_bandwidth_hz: float = 2e6,
) -> MissionAnalysisResult:
    """Run complete mission analysis across all domains.

    Args:
        mission_name: Mission identifier.
        orbit: Orbital configuration.
        chip: Hardware chip profile.
        mission_years: Mission duration in years.
        shielding_mm_al: Aluminium equivalent shielding in mm.
        tmr_strategy: TMR strategy name.
        solar_cycle_phase: Optional solar cycle phase for worst-case analysis.
        downlink_freq_hz: Downlink carrier frequency.
        tx_power_dbw: Transmitter power in dBW.
        tx_antenna_gain_dbi: Transmit antenna gain in dBi.
        rx_antenna_gain_dbi: Receive antenna gain in dBi.
        system_noise_temp_k: System noise temperature in K.
        downlink_bandwidth_hz: Channel bandwidth in Hz.

    Returns:
        MissionAnalysisResult with all domain results.
    """
    # --- Radiation ---
    rad_env = RadiationEnvironment(
        altitude_km=orbit.altitude_km,
        inclination_deg=orbit.inclination_deg,
        shielding_mm_al=shielding_mm_al,
    )

    if solar_cycle_phase:
        rad_env = apply_solar_cycle(rad_env, phase=solar_cycle_phase)

    xsec_factor = chip.seu_cross_section_cm2 / 1e-14
    seu_rate = rad_env.base_seu_rate * xsec_factor

    R_EARTH = 6371.0
    MU = 398600.4418
    a = R_EARTH + orbit.altitude_km
    period_sec = 2 * math.pi * math.sqrt(a**3 / MU)

    seu_rate_per_day = seu_rate * chip.memory_bits * 86400
    seus_per_orbit = seu_rate * chip.memory_bits * period_sec

    mission_days = mission_years * 365.25
    tid_over_mission = rad_env.tid_rate_krad_per_day * mission_days

    # --- Thermal ---
    thermal = generate_thermal_profile(
        orbit=orbit,
        duration_seconds=period_sec * 3,  # 3 orbits
        step_seconds=30.0,
        internal_dissipation_w=chip.tdp_watts * 0.3,  # ~30% as heat at idle
    )

    # --- Link Budget ---
    slant_range_km = orbit.altitude_km / math.sin(math.radians(10))  # 10° min elevation
    slant_range_km = min(slant_range_km, orbit.altitude_km * 4)  # Cap

    link_result = compute_link_budget(
        tx_power_dbw=tx_power_dbw,
        tx_antenna_gain_dbi=tx_antenna_gain_dbi,
        frequency_hz=downlink_freq_hz,
        distance_km=slant_range_km,
        rx_antenna_gain_dbi=rx_antenna_gain_dbi,
        system_noise_temp_k=system_noise_temp_k,
        bandwidth_hz=downlink_bandwidth_hz,
    )

    # --- Risk Assessment ---
    risk_factors: dict[str, str] = {}

    tid_margin = chip.tid_tolerance_krad / tid_over_mission if tid_over_mission > 0 else float("inf")
    if tid_margin < 1.5:
        risk_factors["TID"] = "HIGH"
    elif tid_margin < 3.0:
        risk_factors["TID"] = "MEDIUM"
    else:
        risk_factors["TID"] = "LOW"

    if seus_per_orbit > 10:
        risk_factors["SEU"] = "HIGH"
    elif seus_per_orbit > 1:
        risk_factors["SEU"] = "MEDIUM"
    else:
        risk_factors["SEU"] = "LOW"

    if thermal.temperatures_c:
        if max(thermal.temperatures_c) > chip.max_temp_c:
            risk_factors["THERMAL"] = "HIGH"
        elif max(thermal.temperatures_c) > chip.max_temp_c * 0.8:
            risk_factors["THERMAL"] = "MEDIUM"
        else:
            risk_factors["THERMAL"] = "LOW"

    if link_result.link_margin_db < 3:
        risk_factors["LINK"] = "HIGH"
    elif link_result.link_margin_db < 6:
        risk_factors["LINK"] = "MEDIUM"
    else:
        risk_factors["LINK"] = "LOW"

    if "HIGH" in risk_factors.values():
        overall_risk = "HIGH"
    elif "MEDIUM" in risk_factors.values():
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    # --- ECSS Report ---
    compute_mult = {"none": 1.0, "full_tmr": 3.0, "selective_tmr": 1.5,
                     "checkpoint_rollback": 1.0}.get(tmr_strategy, 1.0)

    ecss_html = generate_ecss_report(
        mission_name=mission_name,
        orbit=orbit,
        mission_duration_years=mission_years,
        chip_name=chip.name,
        model_name="Mission payload model",
        total_parameters=1_000_000,
        seu_rate_per_bit_per_day=seu_rate * 86400,
        tid_rate_rad_per_day=rad_env.tid_rate_krad_per_day * 1000,
        expected_seus_per_orbit=seus_per_orbit,
        tmr_strategy=tmr_strategy,
        protected_layers=[],
        compute_multiplier=compute_mult,
        expected_accuracy_recovery=0.0,
        shielding_mm_al=shielding_mm_al,
    )

    return MissionAnalysisResult(
        mission_name=mission_name,
        seu_rate_per_day=seu_rate_per_day,
        expected_seus_per_orbit=seus_per_orbit,
        tid_over_mission_krad=tid_over_mission,
        min_temperature_c=min(thermal.temperatures_c),
        max_temperature_c=max(thermal.temperatures_c),
        thermal_cycles_per_orbit=thermal.num_cycles,
        downlink_margin_db=link_result.link_margin_db,
        max_data_rate_bps=link_result.max_data_rate_bps,
        overall_risk=overall_risk,
        risk_factors=risk_factors,
        ecss_report_html=ecss_html,
    )
