"""TDD tests for end-to-end mission analysis pipeline.

Single entry point that runs: orbit -> radiation -> thermal -> link budget
-> TMR recommendation -> compliance report. The full consultant deliverable.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.models.chip_profiles import RAD5500, TRILLIUM_V6E


class TestMissionAnalysis:
    """Full mission analysis pipeline."""

    def test_run_returns_result(self) -> None:
        from space_ml_sim.analysis.mission_analysis import (
            run_mission_analysis,
            MissionAnalysisResult,
        )

        result = run_mission_analysis(
            mission_name="TEST-SAT-01",
            orbit=OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0),
            chip=RAD5500,
            mission_years=5.0,
            shielding_mm_al=3.0,
            tmr_strategy="selective_tmr",
        )
        assert isinstance(result, MissionAnalysisResult)

    def test_result_has_all_domains(self) -> None:
        from space_ml_sim.analysis.mission_analysis import run_mission_analysis

        result = run_mission_analysis(
            mission_name="FULL-TEST",
            orbit=OrbitConfig(altitude_km=550, inclination_deg=97.6, raan_deg=0, true_anomaly_deg=0),
            chip=TRILLIUM_V6E,
            mission_years=3.0,
        )
        # Radiation
        assert result.seu_rate_per_day > 0
        assert result.tid_over_mission_krad > 0
        # Thermal
        assert result.min_temperature_c < result.max_temperature_c
        assert result.thermal_cycles_per_orbit >= 0
        # Link budget
        assert result.downlink_margin_db is not None
        # Risk
        assert result.overall_risk in ("LOW", "MEDIUM", "HIGH")
        # Report
        assert result.ecss_report_html is not None
        assert len(result.ecss_report_html) > 100

    def test_different_chips_different_risk(self) -> None:
        from space_ml_sim.analysis.mission_analysis import run_mission_analysis

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        r_rad = run_mission_analysis(
            mission_name="RAD", orbit=orbit, chip=RAD5500, mission_years=5.0
        )
        r_cots = run_mission_analysis(
            mission_name="COTS", orbit=orbit, chip=TRILLIUM_V6E, mission_years=5.0
        )
        # Rad-hard chip should have lower SEU rate
        assert r_rad.seu_rate_per_day < r_cots.seu_rate_per_day

    def test_summary_dict(self) -> None:
        from space_ml_sim.analysis.mission_analysis import run_mission_analysis

        result = run_mission_analysis(
            mission_name="DICT-TEST",
            orbit=OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0),
            chip=RAD5500,
            mission_years=5.0,
        )
        summary = result.to_summary_dict()
        assert isinstance(summary, dict)
        assert "mission_name" in summary
        assert "seu_rate_per_day" in summary
        assert "overall_risk" in summary

    def test_with_solar_cycle(self) -> None:
        from space_ml_sim.analysis.mission_analysis import run_mission_analysis

        result = run_mission_analysis(
            mission_name="SOLAR-MAX",
            orbit=OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0),
            chip=RAD5500,
            mission_years=5.0,
            solar_cycle_phase="solar_max",
        )
        assert result.seu_rate_per_day > 0

    def test_with_link_budget_params(self) -> None:
        from space_ml_sim.analysis.mission_analysis import run_mission_analysis

        result = run_mission_analysis(
            mission_name="LINK-TEST",
            orbit=OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0),
            chip=RAD5500,
            mission_years=5.0,
            downlink_freq_hz=8.4e9,
            tx_power_dbw=5.0,
            tx_antenna_gain_dbi=6.0,
            rx_antenna_gain_dbi=40.0,
        )
        assert result.downlink_margin_db is not None
