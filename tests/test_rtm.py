"""TDD tests for requirements traceability matrix generator."""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig


class TestRTMGeneration:
    """RTM must map requirements to analysis evidence."""

    def test_generate_returns_html(self) -> None:
        from space_ml_sim.reports.rtm import generate_rtm, RequirementEvidence

        entries = [
            RequirementEvidence(
                req_id="RAD-001",
                requirement="SEU rate shall not exceed 10 upsets/orbit",
                standard="ECSS-Q-ST-60-15C",
                analysis_result="3.2 SEU/orbit at 550km SSO",
                status="PASS",
                margin="3.1x",
            ),
            RequirementEvidence(
                req_id="RAD-002",
                requirement="Mission TID shall not exceed 10 krad",
                standard="MIL-STD-883 TM 1019",
                analysis_result="1.5 krad over 5 years",
                status="PASS",
                margin="6.7x",
            ),
        ]
        html = generate_rtm(mission_name="TEST-SAT", entries=entries)
        assert "<html" in html.lower()
        assert "RAD-001" in html
        assert "RAD-002" in html

    def test_contains_pass_fail_status(self) -> None:
        from space_ml_sim.reports.rtm import generate_rtm, RequirementEvidence

        entries = [
            RequirementEvidence(
                req_id="RAD-001",
                requirement="TID < 10 krad",
                standard="ECSS",
                analysis_result="15 krad",
                status="FAIL",
                margin="-1.5x",
            ),
        ]
        html = generate_rtm(mission_name="TEST", entries=entries)
        assert "FAIL" in html

    def test_auto_generate_from_simulation(self) -> None:
        """Auto-generate RTM entries from trade study results."""
        from space_ml_sim.reports.rtm import auto_generate_rtm
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig
        from space_ml_sim.models.chip_profiles import RAD5500

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        config = MissionConfig(
            name="Test",
            orbit=orbit,
            chip=RAD5500,
            tmr_strategy="full_tmr",
            shielding_mm_al=3.0,
            mission_years=5.0,
        )
        study = TradeStudy(configs=[config])
        results = study.run()

        html = auto_generate_rtm(
            mission_name="AUTO-SAT",
            trade_study_result=results[0],
            max_tid_krad=50.0,
            max_seus_per_orbit=100.0,
        )
        assert "AUTO-SAT" in html
        assert "RAD-" in html
