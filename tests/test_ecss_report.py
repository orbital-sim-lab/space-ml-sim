"""TDD tests for ECSS-Q-ST-60-15C compliance report generation.

Written FIRST before implementation (RED phase).

The ECSS report generator:
- Produces structured radiation hardness assurance reports
- Follows ECSS-Q-ST-60-15C section structure
- Includes orbit parameters, radiation environment, fault analysis, and TMR config
- Outputs HTML (printable to PDF via browser)
"""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig


# Simulated mission data for tests
MOCK_MISSION = {
    "mission_name": "LEO-AI-SAT-01",
    "orbit": OrbitConfig(
        altitude_km=550,
        inclination_deg=97.6,
        raan_deg=0.0,
        true_anomaly_deg=0.0,
    ),
    "mission_duration_years": 5.0,
    "chip_name": "TRILLIUM_V6E",
    "model_name": "ResNet-18",
    "total_parameters": 11_689_512,
    "seu_rate_per_bit_per_day": 1.2e-14,
    "tid_rate_rad_per_day": 0.82,
    "expected_seus_per_orbit": 3.2,
    "tmr_strategy": "selective_tmr",
    "protected_layers": ["layer1.0.conv1.weight", "layer2.0.conv1.weight"],
    "compute_multiplier": 1.4,
    "expected_accuracy_recovery": 0.28,
    "shielding_mm_al": 3.0,
}


class TestReportGeneration:
    """Report must be generatable from mission data."""

    def test_generate_returns_html(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_contains_mission_name(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "LEO-AI-SAT-01" in html

    def test_contains_required_sections(self) -> None:
        """ECSS-Q-ST-60-15C requires specific report sections."""
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)

        required_sections = [
            "Mission Overview",
            "Orbital Environment",
            "Radiation Environment",
            "Component Analysis",
            "Fault Tolerance Strategy",
            "Risk Assessment",
        ]
        for section in required_sections:
            assert section in html, f"Missing required section: {section}"

    def test_contains_orbit_parameters(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "550" in html  # altitude
        assert "97.6" in html  # inclination

    def test_contains_radiation_data(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "SEU" in html
        assert "TID" in html

    def test_contains_tmr_recommendation(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "selective_tmr" in html or "Selective TMR" in html


class TestReportSaving:
    """Report must be saveable to file."""

    def test_save_to_file(self, tmp_path) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        output_path = tmp_path / "report.html"
        output_path.write_text(html)

        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Non-trivial content


class TestReportCompleteness:
    """Report must include all required data fields."""

    def test_includes_shielding_info(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "3.0" in html  # shielding mm
        assert "shielding" in html.lower()

    def test_includes_mission_duration(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "5.0" in html or "5 year" in html.lower()

    def test_includes_compute_overhead(self) -> None:
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        html = generate_ecss_report(**MOCK_MISSION)
        assert "1.4" in html  # compute multiplier
