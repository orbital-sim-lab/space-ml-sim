"""TDD tests for MIL-STD-883 TM 1019 test methodology report.

The report documents SEE test procedures and results per
MIL-STD-883 Test Method 1019 (Ionizing Dose Total Dose Test Procedure).
"""

from __future__ import annotations

import pytest


MOCK_TEST_DATA = {
    "device_name": "TRILLIUM_V6E",
    "test_facility": "NSRL Brookhaven",
    "ion_species": "Fe-56",
    "energy_mev": 1000,
    "let_mev_cm2_mg": 28.0,
    "fluence_ions_cm2": 1e7,
    "cross_section_cm2": 1.2e-14,
    "threshold_let": 5.0,
    "saturation_cross_section": 2.5e-14,
    "num_errors_observed": 120,
    "bits_under_test": 1_000_000,
    "test_temperature_c": 25.0,
}


class TestMILSTDReport:
    """MIL-STD-883 TM 1019 report generation."""

    def test_generate_returns_html(self) -> None:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(**MOCK_TEST_DATA)
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_contains_required_sections(self) -> None:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(**MOCK_TEST_DATA)

        required = [
            "Test Configuration",
            "Beam Parameters",
            "Test Results",
            "Cross-Section Analysis",
        ]
        for section in required:
            assert section in html, f"Missing section: {section}"

    def test_contains_device_info(self) -> None:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(**MOCK_TEST_DATA)
        assert "TRILLIUM_V6E" in html
        assert "NSRL Brookhaven" in html

    def test_contains_beam_parameters(self) -> None:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(**MOCK_TEST_DATA)
        assert "Fe-56" in html
        assert "28.0" in html  # LET

    def test_contains_cross_section(self) -> None:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(**MOCK_TEST_DATA)
        assert "1.2" in html  # cross section value
