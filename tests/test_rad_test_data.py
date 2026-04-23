"""TDD tests for radiation test facility data import.

Supports importing SEE test data from CSV files with standard columns
(LET, fluence, cross-section, errors) for comparison with simulation.
"""

from __future__ import annotations

import io


SAMPLE_CSV = """\
ion,energy_mev,let_mev_cm2_mg,fluence_ions_cm2,errors,bits_under_test
Ne-20,200,3.5,1e7,0,1000000
Ar-40,400,10.0,1e7,15,1000000
Fe-56,800,28.0,1e7,120,1000000
Kr-84,1200,40.0,1e7,230,1000000
Xe-131,1500,60.0,1e7,245,1000000
"""


class TestRadTestDataImport:
    """Import radiation test data from CSV."""

    def test_load_from_csv_string(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        assert len(records) == 5

    def test_records_have_required_fields(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        r = records[0]
        assert hasattr(r, "ion")
        assert hasattr(r, "let_mev_cm2_mg")
        assert hasattr(r, "cross_section_cm2_per_bit")

    def test_cross_section_computed(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        # Fe-56: 120 errors / (1e7 fluence * 1e6 bits) = 1.2e-11
        fe = [r for r in records if r.ion == "Fe-56"][0]
        assert abs(fe.cross_section_cm2_per_bit - 1.2e-11) < 1e-13

    def test_zero_errors_zero_cross_section(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        ne = [r for r in records if r.ion == "Ne-20"][0]
        assert ne.cross_section_cm2_per_bit == 0.0

    def test_to_dataframe(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))

        from space_ml_sim.data.rad_test_data import to_dataframe

        df = to_dataframe(records)
        assert len(df) == 5
        assert "let_mev_cm2_mg" in df.columns
        assert "cross_section_cm2_per_bit" in df.columns


class TestCrossSectionCurve:
    """Generate cross-section vs LET curve from test data."""

    def test_curve_is_sorted_by_let(self) -> None:
        from space_ml_sim.data.rad_test_data import load_rad_test_csv, cross_section_curve

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        lets, xsecs = cross_section_curve(records)

        assert lets == sorted(lets)
        assert len(lets) == len(xsecs)

    def test_curve_is_monotonically_nondecreasing(self) -> None:
        """Cross-section should generally increase with LET."""
        from space_ml_sim.data.rad_test_data import load_rad_test_csv, cross_section_curve

        records = load_rad_test_csv(io.StringIO(SAMPLE_CSV))
        lets, xsecs = cross_section_curve(records)

        # Filter to non-zero cross-sections
        nonzero = [(let, x) for let, x in zip(lets, xsecs) if x > 0]
        if len(nonzero) > 1:
            for i in range(len(nonzero) - 1):
                assert nonzero[i + 1][1] >= nonzero[i][1] * 0.5, (
                    "Cross-section should not drop dramatically with increasing LET"
                )
