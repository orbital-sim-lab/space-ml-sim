"""Import radiation test facility data from CSV files.

Parses SEE test data (ion species, LET, fluence, errors) from standard
CSV format and computes cross-sections for comparison with simulation.

Expected CSV columns:
    ion, energy_mev, let_mev_cm2_mg, fluence_ions_cm2, errors, bits_under_test
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TextIO

import pandas as pd


@dataclass(frozen=True)
class RadTestRecord:
    """A single radiation test data point."""

    ion: str
    energy_mev: float
    let_mev_cm2_mg: float
    fluence_ions_cm2: float
    errors: int
    bits_under_test: int
    cross_section_cm2_per_bit: float


def load_rad_test_csv(source: TextIO) -> list[RadTestRecord]:
    """Load radiation test data from a CSV file or string buffer.

    Args:
        source: File-like object with CSV content.

    Returns:
        List of RadTestRecord with computed cross-sections.
    """
    reader = csv.DictReader(source)
    records: list[RadTestRecord] = []

    for row in reader:
        fluence = float(row["fluence_ions_cm2"])
        errors = int(row["errors"])
        bits = int(row["bits_under_test"])

        if fluence > 0 and bits > 0 and errors > 0:
            xsec = errors / (fluence * bits)
        else:
            xsec = 0.0

        records.append(
            RadTestRecord(
                ion=row["ion"].strip(),
                energy_mev=float(row["energy_mev"]),
                let_mev_cm2_mg=float(row["let_mev_cm2_mg"]),
                fluence_ions_cm2=fluence,
                errors=errors,
                bits_under_test=bits,
                cross_section_cm2_per_bit=xsec,
            )
        )

    return records


def to_dataframe(records: list[RadTestRecord]) -> pd.DataFrame:
    """Convert test records to a pandas DataFrame.

    Args:
        records: List of RadTestRecord.

    Returns:
        DataFrame with all fields as columns.
    """
    return pd.DataFrame(
        [
            {
                "ion": r.ion,
                "energy_mev": r.energy_mev,
                "let_mev_cm2_mg": r.let_mev_cm2_mg,
                "fluence_ions_cm2": r.fluence_ions_cm2,
                "errors": r.errors,
                "bits_under_test": r.bits_under_test,
                "cross_section_cm2_per_bit": r.cross_section_cm2_per_bit,
            }
            for r in records
        ]
    )


def cross_section_curve(
    records: list[RadTestRecord],
) -> tuple[list[float], list[float]]:
    """Extract cross-section vs LET curve from test data.

    Args:
        records: Test data records.

    Returns:
        Tuple of (LET values, cross-section values), sorted by LET.
    """
    sorted_records = sorted(records, key=lambda r: r.let_mev_cm2_mg)
    lets = [r.let_mev_cm2_mg for r in sorted_records]
    xsecs = [r.cross_section_cm2_per_bit for r in sorted_records]
    return lets, xsecs
