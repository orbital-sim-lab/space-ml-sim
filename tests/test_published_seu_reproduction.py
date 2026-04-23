"""Regression check: space-ml-sim matches published SEU measurement ranges.

This test imports the same published-case table used by the public
reproduction script (``examples/04_reproduce_published_seu.py``) and
asserts the model stays within the documented ranges. Any drift here is
a signal that calibration constants in ``radiation.py`` have changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing the reproduction script without installing examples/ as a package.
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from importlib import import_module  # noqa: E402

_reproduction = import_module("04_reproduce_published_seu")


@pytest.mark.parametrize("case", _reproduction.CASES, ids=lambda c: c.name)
def test_prediction_within_published_range(case) -> None:
    """Predicted rate must fall inside the published upsets/bit/day band."""
    predicted = _reproduction.predict_upsets_per_bit_per_day(case)
    assert (
        case.reported_upsets_per_bit_per_day_low
        <= predicted
        <= case.reported_upsets_per_bit_per_day_high
    ), (
        f"{case.name}: predicted={predicted:.2e} outside "
        f"[{case.reported_upsets_per_bit_per_day_low:.1e}, "
        f"{case.reported_upsets_per_bit_per_day_high:.1e}] "
        f"(ref: {case.reference})"
    )
