"""Smoke + invariant tests for the built-in chip profiles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from space_ml_sim.models import (
    ALL_CHIPS,
    AURIX_TC4X,
    ChipProfile,
    JETSON_AGX_ORIN,
    RAD5500,
    SAMRH71,
)


class TestAllChipsInvariants:
    """Invariants every built-in chip profile must satisfy."""

    @pytest.mark.parametrize("chip", ALL_CHIPS, ids=lambda c: c.name)
    def test_basic_constraints(self, chip: ChipProfile) -> None:
        assert chip.node_nm > 0
        assert chip.tdp_watts > 0
        assert chip.seu_cross_section_cm2 > 0
        assert chip.tid_tolerance_krad > 0
        assert chip.compute_tops >= 0
        assert chip.memory_bits > 0
        assert chip.name  # non-empty

    def test_no_duplicate_names(self) -> None:
        names = [c.name for c in ALL_CHIPS]
        assert len(names) == len(set(names))

    def test_relative_ordering_makes_sense(self) -> None:
        """Rad-hard chips should have higher TID tolerance than COTS."""
        assert RAD5500.tid_tolerance_krad > JETSON_AGX_ORIN.tid_tolerance_krad
        assert SAMRH71.tid_tolerance_krad > JETSON_AGX_ORIN.tid_tolerance_krad
        # COTS chips should be cheaper to run (lower SEU cross-section is HARDER,
        # so flip the comparison) — Jetson's SEU cross-section should be larger
        # than rad-hardened chips.
        assert JETSON_AGX_ORIN.seu_cross_section_cm2 > RAD5500.seu_cross_section_cm2


class TestAurixProfile:
    """The new AURIX TC4x profile is wired up correctly."""

    def test_aurix_in_all_chips(self) -> None:
        assert AURIX_TC4X in ALL_CHIPS

    def test_aurix_marked_as_not_space_qualified(self) -> None:
        """The notes must clearly state AURIX is not space-qualified."""
        assert "NOT space-qualified" in AURIX_TC4X.notes

    def test_aurix_conservative_tid(self) -> None:
        """AURIX is automotive grade — TID budget should be small."""
        assert AURIX_TC4X.tid_tolerance_krad <= 10  # automotive, not space

    def test_aurix_has_meaningful_compute(self) -> None:
        """AURIX TC4x has a PPU AI extension — non-zero compute."""
        assert AURIX_TC4X.compute_tops > 0


class TestChipProfileValidation:
    """Pydantic validation rejects bad inputs."""

    def test_negative_seu_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChipProfile(
                name="bad",
                node_nm=10,
                tdp_watts=1.0,
                max_temp_c=85,
                seu_cross_section_cm2=-1e-14,
                tid_tolerance_krad=10,
                compute_tops=1.0,
                memory_bits=1024,
            )

    def test_zero_node_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChipProfile(
                name="bad",
                node_nm=0,
                tdp_watts=1.0,
                max_temp_c=85,
                seu_cross_section_cm2=1e-14,
                tid_tolerance_krad=10,
                compute_tops=1.0,
                memory_bits=1024,
            )
