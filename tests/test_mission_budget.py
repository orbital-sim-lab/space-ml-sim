"""TDD tests for mission radiation budget calculator.

Written FIRST before implementation to drive the design.
All tests should FAIL until mission_budget.py is created.

Test coverage:
  - Return type and structure
  - Positive SEU count for any orbit/chip combo
  - SEU per-day is a rate (independent of mission_years)
  - TID scales linearly with mission_years
  - RAD5500 (1000 krad) survives 5 years at 550km
  - Trillium TPU (15 krad) fails 5 years at 2000km
  - Higher altitude produces more SEUs
  - Recommended shielding increases for higher orbits
  - tid_margin algebraic identity
  - years_to_tid_limit algebraic identity
  - Edge cases: zero-shielding, minimum valid inputs
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ISS_ALT = 420.0
ISS_INC = 51.6
STARLINK_ALT = 550.0
STARLINK_INC = 53.0
HIGH_ALT = 2000.0
HIGH_INC = 53.0


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestMissionBudgetReturnType:
    """compute_mission_budget must return a MissionBudget dataclass."""

    def test_returns_mission_budget_instance(self):
        """compute_mission_budget must return a MissionBudget."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget, MissionBudget
        from space_ml_sim.models.chip_profiles import RAD5500

        result = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert isinstance(result, MissionBudget)

    def test_all_fields_present(self):
        """MissionBudget must expose all documented fields."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        required_fields = [
            "mission_duration_years",
            "altitude_km",
            "inclination_deg",
            "chip_name",
            "expected_seu_count",
            "expected_seu_per_day",
            "tid_accumulated_krad",
            "tid_margin_fraction",
            "years_to_tid_limit",
            "tid_ok",
            "recommended_shielding_mm",
        ]
        for field in required_fields:
            assert hasattr(budget, field), f"MissionBudget missing field: {field}"

    def test_chip_name_matches_profile(self):
        """chip_name must match the provided ChipProfile.name."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.chip_name == RAD5500.name

    def test_mission_duration_stored_correctly(self):
        """mission_duration_years must equal the input mission_years."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=5.0,
        )
        assert budget.mission_duration_years == 5.0

    def test_altitude_stored_correctly(self):
        """altitude_km must equal the input altitude_km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.altitude_km == STARLINK_ALT


# ---------------------------------------------------------------------------
# SEU count: positivity and scaling
# ---------------------------------------------------------------------------


class TestSeuCount:
    """expected_seu_count must be positive and well-behaved."""

    def test_expected_seu_count_positive_rad5500(self):
        """SEU count must be positive for RAD5500 at 550km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.expected_seu_count > 0.0

    def test_expected_seu_count_positive_trillium(self):
        """SEU count must be positive for Trillium TPU at 550km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        budget = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.expected_seu_count > 0.0

    def test_expected_seu_count_positive_all_chips(self):
        """SEU count must be positive for every chip in ALL_CHIPS."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import ALL_CHIPS

        for chip in ALL_CHIPS:
            budget = compute_mission_budget(
                chip=chip,
                altitude_km=STARLINK_ALT,
                inclination_deg=STARLINK_INC,
                mission_years=3.0,
            )
            assert budget.expected_seu_count > 0.0, (
                f"expected_seu_count not positive for {chip.name}"
            )

    def test_seu_count_scales_with_mission_years(self):
        """Doubling mission_years must double expected_seu_count."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget_1yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=1.0,
        )
        budget_2yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=2.0,
        )
        ratio = budget_2yr.expected_seu_count / budget_1yr.expected_seu_count
        assert abs(ratio - 2.0) < 1e-9, (
            f"SEU count should double when mission_years doubles, got ratio={ratio:.6f}"
        )


class TestSeuPerDay:
    """expected_seu_per_day is a rate — must not depend on mission_years."""

    def test_seu_per_day_is_positive(self):
        """SEU per-day rate must be positive."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.expected_seu_per_day > 0.0

    def test_seu_per_day_independent_of_mission_years(self):
        """expected_seu_per_day must be the same for 1-year and 5-year missions."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget_1yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=1.0,
        )
        budget_5yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=5.0,
        )
        assert abs(budget_1yr.expected_seu_per_day - budget_5yr.expected_seu_per_day) < 1e-12, (
            "SEU per-day rate must be independent of mission duration"
        )

    def test_seu_per_day_equals_total_divided_by_days(self):
        """expected_seu_per_day must equal expected_seu_count / total_days."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        mission_years = 3.0
        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=mission_years,
        )
        total_days = mission_years * 365.25
        expected_rate = budget.expected_seu_count / total_days
        assert abs(budget.expected_seu_per_day - expected_rate) < 1e-10, (
            f"expected_seu_per_day={budget.expected_seu_per_day} != seu_count/days={expected_rate}"
        )


# ---------------------------------------------------------------------------
# TID accumulation
# ---------------------------------------------------------------------------


class TestTidAccumulation:
    """TID must scale linearly with mission duration and follow algebraic identities."""

    def test_tid_accumulated_positive(self):
        """Accumulated TID must be positive."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.tid_accumulated_krad > 0.0

    def test_tid_scales_linearly_with_mission_years(self):
        """Doubling mission_years must double TID accumulated."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget_1yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=1.0,
        )
        budget_2yr = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=2.0,
        )
        ratio = budget_2yr.tid_accumulated_krad / budget_1yr.tid_accumulated_krad
        assert abs(ratio - 2.0) < 1e-9, (
            f"TID should double when mission_years doubles, got ratio={ratio:.6f}"
        )

    def test_tid_margin_algebraic_identity(self):
        """tid_margin_fraction must equal tid_accumulated / chip_tolerance exactly."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        expected_margin = budget.tid_accumulated_krad / RAD5500.tid_tolerance_krad
        assert abs(budget.tid_margin_fraction - expected_margin) < 1e-10, (
            f"tid_margin_fraction={budget.tid_margin_fraction} != tid/tolerance={expected_margin}"
        )

    def test_years_to_tid_limit_algebraic_identity(self):
        """years_to_tid_limit * annual_tid_rate must equal chip_tolerance."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500
        from space_ml_sim.environment.radiation import RadiationEnvironment

        shielding = 2.0
        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
            shielding_mm_al=shielding,
        )
        env = RadiationEnvironment(
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            shielding_mm_al=shielding,
        )
        annual_tid = env.tid_rate_krad_per_day * 365.25
        computed_limit = budget.years_to_tid_limit * annual_tid
        assert abs(computed_limit - RAD5500.tid_tolerance_krad) < 1e-6, (
            f"years_to_limit * annual_rate = {computed_limit:.4f} krad, "
            f"expected {RAD5500.tid_tolerance_krad} krad"
        )


# ---------------------------------------------------------------------------
# Mission assessment: tid_ok
# ---------------------------------------------------------------------------


class TestTidOk:
    """tid_ok must reflect whether TID stays within chip tolerance."""

    def test_rad5500_survives_5_years_at_550km(self):
        """RAD5500 (1000 krad) must have tid_ok=True for 5 years at 550km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=5.0,
            shielding_mm_al=2.0,
        )
        assert budget.tid_ok is True, (
            f"RAD5500 (1000 krad) should survive 5yr at 550km; "
            f"TID={budget.tid_accumulated_krad:.2f} krad, margin={budget.tid_margin_fraction:.3f}"
        )

    def test_trillium_fails_5_years_at_2000km(self):
        """Trillium TPU (15 krad) must have tid_ok=False for 5 years at 2000km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        budget = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=5.0,
            shielding_mm_al=2.0,
        )
        assert budget.tid_ok is False, (
            f"Trillium TPU (15 krad) should fail 5yr at 2000km; "
            f"TID={budget.tid_accumulated_krad:.2f} krad, margin={budget.tid_margin_fraction:.3f}"
        )

    def test_tid_ok_is_bool(self):
        """tid_ok must be a Python bool, not a numpy bool or similar."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert isinstance(budget.tid_ok, bool)

    def test_tid_ok_false_when_margin_exceeds_one(self):
        """tid_ok must be False whenever tid_margin_fraction > 1.0."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        budget = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=5.0,
        )
        if budget.tid_margin_fraction > 1.0:
            assert budget.tid_ok is False
        else:
            assert budget.tid_ok is True


# ---------------------------------------------------------------------------
# Altitude effects
# ---------------------------------------------------------------------------


class TestAltitudeEffects:
    """Higher altitude must result in more SEUs and more TID."""

    def test_higher_altitude_more_seus(self):
        """SEU count at 2000km must exceed SEU count at 550km."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        low = compute_mission_budget(
            chip=RAD5500,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        high = compute_mission_budget(
            chip=RAD5500,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        assert high.expected_seu_count > low.expected_seu_count, (
            f"Expected more SEUs at 2000km vs 550km; "
            f"high={high.expected_seu_count:.2e}, low={low.expected_seu_count:.2e}"
        )

    def test_higher_altitude_more_tid(self):
        """TID at 2000km must exceed TID at 550km (same mission duration)."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        low = compute_mission_budget(
            chip=RAD5500,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        high = compute_mission_budget(
            chip=RAD5500,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        assert high.tid_accumulated_krad > low.tid_accumulated_krad, (
            f"Expected more TID at 2000km vs 550km; "
            f"high={high.tid_accumulated_krad:.4f}, low={low.tid_accumulated_krad:.4f}"
        )

    def test_recommended_shielding_increases_for_higher_orbit(self):
        """Recommended shielding at 2000km must be >= shielding at 550km for same chip."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        low = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=5.0,
        )
        high = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=5.0,
        )
        assert high.recommended_shielding_mm >= low.recommended_shielding_mm, (
            f"Expected more shielding at 2000km vs 550km; "
            f"high={high.recommended_shielding_mm}, low={low.recommended_shielding_mm}"
        )


# ---------------------------------------------------------------------------
# Shielding effects
# ---------------------------------------------------------------------------


class TestShieldingEffects:
    """More shielding must reduce SEU rate and TID."""

    def test_more_shielding_reduces_tid(self):
        """TID with 5mm shielding must be less than TID with 2mm shielding."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        thin = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
            shielding_mm_al=2.0,
        )
        thick = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
            shielding_mm_al=5.0,
        )
        assert thick.tid_accumulated_krad < thin.tid_accumulated_krad, (
            "Thicker shielding must reduce TID accumulation"
        )

    def test_more_shielding_reduces_seu_rate(self):
        """SEU per-day with 5mm shielding must be less than with 2mm shielding."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        thin = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
            shielding_mm_al=2.0,
        )
        thick = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
            shielding_mm_al=5.0,
        )
        assert thick.expected_seu_per_day < thin.expected_seu_per_day, (
            "Thicker shielding must reduce daily SEU rate"
        )

    def test_recommended_shielding_at_least_input_when_tid_ok(self):
        """When tid_ok=True, recommended_shielding_mm should equal input shielding."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        shielding = 2.0
        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=5.0,
            shielding_mm_al=shielding,
        )
        # When the chip survives, recommended shielding should be the input amount
        if budget.tid_ok:
            assert budget.recommended_shielding_mm == shielding


# ---------------------------------------------------------------------------
# years_to_tid_limit behaviour
# ---------------------------------------------------------------------------


class TestYearsToTidLimit:
    """years_to_tid_limit must be positive and consistent with the radiation model."""

    def test_years_to_tid_limit_positive(self):
        """years_to_tid_limit must be a positive finite number."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert budget.years_to_tid_limit > 0.0
        assert math.isfinite(budget.years_to_tid_limit)

    def test_years_to_tid_limit_shorter_for_sensitive_chip(self):
        """A chip with lower TID tolerance reaches limit sooner (same orbit)."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500, TRILLIUM_V6E

        tough = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        sensitive = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        assert sensitive.years_to_tid_limit < tough.years_to_tid_limit, (
            f"Trillium (15 krad) should reach limit before RAD5500 (1000 krad); "
            f"trillium={sensitive.years_to_tid_limit:.1f}yr, "
            f"rad5500={tough.years_to_tid_limit:.1f}yr"
        )

    def test_years_to_tid_limit_shorter_at_higher_altitude(self):
        """TID limit is reached sooner at higher altitude (same chip)."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        low = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        high = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=2000.0,
            inclination_deg=53.0,
            mission_years=3.0,
        )
        assert high.years_to_tid_limit < low.years_to_tid_limit, (
            "TID limit reached sooner at higher altitude"
        )


# ---------------------------------------------------------------------------
# Determinism: same inputs produce same outputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    """compute_mission_budget is deterministic — no randomness."""

    def test_same_inputs_same_outputs(self):
        """Calling twice with identical inputs must return identical results."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        kwargs = dict(
            chip=RAD5500,
            altitude_km=550.0,
            inclination_deg=53.0,
            mission_years=3.0,
            shielding_mm_al=2.0,
        )
        result1 = compute_mission_budget(**kwargs)
        result2 = compute_mission_budget(**kwargs)

        assert result1.expected_seu_count == result2.expected_seu_count
        assert result1.tid_accumulated_krad == result2.tid_accumulated_krad
        assert result1.tid_ok == result2.tid_ok

    def test_frozen_dataclass_immutable(self):
        """MissionBudget is frozen — attempting to set attributes must raise."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500

        budget = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            budget.expected_seu_count = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cross-chip comparison
# ---------------------------------------------------------------------------


class TestCrossChipComparison:
    """Validate relative ordering across chip profiles."""

    def test_rad5500_fewer_seus_per_day_than_trillium(self):
        """RAD5500 (smaller cross-section) must have fewer SEUs/day than Trillium."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        from space_ml_sim.models.chip_profiles import RAD5500, TRILLIUM_V6E

        rad = compute_mission_budget(
            chip=RAD5500,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        tpu = compute_mission_budget(
            chip=TRILLIUM_V6E,
            altitude_km=STARLINK_ALT,
            inclination_deg=STARLINK_INC,
            mission_years=3.0,
        )
        # Trillium cross-section 5e-13 >> RAD5500 1e-15
        assert tpu.expected_seu_per_day > rad.expected_seu_per_day, (
            "Trillium has 500x larger cross-section — should see far more SEUs/day"
        )

    def test_default_shielding_is_2mm(self):
        """Default shielding_mm_al parameter must be 2.0."""
        from space_ml_sim.metrics.mission_budget import compute_mission_budget
        import inspect

        sig = inspect.signature(compute_mission_budget)
        default = sig.parameters["shielding_mm_al"].default
        assert default == 2.0, f"Default shielding should be 2.0mm, got {default}"
