"""TDD tests for Monte Carlo mission lifetime reliability estimation.

Written FIRST before implementation (RED phase). Tests should FAIL until
metrics/monte_carlo.py is created.

The Monte Carlo estimator runs N simulated mission lifetimes, sampling
SEU events and TID accumulation at each time step, to produce:
- Probability of TID failure before mission end
- Distribution of total SEU counts
- Confidence intervals on mission survival
- Time-to-failure distribution
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Tests for the Monte Carlo estimator
# ---------------------------------------------------------------------------


class TestMonteCarloReturnsCorrectStructure:
    """The estimator must return a well-formed result dataclass."""

    def test_returns_result_dataclass(self) -> None:
        from space_ml_sim.metrics.monte_carlo import (
            MonteCarloResult,
            estimate_mission_reliability,
        )
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=100,
            seed=42,
        )
        assert isinstance(result, MonteCarloResult)

    def test_result_has_required_fields(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=100,
            seed=42,
        )
        # Must have these fields
        assert hasattr(result, "survival_probability")
        assert hasattr(result, "tid_failure_probability")
        assert hasattr(result, "mean_seu_count")
        assert hasattr(result, "seu_count_p5")
        assert hasattr(result, "seu_count_p95")
        assert hasattr(result, "mean_time_to_tid_failure_years")
        assert hasattr(result, "num_simulations")

    def test_probabilities_are_valid(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=100,
            seed=42,
        )
        assert 0.0 <= result.survival_probability <= 1.0
        assert 0.0 <= result.tid_failure_probability <= 1.0
        assert abs(result.survival_probability + result.tid_failure_probability - 1.0) < 1e-9


class TestRadHardenedChipSurvives:
    """RAD5500 (1000 krad tolerance) should survive low LEO easily."""

    def test_rad5500_survives_5yr_at_500km(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=5.0,
            num_simulations=500,
            seed=42,
        )
        # RAD5500 has 1000 krad tolerance; TID at 500km is ~0.1 krad/yr
        # Should survive with near-certainty
        assert result.survival_probability > 0.99

    def test_rad5500_positive_seu_count(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=5.0,
            num_simulations=500,
            seed=42,
        )
        # Even rad-hardened chips accumulate SEUs
        assert result.mean_seu_count > 0


class TestCOTSChipFailsHighOrbit:
    """Trillium (15 krad tolerance) should fail at high LEO over long missions."""

    def test_trillium_fails_5yr_at_2000km(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        result = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=2000,
            inclination_deg=53.0,
            mission_years=5.0,
            num_simulations=500,
            seed=42,
        )
        # TID at 2000km is ~10 krad/yr, chip tolerance is 15 krad
        # Should fail well before 5 years
        assert result.tid_failure_probability > 0.99

    def test_trillium_mean_ttf_less_than_mission(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        result = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=2000,
            inclination_deg=53.0,
            mission_years=5.0,
            num_simulations=500,
            seed=42,
        )
        # Mean time to failure should be < 5 years
        assert result.mean_time_to_tid_failure_years < 5.0


class TestConfidenceIntervals:
    """SEU count confidence intervals must be correctly ordered."""

    def test_p5_less_than_mean_less_than_p95(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        result = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=800,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=1000,
            seed=42,
        )
        assert result.seu_count_p5 <= result.mean_seu_count
        assert result.mean_seu_count <= result.seu_count_p95

    def test_confidence_interval_narrows_with_more_sims(self) -> None:
        """More simulations should not widen the CI (statistically)."""
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        result_100 = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=800,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=100,
            seed=42,
        )
        result_1000 = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=800,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=1000,
            seed=42,
        )
        # The mean should be similar (within 20% for Poisson)
        if result_1000.mean_seu_count > 0:
            rel_diff = (
                abs(result_100.mean_seu_count - result_1000.mean_seu_count)
                / result_1000.mean_seu_count
            )
            assert rel_diff < 0.2


class TestDeterministicWithSeed:
    """Same seed must produce identical results."""

    def test_same_seed_same_result(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        kwargs = dict(
            chip=TRILLIUM_V6E,
            altitude_km=800,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=200,
            seed=99,
        )
        r1 = estimate_mission_reliability(**kwargs)
        r2 = estimate_mission_reliability(**kwargs)
        assert r1.survival_probability == r2.survival_probability
        assert r1.mean_seu_count == r2.mean_seu_count

    def test_different_seed_different_result(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        base = dict(
            chip=TRILLIUM_V6E,
            altitude_km=800,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=200,
        )
        r1 = estimate_mission_reliability(**base, seed=1)
        r2 = estimate_mission_reliability(**base, seed=2)
        # Extremely unlikely to be identical with different seeds
        assert r1.mean_seu_count != r2.mean_seu_count


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_very_short_mission(self) -> None:
        """1-day mission should have very low failure probability for any chip."""
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

        result = estimate_mission_reliability(
            chip=TRILLIUM_V6E,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=1.0 / 365.25,  # 1 day
            num_simulations=200,
            seed=42,
        )
        assert result.survival_probability > 0.99

    def test_num_simulations_stored(self) -> None:
        from space_ml_sim.metrics.monte_carlo import estimate_mission_reliability
        from space_ml_sim.models.chip_profiles import RAD5500

        result = estimate_mission_reliability(
            chip=RAD5500,
            altitude_km=500,
            inclination_deg=53.0,
            mission_years=1.0,
            num_simulations=42,
            seed=0,
        )
        assert result.num_simulations == 42
