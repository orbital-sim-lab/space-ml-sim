"""TDD tests for automated TMR recommendation engine.

Written FIRST before implementation (RED phase).

The TMR recommender:
- Analyzes per-layer vulnerability from sensitivity data
- Recommends optimal TMR configuration (which layers to protect)
- Computes cost/benefit analysis (compute overhead vs accuracy recovery)
- Supports budget-constrained optimization (max compute multiplier)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _make_model() -> nn.Sequential:
    """Small model for testing."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


# Simulated sensitivity data (layer_name -> accuracy_drop)
MOCK_SENSITIVITY = {
    "0.weight": 0.35,  # Most vulnerable
    "0.bias": 0.02,
    "2.weight": 0.20,  # Second most vulnerable
    "2.bias": 0.01,
    "4.weight": 0.08,  # Moderate
    "4.bias": 0.005,
}


class TestTMRRecommendation:
    """Recommender must identify optimal layer protection."""

    def test_recommend_returns_result(self) -> None:
        from space_ml_sim.compute.tmr_recommender import (
            TMRRecommendation,
            recommend_tmr,
        )

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
        )
        assert isinstance(rec, TMRRecommendation)

    def test_protects_most_vulnerable_layers(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
        )
        # Most vulnerable layer should always be protected
        assert "0.weight" in rec.protected_layers

    def test_sorted_by_vulnerability(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
        )
        # Ranked layers should be in descending order of vulnerability
        drops = [MOCK_SENSITIVITY[l] for l in rec.ranked_layers]
        assert drops == sorted(drops, reverse=True)


class TestBudgetConstraint:
    """Recommender must respect compute budget constraints."""

    def test_budget_limits_protected_layers(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        # Very tight budget: only protect the single most impactful layer
        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=1.1,
        )
        # With tight budget, fewer layers should be protected
        rec_full = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=3.0,
        )
        assert len(rec.protected_layers) <= len(rec_full.protected_layers)

    def test_full_budget_protects_all_vulnerable(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=3.0,
        )
        # With full TMR budget, all layers with meaningful vulnerability should be protected
        meaningful = {k for k, v in MOCK_SENSITIVITY.items() if v >= 0.01}
        assert meaningful.issubset(rec.protected_layers)

    def test_compute_multiplier_reported(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=3.0,
        )
        assert 1.0 <= rec.compute_multiplier <= 3.0


class TestCostBenefitAnalysis:
    """Recommendation must include cost/benefit metrics."""

    def test_has_expected_recovery(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
        )
        # Expected recovery is the sum of sensitivity drops for protected layers
        assert rec.expected_accuracy_recovery > 0

    def test_has_cost_benefit_ratio(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
        )
        # Recovery per unit compute cost
        assert rec.cost_benefit_ratio > 0

    def test_more_budget_more_recovery(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec_low = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=1.2,
        )
        rec_high = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=3.0,
        )
        assert rec_high.expected_accuracy_recovery >= rec_low.expected_accuracy_recovery

    def test_unprotected_layers_reported(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            max_compute_multiplier=1.2,
        )
        # Should report which vulnerable layers were left unprotected
        assert len(rec.unprotected_layers) > 0
        assert rec.residual_risk >= 0


class TestThresholdFiltering:
    """Recommender should filter out negligible layers."""

    def test_min_threshold_filters_noise(self) -> None:
        from space_ml_sim.compute.tmr_recommender import recommend_tmr

        rec = recommend_tmr(
            model=_make_model(),
            sensitivity=MOCK_SENSITIVITY,
            min_vulnerability_threshold=0.05,
            max_compute_multiplier=3.0,
        )
        # Layers below threshold should not be protected
        assert "0.bias" not in rec.protected_layers  # 0.02 < 0.05
        assert "4.bias" not in rec.protected_layers  # 0.005 < 0.05
        # Layers above threshold should be
        assert "0.weight" in rec.protected_layers  # 0.35 >= 0.05
