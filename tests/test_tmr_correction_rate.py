"""Benchmark TMR correction rate against theoretical 2-of-3 voting probability.

Theoretical background:
- TMR uses 3 independent replicas with majority voting
- If each replica has independent fault probability p (per-sample),
  TMR fails only when 2 or 3 replicas are wrong on the same sample
- P(TMR_fail) = 3*p^2*(1-p) + p^3 = 3p^2 - 2p^3
- For small p, TMR reduces error rate from p to ~3p^2

This test verifies the TMR implementation matches these theoretical bounds.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.compute.tmr import TMRWrapper
from space_ml_sim.models.chip_profiles import TRILLIUM_V6E


def _model_factory() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


class TestTMRCorrectionRate:
    """TMR must correct faults at or near theoretical 2-of-3 voting rate."""

    def test_tmr_improves_over_unprotected(self) -> None:
        """TMR error rate must be lower than single-replica error rate."""
        rad_env = RadiationEnvironment.leo_500km()
        injector = FaultInjector(rad_env=rad_env, chip_profile=TRILLIUM_V6E)

        torch.manual_seed(42)
        test_input = torch.randn(64, 20)

        baseline = _model_factory().eval()
        with torch.no_grad():
            baseline_preds = baseline(test_input).argmax(dim=1)

        faults = 20
        n_trials = 50

        # Unprotected error rate
        unprotected_errors = 0
        unprotected_total = 0
        for _ in range(n_trials):
            m = copy.deepcopy(baseline)
            injector.inject_weight_faults(m, num_faults=faults)
            with torch.no_grad():
                preds = m(test_input).argmax(dim=1)
            unprotected_errors += (preds != baseline_preds).sum().item()
            unprotected_total += len(baseline_preds)

        # TMR error rate
        tmr_errors = 0
        tmr_total = 0
        for _ in range(n_trials):
            tmr = TMRWrapper(_model_factory, strategy="full_tmr")
            # Load baseline weights into all replicas, then inject faults
            for replica in tmr.replicas:
                replica.load_state_dict(baseline.state_dict())
            tmr.inject_faults_to_replicas(injector, faults_per_replica=faults)

            result = tmr.forward(test_input)
            tmr_preds = result["predictions"]
            tmr_errors += (tmr_preds != baseline_preds).sum().item()
            tmr_total += len(baseline_preds)

        p_unprotected = unprotected_errors / unprotected_total
        p_tmr = tmr_errors / tmr_total

        # TMR should reduce error rate
        assert p_tmr <= p_unprotected, (
            f"TMR error rate ({p_tmr:.4f}) should be <= unprotected ({p_unprotected:.4f})"
        )

    def test_tmr_error_rate_near_theoretical(self) -> None:
        """TMR error rate should be in the neighborhood of 3p^2 - 2p^3.

        We use a generous tolerance because:
        - Faults across replicas are not perfectly independent bit-flips
          (same layers are proportionally targeted)
        - We're comparing per-sample error, not per-bit
        """
        rad_env = RadiationEnvironment.leo_500km()
        injector = FaultInjector(rad_env=rad_env, chip_profile=TRILLIUM_V6E)

        torch.manual_seed(99)
        test_input = torch.randn(128, 20)

        baseline = _model_factory().eval()
        with torch.no_grad():
            baseline_preds = baseline(test_input).argmax(dim=1)

        faults = 15
        n_trials = 100

        # Measure single-replica error rate
        single_errors = 0
        single_total = 0
        for _ in range(n_trials):
            m = copy.deepcopy(baseline)
            injector.inject_weight_faults(m, num_faults=faults)
            with torch.no_grad():
                preds = m(test_input).argmax(dim=1)
            single_errors += (preds != baseline_preds).sum().item()
            single_total += len(baseline_preds)

        p = single_errors / single_total

        # Theoretical TMR failure rate
        p_tmr_theoretical = 3 * p**2 - 2 * p**3

        # Measure TMR error rate
        tmr_errors = 0
        tmr_total = 0
        for _ in range(n_trials):
            tmr = TMRWrapper(_model_factory, strategy="full_tmr")
            for replica in tmr.replicas:
                replica.load_state_dict(baseline.state_dict())
            tmr.inject_faults_to_replicas(injector, faults_per_replica=faults)

            result = tmr.forward(test_input)
            tmr_preds = result["predictions"]
            tmr_errors += (tmr_preds != baseline_preds).sum().item()
            tmr_total += len(baseline_preds)

        p_tmr_observed = tmr_errors / tmr_total

        # TMR observed rate should be within 5x of theoretical
        # (generous due to correlated fault injection across layers)
        if p > 0.01:  # Only test when there are enough errors
            assert p_tmr_observed < p, "TMR must improve over single replica"
            # The observed rate should be in the right order of magnitude
            upper_bound = max(p_tmr_theoretical * 5, 0.02)
            assert p_tmr_observed < upper_bound, (
                f"TMR error rate ({p_tmr_observed:.4f}) too high vs "
                f"theoretical ({p_tmr_theoretical:.4f}), upper bound ({upper_bound:.4f})"
            )

    def test_tmr_unanimous_when_identical_replicas(self) -> None:
        """With identical replicas and no faults, all 3 should agree perfectly."""
        tmr = TMRWrapper(_model_factory, strategy="full_tmr")
        # Make all replicas identical
        state = tmr.replicas[0].state_dict()
        for replica in tmr.replicas[1:]:
            replica.load_state_dict(state)
        x = torch.randn(32, 20)
        result = tmr.forward(x)
        assert result["disagreements"] == 0
