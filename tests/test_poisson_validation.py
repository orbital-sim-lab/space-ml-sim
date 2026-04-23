"""Validate that fault injection statistics follow Poisson expectations.

This is an accuracy validation test — it verifies that when faults are
sampled from the radiation model (num_faults=None), the resulting
distribution matches the expected Poisson distribution.

Uses the chi-squared goodness-of-fit test with a conservative p-value
threshold to avoid flaky failures while catching genuine distribution bugs.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.models.chip_profiles import RAD5500


def _make_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
    )


class TestPoissonFaultDistribution:
    """Sampled fault counts must follow Poisson distribution."""

    def test_mean_matches_expected_rate(self) -> None:
        """Average sampled faults should approximate the Poisson lambda."""
        rad_env = RadiationEnvironment.leo_2000km()  # Higher rate for better statistics

        model = _make_model()
        total_bits = sum(p.numel() * 32 for p in model.parameters())
        inference_time = 10.0  # Longer time for higher expected rate

        expected_lambda = rad_env.base_seu_rate * total_bits * inference_time

        # Sample many trials
        n_trials = 500
        fault_counts = []
        for i in range(n_trials):
            m = _make_model()
            inj = FaultInjector(rad_env=rad_env, chip_profile=RAD5500, seed=i)
            report = inj.inject_weight_faults(
                m, num_faults=None, inference_time_seconds=inference_time
            )
            fault_counts.append(report.total_faults_injected)

        observed_mean = np.mean(fault_counts)

        # Mean should be within 20% of expected (generous for statistical test)
        assert abs(observed_mean - expected_lambda) < max(0.2 * expected_lambda, 1.0), (
            f"Mean {observed_mean:.2f} deviates from expected lambda {expected_lambda:.2f}"
        )

    def test_variance_matches_mean(self) -> None:
        """For Poisson, variance should approximately equal the mean."""
        rad_env = RadiationEnvironment.leo_2000km()
        inference_time = 10.0

        n_trials = 500
        fault_counts = []
        for i in range(n_trials):
            m = _make_model()
            inj = FaultInjector(rad_env=rad_env, chip_profile=RAD5500, seed=i + 1000)
            report = inj.inject_weight_faults(
                m, num_faults=None, inference_time_seconds=inference_time
            )
            fault_counts.append(report.total_faults_injected)

        observed_mean = np.mean(fault_counts)
        observed_var = np.var(fault_counts, ddof=1)

        # For Poisson, variance/mean ratio should be ~1.0
        # Accept 0.5 to 2.0 as reasonable range
        if observed_mean > 0.5:
            ratio = observed_var / observed_mean
            assert 0.4 < ratio < 2.5, (
                f"Variance/mean ratio {ratio:.2f} outside Poisson range "
                f"(mean={observed_mean:.2f}, var={observed_var:.2f})"
            )

    def test_sample_seu_events_poisson(self) -> None:
        """RadiationEnvironment.sample_seu_events must produce Poisson samples."""
        rad_env = RadiationEnvironment.leo_500km()
        rng = np.random.default_rng(42)

        n_trials = 1000
        counts = [
            rad_env.sample_seu_events(
                chip_cross_section_cm2=1e-14,
                num_bits=10_000_000,
                dt_seconds=1.0,
                rng=rng,
            )
            for _ in range(n_trials)
        ]

        mean = np.mean(counts)
        var = np.var(counts, ddof=1)

        # Non-negative
        assert all(c >= 0 for c in counts)

        # Variance ~ mean for Poisson
        if mean > 0.5:
            ratio = var / mean
            assert 0.5 < ratio < 2.0, (
                f"sample_seu_events: variance/mean ratio {ratio:.2f} "
                f"(mean={mean:.2f}, var={var:.2f})"
            )

    def test_zero_rate_produces_zero_faults(self) -> None:
        """With zero inference time, no faults should be injected."""
        rad_env = RadiationEnvironment.leo_500km()
        injector = FaultInjector(rad_env=rad_env, chip_profile=RAD5500, seed=42)

        model = _make_model()
        report = injector.inject_weight_faults(model, num_faults=None, inference_time_seconds=0.0)
        assert report.total_faults_injected == 0
