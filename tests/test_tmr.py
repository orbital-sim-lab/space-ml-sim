"""Tests for Triple Modular Redundancy (TMR) wrapper."""

import pytest
import torch
import torch.nn as nn

from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.compute.tmr import TMRWrapper
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import GOOGLE_TRILLIUM_V6E


def _simple_model_factory():
    """Factory for a tiny deterministic model."""

    def factory():
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        # Set deterministic weights
        torch.manual_seed(42)
        for p in model.parameters():
            nn.init.normal_(p, mean=0, std=0.1)
        return model

    return factory


@pytest.fixture
def model_factory():
    return _simple_model_factory()


@pytest.fixture
def injector():
    return FaultInjector(
        rad_env=RadiationEnvironment.leo_500km(),
        chip_profile=GOOGLE_TRILLIUM_V6E,
    )


class TestFullTMR:
    def test_produces_predictions(self, model_factory):
        tmr = TMRWrapper(model_factory, strategy="full_tmr")
        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert "predictions" in result
        assert result["predictions"].shape == (4,)

    def test_no_disagreements_without_faults(self, model_factory):
        """Identical replicas should have zero disagreements."""
        tmr = TMRWrapper(model_factory, strategy="full_tmr")
        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert result["disagreements"] == 0

    def test_tmr_corrects_single_replica_faults(self, model_factory, injector):
        """If only 1 of 3 replicas is faulted, TMR should recover correct output."""
        tmr = TMRWrapper(model_factory, strategy="full_tmr")
        x = torch.randn(4, 10)

        # Get clean output
        clean_result = tmr.forward(x)
        clean_preds = clean_result["predictions"].clone()

        # Inject heavy faults into just 1 replica
        injector.inject_weight_faults(tmr.replicas[0], num_faults=500)

        # TMR should still produce correct output via majority vote
        faulted_result = tmr.forward(x)
        assert torch.equal(faulted_result["predictions"], clean_preds)

    def test_disagreements_detected(self, model_factory, injector):
        """Faults in one replica should produce disagreements."""
        tmr = TMRWrapper(model_factory, strategy="full_tmr")
        injector.inject_weight_faults(tmr.replicas[0], num_faults=500)
        x = torch.randn(8, 10)
        result = tmr.forward(x)
        assert result["disagreements"] > 0


class TestCheckpointRollback:
    def test_produces_predictions(self, model_factory):
        tmr = TMRWrapper(model_factory, strategy="checkpoint_rollback")
        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert "predictions" in result
        assert result["predictions"].shape == (4,)

    def test_detects_nan_anomaly(self, model_factory):
        """If model produces NaN, should detect and roll back."""
        tmr = TMRWrapper(model_factory, strategy="checkpoint_rollback")

        # Corrupt model to produce NaN
        with torch.no_grad():
            for p in tmr.model.parameters():
                p.fill_(float("nan"))

        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert result["anomaly_detected"]
        assert result["rolled_back"]


class TestInjectFaultsToReplicas:
    def test_injects_to_all_replicas(self, model_factory, injector):
        tmr = TMRWrapper(model_factory, strategy="full_tmr")

        # Save original weights
        originals = []
        for replica in tmr.replicas:
            originals.append({n: p.clone() for n, p in replica.named_parameters()})

        tmr.inject_faults_to_replicas(injector, faults_per_replica=100)

        # Each replica should be different from its original
        for i, replica in enumerate(tmr.replicas):
            any_changed = False
            for name, param in replica.named_parameters():
                if not torch.equal(param, originals[i][name]):
                    any_changed = True
                    break
            assert any_changed

    def test_raises_for_checkpoint_strategy(self, model_factory, injector):
        tmr = TMRWrapper(model_factory, strategy="checkpoint_rollback")
        with pytest.raises(RuntimeError):
            tmr.inject_faults_to_replicas(injector, faults_per_replica=10)


class TestInvalidStrategy:
    def test_rejects_unknown_strategy(self, model_factory):
        with pytest.raises(ValueError, match="Unknown strategy"):
            TMRWrapper(model_factory, strategy="quantum_tmr")
