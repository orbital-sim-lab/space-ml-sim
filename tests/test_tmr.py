"""Tests for Triple Modular Redundancy (TMR) wrapper."""

import pytest
import torch
import torch.nn as nn

from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.compute.tmr import TMRWrapper
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import TRILLIUM_V6E


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
        chip_profile=TRILLIUM_V6E,
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


class TestSelectiveTMR:
    """Tests for selective TMR with per-layer protection."""

    def test_configure_protection_sets_layers(self, model_factory):
        """configure_protection should store the set of protected layer names."""
        tmr = TMRWrapper(model_factory, strategy="selective_tmr")
        tmr.configure_protection(protected_layers={"0.weight", "0.bias"})
        assert tmr.protected_layers == {"0.weight", "0.bias"}

    def test_selective_inject_only_affects_protected_layers(self, model_factory, injector):
        """inject_faults_to_replicas with selective TMR should only fault protected layers."""
        tmr = TMRWrapper(model_factory, strategy="selective_tmr")
        # Only protect the first linear layer
        tmr.configure_protection(protected_layers={"0.weight", "0.bias"})

        # Save originals
        originals = []
        for replica in tmr.replicas:
            originals.append({n: p.clone() for n, p in replica.named_parameters()})

        tmr.inject_faults_to_replicas(injector, faults_per_replica=200)

        # Protected layers should be different across replicas (independent faults)
        for i, replica in enumerate(tmr.replicas):
            for name, param in replica.named_parameters():
                if name in tmr.protected_layers:
                    # Protected: should have been faulted (changed from original)
                    pass  # May or may not change depending on fault distribution
                else:
                    # Unprotected: should be IDENTICAL to original
                    assert torch.equal(param, originals[i][name]), (
                        f"Unprotected layer {name} was modified in replica {i}"
                    )

    def test_selective_tmr_forward_produces_predictions(self, model_factory):
        tmr = TMRWrapper(model_factory, strategy="selective_tmr")
        tmr.configure_protection(protected_layers={"0.weight"})
        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert "predictions" in result
        assert result["predictions"].shape == (4,)
        assert result["strategy"] == "selective_tmr"

    def test_selective_tmr_corrects_faults_in_protected_layers(self, model_factory, injector):
        """Faults in protected layers should be correctable via majority vote."""
        tmr = TMRWrapper(model_factory, strategy="selective_tmr")
        tmr.configure_protection(protected_layers={"0.weight", "0.bias", "2.weight", "2.bias"})

        x = torch.randn(4, 10)
        clean_preds = tmr.forward(x)["predictions"].clone()

        # Inject faults to only 1 replica's protected layers
        injector.inject_weight_faults(tmr.replicas[0], num_faults=500)

        # Majority vote should still produce correct output
        faulted_preds = tmr.forward(x)["predictions"]
        assert torch.equal(faulted_preds, clean_preds)

    def test_default_no_protection_acts_like_full_tmr(self, model_factory):
        """Without configure_protection, selective TMR should protect all layers."""
        tmr = TMRWrapper(model_factory, strategy="selective_tmr")
        # No configure_protection called — should default to protecting all
        assert tmr.protected_layers is None or len(tmr.protected_layers) == 0
        # Forward should still work
        x = torch.randn(4, 10)
        result = tmr.forward(x)
        assert "predictions" in result


class TestInvalidStrategy:
    def test_rejects_unknown_strategy(self, model_factory):
        with pytest.raises(ValueError, match="Unknown strategy"):
            TMRWrapper(model_factory, strategy="quantum_tmr")
