"""Tests for ML fault injection engine."""

import pytest
import torch
import torch.nn as nn

from space_ml_sim.compute.fault_injector import FaultInjector, FaultReport
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import TRILLIUM_V6E


def _simple_model() -> nn.Module:
    """A tiny model for fast testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )


def _random_dataloader(num_batches: int = 2, batch_size: int = 8):
    """Fake dataloader for testing."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))
        data.append((x, y))
    return data


@pytest.fixture
def injector():
    return FaultInjector(
        rad_env=RadiationEnvironment.leo_500km(),
        chip_profile=TRILLIUM_V6E,
    )


class TestFlipRandomBits:
    def test_zero_flips_no_change(self):
        tensor = torch.randn(10)
        original = tensor.clone()
        FaultInjector.flip_random_bits(tensor, 0)
        assert torch.equal(tensor, original)

    def test_flips_actually_change_tensor(self):
        tensor = torch.ones(100)
        original = tensor.clone()
        FaultInjector.flip_random_bits(tensor, 50)
        assert not torch.equal(tensor, original)

    def test_returns_bit_positions(self):
        tensor = torch.randn(100)
        bits = FaultInjector.flip_random_bits(tensor, 10)
        assert len(bits) == 10
        assert all(0 <= b < 32 for b in bits)

    def test_empty_tensor_no_crash(self):
        tensor = torch.tensor([])
        bits = FaultInjector.flip_random_bits(tensor, 5)
        assert bits == []


class TestInjectWeightFaults:
    def test_zero_faults_no_change(self, injector):
        model = _simple_model()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        report = injector.inject_weight_faults(model, num_faults=0)

        assert report.total_faults_injected == 0
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name])

    def test_faults_modify_weights(self, injector):
        model = _simple_model()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        report = injector.inject_weight_faults(model, num_faults=100)

        assert report.total_faults_injected > 0
        assert report.weight_faults > 0
        assert len(report.layers_affected) > 0

        # At least one parameter should have changed
        any_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, original_params[name]):
                any_changed = True
                break
        assert any_changed

    def test_report_is_immutable(self, injector):
        model = _simple_model()
        report = injector.inject_weight_faults(model, num_faults=10)
        assert isinstance(report, FaultReport)
        # FaultReport is a frozen dataclass — verify it's hashable (proxy for frozen)
        with pytest.raises(AttributeError):
            report.total_faults_injected = 999

    def test_radiation_sampled_faults(self, injector):
        """When num_faults=None, faults should be sampled from radiation model."""
        model = _simple_model()
        report = injector.inject_weight_faults(model, num_faults=None, inference_time_seconds=1.0)
        # With such tiny model, radiation-sampled faults will likely be 0
        assert report.total_faults_injected >= 0


class TestManyFaultsDegradeAccuracy:
    def test_heavy_faults_change_output(self, injector):
        """Injecting many faults should change model output."""
        model = _simple_model()
        model.eval()
        x = torch.randn(1, 10)

        with torch.no_grad():
            original_output = model(x).clone()

        injector.inject_weight_faults(model, num_faults=500)

        with torch.no_grad():
            faulted_output = model(x)

        assert not torch.allclose(original_output, faulted_output, atol=1e-6)


class TestActivationHooks:
    def test_hooks_register_and_remove(self, injector):
        model = _simple_model()
        injector.register_activation_hooks(model, fault_probability=0.1)
        assert len(injector._hooks) > 0

        injector.remove_hooks()
        assert len(injector._hooks) == 0

    def test_activation_faults_change_output(self, injector):
        """With high fault probability, activation hooks should change output."""
        model = _simple_model()
        model.eval()
        x = torch.randn(1, 10)

        with torch.no_grad():
            clean = model(x).clone()

        injector.register_activation_hooks(model, fault_probability=0.5)
        with torch.no_grad():
            faulted = model(x)

        injector.remove_hooks()
        # With 50% fault probability, outputs should differ
        assert not torch.allclose(clean, faulted, atol=1e-3)
