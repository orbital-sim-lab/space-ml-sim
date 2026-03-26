"""Tests for InferenceNode: run_inference, can_run."""

from __future__ import annotations

import torch
import torch.nn as nn

from space_ml_sim.compute.inference_node import InferenceNode, InferenceResult, NodeStatus
from space_ml_sim.models.chip_profiles import ChipProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHIP = ChipProfile(
    name="Test Chip",
    node_nm=28,
    tdp_watts=10.0,
    max_temp_c=80.0,
    seu_cross_section_cm2=1e-14,
    tid_tolerance_krad=50.0,
    compute_tops=1.0,
    memory_bits=256 * 8 * 1024 * 1024,  # 256 MB
)


def _make_classifier(in_features: int = 8, num_classes: int = 4) -> nn.Module:
    """Tiny classification model: Linear → ReLU → Linear."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(in_features, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def _make_node(model: nn.Module | None = None, chip: ChipProfile | None = None) -> InferenceNode:
    return InferenceNode(
        model=model or _make_classifier(),
        chip_profile=chip or _CHIP,
    )


# ---------------------------------------------------------------------------
# run_inference: predictions length
# ---------------------------------------------------------------------------


class TestRunInferencePredictionsLength:
    def test_single_sample_returns_one_prediction(self):
        node = _make_node()
        x = torch.randn(1, 8)
        result = node.run_inference(x)
        assert isinstance(result, InferenceResult)
        assert len(result.predictions) == 1

    def test_batch_of_four_returns_four_predictions(self):
        node = _make_node()
        x = torch.randn(4, 8)
        result = node.run_inference(x)
        assert len(result.predictions) == 4

    def test_batch_of_sixteen_returns_sixteen_predictions(self):
        node = _make_node()
        x = torch.randn(16, 8)
        result = node.run_inference(x)
        assert len(result.predictions) == 16

    def test_predictions_are_valid_class_indices(self):
        """Each prediction must be in [0, num_classes)."""
        num_classes = 4
        node = _make_node(model=_make_classifier(num_classes=num_classes))
        x = torch.randn(10, 8)
        result = node.run_inference(x)
        for pred in result.predictions:
            assert 0 <= pred < num_classes, f"Invalid prediction index: {pred}"

    def test_predictions_are_integers(self):
        node = _make_node()
        x = torch.randn(5, 8)
        result = node.run_inference(x)
        for pred in result.predictions:
            assert isinstance(pred, int)

    def test_result_is_inference_result_instance(self):
        node = _make_node()
        x = torch.randn(2, 8)
        result = node.run_inference(x)
        assert isinstance(result, InferenceResult)

    def test_status_is_idle_after_inference(self):
        node = _make_node()
        x = torch.randn(3, 8)
        result = node.run_inference(x)
        assert result.status == NodeStatus.IDLE

    def test_inference_count_increments(self):
        node = _make_node()
        x = torch.randn(2, 8)
        assert node.inference_count == 0
        node.run_inference(x)
        assert node.inference_count == 1
        node.run_inference(x)
        assert node.inference_count == 2

    def test_repeated_inference_same_input_same_predictions(self):
        """Deterministic model with no_grad must produce identical results."""
        node = _make_node()
        x = torch.randn(4, 8)
        result_a = node.run_inference(x)
        result_b = node.run_inference(x)
        assert result_a.predictions == result_b.predictions

    def test_large_batch_returns_correct_length(self):
        node = _make_node()
        batch_size = 128
        x = torch.randn(batch_size, 8)
        result = node.run_inference(x)
        assert len(result.predictions) == batch_size


# ---------------------------------------------------------------------------
# can_run: True when power and temperature within limits
# ---------------------------------------------------------------------------


class TestCanRunTrue:
    def test_exact_tdp_and_max_temp_returns_true(self):
        """Boundary values: power == TDP and temp == max_temp → True."""
        node = _make_node()
        assert (
            node.can_run(
                power_available_w=_CHIP.tdp_watts,
                temperature_c=_CHIP.max_temp_c,
            )
            is True
        )

    def test_excess_power_and_cool_temp_returns_true(self):
        node = _make_node()
        assert node.can_run(power_available_w=100.0, temperature_c=25.0) is True

    def test_exactly_at_limits_returns_true(self):
        chip = ChipProfile(
            name="Limit Chip",
            node_nm=45,
            tdp_watts=15.0,
            max_temp_c=125.0,
            seu_cross_section_cm2=1e-15,
            tid_tolerance_krad=1000.0,
            compute_tops=0.001,
            memory_bits=256 * 8 * 1024 * 1024,
        )
        node = _make_node(chip=chip)
        assert node.can_run(power_available_w=15.0, temperature_c=125.0) is True


# ---------------------------------------------------------------------------
# can_run: False when power too low
# ---------------------------------------------------------------------------


class TestCanRunFalseLowPower:
    def test_zero_power_returns_false(self):
        node = _make_node()
        assert node.can_run(power_available_w=0.0, temperature_c=25.0) is False

    def test_power_just_below_tdp_returns_false(self):
        node = _make_node()
        assert (
            node.can_run(
                power_available_w=_CHIP.tdp_watts - 0.01,
                temperature_c=25.0,
            )
            is False
        )

    def test_negative_power_returns_false(self):
        node = _make_node()
        assert node.can_run(power_available_w=-5.0, temperature_c=25.0) is False

    def test_power_one_watt_below_tdp_returns_false(self):
        node = _make_node()
        assert (
            node.can_run(
                power_available_w=_CHIP.tdp_watts - 1.0,
                temperature_c=0.0,
            )
            is False
        )


# ---------------------------------------------------------------------------
# can_run: False when temperature too high
# ---------------------------------------------------------------------------


class TestCanRunFalseHighTemp:
    def test_temperature_just_above_max_returns_false(self):
        node = _make_node()
        assert (
            node.can_run(
                power_available_w=100.0,
                temperature_c=_CHIP.max_temp_c + 0.01,
            )
            is False
        )

    def test_very_high_temperature_returns_false(self):
        node = _make_node()
        assert node.can_run(power_available_w=100.0, temperature_c=1000.0) is False

    def test_both_power_low_and_temp_high_returns_false(self):
        node = _make_node()
        assert node.can_run(power_available_w=0.0, temperature_c=999.0) is False

    def test_good_power_but_one_degree_over_max_returns_false(self):
        node = _make_node()
        assert (
            node.can_run(
                power_available_w=_CHIP.tdp_watts * 2,
                temperature_c=_CHIP.max_temp_c + 1.0,
            )
            is False
        )


# ---------------------------------------------------------------------------
# can_run: chip profile boundary variations
# ---------------------------------------------------------------------------


class TestCanRunWithDifferentChips:
    def _chip_with(self, tdp: float, max_temp: float) -> ChipProfile:
        return ChipProfile(
            name="Custom",
            node_nm=28,
            tdp_watts=tdp,
            max_temp_c=max_temp,
            seu_cross_section_cm2=1e-14,
            tid_tolerance_krad=50.0,
            compute_tops=1.0,
            memory_bits=256 * 8 * 1024 * 1024,
        )

    def test_high_tdp_chip_requires_more_power(self):
        chip = self._chip_with(tdp=300.0, max_temp=125.0)
        node = _make_node(chip=chip)
        assert node.can_run(power_available_w=299.99, temperature_c=50.0) is False
        assert node.can_run(power_available_w=300.0, temperature_c=50.0) is True

    def test_low_max_temp_chip_fails_sooner(self):
        chip = self._chip_with(tdp=5.0, max_temp=50.0)
        node = _make_node(chip=chip)
        assert node.can_run(power_available_w=10.0, temperature_c=50.1) is False
        assert node.can_run(power_available_w=10.0, temperature_c=50.0) is True
