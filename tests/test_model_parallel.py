"""TDD tests for model-parallel (pipeline) inference across constellation.

Written FIRST before implementation (RED phase).

Model-parallel inference splits a model's layers across multiple satellites,
forming a pipeline where each satellite processes its assigned stage and
forwards activations to the next via ISL links.

Key behaviors:
- A model is partitioned into N sequential stages
- Each stage runs on a different satellite
- Intermediate activations are transferred via ISL
- Total latency = sum(compute per stage) + sum(ISL transfer between stages)
- Pipeline supports batch pipelining for throughput
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _make_sequential_model() -> nn.Sequential:
    """4-layer model that can be split into stages."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )


# ---------------------------------------------------------------------------
# Test: ModelPartition creation
# ---------------------------------------------------------------------------


class TestModelPartition:
    """Model must be splittable into sequential stages."""

    def test_partition_into_stages(self) -> None:
        from space_ml_sim.compute.model_parallel import partition_model

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)
        assert len(stages) == 2

    def test_stages_are_nn_modules(self) -> None:
        from space_ml_sim.compute.model_parallel import partition_model

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)
        for stage in stages:
            assert isinstance(stage, nn.Module)

    def test_stages_compose_to_original(self) -> None:
        """Running all stages sequentially must produce the same output."""
        from space_ml_sim.compute.model_parallel import partition_model

        model = _make_sequential_model()
        model.eval()
        stages = partition_model(model, num_stages=2)

        torch.manual_seed(42)
        x = torch.randn(4, 20)

        with torch.no_grad():
            expected = model(x)

        with torch.no_grad():
            intermediate = x
            for stage in stages:
                intermediate = stage(intermediate)

        assert torch.allclose(expected, intermediate, atol=1e-5)

    def test_single_stage_wraps_entire_model(self) -> None:
        from space_ml_sim.compute.model_parallel import partition_model

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=1)
        assert len(stages) == 1

    def test_more_stages_than_layers_clamps(self) -> None:
        """Requesting more stages than layers should clamp to layer count."""
        from space_ml_sim.compute.model_parallel import partition_model

        model = _make_sequential_model()  # 7 layers
        stages = partition_model(model, num_stages=20)
        assert 1 <= len(stages) <= 7


# ---------------------------------------------------------------------------
# Test: PipelineExecutor
# ---------------------------------------------------------------------------


class TestPipelineExecutor:
    """Pipeline executor must run stages across satellites with ISL costs."""

    def test_execute_returns_result(self) -> None:
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            PipelineResult,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)

        executor = PipelineExecutor(network=network)
        result = executor.execute(
            stages=stages,
            pipeline_nodes=["S0", "S1"],
            input_tensor=torch.randn(4, 20),
        )
        assert isinstance(result, PipelineResult)

    def test_result_has_predictions(self) -> None:
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)

        executor = PipelineExecutor(network=network)
        result = executor.execute(
            stages=stages,
            pipeline_nodes=["S0", "S1"],
            input_tensor=torch.randn(4, 20),
        )
        assert len(result.predictions) == 4
        assert all(0 <= p < 10 for p in result.predictions)

    def test_latency_breakdown(self) -> None:
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)

        executor = PipelineExecutor(network=network)
        result = executor.execute(
            stages=stages,
            pipeline_nodes=["S0", "S1"],
            input_tensor=torch.randn(4, 20),
        )

        assert result.compute_latency_ms > 0
        assert result.communication_latency_ms >= 0
        expected_total = result.compute_latency_ms + result.communication_latency_ms
        assert abs(result.total_latency_ms - expected_total) < 0.01

    def test_more_stages_more_communication(self) -> None:
        """More pipeline stages should increase communication overhead."""
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
            "S2": (6771.0, 0.0, 100.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        x = torch.randn(4, 20)

        model = _make_sequential_model()

        stages_2 = partition_model(model, num_stages=2)
        stages_3 = partition_model(model, num_stages=3)

        executor = PipelineExecutor(network=network)
        r2 = executor.execute(stages=stages_2, pipeline_nodes=["S0", "S1"], input_tensor=x)
        r3 = executor.execute(
            stages=stages_3, pipeline_nodes=["S0", "S1", "S2"], input_tensor=x
        )

        assert r3.communication_latency_ms > r2.communication_latency_ms

    def test_stage_activation_sizes_reported(self) -> None:
        """Result should report activation sizes transferred between stages."""
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        model = _make_sequential_model()
        stages = partition_model(model, num_stages=2)

        executor = PipelineExecutor(network=network)
        result = executor.execute(
            stages=stages,
            pipeline_nodes=["S0", "S1"],
            input_tensor=torch.randn(4, 20),
        )

        assert len(result.activation_sizes_bytes) >= 1
        assert all(s > 0 for s in result.activation_sizes_bytes)


class TestPipelineDeterminism:
    """Pipeline must be deterministic with same inputs."""

    def test_deterministic_output(self) -> None:
        from space_ml_sim.compute.model_parallel import (
            PipelineExecutor,
            partition_model,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        model = _make_sequential_model()
        model.eval()
        stages = partition_model(model, num_stages=2)

        torch.manual_seed(99)
        x = torch.randn(4, 20)

        executor = PipelineExecutor(network=network)
        r1 = executor.execute(stages=stages, pipeline_nodes=["S0", "S1"], input_tensor=x)
        r2 = executor.execute(stages=stages, pipeline_nodes=["S0", "S1"], input_tensor=x)

        assert r1.predictions == r2.predictions
