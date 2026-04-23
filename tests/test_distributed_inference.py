"""TDD tests for distributed inference across constellation.

Written FIRST before implementation (RED phase).
Tests should FAIL until compute/distributed.py is created.

The distributed inference executor:
- Partitions a model's inference work across N satellites
- Accounts for ISL transfer time when sending activations between nodes
- Reports total end-to-end latency (compute + communication)
- Handles node failures by rerouting to available nodes
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _model_factory() -> nn.Module:
    """Tiny model for testing."""
    return nn.Sequential(
        nn.Linear(20, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


# ---------------------------------------------------------------------------
# Test: DistributedInferenceTask structure
# ---------------------------------------------------------------------------


class TestDistributedTaskCreation:
    """Must be able to create a distributed inference task."""

    def test_creates_task(self) -> None:
        from space_ml_sim.compute.distributed import DistributedInferenceTask

        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=3,
            input_shape=(1, 20),
        )
        assert task.num_partitions == 3

    def test_single_partition_is_valid(self) -> None:
        from space_ml_sim.compute.distributed import DistributedInferenceTask

        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(1, 20),
        )
        assert task.num_partitions == 1


class TestDistributedExecutor:
    """Distributed executor must schedule work across nodes with ISL costs."""

    def test_execute_returns_result(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
            DistributedResult,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
            "S2": (6771.0, 0.0, 100.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=2,
            input_shape=(4, 20),
        )

        executor = DistributedExecutor(network=network)
        result = executor.execute(
            task=task,
            source_node="S0",
            worker_nodes=["S1", "S2"],
        )

        assert isinstance(result, DistributedResult)

    def test_result_has_required_fields(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
        )
        executor = DistributedExecutor(network=network)
        result = executor.execute(
            task=task,
            source_node="S0",
            worker_nodes=["S1"],
        )

        assert hasattr(result, "predictions")
        assert hasattr(result, "total_latency_ms")
        assert hasattr(result, "compute_latency_ms")
        assert hasattr(result, "communication_latency_ms")
        assert hasattr(result, "nodes_used")

    def test_predictions_are_valid(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
        )
        executor = DistributedExecutor(network=network)
        result = executor.execute(
            task=task,
            source_node="S0",
            worker_nodes=["S1"],
        )

        assert len(result.predictions) == 4  # batch size
        assert all(0 <= p < 10 for p in result.predictions)  # 10 classes


class TestLatencyAccounting:
    """Latency must correctly account for compute + communication."""

    def test_total_latency_is_sum(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
        )
        executor = DistributedExecutor(network=network)
        result = executor.execute(
            task=task,
            source_node="S0",
            worker_nodes=["S1"],
        )

        expected = result.compute_latency_ms + result.communication_latency_ms
        assert abs(result.total_latency_ms - expected) < 0.01

    def test_more_hops_more_latency(self) -> None:
        """Distributing across distant nodes should increase communication cost."""
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        # Close nodes
        positions_close = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        # Farther nodes
        positions_far = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 4000.0, 0.0),
        }

        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
        )

        net_close = ISLNetwork.from_positions(positions_close, max_range_km=5000.0)
        net_far = ISLNetwork.from_positions(positions_far, max_range_km=5000.0)

        exec_close = DistributedExecutor(network=net_close)
        exec_far = DistributedExecutor(network=net_far)

        result_close = exec_close.execute(task=task, source_node="S0", worker_nodes=["S1"])
        result_far = exec_far.execute(task=task, source_node="S0", worker_nodes=["S1"])

        assert result_far.communication_latency_ms > result_close.communication_latency_ms


class TestDataParallelism:
    """Data-parallel mode must split batches across workers."""

    def test_batch_split_across_workers(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
            "S2": (6771.0, 0.0, 50.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=2,
            input_shape=(8, 20),  # 8 samples, split into 2 partitions of 4
        )

        executor = DistributedExecutor(network=network)
        result = executor.execute(
            task=task,
            source_node="S0",
            worker_nodes=["S1", "S2"],
        )

        # Should still get predictions for all 8 samples
        assert len(result.predictions) == 8
        assert len(result.nodes_used) == 2


class TestDeterministicWithSeed:
    """Same seed must produce identical results."""

    def test_deterministic_execution(self) -> None:
        from space_ml_sim.compute.distributed import (
            DistributedInferenceTask,
            DistributedExecutor,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        task = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
            seed=42,
        )
        executor = DistributedExecutor(network=network)

        r1 = executor.execute(task=task, source_node="S0", worker_nodes=["S1"])
        # Re-create task with same seed
        task2 = DistributedInferenceTask(
            model_factory=_model_factory,
            num_partitions=1,
            input_shape=(4, 20),
            seed=42,
        )
        r2 = executor.execute(task=task2, source_node="S0", worker_nodes=["S1"])

        assert r1.predictions == r2.predictions
