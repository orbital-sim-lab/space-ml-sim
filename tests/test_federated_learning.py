"""TDD tests for bandwidth-constrained federated learning.

Written FIRST before implementation (RED phase).

Federated learning across a constellation:
- Each satellite trains a local model on its data
- Gradients/weights are aggregated at a central node
- Communication is constrained by ISL bandwidth
- Gradient compression reduces transfer size
- FedAvg aggregation produces a new global model
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _make_model() -> nn.Sequential:
    """Small model for federated learning tests."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )


# ---------------------------------------------------------------------------
# Test: Gradient compression
# ---------------------------------------------------------------------------


class TestGradientCompression:
    """Gradient compression must reduce transfer size."""

    def test_top_k_reduces_size(self) -> None:
        from space_ml_sim.compute.federated import compress_gradients

        grads = {"layer.weight": torch.randn(32, 10)}
        compressed = compress_gradients(grads, method="top_k", ratio=0.1)

        # Compressed representation should have fewer non-zero values
        total_original = sum(g.numel() for g in grads.values())
        total_compressed = sum(
            (g != 0).sum().item() for g in compressed.values()
        )
        assert total_compressed < total_original

    def test_none_compression_preserves_all(self) -> None:
        from space_ml_sim.compute.federated import compress_gradients

        grads = {"layer.weight": torch.randn(32, 10)}
        compressed = compress_gradients(grads, method="none", ratio=1.0)

        assert torch.allclose(grads["layer.weight"], compressed["layer.weight"])

    def test_compression_ratio_controls_sparsity(self) -> None:
        from space_ml_sim.compute.federated import compress_gradients

        grads = {"w": torch.randn(100, 100)}
        c10 = compress_gradients(grads, method="top_k", ratio=0.1)
        c50 = compress_gradients(grads, method="top_k", ratio=0.5)

        nnz_10 = (c10["w"] != 0).sum().item()
        nnz_50 = (c50["w"] != 0).sum().item()
        assert nnz_50 > nnz_10


# ---------------------------------------------------------------------------
# Test: FedAvg aggregation
# ---------------------------------------------------------------------------


class TestFedAvg:
    """FedAvg must average model weights from multiple workers."""

    def test_aggregate_two_identical_models(self) -> None:
        from space_ml_sim.compute.federated import fed_avg

        m1 = _make_model()
        m2 = _make_model()
        # Make them identical
        m2.load_state_dict(m1.state_dict())

        avg_state = fed_avg([m1.state_dict(), m2.state_dict()])

        for key in m1.state_dict():
            assert torch.allclose(avg_state[key], m1.state_dict()[key])

    def test_aggregate_produces_average(self) -> None:
        from space_ml_sim.compute.federated import fed_avg

        sd1 = {"w": torch.tensor([2.0, 4.0])}
        sd2 = {"w": torch.tensor([6.0, 8.0])}

        avg = fed_avg([sd1, sd2])
        assert torch.allclose(avg["w"], torch.tensor([4.0, 6.0]))

    def test_weighted_aggregation(self) -> None:
        from space_ml_sim.compute.federated import fed_avg

        sd1 = {"w": torch.tensor([0.0])}
        sd2 = {"w": torch.tensor([10.0])}

        # Weight sd2 three times more
        avg = fed_avg([sd1, sd2], weights=[1.0, 3.0])
        assert torch.allclose(avg["w"], torch.tensor([7.5]))


# ---------------------------------------------------------------------------
# Test: FederatedRound
# ---------------------------------------------------------------------------


class TestFederatedRound:
    """A federated round must train locally, aggregate, and report costs."""

    def test_round_produces_global_model(self) -> None:
        from space_ml_sim.compute.federated import (
            FederatedCoordinator,
            FederatedRoundResult,
        )
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "agg": (6771.0, 0.0, 0.0),
            "w0": (6771.0, 100.0, 0.0),
            "w1": (6771.0, 0.0, 100.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        coord = FederatedCoordinator(
            network=network,
            aggregator_node="agg",
            worker_nodes=["w0", "w1"],
        )

        # Each worker has a local dataset (input, target)
        datasets = {
            "w0": (torch.randn(16, 10), torch.randint(0, 5, (16,))),
            "w1": (torch.randn(16, 10), torch.randint(0, 5, (16,))),
        }

        result = coord.run_round(
            model_factory=_make_model,
            datasets=datasets,
            local_epochs=1,
            lr=0.01,
            compression_method="none",
            compression_ratio=1.0,
        )

        assert isinstance(result, FederatedRoundResult)
        assert result.global_state_dict is not None

    def test_round_reports_communication_cost(self) -> None:
        from space_ml_sim.compute.federated import FederatedCoordinator
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "agg": (6771.0, 0.0, 0.0),
            "w0": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        coord = FederatedCoordinator(
            network=network,
            aggregator_node="agg",
            worker_nodes=["w0"],
        )

        datasets = {
            "w0": (torch.randn(8, 10), torch.randint(0, 5, (8,))),
        }

        result = coord.run_round(
            model_factory=_make_model,
            datasets=datasets,
            local_epochs=1,
            lr=0.01,
        )

        assert result.communication_latency_ms > 0
        assert result.total_bytes_transferred > 0

    def test_compression_reduces_bytes_transferred(self) -> None:
        from space_ml_sim.compute.federated import FederatedCoordinator
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "agg": (6771.0, 0.0, 0.0),
            "w0": (6771.0, 100.0, 0.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        coord = FederatedCoordinator(
            network=network,
            aggregator_node="agg",
            worker_nodes=["w0"],
        )

        datasets = {
            "w0": (torch.randn(8, 10), torch.randint(0, 5, (8,))),
        }

        r_none = coord.run_round(
            model_factory=_make_model,
            datasets=datasets,
            local_epochs=1,
            lr=0.01,
            compression_method="none",
            compression_ratio=1.0,
        )
        r_compressed = coord.run_round(
            model_factory=_make_model,
            datasets=datasets,
            local_epochs=1,
            lr=0.01,
            compression_method="top_k",
            compression_ratio=0.1,
        )

        assert r_compressed.total_bytes_transferred < r_none.total_bytes_transferred

    def test_round_reports_worker_losses(self) -> None:
        from space_ml_sim.compute.federated import FederatedCoordinator
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "agg": (6771.0, 0.0, 0.0),
            "w0": (6771.0, 100.0, 0.0),
            "w1": (6771.0, 0.0, 100.0),
        }
        network = ISLNetwork.from_positions(positions, max_range_km=5000.0)

        coord = FederatedCoordinator(
            network=network,
            aggregator_node="agg",
            worker_nodes=["w0", "w1"],
        )

        datasets = {
            "w0": (torch.randn(8, 10), torch.randint(0, 5, (8,))),
            "w1": (torch.randn(8, 10), torch.randint(0, 5, (8,))),
        }

        result = coord.run_round(
            model_factory=_make_model,
            datasets=datasets,
            local_epochs=2,
            lr=0.01,
        )

        assert len(result.worker_losses) == 2
        assert "w0" in result.worker_losses
        assert "w1" in result.worker_losses
        assert all(loss > 0 for loss in result.worker_losses.values())
