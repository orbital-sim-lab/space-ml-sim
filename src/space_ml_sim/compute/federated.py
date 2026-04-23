"""Bandwidth-constrained federated learning across constellation satellites.

Implements FedAvg with gradient compression for satellite constellations
where ISL bandwidth is limited. Each worker satellite trains locally,
compresses its model updates, and sends them to an aggregator node
that produces a new global model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from space_ml_sim.environment.isl_network import ISLNetwork


def compress_gradients(
    gradients: dict[str, torch.Tensor],
    method: str = "top_k",
    ratio: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compress gradient tensors for bandwidth-constrained transfer.

    Args:
        gradients: Map of parameter name to gradient tensor.
        method: Compression method — "top_k" or "none".
        ratio: Fraction of values to keep (for top_k). 0.1 = keep top 10%.

    Returns:
        Compressed gradients (same shape, zeros for pruned values).
    """
    if method == "none":
        return {k: v.clone() for k, v in gradients.items()}

    if method == "top_k":
        compressed: dict[str, torch.Tensor] = {}
        for name, tensor in gradients.items():
            flat = tensor.abs().flatten()
            k = max(1, int(flat.numel() * ratio))
            _, indices = torch.topk(flat, k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[indices] = True
            result = torch.zeros_like(tensor)
            result.view(-1)[mask] = tensor.view(-1)[mask]
            compressed[name] = result
        return compressed

    raise ValueError(f"Unknown compression method: {method}")


def fed_avg(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Federated averaging of model state dicts.

    Args:
        state_dicts: List of model state dicts from workers.
        weights: Optional per-worker weights (will be normalized).
            If None, uniform weighting is used.

    Returns:
        Averaged state dict.
    """
    n = len(state_dicts)
    if weights is None:
        w = [1.0 / n] * n
    else:
        total = sum(weights)
        w = [wi / total for wi in weights]

    avg: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        avg[key] = sum(sd[key].float() * wi for sd, wi in zip(state_dicts, w))  # type: ignore[assignment]

    return avg


@dataclass(frozen=True)
class FederatedRoundResult:
    """Result of one federated learning round."""

    global_state_dict: dict[str, torch.Tensor]
    worker_losses: dict[str, float]
    communication_latency_ms: float
    total_bytes_transferred: int
    round_index: int = 0


class FederatedCoordinator:
    """Coordinate federated learning across constellation satellites.

    Workflow per round:
    1. Broadcast global model to all workers (aggregator -> workers)
    2. Each worker trains locally for N epochs
    3. Workers compress and send updated weights to aggregator
    4. Aggregator runs FedAvg to produce new global model
    """

    def __init__(
        self,
        network: ISLNetwork,
        aggregator_node: str,
        worker_nodes: list[str],
    ) -> None:
        self.network = network
        self.aggregator_node = aggregator_node
        self.worker_nodes = worker_nodes

    def run_round(
        self,
        model_factory: object,
        datasets: dict[str, tuple[torch.Tensor, torch.Tensor]],
        local_epochs: int = 1,
        lr: float = 0.01,
        compression_method: str = "none",
        compression_ratio: float = 1.0,
        global_state: dict[str, torch.Tensor] | None = None,
    ) -> FederatedRoundResult:
        """Execute one federated learning round.

        Args:
            model_factory: Callable returning a fresh nn.Module.
            datasets: Map of worker_id -> (inputs, targets).
            local_epochs: Number of local training epochs per worker.
            lr: Learning rate for local SGD.
            compression_method: Gradient compression method.
            compression_ratio: Fraction of values to keep.
            global_state: Optional initial global state dict.

        Returns:
            FederatedRoundResult with aggregated model and metrics.
        """
        comm_ms = 0.0
        total_bytes = 0

        # Phase 1: Each worker trains locally
        worker_states: list[dict[str, torch.Tensor]] = []
        worker_losses: dict[str, float] = {}

        for worker_id in self.worker_nodes:
            if worker_id not in datasets:
                continue

            inputs, targets = datasets[worker_id]
            model = model_factory()
            if global_state is not None:
                model.load_state_dict(global_state)
            model.train()

            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            final_loss = 0.0
            for _ in range(local_epochs):
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

            worker_losses[worker_id] = final_loss

            # Phase 2: Compress and "send" to aggregator
            state = model.state_dict()
            compressed = compress_gradients(
                state, method=compression_method, ratio=compression_ratio
            )

            # Calculate transfer size (non-zero elements only for compressed)
            transfer_bytes = sum(
                int((t != 0).sum().item()) * t.element_size() for t in compressed.values()
            )
            total_bytes += transfer_bytes

            # ISL transfer cost: worker -> aggregator
            path = self.network.shortest_path(worker_id, self.aggregator_node)
            if path is not None and len(path) > 1:
                for i in range(len(path) - 1):
                    comm_ms += self.network.transfer_time_ms(path[i], path[i + 1], transfer_bytes)

            worker_states.append(compressed)

        # Phase 3: Aggregation
        global_state_dict = fed_avg(worker_states)

        return FederatedRoundResult(
            global_state_dict=global_state_dict,
            worker_losses=worker_losses,
            communication_latency_ms=comm_ms,
            total_bytes_transferred=total_bytes,
        )
