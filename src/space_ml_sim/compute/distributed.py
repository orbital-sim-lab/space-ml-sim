"""Distributed inference across constellation via ISL links.

Supports data-parallel inference: the input batch is split across
worker satellites, each runs the full model on its partition, and
results are gathered back to the source node. Communication costs
are modeled using the ISL network's latency and bandwidth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from space_ml_sim.environment.isl_network import ISLNetwork


@dataclass(frozen=True)
class DistributedInferenceTask:
    """Specification for a distributed inference job."""

    model_factory: object  # Callable[[], nn.Module]
    num_partitions: int
    input_shape: tuple[int, ...]
    seed: int | None = None


@dataclass(frozen=True)
class DistributedResult:
    """Result of a distributed inference execution."""

    predictions: list[int]
    total_latency_ms: float
    compute_latency_ms: float
    communication_latency_ms: float
    nodes_used: list[str]


class DistributedExecutor:
    """Execute data-parallel inference across constellation nodes.

    Workflow:
    1. Source node splits batch into N partitions
    2. Each partition is sent to a worker node over ISL
    3. Workers run inference locally
    4. Results are sent back to source node
    5. Results are concatenated in original order

    Communication cost accounts for:
    - Input tensor transfer (source -> each worker)
    - Output tensor transfer (each worker -> source)
    - ISL propagation latency per hop
    """

    def __init__(self, network: ISLNetwork) -> None:
        self.network = network

    def execute(
        self,
        task: DistributedInferenceTask,
        source_node: str,
        worker_nodes: list[str],
    ) -> DistributedResult:
        """Execute distributed inference.

        Args:
            task: The inference task specification.
            source_node: Satellite ID that originates the request.
            worker_nodes: Satellite IDs to distribute work across.

        Returns:
            DistributedResult with merged predictions and latency breakdown.
        """
        if task.seed is not None:
            torch.manual_seed(task.seed)

        batch_size = task.input_shape[0]
        num_workers = min(task.num_partitions, len(worker_nodes))
        workers = worker_nodes[:num_workers]

        # Generate input tensor
        inputs = torch.randn(task.input_shape)

        # Split batch across workers
        partitions = _split_batch(inputs, num_workers)

        # Estimate communication cost
        input_bytes_per_partition = partitions[0].nelement() * partitions[0].element_size()
        comm_latency = 0.0

        for worker in workers:
            # Source -> worker (send input partition)
            path = self.network.shortest_path(source_node, worker)
            if path is not None and len(path) > 1:
                send_time = self._path_transfer_time(path, input_bytes_per_partition)
                comm_latency += send_time

        # Run inference on each partition
        model = task.model_factory()
        model.eval()

        compute_start = time.perf_counter()
        all_predictions: list[int] = []
        for partition in partitions:
            with torch.no_grad():
                output = model(partition)
                preds = output.argmax(dim=1).tolist()
                all_predictions.extend(preds)
        compute_ms = (time.perf_counter() - compute_start) * 1000

        # Estimate output transfer back to source
        output_bytes_per_partition = 4 * (batch_size // num_workers) * 10  # int32 * classes
        for worker in workers:
            path = self.network.shortest_path(worker, source_node)
            if path is not None and len(path) > 1:
                recv_time = self._path_transfer_time(path, output_bytes_per_partition)
                comm_latency += recv_time

        total_latency = compute_ms + comm_latency

        return DistributedResult(
            predictions=all_predictions,
            total_latency_ms=total_latency,
            compute_latency_ms=compute_ms,
            communication_latency_ms=comm_latency,
            nodes_used=workers,
        )

    def _path_transfer_time(self, path: list[str], payload_bytes: int) -> float:
        """Total transfer time along a multi-hop path."""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.network.transfer_time_ms(path[i], path[i + 1], payload_bytes)
        return total


def _split_batch(tensor: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Split a batch tensor into n roughly equal partitions."""
    batch_size = tensor.size(0)
    chunk_size = max(1, batch_size // n)
    chunks = []
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size if i < n - 1 else batch_size
        chunks.append(tensor[start:end])
    return chunks
