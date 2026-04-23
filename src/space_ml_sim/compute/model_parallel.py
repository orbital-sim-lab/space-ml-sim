"""Model-parallel (pipeline) inference across constellation satellites.

Splits a sequential model into stages, assigns each stage to a satellite,
and runs inference as a pipeline. Intermediate activations are transferred
between satellites via ISL links, with communication cost modeled using
the ISL network's latency and bandwidth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from space_ml_sim.environment.isl_network import ISLNetwork


def partition_model(model: nn.Sequential, num_stages: int) -> list[nn.Sequential]:
    """Split a Sequential model into pipeline stages.

    Distributes layers as evenly as possible across stages. If num_stages
    exceeds the number of layers, clamps to one layer per stage.

    Args:
        model: A nn.Sequential model to partition.
        num_stages: Desired number of pipeline stages.

    Returns:
        List of nn.Sequential modules, one per stage.
    """
    layers = list(model.children())
    n_layers = len(layers)
    num_stages = min(num_stages, n_layers)
    num_stages = max(num_stages, 1)

    stages: list[nn.Sequential] = []
    chunk_size = n_layers // num_stages
    remainder = n_layers % num_stages

    start = 0
    for i in range(num_stages):
        # Distribute remainder layers across first stages
        end = start + chunk_size + (1 if i < remainder else 0)
        stage = nn.Sequential(*layers[start:end])
        stage.eval()
        stages.append(stage)
        start = end

    return stages


@dataclass(frozen=True)
class PipelineResult:
    """Result of a model-parallel pipeline execution."""

    predictions: list[int]
    total_latency_ms: float
    compute_latency_ms: float
    communication_latency_ms: float
    activation_sizes_bytes: list[int]
    nodes_used: list[str]


class PipelineExecutor:
    """Execute model-parallel inference as a pipeline across satellites.

    Workflow:
    1. Stage 0 runs on node 0, produces activations
    2. Activations are transferred to node 1 via ISL
    3. Stage 1 runs on node 1, produces activations
    4. ... repeat until final stage
    5. Final stage produces predictions
    """

    def __init__(self, network: ISLNetwork) -> None:
        self.network = network

    def execute(
        self,
        stages: list[nn.Sequential],
        pipeline_nodes: list[str],
        input_tensor: torch.Tensor,
    ) -> PipelineResult:
        """Execute pipeline inference across constellation nodes.

        Args:
            stages: Model stages (from partition_model).
            pipeline_nodes: Satellite IDs, one per stage.
            input_tensor: Input batch tensor.

        Returns:
            PipelineResult with predictions and latency breakdown.
        """
        nodes = pipeline_nodes[: len(stages)]
        activation = input_tensor
        compute_ms = 0.0
        comm_ms = 0.0
        activation_sizes: list[int] = []

        for i, stage in enumerate(stages):
            # Run stage computation
            t0 = time.perf_counter()
            with torch.no_grad():
                activation = stage(activation)
            compute_ms += (time.perf_counter() - t0) * 1000

            # Transfer activations to next node (if not last stage)
            if i < len(stages) - 1:
                act_bytes = activation.nelement() * activation.element_size()
                activation_sizes.append(act_bytes)

                src = nodes[i]
                dst = nodes[i + 1]
                path = self.network.shortest_path(src, dst)
                if path is not None and len(path) > 1:
                    for j in range(len(path) - 1):
                        comm_ms += self.network.transfer_time_ms(path[j], path[j + 1], act_bytes)

        # Final predictions
        predictions = activation.argmax(dim=1).tolist()

        return PipelineResult(
            predictions=predictions,
            total_latency_ms=compute_ms + comm_ms,
            compute_latency_ms=compute_ms,
            communication_latency_ms=comm_ms,
            activation_sizes_bytes=activation_sizes,
            nodes_used=nodes,
        )
