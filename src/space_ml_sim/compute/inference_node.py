"""Inference node abstraction for running ML models on a satellite."""

from __future__ import annotations

from enum import Enum

import torch
from pydantic import BaseModel, Field

from space_ml_sim.models.chip_profiles import ChipProfile


class NodeStatus(str, Enum):
    """Inference node operational status."""

    IDLE = "idle"
    RUNNING = "running"
    FAULTED = "faulted"


class InferenceResult(BaseModel):
    """Result of a single inference run."""

    predictions: list[int] = Field(default_factory=list)
    latency_ms: float = 0.0
    faults_during_inference: int = 0
    status: NodeStatus = NodeStatus.IDLE


class InferenceNode:
    """Wraps a PyTorch model as a compute node on a satellite.

    Tracks inference count, fault history, and thermal/power constraints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        chip_profile: ChipProfile,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.chip_profile = chip_profile
        self.device = device
        self.status = NodeStatus.IDLE
        self.inference_count = 0
        self.total_faults = 0

    def run_inference(self, inputs: torch.Tensor) -> InferenceResult:
        """Run inference on a batch of inputs.

        Args:
            inputs: Input tensor batch.

        Returns:
            InferenceResult with predictions and metadata.
        """
        self.status = NodeStatus.RUNNING
        try:
            with torch.no_grad():
                outputs = self.model(inputs.to(self.device))
                predictions = outputs.argmax(dim=1).tolist()
        except Exception:
            self.status = NodeStatus.FAULTED
            raise

        self.inference_count += 1
        self.status = NodeStatus.IDLE

        return InferenceResult(
            predictions=predictions,
            status=NodeStatus.IDLE,
        )

    def can_run(self, power_available_w: float, temperature_c: float) -> bool:
        """Check if the node has enough power and is within thermal limits.

        Args:
            power_available_w: Available power in watts.
            temperature_c: Current temperature in Celsius.

        Returns:
            True if the node can safely run inference.
        """
        return (
            power_available_w >= self.chip_profile.tdp_watts
            and temperature_c <= self.chip_profile.max_temp_c
        )
