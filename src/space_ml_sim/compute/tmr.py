"""Triple Modular Redundancy (TMR) wrapper for ML models.

Strategies:
    - full_tmr: Run 3 complete replicas, majority vote on outputs. 3x compute cost.
    - selective_tmr: TMR only the most vulnerable layers (from sensitivity analysis).
    - checkpoint_rollback: Single model with periodic checkpointing; roll back on detected fault.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from space_ml_sim.compute.fault_injector import FaultInjector, _evaluate_model


class TMRWrapper:
    """Triple Modular Redundancy wrapper for ML models.

    Provides fault-tolerant inference by running multiple model replicas
    and voting on outputs. Supports full TMR, selective TMR, and
    checkpoint-based rollback strategies.
    """

    def __init__(
        self,
        model_factory: callable,
        strategy: str = "full_tmr",
        device: str = "cpu",
    ) -> None:
        """Initialize TMR wrapper.

        Args:
            model_factory: Callable that returns a fresh model instance.
            strategy: One of "full_tmr", "selective_tmr", "checkpoint_rollback".
            device: PyTorch device string.
        """
        if strategy not in ("full_tmr", "selective_tmr", "checkpoint_rollback"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.device = device
        self._model_factory = model_factory

        if strategy in ("full_tmr", "selective_tmr"):
            self.replicas = [model_factory().to(device).eval() for _ in range(3)]
        else:
            self.model = model_factory().to(device).eval()
            self.checkpoint = copy.deepcopy(self.model.state_dict())

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Run fault-tolerant forward pass.

        Args:
            x: Input tensor batch.

        Returns:
            Dict with 'predictions', 'disagreements', and strategy-specific fields.
        """
        if self.strategy == "full_tmr":
            return self._full_tmr_forward(x)
        elif self.strategy == "selective_tmr":
            return self._selective_tmr_forward(x)
        else:
            return self._checkpoint_forward(x)

    def _full_tmr_forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Run all 3 replicas and take majority vote on argmax outputs."""
        outputs = [r(x.to(self.device)) for r in self.replicas]
        votes = [o.argmax(dim=1) for o in outputs]

        # Majority vote per sample in batch
        stacked = torch.stack(votes, dim=0)  # (3, batch)
        result, _ = torch.mode(stacked, dim=0)

        disagreements = int(
            (stacked[0] != stacked[1]).sum().item()
            + (stacked[1] != stacked[2]).sum().item()
            + (stacked[0] != stacked[2]).sum().item()
        )

        return {
            "predictions": result,
            "disagreements": disagreements,
            "raw_outputs": outputs,
            "strategy": "full_tmr",
        }

    def _selective_tmr_forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Selective TMR: same as full TMR but intended for partially-replicated models.

        In v0.1, behaves identically to full TMR. Future versions will support
        per-layer selective replication based on sensitivity analysis.
        """
        return self._full_tmr_forward(x)

    def _checkpoint_forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Single model with anomaly detection and checkpoint rollback."""
        outputs = self.model(x.to(self.device))
        predictions = outputs.argmax(dim=1)

        # Simple anomaly detection: check for NaN/Inf in outputs
        has_anomaly = bool(torch.isnan(outputs).any() or torch.isinf(outputs).any())

        if has_anomaly and self.checkpoint is not None:
            self.model.load_state_dict(copy.deepcopy(self.checkpoint))
            outputs = self.model(x.to(self.device))
            predictions = outputs.argmax(dim=1)

        return {
            "predictions": predictions,
            "disagreements": 0,
            "anomaly_detected": has_anomaly,
            "rolled_back": has_anomaly,
            "strategy": "checkpoint_rollback",
        }

    def save_checkpoint(self) -> None:
        """Save current model state as checkpoint (checkpoint_rollback strategy only)."""
        if self.strategy == "checkpoint_rollback":
            self.checkpoint = copy.deepcopy(self.model.state_dict())

    def inject_faults_to_replicas(
        self, injector: FaultInjector, faults_per_replica: int
    ) -> None:
        """Inject independent faults into each TMR replica.

        Args:
            injector: FaultInjector instance.
            faults_per_replica: Number of faults to inject per replica.
        """
        if not hasattr(self, "replicas"):
            raise RuntimeError("inject_faults_to_replicas requires TMR strategy")

        for replica in self.replicas:
            injector.inject_weight_faults(replica, num_faults=faults_per_replica)

    @staticmethod
    def sensitivity_analysis(
        model_factory: callable,
        dataloader: torch.utils.data.DataLoader,
        injector: FaultInjector,
        faults_per_layer: int = 100,
        num_trials: int = 3,
    ) -> dict[str, float]:
        """Per-layer vulnerability ranking via targeted fault injection.

        Injects faults into each layer independently and measures accuracy drop
        relative to baseline.

        Args:
            model_factory: Callable returning a fresh model.
            dataloader: Evaluation data loader.
            injector: FaultInjector instance.
            faults_per_layer: Number of faults injected per layer per trial.
            num_trials: Number of independent trials per layer.

        Returns:
            Dict of {layer_name: avg_accuracy_drop} sorted by impact (descending).
        """
        baseline_model = model_factory().eval()
        baseline_acc, _ = _evaluate_model(baseline_model, dataloader)

        results: dict[str, float] = {}

        for name, param in baseline_model.named_parameters():
            if not param.requires_grad:
                continue

            drops: list[float] = []
            for _ in range(num_trials):
                test_model = copy.deepcopy(baseline_model)
                test_params = dict(test_model.named_parameters())
                with torch.no_grad():
                    FaultInjector.flip_random_bits(
                        test_params[name].data, faults_per_layer
                    )
                acc, _ = _evaluate_model(test_model, dataloader)
                drops.append(baseline_acc - acc)

            results[name] = float(np.mean(drops))

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
