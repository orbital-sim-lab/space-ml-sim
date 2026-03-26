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

        self.protected_layers: set[str] | None = None  # For selective TMR

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

    def configure_protection(self, protected_layers: set[str]) -> None:
        """Configure which layers are protected by selective TMR.

        Only protected layers will have independent faults across replicas.
        Unprotected layers remain identical, so voting cannot correct their faults.

        Args:
            protected_layers: Set of parameter names (e.g., {"0.weight", "2.bias"})
                to protect with TMR voting.
        """
        self.protected_layers = set(protected_layers)

    def _selective_tmr_forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Selective TMR: majority vote across replicas.

        Same voting mechanism as full TMR. The selectivity comes from
        inject_faults_to_replicas, which only injects faults into protected layers.
        """
        result = self._full_tmr_forward(x)
        return {**result, "strategy": "selective_tmr"}

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

    def inject_faults_to_replicas(self, injector: FaultInjector, faults_per_replica: int) -> None:
        """Inject independent faults into each TMR replica.

        For selective TMR with configured protection, only injects faults
        into the protected layers. Unprotected layers remain identical
        across replicas.

        Args:
            injector: FaultInjector instance.
            faults_per_replica: Number of faults to inject per replica.
        """
        if not hasattr(self, "replicas"):
            raise RuntimeError("inject_faults_to_replicas requires TMR strategy")

        if self.strategy == "selective_tmr" and self.protected_layers:
            for replica in self.replicas:
                self._inject_to_protected_only(replica, injector, faults_per_replica)
        else:
            for replica in self.replicas:
                injector.inject_weight_faults(replica, num_faults=faults_per_replica)

    def _inject_to_protected_only(
        self,
        model: torch.nn.Module,
        injector: FaultInjector,
        num_faults: int,
    ) -> None:
        """Inject faults only into protected layers of a model.

        Distributes faults proportionally across protected parameters.
        """
        protected_params = [
            (name, p)
            for name, p in model.named_parameters()
            if p.requires_grad and name in self.protected_layers
        ]
        if not protected_params:
            return

        total_elements = sum(p.numel() for _, p in protected_params)
        faults_remaining = num_faults

        for name, param in protected_params:
            layer_faults = max(1, int(num_faults * param.numel() / total_elements))
            layer_faults = min(layer_faults, faults_remaining)
            if layer_faults <= 0:
                continue
            with torch.no_grad():
                FaultInjector.flip_random_bits(param.data, layer_faults)
            faults_remaining -= layer_faults
            if faults_remaining <= 0:
                break

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
                    FaultInjector.flip_random_bits(test_params[name].data, faults_per_layer)
                acc, _ = _evaluate_model(test_model, dataloader)
                drops.append(baseline_acc - acc)

            results[name] = float(np.mean(drops))

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
