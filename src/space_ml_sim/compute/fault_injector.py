"""ML-aware radiation fault injection using PyTorch hooks.

Injects radiation-modeled faults into neural network inference:
    1. Weight SEU: Flip random bits in weight tensors (persistent memory fault)
    2. Activation SET: Flip random bits in activations during forward pass (transient)
    3. Stuck-at: Zero out weights in TID-degraded regions (permanent)

Key insight: MSB flips in the IEEE 754 exponent field cause catastrophic errors,
while LSB flips in the mantissa are often benign. Real radiation is uniform
across bit positions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile


@dataclass(frozen=True)
class FaultReport:
    """Summary of faults injected during a simulation run."""

    total_faults_injected: int = 0
    weight_faults: int = 0
    activation_faults: int = 0
    layers_affected: tuple[str, ...] = ()
    bit_positions_flipped: tuple[int, ...] = ()


class FaultInjector:
    """Inject radiation-modeled faults into PyTorch model inference.

    Works independently of the orbital mechanics module — can be used
    standalone for fault tolerance research on any PyTorch model.
    """

    def __init__(
        self,
        rad_env: RadiationEnvironment,
        chip_profile: ChipProfile,
    ) -> None:
        self.rad_env = rad_env
        self.chip = chip_profile
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    @staticmethod
    def flip_random_bits(tensor: torch.Tensor, num_flips: int) -> list[int]:
        """Flip random bits in a float32 tensor's IEEE 754 binary representation.

        Args:
            tensor: A float32 tensor (modified in-place).
            num_flips: Number of random bit flips to inject.

        Returns:
            List of bit positions that were flipped (0-31).
        """
        if num_flips == 0:
            return []

        flat = tensor.view(-1)
        n = flat.numel()
        if n == 0:
            return []

        # Random element indices and bit positions (0-31 for float32)
        indices = torch.randint(0, n, (num_flips,))
        bit_positions = torch.randint(0, 32, (num_flips,))

        # Reinterpret as int32, flip bits, reinterpret back
        int_view = flat.clone().view(torch.int32)
        for idx, bit in zip(indices, bit_positions):
            int_view[idx] ^= 1 << bit.item()
        flat.copy_(int_view.view(torch.float32))

        return bit_positions.tolist()

    def inject_weight_faults(
        self,
        model: torch.nn.Module,
        num_faults: int | None = None,
        inference_time_seconds: float = 0.001,
    ) -> FaultReport:
        """Inject SEU bit flips into model weights.

        If num_faults is None, the count is sampled from a Poisson distribution
        based on the radiation environment, chip profile, and inference time.

        Args:
            model: PyTorch model (weights modified in-place).
            num_faults: Exact number of faults, or None for radiation-sampled.
            inference_time_seconds: Duration used for Poisson rate calculation.

        Returns:
            FaultReport summarizing injected faults.
        """
        if num_faults is None:
            total_weight_bits = sum(p.numel() * 32 for p in model.parameters())
            expected = self.rad_env.base_seu_rate * total_weight_bits * inference_time_seconds
            num_faults = int(np.random.poisson(expected))

        if num_faults == 0:
            return FaultReport()

        # Distribute faults across layers proportional to parameter count
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        sizes = [p.numel() for _, p in params]
        total = sum(sizes)
        if total == 0:
            return FaultReport()

        all_layers: list[str] = []
        all_bits: list[int] = []
        total_injected = 0

        for name, param in params:
            layer_faults = max(1, int(num_faults * param.numel() / total))
            layer_faults = min(layer_faults, num_faults - total_injected)
            if layer_faults <= 0:
                continue

            with torch.no_grad():
                bits = self.flip_random_bits(param.data, layer_faults)

            total_injected += layer_faults
            all_layers.append(name)
            all_bits.extend(bits)

            if total_injected >= num_faults:
                break

        return FaultReport(
            total_faults_injected=total_injected,
            weight_faults=total_injected,
            layers_affected=tuple(all_layers),
            bit_positions_flipped=tuple(all_bits),
        )

    def register_activation_hooks(
        self,
        model: torch.nn.Module,
        fault_probability: float = 0.001,
    ) -> None:
        """Register forward hooks for transient activation fault injection.

        Args:
            model: PyTorch model to hook.
            fault_probability: Probability of a bit flip per tensor element per forward pass.
        """
        self.remove_hooks()

        def make_hook(layer_name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.numel() > 0:
                    num_faults = int(np.random.binomial(output.numel(), fault_probability))
                    if num_faults > 0:
                        FaultInjector.flip_random_bits(output, num_faults)
                return output

            return hook

        for name, module in model.named_modules():
            # Hook leaf modules only
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered activation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def sweep(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        fault_counts: list[int],
        num_trials: int = 5,
    ) -> "pd.DataFrame":
        """Run fault injection sweep across multiple fault counts and trials.

        For each fault count, runs num_trials with independent fault injections
        and measures accuracy degradation.

        Args:
            model: Base PyTorch model (will be deep-copied per trial).
            dataloader: Evaluation data loader.
            fault_counts: List of fault counts to sweep.
            num_trials: Number of independent trials per fault count.

        Returns:
            DataFrame with columns: fault_count, trial, accuracy,
            top5_accuracy, critical_failure, faults_injected, layers_affected.
        """
        import pandas as pd

        results: list[dict] = []

        for fc in fault_counts:
            for trial in range(num_trials):
                test_model = copy.deepcopy(model)
                test_model.eval()

                report = self.inject_weight_faults(test_model, num_faults=fc)

                acc, top5 = _evaluate_model(test_model, dataloader)
                results.append(
                    {
                        "fault_count": fc,
                        "trial": trial,
                        "accuracy": acc,
                        "top5_accuracy": top5,
                        "critical_failure": acc < 0.1,
                        "faults_injected": report.total_faults_injected,
                        "layers_affected": len(report.layers_affected),
                    }
                )

        return pd.DataFrame(results)


def _evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    """Evaluate a model on a dataloader, returning (top1_accuracy, top5_accuracy)."""
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            if outputs.size(1) >= 5:
                _, top5_pred = outputs.topk(5, dim=1)
                for i in range(len(labels)):
                    if labels[i] in top5_pred[i]:
                        correct_top5 += 1
            else:
                correct_top5 += predicted.eq(labels).sum().item()

            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    top5 = correct_top5 / total if total > 0 else 0.0
    return acc, top5
