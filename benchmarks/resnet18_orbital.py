#!/usr/bin/env python3
"""Benchmark: ResNet-18 fault injection across orbital radiation environments.

Quick benchmark to measure:
1. Time to inject N faults
2. Inference latency with/without TMR
3. Accuracy degradation curves
"""

import time

import torch
import torchvision
from rich.console import Console

from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.compute.tmr import TMRWrapper
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import TERAFAB_D3

console = Console()


def main() -> None:
    console.print("[bold cyan]ResNet-18 Orbital Benchmark[/]\n")

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"Model parameters: {total_params:,}")
    console.print(f"Total weight bits: {total_params * 32:,}\n")

    rad_env = RadiationEnvironment.leo_500km()
    injector = FaultInjector(rad_env=rad_env, chip_profile=TERAFAB_D3)

    # Benchmark fault injection speed
    fault_counts = [10, 100, 1000, 10000]
    console.print("[bold]Fault Injection Speed:[/]")
    for fc in fault_counts:
        import copy
        test_model = copy.deepcopy(model)
        start = time.perf_counter()
        injector.inject_weight_faults(test_model, num_faults=fc)
        elapsed = time.perf_counter() - start
        console.print(f"  {fc:>6d} faults: {elapsed*1000:.1f} ms")

    # Benchmark inference latency
    x = torch.randn(1, 3, 224, 224)
    console.print("\n[bold]Inference Latency (batch=1):[/]")

    # Single model
    with torch.no_grad():
        _ = model(x)  # warmup
        start = time.perf_counter()
        for _ in range(10):
            _ = model(x)
        elapsed = (time.perf_counter() - start) / 10
    console.print(f"  Single model:  {elapsed*1000:.1f} ms")

    # Full TMR
    def factory():
        return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    tmr = TMRWrapper(factory, strategy="full_tmr")
    with torch.no_grad():
        _ = tmr.forward(x)  # warmup
        start = time.perf_counter()
        for _ in range(10):
            _ = tmr.forward(x)
        elapsed = (time.perf_counter() - start) / 10
    console.print(f"  Full TMR (3x): {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    main()
