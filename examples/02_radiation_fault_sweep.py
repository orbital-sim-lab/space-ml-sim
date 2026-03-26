#!/usr/bin/env python3
"""Radiation fault sweep: accuracy vs bit flips for different chip profiles.

THE KILLER DEMO. Loads ResNet-18 pretrained on ImageNet, evaluates on CIFAR-10,
and sweeps fault counts across all 4 chip profiles. Saves plot as HTML.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from rich.console import Console
from rich.progress import Progress

from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ALL_CHIPS
from space_ml_sim.viz.plots import plot_fault_sweep

console = Console()

FAULT_COUNTS = [0, 1, 5, 10, 25, 50, 100, 200, 500]
NUM_TRIALS = 3  # Reduce for speed; increase for publication
BATCH_SIZE = 256
MAX_BATCHES = 4  # Limit eval batches for demo speed


def get_cifar10_loader() -> torch.utils.data.DataLoader:
    """Download CIFAR-10 test set and return a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def limited_loader(loader, max_batches: int):
    """Yield at most max_batches from a DataLoader."""
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch


def main() -> None:
    console.print("[bold cyan]Radiation Fault Sweep Demo[/]")
    console.print("Loading ResNet-18 (pretrained) and CIFAR-10...\n")

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    loader = get_cifar10_loader()

    # Use 500km LEO as the radiation environment
    rad_env = RadiationEnvironment.leo_500km()
    all_dfs = []

    with Progress() as progress:
        for chip in ALL_CHIPS:
            task = progress.add_task(f"[cyan]{chip.name}", total=len(FAULT_COUNTS) * NUM_TRIALS)

            injector = FaultInjector(rad_env=rad_env, chip_profile=chip)

            # Wrap the loader to limit batches
            def make_limited():
                return limited_loader(loader, MAX_BATCHES)

            # Manual sweep (using limited loader per eval)
            import copy
            import pandas as pd
            import numpy as np

            results = []
            for fc in FAULT_COUNTS:
                for trial in range(NUM_TRIALS):
                    test_model = copy.deepcopy(model)
                    test_model.eval()

                    report = injector.inject_weight_faults(test_model, num_faults=fc)

                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for images, labels in limited_loader(loader, MAX_BATCHES):
                            outputs = test_model(images)
                            _, predicted = outputs.max(1)
                            correct += predicted.eq(labels).sum().item()
                            total += labels.size(0)

                    acc = correct / total if total > 0 else 0.0
                    results.append({
                        "fault_count": fc,
                        "trial": trial,
                        "accuracy": acc,
                        "top5_accuracy": acc,  # Simplified for CIFAR-10
                        "critical_failure": acc < 0.1,
                        "faults_injected": report.total_faults_injected,
                        "layers_affected": len(report.layers_affected),
                        "chip": chip.name,
                    })
                    progress.advance(task)

            df = pd.DataFrame(results)
            all_dfs.append(df)

            # Print summary for this chip
            baseline = df[df["fault_count"] == 0]["accuracy"].mean()
            worst = df[df["fault_count"] == max(FAULT_COUNTS)]["accuracy"].mean()
            console.print(
                f"  {chip.name}: baseline={baseline:.2%}, "
                f"500-fault={worst:.2%}, "
                f"TID tolerance={chip.tid_tolerance_krad} krad"
            )

    # Combine and plot
    import pandas as pd

    combined = pd.concat(all_dfs, ignore_index=True)

    # Create per-chip plots
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]

    for i, chip in enumerate(ALL_CHIPS):
        chip_df = combined[combined["chip"] == chip.name]
        grouped = chip_df.groupby("fault_count").agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
        ).reset_index()

        fig.add_trace(go.Scatter(
            x=grouped["fault_count"],
            y=grouped["acc_mean"],
            error_y=dict(type="data", array=grouped["acc_std"].fillna(0)),
            mode="lines+markers",
            name=chip.name,
            line=dict(color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        title="ResNet-18 Accuracy vs Radiation-Induced Bit Flips (500km LEO)",
        xaxis_title="Number of Bit Flips Injected",
        yaxis_title="Top-1 Accuracy",
        template="plotly_white",
        legend=dict(x=0.02, y=0.02),
    )

    fig.write_html("fault_sweep_results.html")
    console.print("\n[bold green]Plot saved to fault_sweep_results.html[/]")


if __name__ == "__main__":
    main()
