"""Per-layer sensitivity heatmap for fault injection analysis."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import plotly.graph_objects as go

from space_ml_sim.compute.fault_injector import FaultInjector, _evaluate_model


def sensitivity_heatmap(
    model_factory: Callable[[], torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    injector: FaultInjector,
    faults_per_layer: int = 100,
    num_trials: int = 3,
    title: str = "Layer Sensitivity to Bit Flips",
    save_path: str | None = None,
) -> "go.Figure":
    """Generate a heatmap showing per-layer vulnerability to bit flips.

    Injects faults into each layer independently and measures accuracy
    drop relative to baseline. Produces a horizontal bar chart with
    color-coded vulnerability.

    Args:
        model_factory: Callable returning a fresh model instance.
        dataloader: Evaluation data loader.
        injector: FaultInjector instance.
        faults_per_layer: Number of faults per layer per trial.
        num_trials: Independent trials per layer.
        title: Plot title.
        save_path: Optional path to save as HTML.

    Returns:
        Plotly Figure with per-layer sensitivity visualization.
    """
    import plotly.graph_objects as go

    raw = sensitivity_data(
        model_factory=model_factory,
        dataloader=dataloader,
        injector=injector,
        faults_per_layer=faults_per_layer,
        num_trials=num_trials,
    )

    # Compute baseline accuracy separately for the title annotation
    baseline_model = model_factory().eval()
    baseline_acc, _ = _evaluate_model(baseline_model, dataloader)

    # sensitivity_data returns sorted-descending; unpack preserving order
    layer_names = list(raw.keys())
    drops = list(raw.values())

    # Color scale: green (resilient, drop ~0) to red (vulnerable, drop ~max)
    max_drop = max(abs(d) for d in drops) if drops else 1.0
    if max_drop == 0:
        max_drop = 1.0  # Avoid division by zero when model is fully resilient
    colors = [
        "rgb({r}, {g}, 50)".format(
            r=max(0, min(255, int(255 * d / max_drop))),
            g=max(0, min(255, int(255 * (1.0 - d / max_drop)))),
        )
        for d in drops
    ]

    fig = go.Figure(
        go.Bar(
            x=drops,
            y=layer_names,
            orientation="h",
            marker_color=colors,
            text=[f"{d:.4f}" for d in drops],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"{title} (baseline: {baseline_acc:.2%})",
        xaxis_title="Accuracy Drop (higher = more vulnerable)",
        yaxis_title="Layer",
        template="plotly_white",
        height=max(400, len(layer_names) * 25),
        yaxis=dict(autorange="reversed"),  # Most vulnerable at top
        margin=dict(l=300),  # Room for long layer names
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def sensitivity_data(
    model_factory: Callable[[], torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    injector: FaultInjector,
    faults_per_layer: int = 100,
    num_trials: int = 3,
) -> dict[str, float]:
    """Get raw per-layer sensitivity data without visualization.

    Injects faults into each trainable parameter tensor independently
    across num_trials independent trials and returns the mean accuracy
    drop per layer.

    Args:
        model_factory: Callable returning a fresh model instance.
        dataloader: Evaluation data loader.
        injector: FaultInjector instance (unused directly; flip_random_bits
            is called as a static method to keep injection isolated per layer).
        faults_per_layer: Number of faults injected per parameter tensor per trial.
        num_trials: Number of independent repetitions per parameter tensor.

    Returns:
        Dict of {layer_name: avg_accuracy_drop} sorted by impact descending.
        Returns an empty dict when the model has no trainable parameters.
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
