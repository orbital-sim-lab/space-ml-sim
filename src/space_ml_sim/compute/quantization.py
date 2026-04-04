"""Quantization-aware fault resilience comparison.

Compare how different numerical precisions (FP32, FP16, INT8)
affect a model's vulnerability to radiation-induced bit flips.
INT8 models have fewer critical bits per weight (8 vs 32), making
them inherently more resilient to random bit flips.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

from space_ml_sim.compute.fault_injector import FaultInjector


def quantize_model(
    model: torch.nn.Module,
    mode: str = "dynamic_int8",
) -> torch.nn.Module:
    """Quantize a model to a lower precision.

    Args:
        model: PyTorch model (should be in eval mode).
        mode: One of "fp32" (no-op copy), "fp16", "dynamic_int8".

    Returns:
        Quantized model copy.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "fp32":
        return copy.deepcopy(model)
    elif mode == "fp16":
        return copy.deepcopy(model).half()
    elif mode == "dynamic_int8":
        # Simulate INT8 quantization by clamping and rounding weights
        # to 256 levels. This preserves the PyTorch graph structure
        # (unlike torch.ao.quantization which changes module types).
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            for param in model_copy.parameters():
                if param.is_floating_point():
                    vmin, vmax = param.min(), param.max()
                    scale = (vmax - vmin) / 255.0 if vmax != vmin else 1.0
                    param.data = torch.round((param.data - vmin) / scale) * scale + vmin
        return model_copy
    else:
        raise ValueError(
            f"Unknown quantization mode: {mode!r}. Use 'fp32', 'fp16', or 'dynamic_int8'."
        )


def compare_quantization_resilience(
    model_factory: Callable[[], torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    injector: FaultInjector,
    fault_counts: list[int] | None = None,
    modes: list[str] | None = None,
    num_trials: int = 3,
) -> "pd.DataFrame":
    """Compare fault resilience across quantization levels.

    For each quantization mode, runs a fault injection sweep and
    measures accuracy degradation.

    Args:
        model_factory: Callable returning a fresh FP32 model.
        dataloader: Evaluation data loader.
        injector: FaultInjector instance.
        fault_counts: Fault counts to sweep. Defaults to [0, 1, 5, 10, 25, 50, 100].
        modes: Quantization modes. Defaults to ["fp32", "fp16", "dynamic_int8"].
        num_trials: Trials per fault count per mode.

    Returns:
        DataFrame with columns: mode, fault_count, trial, accuracy, faults_injected.
    """
    import pandas as pd

    if fault_counts is None:
        fault_counts = [0, 1, 5, 10, 25, 50, 100]
    if modes is None:
        modes = ["fp32", "fp16", "dynamic_int8"]

    all_results: list[dict] = []

    for mode in modes:
        base_model = model_factory().eval()
        quant_model = quantize_model(base_model, mode)
        is_half = mode == "fp16"

        for fc in fault_counts:
            for trial in range(num_trials):
                test_model = copy.deepcopy(quant_model)
                test_model.eval()

                # Inject faults into float parameters only
                faults_injected = 0
                if fc > 0:
                    float_params = [
                        (n, p)
                        for n, p in test_model.named_parameters()
                        if p.requires_grad and p.is_floating_point()
                    ]
                    if float_params:
                        total_elements = sum(p.numel() for _, p in float_params)
                        for name, param in float_params:
                            layer_faults = max(1, int(fc * param.numel() / total_elements))
                            layer_faults = min(layer_faults, fc - faults_injected)
                            if layer_faults > 0:
                                with torch.no_grad():
                                    FaultInjector.flip_random_bits(param.data, layer_faults)
                                faults_injected += layer_faults
                            if faults_injected >= fc:
                                break

                # Evaluate
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in dataloader:
                        if is_half:
                            images = images.half()
                        outputs = test_model(images)
                        if outputs.dtype == torch.float16:
                            outputs = outputs.float()
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)

                acc = correct / total if total > 0 else 0.0
                all_results.append(
                    {
                        "mode": mode,
                        "fault_count": fc,
                        "trial": trial,
                        "accuracy": acc,
                        "faults_injected": faults_injected,
                    }
                )

    return pd.DataFrame(all_results)


def plot_quantization_comparison(
    df: "pd.DataFrame",
    title: str = "Fault Resilience by Quantization Level",
    save_path: str | None = None,
) -> "go.Figure":
    """Plot fault resilience comparison across quantization modes.

    Args:
        df: DataFrame from compare_quantization_resilience().
        title: Plot title.
        save_path: Optional path to save as HTML.

    Returns:
        Plotly Figure with one line per quantization mode.
    """
    import plotly.graph_objects as go

    colors = {"fp32": "#636EFA", "fp16": "#EF553B", "dynamic_int8": "#00CC96"}

    fig = go.Figure()

    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]
        grouped = (
            mode_df.groupby("fault_count")
            .agg(
                acc_mean=("accuracy", "mean"),
                acc_std=("accuracy", "std"),
            )
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=grouped["fault_count"],
                y=grouped["acc_mean"],
                error_y=dict(type="data", array=grouped["acc_std"].fillna(0)),
                mode="lines+markers",
                name=mode.upper(),
                line=dict(color=colors.get(mode, "#FFA15A")),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Bit Flips Injected",
        yaxis_title="Accuracy",
        template="plotly_white",
        legend=dict(x=0.02, y=0.02),
    )

    if save_path:
        fig.write_html(save_path)

    return fig
