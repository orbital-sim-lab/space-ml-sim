"""Plotly-based visualization for fault injection and constellation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go


def plot_fault_sweep(
    df: "pd.DataFrame",
    title: str = "Accuracy vs Fault Count",
    save_path: str | None = None,
) -> "go.Figure":
    """Plot fault sweep results showing accuracy degradation.

    Args:
        df: DataFrame from FaultInjector.sweep() with columns:
            fault_count, trial, accuracy, top5_accuracy, critical_failure.
        title: Plot title.
        save_path: Optional path to save as HTML.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    # Aggregate by fault_count: mean and std of accuracy
    grouped = df.groupby("fault_count").agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        top5_mean=("top5_accuracy", "mean"),
        critical_rate=("critical_failure", "mean"),
    ).reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grouped["fault_count"],
            y=grouped["acc_mean"],
            error_y=dict(type="data", array=grouped["acc_std"].fillna(0)),
            mode="lines+markers",
            name="Top-1 Accuracy",
            line=dict(color="#636EFA"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["fault_count"],
            y=grouped["top5_mean"],
            mode="lines+markers",
            name="Top-5 Accuracy",
            line=dict(color="#00CC96", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["fault_count"],
            y=grouped["critical_rate"],
            mode="lines+markers",
            name="Critical Failure Rate",
            line=dict(color="#EF553B", dash="dot"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Bit Flips Injected",
        yaxis_title="Accuracy",
        yaxis2=dict(title="Critical Failure Rate", overlaying="y", side="right", range=[0, 1]),
        template="plotly_white",
        legend=dict(x=0.02, y=0.02),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_constellation_health(
    metrics_history: list[dict],
    title: str = "Constellation Health Over Time",
    save_path: str | None = None,
) -> "go.Figure":
    """Plot constellation health metrics over simulation time.

    Args:
        metrics_history: List of metric dicts from Constellation.step().
        title: Plot title.
        save_path: Optional path to save as HTML.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    times = [m["sim_time"] / 60 for m in metrics_history]  # Convert to minutes

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Satellite Status", "Temperature & SEU Events"),
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Scatter(x=times, y=[m["active_count"] for m in metrics_history],
                   name="Active", line=dict(color="#00CC96")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=[m["degraded_count"] for m in metrics_history],
                   name="Degraded", line=dict(color="#FFA15A")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=[m["failed_count"] for m in metrics_history],
                   name="Failed", line=dict(color="#EF553B")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=times, y=[m["avg_temperature_c"] for m in metrics_history],
                   name="Avg Temp (C)", line=dict(color="#636EFA")),
        row=2, col=1,
    )

    fig.update_layout(title=title, template="plotly_white")
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)

    if save_path:
        fig.write_html(save_path)

    return fig
