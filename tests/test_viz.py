"""Tests for viz/plots.py."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from space_ml_sim.viz.plots import plot_constellation_health, plot_fault_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fault_sweep_df(num_fault_counts: int = 3, trials_per_count: int = 2) -> pd.DataFrame:
    """Build a minimal DataFrame matching the schema expected by plot_fault_sweep."""
    rows = []
    for fault_count in range(num_fault_counts):
        for trial in range(trials_per_count):
            rows.append(
                {
                    "fault_count": fault_count,
                    "trial": trial,
                    "accuracy": 0.9 - fault_count * 0.1,
                    "top5_accuracy": 0.95 - fault_count * 0.05,
                    "critical_failure": 1 if fault_count >= 2 else 0,
                }
            )
    return pd.DataFrame(rows)


def _make_metrics_history(num_steps: int = 4) -> list[dict]:
    """Build a minimal metrics history list matching the schema expected by plot_constellation_health."""
    history = []
    for step in range(num_steps):
        history.append(
            {
                "sim_time": step * 60.0,  # seconds
                "active_count": 10 - step,
                "degraded_count": step,
                "failed_count": 0,
                "avg_temperature_c": 25.0 + step * 2.0,
            }
        )
    return history


# ---------------------------------------------------------------------------
# plot_fault_sweep tests
# ---------------------------------------------------------------------------


class TestPlotFaultSweep:
    """Tests for plot_fault_sweep function."""

    def test_returns_plotly_figure(self):
        """plot_fault_sweep returns a plotly Figure object."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        assert isinstance(fig, go.Figure)

    def test_figure_has_three_traces(self):
        """Figure contains Top-1, Top-5 accuracy and critical failure rate traces."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        assert len(fig.data) == 3

    def test_trace_names(self):
        """Traces are named for Top-1 Accuracy, Top-5 Accuracy, and Critical Failure Rate."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        names = {trace.name for trace in fig.data}
        assert "Top-1 Accuracy" in names
        assert "Top-5 Accuracy" in names
        assert "Critical Failure Rate" in names

    def test_no_file_written_without_save_path(self, tmp_path):
        """No HTML file is written when save_path is None."""
        df = _make_fault_sweep_df()
        plot_fault_sweep(df)

        # Confirm no files were created in a monitored temp directory
        assert list(tmp_path.iterdir()) == []

    def test_writes_html_when_save_path_given(self, tmp_path):
        """plot_fault_sweep writes an HTML file when save_path is provided."""
        df = _make_fault_sweep_df()
        save_file = tmp_path / "fault_sweep.html"
        plot_fault_sweep(df, save_path=str(save_file))

        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_custom_title_applied(self):
        """Custom title is reflected in figure layout."""
        df = _make_fault_sweep_df()
        custom_title = "My Custom Fault Plot"
        fig = plot_fault_sweep(df, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_default_title_applied(self):
        """Default title is used when no title argument is passed."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        assert fig.layout.title.text == "Accuracy vs Fault Count"

    def test_single_fault_count_row(self):
        """Works with a DataFrame that has only one distinct fault_count value."""
        df = pd.DataFrame(
            [
                {
                    "fault_count": 0,
                    "trial": 0,
                    "accuracy": 0.9,
                    "top5_accuracy": 0.95,
                    "critical_failure": 0,
                }
            ]
        )
        fig = plot_fault_sweep(df)

        assert isinstance(fig, go.Figure)

    def test_many_trials_per_fault_count(self):
        """Handles aggregation of many trials without error."""
        df = _make_fault_sweep_df(num_fault_counts=5, trials_per_count=20)
        fig = plot_fault_sweep(df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_x_axis_title_set(self):
        """X-axis title is labelled with fault count description."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        assert "Bit Flips" in fig.layout.xaxis.title.text

    def test_y_axis_title_set(self):
        """Y-axis title is set to Accuracy."""
        df = _make_fault_sweep_df()
        fig = plot_fault_sweep(df)

        assert "Accuracy" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------------------
# plot_constellation_health tests
# ---------------------------------------------------------------------------


class TestPlotConstellationHealth:
    """Tests for plot_constellation_health function."""

    def test_returns_plotly_figure(self):
        """plot_constellation_health returns a plotly Figure object."""
        history = _make_metrics_history()
        fig = plot_constellation_health(history)

        assert isinstance(fig, go.Figure)

    def test_figure_has_four_traces(self):
        """Figure has traces for active, degraded, failed counts, and avg temperature."""
        history = _make_metrics_history()
        fig = plot_constellation_health(history)

        # active, degraded, failed (row 1) + avg temp (row 2)
        assert len(fig.data) == 4

    def test_trace_names_include_status_labels(self):
        """Traces include Active, Degraded, and Failed status series."""
        history = _make_metrics_history()
        fig = plot_constellation_health(history)

        names = {trace.name for trace in fig.data}
        assert "Active" in names
        assert "Degraded" in names
        assert "Failed" in names

    def test_temperature_trace_present(self):
        """Temperature trace is included in the figure."""
        history = _make_metrics_history()
        fig = plot_constellation_health(history)

        names = {trace.name for trace in fig.data}
        assert "Avg Temp (C)" in names

    def test_no_file_written_without_save_path(self, tmp_path):
        """No HTML file is written when save_path is None."""
        history = _make_metrics_history()
        plot_constellation_health(history)

        assert list(tmp_path.iterdir()) == []

    def test_writes_html_when_save_path_given(self, tmp_path):
        """plot_constellation_health writes an HTML file when save_path is provided."""
        history = _make_metrics_history()
        save_file = tmp_path / "health.html"
        plot_constellation_health(history, save_path=str(save_file))

        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_custom_title_applied(self):
        """Custom title is set in figure layout."""
        history = _make_metrics_history()
        custom_title = "My Health Plot"
        fig = plot_constellation_health(history, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_default_title_applied(self):
        """Default title is used when no title argument is supplied."""
        history = _make_metrics_history()
        fig = plot_constellation_health(history)

        assert fig.layout.title.text == "Constellation Health Over Time"

    def test_single_step_history(self):
        """Works correctly with a single-entry metrics history."""
        history = _make_metrics_history(num_steps=1)
        fig = plot_constellation_health(history)

        assert isinstance(fig, go.Figure)

    def test_time_converted_to_minutes(self):
        """X-axis data represents time in minutes, not seconds."""
        history = [
            {
                "sim_time": 120.0,  # 2 minutes in seconds
                "active_count": 5,
                "degraded_count": 1,
                "failed_count": 0,
                "avg_temperature_c": 30.0,
            }
        ]
        fig = plot_constellation_health(history)

        # First trace x-values should be in minutes (120s -> 2.0 min)
        first_trace_x = list(fig.data[0].x)
        assert first_trace_x == pytest.approx([2.0])

    def test_large_history(self):
        """Handles a long simulation history without error."""
        history = _make_metrics_history(num_steps=1000)
        fig = plot_constellation_health(history)

        assert isinstance(fig, go.Figure)
