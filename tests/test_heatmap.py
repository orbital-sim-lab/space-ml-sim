"""TDD tests for viz/heatmap.py -- per-layer sensitivity heatmap.

Written BEFORE implementation (RED phase). Each test must fail until
sensitivity_heatmap and sensitivity_data are implemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import plotly.graph_objects as go

from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import GOOGLE_TRILLIUM_V6E


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def injector() -> FaultInjector:
    return FaultInjector(
        rad_env=RadiationEnvironment.leo_500km(),
        chip_profile=GOOGLE_TRILLIUM_V6E,
        seed=0,
    )


def _tiny_model_factory():
    """Return a fresh 3-layer Linear model with deterministic weights.

    Architecture: Linear(8, 16) -> ReLU -> Linear(16, 8) -> ReLU -> Linear(8, 4)
    Input dim: 8, output dim: 4 (4-class classification).
    """

    def factory() -> nn.Module:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        for p in model.parameters():
            nn.init.normal_(p, mean=0.0, std=0.1)
        return model

    return factory


def _make_dataloader(
    n_samples: int = 32,
    input_dim: int = 8,
    num_classes: int = 4,
    batch_size: int = 16,
) -> torch.utils.data.DataLoader:
    """Create a small synthetic DataLoader with random tensors and integer labels."""
    torch.manual_seed(1)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Trainable param names for the tiny model (ReLU has no params)
_EXPECTED_PARAM_NAMES = {"0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"}


# ---------------------------------------------------------------------------
# sensitivity_heatmap tests
# ---------------------------------------------------------------------------


class TestSensitivityHeatmap:
    """Tests for sensitivity_heatmap()."""

    def test_returns_plotly_figure(self, injector):
        """sensitivity_heatmap must return a plotly Figure."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert isinstance(fig, go.Figure)

    def test_figure_has_one_bar_per_trainable_param(self, injector):
        """Figure bar count equals number of trainable parameter tensors."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        # Tiny model has 6 trainable param tensors (3 weight + 3 bias)
        assert len(fig.data) == 1  # single Bar trace
        bar_trace = fig.data[0]
        assert len(bar_trace.y) == len(_EXPECTED_PARAM_NAMES)

    def test_bar_y_labels_match_param_names(self, injector):
        """Bar y-axis labels are the layer parameter names."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        bar_trace = fig.data[0]
        assert set(bar_trace.y) == _EXPECTED_PARAM_NAMES

    def test_baseline_accuracy_appears_in_title(self, injector):
        """Figure title contains the baseline accuracy percentage string."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        # baseline accuracy is formatted as XX.XX%
        assert "%" in fig.layout.title.text

    def test_custom_title_appears_in_layout(self, injector):
        """A custom title string appears in the figure layout title."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        custom_title = "My Custom Heatmap"
        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
            title=custom_title,
        )
        assert custom_title in fig.layout.title.text

    def test_x_axis_describes_accuracy_drop(self, injector):
        """X-axis title references accuracy drop."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert "Accuracy" in fig.layout.xaxis.title.text

    def test_no_file_written_without_save_path(self, injector, tmp_path):
        """No HTML file is written when save_path is None."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert list(tmp_path.iterdir()) == []

    def test_writes_html_when_save_path_given(self, injector, tmp_path):
        """HTML file is written to save_path when provided."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        save_file = tmp_path / "heatmap.html"
        sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
            save_path=str(save_file),
        )
        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_most_vulnerable_layer_listed_first_on_yaxis(self, injector):
        """The bar with the largest accuracy drop is listed at the top of the y-axis.

        Plotly horizontal bars with yaxis autorange='reversed' render the first
        element at the top, so index 0 of bar_trace.x should be the largest value.
        """
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=10,
            num_trials=2,
        )
        bar_trace = fig.data[0]
        drops = list(bar_trace.x)
        assert drops[0] == max(drops), "Most vulnerable layer should be first (index 0)"

    def test_bar_orientation_is_horizontal(self, injector):
        """Bar chart orientation is horizontal ('h')."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert fig.data[0].orientation == "h"


# ---------------------------------------------------------------------------
# Empty model (no trainable params) edge case
# ---------------------------------------------------------------------------


class TestSensitivityHeatmapEmptyModel:
    """Edge case: a model with no trainable parameters."""

    @staticmethod
    def _frozen_model_factory():
        def factory() -> nn.Module:
            model = nn.Sequential(nn.Linear(4, 2))
            for p in model.parameters():
                p.requires_grad_(False)
            return model

        return factory

    def test_empty_model_returns_figure(self, injector):
        """sensitivity_heatmap returns a Figure even when model has no trainable params."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=self._frozen_model_factory(),
            dataloader=_make_dataloader(input_dim=4, num_classes=2),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert isinstance(fig, go.Figure)

    def test_empty_model_figure_has_no_bars(self, injector):
        """A model with no trainable layers produces a figure with zero bars."""
        from space_ml_sim.viz.heatmap import sensitivity_heatmap

        fig = sensitivity_heatmap(
            model_factory=self._frozen_model_factory(),
            dataloader=_make_dataloader(input_dim=4, num_classes=2),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        bar_trace = fig.data[0]
        assert len(bar_trace.y) == 0


# ---------------------------------------------------------------------------
# sensitivity_data tests
# ---------------------------------------------------------------------------


class TestSensitivityData:
    """Tests for sensitivity_data()."""

    def test_returns_dict(self, injector):
        """sensitivity_data returns a dict."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        result = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert isinstance(result, dict)

    def test_keys_match_trainable_param_names(self, injector):
        """Dict keys equal the trainable parameter names of the model."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        result = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert set(result.keys()) == _EXPECTED_PARAM_NAMES

    def test_all_values_are_floats(self, injector):
        """Every value in the returned dict is a Python float."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        result = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        for key, value in result.items():
            assert isinstance(value, float), f"Expected float for {key}, got {type(value)}"

    def test_dict_sorted_descending_by_value(self, injector):
        """Dict is ordered by accuracy drop descending (most vulnerable first)."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        result = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=10,
            num_trials=2,
        )
        values = list(result.values())
        assert values == sorted(values, reverse=True), "Dict values must be in descending order"

    def test_empty_model_returns_empty_dict(self, injector):
        """A model with no trainable parameters returns an empty dict."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        def frozen_factory() -> nn.Module:
            model = nn.Sequential(nn.Linear(4, 2))
            for p in model.parameters():
                p.requires_grad_(False)
            return model

        result = sensitivity_data(
            model_factory=frozen_factory,
            dataloader=_make_dataloader(input_dim=4, num_classes=2),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        assert result == {}

    def test_multiple_trials_average_drops(self, injector):
        """Running multiple trials produces averaged results (result is still a float per key)."""
        from space_ml_sim.viz.heatmap import sensitivity_data

        result_1 = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=1,
        )
        result_3 = sensitivity_data(
            model_factory=_tiny_model_factory(),
            dataloader=_make_dataloader(),
            injector=injector,
            faults_per_layer=5,
            num_trials=3,
        )
        # Both should have the same keys and float values
        assert set(result_1.keys()) == set(result_3.keys())
        for v in result_3.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# __init__.py export tests
# ---------------------------------------------------------------------------


class TestVizModuleExports:
    """Verify that sensitivity_heatmap and sensitivity_data are exported from viz."""

    def test_sensitivity_heatmap_importable_from_viz(self):
        """sensitivity_heatmap is importable from space_ml_sim.viz."""
        from space_ml_sim.viz import sensitivity_heatmap  # noqa: F401

    def test_sensitivity_data_importable_from_viz(self):
        """sensitivity_data is importable from space_ml_sim.viz."""
        from space_ml_sim.viz import sensitivity_data  # noqa: F401

    def test_both_in_all(self):
        """Both symbols appear in space_ml_sim.viz.__all__."""
        import space_ml_sim.viz as viz_module

        assert "sensitivity_heatmap" in viz_module.__all__
        assert "sensitivity_data" in viz_module.__all__
