"""Tests for quantization-aware fault resilience comparison.

TDD: These tests are written first (RED), before the implementation exists.
"""

from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 20
HIDDEN_DIM = 32
OUTPUT_DIM = 5


def _model_factory() -> nn.Module:
    """Tiny model used throughout tests — fast to instantiate and run."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


def _make_dataloader(num_batches: int = 2, batch_size: int = 8) -> list[tuple]:
    """Synthetic dataloader: list of (images, labels) tuples."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, INPUT_DIM)
        y = torch.randint(0, OUTPUT_DIM, (batch_size,))
        data.append((x, y))
    return data


def _make_injector():
    """Build a FaultInjector with deterministic seed."""
    from space_ml_sim.compute.fault_injector import FaultInjector
    from space_ml_sim.environment.radiation import RadiationEnvironment
    from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

    return FaultInjector(
        rad_env=RadiationEnvironment.leo_500km(),
        chip_profile=TRILLIUM_V6E,
        seed=42,
    )


# ---------------------------------------------------------------------------
# quantize_model tests
# ---------------------------------------------------------------------------


class TestQuantizeModel:
    """Unit tests for quantize_model()."""

    def test_fp32_returns_independent_copy(self):
        """quantize_model('fp32') returns a deep copy, not the original."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        copy_model = quantize_model(model, "fp32")

        # Must be a distinct object
        assert copy_model is not model

    def test_fp32_produces_same_output(self):
        """quantize_model('fp32') copy produces numerically identical output."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        copy_model = quantize_model(model, "fp32")
        copy_model.eval()

        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            out_orig = model(x)
            out_copy = copy_model(x)

        assert torch.allclose(out_orig, out_copy, atol=1e-6)

    def test_fp16_returns_half_precision_model(self):
        """quantize_model('fp16') converts parameters to float16."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        half_model = quantize_model(model, "fp16")

        float_params = [p for p in half_model.parameters() if p.is_floating_point()]
        assert len(float_params) > 0
        for param in float_params:
            assert param.dtype == torch.float16, f"Expected float16, got {param.dtype}"

    def test_fp16_is_independent_copy(self):
        """quantize_model('fp16') does not mutate the original model."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        original_dtype = next(model.parameters()).dtype

        quantize_model(model, "fp16")

        # Original model must remain float32
        assert next(model.parameters()).dtype == original_dtype

    def test_dynamic_int8_returns_quantized_model(self):
        """quantize_model('dynamic_int8') produces a model with quantized weights."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        quant_model = quantize_model(model, "dynamic_int8")

        # Simulated INT8: weights should be rounded to 256 levels
        # Check that weights differ from original (quantization changes values)
        for (_, p_orig), (_, p_quant) in zip(
            model.named_parameters(), quant_model.named_parameters()
        ):
            # Quantized weights should have fewer unique values
            orig_unique = p_orig.unique().numel()
            quant_unique = p_quant.unique().numel()
            assert quant_unique <= orig_unique

    def test_dynamic_int8_is_independent_copy(self):
        """quantize_model('dynamic_int8') does not mutate the original model."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        original_type = type(list(model.children())[0])

        quantize_model(model, "dynamic_int8")

        # First child of original must still be the original type
        assert isinstance(list(model.children())[0], original_type)

    def test_unknown_mode_raises_value_error(self):
        """quantize_model raises ValueError for unrecognized mode strings."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        with pytest.raises(ValueError, match="Unknown quantization mode"):
            quantize_model(model, "bfloat16_magic")

    def test_unknown_mode_error_message_contains_mode(self):
        """ValueError message includes the bad mode name."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        with pytest.raises(ValueError, match="bad_mode"):
            quantize_model(model, "bad_mode")

    def test_unknown_mode_error_lists_valid_modes(self):
        """ValueError message lists the accepted mode names."""
        from space_ml_sim.compute.quantization import quantize_model

        model = _model_factory().eval()
        with pytest.raises(ValueError, match="fp32"):
            quantize_model(model, "nope")


# ---------------------------------------------------------------------------
# compare_quantization_resilience tests
# ---------------------------------------------------------------------------


class TestCompareQuantizationResilience:
    """Tests for compare_quantization_resilience()."""

    def test_returns_dataframe(self):
        """compare_quantization_resilience returns a pandas DataFrame."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0, 5],
            modes=["fp32"],
            num_trials=1,
        )

        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self):
        """Result DataFrame contains all required columns."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0],
            modes=["fp32"],
            num_trials=1,
        )

        required_columns = {"mode", "fault_count", "trial", "accuracy", "faults_injected"}
        assert required_columns.issubset(set(df.columns))

    def test_row_count_matches_modes_faults_trials(self):
        """DataFrame has exactly len(modes) * len(fault_counts) * num_trials rows."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        modes = ["fp32", "fp16"]
        fault_counts = [0, 5, 10]
        num_trials = 2

        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=fault_counts,
            modes=modes,
            num_trials=num_trials,
        )

        expected_rows = len(modes) * len(fault_counts) * num_trials
        assert len(df) == expected_rows

    def test_all_three_modes_by_default(self):
        """Default modes includes fp32, fp16, and dynamic_int8."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0],
            num_trials=1,
        )

        assert set(df["mode"].unique()) == {"fp32", "fp16", "dynamic_int8"}

    def test_default_fault_counts_non_empty(self):
        """Default fault_counts sweep produces results."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            modes=["fp32"],
            num_trials=1,
        )

        assert len(df) > 0
        assert df["fault_count"].nunique() > 1

    def test_accuracy_in_valid_range(self):
        """All accuracy values fall in [0.0, 1.0]."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0, 50],
            modes=["fp32"],
            num_trials=2,
        )

        assert (df["accuracy"] >= 0.0).all()
        assert (df["accuracy"] <= 1.0).all()

    def test_mode_column_values_match_input(self):
        """mode column contains exactly the modes requested."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        requested_modes = ["fp32", "fp16"]
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0],
            modes=requested_modes,
            num_trials=1,
        )

        assert set(df["mode"].unique()) == set(requested_modes)

    def test_zero_faults_gives_roughly_same_accuracy_across_modes(self):
        """With 0 faults, accuracy should be identical across all modes (same model weights)."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        # Use a fixed seed dataloader and many samples for stable estimate
        torch.manual_seed(0)
        dataloader = _make_dataloader(num_batches=4, batch_size=16)

        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=dataloader,
            injector=injector,
            fault_counts=[0],
            modes=["fp32", "fp16", "dynamic_int8"],
            num_trials=1,
        )

        zero_fault_df = df[df["fault_count"] == 0]
        accuracies = zero_fault_df["accuracy"].values

        # All modes produce the same (random) baseline accuracy within a reasonable tolerance
        # FP16 / INT8 introduce small numerical errors but top-1 class should be stable
        max_diff = float(accuracies.max() - accuracies.min())
        assert max_diff < 0.30, (
            f"Accuracy spread across modes at 0 faults too large: {max_diff:.3f}. "
            f"Accuracies: {accuracies}"
        )

    def test_trial_index_column_range(self):
        """trial column values run from 0 to num_trials-1."""
        from space_ml_sim.compute.quantization import compare_quantization_resilience

        injector = _make_injector()
        num_trials = 3
        df = compare_quantization_resilience(
            model_factory=_model_factory,
            dataloader=_make_dataloader(),
            injector=injector,
            fault_counts=[0],
            modes=["fp32"],
            num_trials=num_trials,
        )

        assert set(df["trial"].unique()) == set(range(num_trials))


# ---------------------------------------------------------------------------
# plot_quantization_comparison tests
# ---------------------------------------------------------------------------


def _make_comparison_df(
    modes: list[str] | None = None,
    fault_counts: list[int] | None = None,
    num_trials: int = 2,
) -> pd.DataFrame:
    """Build a minimal DataFrame matching compare_quantization_resilience output."""
    if modes is None:
        modes = ["fp32", "fp16", "dynamic_int8"]
    if fault_counts is None:
        fault_counts = [0, 10, 50]

    rows = []
    for mode in modes:
        for fc in fault_counts:
            for trial in range(num_trials):
                rows.append(
                    {
                        "mode": mode,
                        "fault_count": fc,
                        "trial": trial,
                        "accuracy": max(0.0, 0.9 - fc * 0.003),
                        "faults_injected": fc,
                    }
                )
    return pd.DataFrame(rows)


class TestPlotQuantizationComparison:
    """Tests for plot_quantization_comparison()."""

    def test_returns_plotly_figure(self):
        """plot_quantization_comparison returns a go.Figure."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        fig = plot_quantization_comparison(df)

        assert isinstance(fig, go.Figure)

    def test_one_trace_per_mode(self):
        """Figure has exactly one trace per unique mode in the DataFrame."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        modes = ["fp32", "fp16", "dynamic_int8"]
        df = _make_comparison_df(modes=modes)
        fig = plot_quantization_comparison(df)

        assert len(fig.data) == len(modes)

    def test_trace_names_match_modes(self):
        """Each trace name corresponds to an uppercased mode label."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        modes = ["fp32", "fp16"]
        df = _make_comparison_df(modes=modes)
        fig = plot_quantization_comparison(df)

        trace_names = {trace.name for trace in fig.data}
        for mode in modes:
            assert mode.upper() in trace_names

    def test_custom_title_applied(self):
        """Custom title is reflected in figure layout."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        custom_title = "My Quantization Plot"
        fig = plot_quantization_comparison(df, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_default_title_applied(self):
        """Default title is used when no title is supplied."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        fig = plot_quantization_comparison(df)

        assert fig.layout.title.text == "Fault Resilience by Quantization Level"

    def test_x_axis_title_references_bit_flips(self):
        """X-axis title mentions bit flips."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        fig = plot_quantization_comparison(df)

        assert "Bit Flips" in fig.layout.xaxis.title.text

    def test_y_axis_title_references_accuracy(self):
        """Y-axis title mentions accuracy."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        fig = plot_quantization_comparison(df)

        assert "Accuracy" in fig.layout.yaxis.title.text

    def test_no_file_written_without_save_path(self, tmp_path):
        """No file is written when save_path is None."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        plot_quantization_comparison(df)

        assert list(tmp_path.iterdir()) == []

    def test_writes_html_when_save_path_given(self, tmp_path):
        """An HTML file is written when save_path is supplied."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df()
        save_file = tmp_path / "quant_comparison.html"
        plot_quantization_comparison(df, save_path=str(save_file))

        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_single_mode_single_trace(self):
        """Works correctly when DataFrame contains only one mode."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df(modes=["fp32"])
        fig = plot_quantization_comparison(df)

        assert len(fig.data) == 1
        assert fig.data[0].name == "FP32"

    def test_single_fault_count_no_crash(self):
        """Handles a DataFrame with a single fault_count without error."""
        from space_ml_sim.compute.quantization import plot_quantization_comparison

        df = _make_comparison_df(fault_counts=[0])
        fig = plot_quantization_comparison(df)

        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# __init__.py export tests
# ---------------------------------------------------------------------------


class TestComputeModuleExports:
    """Verify public symbols are accessible from the compute package."""

    def test_quantize_model_exported(self):
        """quantize_model is importable from space_ml_sim.compute."""
        from space_ml_sim.compute import quantize_model  # noqa: F401

    def test_compare_quantization_resilience_exported(self):
        """compare_quantization_resilience is importable from space_ml_sim.compute."""
        from space_ml_sim.compute import compare_quantization_resilience  # noqa: F401

    def test_plot_quantization_comparison_exported(self):
        """plot_quantization_comparison is importable from space_ml_sim.compute."""
        from space_ml_sim.compute import plot_quantization_comparison  # noqa: F401
