"""Tests for ONNX model adapter with fault injection.

All tests are skipped when onnxruntime/onnx are not installed.
The lazy-import tests verify that importing the module itself does NOT
require onnxruntime to be present.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Skip the entire module if onnxruntime or onnx are absent.
onnxruntime = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tiny_onnx(tmp_path: Path) -> str:
    """Export a tiny PyTorch model to ONNX for testing."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    model.eval()
    dummy = torch.randn(1, 10)
    path = str(tmp_path / "tiny.onnx")
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def onnx_path(tmp):
    return _create_tiny_onnx(tmp)


@pytest.fixture()
def onnx_model(onnx_path):
    from space_ml_sim.compute.onnx_adapter import OnnxModel

    return OnnxModel(onnx_path)


# ---------------------------------------------------------------------------
# 1. load_onnx returns an OnnxModel
# ---------------------------------------------------------------------------


class TestLoadOnnx:
    def test_load_onnx_returns_onnx_model(self, onnx_path):
        from space_ml_sim.compute.onnx_adapter import OnnxModel, load_onnx

        model = load_onnx(onnx_path)
        assert isinstance(model, OnnxModel)

    def test_load_onnx_accepts_pathlib_path(self, onnx_path):
        from space_ml_sim.compute.onnx_adapter import load_onnx

        model = load_onnx(Path(onnx_path))
        assert model is not None

    def test_load_onnx_raises_file_not_found_for_missing_path(self, tmp):
        from space_ml_sim.compute.onnx_adapter import load_onnx

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            load_onnx(tmp / "nonexistent.onnx")

    def test_onnx_model_constructor_raises_file_not_found(self, tmp):
        from space_ml_sim.compute.onnx_adapter import OnnxModel

        with pytest.raises(FileNotFoundError):
            OnnxModel(tmp / "missing.onnx")


# ---------------------------------------------------------------------------
# 2. Weight inspection
# ---------------------------------------------------------------------------


class TestWeightInspection:
    def test_weight_count_is_positive(self, onnx_model):
        # Linear(10,20) has 200 + 20 = 220 params; Linear(20,5) has 100 + 5 = 105
        assert onnx_model.weight_count > 0

    def test_weight_count_matches_expected(self, onnx_model):
        # 10*20 + 20 + 20*5 + 5 = 200 + 20 + 100 + 5 = 325
        assert onnx_model.weight_count == 325

    def test_layer_names_is_non_empty_list(self, onnx_model):
        names = onnx_model.layer_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_layer_names_are_strings(self, onnx_model):
        for name in onnx_model.layer_names:
            assert isinstance(name, str)

    def test_state_dict_returns_dict(self, onnx_model):
        sd = onnx_model.state_dict()
        assert isinstance(sd, dict)

    def test_state_dict_contains_tensors(self, onnx_model):
        sd = onnx_model.state_dict()
        for v in sd.values():
            assert isinstance(v, torch.Tensor)

    def test_state_dict_keys_match_layer_names(self, onnx_model):
        assert set(onnx_model.state_dict().keys()) == set(onnx_model.layer_names)

    def test_named_parameters_yields_name_tensor_pairs(self, onnx_model):
        pairs = list(onnx_model.named_parameters())
        assert len(pairs) > 0
        for name, param in pairs:
            assert isinstance(name, str)
            assert isinstance(param, torch.nn.Parameter)

    def test_parameters_yields_tensors(self, onnx_model):
        params = list(onnx_model.parameters())
        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.Tensor)


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------


class TestInference:
    def test_call_with_torch_tensor_returns_tensor(self, onnx_model):
        x = torch.randn(1, 10)
        out = onnx_model(x)
        assert isinstance(out, torch.Tensor)

    def test_output_shape_is_correct(self, onnx_model):
        x = torch.randn(4, 10)
        out = onnx_model(x)
        assert out.shape == (4, 5)

    def test_call_with_numpy_array_returns_tensor(self, onnx_model):
        x = np.random.randn(2, 10).astype(np.float32)
        out = onnx_model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 5)

    def test_inference_is_deterministic_without_faults(self, onnx_model):
        x = torch.randn(1, 10)
        out1 = onnx_model(x)
        out2 = onnx_model(x)
        assert torch.allclose(out1, out2)

    def test_batch_size_one(self, onnx_model):
        x = torch.randn(1, 10)
        out = onnx_model(x)
        assert out.shape == (1, 5)

    def test_batch_size_eight(self, onnx_model):
        x = torch.randn(8, 10)
        out = onnx_model(x)
        assert out.shape == (8, 5)


# ---------------------------------------------------------------------------
# 4. Fault injection — zero faults
# ---------------------------------------------------------------------------


class TestZeroFaultInjection:
    def test_inject_faults_zero_returns_zero(self, onnx_model):
        result = onnx_model.inject_faults(0)
        assert result == 0

    def test_inject_faults_zero_does_not_change_output(self, onnx_model):
        x = torch.randn(1, 10)
        out_before = onnx_model(x).clone()
        onnx_model.inject_faults(0)
        out_after = onnx_model(x)
        assert torch.allclose(out_before, out_after)

    def test_inject_faults_negative_treated_as_zero(self, onnx_model):
        result = onnx_model.inject_faults(-5)
        assert result == 0


# ---------------------------------------------------------------------------
# 5. Fault injection — non-zero faults
# ---------------------------------------------------------------------------


class TestNonZeroFaultInjection:
    def test_inject_faults_returns_positive_count(self, onnx_model):
        injected = onnx_model.inject_faults(100)
        assert injected > 0

    def test_inject_faults_changes_output(self, onnx_model):
        """Injecting 100 bit-flips into 325 weights should alter at least one output."""
        x = torch.randn(1, 10)
        out_before = onnx_model(x).clone()
        onnx_model.inject_faults(100)
        out_after = onnx_model(x)
        assert not torch.allclose(out_before, out_after), (
            "Expected output to change after fault injection"
        )

    def test_inject_faults_does_not_exceed_requested_count(self, onnx_model):
        injected = onnx_model.inject_faults(10)
        assert injected <= 10

    def test_inject_faults_single_fault(self, onnx_model):
        injected = onnx_model.inject_faults(1)
        assert injected >= 1

    def test_multiple_inject_calls_are_cumulative(self, onnx_model):
        """Each inject_faults call further degrades the model."""
        x = torch.randn(1, 10)
        out_clean = onnx_model(x).clone()

        onnx_model.inject_faults(50)
        out_after_first = onnx_model(x).clone()

        onnx_model.inject_faults(50)
        out_after_second = onnx_model(x).clone()

        # After two rounds of injection the model should differ from clean state
        assert not torch.allclose(out_clean, out_after_second)
        # The two faulted outputs should also differ (probabilistically)
        # This is not guaranteed, but with 50 flips into 325 weights it's very likely
        assert not torch.allclose(out_after_first, out_after_second)


# ---------------------------------------------------------------------------
# 6. Lazy-import / optional dependency behaviour
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_importing_module_does_not_raise_without_onnxruntime(self):
        """The module-level import must not fail even if onnxruntime is absent.

        We verify this by temporarily hiding onnxruntime from sys.modules and
        re-importing the adapter.  If it raises at import time the test fails.
        """
        # Save real modules
        saved_onnx = sys.modules.pop("onnx", None)
        saved_ort = sys.modules.pop("onnxruntime", None)
        saved_adapter = sys.modules.pop("space_ml_sim.compute.onnx_adapter", None)

        # Pretend the packages are absent
        sys.modules["onnx"] = None  # type: ignore[assignment]
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]

        try:
            # Should NOT raise ImportError at module import time
            import space_ml_sim.compute.onnx_adapter  # noqa: F401
        except ImportError:
            pytest.fail("Importing onnx_adapter raised ImportError at module level")
        finally:
            # Restore everything
            if saved_onnx is not None:
                sys.modules["onnx"] = saved_onnx
            else:
                sys.modules.pop("onnx", None)

            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort
            else:
                sys.modules.pop("onnxruntime", None)

            if saved_adapter is not None:
                sys.modules["space_ml_sim.compute.onnx_adapter"] = saved_adapter
            else:
                sys.modules.pop("space_ml_sim.compute.onnx_adapter", None)

    def test_require_onnx_succeeds_when_packages_installed(self):
        from space_ml_sim.compute.onnx_adapter import _require_onnx

        onnx_mod, ort = _require_onnx()
        assert onnx_mod is not None
        assert ort is not None

    def test_require_onnx_raises_import_error_with_helpful_message(self):
        """_require_onnx must raise ImportError with pip install hint when missing."""
        saved_onnx = sys.modules.pop("onnx", None)
        saved_ort = sys.modules.pop("onnxruntime", None)
        # Also remove the adapter so _require_onnx re-runs its try/except
        saved_adapter = sys.modules.pop("space_ml_sim.compute.onnx_adapter", None)

        sys.modules["onnx"] = None  # type: ignore[assignment]
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]

        try:
            import space_ml_sim.compute.onnx_adapter as adapter_mod

            importlib.reload(adapter_mod)  # reload with patched sys.modules
            with pytest.raises(ImportError, match="pip install space-ml-sim\\[onnx\\]"):
                adapter_mod._require_onnx()
        finally:
            if saved_onnx is not None:
                sys.modules["onnx"] = saved_onnx
            else:
                sys.modules.pop("onnx", None)

            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort
            else:
                sys.modules.pop("onnxruntime", None)

            if saved_adapter is not None:
                sys.modules["space_ml_sim.compute.onnx_adapter"] = saved_adapter
            else:
                sys.modules.pop("space_ml_sim.compute.onnx_adapter", None)


# ---------------------------------------------------------------------------
# 7. compute __init__.py re-exports OnnxModel and load_onnx
# ---------------------------------------------------------------------------


class TestComputeInitExports:
    def test_onnx_model_importable_from_compute(self):
        from space_ml_sim.compute import OnnxModel  # noqa: F401

    def test_load_onnx_importable_from_compute(self):
        from space_ml_sim.compute import load_onnx  # noqa: F401

    def test_compute_package_still_exports_existing_symbols(self):
        from space_ml_sim.compute import (  # noqa: F401
            FaultInjector,
            FaultReport,
            TMRWrapper,
            CheckpointManager,
            InferenceScheduler,
        )


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_inject_faults_with_large_count(self, onnx_model):
        """Injecting more faults than weight elements should not crash."""
        injected = onnx_model.inject_faults(10_000)
        assert injected > 0

    def test_weight_tensors_are_float32(self, onnx_model):
        for _, tensor in onnx_model.named_parameters():
            assert tensor.dtype == torch.float32

    def test_named_parameters_count_equals_layer_names_count(self, onnx_model):
        pairs = list(onnx_model.named_parameters())
        assert len(pairs) == len(onnx_model.layer_names)

    def test_state_dict_is_immutable_copy(self, onnx_model):
        """Mutating the returned state dict must not change internal weights."""
        sd = onnx_model.state_dict()
        original_val = sd[onnx_model.layer_names[0]].clone()

        # Mutate the returned dict value
        sd[onnx_model.layer_names[0]].fill_(999.0)

        # Internal weights should be unchanged
        internal = list(onnx_model._weights.values())[0]
        assert torch.allclose(internal, original_val), (
            "state_dict() should return a copy, not a reference to internal tensors"
        )

    def test_inject_faults_returns_int(self, onnx_model):
        result = onnx_model.inject_faults(5)
        assert isinstance(result, int)
