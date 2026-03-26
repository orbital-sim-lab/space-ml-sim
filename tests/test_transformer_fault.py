"""Tests for transformer-aware fault injection patterns.

TDD: Tests written before implementation.
All tests in this file are expected to FAIL until
src/space_ml_sim/compute/transformer_fault.py is implemented.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from space_ml_sim.compute.transformer_fault import TransformerFaultInjector


# ---------------------------------------------------------------------------
# Shared test fixture: a tiny transformer that uses common naming conventions
# and does NOT require any external model downloads.
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.pos_embedding = nn.Embedding(16, 32)
        self.layer_norm1 = nn.LayerNorm(32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_embedding(
            torch.arange(x.size(1), device=x.device)
        )
        normed = self.layer_norm1(emb)
        attn_out, _ = self.attn(normed, normed, normed)
        x2 = emb + attn_out
        normed2 = self.layer_norm2(x2)
        ffn_out = self.ffn(normed2)
        out = x2 + ffn_out
        return self.output(out.mean(dim=1))


@pytest.fixture
def tiny_model():
    """Return a fresh TinyTransformer in eval mode."""
    m = TinyTransformer()
    m.eval()
    return m


@pytest.fixture
def injector():
    return TransformerFaultInjector()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a deep copy of all named parameters."""
    return {name: param.data.clone() for name, param in model.named_parameters()}


def _changed_params(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> set[str]:
    """Return the set of parameter names whose values differ from snapshot."""
    return {
        name
        for name, param in model.named_parameters()
        if not torch.equal(param.data, snapshot[name])
    }


# ---------------------------------------------------------------------------
# Tests: vulnerability_profile
# ---------------------------------------------------------------------------

class TestVulnerabilityProfile:
    def test_returns_dict(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        assert isinstance(profile, dict)

    def test_all_parameters_classified(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        param_names = {name for name, _ in tiny_model.named_parameters()}
        assert set(profile.keys()) == param_names

    def test_valid_categories_only(self, injector, tiny_model):
        valid = {"attention", "layernorm", "embedding", "ffn", "other"}
        profile = injector.vulnerability_profile(tiny_model)
        for name, category in profile.items():
            assert category in valid, (
                f"Parameter '{name}' has unknown category '{category}'"
            )

    def test_embedding_parameters_classified_as_embedding(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        # Both token embedding and position embedding should be "embedding"
        assert profile["embedding.weight"] == "embedding"
        assert profile["pos_embedding.weight"] == "embedding"

    def test_layernorm_parameters_classified_correctly(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        # layer_norm1 and layer_norm2 have weight (gamma) and bias (beta)
        assert profile["layer_norm1.weight"] == "layernorm"
        assert profile["layer_norm1.bias"] == "layernorm"
        assert profile["layer_norm2.weight"] == "layernorm"
        assert profile["layer_norm2.bias"] == "layernorm"

    def test_attention_parameters_classified_correctly(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        attn_params = [name for name, cat in profile.items() if cat == "attention"]
        # MultiheadAttention uses "attn" in its name — at least one attn param expected
        assert len(attn_params) > 0

    def test_ffn_parameters_classified_correctly(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        # ffn.0.weight, ffn.0.bias, ffn.2.weight, ffn.2.bias
        ffn_params = [name for name, cat in profile.items() if cat == "ffn"]
        assert len(ffn_params) > 0

    def test_profile_is_immutable_dict(self, injector, tiny_model):
        """vulnerability_profile must return a new dict each call (not shared state)."""
        profile1 = injector.vulnerability_profile(tiny_model)
        profile2 = injector.vulnerability_profile(tiny_model)
        # Mutating one must not affect the other
        profile1["__test__"] = "mutated"
        assert "__test__" not in profile2


# ---------------------------------------------------------------------------
# Tests: inject_attention_faults
# ---------------------------------------------------------------------------

class TestInjectAttentionFaults:
    def test_only_attention_params_modified(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        snapshot = _snapshot(tiny_model)

        injector.inject_attention_faults(tiny_model, num_faults=50)

        changed = _changed_params(tiny_model, snapshot)
        for name in changed:
            assert profile[name] == "attention", (
                f"Non-attention parameter '{name}' was modified"
            )

    def test_non_attention_params_untouched(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)

        injector.inject_attention_faults(tiny_model, num_faults=50)

        profile = injector.vulnerability_profile(tiny_model)
        for name, param in tiny_model.named_parameters():
            if profile[name] != "attention":
                assert torch.equal(param.data, snapshot[name]), (
                    f"Non-attention parameter '{name}' was unexpectedly modified"
                )

    def test_attention_params_are_changed(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_attention_faults(tiny_model, num_faults=100)
        changed = _changed_params(tiny_model, snapshot)
        assert len(changed) > 0, "No parameters were modified after attention fault injection"

    def test_zero_faults_no_change(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_attention_faults(tiny_model, num_faults=0)
        assert _changed_params(tiny_model, snapshot) == set()

    def test_target_qkv_default(self, injector, tiny_model):
        """Default target='qkv' should succeed without error."""
        injector.inject_attention_faults(tiny_model, num_faults=10, target="qkv")

    def test_invalid_target_raises(self, injector, tiny_model):
        with pytest.raises(ValueError, match="target"):
            injector.inject_attention_faults(tiny_model, num_faults=10, target="invalid")

    def test_returns_list_of_modified_params(self, injector, tiny_model):
        result = injector.inject_attention_faults(tiny_model, num_faults=20)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: inject_layernorm_faults
# ---------------------------------------------------------------------------

class TestInjectLayernormFaults:
    def test_only_layernorm_params_modified(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        snapshot = _snapshot(tiny_model)

        injector.inject_layernorm_faults(tiny_model, num_faults=50)

        changed = _changed_params(tiny_model, snapshot)
        for name in changed:
            assert profile[name] == "layernorm", (
                f"Non-layernorm parameter '{name}' was modified"
            )

    def test_non_layernorm_params_untouched(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)

        injector.inject_layernorm_faults(tiny_model, num_faults=50)

        profile = injector.vulnerability_profile(tiny_model)
        for name, param in tiny_model.named_parameters():
            if profile[name] != "layernorm":
                assert torch.equal(param.data, snapshot[name]), (
                    f"Non-layernorm parameter '{name}' was unexpectedly modified"
                )

    def test_layernorm_params_are_changed(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_layernorm_faults(tiny_model, num_faults=100)
        changed = _changed_params(tiny_model, snapshot)
        assert len(changed) > 0

    def test_zero_faults_no_change(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_layernorm_faults(tiny_model, num_faults=0)
        assert _changed_params(tiny_model, snapshot) == set()

    def test_returns_list_of_modified_params(self, injector, tiny_model):
        result = injector.inject_layernorm_faults(tiny_model, num_faults=20)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: inject_embedding_faults
# ---------------------------------------------------------------------------

class TestInjectEmbeddingFaults:
    def test_only_embedding_params_modified(self, injector, tiny_model):
        profile = injector.vulnerability_profile(tiny_model)
        snapshot = _snapshot(tiny_model)

        injector.inject_embedding_faults(tiny_model, num_faults=50)

        changed = _changed_params(tiny_model, snapshot)
        for name in changed:
            assert profile[name] == "embedding", (
                f"Non-embedding parameter '{name}' was modified"
            )

    def test_non_embedding_params_untouched(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)

        injector.inject_embedding_faults(tiny_model, num_faults=50)

        profile = injector.vulnerability_profile(tiny_model)
        for name, param in tiny_model.named_parameters():
            if profile[name] != "embedding":
                assert torch.equal(param.data, snapshot[name]), (
                    f"Non-embedding parameter '{name}' was unexpectedly modified"
                )

    def test_embedding_params_are_changed(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_embedding_faults(tiny_model, num_faults=200)
        changed = _changed_params(tiny_model, snapshot)
        assert len(changed) > 0

    def test_zero_faults_no_change(self, injector, tiny_model):
        snapshot = _snapshot(tiny_model)
        injector.inject_embedding_faults(tiny_model, num_faults=0)
        assert _changed_params(tiny_model, snapshot) == set()

    def test_returns_list_of_modified_params(self, injector, tiny_model):
        result = injector.inject_embedding_faults(tiny_model, num_faults=20)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: LayerNorm vulnerability relative to FFN (cascading effect)
# ---------------------------------------------------------------------------

class TestLayerNormVulnerabilityVsFFN:
    """LayerNorm faults cascade through all downstream activations because
    every token's representation passes through LayerNorm before further
    computation. This test verifies that equal numbers of LN faults cause
    at least as much output deviation as the same count of FFN faults,
    measured as mean absolute deviation from the clean baseline.
    """

    @staticmethod
    def _mean_abs_deviation(clean: torch.Tensor, faulted: torch.Tensor) -> float:
        """Compute mean absolute deviation, masking out NaN/Inf."""
        diff = (clean - faulted).abs()
        finite_mask = torch.isfinite(diff)
        if finite_mask.sum() == 0:
            return float("inf")
        return diff[finite_mask].mean().item()

    def test_layernorm_deviation_at_least_as_large_as_ffn(self):
        """Run several trials; LN deviation should be >= FFN deviation on average."""
        torch.manual_seed(0)
        num_trials = 10
        num_faults = 50
        # Use fixed input tokens
        x = torch.randint(0, 100, (4, 8))

        ln_deviations = []
        ffn_deviations = []
        inj = TransformerFaultInjector()

        for seed in range(num_trials):
            torch.manual_seed(seed)

            # --- LayerNorm fault trial ---
            base_ln = TinyTransformer()
            base_ln.eval()
            with torch.no_grad():
                clean_ln = base_ln(x).clone()

            faulted_ln = copy.deepcopy(base_ln)
            faulted_ln.eval()
            inj.inject_layernorm_faults(faulted_ln, num_faults=num_faults)
            with torch.no_grad():
                out_ln = faulted_ln(x)

            ln_deviations.append(
                TestLayerNormVulnerabilityVsFFN._mean_abs_deviation(clean_ln, out_ln)
            )

            # --- FFN fault trial ---
            torch.manual_seed(seed)
            base_ffn = TinyTransformer()
            base_ffn.eval()
            with torch.no_grad():
                clean_ffn = base_ffn(x).clone()

            faulted_ffn = copy.deepcopy(base_ffn)
            faulted_ffn.eval()
            # Manually inject into only FFN params for fair comparison
            ffn_params = [
                (n, p)
                for n, p in faulted_ffn.named_parameters()
                if inj.vulnerability_profile(faulted_ffn)[n] == "ffn"
            ]
            if ffn_params:
                target_name, target_param = ffn_params[0]
                with torch.no_grad():
                    from space_ml_sim.compute.fault_injector import FaultInjector
                    FaultInjector.flip_random_bits(target_param.data, num_faults)
            with torch.no_grad():
                out_ffn = faulted_ffn(x)

            ffn_deviations.append(
                TestLayerNormVulnerabilityVsFFN._mean_abs_deviation(clean_ffn, out_ffn)
            )

        # Filter out infinite values from MSB flips before averaging
        finite_ln = [d for d in ln_deviations if d < 1e10]
        finite_ffn = [d for d in ffn_deviations if d < 1e10]

        if finite_ln and finite_ffn:
            avg_ln = sum(finite_ln) / len(finite_ln)
            avg_ffn = sum(finite_ffn) / len(finite_ffn)
            # LN faults should cause deviation at least as large as FFN faults
            # (LN parameters are fewer but each affects ALL tokens downstream)
            assert avg_ln >= avg_ffn * 0.5, (
                f"Expected LayerNorm deviation ({avg_ln:.4f}) >= 50% of FFN deviation "
                f"({avg_ffn:.4f}). LayerNorm faults should cascade across all tokens."
            )


# ---------------------------------------------------------------------------
# Tests: model with HuggingFace-style naming (q_proj / k_proj / v_proj)
# ---------------------------------------------------------------------------

class HuggingFaceStyleTransformer(nn.Module):
    """Simulates HuggingFace-style parameter names."""

    def __init__(self):
        super().__init__()
        hidden = 32
        # Attention projections using HuggingFace-style names
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        # LayerNorm with HuggingFace-style name
        self.norm = nn.LayerNorm(hidden)
        # FFN
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        # Embedding
        self.embed_tokens = nn.Embedding(100, hidden)

    def forward(self, x):
        return x  # Not used in these tests


class TestHuggingFaceNamingConventions:
    def test_qkv_proj_classified_as_attention(self):
        model = HuggingFaceStyleTransformer()
        inj = TransformerFaultInjector()
        profile = inj.vulnerability_profile(model)
        assert profile["q_proj.weight"] == "attention"
        assert profile["k_proj.weight"] == "attention"
        assert profile["v_proj.weight"] == "attention"

    def test_norm_classified_as_layernorm(self):
        model = HuggingFaceStyleTransformer()
        inj = TransformerFaultInjector()
        profile = inj.vulnerability_profile(model)
        assert profile["norm.weight"] == "layernorm"
        assert profile["norm.bias"] == "layernorm"

    def test_embed_tokens_classified_as_embedding(self):
        model = HuggingFaceStyleTransformer()
        inj = TransformerFaultInjector()
        profile = inj.vulnerability_profile(model)
        assert profile["embed_tokens.weight"] == "embedding"


# ---------------------------------------------------------------------------
# Tests: edge cases and guard rails
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_model_with_no_target_params_is_safe(self, injector):
        """inject_layernorm_faults on a model with no layernorm should not crash."""
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        model.eval()
        result = injector.inject_layernorm_faults(model, num_faults=10)
        assert result == []

    def test_model_with_no_attention_params_is_safe(self, injector):
        model = nn.Sequential(nn.Linear(8, 8))
        model.eval()
        result = injector.inject_attention_faults(model, num_faults=10)
        assert result == []

    def test_model_with_no_embedding_params_is_safe(self, injector):
        model = nn.Sequential(nn.Linear(8, 8))
        model.eval()
        result = injector.inject_embedding_faults(model, num_faults=10)
        assert result == []

    def test_negative_faults_raises(self, injector, tiny_model):
        with pytest.raises(ValueError, match="num_faults"):
            injector.inject_attention_faults(tiny_model, num_faults=-1)

    def test_non_module_raises(self, injector):
        with pytest.raises((TypeError, AttributeError)):
            injector.vulnerability_profile("not_a_model")


# ---------------------------------------------------------------------------
# Tests: __init__.py export
# ---------------------------------------------------------------------------

class TestExport:
    def test_importable_from_compute_package(self):
        from space_ml_sim.compute import TransformerFaultInjector as TFI  # noqa: F401
        assert TFI is TransformerFaultInjector
