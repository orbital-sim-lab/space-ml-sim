"""Transformer-aware fault injection patterns.

Extends the radiation fault injection concept to target transformer-specific
components:

    - Attention weight matrices (Q, K, V projections)
    - LayerNorm parameters (gamma/beta — extremely vulnerable due to cascading)
    - Embedding layers (token + position embeddings)

Key insight: LayerNorm faults cascade through all token representations because
every downstream computation depends on the normalised output.  A single bit
flip in gamma or beta propagates to every position in the sequence, making
these parameters far more fragile than their small parameter count suggests.

Naming conventions supported:
    - Standard nn.Transformer: "attn", "in_proj", "out_proj"
    - HuggingFace-style: "q_proj", "k_proj", "v_proj", "out_proj"
    - LayerNorm: "layer_norm", "ln", "norm"
    - Embedding: "embed", "pos_embed", "wpe", "wte"
    - FFN: "ffn", "mlp", "fc", "linear" (when not already matched)
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from space_ml_sim.compute.fault_injector import FaultInjector


# ---------------------------------------------------------------------------
# Naming-convention lookup tables
# ---------------------------------------------------------------------------

_ATTENTION_KEYWORDS: Final[tuple[str, ...]] = (
    # standard nn.MultiheadAttention internal names
    "in_proj",
    # HuggingFace-style projection names
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    # Generic attention markers
    "attn",
    "attention",
    "self_attn",
    "cross_attn",
)

_LAYERNORM_KEYWORDS: Final[tuple[str, ...]] = (
    "layer_norm",
    "layernorm",
    "ln_",
    "ln1",
    "ln2",
    # HuggingFace / many GPT variants use bare "norm"
    ".norm",
    "norm.",
    "norm1",
    "norm2",
    # RMSNorm variants
    "rmsnorm",
    "rms_norm",
)

_EMBEDDING_KEYWORDS: Final[tuple[str, ...]] = (
    "embed",
    "embedding",
    "wte",
    "wpe",
    "pos_embed",
    "token_embed",
)

_FFN_KEYWORDS: Final[tuple[str, ...]] = (
    "ffn",
    "mlp",
    "feed_forward",
    "feedforward",
    "fc1",
    "fc2",
    "fc_in",
    "fc_out",
    "dense",
    "intermediate",
    # Note: bare "fc" was removed — it matched too broadly (e.g., "interface").
    # Use specific fc1/fc2/fc_in/fc_out patterns instead.
)

# Subset of attention keywords that specifically identify Q/K/V (not output proj)
_QKV_KEYWORDS: Final[tuple[str, ...]] = (
    "in_proj",  # nn.MultiheadAttention combines Q/K/V
    "q_proj",
    "k_proj",
    "v_proj",
)

_VALID_TARGETS: Final[frozenset[str]] = frozenset({"qkv", "all"})


# ---------------------------------------------------------------------------
# Helper: classify a single parameter name
# ---------------------------------------------------------------------------


def _classify_param(name: str) -> str:
    """Return the vulnerability category for a parameter name.

    Args:
        name: Dot-separated parameter name, e.g. ``"layer_norm1.weight"``.

    Returns:
        One of ``"attention"``, ``"layernorm"``, ``"embedding"``, ``"ffn"``,
        or ``"other"``.
    """
    lower = name.lower()

    # Layernorm check before attention because some models have
    # "attn.ln" components which should be tagged as layernorm.
    for kw in _LAYERNORM_KEYWORDS:
        if kw in lower:
            return "layernorm"

    for kw in _EMBEDDING_KEYWORDS:
        if kw in lower:
            return "embedding"

    for kw in _ATTENTION_KEYWORDS:
        if kw in lower:
            return "attention"

    for kw in _FFN_KEYWORDS:
        if kw in lower:
            return "ffn"

    return "other"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TransformerFaultInjector:
    """Inject radiation-modeled faults targeting transformer-specific components.

    Does not require a ``RadiationEnvironment`` or ``ChipProfile``; it reuses
    ``FaultInjector.flip_random_bits`` for the actual bit-level manipulation
    and adds transformer-aware layer selection on top.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vulnerability_profile(self, model: nn.Module) -> dict[str, str]:
        """Classify every parameter of *model* by its vulnerability category.

        Args:
            model: Any ``nn.Module`` — works best with transformer models
                following standard or HuggingFace naming conventions.

        Returns:
            A new dict mapping each parameter name to one of:
            ``"attention"``, ``"layernorm"``, ``"embedding"``, ``"ffn"``,
            or ``"other"``.

        Raises:
            AttributeError: If *model* does not expose ``named_parameters()``.
        """
        return {name: _classify_param(name) for name, _ in model.named_parameters()}

    def inject_attention_faults(
        self,
        model: nn.Module,
        num_faults: int,
        target: str = "qkv",
    ) -> list[str]:
        """Inject bit flips into attention weight matrices.

        Targets Q/K/V projection parameters (``target="qkv"``) or all
        parameters classified as attention (``target="all"``).

        Args:
            model: PyTorch transformer model (weights modified in-place).
            num_faults: Number of bit flips to distribute across target params.
            target: ``"qkv"`` (default) — restrict to Q/K/V projections only;
                ``"all"`` — include output projection and other attn weights.

        Returns:
            List of parameter names that were modified.

        Raises:
            ValueError: If *num_faults* < 0 or *target* is not recognised.
        """
        self._validate_num_faults(num_faults)
        if target not in _VALID_TARGETS:
            raise ValueError(f"Invalid target '{target}'. Must be one of {sorted(_VALID_TARGETS)}.")

        if target == "qkv":
            params = self._select_params(model, category="attention", extra_filter=_QKV_KEYWORDS)
        else:
            params = self._select_params(model, category="attention")

        return self._distribute_faults(params, num_faults)

    def inject_layernorm_faults(
        self,
        model: nn.Module,
        num_faults: int,
    ) -> list[str]:
        """Inject bit flips into LayerNorm (gamma / beta) parameters.

        LayerNorm is extremely vulnerable: every downstream activation for
        every token passes through the normalised representation.  A single
        MSB flip in gamma or beta cascades across the entire sequence.

        Args:
            model: PyTorch transformer model (weights modified in-place).
            num_faults: Number of bit flips to distribute across LN params.

        Returns:
            List of parameter names that were modified.

        Raises:
            ValueError: If *num_faults* < 0.
        """
        self._validate_num_faults(num_faults)
        params = self._select_params(model, category="layernorm")
        return self._distribute_faults(params, num_faults)

    def inject_embedding_faults(
        self,
        model: nn.Module,
        num_faults: int,
    ) -> list[str]:
        """Inject bit flips into embedding layers (token + position).

        Args:
            model: PyTorch transformer model (weights modified in-place).
            num_faults: Number of bit flips to distribute across embedding params.

        Returns:
            List of parameter names that were modified.

        Raises:
            ValueError: If *num_faults* < 0.
        """
        self._validate_num_faults(num_faults)
        params = self._select_params(model, category="embedding")
        return self._distribute_faults(params, num_faults)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_num_faults(num_faults: int) -> None:
        if num_faults < 0:
            raise ValueError(f"num_faults must be >= 0, got {num_faults}.")

    def _select_params(
        self,
        model: nn.Module,
        category: str,
        extra_filter: tuple[str, ...] | None = None,
    ) -> list[tuple[str, nn.Parameter]]:
        """Return parameters matching *category* (and optionally *extra_filter*).

        Args:
            model: The model to inspect.
            category: One of the five vulnerability categories.
            extra_filter: If given, only include parameters whose lowercased
                name contains at least one keyword from this tuple.

        Returns:
            List of (name, parameter) pairs; returns a new list each call.
        """
        profile = self.vulnerability_profile(model)
        selected = []

        for name, param in model.named_parameters():
            if profile[name] != category:
                continue
            if extra_filter is not None:
                lower = name.lower()
                if not any(kw in lower for kw in extra_filter):
                    continue
            selected.append((name, param))

        return selected

    @staticmethod
    def _distribute_faults(
        params: list[tuple[str, nn.Parameter]],
        num_faults: int,
    ) -> list[str]:
        """Distribute *num_faults* bit flips across *params* proportionally.

        Faults are allocated proportionally to each parameter's element count.
        Allocation uses ``FaultInjector.flip_random_bits`` so the actual bit
        manipulation is consistent with the rest of the fault injection engine.

        Args:
            params: List of (name, parameter) pairs to target.
            num_faults: Total bit flips to inject.

        Returns:
            List of parameter names that received at least one fault.
            Returns an empty list if *params* is empty or *num_faults* == 0.
        """
        if not params or num_faults == 0:
            return []

        total_elements = sum(p.numel() for _, p in params)
        if total_elements == 0:
            return []

        modified: list[str] = []
        faults_remaining = num_faults

        for name, param in params:
            if faults_remaining <= 0:
                break

            # Proportional allocation; at least 1 fault for the first param
            # when total remaining allows it.
            share = int(num_faults * param.numel() / total_elements)
            layer_faults = max(1, share)
            layer_faults = min(layer_faults, faults_remaining)

            with torch.no_grad():
                FaultInjector.flip_random_bits(param.data, layer_faults)

            faults_remaining -= layer_faults
            modified.append(name)

        return modified
