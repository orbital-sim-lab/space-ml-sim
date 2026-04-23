"""Automated TMR recommendation engine with cost/benefit analysis.

Analyzes per-layer vulnerability data from sensitivity analysis and
recommends an optimal selective TMR configuration that maximizes
accuracy recovery within a compute budget constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class TMRRecommendation:
    """Result of TMR optimization analysis."""

    protected_layers: frozenset[str]
    unprotected_layers: frozenset[str]
    ranked_layers: tuple[str, ...]
    compute_multiplier: float
    expected_accuracy_recovery: float
    residual_risk: float
    cost_benefit_ratio: float
    layer_details: tuple[LayerDetail, ...]


@dataclass(frozen=True)
class LayerDetail:
    """Per-layer recommendation detail."""

    name: str
    vulnerability: float
    param_count: int
    param_fraction: float
    protected: bool


def recommend_tmr(
    model: nn.Module,
    sensitivity: dict[str, float],
    max_compute_multiplier: float = 2.0,
    min_vulnerability_threshold: float = 0.0,
) -> TMRRecommendation:
    """Recommend optimal selective TMR configuration.

    Greedily selects layers to protect in descending vulnerability order
    until the compute budget is exhausted.

    Args:
        model: The model to analyze.
        sensitivity: Map of parameter name -> accuracy drop when faulted.
            Typically from TMRWrapper.sensitivity_analysis().
        max_compute_multiplier: Maximum allowed compute overhead (1.0 = no TMR,
            3.0 = full TMR). Selective TMR falls between these.
        min_vulnerability_threshold: Minimum accuracy drop to consider a layer
            worth protecting. Layers below this are treated as negligible.

    Returns:
        TMRRecommendation with optimal configuration and cost/benefit metrics.
    """
    # Gather parameter info
    param_sizes: dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sizes[name] = param.numel()

    total_params = sum(param_sizes.values())

    # Filter and rank layers by vulnerability (descending)
    ranked = sorted(
        ((name, drop) for name, drop in sensitivity.items() if name in param_sizes),
        key=lambda x: x[1],
        reverse=True,
    )
    ranked_names = tuple(name for name, _ in ranked)

    # Greedy selection within budget
    # Compute multiplier: 1.0 + 2.0 * (fraction of params protected by TMR)
    # Because TMR triplicates the protected params: base cost + 2x protected fraction
    max_protected_fraction = (max_compute_multiplier - 1.0) / 2.0
    max_protected_fraction = max(0.0, min(1.0, max_protected_fraction))

    protected: set[str] = set()
    protected_params = 0

    for name, drop in ranked:
        if drop < min_vulnerability_threshold:
            continue
        layer_params = param_sizes.get(name, 0)
        new_fraction = (protected_params + layer_params) / total_params
        if new_fraction <= max_protected_fraction:
            protected.add(name)
            protected_params += layer_params

    # Compute metrics
    protected_fraction = protected_params / total_params if total_params > 0 else 0.0
    compute_multiplier = 1.0 + 2.0 * protected_fraction

    expected_recovery = sum(
        sensitivity.get(name, 0.0) for name in protected
    )

    unprotected_vulnerable = {
        name for name, drop in sensitivity.items()
        if name not in protected and drop >= min_vulnerability_threshold and name in param_sizes
    }
    residual_risk = sum(
        sensitivity.get(name, 0.0) for name in unprotected_vulnerable
    )

    cost_benefit_ratio = (
        expected_recovery / (compute_multiplier - 1.0)
        if compute_multiplier > 1.0
        else 0.0
    )

    # Build layer details
    details = []
    for name, drop in ranked:
        details.append(LayerDetail(
            name=name,
            vulnerability=drop,
            param_count=param_sizes.get(name, 0),
            param_fraction=param_sizes.get(name, 0) / total_params if total_params > 0 else 0.0,
            protected=name in protected,
        ))

    return TMRRecommendation(
        protected_layers=frozenset(protected),
        unprotected_layers=frozenset(unprotected_vulnerable),
        ranked_layers=ranked_names,
        compute_multiplier=compute_multiplier,
        expected_accuracy_recovery=expected_recovery,
        residual_risk=residual_risk,
        cost_benefit_ratio=cost_benefit_ratio,
        layer_details=tuple(details),
    )
