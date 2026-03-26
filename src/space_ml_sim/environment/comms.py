"""Inter-satellite link (ISL) communications model.

Placeholder for v0.1 — models basic line-of-sight distance calculations.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class CommsModel(BaseModel):
    """Inter-satellite link communications model."""

    max_isl_range_km: float = Field(default=5000.0, gt=0, description="Maximum ISL range in km")
    data_rate_gbps: float = Field(default=10.0, gt=0, description="ISL data rate in Gbps")
    latency_overhead_ms: float = Field(default=1.0, ge=0, description="Processing latency overhead in ms")

    def link_latency_ms(self, distance_km: float) -> float:
        """Compute one-way link latency in milliseconds.

        Args:
            distance_km: Distance between satellites in km.

        Returns:
            Total latency (propagation + overhead) in ms, or inf if out of range.
        """
        if distance_km > self.max_isl_range_km:
            return math.inf
        propagation_ms = (distance_km / 299_792.458) * 1000  # speed of light
        return propagation_ms + self.latency_overhead_ms

    @staticmethod
    def distance_km(
        pos_a: tuple[float, float, float], pos_b: tuple[float, float, float]
    ) -> float:
        """Euclidean distance between two ECI positions in km."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))
