"""Performance metrics for inference throughput and latency."""

from __future__ import annotations

from dataclasses import dataclass

from space_ml_sim.core.satellite import Satellite
from space_ml_sim.models.chip_profiles import ChipProfile


@dataclass(frozen=True)
class PerformanceMetrics:
    """Constellation compute performance snapshot."""

    total_tops: float
    active_tops: float
    total_power_watts: float
    active_power_watts: float
    tops_per_watt: float

    @classmethod
    def from_satellites(cls, satellites: list[Satellite]) -> "PerformanceMetrics":
        """Compute performance metrics from satellite list.

        Args:
            satellites: Satellites to analyze.

        Returns:
            PerformanceMetrics snapshot.
        """
        active = [s for s in satellites if s.is_operational]
        total_tops = sum(s.chip_profile.compute_tops for s in satellites)
        active_tops = sum(s.chip_profile.compute_tops for s in active)
        total_power = sum(s.chip_profile.tdp_watts for s in satellites)
        active_power = sum(s.chip_profile.tdp_watts for s in active)
        tops_per_watt = active_tops / active_power if active_power > 0 else 0.0

        return cls(
            total_tops=total_tops,
            active_tops=active_tops,
            total_power_watts=total_power,
            active_power_watts=active_power,
            tops_per_watt=tops_per_watt,
        )
