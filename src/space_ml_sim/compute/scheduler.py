"""Inference scheduling across a constellation.

Decides which satellites run inference based on power, thermal, and
fault state constraints.
"""

from __future__ import annotations

from typing import Any

from space_ml_sim.core.satellite import Satellite, SatelliteState


class InferenceScheduler:
    """Schedule inference tasks across satellite constellation members.

    Simple priority-based scheduler: prefers satellites that are
    nominal, sunlit (more power), and cooler.
    """

    def __init__(self, power_margin_fraction: float = 0.1) -> None:
        """Initialize scheduler.

        Args:
            power_margin_fraction: Fraction of TDP to keep as power margin.
        """
        self.power_margin_fraction = power_margin_fraction

    def select_nodes(
        self,
        satellites: list[Satellite],
        num_needed: int,
    ) -> list[Satellite]:
        """Select the best satellites for running inference.

        Filters by operational state and power availability, then
        ranks by temperature (cooler is better for reliability).

        Args:
            satellites: Available satellites.
            num_needed: Number of compute nodes needed.

        Returns:
            List of selected satellites, up to num_needed.
        """
        candidates = [
            sat
            for sat in satellites
            if sat.is_operational
            and sat.power_available_w
            >= sat.chip_profile.tdp_watts * (1 + self.power_margin_fraction)
        ]

        # Sort by: nominal before degraded, then by temperature ascending
        ranked = sorted(
            candidates,
            key=lambda s: (s.state != SatelliteState.NOMINAL, s.temperature_c),
        )

        return ranked[:num_needed]

    def schedule_summary(self, satellites: list[Satellite]) -> dict[str, Any]:
        """Summarize scheduling availability across the constellation.

        Args:
            satellites: All satellites in constellation.

        Returns:
            Dict with availability metrics.
        """
        operational = [s for s in satellites if s.is_operational]
        powered = [
            s
            for s in operational
            if s.power_available_w >= s.chip_profile.tdp_watts
        ]
        within_thermal = [
            s for s in powered if s.temperature_c <= s.chip_profile.max_temp_c
        ]

        return {
            "total_satellites": len(satellites),
            "operational": len(operational),
            "power_available": len(powered),
            "within_thermal_limits": len(within_thermal),
            "ready_for_inference": len(within_thermal),
        }
