"""Simplified power model for satellite compute payloads.

AI Sat Mini reference: ~100 kW solar when sunlit, ~10 kW battery in eclipse.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PowerModel(BaseModel):
    """Satellite power availability model."""

    solar_power_watts: float = Field(default=100_000.0, gt=0, description="Solar array output in watts")
    battery_power_watts: float = Field(default=10_000.0, gt=0, description="Battery output in watts (eclipse)")

    def available_power(self, in_eclipse: bool) -> float:
        """Return available power in watts based on eclipse state.

        Args:
            in_eclipse: Whether the satellite is in Earth's shadow.

        Returns:
            Available power in watts.
        """
        return self.battery_power_watts if in_eclipse else self.solar_power_watts
