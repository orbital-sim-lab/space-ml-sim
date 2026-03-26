"""Simplified thermal model for satellite compute payloads.

Uses steady-state energy balance: T = T_ambient + P_compute / radiator_conductance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThermalModel(BaseModel):
    """Simplified steady-state thermal model.

    Ambient temperature depends on eclipse state:
        - Direct sun: +80 C (illuminated side thermal load)
        - Eclipse: -40 C (deep space cooling)

    Radiator is sized for 100 m^2 with effective conductance.
    """

    radiator_area_m2: float = Field(default=100.0, gt=0)
    radiator_conductance_w_per_c: float = Field(
        default=50.0, gt=0, description="Effective thermal conductance W/C"
    )
    ambient_sun_c: float = Field(default=80.0)
    ambient_eclipse_c: float = Field(default=-40.0)

    def compute_temperature(
        self, compute_power_watts: float, in_eclipse: bool
    ) -> float:
        """Compute steady-state temperature in Celsius.

        Args:
            compute_power_watts: Heat dissipated by compute payload.
            in_eclipse: Whether satellite is in Earth's shadow.

        Returns:
            Equilibrium temperature in Celsius.
        """
        t_ambient = self.ambient_eclipse_c if in_eclipse else self.ambient_sun_c
        return t_ambient + compute_power_watts / self.radiator_conductance_w_per_c
