"""Satellite model with state tracking for orbital AI compute simulation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.environment.thermal import ThermalModel
from space_ml_sim.environment.power import PowerModel
from space_ml_sim.models.chip_profiles import ChipProfile


class SatelliteState(str, Enum):
    """Operational state of a satellite."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    FAILED = "failed"


class Satellite(BaseModel):
    """A satellite with compute payload, tracking orbital and environmental state.

    Immutable-style updates: methods return new Satellite instances.
    """

    id: str
    orbit_config: OrbitConfig
    chip_profile: ChipProfile
    state: SatelliteState = SatelliteState.NOMINAL
    temperature_c: float = 25.0
    power_available_w: float = 0.0
    total_seu_events: int = 0
    tid_accumulated_krad: float = 0.0
    in_eclipse: bool = False
    position_km: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def with_power_update(self, in_eclipse: bool) -> "Satellite":
        """Return a new Satellite with updated power state.

        Args:
            in_eclipse: Whether the satellite is currently in eclipse.

        Returns:
            New Satellite with updated power_available_w.
        """
        power_model = PowerModel()
        return self.model_copy(
            update={"power_available_w": power_model.available_power(in_eclipse)}
        )

    def with_thermal_update(
        self, compute_load_fraction: float, in_eclipse: bool
    ) -> "Satellite":
        """Return a new Satellite with updated thermal state.

        Args:
            compute_load_fraction: Fraction of TDP being used (0.0 to 1.0).
            in_eclipse: Whether the satellite is in eclipse.

        Returns:
            New Satellite with updated temperature_c.
        """
        thermal = ThermalModel()
        compute_power = self.chip_profile.tdp_watts * compute_load_fraction
        temp = thermal.compute_temperature(compute_power, in_eclipse)
        return self.model_copy(update={"temperature_c": temp})

    def with_radiation_tick(
        self, rad_env: RadiationEnvironment, dt_seconds: float
    ) -> "Satellite":
        """Return a new Satellite after applying radiation effects for dt_seconds.

        Samples SEU events from Poisson distribution and accumulates TID.
        Updates state to DEGRADED or FAILED based on TID thresholds.

        Args:
            rad_env: Radiation environment model.
            dt_seconds: Time step in seconds.

        Returns:
            New Satellite with updated radiation state.
        """
        if self.state == SatelliteState.FAILED:
            return self

        seu_count = rad_env.sample_seu_events(
            chip_cross_section_cm2=self.chip_profile.seu_cross_section_cm2,
            num_bits=self.chip_profile.memory_bits,
            dt_seconds=dt_seconds,
        )
        tid_increment = rad_env.tid_dose(dt_seconds)

        new_tid = self.tid_accumulated_krad + tid_increment
        new_seus = self.total_seu_events + seu_count

        # State transitions based on TID
        tolerance = self.chip_profile.tid_tolerance_krad
        if new_tid > tolerance:
            new_state = SatelliteState.FAILED
        elif new_tid > 0.5 * tolerance:
            new_state = SatelliteState.DEGRADED
        else:
            new_state = self.state

        return self.model_copy(
            update={
                "total_seu_events": new_seus,
                "tid_accumulated_krad": new_tid,
                "state": new_state,
            }
        )

    def with_position(
        self, position_km: tuple[float, float, float], in_eclipse: bool
    ) -> "Satellite":
        """Return a new Satellite with updated position and eclipse state."""
        return self.model_copy(
            update={"position_km": position_km, "in_eclipse": in_eclipse}
        )

    @property
    def is_operational(self) -> bool:
        """Whether the satellite can still perform compute."""
        return self.state != SatelliteState.FAILED
