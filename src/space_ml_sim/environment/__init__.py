"""Space environment models: radiation, thermal, power, communications, and ISL networking."""

from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.environment.thermal import ThermalModel
from space_ml_sim.environment.power import PowerModel
from space_ml_sim.environment.comms import CommsModel
from space_ml_sim.environment.timeline import (
    RadiationTimeline,
    radiation_timeline,
    plot_radiation_timeline,
)
from space_ml_sim.environment.ground_station import (
    GroundStation,
    ContactWindow,
    find_contact_windows,
    GROUND_STATION_PRESETS,
)
from space_ml_sim.environment.isl_network import ISLNetwork
from space_ml_sim.environment.radiation_uncertainty import (
    UncertaintyBand,
    seu_rate_with_uncertainty,
    tid_rate_with_uncertainty,
    mission_tid_with_uncertainty,
    mission_seus_with_uncertainty,
)
from space_ml_sim.environment.solar_cycle import (
    apply_solar_cycle,
    SOLAR_PHASES,
)
from space_ml_sim.environment.thermal_cycling import (
    ThermalProfile,
    generate_thermal_profile,
    derate_at_temperature,
)
from space_ml_sim.environment.sel_model import (
    sel_rate_per_day,
    mission_sel_probability,
    sel_mitigation_requirements,
)

__all__ = [
    "RadiationEnvironment",
    "ThermalModel",
    "PowerModel",
    "CommsModel",
    "RadiationTimeline",
    "radiation_timeline",
    "plot_radiation_timeline",
    "GroundStation",
    "ContactWindow",
    "find_contact_windows",
    "GROUND_STATION_PRESETS",
    "ISLNetwork",
    "UncertaintyBand",
    "seu_rate_with_uncertainty",
    "tid_rate_with_uncertainty",
    "mission_tid_with_uncertainty",
    "mission_seus_with_uncertainty",
    "apply_solar_cycle",
    "SOLAR_PHASES",
    "ThermalProfile",
    "generate_thermal_profile",
    "derate_at_temperature",
    "sel_rate_per_day",
    "mission_sel_probability",
    "sel_mitigation_requirements",
]

from space_ml_sim.environment.dose_depth import (
    DoseDepthCurve,
    generate_dose_depth_curve,
    find_shielding_for_dose,
)

__all__ += [
    "DoseDepthCurve",
    "generate_dose_depth_curve",
    "find_shielding_for_dose",
]
