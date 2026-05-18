# space-ml-sim: Simulate AI inference on orbital satellite constellations
# under realistic space radiation.
#
# Copyright 2026 space-ml-sim contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing is available. See COMMERCIAL_LICENSE.md for details.

"""space-ml-sim: simulate AI inference on orbital satellite constellations under realistic space radiation.

The public API is the set of names re-exported here. Symbols are loaded lazily
on first access (PEP 562) so that `import space_ml_sim` does not eagerly pull
in torch, numpy, or any heavy submodule. The 1.0 stability contract applies
to every name in :data:`__all__`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0rc1"

# Map of public name -> "submodule_path:attribute". Lazy-resolved by __getattr__.
_LAZY_IMPORTS: dict[str, str] = {
    # Core orbital mechanics
    "OrbitConfig": "core:OrbitConfig",
    "Satellite": "core:Satellite",
    "SatelliteState": "core:SatelliteState",
    "Constellation": "core:Constellation",
    "SimClock": "core:SimClock",
    "propagate": "core:propagate",
    "position_at": "core:position_at",
    "walker_delta_orbits": "core:walker_delta_orbits",
    "sun_synchronous_orbits": "core:sun_synchronous_orbits",
    "is_in_eclipse": "core:is_in_eclipse",
    "parse_tle": "core:parse_tle",
    "load_tle_file": "core:load_tle_file",
    "propagate_sgp4": "core:propagate_sgp4",
    "ConstellationPreset": "core:ConstellationPreset",
    "CONSTELLATION_PRESETS": "core:CONSTELLATION_PRESETS",
    "generate_from_preset": "core:generate_from_preset",
    # Radiation environments
    "RadiationEnvironment": "environment:RadiationEnvironment",
    "HeliocentricEnvironment": "environment:HeliocentricEnvironment",
    "SPEStatisticalModel": "environment:SPEStatisticalModel",
    "SolarParticleEvent": "environment:SolarParticleEvent",
    "mission_spe_dose": "environment:mission_spe_dose",
    "PowerModel": "environment:PowerModel",
    "ThermalModel": "environment:ThermalModel",
    "RadiationTimeline": "environment:RadiationTimeline",
    "radiation_timeline": "environment:radiation_timeline",
    "GroundStation": "environment:GroundStation",
    "ContactWindow": "environment:ContactWindow",
    "find_contact_windows": "environment:find_contact_windows",
    "GROUND_STATION_PRESETS": "environment:GROUND_STATION_PRESETS",
    "ISLNetwork": "environment:ISLNetwork",
    # Compute / fault injection / TMR
    "FaultInjector": "compute:FaultInjector",
    "FaultReport": "compute:FaultReport",
    "TransformerFaultInjector": "compute:TransformerFaultInjector",
    "TMRWrapper": "compute:TMRWrapper",
    "CheckpointManager": "compute:CheckpointManager",
    "InferenceScheduler": "compute:InferenceScheduler",
    "quantize_model": "compute:quantize_model",
    "compare_quantization_resilience": "compute:compare_quantization_resilience",
    # Hardware profiles
    "ChipProfile": "models:ChipProfile",
    "ALL_CHIPS": "models:ALL_CHIPS",
    # Metrics
    "MissionBudget": "metrics:MissionBudget",
    "compute_mission_budget": "metrics:compute_mission_budget",
    "MonteCarloResult": "metrics:MonteCarloResult",
    "estimate_mission_reliability": "metrics:estimate_mission_reliability",
    # Analysis
    "MissionConfig": "analysis:MissionConfig",
    "TradeStudy": "analysis:TradeStudy",
    "TradeStudyResult": "analysis:TradeStudyResult",
    "run_mission_analysis": "analysis:run_mission_analysis",
    "MissionAnalysisResult": "analysis:MissionAnalysisResult",
    # Comms / link budget
    "FrequencyBand": "comms:FrequencyBand",
    "LinkBudgetResult": "comms:LinkBudgetResult",
    "FREQUENCY_BANDS": "comms:FREQUENCY_BANDS",
    "compute_link_budget": "comms:compute_link_budget",
}

__all__ = ["__version__", *sorted(_LAZY_IMPORTS)]


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule, attr = target.split(":")
    module = import_module(f"{__name__}.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_IMPORTS))


if TYPE_CHECKING:
    # Static re-exports so IDEs, mypy, and pyright resolve top-level names.
    # ruff: noqa: F401  — runtime-visible via __getattr__/_LAZY_IMPORTS, not here.
    from space_ml_sim.analysis import (
        MissionAnalysisResult,
        MissionConfig,
        TradeStudy,
        TradeStudyResult,
        run_mission_analysis,
    )
    from space_ml_sim.comms import (
        FREQUENCY_BANDS,
        FrequencyBand,
        LinkBudgetResult,
        compute_link_budget,
    )
    from space_ml_sim.compute import (
        CheckpointManager,
        FaultInjector,
        FaultReport,
        InferenceScheduler,
        TMRWrapper,
        TransformerFaultInjector,
        compare_quantization_resilience,
        quantize_model,
    )
    from space_ml_sim.core import (
        CONSTELLATION_PRESETS,
        Constellation,
        ConstellationPreset,
        OrbitConfig,
        Satellite,
        SatelliteState,
        SimClock,
        generate_from_preset,
        is_in_eclipse,
        load_tle_file,
        parse_tle,
        position_at,
        propagate,
        propagate_sgp4,
        sun_synchronous_orbits,
        walker_delta_orbits,
    )
    from space_ml_sim.environment import (
        GROUND_STATION_PRESETS,
        ContactWindow,
        GroundStation,
        HeliocentricEnvironment,
        ISLNetwork,
        PowerModel,
        RadiationEnvironment,
        RadiationTimeline,
        SolarParticleEvent,
        SPEStatisticalModel,
        ThermalModel,
        find_contact_windows,
        mission_spe_dose,
        radiation_timeline,
    )
    from space_ml_sim.metrics import (
        MissionBudget,
        MonteCarloResult,
        compute_mission_budget,
        estimate_mission_reliability,
    )
    from space_ml_sim.models import ALL_CHIPS, ChipProfile
