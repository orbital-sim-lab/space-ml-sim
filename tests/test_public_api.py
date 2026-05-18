"""1.0 API stability snapshot.

Every name in `EXPECTED_PUBLIC_API` is part of the documented public surface.
Removing or renaming any name in this set is a breaking change and requires
a major version bump plus a deprecation cycle (see ROADMAP.md, v1.0 section).

To intentionally extend the API:
1. Add the new name to ``space_ml_sim.__init__._LAZY_IMPORTS``.
2. Add the new name to ``EXPECTED_PUBLIC_API`` below.
3. Document it in CHANGELOG under "Added".

To intentionally remove a name (major version bump only):
1. Remove from both places.
2. Document in CHANGELOG under "Removed" with the migration path.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import space_ml_sim


EXPECTED_PUBLIC_API: frozenset[str] = frozenset(
    {
        # Core orbital mechanics
        "OrbitConfig",
        "Satellite",
        "SatelliteState",
        "Constellation",
        "SimClock",
        "propagate",
        "position_at",
        "walker_delta_orbits",
        "sun_synchronous_orbits",
        "is_in_eclipse",
        "parse_tle",
        "load_tle_file",
        "propagate_sgp4",
        "ConstellationPreset",
        "CONSTELLATION_PRESETS",
        "generate_from_preset",
        # Radiation environments
        "RadiationEnvironment",
        "HeliocentricEnvironment",
        "SPEStatisticalModel",
        "SolarParticleEvent",
        "mission_spe_dose",
        "PowerModel",
        "ThermalModel",
        "RadiationTimeline",
        "radiation_timeline",
        "GroundStation",
        "ContactWindow",
        "find_contact_windows",
        "GROUND_STATION_PRESETS",
        "ISLNetwork",
        # Compute / fault injection / TMR
        "FaultInjector",
        "FaultReport",
        "TransformerFaultInjector",
        "TMRWrapper",
        "CheckpointManager",
        "InferenceScheduler",
        "quantize_model",
        "compare_quantization_resilience",
        # Hardware profiles
        "ChipProfile",
        "ALL_CHIPS",
        # Metrics
        "MissionBudget",
        "compute_mission_budget",
        "MonteCarloResult",
        "estimate_mission_reliability",
        # Analysis
        "MissionConfig",
        "TradeStudy",
        "TradeStudyResult",
        "run_mission_analysis",
        "MissionAnalysisResult",
        # Comms / link budget
        "FrequencyBand",
        "LinkBudgetResult",
        "FREQUENCY_BANDS",
        "compute_link_budget",
    }
)


def test_version_string_present_and_well_formed() -> None:
    """``space_ml_sim.__version__`` must exist and look like a PEP 440 version."""
    version = space_ml_sim.__version__
    assert isinstance(version, str)
    assert version.split(".")[0].isdigit(), f"major version not a digit: {version!r}"


def test_all_export_matches_expected_public_api() -> None:
    """``__all__`` must equal the documented stability snapshot (plus ``__version__``)."""
    declared = set(space_ml_sim.__all__) - {"__version__"}
    assert declared == EXPECTED_PUBLIC_API, (
        f"Public API drift detected.\n"
        f"Missing from __all__: {EXPECTED_PUBLIC_API - declared}\n"
        f"Unexpected in __all__: {declared - EXPECTED_PUBLIC_API}"
    )


def test_lazy_imports_match_all() -> None:
    """The lazy registry and ``__all__`` must agree — drift means a broken export."""
    lazy_names = set(space_ml_sim._LAZY_IMPORTS)  # noqa: SLF001 — invariant check
    declared = set(space_ml_sim.__all__) - {"__version__"}
    assert lazy_names == declared, (
        f"_LAZY_IMPORTS and __all__ disagree.\n"
        f"In __all__ only: {declared - lazy_names}\n"
        f"In _LAZY_IMPORTS only: {lazy_names - declared}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC_API))
def test_each_public_symbol_resolves(name: str) -> None:
    """Every documented symbol must resolve to a non-None object via ``getattr``."""
    obj = getattr(space_ml_sim, name)
    assert obj is not None, f"{name!r} resolved to None"


def test_bare_import_does_not_pull_torch() -> None:
    """``import space_ml_sim`` must stay cheap — no eager torch import.

    Why: torch is a heavy dependency. Users invoking ``space-ml-sim --version``
    or reading ``space_ml_sim.__version__`` should not pay a 1.5s import cost.
    Runs in a subprocess so this test cannot leak state into other tests
    (mutating the parent's ``sys.modules`` would break test isolation).
    """
    script = textwrap.dedent(
        """
        import sys
        import space_ml_sim  # noqa: F401
        print("torch" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "Bare `import space_ml_sim` eagerly imported torch. "
        "Keep top-level __init__.py lazy (PEP 562). "
        f"Subprocess stdout: {result.stdout!r}, stderr: {result.stderr!r}"
    )


def test_dir_includes_public_api() -> None:
    """``dir(space_ml_sim)`` must surface every public symbol for tab completion."""
    public = set(dir(space_ml_sim))
    missing = EXPECTED_PUBLIC_API - public
    assert not missing, f"dir(space_ml_sim) missing public symbols: {missing}"


def test_unknown_attribute_raises_attribute_error() -> None:
    """Accessing a non-public attribute must raise ``AttributeError`` (not a generic exception)."""
    with pytest.raises(AttributeError, match="DoesNotExist"):
        space_ml_sim.DoesNotExist  # noqa: B018
