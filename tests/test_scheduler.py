"""Tests for InferenceScheduler: node selection and schedule summary."""

from __future__ import annotations


from space_ml_sim.compute.scheduler import InferenceScheduler
from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.core.satellite import Satellite, SatelliteState
from space_ml_sim.models.chip_profiles import ChipProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_ORBIT = OrbitConfig(
    altitude_km=550,
    inclination_deg=53.0,
    raan_deg=0.0,
    true_anomaly_deg=0.0,
)

_DEFAULT_CHIP = ChipProfile(
    name="Test Chip",
    node_nm=28,
    tdp_watts=10.0,
    max_temp_c=100.0,
    seu_cross_section_cm2=1e-14,
    tid_tolerance_krad=50.0,
    compute_tops=1.0,
    memory_bits=256 * 8 * 1024 * 1024,  # 256 MB
)


def _make_satellite(
    sat_id: str,
    state: SatelliteState = SatelliteState.NOMINAL,
    power_available_w: float = 20.0,
    temperature_c: float = 30.0,
    chip: ChipProfile | None = None,
) -> Satellite:
    """Create a Satellite with sensible defaults for scheduling tests.

    Default chip TDP is 10 W.  Default power (20 W) is 2x TDP so it easily
    passes the scheduler's power filter (tdp * (1 + margin_fraction)).
    """
    return Satellite(
        id=sat_id,
        orbit_config=_DEFAULT_ORBIT,
        chip_profile=chip or _DEFAULT_CHIP,
        state=state,
        power_available_w=power_available_w,
        temperature_c=temperature_c,
    )


# The default scheduler uses power_margin_fraction=0.1, so the power threshold
# for the default chip (tdp=10 W) is 10 * 1.1 = 11 W.
_SCHEDULER = InferenceScheduler(power_margin_fraction=0.1)
_POWER_THRESHOLD = _DEFAULT_CHIP.tdp_watts * (1 + _SCHEDULER.power_margin_fraction)  # 11 W


# ---------------------------------------------------------------------------
# select_nodes: filtering
# ---------------------------------------------------------------------------

class TestSelectNodesFiltering:
    def test_filters_out_failed_satellites(self):
        sats = [
            _make_satellite("nominal-1", state=SatelliteState.NOMINAL),
            _make_satellite("failed-1", state=SatelliteState.FAILED),
            _make_satellite("nominal-2", state=SatelliteState.NOMINAL),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=5)
        ids = {s.id for s in selected}
        assert "failed-1" not in ids
        assert "nominal-1" in ids
        assert "nominal-2" in ids

    def test_degraded_satellites_are_not_filtered_out(self):
        """DEGRADED is still operational — it should pass the state filter."""
        sats = [
            _make_satellite("degraded-1", state=SatelliteState.DEGRADED),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=5)
        assert len(selected) == 1
        assert selected[0].id == "degraded-1"

    def test_filters_out_satellites_without_enough_power(self):
        """Power must be >= tdp * (1 + margin). Below threshold → excluded."""
        low_power = _make_satellite("low-power", power_available_w=_POWER_THRESHOLD - 0.01)
        ok_power = _make_satellite("ok-power", power_available_w=_POWER_THRESHOLD)
        selected = _SCHEDULER.select_nodes([low_power, ok_power], num_needed=5)
        ids = {s.id for s in selected}
        assert "low-power" not in ids
        assert "ok-power" in ids

    def test_satellite_exactly_at_power_threshold_is_included(self):
        sat = _make_satellite("exact-power", power_available_w=_POWER_THRESHOLD)
        selected = _SCHEDULER.select_nodes([sat], num_needed=5)
        assert len(selected) == 1

    def test_empty_satellite_list_returns_empty(self):
        selected = _SCHEDULER.select_nodes([], num_needed=3)
        assert selected == []

    def test_all_failed_returns_empty(self):
        sats = [
            _make_satellite(f"failed-{i}", state=SatelliteState.FAILED)
            for i in range(5)
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=5)
        assert selected == []

    def test_all_underpowered_returns_empty(self):
        sats = [
            _make_satellite(f"sat-{i}", power_available_w=0.5)
            for i in range(4)
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=4)
        assert selected == []


# ---------------------------------------------------------------------------
# select_nodes: num_needed cap
# ---------------------------------------------------------------------------

class TestSelectNodesNumNeeded:
    def test_returns_at_most_num_needed(self):
        sats = [_make_satellite(f"sat-{i}") for i in range(10)]
        selected = _SCHEDULER.select_nodes(sats, num_needed=3)
        assert len(selected) == 3

    def test_returns_fewer_when_not_enough_candidates(self):
        sats = [_make_satellite(f"sat-{i}") for i in range(2)]
        selected = _SCHEDULER.select_nodes(sats, num_needed=5)
        assert len(selected) == 2

    def test_num_needed_zero_returns_empty(self):
        sats = [_make_satellite(f"sat-{i}") for i in range(5)]
        selected = _SCHEDULER.select_nodes(sats, num_needed=0)
        assert selected == []


# ---------------------------------------------------------------------------
# select_nodes: ranking — nominal before degraded
# ---------------------------------------------------------------------------

class TestSelectNodesRankingNominalBeforeDegraded:
    def test_nominal_preferred_over_degraded(self):
        sats = [
            _make_satellite("degraded-1", state=SatelliteState.DEGRADED, temperature_c=20.0),
            _make_satellite("nominal-1", state=SatelliteState.NOMINAL, temperature_c=25.0),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=1)
        assert selected[0].id == "nominal-1"

    def test_nominal_always_before_degraded_regardless_of_order(self):
        sats = [
            _make_satellite("degraded-a", state=SatelliteState.DEGRADED, temperature_c=10.0),
            _make_satellite("nominal-a", state=SatelliteState.NOMINAL, temperature_c=50.0),
            _make_satellite("degraded-b", state=SatelliteState.DEGRADED, temperature_c=15.0),
            _make_satellite("nominal-b", state=SatelliteState.NOMINAL, temperature_c=45.0),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=2)
        states = [s.state for s in selected]
        assert all(st == SatelliteState.NOMINAL for st in states)

    def test_top_two_from_mixed_list_are_nominal(self):
        """With 2 nominal and 2 degraded, asking for 2 gives both nominals."""
        sats = [
            _make_satellite("d1", state=SatelliteState.DEGRADED),
            _make_satellite("n1", state=SatelliteState.NOMINAL),
            _make_satellite("d2", state=SatelliteState.DEGRADED),
            _make_satellite("n2", state=SatelliteState.NOMINAL),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=2)
        ids = {s.id for s in selected}
        assert ids == {"n1", "n2"}


# ---------------------------------------------------------------------------
# select_nodes: ranking — cooler satellites preferred
# ---------------------------------------------------------------------------

class TestSelectNodesRankingCooler:
    def test_cooler_satellite_ranked_first_when_same_state(self):
        sats = [
            _make_satellite("hot", state=SatelliteState.NOMINAL, temperature_c=80.0),
            _make_satellite("cool", state=SatelliteState.NOMINAL, temperature_c=20.0),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=1)
        assert selected[0].id == "cool"

    def test_coolest_three_selected_from_five(self):
        temps = [50.0, 10.0, 90.0, 30.0, 70.0]
        sats = [
            _make_satellite(f"sat-{i}", temperature_c=t)
            for i, t in enumerate(temps)
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=3)
        selected_temps = sorted(s.temperature_c for s in selected)
        assert selected_temps == [10.0, 30.0, 50.0]

    def test_nominal_cooler_beats_nominal_hotter(self):
        sats = [
            _make_satellite("warm-nominal", state=SatelliteState.NOMINAL, temperature_c=60.0),
            _make_satellite("cold-nominal", state=SatelliteState.NOMINAL, temperature_c=15.0),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=1)
        assert selected[0].id == "cold-nominal"

    def test_nominal_hot_still_beats_degraded_cool(self):
        """State (nominal > degraded) takes priority over temperature."""
        sats = [
            _make_satellite("hot-nominal", state=SatelliteState.NOMINAL, temperature_c=99.0),
            _make_satellite("cool-degraded", state=SatelliteState.DEGRADED, temperature_c=1.0),
        ]
        selected = _SCHEDULER.select_nodes(sats, num_needed=1)
        assert selected[0].id == "hot-nominal"


# ---------------------------------------------------------------------------
# schedule_summary
# ---------------------------------------------------------------------------

class TestScheduleSummary:
    def _make_constellation(self) -> list[Satellite]:
        """Build a known constellation for predictable summary assertions.

        Composition:
          - 2 nominal, fully powered, within thermal limits  → ready
          - 1 degraded, fully powered, within thermal limits → ready
          - 1 nominal, low power (< TDP)                    → not power_available
          - 1 failed                                         → not operational
          - 1 nominal, fully powered, over thermal limit    → not within_thermal_limits

        TDP = 10 W, max_temp = 100°C.
        """
        ready_1 = _make_satellite("ready-1", state=SatelliteState.NOMINAL, power_available_w=15.0, temperature_c=50.0)
        ready_2 = _make_satellite("ready-2", state=SatelliteState.NOMINAL, power_available_w=15.0, temperature_c=60.0)
        ready_3 = _make_satellite("ready-3", state=SatelliteState.DEGRADED, power_available_w=15.0, temperature_c=70.0)
        low_pwr = _make_satellite("low-pwr", state=SatelliteState.NOMINAL, power_available_w=5.0, temperature_c=40.0)
        failed  = _make_satellite("failed-1", state=SatelliteState.FAILED, power_available_w=15.0, temperature_c=30.0)
        hot_sat = _make_satellite("hot-sat", state=SatelliteState.NOMINAL, power_available_w=15.0, temperature_c=110.0)
        return [ready_1, ready_2, ready_3, low_pwr, failed, hot_sat]

    def test_total_satellites_count(self):
        sats = self._make_constellation()
        summary = _SCHEDULER.schedule_summary(sats)
        assert summary["total_satellites"] == 6

    def test_operational_count_excludes_failed(self):
        sats = self._make_constellation()
        summary = _SCHEDULER.schedule_summary(sats)
        # failed-1 excluded → 5 operational
        assert summary["operational"] == 5

    def test_power_available_count(self):
        sats = self._make_constellation()
        summary = _SCHEDULER.schedule_summary(sats)
        # low-pwr (5 W < 10 W TDP) and failed excluded → 4 power_available
        assert summary["power_available"] == 4

    def test_within_thermal_limits_count(self):
        sats = self._make_constellation()
        summary = _SCHEDULER.schedule_summary(sats)
        # hot-sat (110°C > 100°C max) excluded → 3 within_thermal_limits
        assert summary["within_thermal_limits"] == 3

    def test_ready_for_inference_equals_within_thermal(self):
        sats = self._make_constellation()
        summary = _SCHEDULER.schedule_summary(sats)
        assert summary["ready_for_inference"] == summary["within_thermal_limits"]

    def test_summary_keys_present(self):
        summary = _SCHEDULER.schedule_summary([])
        expected_keys = {
            "total_satellites",
            "operational",
            "power_available",
            "within_thermal_limits",
            "ready_for_inference",
        }
        assert expected_keys == set(summary.keys())

    def test_empty_constellation_all_zeros(self):
        summary = _SCHEDULER.schedule_summary([])
        for key, value in summary.items():
            assert value == 0, f"Expected 0 for {key}, got {value}"

    def test_all_nominal_fully_powered_and_cool(self):
        sats = [
            _make_satellite(f"sat-{i}", power_available_w=50.0, temperature_c=25.0)
            for i in range(4)
        ]
        summary = _SCHEDULER.schedule_summary(sats)
        assert summary["total_satellites"] == 4
        assert summary["operational"] == 4
        assert summary["power_available"] == 4
        assert summary["within_thermal_limits"] == 4
        assert summary["ready_for_inference"] == 4
