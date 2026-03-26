"""Tests for metrics/reliability.py and metrics/performance.py."""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.core.satellite import Satellite, SatelliteState
from space_ml_sim.metrics.performance import PerformanceMetrics
from space_ml_sim.metrics.reliability import ReliabilityMetrics
from space_ml_sim.models.chip_profiles import ChipProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEO_ORBIT = OrbitConfig(
    altitude_km=550,
    inclination_deg=53.0,
    raan_deg=0.0,
    true_anomaly_deg=0.0,
)

CHIP_A = ChipProfile(
    name="TestChip-A",
    node_nm=28,
    tdp_watts=10.0,
    max_temp_c=125.0,
    seu_cross_section_cm2=1e-14,
    tid_tolerance_krad=100.0,
    compute_tops=5.0,
    memory_bits=256 * 8 * 1024**2,
)

CHIP_B = ChipProfile(
    name="TestChip-B",
    node_nm=14,
    tdp_watts=20.0,
    max_temp_c=85.0,
    seu_cross_section_cm2=5e-13,
    tid_tolerance_krad=50.0,
    compute_tops=10.0,
    memory_bits=512 * 8 * 1024**2,
)


def _make_satellite(
    sat_id: str,
    state: SatelliteState = SatelliteState.NOMINAL,
    chip: ChipProfile | None = None,
    seu_events: int = 0,
    tid_krad: float = 0.0,
) -> Satellite:
    """Helper to construct a Satellite with controlled state."""
    return Satellite(
        id=sat_id,
        orbit_config=LEO_ORBIT,
        chip_profile=chip if chip is not None else CHIP_A,
        state=state,
        total_seu_events=seu_events,
        tid_accumulated_krad=tid_krad,
    )


# ---------------------------------------------------------------------------
# ReliabilityMetrics tests
# ---------------------------------------------------------------------------


class TestReliabilityMetricsFromSatellites:
    """Tests for ReliabilityMetrics.from_satellites factory."""

    def test_counts_mixed_states(self):
        """from_satellites correctly counts nominal, degraded, and failed satellites."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL),
            _make_satellite("sat-2", SatelliteState.NOMINAL),
            _make_satellite("sat-3", SatelliteState.DEGRADED),
            _make_satellite("sat-4", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.total_satellites == 4
        assert metrics.nominal_count == 2
        assert metrics.degraded_count == 1
        assert metrics.failed_count == 1

    def test_all_nominal(self):
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL),
            _make_satellite("sat-2", SatelliteState.NOMINAL),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.nominal_count == 2
        assert metrics.degraded_count == 0
        assert metrics.failed_count == 0

    def test_all_failed(self):
        satellites = [
            _make_satellite("sat-1", SatelliteState.FAILED),
            _make_satellite("sat-2", SatelliteState.FAILED),
            _make_satellite("sat-3", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.nominal_count == 0
        assert metrics.degraded_count == 0
        assert metrics.failed_count == 3

    def test_total_seu_events_summed(self):
        satellites = [
            _make_satellite("sat-1", seu_events=5),
            _make_satellite("sat-2", seu_events=12),
            _make_satellite("sat-3", seu_events=3),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.total_seu_events == 20

    def test_max_tid_krad(self):
        satellites = [
            _make_satellite("sat-1", tid_krad=10.0),
            _make_satellite("sat-2", tid_krad=50.0),
            _make_satellite("sat-3", tid_krad=25.0),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.max_tid_krad == pytest.approx(50.0)

    def test_mean_tid_krad(self):
        satellites = [
            _make_satellite("sat-1", tid_krad=10.0),
            _make_satellite("sat-2", tid_krad=20.0),
            _make_satellite("sat-3", tid_krad=30.0),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.mean_tid_krad == pytest.approx(20.0)

    def test_empty_satellite_list(self):
        """from_satellites on empty list returns zero counts and zero TID."""
        metrics = ReliabilityMetrics.from_satellites([])

        assert metrics.total_satellites == 0
        assert metrics.nominal_count == 0
        assert metrics.degraded_count == 0
        assert metrics.failed_count == 0
        assert metrics.total_seu_events == 0
        assert metrics.max_tid_krad == 0.0
        assert metrics.mean_tid_krad == 0.0

    def test_single_satellite(self):
        satellites = [
            _make_satellite("sat-1", SatelliteState.DEGRADED, seu_events=7, tid_krad=30.0)
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.total_satellites == 1
        assert metrics.degraded_count == 1
        assert metrics.total_seu_events == 7
        assert metrics.max_tid_krad == pytest.approx(30.0)
        assert metrics.mean_tid_krad == pytest.approx(30.0)


class TestReliabilityMetricsAvailability:
    """Tests for ReliabilityMetrics.availability property."""

    def test_availability_nominal_and_degraded_over_total(self):
        """availability = (nominal + degraded) / total."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL),
            _make_satellite("sat-2", SatelliteState.NOMINAL),
            _make_satellite("sat-3", SatelliteState.DEGRADED),
            _make_satellite("sat-4", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        # (2 nominal + 1 degraded) / 4 total = 0.75
        assert metrics.availability == pytest.approx(0.75)

    def test_availability_all_nominal(self):
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL),
            _make_satellite("sat-2", SatelliteState.NOMINAL),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.availability == pytest.approx(1.0)

    def test_availability_zero_for_empty_list(self):
        """Availability is 0.0 when no satellites exist."""
        metrics = ReliabilityMetrics.from_satellites([])

        assert metrics.availability == 0.0

    def test_availability_zero_for_all_failed(self):
        """All-failed constellation has availability 0."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.FAILED),
            _make_satellite("sat-2", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.availability == 0.0

    def test_availability_only_degraded_counts_as_operational(self):
        """Degraded satellites contribute to availability, failed do not."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.DEGRADED),
            _make_satellite("sat-2", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.availability == pytest.approx(0.5)

    def test_availability_fractional_precision(self):
        """Availability is computed as a float fraction."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL),
            _make_satellite("sat-2", SatelliteState.NOMINAL),
            _make_satellite("sat-3", SatelliteState.FAILED),
        ]
        metrics = ReliabilityMetrics.from_satellites(satellites)

        assert metrics.availability == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# PerformanceMetrics tests
# ---------------------------------------------------------------------------


class TestPerformanceMetricsFromSatellites:
    """Tests for PerformanceMetrics.from_satellites factory."""

    def test_total_tops_sums_all_satellites(self):
        """total_tops sums compute_tops across all satellites, regardless of state."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),  # 5 TOPS
            _make_satellite("sat-2", SatelliteState.DEGRADED, chip=CHIP_B),  # 10 TOPS
            _make_satellite("sat-3", SatelliteState.FAILED, chip=CHIP_A),  # 5 TOPS
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.total_tops == pytest.approx(20.0)

    def test_active_tops_sums_operational_satellites_only(self):
        """active_tops excludes failed satellites (only nominal and degraded)."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),  # 5 TOPS — operational
            _make_satellite("sat-2", SatelliteState.DEGRADED, chip=CHIP_B),  # 10 TOPS — operational
            _make_satellite(
                "sat-3", SatelliteState.FAILED, chip=CHIP_A
            ),  # 5 TOPS — not operational
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.active_tops == pytest.approx(15.0)

    def test_tops_per_watt_is_zero_when_no_active_power(self):
        """tops_per_watt is 0.0 when there are no operational satellites."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.FAILED, chip=CHIP_A),
            _make_satellite("sat-2", SatelliteState.FAILED, chip=CHIP_B),
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.tops_per_watt == 0.0

    def test_tops_per_watt_calculated_from_active_only(self):
        """tops_per_watt = active_tops / active_power_watts."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),  # 5 TOPS, 10W
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        expected = CHIP_A.compute_tops / CHIP_A.tdp_watts  # 5 / 10 = 0.5
        assert metrics.tops_per_watt == pytest.approx(expected)

    def test_total_power_watts_sums_all_satellites(self):
        """total_power_watts sums tdp_watts across all satellites."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),  # 10W
            _make_satellite("sat-2", SatelliteState.FAILED, chip=CHIP_B),  # 20W
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.total_power_watts == pytest.approx(30.0)

    def test_active_power_watts_sums_operational_only(self):
        """active_power_watts excludes failed satellites."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),  # 10W — operational
            _make_satellite("sat-2", SatelliteState.FAILED, chip=CHIP_B),  # 20W — not operational
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.active_power_watts == pytest.approx(10.0)

    def test_empty_satellite_list_returns_zero_metrics(self):
        """Empty satellite list produces all-zero PerformanceMetrics."""
        metrics = PerformanceMetrics.from_satellites([])

        assert metrics.total_tops == 0.0
        assert metrics.active_tops == 0.0
        assert metrics.total_power_watts == 0.0
        assert metrics.active_power_watts == 0.0
        assert metrics.tops_per_watt == 0.0

    def test_all_operational_gives_equal_total_and_active_tops(self):
        """When no satellites are failed, total_tops equals active_tops."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A),
            _make_satellite("sat-2", SatelliteState.DEGRADED, chip=CHIP_B),
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.total_tops == pytest.approx(metrics.active_tops)

    def test_all_failed_gives_zero_active_metrics(self):
        """All-failed constellation has zero active TOPS and power."""
        satellites = [
            _make_satellite("sat-1", SatelliteState.FAILED, chip=CHIP_A),
            _make_satellite("sat-2", SatelliteState.FAILED, chip=CHIP_B),
        ]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.active_tops == 0.0
        assert metrics.active_power_watts == 0.0
        assert metrics.tops_per_watt == 0.0

    def test_single_operational_satellite(self):
        satellites = [_make_satellite("sat-1", SatelliteState.NOMINAL, chip=CHIP_A)]
        metrics = PerformanceMetrics.from_satellites(satellites)

        assert metrics.total_tops == pytest.approx(CHIP_A.compute_tops)
        assert metrics.active_tops == pytest.approx(CHIP_A.compute_tops)
        assert metrics.total_power_watts == pytest.approx(CHIP_A.tdp_watts)
        assert metrics.active_power_watts == pytest.approx(CHIP_A.tdp_watts)
        assert metrics.tops_per_watt == pytest.approx(CHIP_A.compute_tops / CHIP_A.tdp_watts)
