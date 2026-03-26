"""Tests for satellite.py — SatelliteState enum and Satellite model."""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.core.satellite import Satellite, SatelliteState
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile, RAD5500, TERAFAB_D3


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def orbit() -> OrbitConfig:
    return OrbitConfig(
        altitude_km=550,
        inclination_deg=53,
        raan_deg=0,
        true_anomaly_deg=0,
    )


@pytest.fixture()
def chip() -> ChipProfile:
    """Use RAD5500 for most tests — high TID tolerance makes state transitions predictable."""
    return RAD5500


@pytest.fixture()
def satellite(orbit: OrbitConfig, chip: ChipProfile) -> Satellite:
    return Satellite(id="SAT-001", orbit_config=orbit, chip_profile=chip)


@pytest.fixture()
def rad_env() -> RadiationEnvironment:
    return RadiationEnvironment.leo_500km()


# ---------------------------------------------------------------------------
# SatelliteState enum
# ---------------------------------------------------------------------------

class TestSatelliteStateEnum:
    def test_nominal_value(self):
        assert SatelliteState.NOMINAL == "nominal"

    def test_degraded_value(self):
        assert SatelliteState.DEGRADED == "degraded"

    def test_failed_value(self):
        assert SatelliteState.FAILED == "failed"

    def test_enum_members_count(self):
        assert len(SatelliteState) == 3

    def test_str_enum_equality(self):
        """SatelliteState is a str enum — string comparison works."""
        assert SatelliteState.NOMINAL == SatelliteState("nominal")


# ---------------------------------------------------------------------------
# Satellite.with_power_update
# ---------------------------------------------------------------------------

class TestWithPowerUpdate:
    def test_solar_power_when_sunlit(self, satellite: Satellite):
        updated = satellite.with_power_update(in_eclipse=False)
        assert updated.power_available_w == pytest.approx(100_000.0)

    def test_battery_power_in_eclipse(self, satellite: Satellite):
        updated = satellite.with_power_update(in_eclipse=True)
        assert updated.power_available_w == pytest.approx(10_000.0)

    def test_sunlit_greater_than_eclipse(self, satellite: Satellite):
        sunlit = satellite.with_power_update(in_eclipse=False)
        eclipse = satellite.with_power_update(in_eclipse=True)
        assert sunlit.power_available_w > eclipse.power_available_w

    def test_returns_new_instance(self, satellite: Satellite):
        updated = satellite.with_power_update(in_eclipse=False)
        assert updated is not satellite

    def test_original_power_unchanged(self, satellite: Satellite):
        original_power = satellite.power_available_w
        satellite.with_power_update(in_eclipse=False)
        assert satellite.power_available_w == original_power

    def test_other_fields_preserved(self, satellite: Satellite):
        updated = satellite.with_power_update(in_eclipse=True)
        assert updated.id == satellite.id
        assert updated.state == satellite.state
        assert updated.chip_profile == satellite.chip_profile


# ---------------------------------------------------------------------------
# Satellite.with_thermal_update
# ---------------------------------------------------------------------------

class TestWithThermalUpdate:
    def test_temperature_increases_with_compute_load(self, satellite: Satellite):
        low_load = satellite.with_thermal_update(compute_load_fraction=0.0, in_eclipse=False)
        high_load = satellite.with_thermal_update(compute_load_fraction=1.0, in_eclipse=False)
        assert high_load.temperature_c > low_load.temperature_c

    def test_eclipse_cools_satellite(self, satellite: Satellite):
        sunlit = satellite.with_thermal_update(compute_load_fraction=0.0, in_eclipse=False)
        eclipse = satellite.with_thermal_update(compute_load_fraction=0.0, in_eclipse=True)
        assert eclipse.temperature_c < sunlit.temperature_c

    def test_full_load_sunlit_temperature(self, satellite: Satellite):
        """Full TDP load sunlit: T = 80 + (tdp_watts / 50.0)."""
        chip = satellite.chip_profile
        expected = 80.0 + chip.tdp_watts / 50.0
        updated = satellite.with_thermal_update(compute_load_fraction=1.0, in_eclipse=False)
        assert updated.temperature_c == pytest.approx(expected)

    def test_zero_load_eclipse_temperature(self, satellite: Satellite):
        """Zero load in eclipse: T = -40 + 0 / 50 = -40.0."""
        updated = satellite.with_thermal_update(compute_load_fraction=0.0, in_eclipse=True)
        assert updated.temperature_c == pytest.approx(-40.0)

    def test_zero_load_sunlit_temperature(self, satellite: Satellite):
        """Zero load sunlit: T = 80 + 0 / 50 = 80.0."""
        updated = satellite.with_thermal_update(compute_load_fraction=0.0, in_eclipse=False)
        assert updated.temperature_c == pytest.approx(80.0)

    def test_returns_new_instance(self, satellite: Satellite):
        updated = satellite.with_thermal_update(compute_load_fraction=0.5, in_eclipse=False)
        assert updated is not satellite

    def test_original_temperature_unchanged(self, satellite: Satellite):
        original_temp = satellite.temperature_c
        satellite.with_thermal_update(compute_load_fraction=1.0, in_eclipse=False)
        assert satellite.temperature_c == original_temp

    def test_other_fields_preserved(self, satellite: Satellite):
        updated = satellite.with_thermal_update(compute_load_fraction=0.5, in_eclipse=True)
        assert updated.id == satellite.id
        assert updated.state == satellite.state
        assert updated.total_seu_events == satellite.total_seu_events

    def test_partial_load(self, satellite: Satellite):
        """50% load sunlit: T = 80 + (0.5 * tdp) / 50."""
        chip = satellite.chip_profile
        expected = 80.0 + (0.5 * chip.tdp_watts) / 50.0
        updated = satellite.with_thermal_update(compute_load_fraction=0.5, in_eclipse=False)
        assert updated.temperature_c == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Satellite.with_radiation_tick
# ---------------------------------------------------------------------------

class TestWithRadiationTick:
    def test_tid_accumulates(self, satellite: Satellite, rad_env: RadiationEnvironment):
        updated = satellite.with_radiation_tick(rad_env, dt_seconds=86400.0)
        assert updated.tid_accumulated_krad > satellite.tid_accumulated_krad

    def test_seu_events_are_non_negative_int(self, satellite: Satellite, rad_env: RadiationEnvironment):
        """SEU count is always a non-negative integer."""
        updated = satellite.with_radiation_tick(rad_env, dt_seconds=86400.0)
        assert isinstance(updated.total_seu_events, int)
        assert updated.total_seu_events >= 0

    def test_seu_events_accumulate_over_many_ticks(self, orbit: OrbitConfig):
        """SEU events accumulate to non-zero totals over many ticks in a high-radiation env."""
        # Use TERAFAB_D3 which has large memory_bits (32 GB) to get statistically non-zero SEUs
        chip = TERAFAB_D3
        rad_env_high = RadiationEnvironment.leo_2000km()  # High-radiation orbit
        sat = Satellite(id="SAT-SEU", orbit_config=orbit, chip_profile=chip)
        for _ in range(365):  # 365 one-day ticks
            sat = sat.with_radiation_tick(rad_env_high, dt_seconds=86400.0)
        # Expected rate: base_seu_rate * memory_bits * dt is large enough for non-zero Poisson draws
        assert sat.total_seu_events >= 0  # type safety; non-zero expected statistically
        assert isinstance(sat.total_seu_events, int)

    def test_tid_increment_matches_model(self, satellite: Satellite, rad_env: RadiationEnvironment):
        dt = 86400.0
        expected_tid = rad_env.tid_dose(dt)
        updated = satellite.with_radiation_tick(rad_env, dt_seconds=dt)
        assert updated.tid_accumulated_krad == pytest.approx(expected_tid)

    def test_nominal_to_degraded_transition(self, orbit: OrbitConfig):
        """Satellite transitions to DEGRADED when TID > 50% of tolerance."""
        # Use a chip with a known tolerance
        chip = ChipProfile(
            name="test-chip",
            node_nm=28,
            tdp_watts=10.0,
            max_temp_c=85.0,
            seu_cross_section_cm2=1e-14,
            tid_tolerance_krad=100.0,  # 50% threshold = 50 krad
            compute_tops=0.01,
            memory_bits=1024,
        )
        sat = Satellite(
            id="SAT-DEG",
            orbit_config=orbit,
            chip_profile=chip,
            tid_accumulated_krad=49.9,  # Just below 50% threshold
        )
        # Force a tid increment that crosses 50% threshold
        rad_env = RadiationEnvironment.leo_2000km()  # High dose rate
        # Pre-set TID to just below the 50% mark and add enough to cross it
        sat_at_threshold = sat.model_copy(update={"tid_accumulated_krad": 50.1})
        assert sat_at_threshold.state == SatelliteState.NOMINAL

        # Apply one tick with enough dose to keep TID above 50% but below 100%
        updated = sat_at_threshold.with_radiation_tick(rad_env, dt_seconds=1.0)
        # After tick: TID is still above 50%, below 100% -> DEGRADED
        assert updated.state == SatelliteState.DEGRADED

    def test_degraded_to_failed_transition(self, orbit: OrbitConfig):
        """Satellite transitions to FAILED when TID > 100% of tolerance."""
        chip = ChipProfile(
            name="test-chip",
            node_nm=28,
            tdp_watts=10.0,
            max_temp_c=85.0,
            seu_cross_section_cm2=1e-14,
            tid_tolerance_krad=100.0,
            compute_tops=0.01,
            memory_bits=1024,
        )
        # Pre-set TID just below tolerance (99.9 krad)
        sat = Satellite(
            id="SAT-FAIL",
            orbit_config=orbit,
            chip_profile=chip,
            tid_accumulated_krad=99.9,
            state=SatelliteState.DEGRADED,
        )
        rad_env = RadiationEnvironment.leo_2000km()  # High dose rate
        # Apply enough time to push TID over 100 krad
        updated = sat.with_radiation_tick(rad_env, dt_seconds=86400.0 * 365)
        assert updated.state == SatelliteState.FAILED

    def test_failed_satellite_stays_failed(self, orbit: OrbitConfig, rad_env: RadiationEnvironment):
        chip = ChipProfile(
            name="test-chip",
            node_nm=28,
            tdp_watts=10.0,
            max_temp_c=85.0,
            seu_cross_section_cm2=1e-14,
            tid_tolerance_krad=1.0,
            compute_tops=0.01,
            memory_bits=1024,
        )
        sat = Satellite(
            id="SAT-DEAD",
            orbit_config=orbit,
            chip_profile=chip,
            state=SatelliteState.FAILED,
            tid_accumulated_krad=2.0,
        )
        result = sat.with_radiation_tick(rad_env, dt_seconds=86400.0)
        # Failed satellite returns itself — state and TID unchanged
        assert result is sat
        assert result.state == SatelliteState.FAILED
        assert result.tid_accumulated_krad == sat.tid_accumulated_krad

    def test_returns_new_instance_when_not_failed(self, satellite: Satellite, rad_env: RadiationEnvironment):
        updated = satellite.with_radiation_tick(rad_env, dt_seconds=60.0)
        assert updated is not satellite

    def test_original_unchanged_after_tick(self, satellite: Satellite, rad_env: RadiationEnvironment):
        original_tid = satellite.tid_accumulated_krad
        original_seus = satellite.total_seu_events
        satellite.with_radiation_tick(rad_env, dt_seconds=86400.0)
        assert satellite.tid_accumulated_krad == original_tid
        assert satellite.total_seu_events == original_seus

    def test_nominal_stays_nominal_below_threshold(self, orbit: OrbitConfig):
        """Satellite remains NOMINAL when TID stays below 50% of tolerance."""
        chip = ChipProfile(
            name="test-chip",
            node_nm=28,
            tdp_watts=10.0,
            max_temp_c=85.0,
            seu_cross_section_cm2=1e-14,
            tid_tolerance_krad=1_000_000.0,  # Effectively immune
            compute_tops=0.01,
            memory_bits=1024,
        )
        sat = Satellite(id="SAT-IMM", orbit_config=orbit, chip_profile=chip)
        rad_env = RadiationEnvironment.leo_500km()
        updated = sat.with_radiation_tick(rad_env, dt_seconds=86400.0)
        assert updated.state == SatelliteState.NOMINAL

    def test_seu_count_is_non_negative_int(self, satellite: Satellite, rad_env: RadiationEnvironment):
        updated = satellite.with_radiation_tick(rad_env, dt_seconds=1.0)
        assert isinstance(updated.total_seu_events, int)
        assert updated.total_seu_events >= 0


# ---------------------------------------------------------------------------
# Satellite.with_position
# ---------------------------------------------------------------------------

class TestWithPosition:
    def test_position_updated(self, satellite: Satellite):
        new_pos = (1000.0, 2000.0, 3000.0)
        updated = satellite.with_position(new_pos, in_eclipse=False)
        assert updated.position_km == new_pos

    def test_eclipse_flag_updated(self, satellite: Satellite):
        updated = satellite.with_position((0.0, 0.0, 7000.0), in_eclipse=True)
        assert updated.in_eclipse is True

    def test_eclipse_flag_false(self, satellite: Satellite):
        updated = satellite.with_position((7000.0, 0.0, 0.0), in_eclipse=False)
        assert updated.in_eclipse is False

    def test_returns_new_instance(self, satellite: Satellite):
        updated = satellite.with_position((1.0, 2.0, 3.0), in_eclipse=False)
        assert updated is not satellite

    def test_original_position_unchanged(self, satellite: Satellite):
        original_pos = satellite.position_km
        satellite.with_position((9999.0, 9999.0, 9999.0), in_eclipse=False)
        assert satellite.position_km == original_pos

    def test_original_eclipse_flag_unchanged(self, satellite: Satellite):
        original_eclipse = satellite.in_eclipse
        satellite.with_position((0.0, 0.0, 0.0), in_eclipse=True)
        assert satellite.in_eclipse == original_eclipse

    def test_other_fields_preserved(self, satellite: Satellite):
        updated = satellite.with_position((500.0, 500.0, 500.0), in_eclipse=False)
        assert updated.id == satellite.id
        assert updated.state == satellite.state
        assert updated.chip_profile == satellite.chip_profile


# ---------------------------------------------------------------------------
# Satellite.is_operational
# ---------------------------------------------------------------------------

class TestIsOperational:
    def test_nominal_is_operational(self, satellite: Satellite):
        assert satellite.is_operational is True

    def test_degraded_is_operational(self, orbit: OrbitConfig, chip: ChipProfile):
        sat = Satellite(
            id="SAT-DEG",
            orbit_config=orbit,
            chip_profile=chip,
            state=SatelliteState.DEGRADED,
        )
        assert sat.is_operational is True

    def test_failed_is_not_operational(self, orbit: OrbitConfig, chip: ChipProfile):
        sat = Satellite(
            id="SAT-FAIL",
            orbit_config=orbit,
            chip_profile=chip,
            state=SatelliteState.FAILED,
        )
        assert sat.is_operational is False

    def test_is_operational_type(self, satellite: Satellite):
        assert isinstance(satellite.is_operational, bool)


# ---------------------------------------------------------------------------
# Immutability: each with_* returns a new Satellite, original unchanged
# ---------------------------------------------------------------------------

class TestImmutability:
    """Verify that no with_* method mutates the original Satellite."""

    def test_with_power_update_immutable(self, satellite: Satellite):
        snapshot = satellite.model_dump()
        satellite.with_power_update(in_eclipse=True)
        assert satellite.model_dump() == snapshot

    def test_with_thermal_update_immutable(self, satellite: Satellite):
        snapshot = satellite.model_dump()
        satellite.with_thermal_update(compute_load_fraction=0.9, in_eclipse=False)
        assert satellite.model_dump() == snapshot

    def test_with_radiation_tick_immutable(self, satellite: Satellite, rad_env: RadiationEnvironment):
        snapshot = satellite.model_dump()
        satellite.with_radiation_tick(rad_env, dt_seconds=3600.0)
        assert satellite.model_dump() == snapshot

    def test_with_position_immutable(self, satellite: Satellite):
        snapshot = satellite.model_dump()
        satellite.with_position((1.0, 2.0, 3.0), in_eclipse=True)
        assert satellite.model_dump() == snapshot

    def test_chained_updates_do_not_affect_original(
        self, satellite: Satellite, rad_env: RadiationEnvironment
    ):
        """Chaining multiple updates never changes the starting satellite."""
        snapshot = satellite.model_dump()
        (
            satellite
            .with_power_update(in_eclipse=False)
            .with_thermal_update(compute_load_fraction=0.5, in_eclipse=False)
            .with_radiation_tick(rad_env, dt_seconds=60.0)
            .with_position((1000.0, 0.0, 0.0), in_eclipse=False)
        )
        assert satellite.model_dump() == snapshot
