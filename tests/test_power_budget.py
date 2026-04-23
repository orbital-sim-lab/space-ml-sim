"""TDD tests for orbital power budget calculator.

Models power generation (solar panels) and consumption (compute, comms,
thermal control) across orbital phases (sunlit/eclipse) to determine
if the satellite has sufficient power for AI inference.
"""

from __future__ import annotations


from space_ml_sim.core.orbit import OrbitConfig


class TestPowerBudgetCreation:
    """Power budget must accept satellite power parameters."""

    def test_create_budget(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            PowerBudgetResult,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=2.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=100.0, dod_limit=0.4),
            base_load_w=10.0,
            compute_load_w=15.0,
            comms_load_w=5.0,
            thermal_load_w=3.0,
        ).analyze()

        assert isinstance(result, PowerBudgetResult)

    def test_result_has_key_metrics(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=2.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=100.0, dod_limit=0.4),
            base_load_w=10.0,
            compute_load_w=15.0,
        ).analyze()

        assert result.solar_generation_w > 0
        assert result.total_load_sunlit_w > 0
        assert result.total_load_eclipse_w > 0
        assert result.eclipse_duration_minutes > 0
        assert result.sunlit_duration_minutes > 0
        assert isinstance(result.power_positive, bool)


class TestPowerMargins:
    """Power budget must correctly compute margins."""

    def test_sufficient_power_is_positive(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        # Large solar array, small load
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=5.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=200.0, dod_limit=0.4),
            base_load_w=5.0,
            compute_load_w=10.0,
        ).analyze()

        assert result.power_positive is True
        assert result.margin_w > 0

    def test_insufficient_power_is_negative(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        # Tiny solar array, huge load
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=0.1, efficiency=0.10),
            battery=BatteryConfig(capacity_wh=10.0, dod_limit=0.4),
            base_load_w=50.0,
            compute_load_w=100.0,
        ).analyze()

        assert result.power_positive is False

    def test_tmr_increases_compute_load(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        kwargs = dict(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=2.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=100.0, dod_limit=0.4),
            base_load_w=10.0,
            compute_load_w=15.0,
        )
        r_none = PowerBudget(**kwargs, tmr_multiplier=1.0).analyze()
        r_tmr = PowerBudget(**kwargs, tmr_multiplier=3.0).analyze()

        assert r_tmr.total_load_sunlit_w > r_none.total_load_sunlit_w


class TestBatteryAnalysis:
    """Battery must sustain eclipse loads."""

    def test_battery_sufficient_for_eclipse(self) -> None:
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=2.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=200.0, dod_limit=0.4),
            base_load_w=10.0,
            compute_load_w=15.0,
        ).analyze()

        assert result.battery_eclipse_wh > 0
        assert result.battery_dod_fraction <= 1.0

    def test_eclipse_compute_duty_cycle(self) -> None:
        """Report how much compute can run during eclipse."""
        from space_ml_sim.analysis.power_budget import (
            PowerBudget,
            SolarArrayConfig,
            BatteryConfig,
        )

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        result = PowerBudget(
            orbit=orbit,
            solar_array=SolarArrayConfig(area_m2=2.0, efficiency=0.30),
            battery=BatteryConfig(capacity_wh=100.0, dod_limit=0.4),
            base_load_w=10.0,
            compute_load_w=30.0,
        ).analyze()

        assert 0.0 <= result.eclipse_compute_duty_cycle <= 1.0
