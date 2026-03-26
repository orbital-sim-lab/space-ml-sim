"""Tests for power environment model."""

import pytest

from space_ml_sim.environment.power import PowerModel


class TestAvailablePowerDefaults:
    def test_returns_solar_power_when_not_in_eclipse(self):
        model = PowerModel()
        assert model.available_power(in_eclipse=False) == model.solar_power_watts

    def test_returns_battery_power_when_in_eclipse(self):
        model = PowerModel()
        assert model.available_power(in_eclipse=True) == model.battery_power_watts

    def test_default_solar_power_is_100kw(self):
        model = PowerModel()
        assert model.solar_power_watts == 100_000.0

    def test_default_battery_power_is_10kw(self):
        model = PowerModel()
        assert model.battery_power_watts == 10_000.0

    def test_solar_power_greater_than_battery_power_by_default(self):
        model = PowerModel()
        assert model.solar_power_watts > model.battery_power_watts


class TestAvailablePowerCustomValues:
    def test_custom_solar_power_returned_when_sunlit(self):
        model = PowerModel(solar_power_watts=50_000.0, battery_power_watts=5_000.0)
        assert model.available_power(in_eclipse=False) == 50_000.0

    def test_custom_battery_power_returned_when_in_eclipse(self):
        model = PowerModel(solar_power_watts=50_000.0, battery_power_watts=5_000.0)
        assert model.available_power(in_eclipse=True) == 5_000.0

    def test_custom_equal_solar_and_battery_power(self):
        model = PowerModel(solar_power_watts=20_000.0, battery_power_watts=20_000.0)
        assert model.available_power(in_eclipse=False) == 20_000.0
        assert model.available_power(in_eclipse=True) == 20_000.0

    def test_small_power_values_are_accepted(self):
        model = PowerModel(solar_power_watts=1.0, battery_power_watts=0.5)
        assert model.available_power(in_eclipse=False) == 1.0
        assert model.available_power(in_eclipse=True) == 0.5


class TestPowerModelValidation:
    def test_solar_power_must_be_positive(self):
        with pytest.raises(Exception):
            PowerModel(solar_power_watts=0.0)

    def test_battery_power_must_be_positive(self):
        with pytest.raises(Exception):
            PowerModel(battery_power_watts=0.0)

    def test_negative_solar_power_rejected(self):
        with pytest.raises(Exception):
            PowerModel(solar_power_watts=-100.0)

    def test_negative_battery_power_rejected(self):
        with pytest.raises(Exception):
            PowerModel(battery_power_watts=-1.0)


class TestPowerModelImmutability:
    def test_model_is_frozen_or_independent_between_calls(self):
        """Calling available_power twice should return the same value."""
        model = PowerModel()
        first = model.available_power(in_eclipse=False)
        second = model.available_power(in_eclipse=False)
        assert first == second

    def test_eclipse_state_does_not_persist(self):
        """Eclipse state is passed per call, not stored on the model."""
        model = PowerModel()
        _ = model.available_power(in_eclipse=True)
        result = model.available_power(in_eclipse=False)
        assert result == model.solar_power_watts
