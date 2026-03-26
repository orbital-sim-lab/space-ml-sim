"""Tests for thermal environment model."""

import pytest

from space_ml_sim.environment.thermal import ThermalModel


class TestAmbientTemperature:
    def test_zero_compute_in_eclipse_returns_ambient_eclipse(self):
        model = ThermalModel()
        result = model.compute_temperature(compute_power_watts=0.0, in_eclipse=True)
        assert result == model.ambient_eclipse_c

    def test_zero_compute_in_sun_returns_ambient_sun(self):
        model = ThermalModel()
        result = model.compute_temperature(compute_power_watts=0.0, in_eclipse=False)
        assert result == model.ambient_sun_c

    def test_default_eclipse_ambient_is_minus_40(self):
        model = ThermalModel()
        assert model.ambient_eclipse_c == -40.0

    def test_default_sun_ambient_is_plus_80(self):
        model = ThermalModel()
        assert model.ambient_sun_c == 80.0


class TestComputeTemperatureIncrease:
    def test_higher_compute_power_yields_higher_temperature_in_sun(self):
        model = ThermalModel()
        low = model.compute_temperature(compute_power_watts=100.0, in_eclipse=False)
        high = model.compute_temperature(compute_power_watts=1000.0, in_eclipse=False)
        assert high > low

    def test_higher_compute_power_yields_higher_temperature_in_eclipse(self):
        model = ThermalModel()
        low = model.compute_temperature(compute_power_watts=100.0, in_eclipse=True)
        high = model.compute_temperature(compute_power_watts=1000.0, in_eclipse=True)
        assert high > low

    def test_temperature_proportional_to_power_over_conductance_in_sun(self):
        model = ThermalModel(radiator_conductance_w_per_c=50.0, ambient_sun_c=80.0)
        result = model.compute_temperature(compute_power_watts=500.0, in_eclipse=False)
        expected = 80.0 + 500.0 / 50.0
        assert result == pytest.approx(expected)

    def test_temperature_proportional_to_power_over_conductance_in_eclipse(self):
        model = ThermalModel(radiator_conductance_w_per_c=50.0, ambient_eclipse_c=-40.0)
        result = model.compute_temperature(compute_power_watts=500.0, in_eclipse=True)
        expected = -40.0 + 500.0 / 50.0
        assert result == pytest.approx(expected)

    def test_doubling_compute_power_doubles_temperature_rise(self):
        model = ThermalModel(ambient_sun_c=0.0)
        temp_1x = model.compute_temperature(compute_power_watts=200.0, in_eclipse=False)
        temp_2x = model.compute_temperature(compute_power_watts=400.0, in_eclipse=False)
        assert temp_2x == pytest.approx(2 * temp_1x)

    def test_higher_conductance_reduces_temperature_rise(self):
        low_cond = ThermalModel(radiator_conductance_w_per_c=10.0)
        high_cond = ThermalModel(radiator_conductance_w_per_c=100.0)
        compute = 500.0
        temp_low = low_cond.compute_temperature(compute_power_watts=compute, in_eclipse=False)
        temp_high = high_cond.compute_temperature(compute_power_watts=compute, in_eclipse=False)
        assert temp_high < temp_low


class TestEclipseVsSunDifference:
    def test_same_compute_eclipse_is_cooler_than_sun(self):
        model = ThermalModel()
        in_eclipse = model.compute_temperature(compute_power_watts=200.0, in_eclipse=True)
        in_sun = model.compute_temperature(compute_power_watts=200.0, in_eclipse=False)
        assert in_eclipse < in_sun

    def test_eclipse_sun_delta_equals_ambient_difference(self):
        model = ThermalModel(ambient_sun_c=80.0, ambient_eclipse_c=-40.0)
        compute = 300.0
        temp_sun = model.compute_temperature(compute_power_watts=compute, in_eclipse=False)
        temp_eclipse = model.compute_temperature(compute_power_watts=compute, in_eclipse=True)
        expected_delta = 80.0 - (-40.0)
        assert (temp_sun - temp_eclipse) == pytest.approx(expected_delta)


class TestCustomThermalModel:
    def test_custom_ambient_and_conductance_values(self):
        model = ThermalModel(
            ambient_sun_c=60.0,
            ambient_eclipse_c=-20.0,
            radiator_conductance_w_per_c=25.0,
        )
        result = model.compute_temperature(compute_power_watts=250.0, in_eclipse=False)
        expected = 60.0 + 250.0 / 25.0
        assert result == pytest.approx(expected)

    def test_custom_model_eclipse_path(self):
        model = ThermalModel(
            ambient_eclipse_c=-60.0,
            radiator_conductance_w_per_c=40.0,
        )
        result = model.compute_temperature(compute_power_watts=400.0, in_eclipse=True)
        expected = -60.0 + 400.0 / 40.0
        assert result == pytest.approx(expected)


class TestThermalModelValidation:
    def test_radiator_conductance_must_be_positive(self):
        with pytest.raises(Exception):
            ThermalModel(radiator_conductance_w_per_c=0.0)

    def test_negative_radiator_conductance_rejected(self):
        with pytest.raises(Exception):
            ThermalModel(radiator_conductance_w_per_c=-10.0)

    def test_radiator_area_must_be_positive(self):
        with pytest.raises(Exception):
            ThermalModel(radiator_area_m2=0.0)
