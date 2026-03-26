"""Tests for inter-satellite communications model."""

import math

import pytest

from space_ml_sim.environment.comms import CommsModel


SPEED_OF_LIGHT_KM_PER_MS = 299_792.458 / 1000  # km/ms


class TestLinkLatencyInRange:
    def test_in_range_distance_returns_finite_latency(self):
        model = CommsModel()
        result = model.link_latency_ms(distance_km=1000.0)
        assert math.isfinite(result)

    def test_zero_distance_returns_overhead_only(self):
        model = CommsModel(latency_overhead_ms=1.0)
        result = model.link_latency_ms(distance_km=0.0)
        assert result == pytest.approx(1.0)

    def test_latency_includes_propagation_delay(self):
        model = CommsModel(latency_overhead_ms=0.0)
        distance_km = 3000.0
        expected_propagation_ms = (distance_km / 299_792.458) * 1000
        result = model.link_latency_ms(distance_km=distance_km)
        assert result == pytest.approx(expected_propagation_ms)

    def test_latency_includes_overhead_plus_propagation(self):
        model = CommsModel(latency_overhead_ms=2.5, max_isl_range_km=5000.0)
        distance_km = 1500.0
        propagation_ms = (distance_km / 299_792.458) * 1000
        expected = propagation_ms + 2.5
        result = model.link_latency_ms(distance_km=distance_km)
        assert result == pytest.approx(expected)

    def test_latency_increases_with_distance(self):
        model = CommsModel()
        near = model.link_latency_ms(distance_km=500.0)
        far = model.link_latency_ms(distance_km=4000.0)
        assert far > near

    def test_at_exact_max_range_is_finite(self):
        model = CommsModel(max_isl_range_km=5000.0)
        result = model.link_latency_ms(distance_km=5000.0)
        assert math.isfinite(result)


class TestLinkLatencyOutOfRange:
    def test_out_of_range_returns_inf(self):
        model = CommsModel(max_isl_range_km=5000.0)
        result = model.link_latency_ms(distance_km=5001.0)
        assert result == math.inf

    def test_far_out_of_range_returns_inf(self):
        model = CommsModel(max_isl_range_km=5000.0)
        result = model.link_latency_ms(distance_km=100_000.0)
        assert result == math.inf

    def test_custom_max_range_respected(self):
        model = CommsModel(max_isl_range_km=1000.0)
        in_range = model.link_latency_ms(distance_km=999.0)
        out_of_range = model.link_latency_ms(distance_km=1001.0)
        assert math.isfinite(in_range)
        assert out_of_range == math.inf


class TestDistanceKm:
    def test_same_point_distance_is_zero(self):
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert result == 0.0

    def test_same_non_origin_point_distance_is_zero(self):
        pos = (1000.0, 2000.0, 3000.0)
        result = CommsModel.distance_km(pos, pos)
        assert result == 0.0

    def test_unit_distance_along_x_axis(self):
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert result == pytest.approx(1.0)

    def test_unit_distance_along_y_axis(self):
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert result == pytest.approx(1.0)

    def test_unit_distance_along_z_axis(self):
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        assert result == pytest.approx(1.0)

    def test_3d_euclidean_distance(self):
        # 3-4-5 right triangle extended to 3D: sqrt(3^2 + 4^2 + 0^2) = 5
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        assert result == pytest.approx(5.0)

    def test_3d_euclidean_distance_all_axes(self):
        # sqrt(1^2 + 2^2 + 2^2) = sqrt(1 + 4 + 4) = 3
        result = CommsModel.distance_km((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
        assert result == pytest.approx(3.0)

    def test_distance_is_symmetric(self):
        pos_a = (100.0, 200.0, 300.0)
        pos_b = (400.0, 500.0, 600.0)
        assert CommsModel.distance_km(pos_a, pos_b) == pytest.approx(
            CommsModel.distance_km(pos_b, pos_a)
        )

    def test_realistic_leo_separation(self):
        """Two LEO satellites separated by ~1000 km should compute correctly."""
        pos_a = (7000.0, 0.0, 0.0)
        pos_b = (6000.0, 0.0, 0.0)
        result = CommsModel.distance_km(pos_a, pos_b)
        assert result == pytest.approx(1000.0)

    def test_distance_always_non_negative(self):
        result = CommsModel.distance_km((5.0, 3.0, 1.0), (2.0, 7.0, 9.0))
        assert result >= 0.0


class TestCommsModelDefaults:
    def test_default_max_isl_range_is_5000km(self):
        model = CommsModel()
        assert model.max_isl_range_km == 5000.0

    def test_default_data_rate_is_10gbps(self):
        model = CommsModel()
        assert model.data_rate_gbps == 10.0

    def test_default_latency_overhead_is_1ms(self):
        model = CommsModel()
        assert model.latency_overhead_ms == 1.0


class TestCommsModelValidation:
    def test_max_isl_range_must_be_positive(self):
        with pytest.raises(Exception):
            CommsModel(max_isl_range_km=0.0)

    def test_data_rate_must_be_positive(self):
        with pytest.raises(Exception):
            CommsModel(data_rate_gbps=0.0)

    def test_latency_overhead_can_be_zero(self):
        model = CommsModel(latency_overhead_ms=0.0)
        assert model.latency_overhead_ms == 0.0

    def test_negative_latency_overhead_rejected(self):
        with pytest.raises(Exception):
            CommsModel(latency_overhead_ms=-1.0)
