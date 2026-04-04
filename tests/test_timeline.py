"""TDD tests for radiation exposure timeline (RED phase).

Tests are written before the implementation in timeline.py.
Expected to FAIL until the implementation is created.
"""

from __future__ import annotations

import math
import pytest

# ---------------------------------------------------------------------------
# ISS TLE used across all tests
# ---------------------------------------------------------------------------
ISS_LINE1 = "1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075"


# ---------------------------------------------------------------------------
# Tests for _is_in_saa
# ---------------------------------------------------------------------------


class TestIsInSaa:
    """Unit tests for the SAA bounding-box classifier."""

    def test_centre_of_saa_is_inside(self):
        """A point well within the SAA box must return True."""
        from space_ml_sim.environment.timeline import _is_in_saa

        # Centre: lat ~-25, lon ~-35 (clearly inside -50..0, -90..40)
        assert _is_in_saa(-25.0, -35.0) is True

    def test_north_pole_is_outside(self):
        """90 deg latitude is above the SAA box (lat max 0)."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(90.0, 0.0) is False

    def test_positive_latitude_outside(self):
        """Any positive latitude is above the SAA box."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(10.0, -30.0) is False

    def test_too_far_south_is_outside(self):
        """Latitude below -50 deg is outside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-60.0, -30.0) is False

    def test_lon_too_far_east_is_outside(self):
        """Longitude > 40 deg is outside the SAA box."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-25.0, 50.0) is False

    def test_lon_too_far_west_is_outside(self):
        """Longitude < -90 deg is outside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-25.0, -100.0) is False

    def test_boundary_lat_min_inclusive(self):
        """lat == -50 (lower boundary) is inside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-50.0, 0.0) is True

    def test_boundary_lat_max_inclusive(self):
        """lat == 0 (upper boundary) is inside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(0.0, 0.0) is True

    def test_boundary_lon_min_inclusive(self):
        """lon == -90 (left boundary) is inside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-25.0, -90.0) is True

    def test_boundary_lon_max_inclusive(self):
        """lon == 40 (right boundary) is inside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(-25.0, 40.0) is True

    def test_equator_prime_meridian_is_inside(self):
        """lat=0, lon=0 sits on the corner of the SAA box and is inside."""
        from space_ml_sim.environment.timeline import _is_in_saa

        assert _is_in_saa(0.0, 0.0) is True


# ---------------------------------------------------------------------------
# Tests for _eci_to_geodetic
# ---------------------------------------------------------------------------


class TestEciToGeodetic:
    """Sanity checks for the ECI-to-geodetic conversion."""

    def test_equatorial_point_has_zero_latitude(self):
        """A point on the equatorial plane (z=0) must have latitude ~0."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0  # 400 km altitude
        lat, lon, alt = _eci_to_geodetic(R, 0.0, 0.0, gmst_rad=0.0)
        assert abs(lat) < 1e-6, f"Expected lat~0, got {lat}"

    def test_equatorial_point_altitude_is_positive(self):
        """A point above Earth's surface must have positive altitude."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0
        lat, lon, alt = _eci_to_geodetic(R, 0.0, 0.0, gmst_rad=0.0)
        assert alt > 0.0, f"Expected alt>0, got {alt}"

    def test_equatorial_altitude_close_to_input(self):
        """Altitude should match what was put in (400 km)."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0
        lat, lon, alt = _eci_to_geodetic(R, 0.0, 0.0, gmst_rad=0.0)
        assert abs(alt - 400.0) < 1.0, f"Expected alt~400, got {alt}"

    def test_north_pole_point_has_positive_latitude(self):
        """A point directly above the north pole has lat = +90."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0
        lat, lon, alt = _eci_to_geodetic(0.0, 0.0, R, gmst_rad=0.0)
        assert abs(lat - 90.0) < 1e-4, f"Expected lat~90, got {lat}"

    def test_south_pole_point_has_negative_latitude(self):
        """A point below the south pole has lat = -90."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0
        lat, lon, alt = _eci_to_geodetic(0.0, 0.0, -R, gmst_rad=0.0)
        assert abs(lat + 90.0) < 1e-4, f"Expected lat~-90, got {lat}"

    def test_gmst_rotation_changes_longitude(self):
        """Applying a non-zero GMST shifts longitude from what zero GMST gives."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 400.0
        _, lon0, _ = _eci_to_geodetic(R, 0.0, 0.0, gmst_rad=0.0)
        _, lon1, _ = _eci_to_geodetic(R, 0.0, 0.0, gmst_rad=math.pi / 4)
        assert abs(lon0 - lon1) > 1.0, "GMST rotation should change longitude"

    def test_zero_radius_returns_zero_lat(self):
        """If r == 0 the function should not raise and returns lat=0."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        lat, lon, alt = _eci_to_geodetic(0.0, 0.0, 0.0, gmst_rad=0.0)
        assert lat == 0.0

    def test_longitude_within_range(self):
        """Longitude must always be within [-180, 180]."""
        from space_ml_sim.environment.timeline import _eci_to_geodetic

        R = 6371.0 + 500.0
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x = R * math.cos(rad)
            y = R * math.sin(rad)
            _, lon, _ = _eci_to_geodetic(x, y, 0.0, gmst_rad=0.0)
            assert -180.0 <= lon <= 180.0, f"lon {lon} out of range for angle {angle}"


# ---------------------------------------------------------------------------
# Tests for radiation_timeline
# ---------------------------------------------------------------------------


class TestRadiationTimeline:
    """Integration tests for the full timeline generator."""

    def test_returns_radiation_timeline_type(self):
        """Function must return a RadiationTimeline instance."""
        from space_ml_sim.environment.timeline import radiation_timeline, RadiationTimeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        assert isinstance(tl, RadiationTimeline)

    def test_correct_number_of_points(self):
        """For 1h / 60s step, expect 61 points (0..60 inclusive)."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        expected = int(1.0 * 3600 / 60.0) + 1  # 61
        assert len(tl.points) == expected

    def test_duration_stored_correctly(self):
        """total_duration_hours must match the requested value."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=2.0, step_seconds=60.0)
        assert tl.total_duration_hours == 2.0

    def test_iss_24h_has_at_least_one_saa_crossing(self):
        """ISS at 51.6 deg incl. crosses the SAA multiple times per day."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=24.0, step_seconds=120.0)
        assert tl.saa_crossing_count >= 1, (
            f"Expected >=1 SAA crossings in 24h, got {tl.saa_crossing_count}"
        )

    def test_seu_rate_always_positive(self):
        """No timeline point should have a zero or negative SEU rate."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        for p in tl.points:
            assert p.seu_rate > 0.0, f"SEU rate must be positive, got {p.seu_rate}"

    def test_peak_seu_rate_gte_mean(self):
        """Peak SEU rate must be >= mean SEU rate."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        assert tl.peak_seu_rate >= tl.mean_seu_rate

    def test_saa_points_have_higher_seu_rate_than_non_saa(self):
        """Points inside SAA should have strictly higher SEU rate than outside."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=24.0, step_seconds=120.0)
        saa_rates = [p.seu_rate for p in tl.points if p.in_saa]
        non_saa_rates = [p.seu_rate for p in tl.points if not p.in_saa]

        if saa_rates and non_saa_rates:
            assert min(saa_rates) > max(non_saa_rates) or (
                # Weaker check: mean SAA rate > mean non-SAA rate
                sum(saa_rates) / len(saa_rates) > sum(non_saa_rates) / len(non_saa_rates)
            ), "SAA points should have higher SEU rates than non-SAA points"

    def test_time_series_is_monotonic(self):
        """time_seconds must strictly increase across timeline points."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        times = [p.time_seconds for p in tl.points]
        assert times == sorted(times), "time_seconds must be monotonically increasing"

    def test_first_point_time_is_zero(self):
        """First point must be at t=0."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        assert tl.points[0].time_seconds == 0.0

    def test_last_point_time_matches_duration(self):
        """Last point time should equal duration_hours * 3600."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        expected_end = 1.0 * 3600.0
        assert tl.points[-1].time_seconds == expected_end

    def test_altitudes_are_physically_plausible(self):
        """ISS altitude should stay in a reasonable LEO range (200-500 km)."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        for p in tl.points:
            assert 100.0 <= p.altitude_km <= 600.0, (
                f"Altitude {p.altitude_km} km is implausible for ISS"
            )

    def test_saa_total_seconds_consistent_with_crossing_count(self):
        """saa_total_seconds must be >= 0 and consistent with crossing count."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=24.0, step_seconds=120.0)
        assert tl.saa_total_seconds >= 0.0
        if tl.saa_crossing_count == 0:
            assert tl.saa_total_seconds == 0.0

    def test_tid_rate_positive_at_every_point(self):
        """TID rate must be positive at every timeline point."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        for p in tl.points:
            assert p.tid_rate_krad_day > 0.0

    def test_custom_shielding_reduces_seu_rate(self):
        """More shielding should reduce SEU rates throughout the timeline."""
        from space_ml_sim.environment.timeline import radiation_timeline

        thin = radiation_timeline(
            ISS_LINE1,
            ISS_LINE2,
            duration_hours=1.0,
            step_seconds=60.0,
            shielding_mm_al=1.0,
        )
        thick = radiation_timeline(
            ISS_LINE1,
            ISS_LINE2,
            duration_hours=1.0,
            step_seconds=60.0,
            shielding_mm_al=10.0,
        )
        assert thick.peak_seu_rate < thin.peak_seu_rate

    def test_invalid_tle_raises_value_error(self):
        """Malformed TLE lines must raise ValueError."""
        from space_ml_sim.environment.timeline import radiation_timeline

        with pytest.raises(ValueError):
            radiation_timeline("bad line1", "bad line2", duration_hours=1.0)

    def test_zero_duration_raises_or_returns_minimal_points(self):
        """Duration of 0 should either raise or return at minimum the epoch point."""
        from space_ml_sim.environment.timeline import radiation_timeline

        try:
            tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=0.0, step_seconds=60.0)
            # If it doesn't raise, it should return at least 1 point (the epoch)
            assert len(tl.points) >= 1
        except (ValueError, ZeroDivisionError):
            pass  # Raising is also acceptable

    def test_points_is_a_tuple(self):
        """The points attribute should be immutable (tuple, not list)."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=60.0)
        assert isinstance(tl.points, tuple)


# ---------------------------------------------------------------------------
# Tests for plot_radiation_timeline
# ---------------------------------------------------------------------------


class TestPlotRadiationTimeline:
    """Tests that the plotting function produces valid Plotly figures."""

    @pytest.fixture
    def sample_timeline(self):
        from space_ml_sim.environment.timeline import radiation_timeline

        return radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=2.0, step_seconds=120.0)

    def test_returns_plotly_figure(self, sample_timeline):
        """plot_radiation_timeline must return a plotly Figure object."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline
        import plotly.graph_objects as go

        fig = plot_radiation_timeline(sample_timeline)
        assert isinstance(fig, go.Figure)

    def test_figure_has_two_traces(self, sample_timeline):
        """Figure must contain at least 2 traces: SEU rate and altitude."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        fig = plot_radiation_timeline(sample_timeline)
        assert len(fig.data) >= 2

    def test_figure_has_title(self, sample_timeline):
        """Figure layout must include a title."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        fig = plot_radiation_timeline(sample_timeline, title="Test Plot")
        assert fig.layout.title.text is not None
        assert len(fig.layout.title.text) > 0

    def test_save_path_writes_html(self, sample_timeline, tmp_path):
        """When save_path is given, an HTML file must be written."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        out = str(tmp_path / "timeline.html")
        plot_radiation_timeline(sample_timeline, save_path=out)
        import os

        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_custom_title_appears_in_figure(self, sample_timeline):
        """Custom title string must appear in the figure title."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        fig = plot_radiation_timeline(sample_timeline, title="My Custom Title")
        assert "My Custom Title" in fig.layout.title.text

    def test_saa_crossing_count_in_title(self, sample_timeline):
        """Figure title should mention SAA crossing count."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        fig = plot_radiation_timeline(sample_timeline)
        assert "SAA" in fig.layout.title.text

    def test_no_save_path_does_not_raise(self, sample_timeline):
        """Calling without save_path should succeed without raising."""
        from space_ml_sim.environment.timeline import plot_radiation_timeline

        fig = plot_radiation_timeline(sample_timeline, save_path=None)
        assert fig is not None


# ---------------------------------------------------------------------------
# Tests for TimelinePoint and RadiationTimeline dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Immutability and structural tests for the frozen dataclasses."""

    def test_timeline_point_is_frozen(self):
        """TimelinePoint must be immutable (frozen dataclass)."""
        from space_ml_sim.environment.timeline import TimelinePoint

        pt = TimelinePoint(
            time_seconds=0.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_km=400.0,
            seu_rate=1e-12,
            tid_rate_krad_day=1e-5,
            in_saa=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            pt.seu_rate = 999.0  # type: ignore[misc]

    def test_radiation_timeline_is_frozen(self):
        """RadiationTimeline must be immutable."""
        from space_ml_sim.environment.timeline import radiation_timeline

        tl = radiation_timeline(ISS_LINE1, ISS_LINE2, duration_hours=1.0, step_seconds=300.0)
        with pytest.raises((AttributeError, TypeError)):
            tl.peak_seu_rate = 999.0  # type: ignore[misc]

    def test_timeline_point_fields_accessible(self):
        """All documented fields must be accessible on TimelinePoint."""
        from space_ml_sim.environment.timeline import TimelinePoint

        pt = TimelinePoint(
            time_seconds=60.0,
            latitude_deg=-20.0,
            longitude_deg=-45.0,
            altitude_km=415.0,
            seu_rate=3e-12,
            tid_rate_krad_day=2e-5,
            in_saa=True,
        )
        assert pt.time_seconds == 60.0
        assert pt.latitude_deg == -20.0
        assert pt.longitude_deg == -45.0
        assert pt.altitude_km == 415.0
        assert pt.seu_rate == 3e-12
        assert pt.tid_rate_krad_day == 2e-5
        assert pt.in_saa is True
