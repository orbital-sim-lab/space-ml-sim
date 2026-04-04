"""TDD tests for ground track visualization (RED phase).

Tests are written before the implementation in ground_track.py.
Expected to FAIL until the implementation is created.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# ISS TLE used across all tests
# ---------------------------------------------------------------------------

ISS_L1 = "1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075"


# ---------------------------------------------------------------------------
# Shared fixture: 2-hour timeline (short and fast for test runs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_timeline():
    """Generate a 2-hour ISS timeline at 120-second steps (61 points)."""
    from space_ml_sim.environment.timeline import radiation_timeline

    return radiation_timeline(ISS_L1, ISS_L2, duration_hours=2.0, step_seconds=120.0)


# ---------------------------------------------------------------------------
# Tests for plot_ground_track
# ---------------------------------------------------------------------------


class TestPlotGroundTrackReturnType:
    """plot_ground_track must return a valid Plotly Figure."""

    def test_returns_plotly_figure(self, short_timeline):
        """Return type must be plotly.graph_objects.Figure."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        assert isinstance(fig, go.Figure)

    def test_figure_is_not_none(self, short_timeline):
        """Function must return a non-None value."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        assert fig is not None


class TestPlotGroundTrackTraces:
    """Figure must contain the expected trace structure."""

    def test_figure_has_at_least_one_trace(self, short_timeline):
        """Figure must have at least 1 trace for the ground track segments."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        assert len(fig.data) >= 1

    def test_figure_has_saa_boundary_trace_when_show_saa_true(self, short_timeline):
        """When show_saa=True, figure must contain a trace named 'SAA Boundary'."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, show_saa=True)

        trace_names = [trace.name for trace in fig.data]
        assert "SAA Boundary" in trace_names

    def test_figure_has_no_saa_trace_when_show_saa_false(self, short_timeline):
        """When show_saa=False, figure must NOT contain 'SAA Boundary' trace."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, show_saa=False)

        trace_names = [trace.name for trace in fig.data]
        assert "SAA Boundary" not in trace_names

    def test_figure_has_ground_track_trace(self, short_timeline):
        """At least one trace must be named 'Ground Track'."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        trace_names = [trace.name for trace in fig.data]
        assert "Ground Track" in trace_names

    def test_saa_boundary_trace_uses_dashed_line(self, short_timeline):
        """SAA boundary trace must use a dashed line style."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, show_saa=True)

        saa_traces = [t for t in fig.data if t.name == "SAA Boundary"]
        assert len(saa_traces) == 1
        assert saa_traces[0].line.dash == "dash"

    def test_saa_boundary_trace_is_red(self, short_timeline):
        """SAA boundary trace line color must be red."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, show_saa=True)

        saa_traces = [t for t in fig.data if t.name == "SAA Boundary"]
        assert len(saa_traces) == 1
        assert saa_traces[0].line.color == "red"


class TestPlotGroundTrackTitle:
    """Figure title must contain required metadata."""

    def test_figure_title_contains_saa_crossing_count(self, short_timeline):
        """Figure title must embed the SAA crossing count from the timeline."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        title_text = fig.layout.title.text
        assert str(short_timeline.saa_crossing_count) in title_text

    def test_figure_title_uses_custom_title_string(self, short_timeline):
        """Custom title argument must appear in the figure title."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        custom = "My Custom Ground Track"
        fig = plot_ground_track(short_timeline, title=custom)

        assert custom in fig.layout.title.text

    def test_figure_title_contains_duration(self, short_timeline):
        """Figure title must include the total duration in hours."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        # Duration is 2.0 hours - should appear as "2h" or "2.0h" etc.
        title_text = fig.layout.title.text
        assert "2" in title_text

    def test_default_title_applied(self, short_timeline):
        """Default title string must appear in layout when no title given."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        assert "Satellite Ground Track" in fig.layout.title.text


class TestPlotGroundTrackSavePath:
    """save_path parameter must write a non-empty HTML file."""

    def test_save_path_writes_html_file(self, short_timeline, tmp_path):
        """When save_path is given, an HTML file must exist after the call."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        out = str(tmp_path / "ground_track.html")
        plot_ground_track(short_timeline, save_path=out)

        assert os.path.exists(out)

    def test_saved_html_file_is_non_empty(self, short_timeline, tmp_path):
        """The saved HTML file must have content (not 0 bytes)."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        out = str(tmp_path / "ground_track.html")
        plot_ground_track(short_timeline, save_path=out)

        assert os.path.getsize(out) > 0

    def test_no_save_path_does_not_raise(self, short_timeline):
        """Calling without save_path must succeed without raising."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, save_path=None)

        assert fig is not None

    def test_no_file_written_without_save_path(self, short_timeline, tmp_path):
        """No HTML file is created when save_path is None."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        plot_ground_track(short_timeline, save_path=None)

        assert list(tmp_path.iterdir()) == []


class TestPlotGroundTrackSingleOrbit:
    """Works correctly with a short single-orbit timeline."""

    def test_works_with_one_orbit_timeline(self):
        """plot_ground_track must succeed for a ~92-minute (1 orbit) timeline."""
        from space_ml_sim.environment.timeline import radiation_timeline
        from space_ml_sim.viz.ground_track import plot_ground_track

        tl = radiation_timeline(ISS_L1, ISS_L2, duration_hours=1.5, step_seconds=120.0)
        fig = plot_ground_track(tl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_one_orbit_has_ground_track_trace(self):
        """Single-orbit figure must still contain a 'Ground Track' trace."""
        from space_ml_sim.environment.timeline import radiation_timeline
        from space_ml_sim.viz.ground_track import plot_ground_track

        tl = radiation_timeline(ISS_L1, ISS_L2, duration_hours=1.5, step_seconds=120.0)
        fig = plot_ground_track(tl)

        trace_names = [trace.name for trace in fig.data]
        assert "Ground Track" in trace_names


class TestPlotGroundTrackGeoLayout:
    """The figure must use geo (world map) projection, not cartesian axes."""

    def test_figure_uses_scattergeo_trace_type(self, short_timeline):
        """At least one trace must be a Scattergeo (not Scatter)."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scattergeo" in trace_types

    def test_saa_boundary_is_scattergeo(self, short_timeline):
        """The SAA boundary trace must also be a Scattergeo type."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline, show_saa=True)

        saa_traces = [t for t in fig.data if t.name == "SAA Boundary"]
        assert len(saa_traces) == 1
        assert type(saa_traces[0]).__name__ == "Scattergeo"

    def test_figure_height_is_set(self, short_timeline):
        """Figure layout height must be explicitly set (>= 400 px)."""
        from space_ml_sim.viz.ground_track import plot_ground_track

        fig = plot_ground_track(short_timeline)

        assert fig.layout.height is not None
        assert fig.layout.height >= 400


class TestPlotGroundTrackExport:
    """plot_ground_track must be importable from the viz package."""

    def test_importable_from_viz_package(self):
        """plot_ground_track is exported from space_ml_sim.viz."""
        from space_ml_sim.viz import plot_ground_track  # noqa: F401

    def test_viz_all_contains_plot_ground_track(self):
        """__all__ in space_ml_sim.viz must include 'plot_ground_track'."""
        import space_ml_sim.viz as viz_pkg

        assert "plot_ground_track" in viz_pkg.__all__
