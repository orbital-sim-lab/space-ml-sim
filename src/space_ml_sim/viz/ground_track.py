"""Ground track visualization with radiation overlay.

Shows the satellite's ground track on a world map with:
- Path color-coded by SEU rate (green=low, red=high)
- SAA boundary drawn as a dashed rectangle
- Markers at SAA entry/exit points
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from space_ml_sim.environment.timeline import RadiationTimeline


# SAA boundary (same as timeline.py)
_SAA_LAT_MIN = -50.0
_SAA_LAT_MAX = 0.0
_SAA_LON_MIN = -90.0
_SAA_LON_MAX = 40.0


def _normalize_rates(rates: list[float]) -> list[float]:
    """Normalize a list of rates to the [0, 1] range.

    Args:
        rates: Raw SEU rate values.

    Returns:
        Normalized values in [0.0, 1.0]. Returns 0.5 for all entries when
        min == max (uniform field).
    """
    min_rate = min(rates)
    max_rate = max(rates)
    if max_rate > min_rate:
        return [(r - min_rate) / (max_rate - min_rate) for r in rates]
    return [0.5] * len(rates)


def _rate_to_rgb(norm: float) -> str:
    """Map a normalized rate value (0–1) to an RGB color string.

    Color scale: green (low) -> red (high), with a fixed blue channel of 50
    to keep the palette saturated and readable against ocean/land backgrounds.

    Args:
        norm: Normalized rate in [0.0, 1.0].

    Returns:
        CSS rgb() string, e.g. ``"rgb(255, 0, 50)"``.
    """
    r = int(255 * norm)
    g = int(255 * (1.0 - norm))
    return f"rgb({r}, {g}, 50)"


def _split_segments(
    lats: list[float],
    lons: list[float],
    rates: list[float],
    texts: list[str],
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[str]],
]:
    """Split parallel lat/lon/rate/text lists at antimeridian crossings.

    When consecutive longitude values differ by more than 180 degrees the
    satellite has wrapped around the antimeridian.  Connecting those two
    points directly would draw a spurious line across the entire map, so we
    break the path into separate segments at each such jump.

    Args:
        lats: Geodetic latitudes in degrees.
        lons: Geodetic longitudes in degrees (-180 to 180).
        rates: SEU rates parallel to lats/lons.
        texts: Hover text strings parallel to lats/lons.

    Returns:
        Four lists of lists: (seg_lats, seg_lons, seg_rates, seg_texts).
        Each inner list is one contiguous segment.
    """
    if not lats:
        return [], [], [], []

    seg_lats: list[list[float]] = []
    seg_lons: list[list[float]] = []
    seg_rates: list[list[float]] = []
    seg_texts: list[list[str]] = []

    cur_lat = [lats[0]]
    cur_lon = [lons[0]]
    cur_rate = [rates[0]]
    cur_text = [texts[0]]

    for i in range(1, len(lats)):
        if abs(lons[i] - lons[i - 1]) > 180:
            seg_lats.append(cur_lat)
            seg_lons.append(cur_lon)
            seg_rates.append(cur_rate)
            seg_texts.append(cur_text)
            cur_lat = []
            cur_lon = []
            cur_rate = []
            cur_text = []

        cur_lat.append(lats[i])
        cur_lon.append(lons[i])
        cur_rate.append(rates[i])
        cur_text.append(texts[i])

    if cur_lat:
        seg_lats.append(cur_lat)
        seg_lons.append(cur_lon)
        seg_rates.append(cur_rate)
        seg_texts.append(cur_text)

    return seg_lats, seg_lons, seg_rates, seg_texts


def plot_ground_track(
    timeline: RadiationTimeline,
    title: str = "Satellite Ground Track with Radiation Overlay",
    save_path: str | None = None,
    show_saa: bool = True,
) -> "go.Figure":
    """Plot satellite ground track on world map with radiation color coding.

    Produces a Plotly Figure with a geographic projection (Natural Earth)
    showing:

    - The satellite ground track split at antimeridian crossings.
    - Each point colored from green (low SEU rate) to red (high SEU rate).
    - An optional dashed red rectangle marking the SAA bounding box.

    Args:
        timeline: RadiationTimeline from radiation_timeline().
        title: Plot title prefix shown in the figure header.
        save_path: Optional filesystem path to save the figure as HTML.
        show_saa: Whether to draw the SAA bounding-box rectangle.

    Returns:
        Plotly Figure with geo scatter plot.
    """
    import plotly.graph_objects as go  # noqa: PLC0415

    points = timeline.points

    lats = [p.latitude_deg for p in points]
    lons = [p.longitude_deg for p in points]
    rates = [p.seu_rate for p in points]
    times_h = [p.time_seconds / 3600.0 for p in points]
    in_saa_flags = [p.in_saa for p in points]

    # Build hover text labels for every point.
    hover_texts = [
        f"t={times_h[i]:.1f}h, SEU={rates[i]:.2e}" + (" [SAA]" if in_saa_flags[i] else "")
        for i in range(len(points))
    ]

    # Split path at antimeridian crossings to avoid map-crossing artifacts.
    seg_lats, seg_lons, seg_rates, seg_texts = _split_segments(lats, lons, rates, hover_texts)

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Ground track: one Scattergeo trace per contiguous segment.
    # ------------------------------------------------------------------
    global_min = min(rates) if rates else 0.0
    global_max = max(rates) if rates else 1.0

    for seg_idx, (slat, slon, srate, stext) in enumerate(
        zip(seg_lats, seg_lons, seg_rates, seg_texts)
    ):
        # Re-normalize per-segment rates against the global scale so that
        # colors are consistent across all segments.
        if global_max > global_min:
            seg_norm = [(r - global_min) / (global_max - global_min) for r in srate]
        else:
            seg_norm = [0.5] * len(srate)

        colors = [_rate_to_rgb(n) for n in seg_norm]

        fig.add_trace(
            go.Scattergeo(
                lat=slat,
                lon=slon,
                mode="markers+lines",
                marker=dict(size=4, color=colors),
                line=dict(width=1, color="rgba(100,100,100,0.3)"),
                text=stext,
                hoverinfo="text",
                showlegend=(seg_idx == 0),
                name="Ground Track",
            )
        )

    # ------------------------------------------------------------------
    # SAA boundary rectangle (optional).
    # ------------------------------------------------------------------
    if show_saa:
        saa_lats = [
            _SAA_LAT_MIN,
            _SAA_LAT_MIN,
            _SAA_LAT_MAX,
            _SAA_LAT_MAX,
            _SAA_LAT_MIN,
        ]
        saa_lons = [
            _SAA_LON_MIN,
            _SAA_LON_MAX,
            _SAA_LON_MAX,
            _SAA_LON_MIN,
            _SAA_LON_MIN,
        ]
        fig.add_trace(
            go.Scattergeo(
                lat=saa_lats,
                lon=saa_lons,
                mode="lines",
                line=dict(width=2, color="red", dash="dash"),
                name="SAA Boundary",
                hoverinfo="name",
            )
        )

    # ------------------------------------------------------------------
    # Geo layout: Natural Earth projection with coastlines / land.
    # ------------------------------------------------------------------
    fig.update_geos(
        showcountries=True,
        countrycolor="lightgray",
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showocean=True,
        oceancolor="rgb(230, 240, 255)",
        projection_type="natural earth",
    )

    fig.update_layout(
        title=(
            f"{title} ({timeline.saa_crossing_count} SAA crossings, "
            f"{timeline.total_duration_hours:.0f}h)"
        ),
        template="plotly_white",
        height=600,
    )

    if save_path is not None:
        fig.write_html(save_path)

    return fig
