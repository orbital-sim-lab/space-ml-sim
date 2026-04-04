"""Radiation exposure timeline for orbital trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimelinePoint:
    """Single point in a radiation exposure timeline."""

    time_seconds: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    seu_rate: float  # upsets/bit/sec at this point
    tid_rate_krad_day: float  # krad/day at this point
    in_saa: bool  # whether inside South Atlantic Anomaly


@dataclass(frozen=True)
class RadiationTimeline:
    """Complete radiation exposure timeline for an orbit."""

    points: tuple[TimelinePoint, ...]
    total_duration_hours: float
    saa_crossing_count: int
    saa_total_seconds: float  # total time spent in SAA
    peak_seu_rate: float
    mean_seu_rate: float


# SAA bounding box (approximate)
_SAA_LAT_MIN = -50.0
_SAA_LAT_MAX = 0.0
_SAA_LON_MIN = -90.0
_SAA_LON_MAX = 40.0


def _is_in_saa(lat_deg: float, lon_deg: float) -> bool:
    """Check if a lat/lon point is inside the SAA bounding box.

    Uses the approximate rectangular bounding box for the South Atlantic
    Anomaly: lat [-50, 0] deg, lon [-90, 40] deg.

    Args:
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Geodetic longitude in degrees.

    Returns:
        True if the point is inside the SAA box.
    """
    return _SAA_LAT_MIN <= lat_deg <= _SAA_LAT_MAX and _SAA_LON_MIN <= lon_deg <= _SAA_LON_MAX


def _eci_to_geodetic(
    x_km: float,
    y_km: float,
    z_km: float,
    gmst_rad: float,
) -> tuple[float, float, float]:
    """Convert ECI position to geodetic lat/lon/alt (simplified spherical Earth).

    Args:
        x_km: ECI X position in km.
        y_km: ECI Y position in km.
        z_km: ECI Z position in km.
        gmst_rad: Greenwich Mean Sidereal Time in radians.

    Returns:
        (latitude_deg, longitude_deg, altitude_km)
    """
    R_EARTH = 6371.0

    # Rotate from ECI to ECEF using GMST
    cos_g = math.cos(gmst_rad)
    sin_g = math.sin(gmst_rad)
    x_ecef = cos_g * x_km + sin_g * y_km
    y_ecef = -sin_g * x_km + cos_g * y_km
    z_ecef = z_km

    r = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
    lat = math.degrees(math.asin(z_ecef / r)) if r > 0 else 0.0
    lon = math.degrees(math.atan2(y_ecef, x_ecef))
    alt = r - R_EARTH

    return lat, lon, alt


def radiation_timeline(
    line1: str,
    line2: str,
    duration_hours: float = 24.0,
    step_seconds: float = 60.0,
    shielding_mm_al: float = 2.0,
) -> RadiationTimeline:
    """Generate radiation exposure timeline from TLE.

    Propagates the orbit using SGP4, computes radiation rates at each point
    including SAA crossing detection.

    Args:
        line1: TLE line 1.
        line2: TLE line 2.
        duration_hours: Duration to simulate in hours.
        step_seconds: Time step in seconds.
        shielding_mm_al: Aluminum shielding thickness in mm.

    Returns:
        RadiationTimeline with time-series data.

    Raises:
        ValueError: If TLE lines are malformed or SGP4 propagation fails.
    """
    from space_ml_sim.core.tle import propagate_sgp4, _build_satrec  # noqa: PLC0415
    from space_ml_sim.environment.radiation import RadiationEnvironment  # noqa: PLC0415

    # Validate TLE and extract inclination before the loop.
    sat = _build_satrec(line1, line2)
    inc_deg = math.degrees(sat.inclo)

    num_steps = int((duration_hours * 3600.0) / step_seconds)

    # Earth's rotation rate (rad/s) for ECI -> ECEF conversion.
    _EARTH_ROT_RAD_S = 7.2921159e-5

    points: list[TimelinePoint] = []
    saa_crossings = 0
    saa_seconds = 0.0
    prev_in_saa = False

    for i in range(num_steps + 1):
        t_min = i * step_seconds / 60.0  # minutes from epoch

        pos = propagate_sgp4(line1, line2, t_min)

        # GMST approximation: simple linear rotation from epoch.
        gmst = _EARTH_ROT_RAD_S * i * step_seconds

        lat, lon, alt = _eci_to_geodetic(pos[0], pos[1], pos[2], gmst)

        # Clamp altitude: keep it physically above 100 km for the model.
        safe_alt = max(alt, 100.0)

        rad_env = RadiationEnvironment(
            altitude_km=safe_alt,
            inclination_deg=inc_deg,
            shielding_mm_al=shielding_mm_al,
        )

        in_saa = _is_in_saa(lat, lon)

        # SAA enhances SEU ~5x at the actual position (the orbit-averaged
        # factor in RadiationEnvironment already accounts for time-average
        # SAA exposure; here we override to reflect the instantaneous position).
        saa_mult = 5.0 if in_saa else 1.0
        seu_rate = rad_env.base_seu_rate * saa_mult
        # TID scales sub-linearly with the same factor.
        tid_rate = rad_env.tid_rate_krad_per_day * (saa_mult**0.5)

        points.append(
            TimelinePoint(
                time_seconds=float(i * step_seconds),
                latitude_deg=round(lat, 4),
                longitude_deg=round(lon, 4),
                altitude_km=round(alt, 2),
                seu_rate=seu_rate,
                tid_rate_krad_day=tid_rate,
                in_saa=in_saa,
            )
        )

        if in_saa and not prev_in_saa:
            saa_crossings += 1
        if in_saa:
            saa_seconds += step_seconds

        prev_in_saa = in_saa

    seu_rates = [p.seu_rate for p in points]

    return RadiationTimeline(
        points=tuple(points),
        total_duration_hours=duration_hours,
        saa_crossing_count=saa_crossings,
        saa_total_seconds=saa_seconds,
        peak_seu_rate=max(seu_rates),
        mean_seu_rate=sum(seu_rates) / len(seu_rates),
    )


def plot_radiation_timeline(
    timeline: RadiationTimeline,
    title: str = "Radiation Exposure Timeline",
    save_path: str | None = None,
):
    """Plot radiation timeline with SAA crossings highlighted.

    Produces a Plotly Figure with:
    - SEU rate over time (primary y-axis)
    - Altitude on secondary y-axis (dotted line)
    - SAA crossing intervals highlighted as red bands

    Args:
        timeline: RadiationTimeline to visualise.
        title: Figure title prefix.
        save_path: Optional filesystem path to save the figure as HTML.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    times_h = [p.time_seconds / 3600.0 for p in timeline.points]
    seu_rates = [p.seu_rate for p in timeline.points]
    altitudes = [p.altitude_km for p in timeline.points]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=times_h,
            y=seu_rates,
            name="SEU Rate",
            line=dict(color="#636EFA"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=times_h,
            y=altitudes,
            name="Altitude (km)",
            line=dict(color="#00CC96", dash="dot"),
        ),
        secondary_y=True,
    )

    # Highlight SAA crossing bands.
    in_saa_start: float | None = None
    for i, p in enumerate(timeline.points):
        if p.in_saa and in_saa_start is None:
            in_saa_start = times_h[i]
        elif not p.in_saa and in_saa_start is not None:
            annotation = "SAA" if i < 10 else None
            fig.add_vrect(
                x0=in_saa_start,
                x1=times_h[i],
                fillcolor="red",
                opacity=0.1,
                line_width=0,
                annotation_text=annotation,
            )
            in_saa_start = None

    # Close any open SAA band that extends to the end of the timeline.
    if in_saa_start is not None and times_h:
        fig.add_vrect(
            x0=in_saa_start,
            x1=times_h[-1],
            fillcolor="red",
            opacity=0.1,
            line_width=0,
        )

    full_title = f"{title} ({timeline.saa_crossing_count} SAA crossings)"

    fig.update_layout(
        title=full_title,
        xaxis_title="Time (hours)",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="SEU Rate (upsets/bit/sec)", secondary_y=False)
    fig.update_yaxes(title_text="Altitude (km)", secondary_y=True)

    if save_path:
        fig.write_html(save_path)

    return fig
