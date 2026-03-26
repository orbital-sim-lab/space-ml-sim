#!/usr/bin/env python3
"""Basic constellation setup: 100 satellites at 550km Walker-Delta.

Propagates for 1 orbit, prints positions, eclipse states, and radiation environment.
"""

from rich.console import Console
from rich.table import Table

from space_ml_sim.core.constellation import Constellation
from space_ml_sim.core.orbit import propagate
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.metrics.reliability import ReliabilityMetrics
from space_ml_sim.models.chip_profiles import TERAFAB_D3

console = Console()


def main() -> None:
    # Build constellation: 10 planes x 10 sats, 550km, 53 deg inclination
    console.print("[bold cyan]Building Walker-Delta constellation...[/]")
    constellation = Constellation.walker_delta(
        num_planes=10,
        sats_per_plane=10,
        altitude_km=550,
        inclination_deg=53,
        chip_profile=TERAFAB_D3,
    )
    console.print(f"  Satellites: {len(constellation.satellites)}")

    # Radiation environment
    rad_env = RadiationEnvironment.leo_500km()
    console.print(f"\n[bold cyan]Radiation Environment (500km LEO):[/]")
    console.print(f"  SEU rate: {rad_env.base_seu_rate:.2e} upsets/bit/sec")
    console.print(f"  TID rate: {rad_env.tid_rate_krad_per_day:.4e} krad/day")

    # Propagate for 1 orbit (~95 minutes at 550km)
    period_minutes = constellation.satellites[0].orbit_config.orbital_period_seconds / 60
    console.print(f"\n[bold cyan]Propagating for 1 orbit ({period_minutes:.1f} minutes)...[/]")

    dt = 60.0  # 1-minute steps
    num_steps = int(period_minutes)
    metrics_history = []

    for _ in range(num_steps):
        metrics = constellation.step(dt_seconds=dt)
        metrics_history.append(metrics)

    # Print final state
    final = metrics_history[-1]
    console.print(f"\n[bold green]After 1 orbit:[/]")
    console.print(f"  Active: {final['active_count']}")
    console.print(f"  Degraded: {final['degraded_count']}")
    console.print(f"  Failed: {final['failed_count']}")
    console.print(f"  Avg Temperature: {final['avg_temperature_c']}°C")
    console.print(f"  Total SEU events: {final['total_seus']}")

    # Show first 10 satellite positions
    table = Table(title="First 10 Satellites (end of orbit)")
    table.add_column("ID")
    table.add_column("Position (km)")
    table.add_column("Eclipse")
    table.add_column("State")
    table.add_column("Temp (°C)")
    table.add_column("SEUs")

    for sat in constellation.satellites[:10]:
        pos_str = f"({sat.position_km[0]:.0f}, {sat.position_km[1]:.0f}, {sat.position_km[2]:.0f})"
        table.add_row(
            sat.id,
            pos_str,
            str(sat.in_eclipse),
            sat.state.value,
            f"{sat.temperature_c:.1f}",
            str(sat.total_seu_events),
        )

    console.print(table)

    # ISL connectivity
    isl_pairs = constellation.get_isl_pairs(max_distance_km=2000)
    console.print(f"\n[bold cyan]ISL pairs within 2000km: {len(isl_pairs)}[/]")
    if isl_pairs:
        console.print(f"  Closest pair: {isl_pairs[0][0]} <-> {isl_pairs[0][1]} at {isl_pairs[0][2]} km")

    # Reliability summary
    reliability = ReliabilityMetrics.from_satellites(constellation.satellites)
    console.print(f"\n[bold cyan]Reliability:[/]")
    console.print(f"  Availability: {reliability.availability:.2%}")
    console.print(f"  Max TID: {reliability.max_tid_krad:.6f} krad")
    console.print(f"  Mean TID: {reliability.mean_tid_krad:.6f} krad")


if __name__ == "__main__":
    main()
