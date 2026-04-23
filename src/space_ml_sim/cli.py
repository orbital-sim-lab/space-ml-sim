"""Command-line interface for space-ml-sim.

Quick analysis without writing Python:
  space-ml-sim chips
  space-ml-sim trade-study --orbit 550/53 --chip RAD5500 --tmr full_tmr
  space-ml-sim report --type ecss --orbit 550/97.6 --chip TRILLIUM_V6E --output report.html
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _parse_orbit(orbit_str: str):
    """Parse 'altitude/inclination' string into OrbitConfig."""
    from space_ml_sim.core.orbit import OrbitConfig

    parts = orbit_str.split("/")
    if len(parts) != 2:
        raise click.BadParameter(f"Orbit must be 'altitude/inclination', got '{orbit_str}'")
    return OrbitConfig(
        altitude_km=float(parts[0]),
        inclination_deg=float(parts[1]),
        raan_deg=0.0,
        true_anomaly_deg=0.0,
    )


def _resolve_chip(name: str):
    """Resolve chip name to ChipProfile."""
    from space_ml_sim.models.chip_profiles import ALL_CHIPS

    for chip in ALL_CHIPS:
        if name.upper() in chip.name.upper() or name.upper() == chip.name.split()[0].upper():
            return chip
    # Try exact match on constant names
    import space_ml_sim.models.chip_profiles as cp

    obj = getattr(cp, name.upper(), None)
    if obj is not None:
        return obj
    raise click.BadParameter(
        f"Unknown chip '{name}'. Use 'space-ml-sim chips' to list available chips."
    )


@click.group()
@click.version_option(package_name="space-ml-sim")
def cli():
    """space-ml-sim: Simulate AI inference on orbital satellite constellations."""


@cli.command()
@click.option("--name", default=None, help="Show details for a specific chip")
def chips(name: str | None):
    """List available hardware chip profiles."""
    from space_ml_sim.models.chip_profiles import ALL_CHIPS

    if name:
        chip = _resolve_chip(name)
        table = Table(title=f"Chip Profile: {chip.name}")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Process Node", f"{chip.node_nm} nm")
        table.add_row("TDP", f"{chip.tdp_watts} W")
        table.add_row("Max Temp", f"{chip.max_temp_c} °C")
        table.add_row("SEU Cross-Section", f"{chip.seu_cross_section_cm2:.2e} cm²/bit")
        table.add_row("TID Tolerance", f"{chip.tid_tolerance_krad} krad(Si)")
        table.add_row("Compute", f"{chip.compute_tops} TOPS")
        table.add_row("Notes", chip.notes)
        console.print(table)
        return

    table = Table(title="Available Chip Profiles")
    table.add_column("Name")
    table.add_column("Node")
    table.add_column("TOPS")
    table.add_column("TDP (W)")
    table.add_column("SEU xsec (cm²)")
    table.add_column("TID (krad)")

    for chip in ALL_CHIPS:
        table.add_row(
            chip.name,
            f"{chip.node_nm}nm",
            f"{chip.compute_tops}",
            f"{chip.tdp_watts}",
            f"{chip.seu_cross_section_cm2:.1e}",
            f"{chip.tid_tolerance_krad}",
        )
    console.print(table)


@cli.command("trade-study")
@click.option("--orbit", required=True, help="Orbit as 'altitude_km/inclination_deg' (e.g. 550/53)")
@click.option("--chip", "chip_names", multiple=True, required=True, help="Chip name(s)")
@click.option(
    "--tmr", default="none", help="TMR strategy: none, full_tmr, selective_tmr, checkpoint_rollback"
)
@click.option("--shielding", default=2.0, type=float, help="Shielding in mm Al (default: 2.0)")
@click.option("--mission-years", default=5.0, type=float, help="Mission duration in years")
def trade_study(
    orbit: str, chip_names: tuple[str, ...], tmr: str, shielding: float, mission_years: float
):
    """Run a mission trade study comparing configurations."""
    from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

    orbit_config = _parse_orbit(orbit)
    configs = []
    for name in chip_names:
        chip = _resolve_chip(name)
        configs.append(
            MissionConfig(
                name=chip.name,
                orbit=orbit_config,
                chip=chip,
                tmr_strategy=tmr,
                shielding_mm_al=shielding,
                mission_years=mission_years,
            )
        )

    study = TradeStudy(configs=configs)
    results = study.run()

    table = Table(title=f"Trade Study: {orbit} orbit, {shielding}mm Al, {tmr}")
    table.add_column("Configuration")
    table.add_column("SEU/day")
    table.add_column("SEU/orbit")
    table.add_column("TID (krad)")
    table.add_column("Power (W)")
    table.add_column("TOPS")
    table.add_column("Risk")

    for r in results:
        risk_style = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(r.risk_level, "white")
        table.add_row(
            r.name,
            f"{r.seu_rate_per_day:.2e}",
            f"{r.expected_seus_per_orbit:.2f}",
            f"{r.tid_over_mission_krad:.1f}",
            f"{r.power_watts:.0f}",
            f"{r.compute_tops}",
            f"[{risk_style}]{r.risk_level}[/]",
        )
    console.print(table)


@cli.command()
@click.option("--type", "report_type", required=True, type=click.Choice(["ecss", "milstd"]))
@click.option("--orbit", required=True, help="Orbit as 'altitude_km/inclination_deg'")
@click.option("--chip", "chip_name", required=True, help="Chip name")
@click.option("--mission-years", default=5.0, type=float)
@click.option("--shielding", default=3.0, type=float)
@click.option("--tmr", default="selective_tmr")
@click.option("--output", required=True, help="Output HTML file path")
def report(
    report_type: str,
    orbit: str,
    chip_name: str,
    mission_years: float,
    shielding: float,
    tmr: str,
    output: str,
):
    """Generate a compliance report (ECSS or MIL-STD)."""
    import math

    orbit_config = _parse_orbit(orbit)
    chip = _resolve_chip(chip_name)

    from space_ml_sim.environment.radiation import RadiationEnvironment

    rad_env = RadiationEnvironment(
        altitude_km=orbit_config.altitude_km,
        inclination_deg=orbit_config.inclination_deg,
        shielding_mm_al=shielding,
    )

    if report_type == "ecss":
        from space_ml_sim.reports.ecss_report import generate_ecss_report

        R_EARTH = 6371.0
        MU = 398600.4418
        a = R_EARTH + orbit_config.altitude_km
        period_sec = 2 * math.pi * math.sqrt(a**3 / MU)

        xsec_factor = chip.seu_cross_section_cm2 / 1e-14
        seu_rate = rad_env.base_seu_rate * xsec_factor
        seus_per_orbit = seu_rate * chip.memory_bits * period_sec

        html = generate_ecss_report(
            mission_name=f"{chip.name} @ {orbit_config.altitude_km}km",
            orbit=orbit_config,
            mission_duration_years=mission_years,
            chip_name=chip.name,
            model_name="User-specified model",
            total_parameters=1_000_000,
            seu_rate_per_bit_per_day=seu_rate * 86400,
            tid_rate_rad_per_day=rad_env.tid_rate_krad_per_day * 1000,
            expected_seus_per_orbit=seus_per_orbit,
            tmr_strategy=tmr,
            protected_layers=[],
            compute_multiplier={
                "none": 1.0,
                "full_tmr": 3.0,
                "selective_tmr": 1.5,
                "checkpoint_rollback": 1.0,
            }.get(tmr, 1.0),
            expected_accuracy_recovery=0.0,
            shielding_mm_al=shielding,
        )
    else:
        from space_ml_sim.reports.milstd_report import generate_milstd_report

        html = generate_milstd_report(
            device_name=chip.name,
            test_facility="Simulation (space-ml-sim)",
            ion_species="Fe-56",
            energy_mev=1000,
            let_mev_cm2_mg=28.0,
            fluence_ions_cm2=1e7,
            cross_section_cm2=chip.seu_cross_section_cm2,
            threshold_let=5.0,
            saturation_cross_section=chip.seu_cross_section_cm2 * 2,
            num_errors_observed=int(chip.seu_cross_section_cm2 * 1e7 * chip.memory_bits),
            bits_under_test=chip.memory_bits,
            test_temperature_c=25.0,
        )

    with open(output, "w") as f:
        f.write(html)

    console.print(f"[green]Report saved to {output}[/]")


@cli.command("link-budget")
@click.option("--orbit", required=True, help="Orbit as 'altitude_km/inclination_deg'")
@click.option("--freq", default="S", help="Frequency band: UHF, S, X, Ku, Ka, V, optical")
@click.option("--tx-power", default=5.0, type=float, help="Transmitter power in dBW")
@click.option("--tx-gain", default=6.0, type=float, help="Tx antenna gain in dBi")
@click.option("--rx-gain", default=30.0, type=float, help="Rx antenna gain in dBi")
@click.option("--noise-temp", default=300.0, type=float, help="System noise temp in K")
@click.option("--bandwidth", default=2e6, type=float, help="Bandwidth in Hz")
def link_budget(
    orbit: str,
    freq: str,
    tx_power: float,
    tx_gain: float,
    rx_gain: float,
    noise_temp: float,
    bandwidth: float,
):
    """Compute satellite downlink budget."""
    import math
    from space_ml_sim.comms.link_budget import compute_link_budget, FREQUENCY_BANDS

    orbit_config = _parse_orbit(orbit)
    band = FREQUENCY_BANDS.get(freq)
    if band is None:
        raise click.BadParameter(f"Unknown band '{freq}'. Options: {list(FREQUENCY_BANDS.keys())}")

    slant_range = orbit_config.altitude_km / math.sin(math.radians(10))
    slant_range = min(slant_range, orbit_config.altitude_km * 4)

    result = compute_link_budget(
        tx_power_dbw=tx_power,
        tx_antenna_gain_dbi=tx_gain,
        frequency_hz=band.center_freq_hz,
        distance_km=slant_range,
        rx_antenna_gain_dbi=rx_gain,
        system_noise_temp_k=noise_temp,
        bandwidth_hz=bandwidth,
        atmospheric_loss_db=band.atmospheric_loss_db,
    )

    table = Table(title=f"Link Budget: {band.name} @ {orbit}")
    table.add_column("Parameter")
    table.add_column("Value")
    table.add_row("EIRP", f"{result.eirp_dbw:.1f} dBW")
    table.add_row("Slant Range", f"{slant_range:.0f} km")
    table.add_row("Free Space Loss", f"{result.free_space_loss_db:.1f} dB")
    table.add_row("Atmospheric Loss", f"{result.atmospheric_loss_db:.1f} dB")
    table.add_row("G/T", f"{result.gt_db_k:.1f} dB/K")
    table.add_row("C/N", f"{result.cn_db:.1f} dB")
    table.add_row("Eb/No", f"{result.eb_no_db:.1f} dB")

    margin_style = (
        "green" if result.link_margin_db > 3 else ("yellow" if result.link_margin_db > 0 else "red")
    )
    table.add_row("Link Margin", f"[{margin_style}]{result.link_margin_db:.1f} dB[/]")
    table.add_row("Max Data Rate", f"{result.max_data_rate_bps / 1e6:.1f} Mbps")
    console.print(table)


@cli.command()
def constellations():
    """List available constellation presets."""
    from space_ml_sim.core.constellation_presets import CONSTELLATION_PRESETS

    table = Table(title="Constellation Presets")
    table.add_column("Key")
    table.add_column("Name")
    table.add_column("Operator")
    table.add_column("Sats")
    table.add_column("Planes")
    table.add_column("Alt (km)")
    table.add_column("Inc (°)")

    for key, preset in CONSTELLATION_PRESETS.items():
        table.add_row(
            key,
            preset.name,
            preset.operator,
            str(preset.num_satellites),
            str(preset.num_planes),
            str(preset.altitude_km),
            str(preset.inclination_deg),
        )
    console.print(table)


@cli.command()
@click.option("--orbit", required=True, help="Orbit as 'altitude_km/inclination_deg'")
@click.option("--chip", "chip_name", required=True, help="Chip name")
@click.option("--mission-years", default=5.0, type=float)
@click.option("--shielding", default=2.0, type=float)
@click.option("--tmr", default="none")
@click.option("--solar", default=None, help="Solar cycle phase: solar_min, solar_max, average")
@click.option("--output", default=None, help="Save ECSS report to HTML file")
def analyze(
    orbit: str,
    chip_name: str,
    mission_years: float,
    shielding: float,
    tmr: str,
    solar: str | None,
    output: str | None,
):
    """Run full mission analysis (radiation + thermal + link + risk)."""
    from space_ml_sim.analysis.mission_analysis import run_mission_analysis

    orbit_config = _parse_orbit(orbit)
    chip = _resolve_chip(chip_name)

    result = run_mission_analysis(
        mission_name=f"{chip.name} @ {orbit_config.altitude_km}km",
        orbit=orbit_config,
        chip=chip,
        mission_years=mission_years,
        shielding_mm_al=shielding,
        tmr_strategy=tmr,
        solar_cycle_phase=solar,
    )

    table = Table(title=f"Mission Analysis: {result.mission_name}")
    table.add_column("Domain")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Risk")

    # Radiation
    risk_style = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}
    rad_risk = result.risk_factors.get("SEU", "LOW")
    table.add_row(
        "Radiation",
        "SEU/day",
        f"{result.seu_rate_per_day:.2e}",
        f"[{risk_style[rad_risk]}]{rad_risk}[/]",
    )
    table.add_row("", "SEU/orbit", f"{result.expected_seus_per_orbit:.2f}", "")
    tid_risk = result.risk_factors.get("TID", "LOW")
    table.add_row(
        "",
        "TID (mission)",
        f"{result.tid_over_mission_krad:.1f} krad",
        f"[{risk_style[tid_risk]}]{tid_risk}[/]",
    )

    # Thermal
    thermal_risk = result.risk_factors.get("THERMAL", "LOW")
    table.add_row(
        "Thermal",
        "Min temp",
        f"{result.min_temperature_c:.0f} °C",
        f"[{risk_style[thermal_risk]}]{thermal_risk}[/]",
    )
    table.add_row("", "Max temp", f"{result.max_temperature_c:.0f} °C", "")
    table.add_row("", "Cycles/orbit", f"{result.thermal_cycles_per_orbit}", "")

    # Link
    link_risk = result.risk_factors.get("LINK", "LOW")
    margin_str = f"{result.downlink_margin_db:.1f} dB" if result.downlink_margin_db else "N/A"
    rate_str = f"{result.max_data_rate_bps / 1e6:.1f} Mbps" if result.max_data_rate_bps else "N/A"
    table.add_row("Link", "Margin", margin_str, f"[{risk_style[link_risk]}]{link_risk}[/]")
    table.add_row("", "Max rate", rate_str, "")

    # Overall
    overall_style = risk_style[result.overall_risk]
    table.add_row("[bold]Overall[/]", "", "", f"[bold {overall_style}]{result.overall_risk}[/]")

    console.print(table)

    if output:
        with open(output, "w") as f:
            f.write(result.ecss_report_html)
        console.print(f"[green]ECSS report saved to {output}[/]")
