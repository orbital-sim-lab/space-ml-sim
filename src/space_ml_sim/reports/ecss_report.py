"""ECSS-Q-ST-60-15C compliance report generator.

Generates structured HTML reports for radiation hardness assurance
following the European Cooperation for Space Standardization (ECSS)
guidelines for EEE components.

The report can be saved as HTML and printed/exported to PDF via any
web browser (File > Print > Save as PDF).
"""

from __future__ import annotations

import datetime
from textwrap import dedent

from space_ml_sim.core.orbit import OrbitConfig


_REPORT_TEMPLATE = dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECSS-Q-ST-60-15C Radiation Hardness Assurance Report — {mission_name}</title>
<style>
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
    color: #1a1a1a;
    line-height: 1.6;
  }}
  h1 {{
    border-bottom: 3px solid #1a1a1a;
    padding-bottom: 10px;
    font-size: 1.8em;
  }}
  h2 {{
    color: #2c5282;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 6px;
    margin-top: 2em;
  }}
  h3 {{
    color: #2d3748;
    margin-top: 1.5em;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
  }}
  th, td {{
    border: 1px solid #cbd5e0;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background-color: #edf2f7;
    font-weight: 600;
  }}
  .metric {{
    font-size: 1.1em;
    font-weight: 600;
    color: #2c5282;
  }}
  .warning {{
    background-color: #fffbeb;
    border-left: 4px solid #d69e2e;
    padding: 10px 15px;
    margin: 1em 0;
  }}
  .footer {{
    margin-top: 3em;
    padding-top: 1em;
    border-top: 1px solid #e2e8f0;
    font-size: 0.85em;
    color: #718096;
  }}
  @media print {{
    body {{ padding: 20px; }}
    h2 {{ page-break-before: auto; }}
  }}
</style>
</head>
<body>

<h1>Radiation Hardness Assurance Report</h1>
<p><strong>Standard:</strong> ECSS-Q-ST-60-15C (Radiation Hardness Assurance — EEE Components)</p>
<p><strong>Generated:</strong> {generation_date}</p>
<p><strong>Tool:</strong> space-ml-sim v0.5.0</p>

<hr>

<h2>1. Mission Overview</h2>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Mission Name</td><td>{mission_name}</td></tr>
  <tr><td>Mission Duration</td><td>{mission_duration_years} years</td></tr>
  <tr><td>ML Model</td><td>{model_name}</td></tr>
  <tr><td>Total Parameters</td><td>{total_parameters:,}</td></tr>
  <tr><td>Compute Platform</td><td>{chip_name}</td></tr>
  <tr><td>Shielding</td><td>{shielding_mm_al} mm Aluminium equivalent</td></tr>
</table>

<h2>2. Orbital Environment</h2>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Altitude</td><td>{altitude_km} km</td></tr>
  <tr><td>Inclination</td><td>{inclination_deg}&deg;</td></tr>
  <tr><td>RAAN</td><td>{raan_deg}&deg;</td></tr>
  <tr><td>Orbit Type</td><td>{orbit_type}</td></tr>
  <tr><td>Orbital Period</td><td>~{period_min:.0f} minutes</td></tr>
</table>

<h2>3. Radiation Environment</h2>

<h3>3.1 Single Event Upset (SEU) Environment</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>SEU Rate</td><td class="metric">{seu_rate_per_bit_per_day:.2e} upsets/bit/day</td></tr>
  <tr><td>Expected SEUs per Orbit</td><td class="metric">{expected_seus_per_orbit:.1f}</td></tr>
  <tr><td>Expected SEUs per Day</td><td>{seus_per_day:.1f}</td></tr>
  <tr><td>Expected SEUs over Mission</td><td>{seus_over_mission:,.0f}</td></tr>
</table>

<h3>3.2 Total Ionizing Dose (TID) Environment</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>TID Rate</td><td class="metric">{tid_rate_rad_per_day:.2f} rad(Si)/day</td></tr>
  <tr><td>TID over Mission</td><td class="metric">{tid_over_mission:,.0f} rad(Si)</td></tr>
</table>

{tid_warning}

<h2>4. Component Analysis</h2>

<h3>4.1 ML Inference Processor</h3>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Chip</td><td>{chip_name}</td></tr>
  <tr><td>Model Architecture</td><td>{model_name}</td></tr>
  <tr><td>Total Trainable Parameters</td><td>{total_parameters:,}</td></tr>
  <tr><td>Memory Footprint (FP32)</td><td>{memory_footprint_mb:.1f} MB</td></tr>
  <tr><td>Vulnerable Bits</td><td>{vulnerable_bits:,}</td></tr>
</table>

<h2>5. Fault Tolerance Strategy</h2>

<h3>5.1 Mitigation Approach</h3>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Strategy</td><td>{tmr_strategy_display}</td></tr>
  <tr><td>Compute Multiplier</td><td>{compute_multiplier}x</td></tr>
  <tr><td>Expected Accuracy Recovery</td><td>{expected_accuracy_recovery_pct:.1f}%</td></tr>
  <tr><td>Protected Layers</td><td>{num_protected_layers}</td></tr>
</table>

<h3>5.2 Protected Layer Configuration</h3>
{protected_layers_table}

<h2>6. Risk Assessment</h2>

<h3>6.1 Residual Risk Summary</h3>
<table>
  <tr><th>Risk Factor</th><th>Level</th><th>Mitigation</th></tr>
  <tr>
    <td>SEU-induced inference error</td>
    <td>{seu_risk_level}</td>
    <td>{tmr_strategy_display} ({compute_multiplier}x overhead)</td>
  </tr>
  <tr>
    <td>TID degradation over mission</td>
    <td>{tid_risk_level}</td>
    <td>{shielding_mm_al} mm Al shielding</td>
  </tr>
  <tr>
    <td>Unprotected layer vulnerability</td>
    <td>{unprotected_risk_level}</td>
    <td>Accept or increase compute budget</td>
  </tr>
</table>

<h3>6.2 Recommendations</h3>
<ul>
{recommendations}
</ul>

<div class="footer">
  <p>This report was generated by space-ml-sim and is intended to support radiation hardness
  assurance analysis per ECSS-Q-ST-60-15C. It does not constitute formal qualification and
  should be reviewed by a radiation effects engineer before use in flight qualification.</p>
  <p>Report generated: {generation_date}</p>
</div>

</body>
</html>
""")


def generate_ecss_report(
    mission_name: str,
    orbit: OrbitConfig,
    mission_duration_years: float,
    chip_name: str,
    model_name: str,
    total_parameters: int,
    seu_rate_per_bit_per_day: float,
    tid_rate_rad_per_day: float,
    expected_seus_per_orbit: float,
    tmr_strategy: str,
    protected_layers: list[str],
    compute_multiplier: float,
    expected_accuracy_recovery: float,
    shielding_mm_al: float,
) -> str:
    """Generate an ECSS-Q-ST-60-15C compliance report as HTML.

    Args:
        mission_name: Mission identifier.
        orbit: Orbital configuration.
        mission_duration_years: Mission duration in years.
        chip_name: Compute platform name.
        model_name: ML model architecture name.
        total_parameters: Total trainable parameters in the model.
        seu_rate_per_bit_per_day: SEU rate per bit per day.
        tid_rate_rad_per_day: TID rate in rad(Si) per day.
        expected_seus_per_orbit: Expected SEU events per orbit.
        tmr_strategy: TMR strategy name.
        protected_layers: List of protected layer names.
        compute_multiplier: Compute overhead factor.
        expected_accuracy_recovery: Expected accuracy recovery fraction.
        shielding_mm_al: Aluminium equivalent shielding in mm.

    Returns:
        Complete HTML report string.
    """
    import math

    mission_days = mission_duration_years * 365.25
    seus_per_day = expected_seus_per_orbit * (86400 / _orbital_period_seconds(orbit))
    seus_over_mission = seus_per_day * mission_days
    tid_over_mission = tid_rate_rad_per_day * mission_days

    vulnerable_bits = total_parameters * 32  # FP32
    memory_footprint_mb = total_parameters * 4 / (1024 * 1024)

    # Orbit classification
    if orbit.inclination_deg > 96 and orbit.inclination_deg < 100:
        orbit_type = "Sun-Synchronous (SSO)"
    elif orbit.inclination_deg > 50 and orbit.inclination_deg < 60:
        orbit_type = "Mid-inclination LEO"
    else:
        orbit_type = f"LEO ({orbit.inclination_deg:.1f} deg)"

    period_min = _orbital_period_seconds(orbit) / 60.0

    # TMR display
    strategy_display = {
        "full_tmr": "Full Triple Modular Redundancy",
        "selective_tmr": "Selective TMR (per-layer)",
        "checkpoint_rollback": "Checkpoint Rollback",
    }.get(tmr_strategy, tmr_strategy)

    # Protected layers table
    if protected_layers:
        rows = "\n".join(
            f"  <tr><td>{i + 1}</td><td><code>{name}</code></td></tr>"
            for i, name in enumerate(protected_layers)
        )
        protected_layers_table = (
            f"<table>\n  <tr><th>#</th><th>Layer Name</th></tr>\n{rows}\n</table>"
        )
    else:
        protected_layers_table = "<p><em>No individual layer protection configured.</em></p>"

    # TID warning
    tid_warning = ""
    if tid_over_mission > 10_000:
        tid_warning = (
            '<div class="warning"><strong>Warning:</strong> Total mission TID '
            f"({tid_over_mission:,.0f} rad) exceeds 10 krad. Additional shielding "
            "or radiation-hardened components recommended.</div>"
        )

    # Risk levels
    seu_risk_level = (
        "LOW" if expected_seus_per_orbit < 1
        else "MEDIUM" if expected_seus_per_orbit < 10
        else "HIGH"
    )
    tid_risk_level = (
        "LOW" if tid_over_mission < 5_000
        else "MEDIUM" if tid_over_mission < 50_000
        else "HIGH"
    )
    unprotected_risk_level = (
        "LOW" if compute_multiplier >= 2.5
        else "MEDIUM" if compute_multiplier >= 1.5
        else "HIGH"
    )

    # Recommendations
    recs: list[str] = []
    if seu_risk_level == "HIGH":
        recs.append("<li>Consider full TMR for all critical inference layers</li>")
    if tid_risk_level in ("MEDIUM", "HIGH"):
        recs.append("<li>Increase shielding or use radiation-hardened memory</li>")
    if compute_multiplier < 1.5:
        recs.append(
            "<li>Current compute overhead is minimal — consider protecting more layers "
            "if power budget allows</li>"
        )
    if expected_accuracy_recovery < 0.1:
        recs.append(
            "<li>Expected accuracy recovery is low — review sensitivity analysis "
            "and adjust protected layer selection</li>"
        )
    recs.append(
        "<li>Validate simulation results against radiation test facility data "
        "before flight qualification</li>"
    )
    recs.append(
        "<li>Perform periodic on-orbit health checks to detect TID degradation</li>"
    )

    return _REPORT_TEMPLATE.format(
        mission_name=mission_name,
        generation_date=datetime.date.today().isoformat(),
        mission_duration_years=mission_duration_years,
        model_name=model_name,
        total_parameters=total_parameters,
        chip_name=chip_name,
        shielding_mm_al=shielding_mm_al,
        altitude_km=orbit.altitude_km,
        inclination_deg=orbit.inclination_deg,
        raan_deg=orbit.raan_deg,
        orbit_type=orbit_type,
        period_min=period_min,
        seu_rate_per_bit_per_day=seu_rate_per_bit_per_day,
        expected_seus_per_orbit=expected_seus_per_orbit,
        seus_per_day=seus_per_day,
        seus_over_mission=seus_over_mission,
        tid_rate_rad_per_day=tid_rate_rad_per_day,
        tid_over_mission=tid_over_mission,
        tid_warning=tid_warning,
        memory_footprint_mb=memory_footprint_mb,
        vulnerable_bits=vulnerable_bits,
        tmr_strategy_display=strategy_display,
        compute_multiplier=compute_multiplier,
        expected_accuracy_recovery_pct=expected_accuracy_recovery * 100,
        num_protected_layers=len(protected_layers),
        protected_layers_table=protected_layers_table,
        seu_risk_level=seu_risk_level,
        tid_risk_level=tid_risk_level,
        unprotected_risk_level=unprotected_risk_level,
        recommendations="\n".join(recs),
    )


def _orbital_period_seconds(orbit: OrbitConfig) -> float:
    """Approximate orbital period from altitude."""
    import math

    R_EARTH = 6371.0
    MU = 398600.4418  # km^3/s^2
    a = R_EARTH + orbit.altitude_km
    return 2 * math.pi * math.sqrt(a**3 / MU)
