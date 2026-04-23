"""MIL-STD-883 Test Method 1019 test methodology report generator.

Generates structured HTML reports documenting Single Event Effects (SEE)
test procedures and results per MIL-STD-883 TM 1019.8.

The report can be saved as HTML and printed/exported to PDF.
"""

from __future__ import annotations

import datetime


def generate_milstd_report(
    device_name: str,
    test_facility: str,
    ion_species: str,
    energy_mev: float,
    let_mev_cm2_mg: float,
    fluence_ions_cm2: float,
    cross_section_cm2: float,
    threshold_let: float,
    saturation_cross_section: float,
    num_errors_observed: int,
    bits_under_test: int,
    test_temperature_c: float,
) -> str:
    """Generate a MIL-STD-883 TM 1019 SEE test report.

    Args:
        device_name: Device under test identifier.
        test_facility: Radiation test facility name.
        ion_species: Heavy ion species used (e.g., "Fe-56").
        energy_mev: Beam energy in MeV.
        let_mev_cm2_mg: Linear Energy Transfer in MeV*cm^2/mg.
        fluence_ions_cm2: Total fluence in ions/cm^2.
        cross_section_cm2: Measured SEU cross-section in cm^2/bit.
        threshold_let: LET threshold for SEU onset in MeV*cm^2/mg.
        saturation_cross_section: Saturated cross-section in cm^2/bit.
        num_errors_observed: Total SEU errors counted.
        bits_under_test: Number of bits under test.
        test_temperature_c: Test temperature in Celsius.

    Returns:
        Complete HTML report string.
    """
    error_rate = num_errors_observed / fluence_ions_cm2 if fluence_ions_cm2 > 0 else 0
    cross_section_per_bit = num_errors_observed / (fluence_ions_cm2 * bits_under_test) if (fluence_ions_cm2 > 0 and bits_under_test > 0) else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIL-STD-883 TM 1019 SEE Test Report — {device_name}</title>
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
  }}
  h2 {{
    color: #2c5282;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 6px;
    margin-top: 2em;
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
    font-weight: 600;
    color: #2c5282;
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
  }}
</style>
</head>
<body>

<h1>Single Event Effects Test Report</h1>
<p><strong>Standard:</strong> MIL-STD-883 Test Method 1019.8</p>
<p><strong>Generated:</strong> {datetime.date.today().isoformat()}</p>
<p><strong>Tool:</strong> space-ml-sim v0.5.0</p>

<hr>

<h2>1. Test Configuration</h2>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Device Under Test</td><td>{device_name}</td></tr>
  <tr><td>Test Facility</td><td>{test_facility}</td></tr>
  <tr><td>Bits Under Test</td><td>{bits_under_test:,}</td></tr>
  <tr><td>Test Temperature</td><td>{test_temperature_c} &deg;C</td></tr>
</table>

<h2>2. Beam Parameters</h2>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Ion Species</td><td>{ion_species}</td></tr>
  <tr><td>Energy</td><td>{energy_mev} MeV</td></tr>
  <tr><td>LET</td><td class="metric">{let_mev_cm2_mg} MeV&middot;cm&sup2;/mg</td></tr>
  <tr><td>Total Fluence</td><td>{fluence_ions_cm2:.2e} ions/cm&sup2;</td></tr>
</table>

<h2>3. Test Results</h2>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Errors Observed</td><td class="metric">{num_errors_observed}</td></tr>
  <tr><td>Error Rate</td><td>{error_rate:.2e} errors/ion/cm&sup2;</td></tr>
  <tr><td>Measured Cross-Section</td><td class="metric">{cross_section_cm2:.2e} cm&sup2;/bit</td></tr>
  <tr><td>Calculated Cross-Section</td><td>{cross_section_per_bit:.2e} cm&sup2;/bit</td></tr>
</table>

<h2>4. Cross-Section Analysis</h2>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>LET Threshold</td><td>{threshold_let} MeV&middot;cm&sup2;/mg</td></tr>
  <tr><td>Saturation Cross-Section</td><td>{saturation_cross_section:.2e} cm&sup2;/bit</td></tr>
  <tr><td>Test LET / Threshold LET Ratio</td><td>{let_mev_cm2_mg / threshold_let:.1f}x</td></tr>
</table>

<h3>4.1 Weibull Fit Parameters (for rate prediction)</h3>
<p>To predict on-orbit SEE rates, the cross-section vs LET curve should be fit
to a Weibull function. This report provides the single-LET data point above.
A complete characterization requires measurements at multiple LET values
(typically 5+ points from threshold to saturation).</p>

<h2>5. Compliance Notes</h2>
<ul>
  <li>Test performed per MIL-STD-883 TM 1019.8 methodology</li>
  <li>Device was powered and operational during irradiation</li>
  <li>Error detection via comparison with reference pattern</li>
  <li>Minimum fluence requirement: 1&times;10<sup>7</sup> ions/cm&sup2; per TM 1019 &mdash;
      {"PASS" if fluence_ions_cm2 >= 1e7 else "FAIL: insufficient fluence"}</li>
</ul>

<div class="footer">
  <p>This report was generated by space-ml-sim for documentation purposes.
  It should be reviewed alongside actual facility test data before use in
  qualification decisions. Simulation-derived cross-sections should be validated
  against beam test measurements.</p>
</div>

</body>
</html>"""
