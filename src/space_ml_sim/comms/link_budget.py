"""Satellite link budget calculator.

Computes RF link budgets for LEO satellite downlink/uplink including:
- EIRP (Effective Isotropic Radiated Power)
- Free-space path loss (FSPL)
- Atmospheric attenuation
- G/T (receiver figure of merit)
- C/N (carrier-to-noise ratio)
- Eb/No and link margin
- Shannon capacity data rate estimate
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Speed of light in m/s
_C = 299792458.0

# Boltzmann constant in dBW/K/Hz
_K_DBW = -228.6  # 10*log10(1.38e-23)


@dataclass(frozen=True)
class FrequencyBand:
    """Standard satellite communication frequency band."""

    name: str
    center_freq_hz: float
    typical_bandwidth_hz: float
    atmospheric_loss_db: float


FREQUENCY_BANDS: dict[str, FrequencyBand] = {
    "UHF": FrequencyBand("UHF", 400e6, 50e3, 0.3),
    "S": FrequencyBand("S-band", 2.2e9, 2e6, 0.5),
    "X": FrequencyBand("X-band", 8.4e9, 10e6, 1.0),
    "Ku": FrequencyBand("Ku-band", 14.5e9, 36e6, 2.0),
    "Ka": FrequencyBand("Ka-band", 26.5e9, 500e6, 4.0),
    "V": FrequencyBand("V-band", 50.0e9, 1e9, 8.0),
    "optical": FrequencyBand("Optical (1550nm)", 193.1e12, 10e9, 0.5),
}


@dataclass(frozen=True)
class LinkBudgetResult:
    """Complete link budget analysis result."""

    eirp_dbw: float
    free_space_loss_db: float
    atmospheric_loss_db: float
    pointing_loss_db: float
    rx_antenna_gain_dbi: float
    system_noise_temp_k: float
    gt_db_k: float
    cn_db: float
    eb_no_db: float
    required_eb_no_db: float
    link_margin_db: float
    max_data_rate_bps: float
    bandwidth_hz: float


def free_space_path_loss_db(distance_km: float, frequency_hz: float) -> float:
    """Compute free-space path loss in dB.

    FSPL = 20*log10(4*pi*d*f/c) where d is in meters, f in Hz.

    Args:
        distance_km: Slant range in km.
        frequency_hz: Carrier frequency in Hz.

    Returns:
        Path loss in dB (positive value).
    """
    d_m = distance_km * 1000.0
    fspl = 20.0 * math.log10(4.0 * math.pi * d_m * frequency_hz / _C)
    return fspl


def compute_link_budget(
    tx_power_dbw: float,
    tx_antenna_gain_dbi: float,
    frequency_hz: float,
    distance_km: float,
    rx_antenna_gain_dbi: float,
    system_noise_temp_k: float,
    bandwidth_hz: float,
    atmospheric_loss_db: float = 0.0,
    pointing_loss_db: float = 0.0,
    required_eb_no_db: float = 10.0,
) -> LinkBudgetResult:
    """Compute a complete satellite link budget.

    Args:
        tx_power_dbw: Transmitter power in dBW.
        tx_antenna_gain_dbi: Transmit antenna gain in dBi.
        frequency_hz: Carrier frequency in Hz.
        distance_km: Slant range in km.
        rx_antenna_gain_dbi: Receive antenna gain in dBi.
        system_noise_temp_k: System noise temperature in Kelvin.
        bandwidth_hz: Channel bandwidth in Hz.
        atmospheric_loss_db: Atmospheric attenuation in dB.
        pointing_loss_db: Antenna pointing loss in dB.
        required_eb_no_db: Required Eb/No for target BER.

    Returns:
        LinkBudgetResult with all intermediate and final values.
    """
    eirp = tx_power_dbw + tx_antenna_gain_dbi
    fspl = free_space_path_loss_db(distance_km, frequency_hz)

    # G/T = receiver antenna gain - 10*log10(noise temp)
    gt = rx_antenna_gain_dbi - 10.0 * math.log10(system_noise_temp_k)

    # C/N = EIRP - FSPL - atmo_loss - pointing_loss + G/T - k
    # where k = Boltzmann constant in dBW/K/Hz = -228.6
    cn = (
        eirp
        - fspl
        - atmospheric_loss_db
        - pointing_loss_db
        + gt
        - _K_DBW
        - 10.0 * math.log10(bandwidth_hz)
    )

    # Eb/No = C/N + 10*log10(BW/Rb) — for BW-limited, Eb/No ≈ C/N when Rb ≈ BW
    eb_no = cn  # Simplified: assumes Rb ≈ BW

    margin = eb_no - required_eb_no_db

    # Shannon capacity: C = BW * log2(1 + SNR)
    snr_linear = 10.0 ** (cn / 10.0)
    max_data_rate = bandwidth_hz * math.log2(1.0 + snr_linear)

    return LinkBudgetResult(
        eirp_dbw=eirp,
        free_space_loss_db=fspl,
        atmospheric_loss_db=atmospheric_loss_db,
        pointing_loss_db=pointing_loss_db,
        rx_antenna_gain_dbi=rx_antenna_gain_dbi,
        system_noise_temp_k=system_noise_temp_k,
        gt_db_k=gt,
        cn_db=cn,
        eb_no_db=eb_no,
        required_eb_no_db=required_eb_no_db,
        link_margin_db=margin,
        max_data_rate_bps=max_data_rate,
        bandwidth_hz=bandwidth_hz,
    )
