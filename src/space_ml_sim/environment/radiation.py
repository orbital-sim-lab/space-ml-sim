"""Parametric radiation environment model for LEO orbits.

Models Single Event Upsets (SEU) from galactic cosmic rays and trapped protons,
and Total Ionizing Dose (TID) accumulation. Uses simplified parametric fits
suitable for simulation — not intended for flight qualification.

References:
    - NASA AP-8/AE-8 trapped particle models (parametric approximations)
    - CREME96 GCR model (simplified)
    - ESA SPENVIS radiation analysis tool (validation targets)
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from space_ml_sim.models.rad_profiles import RadPreset


class RadiationEnvironment(BaseModel):
    """Parametric radiation model for LEO satellites.

    SEU rate model (simplified):
        - Base GCR rate: ~1e-7 upsets/bit/day at 500km with 2mm Al shielding
        - Trapped proton enhancement: increases ~10x near SAA, ~3x at 2000km
        - Rate scales linearly with chip cross-section

    TID model:
        - ~0.1 krad(Si)/year at 500km with 2mm Al
        - ~1 krad(Si)/year at 800km
        - ~10 krad(Si)/year at 2000km (inner belt proximity)
    """

    altitude_km: float = Field(gt=0)
    inclination_deg: float = Field(ge=0, le=180)
    shielding_mm_al: float = Field(default=2.0, ge=0)

    # Computed rates (set during model_post_init)
    base_seu_rate: float = Field(default=0.0, description="Upsets/bit/second")
    tid_rate_krad_per_day: float = Field(default=0.0, description="krad(Si) per day")

    def model_post_init(self, __context: object) -> None:
        """Compute base radiation rates from orbital parameters."""
        object.__setattr__(
            self,
            "base_seu_rate",
            self._compute_seu_rate(self.altitude_km, self.inclination_deg, self.shielding_mm_al),
        )
        object.__setattr__(
            self,
            "tid_rate_krad_per_day",
            self._compute_tid_rate(self.altitude_km, self.shielding_mm_al),
        )

    @staticmethod
    def _compute_seu_rate(alt: float, inc: float, shield: float) -> float:
        """Compute SEU rate in upsets/bit/second.

        GCR contribution is relatively flat with altitude.
        Trapped protons increase exponentially above ~800km.
        SAA enhancement applies for inclinations 20-60 deg below 1500km.
        """
        gcr_base = 1e-12  # upsets/bit/sec baseline at 500km/2mm Al

        # Trapped protons increase exponentially above 800km
        trapped = 0.0
        if alt > 800:
            trapped = gcr_base * 10 * ((alt - 800) / 1200) ** 2

        # SAA enhancement for relevant inclinations
        saa_factor = 1.0
        if 20 < inc < 60 and alt < 1500:
            saa_factor = 3.0  # Average enhancement (SAA is periodic)

        # Shielding attenuation (roughly exponential, normalized to 1.0 at 0mm)
        shield_factor = math.exp(-0.3 * shield)

        return (gcr_base + trapped) * saa_factor * shield_factor

    @staticmethod
    def _compute_tid_rate(alt: float, shield: float) -> float:
        """Compute TID rate in krad(Si) per day.

        Dominated by trapped protons; exponential increase with altitude.
        """
        base_krad_per_year = 0.1 * math.exp(0.003 * (alt - 500))
        shield_atten = math.exp(-0.5 * shield)
        return (base_krad_per_year / 365.25) * shield_atten

    def sample_seu_events(
        self, chip_cross_section_cm2: float, num_bits: int, dt_seconds: float
    ) -> int:
        """Sample number of SEU events in a time interval from Poisson distribution.

        Args:
            chip_cross_section_cm2: SEU cross-section per bit in cm^2.
            num_bits: Total number of bits exposed.
            dt_seconds: Time interval in seconds.

        Returns:
            Number of single-event upsets sampled.
        """
        # SEU rate = flux * cross_section_per_bit * num_bits * time
        # base_seu_rate encodes the environmental particle flux (upsets/bit/sec
        # at a reference cross-section of 1e-14 cm^2). Scale by the chip's
        # actual cross-section normalized to this reference.
        reference_cross_section = 1e-14  # cm^2, matches gcr_base calibration
        cross_section_factor = chip_cross_section_cm2 / reference_cross_section
        rate = self.base_seu_rate * cross_section_factor * num_bits * dt_seconds
        return int(np.random.poisson(rate))

    def tid_dose(self, dt_seconds: float) -> float:
        """TID accumulated over dt_seconds in krad(Si)."""
        return self.tid_rate_krad_per_day * (dt_seconds / 86400.0)

    # --- Factory presets ---

    @classmethod
    def leo_500km(cls) -> "RadiationEnvironment":
        """SpaceX lower shell (~500km, 53 deg)."""
        return cls(altitude_km=500, inclination_deg=53, shielding_mm_al=2.0)

    @classmethod
    def sso_650km(cls) -> "RadiationEnvironment":
        """Sun-synchronous orbit (~650km, 98 deg)."""
        return cls(altitude_km=650, inclination_deg=98, shielding_mm_al=2.0)

    @classmethod
    def leo_2000km(cls) -> "RadiationEnvironment":
        """High LEO near inner belt (~2000km, 53 deg)."""
        return cls(altitude_km=2000, inclination_deg=53, shielding_mm_al=2.0)

    @classmethod
    def from_preset(cls, preset: RadPreset) -> "RadiationEnvironment":
        """Create from a named preset."""
        factories = {
            RadPreset.LEO_500KM: cls.leo_500km,
            RadPreset.SSO_650KM: cls.sso_650km,
            RadPreset.LEO_2000KM: cls.leo_2000km,
        }
        return factories[preset]()
