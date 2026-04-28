"""Heliocentric (interplanetary) radiation environment model.

Models galactic cosmic ray (GCR) and quiet-time solar background flux for
spacecraft outside Earth's magnetosphere — lunar transfer, cislunar,
interplanetary cruise, Mars and Venus encounter orbits.

This is a parametric, first-order model. It captures the two dominant
effects on GCR exposure outside the magnetosphere:

  1. **Solar cycle modulation** — GCR flux above ~100 MeV roughly doubles
     between solar maximum (more solar wind shielding) and solar minimum
     (less shielding, more cosmic rays through). Modeled as a 2x ratio.

  2. **Heliocentric distance** — GCR flux increases gradually with
     distance from the Sun as solar modulation weakens. Captured as a
     piecewise linear ramp from 0.7x at 0.3 AU to 1.5x at 10 AU+.

Solar Particle Events (SPEs) are deliberately NOT included here — they
are episodic, dominated by single events that can deliver tens of krad
in hours, and warrant a separate event-based model. Use this class for
*background* exposure budgeting.

The class deliberately mirrors the public surface of
:class:`~space_ml_sim.environment.radiation.RadiationEnvironment` so it is
a drop-in replacement for fault-injection, timeline, and budget tools
that take ``base_seu_rate`` and ``tid_rate_krad_per_day``.

References:

    - Matthiä, Berger, Mrigakshi, Reitz, "A ready-to-use galactic cosmic
      ray model", Adv. Space Res. 51 (2013) 329-338.
    - O'Neill, Golge, Slaba, "Badhwar–O'Neill 2014 Galactic Cosmic Ray
      Flux Model Description", NASA/TP-2015-218569.
    - Mewaldt et al., "Record-setting cosmic ray intensities in 2009 and
      2010", ApJ Letters 723 (2010) L1.
    - Spence et al., "CRaTER: The Cosmic Ray Telescope for the Effects
      of Radiation Experiment on the LRO Mission", Space Sci. Rev. 150
      (2010) 243-284.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SolarPhase = Literal["min", "max"]

# --- Calibration constants (1 AU, solar minimum, 2 mm Al equivalent) -------
#
# Tuned so the model lands inside published order-of-magnitude bands for
# CRaTER lunar-orbit dose rates and quiet-time GCR-induced upsets reported
# on Voyager-class SRAM. The reference cross-section is the same 1e-14 cm^2
# used by RadiationEnvironment so the two models are interoperable.

_GCR_SEU_BASE_PER_BIT_PER_SEC = 2.0e-12
"""SEU rate at 1 AU, solar minimum, 2mm Al, 1e-14 cm^2 per bit cross-section.

≈ 1.7e-7 upsets/bit/day — slightly above the upper ISS SRAM band, which
matches expectations: outside the magnetosphere there is no geomagnetic
shielding contribution.
"""

_GCR_TID_BASE_KRAD_PER_DAY = 1.0e-4
"""Background GCR-only TID rate at 1 AU, solar minimum, 2mm Al.

≈ 36 rad/year. CRaTER reports ~14 rad/yr behind heavier ~5mm Al equivalent,
so this is in the right neighborhood for thinner CubeSat-class shielding.
"""

_REFERENCE_SHIELDING_MM_AL = 2.0
"""Reference shielding thickness at which the base constants are calibrated."""


def _solar_phase_factor(phase: SolarPhase) -> float:
    """GCR modulation factor for the named solar phase.

    Solar maximum suppresses GCR by a factor of ~2 relative to solar
    minimum (Mewaldt 2010). We use 1.0 / 0.5 endpoints as a stand-in for
    a continuous modulation parameter; this is sufficient for first-order
    mission budgeting.
    """
    return {"min": 1.0, "max": 0.5}[phase]


def _heliocentric_distance_factor(distance_au: float) -> float:
    """Piecewise linear scaling of GCR flux vs heliocentric distance.

    Anchored to:
        - 0.3 AU (Mercury orbit): 0.7x — strong solar modulation
        - 1.0 AU (Earth's orbit): 1.0x — calibration point
        - 5.0 AU (Jupiter): 1.3x — modulation weakening
        - ≥10 AU: 1.5x — saturated, approaching local interstellar medium
    """
    if distance_au <= 0.3:
        return 0.7
    if distance_au <= 1.0:
        # 0.3 AU → 0.7,  1.0 AU → 1.0
        return 0.7 + (distance_au - 0.3) * (0.3 / 0.7)
    if distance_au <= 5.0:
        # 1.0 AU → 1.0,  5.0 AU → 1.3
        return 1.0 + (distance_au - 1.0) * (0.3 / 4.0)
    if distance_au <= 10.0:
        # 5.0 AU → 1.3,  10.0 AU → 1.5
        return 1.3 + (distance_au - 5.0) * (0.2 / 5.0)
    return 1.5


def _shielding_attenuation(shielding_mm_al: float, decay_per_mm: float) -> float:
    """Exponential shielding attenuation, normalized to the reference thickness.

    Returns 1.0 at the reference (so the calibration constants apply
    directly), <1 for thicker shielding, >1 for thinner.
    """
    delta = shielding_mm_al - _REFERENCE_SHIELDING_MM_AL
    return math.exp(-decay_per_mm * delta)


class HeliocentricEnvironment(BaseModel):
    """Background interplanetary radiation environment.

    Drop-in replacement for ``RadiationEnvironment`` that exposes the
    same ``base_seu_rate`` and ``tid_rate_krad_per_day`` contract, so
    any code that integrates over a radiation environment (fault
    injection, timeline plotting, mission-budget calculator) accepts
    this class without modification.

    Attributes:
        heliocentric_distance_au: Distance from the Sun in AU. Valid
            range 0.3 (Mercury) to 50 (Voyager / heliopause).
        shielding_mm_al: Aluminum-equivalent shielding thickness in mm.
        solar_phase: ``"min"`` or ``"max"`` of the 11-year solar cycle.
            Solar minimum is the conservative (high-GCR) case.
        base_seu_rate: Computed at construction. Upsets per bit per
            second at the model's reference cross-section (1e-14 cm²).
        tid_rate_krad_per_day: Computed at construction. Background
            ionizing dose rate from GCR only (no SPE contribution).

    Example::

        env = HeliocentricEnvironment(
            heliocentric_distance_au=1.0,
            shielding_mm_al=2.0,
            solar_phase="min",
        )
        print(env.base_seu_rate)            # 2e-12 upsets/bit/sec
        print(env.tid_rate_krad_per_day)    # 1e-4 krad/day
    """

    model_config = ConfigDict(frozen=True)

    heliocentric_distance_au: float = Field(gt=0, le=50)
    shielding_mm_al: float = Field(default=2.0, ge=0)
    solar_phase: SolarPhase = Field(default="min")

    base_seu_rate: float = Field(default=0.0, description="Upsets/bit/second at 1e-14 cm² ref")
    tid_rate_krad_per_day: float = Field(default=0.0, description="krad(Si) per day")

    def model_post_init(self, __context: object) -> None:
        """Compute SEU and TID rates from the heliocentric inputs."""
        phase = _solar_phase_factor(self.solar_phase)
        distance = _heliocentric_distance_factor(self.heliocentric_distance_au)

        # GCR is more penetrating than trapped protons; use a smaller
        # decay constant than RadiationEnvironment's 0.3/mm. The 0.15/mm
        # figure roughly matches Matthiä 2013 attenuation curves for
        # 2-10mm Al against integral GCR LET.
        seu_atten = _shielding_attenuation(self.shielding_mm_al, decay_per_mm=0.15)
        tid_atten = _shielding_attenuation(self.shielding_mm_al, decay_per_mm=0.20)

        seu = _GCR_SEU_BASE_PER_BIT_PER_SEC * phase * distance * seu_atten
        tid = _GCR_TID_BASE_KRAD_PER_DAY * phase * distance * tid_atten

        object.__setattr__(self, "base_seu_rate", seu)
        object.__setattr__(self, "tid_rate_krad_per_day", tid)

    # ------------------------------------------------------------------
    # API parity with RadiationEnvironment
    # ------------------------------------------------------------------

    def sample_seu_events(
        self,
        chip_cross_section_cm2: float,
        num_bits: int,
        dt_seconds: float,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Sample SEU events over a time interval (Poisson)."""
        reference_cross_section = 1e-14
        cross_section_factor = chip_cross_section_cm2 / reference_cross_section
        rate = self.base_seu_rate * cross_section_factor * num_bits * dt_seconds
        gen = rng or np.random.default_rng()
        return int(gen.poisson(rate))

    def tid_dose(self, dt_seconds: float) -> float:
        """Background TID accumulated over ``dt_seconds`` in krad(Si)."""
        return self.tid_rate_krad_per_day * (dt_seconds / 86400.0)

    # ------------------------------------------------------------------
    # Mission presets
    # ------------------------------------------------------------------

    @classmethod
    def cruise_1au_solar_min(cls) -> "HeliocentricEnvironment":
        """Generic cruise phase at 1 AU during solar minimum (worst-case GCR)."""
        return cls(heliocentric_distance_au=1.0, shielding_mm_al=2.0, solar_phase="min")

    @classmethod
    def cruise_1au_solar_max(cls) -> "HeliocentricEnvironment":
        """Generic cruise phase at 1 AU during solar maximum (best-case GCR)."""
        return cls(heliocentric_distance_au=1.0, shielding_mm_al=2.0, solar_phase="max")

    @classmethod
    def lunar_transfer(cls) -> "HeliocentricEnvironment":
        """Lunar transfer / cislunar — outside the magnetosphere at ~1 AU."""
        return cls(heliocentric_distance_au=1.0, shielding_mm_al=2.0, solar_phase="min")

    @classmethod
    def venus_flyby(cls) -> "HeliocentricEnvironment":
        """Venus orbit / flyby (~0.72 AU). No intrinsic magnetic field."""
        return cls(heliocentric_distance_au=0.72, shielding_mm_al=2.0, solar_phase="min")

    @classmethod
    def mars_transit(cls) -> "HeliocentricEnvironment":
        """Mars transit cruise at ~1.5 AU."""
        return cls(heliocentric_distance_au=1.5, shielding_mm_al=2.0, solar_phase="min")
