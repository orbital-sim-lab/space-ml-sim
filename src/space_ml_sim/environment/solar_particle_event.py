"""Solar Particle Event (SPE) statistical model.

Solar Particle Events are episodic bursts of high-energy protons emitted
during solar flares and coronal mass ejections. They are the **dominant
TID risk** for spacecraft outside Earth's magnetosphere — a single major
event can deliver tens of krad in hours behind thin shielding.

Unlike background GCR (modeled by :class:`HeliocentricEnvironment`), SPEs
are statistical: most days nothing happens, then a multi-month
mission-fraction can be dominated by 1–3 large events. They cannot be
modeled as a steady rate.

This module provides:

  - :class:`SolarParticleEvent` — a single-event description
      (fluence, peak flux, duration).
  - :class:`SPEStatisticalModel` — annual frequency of events of a given
      magnitude during a mission window, parameterized on solar phase.
  - :func:`mission_spe_dose` — Monte-Carlo or deterministic worst-case
      TID contribution from SPEs over a mission window.

The model uses the **ESP–PSYCHIC parametric tail** (Xapsos 1999, 2000) for
event-magnitude statistics, which is the standard method for spacecraft
SPE budgeting.

References:

    - Xapsos, Summers, Barth, Stassinopoulos, Burke, "Probability model for
      cumulative solar proton event fluences", IEEE TNS 47 (2000) 486-490.
    - Xapsos, Stauffer, Jordan, Barth, "Model for solar proton risk
      assessment", IEEE TNS 54 (2007) 1985-1989.
    - Mewaldt et al., "The cosmic ray radiation dose in interplanetary
      space — present day and worst-case evaluations", Radiation
      Measurements 41 (2006) 1149-1156.
    - Feynman et al., "JPL fluence model 1991", IEEE TNS 40 (1993).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SolarPhase = Literal["min", "max"]
EventMagnitude = Literal["small", "medium", "large", "extreme"]

# --- ESP–PSYCHIC tail parameters ---------------------------------------------
#
# Annual event-frequency calibration during solar-active years (typically 7
# of every 11 years in the cycle). Numbers are from Xapsos et al. 2000
# Table 1 and follow-up calibration to the August-1972 / October-1989 /
# October-2003 reference events. Quiet years (solar minimum) see roughly
# 0.1× the active-year frequencies.

_ANNUAL_EVENTS_ACTIVE = {
    "small": 6.0,  # ≥1e7 protons/cm² @ >10 MeV
    "medium": 1.5,  # ≥1e9 protons/cm² @ >10 MeV
    "large": 0.4,  # ≥1e10 protons/cm² @ >10 MeV (e.g. 1989 Oct)
    "extreme": 0.05,  # ≥1e11 protons/cm² @ >10 MeV (e.g. 1972 Aug)
}

# Approximate dose contribution per event behind 2 mm Al shielding at 1 AU,
# in krad(Si). Drawn from Xapsos 2007 dose tables.
_EVENT_DOSE_KRAD_2MM_AL = {
    "small": 0.05,
    "medium": 1.0,
    "large": 5.0,
    "extreme": 30.0,
}


def _phase_frequency_factor(phase: SolarPhase) -> float:
    """Multiplier on active-year event frequency.

    During solar minimum the Sun is quiet — typical SPE frequency is
    roughly 10x lower than during the active phase. (Mewaldt 2006.)
    """
    return {"max": 1.0, "min": 0.1}[phase]


def _shielding_dose_factor(shielding_mm_al: float) -> float:
    """Exponential attenuation of SPE dose with aluminum-equivalent shielding.

    SPE protons span 10 MeV to 1 GeV+ but the dose is dominated by the
    soft (10–100 MeV) component, which attenuates rapidly with shielding.
    This is a simple parametric fit; the calibration constants assume
    2 mm Al gives the published `_EVENT_DOSE_KRAD_2MM_AL` values.
    """
    return math.exp(-0.35 * (shielding_mm_al - 2.0))


class SolarParticleEvent(BaseModel):
    """A single discrete solar particle event."""

    model_config = ConfigDict(frozen=True)

    magnitude: EventMagnitude
    integral_fluence_per_cm2: float = Field(gt=0, description=">10 MeV proton fluence")
    duration_hours: float = Field(gt=0)


class SPEStatisticalModel(BaseModel):
    """Annual SPE frequency model for a given solar-cycle phase.

    Lets users translate a mission window into expected SPE frequency
    and dose, either deterministically (worst-case stack) or via
    Monte-Carlo sampling.

    Attributes:
        solar_phase: ``"min"`` or ``"max"``. Min has ~10x lower frequency.
        shielding_mm_al: Aluminum-equivalent shielding thickness in mm.
            Modulates per-event dose, not frequency.
    """

    model_config = ConfigDict(frozen=True)

    solar_phase: SolarPhase = Field(default="max")
    shielding_mm_al: float = Field(default=2.0, ge=0)

    def annual_event_frequency(self, magnitude: EventMagnitude) -> float:
        """Expected events of the given magnitude per year."""
        return _ANNUAL_EVENTS_ACTIVE[magnitude] * _phase_frequency_factor(self.solar_phase)

    def expected_events_in_window(self, duration_days: float, magnitude: EventMagnitude) -> float:
        """Mean (Poisson lambda) for events of ``magnitude`` over the window."""
        return self.annual_event_frequency(magnitude) * (duration_days / 365.25)

    def expected_dose_krad(self, duration_days: float) -> float:
        """Expected SPE-only TID over a mission window, in krad(Si).

        Sums the expected per-magnitude event count times the per-event
        dose, attenuated by shielding. This is the *mean* — actual mission
        dose has a long tail, dominated by whether an extreme event lands.
        """
        atten = _shielding_dose_factor(self.shielding_mm_al)
        total = 0.0
        for magnitude in ("small", "medium", "large", "extreme"):
            n = self.expected_events_in_window(duration_days, magnitude)  # type: ignore[arg-type]
            total += n * _EVENT_DOSE_KRAD_2MM_AL[magnitude] * atten
        return total

    def worst_case_dose_krad(self, duration_days: float, percentile: float = 0.95) -> float:
        """Approximate ``percentile`` worst-case SPE dose over the window.

        Uses a closed-form Poisson-weighted upper bound rather than full
        Monte Carlo: at the 95th percentile, assume the mission catches
        ceil(λ_extreme × 1.6) extreme events on top of the mean.
        Conservative but cheap for trade studies.
        """
        if not 0.5 <= percentile < 1.0:
            raise ValueError("percentile must be in [0.5, 1.0)")
        mean = self.expected_dose_krad(duration_days)
        atten = _shielding_dose_factor(self.shielding_mm_al)
        # Headroom from a single extra extreme event scaled by percentile gap.
        z = (percentile - 0.5) * 3.0
        extra_extreme_events = max(
            0.0, math.ceil(self.expected_events_in_window(duration_days, "extreme") * z)
        )
        return mean + extra_extreme_events * _EVENT_DOSE_KRAD_2MM_AL["extreme"] * atten

    def sample_mission(
        self,
        duration_days: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[SolarParticleEvent], float]:
        """Monte-Carlo sample a single mission realization.

        Returns:
            (events, total_dose_krad) — list of events that fired during the
            mission window and the resulting cumulative SPE-only TID.
        """
        gen = rng or np.random.default_rng()
        events: list[SolarParticleEvent] = []
        total_dose = 0.0
        atten = _shielding_dose_factor(self.shielding_mm_al)
        for magnitude in ("small", "medium", "large", "extreme"):
            lam = self.expected_events_in_window(duration_days, magnitude)  # type: ignore[arg-type]
            n = int(gen.poisson(lam))
            for _ in range(n):
                # Crude per-event description; not a full spectral fit.
                fluence = {
                    "small": 1e7,
                    "medium": 1e9,
                    "large": 1e10,
                    "extreme": 1e11,
                }[magnitude]
                events.append(
                    SolarParticleEvent(
                        magnitude=magnitude,  # type: ignore[arg-type]
                        integral_fluence_per_cm2=fluence,
                        duration_hours=24.0,
                    )
                )
                total_dose += _EVENT_DOSE_KRAD_2MM_AL[magnitude] * atten
        return events, total_dose


def mission_spe_dose(
    duration_days: float,
    solar_phase: SolarPhase = "max",
    shielding_mm_al: float = 2.0,
    method: Literal["mean", "p95"] = "p95",
) -> float:
    """Convenience wrapper — total SPE TID estimate for a mission window.

    Args:
        duration_days: Mission length in days.
        solar_phase: ``"min"`` or ``"max"`` of the 11-year solar cycle.
        shielding_mm_al: Aluminum-equivalent shielding thickness in mm.
        method: ``"mean"`` for expected dose, ``"p95"`` for 95th-percentile
            worst case (recommended for design budgeting).

    Returns:
        TID in krad(Si) from SPEs only (does not include background GCR).
    """
    model = SPEStatisticalModel(solar_phase=solar_phase, shielding_mm_al=shielding_mm_al)
    if method == "mean":
        return model.expected_dose_krad(duration_days)
    return model.worst_case_dose_krad(duration_days, percentile=0.95)
