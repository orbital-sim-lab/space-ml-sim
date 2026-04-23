"""Validate radiation model against published SPENVIS reference data.

These tests compare our parametric radiation model's outputs against
published reference values from ESA's SPENVIS tool for standard orbits.
They ensure our simplified model stays within acceptable bounds of the
authoritative trapped particle and GCR models.

Reference data sources:
    - ESA SPENVIS AP-9/AE-9 reference orbits (publicly available)
    - Xapsos et al., "Probability Model for Cumulative Solar Proton
      Event Fluences", IEEE TNS, 2000
    - Barth et al., "Space, Atmospheric, and Terrestrial Radiation
      Environments", IEEE TNS, 2003
    - ECSS-E-ST-10-04C, "Space environment", ESA, 2020

Tolerance philosophy:
    Our parametric model is intentionally simplified for simulation speed.
    We accept order-of-magnitude agreement with SPENVIS for TID and SEU
    rates. The goal is "physically reasonable", not "flight qualification".
"""

from __future__ import annotations


import pytest

from space_ml_sim.environment.radiation import RadiationEnvironment


# ============================================================================
# Published SPENVIS reference values for standard orbits
# ============================================================================
#
# These are approximate annual TID values behind 2mm Al shielding from
# SPENVIS AP-9/AE-9 runs at solar maximum, widely cited in literature.
# Sources: Barth et al. 2003, ECSS-E-ST-10-04C Table A-1

SPENVIS_REFERENCES = {
    # (altitude_km, inclination_deg): tid_krad_per_year (behind 2mm Al)
    "leo_400km_51deg": {
        "alt": 400,
        "inc": 51.6,
        "tid_krad_yr_low": 0.05,
        "tid_krad_yr_high": 0.5,
        "notes": "ISS orbit, low radiation, SAA dominant",
    },
    "leo_800km_98deg": {
        "alt": 800,
        "inc": 98.0,
        "tid_krad_yr_low": 0.5,
        "tid_krad_yr_high": 5.0,
        "notes": "Sun-synchronous, significant trapped protons",
    },
    "leo_1400km_53deg": {
        "alt": 1400,
        "inc": 53.0,
        "tid_krad_yr_low": 2.0,
        "tid_krad_yr_high": 30.0,
        "notes": "Approaching inner belt, high radiation",
    },
}

# Published SEU rate ranges for LEO (upsets/bit/day) from Petersen 2011
# "Single Event Effects in Aerospace" and JEDEC JESD89B
# These are for a typical SRAM with cross-section ~1e-14 cm^2
SEU_REFERENCE_RANGES = {
    "leo_500km": {
        "alt": 500,
        "inc": 53.0,
        "seu_per_bit_per_day_low": 1e-12,
        "seu_per_bit_per_day_high": 1e-6,
        "notes": "GCR + SAA protons at 53deg inclination, wide range due to geometry",
    },
    "leo_2000km": {
        "alt": 2000,
        "inc": 53.0,
        "seu_per_bit_per_day_low": 1e-10,
        "seu_per_bit_per_day_high": 5e-6,
        "notes": "Near inner Van Allen belt, high trapped proton rates",
    },
}


class TestTIDAgainstSPENVIS:
    """Our TID model must produce values within the published SPENVIS range.

    We use generous bounds (order-of-magnitude) because:
    1. Our model is parametric, not a full particle transport calculation
    2. SPENVIS values themselves vary with solar cycle assumptions
    3. Shielding geometry effects are simplified to 1D slab
    """

    @pytest.mark.parametrize(
        "ref_key",
        list(SPENVIS_REFERENCES.keys()),
        ids=list(SPENVIS_REFERENCES.keys()),
    )
    def test_tid_within_spenvis_range(self, ref_key: str) -> None:
        ref = SPENVIS_REFERENCES[ref_key]
        env = RadiationEnvironment(
            altitude_km=ref["alt"],
            inclination_deg=ref["inc"],
            shielding_mm_al=2.0,
        )
        tid_krad_per_year = env.tid_rate_krad_per_day * 365.25

        assert tid_krad_per_year >= ref["tid_krad_yr_low"] * 0.1, (
            f"{ref_key}: TID {tid_krad_per_year:.4f} krad/yr is more than 10x "
            f"below SPENVIS lower bound {ref['tid_krad_yr_low']} krad/yr"
        )
        assert tid_krad_per_year <= ref["tid_krad_yr_high"] * 10, (
            f"{ref_key}: TID {tid_krad_per_year:.4f} krad/yr is more than 10x "
            f"above SPENVIS upper bound {ref['tid_krad_yr_high']} krad/yr"
        )

    def test_tid_increases_with_altitude(self) -> None:
        """TID must increase monotonically with altitude (fundamental physics)."""
        altitudes = [400, 600, 800, 1000, 1500, 2000]
        tids = []
        for alt in altitudes:
            env = RadiationEnvironment(altitude_km=alt, inclination_deg=53.0, shielding_mm_al=2.0)
            tids.append(env.tid_rate_krad_per_day)

        for i in range(1, len(tids)):
            assert tids[i] > tids[i - 1], (
                f"TID did not increase from {altitudes[i - 1]}km "
                f"({tids[i - 1]:.6f}) to {altitudes[i]}km ({tids[i]:.6f})"
            )

    def test_tid_decreases_with_shielding(self) -> None:
        """TID must decrease with more shielding (fundamental physics)."""
        shields = [0.5, 1.0, 2.0, 5.0, 10.0]
        tids = []
        for s in shields:
            env = RadiationEnvironment(altitude_km=800, inclination_deg=98.0, shielding_mm_al=s)
            tids.append(env.tid_rate_krad_per_day)

        for i in range(1, len(tids)):
            assert tids[i] < tids[i - 1], (
                f"TID did not decrease from {shields[i - 1]}mm "
                f"({tids[i - 1]:.6f}) to {shields[i]}mm ({tids[i]:.6f})"
            )

    def test_2000km_tid_much_higher_than_500km(self) -> None:
        """Inner belt proximity at 2000km should give >>10x TID vs 500km.

        This is a well-established result from trapped proton models.
        """
        env_500 = RadiationEnvironment(altitude_km=500, inclination_deg=53.0, shielding_mm_al=2.0)
        env_2000 = RadiationEnvironment(altitude_km=2000, inclination_deg=53.0, shielding_mm_al=2.0)
        ratio = env_2000.tid_rate_krad_per_day / env_500.tid_rate_krad_per_day
        assert ratio > 10, f"2000km/500km TID ratio is only {ratio:.1f}x, expected >10x"


class TestSEURatesAgainstLiterature:
    """Our SEU rates must be physically reasonable compared to published data."""

    @pytest.mark.parametrize(
        "ref_key",
        list(SEU_REFERENCE_RANGES.keys()),
        ids=list(SEU_REFERENCE_RANGES.keys()),
    )
    def test_seu_rate_within_literature_range(self, ref_key: str) -> None:
        """SEU rate per bit per day must fall within published bounds."""
        ref = SEU_REFERENCE_RANGES[ref_key]
        env = RadiationEnvironment(
            altitude_km=ref["alt"],
            inclination_deg=ref["inc"],
            shielding_mm_al=2.0,
        )
        # base_seu_rate is upsets/bit/second at reference cross-section 1e-14
        # Convert to per-day at reference cross-section
        seu_per_bit_per_day = env.base_seu_rate * 86400.0

        assert seu_per_bit_per_day >= ref["seu_per_bit_per_day_low"], (
            f"{ref_key}: SEU rate {seu_per_bit_per_day:.2e}/bit/day below "
            f"literature minimum {ref['seu_per_bit_per_day_low']:.2e}"
        )
        assert seu_per_bit_per_day <= ref["seu_per_bit_per_day_high"], (
            f"{ref_key}: SEU rate {seu_per_bit_per_day:.2e}/bit/day above "
            f"literature maximum {ref['seu_per_bit_per_day_high']:.2e}"
        )

    def test_seu_increases_with_altitude(self) -> None:
        """SEU rate must increase with altitude overall (more particle flux)."""
        env_500 = RadiationEnvironment(altitude_km=500, inclination_deg=53.0, shielding_mm_al=2.0)
        env_2000 = RadiationEnvironment(altitude_km=2000, inclination_deg=53.0, shielding_mm_al=2.0)
        assert env_2000.base_seu_rate > env_500.base_seu_rate

    def test_saa_enhancement_for_mid_inclination(self) -> None:
        """SAA should enhance SEU rates for ISS-like inclinations vs polar."""
        # ISS-like (passes through SAA)
        env_iss = RadiationEnvironment(altitude_km=500, inclination_deg=51.6, shielding_mm_al=2.0)
        # High inclination (less SAA exposure)
        env_polar = RadiationEnvironment(altitude_km=500, inclination_deg=10.0, shielding_mm_al=2.0)
        assert env_iss.base_seu_rate > env_polar.base_seu_rate, (
            "SAA enhancement should make ISS-inclination SEU rate higher than "
            "equatorial orbit at same altitude"
        )


class TestSEUPoissonStatistics:
    """Validate that SEU sampling follows correct Poisson statistics.

    The expected value of a Poisson-sampled SEU count should converge to
    the analytical mean over many samples.
    """

    def test_mean_converges_to_expected(self) -> None:
        """Mean of N Poisson samples should converge to analytical lambda."""
        import numpy as np

        env = RadiationEnvironment(altitude_km=800, inclination_deg=53.0, shielding_mm_al=2.0)
        chip_xsec = 5e-13  # Trillium-like
        num_bits = 1_000_000
        dt = 3600.0  # 1 hour

        rng = np.random.default_rng(42)
        n_samples = 10_000
        counts = [env.sample_seu_events(chip_xsec, num_bits, dt, rng=rng) for _ in range(n_samples)]

        sample_mean = sum(counts) / n_samples

        # Analytical expected value
        ref_xsec = 1e-14
        xsec_factor = chip_xsec / ref_xsec
        analytical_mean = env.base_seu_rate * xsec_factor * num_bits * dt

        # With 10k samples, the sample mean should be within 5% of analytical
        # (for large enough lambda this is very likely)
        if analytical_mean > 1.0:
            rel_error = abs(sample_mean - analytical_mean) / analytical_mean
            assert rel_error < 0.05, (
                f"Sample mean {sample_mean:.2f} differs from analytical "
                f"{analytical_mean:.2f} by {rel_error:.1%}"
            )
        else:
            # For small lambda, use absolute tolerance
            assert abs(sample_mean - analytical_mean) < 1.0


class TestShieldingPhysics:
    """Validate shielding attenuation behaves physically."""

    def test_zero_shielding_gives_maximum_rates(self) -> None:
        """No shielding should give the highest radiation rates."""
        env_0 = RadiationEnvironment(altitude_km=800, inclination_deg=53.0, shielding_mm_al=0.0)
        env_2 = RadiationEnvironment(altitude_km=800, inclination_deg=53.0, shielding_mm_al=2.0)
        assert env_0.base_seu_rate > env_2.base_seu_rate
        assert env_0.tid_rate_krad_per_day > env_2.tid_rate_krad_per_day

    def test_heavy_shielding_significantly_reduces_tid(self) -> None:
        """10mm Al should reduce TID by at least 90% vs 0.5mm."""
        env_light = RadiationEnvironment(altitude_km=800, inclination_deg=53.0, shielding_mm_al=0.5)
        env_heavy = RadiationEnvironment(
            altitude_km=800, inclination_deg=53.0, shielding_mm_al=10.0
        )
        reduction = 1.0 - (env_heavy.tid_rate_krad_per_day / env_light.tid_rate_krad_per_day)
        assert reduction > 0.9, f"10mm Al only reduced TID by {reduction:.0%}, expected >90%"

    def test_shielding_attenuation_is_monotonic(self) -> None:
        """More shielding must always reduce rates (no anomalous behavior)."""
        shields = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
        rates = []
        for s in shields:
            env = RadiationEnvironment(altitude_km=600, inclination_deg=53.0, shielding_mm_al=s)
            rates.append(env.tid_rate_krad_per_day)

        for i in range(1, len(rates)):
            assert rates[i] < rates[i - 1], (
                f"TID rate not monotonically decreasing: "
                f"{shields[i - 1]}mm={rates[i - 1]:.6f} vs {shields[i]}mm={rates[i]:.6f}"
            )
