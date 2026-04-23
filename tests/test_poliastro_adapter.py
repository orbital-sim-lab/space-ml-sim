"""Tests for poliastro orbit import adapter.

These tests validate the conversion from poliastro Orbit objects
to our OrbitConfig. Tests that require poliastro are skipped if
the package is not installed.
"""

from __future__ import annotations


import pytest

from space_ml_sim.core.orbit import OrbitConfig


# Check if poliastro is available
try:
    import poliastro  # noqa: F401

    HAS_POLIASTRO = True
except ImportError:
    HAS_POLIASTRO = False

skip_no_poliastro = pytest.mark.skipif(not HAS_POLIASTRO, reason="poliastro not installed")


class TestFromPoliastroImport:
    """Test that the adapter function exists and has correct signature."""

    def test_adapter_function_exists(self) -> None:
        from space_ml_sim.core.poliastro_adapter import from_poliastro

        assert callable(from_poliastro)

    def test_adapter_function_from_elements(self) -> None:
        """Test adapter with a mock-like dict when poliastro isn't available."""
        from space_ml_sim.core.poliastro_adapter import from_elements

        config = from_elements(
            altitude_km=500.0,
            inclination_deg=51.6,
            raan_deg=100.0,
            true_anomaly_deg=0.0,
        )
        assert isinstance(config, OrbitConfig)
        assert config.altitude_km == 500.0
        assert config.inclination_deg == 51.6
        assert config.raan_deg == 100.0

    @skip_no_poliastro
    def test_from_poliastro_iss_orbit(self) -> None:
        """Convert a poliastro ISS orbit to OrbitConfig."""
        from astropy import units as u
        from poliastro.bodies import Earth
        from poliastro.twobody import Orbit

        from space_ml_sim.core.poliastro_adapter import from_poliastro

        iss = Orbit.circular(Earth, alt=400 * u.km, inc=51.6 * u.deg)
        config = from_poliastro(iss)

        assert isinstance(config, OrbitConfig)
        assert abs(config.altitude_km - 400.0) < 5.0
        assert abs(config.inclination_deg - 51.6) < 0.5

    @skip_no_poliastro
    def test_from_poliastro_sso_orbit(self) -> None:
        """Convert a poliastro SSO orbit to OrbitConfig."""
        from astropy import units as u
        from poliastro.bodies import Earth
        from poliastro.twobody import Orbit

        from space_ml_sim.core.poliastro_adapter import from_poliastro

        sso = Orbit.circular(Earth, alt=650 * u.km, inc=98.0 * u.deg)
        config = from_poliastro(sso)

        assert abs(config.altitude_km - 650.0) < 5.0
        assert abs(config.inclination_deg - 98.0) < 0.5

    @skip_no_poliastro
    def test_roundtrip_preserves_elements(self) -> None:
        """Orbit elements should survive poliastro -> OrbitConfig conversion."""
        from astropy import units as u
        from poliastro.bodies import Earth
        from poliastro.twobody import Orbit

        from space_ml_sim.core.poliastro_adapter import from_poliastro

        orbit = Orbit.circular(Earth, alt=800 * u.km, inc=53.0 * u.deg, raan=120 * u.deg)
        config = from_poliastro(orbit)

        assert abs(config.altitude_km - 800.0) < 10.0
        assert abs(config.inclination_deg - 53.0) < 1.0
        assert abs(config.raan_deg - 120.0) < 1.0
