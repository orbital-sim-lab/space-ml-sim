"""Tests for MEO/GEO radiation environment extensions.

Van Allen belts create a complex radiation profile:
- LEO (200-2000km): Low trapped flux, GCR-dominated
- Inner belt peak (~3000-6000km): Intense trapped protons
- Slot region (~6000-13000km): Lower flux
- Outer belt peak (~15000-25000km): Intense trapped electrons
- GEO (~35786km): Moderate, electron-dominated
"""

from __future__ import annotations


from space_ml_sim.environment.radiation import RadiationEnvironment


class TestMEORadiation:
    """MEO orbits must have higher radiation than LEO."""

    def test_meo_orbit_creates(self) -> None:
        """MEO altitude should be accepted."""
        env = RadiationEnvironment(altitude_km=20200, inclination_deg=55, shielding_mm_al=2.0)
        assert env.base_seu_rate > 0
        assert env.tid_rate_krad_per_day > 0

    def test_inner_belt_peak(self) -> None:
        """Inner belt (~3000-6000km) should have highest proton flux."""
        env_leo = RadiationEnvironment(altitude_km=500, inclination_deg=53, shielding_mm_al=2.0)
        env_inner = RadiationEnvironment(altitude_km=4000, inclination_deg=53, shielding_mm_al=2.0)

        assert env_inner.tid_rate_krad_per_day > env_leo.tid_rate_krad_per_day * 5

    def test_slot_region_lower_than_inner(self) -> None:
        """Slot region (~10000km) should have less flux than inner belt peak."""
        env_inner = RadiationEnvironment(altitude_km=4000, inclination_deg=53, shielding_mm_al=2.0)
        env_slot = RadiationEnvironment(altitude_km=10000, inclination_deg=53, shielding_mm_al=2.0)

        assert env_slot.tid_rate_krad_per_day < env_inner.tid_rate_krad_per_day

    def test_gps_orbit(self) -> None:
        """GPS orbit (~20200km) should be characterized."""
        env = RadiationEnvironment(altitude_km=20200, inclination_deg=55, shielding_mm_al=3.0)
        assert env.base_seu_rate > 0
        assert env.tid_rate_krad_per_day > 0


class TestGEORadiation:
    """GEO orbits must have moderate radiation."""

    def test_geo_orbit_creates(self) -> None:
        env = RadiationEnvironment(altitude_km=35786, inclination_deg=0, shielding_mm_al=2.0)
        assert env.base_seu_rate > 0
        assert env.tid_rate_krad_per_day > 0

    def test_geo_lower_than_outer_belt(self) -> None:
        """GEO should have less trapped flux than outer belt peak."""
        env_outer = RadiationEnvironment(altitude_km=20000, inclination_deg=0, shielding_mm_al=2.0)
        env_geo = RadiationEnvironment(altitude_km=35786, inclination_deg=0, shielding_mm_al=2.0)

        assert env_geo.tid_rate_krad_per_day < env_outer.tid_rate_krad_per_day


class TestRadiationPresets:
    """Factory presets for standard orbit regimes."""

    def test_meo_preset(self) -> None:
        from space_ml_sim.environment.radiation import RadiationEnvironment

        env = RadiationEnvironment.meo_20200km()
        assert 20000 < env.altitude_km < 21000

    def test_geo_preset(self) -> None:
        from space_ml_sim.environment.radiation import RadiationEnvironment

        env = RadiationEnvironment.geo()
        assert 35000 < env.altitude_km < 36000
