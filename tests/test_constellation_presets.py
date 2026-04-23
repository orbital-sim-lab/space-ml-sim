"""TDD tests for constellation presets."""

from __future__ import annotations


class TestConstellationPresets:
    """Pre-built constellation configurations for popular constellations."""

    def test_starlink_preset(self) -> None:
        from space_ml_sim.core.constellation_presets import CONSTELLATION_PRESETS

        starlink = CONSTELLATION_PRESETS["starlink_shell1"]
        assert starlink.num_satellites > 0
        assert 540 <= starlink.altitude_km <= 560
        assert 52 <= starlink.inclination_deg <= 54

    def test_oneweb_preset(self) -> None:
        from space_ml_sim.core.constellation_presets import CONSTELLATION_PRESETS

        oneweb = CONSTELLATION_PRESETS["oneweb"]
        assert oneweb.num_satellites > 0
        assert 1100 <= oneweb.altitude_km <= 1300

    def test_kuiper_preset(self) -> None:
        from space_ml_sim.core.constellation_presets import CONSTELLATION_PRESETS

        kuiper = CONSTELLATION_PRESETS["kuiper_shell1"]
        assert kuiper.num_satellites > 0

    def test_generate_orbits_from_preset(self) -> None:
        from space_ml_sim.core.constellation_presets import (
            CONSTELLATION_PRESETS,
            generate_from_preset,
        )

        orbits = generate_from_preset("starlink_shell1")
        preset = CONSTELLATION_PRESETS["starlink_shell1"]
        assert len(orbits) == preset.num_satellites

    def test_list_all_presets(self) -> None:
        from space_ml_sim.core.constellation_presets import CONSTELLATION_PRESETS

        assert len(CONSTELLATION_PRESETS) >= 3
