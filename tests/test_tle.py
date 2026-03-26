"""Tests for TLE ingestion and sgp4-based propagation.

TDD order: tests written first, implementation follows.

ISS TLE used throughout (epoch ~2024-02-14):
  1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993
  2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075

Expected orbital characteristics (ISS):
  - Inclination: ~51.6 deg
  - Altitude: ~400-420 km
  - Orbital period: ~92 min  (mean motion ≈ 15.5 rev/day)
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

# --- ISS sample TLE ---
ISS_LINE1 = "1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075"
ISS_NAME = "ISS (ZARYA)"

# A second TLE for multi-satellite file tests (Hubble Space Telescope)
HST_LINE1 = "1 20580U 90037B   24045.12345678  .00000600  00000+0  25000-4 0  9991"
HST_LINE2 = "2 20580  28.4700  45.0000 0002500  90.0000 270.0000 15.09177300123456"
HST_NAME = "HST"

# A deliberately malformed TLE line
BAD_LINE1 = "THIS IS NOT A VALID TLE LINE 1"
BAD_LINE2 = "2 99999  99.9999 999.9999 9999999 999.9999 999.9999 00.00000000000000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tle_file(entries: list[tuple[str, str, str]], path: str) -> None:
    """Write a standard 3-line TLE file: name / line1 / line2 per entry."""
    with open(path, "w") as fh:
        for name, l1, l2 in entries:
            fh.write(f"{name}\n{l1}\n{l2}\n")


# ---------------------------------------------------------------------------
# parse_tle
# ---------------------------------------------------------------------------


class TestParseTle:
    """Unit tests for parse_tle(line1, line2) -> OrbitConfig."""

    def test_returns_orbit_config(self):
        """parse_tle must return an OrbitConfig instance."""
        from space_ml_sim.core.tle import parse_tle
        from space_ml_sim.core.orbit import OrbitConfig

        result = parse_tle(ISS_LINE1, ISS_LINE2)
        assert isinstance(result, OrbitConfig)

    def test_iss_inclination_approx_51_6(self):
        """ISS inclination must be within 0.01 deg of 51.6416 deg."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        assert abs(config.inclination_deg - 51.6416) < 0.01

    def test_iss_altitude_in_range(self):
        """ISS altitude derived from mean motion must be ~400-430 km.

        The exact altitude depends on the TLE epoch and eccentricity.
        The mean semi-major axis gives an average altitude which may exceed
        the typical 408-420km periapsis range.
        """
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        assert 390.0 <= config.altitude_km <= 440.0, (
            f"Expected ISS altitude ~400-430 km, got {config.altitude_km:.1f} km"
        )

    def test_iss_raan_approx_247(self):
        """RAAN must be parsed from TLE line 2 field (247.4627 deg)."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        assert abs(config.raan_deg - 247.4627) < 0.01

    def test_true_anomaly_in_valid_range(self):
        """True anomaly converted from mean anomaly must be in [0, 360)."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        assert 0.0 <= config.true_anomaly_deg < 360.0

    def test_true_anomaly_near_mean_anomaly_for_low_eccentricity(self):
        """For near-circular orbits (ecc ≈ 0), true anomaly ≈ mean anomaly."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        # ISS eccentricity is ~0.0007, so difference should be < 1 deg
        mean_anomaly_deg = 229.6116  # from TLE line 2
        diff = abs(config.true_anomaly_deg - mean_anomaly_deg)
        # Normalise to [0, 180]
        diff = min(diff, 360.0 - diff)
        assert diff < 1.0, (
            f"Expected true anomaly close to mean anomaly for near-circular orbit, "
            f"true={config.true_anomaly_deg:.4f} mean={mean_anomaly_deg}"
        )

    def test_altitude_positive(self):
        """Altitude must always be positive (OrbitConfig field constraint)."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(ISS_LINE1, ISS_LINE2)
        assert config.altitude_km > 0

    def test_invalid_tle_raises_value_error(self):
        """Garbage TLE strings must raise ValueError."""
        from space_ml_sim.core.tle import parse_tle

        with pytest.raises(ValueError, match="[Ii]nvalid|[Pp]arse|[Ff]ail|[Ee]rror|[Tt]LE"):
            parse_tle(BAD_LINE1, BAD_LINE2)

    def test_empty_strings_raise_value_error(self):
        """Empty TLE strings must raise ValueError."""
        from space_ml_sim.core.tle import parse_tle

        with pytest.raises(ValueError):
            parse_tle("", "")

    def test_whitespace_only_raises_value_error(self):
        """Whitespace-only TLE strings must raise ValueError."""
        from space_ml_sim.core.tle import parse_tle

        with pytest.raises(ValueError):
            parse_tle("   ", "   ")

    def test_swapped_lines_raise_value_error(self):
        """Passing TLE lines in wrong order must raise ValueError."""
        from space_ml_sim.core.tle import parse_tle

        with pytest.raises(ValueError):
            parse_tle(ISS_LINE2, ISS_LINE1)  # intentionally swapped

    def test_hst_inclination_approx_28_5(self):
        """HST inclination must be within 0.1 deg of 28.47 deg."""
        from space_ml_sim.core.tle import parse_tle

        config = parse_tle(HST_LINE1, HST_LINE2)
        assert abs(config.inclination_deg - 28.47) < 0.1


# ---------------------------------------------------------------------------
# load_tle_file
# ---------------------------------------------------------------------------


class TestLoadTleFile:
    """Unit tests for load_tle_file(path) -> list[OrbitConfig]."""

    def test_loads_single_entry(self):
        """A file with one TLE entry returns a list with one OrbitConfig."""
        from space_ml_sim.core.tle import load_tle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n")
            path = fh.name

        try:
            configs = load_tle_file(path)
            assert len(configs) == 1
            assert isinstance(configs[0].inclination_deg, float)
        finally:
            os.unlink(path)

    def test_loads_multiple_entries(self):
        """A file with two TLE entries returns a list with two OrbitConfigs."""
        from space_ml_sim.core.tle import load_tle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n")
            fh.write(f"{HST_NAME}\n{HST_LINE1}\n{HST_LINE2}\n")
            path = fh.name

        try:
            configs = load_tle_file(path)
            assert len(configs) == 2
        finally:
            os.unlink(path)

    def test_entry_values_match_parse_tle(self):
        """OrbitConfig values from file must match direct parse_tle output."""
        from space_ml_sim.core.tle import load_tle_file, parse_tle

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n")
            path = fh.name

        try:
            file_config = load_tle_file(path)[0]
            direct_config = parse_tle(ISS_LINE1, ISS_LINE2)
            assert file_config.inclination_deg == direct_config.inclination_deg
            assert file_config.altitude_km == direct_config.altitude_km
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises_os_error(self):
        """A path that does not exist must raise OSError (or FileNotFoundError)."""
        from space_ml_sim.core.tle import load_tle_file

        with pytest.raises((OSError, FileNotFoundError)):
            load_tle_file("/tmp/this_file_does_not_exist_space_ml_sim.txt")

    def test_empty_file_returns_empty_list(self):
        """An empty file must return an empty list, not raise."""
        from space_ml_sim.core.tle import load_tle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            path = fh.name  # write nothing

        try:
            configs = load_tle_file(path)
            assert configs == []
        finally:
            os.unlink(path)

    def test_file_with_trailing_blank_lines(self):
        """Extra blank lines at end of file must not cause extra entries or errors."""
        from space_ml_sim.core.tle import load_tle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n\n\n")
            path = fh.name

        try:
            configs = load_tle_file(path)
            assert len(configs) == 1
        finally:
            os.unlink(path)

    def test_returns_list_type(self):
        """Return type must be a list, not a generator or other iterable."""
        from space_ml_sim.core.tle import load_tle_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n")
            path = fh.name

        try:
            result = load_tle_file(path)
            assert isinstance(result, list)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# propagate_sgp4
# ---------------------------------------------------------------------------


class TestPropagateSgp4:
    """Unit tests for propagate_sgp4(line1, line2, minutes) -> (x, y, z)."""

    def test_returns_three_element_tuple(self):
        """propagate_sgp4 must return a 3-tuple of floats."""
        from space_ml_sim.core.tle import propagate_sgp4

        result = propagate_sgp4(ISS_LINE1, ISS_LINE2, 0.0)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_position_at_epoch_is_at_orbital_altitude(self):
        """At epoch (t=0), position magnitude must be near ISS orbital radius (~6780 km)."""
        from space_ml_sim.core.tle import propagate_sgp4

        x, y, z = propagate_sgp4(ISS_LINE1, ISS_LINE2, 0.0)
        r = math.sqrt(x**2 + y**2 + z**2)
        # ISS: altitude 410 km -> radius ~6781 km; allow ±50 km tolerance
        assert 6700.0 <= r <= 6850.0, f"Orbital radius {r:.1f} km out of expected range"

    def test_position_changes_with_time(self):
        """Position at t=45 min must differ from position at t=0."""
        from space_ml_sim.core.tle import propagate_sgp4

        pos0 = propagate_sgp4(ISS_LINE1, ISS_LINE2, 0.0)
        pos45 = propagate_sgp4(ISS_LINE1, ISS_LINE2, 45.0)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos0, pos45)))
        # After 45 min (≈ half orbit) satellite should be far from initial position
        assert dist > 1000.0, f"Satellite moved only {dist:.1f} km in 45 min"

    def test_position_radius_consistent_over_orbit(self):
        """Orbital radius must stay within ±100 km across multiple points (circular orbit)."""
        from space_ml_sim.core.tle import propagate_sgp4

        radii = []
        for t in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
            x, y, z = propagate_sgp4(ISS_LINE1, ISS_LINE2, t)
            radii.append(math.sqrt(x**2 + y**2 + z**2))

        assert max(radii) - min(radii) < 100.0, (
            f"Radius variation {max(radii) - min(radii):.1f} km exceeds 100 km threshold"
        )

    def test_zero_minutes_from_epoch(self):
        """t=0.0 must succeed and return a non-zero position."""
        from space_ml_sim.core.tle import propagate_sgp4

        x, y, z = propagate_sgp4(ISS_LINE1, ISS_LINE2, 0.0)
        assert x != 0.0 or y != 0.0 or z != 0.0

    def test_negative_minutes_before_epoch(self):
        """Negative time (before epoch) must return a valid position tuple."""
        from space_ml_sim.core.tle import propagate_sgp4

        result = propagate_sgp4(ISS_LINE1, ISS_LINE2, -10.0)
        assert len(result) == 3
        x, y, z = result
        r = math.sqrt(x**2 + y**2 + z**2)
        assert 6700.0 <= r <= 6850.0

    def test_invalid_tle_raises_value_error(self):
        """Invalid TLE must raise ValueError."""
        from space_ml_sim.core.tle import propagate_sgp4

        with pytest.raises(ValueError):
            propagate_sgp4(BAD_LINE1, BAD_LINE2, 0.0)


# ---------------------------------------------------------------------------
# Constellation.from_tle
# ---------------------------------------------------------------------------


class TestConstellationFromTle:
    """Integration tests for Constellation.from_tle class method."""

    def test_creates_correct_number_of_satellites(self):
        """from_tle with 2 TLE pairs must create a constellation with 2 satellites."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [
            (ISS_LINE1, ISS_LINE2),
            (HST_LINE1, HST_LINE2),
        ]
        constellation = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        assert len(constellation.satellites) == 2

    def test_single_tle_pair_creates_one_satellite(self):
        """A single TLE pair must yield exactly one satellite."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [(ISS_LINE1, ISS_LINE2)]
        constellation = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        assert len(constellation.satellites) == 1

    def test_satellites_have_correct_inclination(self):
        """Satellites created from ISS TLE must have ISS inclination."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [(ISS_LINE1, ISS_LINE2)]
        constellation = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        sat = constellation.satellites[0]
        assert abs(sat.orbit_config.inclination_deg - 51.6416) < 0.01

    def test_returns_constellation_instance(self):
        """from_tle must return a Constellation object."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [(ISS_LINE1, ISS_LINE2)]
        result = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        assert isinstance(result, Constellation)

    def test_empty_tle_pairs_creates_empty_constellation(self):
        """An empty list of TLE pairs must produce a constellation with no satellites."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        constellation = Constellation.from_tle([], chip_profile=TERAFAB_D3)
        assert len(constellation.satellites) == 0

    def test_satellites_are_steppable(self):
        """Constellation built from TLE must be steppable without error."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [(ISS_LINE1, ISS_LINE2)]
        constellation = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        # Should not raise
        metrics = constellation.step(dt_seconds=60.0)
        assert "active_count" in metrics

    def test_satellite_ids_are_unique(self):
        """Each satellite in the constellation must have a unique ID."""
        from space_ml_sim.core.constellation import Constellation
        from space_ml_sim.models.chip_profiles import TERAFAB_D3

        tle_pairs = [
            (ISS_LINE1, ISS_LINE2),
            (HST_LINE1, HST_LINE2),
        ]
        constellation = Constellation.from_tle(tle_pairs, chip_profile=TERAFAB_D3)
        ids = [s.id for s in constellation.satellites]
        assert len(ids) == len(set(ids)), "Satellite IDs must be unique"


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify that parse_tle, load_tle_file, propagate_sgp4 are exported from core."""

    def test_parse_tle_exported_from_core(self):
        """parse_tle must be importable from space_ml_sim.core."""
        from space_ml_sim.core import parse_tle  # noqa: F401

    def test_load_tle_file_exported_from_core(self):
        """load_tle_file must be importable from space_ml_sim.core."""
        from space_ml_sim.core import load_tle_file  # noqa: F401

    def test_propagate_sgp4_exported_from_core(self):
        """propagate_sgp4 must be importable from space_ml_sim.core."""
        from space_ml_sim.core import propagate_sgp4  # noqa: F401
