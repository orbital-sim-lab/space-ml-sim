"""Constellation management for groups of satellites."""

from __future__ import annotations

import math
from typing import Any

from space_ml_sim.core.orbit import (
    position_at,
    walker_delta_orbits,
    sun_synchronous_orbits,
    is_in_eclipse,
)
from space_ml_sim.core.satellite import Satellite, SatelliteState
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import ChipProfile


class Constellation:
    """A collection of satellites forming an orbital constellation.

    Manages bulk operations: propagation, environment updates, and metrics.
    """

    def __init__(
        self,
        satellites: list[Satellite],
        rad_env: RadiationEnvironment | None = None,
    ) -> None:
        self.satellites = list(satellites)
        self.rad_env = rad_env or RadiationEnvironment.leo_500km()
        self._sim_time = 0.0

    @classmethod
    def walker_delta(
        cls,
        num_planes: int,
        sats_per_plane: int,
        altitude_km: float,
        inclination_deg: float,
        chip_profile: ChipProfile,
        phasing: int = 1,
        rad_env: RadiationEnvironment | None = None,
    ) -> "Constellation":
        """Create a Walker-Delta constellation.

        Args:
            num_planes: Number of orbital planes.
            sats_per_plane: Satellites per plane.
            altitude_km: Orbital altitude in km.
            inclination_deg: Orbital inclination in degrees.
            chip_profile: Hardware profile for all satellites.
            phasing: Walker phasing parameter.
            rad_env: Radiation environment (defaults to altitude-matched).

        Returns:
            Constellation with all satellites initialized.
        """
        orbits = walker_delta_orbits(
            num_planes, sats_per_plane, altitude_km, inclination_deg, phasing
        )
        satellites = [
            Satellite(
                id=f"WD-P{i // sats_per_plane:02d}-S{i % sats_per_plane:02d}",
                orbit_config=orbit,
                chip_profile=chip_profile,
            )
            for i, orbit in enumerate(orbits)
        ]
        env = rad_env or RadiationEnvironment(
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
        )
        return cls(satellites=satellites, rad_env=env)

    @classmethod
    def sun_synchronous(
        cls,
        num_sats: int,
        altitude_km: float,
        chip_profile: ChipProfile,
        ltan_hours: float = 6.0,
        rad_env: RadiationEnvironment | None = None,
    ) -> "Constellation":
        """Create a sun-synchronous constellation.

        Args:
            num_sats: Number of satellites.
            altitude_km: Orbital altitude in km.
            chip_profile: Hardware profile for all satellites.
            ltan_hours: Local time of ascending node.
            rad_env: Radiation environment (defaults to altitude-matched).

        Returns:
            Constellation with SSO satellites.
        """
        orbits = sun_synchronous_orbits(num_sats, altitude_km, ltan_hours)
        satellites = [
            Satellite(
                id=f"SSO-{i:03d}",
                orbit_config=orbit,
                chip_profile=chip_profile,
            )
            for i, orbit in enumerate(orbits)
        ]
        inc = orbits[0].inclination_deg if orbits else 98.0
        env = rad_env or RadiationEnvironment(altitude_km=altitude_km, inclination_deg=inc)
        return cls(satellites=satellites, rad_env=env)

    def step(
        self,
        dt_seconds: float,
        sun_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        compute_load_fraction: float = 1.0,
    ) -> dict[str, Any]:
        """Advance all satellites by one time step.

        For each satellite:
            1. Propagate orbit position
            2. Check eclipse state
            3. Update power
            4. Update thermal
            5. Apply radiation tick

        Args:
            dt_seconds: Time step in seconds.
            sun_direction: Unit vector from Earth to Sun in ECI.
            compute_load_fraction: Fraction of compute capacity in use.

        Returns:
            Metrics dict with counts and averages.
        """
        self._sim_time += dt_seconds
        updated: list[Satellite] = []

        for sat in self.satellites:
            # 1. Compute position at current sim time
            pos = position_at(sat.orbit_config, self._sim_time)

            # 2. Eclipse check
            eclipse = is_in_eclipse(pos, sun_direction)

            # 3-5. Update satellite state (immutable chain)
            new_sat = (
                sat.with_position(pos, eclipse)
                .with_power_update(eclipse)
                .with_thermal_update(compute_load_fraction, eclipse)
                .with_radiation_tick(self.rad_env, dt_seconds)
            )
            updated.append(new_sat)

        self.satellites = updated

        # Compute metrics in a single pass
        active = degraded = failed = total_seus = 0
        temp_sum = 0.0
        temp_count = 0
        for s in self.satellites:
            if s.state == SatelliteState.NOMINAL:
                active += 1
            elif s.state == SatelliteState.DEGRADED:
                degraded += 1
            else:
                failed += 1
            if s.is_operational:
                temp_sum += s.temperature_c
                temp_count += 1
            total_seus += s.total_seu_events
        avg_temp = temp_sum / temp_count if temp_count > 0 else 0.0

        return {
            "sim_time": self._sim_time,
            "active_count": active,
            "degraded_count": degraded,
            "failed_count": failed,
            "avg_temperature_c": round(avg_temp, 2),
            "total_seus": total_seus,
        }

    def get_isl_pairs(self, max_distance_km: float = 5000.0) -> list[tuple[str, str, float]]:
        """Find satellite pairs within inter-satellite link range.

        Args:
            max_distance_km: Maximum distance for ISL connectivity.

        Returns:
            List of (sat_id_a, sat_id_b, distance_km) tuples.
        """
        pairs: list[tuple[str, str, float]] = []
        n = len(self.satellites)

        for i in range(n):
            for j in range(i + 1, n):
                a = self.satellites[i]
                b = self.satellites[j]
                dist = math.sqrt(
                    sum((pa - pb) ** 2 for pa, pb in zip(a.position_km, b.position_km))
                )
                if dist <= max_distance_km:
                    pairs.append((a.id, b.id, round(dist, 2)))

        return pairs

    @classmethod
    def from_tle(
        cls,
        tle_pairs: list[tuple[str, str]],
        chip_profile: ChipProfile,
        rad_env: RadiationEnvironment | None = None,
    ) -> "Constellation":
        """Build a constellation from a list of TLE line pairs.

        Each element of *tle_pairs* is a ``(line1, line2)`` tuple in the
        standard TLE format.  The satellite IDs are assigned as
        ``TLE-000``, ``TLE-001``, … preserving the input order.

        The radiation environment defaults to a LEO 500 km profile when not
        provided; callers may pass a custom *rad_env* for more accurate
        modelling.

        Args:
            tle_pairs: Sequence of (line1, line2) TLE string pairs.
            chip_profile: Hardware profile applied to every satellite.
            rad_env: Optional radiation environment.  Defaults to LEO 500 km.

        Returns:
            Constellation with one satellite per TLE pair.

        Raises:
            ValueError: If any TLE pair is invalid.
        """
        # Import here to avoid a circular dependency at module load time.
        from space_ml_sim.core.tle import parse_tle  # noqa: PLC0415

        satellites: list[Satellite] = []
        for idx, (l1, l2) in enumerate(tle_pairs):
            orbit_config = parse_tle(l1, l2)
            satellites.append(
                Satellite(
                    id=f"TLE-{idx:03d}",
                    orbit_config=orbit_config,
                    chip_profile=chip_profile,
                )
            )

        env = rad_env or RadiationEnvironment.leo_500km()
        return cls(satellites=satellites, rad_env=env)

    @property
    def operational_count(self) -> int:
        """Number of satellites that can still compute."""
        return sum(1 for s in self.satellites if s.is_operational)
