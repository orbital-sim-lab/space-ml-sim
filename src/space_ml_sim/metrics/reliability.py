"""Reliability metrics for constellation health tracking."""

from __future__ import annotations

from dataclasses import dataclass

from space_ml_sim.core.satellite import Satellite, SatelliteState


@dataclass(frozen=True)
class ReliabilityMetrics:
    """Snapshot of constellation reliability."""

    total_satellites: int
    nominal_count: int
    degraded_count: int
    failed_count: int
    total_seu_events: int
    max_tid_krad: float
    mean_tid_krad: float

    @classmethod
    def from_satellites(cls, satellites: list[Satellite]) -> "ReliabilityMetrics":
        """Compute metrics from a list of satellites.

        Args:
            satellites: Satellites to analyze.

        Returns:
            ReliabilityMetrics snapshot.
        """
        tids = [s.tid_accumulated_krad for s in satellites]
        return cls(
            total_satellites=len(satellites),
            nominal_count=sum(1 for s in satellites if s.state == SatelliteState.NOMINAL),
            degraded_count=sum(1 for s in satellites if s.state == SatelliteState.DEGRADED),
            failed_count=sum(1 for s in satellites if s.state == SatelliteState.FAILED),
            total_seu_events=sum(s.total_seu_events for s in satellites),
            max_tid_krad=max(tids) if tids else 0.0,
            mean_tid_krad=sum(tids) / len(tids) if tids else 0.0,
        )

    @property
    def availability(self) -> float:
        """Fraction of satellites that are operational (nominal or degraded)."""
        if self.total_satellites == 0:
            return 0.0
        return (self.nominal_count + self.degraded_count) / self.total_satellites
