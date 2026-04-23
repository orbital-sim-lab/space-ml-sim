"""Inter-satellite link (ISL) network graph with routing.

Models the communication topology of an orbital constellation as an
undirected weighted graph. Edge weights are propagation latency.
Supports multi-hop shortest-path routing via Dijkstra's algorithm.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


# Speed of light in km/ms
_C_KM_PER_MS = 299_792.458 / 1000.0  # ~299.79 km/ms

# Default processing overhead per hop
_DEFAULT_OVERHEAD_MS = 1.0


@dataclass(frozen=True)
class _Link:
    """A single ISL link between two satellites."""

    distance_km: float
    latency_ms: float


@dataclass(frozen=True)
class ISLNetwork:
    """Undirected graph of inter-satellite links.

    Nodes are satellite IDs (strings). Edges exist between satellites
    within max_range_km of each other, weighted by propagation latency.
    """

    adjacency: dict[str, dict[str, _Link]] = field(default_factory=dict)
    max_range_km: float = 5000.0
    bandwidth_gbps: float = 10.0
    overhead_ms: float = _DEFAULT_OVERHEAD_MS

    @classmethod
    def from_positions(
        cls,
        positions: dict[str, tuple[float, float, float]],
        max_range_km: float = 5000.0,
        bandwidth_gbps: float = 10.0,
        overhead_ms: float = _DEFAULT_OVERHEAD_MS,
    ) -> ISLNetwork:
        """Build an ISL network from satellite ECI positions.

        Args:
            positions: Map of sat_id -> (x, y, z) in km.
            max_range_km: Maximum link distance in km.
            bandwidth_gbps: Per-link bandwidth in Gbps.
            overhead_ms: Processing overhead per hop in ms.

        Returns:
            ISLNetwork with links between nearby satellites.
        """
        adj: dict[str, dict[str, _Link]] = {sid: {} for sid in positions}
        ids = list(positions.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                dist = _euclidean(positions[a], positions[b])
                if dist <= max_range_km:
                    link = _Link(
                        distance_km=dist,
                        latency_ms=(dist / _C_KM_PER_MS) + overhead_ms,
                    )
                    adj[a][b] = link
                    adj[b][a] = link

        return cls(
            adjacency=adj,
            max_range_km=max_range_km,
            bandwidth_gbps=bandwidth_gbps,
            overhead_ms=overhead_ms,
        )

    def with_updated_positions(
        self,
        positions: dict[str, tuple[float, float, float]],
    ) -> ISLNetwork:
        """Return a new ISLNetwork with updated satellite positions.

        Args:
            positions: New sat_id -> (x, y, z) map.

        Returns:
            New ISLNetwork (original is unchanged).
        """
        return ISLNetwork.from_positions(
            positions,
            max_range_km=self.max_range_km,
            bandwidth_gbps=self.bandwidth_gbps,
            overhead_ms=self.overhead_ms,
        )

    def has_link(self, a: str, b: str) -> bool:
        """Check if two satellites have a direct link."""
        return b in self.adjacency.get(a, {})

    @property
    def num_links(self) -> int:
        """Number of undirected links in the network."""
        return sum(len(neighbors) for neighbors in self.adjacency.values()) // 2

    def link_latency_ms(self, a: str, b: str) -> float:
        """One-way latency between two directly linked satellites.

        Returns inf if no direct link exists.
        """
        link = self.adjacency.get(a, {}).get(b)
        return link.latency_ms if link is not None else math.inf

    def transfer_time_ms(
        self,
        a: str,
        b: str,
        payload_bytes: int,
    ) -> float:
        """Total time to transfer a payload over a direct link.

        Includes propagation latency plus serialization time.

        Args:
            a: Source satellite ID.
            b: Destination satellite ID.
            payload_bytes: Payload size in bytes.

        Returns:
            Total transfer time in ms, or inf if no link.
        """
        latency = self.link_latency_ms(a, b)
        if latency == math.inf:
            return math.inf
        # Serialization time: bytes -> bits -> Gbps -> ms
        bits = payload_bytes * 8
        serialization_ms = bits / (self.bandwidth_gbps * 1e9) * 1000
        return latency + serialization_ms

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find the shortest-latency path using Dijkstra's algorithm.

        Args:
            source: Start satellite ID.
            target: End satellite ID.

        Returns:
            List of satellite IDs forming the path, or None if unreachable.
        """
        if source not in self.adjacency or target not in self.adjacency:
            return None

        dist: dict[str, float] = {sid: math.inf for sid in self.adjacency}
        dist[source] = 0.0
        prev: dict[str, str | None] = {sid: None for sid in self.adjacency}
        heap: list[tuple[float, str]] = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == target:
                break
            for v, link in self.adjacency[u].items():
                alt = d + link.latency_ms
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        if dist[target] == math.inf:
            return None

        # Reconstruct path
        path: list[str] = []
        node: str | None = target
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def path_latency_ms(self, path: list[str]) -> float:
        """Total latency along a multi-hop path.

        Args:
            path: Ordered list of satellite IDs.

        Returns:
            Sum of per-hop latencies in ms.
        """
        total = 0.0
        for i in range(len(path) - 1):
            lat = self.link_latency_ms(path[i], path[i + 1])
            if lat == math.inf:
                return math.inf
            total += lat
        return total


def _euclidean(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
