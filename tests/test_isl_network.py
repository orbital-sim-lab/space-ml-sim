"""TDD tests for ISL network graph with bandwidth, latency, and routing.

Written FIRST before implementation (RED phase).
Tests should FAIL until environment/isl_network.py is created.

The ISL network models:
- Directed graph of satellite-to-satellite links
- Per-link latency (propagation + processing)
- Per-link bandwidth with transfer time for payloads
- Multi-hop shortest-path routing (Dijkstra on latency)
- Link state tracking (up/down based on distance and eclipse)
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Test: ISLNetwork construction and topology
# ---------------------------------------------------------------------------


class TestISLNetworkConstruction:
    """Network must be constructable from satellite positions."""

    def test_creates_from_positions(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
            "S2": (6771.0, 0.0, 100.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert isinstance(net, ISLNetwork)

    def test_nearby_sats_are_linked(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),  # ~100 km away
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert net.has_link("S0", "S1")

    def test_distant_sats_not_linked(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (-6771.0, 0.0, 0.0),  # ~13542 km away (opposite side of Earth)
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert not net.has_link("S0", "S1")

    def test_link_count(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        # Triangle of 3 close sats
        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
            "S2": (6771.0, 0.0, 50.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert net.num_links == 3  # bidirectional: 3 undirected edges


class TestLinkLatency:
    """Latency model must account for propagation and overhead."""

    def test_latency_increases_with_distance(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 100.0, 0.0),
            "S2": (6771.0, 500.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        lat_01 = net.link_latency_ms("S0", "S1")
        lat_02 = net.link_latency_ms("S0", "S2")
        assert lat_02 > lat_01

    def test_latency_is_positive(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert net.link_latency_ms("S0", "S1") > 0

    def test_no_link_returns_inf(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (-6771.0, 0.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert net.link_latency_ms("S0", "S1") == math.inf


class TestTransferTime:
    """Transfer time must depend on payload size and bandwidth."""

    def test_larger_payload_takes_longer(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net = ISLNetwork.from_positions(
            positions, max_range_km=5000.0, bandwidth_gbps=10.0
        )
        t_small = net.transfer_time_ms("S0", "S1", payload_bytes=1_000_000)
        t_large = net.transfer_time_ms("S0", "S1", payload_bytes=100_000_000)
        assert t_large > t_small

    def test_transfer_includes_latency(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net = ISLNetwork.from_positions(
            positions, max_range_km=5000.0, bandwidth_gbps=10.0
        )
        latency = net.link_latency_ms("S0", "S1")
        transfer = net.transfer_time_ms("S0", "S1", payload_bytes=1)
        # Transfer time must be at least the propagation latency
        assert transfer >= latency


class TestMultiHopRouting:
    """Shortest-path routing must find optimal multi-hop paths."""

    def test_direct_path_when_linked(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        path = net.shortest_path("S0", "S1")
        assert path == ["S0", "S1"]

    def test_multi_hop_path(self) -> None:
        """When S0-S2 not linked directly, find S0->S1->S2."""
        from space_ml_sim.environment.isl_network import ISLNetwork

        # S0 and S2 are far apart, but both close to S1
        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 3000.0, 0.0),
            "S2": (6771.0, 4500.0, 2500.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=4000.0)
        # S0-S1: ~3000km (linked), S1-S2: ~2915km (linked), S0-S2: ~5315km (not linked)
        path = net.shortest_path("S0", "S2")
        assert path is not None
        assert path[0] == "S0"
        assert path[-1] == "S2"
        assert len(path) >= 2

    def test_no_path_returns_none(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (-6771.0, 0.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        path = net.shortest_path("S0", "S1")
        assert path is None

    def test_path_latency(self) -> None:
        """Total path latency must sum per-hop latencies."""
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 1000.0, 0.0),
            "S2": (6771.0, 2000.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        total = net.path_latency_ms(["S0", "S1", "S2"])
        hop1 = net.link_latency_ms("S0", "S1")
        hop2 = net.link_latency_ms("S1", "S2")
        assert abs(total - (hop1 + hop2)) < 0.001


class TestNetworkUpdate:
    """Network topology must be updatable with new positions."""

    def test_update_positions_changes_topology(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        assert net.has_link("S0", "S1")

        # Move S1 to opposite side of Earth
        new_positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (-6771.0, 0.0, 0.0),
        }
        net = net.with_updated_positions(new_positions)
        assert not net.has_link("S0", "S1")

    def test_update_is_immutable(self) -> None:
        from space_ml_sim.environment.isl_network import ISLNetwork

        positions = {
            "S0": (6771.0, 0.0, 0.0),
            "S1": (6771.0, 50.0, 0.0),
        }
        net1 = ISLNetwork.from_positions(positions, max_range_km=5000.0)
        net2 = net1.with_updated_positions({
            "S0": (6771.0, 0.0, 0.0),
            "S1": (-6771.0, 0.0, 0.0),
        })
        # Original unchanged
        assert net1.has_link("S0", "S1")
        assert not net2.has_link("S0", "S1")
