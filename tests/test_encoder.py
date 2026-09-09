"""Tests for the encoder's feature layout and node ordering. Requires torch."""

import importlib.util
import json
import os
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and (
    importlib.util.find_spec("torch_geometric") is not None
)

if HAS_TORCH:
    from nsg_surrogate.attempt_counts import AttemptCounts
    from nsg_surrogate.encoder import FEATURE_DIM, state_summary, state_to_pyg
    from nsg_surrogate.state_adapter import AdapterConfig, project_graph_to_game_state

from netsecgame.game_components import ActionType, GameState, IP, Network, Service

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "strategic_graph.json")
EXTERNAL = "203.0.113.9"


def load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return json.load(handle)


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.projection = project_graph_to_game_state(
            load_fixture(), config=AdapterConfig(external_hosts=frozenset({EXTERNAL}))
        )
        self.state = self.projection.state
        self.provenance = self.projection.provenance
        self.graph, self.object_to_idx, self.idx_to_object = state_to_pyg(
            self.state, provenance=self.provenance
        )

    def test_node_counts_and_feature_width(self):
        self.assertEqual(self.graph["host"].x.shape, (4, FEATURE_DIM))  # 3 observed + external
        self.assertEqual(self.graph["network"].x.shape, (1, FEATURE_DIM))
        self.assertEqual(self.graph["service"].x.shape, (2, FEATURE_DIM))
        self.assertEqual(self.graph["data"].x.shape, (1, FEATURE_DIM))

    def test_controlled_and_external_host_bits(self):
        hosts = self.idx_to_object["host"]
        features = self.graph["host"].x
        local_idx = hosts.index(IP("10.0.0.5"))
        external_idx = hosts.index(IP(EXTERNAL))
        gateway_idx = hosts.index(IP("10.0.0.1"))

        self.assertEqual(features[local_idx][0].item(), 1.0, "local container is controlled")
        self.assertEqual(features[local_idx][1].item(), 0.0, "local container is not external")
        self.assertEqual(features[external_idx][1].item(), 1.0, "designated host is external")
        self.assertEqual(features[gateway_idx][0].item(), 0.0, "gateway is not controlled")

    def test_service_port_feature_is_populated_from_provenance(self):
        """The simulator's conversion of `Service.name` leaves this feature zero."""
        ports = sorted(feature[0].item() for feature in self.graph["service"].x)
        for actual, expected in zip(ports, [22 / 65535.0, 443 / 65535.0]):
            self.assertAlmostEqual(actual, expected, places=6)

        legacy_graph, _, _ = state_to_pyg(
            self.state, provenance=self.provenance, legacy_service_port=True
        )
        self.assertEqual(
            [feature[0].item() for feature in legacy_graph["service"].x],
            [0.0, 0.0],
            "legacy mode must reproduce the simulator's always-zero port feature",
        )

    def test_attempt_counters_land_in_the_documented_slots(self):
        counts = AttemptCounts()
        target = IP("10.0.0.9")
        for _ in range(3):
            counts.findservices[target] = counts.findservices.get(target, 0) + 1
        counts.exploit[target] = 10
        counts.scan[Network("10.0.0.0", 24)] = 5

        graph, _, idx_to_object = state_to_pyg(
            self.state, attempt_counts=counts, provenance=self.provenance
        )
        host_idx = idx_to_object["host"].index(target)
        self.assertAlmostEqual(graph["host"].x[host_idx][2].item(), 0.3, places=5)
        self.assertAlmostEqual(graph["host"].x[host_idx][3].item(), 0.0, places=5)
        self.assertAlmostEqual(graph["host"].x[host_idx][4].item(), 1.0, places=5)
        self.assertAlmostEqual(graph["network"].x[0][0].item(), 0.5, places=5)

    def test_exfiltrated_data_flag_tracks_the_external_host(self):
        state = self.projection.state
        datapoint = next(iter(state.known_data[IP("10.0.0.5")]))
        graph, _, _ = state_to_pyg(state, provenance=self.provenance)
        self.assertEqual(graph["data"].x[0][0].item(), 0.0)

        # A copy now sits on the controlled external host: that is exfiltrated.
        state.known_data[IP(EXTERNAL)] = {datapoint}
        graph, _, idx_to_object = state_to_pyg(state, provenance=self.provenance)
        self.assertEqual(len(idx_to_object["data"]), 1, "data dedupes across hosts")
        self.assertEqual(graph["data"].x[0][0].item(), 1.0)

    def test_edges_link_hosts_to_their_services_networks_and_data(self):
        hosts = self.idx_to_object["host"]
        target_idx = hosts.index(IP("10.0.0.9"))
        runs = self.graph["host", "runs", "service"].edge_index
        self.assertEqual(set(runs[0].tolist()), {target_idx})
        self.assertEqual(runs.shape[1], 2)

        # Reverse edges must mirror exactly, or message passing is one-way.
        rev = self.graph["service", "rev_runs", "host"].edge_index
        self.assertEqual(rev[0].tolist(), runs[1].tolist())
        self.assertEqual(rev[1].tolist(), runs[0].tolist())

        in_network = self.graph["host", "in", "network"].edge_index
        # The external host is outside 10.0.0.0/24, so 3 of 4 hosts are members.
        self.assertEqual(in_network.shape[1], 3)

    def test_node_order_is_deterministic(self):
        for _ in range(5):
            _, _, again = state_to_pyg(self.state, provenance=self.provenance)
            self.assertEqual(
                [str(host) for host in again["host"]],
                [str(host) for host in self.idx_to_object["host"]],
            )

    def test_empty_state_encodes_without_error(self):
        graph, object_to_idx, _ = state_to_pyg(GameState())
        for node_type in ("network", "host", "service", "data"):
            self.assertEqual(graph[node_type].x.shape, (0, FEATURE_DIM))
            self.assertEqual(object_to_idx[node_type], {})

    def test_state_summary_shape_and_semantics(self):
        summary = state_summary(self.state, provenance=self.provenance)
        self.assertEqual(tuple(summary.shape), (3,))
        # 10.0.0.9 is uncontrolled and has services -> exploitable == 1 -> 0.1
        self.assertAlmostEqual(summary[2].item(), 0.1, places=5)
        # All known data sits on a controlled host, so the "not yet exfiltrated"
        # term reads 0 even though nothing has been exfiltrated.
        self.assertAlmostEqual(summary[1].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
