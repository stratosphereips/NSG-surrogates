"""Golden tests for the docker -> NetSecGame state projection.

Written as `unittest.TestCase` deliberately: these run with the standard library
alone (`python3 -m unittest discover tests`), so the mapping half of the PoC is
testable before torch is installed, and pytest still collects them unchanged.
"""

import json
import os
import unittest

from netsecgame.game_components import ActionType, Data, IP, Network, Service

from nsg_surrogate import candidates
from nsg_surrogate.state_adapter import AdapterConfig, project_graph_to_game_state

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "strategic_graph.json")

LOCAL = IP("10.0.0.5")
GATEWAY = IP("10.0.0.1")
TARGET = IP("10.0.0.9")


def load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.projection = project_graph_to_game_state(load_fixture(), source=FIXTURE)
        self.state = self.projection.state
        self.report = self.projection.report

    def test_hosts_and_control(self):
        self.assertEqual(self.state.known_hosts, {LOCAL, GATEWAY, TARGET})
        self.assertEqual(self.state.controlled_hosts, {LOCAL})

    def test_local_host_identity_prefers_its_own_subnet_over_a_gateway(self):
        """The container claims its gateway's address too; NSG needs exactly one.

        `10.0.0.1` is numerically smallest and would win a naive sort, but it is
        the gateway (a separate host node). `192.168.50.2` is routable but not in
        any network the container is a MEMBER_OF. `10.0.0.5` is the real identity.
        """
        self.assertEqual(self.projection.provenance.node_id[LOCAL], "host:local")
        self.assertIn(LOCAL, self.projection.provenance.local_hosts)
        self.assertTrue(
            any("also claims" in note for note in self.report.notes),
            "address aliasing between the container and the gateway must be reported",
        )

    def test_ipv6_only_host_is_dropped_not_guessed(self):
        reasons = self.report.dropped_by_reason
        self.assertEqual(reasons.get("host/no routable IPv4 address"), 1)
        # ...and its service goes with it rather than attaching to nothing.
        self.assertEqual(reasons.get("service/host not projected"), 1)

    def test_low_confidence_host_is_dropped(self):
        self.assertEqual(self.report.dropped_by_reason.get("host/below min_confidence"), 1)
        self.assertNotIn(IP("10.0.0.77"), self.state.known_hosts)

    def test_networks_exclude_default_route_and_inferred_prefixes(self):
        self.assertEqual(self.state.known_networks, {Network("10.0.0.0", 24)})
        reasons = self.report.dropped_by_reason
        self.assertEqual(reasons.get("network/excluded cidr"), 1)
        self.assertEqual(reasons.get("network/inferred network"), 1)

    def test_inferred_networks_can_be_kept(self):
        projection = project_graph_to_game_state(
            load_fixture(), config=AdapterConfig(min_confidence=0.3, include_inferred_networks=True)
        )
        self.assertIn(Network("8.8.8.0", 24), projection.state.known_networks)

    def test_service_name_list_collapses_to_first_alias(self):
        services = self.state.known_services[TARGET]
        names = {service.name for service in services}
        self.assertEqual(names, {"ssh", "https"})
        self.assertNotIn("['https', 'ssl']", names)

    def test_service_ports_survive_in_provenance(self):
        ports = {
            port
            for (host, _service), port in self.projection.provenance.service_ports.items()
            if host == TARGET
        }
        self.assertEqual(ports, {22, 443})

    def test_data_keeps_only_files_observed_present(self):
        items = self.state.known_data[LOCAL]
        self.assertEqual({item.id for item in items}, {"/home/user/secrets.txt"})
        reasons = self.report.dropped_by_reason
        self.assertEqual(
            reasons.get("data/existence unstable (observed present and absent)"), 1
        )
        self.assertEqual(reasons.get("data/denylisted path"), 1)

    def test_blocks_are_reported_unmapped(self):
        self.assertEqual(self.state.known_blocks, {})
        self.assertEqual(len(self.projection.provenance.blocks), 1)
        self.assertTrue(any("block node(s) not projected" in note for note in self.report.notes))

    def test_projection_is_deterministic(self):
        again = project_graph_to_game_state(load_fixture(), source=FIXTURE)
        self.assertEqual(self.state.as_json(), again.state.as_json())

    def test_counts_out_match_the_state(self):
        self.assertEqual(
            self.report.counts_out,
            {
                "known_networks": 1,
                "known_hosts": 3,
                "controlled_hosts": 1,
                "known_services": 2,
                "known_data": 1,
                "known_blocks": 0,
            },
        )


class ExfiltrationReachabilityTests(unittest.TestCase):
    """A single controlled host makes NSG's exfiltration action unreachable."""

    def test_no_exfiltration_without_a_second_controlled_host(self):
        projection = project_graph_to_game_state(load_fixture())
        actions = candidates.enumerate_actions(projection.state)
        self.assertEqual(
            [action for action in actions if action.type == ActionType.ExfiltrateData], []
        )
        self.assertTrue(
            any("exfiltration is unreachable" in note for note in projection.report.notes)
        )

    def test_declared_external_host_restores_exfiltration(self):
        projection = project_graph_to_game_state(
            load_fixture(), config=AdapterConfig(external_hosts=frozenset({"203.0.113.9"}))
        )
        external = IP("203.0.113.9")
        self.assertIn(external, projection.state.controlled_hosts)
        self.assertIn(external, projection.provenance.external_hosts)

        exfiltration = [
            action
            for action in candidates.enumerate_actions(projection.state)
            if action.type == ActionType.ExfiltrateData
        ]
        self.assertEqual(len(exfiltration), 1)
        self.assertEqual(exfiltration[0].parameters["target_host"], external)
        self.assertEqual(exfiltration[0].parameters["data"].id, "/home/user/secrets.txt")
        self.assertTrue(
            any("configuration, not observation" in note for note in projection.report.notes),
            "adding a controlled host is a declared fact and must be reported as such",
        )


class ScopeTests(unittest.TestCase):
    """Every host the container talked to becomes an NSG target unless scoped."""

    def test_out_of_scope_hosts_and_their_services_are_dropped(self):
        projection = project_graph_to_game_state(
            load_fixture(), config=AdapterConfig(scope_cidrs=frozenset({"10.0.0.0/29"}))
        )
        self.assertEqual(projection.state.known_hosts, {LOCAL, GATEWAY})
        self.assertNotIn(TARGET, projection.state.known_hosts)
        self.assertEqual(projection.state.known_services, {})
        reasons = projection.report.dropped_by_reason
        self.assertEqual(reasons.get("host/outside declared scope"), 1)
        self.assertEqual(reasons.get("service/host not projected"), 3)

    def test_declared_external_host_is_exempt_from_scope(self):
        projection = project_graph_to_game_state(
            load_fixture(),
            config=AdapterConfig(
                scope_cidrs=frozenset({"10.0.0.0/29"}),
                external_hosts=frozenset({"203.0.113.9"}),
            ),
        )
        self.assertIn(IP("203.0.113.9"), projection.state.controlled_hosts)

    def test_public_hosts_without_a_scope_are_flagged(self):
        graph = load_fixture()
        graph["nodes"].append(
            {
                "id": "host:ip:8.8.8.8",
                "type": "host",
                "label": "8.8.8.8",
                "confidence": 0.95,
                "attributes": {"addresses": ["8.8.8.8"], "local": False},
            }
        )
        graph["edges"].append(
            {"id": "e99", "type": "KNOWS_HOST", "source": "agent:observed", "target": "host:ip:8.8.8.8"}
        )
        projection = project_graph_to_game_state(graph)
        self.assertIn(IP("8.8.8.8"), projection.state.known_hosts)
        self.assertTrue(
            any("declare a scope before any live run" in note for note in projection.report.notes)
        )

    def test_networks_outside_scope_are_dropped(self):
        projection = project_graph_to_game_state(
            load_fixture(), config=AdapterConfig(scope_cidrs=frozenset({"192.168.0.0/16"}))
        )
        self.assertEqual(projection.state.known_networks, set())
        self.assertEqual(
            projection.report.dropped_by_reason.get("network/outside declared scope"), 1
        )


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.projection = project_graph_to_game_state(load_fixture())
        self.actions = candidates.enumerate_actions(self.projection.state)

    def test_action_space_shape(self):
        # One controlled source host: 1 network to scan, 3 hosts to probe,
        # 1 FindData (source is forced equal to target), and the two services on
        # 10.0.0.9 collapsed to a single host-level exploit.
        self.assertEqual(
            candidates.breakdown(self.actions),
            {"ExploitService": 1, "FindData": 1, "FindServices": 3, "ScanNetwork": 1},
        )

    def test_exploits_collapse_to_one_service_per_host(self):
        state = self.projection.state
        state.known_services[TARGET] = {
            Service("ssh", "tcp", "unknown", False),
            Service("https", "tcp", "unknown", False),
        }
        raw = candidates.enumerate_actions(state, canonicalize_exploit_services=False)
        collapsed = candidates.enumerate_actions(state)
        exploits_raw = [a for a in raw if a.type == ActionType.ExploitService]
        exploits_collapsed = [a for a in collapsed if a.type == ActionType.ExploitService]
        self.assertEqual(len(exploits_raw), 2)
        self.assertEqual(len(exploits_collapsed), 1)
        self.assertEqual(exploits_collapsed[0].parameters["target_service"].name, "https")

    def test_enumeration_order_is_stable(self):
        again = candidates.enumerate_actions(self.projection.state)
        self.assertEqual([a.to_json() for a in self.actions], [a.to_json() for a in again])

    def test_translator_support_split(self):
        executable = candidates.executable(self.actions)
        self.assertEqual(
            {candidates.support_status(a) for a in executable}, {"live"}
        )
        self.assertEqual(
            {a.type for a in self.actions if candidates.support_status(a) != "live"},
            {ActionType.FindData, ActionType.ExploitService},
        )


class DataIdentityTests(unittest.TestCase):
    """NSG `Data` equality is (owner, id, type); the locator carries identity."""

    def test_same_file_on_two_hosts_is_two_data_objects(self):
        first = Data(owner="10.0.0.5", id="/etc/shadow", size=0, type="file", content="/etc/shadow")
        second = Data(owner="10.0.0.9", id="/etc/shadow", size=0, type="file", content="/etc/shadow")
        self.assertNotEqual(first, second)

    def test_size_and_content_do_not_affect_identity(self):
        first = Data(owner="10.0.0.5", id="/etc/shadow", size=0, type="file", content="a")
        second = Data(owner="10.0.0.5", id="/etc/shadow", size=99, type="file", content="b")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
