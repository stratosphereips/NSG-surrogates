"""Tests for reading a trajectory and writing a dataset.

A synthetic trajectory is constructed on disk in the layout
NSG-docker-state-creator produces, so these cover the file format as well as the
labelling.
"""

import copy
import json
import os
import tempfile
import unittest

from netsecgame.game_components import ActionType, GameState, IP

from nsg_surrogate import dataset
from nsg_surrogate.state_adapter import AdapterConfig, project_graph_to_game_state

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "strategic_graph.json")
GATEWAY = IP("10.0.0.1")

NMAP_RECORD = {
    "id": "bash:c0ffee:100:1",
    "source": "bash-command",
    "confidence": 0.995,
    "started_at": "2026-09-08T08:36:08.127231Z",
    "changed_domains": ["known_services"],
    "command": {"argv": ["nmap", "-p22", "10.0.0.1"], "executable": "nmap",
                "shell_text": "nmap -p22 10.0.0.1"},
}
#: The same execution as seen by the low-confidence fallback collector.
EBPF_DUPLICATE = dict(
    NMAP_RECORD,
    id="action:ebpf:dead",
    source="ebpf-exec-fallback",
    confidence=0.8,
)


def build_trajectory(root: str, extra_records=()) -> str:
    """Write a two-state trajectory in which a service is discovered."""
    trajectory = os.path.join(root, "trajectory")
    os.makedirs(os.path.join(trajectory, "actions"))

    with open(FIXTURE, "r", encoding="utf-8") as handle:
        before = json.load(handle)
    after = copy.deepcopy(before)
    after["nodes"].append(
        {
            "id": "service:ssh-10.0.0.1",
            "type": "service",
            "label": "ssh tcp/22 on 10.0.0.1",
            "confidence": 0.95,
            "attributes": {
                "host_id": "host:ip:10.0.0.1",
                "port": 22,
                "protocol": "tcp",
                "service_name": "ssh",
            },
        }
    )
    after["edges"].append(
        {"id": "e50", "type": "KNOWS_SERVICE", "source": "agent:observed",
         "target": "service:ssh-10.0.0.1"}
    )

    for state_id, graph in (("state-000000", before), ("state-000001", after)):
        state_dir = os.path.join(trajectory, "states", state_id)
        os.makedirs(state_dir)
        with open(os.path.join(state_dir, "graph.json"), "w", encoding="utf-8") as handle:
            json.dump(graph, handle)

    records = [NMAP_RECORD, *extra_records]
    sequence = [{"kind": "state", "id": "state-000000", "sequence": 0}]
    for index, record in enumerate(records):
        path = os.path.join("actions", f"a{index}.json")
        with open(os.path.join(trajectory, path), "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        sequence.append(
            {
                "kind": "action",
                "id": record["id"],
                "record": path,
                "state_before": "state-000000",
                "state_after": "state-000001",
                "state_changed": True,
                "action_type": "network_probe",
                "sequence": index + 1,
            }
        )
    sequence.append({"kind": "state", "id": "state-000001", "sequence": len(records) + 1})

    with open(os.path.join(trajectory, "sequence.jsonl"), "w", encoding="utf-8") as handle:
        for entry in sequence:
            handle.write(json.dumps(entry) + "\n")
    return root


class BuildTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = build_trajectory(self._tmp.name)
        self.result = dataset.build(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_pair_is_produced(self):
        self.assertEqual(len(self.result.rows), 1)
        row = self.result.rows[0]
        self.assertEqual(row["schema_version"], dataset.PAIR_SCHEMA_VERSION)
        self.assertEqual(row["state_before"]["id"], "state-000000")
        self.assertEqual(row["state_after"]["id"], "state-000001")
        self.assertEqual(row["nsg_diff_categories"], ["known_services"])

    def test_label_is_the_service_scan_corroborated_by_the_command(self):
        label = self.result.rows[0]["label"]
        self.assertEqual(label["action_type"], ActionType.FindServices.name)
        self.assertEqual(
            label["action"]["parameters"]["target_host"], {"ip": str(GATEWAY)}
        )
        self.assertEqual(label["label_source"], "effect+command")
        self.assertIn("nmap -p22 10.0.0.1", label["evidence"]["corroborating_commands"])

    def test_label_is_inside_the_before_state_candidate_set(self):
        cache = self.result.rows[0]["projection_cache"]
        self.assertTrue(cache["label_in_candidates"])
        self.assertEqual(cache["candidate_breakdown"]["FindServices"], 3)

    def test_report_counts(self):
        report = self.result.report
        self.assertEqual(report.states, 2)
        self.assertEqual(report.transitions, 1)
        self.assertEqual(report.action_records, 1)
        self.assertEqual(report.labels, 1)
        self.assertEqual(report.labels_by_source, {"effect+command": 1})
        self.assertEqual(report.transitions_with_no_label, 0)

    def test_duplicate_collector_records_collapse(self):
        with tempfile.TemporaryDirectory() as other:
            root = build_trajectory(other, extra_records=[EBPF_DUPLICATE])
            result = dataset.build(root)
            self.assertEqual(result.report.action_records, 2)
            self.assertEqual(result.report.actions_after_dedupe, 1)
            self.assertEqual(len(result.rows), 1)


class WriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = build_trajectory(self._tmp.name)
        self.config = AdapterConfig(scope_cidrs=frozenset({"10.0.0.0/24"}))
        self.result = dataset.build(self.root, config=self.config)
        self.out = os.path.join(self._tmp.name, "ds")
        self.paths = dataset.write(
            self.result, self.out, dataset._trajectory_dir(self.root)
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_dataset_directory_is_self_contained(self):
        for path in self.paths.values():
            self.assertTrue(os.path.isfile(path), path)
        for state_id in ("state-000000", "state-000001"):
            self.assertTrue(
                os.path.isfile(os.path.join(self.out, "graphs", f"{state_id}.json")),
                f"{state_id} graph must be copied, not referenced",
            )

    def test_rows_round_trip(self):
        rows = list(dataset.read_pairs(self.paths["pairs"]))
        self.assertEqual(len(rows), len(self.result.rows))
        self.assertEqual(rows[0]["pair_id"], self.result.rows[0]["pair_id"])

    def test_projection_cache_is_regenerable_from_the_copied_graph(self):
        """The dataset must survive an adapter change: graphs are the truth."""
        row = list(dataset.read_pairs(self.paths["pairs"]))[0]
        with open(os.path.join(self.out, row["state_before"]["graph"]), "r", encoding="utf-8") as handle:
            graph = json.load(handle)
        reprojected = project_graph_to_game_state(graph, config=self.config)
        # Compared by value, not as JSON text: `as_json` walks sets, so its list
        # order varies between processes even for identical states.
        self.assertEqual(
            GameState.from_json(json.dumps(row["projection_cache"]["game_state"])),
            reprojected.state,
        )
        self.assertEqual(
            dataset.canonical_game_state(reprojected.state),
            row["projection_cache"]["game_state"],
            "stored projections must be in canonical (byte-stable) form",
        )

    def test_config_hash_changes_with_the_adapter_config(self):
        other = dataset.build(
            self.root, config=AdapterConfig(scope_cidrs=frozenset({"192.168.0.0/16"}))
        )
        if other.rows:
            self.assertNotEqual(
                other.rows[0]["projection_cache"]["config_hash"],
                self.result.rows[0]["projection_cache"]["config_hash"],
            )
        else:
            self.assertNotEqual(
                dataset._config_hash(self.config),
                dataset._config_hash(AdapterConfig(scope_cidrs=frozenset({"192.168.0.0/16"}))),
            )

    def test_report_json_is_written(self):
        with open(self.paths["report"], "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["labels"], self.result.report.labels)
        self.assertIn("unmappable", report)


if __name__ == "__main__":
    unittest.main()
