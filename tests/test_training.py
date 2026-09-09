"""Tests for the factored targets and the training pipeline.

The round-trip tests are the load-bearing ones: if `head_targets` and the
decoder disagree about what index `i` means, training silently optimises the
wrong thing and no loss curve would show it.
"""

import importlib.util
import json
import os
import tempfile
import unittest

from netsecgame.game_components import Action, ActionType, Data, GameState, IP, Network, Service

from nsg_surrogate import candidates, factorization
from nsg_surrogate.factorization import decode_targets, head_targets

HAS_TORCH = importlib.util.find_spec("torch") is not None and (
    importlib.util.find_spec("torch_geometric") is not None
)

if HAS_TORCH:
    import torch

    from nsg_surrogate import training
    from nsg_surrogate.encoder import state_to_pyg
    from nsg_surrogate.policy import FactoredGNNPolicy

    from tests.test_dataset import build_trajectory  # reuse the synthetic run
    from nsg_surrogate import dataset as dataset_mod

LOCAL = IP("10.0.0.5")
TARGET = IP("10.0.0.9")
DROP_BOX = IP("203.0.113.9")
NET = Network("10.0.0.0", 24)
SSH = Service("ssh", "tcp", "unknown", False)
HTTP = Service("http", "tcp", "unknown", False)
SECRET = Data(owner="10.0.0.5", id="/root/secret", size=0, type="file", content="")


def rich_state() -> GameState:
    """A state where every action type has several candidates."""
    return GameState(
        controlled_hosts={LOCAL, DROP_BOX},
        known_hosts={LOCAL, TARGET, DROP_BOX, IP("10.0.0.7")},
        known_services={TARGET: {SSH, HTTP}, IP("10.0.0.7"): {SSH}},
        known_data={LOCAL: {SECRET}},
        known_networks={NET, Network("10.1.0.0", 24)},
        known_blocks={},
    )


def object_to_idx(state: GameState):
    if not HAS_TORCH:
        raise unittest.SkipTest("torch required for the encoder")
    _, mapping, _ = state_to_pyg(state)
    return mapping


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
class FactorizationRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.state = rich_state()
        self.candidates = candidates.enumerate_actions(self.state)
        self.mapping = object_to_idx(self.state)

    def test_every_candidate_round_trips_through_its_targets(self):
        """head_targets -> decode_targets must be the identity on candidates."""
        self.assertGreater(len(self.candidates), 10)
        for action in self.candidates:
            targets = head_targets(action, self.candidates, self.mapping)
            self.assertIsNotNone(targets, candidates.describe(action))
            decoded = decode_targets(targets, self.candidates, self.mapping)
            self.assertIsNotNone(decoded, candidates.describe(action))
            self.assertEqual(
                candidates.action_key(decoded),
                candidates.action_key(action),
                f"{candidates.describe(action)} decoded to {candidates.describe(decoded)}",
            )

    def test_all_five_action_types_are_covered(self):
        present = {action.type for action in self.candidates}
        self.assertEqual(
            present,
            {
                ActionType.ScanNetwork,
                ActionType.FindServices,
                ActionType.FindData,
                ActionType.ExploitService,
                ActionType.ExfiltrateData,
            },
        )

    def test_exfiltration_uses_the_secondary_head(self):
        exfil = next(a for a in self.candidates if a.type == ActionType.ExfiltrateData)
        targets = head_targets(exfil, self.candidates, self.mapping)
        self.assertIsNotNone(targets.secondary_index)
        self.assertEqual(targets.active_heads, ("type", "source", "target", "secondary"))

    def test_other_types_leave_the_secondary_head_inactive(self):
        scan = next(a for a in self.candidates if a.type == ActionType.ScanNetwork)
        targets = head_targets(scan, self.candidates, self.mapping)
        self.assertIsNone(targets.secondary_index)

    def test_action_outside_the_candidate_set_has_no_targets(self):
        stranger = Action(
            ActionType.FindServices,
            {"source_host": LOCAL, "target_host": IP("192.0.2.99")},
        )
        self.assertIsNone(head_targets(stranger, self.candidates, self.mapping))

    def test_greedy_decode_agrees_with_the_argmax_of_the_teacher_logits(self):
        """The decoder and the training path must score the same candidates."""
        torch.manual_seed(0)
        policy = FactoredGNNPolicy()
        graph, mapping, _ = state_to_pyg(self.state)
        decision = policy.decide(graph, mapping, self.candidates, greedy=True)
        self.assertFalse(decision.fallback)

        targets = head_targets(decision.action, self.candidates, mapping)
        logits = policy.head_logits(graph, mapping, self.candidates, targets)
        self.assertEqual(int(logits["type"].argmax().item()), targets.type_index)
        for head, index in (
            ("source", targets.source_index),
            ("target", targets.target_index),
            ("secondary", targets.secondary_index),
        ):
            if index is None or head not in logits:
                continue
            self.assertEqual(int(logits[head].argmax().item()), index, head)


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = build_trajectory(self._tmp.name)
        result = dataset_mod.build(root)
        self.out = os.path.join(self._tmp.name, "ds")
        dataset_mod.write(result, self.out, dataset_mod._trajectory_dir(root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_samples_load_and_re_project_cleanly(self):
        samples, report = training.load_samples([self.out])
        self.assertEqual(report.rows, 1)
        self.assertEqual(report.samples, 1)
        self.assertEqual(
            report.projection_mismatches,
            0,
            "a freshly written dataset must re-project identically",
        )
        self.assertEqual(samples[0].action.type, ActionType.FindServices)

    def test_confidence_becomes_the_sample_weight(self):
        weighted, _ = training.load_samples([self.out])
        flat, _ = training.load_samples([self.out], weight_by_confidence=False)
        self.assertAlmostEqual(weighted[0].weight, 0.9, places=3)
        self.assertEqual(flat[0].weight, 1.0)

    def test_training_overfits_a_single_sample(self):
        """The pipeline must be able to drive loss down on data it has seen."""
        samples, report = training.load_samples([self.out])
        policy = FactoredGNNPolicy()
        result = training.train(
            samples, epochs=30, learning_rate=5e-3, policy=policy, load=report, verbose=False
        )
        first, last = result.epochs[0]["loss"], result.epochs[-1]["loss"]
        self.assertLess(last, first, f"loss did not fall: {first} -> {last}")
        self.assertEqual(result.train_eval.action_accuracy, 1.0)
        self.assertEqual(result.train_eval.fallbacks, 0)

    def test_split_by_run_keeps_runs_whole(self):
        samples, _ = training.load_samples([self.out])
        train_part, holdout = training.split_by_run(samples, ["manual-run-strategic"])
        self.assertEqual(len(holdout), 0)  # this synthetic run has another id
        self.assertEqual(len(train_part), len(samples))

        run_id = samples[0].run_id
        train_part, holdout = training.split_by_run(samples, [run_id])
        self.assertEqual(len(train_part), 0)
        self.assertEqual(len(holdout), len(samples))
        self.assertFalse(
            {s.state_key for s in train_part} & {s.state_key for s in holdout},
            "no state may appear on both sides of the split",
        )

    def test_evaluate_reports_the_majority_baseline(self):
        samples, _ = training.load_samples([self.out])
        result = training.evaluate(FactoredGNNPolicy(), samples)
        self.assertEqual(result.samples, 1)
        self.assertEqual(result.majority_type_baseline, 1.0)

    def test_checkpoint_saves_and_reloads(self):
        samples, report = training.load_samples([self.out])
        policy = FactoredGNNPolicy()
        training.train(samples, epochs=2, policy=policy, load=report, verbose=False)
        path = os.path.join(self._tmp.name, "weights.pth")
        training.save(policy, path)

        reloaded = FactoredGNNPolicy()
        # Lazy GATv2 parameters materialise only after a forward pass.
        graph, mapping, _ = state_to_pyg(samples[0].state)
        reloaded.decide(graph, mapping, samples[0].candidates, greedy=True)
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, reloaded.state_dict()[key]), key)


class CanonicalStateTests(unittest.TestCase):
    """Dataset rows must be byte-stable across processes."""

    def test_lists_are_sorted_deterministically(self):
        from nsg_surrogate.dataset import canonical_game_state

        state = rich_state()
        first = json.dumps(canonical_game_state(state), sort_keys=True)
        second = json.dumps(canonical_game_state(state), sort_keys=True)
        self.assertEqual(first, second)
        hosts = canonical_game_state(state)["known_hosts"]
        self.assertEqual(hosts, sorted(hosts, key=lambda item: json.dumps(item, sort_keys=True)))


if __name__ == "__main__":
    unittest.main()
