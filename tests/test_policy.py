"""Tests for action validity and checkpoint round-tripping."""

import importlib.util
import json
import os
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and (
    importlib.util.find_spec("torch_geometric") is not None
)

if HAS_TORCH:
    import torch

    from nsg_surrogate.candidates import enumerate_actions
    from nsg_surrogate.encoder import state_summary, state_to_pyg
    from nsg_surrogate.policy import SurrogatePolicy
    from nsg_surrogate.state_adapter import AdapterConfig, project_graph_to_game_state

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "strategic_graph.json")
EXTERNAL = "203.0.113.9"


def load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def projected_state():
    return project_graph_to_game_state(
        load_fixture(), config=AdapterConfig(external_hosts=frozenset({EXTERNAL}))
    )


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
class DecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.projection = projected_state()
        self.actions = enumerate_actions(self.projection.state)
        self.surrogate = SurrogatePolicy()

    def test_every_sampled_action_is_a_candidate(self):
        legal = {action.to_json() for action in self.actions}
        for _ in range(50):
            self.surrogate.reset()
            decision = self.surrogate.act(
                self.projection.state,
                self.actions,
                provenance=self.projection.provenance,
                temperature=1.0,
            )
            self.assertIn(decision.action.to_json(), legal)
            self.assertFalse(decision.fallback, decision.head_choices)

    def test_head_choices_agree_with_the_emitted_action(self):
        decision = self.surrogate.act(
            self.projection.state,
            self.actions,
            provenance=self.projection.provenance,
            temperature=1.0,
        )
        choices = decision.head_choices
        self.assertEqual(choices["type"], decision.action.type.name)
        self.assertEqual(choices["source_host"], str(decision.action.parameters["source_host"]))
        self.assertEqual(decision.candidate_count, len(self.actions))
        self.assertIsNotNone(decision.log_prob)

    def test_attempt_counters_advance_with_each_action(self):
        self.surrogate.reset()
        self.assertEqual(self.surrogate.attempt_counts.total(), 0)
        for step in range(1, 4):
            self.surrogate.act(
                self.projection.state,
                self.actions,
                provenance=self.projection.provenance,
                temperature=1.0,
            )
            self.assertEqual(self.surrogate.attempt_counts.total(), step)

    def test_empty_candidate_set_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            self.surrogate.act(self.projection.state, [], provenance=self.projection.provenance)

    def test_checkpoint_round_trip(self):
        graph, object_to_idx, _ = state_to_pyg(
            self.projection.state, provenance=self.projection.provenance
        )
        summary = state_summary(self.projection.state, provenance=self.projection.provenance)
        self.surrogate.policy.decide(graph, object_to_idx, self.actions, summary=summary)

        path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "nsg_surrogate_roundtrip.pth"
        )
        torch.save(self.surrogate.policy.state_dict(), path)
        try:
            reloaded = SurrogatePolicy.load(path)
        finally:
            os.remove(path)

        for key, value in self.surrogate.policy.state_dict().items():
            self.assertTrue(torch.equal(value, reloaded.policy.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
