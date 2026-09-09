"""Tests for action validity and for checkpoint compatibility.

The compatibility test matters for the comparison this repository is built to
support: if the parameters of this policy diverge from
`sgrl_netsec`'s `FactoredGNNPolicy`, a checkpoint from the simulator agent
cannot be evaluated on projected container states without retraining.
"""

import importlib.util
import json
import os
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and (
    importlib.util.find_spec("torch_geometric") is not None
)

if HAS_TORCH:
    import torch

    from nsg_surrogate.candidates import enumerate_actions
    from nsg_surrogate.encoder import state_summary, state_to_pyg
    from nsg_surrogate.policy import FactoredGNNPolicy, SurrogatePolicy
    from nsg_surrogate.state_adapter import AdapterConfig, project_graph_to_game_state

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "strategic_graph.json")
SGRL_REPO = "/home/rigakmar/repos/sgrl_netsec"
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


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
@unittest.skipUnless(
    os.path.isdir(SGRL_REPO), f"simulator agent not available at {SGRL_REPO}"
)
class SimulatorCheckpointCompatibilityTests(unittest.TestCase):
    """The surrogate must accept weights trained by the simulator agent."""

    def _simulator_policy(self):
        if SGRL_REPO not in sys.path:
            sys.path.insert(0, SGRL_REPO)
        from blackbox_pure_gnn_agent import FactoredGNNPolicy as SimulatorPolicy

        return SimulatorPolicy()

    def test_state_dict_keys_and_shapes_match(self):
        try:
            simulator = self._simulator_policy()
        except ImportError as error:  # pragma: no cover - environment dependent
            self.skipTest(f"cannot import the simulator agent: {error}")

        surrogate = FactoredGNNPolicy()
        projection = projected_state()
        actions = enumerate_actions(projection.state)
        graph, object_to_idx, _ = state_to_pyg(
            projection.state, provenance=projection.provenance
        )
        summary = state_summary(projection.state, provenance=projection.provenance)

        # GATv2Conv is lazily initialised: parameters only exist after a forward.
        surrogate.decide(graph, object_to_idx, actions, summary=summary)
        simulator(graph, object_to_idx, actions, state_summary=summary)

        surrogate_shapes = {
            key: tuple(value.shape) for key, value in surrogate.state_dict().items()
        }
        simulator_shapes = {
            key: tuple(value.shape) for key, value in simulator.state_dict().items()
        }
        self.assertEqual(
            sorted(surrogate_shapes), sorted(simulator_shapes), "parameter names diverged"
        )
        self.assertEqual(surrogate_shapes, simulator_shapes, "parameter shapes diverged")

    def test_simulator_weights_load_into_the_surrogate(self):
        try:
            simulator = self._simulator_policy()
        except ImportError as error:  # pragma: no cover - environment dependent
            self.skipTest(f"cannot import the simulator agent: {error}")

        projection = projected_state()
        actions = enumerate_actions(projection.state)
        graph, object_to_idx, _ = state_to_pyg(
            projection.state, provenance=projection.provenance
        )
        summary = state_summary(projection.state, provenance=projection.provenance)
        simulator(graph, object_to_idx, actions, state_summary=summary)

        surrogate = FactoredGNNPolicy()
        surrogate.decide(graph, object_to_idx, actions, summary=summary)
        surrogate.load_state_dict(simulator.state_dict())

        for key, value in simulator.state_dict().items():
            self.assertTrue(torch.equal(value, surrogate.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
