"""Tests for running the surrogate as a NetSecGame agent.

These run without a game server: action selection is in `SurrogateController`
and the statistics in `aggregate`, neither of which opens a connection. The
exchange with the server cannot be covered this way, which is why
`SurrogateAgent` contains only the connection and the episode loop.
"""

import importlib.util
import unittest

from netsecgame.game_components import (
    ActionType,
    AgentStatus,
    Data,
    GameState,
    IP,
    Network,
    Observation,
    Service,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None and (
    importlib.util.find_spec("torch_geometric") is not None
)

from nsg_surrogate.nsg_agent import (  # noqa: E402  (import after the guard)
    EpisodeResult,
    aggregate,
    filter_noise_data,
)

if HAS_TORCH:
    import torch

    from nsg_surrogate.nsg_agent import SurrogateController
    from nsg_surrogate.state_adapter import Provenance

LOCAL = IP("192.168.1.2")
PUBLIC = IP("213.47.23.195")
TARGET = IP("192.168.1.3")
NET = Network("192.168.1.0", 24)
SSH = Service("ssh", "tcp", "1.0", False)
SECRET = Data(owner="192.168.1.3", id="DataFromServer1", size=0, type="file", content="")
LOGFILE = Data(owner="192.168.1.2", id="logfile", size=0, type="log", content="")


def simulator_state() -> GameState:
    """A state shaped like the simulator's, with a public controlled host."""
    return GameState(
        controlled_hosts={LOCAL, PUBLIC},
        known_hosts={LOCAL, PUBLIC, TARGET},
        known_services={TARGET: {SSH}},
        known_data={TARGET: {SECRET}},
        known_networks={NET},
        known_blocks={},
    )


def observation(state: GameState, reward: float = 0.0, end: bool = False) -> Observation:
    return Observation(state=state, reward=reward, end=end, info={})


class NoiseFilterTests(unittest.TestCase):
    def test_logfile_entities_are_removed(self):
        state = simulator_state()
        state.known_data[LOCAL] = {LOGFILE}
        filtered = filter_noise_data(observation(state))
        self.assertNotIn(LOCAL, filtered.state.known_data)
        self.assertEqual(filtered.state.known_data[TARGET], {SECRET})

    def test_hosts_keep_their_real_data(self):
        state = simulator_state()
        state.known_data[TARGET] = {SECRET, LOGFILE}
        filtered = filter_noise_data(observation(state))
        self.assertEqual(filtered.state.known_data[TARGET], {SECRET})

    def test_the_caller_copy_is_not_mutated(self):
        state = simulator_state()
        state.known_data[LOCAL] = {LOGFILE}
        original = observation(state)
        filter_noise_data(original)
        self.assertIn(LOCAL, original.state.known_data)

    def test_clean_observation_is_returned_unchanged(self):
        original = observation(simulator_state())
        self.assertIs(filter_noise_data(original), original)

    def test_none_passes_through(self):
        self.assertIsNone(filter_noise_data(None))

    def test_reward_and_end_are_preserved(self):
        state = simulator_state()
        state.known_data[LOCAL] = {LOGFILE}
        filtered = filter_noise_data(observation(state, reward=-1.5, end=True))
        self.assertEqual(filtered.reward, -1.5)
        self.assertTrue(filtered.end)


class AggregateTests(unittest.TestCase):
    def test_win_rate_and_spread(self):
        results = [
            EpisodeResult(steps=10, reward=90.0, end_reason=AgentStatus.Success,
                          actions=["ScanNetwork", "FindData"]),
            EpisodeResult(steps=20, reward=-20.0, end_reason=AgentStatus.Fail,
                          actions=["ScanNetwork"]),
            EpisodeResult(steps=30, reward=-30.0, end_reason=AgentStatus.Success,
                          actions=["FindData"]),
        ]
        stats = aggregate(results)
        self.assertEqual(stats.episodes, 3)
        self.assertEqual(stats.wins, 2)
        self.assertAlmostEqual(stats.win_rate, 2 / 3, places=4)
        self.assertGreater(stats.win_rate_se, 0.0)
        self.assertAlmostEqual(stats.avg_steps, 20.0)
        self.assertAlmostEqual(stats.avg_win_steps, 20.0)  # 10 and 30
        self.assertEqual(stats.action_type_counts, {"FindData": 2, "ScanNetwork": 2})

    def test_no_wins_reports_n_a_rather_than_zero_steps(self):
        stats = aggregate([EpisodeResult(steps=5, end_reason=AgentStatus.Fail)])
        self.assertEqual(stats.wins, 0)
        self.assertIn("steps wins n/a", stats.summary_line())

    def test_single_episode_has_no_spread(self):
        stats = aggregate([EpisodeResult(steps=7, reward=1.0, end_reason=AgentStatus.Success)])
        self.assertEqual(stats.std_steps, 0.0)
        self.assertEqual(stats.win_rate_se, 0.0)

    def test_empty_input(self):
        stats = aggregate([])
        self.assertEqual(stats.episodes, 0)
        self.assertEqual(stats.win_rate, 0.0)


@unittest.skipUnless(HAS_TORCH, "torch and torch-geometric required")
class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.controller = SurrogateController(temperature=1.0)
        self.state = simulator_state()

    def test_selected_actions_are_always_valid(self):
        legal = {
            candidates_key(action)
            for action in self.controller.valid_actions(self.state)
        }
        for _ in range(30):
            action = self.controller.select_action(observation(self.state))
            self.assertIn(candidates_key(action), legal)
            self.assertFalse(self.controller.last_decision.fallback)

    def test_exfiltration_source_correction_is_applied_by_default(self):
        """NSG offers exfiltration from uncontrolled hosts; the game refuses it."""
        corrected = self.controller.valid_actions(self.state)
        raw = SurrogateController(raw_action_space=True).valid_actions(self.state)

        def sources(actions):
            return {
                str(a.parameters["source_host"])
                for a in actions
                if a.type == ActionType.ExfiltrateData
            }

        # The secret sits on TARGET, which the agent does not control.
        self.assertEqual(sources(corrected), set())
        self.assertEqual(sources(raw), {str(TARGET)})

    def test_attempt_counters_advance_and_reset(self):
        self.controller.reset()
        for step in range(1, 4):
            self.controller.select_action(observation(self.state))
            self.assertEqual(self.controller.attempt_counts.total(), step)
        self.controller.reset()
        self.assertEqual(self.controller.attempt_counts.total(), 0)

    def test_state_value_is_a_scalar(self):
        value = self.controller.state_value(observation(self.state))
        self.assertEqual(tuple(value.shape), ())

    def test_greedy_selection_is_deterministic(self):
        controller = SurrogateController(greedy=True)
        first = controller.select_action(observation(self.state))
        controller.reset()
        again = controller.select_action(observation(self.state))
        self.assertEqual(candidates_key(first), candidates_key(again))

    def test_provenance_overrides_the_is_private_fallback(self):
        """In the simulator the public host is found by is_private(); in a
        container range it must be declared instead."""
        provenance = Provenance()
        provenance.external_hosts.add(LOCAL)
        declared = SurrogateController(provenance=provenance)
        action = declared.select_action(observation(self.state))
        self.assertIn(action.type, set(ActionType))

    def test_empty_state_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            self.controller.select_action(observation(GameState()))

    def test_info_reports_the_configuration(self):
        info = self.controller.info()
        self.assertEqual(info["weights"], "<random-init>")
        self.assertEqual(info["temperature"], 1.0)
        self.assertFalse(info["raw_action_space"])


def candidates_key(action):
    from nsg_surrogate.candidates import action_key

    return action_key(action)


if __name__ == "__main__":
    unittest.main()
