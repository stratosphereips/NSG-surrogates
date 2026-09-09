"""Play the surrogate inside NetSecGame, against the real game server.

This is the return leg of the emulation -> simulation -> emulation loop. The
surrogate is trained on states projected from a real container
(`state_adapter`), and this module runs that same policy as an ordinary NSG
agent over the coordinator's socket, exactly the way
`sgrl_netsec/blackbox_pure_gnn_agent.py` does. Two things follow:

* other agents can be trained in NetSecGame against a surrogate whose behaviour
  came from a real operator, rather than against a hand-written baseline;
* the surrogate can be *measured* — win rate, steps, reward — on the same
  scenarios and with the same statistics as the simulator-native agents, which
  is the only way to compare them.

`SurrogateController` holds everything that decides an action and needs no
socket, so it can be tested without a running server. `SurrogateAgent` is the
`BaseAgent` subclass that owns the connection and the episode loop.

The candidate set applies the exfiltration source-host filter documented in
`candidates`: NetSecGame's generator currently offers exfiltrations from hosts
the agent does not control, and the game refuses them, so selecting one wastes a
step. Pass `raw_action_space=True` to enumerate exactly what the installed
`netsecgame` generates instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from netsecgame import AgentRole, BaseAgent
from netsecgame.game_components import (
    Action,
    AgentStatus,
    GameState,
    Observation,
)

from . import candidates as candidates_mod
from .attempt_counts import AttemptCounts
from .encoder import state_summary, state_to_pyg
from .policy import Decision, FactoredGNNPolicy, SurrogatePolicy
from .state_adapter import Provenance

#: Data entities the simulator emits in bulk and no policy should reason over.
NOISE_DATA_IDS = frozenset({"logfile"})


def filter_noise_data(observation: Optional[Observation]) -> Optional[Observation]:
    """Drop bulk `logfile` data entities from an observation.

    The simulator hands out one of these per host per step, which inflates the
    data node count and the `ExfiltrateData` branching factor without carrying
    information. `sgrl_netsec` filters them in place; this returns a new
    observation instead, so the caller's copy is never mutated underneath it.

    `None` passes through: `make_step` returns `None` when the server reply
    carries no observation (error or timeout).
    """
    if observation is None:
        return None

    known_data = {}
    for host, items in observation.state.known_data.items():
        kept = {item for item in items if item.id not in NOISE_DATA_IDS}
        if kept:
            known_data[host] = kept

    if known_data == observation.state.known_data:
        return observation

    state = GameState(
        controlled_hosts=observation.state.controlled_hosts,
        known_hosts=observation.state.known_hosts,
        known_services=observation.state.known_services,
        known_data=known_data,
        known_networks=observation.state.known_networks,
        known_blocks=observation.state.known_blocks,
    )
    return observation._replace(state=state)


class SurrogateController:
    """Everything needed to choose an action. No connection, no episode loop."""

    def __init__(
        self,
        weights: Optional[str] = None,
        temperature: float = 0.1,
        provenance: Optional[Provenance] = None,
        greedy: bool = False,
        raw_action_space: bool = False,
        legacy_service_port: bool = False,
        policy: Optional[FactoredGNNPolicy] = None,
        device: Optional[torch.device] = None,
    ):
        self.surrogate = (
            SurrogatePolicy(policy=policy, device=device, legacy_service_port=legacy_service_port)
            if policy is not None
            else SurrogatePolicy.load(weights, legacy_service_port=legacy_service_port)
        )
        self.weights = weights
        self.temperature = temperature
        self.provenance = provenance
        self.greedy = greedy
        self.raw_action_space = raw_action_space
        self.last_decision: Optional[Decision] = None
        self.fallbacks = 0

    @property
    def policy(self) -> FactoredGNNPolicy:
        return self.surrogate.policy

    @property
    def attempt_counts(self) -> AttemptCounts:
        return self.surrogate.attempt_counts

    def reset(self) -> None:
        """Clear per-episode interaction history."""
        self.surrogate.reset()
        self.fallbacks = 0

    def valid_actions(self, state: GameState) -> List[Action]:
        return candidates_mod.enumerate_actions(
            state, require_controlled_exfil_source=not self.raw_action_space
        )

    def select_action(
        self, observation: Observation, temperature: Optional[float] = None
    ) -> Action:
        """Choose one action for an observation, recording the attempt."""
        actions = self.valid_actions(observation.state)
        if not actions:
            raise ValueError("no valid action for this observation")

        decision = self.surrogate.act(
            observation.state,
            actions,
            provenance=self.provenance,
            temperature=self.temperature if temperature is None else temperature,
            greedy=self.greedy,
        )
        self.last_decision = decision
        if decision.fallback:
            self.fallbacks += 1
        return decision.action

    def state_value(self, observation: Observation) -> torch.Tensor:
        """V(s) from the shared backbone, for anyone wiring this into RL."""
        graph, _, _ = state_to_pyg(
            observation.state,
            attempt_counts=self.attempt_counts,
            provenance=self.provenance,
        )
        summary = state_summary(observation.state, provenance=self.provenance)
        return self.policy.state_value(
            graph.to(self.surrogate.device), summary.to(self.surrogate.device)
        )

    def info(self) -> Dict[str, Any]:
        return {
            "policy": "nsg_surrogate.FactoredGNNPolicy",
            "weights": self.weights or "<random-init>",
            "temperature": self.temperature,
            "greedy": self.greedy,
            "raw_action_space": self.raw_action_space,
        }


@dataclass
class EpisodeResult:
    steps: int = 0
    reward: float = 0.0
    end_reason: Optional[Any] = None
    fallbacks: int = 0
    actions: List[str] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return self.end_reason == AgentStatus.Success


@dataclass
class EvalStats:
    """Aggregate over episodes, in the same shape the simulator agent reports.

    Spread travels with every mean: two runs with the same average length can
    behave very differently, and a win rate from 10 episodes is not a
    measurement without its standard error.
    """

    episodes: int = 0
    win_rate: float = 0.0
    win_rate_se: float = 0.0
    wins: int = 0
    avg_steps: float = 0.0
    std_steps: float = 0.0
    avg_win_steps: float = 0.0
    std_win_steps: float = 0.0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    fallbacks: int = 0
    action_type_counts: Dict[str, int] = field(default_factory=dict)
    end_reasons: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "win_rate": round(self.win_rate, 4),
            "win_rate_se": round(self.win_rate_se, 4),
            "wins": self.wins,
            "avg_steps": round(self.avg_steps, 2),
            "std_steps": round(self.std_steps, 2),
            "avg_win_steps": round(self.avg_win_steps, 2),
            "std_win_steps": round(self.std_win_steps, 2),
            "avg_reward": round(self.avg_reward, 2),
            "std_reward": round(self.std_reward, 2),
            "fallbacks": self.fallbacks,
            "action_type_counts": self.action_type_counts,
            "end_reasons": self.end_reasons,
        }

    def summary_line(self) -> str:
        wins = (
            f"{self.avg_win_steps:.1f} ± {self.std_win_steps:.1f}"
            if self.wins
            else "n/a"  # "0.0 ± 0.0" would read as "wins take no steps"
        )
        return (
            f"win rate {self.win_rate * 100:.1f}% ± {self.win_rate_se * 100:.1f} SE "
            f"({self.wins}/{self.episodes}) | steps all {self.avg_steps:.1f} ± "
            f"{self.std_steps:.1f} | steps wins {wins} | reward "
            f"{self.avg_reward:.2f} ± {self.std_reward:.2f}"
        )


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _proportion_se(proportion: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return math.sqrt(proportion * (1.0 - proportion) / count)


def aggregate(results: List[EpisodeResult]) -> EvalStats:
    """Summarise episode results. Separate from the loop so it is testable."""
    stats = EvalStats(episodes=len(results))
    if not results:
        return stats

    stats.wins = sum(1 for result in results if result.won)
    stats.win_rate = stats.wins / len(results)
    stats.win_rate_se = _proportion_se(stats.win_rate, len(results))
    stats.avg_steps, stats.std_steps = _mean_std([float(r.steps) for r in results])
    stats.avg_win_steps, stats.std_win_steps = _mean_std(
        [float(r.steps) for r in results if r.won]
    )
    stats.avg_reward, stats.std_reward = _mean_std([r.reward for r in results])
    stats.fallbacks = sum(result.fallbacks for result in results)

    types: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for result in results:
        for action_type in result.actions:
            types[action_type] = types.get(action_type, 0) + 1
        key = str(result.end_reason)
        reasons[key] = reasons.get(key, 0) + 1
    stats.action_type_counts = dict(sorted(types.items()))
    stats.end_reasons = dict(sorted(reasons.items()))
    return stats


class SurrogateAgent(BaseAgent):
    """The surrogate as an NSG agent: connect, register, play episodes."""

    def __init__(
        self,
        host: str,
        port: int,
        role: AgentRole = AgentRole.Attacker,
        controller: Optional[SurrogateController] = None,
        seed: int = 42,
        max_steps: Optional[int] = None,
        **controller_kwargs: Any,
    ):
        super().__init__(host, port, role)
        torch.manual_seed(seed)
        random.seed(seed)
        self.controller = controller or SurrogateController(**controller_kwargs)
        self.max_steps = max_steps
        self.policy.eval()

    @property
    def policy(self) -> FactoredGNNPolicy:
        return self.controller.policy

    def play_episode(
        self, observation: Optional[Observation], verbose: bool = False
    ) -> Tuple[EpisodeResult, Optional[Observation]]:
        """Run one episode from `observation` until the server ends it."""
        self.controller.reset()
        observation = filter_noise_data(observation)
        result = EpisodeResult()

        while observation and not observation.end:
            if self.max_steps is not None and result.steps >= self.max_steps:
                # The server enforces its own limit; this is only a guard for
                # scenarios configured without one.
                break

            action = self.controller.select_action(observation)
            result.actions.append(action.type.name)
            if verbose:
                print(f"    step {result.steps + 1}: {candidates_mod.describe(action)}")

            observation = filter_noise_data(self.make_step(action))
            if observation is None:
                break
            result.reward += observation.reward
            result.steps += 1
            if verbose:
                print(f"      reward={observation.reward} end={observation.end}")

        if observation is not None and observation.info:
            result.end_reason = observation.info.get("end_reason")
        result.fallbacks = self.controller.fallbacks
        return result, observation

    def evaluate(self, episodes: int, verbose: bool = False) -> Tuple[EvalStats, List[EpisodeResult]]:
        """Play `episodes` episodes and aggregate. Requires a live server."""
        results: List[EpisodeResult] = []
        with torch.no_grad():
            # The episode already on the server belongs to whatever ran before
            # us, so start from a clean reset.
            observation = self.request_game_reset(request_trajectory=False)
            for index in range(1, episodes + 1):
                result, observation = self.play_episode(observation, verbose=verbose)
                results.append(result)
                if verbose or True:
                    print(
                        f"  episode {index:>3} | steps {result.steps:>3} | "
                        f"reward {result.reward:>8.2f} | {result.end_reason}"
                    )
                if index < episodes:
                    observation = self.request_game_reset(request_trajectory=False)
        return aggregate(results), results


def play(
    host: str,
    port: int,
    episodes: int,
    weights: Optional[str] = None,
    temperature: float = 0.1,
    seed: int = 42,
    greedy: bool = False,
    raw_action_space: bool = False,
    provenance: Optional[Provenance] = None,
    verbose: bool = False,
) -> EvalStats:
    """Connect, register, evaluate, disconnect."""
    agent = SurrogateAgent(
        host,
        port,
        AgentRole.Attacker,
        seed=seed,
        weights=weights,
        temperature=temperature,
        greedy=greedy,
        raw_action_space=raw_action_space,
        provenance=provenance,
    )
    if agent.socket is None:
        raise ConnectionError(
            f"could not connect to the NetSecGame coordinator at {host}:{port} — "
            "start the server first (see README)"
        )

    try:
        observation = agent.register()
        if observation is None:
            raise ConnectionError("registration with the coordinator failed")
        stats, _ = agent.evaluate(episodes, verbose=verbose)
        return stats
    finally:
        agent.terminate_connection()
